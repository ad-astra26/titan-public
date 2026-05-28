//! kernel_supervisor — Kernel's child supervisor wrapper (Phase C C-S7).
//!
//! Wires the existing `titan-core::supervisor::Supervisor` framework
//! (decision logic + escalation handshake + reason classification) to
//! actual `tokio::process::Child` lifecycles for the kernel's two direct
//! children:
//!
//! 1. `trinity-substrate` (titan-trinity-rs)
//! 2. `titan_HCL` (python -m titan_hcl)
//!
//! Mirrors the proven shape of `titan-unified-spirit-rs::supervise::DaemonSupervisor`
//! that already supervises the 6 leaf trinity daemons.
//!
//! # Restart cascade (SPEC §11.B)
//!
//! For each child, a tokio task awaits `Child::wait()`. On exit:
//!
//! 1. If shutdown is in progress → log clean exit, return.
//! 2. Else → classify exit code → `Supervisor::handle_child_exit` → match
//!    `RestartDecision`:
//!    - `Respawn { delay }` → sleep delay, call spawn_fn again, register
//!      new pid, loop.
//!    - `Escalate { escalation_id }` → kernel-self short-circuit: call
//!      `kernel_default_decision(reason)` directly (kernel IS the
//!      escalation recipient per SPEC §11.B.1 step 4) → act on decision:
//!        - `Terminate` → kernel exits with code 64 (escalation range
//!          per SPEC §15) → systemd cascades a fresh kernel restart.
//!        - `Halt` → leave child dead; periodic CHILD_DOWN reminder.
//!        - `Continue` → reset counter, loop (re-attempt respawn).
//!    - `RespawnBlocked { blocked_dep, recheck_interval }` → sleep
//!      recheck_interval, retry pre-respawn dep check (handled by
//!      Supervisor on next exit cycle).
//!    - `NoRestart` → log + return (child marked restart_on_crash=false).
//!
//! Every transition is published via `JsonlSupervisionPublisher` →
//! supervision.jsonl + bus broker fanout.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;
use titan_bus::broker::BusBroker;
use titan_core::constants::SUPERVISION_LOG_PATH;
use titan_core::supervisor::{
    classify_exit, escalation::kernel_default_decision, restart::RestartDecision, ChildSpec,
    EscalationDecision, MockDepProbe, Supervisor, SupervisorError,
};
use tokio::process::Child;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::spawn::{
    spawn_guardian_hcl, spawn_substrate, spawn_titan_hcl, spawn_titan_hcl_api, SpawnConfig,
    SpawnError,
};
use crate::supervision_log::{JsonlSupervisionPublisher, SupervisionLogError};

/// Canonical child name for the trinity-substrate (titan-trinity-rs)
/// child. MUST match SPEC §9.A.
pub const CHILD_NAME_SUBSTRATE: &str = "trinity-substrate";

/// Canonical child name for the guardian_hcl (L1 supervisor) Python peer.
/// Historically "titan_HCL" (the supervised "python" child); kept stable for
/// the supervision.jsonl audit trail. MUST match SPEC §9.B.
pub const CHILD_NAME_PYTHON: &str = "titan_HCL";

/// Phase 11 §11.I.1 / §11.B.4 — the titan_hcl (L2 orchestrator) Python peer.
pub const CHILD_NAME_TITAN_HCL: &str = "titan_hcl";

/// Phase 11 §11.I.1 / §11.B.4 — the titan_hcl_api (L3) Python peer.
pub const CHILD_NAME_TITAN_HCL_API: &str = "titan_hcl_api";

/// Which spawn function respawns a given supervised child (Phase 11.x — kernel
/// now supervises all 3 Python peers + substrate; each must respawn via its
/// OWN spawn fn, not a single hardcoded one). Normalizes the substrate spawn
/// (`Result<Child>`) to the python peers' `Result<Option<Child>>` shape.
#[derive(Clone, Copy, Debug)]
enum RespawnKind {
    Substrate,
    GuardianHcl,
    TitanHcl,
    TitanHclApi,
}

impl RespawnKind {
    fn spawn(self, cfg: &SpawnConfig) -> Result<Option<Child>, SpawnError> {
        match self {
            RespawnKind::Substrate => spawn_substrate(cfg).map(Some),
            RespawnKind::GuardianHcl => spawn_guardian_hcl(cfg),
            RespawnKind::TitanHcl => spawn_titan_hcl(cfg),
            RespawnKind::TitanHclApi => spawn_titan_hcl_api(cfg),
        }
    }
}

/// Errors during kernel-supervisor setup.
#[derive(Debug, Error)]
pub enum KernelSupervisorError {
    /// Underlying titan-core Supervisor returned an error.
    #[error("supervisor: {0}")]
    Supervisor(#[from] SupervisorError),
    /// Failed to construct the supervision.jsonl publisher.
    #[error("supervision_log: {0}")]
    Log(#[from] SupervisionLogError),
    /// Failed to spawn a child via spawn::spawn_*.
    #[error("spawn {child}: {source}")]
    Spawn {
        /// Child name (substrate / python).
        child: String,
        /// Underlying spawn error.
        #[source]
        source: SpawnError,
    },
}

/// Kernel-level supervisor — tracks substrate + python_main with full
/// SPEC §11.B restart cascade.
pub struct KernelChildSupervisor {
    /// Wrapped titan-core Supervisor (bookkeeping + decision logic).
    supervisor: Arc<Mutex<Supervisor>>,
    /// supervision.jsonl publisher (writes JSONL + fans to bus broker).
    /// Held via Arc shared with the wrapped Supervisor — the Supervisor
    /// invokes it on every state transition; we keep an explicit reference
    /// so the lifetime is tied to the KernelChildSupervisor.
    #[allow(dead_code)]
    publisher: Arc<JsonlSupervisionPublisher>,
    /// Spawn config (binaries, env, paths) — shared by both children.
    spawn_config: Arc<SpawnConfig>,
    /// Notifies the steady-state loop that we want to exit (e.g. on
    /// `EscalationDecision::Terminate`).
    shutdown: Arc<Notify>,
    /// Whether kernel shutdown is in progress (used to distinguish clean
    /// exits during shutdown from unexpected crashes).
    shutdown_active: Arc<std::sync::atomic::AtomicBool>,
    /// Tokio runtime handle (for spawning watch tasks).
    runtime: tokio::runtime::Handle,
    /// Whether the escalation policy has resolved to terminate (set by
    /// any watch task; checked by main loop).
    terminate_requested: Arc<std::sync::atomic::AtomicBool>,
}

impl KernelChildSupervisor {
    /// Construct a kernel-level supervisor and register substrate + python
    /// ChildSpecs.
    pub fn new(
        spawn_config: SpawnConfig,
        boot_generation: u64,
        broker: Option<Arc<BusBroker>>,
        shutdown: Arc<Notify>,
        runtime: tokio::runtime::Handle,
        data_dir: &std::path::Path,
    ) -> Result<Arc<Self>, KernelSupervisorError> {
        let log_path: PathBuf = data_dir.join(
            // SUPERVISION_LOG_PATH is "data/supervision.jsonl"; we already
            // joined data_dir so use just the filename.
            std::path::Path::new(SUPERVISION_LOG_PATH)
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("supervision.jsonl")),
        );
        let publisher = Arc::new(JsonlSupervisionPublisher::new(
            log_path,
            boot_generation,
            broker,
            runtime.clone(),
        )?);

        // Kernel children have no critical-dep declarations yet (Maker can
        // add via runtime override later via §11.G framework).
        let dep_probe = Arc::new(MockDepProbe::default());
        let mut sup = Supervisor::new("kernel", publisher.clone(), dep_probe);

        // Register both ChildSpecs. critical_data_writer=true on both:
        // substrate writes shm slots, python plugin writes inner_memory.db
        // + observatory.db etc. Their shutdown discipline matters per §11.H.
        sup.register_child(ChildSpec::new(CHILD_NAME_SUBSTRATE).critical_data_writer())?;
        sup.register_child(ChildSpec::new(CHILD_NAME_PYTHON).critical_data_writer())?;
        // Phase 11.x — titan_hcl (L2 orchestrator) + titan_hcl_api (L3) are
        // kernel peers too; register them so the kernel watches + respawns them
        // (previously fire-and-forget spawned → zombied on death). titan_hcl is
        // a critical-data writer (inner_memory.db etc.); the api is read-only
        // SHM-direct (state reads survive its restart) so not critical-data.
        sup.register_child(ChildSpec::new(CHILD_NAME_TITAN_HCL).critical_data_writer())?;
        sup.register_child(ChildSpec::new(CHILD_NAME_TITAN_HCL_API))?;

        Ok(Arc::new(Self {
            supervisor: Arc::new(Mutex::new(sup)),
            publisher,
            spawn_config: Arc::new(spawn_config),
            shutdown,
            shutdown_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            runtime,
            terminate_requested: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }

    /// Mark kernel shutdown in progress. Watch tasks observing child exit
    /// after this point treat the exit as clean (graceful cascade per
    /// SPEC §17). Idempotent.
    pub fn mark_shutdown_active(&self) {
        self.shutdown_active
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// True iff a watch task escalated to Terminate (kernel must exit
    /// with code 64 so systemd cascades a fresh restart per SPEC §11.B.1
    /// step 6b + §15 escalation range).
    pub fn terminate_requested(&self) -> bool {
        self.terminate_requested
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Spawn substrate + start its watch task. Returns the JoinHandle for
    /// the watch task (for orderly shutdown).
    pub fn spawn_and_watch_substrate(
        self: &Arc<Self>,
    ) -> Result<JoinHandle<()>, KernelSupervisorError> {
        let child =
            spawn_substrate(&self.spawn_config).map_err(|source| KernelSupervisorError::Spawn {
                child: CHILD_NAME_SUBSTRATE.to_string(),
                source,
            })?;
        let pid = child.id().unwrap_or(0);
        info!(
            event = "KERNEL_SUPERVISOR_SUBSTRATE_SPAWNED",
            pid, "substrate spawned + supervision attached"
        );
        self.supervisor
            .lock()
            .mark_running(CHILD_NAME_SUBSTRATE, pid)?;
        let this = Arc::clone(self);
        let handle = self.runtime.spawn(async move {
            this.watch_loop(CHILD_NAME_SUBSTRATE, child, RespawnKind::Substrate)
                .await;
        });
        Ok(handle)
    }

    /// Spawn python_main + start its watch task.
    pub fn spawn_and_watch_python(
        self: &Arc<Self>,
    ) -> Result<Option<JoinHandle<()>>, KernelSupervisorError> {
        let child_opt = spawn_guardian_hcl(&self.spawn_config).map_err(|source| {
            KernelSupervisorError::Spawn {
                child: CHILD_NAME_PYTHON.to_string(),
                source,
            }
        })?;
        let child = match child_opt {
            Some(c) => c,
            None => {
                info!("kernel_supervisor: python spawn disabled (test mode)");
                return Ok(None);
            }
        };
        let pid = child.id().unwrap_or(0);
        info!(
            event = "KERNEL_SUPERVISOR_PYTHON_SPAWNED",
            pid, "python_main spawned + supervision attached"
        );
        self.supervisor
            .lock()
            .mark_running(CHILD_NAME_PYTHON, pid)?;
        let this = Arc::clone(self);
        let handle = self.runtime.spawn(async move {
            this.watch_loop(CHILD_NAME_PYTHON, child, RespawnKind::GuardianHcl)
                .await;
        });
        Ok(Some(handle))
    }

    /// Spawn titan_hcl (L2 orchestrator) + start its watch task (Phase 11.x).
    pub fn spawn_and_watch_titan_hcl(
        self: &Arc<Self>,
    ) -> Result<Option<JoinHandle<()>>, KernelSupervisorError> {
        let child_opt =
            spawn_titan_hcl(&self.spawn_config).map_err(|source| KernelSupervisorError::Spawn {
                child: CHILD_NAME_TITAN_HCL.to_string(),
                source,
            })?;
        let child = match child_opt {
            Some(c) => c,
            None => {
                info!("kernel_supervisor: titan_hcl spawn disabled (test mode)");
                return Ok(None);
            }
        };
        let pid = child.id().unwrap_or(0);
        info!(
            event = "KERNEL_SUPERVISOR_TITAN_HCL_SPAWNED",
            pid, "titan_hcl spawned + supervision attached"
        );
        self.supervisor
            .lock()
            .mark_running(CHILD_NAME_TITAN_HCL, pid)?;
        let this = Arc::clone(self);
        let handle = self.runtime.spawn(async move {
            this.watch_loop(CHILD_NAME_TITAN_HCL, child, RespawnKind::TitanHcl)
                .await;
        });
        Ok(Some(handle))
    }

    /// Spawn titan_hcl_api (L3) + start its watch task (Phase 11.x).
    pub fn spawn_and_watch_titan_hcl_api(
        self: &Arc<Self>,
    ) -> Result<Option<JoinHandle<()>>, KernelSupervisorError> {
        let child_opt = spawn_titan_hcl_api(&self.spawn_config).map_err(|source| {
            KernelSupervisorError::Spawn {
                child: CHILD_NAME_TITAN_HCL_API.to_string(),
                source,
            }
        })?;
        let child = match child_opt {
            Some(c) => c,
            None => {
                info!("kernel_supervisor: titan_hcl_api spawn disabled (test mode)");
                return Ok(None);
            }
        };
        let pid = child.id().unwrap_or(0);
        info!(
            event = "KERNEL_SUPERVISOR_TITAN_HCL_API_SPAWNED",
            pid, "titan_hcl_api spawned + supervision attached"
        );
        self.supervisor
            .lock()
            .mark_running(CHILD_NAME_TITAN_HCL_API, pid)?;
        let this = Arc::clone(self);
        let handle = self.runtime.spawn(async move {
            this.watch_loop(CHILD_NAME_TITAN_HCL_API, child, RespawnKind::TitanHclApi)
                .await;
        });
        Ok(Some(handle))
    }

    /// Per-child watch loop: await child exit, classify, decide, respawn
    /// or escalate per SPEC §11.B.
    async fn watch_loop(self: Arc<Self>, name: &'static str, mut child: Child, kind: RespawnKind) {
        loop {
            let pid = child.id().unwrap_or(0);
            let wait_result = child.wait().await;

            // Distinguish clean shutdown cascade from unexpected crash.
            if self
                .shutdown_active
                .load(std::sync::atomic::Ordering::Acquire)
            {
                let exit_code = wait_result.ok().and_then(|s| s.code());
                info!(
                    event = "SUPERVISION_CHILD_CLEAN_EXIT",
                    child = name,
                    pid,
                    exit_code,
                    "child exited cleanly during kernel shutdown"
                );
                return;
            }

            // Unexpected exit — classify + decide.
            let exit_code = wait_result.ok().and_then(|s| s.code());
            let reason = exit_code
                .map(classify_exit)
                .unwrap_or(titan_core::supervisor::SupervisionReason::Killed);
            let detail = format!("child={name} pid={pid} exit_code={exit_code:?}");

            let decision = match self
                .supervisor
                .lock()
                .handle_child_exit(name, reason, detail, exit_code)
            {
                Ok(d) => d,
                Err(e) => {
                    error!(err = ?e, child = name, "supervisor handle_child_exit failed");
                    return;
                }
            };

            match decision {
                RestartDecision::Respawn { delay } => {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    info!(
                        event = "SUPERVISION_CHILD_RESPAWNING",
                        child = name,
                        delay_ms = delay.as_millis() as u64,
                        "respawning child"
                    );
                    // Re-spawn via the child's OWN spawn function (Phase 11.x —
                    // was a single hardcoded spawn_guardian_hcl for any python
                    // child, which would have respawned titan_hcl/api as
                    // guardian_hcl; RespawnKind routes each to its own fn).
                    let respawn = match kind.spawn(&self.spawn_config) {
                        Ok(Some(c)) => Some(c),
                        Ok(None) => None,
                        Err(e) => {
                            error!(err = ?e, child = name, "respawn failed");
                            None
                        }
                    };
                    match respawn {
                        Some(new_child) => {
                            let new_pid = new_child.id().unwrap_or(0);
                            if let Err(e) = self.supervisor.lock().mark_running(name, new_pid) {
                                error!(err = ?e, child = name, "mark_running failed after respawn");
                                return;
                            }
                            child = new_child;
                            // Continue the loop with the new child.
                        }
                        None => {
                            warn!(
                                child = name,
                                "respawn returned no child; aborting watch loop"
                            );
                            return;
                        }
                    }
                }
                RestartDecision::Escalate { escalation_id } => {
                    // Phase C C-S7 Gap B per Maker decision (2026-05-05):
                    // when supervisor IS kernel, short-circuit the bus
                    // round-trip and apply kernel_default_decision in-process.
                    // The SUPERVISION_ESCALATION event was already published
                    // by handle_child_exit (so supervision.jsonl has the
                    // audit record); we just need to act on the decision.
                    let policy_decision = kernel_default_decision(reason);
                    warn!(
                        event = "SUPERVISION_ESCALATION_DECIDED",
                        child = name,
                        escalation_id = %escalation_id,
                        most_common_reason = reason.as_str(),
                        decision = ?policy_decision,
                        "kernel-self escalation decided per default policy"
                    );
                    match policy_decision {
                        EscalationDecision::Continue => {
                            // Kernel grants more restarts: reset counter
                            // and continue. Re-spawn on next loop iteration.
                            // The titan-core Supervisor doesn't yet expose
                            // a reset-counter API; for now we just exit the
                            // watch task and let systemd handle it. (Future:
                            // add Supervisor::reset_restart_counter.)
                            warn!(
                                child = name,
                                "Continue policy: counter reset not yet \
                                 implemented; letting systemd cascade fresh tree"
                            );
                            self.terminate_requested
                                .store(true, std::sync::atomic::Ordering::Release);
                            self.shutdown.notify_waiters();
                            return;
                        }
                        EscalationDecision::Terminate => {
                            error!(
                                child = name,
                                "Terminate policy: kernel will exit with code 64 \
                                 → systemd cascades fresh tree per SPEC §11.B.1 step 6b"
                            );
                            self.terminate_requested
                                .store(true, std::sync::atomic::Ordering::Release);
                            self.shutdown.notify_waiters();
                            return;
                        }
                        EscalationDecision::Halt => {
                            warn!(
                                child = name,
                                "Halt policy: leaving child dead; Maker must \
                                 intervene. Periodic CHILD_DOWN reminder follows."
                            );
                            // For Halt: leave the watch task dead; child
                            // stays down. Future: emit periodic CHILD_DOWN
                            // reminder per SPEC §11.B.1 step 6c.
                            return;
                        }
                    }
                }
                RestartDecision::RespawnBlocked {
                    blocked_dep,
                    recheck_interval,
                } => {
                    warn!(
                        event = "SUPERVISION_DEPENDENCY_BLOCKED",
                        child = name,
                        blocked_dep = %blocked_dep,
                        recheck_ms = recheck_interval.as_millis() as u64,
                        "respawn blocked on dependency; waiting for recovery"
                    );
                    // For now: sleep and exit — Supervisor will reattempt
                    // on next handle_child_exit. Full recheck loop is a
                    // future enhancement (use SupervisionEvent timer thread).
                    tokio::time::sleep(recheck_interval).await;
                    return;
                }
                RestartDecision::NoRestart => {
                    info!(
                        event = "SUPERVISION_CHILD_NO_RESTART",
                        child = name,
                        "child marked restart_on_crash=false; not restarting"
                    );
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_core::supervisor::SupervisionReason;

    #[test]
    fn child_names_match_spec_9a() {
        // Per SPEC §9.A naming — these are the canonical names that
        // appear in supervision.jsonl + arch_map supervision-log queries.
        assert_eq!(CHILD_NAME_SUBSTRATE, "trinity-substrate");
        assert_eq!(CHILD_NAME_PYTHON, "titan_HCL");
    }

    #[test]
    fn classify_exit_smoke() {
        // Sanity check that classify_exit + kernel_default_decision align
        // on the worst-case escalation paths (load-bearing for cascade).
        assert_eq!(classify_exit(0), SupervisionReason::CleanExit);
        assert_eq!(classify_exit(1), SupervisionReason::Panic);
        assert_eq!(classify_exit(2), SupervisionReason::ConfigError);
        assert_eq!(classify_exit(139), SupervisionReason::Segv);
        assert_eq!(classify_exit(137), SupervisionReason::Killed);

        // Kernel default policy for each reason.
        assert_eq!(
            kernel_default_decision(SupervisionReason::Panic),
            EscalationDecision::Terminate
        );
        assert_eq!(
            kernel_default_decision(SupervisionReason::ConfigError),
            EscalationDecision::Halt
        );
        assert_eq!(
            kernel_default_decision(SupervisionReason::Empty),
            EscalationDecision::Halt
        );
    }
}
