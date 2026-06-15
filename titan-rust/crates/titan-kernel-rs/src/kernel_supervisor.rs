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

use std::path::Path;
use std::time::Duration;

use parking_lot::Mutex;
use thiserror::Error;
use titan_bus::broker::BusBroker;
use titan_core::constants::{
    API_RELOAD_DRAIN_TIMEOUT_S, API_RELOAD_HEALTH_TIMEOUT_S, SUPERVISION_LOG_PATH,
};
use titan_core::supervisor::{
    classify_exit, compute_backoff, escalation::kernel_default_decision, restart::RestartDecision,
    ChildSpec, EscalationDecision, MockDepProbe, Supervisor, SupervisorError,
};
use tokio::process::Child;
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::spawn::{
    spawn_guardian_hcl, spawn_substrate, spawn_titan_hcl, spawn_titan_hcl_api,
    spawn_titan_hcl_api_reload_child, SpawnConfig, SpawnError,
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

/// SPEC §11.B.5 / D-SPEC-149 — a deliberate, zero-downtime api code-reload
/// request delivered to the api `watch_loop` from the kernel's inbound
/// `KERNEL_API_RELOAD_REQUEST` bus subscriber. Carries the bus-payload
/// provenance fields (`{reason, requested_by}` per §8.1) for the audit log.
#[derive(Debug, Clone)]
pub struct ApiReloadCommand {
    /// Free-text reason from the operator (`reload-api` surface).
    pub reason: String,
    /// Who requested the reload (operator identity / host).
    pub requested_by: String,
}

/// What a `watch_loop` iteration woke on: the child exited (crash path) or a
/// planned api-reload command arrived (api child only). Distinguishing these
/// is load-bearing — a planned drain must NOT be classified as a crash (the
/// §11.B.5 crash-vs-planned-swap trap).
enum WatchEvent {
    /// `child.wait()` resolved — the child process exited.
    Exit(std::io::Result<std::process::ExitStatus>),
    /// A reload command arrived on the api reload channel (`Some`), or the
    /// channel closed (`None` → drop reload capability, keep crash-watching).
    Reload(Option<ApiReloadCommand>),
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
    /// SPEC §11.B.5 — sender half of the api-only reload channel. The
    /// kernel's `KERNEL_API_RELOAD_REQUEST` bus subscriber clones this to
    /// hand reload commands to the api `watch_loop`. `None` for every other
    /// child; one bounded channel for the api.
    api_reload_tx: mpsc::Sender<ApiReloadCommand>,
    /// Receiver half — taken once by `spawn_and_watch_titan_hcl_api` and
    /// moved into the api watch task's `tokio::select!`. Wrapped so the
    /// single-consumer move is explicit + idempotent-guarded.
    api_reload_rx: Mutex<Option<mpsc::Receiver<ApiReloadCommand>>>,
    /// Broker handle for emitting the §11.B.5 `SUPERVISION_API_*` bus events
    /// (reloaded / failed / rejected / degraded). `None` in test fixtures.
    broker: Option<Arc<BusBroker>>,
    /// shm dir (for reading the NEW api's `module_api_reload_state.bin`
    /// readiness slot during the health-gate). Mirrors `spawn_config.shm_dir`.
    shm_dir: PathBuf,
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
        // Keep a broker handle for §11.B.5 SUPERVISION_API_* emits + capture
        // shm_dir for the reload health-gate before spawn_config is moved.
        let broker_for_self = broker.clone();
        let shm_dir = spawn_config.shm_dir.clone();
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

        // SPEC §11.B.5 — bounded api reload channel (depth 8: a reload is a
        // rare operator action; the concurrency guard in P3 rejects overlap
        // anyway, so a small buffer is ample).
        let (api_reload_tx, api_reload_rx) = mpsc::channel::<ApiReloadCommand>(8);

        Ok(Arc::new(Self {
            supervisor: Arc::new(Mutex::new(sup)),
            publisher,
            spawn_config: Arc::new(spawn_config),
            shutdown,
            shutdown_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            runtime,
            terminate_requested: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            api_reload_tx,
            api_reload_rx: Mutex::new(Some(api_reload_rx)),
            broker: broker_for_self,
            shm_dir,
        }))
    }

    /// SPEC §11.B.5 — a clone of the api reload-channel sender for the
    /// kernel's `KERNEL_API_RELOAD_REQUEST` bus subscriber to publish
    /// reload commands onto. The api `watch_loop` owns the receiver.
    pub fn api_reload_sender(&self) -> mpsc::Sender<ApiReloadCommand> {
        self.api_reload_tx.clone()
    }

    /// Mark kernel shutdown in progress. Watch tasks observing child exit
    /// after this point treat the exit as clean (graceful cascade per
    /// SPEC §17). Idempotent.
    pub fn mark_shutdown_active(&self) {
        self.shutdown_active
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// SPEC §18.4 / RFP_supervision_lifecycle §7.D — explicitly SIGTERM every
    /// supervised child (substrate, guardian_hcl, titan_hcl, titan_hcl_api) at
    /// SHUTDOWN_BEGIN, while the bus broker is still alive.
    ///
    /// This is the ordered-drain trigger under `KillMode=mixed`: systemd
    /// signals only the kernel ($MAINPID), so the kernel — not systemd, not
    /// PDEATHSIG — must start the children draining. Without this, the children
    /// would only get SIGTERM via PDEATHSIG when the kernel EXITS, i.e. AFTER
    /// `broker.stop()`, so every worker's bus-dependent save-handshake would
    /// hang → SIGKILL (the pre-fix bug). The kernel then awaits the watch
    /// handles (children drain over the LIVE bus) before stopping the broker.
    ///
    /// `mark_shutdown_active()` MUST be called first so the resulting child
    /// exits are classified clean (no respawn). Best-effort: a missing/already-
    /// dead pid is logged, not fatal.
    pub async fn sigterm_children(&self) {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        let pids = self.supervisor.lock().running_pids();
        for (name, pid) in pids {
            if pid == 0 {
                continue;
            }
            info!(child = %name, pid, "kernel shutdown: SIGTERM to supervised child");
            if let Err(e) = kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
                warn!(child = %name, pid, err = ?e,
                    "kernel shutdown: SIGTERM to child failed (already exited?)");
            }
        }
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
            this.watch_loop(CHILD_NAME_SUBSTRATE, child, RespawnKind::Substrate, None)
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
            this.watch_loop(CHILD_NAME_PYTHON, child, RespawnKind::GuardianHcl, None)
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
            this.watch_loop(CHILD_NAME_TITAN_HCL, child, RespawnKind::TitanHcl, None)
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
        // SPEC §11.B.5 — take the single api reload receiver (idempotent: a
        // second spawn after a watch-task exit reuses None → crash-watch only,
        // matching the channel's single-consumer contract).
        let reload_rx = self.api_reload_rx.lock().take();
        let this = Arc::clone(self);
        let handle = self.runtime.spawn(async move {
            this.watch_loop(
                CHILD_NAME_TITAN_HCL_API,
                child,
                RespawnKind::TitanHclApi,
                reload_rx,
            )
            .await;
        });
        Ok(Some(handle))
    }

    /// Per-child watch loop: await child exit, classify, decide, respawn
    /// or escalate per SPEC §11.B.
    ///
    /// `reload_rx` is `Some` only for the api child (SPEC §11.B.5): the loop
    /// then `select!`s between `child.wait()` (the unchanged crash path) and a
    /// planned `KERNEL_API_RELOAD_REQUEST` reload command. The reload arm runs
    /// the zero-downtime swap inline (P3) — for P2 it logs that the command
    /// reached the loop. **Crash-vs-planned distinction is load-bearing**: a
    /// planned drain must never be misread as a crash (it is consumed by the
    /// swap's own `wait()`, not the loop's top-level `child.wait()`).
    async fn watch_loop(
        self: Arc<Self>,
        name: &'static str,
        mut child: Child,
        kind: RespawnKind,
        mut reload_rx: Option<mpsc::Receiver<ApiReloadCommand>>,
    ) {
        loop {
            let pid = child.id().unwrap_or(0);

            // Wake on EITHER child exit (crash path) OR — api child only — a
            // planned reload command. `child.wait()` is cancel-safe, so when
            // the reload arm fires we simply re-await on the next iteration.
            let event = match reload_rx.as_mut() {
                Some(rx) => {
                    tokio::select! {
                        wr = child.wait() => WatchEvent::Exit(wr),
                        cmd = rx.recv() => WatchEvent::Reload(cmd),
                    }
                }
                None => WatchEvent::Exit(child.wait().await),
            };

            let wait_result = match event {
                WatchEvent::Reload(Some(cmd)) => {
                    // SPEC §11.B.5 — run the zero-downtime swap inline. The
                    // swap owns the OLD `child` handle, so the PID swap is
                    // local + race-free. On success `child` becomes NEW; on
                    // rollback OLD is untouched. Either way we `continue` (OLD
                    // never enters the crash path — the planned drain consumes
                    // OLD's exit inside the swap).
                    info!(
                        event = "KERNEL_API_RELOAD_RECEIVED",
                        child = name,
                        pid,
                        reason = %cmd.reason,
                        requested_by = %cmd.requested_by,
                        "api reload command received; starting zero-downtime swap"
                    );
                    match self.run_api_swap(&mut child).await {
                        Some(new_child) => child = new_child,
                        None => { /* rollback: keep OLD child serving */ }
                    }
                    // INV-API-RELOAD-2 — reject (not queue) any reload commands
                    // that arrived during the swap. A single swap at a time.
                    if let Some(rx) = reload_rx.as_mut() {
                        while let Ok(rejected) = rx.try_recv() {
                            warn!(
                                event = "SUPERVISION_API_RELOAD_REJECTED",
                                reason = "swap_in_flight",
                                rejected_reason = %rejected.reason,
                                "api reload rejected — a swap was already in-flight (INV-API-RELOAD-2)"
                            );
                            self.emit_api_event(
                                "SUPERVISION_API_RELOAD_REJECTED",
                                vec![(
                                    "reason".to_string(),
                                    rmpv::Value::String("swap_in_flight".into()),
                                )],
                            );
                        }
                    }
                    continue;
                }
                WatchEvent::Reload(None) => {
                    // Sender dropped (kernel shutdown / no subscriber). Drop the
                    // reload capability and fall back to pure crash-watch so the
                    // loop never spins on a closed channel.
                    warn!(
                        child = name,
                        "api reload channel closed; continuing with crash-watch only"
                    );
                    reload_rx = None;
                    continue;
                }
                WatchEvent::Exit(wr) => wr,
            };

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
                    // SPEC §11.B.5 D6 / INV-API-HA — the api child must NEVER
                    // escalate to Terminate (a full-tree bounce would take the
                    // whole Titan down — exactly the crashloop INV-API-HA
                    // forbids). On api escalation we override the default
                    // policy: keep retrying the last-known-good with a CAPPED
                    // exponential backoff (never a tight loop), reset the
                    // counter so we never give up, and emit SUPERVISION_API_
                    // DEGRADED so the operator sees the degraded state. A bad
                    // *reload* can't reach here (it's caught at the §5.1 health-
                    // gate → rollback before adoption); this is the safety net
                    // for an adopted api that later crashes repeatedly.
                    if matches!(kind, RespawnKind::TitanHclApi) {
                        let restart_count = self
                            .supervisor
                            .lock()
                            .restart_count_in_window(name)
                            .unwrap_or(0);
                        let backoff = compute_backoff(restart_count.max(1));
                        warn!(
                            event = "SUPERVISION_API_DEGRADED",
                            child = name,
                            escalation_id = %escalation_id,
                            restart_count,
                            backoff_s = backoff.as_secs_f64(),
                            "api repeated crash — capped-backoff Continue (NEVER Terminate, INV-API-HA)"
                        );
                        self.emit_api_event(
                            "SUPERVISION_API_DEGRADED",
                            vec![
                                ("pid".to_string(), rmpv::Value::Nil),
                                (
                                    "restart_count".to_string(),
                                    rmpv::Value::from(restart_count),
                                ),
                                (
                                    "backoff_s".to_string(),
                                    rmpv::Value::from(backoff.as_secs_f64()),
                                ),
                                (
                                    "detail".to_string(),
                                    rmpv::Value::String(
                                        "api escalation → capped-backoff continue".into(),
                                    ),
                                ),
                            ],
                        );
                        // Reset the window so the next exit starts fresh (never
                        // re-escalates immediately → never spins, never gives up).
                        let _ = self.supervisor.lock().reset_restart_counter(name);
                        tokio::time::sleep(backoff).await;
                        let respawn = match kind.spawn(&self.spawn_config) {
                            Ok(Some(c)) => Some(c),
                            Ok(None) => None,
                            Err(e) => {
                                error!(err = ?e, child = name, "api degraded respawn failed");
                                None
                            }
                        };
                        match respawn {
                            Some(new_child) => {
                                let new_pid = new_child.id().unwrap_or(0);
                                if let Err(e) = self.supervisor.lock().mark_running(name, new_pid) {
                                    error!(err = ?e, child = name, "mark_running failed after api degraded respawn");
                                    return;
                                }
                                child = new_child;
                                continue;
                            }
                            None => {
                                warn!(
                                    child = name,
                                    "api degraded respawn returned no child; aborting watch"
                                );
                                return;
                            }
                        }
                    }

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

    /// SPEC §11.B.5 — the zero-downtime api swap (runs inside the api
    /// watch_loop's reload arm; `old_child` is the tracked OLD api Child).
    ///
    /// spawn-NEW(reload-child) → SHM health-gate → drain-OLD → adopt. NEW
    /// inherits the kernel-owned listen fd (socket activation) so it shares
    /// OLD's accept queue — draining OLD drops zero connections.
    /// Returns `Some(new_child)` to adopt NEW, or `None` (rollback) to keep OLD
    /// serving. INV-API-HA: OLD is never signalled until NEW is proven RUNNING;
    /// a failure at any gate is a no-op on OLD.
    async fn run_api_swap(self: &Arc<Self>, old_child: &mut Child) -> Option<Child> {
        let old_pid = old_child.id().unwrap_or(0);

        // 1. Spawn NEW (inherits the kernel-owned listen fd; flagged
        //    reload-child so it writes the dedicated readiness slot + defers its
        //    accept loop until warm + self-promotes after OLD exits).
        let mut new_child = match spawn_titan_hcl_api_reload_child(&self.spawn_config) {
            Ok(Some(c)) => c,
            Ok(None) => {
                warn!("api swap: spawn returned None (disabled); reload is a no-op");
                self.emit_api_event(
                    "SUPERVISION_API_RELOAD_FAILED",
                    vec![
                        ("stage".to_string(), rmpv::Value::String("spawn".into())),
                        ("old_pid".to_string(), rmpv::Value::from(old_pid)),
                        (
                            "detail".to_string(),
                            rmpv::Value::String("spawn disabled".into()),
                        ),
                    ],
                );
                return None;
            }
            Err(e) => {
                error!(err = ?e, "api swap: spawn NEW failed; OLD keeps serving");
                self.emit_api_event(
                    "SUPERVISION_API_RELOAD_FAILED",
                    vec![
                        ("stage".to_string(), rmpv::Value::String("spawn".into())),
                        ("old_pid".to_string(), rmpv::Value::from(old_pid)),
                        (
                            "detail".to_string(),
                            rmpv::Value::String(format!("{e:?}").into()),
                        ),
                    ],
                );
                return None;
            }
        };
        let new_pid = new_child.id().unwrap_or(0);
        info!(
            event = "KERNEL_API_SWAP_SPAWNED_NEW",
            old_pid, new_pid, "api swap: NEW spawned; health-gating"
        );

        // 2. Health-gate NEW (SHM, pid-specific). NEW must reach RUNNING before
        //    OLD is touched (INV-API-RELOAD-1).
        let health_start = tokio::time::Instant::now();
        let healthy = self.gate_api_health(new_pid).await;
        let health_wait_s = health_start.elapsed().as_secs_f64();

        if !healthy {
            // 3. Rollback — SIGKILL NEW, leave OLD serving (INV-API-RELOAD-3).
            warn!(
                event = "KERNEL_API_SWAP_ROLLBACK",
                old_pid,
                new_pid,
                health_wait_s,
                "api swap: NEW did not reach RUNNING in budget; rolling back (OLD keeps serving)"
            );
            let _ = new_child.start_kill();
            let _ = new_child.wait().await;
            self.emit_api_event(
                "SUPERVISION_API_RELOAD_FAILED",
                vec![
                    ("stage".to_string(), rmpv::Value::String("health".into())),
                    ("old_pid".to_string(), rmpv::Value::from(old_pid)),
                    ("new_pid".to_string(), rmpv::Value::from(new_pid)),
                    (
                        "detail".to_string(),
                        rmpv::Value::String("NEW api never reached state=running".into()),
                    ),
                ],
            );
            return None;
        }

        // 4. Drain + retire OLD (SIGTERM graceful; SIGKILL on drain-timeout).
        let drain_start = tokio::time::Instant::now();
        self.drain_old_api(old_child, old_pid).await;
        let drain_s = drain_start.elapsed().as_secs_f64();

        // 5. Adopt NEW (PID swap) + reset the api restart counter (fresh, clean
        //    process → don't carry OLD's crash history into NEW's window).
        {
            let mut sup = self.supervisor.lock();
            if let Err(e) = sup.mark_running(CHILD_NAME_TITAN_HCL_API, new_pid) {
                error!(err = ?e, "api swap: mark_running(NEW) failed");
            }
            let _ = sup.reset_restart_counter(CHILD_NAME_TITAN_HCL_API);
        }
        info!(
            event = "SUPERVISION_API_RELOADED",
            old_pid, new_pid, health_wait_s, drain_s, "api zero-downtime reload complete"
        );
        self.emit_api_event(
            "SUPERVISION_API_RELOADED",
            vec![
                ("old_pid".to_string(), rmpv::Value::from(old_pid)),
                ("new_pid".to_string(), rmpv::Value::from(new_pid)),
                (
                    "health_wait_s".to_string(),
                    rmpv::Value::from(health_wait_s),
                ),
                ("drain_s".to_string(), rmpv::Value::from(drain_s)),
            ],
        );
        Some(new_child)
    }

    /// Poll the NEW api's `module_api_reload_state.bin` readiness slot until
    /// `state=="running"` AND `pid==new_pid` (pid-specific — Preamble G18 state
    /// via SHM), bounded by `API_RELOAD_HEALTH_TIMEOUT_S`. Returns `true` once
    /// NEW is healthy, `false` on timeout.
    async fn gate_api_health(self: &Arc<Self>, new_pid: u32) -> bool {
        let slot_path = self.shm_dir.join("module_api_reload_state.bin");
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs_f64(API_RELOAD_HEALTH_TIMEOUT_S);
        loop {
            if let Some((state, pid)) = read_api_reload_state(&slot_path) {
                if pid == new_pid && state == "running" {
                    return true;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Drain + retire the OLD api: SIGTERM (uvicorn graceful — finishes
    /// in-flight requests, closes WebSockets), bounded by
    /// `API_RELOAD_DRAIN_TIMEOUT_S`; SIGKILL only on drain-timeout. The OLD
    /// exit is consumed HERE (not by the watch_loop's crash arm).
    async fn drain_old_api(self: &Arc<Self>, old_child: &mut Child, old_pid: u32) {
        // Graceful SIGTERM (tokio's start_kill sends SIGKILL — we want a clean
        // uvicorn shutdown, so signal explicitly via nix).
        if old_pid != 0 {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            if let Err(e) = kill(Pid::from_raw(old_pid as i32), Signal::SIGTERM) {
                warn!(old_pid, err = ?e, "api drain: SIGTERM to OLD failed; will SIGKILL");
            }
        }
        let drain_timeout = Duration::from_secs_f64(API_RELOAD_DRAIN_TIMEOUT_S);
        match tokio::time::timeout(drain_timeout, old_child.wait()).await {
            Ok(_) => {
                info!(old_pid, "api drain: OLD exited gracefully");
            }
            Err(_) => {
                warn!(
                    old_pid,
                    drain_timeout_s = API_RELOAD_DRAIN_TIMEOUT_S,
                    "api drain: OLD did not exit in budget; SIGKILL"
                );
                let _ = old_child.start_kill();
                let _ = old_child.wait().await;
            }
        }
    }

    /// Emit a `SUPERVISION_API_*` bus event (§8.1) via the kernel broker. A
    /// `ts` field is appended automatically. No-op if no broker (tests).
    fn emit_api_event(&self, topic: &str, mut fields: Vec<(String, rmpv::Value)>) {
        let broker = match &self.broker {
            Some(b) => b.clone(),
            None => return,
        };
        // Append a wall-clock ts (seconds) per the §8.1 payload schemas.
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        fields.push(("ts".to_string(), rmpv::Value::from(ts)));
        let payload = rmpv::Value::Map(
            fields
                .into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), v))
                .collect(),
        );
        let envelope = match titan_bus::message::encode_simple(
            topic,
            Some("kernel"),
            Some("all"),
            Some(payload),
        ) {
            Ok(b) => b,
            Err(e) => {
                warn!(err = ?e, topic, "emit_api_event: encode failed");
                return;
            }
        };
        let topic_owned = topic.to_string();
        self.runtime.spawn(async move {
            broker
                .publish_local(&topic_owned, "kernel:supervisor", envelope)
                .await;
        });
    }
}

/// Read the api reload-readiness slot (`module_api_reload_state.bin`) and
/// return `(state, pid)` from the msgpack `ModuleStateEntry`, or `None` if the
/// slot is missing / uninitialized / undecodable. Uses the canonical
/// `titan_state::Slot` SeqLock-safe reader (never hand-rolled offset reads).
fn read_api_reload_state(slot_path: &Path) -> Option<(String, u32)> {
    let slot = titan_state::Slot::open(slot_path).ok()?;
    let bytes = slot.read().ok()?;
    if bytes.is_empty() {
        return None;
    }
    let val: rmpv::Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes[..])).ok()?;
    let entries = match val {
        rmpv::Value::Map(m) => m,
        _ => return None,
    };
    let mut state: Option<String> = None;
    let mut pid: Option<u32> = None;
    for (k, v) in entries {
        if let rmpv::Value::String(ks) = &k {
            match ks.as_str() {
                Some("state") => state = v.as_str().map(|s| s.to_string()),
                Some("pid") => pid = v.as_u64().map(|p| p as u32),
                _ => {}
            }
        }
    }
    match (state, pid) {
        (Some(s), Some(p)) => Some((s, p)),
        _ => None,
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

    fn write_reload_slot(path: &std::path::Path, state: &str, pid: u32) {
        let mut slot = titan_state::Slot::create(path, 1, 16384).unwrap();
        let payload = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("state".into()),
                rmpv::Value::String(state.into()),
            ),
            (rmpv::Value::String("pid".into()), rmpv::Value::from(pid)),
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &payload).unwrap();
        slot.write(&buf).unwrap();
    }

    #[test]
    fn read_api_reload_state_decodes_state_and_pid() {
        // SPEC §11.B.5 health-gate reader: decodes {state, pid} from the
        // dedicated reload-readiness slot via the canonical SeqLock Slot.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module_api_reload_state.bin");
        write_reload_slot(&path, "running", 54321);
        assert_eq!(
            read_api_reload_state(&path),
            Some(("running".to_string(), 54321))
        );
    }

    #[test]
    fn read_api_reload_state_reflects_pre_running_state() {
        // A NEW api still booting reports "starting"/"booted" — the gate must
        // see the non-running state (and so keep waiting), never false-pass.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module_api_reload_state.bin");
        write_reload_slot(&path, "starting", 777);
        let (state, pid) = read_api_reload_state(&path).unwrap();
        assert_eq!(state, "starting");
        assert_eq!(pid, 777);
    }

    #[test]
    fn read_api_reload_state_missing_slot_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.bin");
        assert_eq!(read_api_reload_state(&path), None);
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
