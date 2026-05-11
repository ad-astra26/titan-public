//! supervise — Substrate-side supervisor for `titan-unified-spirit-rs`
//! (placeholder in C-S3, real binary in C-S4).
//!
//! C-S3 chunk C3-3 ships the SPAWN-AND-WAIT scaffold:
//!   - Builds the canonical child env per SPEC §3 D18 + §5
//!   - Spawns the placeholder binary
//!   - Tracks child PID for shutdown SIGTERM
//!   - Waits via `tokio::process::Child::wait()` so SIGCHLD is observed
//!
//! C-S5+ chunks add full one_for_one supervision via
//! `titan-core::supervisor` primitives (max_restarts=5, escalation handshake
//! per SPEC §11.B.1, dependency-aware respawn per §11.G).

use std::collections::HashMap;

use parking_lot::Mutex;
use thiserror::Error;
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::boot::BootConfig;
use crate::exit::SubstrateExitCode;

/// Errors during child spawn.
#[derive(Debug, Error)]
pub enum SuperviseError {
    /// Failed to spawn the child process (binary not found, perms, etc.).
    #[error("spawn {binary} failed: {source}")]
    SpawnFailed {
        /// Binary path attempted.
        binary: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

impl SuperviseError {
    /// Map to canonical exit code per SPEC §15.
    pub fn to_exit_code(&self) -> SubstrateExitCode {
        match self {
            SuperviseError::SpawnFailed { .. } => SubstrateExitCode::ChildLimitReached,
        }
    }
}

/// Build the canonical env passed to the unified-spirit child. Mirrors
/// `titan-kernel-rs::spawn::build_child_env` but with the substrate's PID as
/// parent + `unified-spirit` as daemon name.
pub fn build_unified_spirit_env(cfg: &BootConfig) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = HashMap::new();

    // Canonical names per SPEC §5
    env.insert("TITAN_KERNEL_TITAN_ID".into(), cfg.titan_id.clone());
    env.insert(
        "TITAN_KERNEL_BOOT_GENERATION".into(),
        cfg.boot_generation.to_string(),
    );
    if let Some(bus) = cfg.bus_socket.as_ref() {
        env.insert(
            "TITAN_KERNEL_BUS_SOCKET_PATH".into(),
            bus.to_string_lossy().into_owned(),
        );
    }
    env.insert(
        "TITAN_KERNEL_FASTBUS_PATH".into(),
        cfg.fastbus_path.to_string_lossy().into_owned(),
    );
    env.insert(
        "TITAN_KERNEL_SHM_DIR".into(),
        cfg.shm_dir.to_string_lossy().into_owned(),
    );

    // Pass through kernel-set authkey + log level + data dir captured by
    // BootConfig at boot time (avoids test-parallelism env races).
    if let Some(ref hex) = cfg.authkey_hex {
        env.insert("TITAN_AUTHKEY_HEX".into(), hex.clone());
    }
    if let Some(ref level) = cfg.log_level {
        env.insert("TITAN_KERNEL_LOG_LEVEL".into(), level.clone());
    }
    if let Some(ref dir) = cfg.data_dir {
        env.insert(
            "TITAN_KERNEL_DATA_DIR".into(),
            dir.to_string_lossy().into_owned(),
        );
    }

    // Daemon-side metadata: substrate is the new parent
    env.insert("TITAN_DAEMON_NAME".into(), "unified-spirit".into());
    env.insert(
        "TITAN_DAEMON_PARENT_PID".into(),
        std::process::id().to_string(),
    );

    // Legacy aliases per SPEC §3 D18
    env.insert("TITAN_BUS_TITAN_ID".into(), cfg.titan_id.clone());
    env.insert("TITAN_ID".into(), cfg.titan_id.clone());
    env.insert(
        "TITAN_SHM_ROOT".into(),
        cfg.shm_dir.to_string_lossy().into_owned(),
    );

    // Phase C C-S7 (2026-05-05) — forward TITAN_DAEMON_BINARY_DIR if set
    // in substrate's env (kernel-rs forwards from systemd Environment=).
    // unified-spirit-rs's clap arg reads this env var to override its
    // /usr/local/bin default. Lets each Titan have its own bin/ without
    // /usr/local/bin symlinks.
    if let Ok(daemon_bin_dir) = std::env::var("TITAN_DAEMON_BINARY_DIR") {
        env.insert("TITAN_DAEMON_BINARY_DIR".into(), daemon_bin_dir);
    }

    env
}

/// Spawn the unified-spirit child per PLAN §7.1 step S10.
pub fn spawn_unified_spirit(cfg: &BootConfig) -> Result<Child, SuperviseError> {
    let binary = &cfg.unified_spirit_binary;
    let env = build_unified_spirit_env(cfg);
    let mut cmd = Command::new(binary);
    cmd.env_clear().envs(env).kill_on_drop(false);

    info!(
        binary = ?binary,
        titan_id = %cfg.titan_id,
        "substrate: spawning unified-spirit-placeholder per PLAN §7.1 S10"
    );

    cmd.spawn().map_err(|source| SuperviseError::SpawnFailed {
        binary: binary.to_string_lossy().into_owned(),
        source,
    })
}

/// Tracked supervised children — `Mutex<Option<Child>>` so shutdown can take
/// ownership + send SIGTERM. Only the unified-spirit child today.
pub struct SupervisedChildren {
    /// Unified spirit (placeholder in C-S3; real binary in C-S4).
    pub unified_spirit: Mutex<Option<Child>>,
}

impl SupervisedChildren {
    /// New empty registry.
    pub fn new() -> Self {
        Self {
            unified_spirit: Mutex::new(None),
        }
    }

    /// Send SIGTERM to all live children (best-effort).
    pub fn sigterm_all(&self) {
        if let Some(mut child) = self.unified_spirit.lock().take() {
            let pid = child.id();
            info!(pid, "substrate: sending SIGTERM to unified-spirit child");
            if let Err(e) = child.start_kill() {
                warn!(err = ?e, pid, "substrate: SIGTERM to unified-spirit failed");
            }
        }
    }
}

impl Default for SupervisedChildren {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolves the placeholder binary relative to the substrate's own binary
/// when no explicit path is provided. Used at boot config resolution + tests.
pub fn default_unified_spirit_binary() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("titan-unified-spirit-rs")))
        .unwrap_or_else(|| std::path::PathBuf::from("titan-unified-spirit-rs"))
}

// ─── Phase C C-S7 Gap B — substrate→unified-spirit supervision ────────
//
// Wires titan-core::supervisor::Supervisor to actual unified-spirit child
// lifecycle so unexpected exits trigger the SPEC §11.0 row 3 cascade
// (substrate restarts unified-spirit; 6 daemons stay alive). Mirrors the
// kernel-side KernelChildSupervisor + leaf-level DaemonSupervisor patterns.

/// Canonical child name for unified-spirit per SPEC §9.A.
pub const CHILD_NAME_UNIFIED_SPIRIT: &str = "unified-spirit";

/// Substrate's supervisor for the unified-spirit child.
pub struct UnifiedSpiritSupervisor {
    /// Wrapped titan-core Supervisor (decision logic + classification).
    supervisor: std::sync::Arc<Mutex<titan_core::supervisor::Supervisor>>,
    /// Boot config (binaries, env, paths) — shared across respawns.
    boot_config: std::sync::Arc<BootConfig>,
    /// Notifies substrate's main loop when escalation requests shutdown.
    shutdown: std::sync::Arc<tokio::sync::Notify>,
    /// Whether substrate shutdown is in progress.
    shutdown_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether escalation resolved to terminate.
    terminate_requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl UnifiedSpiritSupervisor {
    /// Construct + register unified-spirit ChildSpec.
    pub fn new(
        boot_config: BootConfig,
        shutdown: std::sync::Arc<tokio::sync::Notify>,
    ) -> Result<std::sync::Arc<Self>, titan_core::supervisor::SupervisorError> {
        // RecordingPublisher captures events for tracing/tests; substrate
        // doesn't own supervision.jsonl (kernel does).
        let publisher = std::sync::Arc::new(titan_core::supervisor::RecordingPublisher::default());
        // No critical-dep declarations on unified-spirit at substrate level
        // today (substrate IS the dep that unified-spirit depends on, not
        // the other way).
        let dep_probe = std::sync::Arc::new(titan_core::supervisor::MockDepProbe::default());
        let mut sup = titan_core::supervisor::Supervisor::new("substrate", publisher, dep_probe);
        sup.register_child(
            titan_core::supervisor::ChildSpec::new(CHILD_NAME_UNIFIED_SPIRIT)
                .critical_data_writer(),
        )?;
        Ok(std::sync::Arc::new(Self {
            supervisor: std::sync::Arc::new(Mutex::new(sup)),
            boot_config: std::sync::Arc::new(boot_config),
            shutdown,
            shutdown_active: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            terminate_requested: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }

    /// Mark substrate shutdown active so the watch task treats subsequent
    /// child exit as clean.
    pub fn mark_shutdown_active(&self) {
        self.shutdown_active
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// True iff watch loop escalated to Terminate (substrate must exit
    /// with code 64 → kernel cascades fresh tree per SPEC §11.B.1 + §15).
    pub fn terminate_requested(&self) -> bool {
        self.terminate_requested
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Spawn unified-spirit + start its watch task.
    pub fn spawn_and_watch(
        self: &std::sync::Arc<Self>,
    ) -> Result<tokio::task::JoinHandle<()>, SuperviseError> {
        let child = spawn_unified_spirit(&self.boot_config)?;
        let pid = child.id().unwrap_or(0);
        info!(
            event = "SUBSTRATE_SUPERVISOR_UNIFIED_SPIRIT_SPAWNED",
            pid, "unified-spirit spawned + supervision attached"
        );
        if let Err(e) = self
            .supervisor
            .lock()
            .mark_running(CHILD_NAME_UNIFIED_SPIRIT, pid)
        {
            warn!(err = ?e, "mark_running failed; aborting watch");
            return Err(SuperviseError::SpawnFailed {
                binary: "unified-spirit".into(),
                source: std::io::Error::other("mark_running failed"),
            });
        }
        let this = std::sync::Arc::clone(self);
        let handle = tokio::spawn(async move {
            this.watch_loop(child).await;
        });
        Ok(handle)
    }

    /// Watch loop — await child exit, classify, decide, respawn or escalate.
    async fn watch_loop(self: std::sync::Arc<Self>, mut child: Child) {
        use titan_core::supervisor::{
            classify_exit, escalation::kernel_default_decision, restart::RestartDecision,
            EscalationDecision,
        };
        loop {
            let pid = child.id().unwrap_or(0);
            let wait_result = child.wait().await;

            if self
                .shutdown_active
                .load(std::sync::atomic::Ordering::Acquire)
            {
                let exit_code = wait_result.ok().and_then(|s| s.code());
                info!(
                    event = "SUPERVISION_CHILD_CLEAN_EXIT",
                    child = CHILD_NAME_UNIFIED_SPIRIT,
                    pid,
                    exit_code,
                    "unified-spirit exited cleanly during substrate shutdown"
                );
                return;
            }

            let exit_code = wait_result.ok().and_then(|s| s.code());
            let reason = exit_code
                .map(classify_exit)
                .unwrap_or(titan_core::supervisor::SupervisionReason::Killed);
            let detail = format!("unified-spirit pid={pid} exit_code={exit_code:?}");

            let decision = match self.supervisor.lock().handle_child_exit(
                CHILD_NAME_UNIFIED_SPIRIT,
                reason,
                detail,
                exit_code,
            ) {
                Ok(d) => d,
                Err(e) => {
                    warn!(err = ?e, "supervisor handle_child_exit failed");
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
                        child = CHILD_NAME_UNIFIED_SPIRIT,
                        delay_ms = delay.as_millis() as u64,
                        "respawning unified-spirit"
                    );
                    match spawn_unified_spirit(&self.boot_config) {
                        Ok(new_child) => {
                            let new_pid = new_child.id().unwrap_or(0);
                            if let Err(e) = self
                                .supervisor
                                .lock()
                                .mark_running(CHILD_NAME_UNIFIED_SPIRIT, new_pid)
                            {
                                warn!(err = ?e, "mark_running failed after respawn");
                                return;
                            }
                            child = new_child;
                        }
                        Err(e) => {
                            warn!(err = ?e, "unified-spirit respawn failed; aborting watch");
                            return;
                        }
                    }
                }
                RestartDecision::Escalate { escalation_id } => {
                    // Substrate has no Maker UI; apply kernel_default_decision
                    // in-process per Maker decision (kernel is the canonical
                    // escalation recipient — substrate mirrors that policy).
                    let policy_decision = kernel_default_decision(reason);
                    warn!(
                        event = "SUPERVISION_ESCALATION_DECIDED",
                        child = CHILD_NAME_UNIFIED_SPIRIT,
                        escalation_id = %escalation_id,
                        most_common_reason = reason.as_str(),
                        decision = ?policy_decision,
                        "substrate-self escalation decided per default policy"
                    );
                    match policy_decision {
                        EscalationDecision::Continue => {
                            warn!(
                                child = CHILD_NAME_UNIFIED_SPIRIT,
                                "Continue policy: substrate exits → kernel cascades fresh tree"
                            );
                            self.terminate_requested
                                .store(true, std::sync::atomic::Ordering::Release);
                            self.shutdown.notify_waiters();
                            return;
                        }
                        EscalationDecision::Terminate => {
                            warn!(
                                child = CHILD_NAME_UNIFIED_SPIRIT,
                                "Terminate policy: substrate exits → kernel cascades fresh tree"
                            );
                            self.terminate_requested
                                .store(true, std::sync::atomic::Ordering::Release);
                            self.shutdown.notify_waiters();
                            return;
                        }
                        EscalationDecision::Halt => {
                            warn!(
                                child = CHILD_NAME_UNIFIED_SPIRIT,
                                "Halt policy: leaving unified-spirit dead; \
                                 Maker must intervene"
                            );
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
                        child = CHILD_NAME_UNIFIED_SPIRIT,
                        blocked_dep = %blocked_dep,
                        recheck_ms = recheck_interval.as_millis() as u64,
                        "respawn blocked on dependency"
                    );
                    tokio::time::sleep(recheck_interval).await;
                    return;
                }
                RestartDecision::NoRestart => {
                    info!(
                        event = "SUPERVISION_CHILD_NO_RESTART",
                        child = CHILD_NAME_UNIFIED_SPIRIT,
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
    use std::path::PathBuf;

    fn test_cfg() -> BootConfig {
        BootConfig {
            titan_id: "T1".into(),
            fastbus_path: PathBuf::from("/tmp/fastbus.bin"),
            shm_dir: PathBuf::from("/tmp/shm"),
            data_dir: None,
            bus_socket: Some(PathBuf::from("/tmp/bus.sock")),
            boot_generation: 7,
            parent_pid: Some(123),
            authkey_hex: None,
            log_level: None,
            unified_spirit_binary: PathBuf::from("/nonexistent/binary"),
            spawn_unified_spirit: true,
        }
    }

    #[test]
    fn build_env_sets_canonical_and_legacy_names() {
        let env = build_unified_spirit_env(&test_cfg());
        assert_eq!(env.get("TITAN_KERNEL_TITAN_ID"), Some(&"T1".to_string()));
        assert_eq!(env.get("TITAN_ID"), Some(&"T1".to_string())); // legacy
        assert_eq!(
            env.get("TITAN_KERNEL_BOOT_GENERATION"),
            Some(&"7".to_string())
        );
        assert_eq!(
            env.get("TITAN_DAEMON_NAME"),
            Some(&"unified-spirit".to_string())
        );
        assert_eq!(
            env.get("TITAN_DAEMON_PARENT_PID"),
            Some(&std::process::id().to_string())
        );
        assert_eq!(
            env.get("TITAN_KERNEL_FASTBUS_PATH"),
            Some(&"/tmp/fastbus.bin".to_string())
        );
        // No authkey / log_level / data_dir → those env vars NOT set in child
        assert!(!env.contains_key("TITAN_AUTHKEY_HEX"));
        assert!(!env.contains_key("TITAN_KERNEL_LOG_LEVEL"));
        assert!(!env.contains_key("TITAN_KERNEL_DATA_DIR"));
    }

    #[test]
    fn build_env_passes_through_authkey_when_present_in_cfg() {
        let mut cfg = test_cfg();
        cfg.authkey_hex = Some("deadbeef".repeat(8));
        cfg.log_level = Some("info".into());
        cfg.data_dir = Some(PathBuf::from("data/"));
        let env = build_unified_spirit_env(&cfg);
        assert_eq!(env.get("TITAN_AUTHKEY_HEX"), Some(&"deadbeef".repeat(8)));
        assert_eq!(env.get("TITAN_KERNEL_LOG_LEVEL"), Some(&"info".to_string()));
        assert_eq!(env.get("TITAN_KERNEL_DATA_DIR"), Some(&"data/".to_string()));
    }

    #[test]
    fn spawn_returns_error_for_missing_binary() {
        let cfg = test_cfg();
        let result = spawn_unified_spirit(&cfg);
        assert!(matches!(result, Err(SuperviseError::SpawnFailed { .. })));
    }

    #[test]
    fn supervise_error_maps_to_child_limit_exit_code() {
        let err = SuperviseError::SpawnFailed {
            binary: "x".into(),
            source: std::io::Error::other("test"),
        };
        assert_eq!(err.to_exit_code() as u8, 6);
    }

    #[test]
    fn supervised_children_default_empty() {
        let s = SupervisedChildren::default();
        assert!(s.unified_spirit.lock().is_none());
    }

    #[test]
    fn sigterm_all_no_op_when_empty() {
        let s = SupervisedChildren::new();
        s.sigterm_all(); // shouldn't panic
    }
}
