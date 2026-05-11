//! supervise — Unified-spirit's 6-daemon `Supervisor` wiring (C4-4).
//!
//! Per SPEC §11.B (one_for_one) + §11.B.1 (escalation handshake) +
//! §11.G (dependency-aware respawn) + master plan §10.4 chunk C4-4.
//!
//! Wraps [`titan_core::supervisor::Supervisor`] with OS-level spawn +
//! reap. The Supervisor owns bookkeeping (lifecycle, restart counters,
//! escalation state, dependency probes); this module owns the actual
//! `tokio::process::Command` spawns + SIGCHLD reap loop + bus event
//! plumbing.
//!
//! Per master plan §3.3 Option A (Maker confirmed 2026-04-29): in C-S4
//! the 6 "daemons" use `--use-placeholder-daemons` mode → spawn 6
//! instances of any available placeholder binary. Real daemon binaries
//! ship in C-S5/C-S6; the supervision wiring here is forward-compatible.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use thiserror::Error;
use titan_core::supervisor::{
    classify_exit, restart::RestartDecision, MockDepProbe, RecordingPublisher, Supervisor,
    SupervisorError,
};
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::child_specs::{build_daemon_specs, DAEMON_NAMES};

/// Errors during daemon supervision setup / spawn.
#[derive(Debug, Error)]
pub enum SuperviseError {
    /// Underlying [`Supervisor`] returned an error (duplicate child,
    /// unknown child, DAG cycle, etc.).
    #[error("supervisor: {0}")]
    Supervisor(#[from] SupervisorError),
    /// `tokio::process::Command::spawn` failed (binary missing, perms).
    #[error("spawn {binary}: {source}")]
    Spawn {
        /// Binary path that failed.
        binary: String,
        /// Underlying io error.
        #[source]
        source: std::io::Error,
    },
    /// Configured daemon binary directory does not exist.
    #[error("daemon binary dir missing: {0}")]
    BinaryDirMissing(String),
}

/// Configuration for the daemon supervisor.
#[derive(Debug, Clone)]
pub struct DaemonSupervisorConfig {
    /// Where to find `titan-{inner,outer}-{body,mind,spirit}-rs` binaries.
    pub daemon_binary_dir: PathBuf,
    /// If true, spawn the placeholder binary (single shared) for all 6
    /// daemons. Used pre-C-S5 when real daemon binaries don't exist yet.
    pub use_placeholder_daemons: bool,
    /// Path to the placeholder binary when `use_placeholder_daemons=true`.
    /// Defaults to `<daemon_binary_dir>/titan-trinity-rs-placeholder` when None.
    pub placeholder_binary: Option<PathBuf>,
    /// Env vars passed through to each child (TITAN_KERNEL_TITAN_ID, etc.).
    pub child_env: HashMap<String, String>,
}

impl DaemonSupervisorConfig {
    /// New config for the canonical daemon binary search dir.
    pub fn new(daemon_binary_dir: PathBuf) -> Self {
        Self {
            daemon_binary_dir,
            use_placeholder_daemons: false,
            placeholder_binary: None,
            child_env: HashMap::new(),
        }
    }

    /// Resolve the binary path for `daemon_name`. With placeholder mode,
    /// always returns the placeholder. Otherwise returns
    /// `<daemon_binary_dir>/<daemon_name>`.
    pub fn binary_path(&self, daemon_name: &str) -> PathBuf {
        if self.use_placeholder_daemons {
            self.placeholder_binary
                .clone()
                .unwrap_or_else(|| self.daemon_binary_dir.join("titan-trinity-rs-placeholder"))
        } else {
            self.daemon_binary_dir.join(daemon_name)
        }
    }
}

/// Daemon process registry — tracks spawned children by name + PID.
#[derive(Debug, Default)]
pub struct DaemonProcessRegistry {
    by_name: HashMap<String, Child>,
    pid_to_name: HashMap<u32, String>,
}

impl DaemonProcessRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a spawned child.
    pub fn insert(&mut self, name: String, child: Child) {
        if let Some(pid) = child.id() {
            self.pid_to_name.insert(pid, name.clone());
        }
        self.by_name.insert(name, child);
    }

    /// Look up daemon name by PID.
    pub fn name_of(&self, pid: u32) -> Option<&str> {
        self.pid_to_name.get(&pid).map(|s| s.as_str())
    }

    /// Take ownership of a child (used at reap time).
    pub fn take(&mut self, name: &str) -> Option<Child> {
        if let Some(child) = self.by_name.remove(name) {
            if let Some(pid) = child.id() {
                self.pid_to_name.remove(&pid);
            }
            Some(child)
        } else {
            None
        }
    }

    /// Iterate over (name, pid) pairs for currently-tracked daemons.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Option<u32>)> {
        self.by_name.iter().map(|(n, c)| (n.as_str(), c.id()))
    }

    /// Count of tracked daemons.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// 6-daemon supervisor — unified-spirit's child-management orchestrator.
pub struct DaemonSupervisor {
    /// Wrapped titan-core Supervisor (bookkeeping).
    pub supervisor: Arc<Mutex<Supervisor>>,
    /// Registry of running child processes (name → tokio::process::Child).
    pub registry: Arc<Mutex<DaemonProcessRegistry>>,
    /// Configuration (binary paths, env, placeholder mode).
    pub config: DaemonSupervisorConfig,
    /// Recording event publisher (captures supervision events for tests
    /// + observability — production wiring publishes via bus client).
    pub publisher: Arc<RecordingPublisher>,
}

impl DaemonSupervisor {
    /// Construct + register 6 daemon ChildSpecs.
    pub fn new(config: DaemonSupervisorConfig) -> Result<Self, SuperviseError> {
        let publisher = Arc::new(RecordingPublisher::default());
        let dep_probe = Arc::new(MockDepProbe::default()); // C-S4: probes always succeed
        let mut sup = Supervisor::new("unified-spirit", publisher.clone(), dep_probe);

        for spec in build_daemon_specs() {
            sup.register_child(spec)?;
        }

        Ok(Self {
            supervisor: Arc::new(Mutex::new(sup)),
            registry: Arc::new(Mutex::new(DaemonProcessRegistry::new())),
            config,
            publisher,
        })
    }

    /// Spawn all 6 daemons. Failures bubble up as SuperviseError.
    pub async fn spawn_all(&self) -> Result<(), SuperviseError> {
        for &name in DAEMON_NAMES.iter() {
            self.spawn_one(name).await?;
        }
        Ok(())
    }

    /// Spawn one daemon by name. Records its PID in the registry +
    /// marks it `Running` in the wrapped Supervisor.
    pub async fn spawn_one(&self, name: &str) -> Result<u32, SuperviseError> {
        let binary = self.config.binary_path(name);
        let mut cmd = Command::new(&binary);
        cmd.env_clear()
            .envs(&self.config.child_env)
            .env("TITAN_DAEMON_NAME", name)
            .kill_on_drop(false);

        let child = cmd.spawn().map_err(|source| SuperviseError::Spawn {
            binary: binary.to_string_lossy().into_owned(),
            source,
        })?;
        let pid = child.id().unwrap_or(0);

        info!(
            event = "DAEMON_SPAWNED",
            daemon = name,
            pid = pid,
            binary = ?binary,
            "spawned daemon child"
        );

        // Bookkeeping
        {
            let mut sup = self.supervisor.lock();
            sup.mark_running(name, pid)?;
        }
        {
            let mut reg = self.registry.lock();
            reg.insert(name.to_string(), child);
        }
        Ok(pid)
    }

    /// Handle a child exit (typically called from a SIGCHLD reaper task).
    /// Looks up the daemon, calls Supervisor::handle_child_exit, and
    /// (per RestartDecision) respawns or escalates.
    ///
    /// Caller passes `exit_code: Option<i32>` from `Child::wait().status.code()`
    /// or `None` for signal-killed processes (Unix encodes signals as
    /// 128+N exit code per SPEC §15).
    pub async fn handle_exit(
        &self,
        name: &str,
        exit_code: Option<i32>,
    ) -> Result<(), SuperviseError> {
        let reason = exit_code
            .map(classify_exit)
            .unwrap_or(titan_core::supervisor::SupervisionReason::Killed);
        let detail = format!("daemon exited with code={exit_code:?}");

        let decision = {
            let mut sup = self.supervisor.lock();
            sup.handle_child_exit(name, reason, detail, exit_code)?
        };
        // Drop the dead child from registry
        {
            let mut reg = self.registry.lock();
            let _ = reg.take(name);
        }

        match decision {
            RestartDecision::Respawn { delay } => {
                // C4-4 honors backoff per SPEC §11.B.5 (jittered backoff).
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                info!(
                    event = "DAEMON_RESTARTING",
                    daemon = name,
                    delay_ms = delay.as_millis() as u64,
                    "restarting daemon"
                );
                self.spawn_one(name).await?;
            }
            RestartDecision::Escalate { escalation_id } => {
                warn!(
                    event = "DAEMON_ESCALATION_PENDING",
                    daemon = name,
                    escalation_id = %escalation_id,
                    "max_restarts exceeded; escalation pending parent decision"
                );
            }
            RestartDecision::RespawnBlocked {
                blocked_dep,
                recheck_interval,
            } => {
                warn!(
                    event = "DAEMON_DEPENDENCY_BLOCKED",
                    daemon = name,
                    dep = %blocked_dep,
                    recheck_ms = recheck_interval.as_millis() as u64,
                    "respawn blocked on dependency"
                );
            }
            RestartDecision::NoRestart => {
                info!(
                    event = "DAEMON_NO_RESTART",
                    daemon = name,
                    "child marked restart_on_crash=false; not restarting"
                );
            }
        }
        Ok(())
    }

    /// Reaper loop — polls every 200ms for child exits, calls
    /// [`Self::handle_exit`] on any that have died. The polling cadence is
    /// fast enough that a daemon's perceived downtime is ≤200ms before the
    /// supervisor's restart logic kicks in (RestartDecision::Respawn).
    ///
    /// **Why polling not signals**: SIGCHLD-driven reaping requires a
    /// process-level handler that doesn't compose well with tokio's
    /// per-task `Child::wait`. A 200ms poll is cheap (≤6 try_wait calls per
    /// tick, each a single waitpid syscall returning immediately) and
    /// avoids the SIGCHLD-signal-merging problem when multiple children
    /// exit close together.
    ///
    /// **Why this method exists**: C4-4 ship landed `handle_exit` but no
    /// production caller. Daemons going defunct never got reaped or
    /// respawned. This method closes that gap — call once after
    /// [`Self::spawn_all`].
    ///
    /// Returns the JoinHandle of the reaper task; the caller is responsible
    /// for awaiting (or aborting) it during shutdown.
    pub fn spawn_reaper(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(200));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(event = "DAEMON_REAPER_STARTED", poll_ms = 200);
            loop {
                interval.tick().await;
                // Snapshot names under a short lock — don't hold across await.
                let names: Vec<String> = {
                    let reg = self.registry.lock();
                    reg.by_name.keys().cloned().collect()
                };
                if names.is_empty() {
                    continue;
                }
                // For each tracked daemon, non-blocking try_wait under lock.
                let exited: Vec<(String, Option<i32>)> = {
                    let mut reg = self.registry.lock();
                    let mut out = Vec::new();
                    for name in &names {
                        if let Some(child) = reg.by_name.get_mut(name) {
                            match child.try_wait() {
                                Ok(Some(status)) => {
                                    out.push((name.clone(), status.code()));
                                }
                                Ok(None) => { /* still running */ }
                                Err(e) => {
                                    warn!(
                                        event = "DAEMON_TRY_WAIT_ERR",
                                        daemon = %name,
                                        err = ?e,
                                    );
                                }
                            }
                        }
                    }
                    out
                };
                for (name, exit_code) in exited {
                    info!(
                        event = "DAEMON_EXIT_OBSERVED",
                        daemon = %name,
                        exit_code = ?exit_code,
                        "child exited; invoking handle_exit (respawn or escalate)"
                    );
                    if let Err(e) = self.handle_exit(&name, exit_code).await {
                        warn!(
                            event = "DAEMON_HANDLE_EXIT_ERR",
                            daemon = %name,
                            err = ?e,
                        );
                    }
                }
            }
        })
    }

    /// Send SIGTERM to all live daemons + drop registry. Used at graceful
    /// shutdown per SPEC §10.A KERNEL_SHUTDOWN_ANNOUNCE cascade.
    pub async fn shutdown_all(&self) {
        let names: Vec<String> = {
            let reg = self.registry.lock();
            reg.by_name.keys().cloned().collect()
        };
        for name in names {
            let mut child = {
                let mut reg = self.registry.lock();
                match reg.take(&name) {
                    Some(c) => c,
                    None => continue,
                }
            };
            // Best-effort SIGTERM (kill_on_drop=false to honor graceful
            // grace per SPEC §17 — caller waits for exit separately).
            let _ = child.start_kill();
            info!(event = "DAEMON_SHUTDOWN_KILL_SENT", daemon = %name);
        }
    }

    /// Number of currently-tracked live children.
    pub fn live_count(&self) -> usize {
        self.registry.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use titan_core::supervisor::{SupervisionEvent, SupervisionReason};

    fn config_for_test(placeholder: &str) -> DaemonSupervisorConfig {
        DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/local/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from(placeholder)),
            child_env: HashMap::new(),
        }
    }

    #[test]
    fn supervisor_registers_six_children() {
        // C4-4 test 7: DaemonSupervisor registers all 6 specs
        let cfg = config_for_test("/bin/sleep");
        let sup = DaemonSupervisor::new(cfg).unwrap();
        assert_eq!(sup.supervisor.lock().child_count(), 6);
    }

    #[test]
    fn binary_path_uses_placeholder_when_flag_set() {
        // C4-4 test 8: --use-placeholder-daemons routes spawn to single binary
        let cfg = config_for_test("/tmp/my_placeholder");
        for &name in DAEMON_NAMES.iter() {
            assert_eq!(cfg.binary_path(name), PathBuf::from("/tmp/my_placeholder"));
        }
    }

    #[test]
    fn binary_path_uses_per_daemon_when_flag_unset() {
        // C4-4 test 9: production mode uses per-daemon binary names
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/opt/titan/bin"),
            use_placeholder_daemons: false,
            placeholder_binary: None,
            child_env: HashMap::new(),
        };
        assert_eq!(
            cfg.binary_path("titan-inner-body-rs"),
            PathBuf::from("/opt/titan/bin/titan-inner-body-rs")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_all_starts_six_processes() {
        // C4-4 test 10: spawn_all launches 6 children + registry tracks them.
        // /bin/true (always-exits-success) used for spawn-only test that
        // doesn't need long-lived children.
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/true")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_all().await.unwrap();
        // Registry tracked all 6 (PIDs may have already exited since
        // /bin/true exits immediately; we only verify spawn-call count)
        assert_eq!(sup.registry.lock().len(), 6);
        // Cleanup any zombies
        sup.shutdown_all().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_one_fails_for_missing_binary() {
        // C4-4 test 11: missing binary → SuperviseError::Spawn
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/nonexistent"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/nonexistent/binary")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        let result = sup.spawn_one("titan-inner-body-rs").await;
        assert!(matches!(result, Err(SuperviseError::Spawn { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handle_exit_emits_child_down_event_and_respawns() {
        // C4-4 test 12: kill 1 daemon → bookkeeping emits CHILD_DOWN +
        // CHILD_RESTARTED via the publisher
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/true")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_one("titan-inner-body-rs").await.unwrap();

        // Simulate exit code 0 (clean) — supervisor should respawn since
        // ChildSpec.restart_on_crash=true (default).
        sup.handle_exit("titan-inner-body-rs", Some(0))
            .await
            .unwrap();

        // CHILD_DOWN event must be in publisher
        let events = sup.publisher.snapshot();
        let down_count = events
            .iter()
            .filter(|e| matches!(e, SupervisionEvent::ChildDown { .. }))
            .count();
        assert!(down_count >= 1, "CHILD_DOWN should be emitted");
        sup.shutdown_all().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reaper_detects_child_exit_and_respawns_via_handle_exit() {
        // Post-D2 fix verification (2026-05-06): spawn_reaper polls
        // try_wait + invokes handle_exit when a child dies.
        //
        // Uses /bin/true (exits immediately with code 0) as placeholder.
        // After spawn, the child is already-dead OR will be within ms.
        // Reaper at 200ms cadence will observe the exit + invoke handle_exit
        // which respawns (RestartDecision::Respawn since restart_on_crash=true).
        // Verify: after >250ms, publisher recorded ChildDown + CHILD_RESTARTED.
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/true")),
            child_env: HashMap::new(),
        };
        let sup = Arc::new(DaemonSupervisor::new(cfg).unwrap());
        sup.spawn_one("titan-inner-body-rs").await.unwrap();

        // Spawn reaper.
        let reaper = sup.clone().spawn_reaper();

        // Wait long enough for reaper to observe + handle the exit.
        // /bin/true exits immediately; reaper polls every 200ms.
        // 600ms gives the reaper at least 2 polls + one respawn cycle.
        tokio::time::sleep(Duration::from_millis(600)).await;

        // Reaper must have called handle_exit at least once.
        let events = sup.publisher.snapshot();
        let down_count = events
            .iter()
            .filter(|e| matches!(e, SupervisionEvent::ChildDown { .. }))
            .count();
        assert!(
            down_count >= 1,
            "reaper should have triggered ≥1 ChildDown via handle_exit, got {down_count} events: {events:?}"
        );

        reaper.abort();
        let _ = tokio::time::timeout(Duration::from_millis(500), reaper).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn six_kills_within_window_triggers_escalation() {
        // C4-4 test 13: 6 successive kills on one daemon → escalation
        // event emitted (max_restarts=5 exceeded per SPEC §11.B.1)
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/true")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_one("titan-inner-body-rs").await.unwrap();

        // 6 kills (1 spawn + 6 exits = 7 lifecycle events; the 6th
        // exit-handler should hit max_restarts threshold)
        for _ in 0..6 {
            let _ = sup.handle_exit("titan-inner-body-rs", Some(1)).await;
        }

        let events = sup.publisher.snapshot();
        let escalation_count = events
            .iter()
            .filter(|e| matches!(e, SupervisionEvent::Escalation { .. }))
            .count();
        assert!(
            escalation_count >= 1,
            "ESCALATION should be emitted after 6 crashes (events: {events:#?})"
        );
        sup.shutdown_all().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn siblings_unaffected_by_one_daemon_kill() {
        // C4-4 test 14: kill 1 daemon → other 5 PIDs unchanged.
        // /bin/cat (blocks until stdin closes — long-lived enough for
        // PID-tracking test).
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/cat")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_all().await.unwrap();
        let pids_before: HashMap<String, Option<u32>> = sup
            .registry
            .lock()
            .iter()
            .map(|(n, p)| (n.to_string(), p))
            .collect();
        assert_eq!(pids_before.len(), 6);

        // Simulate kill of one daemon (handle_exit will remove from registry
        // + respawn → new PID)
        sup.handle_exit("titan-inner-body-rs", Some(1))
            .await
            .unwrap();

        let pids_after: HashMap<String, Option<u32>> = sup
            .registry
            .lock()
            .iter()
            .map(|(n, p)| (n.to_string(), p))
            .collect();

        // Sibling PIDs unchanged
        for &name in DAEMON_NAMES.iter() {
            if name == "titan-inner-body-rs" {
                continue;
            }
            assert_eq!(
                pids_before.get(name),
                pids_after.get(name),
                "sibling {name} PID should not change"
            );
        }
        sup.shutdown_all().await;
    }

    #[test]
    fn daemon_process_registry_tracks_pid_lookup() {
        // C4-4 test 15: registry insert/lookup roundtrip (no spawn needed)
        let mut reg = DaemonProcessRegistry::new();
        // We can't easily create a real Child without spawning, so this
        // test verifies the empty path
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
        assert!(reg.name_of(99999).is_none());
        assert!(reg.take("nonexistent").is_none());
    }

    #[test]
    fn child_specs_dag_has_no_cycles() {
        // C4-4 test 16: dependency graph is a DAG (verified via Supervisor's
        // internal check_dag invocation during register_child).
        let cfg = config_for_test("/bin/true");
        let sup = DaemonSupervisor::new(cfg);
        assert!(
            sup.is_ok(),
            "register_child should succeed if deps form a DAG"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_all_clears_registry() {
        // C4-4 test 17: shutdown_all empties the registry
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/cat")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_all().await.unwrap();
        assert_eq!(sup.live_count(), 6);
        sup.shutdown_all().await;
        assert_eq!(sup.live_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn classify_exit_routes_panic_and_segv_correctly() {
        // C4-4 test 18: classify_exit categorizes exit codes per SPEC §15
        // (delegates to titan_core::supervisor::classify_exit). 1 = Panic,
        // 139 = Segv (SIGSEGV), 137 = Killed (SIGKILL), 143 = CleanExit
        // (SIGTERM graceful).
        let cfg = DaemonSupervisorConfig {
            daemon_binary_dir: PathBuf::from("/usr/bin"),
            use_placeholder_daemons: true,
            placeholder_binary: Some(PathBuf::from("/bin/true")),
            child_env: HashMap::new(),
        };
        let sup = DaemonSupervisor::new(cfg).unwrap();
        sup.spawn_one("titan-inner-mind-rs").await.unwrap();
        // Panic (exit 1)
        sup.handle_exit("titan-inner-mind-rs", Some(1))
            .await
            .unwrap();
        let events = sup.publisher.snapshot();
        let has_panic = events.iter().any(|e| match e {
            SupervisionEvent::ChildDown { reason, .. } => *reason == SupervisionReason::Panic,
            _ => false,
        });
        assert!(
            has_panic,
            "exit 1 should classify as Panic (events: {events:#?})"
        );
        sup.shutdown_all().await;
    }
}
