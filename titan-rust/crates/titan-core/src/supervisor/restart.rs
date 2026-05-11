//! restart — `Supervisor` struct + restart logic + reason classification.
//!
//! Per SPEC §11.B (one_for_one restart strategy) + §11.G (dependency-aware
//! respawn).
//!
//! # Restart flow (SPEC §11.B step 1-6)
//!
//! When a child crashes:
//! 1. Note crash details (exit, signal, log excerpt)
//! 2. Classify into `SupervisionReason` via [`classify_exit`]
//! 3. Append to per-child rolling reason buffer (last 16)
//! 4. Increment restart counter within rolling intensity window
//! 5. If counter ≤ `SUPERVISION_MAX_RESTARTS`:
//!    - Wait jittered backoff
//!    - Pre-respawn dependency check (§11.G.2)
//!    - If all critical deps OK → spawn replacement
//!    - Else → enter `respawn_blocked` state
//!    - Emit `SUPERVISION_CHILD_RESTARTED`
//! 6. If counter > max → escalation handshake (§11.B.1)
//!
//! C-S5 ships the data structures + decision logic. C-S6 (kernel binary)
//! wires actual `tokio::process::Command::spawn` + signal handling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::Mutex;
use uuid::Uuid;

use crate::supervisor::child::{ChildLifecycle, ChildSpec, ChildState};
use crate::supervisor::dependency::{DepProbe, DependencyCheckOutcome};
use crate::supervisor::escalation::EscalationState;
use crate::supervisor::event::{SupervisionEvent, SupervisionPublisher};
use crate::supervisor::types::{ReasonRecord, SupervisionReason};

use crate::constants::{
    SUPERVISION_DEPENDENCY_BLOCKED_TIMEOUT_S, SUPERVISION_DEPENDENCY_RECHECK_INTERVAL_S,
    SUPERVISION_ESCALATION_TIMEOUT_S, SUPERVISION_INTENSITY_WINDOW_S, SUPERVISION_MAX_RESTARTS,
    SUPERVISION_SUSTAINED_UPTIME_RESET_S,
};

/// Configurable per-supervisor knobs (mostly defaults from SPEC; tests can
/// override).
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// Max restarts within `intensity_window` before escalation fires.
    pub max_restarts: u64,
    /// Rolling window for restart counting.
    pub intensity_window: Duration,
    /// Stable uptime threshold to reset restart counter.
    pub sustained_uptime_reset: Duration,
    /// Max wait for kernel ESCALATION_RESPONSE before defaulting to terminate.
    pub escalation_timeout: Duration,
    /// How often to recheck blocked dependencies in respawn_blocked state.
    pub dependency_recheck_interval: Duration,
    /// Time blocked-respawn waits before escalating to kernel.
    pub dependency_blocked_timeout: Duration,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            max_restarts: SUPERVISION_MAX_RESTARTS,
            intensity_window: Duration::from_secs_f64(SUPERVISION_INTENSITY_WINDOW_S),
            sustained_uptime_reset: Duration::from_secs_f64(SUPERVISION_SUSTAINED_UPTIME_RESET_S),
            escalation_timeout: Duration::from_secs_f64(SUPERVISION_ESCALATION_TIMEOUT_S),
            dependency_recheck_interval: Duration::from_secs_f64(
                SUPERVISION_DEPENDENCY_RECHECK_INTERVAL_S,
            ),
            dependency_blocked_timeout: Duration::from_secs_f64(
                SUPERVISION_DEPENDENCY_BLOCKED_TIMEOUT_S,
            ),
        }
    }
}

/// Supervisor errors.
#[derive(Debug, thiserror::Error)]
pub enum SupervisorError {
    /// Child name not found in registry.
    #[error("unknown child: {0}")]
    UnknownChild(String),
    /// Child already registered.
    #[error("child already registered: {0}")]
    DuplicateChild(String),
}

/// Restart-decision outcome — what the supervisor should do next.
#[derive(Debug, Clone)]
pub enum RestartDecision {
    /// Respawn the child after `delay`.
    Respawn {
        /// Backoff duration before respawn.
        delay: Duration,
    },
    /// Block respawn — critical dependency is down. Recheck after
    /// `recheck_interval`.
    RespawnBlocked {
        /// Name of the blocking dependency.
        blocked_dep: String,
        /// Recheck cadence.
        recheck_interval: Duration,
    },
    /// Restart counter exceeded; escalation request emitted.
    Escalate {
        /// Correlation ID for the escalation.
        escalation_id: Uuid,
    },
    /// `restart_on_crash=false` — log + emit CHILD_DOWN, do not restart.
    NoRestart,
}

/// Top-level supervisor. One per parent in the tree (per SPEC §4.1
/// hierarchy: kernel + substrate + unified-spirit + 6 daemons + Python tree).
pub struct Supervisor {
    /// Supervisor name (e.g. `"kernel"`, `"trinity-substrate"`).
    pub name: String,
    /// Configurable knobs.
    pub config: SupervisorConfig,
    /// Per-child registry.
    children: HashMap<String, ChildState>,
    /// Pending escalations (shared across children).
    escalation_state: EscalationState,
    /// Event publisher.
    publisher: Arc<dyn SupervisionPublisher>,
    /// Dependency probe.
    dep_probe: Arc<dyn DepProbe>,
    /// Internal mutex when used from async — wrap in `Arc<Mutex<>>` outside.
    _lock_guard: parking_lot::Mutex<()>,
}

impl Supervisor {
    /// Construct a new supervisor with default config.
    pub fn new(
        name: impl Into<String>,
        publisher: Arc<dyn SupervisionPublisher>,
        dep_probe: Arc<dyn DepProbe>,
    ) -> Self {
        Self {
            name: name.into(),
            config: SupervisorConfig::default(),
            children: HashMap::new(),
            escalation_state: EscalationState::new(),
            publisher,
            dep_probe,
            _lock_guard: Mutex::new(()),
        }
    }

    /// Register a new child. Fails if name already registered.
    pub fn register_child(&mut self, spec: ChildSpec) -> Result<(), SupervisorError> {
        if self.children.contains_key(&spec.name) {
            return Err(SupervisorError::DuplicateChild(spec.name));
        }
        let name = spec.name.clone();
        self.children.insert(name, ChildState::new(spec));
        Ok(())
    }

    /// Number of registered children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Snapshot of a child's lifecycle (for tests + observability).
    pub fn lifecycle_of(&self, child_name: &str) -> Option<ChildLifecycle> {
        self.children.get(child_name).map(|s| s.lifecycle.clone())
    }

    /// Mark a child as Running (called by kernel after successful spawn).
    pub fn mark_running(&mut self, child_name: &str, pid: u32) -> Result<(), SupervisorError> {
        let state = self
            .children
            .get_mut(child_name)
            .ok_or_else(|| SupervisorError::UnknownChild(child_name.to_string()))?;
        state.lifecycle = ChildLifecycle::Running { pid };
        state.last_uptime_start = Instant::now();
        if state.first_started_at.is_none() {
            state.first_started_at = Some(SystemTime::now());
        }
        Ok(())
    }

    /// Handle a child exit event. Returns the next action the supervisor
    /// should take.
    ///
    /// This is the core of SPEC §11.B steps 1-6 + §11.G.2 pre-respawn check.
    pub fn handle_child_exit(
        &mut self,
        child_name: &str,
        reason: SupervisionReason,
        detail: impl Into<String>,
        exit_code: Option<i32>,
    ) -> Result<RestartDecision, SupervisorError> {
        let now = Instant::now();
        let supervisor_name = self.name.clone();
        let max_restarts = self.config.max_restarts;
        let intensity_window = self.config.intensity_window;
        let escalation_timeout = self.config.escalation_timeout;
        let dependency_recheck_interval = self.config.dependency_recheck_interval;
        let dependency_blocked_timeout = self.config.dependency_blocked_timeout;

        // First borrow: check exists + advance window (no other borrows yet).
        {
            let state = self
                .children
                .get_mut(child_name)
                .ok_or_else(|| SupervisorError::UnknownChild(child_name.to_string()))?;
            state.maybe_advance_window(now, intensity_window);
            state.lifecycle = ChildLifecycle::Dead;
            state.record_reason(ReasonRecord::new(reason, detail, exit_code));
        }

        // Read snapshot fields needed downstream
        let (restart_on_crash, deps, restart_count_before) = {
            let state = self.children.get(child_name).unwrap();
            (
                state.spec.restart_on_crash,
                state.spec.dependencies.clone(),
                state.restart_count_in_window,
            )
        };

        // Emit CHILD_DOWN regardless of restart decision (per SPEC §11.B step 1)
        let down_event = SupervisionEvent::ChildDown {
            child_name: child_name.to_string(),
            supervisor: supervisor_name.clone(),
            reason,
            reason_detail: format!("exit_code={exit_code:?}"),
            restart_count: restart_count_before,
            ts: SystemTime::now(),
        };
        let _ = self.publisher.publish(&down_event);

        // restart_on_crash=false → no restart attempt
        if !restart_on_crash {
            return Ok(RestartDecision::NoRestart);
        }

        // Increment restart counter
        {
            let state = self.children.get_mut(child_name).unwrap();
            state.restart_count_in_window = state.restart_count_in_window.saturating_add(1);
        }

        let (restart_count_after, most_common_reason) = {
            let state = self.children.get(child_name).unwrap();
            (state.restart_count_in_window, state.most_common_reason())
        };

        // Check if we should escalate
        if restart_count_after as u64 > max_restarts {
            let escalation_id = Uuid::new_v4();
            self.escalation_state.register(
                escalation_id,
                child_name.to_string(),
                most_common_reason,
                escalation_timeout,
            );
            // Update child lifecycle
            {
                let state = self.children.get_mut(child_name).unwrap();
                let timeout_at = now + escalation_timeout;
                state.lifecycle = ChildLifecycle::EscalationPending {
                    escalation_id,
                    started_at: now,
                    timeout_at,
                };
            }
            // Emit ESCALATION event
            let reasons_observed: Vec<String> = {
                let state = self.children.get(child_name).unwrap();
                state
                    .reason_buffer
                    .iter()
                    .map(|r| r.reason.as_str().to_string())
                    .collect()
            };
            let last_detail = self
                .children
                .get(child_name)
                .and_then(|s| s.reason_buffer.back())
                .map(|r| r.detail.clone())
                .unwrap_or_default();
            let _ = self.publisher.publish(&SupervisionEvent::Escalation {
                escalation_id,
                child_name: child_name.to_string(),
                supervisor: supervisor_name,
                restart_count: restart_count_after,
                window_s: intensity_window.as_secs_f64(),
                reasons_observed,
                most_common_reason,
                last_reason_detail: last_detail,
                ts: SystemTime::now(),
            });
            return Ok(RestartDecision::Escalate { escalation_id });
        }

        // Pre-respawn dependency check (SPEC §11.G.2)
        for dep in &deps {
            if matches!(
                dep.severity,
                crate::supervisor::types::DependencySeverity::Critical
            ) {
                let outcome = self.dep_probe.check(dep);
                if let DependencyCheckOutcome::Down {
                    reason: _dep_reason,
                } = outcome
                {
                    // Block respawn
                    let escalation_at = now + dependency_blocked_timeout;
                    {
                        let state = self.children.get_mut(child_name).unwrap();
                        state.lifecycle = ChildLifecycle::RespawnBlocked {
                            since: now,
                            blocked_dep: dep.name.clone(),
                            escalation_at,
                        };
                    }
                    let _ = self
                        .publisher
                        .publish(&SupervisionEvent::DependencyBlocked {
                            child_name: child_name.to_string(),
                            supervisor: self.name.clone(),
                            blocked_dependency: dep.name.clone(),
                            dependency_kind: dep.kind.as_str().to_string(),
                            severity: "critical".to_string(),
                            since_ts: SystemTime::now(),
                            recheck_interval_s: dependency_recheck_interval.as_secs_f64(),
                            escalation_at_ts: SystemTime::now() + dependency_blocked_timeout,
                        });
                    return Ok(RestartDecision::RespawnBlocked {
                        blocked_dep: dep.name.clone(),
                        recheck_interval: dependency_recheck_interval,
                    });
                }
            }
        }

        // All critical deps OK → respawn with backoff
        let backoff = crate::supervisor::backoff::compute_backoff(restart_count_after);
        Ok(RestartDecision::Respawn { delay: backoff })
    }

    /// Recheck a blocked child's dependencies. If all critical deps now OK,
    /// transitions out of `RespawnBlocked` and returns `Some(decision)`
    /// (typically `RestartDecision::Respawn`). If still blocked, returns
    /// `None` (caller schedules another recheck).
    pub fn recheck_blocked_child(
        &mut self,
        child_name: &str,
    ) -> Result<Option<RestartDecision>, SupervisorError> {
        let now = Instant::now();
        let supervisor_name = self.name.clone();

        let (deps, was_blocked_since, _was_escalation_at) = {
            let state = self
                .children
                .get(child_name)
                .ok_or_else(|| SupervisorError::UnknownChild(child_name.to_string()))?;
            match &state.lifecycle {
                ChildLifecycle::RespawnBlocked {
                    since,
                    escalation_at,
                    ..
                } => (state.spec.dependencies.clone(), *since, *escalation_at),
                _ => return Ok(None), // not blocked
            }
        };

        // Check all critical deps
        for dep in &deps {
            if matches!(
                dep.severity,
                crate::supervisor::types::DependencySeverity::Critical
            ) {
                let outcome = self.dep_probe.check(dep);
                if let DependencyCheckOutcome::Down { .. } = outcome {
                    // Still blocked
                    return Ok(None);
                }
            }
        }

        // All recovered — transition out + emit DependencyRecovered + restart
        let blocked_for_s = now
            .saturating_duration_since(was_blocked_since)
            .as_secs_f64();
        let _ = self
            .publisher
            .publish(&SupervisionEvent::DependencyRecovered {
                child_name: child_name.to_string(),
                supervisor: supervisor_name,
                dependency_name: "all_critical".to_string(),
                restored_at_ts: SystemTime::now(),
                total_blocked_s: blocked_for_s,
            });

        let restart_count = {
            let state = self.children.get(child_name).unwrap();
            state.restart_count_in_window
        };
        let backoff = crate::supervisor::backoff::compute_backoff(restart_count);
        Ok(Some(RestartDecision::Respawn { delay: backoff }))
    }
}

/// Classify a process exit into a `SupervisionReason`.
///
/// Per SPEC §11.B step 2 + §15 exit codes:
/// - `0` → `CleanExit`
/// - `1` → `Panic` (generic; Rust panic / Python uncaught)
/// - `2` → `ConfigError` (invalid TOML, missing env)
/// - `3` → `ConfigError` (identity load failure → config-class)
/// - `4` → `ConfigError` (bus bind failure)
/// - `5` → `ConfigError` (shm create failure)
/// - `6` → `ConfigError` (child process limit)
/// - `7` → `Other` (parent died — log only)
/// - `8` → `Other` (adoption rejected — restart fresh)
/// - `64-127` → `Other` (escalation range)
/// - `128 + N` (signal) → `Killed` if `N=9` (SIGKILL); `Panic` if `N=11`
///   (SIGSEGV/SIGBUS/SIGABRT); `CleanExit` if `N=15` (SIGTERM expected);
///   `Other` otherwise
pub fn classify_exit(exit_code: i32) -> SupervisionReason {
    match exit_code {
        0 => SupervisionReason::CleanExit,
        1 => SupervisionReason::Panic,
        2..=6 => SupervisionReason::ConfigError,
        7 => SupervisionReason::Other,
        8 => SupervisionReason::Other,
        137 => SupervisionReason::Killed,    // 128 + 9 = SIGKILL
        139 => SupervisionReason::Segv,      // 128 + 11 = SIGSEGV
        134 => SupervisionReason::Segv,      // 128 + 6 = SIGABRT
        135 => SupervisionReason::Segv,      // 128 + 7 = SIGBUS
        143 => SupervisionReason::CleanExit, // 128 + 15 = SIGTERM (graceful)
        64..=127 => SupervisionReason::Other,
        _ => SupervisionReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::dependency::{Dependency, MockDepProbe};
    use crate::supervisor::event::RecordingPublisher;

    fn make_supervisor() -> (Arc<RecordingPublisher>, Arc<MockDepProbe>, Supervisor) {
        let pub_ = Arc::new(RecordingPublisher::new());
        let probe = Arc::new(MockDepProbe::new());
        let pub_dyn: Arc<dyn SupervisionPublisher> = pub_.clone();
        let probe_dyn: Arc<dyn DepProbe> = probe.clone();
        (
            pub_,
            probe,
            Supervisor::new("test_supervisor", pub_dyn, probe_dyn),
        )
    }

    // ── classify_exit tests ───────────────────────────────────────────

    #[test]
    fn classify_clean_exit() {
        assert_eq!(classify_exit(0), SupervisionReason::CleanExit);
    }

    #[test]
    fn classify_generic_panic() {
        assert_eq!(classify_exit(1), SupervisionReason::Panic);
    }

    #[test]
    fn classify_config_errors_2_through_6() {
        for code in 2..=6 {
            assert_eq!(classify_exit(code), SupervisionReason::ConfigError);
        }
    }

    #[test]
    fn classify_sigkill_137() {
        assert_eq!(classify_exit(137), SupervisionReason::Killed);
    }

    #[test]
    fn classify_sigsegv_139() {
        assert_eq!(classify_exit(139), SupervisionReason::Segv);
    }

    #[test]
    fn classify_sigterm_143_clean() {
        assert_eq!(classify_exit(143), SupervisionReason::CleanExit);
    }

    #[test]
    fn classify_other_signals_other() {
        assert_eq!(classify_exit(200), SupervisionReason::Other);
    }

    // ── Supervisor tests ──────────────────────────────────────────────

    #[test]
    fn register_child_and_count() {
        let (_, _, mut sup) = make_supervisor();
        sup.register_child(ChildSpec::new("a")).unwrap();
        sup.register_child(ChildSpec::new("b")).unwrap();
        assert_eq!(sup.child_count(), 2);
    }

    #[test]
    fn duplicate_child_rejected() {
        let (_, _, mut sup) = make_supervisor();
        sup.register_child(ChildSpec::new("a")).unwrap();
        let err = sup.register_child(ChildSpec::new("a")).unwrap_err();
        assert!(matches!(err, SupervisorError::DuplicateChild(_)));
    }

    #[test]
    fn mark_running_updates_lifecycle() {
        let (_, _, mut sup) = make_supervisor();
        sup.register_child(ChildSpec::new("a")).unwrap();
        sup.mark_running("a", 1234).unwrap();
        assert_eq!(
            sup.lifecycle_of("a").unwrap(),
            ChildLifecycle::Running { pid: 1234 }
        );
    }

    #[test]
    fn child_exit_first_time_returns_respawn() {
        let (pub_, _, mut sup) = make_supervisor();
        sup.register_child(ChildSpec::new("a")).unwrap();
        sup.mark_running("a", 100).unwrap();

        let decision = sup
            .handle_child_exit("a", SupervisionReason::Panic, "test", Some(1))
            .unwrap();
        match decision {
            RestartDecision::Respawn { delay } => {
                // 100 → 200ms band ±25% → roughly [75ms, 125ms] for restart_count=1
                assert!(delay.as_millis() >= 75 && delay.as_millis() <= 200);
            }
            other => panic!("expected Respawn, got {other:?}"),
        }
        // Should have emitted CHILD_DOWN
        assert_eq!(pub_.count_of("SUPERVISION_CHILD_DOWN"), 1);
    }

    #[test]
    fn child_exit_restart_on_crash_false_returns_no_restart() {
        let (pub_, _, mut sup) = make_supervisor();
        let mut spec = ChildSpec::new("a");
        spec.restart_on_crash = false;
        sup.register_child(spec).unwrap();
        sup.mark_running("a", 100).unwrap();

        let decision = sup
            .handle_child_exit("a", SupervisionReason::Panic, "test", Some(1))
            .unwrap();
        assert!(matches!(decision, RestartDecision::NoRestart));
        // CHILD_DOWN still emitted
        assert_eq!(pub_.count_of("SUPERVISION_CHILD_DOWN"), 1);
    }

    #[test]
    fn child_exit_with_blocked_dep_returns_blocked() {
        let (pub_, probe, mut sup) = make_supervisor();
        let dep = Dependency::critical_module("memory_module");
        let spec = ChildSpec::new("social").with_dependency(dep);
        sup.register_child(spec).unwrap();
        sup.mark_running("social", 100).unwrap();

        // Mark dep as down
        probe.down("memory_module", "process not running");

        let decision = sup
            .handle_child_exit("social", SupervisionReason::Panic, "test", Some(1))
            .unwrap();
        match decision {
            RestartDecision::RespawnBlocked { blocked_dep, .. } => {
                assert_eq!(blocked_dep, "memory_module");
            }
            other => panic!("expected RespawnBlocked, got {other:?}"),
        }

        // Should have emitted CHILD_DOWN + DEPENDENCY_BLOCKED
        assert_eq!(pub_.count_of("SUPERVISION_CHILD_DOWN"), 1);
        assert_eq!(pub_.count_of("SUPERVISION_DEPENDENCY_BLOCKED"), 1);
    }

    #[test]
    fn six_crashes_in_window_triggers_escalation() {
        let (pub_, _, mut sup) = make_supervisor();
        sup.register_child(ChildSpec::new("a")).unwrap();
        sup.mark_running("a", 100).unwrap();

        // 5 restarts allowed; 6th triggers escalation
        for i in 1..=6 {
            let decision = sup
                .handle_child_exit("a", SupervisionReason::Panic, format!("crash {i}"), Some(1))
                .unwrap();
            if i <= 5 {
                assert!(matches!(decision, RestartDecision::Respawn { .. }));
            } else {
                assert!(matches!(decision, RestartDecision::Escalate { .. }));
            }
        }
        assert_eq!(pub_.count_of("SUPERVISION_CHILD_DOWN"), 6);
        assert_eq!(pub_.count_of("SUPERVISION_ESCALATION"), 1);
    }

    #[test]
    fn recheck_blocked_child_restarts_when_dep_recovers() {
        let (pub_, probe, mut sup) = make_supervisor();
        let dep = Dependency::critical_module("memory_module");
        sup.register_child(ChildSpec::new("social").with_dependency(dep))
            .unwrap();
        sup.mark_running("social", 100).unwrap();

        probe.down("memory_module", "down");
        sup.handle_child_exit("social", SupervisionReason::Panic, "test", Some(1))
            .unwrap();

        // Dep still down → recheck returns None
        assert!(sup.recheck_blocked_child("social").unwrap().is_none());

        // Dep recovers → recheck returns Some(Respawn)
        probe.healthy("memory_module");
        let decision = sup.recheck_blocked_child("social").unwrap();
        assert!(matches!(decision, Some(RestartDecision::Respawn { .. })));
        assert_eq!(pub_.count_of("SUPERVISION_DEPENDENCY_RECOVERED"), 1);
    }

    #[test]
    fn unknown_child_returns_unknown_child_error() {
        let (_, _, mut sup) = make_supervisor();
        let err = sup
            .handle_child_exit("ghost", SupervisionReason::Panic, "test", Some(1))
            .unwrap_err();
        assert!(matches!(err, SupervisorError::UnknownChild(_)));
    }
}
