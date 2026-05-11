//! child — `ChildSpec` (declarative) + `ChildState` (runtime) + `ChildLifecycle`.
//!
//! Per SPEC §11.B + §11.G: every supervised child has:
//! - A `ChildSpec` declared at compile/boot time (name, dependencies,
//!   restart policy, RSS limit).
//! - A `ChildState` tracked at runtime (lifecycle, restart count, reason
//!   buffer, current PID).
//!
//! For Phase C C-S5 we ship the data structures + state-transition logic
//! WITHOUT actually spawning OS processes — that integration ships in C2-6
//! (kernel binary). This keeps C2-5 unit-testable.

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use crate::supervisor::dependency::Dependency;
use crate::supervisor::types::{ReasonRecord, SupervisionReason};

/// Maximum entries in a per-child rolling reason buffer per SPEC §11.B step 3.
pub const REASON_BUFFER_SIZE: usize = 16;

/// Lifecycle state of a supervised child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildLifecycle {
    /// Child is up and running.
    Running {
        /// Process ID.
        pid: u32,
    },
    /// Child crashed; supervisor refused to respawn because a critical
    /// dependency is down (SPEC §11.G.2 step 4).
    RespawnBlocked {
        /// When the block began.
        since: Instant,
        /// Name of the blocking dependency.
        blocked_dep: String,
        /// When escalation will fire if dep stays blocked.
        escalation_at: Instant,
    },
    /// Child crashed; supervisor exhausted intensity-window restart budget
    /// and emitted `SUPERVISION_ESCALATION`. Awaiting kernel decision.
    EscalationPending {
        /// Correlation ID for the escalation request.
        escalation_id: uuid::Uuid,
        /// When escalation was emitted.
        started_at: Instant,
        /// When the supervisor will default-to-`terminate` if no response.
        timeout_at: Instant,
    },
    /// Kernel decided `halt`; child stays dead until Maker intervention.
    Halted,
    /// Child has never been started OR was halted permanently.
    Dead,
}

impl ChildLifecycle {
    /// Returns `true` if the child is in a terminal state (no more
    /// auto-restart attempts will happen without external intervention).
    pub fn is_terminal(&self) -> bool {
        matches!(self, ChildLifecycle::Halted | ChildLifecycle::Dead)
    }

    /// Returns `true` if the child is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self, ChildLifecycle::Running { .. })
    }
}

/// Declarative child specification — built at boot or from `ModuleSpec`.
#[derive(Debug, Clone)]
pub struct ChildSpec {
    /// Canonical child name (matches SPEC §1 glossary if applicable).
    pub name: String,
    /// Dependencies (per SPEC §11.G).
    pub dependencies: Vec<Dependency>,
    /// `false` = supervisor logs death + emits `CHILD_DOWN` but does NOT
    /// auto-restart (per SPEC §11.E "Restart-disabled modules").
    pub restart_on_crash: bool,
    /// Per-module RSS limit override; `None` = use
    /// `MODULE_DEFAULT_RSS_LIMIT_MB`.
    pub rss_limit_mb: Option<u32>,
    /// Per-module heartbeat timeout override; `None` = default
    /// `MODULE_HEARTBEAT_TIMEOUT_S`.
    pub heartbeat_timeout_s: Option<f32>,
    /// `true` → must go through SPEC §11.H.3 DB checkpoint sequence on
    /// shutdown; supervisor never SIGKILLs without forced-kill log entry.
    pub critical_data_writer: bool,
}

impl ChildSpec {
    /// Construct a default child spec (auto-restart, no deps).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dependencies: Vec::new(),
            restart_on_crash: true,
            rss_limit_mb: None,
            heartbeat_timeout_s: None,
            critical_data_writer: false,
        }
    }

    /// Append a dependency to the child's declared dep list.
    pub fn with_dependency(mut self, dep: Dependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Mark as a critical-data writer (SPEC §11.H.3 shutdown discipline).
    pub fn critical_data_writer(mut self) -> Self {
        self.critical_data_writer = true;
        self
    }
}

/// Runtime state per child.
#[derive(Debug)]
pub struct ChildState {
    /// Declarative spec.
    pub spec: ChildSpec,
    /// Current lifecycle.
    pub lifecycle: ChildLifecycle,
    /// Restart count within the current intensity window.
    pub restart_count_in_window: u32,
    /// When the current intensity window started.
    pub window_start: Instant,
    /// Per-child rolling reason buffer (last 16 records).
    pub reason_buffer: VecDeque<ReasonRecord>,
    /// When the current "stable uptime" timer started (resets restart
    /// counter after `SUPERVISION_SUSTAINED_UPTIME_RESET_S`).
    pub last_uptime_start: Instant,
    /// When the child was first started (for total uptime tracking).
    pub first_started_at: Option<SystemTime>,
}

impl ChildState {
    /// New state for a never-started child.
    pub fn new(spec: ChildSpec) -> Self {
        let now = Instant::now();
        Self {
            spec,
            lifecycle: ChildLifecycle::Dead,
            restart_count_in_window: 0,
            window_start: now,
            reason_buffer: VecDeque::with_capacity(REASON_BUFFER_SIZE),
            last_uptime_start: now,
            first_started_at: None,
        }
    }

    /// Append a reason to the rolling buffer (drops oldest if at cap).
    pub fn record_reason(&mut self, reason: ReasonRecord) {
        if self.reason_buffer.len() == REASON_BUFFER_SIZE {
            self.reason_buffer.pop_front();
        }
        self.reason_buffer.push_back(reason);
    }

    /// Most-common reason in the buffer (mode). Used by escalation per
    /// SPEC §11.B.1 step 1.
    pub fn most_common_reason(&self) -> SupervisionReason {
        if self.reason_buffer.is_empty() {
            return SupervisionReason::Other;
        }
        let mut counts: std::collections::HashMap<SupervisionReason, u32> =
            std::collections::HashMap::new();
        for r in &self.reason_buffer {
            *counts.entry(r.reason).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(r, _)| r)
            .unwrap_or(SupervisionReason::Other)
    }

    /// Reset the restart counter if sustained uptime exceeded
    /// `SUPERVISION_SUSTAINED_UPTIME_RESET_S` (per SPEC §11.B "Reset rule").
    pub fn maybe_reset_counter(&mut self, now: Instant, sustained_uptime: Duration) {
        if !self.lifecycle.is_running() {
            return;
        }
        let stable_for = now.saturating_duration_since(self.last_uptime_start);
        if stable_for >= sustained_uptime {
            self.restart_count_in_window = 0;
            self.window_start = now;
            self.reason_buffer.clear();
            self.last_uptime_start = now;
        }
    }

    /// Advance the intensity window: if `now - window_start > window_s`,
    /// reset counter to 0 + window_start to now.
    pub fn maybe_advance_window(&mut self, now: Instant, window: Duration) {
        let elapsed = now.saturating_duration_since(self.window_start);
        if elapsed >= window {
            self.restart_count_in_window = 0;
            self.window_start = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_is_terminal() {
        assert!(!ChildLifecycle::Running { pid: 1 }.is_terminal());
        assert!(!ChildLifecycle::RespawnBlocked {
            since: Instant::now(),
            blocked_dep: "x".into(),
            escalation_at: Instant::now(),
        }
        .is_terminal());
        assert!(ChildLifecycle::Halted.is_terminal());
        assert!(ChildLifecycle::Dead.is_terminal());
    }

    #[test]
    fn lifecycle_is_running() {
        assert!(ChildLifecycle::Running { pid: 1 }.is_running());
        assert!(!ChildLifecycle::Dead.is_running());
    }

    #[test]
    fn child_spec_new_defaults() {
        let s = ChildSpec::new("test");
        assert_eq!(s.name, "test");
        assert!(s.restart_on_crash);
        assert!(s.dependencies.is_empty());
        assert!(!s.critical_data_writer);
    }

    #[test]
    fn child_spec_critical_data_writer_builder() {
        let s = ChildSpec::new("memory_module").critical_data_writer();
        assert!(s.critical_data_writer);
    }

    #[test]
    fn record_reason_caps_buffer_at_16() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        for i in 0..20 {
            state.record_reason(ReasonRecord::new(
                SupervisionReason::Panic,
                format!("crash {i}"),
                None,
            ));
        }
        assert_eq!(state.reason_buffer.len(), REASON_BUFFER_SIZE);
        // Oldest 4 dropped; first remaining is "crash 4"
        assert_eq!(state.reason_buffer.front().unwrap().detail, "crash 4");
        assert_eq!(state.reason_buffer.back().unwrap().detail, "crash 19");
    }

    #[test]
    fn most_common_reason_returns_mode() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        for _ in 0..3 {
            state.record_reason(ReasonRecord::new(SupervisionReason::Panic, "p", None));
        }
        for _ in 0..5 {
            state.record_reason(ReasonRecord::new(SupervisionReason::Oom, "o", None));
        }
        for _ in 0..2 {
            state.record_reason(ReasonRecord::new(SupervisionReason::Hang, "h", None));
        }
        assert_eq!(state.most_common_reason(), SupervisionReason::Oom);
    }

    #[test]
    fn most_common_reason_empty_buffer_returns_other() {
        let state = ChildState::new(ChildSpec::new("test"));
        assert_eq!(state.most_common_reason(), SupervisionReason::Other);
    }

    #[test]
    fn maybe_reset_counter_on_sustained_uptime() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        state.lifecycle = ChildLifecycle::Running { pid: 100 };
        state.restart_count_in_window = 3;
        state.last_uptime_start = Instant::now() - Duration::from_secs(400);
        state.maybe_reset_counter(Instant::now(), Duration::from_secs(300));
        assert_eq!(state.restart_count_in_window, 0);
        assert!(state.reason_buffer.is_empty());
    }

    #[test]
    fn maybe_reset_counter_no_op_under_sustained_uptime() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        state.lifecycle = ChildLifecycle::Running { pid: 100 };
        state.restart_count_in_window = 3;
        state.last_uptime_start = Instant::now() - Duration::from_secs(100);
        state.maybe_reset_counter(Instant::now(), Duration::from_secs(300));
        assert_eq!(state.restart_count_in_window, 3);
    }

    #[test]
    fn maybe_reset_counter_skipped_when_not_running() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        state.lifecycle = ChildLifecycle::Dead;
        state.restart_count_in_window = 3;
        state.last_uptime_start = Instant::now() - Duration::from_secs(400);
        state.maybe_reset_counter(Instant::now(), Duration::from_secs(300));
        // Counter NOT reset — child isn't running
        assert_eq!(state.restart_count_in_window, 3);
    }

    #[test]
    fn maybe_advance_window_after_window_elapsed() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        state.restart_count_in_window = 4;
        state.window_start = Instant::now() - Duration::from_secs(70);
        state.maybe_advance_window(Instant::now(), Duration::from_secs(60));
        assert_eq!(state.restart_count_in_window, 0);
    }

    #[test]
    fn maybe_advance_window_no_op_within_window() {
        let mut state = ChildState::new(ChildSpec::new("test"));
        state.restart_count_in_window = 4;
        state.window_start = Instant::now() - Duration::from_secs(30);
        state.maybe_advance_window(Instant::now(), Duration::from_secs(60));
        assert_eq!(state.restart_count_in_window, 4);
    }
}
