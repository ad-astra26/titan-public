//! event — `SupervisionEvent` enum + `SupervisionPublisher` trait.
//!
//! Per SPEC §8.1 (kernel + supervision messages, all P0). Every event the
//! supervisor emits ends up as a message on the main bus AND a JSONL line
//! in `data/supervision.jsonl` (per SPEC §11.G.4).
//!
//! The `SupervisionPublisher` trait abstracts the broker so the supervisor
//! is testable with `NoopPublisher` / `RecordingPublisher` and the kernel
//! binary in C2-6 wires it to `titan-bus::BusBroker`.

use std::time::SystemTime;

use parking_lot::Mutex;
use uuid::Uuid;

use crate::supervisor::types::SupervisionReason;

/// Per-supervision event emitted to the bus + supervision.jsonl.
///
/// Variant names + field names match SPEC §8.1 message catalog.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupervisionEvent {
    /// `SUPERVISION_CHILD_DOWN` — a supervised child crashed.
    ChildDown {
        /// Canonical child name.
        child_name: String,
        /// Supervisor name emitting the event.
        supervisor: String,
        /// Classified reason.
        reason: SupervisionReason,
        /// Free-text detail (capped 256 chars).
        reason_detail: String,
        /// Number of restarts within current intensity window.
        restart_count: u32,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },

    /// `SUPERVISION_CHILD_RESTARTED` — child has been restarted.
    ChildRestarted {
        /// Canonical child name.
        child_name: String,
        /// Supervisor name.
        supervisor: String,
        /// Classification of the crash that triggered the restart.
        reason: SupervisionReason,
        /// Restart count after this restart.
        restart_count: u32,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },

    /// `SUPERVISION_ESCALATION` — max_restarts exceeded; supervisor
    /// requesting kernel decision (continue / terminate / halt).
    Escalation {
        /// UUID correlating request → response.
        escalation_id: Uuid,
        /// Child whose restart-storm triggered escalation.
        child_name: String,
        /// Supervisor emitting the request.
        supervisor: String,
        /// Restart count at escalation time.
        restart_count: u32,
        /// Window size (`SUPERVISION_INTENSITY_WINDOW_S` typical).
        window_s: f64,
        /// All reasons observed in this child's rolling buffer.
        reasons_observed: Vec<String>,
        /// Mode of `reasons_observed`.
        most_common_reason: SupervisionReason,
        /// Last reason's detail field.
        last_reason_detail: String,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },

    /// `SUPERVISION_DEPENDENCY_BLOCKED` — pre-respawn check failed; child
    /// stays dead until dep recovers (or escalation timeout).
    DependencyBlocked {
        /// Child whose respawn was blocked.
        child_name: String,
        /// Supervisor.
        supervisor: String,
        /// Name of the blocking dep (e.g. `"x_api_reachable"`).
        blocked_dependency: String,
        /// Dep kind (canonical snake_case).
        dependency_kind: String,
        /// Severity (`"critical"` or `"soft"`).
        severity: String,
        /// When the block began.
        since_ts: SystemTime,
        /// Recheck interval per SPEC §11.G.3.
        recheck_interval_s: f64,
        /// When escalation will fire if dep stays blocked.
        escalation_at_ts: SystemTime,
    },

    /// `SUPERVISION_DEPENDENCY_RECOVERED` — dep came back; supervisor
    /// proceeds with respawn.
    DependencyRecovered {
        /// Child whose respawn is unblocked.
        child_name: String,
        /// Supervisor.
        supervisor: String,
        /// Dep that recovered.
        dependency_name: String,
        /// When the recovery was detected.
        restored_at_ts: SystemTime,
        /// Total seconds the child was blocked.
        total_blocked_s: f64,
    },

    /// `SUPERVISION_DEPENDENCY_DEGRADED` — soft dep failed; respawn
    /// continued with degraded service. Informational (P1).
    DependencyDegraded {
        /// Child name.
        child_name: String,
        /// Supervisor.
        supervisor: String,
        /// Dep that's degraded.
        soft_dependency: String,
        /// When degradation began.
        degraded_since_ts: SystemTime,
    },

    /// `SUPERVISION_DATA_RESTORE` — boot integrity check failed; restored
    /// from `.bak` per SPEC §11.H.4.
    DataRestore {
        /// File path that was restored.
        path: String,
        /// Why the integrity check failed.
        failure_reason: String,
        /// Source of the restore (`"bak"` or `"bak.prev"`).
        restored_from: String,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },

    /// `SUPERVISION_DATA_LOST` — both `.bak` files also corrupted; halt.
    DataLost {
        /// File path that was lost.
        path: String,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },

    /// `SUPERVISION_FORCED_KILL` — critical-data writer SIGKILL'd after
    /// shutdown grace exhausted (only acceptable kill path per G16).
    ForcedKill {
        /// Child SIGKILL'd.
        child_name: String,
        /// Supervisor.
        supervisor: String,
        /// Why the kill was forced.
        reason: String,
        /// Grace window that was exhausted.
        grace_exhausted_s: f64,
        /// Wall-clock timestamp.
        ts: SystemTime,
    },
}

impl SupervisionEvent {
    /// Canonical bus message type (matches SPEC §8.1 names).
    pub fn msg_type(&self) -> &'static str {
        match self {
            SupervisionEvent::ChildDown { .. } => "SUPERVISION_CHILD_DOWN",
            SupervisionEvent::ChildRestarted { .. } => "SUPERVISION_CHILD_RESTARTED",
            SupervisionEvent::Escalation { .. } => "SUPERVISION_ESCALATION",
            SupervisionEvent::DependencyBlocked { .. } => "SUPERVISION_DEPENDENCY_BLOCKED",
            SupervisionEvent::DependencyRecovered { .. } => "SUPERVISION_DEPENDENCY_RECOVERED",
            SupervisionEvent::DependencyDegraded { .. } => "SUPERVISION_DEPENDENCY_DEGRADED",
            SupervisionEvent::DataRestore { .. } => "SUPERVISION_DATA_RESTORE",
            SupervisionEvent::DataLost { .. } => "SUPERVISION_DATA_LOST",
            SupervisionEvent::ForcedKill { .. } => "SUPERVISION_FORCED_KILL",
        }
    }
}

/// Trait abstracting how supervision events are published.
///
/// Implemented by the kernel binary against `titan-bus::BusBroker` (C2-6).
/// Tests use [`NoopPublisher`] or [`RecordingPublisher`].
pub trait SupervisionPublisher: Send + Sync {
    /// Publish one event. Returning an error logs but does NOT halt the
    /// supervisor — supervision events are best-effort under bus pressure;
    /// `data/supervision.jsonl` is the durable record.
    fn publish(&self, event: &SupervisionEvent) -> Result<(), String>;
}

/// No-op publisher — for tests + boot-time scenarios pre-broker.
pub struct NoopPublisher;
impl SupervisionPublisher for NoopPublisher {
    fn publish(&self, _: &SupervisionEvent) -> Result<(), String> {
        Ok(())
    }
}

/// Records all events for inspection in tests.
pub struct RecordingPublisher {
    events: Mutex<Vec<SupervisionEvent>>,
}

impl RecordingPublisher {
    /// New empty recorder.
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    /// Number of events captured so far.
    pub fn count(&self) -> usize {
        self.events.lock().len()
    }

    /// Snapshot of all captured events.
    pub fn snapshot(&self) -> Vec<SupervisionEvent> {
        self.events.lock().clone()
    }

    /// First event matching a predicate, if any.
    pub fn find<F: Fn(&SupervisionEvent) -> bool>(&self, predicate: F) -> Option<SupervisionEvent> {
        self.events.lock().iter().find(|e| predicate(e)).cloned()
    }

    /// Count events whose `msg_type()` matches.
    pub fn count_of(&self, msg_type: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| e.msg_type() == msg_type)
            .count()
    }
}

impl Default for RecordingPublisher {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisionPublisher for RecordingPublisher {
    fn publish(&self, event: &SupervisionEvent) -> Result<(), String> {
        self.events.lock().push(event.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child_down_event() -> SupervisionEvent {
        SupervisionEvent::ChildDown {
            child_name: "social_module".into(),
            supervisor: "guardian_HCL".into(),
            reason: SupervisionReason::Panic,
            reason_detail: "uncaught exception in social loop".into(),
            restart_count: 1,
            ts: SystemTime::now(),
        }
    }

    #[test]
    fn msg_type_matches_spec_8_1() {
        let e = child_down_event();
        assert_eq!(e.msg_type(), "SUPERVISION_CHILD_DOWN");

        let restart = SupervisionEvent::ChildRestarted {
            child_name: "x".into(),
            supervisor: "y".into(),
            reason: SupervisionReason::Other,
            restart_count: 2,
            ts: SystemTime::now(),
        };
        assert_eq!(restart.msg_type(), "SUPERVISION_CHILD_RESTARTED");

        let esc = SupervisionEvent::Escalation {
            escalation_id: Uuid::new_v4(),
            child_name: "x".into(),
            supervisor: "y".into(),
            restart_count: 5,
            window_s: 60.0,
            reasons_observed: vec!["PANIC".into()],
            most_common_reason: SupervisionReason::Panic,
            last_reason_detail: "...".into(),
            ts: SystemTime::now(),
        };
        assert_eq!(esc.msg_type(), "SUPERVISION_ESCALATION");
    }

    #[test]
    fn noop_publisher_succeeds() {
        let p = NoopPublisher;
        assert!(p.publish(&child_down_event()).is_ok());
    }

    #[test]
    fn recording_publisher_captures_events() {
        let p = RecordingPublisher::new();
        assert_eq!(p.count(), 0);
        p.publish(&child_down_event()).unwrap();
        assert_eq!(p.count(), 1);
        assert_eq!(p.count_of("SUPERVISION_CHILD_DOWN"), 1);
        assert_eq!(p.count_of("SUPERVISION_ESCALATION"), 0);
    }

    #[test]
    fn recording_publisher_find() {
        let p = RecordingPublisher::new();
        p.publish(&child_down_event()).unwrap();
        let found = p.find(|e| matches!(e, SupervisionEvent::ChildDown { .. }));
        assert!(found.is_some());
        let none = p.find(|e| matches!(e, SupervisionEvent::DataLost { .. }));
        assert!(none.is_none());
    }
}
