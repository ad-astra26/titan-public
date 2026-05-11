//! escalation — Reason-tracking escalation handshake state machine.
//!
//! Per SPEC §11.B.1 (handshake) + §11.B.2 (kernel default policy).
//!
//! # Flow (SPEC §11.B.1)
//!
//! 1. Supervisor's restart counter exceeds `SUPERVISION_MAX_RESTARTS=5`
//!    within `SUPERVISION_INTENSITY_WINDOW_S=60` for a child.
//! 2. Supervisor computes `most_common_reason` from per-child reason buffer.
//! 3. Supervisor emits `SUPERVISION_ESCALATION` event with `escalation_id`.
//! 4. Supervisor enters `EscalationPending` state for that child; child
//!    stays dead; timer starts (`SUPERVISION_ESCALATION_TIMEOUT_S=10`).
//! 5. Kernel decides per §11.B.2 policy → emits
//!    `SUPERVISION_ESCALATION_RESPONSE` matched on `escalation_id`.
//! 6. Supervisor handles decision:
//!    - `continue` → reset counter; resume normal restart flow
//!    - `terminate` → supervisor exits with code 64
//!    - `halt` → child stays dead; periodic `CHILD_DOWN` reminder
//! 7. If no response within timeout → default to `terminate` (vanilla
//!    OTP fallback per SPEC §11.B.1 step 7).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::supervisor::types::{EscalationDecision, SupervisionReason};

/// Outcome of feeding an event into the escalation state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationOutcome {
    /// Decision applied; supervisor should `continue` restart flow with
    /// counter reset.
    Continued,
    /// Decision applied; supervisor should terminate self with exit 64.
    Terminate,
    /// Decision applied; child stays dead; supervisor emits periodic
    /// `CHILD_DOWN` reminders + awaits Maker intervention.
    Halt,
    /// Pending — no decision yet AND timeout not yet reached.
    Pending,
}

/// Tracks all in-flight escalation requests by `escalation_id`. One
/// `EscalationState` per supervisor (not per child — children share it
/// because escalation_ids are globally unique).
pub struct EscalationState {
    pending: HashMap<Uuid, PendingEscalation>,
}

#[derive(Debug, Clone)]
struct PendingEscalation {
    /// Child whose restart-storm triggered escalation.
    child_name: String,
    /// When the request was emitted (kept for future telemetry; not yet read).
    #[allow(dead_code)]
    started_at: Instant,
    /// When default-to-terminate fires.
    timeout_at: Instant,
    /// Mode of reason buffer at emission time (kept for future telemetry; not yet read).
    #[allow(dead_code)]
    most_common_reason: SupervisionReason,
}

impl EscalationState {
    /// New empty state.
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
        }
    }

    /// Register a new pending escalation. Caller publishes the
    /// `SUPERVISION_ESCALATION` event separately.
    pub fn register(
        &mut self,
        escalation_id: Uuid,
        child_name: String,
        most_common_reason: SupervisionReason,
        timeout: Duration,
    ) {
        let now = Instant::now();
        self.pending.insert(
            escalation_id,
            PendingEscalation {
                child_name,
                started_at: now,
                timeout_at: now + timeout,
                most_common_reason,
            },
        );
    }

    /// Number of in-flight escalations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if an escalation with this id is in-flight.
    pub fn is_pending(&self, id: &Uuid) -> bool {
        self.pending.contains_key(id)
    }

    /// Apply a decision received from the kernel.
    ///
    /// Returns the outcome the supervisor should act on. If the id is
    /// unknown (e.g. response arrived after timeout already triggered
    /// `Terminate`), returns `EscalationOutcome::Pending` to mean "no-op".
    pub fn apply_decision(&mut self, id: Uuid, decision: EscalationDecision) -> EscalationOutcome {
        if self.pending.remove(&id).is_none() {
            return EscalationOutcome::Pending;
        }
        match decision {
            EscalationDecision::Continue => EscalationOutcome::Continued,
            EscalationDecision::Terminate => EscalationOutcome::Terminate,
            EscalationDecision::Halt => EscalationOutcome::Halt,
        }
    }

    /// Sweep timed-out escalations. Returns the list of `(id, child_name)`
    /// that defaulted to terminate. Caller acts on each per SPEC §11.B.1
    /// step 7.
    pub fn sweep_timeouts(&mut self, now: Instant) -> Vec<(Uuid, String)> {
        let timed_out: Vec<Uuid> = self
            .pending
            .iter()
            .filter(|(_, p)| p.timeout_at <= now)
            .map(|(id, _)| *id)
            .collect();

        timed_out
            .iter()
            .filter_map(|id| {
                let pending = self.pending.remove(id)?;
                Some((*id, pending.child_name))
            })
            .collect()
    }
}

impl Default for EscalationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel's default escalation policy per SPEC §11.B.2.
///
/// - `Oom` → `Terminate` (limit too low or genuine bloat)
/// - `Panic` / `Segv` → `Terminate` (reproducible bug)
/// - `Hang` → `Terminate` (deadlock)
/// - `Empty` → `Halt` (almost certainly wiring bug; lower threshold)
/// - `DependencyBlocked` → `Continue` (keep waiting; dep recovery is correct)
/// - `ConfigError` → `Halt` (Maker must fix config)
/// - `BootFailure` → `Halt` (child can't even start)
/// - All others → `Terminate`
///
/// Maker can override per-child at runtime via
/// `kernel_rpc.set_escalation_policy(child_name, decision)` per SPEC §11.B.2.
pub fn kernel_default_decision(reason: SupervisionReason) -> EscalationDecision {
    match reason {
        SupervisionReason::Oom => EscalationDecision::Terminate,
        SupervisionReason::Panic => EscalationDecision::Terminate,
        SupervisionReason::Segv => EscalationDecision::Terminate,
        SupervisionReason::Hang => EscalationDecision::Terminate,
        SupervisionReason::Empty => EscalationDecision::Halt,
        SupervisionReason::DependencyBlocked => EscalationDecision::Continue,
        SupervisionReason::ConfigError => EscalationDecision::Halt,
        SupervisionReason::BootFailure => EscalationDecision::Halt,
        SupervisionReason::CleanExit => EscalationDecision::Continue,
        SupervisionReason::Killed => EscalationDecision::Terminate,
        SupervisionReason::Other => EscalationDecision::Terminate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_increments_pending() {
        let mut state = EscalationState::new();
        assert_eq!(state.pending_count(), 0);
        state.register(
            Uuid::new_v4(),
            "x".into(),
            SupervisionReason::Panic,
            Duration::from_secs(10),
        );
        assert_eq!(state.pending_count(), 1);
    }

    #[test]
    fn apply_continue_resolves_to_continued() {
        let mut state = EscalationState::new();
        let id = Uuid::new_v4();
        state.register(
            id,
            "x".into(),
            SupervisionReason::Panic,
            Duration::from_secs(10),
        );
        let outcome = state.apply_decision(id, EscalationDecision::Continue);
        assert_eq!(outcome, EscalationOutcome::Continued);
        assert_eq!(state.pending_count(), 0);
    }

    #[test]
    fn apply_terminate_resolves_to_terminate() {
        let mut state = EscalationState::new();
        let id = Uuid::new_v4();
        state.register(
            id,
            "x".into(),
            SupervisionReason::Oom,
            Duration::from_secs(10),
        );
        assert_eq!(
            state.apply_decision(id, EscalationDecision::Terminate),
            EscalationOutcome::Terminate
        );
    }

    #[test]
    fn apply_halt_resolves_to_halt() {
        let mut state = EscalationState::new();
        let id = Uuid::new_v4();
        state.register(
            id,
            "x".into(),
            SupervisionReason::Empty,
            Duration::from_secs(10),
        );
        assert_eq!(
            state.apply_decision(id, EscalationDecision::Halt),
            EscalationOutcome::Halt
        );
    }

    #[test]
    fn unknown_id_returns_pending() {
        let mut state = EscalationState::new();
        let unknown = Uuid::new_v4();
        let outcome = state.apply_decision(unknown, EscalationDecision::Terminate);
        assert_eq!(outcome, EscalationOutcome::Pending);
    }

    #[test]
    fn sweep_timeouts_drains_expired() {
        let mut state = EscalationState::new();
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        state.register(
            id_a,
            "a".into(),
            SupervisionReason::Panic,
            Duration::from_millis(10),
        );
        state.register(
            id_b,
            "b".into(),
            SupervisionReason::Hang,
            Duration::from_secs(60),
        );
        std::thread::sleep(Duration::from_millis(20));
        let timed_out = state.sweep_timeouts(Instant::now());
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0].0, id_a);
        assert_eq!(timed_out[0].1, "a");
        assert_eq!(state.pending_count(), 1);
    }

    #[test]
    fn sweep_idempotent_when_no_timeouts() {
        let mut state = EscalationState::new();
        state.register(
            Uuid::new_v4(),
            "x".into(),
            SupervisionReason::Panic,
            Duration::from_secs(60),
        );
        let timed_out = state.sweep_timeouts(Instant::now());
        assert!(timed_out.is_empty());
    }

    // ── Kernel default policy tests ───────────────────────────────────

    #[test]
    fn kernel_policy_oom_terminate() {
        assert_eq!(
            kernel_default_decision(SupervisionReason::Oom),
            EscalationDecision::Terminate
        );
    }

    #[test]
    fn kernel_policy_empty_halts_aggressively() {
        // Per SPEC §11.B.2: EMPTY threshold lower than other reasons (3
        // vs 5) because it's near-certain to be a wiring bug.
        assert_eq!(
            kernel_default_decision(SupervisionReason::Empty),
            EscalationDecision::Halt
        );
    }

    #[test]
    fn kernel_policy_dep_blocked_continues_waiting() {
        // Per SPEC §11.B.2: DEPENDENCY_BLOCKED → continue (keep waiting +
        // rechecking; dep recovery is the correct action).
        assert_eq!(
            kernel_default_decision(SupervisionReason::DependencyBlocked),
            EscalationDecision::Continue
        );
    }

    #[test]
    fn kernel_policy_config_error_halts() {
        assert_eq!(
            kernel_default_decision(SupervisionReason::ConfigError),
            EscalationDecision::Halt
        );
    }

    #[test]
    fn kernel_policy_boot_failure_halts() {
        assert_eq!(
            kernel_default_decision(SupervisionReason::BootFailure),
            EscalationDecision::Halt
        );
    }

    #[test]
    fn kernel_policy_panic_terminates() {
        assert_eq!(
            kernel_default_decision(SupervisionReason::Panic),
            EscalationDecision::Terminate
        );
    }

    #[test]
    fn kernel_policy_other_terminates() {
        assert_eq!(
            kernel_default_decision(SupervisionReason::Other),
            EscalationDecision::Terminate
        );
    }
}
