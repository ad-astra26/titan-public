//! supervisor — Erlang/OTP-style one_for_one supervisor primitives.
//!
//! Per SPEC §11.B (restart strategy) + §11.B.1 (reason-tracking escalation
//! handshake) + §11.B.2 (kernel escalation policy) + §11.C (PR_SET_PDEATHSIG +
//! PR_SET_CHILD_SUBREAPER) + §11.G (dependency-aware respawn).
//!
//! # Submodules
//!
//! - [`types`] — `SupervisionReason`, `ReasonRecord`, `EscalationDecision`,
//!   `DependencyKind`, `DependencySeverity`. Stable enums used by the rest
//!   of the workspace (e.g. `titan-bus::heartbeat`).
//! - [`backoff`] — Jittered exponential backoff per §11.B step 5
//!   (`100ms → 200 → 400 → 800 → 1600` capped at `2s`, ±25% jitter).
//! - [`event`] — `SupervisionEvent` enum + `SupervisionPublisher` trait.
//!   The supervisor emits events to a publisher; `titan-bus` provides the
//!   real broker-backed impl in C2-6, tests use `NoopPublisher` /
//!   `RecordingPublisher`.
//! - [`child`] — `ChildSpec` (declarative) + `ChildState` (live runtime
//!   state) + `ChildLifecycle` (Running / RespawnBlocked /
//!   EscalationPending / Halted / Dead).
//! - [`restart`] — `Supervisor` struct: child registry, intensity window
//!   per child, restart counter, sustained-uptime reset.
//! - [`escalation`] — Reason-tracking escalation handshake state machine.
//! - [`dependency`] — `Dependency` + `DepProbe` trait + DAG cycle check.
//! - [`prctl_unix`] — `PR_SET_PDEATHSIG` + `PR_SET_CHILD_SUBREAPER` syscall
//!   wrappers for Linux. Per SPEC §11.C.
//!
//! # Public API stability
//!
//! All types previously exported from the flat `supervisor.rs` (C2-1
//! skeleton) remain at the same paths via `pub use` re-exports below.
//! Downstream crates (`titan-bus`, `titan-state`) require no changes.

pub mod backoff;
pub mod child;
pub mod dependency;
pub mod escalation;
pub mod event;
pub mod prctl_unix;
pub mod restart;
pub mod types;

// ── Public API (preserved from C2-1 skeleton) ────────────────────────────

pub use crate::constants::{
    SUPERVISION_DEPENDENCY_BLOCKED_TIMEOUT_S, SUPERVISION_DEPENDENCY_PROBE_TIMEOUT_S,
    SUPERVISION_DEPENDENCY_RECHECK_INTERVAL_S, SUPERVISION_EMPTY_GRACE_S,
    SUPERVISION_ESCALATION_TIMEOUT_S, SUPERVISION_INTENSITY_WINDOW_S,
    SUPERVISION_LOG_ARCHIVE_COUNT, SUPERVISION_LOG_MAX_BYTES, SUPERVISION_LOG_PATH,
    SUPERVISION_MAX_RESTARTS, SUPERVISION_RESTART_BACKOFF_INITIAL_MS,
    SUPERVISION_RESTART_BACKOFF_MAX_S, SUPERVISION_RESTART_JITTER_PCT,
    SUPERVISION_SUSTAINED_UPTIME_RESET_S,
};

pub use crate::supervisor::backoff::{backoff_base_unjittered, compute_backoff};
pub use crate::supervisor::child::{ChildLifecycle, ChildSpec, ChildState};
pub use crate::supervisor::dependency::{
    check_dag, DagCheckError, DepProbe, Dependency, DependencyCheckOutcome, MockDepProbe,
};
pub use crate::supervisor::escalation::{
    kernel_default_decision, EscalationOutcome, EscalationState,
};
pub use crate::supervisor::event::{
    NoopPublisher, RecordingPublisher, SupervisionEvent, SupervisionPublisher,
};
pub use crate::supervisor::restart::{
    classify_exit, Supervisor, SupervisorConfig, SupervisorError,
};
pub use crate::supervisor::types::{
    DependencyKind, DependencySeverity, EscalationDecision, ReasonRecord, SupervisionReason,
};
