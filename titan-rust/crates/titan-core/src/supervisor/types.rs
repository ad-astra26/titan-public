//! types — Enums + small structs used across the supervisor module.
//!
//! These are the stable primitives downstream crates depend on. Keep this
//! file backwards-compatible per SPEC §2.6 (MAJOR bump on rename/removal).

use std::time::SystemTime;

/// Classification of why a child crashed (per SPEC §11.B step 2).
///
/// Used in:
/// - Per-child rolling buffer of last 16 records (SPEC §11.B.1)
/// - `SUPERVISION_ESCALATION` payload `most_common_reason` field (SPEC §8.1)
/// - Kernel escalation policy decisions (SPEC §11.B.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupervisionReason {
    /// RSS exceeded limit.
    Oom,
    /// Rust panic / Python uncaught exception.
    Panic,
    /// SIGSEGV / SIGBUS / SIGABRT.
    Segv,
    /// Heartbeat timeout / starved-cycle threshold.
    Hang,
    /// Module alive (heartbeat OK) but primary output never populated within
    /// `SUPERVISION_EMPTY_GRACE_S`. High-signal — usually wiring bug.
    Empty,
    /// Critical dependency down (per SPEC §11.G); refused to respawn.
    DependencyBlocked,
    /// Exit codes 2/3/4/5/6 per SPEC §15 — Maker config fix needed.
    ConfigError,
    /// Failed `MODULE_READY` / primary-slot write within boot timeout.
    BootFailure,
    /// Exit 0 (only relevant when `restart_on_crash=true`).
    CleanExit,
    /// SIGKILL from outside the supervisor (e.g. Maker emergency stop).
    Killed,
    /// Unclassified.
    Other,
}

impl SupervisionReason {
    /// Canonical string form used in `data/supervision.jsonl` log lines.
    pub fn as_str(&self) -> &'static str {
        match self {
            SupervisionReason::Oom => "OOM",
            SupervisionReason::Panic => "PANIC",
            SupervisionReason::Segv => "SEGV",
            SupervisionReason::Hang => "HANG",
            SupervisionReason::Empty => "EMPTY",
            SupervisionReason::DependencyBlocked => "DEPENDENCY_BLOCKED",
            SupervisionReason::ConfigError => "CONFIG_ERROR",
            SupervisionReason::BootFailure => "BOOT_FAILURE",
            SupervisionReason::CleanExit => "CLEAN_EXIT",
            SupervisionReason::Killed => "KILLED",
            SupervisionReason::Other => "OTHER",
        }
    }
}

/// One record in a per-child rolling reason buffer (size 16).
#[derive(Debug, Clone)]
pub struct ReasonRecord {
    /// Classified reason.
    pub reason: SupervisionReason,
    /// Free-text detail (capped 256 chars per SPEC §11.B step 2).
    pub detail: String,
    /// When the crash was observed.
    pub ts: SystemTime,
    /// Process exit code, if any.
    pub exit_code: Option<i32>,
}

impl ReasonRecord {
    /// Truncate detail to 256 chars per SPEC §11.B step 2.
    pub fn new(
        reason: SupervisionReason,
        detail: impl Into<String>,
        exit_code: Option<i32>,
    ) -> Self {
        let mut detail: String = detail.into();
        if detail.len() > 256 {
            detail.truncate(256);
        }
        Self {
            reason,
            detail,
            ts: SystemTime::now(),
            exit_code,
        }
    }
}

/// Kernel decision on an escalation request (SPEC §11.B.1 step 5/6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EscalationDecision {
    /// Reset counter; resume normal restart flow (optionally with new
    /// `max_restarts` / `window_s` from kernel).
    Continue,
    /// Supervisor terminates self with exit 64 (SPEC §15 supervision range).
    Terminate,
    /// Leave child dead; periodic `SUPERVISION_CHILD_DOWN` reminder; await
    /// Maker intervention.
    Halt,
}

/// Severity of a declared child dependency (SPEC §11.G.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DependencySeverity {
    /// Refuse to respawn; escalate to kernel after 5 min.
    Critical,
    /// Log warning; respawn anyway; module degrades.
    Soft,
}

/// Kind of a declared child dependency (SPEC §11.G.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyKind {
    /// Sibling Python module (must be RUNNING in `guardian_HCL`).
    Module,
    /// Rust binary (must be alive — supervisor's child registry).
    Binary,
    /// `/dev/shm/titan_<id>/<slot>.bin` exists + populated + fresh `wall_ns`.
    ShmSlot,
    /// External service (HTTP/RPC probe — Solana RPC, X API, Ollama, etc.).
    ExternalService,
    /// `data/*.db` exists + readable + schema version OK.
    DbFile,
    /// HTTP 200 on URL within timeout.
    Endpoint,
}

impl DependencyKind {
    /// Canonical snake_case form for log fields.
    pub fn as_str(&self) -> &'static str {
        match self {
            DependencyKind::Module => "module",
            DependencyKind::Binary => "binary",
            DependencyKind::ShmSlot => "shm_slot",
            DependencyKind::ExternalService => "external_service",
            DependencyKind::DbFile => "db_file",
            DependencyKind::Endpoint => "endpoint",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervision_reason_canonical_strings() {
        assert_eq!(SupervisionReason::Oom.as_str(), "OOM");
        assert_eq!(
            SupervisionReason::DependencyBlocked.as_str(),
            "DEPENDENCY_BLOCKED"
        );
        assert_eq!(SupervisionReason::Empty.as_str(), "EMPTY");
        assert_eq!(SupervisionReason::Other.as_str(), "OTHER");
    }

    #[test]
    fn supervision_reason_serialize_uppercase() {
        let json = serde_json::to_string(&SupervisionReason::DependencyBlocked).unwrap();
        assert_eq!(json, "\"DEPENDENCY_BLOCKED\"");
    }

    #[test]
    fn escalation_decision_serialize_lowercase() {
        let json = serde_json::to_string(&EscalationDecision::Halt).unwrap();
        assert_eq!(json, "\"halt\"");
    }

    #[test]
    fn dependency_kind_canonical_strings() {
        assert_eq!(DependencyKind::Module.as_str(), "module");
        assert_eq!(DependencyKind::ShmSlot.as_str(), "shm_slot");
        assert_eq!(DependencyKind::ExternalService.as_str(), "external_service");
    }

    #[test]
    fn reason_record_truncates_long_detail() {
        let long_detail = "a".repeat(500);
        let r = ReasonRecord::new(SupervisionReason::Panic, long_detail, Some(1));
        assert_eq!(r.detail.len(), 256);
        assert_eq!(r.exit_code, Some(1));
    }

    #[test]
    fn reason_record_short_detail_unchanged() {
        let r = ReasonRecord::new(SupervisionReason::Oom, "short", None);
        assert_eq!(r.detail, "short");
    }
}
