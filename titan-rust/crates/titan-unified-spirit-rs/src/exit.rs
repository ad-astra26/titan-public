//! exit — Unified-spirit exit code taxonomy per SPEC §15.
//!
//! Same taxonomy as kernel (SPEC §15 is global) but with binary-specific
//! contexts. Only the codes meaningful for unified-spirit are listed; the
//! kernel-only codes (e.g. ChildLimitReached) are still reachable via the
//! generic `from_u8` path in case future config triggers them.

use std::process::ExitCode;

/// Canonical unified-spirit exit codes per SPEC §15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UnifiedSpiritExitCode {
    /// Clean shutdown (SIGTERM → graceful).
    Clean = 0,
    /// Generic error (panic, uncaught exception). Parent restarts per §11.B.
    Generic = 1,
    /// Config error (invalid TOML, missing required env). Parent escalates `halt`.
    ConfigError = 2,
    /// Identity load failure (shm slot missing, not yet populated by kernel).
    IdentityLoadFailure = 3,
    /// Bus connect failure (socket missing, auth handshake failed).
    BusConnectFailure = 4,
    /// Shm slot open failure (kernel did not create the expected slot).
    ShmOpenFailure = 5,
    /// Child process limit reached (6 daemon spawn budget exceeded).
    ChildLimitReached = 6,
    /// Parent died (PDEATHSIG → kernel/substrate dropped us).
    ParentDied = 7,
    /// Adoption rejected post-swap (B.2.1 supervision transfer failed).
    AdoptionRejected = 8,
    /// Supervisor self-terminated after escalation `terminate` decision.
    SupervisorSelfTerminate = 64,
}

impl UnifiedSpiritExitCode {
    /// Convert to `std::process::ExitCode`.
    pub fn to_exit_code(self) -> ExitCode {
        ExitCode::from(self as u8)
    }

    /// Stable string label for structured logs / `EXIT` event.
    pub fn as_str(self) -> &'static str {
        match self {
            UnifiedSpiritExitCode::Clean => "CLEAN",
            UnifiedSpiritExitCode::Generic => "GENERIC",
            UnifiedSpiritExitCode::ConfigError => "CONFIG_ERROR",
            UnifiedSpiritExitCode::IdentityLoadFailure => "IDENTITY_LOAD_FAILURE",
            UnifiedSpiritExitCode::BusConnectFailure => "BUS_CONNECT_FAILURE",
            UnifiedSpiritExitCode::ShmOpenFailure => "SHM_OPEN_FAILURE",
            UnifiedSpiritExitCode::ChildLimitReached => "CHILD_LIMIT_REACHED",
            UnifiedSpiritExitCode::ParentDied => "PARENT_DIED",
            UnifiedSpiritExitCode::AdoptionRejected => "ADOPTION_REJECTED",
            UnifiedSpiritExitCode::SupervisorSelfTerminate => "SUPERVISOR_SELF_TERMINATE",
        }
    }

    /// Returns `true` if the parent supervisor should auto-restart on this
    /// exit code. Mirrors SPEC §11.B classification + §15 taxonomy.
    pub fn parent_auto_restarts(self) -> bool {
        // Config-class errors are non-auto-restart per SPEC §11.B.2 (halt).
        !matches!(
            self,
            UnifiedSpiritExitCode::ConfigError
                | UnifiedSpiritExitCode::IdentityLoadFailure
                | UnifiedSpiritExitCode::BusConnectFailure
                | UnifiedSpiritExitCode::ShmOpenFailure
                | UnifiedSpiritExitCode::ChildLimitReached
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_spec_section_15() {
        // SPEC §15 numeric values — locked.
        assert_eq!(UnifiedSpiritExitCode::Clean as u8, 0);
        assert_eq!(UnifiedSpiritExitCode::Generic as u8, 1);
        assert_eq!(UnifiedSpiritExitCode::ConfigError as u8, 2);
        assert_eq!(UnifiedSpiritExitCode::IdentityLoadFailure as u8, 3);
        assert_eq!(UnifiedSpiritExitCode::BusConnectFailure as u8, 4);
        assert_eq!(UnifiedSpiritExitCode::ShmOpenFailure as u8, 5);
        assert_eq!(UnifiedSpiritExitCode::ChildLimitReached as u8, 6);
        assert_eq!(UnifiedSpiritExitCode::ParentDied as u8, 7);
        assert_eq!(UnifiedSpiritExitCode::AdoptionRejected as u8, 8);
        assert_eq!(UnifiedSpiritExitCode::SupervisorSelfTerminate as u8, 64);
    }

    #[test]
    fn config_class_errors_do_not_auto_restart() {
        // Per SPEC §11.B.2: config-class errors → escalate `halt`, no auto-restart.
        assert!(!UnifiedSpiritExitCode::ConfigError.parent_auto_restarts());
        assert!(!UnifiedSpiritExitCode::IdentityLoadFailure.parent_auto_restarts());
        assert!(!UnifiedSpiritExitCode::BusConnectFailure.parent_auto_restarts());
        assert!(!UnifiedSpiritExitCode::ShmOpenFailure.parent_auto_restarts());
        // Generic/transient errors → auto-restart per §11.B ladder.
        assert!(UnifiedSpiritExitCode::Generic.parent_auto_restarts());
        assert!(UnifiedSpiritExitCode::ParentDied.parent_auto_restarts());
        assert!(UnifiedSpiritExitCode::Clean.parent_auto_restarts());
    }
}
