//! exit — Kernel exit code taxonomy per SPEC §15.
//!
//! Exit codes drive systemd's `RestartPreventExitStatus` policy. SPEC §15
//! lists which codes auto-restart vs which escalate (`halt` per §11.B.2).

use std::process::ExitCode;

/// Canonical kernel exit codes per SPEC §15.
///
/// Ranges:
/// - `0`: clean shutdown (SIGTERM → graceful). Counted as healthy.
/// - `1`: generic error (panic, uncaught exception). systemd auto-restarts.
/// - `2-6`: config-level errors. systemd does NOT auto-restart (per
///   `RestartPreventExitStatus=2 3 4 5 6` in the systemd unit). Maker
///   must intervene.
/// - `7`: parent died (PDEATHSIG). Logged only; parent will respawn or I exit.
/// - `8`: adoption rejected post-swap. Restart fresh.
/// - `64-127`: reserved for future Phase D supervisor signals (incl. 64 =
///   supervisor self-terminated after escalation `terminate`).
/// - `128 + N`: process killed by signal N (137 = SIGKILL, 143 = SIGTERM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelExitCode {
    /// Clean shutdown (SIGTERM → graceful).
    Clean = 0,
    /// Generic error (panic, uncaught exception).
    Generic = 1,
    /// Config error (invalid TOML, missing required env).
    ConfigError = 2,
    /// Identity load failure (missing keypair, corrupt, hex decode fail).
    IdentityLoadFailure = 3,
    /// Bus bind failure (socket already in use, perms).
    BusBindFailure = 4,
    /// Shm create failure (no `/dev/shm` space, perms).
    ShmCreateFailure = 5,
    /// Child process limit reached.
    ChildLimitReached = 6,
    /// Parent died (PDEATHSIG).
    ParentDied = 7,
    /// Adoption rejected (post-swap).
    AdoptionRejected = 8,
    /// Supervisor self-terminated after escalation `terminate` decision.
    SupervisorSelfTerminate = 64,
}

impl KernelExitCode {
    /// Convert to `std::process::ExitCode`.
    pub fn to_exit_code(self) -> ExitCode {
        ExitCode::from(self as u8)
    }

    /// Returns `true` if systemd should auto-restart on this exit code.
    /// Mirrors `RestartPreventExitStatus=2 3 4 5 6` in the systemd unit.
    pub fn systemd_auto_restarts(self) -> bool {
        !matches!(
            self,
            KernelExitCode::ConfigError
                | KernelExitCode::IdentityLoadFailure
                | KernelExitCode::BusBindFailure
                | KernelExitCode::ShmCreateFailure
                | KernelExitCode::ChildLimitReached
        )
    }

    /// Human-readable label for log lines.
    pub fn as_str(&self) -> &'static str {
        match self {
            KernelExitCode::Clean => "clean",
            KernelExitCode::Generic => "generic",
            KernelExitCode::ConfigError => "config_error",
            KernelExitCode::IdentityLoadFailure => "identity_load_failure",
            KernelExitCode::BusBindFailure => "bus_bind_failure",
            KernelExitCode::ShmCreateFailure => "shm_create_failure",
            KernelExitCode::ChildLimitReached => "child_limit_reached",
            KernelExitCode::ParentDied => "parent_died",
            KernelExitCode::AdoptionRejected => "adoption_rejected",
            KernelExitCode::SupervisorSelfTerminate => "supervisor_self_terminate",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_spec_15() {
        assert_eq!(KernelExitCode::Clean as u8, 0);
        assert_eq!(KernelExitCode::Generic as u8, 1);
        assert_eq!(KernelExitCode::ConfigError as u8, 2);
        assert_eq!(KernelExitCode::IdentityLoadFailure as u8, 3);
        assert_eq!(KernelExitCode::BusBindFailure as u8, 4);
        assert_eq!(KernelExitCode::ShmCreateFailure as u8, 5);
        assert_eq!(KernelExitCode::ChildLimitReached as u8, 6);
        assert_eq!(KernelExitCode::ParentDied as u8, 7);
        assert_eq!(KernelExitCode::AdoptionRejected as u8, 8);
        assert_eq!(KernelExitCode::SupervisorSelfTerminate as u8, 64);
    }

    #[test]
    fn config_class_codes_do_not_auto_restart() {
        // Per SPEC §15: codes 2-6 escalate; do not auto-restart.
        for code in [
            KernelExitCode::ConfigError,
            KernelExitCode::IdentityLoadFailure,
            KernelExitCode::BusBindFailure,
            KernelExitCode::ShmCreateFailure,
            KernelExitCode::ChildLimitReached,
        ] {
            assert!(
                !code.systemd_auto_restarts(),
                "{} should NOT auto-restart",
                code.as_str()
            );
        }
    }

    #[test]
    fn other_codes_auto_restart() {
        for code in [
            KernelExitCode::Clean,
            KernelExitCode::Generic,
            KernelExitCode::ParentDied,
            KernelExitCode::AdoptionRejected,
            KernelExitCode::SupervisorSelfTerminate,
        ] {
            assert!(
                code.systemd_auto_restarts(),
                "{} should auto-restart",
                code.as_str()
            );
        }
    }

    #[test]
    fn as_str_canonical_labels() {
        assert_eq!(KernelExitCode::Clean.as_str(), "clean");
        assert_eq!(
            KernelExitCode::IdentityLoadFailure.as_str(),
            "identity_load_failure"
        );
        assert_eq!(
            KernelExitCode::SupervisorSelfTerminate.as_str(),
            "supervisor_self_terminate"
        );
    }
}
