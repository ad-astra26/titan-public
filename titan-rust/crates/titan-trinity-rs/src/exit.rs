//! exit — Substrate exit code taxonomy.
//!
//! Per SPEC §15: shared canonical taxonomy across all Rust binaries. Substrate
//! uses the same codes as kernel-rs so kernel's supervisor can interpret child
//! exit consistently per §11.B.2 escalation policy.

use std::process::ExitCode;

/// Canonical substrate exit codes per SPEC §15.
///
/// Codes 0..=8 match `titan-kernel-rs::exit::KernelExitCode` semantics; codes
/// ≥ 64 are reserved supervision codes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum SubstrateExitCode {
    /// Clean shutdown (SIGTERM → graceful sequence completed).
    Clean = 0,
    /// Generic error (panic, uncaught exception, unhandled error path).
    Generic = 1,
    /// Config error (invalid CLI / missing required env). Maker must fix.
    ConfigError = 2,
    /// Identity-related error. Substrate doesn't load identity directly
    /// (kernel passes authkey via env per §3 D08); this code is reserved
    /// for future use + parity with kernel.
    IdentityError = 3,
    /// Bus connect failure — can't connect to `/tmp/titan_bus_<id>.sock`.
    BusConnectFailure = 4,
    /// Shm open / fastbus attach failure — can't open kernel-pre-created
    /// slot file or fastbus.bin.
    ShmOpenFailure = 5,
    /// Child spawn limit reached — substrate couldn't spawn
    /// `titan-unified-spirit-rs` (or future real
    /// `titan-unified-spirit-rs`).
    ChildLimitReached = 6,
    /// `prctl(PR_SET_PDEATHSIG, SIGTERM)` syscall failed at boot.
    PdeathSigFailure = 7,
    /// Adoption rejected (unused in C-S3 — substrate has no spawn-mode peer
    /// today; reserved for parity with B.2.1 supervision-transfer).
    AdoptionRejected = 8,
    /// Supervisor self-terminated after escalation `terminate` decision.
    /// Per SPEC §15 + §11.B.1 step 6b — kernel cascades a fresh substrate
    /// when this code is observed.
    SupervisorSelfTerminate = 64,
    /// Boot didn't reach `MODULE_READY` within budget. SPEC §15 supervision
    /// range (64..=127). Was 64 pre-C-S7; bumped to 65 to free 64 for the
    /// canonical SupervisorSelfTerminate semantic per SPEC §15.
    BootFailure = 65,
}

impl SubstrateExitCode {
    /// Convert to a [`std::process::ExitCode`] for use as `main()` return.
    pub fn to_exit_code(self) -> ExitCode {
        ExitCode::from(self as u8)
    }

    /// Human-readable label for log lines + diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            SubstrateExitCode::Clean => "clean",
            SubstrateExitCode::Generic => "generic",
            SubstrateExitCode::ConfigError => "config_error",
            SubstrateExitCode::IdentityError => "identity_error",
            SubstrateExitCode::BusConnectFailure => "bus_connect_failure",
            SubstrateExitCode::ShmOpenFailure => "shm_open_failure",
            SubstrateExitCode::ChildLimitReached => "child_limit_reached",
            SubstrateExitCode::PdeathSigFailure => "pdeathsig_failure",
            SubstrateExitCode::AdoptionRejected => "adoption_rejected",
            SubstrateExitCode::SupervisorSelfTerminate => "supervisor_self_terminate",
            SubstrateExitCode::BootFailure => "boot_failure",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_spec_15_canonical_taxonomy() {
        assert_eq!(SubstrateExitCode::Clean as u8, 0);
        assert_eq!(SubstrateExitCode::Generic as u8, 1);
        assert_eq!(SubstrateExitCode::ConfigError as u8, 2);
        assert_eq!(SubstrateExitCode::IdentityError as u8, 3);
        assert_eq!(SubstrateExitCode::BusConnectFailure as u8, 4);
        assert_eq!(SubstrateExitCode::ShmOpenFailure as u8, 5);
        assert_eq!(SubstrateExitCode::ChildLimitReached as u8, 6);
        assert_eq!(SubstrateExitCode::PdeathSigFailure as u8, 7);
        assert_eq!(SubstrateExitCode::AdoptionRejected as u8, 8);
        assert_eq!(SubstrateExitCode::SupervisorSelfTerminate as u8, 64);
        assert_eq!(SubstrateExitCode::BootFailure as u8, 65);
    }

    #[test]
    fn as_str_distinct_per_variant() {
        let labels = [
            SubstrateExitCode::Clean,
            SubstrateExitCode::Generic,
            SubstrateExitCode::ConfigError,
            SubstrateExitCode::IdentityError,
            SubstrateExitCode::BusConnectFailure,
            SubstrateExitCode::ShmOpenFailure,
            SubstrateExitCode::ChildLimitReached,
            SubstrateExitCode::PdeathSigFailure,
            SubstrateExitCode::AdoptionRejected,
            SubstrateExitCode::SupervisorSelfTerminate,
            SubstrateExitCode::BootFailure,
        ]
        .iter()
        .map(|c| c.as_str())
        .collect::<Vec<_>>();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            labels.len(),
            "exit code labels must be distinct"
        );
    }
}
