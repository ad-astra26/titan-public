//! error — `DaemonError` + `Result<T>` alias for the whole crate.
//!
//! All errors flow up to the daemon binary's `main.rs`, which logs them via
//! `tracing` (SPEC §16) and exits with a stable exit code (per `Cli::exit`
//! convention from C-S2's titan-kernel-rs).

use thiserror::Error;

use titan_state::SlotIoError;

/// Errors during daemon operation. Each variant maps to a stable exit code
/// when raised from `main.rs` (per SPEC §13 per-binary contract — exit codes
/// are part of the supervision contract between kernel and child).
#[derive(Debug, Error)]
pub enum DaemonError {
    /// Failed to open a kernel-created shm slot (file missing, mmap failed,
    /// schema mismatch). Maps to exit code 5 (SHM_OPEN_FAILED).
    #[error("shm slot open failed: {source}")]
    ShmOpen {
        /// Underlying titan-state error.
        #[source]
        source: SlotIoError,
    },

    /// Failed to read a sibling slot — SeqLock retries exhausted, payload
    /// corruption, or schema mismatch. Maps to exit code 6 (SHM_READ_FAILED).
    #[error("shm slot read failed: {source}")]
    ShmRead {
        /// Underlying titan-state error.
        #[source]
        source: SlotIoError,
    },

    /// Failed to write own slot (oversized payload, mmap msync failure).
    /// Maps to exit code 7 (SHM_WRITE_FAILED).
    #[error("shm slot write failed: {source}")]
    ShmWrite {
        /// Underlying titan-state error.
        #[source]
        source: SlotIoError,
    },

    /// A 4-byte float slice did not match the expected dim count
    /// (e.g. payload claims 20 bytes but caller asked for [f32; 5] which
    /// requires exactly 20 bytes — caught earlier; this variant fires only on
    /// payload-size drift detected at decode time).
    #[error(
        "dim count mismatch: expected {expected}D ({expected_bytes}B payload), got {actual_bytes}B"
    )]
    DimMismatch {
        /// Expected dimensionality (5, 15, 30, 45, etc.).
        expected: usize,
        /// Expected payload size in bytes (`expected * 4`).
        expected_bytes: usize,
        /// Actual payload size encountered.
        actual_bytes: usize,
    },

    /// msgpack decode of a B.2.1 ADOPTION payload failed (corrupt vector,
    /// schema mismatch). Maps to exit code 8 (BUS_PROTOCOL_VIOLATION).
    #[error("msgpack decode failed: {0}")]
    MsgpackDecode(String),

    /// msgpack encode of an outbound payload failed.
    #[error("msgpack encode failed: {0}")]
    MsgpackEncode(String),
}

/// Convenience `Result<T, DaemonError>` alias.
pub type DaemonResult<T> = Result<T, DaemonError>;

impl DaemonError {
    /// Stable exit code for this error class. Used by `main.rs` when
    /// converting an error to a process exit (SPEC §13).
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::ShmOpen { .. } => 5,
            Self::ShmRead { .. } => 6,
            Self::ShmWrite { .. } => 7,
            Self::MsgpackDecode(_) | Self::MsgpackEncode(_) => 8,
            Self::DimMismatch { .. } => 9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_stable_per_variant() {
        let dim_err = DaemonError::DimMismatch {
            expected: 5,
            expected_bytes: 20,
            actual_bytes: 24,
        };
        assert_eq!(dim_err.exit_code(), 9);
        assert_eq!(
            DaemonError::MsgpackEncode("x".into()).exit_code(),
            DaemonError::MsgpackDecode("y".into()).exit_code()
        );
    }

    #[test]
    fn dim_mismatch_message_includes_byte_counts() {
        let err = DaemonError::DimMismatch {
            expected: 5,
            expected_bytes: 20,
            actual_bytes: 24,
        };
        let msg = format!("{err}");
        assert!(msg.contains("5D"));
        assert!(msg.contains("20B"));
        assert!(msg.contains("24B"));
    }
}
