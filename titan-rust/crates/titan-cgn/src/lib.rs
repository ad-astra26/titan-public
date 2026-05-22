//! titan-cgn — CGN slot lifecycle for Titan microkernel v2 Phase C.
//!
//! **Per SPEC §18.2 (resolved open question):** Phase C kernel creates ONLY
//! `cgn_live_weights.bin` (variable-size, ≤ 256 KB) at boot. All other CGN
//! slots remain Python-managed (created by `cgn_module` lazily). The kernel
//! does NOT understand CGN semantics — it only owns slot lifecycle for the
//! one CGN slot that's read by many consumers.
//!
//! Per-CGN-domain slots (alpha-beta tables, etc.) stay in Python's slot
//! path until Phase D ports `cgn_worker` to Rust.
//!
//! # Lifecycle
//!
//! ```ignore
//! use titan_cgn::create_cgn_live_weights;
//! let slot = create_cgn_live_weights(shm_dir)?;
//! // slot is now ready; cgn_module (Python) writes to it via mmap
//! ```
//!
//! # SPEC discipline
//!
//! The single CGN slot's byte layout is variable (≤ `CGN_LIVE_WEIGHTS_MAX_BYTES`)
//! per SPEC §7.1. Rust kernel creates the file with the maximum capacity
//! and a SeqLock header; Python writes payload+1 ≤ max_bytes via the same
//! 24-byte SeqLock header per SPEC §7.0.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::path::{Path, PathBuf};

use tracing::info;

use titan_core::constants::CGN_LIVE_WEIGHTS_SCHEMA_VERSION;
use titan_state::{Slot, SlotIoError};

/// Maximum payload bytes for `cgn_live_weights.bin` per SPEC §7.1
/// (variable, ≤ 256 KB; typical 40–50 KB).
pub const CGN_LIVE_WEIGHTS_MAX_BYTES: u32 = 262144;

/// Filename of the lone CGN slot kernel creates.
pub const CGN_LIVE_WEIGHTS_FILENAME: &str = "cgn_live_weights.bin";

/// Errors during CGN slot creation.
#[derive(Debug, thiserror::Error)]
pub enum CgnError {
    /// Slot creation failed.
    #[error("cgn_live_weights.bin creation failed at {path}: {source}")]
    SlotCreate {
        /// Slot path.
        path: PathBuf,
        /// Underlying slot error.
        source: SlotIoError,
    },
}

/// Create the `cgn_live_weights.bin` slot in `shm_dir`. Idempotent: stale
/// files are unlinked first.
///
/// Per SPEC §18.2 + §7.1: variable-size payload up to 256 KB,
/// `schema_version = CGN_LIVE_WEIGHTS_SCHEMA_VERSION = 1` at v0.1.0.
pub fn create_cgn_live_weights(shm_dir: impl AsRef<Path>) -> Result<Slot, CgnError> {
    let path = shm_dir.as_ref().join(CGN_LIVE_WEIGHTS_FILENAME);

    if path.exists() {
        let _ = std::fs::remove_file(&path);
    }

    let slot = Slot::create(
        &path,
        CGN_LIVE_WEIGHTS_SCHEMA_VERSION as u32,
        CGN_LIVE_WEIGHTS_MAX_BYTES,
    )
    .map_err(|source| CgnError::SlotCreate {
        path: path.clone(),
        source,
    })?;

    info!(
        path = ?path,
        capacity_bytes = CGN_LIVE_WEIGHTS_MAX_BYTES,
        "cgn_live_weights slot initialized"
    );

    Ok(slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn creates_cgn_live_weights_slot() {
        let dir = tempdir();
        std::fs::create_dir_all(dir.path()).unwrap();
        let slot = create_cgn_live_weights(dir.path()).unwrap();
        let path = dir.path().join(CGN_LIVE_WEIGHTS_FILENAME);
        assert!(path.exists());
        let meta = std::fs::metadata(&path).unwrap();
        // §7.0 v1.0.0: 16-byte fixed header + 3 × (16-byte buffer meta + 256 KB max payload)
        assert_eq!(
            meta.len(),
            16 + 3 * (16 + CGN_LIVE_WEIGHTS_MAX_BYTES as u64)
        );
        let _ = slot;
    }

    #[test]
    fn cgn_slot_schema_version_matches_constant() {
        let dir = tempdir();
        let slot = create_cgn_live_weights(dir.path()).unwrap();
        let h = slot.header();
        assert_eq!(h.schema_version, CGN_LIVE_WEIGHTS_SCHEMA_VERSION as u32);
    }

    #[test]
    fn idempotent_unlinks_stale() {
        let dir = tempdir();
        // Create a stale file at the path
        let path = dir.path().join(CGN_LIVE_WEIGHTS_FILENAME);
        std::fs::write(&path, b"stale").unwrap();
        // Re-create — should unlink + recreate clean
        let slot = create_cgn_live_weights(dir.path()).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(
            meta.len(),
            16 + 3 * (16 + CGN_LIVE_WEIGHTS_MAX_BYTES as u64)
        );
        let _ = slot;
    }

    #[test]
    fn cgn_slot_writable_within_capacity() {
        let dir = tempdir();
        let mut slot = create_cgn_live_weights(dir.path()).unwrap();
        let payload = vec![0xAB; 50 * 1024]; // 50 KB typical
        slot.write(&payload).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, payload);
    }

    #[test]
    fn cgn_slot_rejects_oversized_payload() {
        let dir = tempdir();
        let mut slot = create_cgn_live_weights(dir.path()).unwrap();
        let too_big = vec![0u8; (CGN_LIVE_WEIGHTS_MAX_BYTES + 1) as usize];
        let result = slot.write(&too_big);
        assert!(matches!(result, Err(SlotIoError::PayloadTooLarge { .. })));
    }
}
