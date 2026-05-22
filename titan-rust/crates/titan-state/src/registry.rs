//! registry — Boot-time creation of all kernel-owned shm slots.
//!
//! Per SPEC §10.A B3 step (T+15ms→T+50ms boot budget): kernel creates
//! `/dev/shm/titan_<id>/` directory + every slot whose creator is `Kernel`
//! per [`crate::spec::SLOT_SPECS`].
//!
//! # D04 symlink (SPEC §3.1 D04)
//!
//! After creating `self_162d.bin`, the registry creates a symlink
//! `trinity_state.bin → self_162d.bin` so legacy Python consumers reading
//! the old name still resolve to the canonical file. The symlink is
//! deleted in C-S8 cleanup once all consumers migrate.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::slot::{Slot, SlotIoError};
use crate::spec::{kernel_slots, SlotSpec};

/// Errors during registry boot.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Failed to create or set permissions on the shm root directory.
    #[error("shm dir setup failed at {path}: {source}")]
    DirSetup {
        /// Directory path attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Failed to create one of the spec'd slots.
    #[error("slot creation failed for {slot_name}: {source}")]
    SlotCreate {
        /// Slot filename.
        slot_name: &'static str,
        /// Underlying slot error.
        source: SlotIoError,
    },

    /// Failed to create the D04 symlink.
    #[error("symlink trinity_state.bin → self_162d.bin failed: {0}")]
    SymlinkFailed(std::io::Error),
}

/// Top-level registry handle. Holds the open `Slot` handles for kernel-
/// created slots so the kernel boot path can write initial values without
/// re-opening.
pub struct SlotRegistry {
    /// Root shm directory (e.g. `/dev/shm/titan_T1/`).
    pub shm_dir: PathBuf,
    /// Open slot handles, one per kernel-created slot.
    pub slots: Vec<(SlotSpec, Slot)>,
}

impl SlotRegistry {
    /// Create the shm directory + every kernel-owned slot per `SLOT_SPECS`.
    /// Idempotent: stale files from previous boots are unlinked first
    /// (per SPEC §7.3 — `/dev/shm/` is tmpfs and gets re-created on every
    /// kernel boot anyway).
    pub fn create_all(shm_dir: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let shm_dir = shm_dir.as_ref().to_path_buf();

        // 1. Ensure shm_dir exists with mode 0700
        std::fs::create_dir_all(&shm_dir).map_err(|e| RegistryError::DirSetup {
            path: shm_dir.clone(),
            source: e,
        })?;
        std::fs::set_permissions(&shm_dir, std::fs::Permissions::from_mode(0o700)).map_err(
            |e| RegistryError::DirSetup {
                path: shm_dir.clone(),
                source: e,
            },
        )?;

        // 2. Create each kernel slot
        let mut slots = Vec::with_capacity(kernel_slots().count());
        for spec in kernel_slots() {
            let slot_path = shm_dir.join(spec.name);

            // Idempotent-failsafe: unlink stale file if present
            if slot_path.exists() {
                let _ = std::fs::remove_file(&slot_path);
            }

            // fastbus.bin is a self-contained SPSC ring (per SPEC §7.1 note);
            // it does NOT use the universal §7.0 triple-buffer header. The
            // titan-fastbus crate validates its own file size + writes its
            // own ring header at attach time. Allocate a flat zero-filled
            // file of the size titan-fastbus expects and skip universal-slot
            // wrapping. We do NOT track fastbus.bin in `slots` because it
            // has no Slot handle (Slot::open would fail on the size).
            if spec.name == "fastbus.bin" {
                Self::create_raw_file(&slot_path, spec.fastbus_total_bytes())?;
                debug!(
                    slot = "fastbus.bin",
                    "created raw SPSC fastbus file (excluded from §7.0 universal slot wrapping)"
                );
                continue;
            }

            let slot = Slot::create(&slot_path, spec.schema_version, spec.payload_bytes).map_err(
                |source| RegistryError::SlotCreate {
                    slot_name: spec.name,
                    source,
                },
            )?;
            slots.push((*spec, slot));
            debug!(slot = %spec.name, "created shm slot");
        }

        // 3. Create D04 symlink trinity_state.bin → self_162d.bin
        Self::create_d04_symlink(&shm_dir)?;

        info!(
            dir = ?shm_dir,
            kernel_slot_count = slots.len(),
            "shm slot registry initialized"
        );
        Ok(Self { shm_dir, slots })
    }

    /// Allocate a flat zero-filled file of exact size, with mode 0600.
    /// Used for slots that do NOT use the universal §7.0 triple-buffer
    /// header (currently only `fastbus.bin` — self-contained SPSC ring).
    /// Atomic via tmp + rename.
    fn create_raw_file(path: &Path, total_bytes: u64) -> Result<(), RegistryError> {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp_path = std::path::PathBuf::from(tmp);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| RegistryError::SlotCreate {
                slot_name: "fastbus.bin",
                source: SlotIoError::Io {
                    path: tmp_path.clone(),
                    source: e,
                },
            })?;
        file.set_len(total_bytes)
            .map_err(|e| RegistryError::SlotCreate {
                slot_name: "fastbus.bin",
                source: SlotIoError::Io {
                    path: tmp_path.clone(),
                    source: e,
                },
            })?;
        drop(file);
        std::fs::rename(&tmp_path, path).map_err(|e| RegistryError::SlotCreate {
            slot_name: "fastbus.bin",
            source: SlotIoError::Io {
                path: path.to_path_buf(),
                source: e,
            },
        })?;
        Ok(())
    }

    /// Create the D04 symlink. Per SPEC §3.1 D04 — Phase C transition only;
    /// C-S8 deletes the symlink once consumers migrate to canonical name.
    pub fn create_d04_symlink(shm_dir: &Path) -> Result<(), RegistryError> {
        let canonical = shm_dir.join("self_162d.bin");
        let legacy = shm_dir.join("trinity_state.bin");

        // Sanity: canonical must exist (we create it before the symlink)
        if !canonical.exists() {
            return Err(RegistryError::SymlinkFailed(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "self_162d.bin missing — symlink target invalid",
            )));
        }

        // Remove stale symlink/file if present
        if legacy.exists() || legacy.is_symlink() {
            let _ = std::fs::remove_file(&legacy);
        }

        // Create relative symlink (self_162d.bin in same dir)
        std::os::unix::fs::symlink("self_162d.bin", &legacy)
            .map_err(RegistryError::SymlinkFailed)?;

        debug!(
            canonical = ?canonical,
            legacy = ?legacy,
            "D04 symlink created"
        );
        Ok(())
    }

    /// Returns a mutable handle to a kernel-created slot by its filename.
    /// Used by the kernel boot path to write initial values (e.g. circadian
    /// phase=0, π-heartbeat phase=0).
    pub fn slot_mut(&mut self, name: &str) -> Option<&mut Slot> {
        self.slots
            .iter_mut()
            .find(|(spec, _)| spec.name == name)
            .map(|(_, slot)| slot)
    }

    /// Number of kernel-created slots in the registry.
    pub fn count(&self) -> usize {
        self.slots.len()
    }

    /// Try to clean up: flush all slots + remove the shm directory. Used
    /// during graceful shutdown OR test teardown. Errors are logged but
    /// not returned (best-effort).
    pub fn cleanup(&self) {
        for (spec, slot) in &self.slots {
            if let Err(e) = slot.flush() {
                warn!(slot = %spec.name, err = ?e, "flush failed during cleanup");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn create_all_creates_shm_dir() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let _ = SlotRegistry::create_all(&shm_dir).unwrap();
        assert!(shm_dir.exists());
    }

    #[test]
    fn create_all_creates_16_kernel_slots() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let registry = SlotRegistry::create_all(&shm_dir).unwrap();
        // fastbus.bin is excluded from the universal-slot count (raw SPSC ring,
        // no Slot handle held in registry.slots). All other kernel-created
        // slots count = KERNEL_CREATED_COUNT - 1.
        assert_eq!(registry.count(), crate::spec::KERNEL_CREATED_COUNT - 1);
    }

    #[test]
    fn create_all_creates_self_162d_with_correct_size() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let _ = SlotRegistry::create_all(&shm_dir).unwrap();
        let path = shm_dir.join("self_162d.bin");
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 16 + 3 * (16 + 648)); // §7.0 v1.0.0: 16-byte fixed header + 3 × (16 buffer meta + 162 × float32)
    }

    #[test]
    fn create_all_creates_d04_symlink() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let _ = SlotRegistry::create_all(&shm_dir).unwrap();
        let symlink = shm_dir.join("trinity_state.bin");
        let canonical = shm_dir.join("self_162d.bin");
        assert!(symlink.is_symlink());
        // Both paths should refer to the same file content
        let canonical_meta = std::fs::metadata(&canonical).unwrap();
        let symlink_meta = std::fs::metadata(&symlink).unwrap();
        assert_eq!(canonical_meta.len(), symlink_meta.len());
    }

    #[test]
    fn create_all_idempotent_unlinks_stale() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        // Create once
        let _ = SlotRegistry::create_all(&shm_dir).unwrap();
        // Tamper with one slot — write garbage
        let path = shm_dir.join("circadian.bin");
        std::fs::write(&path, b"garbage").unwrap();
        // Re-create — should unlink + recreate clean
        let registry = SlotRegistry::create_all(&shm_dir).unwrap();
        // fastbus.bin is excluded from the universal-slot count (raw SPSC ring,
        // no Slot handle held in registry.slots). All other kernel-created
        // slots count = KERNEL_CREATED_COUNT - 1.
        assert_eq!(registry.count(), crate::spec::KERNEL_CREATED_COUNT - 1);
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 16 + 3 * (16 + 12)); // §7.0 v1.0.0: 16-byte fixed header + 3 × (16 buffer meta + 12 circadian payload)
    }

    #[test]
    fn shm_dir_mode_0700() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let _ = SlotRegistry::create_all(&shm_dir).unwrap();
        let mode = std::fs::metadata(&shm_dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn slot_mut_finds_kernel_slot_by_name() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let mut registry = SlotRegistry::create_all(&shm_dir).unwrap();
        let slot = registry.slot_mut("epoch_counter.bin");
        assert!(slot.is_some());
    }

    #[test]
    fn slot_mut_returns_none_for_python_managed() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let mut registry = SlotRegistry::create_all(&shm_dir).unwrap();
        // neuromod_state is Python-managed; not in registry
        let slot = registry.slot_mut("neuromod_state.bin");
        assert!(slot.is_none());
    }

    #[test]
    fn write_to_kernel_slot_via_registry() {
        let dir = tempdir();
        let shm_dir = dir.path().join("titan_T1");
        let mut registry = SlotRegistry::create_all(&shm_dir).unwrap();
        let slot = registry.slot_mut("epoch_counter.bin").unwrap();
        let payload = 42u64.to_le_bytes();
        slot.write(&payload).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, payload);
    }
}
