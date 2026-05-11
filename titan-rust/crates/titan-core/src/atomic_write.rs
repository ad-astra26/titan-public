//! atomic_write — Atomic-write helper with N-generation backup rotation.
//!
//! Per SPEC §11.H.2 (Data Integrity Invariants). Used by every Phase C
//! critical-data writer. Mandatory for all paths in `SPEC §11.H.1`.
//!
//! # Algorithm
//!
//! 1. Write `<path>.tmp` with full data
//! 2. `fsync(tmp_fd)` — guarantee data hits disk before rename
//! 3. If `<path>` exists and `keep_backups > 0`:
//!    - rotate `<path>.bak.prev` → DELETE
//!    - rotate `<path>.bak` → `<path>.bak.prev`
//!    - rotate `<path>` → `<path>.bak`
//! 4. `rename(<path>.tmp, <path>)` — POSIX atomic
//! 5. `fsync(parent_dir_fd)` — commit rename to disk
//!
//! # Forbidden patterns
//!
//! Per SPEC §11.H.2: direct `File::create()` / `std::fs::write()` /
//! `OpenOptions::create(true).write(true).open()` on any §11.H.1
//! critical-data path is a SPEC violation. arch_map static-checks codebase.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub use crate::constants::DATA_BACKUP_RETENTION_GENERATIONS;

/// Errors during atomic write.
#[derive(Debug, thiserror::Error)]
pub enum AtomicWriteError {
    /// I/O error during write/rename/fsync.
    #[error("atomic_write I/O at {path}: {source}")]
    Io {
        /// Path where the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Provided path has no parent directory (cannot fsync).
    #[error("path has no parent directory: {0}")]
    NoParent(PathBuf),
}

impl AtomicWriteError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Atomic-write helper.
///
/// # Arguments
///
/// - `path` — final destination path (canonical name)
/// - `data` — bytes to write
/// - `keep_backups` — number of backup generations to retain
///   (`DATA_BACKUP_RETENTION_GENERATIONS=2` is canonical for SPEC §11.H.1
///   files; `0` = no rotation, just atomic write)
///
/// # Errors
///
/// Returns `AtomicWriteError` on any I/O failure. Caller should treat
/// failure as critical (per SPEC §15 exit code 1 generic error in kernel
/// context).
pub fn atomic_write(path: &Path, data: &[u8], keep_backups: usize) -> Result<(), AtomicWriteError> {
    let tmp = tmp_path(path);

    // 1. Write tmp file
    {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .map_err(|e| AtomicWriteError::io(&tmp, e))?;
        file.write_all(data)
            .map_err(|e| AtomicWriteError::io(&tmp, e))?;
        file.sync_all().map_err(|e| AtomicWriteError::io(&tmp, e))?;
    }

    // 2. Rotate backups if requested + path already exists
    if keep_backups > 0 && path.exists() {
        rotate_backups(path, keep_backups)?;
    }

    // 3. POSIX atomic rename
    fs::rename(&tmp, path).map_err(|e| AtomicWriteError::io(path, e))?;

    // 4. fsync parent dir to commit the rename
    let parent = path
        .parent()
        .ok_or_else(|| AtomicWriteError::NoParent(path.to_path_buf()))?;
    let dir_fd = fs::File::open(parent).map_err(|e| AtomicWriteError::io(parent, e))?;
    dir_fd
        .sync_all()
        .map_err(|e| AtomicWriteError::io(parent, e))?;

    Ok(())
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

fn bak_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".bak");
    PathBuf::from(s)
}

fn bak_prev_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".bak.prev");
    PathBuf::from(s)
}

fn rotate_backups(path: &Path, keep_backups: usize) -> Result<(), AtomicWriteError> {
    debug_assert!(keep_backups > 0);
    let bak = bak_path(path);
    let bak_prev = bak_prev_path(path);

    // Rotate: .bak.prev → DELETE; .bak → .bak.prev; current → .bak
    if keep_backups >= 2 && bak_prev.exists() {
        // Sliding window: drop oldest
        fs::remove_file(&bak_prev).map_err(|e| AtomicWriteError::io(&bak_prev, e))?;
    }
    if keep_backups >= 2 && bak.exists() {
        fs::rename(&bak, &bak_prev).map_err(|e| AtomicWriteError::io(&bak, e))?;
    }
    fs::rename(path, &bak).map_err(|e| AtomicWriteError::io(path, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn writes_new_file() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"hello", 2).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn overwrites_creates_bak() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"first", 2).unwrap();
        atomic_write(&path, b"second", 2).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        assert_eq!(fs::read(bak_path(&path)).unwrap(), b"first");
    }

    #[test]
    fn three_writes_create_bak_and_bak_prev() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"v1", 2).unwrap();
        atomic_write(&path, b"v2", 2).unwrap();
        atomic_write(&path, b"v3", 2).unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"v3");
        assert_eq!(fs::read(bak_path(&path)).unwrap(), b"v2");
        assert_eq!(fs::read(bak_prev_path(&path)).unwrap(), b"v1");
    }

    #[test]
    fn fourth_write_drops_oldest() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"v1", 2).unwrap();
        atomic_write(&path, b"v2", 2).unwrap();
        atomic_write(&path, b"v3", 2).unwrap();
        atomic_write(&path, b"v4", 2).unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"v4");
        assert_eq!(fs::read(bak_path(&path)).unwrap(), b"v3");
        assert_eq!(fs::read(bak_prev_path(&path)).unwrap(), b"v2");
        // v1 dropped
    }

    #[test]
    fn keep_backups_zero_no_rotation() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"first", 0).unwrap();
        atomic_write(&path, b"second", 0).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        // No .bak created
        assert!(!bak_path(&path).exists());
    }

    #[test]
    fn no_tmp_file_left_behind() {
        let dir = tempdir();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"hello", 2).unwrap();
        assert!(!tmp_path(&path).exists());
    }

    #[test]
    fn data_backup_retention_constant_matches_spec() {
        // SPEC §11.H.2 + §3.D04 + canonical TOML
        assert_eq!(DATA_BACKUP_RETENTION_GENERATIONS, 2);
    }
}
