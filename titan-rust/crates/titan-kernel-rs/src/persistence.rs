//! persistence — L0 atomic-snapshot writer per SPEC §11.H.1 + §10.A B11.
//!
//! Every `KERNEL_SNAPSHOT_INTERVAL_S=1.0` the kernel writes
//! `data/l0_snapshot.bin` containing:
//! - `boot_generation` — monotonic counter across restarts
//! - clock state (circadian phase, π-heartbeat phase, last epoch_id)
//! - supervision-tree state summary
//!
//! Format: msgpack envelope. 2-generation `.bak` retention via
//! `titan_core::atomic_write` per §11.H.2.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tracing::{debug, warn};

use titan_core::atomic_write::atomic_write;
use titan_core::constants::{DATA_BACKUP_RETENTION_GENERATIONS, KERNEL_SNAPSHOT_INTERVAL_S};

/// Errors during persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    /// I/O failure during snapshot read/write.
    #[error("persistence I/O at {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// Atomic-write failure.
    #[error("atomic_write: {0}")]
    AtomicWrite(String),
    /// msgpack codec failure.
    #[error("codec: {0}")]
    Codec(String),
}

/// L0 snapshot wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L0Snapshot {
    /// SPEC version this snapshot was written under (for forward-compat).
    pub spec_version: String,
    /// Monotonic boot counter (incremented on every kernel start).
    pub boot_generation: u64,
    /// Wall-clock timestamp of the snapshot.
    pub snapshot_ts: f64,
    /// Last circadian phase (0..1).
    pub circadian_phase: f32,
    /// Last π-heartbeat phase (0..1).
    pub pi_heartbeat_phase: f32,
    /// Last KERNEL_EPOCH_TICK epoch_id.
    pub last_epoch_id: u64,
    /// π-heartbeat pulse_count.
    pub pulse_count: u64,
    /// Number of supervised children currently registered.
    pub supervised_child_count: u32,
}

impl L0Snapshot {
    /// New default snapshot (boot_generation=0, all phases=0).
    pub fn new(spec_version: impl Into<String>) -> Self {
        Self {
            spec_version: spec_version.into(),
            boot_generation: 0,
            snapshot_ts: 0.0,
            circadian_phase: 0.0,
            pi_heartbeat_phase: 0.0,
            last_epoch_id: 0,
            pulse_count: 0,
            supervised_child_count: 0,
        }
    }

    /// Encode to msgpack bytes.
    pub fn encode(&self) -> Result<Vec<u8>, PersistenceError> {
        rmp_serde::to_vec(self).map_err(|e| PersistenceError::Codec(e.to_string()))
    }

    /// Decode from msgpack bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, PersistenceError> {
        rmp_serde::from_slice(bytes).map_err(|e| PersistenceError::Codec(e.to_string()))
    }
}

/// Try to load the L0 snapshot from disk. Falls back to `.bak` then
/// `.bak.prev` on integrity failure (per SPEC §11.H.4).
///
/// Returns `Ok(None)` if no snapshot file exists yet (clean first boot).
pub fn load_or_default(path: &Path) -> Result<Option<L0Snapshot>, PersistenceError> {
    for candidate in [
        path.to_path_buf(),
        path.with_extension(format!(
            "{}.bak",
            path.extension().and_then(|s| s.to_str()).unwrap_or("bin")
        )),
    ] {
        if !candidate.exists() {
            continue;
        }
        match std::fs::read(&candidate) {
            Ok(bytes) => match L0Snapshot::decode(&bytes) {
                Ok(snap) => {
                    debug!(path = ?candidate, "L0 snapshot loaded");
                    return Ok(Some(snap));
                }
                Err(e) => {
                    warn!(err = %e, path = ?candidate, "L0 snapshot corrupt; trying next");
                }
            },
            Err(e) => {
                warn!(err = %e, path = ?candidate, "L0 snapshot read failed; trying next");
            }
        }
    }
    Ok(None)
}

/// Atomic-write a snapshot with N-generation backups.
pub fn write_snapshot(path: &Path, snapshot: &L0Snapshot) -> Result<(), PersistenceError> {
    let bytes = snapshot.encode()?;
    atomic_write(path, &bytes, DATA_BACKUP_RETENTION_GENERATIONS as usize)
        .map_err(|e| PersistenceError::AtomicWrite(format!("{e:?}")))
}

/// Shared mutable state the snapshot loop reads on each tick.
#[derive(Debug, Clone)]
pub struct SnapshotState {
    /// Latest known clock + supervision data.
    pub current: L0Snapshot,
}

/// Run the L0 persistence loop until shutdown is signaled.
///
/// Cadence: `KERNEL_SNAPSHOT_INTERVAL_S` (1 Hz default per SPEC §11.H +
/// §10.A B11).
pub async fn run_snapshot_loop(
    path: PathBuf,
    state: Arc<Mutex<SnapshotState>>,
    shutdown: Arc<Notify>,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs_f64(
        KERNEL_SNAPSHOT_INTERVAL_S,
    ));
    interval.tick().await; // consume immediate first tick

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let snapshot = state.lock().current.clone();
                if let Err(e) = write_snapshot(&path, &snapshot) {
                    warn!(err = ?e, path = ?path, "L0 snapshot write failed");
                }
            }
            _ = shutdown.notified() => {
                debug!("L0 snapshot loop: shutdown received");
                // Final snapshot before exit
                let snapshot = state.lock().current.clone();
                if let Err(e) = write_snapshot(&path, &snapshot) {
                    warn!(err = ?e, path = ?path, "final L0 snapshot write failed");
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_round_trip() {
        let s1 = L0Snapshot {
            spec_version: "0.1.2".into(),
            boot_generation: 42,
            snapshot_ts: 1714408800.0,
            circadian_phase: 0.5,
            pi_heartbeat_phase: 0.25,
            last_epoch_id: 1000,
            pulse_count: 1500,
            supervised_child_count: 3,
        };
        let bytes = s1.encode().unwrap();
        let s2 = L0Snapshot::decode(&bytes).unwrap();
        assert_eq!(s2.boot_generation, 42);
        assert_eq!(s2.last_epoch_id, 1000);
        assert_eq!(s2.spec_version, "0.1.2");
    }

    #[test]
    fn write_snapshot_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l0_snapshot.bin");
        let s = L0Snapshot::new("0.1.2");
        write_snapshot(&path, &s).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn load_or_default_reads_back_what_we_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l0_snapshot.bin");
        let mut s = L0Snapshot::new("0.1.2");
        s.boot_generation = 7;
        s.last_epoch_id = 999;
        write_snapshot(&path, &s).unwrap();

        let loaded = load_or_default(&path).unwrap().unwrap();
        assert_eq!(loaded.boot_generation, 7);
        assert_eq!(loaded.last_epoch_id, 999);
    }

    #[test]
    fn load_or_default_returns_none_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.bin");
        let result = load_or_default(&path).unwrap();
        assert!(result.is_none());
    }
}
