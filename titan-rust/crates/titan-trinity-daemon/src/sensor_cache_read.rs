//! sensor_cache_read — Phase C C-S6 outer-daemon sensor cache reader.
//!
//! Outer trinity daemons (titan-outer-{body,mind,spirit}-rs) consume
//! Python-written sensor caches per SPEC §9.D — variable-size
//! msgpack-encoded source dicts at:
//!
//!   /dev/shm/titan_<id>/sensor_cache_outer_body.bin    (≤ 8192 B msgpack)
//!   /dev/shm/titan_<id>/sensor_cache_outer_mind.bin    (≤ 8192 B msgpack)
//!   /dev/shm/titan_<id>/sensor_cache_outer_spirit.bin  (≤ 8192 B msgpack)
//!
//! These slots use the same 24-byte SeqLock header as fixed-size slots
//! (per SPEC §7.0); only the payload is variable + msgpack-encoded.
//!
//! # Stale check (SPEC §18.1 line 1867)
//!
//! Daemon-side: if `wall_ns < now − OUTER_CACHE_STALE_CADENCE_MULTIPLIER ×
//! cadence` the cache is treated as STALE; the daemon writes its slot
//! with last-known dims + emits structured-log line `confidence=0.0`.
//!
//! Per cadence in SPEC §18.1 + D-SPEC-100 (G13 1:3:9, spirit fastest):
//!
//!   - outer_body : stale at 135s (3 × OUTER_BODY_TICK_BASE_S = 45 s)
//!   - outer_mind : stale at  45s (3 × OUTER_MIND_TICK_BASE_S = 15 s)
//!   - outer_spirit: stale at 15s (3 × OUTER_SPIRIT_TICK_BASE_S =  5 s)
//!
//! # API
//!
//! [`read_sensor_cache`] returns a [`SensorCacheRead`] enum:
//!   - `Fresh { payload, age_s, wall_ns }` — slot is fresh; payload is
//!     msgpack-encoded source dict for caller to decode.
//!   - `Stale { age_s, wall_ns }` — slot is past `stale_threshold_s`;
//!     caller falls back to last-known + emits confidence=0.0 log.
//!   - `Missing` — slot file doesn't exist (cold boot before sidecar runs).
//!
//! All variants are non-fatal — outer daemons NEVER crash on cache
//! issues (per SPEC §3.0 graceful degradation discipline).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use titan_state::Slot;

use crate::error::DaemonError;

/// Outcome of a sensor cache read.
#[derive(Debug, Clone)]
pub enum SensorCacheRead {
    /// Cache is fresh (`now - wall_ns < stale_threshold_s`). Payload is
    /// msgpack-encoded raw bytes — caller decodes per its own schema
    /// (e.g. [`OuterBodySources`] / [`OuterMindSources`] /
    /// [`OuterSpiritSources`]).
    Fresh {
        /// msgpack-encoded source dict (length ≤ slot's max payload).
        payload: Vec<u8>,
        /// Age of the cache when read, in seconds.
        age_s: f64,
        /// Cache's last-write wall clock (nanoseconds since UNIX epoch).
        wall_ns: u64,
    },
    /// Cache is past the stale threshold (Python sidecar likely
    /// crashed or is far behind). Daemon falls back to last-known.
    Stale {
        /// Age of the cache when read, in seconds (greater than threshold).
        age_s: f64,
        /// Cache's last-write wall clock.
        wall_ns: u64,
    },
    /// Slot file does not exist yet (cold boot before sidecar's first write).
    Missing,
}

/// Read a variable-size msgpack-encoded sensor cache slot, applying the
/// SPEC §18.1 staleness threshold.
///
/// `slot_path` — absolute path to the cache slot file.
/// `stale_threshold_s` — daemon's `OUTER_CACHE_STALE_CADENCE_MULTIPLIER ×
/// natural_cadence_s` (e.g. `30.0` for outer_body).
///
/// On any I/O error other than "file missing" returns
/// [`DaemonError::ShmRead`] / [`DaemonError::ShmOpen`] — caller should
/// treat as `Stale` for safety. Common case is a clean read returning
/// either `Fresh` or `Stale`.
pub fn read_sensor_cache(
    slot_path: impl AsRef<Path>,
    stale_threshold_s: f64,
) -> Result<SensorCacheRead, DaemonError> {
    let path = slot_path.as_ref();
    if !path.exists() {
        return Ok(SensorCacheRead::Missing);
    }
    let slot = Slot::open(path).map_err(|source| DaemonError::ShmOpen { source })?;
    // §7.0 v1.0.0: wall_ns lives in per-buffer metadata, not the fixed header.
    // read_with_meta returns (payload, wall_ns) atomically with the publish.
    let (payload, wall_ns) = slot
        .read_with_meta()
        .map_err(|source| DaemonError::ShmRead { source })?;
    let now_ns = current_wall_ns();
    let age_s = age_seconds(now_ns, wall_ns);
    if age_s > stale_threshold_s {
        Ok(SensorCacheRead::Stale { age_s, wall_ns })
    } else {
        Ok(SensorCacheRead::Fresh {
            payload,
            age_s,
            wall_ns,
        })
    }
}

/// Return the current wall clock as nanoseconds since UNIX epoch.
/// Mirrors Python's `time.time_ns()` so wall_ns is comparable across
/// the language boundary.
pub fn current_wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Compute age in seconds, returning 0.0 when `now < wall_ns` (clock
/// skew across writer/reader processes — extremely rare on a single
/// host, but be defensive).
pub fn age_seconds(now_ns: u64, wall_ns: u64) -> f64 {
    if now_ns <= wall_ns {
        return 0.0;
    }
    (now_ns - wall_ns) as f64 / 1.0e9
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tmp slot file with the given payload.
    /// Returns the slot's path (in a tempdir kept alive by the caller).
    /// Uses `Slot::create(path, schema_version, max_payload_bytes)` per the
    /// titan-state API; the actual write payload may be ≤ max_payload_bytes.
    fn make_slot_with(
        dir: &tempfile::TempDir,
        name: &str,
        max_payload: u32,
        payload: &[u8],
    ) -> std::path::PathBuf {
        let path = dir.path().join(format!("{}.bin", name));
        let mut slot = Slot::create(&path, 1u32, max_payload).unwrap();
        slot.write(payload).unwrap();
        path
    }

    #[test]
    fn missing_slot_returns_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.bin");
        let result = read_sensor_cache(&path, 30.0).unwrap();
        assert!(matches!(result, SensorCacheRead::Missing));
    }

    #[test]
    fn fresh_slot_returns_payload() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x42];
        let path = make_slot_with(&dir, "sensor_cache_outer_body", 8192, &payload);

        let result = read_sensor_cache(&path, 30.0).unwrap();
        match result {
            SensorCacheRead::Fresh {
                payload: got,
                age_s,
                wall_ns,
            } => {
                assert_eq!(got, payload);
                assert!(age_s >= 0.0);
                assert!(age_s < 1.0, "fresh slot age {}s should be < 1s", age_s);
                assert!(wall_ns > 0);
            }
            other => panic!("expected Fresh, got {:?}", other),
        }
    }

    #[test]
    fn empty_payload_is_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_slot_with(&dir, "sensor_cache_outer_body", 8192, &[]);
        let result = read_sensor_cache(&path, 30.0).unwrap();
        assert!(matches!(result, SensorCacheRead::Fresh { .. }));
    }

    #[test]
    fn stale_threshold_zero_always_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_slot_with(&dir, "sensor_cache_outer_mind", 8192, &[0x01, 0x02]);
        // threshold of 0 ⇒ ANY positive age is stale (microsecond-scale)
        // Some test hosts may report exactly 0 for sub-microsecond ages;
        // sleep briefly to ensure age > 0.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let result = read_sensor_cache(&path, 0.0).unwrap();
        assert!(
            matches!(result, SensorCacheRead::Stale { .. }),
            "expected Stale with threshold=0 + 2ms age, got {:?}",
            result,
        );
    }

    #[test]
    fn age_seconds_zero_when_now_le_wall() {
        assert_eq!(age_seconds(0, 100), 0.0);
        assert_eq!(age_seconds(100, 100), 0.0);
        assert_eq!(age_seconds(99, 100), 0.0); // clock skew defensive
    }

    #[test]
    fn age_seconds_positive_difference() {
        // 1 second = 1_000_000_000 ns
        assert!((age_seconds(2_000_000_000, 1_000_000_000) - 1.0).abs() < 1e-9);
        // 0.5 s
        assert!((age_seconds(1_500_000_000, 1_000_000_000) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn current_wall_ns_is_positive_and_recent() {
        let ns = current_wall_ns();
        // Lower bound: 2026-01-01 UTC = 1767225600 s × 1e9 ns
        let early_2026_ns: u64 = 1_767_225_600 * 1_000_000_000;
        assert!(ns > early_2026_ns, "wall_ns {} too early", ns);
    }
}
