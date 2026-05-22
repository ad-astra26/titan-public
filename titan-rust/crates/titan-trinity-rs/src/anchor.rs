//! anchor — `data/anchor_state.json` reader with 60s freshness cache.
//!
//! Closes rFP_phase_c_substrate_observable_closure.md §2.2: previously the
//! Rust substrate hardcoded `anchor_factor = 1.0` regardless of whether
//! `data/anchor_state.json` existed. Python's `topology.py:288-302` reads
//! the file each call; this Rust port re-reads on a 60s cadence (anchor
//! freshness has sub-minute granularity per SPEC §11.D).
//!
//! # Anchor factor semantics (matches topology.py:298-299)
//!
//! - Fresh anchor (just landed, _since = 0): `factor = 0.5` (low grounding tension)
//! - 300s ago:                                `factor = 0.75`
//! - ≥600s ago / no file / parse error:       `factor = 1.0` (full tension)
//!
//! The factor modulates `grounding_tension` at `topology.rs::compute_whole_10d`
//! position [6] of the WHOLE-10D synthesis.

use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Re-read cadence — 60s matches sub-minute anchor freshness per SPEC §11.D.
const READ_CADENCE_SECS: u64 = 60;

/// Filename fixed at SPEC convention; relative to `data_dir` passed at boot.
const ANCHOR_FILE_NAME: &str = "anchor_state.json";

/// Default factor when no file / parse error — full grounding tension
/// (matches Python `anchor_factor = 1.0` fallback at topology.py:289).
const DEFAULT_FACTOR: f32 = 1.0;

#[derive(Debug, Deserialize)]
struct AnchorStateFile {
    /// Unix timestamp seconds since last on-chain anchor commit. Optional —
    /// absent fields default per `serde(default)`.
    #[serde(default)]
    last_anchor_time: f64,
}

/// 60s-cached reader for `data/anchor_state.json`. Construct once per
/// `SubstrateState`; call `factor()` per body tick.
#[derive(Debug)]
pub struct AnchorReader {
    path: PathBuf,
    cache: Mutex<CacheEntry>,
}

#[derive(Debug)]
struct CacheEntry {
    factor: f32,
    last_read: Option<Instant>,
}

impl AnchorReader {
    /// Build a reader rooted at `data_dir/anchor_state.json`.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            path: data_dir.join(ANCHOR_FILE_NAME),
            cache: Mutex::new(CacheEntry {
                factor: DEFAULT_FACTOR,
                last_read: None,
            }),
        }
    }

    /// Returns the current anchor_factor in [0.5, 1.0]. Re-reads at most
    /// once per `READ_CADENCE_SECS`; otherwise returns cached value.
    pub fn factor(&self) -> f32 {
        let now = Instant::now();
        let mut guard = self.cache.lock().expect("anchor cache lock poisoned");
        let needs_refresh = match guard.last_read {
            None => true,
            Some(t) => now.duration_since(t).as_secs() >= READ_CADENCE_SECS,
        };
        if needs_refresh {
            guard.factor = read_factor_from_disk(&self.path);
            guard.last_read = Some(now);
        }
        guard.factor
    }
}

impl Clone for AnchorReader {
    /// Clone re-initializes the cache (Mutex isn't Clone). Fresh cache means
    /// the first `factor()` call on the clone re-reads from disk, which is
    /// the conservative-correct behavior for state forking (e.g., during
    /// shadow-swap snapshots).
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            cache: Mutex::new(CacheEntry {
                factor: DEFAULT_FACTOR,
                last_read: None,
            }),
        }
    }
}

/// One-shot disk read; failure-tolerant. Returns `DEFAULT_FACTOR` on any
/// error (file missing, parse failure, system clock skew). Matches
/// `topology.py:288-302` swallow-on-exception semantics.
fn read_factor_from_disk(path: &PathBuf) -> f32 {
    let Ok(raw) = fs::read_to_string(path) else {
        return DEFAULT_FACTOR;
    };
    let Ok(state): Result<AnchorStateFile, _> = serde_json::from_str(&raw) else {
        return DEFAULT_FACTOR;
    };
    if state.last_anchor_time <= 0.0 {
        return DEFAULT_FACTOR;
    }
    let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return DEFAULT_FACTOR;
    };
    let now_s = now.as_secs_f64();
    let since_s = (now_s - state.last_anchor_time).max(0.0);

    // Match topology.py:299:
    //   anchor_factor = 0.5 + 0.5 * min(1.0, _since / 600.0)
    let ratio = (since_s / 600.0).min(1.0) as f32;
    0.5 + 0.5 * ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_anchor(dir: &std::path::Path, ts: f64) {
        let path = dir.join(ANCHOR_FILE_NAME);
        let mut f = fs::File::create(&path).expect("create anchor file");
        writeln!(f, r#"{{"last_anchor_time": {ts}}}"#).expect("write anchor file");
    }

    #[test]
    fn missing_file_returns_default_factor() {
        let dir = tempdir().unwrap();
        let r = AnchorReader::new(dir.path().to_path_buf());
        assert!((r.factor() - DEFAULT_FACTOR).abs() < 1e-6);
    }

    #[test]
    fn parse_error_returns_default_factor() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(ANCHOR_FILE_NAME);
        fs::write(&path, "not json").unwrap();
        let r = AnchorReader::new(dir.path().to_path_buf());
        assert!((r.factor() - DEFAULT_FACTOR).abs() < 1e-6);
    }

    #[test]
    fn fresh_anchor_returns_low_factor() {
        let dir = tempdir().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        // Very recent anchor (~now): factor ≈ 0.5
        write_anchor(dir.path(), now);
        let r = AnchorReader::new(dir.path().to_path_buf());
        let f = r.factor();
        assert!(f >= 0.5);
        assert!(f < 0.55, "fresh-anchor factor = {f}, expected ~0.5");
    }

    #[test]
    fn stale_anchor_returns_full_tension() {
        let dir = tempdir().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        // 10 minutes ago: factor = 1.0 (full tension)
        write_anchor(dir.path(), now - 700.0);
        let r = AnchorReader::new(dir.path().to_path_buf());
        let f = r.factor();
        assert!((f - 1.0).abs() < 1e-6, "stale-anchor factor = {f}");
    }

    #[test]
    fn cache_does_not_rerun_within_cadence() {
        let dir = tempdir().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        write_anchor(dir.path(), now);
        let r = AnchorReader::new(dir.path().to_path_buf());
        let f1 = r.factor();
        // Mutate file mid-cadence — should NOT be picked up (cache active).
        write_anchor(dir.path(), now - 1000.0);
        let f2 = r.factor();
        assert!(
            (f1 - f2).abs() < 1e-6,
            "cache should hold within READ_CADENCE_SECS"
        );
    }
}
