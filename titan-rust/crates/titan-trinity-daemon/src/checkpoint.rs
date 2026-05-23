//! checkpoint — per-part traveling-tensor state persistence.
//!
//! Per SPEC §G5.2 item 4 + PLAN_trinity_homeostasis_p0 §1.3 (Maker override
//! 2026-05-23: full closure, no deferral): each of the 6 trinity-part daemons
//! writes a fixed-layout `<part>_checkpoint.bin` sidecar periodically
//! containing the exact `prev` (= x[t-1]), `prev2` (= x[t-2]) and the latest
//! 5D observable signature. At boot the daemon retry-loads the checkpoint —
//! if present + valid, restores tensor state so a restart never erases the
//! §G5.2 traveling tensor's journey. If absent / invalid: cold-start at 0.5
//! (the prior behavior — substrate continues per §11.B).
//!
//! G21/INV-4: each daemon is the SOLE writer of its own `<part>_checkpoint.bin`.
//! Atomic write via tmp + `rename(2)` so a crash can't leave the sidecar half-
//! written; readers either see the prior snapshot or the new one, never garbage.
//!
//! ## Byte layout (all `f32` little-endian, fixed-size per part)
//!
//! ```text
//!   [0..4]               schema version (currently 1.0)
//!   [4..8]               ts (Unix seconds, host clock)
//!   [8..8+4N]            prev      (N floats — N = 5 / 15 / 45)
//!   [8+4N..8+8N]         prev2     (N floats)
//!   [8+8N..8+8N+20]      last_obs  (5 floats: coh, mag, vel, dir, pol)
//! ```
//!
//! `N` is the layer's dim count (5 body / 15 mind / 45 spirit). Total payload
//! `= 4 + 4 + 8N + 20 = 28 + 8N` bytes (= 68 / 148 / 388 for body / mind / spirit).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::homeostasis::LayerObs;

/// Current checkpoint payload schema version. Bump when the byte layout
/// changes; readers fall back to cold-start when they see a mismatched
/// version (substrate continues per §11.B).
pub const CHECKPOINT_SCHEMA_VERSION_F32: f32 = 1.0;

/// Per-part checkpoint sidecar names. Each daemon owns its own (single writer).
pub fn checkpoint_filename(part: &str) -> String {
    format!("{part}_checkpoint.bin")
}

/// Compute the fixed payload size for an N-dim layer's checkpoint.
pub const fn payload_bytes(n: usize) -> usize {
    4 /*ver*/ + 4 /*ts*/ + 8 * n /*prev + prev2*/ + 20 /*last_obs*/
}

/// A loaded checkpoint snapshot — what [`load_for_part`] returns on success.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointSnapshot<const N: usize> {
    /// Unix timestamp the snapshot was written at.
    pub ts: f32,
    /// x[t-1] at write time — restored as the daemon's `prev`.
    pub prev: [f32; N],
    /// x[t-2] at write time — restored as the daemon's `prev2`.
    pub prev2: [f32; N],
    /// 5D observable signature at write time.
    pub last_obs: LayerObs,
}

fn now_secs_f32() -> f32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f32())
        .unwrap_or(0.0)
}

/// Path to the part's checkpoint sidecar under `shm_dir/`.
pub fn checkpoint_path(shm_dir: &Path, part: &str) -> PathBuf {
    shm_dir.join(checkpoint_filename(part))
}

/// Atomically write the checkpoint for one part. Writes to a same-FS tmp file
/// + `rename(2)` so daemons that mid-tick crash can never publish a torn
/// payload. Surfaces I/O failures (`directive_error_visibility`) — caller logs
/// + continues; never blocks the tick loop.
pub fn write_for_part<const N: usize>(
    shm_dir: &Path,
    part: &str,
    prev: &[f32; N],
    prev2: &[f32; N],
    last_obs: &LayerObs,
) -> std::io::Result<()> {
    let mut payload = Vec::with_capacity(payload_bytes(N));
    payload.extend_from_slice(&CHECKPOINT_SCHEMA_VERSION_F32.to_le_bytes());
    payload.extend_from_slice(&now_secs_f32().to_le_bytes());
    for v in prev.iter() {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for v in prev2.iter() {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload.extend_from_slice(&last_obs.coherence.to_le_bytes());
    payload.extend_from_slice(&last_obs.magnitude.to_le_bytes());
    payload.extend_from_slice(&last_obs.velocity.to_le_bytes());
    payload.extend_from_slice(&last_obs.direction.to_le_bytes());
    payload.extend_from_slice(&last_obs.polarity.to_le_bytes());
    debug_assert_eq!(payload.len(), payload_bytes(N));

    let target = checkpoint_path(shm_dir, part);
    let mut tmp = target.clone();
    let tmp_name = format!(".{}.tmp.{}", checkpoint_filename(part), std::process::id());
    tmp.set_file_name(tmp_name);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&payload)?;
        f.sync_data()?;
    }
    fs::rename(&tmp, &target)?;
    Ok(())
}

/// Load + parse the checkpoint for one part. Returns `None` on absent /
/// short / version-mismatched payload — caller cold-starts at 0.5 (no panic).
pub fn load_for_part<const N: usize>(shm_dir: &Path, part: &str) -> Option<CheckpointSnapshot<N>> {
    let path = checkpoint_path(shm_dir, part);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            info!(
                event = "CHECKPOINT_ABSENT",
                part,
                path = %path.display(),
                reason = format!("{e}"),
                "no checkpoint at boot \u{2014} cold-starting at 0.5"
            );
            return None;
        }
    };
    let expected = payload_bytes(N);
    if bytes.len() < expected {
        warn!(
            event = "CHECKPOINT_SHORT",
            part,
            bytes = bytes.len(),
            expected,
            "checkpoint payload too short \u{2014} cold-starting"
        );
        return None;
    }
    let ver = f32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes after length check"));
    if (ver - CHECKPOINT_SCHEMA_VERSION_F32).abs() > 1e-6 {
        warn!(
            event = "CHECKPOINT_VERSION_MISMATCH",
            part,
            ver,
            expected = CHECKPOINT_SCHEMA_VERSION_F32,
            "checkpoint schema version mismatch \u{2014} cold-starting"
        );
        return None;
    }
    let ts = f32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes after length check"));
    let mut prev = [0.0_f32; N];
    let mut prev2 = [0.0_f32; N];
    for i in 0..N {
        let o = 8 + i * 4;
        prev[i] = f32::from_le_bytes(
            bytes[o..o + 4]
                .try_into()
                .expect("4 bytes after length check"),
        );
    }
    for i in 0..N {
        let o = 8 + 4 * N + i * 4;
        prev2[i] = f32::from_le_bytes(
            bytes[o..o + 4]
                .try_into()
                .expect("4 bytes after length check"),
        );
    }
    let obs_base = 8 + 8 * N;
    let coherence = f32::from_le_bytes(
        bytes[obs_base..obs_base + 4]
            .try_into()
            .expect("4 bytes after length check"),
    );
    let magnitude = f32::from_le_bytes(
        bytes[obs_base + 4..obs_base + 8]
            .try_into()
            .expect("4 bytes after length check"),
    );
    let velocity = f32::from_le_bytes(
        bytes[obs_base + 8..obs_base + 12]
            .try_into()
            .expect("4 bytes after length check"),
    );
    let direction = f32::from_le_bytes(
        bytes[obs_base + 12..obs_base + 16]
            .try_into()
            .expect("4 bytes after length check"),
    );
    let polarity = f32::from_le_bytes(
        bytes[obs_base + 16..obs_base + 20]
            .try_into()
            .expect("4 bytes after length check"),
    );

    // Defensive: any non-finite value invalidates the snapshot.
    if !ts.is_finite()
        || prev.iter().any(|v| !v.is_finite())
        || prev2.iter().any(|v| !v.is_finite())
        || ![coherence, magnitude, velocity, direction, polarity]
            .iter()
            .all(|v| v.is_finite())
    {
        warn!(
            event = "CHECKPOINT_NONFINITE",
            part, "checkpoint contained NaN/Inf \u{2014} cold-starting"
        );
        return None;
    }

    info!(
        event = "CHECKPOINT_LOADED",
        part,
        ts,
        path = %path.display(),
        "tensor + observable state restored from checkpoint (no cold-start)"
    );
    Some(CheckpointSnapshot {
        ts,
        prev,
        prev2,
        last_obs: LayerObs {
            coherence,
            magnitude,
            velocity,
            direction,
            polarity,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_obs() -> LayerObs {
        LayerObs {
            coherence: 0.75,
            magnitude: 0.42,
            velocity: 0.05,
            direction: 1.0,
            polarity: -0.20,
        }
    }

    #[test]
    fn roundtrip_inner_body_5d() {
        let dir = tempdir().unwrap();
        let prev: [f32; 5] = [0.10, 0.20, 0.30, 0.40, 0.50];
        let prev2: [f32; 5] = [0.05, 0.15, 0.25, 0.35, 0.45];
        let obs = sample_obs();
        write_for_part(dir.path(), "inner_body", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<5> = load_for_part(dir.path(), "inner_body").expect("loaded");
        assert!(snap.ts > 0.0);
        assert_eq!(snap.prev, prev);
        assert_eq!(snap.prev2, prev2);
        assert!((snap.last_obs.coherence - obs.coherence).abs() < 1e-6);
        assert!((snap.last_obs.magnitude - obs.magnitude).abs() < 1e-6);
        assert!((snap.last_obs.velocity - obs.velocity).abs() < 1e-6);
        assert!((snap.last_obs.direction - obs.direction).abs() < 1e-6);
        assert!((snap.last_obs.polarity - obs.polarity).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_inner_spirit_45d() {
        let dir = tempdir().unwrap();
        let prev: [f32; 45] = std::array::from_fn(|i| (i as f32) * 0.01);
        let prev2: [f32; 45] = std::array::from_fn(|i| (i as f32) * 0.011);
        let obs = sample_obs();
        write_for_part(dir.path(), "inner_spirit", &prev, &prev2, &obs).unwrap();
        let snap: CheckpointSnapshot<45> =
            load_for_part(dir.path(), "inner_spirit").expect("loaded");
        assert_eq!(snap.prev, prev);
        assert_eq!(snap.prev2, prev2);
    }

    #[test]
    fn absent_returns_none() {
        let dir = tempdir().unwrap();
        let snap: Option<CheckpointSnapshot<5>> = load_for_part(dir.path(), "inner_body");
        assert!(snap.is_none());
    }

    #[test]
    fn version_mismatch_rejected() {
        let dir = tempdir().unwrap();
        let target = checkpoint_path(dir.path(), "inner_body");
        let mut bytes = vec![0u8; payload_bytes(5)];
        // Wrong version 99.0.
        bytes[0..4].copy_from_slice(&99.0_f32.to_le_bytes());
        std::fs::write(&target, &bytes).unwrap();
        let snap: Option<CheckpointSnapshot<5>> = load_for_part(dir.path(), "inner_body");
        assert!(snap.is_none());
    }

    #[test]
    fn nan_payload_rejected() {
        let dir = tempdir().unwrap();
        let prev: [f32; 5] = [0.0, f32::NAN, 0.0, 0.0, 0.0];
        let prev2: [f32; 5] = [0.0; 5];
        write_for_part(dir.path(), "inner_body", &prev, &prev2, &sample_obs()).unwrap();
        let snap: Option<CheckpointSnapshot<5>> = load_for_part(dir.path(), "inner_body");
        assert!(snap.is_none());
    }

    #[test]
    fn payload_bytes_layout() {
        // 5D body: 28 + 40 = 68 bytes.
        assert_eq!(payload_bytes(5), 68);
        // 15D mind: 28 + 120 = 148 bytes.
        assert_eq!(payload_bytes(15), 148);
        // 45D spirit: 28 + 360 = 388 bytes.
        assert_eq!(payload_bytes(45), 388);
    }
}
