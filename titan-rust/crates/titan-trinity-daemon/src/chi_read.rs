//! chi_read — SHM-direct reader for the substrate-scoped `chi_state.bin`
//! slot, used by spirit daemons to source `chi_health` for the §6.6.4
//! emergent CORRECTIVE_NUDGE amplitude formula.
//!
//! Slot layout per SPEC §7.1 + `titan-trinity-rs::chi_state` writer: 6 ×
//! float32 LE = 24 bytes, in canonical order:
//!
//! ```text
//! [0]  total      ∈ [0, 1]  ← `chi_health` per PLAN §6.6.4
//! [1]  spirit     ∈ [0, 1]
//! [2]  mind       ∈ [0, 1]
//! [3]  body       ∈ [0, 1]
//! [4]  coherence  ∈ [0, 1]
//! [5]  urgency    ∈ [0, 1]
//! ```
//!
//! Spirit only needs `total`. The shape mirrors [`crate::neuromod_read`]
//! so the spirit tick can fold both reads into a single per-tick SHM
//! traversal (the daemon's existing `~retry_every_n` cadence is enough —
//! chi_state updates at body-cycle 1 Hz so reading it every spirit tick
//! is light over-sampling, not stale).

use std::path::Path;

use titan_state::Slot;
use tracing::debug;

/// Total `chi_state.bin` payload bytes per SPEC §7.1.
pub const CHI_STATE_PAYLOAD_BYTES: usize = 24;

/// Default chi_health when the slot is absent or short-read (graceful
/// degradation: low-chi amplifier no-ops, nudge formula still computes
/// from excess + chronicity factors per PLAN §6.6.4).
pub const CHI_HEALTH_DEFAULT: f32 = 1.0;

/// Try to open `chi_state.bin` under `shm_dir/`. Returns `None` when the
/// slot doesn't exist yet (substrate writer may boot after the consumer);
/// caller retries via [`retry_open`] at the daemon's slot-retry cadence.
pub fn open_if_present(shm_dir: &Path) -> Option<Slot> {
    Slot::open(shm_dir.join("chi_state.bin")).ok()
}

/// Retry open the slot if currently `None`. No-op when already open.
pub fn retry_open(slot: &mut Option<Slot>, shm_dir: &Path) {
    if slot.is_none() {
        *slot = Slot::open(shm_dir.join("chi_state.bin")).ok();
    }
}

/// Read `chi.total` from the slot or return [`CHI_HEALTH_DEFAULT`] when the
/// slot is absent / short / unreadable. Never errors — graceful default.
pub fn read_chi_health(slot: Option<&Slot>) -> f32 {
    let Some(s) = slot else {
        return CHI_HEALTH_DEFAULT;
    };
    let bytes = match s.read() {
        Ok(b) => b,
        Err(e) => {
            debug!(err = ?e, "chi_state.bin read failed; using default chi_health");
            return CHI_HEALTH_DEFAULT;
        }
    };
    if bytes.len() < 4 {
        return CHI_HEALTH_DEFAULT;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[..4]);
    let total = f32::from_le_bytes(buf);
    if total.is_finite() {
        total.clamp(0.0, 1.0)
    } else {
        CHI_HEALTH_DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn encode_chi(total: f32, others: [f32; 5]) -> Vec<u8> {
        let mut out = vec![0u8; CHI_STATE_PAYLOAD_BYTES];
        out[0..4].copy_from_slice(&total.to_le_bytes());
        for (i, v) in others.iter().enumerate() {
            out[4 + i * 4..4 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn absent_slot_returns_default() {
        assert_eq!(read_chi_health(None), CHI_HEALTH_DEFAULT);
    }

    #[test]
    fn reads_total_from_first_4_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chi_state.bin");
        let mut slot = Slot::create(&path, 1, CHI_STATE_PAYLOAD_BYTES as u32).unwrap();
        let bytes = encode_chi(0.42, [0.5, 0.3, 0.4, 0.6, 0.1]);
        slot.write(&bytes).unwrap();
        let opened = Slot::open(&path).unwrap();
        let read = read_chi_health(Some(&opened));
        assert!((read - 0.42).abs() < 1e-6, "got {read}");
    }

    #[test]
    fn clamps_into_unit_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chi_state.bin");
        let mut slot = Slot::create(&path, 1, CHI_STATE_PAYLOAD_BYTES as u32).unwrap();
        // Out-of-range total — must clamp.
        slot.write(&encode_chi(-0.5, [0.0; 5])).unwrap();
        assert_eq!(read_chi_health(Some(&Slot::open(&path).unwrap())), 0.0);
        slot.write(&encode_chi(1.5, [0.0; 5])).unwrap();
        assert_eq!(read_chi_health(Some(&Slot::open(&path).unwrap())), 1.0);
    }

    #[test]
    fn non_finite_returns_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chi_state.bin");
        let mut slot = Slot::create(&path, 1, CHI_STATE_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_chi(f32::NAN, [0.0; 5])).unwrap();
        assert_eq!(
            read_chi_health(Some(&Slot::open(&path).unwrap())),
            CHI_HEALTH_DEFAULT,
        );
    }

    #[test]
    fn open_if_present_returns_none_for_missing() {
        let dir = tempdir().unwrap();
        assert!(open_if_present(dir.path()).is_none());
    }

    #[test]
    fn retry_open_lazy_picks_up_late_create() {
        let dir = tempdir().unwrap();
        let mut slot: Option<Slot> = None;
        retry_open(&mut slot, dir.path());
        assert!(slot.is_none());
        // Substrate creates the slot later.
        let path = dir.path().join("chi_state.bin");
        let _writer = Slot::create(&path, 1, CHI_STATE_PAYLOAD_BYTES as u32).unwrap();
        retry_open(&mut slot, dir.path());
        assert!(slot.is_some());
    }
}
