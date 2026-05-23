//! neuromod_read — read `neuromod_state.bin` and project to a scalar
//! `neuromod_gain` for the §G5.2 restoring spring.
//!
//! Per SPEC §G5.2 item 2: `k_restore is scaled by the neuromod gain (read
//! from neuromod_state.bin) per §G5.2`. The slot's v2 schema is
//! `(NEUROMOD_FIELD_COUNT × NEUROMOD_FIELDS_PER_MOD) × float32 LE = 96 bytes`
//! with per-modulator fields `(level, gain, phasic, tonic)` for the 6
//! canonical neuromodulators (DA, 5HT, NE, ACh, Endorphin, GABA — Python
//! ordering per `titan_hcl/logic/neuromodulator.py:compute_modulation_from_state`).
//!
//! **Scalar projection:** mean of the 6 per-modulator `gain` fields. Each
//! modulator's gain sits at 1.0 baseline (no modulation) and ranges through
//! the homeostatic operating window (`titan_hcl/logic/neuromodulator.py` clamps
//! around 0.3..3.0). Averaging is symmetric (no single modulator privileged),
//! sits at 1.0 baseline (so neutral state = no spring modulation), and
//! globally tracks neuromod activation: when modulators are elevated (high
//! arousal / vigilance), the restoring spring pulls harder back to centre,
//! per §G5.2's interpretation of neuromod gain as homeostatic strength.
//!
//! The gain is clamped to `[NEUROMOD_GAIN_MIN, NEUROMOD_GAIN_MAX]` so a
//! pathological neuromod_state.bin value cannot freeze the spring or send
//! it negative. On absent/short-read/error, returns the 1.0 baseline so the
//! spring runs at its config-specified `k_restore` with no modulation
//! (substrate continues per §11.B + `directive_error_visibility`).

use std::path::Path;

use titan_core::constants::{
    NEUROMOD_FIELDS_PER_MOD, NEUROMOD_FIELD_COUNT, NEUROMOD_PAYLOAD_BYTES,
};
use titan_state::Slot;
use tracing::{debug, warn};

/// Lower clamp on the projected `neuromod_gain` — prevents zero/negative
/// spring strength under pathological neuromod state.
pub const NEUROMOD_GAIN_MIN: f32 = 0.3;
/// Upper clamp on the projected `neuromod_gain` — caps over-strong spring
/// (would pin tensors at 0.5 per PLAN §7 risk).
pub const NEUROMOD_GAIN_MAX: f32 = 3.0;
/// Baseline (no-modulation) gain. Returned on absent/short-read/error.
pub const NEUROMOD_GAIN_NEUTRAL: f32 = 1.0;

/// Byte offset of the per-modulator `gain` field in the v2 layout:
/// per modulator stride = `NEUROMOD_FIELDS_PER_MOD × 4` = 16 bytes;
/// fields are `(level, gain, phasic, tonic)` so `gain` is at +4 bytes.
const GAIN_FIELD_OFFSET: usize = 4;

/// Open `neuromod_state.bin` under `shm_dir/` if present. Returns `None` if
/// the slot file is absent at boot (Python L2's `neuromod_worker` creates it
/// after the Rust daemon has started — retry-open at tick cadence using
/// [`open_if_present`]).
pub fn open_if_present(shm_dir: &Path) -> Option<Slot> {
    Slot::open(shm_dir.join("neuromod_state.bin")).ok()
}

/// Read `neuromod_state.bin` and project the 6 `gain` fields to a single
/// scalar (mean, clamped). Returns [`NEUROMOD_GAIN_NEUTRAL`] (1.0) if the
/// slot is absent, the read fails, or the payload is shorter than v2 spec.
pub fn read_gain(slot: Option<&Slot>) -> f32 {
    let Some(slot) = slot else {
        return NEUROMOD_GAIN_NEUTRAL;
    };
    let bytes = match slot.read() {
        Ok(b) => b,
        Err(e) => {
            // Surfaced once per N ticks would be ideal; daemons do this via
            // DriftAggregator for drift. Here a debug-level log is enough —
            // sustained read failure surfaces via the missing dimensional
            // response in the tensor (loud-in-aggregate per
            // `directive_error_visibility`).
            debug!(err = ?e, "neuromod_state.bin read failed; using neutral gain");
            return NEUROMOD_GAIN_NEUTRAL;
        }
    };
    if bytes.len() < NEUROMOD_PAYLOAD_BYTES as usize {
        warn!(
            event = "NEUROMOD_STATE_SHORT_READ",
            bytes = bytes.len(),
            expected = NEUROMOD_PAYLOAD_BYTES,
            "neuromod_state.bin shorter than v2 spec — using neutral gain"
        );
        return NEUROMOD_GAIN_NEUTRAL;
    }
    let stride = NEUROMOD_FIELDS_PER_MOD as usize * 4;
    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for i in 0..(NEUROMOD_FIELD_COUNT as usize) {
        let off = i * stride + GAIN_FIELD_OFFSET;
        let g = f32::from_le_bytes(
            bytes[off..off + 4]
                .try_into()
                .expect("4 bytes after length check"),
        );
        if g.is_finite() {
            sum += g;
            n += 1;
        }
    }
    if n == 0 {
        return NEUROMOD_GAIN_NEUTRAL;
    }
    (sum / n as f32).clamp(NEUROMOD_GAIN_MIN, NEUROMOD_GAIN_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Encode a v2 neuromod_state payload: 6 modulators × (level, gain, phasic, tonic) f32 LE.
    fn encode_v2(gains: [f32; 6]) -> Vec<u8> {
        let mut out = Vec::with_capacity(NEUROMOD_PAYLOAD_BYTES as usize);
        for &g in gains.iter() {
            out.extend_from_slice(&0.5_f32.to_le_bytes()); // level
            out.extend_from_slice(&g.to_le_bytes()); // gain
            out.extend_from_slice(&0.0_f32.to_le_bytes()); // phasic
            out.extend_from_slice(&0.0_f32.to_le_bytes()); // tonic
        }
        out
    }

    #[test]
    fn absent_slot_returns_neutral() {
        assert_eq!(read_gain(None), NEUROMOD_GAIN_NEUTRAL);
    }

    #[test]
    fn baseline_gains_project_to_one() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("neuromod_state.bin");
        let mut slot = Slot::create(&path, 2, NEUROMOD_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_v2([1.0; 6])).unwrap();
        assert!((read_gain(Some(&slot)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn elevated_gains_project_above_one() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("neuromod_state.bin");
        let mut slot = Slot::create(&path, 2, NEUROMOD_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_v2([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]))
            .unwrap();
        assert!((read_gain(Some(&slot)) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn clamped_into_safe_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("neuromod_state.bin");
        let mut slot = Slot::create(&path, 2, NEUROMOD_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_v2([10.0; 6])).unwrap();
        assert_eq!(read_gain(Some(&slot)), NEUROMOD_GAIN_MAX);

        let mut slot2 = Slot::create(
            dir.path().join("neuromod_state2.bin"),
            2,
            NEUROMOD_PAYLOAD_BYTES as u32,
        )
        .unwrap();
        slot2.write(&encode_v2([0.0; 6])).unwrap();
        assert_eq!(read_gain(Some(&slot2)), NEUROMOD_GAIN_MIN);
    }
}
