//! focus_input — read the §G12 FOCUS cascade nudge slot.
//!
//! Per SPEC §G5.2 item 2 + §G12 (D-SPEC-112) + PLAN §1.4: FOCUS enters every
//! trinity-part daemon via a read-only SHM sidecar `focus_input.bin`, written
//! by the Python L2 `FocusPIDPublisher` (single-writer per G21). The slot is
//! a **fixed-layout** float32 LE payload — daemons need only their layer's
//! slice + the `stale_focus_multiplier` (the §G12 SPIRIT→Lower-Spirit→Mind→Body
//! cascade amplifier when `unified_spirit` reports the layer is STALE).
//!
//! Each daemon reads its own slice, scales by `stale_focus_multiplier`, and
//! composes the result into the per-tick `enrichment_force` it passes to the
//! §G5.2 [`crate::homeostasis::stateful_update`] kernel — so FOCUS lands as a
//! SEPARATE full-weight additive term per the §G5.2 equation, NOT folded into
//! drive (preserving G5.2-enrichment-separate semantics).
//!
//! ## Byte layout (528 bytes, all `f32` little-endian)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 4    | `ts` (Python `time.time()`) |
//! | 4      | 4    | `stale_focus_multiplier` ≥ 1.0 |
//! | 8      | 20   | `inner_body` 5×f32   |
//! | 28     | 60   | `inner_mind` 15×f32  |
//! | 88     | 180  | `inner_spirit` 45×f32 |
//! | 268    | 20   | `outer_body` 5×f32   |
//! | 288    | 60   | `outer_mind` 15×f32  |
//! | 348    | 180  | `outer_spirit` 45×f32 |
//!
//! G21/INV-4 preserved: daemons NEVER write this slot — they only read.

use std::path::Path;

use titan_state::Slot;
use tracing::{debug, warn};

/// Total fixed-layout payload size.
pub const FOCUS_INPUT_PAYLOAD_BYTES: usize = 528;
/// Sidecar file name under `shm_dir/`.
pub const FOCUS_INPUT_SIDECAR: &str = "focus_input.bin";

/// Byte offset of the `stale_focus_multiplier` field.
pub const OFFSET_STALE_FOCUS_MULT: usize = 4;
/// Byte offset of the `inner_body` 5×f32 slice.
pub const OFFSET_INNER_BODY: usize = 8;
/// Byte offset of the `inner_mind` 15×f32 slice.
pub const OFFSET_INNER_MIND: usize = 28;
/// Byte offset of the `inner_spirit` 45×f32 slice.
pub const OFFSET_INNER_SPIRIT: usize = 88;
/// Byte offset of the `outer_body` 5×f32 slice.
pub const OFFSET_OUTER_BODY: usize = 268;
/// Byte offset of the `outer_mind` 15×f32 slice.
pub const OFFSET_OUTER_MIND: usize = 288;
/// Byte offset of the `outer_spirit` 45×f32 slice.
pub const OFFSET_OUTER_SPIRIT: usize = 348;

/// Which trinity part a daemon owns — selects the correct slice + dim count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPart {
    /// inner-body 5D.
    InnerBody,
    /// inner-mind 15D.
    InnerMind,
    /// inner-spirit 45D.
    InnerSpirit,
    /// outer-body 5D.
    OuterBody,
    /// outer-mind 15D.
    OuterMind,
    /// outer-spirit 45D.
    OuterSpirit,
}

impl FocusPart {
    /// Byte offset of this part's slice in `focus_input.bin`.
    pub const fn byte_offset(self) -> usize {
        match self {
            FocusPart::InnerBody => OFFSET_INNER_BODY,
            FocusPart::InnerMind => OFFSET_INNER_MIND,
            FocusPart::InnerSpirit => OFFSET_INNER_SPIRIT,
            FocusPart::OuterBody => OFFSET_OUTER_BODY,
            FocusPart::OuterMind => OFFSET_OUTER_MIND,
            FocusPart::OuterSpirit => OFFSET_OUTER_SPIRIT,
        }
    }

    /// Dim count of this part.
    pub const fn dims(self) -> usize {
        match self {
            FocusPart::InnerBody | FocusPart::OuterBody => 5,
            FocusPart::InnerMind | FocusPart::OuterMind => 15,
            FocusPart::InnerSpirit | FocusPart::OuterSpirit => 45,
        }
    }
}

/// One layer's FOCUS nudge read out of `focus_input.bin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusNudge<const N: usize> {
    /// The `stale_focus_multiplier` (≥ 1.0). When `unified_spirit` reports the
    /// matching layer is STALE, this multiplies the nudge amplifying the
    /// SPIRIT→Lower-Spirit→Mind→Body cascade per §G12.
    pub stale_focus_multiplier: f32,
    /// Per-dim signed nudge (positive = push up, negative = push down).
    pub nudge: [f32; N],
}

impl<const N: usize> FocusNudge<N> {
    /// Neutral nudge — baseline multiplier `1.0` (no cascade amplification),
    /// zeros for every dim. Returned by [`read_nudge`] on absent / short-read.
    pub const fn neutral() -> Self {
        Self {
            stale_focus_multiplier: 1.0,
            nudge: [0.0; N],
        }
    }
}

/// Open `focus_input.bin` under `shm_dir/` if present. Returns `None` when
/// the slot is absent (Python publisher hadn't written it yet at boot —
/// retry at tick cadence).
pub fn open_if_present(shm_dir: &Path) -> Option<Slot> {
    Slot::open(shm_dir.join(FOCUS_INPUT_SIDECAR)).ok()
}

/// Read this daemon's [`FocusNudge`] from `focus_input.bin`. Returns a
/// neutral nudge (multiplier `1.0`, zeros) on absent / short / errored
/// read — substrate continues per §11.B.
pub fn read_nudge<const N: usize>(slot: Option<&Slot>, part: FocusPart) -> FocusNudge<N> {
    debug_assert_eq!(part.dims(), N, "FocusPart dims mismatch");
    let neutral = FocusNudge::<N>::neutral();
    let Some(slot) = slot else { return neutral };
    let bytes = match slot.read() {
        Ok(b) => b,
        Err(e) => {
            debug!(err = ?e, "focus_input.bin read failed; using neutral nudge");
            return neutral;
        }
    };
    if bytes.len() < FOCUS_INPUT_PAYLOAD_BYTES {
        warn!(
            event = "FOCUS_INPUT_SHORT_READ",
            bytes = bytes.len(),
            expected = FOCUS_INPUT_PAYLOAD_BYTES,
            "focus_input.bin shorter than spec — using neutral nudge"
        );
        return neutral;
    }
    let mut mult = f32::from_le_bytes(
        bytes[OFFSET_STALE_FOCUS_MULT..OFFSET_STALE_FOCUS_MULT + 4]
            .try_into()
            .expect("4 bytes after length check"),
    );
    // Defensive clamp — pathological multiplier cannot blow the integrator.
    // 1.0 = no cascade amplification (baseline); 8.0 = aggressive cascade.
    if !mult.is_finite() || mult < 1.0 {
        mult = 1.0;
    }
    mult = mult.min(8.0);
    let off = part.byte_offset();
    let mut nudge = [0.0_f32; N];
    for i in 0..N {
        let o = off + i * 4;
        nudge[i] = f32::from_le_bytes(
            bytes[o..o + 4]
                .try_into()
                .expect("4 bytes after length check"),
        );
        if !nudge[i].is_finite() {
            nudge[i] = 0.0;
        }
    }
    FocusNudge {
        stale_focus_multiplier: mult,
        nudge,
    }
}

/// Compose the cascade-amplified FOCUS nudge into an existing
/// `enrichment_force` vector in place — element-wise add of
/// `nudge[i] * stale_focus_multiplier`. The daemon calls this AFTER
/// computing its filter_down/ground_up enrichment delta so the §G5.2
/// kernel sees one unified `enrichment_force` term (still SEPARATE from
/// drive per G5.2-enrichment-separate).
pub fn compose_into<const N: usize>(enrichment: &mut [f32; N], focus: &FocusNudge<N>) {
    for i in 0..N {
        enrichment[i] += focus.nudge[i] * focus.stale_focus_multiplier;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[allow(clippy::too_many_arguments)]
    fn encode_payload(
        ts: f32,
        mult: f32,
        ib: [f32; 5],
        im: [f32; 15],
        is_: [f32; 45],
        ob: [f32; 5],
        om: [f32; 15],
        os: [f32; 45],
    ) -> Vec<u8> {
        let mut out = vec![0u8; FOCUS_INPUT_PAYLOAD_BYTES];
        out[0..4].copy_from_slice(&ts.to_le_bytes());
        out[OFFSET_STALE_FOCUS_MULT..OFFSET_STALE_FOCUS_MULT + 4]
            .copy_from_slice(&mult.to_le_bytes());
        for (i, v) in ib.iter().enumerate() {
            let o = OFFSET_INNER_BODY + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in im.iter().enumerate() {
            let o = OFFSET_INNER_MIND + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in is_.iter().enumerate() {
            let o = OFFSET_INNER_SPIRIT + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in ob.iter().enumerate() {
            let o = OFFSET_OUTER_BODY + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in om.iter().enumerate() {
            let o = OFFSET_OUTER_MIND + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in os.iter().enumerate() {
            let o = OFFSET_OUTER_SPIRIT + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn part_offsets_pack_correctly() {
        assert_eq!(FocusPart::InnerBody.byte_offset(), 8);
        assert_eq!(FocusPart::InnerMind.byte_offset(), 28);
        assert_eq!(FocusPart::InnerSpirit.byte_offset(), 88);
        assert_eq!(FocusPart::OuterBody.byte_offset(), 268);
        assert_eq!(FocusPart::OuterMind.byte_offset(), 288);
        assert_eq!(FocusPart::OuterSpirit.byte_offset(), 348);
        // Total = OuterSpirit offset + 45×4 = 348 + 180 = 528.
        assert_eq!(
            FocusPart::OuterSpirit.byte_offset() + FocusPart::OuterSpirit.dims() * 4,
            FOCUS_INPUT_PAYLOAD_BYTES,
        );
    }

    #[test]
    fn absent_slot_returns_neutral_nudge() {
        let n: FocusNudge<5> = read_nudge(None, FocusPart::InnerBody);
        assert_eq!(n.stale_focus_multiplier, 1.0);
        for v in n.nudge.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn reads_correct_part_slice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(FOCUS_INPUT_SIDECAR);
        let ib = [0.1, 0.2, 0.3, 0.4, 0.5];
        let im = std::array::from_fn(|i| (i as f32) * 0.01);
        let is_: [f32; 45] = std::array::from_fn(|i| -(i as f32) * 0.001);
        let ob = [-0.1, -0.2, -0.3, -0.4, -0.5];
        let om: [f32; 15] = [0.0; 15];
        let os: [f32; 45] = [0.0; 45];
        let payload = encode_payload(1.0, 2.5, ib, im, is_, ob, om, os);
        let mut slot = Slot::create(&path, 1, FOCUS_INPUT_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&payload).unwrap();

        let inner_body: FocusNudge<5> = read_nudge(Some(&slot), FocusPart::InnerBody);
        assert_eq!(inner_body.stale_focus_multiplier, 2.5);
        assert_eq!(inner_body.nudge, ib);

        let inner_spirit: FocusNudge<45> = read_nudge(Some(&slot), FocusPart::InnerSpirit);
        assert_eq!(inner_spirit.nudge, is_);

        let outer_body: FocusNudge<5> = read_nudge(Some(&slot), FocusPart::OuterBody);
        assert_eq!(outer_body.nudge, ob);
    }

    #[test]
    fn multiplier_clamped_safe() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(FOCUS_INPUT_SIDECAR);
        // Bogus multiplier 100.0 must clamp to 8.0; below-1.0 must floor at 1.0.
        let payload_hi = encode_payload(
            0.0, 100.0, [0.0; 5], [0.0; 15], [0.0; 45], [0.0; 5], [0.0; 15], [0.0; 45],
        );
        let mut slot = Slot::create(&path, 1, FOCUS_INPUT_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&payload_hi).unwrap();
        let n: FocusNudge<5> = read_nudge(Some(&slot), FocusPart::InnerBody);
        assert_eq!(n.stale_focus_multiplier, 8.0);

        let path2 = dir.path().join("focus_input2.bin");
        let mut slot2 = Slot::create(&path2, 1, FOCUS_INPUT_PAYLOAD_BYTES as u32).unwrap();
        let payload_lo = encode_payload(
            0.0, 0.1, [0.0; 5], [0.0; 15], [0.0; 45], [0.0; 5], [0.0; 15], [0.0; 45],
        );
        slot2.write(&payload_lo).unwrap();
        let n2: FocusNudge<5> = read_nudge(Some(&slot2), FocusPart::InnerBody);
        assert_eq!(n2.stale_focus_multiplier, 1.0);
    }

    #[test]
    fn compose_into_amplifies_by_multiplier() {
        let nudge = FocusNudge::<5> {
            stale_focus_multiplier: 2.0,
            nudge: [0.1, 0.0, -0.1, 0.0, 0.0],
        };
        let mut enrichment = [0.05_f32; 5];
        compose_into(&mut enrichment, &nudge);
        // enrichment[0] = 0.05 + 0.1·2.0 = 0.25
        // enrichment[2] = 0.05 + (-0.1)·2.0 = -0.15
        assert!((enrichment[0] - 0.25).abs() < 1e-6);
        assert!((enrichment[1] - 0.05).abs() < 1e-6);
        assert!((enrichment[2] + 0.15).abs() < 1e-6);
    }
}
