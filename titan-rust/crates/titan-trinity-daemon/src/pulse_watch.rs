//! pulse_watch — SHM-direct rising-edge detector for the 6 sphere clocks.
//!
//! Per SPEC §G5.1 (D-SPEC-112 amendment) + PLAN_trinity_homeostasis_p0 §4
//! (Phase 0a): the small filter_down DOWN-leg fires on the **spirit sphere-clock
//! PULSE event**, NOT on `KERNEL_EPOCH_TICK`. The UP-leg fires an additive
//! snapshot bonus to spirit on a **body/mind sphere-clock pulse**. Both
//! triggers read `sphere_clocks.bin` directly — the D-SPEC-117 SHM-direct
//! pattern that bypasses the `SPHERE_PULSE` bus event (which has a
//! substrate→broker transport gap; see `project_sphere_pulse_not_reaching_broker_freeze_20260522.md`).
//!
//! Pulse-count is the canonical witness: each [`SphereClock`] increments its
//! `pulse_count` (field index 4 in `sphere_clocks.bin`) at the exact moment
//! its scalar reaches centre. A rising edge `prev_pulse_count != cur_pulse_count`
//! is the cleanest, race-free detector — equivalent to the original
//! `SphereClock::tick(...) -> Option<PulseEvent>` semantics in the resonance
//! detector, but readable from any consumer process.
//!
//! `sphere_clocks.bin` layout (mirrored from
//! `titan-trinity-rs::sphere_clocks`): 6 clocks × 7 fields × `f32` LE
//! = 168 bytes. Field order per clock: `[radius, scalar_position, phase,
//! contraction_velocity, pulse_count, consecutive_balanced, last_pulse_age_s]`.
//! Clock order in file: inner_body, inner_mind, inner_spirit, outer_body,
//! outer_mind, outer_spirit.

use std::path::Path;

use titan_state::Slot;
use tracing::{debug, warn};

/// Per-clock stride in bytes (7 fields × 4 bytes).
pub const PULSE_WATCH_CLOCK_STRIDE: usize = 28;
/// Pulse-count field byte offset within a clock entry (field index 4 × 4).
pub const PULSE_WATCH_COUNT_OFFSET: usize = 16;
/// `consecutive_balanced` field byte offset within a clock entry (field index 5 × 4).
/// SPEC §G5.1 D-SPEC-121: a small filter_down emission requires the spirit
/// pulse to be balanced (`consecutive_balanced ≥ 1` at pulse moment).
pub const PULSE_WATCH_CONSEC_BALANCED_OFFSET: usize = 20;
/// Total payload bytes — must match `SPHERE_CLOCKS_PAYLOAD_BYTES`.
pub const PULSE_WATCH_PAYLOAD_BYTES: usize = 168;

/// Six clock roles in `sphere_clocks.bin` canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseClockRole {
    /// Inner-body sphere clock (file index 0).
    InnerBody,
    /// Inner-mind sphere clock (file index 1).
    InnerMind,
    /// Inner-spirit sphere clock (file index 2).
    InnerSpirit,
    /// Outer-body sphere clock (file index 3).
    OuterBody,
    /// Outer-mind sphere clock (file index 4).
    OuterMind,
    /// Outer-spirit sphere clock (file index 5).
    OuterSpirit,
}

impl PulseClockRole {
    /// File index in `sphere_clocks.bin` (0..6).
    pub const fn index(self) -> usize {
        match self {
            PulseClockRole::InnerBody => 0,
            PulseClockRole::InnerMind => 1,
            PulseClockRole::InnerSpirit => 2,
            PulseClockRole::OuterBody => 3,
            PulseClockRole::OuterMind => 4,
            PulseClockRole::OuterSpirit => 5,
        }
    }

    /// Byte offset of this clock's `pulse_count` field (field index 4).
    pub const fn count_byte_offset(self) -> usize {
        self.index() * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_COUNT_OFFSET
    }

    /// Byte offset of this clock's `consecutive_balanced` field (field index 5).
    /// Used by the §G5.1 D-SPEC-121 balanced-pulse gate: a small filter_down
    /// fires only when the spirit clock pulses AND was balanced at pulse time
    /// (`consecutive_balanced` ≥ 1 means the latest tick was within balance).
    pub const fn consecutive_balanced_byte_offset(self) -> usize {
        self.index() * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_CONSEC_BALANCED_OFFSET
    }
}

/// Per-tick edge report: which clocks pulsed since the last [`PulseWatcher::tick`]
/// call. `[inner_body, inner_mind, inner_spirit, outer_body, outer_mind, outer_spirit]`.
pub type PulseEdges = [bool; 6];

/// Per-tick balanced-edge report: which clocks pulsed THIS tick AND were
/// balanced at pulse time (`consecutive_balanced ≥ 1`). The §G5.1 D-SPEC-121
/// gate for emitting small filter_down: spirit pulse rising-edge AND
/// balanced. Per-index alignment matches [`PulseEdges`].
pub type BalancedPulseEdges = [bool; 6];

/// SHM-direct sphere-clock pulse-count edge detector.
///
/// At construction, opens `sphere_clocks.bin` (Option — substrate may not
/// have created it at boot) and seeds the previous-counts vector lazily.
/// First tick after slot becomes available reads counts but reports NO
/// edges (no prior to diff against) — subsequent ticks emit a `true` for
/// any clock whose `pulse_count` advanced (delta ≥ 1).
///
/// `f32 → u32` cast of `pulse_count` follows the substrate writer in
/// `titan_state::registry::SphereClocksWriter::serialize_clock` which
/// stores `pulse_count as f32`; values are integer-valued non-negative,
/// so `as u32` is exact within the 24-bit f32 mantissa for ≥ 16M pulses
/// — well past the production fleet ceiling.
pub struct PulseWatcher {
    slot: Option<Slot>,
    seeded: bool,
    prev_counts: [u32; 6],
}

impl PulseWatcher {
    /// Open `sphere_clocks.bin` under `shm_dir/` if present.
    pub fn open(shm_dir: &Path) -> Self {
        let slot = Slot::open(shm_dir.join("sphere_clocks.bin")).ok();
        Self {
            slot,
            seeded: false,
            prev_counts: [0; 6],
        }
    }

    /// Lazy retry-open for the substrate-created slot. Daemons retry every
    /// ~1 s before the slot exists; once opened, stays open.
    pub fn retry_open(&mut self, shm_dir: &Path) {
        if self.slot.is_none() {
            self.slot = Slot::open(shm_dir.join("sphere_clocks.bin")).ok();
        }
    }

    /// Returns `true` when the watcher has an open slot.
    pub fn is_open(&self) -> bool {
        self.slot.is_some()
    }

    /// Read latest pulse_counts + emit rising-edge mask.
    /// On absent/short-read returns no edges (substrate continues per §11.B).
    /// First call after seeding emits no edges (no prior); subsequent calls
    /// emit `true` for every clock whose pulse_count advanced.
    pub fn tick(&mut self) -> PulseEdges {
        self.tick_with_balanced().0
    }

    /// Read latest pulse_counts + `consecutive_balanced` + emit BOTH
    /// rising-edge masks: `(edges, balanced_edges)` where `balanced_edges[i]`
    /// is true iff `edges[i]` is true AND `consecutive_balanced[i] >= 1` at
    /// pulse time. SPEC §G5.1 D-SPEC-121 gate for small filter_down emission:
    /// spirit pulse rising-edge AND balanced.
    pub fn tick_with_balanced(&mut self) -> (PulseEdges, BalancedPulseEdges) {
        let mut edges: PulseEdges = [false; 6];
        let mut balanced_edges: BalancedPulseEdges = [false; 6];
        let Some(slot) = self.slot.as_ref() else {
            return (edges, balanced_edges);
        };
        let bytes = match slot.read() {
            Ok(b) => b,
            Err(e) => {
                debug!(err = ?e, "sphere_clocks.bin read failed; no pulse edge this tick");
                return (edges, balanced_edges);
            }
        };
        if bytes.len() < PULSE_WATCH_PAYLOAD_BYTES {
            warn!(
                event = "PULSE_WATCH_SHORT_READ",
                bytes = bytes.len(),
                expected = PULSE_WATCH_PAYLOAD_BYTES,
                "sphere_clocks.bin shorter than spec — no pulse edges"
            );
            return (edges, balanced_edges);
        }
        let mut cur = [0u32; 6];
        let mut cur_cb = [0u32; 6];
        let roles = [
            PulseClockRole::InnerBody,
            PulseClockRole::InnerMind,
            PulseClockRole::InnerSpirit,
            PulseClockRole::OuterBody,
            PulseClockRole::OuterMind,
            PulseClockRole::OuterSpirit,
        ];
        for (i, r) in roles.iter().enumerate() {
            let off = r.count_byte_offset();
            let pc = f32::from_le_bytes(
                bytes[off..off + 4]
                    .try_into()
                    .expect("4 bytes after length check"),
            );
            cur[i] = if pc.is_finite() && pc >= 0.0 {
                pc as u32
            } else {
                0
            };
            let cb_off = r.consecutive_balanced_byte_offset();
            let cb = f32::from_le_bytes(
                bytes[cb_off..cb_off + 4]
                    .try_into()
                    .expect("4 bytes after length check"),
            );
            cur_cb[i] = if cb.is_finite() && cb >= 0.0 {
                cb as u32
            } else {
                0
            };
        }
        if self.seeded {
            for i in 0..6 {
                edges[i] = cur[i] > self.prev_counts[i];
                // Balanced edge: pulse fired this tick AND the clock was within
                // balance (`consecutive_balanced ≥ 1` at pulse moment per
                // `sphere_clock.py` — `is_balanced` increments cb BEFORE the
                // pulse fires, so a balanced pulse sees cb ≥ 1).
                balanced_edges[i] = edges[i] && cur_cb[i] >= 1;
            }
        } else {
            self.seeded = true;
        }
        self.prev_counts = cur;
        (edges, balanced_edges)
    }

    /// Test-only helper: inject a synthetic previous-count baseline so
    /// integration tests don't have to drive a full Schumann tick to seed.
    #[cfg(test)]
    pub fn seed_prev(&mut self, counts: [u32; 6]) {
        self.seeded = true;
        self.prev_counts = counts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn encode_clocks(counts: [u32; 6]) -> Vec<u8> {
        encode_clocks_with_balanced(counts, [0; 6])
    }

    fn encode_clocks_with_balanced(counts: [u32; 6], consecutive_balanced: [u32; 6]) -> Vec<u8> {
        let mut out = vec![0u8; PULSE_WATCH_PAYLOAD_BYTES];
        for (i, c) in counts.iter().enumerate() {
            let off = i * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_COUNT_OFFSET;
            let f = *c as f32;
            out[off..off + 4].copy_from_slice(&f.to_le_bytes());
        }
        for (i, cb) in consecutive_balanced.iter().enumerate() {
            let off = i * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_CONSEC_BALANCED_OFFSET;
            let f = *cb as f32;
            out[off..off + 4].copy_from_slice(&f.to_le_bytes());
        }
        out
    }

    #[test]
    fn role_offsets_pack_correctly() {
        assert_eq!(PulseClockRole::InnerBody.count_byte_offset(), 16);
        assert_eq!(PulseClockRole::InnerSpirit.count_byte_offset(), 72); // 2·28+16
        assert_eq!(PulseClockRole::OuterSpirit.count_byte_offset(), 156); // 5·28+16
    }

    #[test]
    fn first_tick_emits_no_edges_then_subsequent_rising_edge() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sphere_clocks.bin");
        let mut slot = Slot::create(&path, 1, PULSE_WATCH_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_clocks([3, 5, 7, 11, 13, 17])).unwrap();
        let mut w = PulseWatcher::open(dir.path());
        let edges_seed = w.tick(); // seeds; reports nothing
        for e in edges_seed.iter() {
            assert!(!e);
        }
        // Now spirit pulses (inner_spirit count advances 7 → 8) — edge fires.
        slot.write(&encode_clocks([3, 5, 8, 11, 13, 17])).unwrap();
        let edges = w.tick();
        assert!(!edges[PulseClockRole::InnerBody.index()]);
        assert!(!edges[PulseClockRole::InnerMind.index()]);
        assert!(edges[PulseClockRole::InnerSpirit.index()]);
        assert!(!edges[PulseClockRole::OuterBody.index()]);
    }

    #[test]
    fn body_and_mind_edges_independently_detected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sphere_clocks.bin");
        let mut slot = Slot::create(&path, 1, PULSE_WATCH_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_clocks([0; 6])).unwrap();
        let mut w = PulseWatcher::open(dir.path());
        w.tick();
        slot.write(&encode_clocks([1, 0, 0, 0, 1, 0])).unwrap();
        let edges = w.tick();
        assert!(edges[PulseClockRole::InnerBody.index()]);
        assert!(!edges[PulseClockRole::InnerMind.index()]);
        assert!(edges[PulseClockRole::OuterMind.index()]);
    }

    #[test]
    fn absent_slot_returns_no_edges() {
        let dir = tempdir().unwrap();
        let mut w = PulseWatcher::open(dir.path()); // file does not exist
        let edges = w.tick();
        for e in edges.iter() {
            assert!(!e);
        }
        assert!(!w.is_open());
    }

    #[test]
    fn balanced_edge_requires_pulse_and_consecutive_balanced_ge_1() {
        // SPEC §G5.1 D-SPEC-121: a small filter_down emits only when the spirit
        // clock pulses AND was balanced at pulse moment (consecutive_balanced ≥ 1).
        // tick_with_balanced returns (edges, balanced_edges) where balanced_edges[i]
        // ⇔ edges[i] AND cb[i] ≥ 1.
        let dir = tempdir().unwrap();
        let path = dir.path().join("sphere_clocks.bin");
        let mut slot = Slot::create(&path, 1, PULSE_WATCH_PAYLOAD_BYTES as u32).unwrap();
        slot.write(&encode_clocks_with_balanced([0; 6], [0; 6]))
            .unwrap();
        let mut w = PulseWatcher::open(dir.path());
        let _seed = w.tick_with_balanced(); // seed

        // Inner-spirit pulses but is NOT balanced (cb=0) → edge fires, balanced_edge does not.
        slot.write(&encode_clocks_with_balanced([0, 0, 1, 0, 0, 0], [0; 6]))
            .unwrap();
        let (edges, balanced_edges) = w.tick_with_balanced();
        assert!(edges[PulseClockRole::InnerSpirit.index()]);
        assert!(
            !balanced_edges[PulseClockRole::InnerSpirit.index()],
            "unbalanced pulse must NOT emit a balanced edge"
        );

        // Inner-spirit pulses AND is balanced (cb=5) → both edges fire.
        slot.write(&encode_clocks_with_balanced(
            [0, 0, 2, 0, 0, 0],
            [0, 0, 5, 0, 0, 0],
        ))
        .unwrap();
        let (edges2, balanced_edges2) = w.tick_with_balanced();
        assert!(edges2[PulseClockRole::InnerSpirit.index()]);
        assert!(
            balanced_edges2[PulseClockRole::InnerSpirit.index()],
            "balanced pulse must emit a balanced edge — the D-SPEC-121 small filter_down gate"
        );

        // No pulse, but cb ≥ 1 (clock has been balanced sustainedly without a
        // new pulse this tick) → no edge, no balanced edge.
        slot.write(&encode_clocks_with_balanced(
            [0, 0, 2, 0, 0, 0],
            [0, 0, 10, 0, 0, 0],
        ))
        .unwrap();
        let (edges3, balanced_edges3) = w.tick_with_balanced();
        assert!(!edges3[PulseClockRole::InnerSpirit.index()]);
        assert!(!balanced_edges3[PulseClockRole::InnerSpirit.index()]);
    }
}
