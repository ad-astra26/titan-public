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
/// Pulse-count field byte offset within a clock entry.
pub const PULSE_WATCH_COUNT_OFFSET: usize = 16; // 4 × 4
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

    /// Byte offset of this clock's pulse_count field.
    pub const fn count_byte_offset(self) -> usize {
        self.index() * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_COUNT_OFFSET
    }
}

/// Per-tick edge report: which clocks pulsed since the last [`PulseWatcher::tick`]
/// call. `[inner_body, inner_mind, inner_spirit, outer_body, outer_mind, outer_spirit]`.
pub type PulseEdges = [bool; 6];

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
        let mut edges: PulseEdges = [false; 6];
        let Some(slot) = self.slot.as_ref() else {
            return edges;
        };
        let bytes = match slot.read() {
            Ok(b) => b,
            Err(e) => {
                debug!(err = ?e, "sphere_clocks.bin read failed; no pulse edge this tick");
                return edges;
            }
        };
        if bytes.len() < PULSE_WATCH_PAYLOAD_BYTES {
            warn!(
                event = "PULSE_WATCH_SHORT_READ",
                bytes = bytes.len(),
                expected = PULSE_WATCH_PAYLOAD_BYTES,
                "sphere_clocks.bin shorter than spec — no pulse edges"
            );
            return edges;
        }
        let mut cur = [0u32; 6];
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
            // Defensive: NaN / negative → 0 (pulse counts are monotone u32).
            cur[i] = if pc.is_finite() && pc >= 0.0 {
                pc as u32
            } else {
                0
            };
        }
        if self.seeded {
            for i in 0..6 {
                edges[i] = cur[i] > self.prev_counts[i];
            }
        } else {
            self.seeded = true;
        }
        self.prev_counts = cur;
        edges
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
        let mut out = vec![0u8; PULSE_WATCH_PAYLOAD_BYTES];
        for (i, c) in counts.iter().enumerate() {
            let off = i * PULSE_WATCH_CLOCK_STRIDE + PULSE_WATCH_COUNT_OFFSET;
            let f = *c as f32;
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
}
