//! filter_down — Substrate-side FILTER_DOWN coordination passthrough.
//!
//! Per Preamble G6 + G7 + G8 + G9 + SPEC §10.F V5 cascade. C-S3 SUBSTRATE
//! ROLE = passthrough recorder only:
//!
//! - **Receives** `UNIFIED_SPIRIT_FILTER_DOWN` (canonical for V5 per SPEC
//!   §8.6) when unified-spirit-rs publishes it (C-S4 ships the publisher).
//!   In C-S3 the publisher doesn't exist yet, so this struct exists primarily
//!   as a wiring contract — bus subscriber registration ships in C3-6.
//! - **Records** the latest received state (multipliers + epoch_id + ts) for
//!   diagnostics + future fastbus-forwarding to daemons.
//! - **Does NOT compute** filter_down. That's unified-spirit-rs (C-S4)'s job
//!   per Preamble G7 (162D in → 120 multipliers out).
//! - **Does NOT apply** filter_down to slot tensors. That's the daemons'
//!   job per SPEC §10.F step 7-8 (clamp to `[FILTER_DOWN_MULTIPLIER_FLOOR,
//!   FILTER_DOWN_MULTIPLIER_CEIL]` per G7).
//!
//! # Why a passthrough struct in C-S3?
//!
//! Per memory rule `feedback_wire_now_gate_later.md`: ship the substrate side
//! of the pipeline in C-S3 (this commit) so C-S4 has a stable API to publish
//! into. The struct's behavior is observable: `last_unified_epoch` advances
//! monotonically when bus messages arrive, and 0 means "no message yet."
//!
//! # Mask invariants per Preamble G8
//!
//! Inner-spirit observer dims `[20:25]` (5D) and outer-spirit observer dims
//! `[85:90]` (5D) — total 10 dims — are NEVER part of the published filter_down
//! multipliers. Substrate doesn't enforce this directly (publisher does), but
//! exposes the dim layout via [`UnifiedFilterDownPayload`] so downstream
//! consumers can statically inspect the contract.
//!
//! # Dim layout per Preamble G7 (120 multipliers total)
//!
//! - `inner_body[5]`           dims 0..5    (all 5D filtered)
//! - `inner_mind[15]`          dims 0..15   (all 15D filtered)
//! - `inner_spirit_content[40]` dims 0..40 (= absolute 25..65; observer 20..25 masked)
//! - `outer_body[5]`           dims 0..5    (all 5D filtered)
//! - `outer_mind[15]`          dims 0..15   (all 15D filtered)
//! - `outer_spirit_content[40]` dims 0..40 (= absolute 90..130; observer 85..90 masked)
//!
//! Total: 5 + 15 + 40 + 5 + 15 + 40 = 120 dims.

use titan_core::constants::{
    FILTER_DOWN_COLD_START_FLOOR_EPOCHS, FILTER_DOWN_MULTIPLIER_CEIL, FILTER_DOWN_MULTIPLIER_FLOOR,
    FILTER_DOWN_SPIRIT_STRENGTH_MULT,
};

/// Body multiplier vector (5D) per Preamble G7.
pub const BODY_MULT_DIMS: usize = 5;
/// Mind multiplier vector (15D) per Preamble G7.
pub const MIND_MULT_DIMS: usize = 15;
/// Spirit content multiplier vector (40D, observer-masked) per Preamble G9.
pub const SPIRIT_CONTENT_MULT_DIMS: usize = 40;
/// Total multiplier count: 2 trinities × (5 body + 15 mind + 40 spirit content) = 120.
pub const TOTAL_MULT_DIMS: usize = 2 * (BODY_MULT_DIMS + MIND_MULT_DIMS + SPIRIT_CONTENT_MULT_DIMS);

const _: () = assert!(TOTAL_MULT_DIMS == 120);

/// Multiplier set published in `UNIFIED_SPIRIT_FILTER_DOWN`. Per Preamble G7
/// + SPEC §10.F V5 cascade payload shape.
///
/// All multipliers are pre-clamped to `[FILTER_DOWN_MULTIPLIER_FLOOR=0.3,
/// FILTER_DOWN_MULTIPLIER_CEIL=3.0]` by the publisher (C-S4); spirit content
/// multipliers additionally pre-multiplied by `FILTER_DOWN_SPIRIT_STRENGTH_MULT
/// = 0.3` toward 1.0 per G9 gentle-filter rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnifiedMultipliers {
    /// Inner-body dims 0..5.
    pub inner_body: [f32; BODY_MULT_DIMS],
    /// Inner-mind dims 0..15.
    pub inner_mind: [f32; MIND_MULT_DIMS],
    /// Inner-spirit content dims (observer-masked per G8) — 40D.
    pub inner_spirit_content: [f32; SPIRIT_CONTENT_MULT_DIMS],
    /// Outer-body dims 0..5.
    pub outer_body: [f32; BODY_MULT_DIMS],
    /// Outer-mind dims 0..15.
    pub outer_mind: [f32; MIND_MULT_DIMS],
    /// Outer-spirit content dims (observer-masked per G8) — 40D.
    pub outer_spirit_content: [f32; SPIRIT_CONTENT_MULT_DIMS],
}

impl UnifiedMultipliers {
    /// Identity multipliers — every dim = 1.0. Cold-start state per
    /// SPEC §10.F until `FILTER_DOWN_COLD_START_FLOOR_EPOCHS` (= 2000)
    /// is reached.
    pub fn identity() -> Self {
        Self {
            inner_body: [1.0; BODY_MULT_DIMS],
            inner_mind: [1.0; MIND_MULT_DIMS],
            inner_spirit_content: [1.0; SPIRIT_CONTENT_MULT_DIMS],
            outer_body: [1.0; BODY_MULT_DIMS],
            outer_mind: [1.0; MIND_MULT_DIMS],
            outer_spirit_content: [1.0; SPIRIT_CONTENT_MULT_DIMS],
        }
    }

    /// Returns `true` if every multiplier lies within the SPEC G7 clamp
    /// `[0.3, 3.0]`. Diagnostic helper for substrate-side parity check —
    /// detects malformed payloads from a buggy publisher.
    pub fn all_within_clamp(&self) -> bool {
        let lo = FILTER_DOWN_MULTIPLIER_FLOOR as f32;
        let hi = FILTER_DOWN_MULTIPLIER_CEIL as f32;
        let in_range = |v: f32| v.is_finite() && v >= lo && v <= hi;
        self.inner_body.iter().all(|v| in_range(*v))
            && self.inner_mind.iter().all(|v| in_range(*v))
            && self.inner_spirit_content.iter().all(|v| in_range(*v))
            && self.outer_body.iter().all(|v| in_range(*v))
            && self.outer_mind.iter().all(|v| in_range(*v))
            && self.outer_spirit_content.iter().all(|v| in_range(*v))
    }
}

impl Default for UnifiedMultipliers {
    fn default() -> Self {
        Self::identity()
    }
}

/// Decoded payload of `UNIFIED_SPIRIT_FILTER_DOWN`. Used by [`FilterDownCoordinator::on_unified_received`].
#[derive(Debug, Clone, Copy)]
pub struct UnifiedFilterDownPayload {
    /// 120 multipliers per Preamble G7.
    pub multipliers: UnifiedMultipliers,
    /// Consciousness epoch this payload was computed at.
    pub epoch_id: u64,
    /// Wall-clock seconds at publish time (Python `time.time()` semantics).
    pub ts: f64,
}

/// Substrate-side passthrough recorder.
#[derive(Debug, Clone)]
pub struct FilterDownCoordinator {
    last_multipliers: UnifiedMultipliers,
    last_epoch: u64,
    last_ts: f64,
    total_received: u64,
}

impl FilterDownCoordinator {
    /// Construct with cold-start identity multipliers + zero epoch.
    pub fn new() -> Self {
        Self {
            last_multipliers: UnifiedMultipliers::identity(),
            last_epoch: 0,
            last_ts: 0.0,
            total_received: 0,
        }
    }

    /// Called by C3-6 main bus subscription handler when
    /// `UNIFIED_SPIRIT_FILTER_DOWN` arrives. Records latest state for
    /// observability + future fastbus forwarding.
    pub fn on_unified_received(&mut self, payload: &UnifiedFilterDownPayload) {
        self.last_multipliers = payload.multipliers;
        self.last_epoch = payload.epoch_id;
        self.last_ts = payload.ts;
        self.total_received += 1;
    }

    /// Last received multipliers. Cold-start = identity (all 1.0) until first
    /// publication arrives.
    pub fn last_multipliers(&self) -> &UnifiedMultipliers {
        &self.last_multipliers
    }

    /// Last received epoch_id. `0` = no publication received yet (cold-start).
    pub fn last_epoch(&self) -> u64 {
        self.last_epoch
    }

    /// Last received wall-clock timestamp. `0.0` = no publication received yet.
    pub fn last_ts(&self) -> f64 {
        self.last_ts
    }

    /// Total publications received since substrate boot.
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Returns `true` if the substrate is still in cold-start (no V5
    /// publication received OR sender is below
    /// `FILTER_DOWN_COLD_START_FLOOR_EPOCHS=2000`). Used by diagnostics —
    /// substrate's behavior doesn't actually change based on this in C-S3.
    pub fn is_cold_start(&self) -> bool {
        self.last_epoch < FILTER_DOWN_COLD_START_FLOOR_EPOCHS
    }
}

impl Default for FilterDownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export the gentle-filter strength constant for callers that need to
/// reason about V5 spirit-content scaling per Preamble G9 + SPEC §10.F step 4.
pub const SPIRIT_GENTLE_FILTER_STRENGTH: f32 = FILTER_DOWN_SPIRIT_STRENGTH_MULT as f32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_mult_dims_equals_120_per_preamble_g7() {
        assert_eq!(TOTAL_MULT_DIMS, 120);
        // Construction sanity: each dim count matches G7
        assert_eq!(BODY_MULT_DIMS, 5);
        assert_eq!(MIND_MULT_DIMS, 15);
        assert_eq!(SPIRIT_CONTENT_MULT_DIMS, 40);
    }

    #[test]
    fn identity_multipliers_all_one() {
        let m = UnifiedMultipliers::identity();
        assert!(m.inner_body.iter().all(|v| *v == 1.0));
        assert!(m.inner_mind.iter().all(|v| *v == 1.0));
        assert!(m.inner_spirit_content.iter().all(|v| *v == 1.0));
        assert!(m.outer_body.iter().all(|v| *v == 1.0));
        assert!(m.outer_mind.iter().all(|v| *v == 1.0));
        assert!(m.outer_spirit_content.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn identity_multipliers_pass_clamp_check() {
        assert!(UnifiedMultipliers::identity().all_within_clamp());
    }

    #[test]
    fn out_of_range_multipliers_fail_clamp_check() {
        let mut m = UnifiedMultipliers::identity();
        // Below floor (G7 LOCKED 0.3)
        m.inner_body[0] = 0.1;
        assert!(!m.all_within_clamp());

        let mut m = UnifiedMultipliers::identity();
        // Above ceil (G7 LOCKED 3.0)
        m.outer_spirit_content[39] = 5.0;
        assert!(!m.all_within_clamp());

        let mut m = UnifiedMultipliers::identity();
        m.inner_mind[7] = f32::NAN;
        assert!(!m.all_within_clamp(), "NaN must fail clamp check");

        let mut m = UnifiedMultipliers::identity();
        m.outer_body[2] = f32::INFINITY;
        assert!(!m.all_within_clamp());
    }

    #[test]
    fn floor_and_ceil_constants_match_g7() {
        // Preamble G7 LOCKED: floor=0.3, ceil=3.0 — read from SPEC TOML
        assert_eq!(FILTER_DOWN_MULTIPLIER_FLOOR, 0.3);
        assert_eq!(FILTER_DOWN_MULTIPLIER_CEIL, 3.0);
    }

    #[test]
    fn spirit_strength_constant_matches_g9() {
        // G9: gentle filter scales spirit content multipliers by 0.3 toward 1.0
        assert_eq!(FILTER_DOWN_SPIRIT_STRENGTH_MULT, 0.3);
        assert_eq!(SPIRIT_GENTLE_FILTER_STRENGTH, 0.3);
    }

    #[test]
    fn cold_start_floor_constant_matches_spec_10f() {
        assert_eq!(FILTER_DOWN_COLD_START_FLOOR_EPOCHS, 2000);
    }

    #[test]
    fn coordinator_new_starts_in_cold_start() {
        let c = FilterDownCoordinator::new();
        assert_eq!(c.last_epoch(), 0);
        assert_eq!(c.last_ts(), 0.0);
        assert_eq!(c.total_received(), 0);
        assert!(c.is_cold_start());
        // Identity multipliers
        assert!(c.last_multipliers().inner_body.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn coordinator_on_unified_received_records_latest() {
        let mut c = FilterDownCoordinator::new();
        let mut mults = UnifiedMultipliers::identity();
        mults.inner_body[0] = 0.5;
        let payload = UnifiedFilterDownPayload {
            multipliers: mults,
            epoch_id: 3000,
            ts: 1_700_000_000.0,
        };
        c.on_unified_received(&payload);
        assert_eq!(c.last_epoch(), 3000);
        assert_eq!(c.last_ts(), 1_700_000_000.0);
        assert_eq!(c.total_received(), 1);
        assert_eq!(c.last_multipliers().inner_body[0], 0.5);
        // Past cold-start floor (epoch >= 2000) → no longer cold
        assert!(!c.is_cold_start());
    }

    #[test]
    fn coordinator_total_received_increments_per_publication() {
        let mut c = FilterDownCoordinator::new();
        for i in 1..=10 {
            let p = UnifiedFilterDownPayload {
                multipliers: UnifiedMultipliers::identity(),
                epoch_id: i,
                ts: i as f64,
            };
            c.on_unified_received(&p);
        }
        assert_eq!(c.total_received(), 10);
        assert_eq!(c.last_epoch(), 10);
    }

    #[test]
    fn coordinator_is_cold_start_until_2000_epochs() {
        let mut c = FilterDownCoordinator::new();
        // Below floor
        c.on_unified_received(&UnifiedFilterDownPayload {
            multipliers: UnifiedMultipliers::identity(),
            epoch_id: 1999,
            ts: 0.0,
        });
        assert!(c.is_cold_start());
        // At floor
        c.on_unified_received(&UnifiedFilterDownPayload {
            multipliers: UnifiedMultipliers::identity(),
            epoch_id: 2000,
            ts: 0.0,
        });
        assert!(!c.is_cold_start());
    }
}
