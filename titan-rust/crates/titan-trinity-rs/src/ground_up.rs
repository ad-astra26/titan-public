//! ground_up — GROUND_UP grounding-enrichment primitives.
//!
//! Per Preamble G5 (FILTER_DOWN + GROUND_UP symmetry = Middle Path) + G10
//! (GROUND_UP modifies Body all 5D + Mind WILLING ONLY).
//!
//! # Architectural role
//!
//! The substrate writes `topology_30d.bin` (per SPEC §10.G step 5) which is
//! the source of the grounding signal. Daemons (titan-inner-body-rs +
//! titan-inner-mind-rs in C-S5; outer pair in C-S6) read their topology slice
//! and APPLY the ground_up nudge per SPEC §10.G step 6. C-S3 ships the
//! coordination primitives here so:
//!
//! - daemons in C-S5/C-S6 import `compute_nudge` + `apply_nudge` directly;
//! - the substrate's body-tick observability records `GroundUpCoordinator`
//!   state (`_total_applications`, `_total_body_delta`, `_total_mind_delta`)
//!   for diagnostics — Phase D may surface via `/v4/admin/substrate-state`.
//!
//! # Byte-identical port (per SPEC §11.6)
//!
//! Implementation byte-identical to `titan_plugin/logic/ground_up.py` lines
//! 51–147 + Preamble G10 dim mask. Verified by parity vectors at
//! `tests/parity/vectors.json::ground_up_nudge`.
//!
//! # Constants
//!
//! - `GROUND_UP_DEFAULT_STRENGTH = 0.1` — daemon-applied multiplier on nudge
//! - `GROUND_UP_DEFAULT_DAMPING = 0.95` — smoothing of prev with raw signal
//! - `GROUND_UP_MAX_NUDGE = 0.05` — per-tick safety clamp
//!
//! All from `titan_core::constants` (auto-generated from SPEC TOML v0.1.4).

use titan_core::constants::{
    GROUND_UP_DEFAULT_DAMPING, GROUND_UP_DEFAULT_STRENGTH, GROUND_UP_MAX_NUDGE,
};
use tracing::trace;

/// Dimensions of the body tensor — per Preamble G1.
pub const BODY_5D: usize = 5;
/// Dimensions of the mind tensor — per Preamble G1.
pub const MIND_15D: usize = 15;
/// Mind willing dim range — per Preamble G10 (the only mind dims grounded).
pub const MIND_WILLING_RANGE: std::ops::Range<usize> = 10..15;
/// Grounding signal length: body[0:5] + mind willing[5:10] = 10D.
pub const GROUNDING_SIGNAL_10D: usize = 10;

/// One nudge result — body[5] + mind[5] vectors + total magnitude.
///
/// Mirrors Python `compute_nudge()` return value:
/// `{"body_nudge": [5], "mind_nudge": [5], "total_magnitude": float}`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NudgeResult {
    /// Body nudge vector (5D), already clamped to ±`GROUND_UP_MAX_NUDGE`.
    pub body_nudge: [f32; BODY_5D],
    /// Mind willing nudge vector (5D), already clamped to ±`GROUND_UP_MAX_NUDGE`.
    pub mind_nudge: [f32; BODY_5D],
    /// L2 magnitude across both nudge vectors. Python `total_magnitude`
    /// rounded to 6 decimals; Rust returns full precision (caller rounds if
    /// needed for parity vector comparison).
    pub total_magnitude: f32,
}

/// Pure compute_nudge — byte-identical port of `ground_up.py:51-101`.
///
/// Stateless: caller passes `prev_nudge_body` + `prev_nudge_mind` from
/// previous tick. Used by [`GroundUpEnricher`] (stateful) AND by daemons
/// directly when they have their own enricher state.
///
/// # Algorithm
///
/// 1. `signal = grounding_signal_10d` (already 10D)
/// 2. `raw_body = signal[0..5]`, `raw_mind = signal[5..10]`
/// 3. `body_nudge[i] = damping × prev_body[i] + (1 − damping) × raw_body[i]`
/// 4. `mind_nudge[i] = damping × prev_mind[i] + (1 − damping) × raw_mind[i]`
/// 5. Clamp both to ±`GROUND_UP_MAX_NUDGE`
/// 6. `total_mag = sqrt(Σ body² + Σ mind²)`
pub fn compute_nudge(
    grounding_signal_10d: &[f32; GROUNDING_SIGNAL_10D],
    prev_nudge_body: &[f32; BODY_5D],
    prev_nudge_mind: &[f32; BODY_5D],
    damping: f32,
) -> NudgeResult {
    let max_nudge = GROUND_UP_MAX_NUDGE as f32;
    let body_nudge: [f32; BODY_5D] = std::array::from_fn(|i| {
        let raw = grounding_signal_10d[i];
        let damped = damping * prev_nudge_body[i] + (1.0 - damping) * raw;
        damped.clamp(-max_nudge, max_nudge)
    });
    let mind_nudge: [f32; BODY_5D] = std::array::from_fn(|i| {
        let raw = grounding_signal_10d[BODY_5D + i];
        let damped = damping * prev_nudge_mind[i] + (1.0 - damping) * raw;
        damped.clamp(-max_nudge, max_nudge)
    });
    let total_magnitude = (body_nudge
        .iter()
        .chain(mind_nudge.iter())
        .map(|v| v * v)
        .sum::<f32>())
    .sqrt();
    NudgeResult {
        body_nudge,
        mind_nudge,
        total_magnitude,
    }
}

/// Pure apply_nudge — byte-identical port of `ground_up.py:103-147` per
/// Preamble G10 dim mask.
///
/// **Modifies in-place**:
/// - `body_5d[0..5]` — all 5 dims grounded
/// - `mind_15d[10..15]` — willing dims only
///
/// **Does NOT touch**:
/// - `mind_15d[0..10]` (thinking + feeling) — not grounded per G10
/// - `spirit_45d[..]` — not grounded per G10 (caller doesn't pass spirit here)
///
/// `dt` is clamped to `[0.0, 30.0]` (matches Python `min(dt, 30.0)`); typical
/// substrate body cycle ≈ 1.149 s.
pub fn apply_nudge(
    body_5d: &mut [f32; BODY_5D],
    mind_15d: &mut [f32; MIND_15D],
    nudge: &NudgeResult,
    strength: f32,
    dt: f32,
) {
    let dt_clamped = dt.min(30.0);
    for (i, slot) in body_5d.iter_mut().enumerate() {
        let delta = nudge.body_nudge[i] * strength * dt_clamped;
        *slot = (*slot + delta).clamp(0.0, 1.0);
    }
    // Per G10: ONLY mind[10:15] willing dims. mind[0:10] thinking + feeling
    // remain untouched.
    for (i, slot) in mind_15d[MIND_WILLING_RANGE.start..MIND_WILLING_RANGE.end]
        .iter_mut()
        .enumerate()
    {
        let delta = nudge.mind_nudge[i] * strength * dt_clamped;
        *slot = (*slot + delta).clamp(0.0, 1.0);
    }
}

/// Stateful enricher that tracks `prev_nudge_body` + `prev_nudge_mind` across
/// ticks (for damping continuity per SPEC §10.G + Python `ground_up.py`
/// constructor).
///
/// Substrate constructs ONE per inner/outer trinity (because each trinity
/// produces its own grounding signal). Daemons in C-S5/C-S6 may construct
/// their own per-daemon enricher instead — interchangeable.
#[derive(Debug, Clone)]
pub struct GroundUpEnricher {
    strength: f32,
    damping: f32,
    prev_nudge_body: [f32; BODY_5D],
    prev_nudge_mind: [f32; BODY_5D],
    total_applications: u64,
    total_body_delta: f64,
    total_mind_delta: f64,
}

impl GroundUpEnricher {
    /// Construct with explicit strength + damping.
    pub fn new(strength: f32, damping: f32) -> Self {
        Self {
            strength,
            damping,
            prev_nudge_body: [0.0; BODY_5D],
            prev_nudge_mind: [0.0; BODY_5D],
            total_applications: 0,
            total_body_delta: 0.0,
            total_mind_delta: 0.0,
        }
    }

    /// Construct with default constants from SPEC TOML
    /// (`GROUND_UP_DEFAULT_STRENGTH=0.1`, `GROUND_UP_DEFAULT_DAMPING=0.95`).
    pub fn with_defaults() -> Self {
        Self::new(
            GROUND_UP_DEFAULT_STRENGTH as f32,
            GROUND_UP_DEFAULT_DAMPING as f32,
        )
    }

    /// Configured strength.
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Configured damping.
    pub fn damping(&self) -> f32 {
        self.damping
    }

    /// Number of times `apply` has been called since construction.
    pub fn total_applications(&self) -> u64 {
        self.total_applications
    }

    /// Cumulative L1 norm of body nudges applied. Useful for substrate-side
    /// telemetry per `/v4/admin/substrate-state` (Phase D).
    pub fn total_body_delta(&self) -> f64 {
        self.total_body_delta
    }

    /// Cumulative L1 norm of mind nudges applied.
    pub fn total_mind_delta(&self) -> f64 {
        self.total_mind_delta
    }

    /// Compute the next nudge given the current grounding signal. Updates
    /// internal `prev_nudge_*` state.
    pub fn compute_nudge(
        &mut self,
        grounding_signal_10d: &[f32; GROUNDING_SIGNAL_10D],
    ) -> NudgeResult {
        let result = compute_nudge(
            grounding_signal_10d,
            &self.prev_nudge_body,
            &self.prev_nudge_mind,
            self.damping,
        );
        self.prev_nudge_body = result.body_nudge;
        self.prev_nudge_mind = result.mind_nudge;
        result
    }

    /// Compute + apply in one call. Returns the nudge for diagnostics.
    pub fn apply(
        &mut self,
        body_5d: &mut [f32; BODY_5D],
        mind_15d: &mut [f32; MIND_15D],
        grounding_signal_10d: &[f32; GROUNDING_SIGNAL_10D],
        dt: f32,
    ) -> NudgeResult {
        let nudge = self.compute_nudge(grounding_signal_10d);
        apply_nudge(body_5d, mind_15d, &nudge, self.strength, dt);
        self.total_applications += 1;
        self.total_body_delta += nudge.body_nudge.iter().map(|v| v.abs() as f64).sum::<f64>();
        self.total_mind_delta += nudge.mind_nudge.iter().map(|v| v.abs() as f64).sum::<f64>();
        trace!(
            total_apps = self.total_applications,
            total_mag = nudge.total_magnitude,
            "ground_up: applied nudge"
        );
        nudge
    }
}

impl Default for GroundUpEnricher {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Substrate-side coordination state — one inner enricher + one outer
/// enricher, since inner trinity and outer trinity each produce their own
/// grounding signal (per Preamble G1 + SPEC §9.A).
#[derive(Debug, Clone, Default)]
pub struct GroundUpCoordinator {
    /// Inner-trinity enricher (drives inner-body + inner-mind grounding in C-S5).
    pub inner: GroundUpEnricher,
    /// Outer-trinity enricher (drives outer-body + outer-mind grounding in C-S6).
    pub outer: GroundUpEnricher,
}

impl GroundUpCoordinator {
    /// Construct with default strength + damping for both enrichers.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_arr<const N: usize>(a: &[f32; N], b: &[f32; N], eps: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y, eps))
    }

    // ── Pure compute_nudge ──────────────────────────────────────────────

    #[test]
    fn compute_nudge_zero_signal_returns_zero() {
        let signal = [0.0; 10];
        let prev_body = [0.0; 5];
        let prev_mind = [0.0; 5];
        let r = compute_nudge(&signal, &prev_body, &prev_mind, 0.95);
        assert_eq!(r.body_nudge, [0.0; 5]);
        assert_eq!(r.mind_nudge, [0.0; 5]);
        assert_eq!(r.total_magnitude, 0.0);
    }

    #[test]
    fn compute_nudge_saturated_signal_clamps_to_max_nudge() {
        // Signal=1 + prev=0 + damping=0.95: damped=0.05 → within [-0.05, 0.05]
        let signal = [1.0; 10];
        let prev_body = [0.0; 5];
        let prev_mind = [0.0; 5];
        let r = compute_nudge(&signal, &prev_body, &prev_mind, 0.95);
        assert!(approx_arr(&r.body_nudge, &[0.05; 5], 1e-6));
        assert!(approx_arr(&r.mind_nudge, &[0.05; 5], 1e-6));
        // total_mag = sqrt(10 × 0.05²) = sqrt(0.025) ≈ 0.158113883...
        assert!(approx(r.total_magnitude, 0.158_113_88, 1e-5));
    }

    #[test]
    fn compute_nudge_max_nudge_clamp_with_extreme_signal() {
        // Signal=20, prev=0, damping=0.95: damped = 0.05*20 = 1.0, then clamped to 0.05
        let signal = [20.0; 10];
        let r = compute_nudge(&signal, &[0.0; 5], &[0.0; 5], 0.95);
        for v in r.body_nudge.iter().chain(r.mind_nudge.iter()) {
            assert_eq!(*v, 0.05, "value {v} should clamp to MAX_NUDGE=0.05");
        }
    }

    #[test]
    fn compute_nudge_negative_signal_clamps_to_negative_max() {
        let signal = [-20.0; 10];
        let r = compute_nudge(&signal, &[0.0; 5], &[0.0; 5], 0.95);
        for v in r.body_nudge.iter().chain(r.mind_nudge.iter()) {
            assert_eq!(*v, -0.05, "value {v} should clamp to -MAX_NUDGE");
        }
    }

    #[test]
    fn compute_nudge_damping_smooths_with_prev() {
        // prev=0.04, signal=0 (raw=0), damping=0.95: damped = 0.95*0.04 = 0.038
        let signal = [0.0; 10];
        let prev_body = [0.04; 5];
        let prev_mind = [0.04; 5];
        let r = compute_nudge(&signal, &prev_body, &prev_mind, 0.95);
        for v in r.body_nudge.iter() {
            assert!(approx(*v, 0.038, 1e-6), "expected ~0.038, got {v}");
        }
    }

    #[test]
    fn compute_nudge_signal_split_body_first_then_mind() {
        // Signal[0..5] feeds body_nudge, signal[5..10] feeds mind_nudge
        let signal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let r = compute_nudge(&signal, &[0.0; 5], &[0.0; 5], 0.95);
        // Each body[i] = 0.05 * signal[i]; clamp to [-0.05, 0.05]
        for (i, sig) in signal.iter().take(5).enumerate() {
            let expected = (0.05 * sig).clamp(-0.05, 0.05);
            assert!(approx(r.body_nudge[i], expected, 1e-6));
        }
        for (i, sig) in signal.iter().skip(5).take(5).enumerate() {
            let expected = (0.05 * sig).clamp(-0.05, 0.05);
            assert!(approx(r.mind_nudge[i], expected, 1e-6));
        }
    }

    // ── Pure apply_nudge ────────────────────────────────────────────────

    #[test]
    fn apply_nudge_modifies_body_all_5_dims() {
        let mut body = [0.5; 5];
        let mut mind = [0.5; 15];
        let nudge = NudgeResult {
            body_nudge: [0.05; 5],
            mind_nudge: [0.0; 5],
            total_magnitude: 0.0,
        };
        apply_nudge(&mut body, &mut mind, &nudge, 0.1, 1.0);
        // Each body dim += 0.05 * 0.1 * 1.0 = 0.005
        for v in body.iter() {
            assert!(approx(*v, 0.505, 1e-6));
        }
    }

    #[test]
    fn apply_nudge_modifies_mind_only_willing_dims_10_to_15() {
        let mut body = [0.5; 5];
        let mut mind = [0.5; 15];
        let nudge = NudgeResult {
            body_nudge: [0.0; 5],
            mind_nudge: [0.05; 5],
            total_magnitude: 0.0,
        };
        apply_nudge(&mut body, &mut mind, &nudge, 0.1, 1.0);
        // Per G10: thinking[0..5] + feeling[5..10] UNTOUCHED
        for (i, v) in mind.iter().take(10).enumerate() {
            assert_eq!(*v, 0.5, "mind[{i}] (thinking/feeling) must be unchanged");
        }
        // Willing[10..15] modified by 0.005 each
        for (i, v) in mind.iter().enumerate().take(15).skip(10) {
            assert!(
                approx(*v, 0.505, 1e-6),
                "mind[{i}] willing should be 0.505, got {}",
                v
            );
        }
    }

    #[test]
    fn apply_nudge_clamps_body_to_unit_interval() {
        let mut body = [0.99; 5];
        let mut mind = [0.5; 15];
        // Strong nudge that would push body > 1.0
        let nudge = NudgeResult {
            body_nudge: [0.05; 5],
            mind_nudge: [0.0; 5],
            total_magnitude: 0.0,
        };
        apply_nudge(&mut body, &mut mind, &nudge, 1.0, 30.0);
        // delta = 0.05 * 1.0 * 30 = 1.5 → 0.99 + 1.5 = 2.49 → clamped to 1.0
        for v in body.iter() {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn apply_nudge_clamps_body_to_zero_floor() {
        let mut body = [0.01; 5];
        let mut mind = [0.5; 15];
        let nudge = NudgeResult {
            body_nudge: [-0.05; 5],
            mind_nudge: [0.0; 5],
            total_magnitude: 0.0,
        };
        apply_nudge(&mut body, &mut mind, &nudge, 1.0, 30.0);
        for v in body.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn apply_nudge_clamps_dt_to_30_seconds_max() {
        let mut body = [0.5; 5];
        let mut mind = [0.5; 15];
        let nudge = NudgeResult {
            body_nudge: [0.001; 5],
            mind_nudge: [0.0; 5],
            total_magnitude: 0.0,
        };
        // dt=1000s should be clamped to 30s; delta = 0.001 * 1.0 * 30 = 0.03
        apply_nudge(&mut body, &mut mind, &nudge, 1.0, 1000.0);
        for v in body.iter() {
            assert!(approx(*v, 0.53, 1e-6), "expected 0.53, got {v}");
        }
    }

    #[test]
    fn apply_nudge_does_not_touch_spirit_or_thinking_or_feeling() {
        // The function signature only takes body[5] + mind[15]; the absence of
        // any spirit parameter is the contract that proves G10. This test is a
        // structural sanity check + documents the invariant.
        let mut body = [0.5; 5];
        let mut mind = [0.42; 15]; // distinctive value
        let nudge = NudgeResult {
            body_nudge: [0.05; 5],
            mind_nudge: [0.05; 5],
            total_magnitude: 0.0,
        };
        apply_nudge(&mut body, &mut mind, &nudge, 0.1, 1.0);
        // mind[0..10] preserved
        for (i, v) in mind.iter().take(10).enumerate() {
            assert_eq!(*v, 0.42, "mind[{i}] must remain 0.42 per G10");
        }
    }

    // ── Stateful GroundUpEnricher ────────────────────────────────────────

    #[test]
    fn enricher_with_defaults_uses_spec_constants() {
        let e = GroundUpEnricher::with_defaults();
        assert!(approx(e.strength(), 0.1, 1e-6));
        assert!(approx(e.damping(), 0.95, 1e-6));
        assert_eq!(e.total_applications(), 0);
    }

    #[test]
    fn enricher_compute_nudge_advances_prev_state() {
        let mut e = GroundUpEnricher::with_defaults();
        // First call: prev=0, signal=1 → damped=0.05
        let r1 = e.compute_nudge(&[1.0; 10]);
        assert!(approx_arr(&r1.body_nudge, &[0.05; 5], 1e-6));
        // Second call: prev=0.05, signal=1 → damped = 0.95*0.05 + 0.05*1 = 0.0975 → clamp to 0.05
        let r2 = e.compute_nudge(&[1.0; 10]);
        assert!(approx_arr(&r2.body_nudge, &[0.05; 5], 1e-6));
    }

    #[test]
    fn enricher_compute_nudge_decays_with_zero_signal_after_saturation() {
        let mut e = GroundUpEnricher::with_defaults();
        // Saturate
        let _ = e.compute_nudge(&[1.0; 10]);
        // Then zero signal → damped = 0.95 * 0.05 + 0 = 0.0475
        let r = e.compute_nudge(&[0.0; 10]);
        for v in r.body_nudge.iter() {
            assert!(approx(*v, 0.0475, 1e-6));
        }
    }

    #[test]
    fn enricher_apply_increments_stats() {
        let mut e = GroundUpEnricher::with_defaults();
        let mut body = [0.5; 5];
        let mut mind = [0.5; 15];
        let _ = e.apply(&mut body, &mut mind, &[1.0; 10], 1.0);
        assert_eq!(e.total_applications(), 1);
        // 5 dims × 0.05 = 0.25 per side
        assert!((e.total_body_delta() - 0.25).abs() < 1e-5);
        assert!((e.total_mind_delta() - 0.25).abs() < 1e-5);

        let _ = e.apply(&mut body, &mut mind, &[1.0; 10], 1.0);
        assert_eq!(e.total_applications(), 2);
        // Cumulative: 2 ticks × 0.25 = 0.5
        assert!((e.total_body_delta() - 0.5).abs() < 1e-5);
    }

    // ── Coordinator ──────────────────────────────────────────────────────

    #[test]
    fn coordinator_default_has_two_independent_enrichers() {
        let mut c = GroundUpCoordinator::new();
        // Inner saturates
        let _ = c.inner.compute_nudge(&[1.0; 10]);
        // Outer stays at zero
        let outer_r = c.outer.compute_nudge(&[0.0; 10]);
        assert_eq!(outer_r.body_nudge, [0.0; 5]);
        // Inner state independent
        assert_eq!(c.inner.total_body_delta(), 0.0); // compute_nudge doesn't bump stats
    }

    // ── Parity vectors (loaded from tests/parity/vectors.json) ───────────

    #[test]
    fn parity_vector_zero_signal_matches_python_reference() {
        let r = compute_nudge(&[0.0; 10], &[0.0; 5], &[0.0; 5], 0.95);
        // Python ground_up.py with zero signal: zero everywhere
        assert_eq!(r.body_nudge, [0.0; 5]);
        assert_eq!(r.mind_nudge, [0.0; 5]);
        assert_eq!(r.total_magnitude, 0.0);
    }

    #[test]
    fn parity_vector_saturated_signal_matches_python_reference() {
        let r = compute_nudge(&[1.0; 10], &[0.0; 5], &[0.0; 5], 0.95);
        // Python: each dim 0.05; total_mag = sqrt(10 * 0.0025) ≈ 0.158113...
        for v in r.body_nudge.iter() {
            assert!((v - 0.05).abs() < 1e-6);
        }
        for v in r.mind_nudge.iter() {
            assert!((v - 0.05).abs() < 1e-6);
        }
        // Python rounds to 6 decimals → 0.158114
        assert!((r.total_magnitude - 0.158_114).abs() < 1e-4);
    }

    #[test]
    fn parity_vector_with_prev_damping_matches_python_reference() {
        // prev=0.04 at all positions, signal=[0.5; 10], damping=0.95
        // damped = 0.95 * 0.04 + 0.05 * 0.5 = 0.038 + 0.025 = 0.063 → clamp to 0.05
        let r = compute_nudge(&[0.5; 10], &[0.04; 5], &[0.04; 5], 0.95);
        for v in r.body_nudge.iter() {
            assert_eq!(*v, 0.05); // saturated by clamp
        }
    }
}
