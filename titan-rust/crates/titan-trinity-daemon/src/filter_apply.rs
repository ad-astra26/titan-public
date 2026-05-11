//! filter_apply — Daemon-side application of FILTER_DOWN multipliers.
//!
//! Mirrors `filter_down.py:483-498` + `titan_params.toml:256-284` byte-
//! identically. Inner daemons receive UNIFIED_SPIRIT_FILTER_DOWN multipliers
//! from titan-unified-spirit-rs (C-S4) AND optionally local
//! INNER_SPIRIT_FILTER_DOWN multipliers from titan-inner-spirit-rs (THIS
//! session). The two are composed multiplicatively + clipped, then applied
//! multiplicatively to the daemon's tensor with `[0, 1]` clipping.
//!
//! # SPEC ground truths
//!
//! - **G5** FILTER_DOWN + GROUND_UP push toward 0.5 — productive tension.
//! - **G7** Output is 120 multipliers (60 inner + 60 outer); inner consumed
//!   here = body[5] + mind[15] + spirit_content[40] = 60.
//! - **G8** Inner-Spirit observer dims `[0:5]` (= absolute `[20:25]`) NEVER
//!   appear as multipliers. Daemon receives 40D for inner_spirit_content.
//! - **G9** Spirit content multipliers scaled by
//!   [`SPIRIT_FILTER_STRENGTH_MULT`] toward 1.0 — applied at PUBLISH side
//!   (V5 engine's `_scale` per filter_down.py:737-740). Daemon-side apply
//!   does NOT re-scale.
//!
//! # Constants
//!
//! All values lifted from SPEC G7 prose + `filter_down.py:50-52` +
//! `titan_params.toml:256/262/284`. C-S3 lands them in
//! `titan-core::constants` as `FILTER_DOWN_MULTIPLIER_FLOOR/CEIL/...` per
//! SPEC TOML v0.1.3 (already in progress at
//! `.claude/worktrees/phase_c_s3_substrate`). When C-S3 merges to titan-v6,
//! refactor to `use titan_core::constants::FILTER_DOWN_*` and delete the
//! local consts below.
//!
//! # EMA smoothing on the publisher side
//!
//! Per `filter_down.py:429` + `:742-743`, EMA happens at the V5 engine's
//! publish path (smoothing 0.9 across consecutive UNIFIED_SPIRIT_FILTER_DOWN
//! emissions). The DAEMON receives ALREADY-SMOOTHED multipliers and does
//! NOT apply EMA again. The [`EmaSmoother`] helper in this module is for
//! the **inner-spirit daemon's INNER_SPIRIT_FILTER_DOWN publish path** —
//! when inner-spirit publishes its LOCAL multipliers, it smooths them
//! exactly like V5 does.

// SPEC v0.1.3 (C-S3) ships the canonical f64 versions in
// titan-core::constants. Daemon-facing f32 aliases below — slot payloads
// + multipliers are float32 per SPEC §7.1.

/// Minimum severity multiplier — never fully mute a sense.
/// Sourced from `titan_core::constants::FILTER_DOWN_MULTIPLIER_FLOOR`.
/// Per SPEC G7 + `filter_down.py:50`.
pub const MULTIPLIER_FLOOR: f32 = titan_core::constants::FILTER_DOWN_MULTIPLIER_FLOOR as f32;

/// Maximum severity multiplier.
/// Sourced from `titan_core::constants::FILTER_DOWN_MULTIPLIER_CEIL`.
/// Per SPEC G7 + `filter_down.py:51`.
pub const MULTIPLIER_CEIL: f32 = titan_core::constants::FILTER_DOWN_MULTIPLIER_CEIL as f32;

/// Spirit content multiplier strength (pulls spirit mults toward 1.0).
/// Sourced from `titan_core::constants::FILTER_DOWN_SPIRIT_STRENGTH_MULT`.
/// Per SPEC G9 + `filter_down.py:737-740`. Applied at PUBLISH side.
pub const SPIRIT_FILTER_STRENGTH_MULT: f32 =
    titan_core::constants::FILTER_DOWN_SPIRIT_STRENGTH_MULT as f32;

/// EMA smoothing factor for multiplier updates between consecutive publishes.
/// `new_mult = SMOOTHING * old + (1 - SMOOTHING) * incoming`.
/// Per `filter_down.py:52 SMOOTHING=0.9` + `:742-743`.
pub const FILTER_DOWN_EMA_SMOOTHING: f32 = 0.9;

/// Tensor value lower bound (Titan tensors live in `[0, 1]`).
pub const TENSOR_MIN: f32 = 0.0;

/// Tensor value upper bound.
pub const TENSOR_MAX: f32 = 1.0;

/// Apply a multiplier vector to a tensor in place. `out[i] = clip(tensor[i] *
/// mults[i], TENSOR_MIN, TENSOR_MAX)`.
///
/// `tensor.len()` and `mults.len()` MUST match. Shorter `mults` panics.
pub fn apply_multipliers(tensor: &mut [f32], mults: &[f32]) {
    debug_assert_eq!(tensor.len(), mults.len(), "tensor + mults dim mismatch");
    for i in 0..tensor.len() {
        tensor[i] = (tensor[i] * mults[i]).clamp(TENSOR_MIN, TENSOR_MAX);
    }
}

/// Compose two multiplier vectors element-wise: `out[i] = clip(a[i] * b[i],
/// FLOOR, CEIL)`. Used to combine UNIFIED + LOCAL filter_down multipliers
/// before apply.
///
/// Returns a new `Vec<f32>` since fixed-size composition lives at the call
/// site.
pub fn compose_multipliers(a: &[f32], b: &[f32], floor: f32, ceil: f32) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "a + b dim mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| (av * bv).clamp(floor, ceil))
        .collect()
}

/// Compose two multiplier vectors with the SPEC-default floor / ceil.
pub fn compose_multipliers_default(a: &[f32], b: &[f32]) -> Vec<f32> {
    compose_multipliers(a, b, MULTIPLIER_FLOOR, MULTIPLIER_CEIL)
}

/// Apply spirit-strength scaling (pulls multipliers toward 1.0): `out[i] =
/// (m[i] - 1.0) * k + 1.0`. Used by inner-spirit daemon when publishing
/// INNER_SPIRIT_FILTER_DOWN. Per `filter_down.py:737-740`.
pub fn apply_spirit_strength(mults: &mut [f32], strength: f32) {
    for m in mults.iter_mut() {
        *m = (*m - 1.0) * strength + 1.0;
    }
}

/// Stateful EMA smoother for multiplier publish paths. Inner-spirit holds
/// one of these per multiplier vector field (inner_body / inner_mind /
/// inner_spirit_content) and runs `update(new_mults)` per publish to get
/// smoothed values.
///
/// Mirrors `filter_down.py:742-743` byte-identically.
#[derive(Debug, Clone)]
pub struct EmaSmoother {
    smoothing: f32,
    state: Vec<f32>,
    initialized: bool,
}

impl EmaSmoother {
    /// New smoother of size `dim`. State initialized to zeros (overwritten
    /// on first `update` call to avoid bias toward zero).
    pub fn new(dim: usize) -> Self {
        Self::with_smoothing(dim, FILTER_DOWN_EMA_SMOOTHING)
    }

    /// New smoother with explicit smoothing factor (for testing).
    pub fn with_smoothing(dim: usize, smoothing: f32) -> Self {
        Self {
            smoothing,
            state: vec![0.0; dim],
            initialized: false,
        }
    }

    /// First-call replaces state directly with `new` (avoids bias toward
    /// zero start). Subsequent calls EMA-smooth.
    pub fn update(&mut self, new: &[f32]) -> &[f32] {
        debug_assert_eq!(new.len(), self.state.len(), "EMA dim mismatch");
        if !self.initialized {
            self.state.copy_from_slice(new);
            self.initialized = true;
        } else {
            for (s, n) in self.state.iter_mut().zip(new.iter()) {
                *s = self.smoothing * *s + (1.0 - self.smoothing) * n;
            }
        }
        &self.state
    }

    /// Current state (read-only).
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Restore from snapshot (for hot-reload). Marks initialized.
    pub fn restore(&mut self, state: Vec<f32>) {
        debug_assert_eq!(state.len(), self.state.len());
        self.state = state;
        self.initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn apply_multipliers_clips_to_tensor_range() {
        let mut tensor = [0.5, 0.5, 0.5, 0.5, 0.5];
        // 3.0 * 0.5 = 1.5 → clip 1.0; 0.3 * 0.5 = 0.15 (no clip)
        let mults = [3.0, 0.3, 1.0, 2.0, 0.5];
        apply_multipliers(&mut tensor, &mults);
        assert!(approx_eq(tensor[0], 1.0, 1e-6)); // clipped
        assert!(approx_eq(tensor[1], 0.15, 1e-6));
        assert!(approx_eq(tensor[2], 0.5, 1e-6));
        assert!(approx_eq(tensor[3], 1.0, 1e-6)); // 2.0*0.5=1.0 exactly
        assert!(approx_eq(tensor[4], 0.25, 1e-6));
    }

    #[test]
    fn apply_multipliers_neutral_unchanged() {
        let mut tensor = [0.1, 0.2, 0.3, 0.4, 0.5];
        let mults = [1.0; 5];
        apply_multipliers(&mut tensor, &mults);
        assert!(approx_eq(tensor[0], 0.1, 1e-6));
        assert!(approx_eq(tensor[4], 0.5, 1e-6));
    }

    #[test]
    fn apply_multipliers_zero_floors_to_min() {
        // Tensor=0; any mult; result clips to TENSOR_MIN=0
        let mut tensor = [0.0; 5];
        let mults = [3.0; 5];
        apply_multipliers(&mut tensor, &mults);
        for v in tensor.iter() {
            assert!(approx_eq(*v, TENSOR_MIN, 1e-6));
        }
    }

    #[test]
    fn compose_multipliers_multiplies_and_clips() {
        let a = vec![1.0, 2.0, 0.5, 3.0, 0.2];
        let b = vec![1.0, 2.0, 1.0, 2.0, 0.5];
        let composed = compose_multipliers_default(&a, &b);
        // [1*1=1, 2*2=4→clip 3, 0.5*1=0.5, 3*2=6→clip 3, 0.2*0.5=0.1→clip 0.3]
        assert!(approx_eq(composed[0], 1.0, 1e-6));
        assert!(approx_eq(composed[1], MULTIPLIER_CEIL, 1e-6));
        assert!(approx_eq(composed[2], 0.5, 1e-6));
        assert!(approx_eq(composed[3], MULTIPLIER_CEIL, 1e-6));
        assert!(approx_eq(composed[4], MULTIPLIER_FLOOR, 1e-6));
    }

    #[test]
    fn compose_multipliers_neutral_returns_a() {
        let a = vec![0.5, 1.0, 1.5, 2.0, 0.3];
        let b = vec![1.0; 5];
        let composed = compose_multipliers_default(&a, &b);
        for i in 0..5 {
            assert!(approx_eq(composed[i], a[i], 1e-6));
        }
    }

    #[test]
    fn apply_spirit_strength_pulls_toward_one() {
        // strength=0.3: m=2.0 → (2-1)*0.3+1 = 1.3; m=0.5 → (0.5-1)*0.3+1 = 0.85
        let mut mults = [2.0, 0.5, 1.0, 3.0, 0.3];
        apply_spirit_strength(&mut mults, SPIRIT_FILTER_STRENGTH_MULT);
        assert!(approx_eq(mults[0], 1.3, 1e-6));
        assert!(approx_eq(mults[1], 0.85, 1e-6));
        assert!(approx_eq(mults[2], 1.0, 1e-6)); // 1.0 unchanged
        assert!(approx_eq(mults[3], 1.6, 1e-6));
        assert!(approx_eq(mults[4], 0.79, 1e-6));
    }

    #[test]
    fn apply_spirit_strength_zero_collapses_to_neutral() {
        // strength=0: every mult becomes 1.0
        let mut mults = [0.5, 2.0, 3.0, 0.3, 1.5];
        apply_spirit_strength(&mut mults, 0.0);
        for v in mults.iter() {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
    }

    #[test]
    fn ema_first_update_replaces_state() {
        let mut s = EmaSmoother::new(3);
        let out = s.update(&[0.5, 1.0, 2.0]);
        // First call: no smoothing (avoid bias toward 0)
        assert!(approx_eq(out[0], 0.5, 1e-6));
        assert!(approx_eq(out[1], 1.0, 1e-6));
        assert!(approx_eq(out[2], 2.0, 1e-6));
    }

    #[test]
    fn ema_subsequent_smooths_with_0_9() {
        let mut s = EmaSmoother::new(1);
        s.update(&[1.0]);
        // Second: 0.9*1.0 + 0.1*5.0 = 1.4
        let out = s.update(&[5.0]);
        assert!(approx_eq(out[0], 1.4, 1e-6));
    }

    #[test]
    fn ema_steady_state_converges() {
        // Repeatedly feed same value → EMA converges to it
        let mut s = EmaSmoother::new(1);
        s.update(&[2.0]);
        for _ in 0..200 {
            s.update(&[2.0]);
        }
        assert!(approx_eq(s.state()[0], 2.0, 1e-4));
    }

    #[test]
    fn ema_restore_marks_initialized() {
        let mut s = EmaSmoother::new(3);
        s.restore(vec![1.0, 2.0, 3.0]);
        // After restore, next update should EMA-smooth (NOT replace)
        let out = s.update(&[5.0, 5.0, 5.0]);
        // 0.9*1 + 0.1*5 = 1.4; 0.9*2 + 0.1*5 = 2.3; 0.9*3 + 0.1*5 = 3.2
        assert!(approx_eq(out[0], 1.4, 1e-6));
        assert!(approx_eq(out[1], 2.3, 1e-6));
        assert!(approx_eq(out[2], 3.2, 1e-6));
    }

    #[test]
    fn ema_with_custom_smoothing_factor() {
        let mut s = EmaSmoother::with_smoothing(1, 0.5);
        s.update(&[10.0]);
        // 0.5*10 + 0.5*0 = 5.0
        let out = s.update(&[0.0]);
        assert!(approx_eq(out[0], 5.0, 1e-6));
    }

    #[test]
    fn constants_match_python_reference() {
        // Sanity: compile-time guard against accidental drift from
        // filter_down.py:50-52 + titan_params.toml.
        assert!(approx_eq(MULTIPLIER_FLOOR, 0.3, 1e-9));
        assert!(approx_eq(MULTIPLIER_CEIL, 3.0, 1e-9));
        assert!(approx_eq(SPIRIT_FILTER_STRENGTH_MULT, 0.3, 1e-9));
        assert!(approx_eq(FILTER_DOWN_EMA_SMOOTHING, 0.9, 1e-9));
        assert!(approx_eq(TENSOR_MIN, 0.0, 1e-9));
        assert!(approx_eq(TENSOR_MAX, 1.0, 1e-9));
    }
}
