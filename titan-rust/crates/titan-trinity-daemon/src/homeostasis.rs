//! homeostasis — the stateful traveling-tensor kernel (SPEC §G5.2, D-SPEC-112).
//!
//! Trinity Middle-Path Homeostasis P0-0b. Closes the verified oscillator gap:
//! the trinity-part tensors were *recomputed from scratch each tick* (stateless),
//! so a restoring force applied after the recompute was erased — the tensor could
//! not "travel" and balance was not an attractor. This module makes each part
//! tensor a **persistent state that travels**, integrating:
//!
//! ```text
//! x[t] = clamp[0,1]( x[t-1]
//!   + k_drive    · (enriched − x[t-1])            // experience drive (raw producer + filter_down/ground_up
//!                                                 //   folded into the enriched target: enrichment_force =
//!                                                 //   k_drive·(enriched−raw), per SPEC §G5.2)
//!   − k_restore·g · (x[t-1] − 0.5)                // PD restoring spring toward the 0.5 Divine Centre
//!                                                 //   (covers ALL layers incl. spirit's 45D; g = neuromod gain)
//!   + w_quant·k_momentum · (x[t-1] − x[t-2])      // QUANTITATIVE observable-feedback: velocity-momentum carry
//!   + w_qual ·k_cohesion · (mean(x[t-1]) − x[t-1]) ) // QUALITATIVE observable-feedback: coherence cohesion
//! ```
//!
//! The 5 observables (coherence/magnitude/velocity/direction/polarity) split into
//! the **quantitative** half (magnitude/velocity) and **qualitative** half
//! (coherence/direction/polarity); the per-layer **quant→qual gradient** (body
//! leans quantitative, spirit qualitative — INV-9) weights the feedback. Here the
//! velocity observable manifests as the momentum carry; the coherence observable
//! as the cohesion-to-layer-mean (variance-reduction) term.
//!
//! **No hardcoded no-signal floor** (SPEC §G5.2 item 3): a quiet dim holds its
//! last value and drifts toward centre under the spring; a *persistently* dead dim
//! is a formula bug to re-ground (P0-0c / D-SPEC-104), never floored here.
//!
//! Gains are crate-local tuning constants (the `sphere_clocks` precedent — tunable
//! per rebuild, not SPEC-TOML constants), conservative starting values; an explicit
//! tuning pass is expected (SPEC §G5.2 item 5 / OPEN-5).

/// Which trinity layer a daemon owns — selects the quant→qual gradient (INV-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Body (material) — leans quantitative.
    Body,
    /// Mind (transitional).
    Mind,
    /// Spirit (ethereal) — leans qualitative.
    Spirit,
}

// ── Crate-local tuning constants (sphere_clocks precedent; tunable per rebuild) ──

/// Drive coefficient — how fast the tensor tracks the enriched producer target
/// per tick. Conservative: the tensor follows the producer but with lag, so it
/// genuinely *travels* rather than snapping. (SPEC §G5.2 / OPEN-5 tunable.)
pub const DEFAULT_K_DRIVE: f32 = 0.30;

/// Restoring-spring coefficient — gentle PD pull toward the 0.5 centre, ∝ delta.
/// Conservative (errs toward preserving signal); raise during the tuning pass if
/// coherence does not reach the §G11 balance threshold.
pub const DEFAULT_K_RESTORE: f32 = 0.05;

/// Momentum coefficient — quantitative observable-feedback (velocity carry).
pub const DEFAULT_K_MOMENTUM: f32 = 0.10;

/// Cohesion coefficient — qualitative observable-feedback (pull toward layer mean
/// = variance-reduction = coherence-raising).
pub const DEFAULT_K_COHESION: f32 = 0.10;

/// Per-layer quant→qual gradient (INV-9): `(w_quant, w_qual)`.
/// body 0.7/0.3, mind 0.5/0.5, spirit 0.3/0.7.
pub const fn gradient(layer: Layer) -> (f32, f32) {
    match layer {
        Layer::Body => (0.7, 0.3),
        Layer::Mind => (0.5, 0.5),
        Layer::Spirit => (0.3, 0.7),
    }
}

/// The 0.5 Divine Centre — the per-dim restoring-force target (SPEC §G5 GROUND_REFERENCE).
pub const CENTRE: f32 = 0.5;

/// Restoring-force + observable-feedback configuration for one layer.
#[derive(Debug, Clone, Copy)]
pub struct RestoringCfg {
    /// Drive coefficient (toward the enriched producer target).
    pub k_drive: f32,
    /// Restoring-spring coefficient (toward 0.5 centre).
    pub k_restore: f32,
    /// Momentum coefficient (quantitative feedback).
    pub k_momentum: f32,
    /// Cohesion coefficient (qualitative feedback).
    pub k_cohesion: f32,
    /// Quantitative gradient weight for this layer.
    pub w_quant: f32,
    /// Qualitative gradient weight for this layer.
    pub w_qual: f32,
    /// Neuromod gain — multiplies `k_restore` (stronger pull under sustained
    /// imbalance). 1.0 = neutral; read from `neuromod_state.bin` at runtime.
    pub neuromod_gain: f32,
}

impl RestoringCfg {
    /// Construct with the crate-local default gains + the layer's gradient.
    pub fn for_layer(layer: Layer) -> Self {
        let (w_quant, w_qual) = gradient(layer);
        Self {
            k_drive: DEFAULT_K_DRIVE,
            k_restore: DEFAULT_K_RESTORE,
            k_momentum: DEFAULT_K_MOMENTUM,
            k_cohesion: DEFAULT_K_COHESION,
            w_quant,
            w_qual,
            neuromod_gain: 1.0,
        }
    }
}

/// One layer's 5D observable signature (SPEC §G4 / topology.py OBSERVABLE_KEYS).
/// Computed for telemetry, the balance-gated UP-leg snapshot (P0-0a), MSL reward
/// (P0 §6), and the journey checkpoint. The update kernel realizes the velocity +
/// coherence observables directly as the momentum + cohesion forces.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerObs {
    /// `1 − variance/0.25`, clamped [0,1] (SPEC §G4 / D-SPEC-84).
    pub coherence: f32,
    /// `l2_norm / sqrt(N)`, clamped [0,1].
    pub magnitude: f32,
    /// `|magnitude − prev_magnitude|`, clamped [0,1].
    pub velocity: f32,
    /// `sign(magnitude − prev_magnitude)` ∈ {−1,0,+1}.
    pub direction: f32,
    /// `clamp((mean − 0.5)·2, −1, 1)`.
    pub polarity: f32,
}

/// Mean of a tensor.
fn mean(t: &[f32]) -> f32 {
    if t.is_empty() {
        return CENTRE;
    }
    t.iter().sum::<f32>() / t.len() as f32
}

/// `coherence = 1 − variance/0.25`, clamped [0,1] (parity with
/// `middle_path.layer_coherence` / topology.rs).
fn coherence(t: &[f32]) -> f32 {
    if t.len() < 2 {
        return 1.0;
    }
    let m = mean(t);
    let var = t.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / t.len() as f32;
    (1.0 - var / 0.25).clamp(0.0, 1.0)
}

/// `magnitude = l2_norm / sqrt(N)`, clamped [0,1].
fn magnitude(t: &[f32]) -> f32 {
    if t.is_empty() {
        return 0.0;
    }
    let l2 = t.iter().map(|&v| v * v).sum::<f32>().sqrt();
    (l2 / (t.len() as f32).sqrt()).clamp(0.0, 1.0)
}

/// Compute the 5D observable signature of `cur` relative to `prev` (parity with
/// `TopologyEngine::derive_layer_observables`). On the first tick (`prev == cur`)
/// velocity/direction are 0.
pub fn observe(cur: &[f32], prev: &[f32]) -> LayerObs {
    let mag = magnitude(cur);
    let prev_mag = magnitude(prev);
    let dmag = mag - prev_mag;
    LayerObs {
        coherence: coherence(cur),
        magnitude: mag,
        velocity: dmag.abs().clamp(0.0, 1.0),
        direction: if dmag > 0.0 {
            1.0
        } else if dmag < 0.0 {
            -1.0
        } else {
            0.0
        },
        polarity: ((mean(cur) - CENTRE) * 2.0).clamp(-1.0, 1.0),
    }
}

/// The §G5.2 stateful-update kernel. Integrates the previous tensor (`prev` =
/// x[t-1], `prev2` = x[t-2]) toward the enriched producer target under the
/// restoring spring + observable-feedback. Returns x[t], clamped [0,1].
///
/// `enriched` is the current per-tick producer output AFTER filter_down + ground_up
/// (its delta from raw is the §G5.2 `enrichment_force`, folded into the drive).
pub fn stateful_update(
    prev: &[f32],
    prev2: &[f32],
    enriched: &[f32],
    cfg: &RestoringCfg,
) -> Vec<f32> {
    let n = prev.len();
    debug_assert_eq!(n, enriched.len());
    debug_assert_eq!(n, prev2.len());
    let mean_prev = mean(prev);
    let k_restore_eff = cfg.k_restore * cfg.neuromod_gain;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let drive = cfg.k_drive * (enriched[i] - prev[i]);
        let spring = -k_restore_eff * (prev[i] - CENTRE);
        let momentum = cfg.w_quant * cfg.k_momentum * (prev[i] - prev2[i]);
        let cohesion = cfg.w_qual * cfg.k_cohesion * (mean_prev - prev[i]);
        let x = prev[i] + drive + spring + momentum + cohesion;
        out.push(x.clamp(0.0, 1.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn gradient_per_layer() {
        assert_eq!(gradient(Layer::Body), (0.7, 0.3));
        assert_eq!(gradient(Layer::Mind), (0.5, 0.5));
        assert_eq!(gradient(Layer::Spirit), (0.3, 0.7));
    }

    #[test]
    fn first_tick_no_velocity() {
        let t = vec![0.2, 0.8, 0.5];
        let o = observe(&t, &t);
        assert_eq!(o.velocity, 0.0);
        assert_eq!(o.direction, 0.0);
    }

    #[test]
    fn coherence_uniform_is_one() {
        assert!(approx(coherence(&[0.4, 0.4, 0.4, 0.4]), 1.0, 1e-6));
        // bimodal 0/1 → variance 0.25 → coherence 0
        assert!(approx(coherence(&[0.0, 1.0, 0.0, 1.0]), 0.0, 1e-6));
    }

    #[test]
    fn drift_then_return_toward_centre() {
        // A tensor perturbed far from centre, with a NEUTRAL producer target at
        // its own value (no drive), returns toward 0.5 under the spring over ticks.
        let cfg = RestoringCfg {
            k_drive: 0.0, // isolate the spring
            ..RestoringCfg::for_layer(Layer::Spirit)
        };
        let mut prev2 = vec![0.9_f32; 4];
        let mut prev = vec![0.9_f32; 4];
        for _ in 0..50 {
            let x = stateful_update(&prev, &prev2, &prev, &cfg); // enriched = prev → drive 0
            prev2 = prev;
            prev = x;
        }
        // Should have moved from 0.9 toward 0.5 (not past it, not pinned).
        assert!(prev[0] < 0.9 && prev[0] > 0.5, "got {}", prev[0]);
    }

    #[test]
    fn zero_restore_pure_drift_regression() {
        // k_restore=0 → no centre-pull → tensor tracks the producer target only.
        let cfg = RestoringCfg {
            k_restore: 0.0,
            k_momentum: 0.0,
            k_cohesion: 0.0,
            ..RestoringCfg::for_layer(Layer::Body)
        };
        let prev2 = vec![0.5_f32; 5];
        let prev = vec![0.5_f32; 5];
        let enriched = vec![0.8_f32; 5];
        let x = stateful_update(&prev, &prev2, &enriched, &cfg);
        // moves k_drive*(0.8-0.5)=0.09 toward target → 0.59
        assert!(approx(x[0], 0.59, 1e-5), "got {}", x[0]);
    }

    #[test]
    fn cohesion_reduces_variance() {
        // A spread tensor with only the cohesion term should tighten toward its mean.
        let cfg = RestoringCfg {
            k_drive: 0.0,
            k_restore: 0.0,
            k_momentum: 0.0,
            k_cohesion: 0.5,
            w_qual: 1.0,
            w_quant: 0.0,
            neuromod_gain: 1.0,
        };
        let prev = vec![0.2_f32, 0.8, 0.3, 0.7];
        let prev2 = prev.clone();
        let coh_before = coherence(&prev);
        let x = stateful_update(&prev, &prev2, &prev, &cfg);
        assert!(coherence(&x) > coh_before, "coherence should rise");
    }

    #[test]
    fn clamped_to_unit_interval() {
        let cfg = RestoringCfg::for_layer(Layer::Body);
        let prev = vec![0.95_f32; 5];
        let prev2 = vec![0.0_f32; 5];
        let enriched = vec![1.5_f32; 5]; // out-of-range producer
        let x = stateful_update(&prev, &prev2, &enriched, &cfg);
        for v in x {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn neuromod_gain_scales_restore() {
        let mut cfg = RestoringCfg {
            k_drive: 0.0,
            k_momentum: 0.0,
            k_cohesion: 0.0,
            ..RestoringCfg::for_layer(Layer::Body)
        };
        let prev = vec![0.9_f32; 3];
        let prev2 = prev.clone();
        cfg.neuromod_gain = 1.0;
        let x1 = stateful_update(&prev, &prev2, &prev, &cfg);
        cfg.neuromod_gain = 2.0;
        let x2 = stateful_update(&prev, &prev2, &prev, &cfg);
        // higher gain → stronger pull toward 0.5 → lower value
        assert!(x2[0] < x1[0]);
    }
}
