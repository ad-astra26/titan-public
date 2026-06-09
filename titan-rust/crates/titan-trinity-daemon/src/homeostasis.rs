//! homeostasis — the stateful traveling-tensor kernel (SPEC §G5.2, D-SPEC-112).
//!
//! Trinity Middle-Path Homeostasis P0-0b. Closes the verified oscillator gap:
//! the trinity-part tensors were *recomputed from scratch each tick* (stateless),
//! so a restoring force applied after the recompute was erased — the tensor could
//! not "travel" and balance was not an attractor. This module makes each part
//! tensor a **persistent state that travels**, integrating the §G5.2 equation
//! with **all 5 observables** in their §9.0 / §9.2 roles:
//!
//! ```text
//! x[t] = clamp[0,1](
//!    x[t-1]
//!  + k_drive · (1 + w_quant·a_mag·(2·magnitude − 1)) · (raw[t] − x[t-1])
//!       // DRIVE with magnitude LANDING-GAIN (§9.0 "magnitude → how hard
//!       //                                     enrichment lands"; quant-weighted)
//!  + enrichment_force[t]
//!       // separate FULL-WEIGHT additive term — held filter_down/ground_up
//!       // (§G5.1 / §G10 held-value model); NOT folded into the drive.
//!  − k_restore · g_neuro · (1 + w_qual·a_drift·((1 − coherence) + |polarity|)) · (x[t-1] − 0.5)
//!       // RESTORING SPRING — §G5.2 item 2 + §9.0 "coherence + polarity →
//!       //                    how strongly the spring pulls to centre"
//!       // Drift signal amplifies pull off-centre. Qual-weighted (spirit pulls
//!       // harder). g_neuro = neuromod gain read from neuromod_state.bin.
//!  − k_damp · (1 + w_quant·a_dmag·magnitude) · (x[t-1] − x[t-2])
//!       // PD DAMPING — §9.2 "PD-shaped: spring on position/coherence + damping
//!       //                    on velocity". Quant-weighted (body damps fast/large
//!       //                    excursions hardest); magnitude scales authority.
//!  + w_quant · k_mom · (x[t-1] − x[t-2])
//!       // MOMENTUM — §9.0 "velocity → carry-forward momentum". Quant-weighted.
//!  + w_qual  · k_dir · direction · |x[t-1] − x[t-2]|
//!       // CONTINUITY — §9.0 "direction → trajectory continuity". Qual-weighted.
//! )
//! ```
//!
//! **All 5 observables present, no double-count:** velocity → momentum + damping;
//! magnitude → drive-landing + damping authority; coherence → spring strength;
//! polarity → spring strength; direction → continuity. **Gradient-weighted** per
//! INV-9 (body 0.7/0.3 quant/qual, mind 0.5/0.5, spirit 0.3/0.7). **Neuromod gain**
//! modulates the spring per §G5.2 item 2 (caller passes `cfg.neuromod_gain` read
//! from `neuromod_state.bin`). **Enrichment is a separate full-weight additive
//! term** per the §G5.2 equation literal. **No hardcoded no-signal floor** per
//! §G5.2 item 3.
//!
//! Gain coefficients (`k_drive`, `k_restore`, `k_damp`, `k_mom`, `k_dir`,
//! `a_mag`, `a_drift`, `a_dmag`) are conservative starting values here; per
//! SPEC §G5.2 item 5 they MUST be sourced from `titan_params.toml
//! [trinity_restoring]` at the daemon's construction time (gate clause
//! G5.2-config — separate slice). The values here are the rebuild-defaults
//! until the TOML loader lands.

/// Which trinity layer a daemon owns — selects the quant→qual gradient (INV-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    /// Body (material) — leans quantitative.
    Body,
    /// Mind (transitional).
    Mind,
    /// Spirit (ethereal) — leans qualitative.
    Spirit,
}

// ── Crate-local tuning defaults (sphere_clocks precedent; conservative starts) ──

/// Drive coefficient — how fast the tensor tracks the raw producer per tick.
pub const DEFAULT_K_DRIVE: f32 = 0.30;
/// Restoring-spring coefficient — gentle PD position pull toward 0.5.
pub const DEFAULT_K_RESTORE: f32 = 0.05;
/// PD damping coefficient — gentle opposition to velocity (§9.2).
pub const DEFAULT_K_DAMP: f32 = 0.05;
/// Momentum coefficient — velocity-carry feedback (§9.0).
pub const DEFAULT_K_MOMENTUM: f32 = 0.10;
/// Direction-continuity coefficient — trajectory bias (§9.0).
pub const DEFAULT_K_DIRECTION: f32 = 0.05;
/// Magnitude landing-gain amplitude — `drive_gain ∈ [1−wq·a_mag, 1+wq·a_mag]`.
pub const DEFAULT_A_MAG: f32 = 0.50;
/// Spring drift-modulation amplitude — `spring_gain = 1 + wq·a_drift·((1−coh)+|pol|)`.
pub const DEFAULT_A_DRIFT: f32 = 1.00;
/// Damping magnitude-modulation amplitude — `damp_gain = 1 + wq·a_dmag·magnitude`.
pub const DEFAULT_A_DMAG: f32 = 0.50;

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

/// Restoring-force + observable-feedback configuration for one layer (SPEC §G5.2 item 5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestoringCfg {
    /// Drive coefficient — how fast tensor tracks the raw producer.
    pub k_drive: f32,
    /// Restoring-spring coefficient (position pull toward 0.5 centre).
    pub k_restore: f32,
    /// PD damping coefficient (opposes velocity; §9.2).
    pub k_damp: f32,
    /// Momentum (velocity-carry) coefficient.
    pub k_mom: f32,
    /// Direction-continuity coefficient.
    pub k_dir: f32,
    /// Magnitude landing-gain amplitude.
    pub a_mag: f32,
    /// Spring drift-modulation amplitude.
    pub a_drift: f32,
    /// Damping magnitude-modulation amplitude.
    pub a_dmag: f32,
    /// Quantitative gradient weight for this layer.
    pub w_quant: f32,
    /// Qualitative gradient weight for this layer.
    pub w_qual: f32,
    /// Neuromod gain — multiplies `k_restore` per SPEC §G5.2 item 2.
    /// `1.0` = neutral; read from `neuromod_state.bin` at runtime by the daemon.
    pub neuromod_gain: f32,
}

impl RestoringCfg {
    /// Construct with the crate-local default gains + the layer's gradient.
    /// Per SPEC §G5.2 item 5 the daemons SHOULD override these from
    /// `titan_params.toml [trinity_restoring]` at boot (gate clause G5.2-config).
    pub fn for_layer(layer: Layer) -> Self {
        let (w_quant, w_qual) = gradient(layer);
        Self {
            k_drive: DEFAULT_K_DRIVE,
            k_restore: DEFAULT_K_RESTORE,
            k_damp: DEFAULT_K_DAMP,
            k_mom: DEFAULT_K_MOMENTUM,
            k_dir: DEFAULT_K_DIRECTION,
            a_mag: DEFAULT_A_MAG,
            a_drift: DEFAULT_A_DRIFT,
            a_dmag: DEFAULT_A_DMAG,
            w_quant,
            w_qual,
            neuromod_gain: 1.0,
        }
    }
}

/// One layer's 5D observable signature (SPEC §G4 / topology.py OBSERVABLE_KEYS).
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
/// `TopologyEngine::derive_layer_observables`).
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

/// The §G5.2 stateful-update kernel — the ratified closed-form realizing ALL 5
/// observables in their §9.0 roles, gradient-weighted (INV-9), with neuromod-modulated
/// PD restoring force (§G5.2 item 2 + §9.2) and separate full-weight enrichment.
///
/// Caller (daemon) is responsible for:
/// - holding `prev` (= x[t-1]) and `prev2` (= x[t-2]) across ticks;
/// - splitting the producer pipeline into `raw` (the un-enriched producer output)
///   and `enrichment_force` (the held filter_down + ground_up delta, §G5.1/§G10);
/// - computing `obs = observe(prev, prev2)` (5-observable signature of x[t-1]);
/// - reading `cfg.neuromod_gain` from `neuromod_state.bin` per tick (§G5.2 item 2);
/// - sourcing the gain coefficients from `titan_params.toml [trinity_restoring]`
///   (§G5.2 item 5).
///
/// Returns `x[t]`, clamped [0,1].
pub fn stateful_update(
    prev: &[f32],
    prev2: &[f32],
    raw: &[f32],
    enrichment_force: &[f32],
    obs: &LayerObs,
    cfg: &RestoringCfg,
) -> Vec<f32> {
    let n = prev.len();
    debug_assert_eq!(n, prev2.len());
    debug_assert_eq!(n, raw.len());
    debug_assert_eq!(n, enrichment_force.len());

    // ── observable-derived per-tick gains (all 5 observables in their §9.0 roles) ──
    let mag_landing = 1.0 + cfg.w_quant * cfg.a_mag * (2.0 * obs.magnitude - 1.0);
    let drift_signal = (1.0 - obs.coherence) + obs.polarity.abs();
    let spring_gain = 1.0 + cfg.w_qual * cfg.a_drift * drift_signal;
    let damp_gain = 1.0 + cfg.w_quant * cfg.a_dmag * obs.magnitude;
    let k_restore_eff = cfg.k_restore * cfg.neuromod_gain * spring_gain;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let step = prev[i] - prev2[i]; // velocity (per-dim signed)

        let drive = cfg.k_drive * mag_landing * (raw[i] - prev[i]);
        let enrichment = enrichment_force[i]; // full weight, separate
        let spring = -k_restore_eff * (prev[i] - CENTRE);
        let damping = -cfg.k_damp * damp_gain * step;
        let momentum = cfg.w_quant * cfg.k_mom * step;
        let continuity = cfg.w_qual * cfg.k_dir * obs.direction * step.abs();

        let x = prev[i] + drive + enrichment + spring + damping + momentum + continuity;
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

    /// Zero-driver scenario: perturbed centred tensor returns toward 0.5 under spring.
    #[test]
    fn drift_then_return_toward_centre() {
        let cfg = RestoringCfg {
            k_drive: 0.0, // isolate the spring
            k_damp: 0.0,
            k_mom: 0.0,
            k_dir: 0.0,
            ..RestoringCfg::for_layer(Layer::Spirit)
        };
        let mut prev2 = vec![0.9_f32; 4];
        let mut prev = vec![0.9_f32; 4];
        let zeros = vec![0.0_f32; 4];
        for _ in 0..80 {
            let obs = observe(&prev, &prev2);
            let x = stateful_update(&prev, &prev2, &prev, &zeros, &obs, &cfg);
            prev2 = prev;
            prev = x;
        }
        // Should have moved from 0.9 toward 0.5 (not past it, not pinned).
        assert!(prev[0] < 0.9 && prev[0] > 0.5, "got {}", prev[0]);
    }

    /// k_restore=0 + a_mag=0 → pure drive; verifies the arithmetic of the drive term.
    #[test]
    fn zero_restore_pure_drive_regression() {
        let cfg = RestoringCfg {
            k_restore: 0.0,
            k_damp: 0.0,
            k_mom: 0.0,
            k_dir: 0.0,
            a_mag: 0.0, // disable mag-landing for pure-arithmetic check
            ..RestoringCfg::for_layer(Layer::Body)
        };
        let prev = vec![0.5_f32; 5];
        let prev2 = vec![0.5_f32; 5];
        let raw = vec![0.8_f32; 5];
        let enrich = vec![0.0_f32; 5];
        let obs = observe(&prev, &prev2);
        let x = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
        // pure drive: k_drive·(0.8 − 0.5) = 0.30·0.30 = 0.09 → 0.59
        assert!(approx(x[0], 0.59, 1e-5), "got {}", x[0]);
    }

    #[test]
    fn clamped_to_unit_interval() {
        let cfg = RestoringCfg::for_layer(Layer::Body);
        let prev = vec![0.95_f32; 5];
        let prev2 = vec![0.0_f32; 5];
        let raw = vec![1.5_f32; 5]; // out-of-range producer
        let enrich = vec![0.5_f32; 5]; // also large enrichment
        let obs = observe(&prev, &prev2);
        let x = stateful_update(&prev, &prev2, &raw, &enrich, &obs, &cfg);
        for v in x {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
