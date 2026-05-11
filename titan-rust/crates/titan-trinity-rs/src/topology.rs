//! topology — 30D Space Topology computation per Preamble G4 + SPEC §9.A.
//!
//! # Layout (per Preamble G4 + SPEC §7.1 topology_30d.bin)
//!
//! ```text
//! [0:10]  outer_lower  — outer_body[0:5] + outer_mind willing[10:15] → 10D + grounding signal
//! [10:20] inner_lower  — inner_body[0:5] + inner_mind willing[10:15] → 10D + grounding signal
//! [20:30] WHOLE        — synthesis: volume, curvature, density, mean_distance,
//!                        cross_layer_mirror, cluster_count, grounding_tension,
//!                        matter_spirit_ratio, willing_coherence, field_polarity
//! ```
//!
//! # Byte-identical port (per SPEC §11.6)
//!
//! - `LowerTopology` ports `titan_plugin/logic/lower_topology.py:64-204`
//! - `compute_whole_10d` ports `titan_plugin/logic/topology.py:241-338`
//! - Anchor-freshness handling (Option B per PLAN §10.5.1) — substrate uses
//!   `anchor_factor = 1.0` constant when `data/anchor_state.json` absent;
//!   matches Python's `except` swallow path. C-S7 may upgrade to a Python
//!   sidecar (Option C) if `OBS-c-s3-topology-byte-identical` flags drift.
//!
//! # Cluster_count + cross_layer_mirror inputs (C-S3 simplification)
//!
//! C-S3 substrate reads daemon slots that are zero-initialized at boot
//! (daemons ship in C-S5/C-S6). Without per-body-part observable data, the
//! substrate populates `BasicTopology` with zeros for cluster-count, volume,
//! curvature, mean_distance, and cross_layer_mirror. This produces a
//! computable, valid `topology_30d.bin` in C-S3 — same path Python takes
//! when called with empty observables (per `topology.py:57-59`).
//!
//! Phase D enhancement: substrate may add body-part observable derivation
//! from daemon tensors to compute non-trivial basic topology.

use titan_core::constants::{GROUND_UP_DEFAULT_STRENGTH, TOPOLOGY_VOLUME_HISTORY_SIZE};

/// Body dim count per Preamble G1.
pub const BODY_5D: usize = 5;
/// Mind dim count per Preamble G1.
pub const MIND_15D: usize = 15;
/// Mind willing dim range per Preamble G10 (the only mind dims the lower
/// topology incorporates).
pub const MIND_WILLING_RANGE: std::ops::Range<usize> = 10..15;
/// Spirit dim count per Preamble G1 (per inner OR outer; total Trinity = 2×).
pub const SPIRIT_45D: usize = 45;
/// Lower topology dim count.
pub const LOWER_10D: usize = 10;
/// Whole-topology dim count.
pub const WHOLE_10D: usize = 10;
/// Total topology output per Preamble G4 (3 × 10D = 30D).
pub const TOPOLOGY_30D: usize = 30;

/// Per-Titan ground-equilibrium reference state — matches Python
/// `lower_topology.py:29 GROUND_REFERENCE = [0.5] * 10`.
pub const GROUND_REFERENCE: [f32; LOWER_10D] = [0.5; LOWER_10D];

/// Numerical-zero threshold matching Python `1e-10` magnitude check.
pub const MIN_MAGNITUDE: f32 = 1e-10;

/// Lower-topology variant — affects anchor modulation + ground-center bias
/// per `lower_topology.py:35-42`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerVariant {
    /// "Inner" — ethereal matter; ground center stays close to reference.
    Inner,
    /// "Outer" — dense/physical matter; willing dims modulated by anchor
    /// freshness (Option B in C-S3 = constant 1.0; Phase D may switch to
    /// Python sidecar).
    Outer,
}

/// Observable metrics produced by `LowerTopology::compute()`.
///
/// Mirrors Python `lower_topology.py:53-58 _observables` dict. All values
/// rounded to 6 decimals at construction time (matches Python `round(_, 6)`
/// in `LowerTopology._observables` final assignment).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowerObservables {
    /// Cosine similarity with reference balanced state (1.0 = balanced).
    pub coherence: f32,
    /// L2 norm of the 10D state vector.
    pub magnitude: f32,
    /// Rate of magnitude change (delta from previous tick).
    pub velocity: f32,
    /// `+1.0` contracting (velocity < 0), `-1.0` expanding, `0.0` stable.
    pub direction: f32,
    /// Cosine similarity with whole-10D topology (alignment).
    pub polarity: f32,
}

/// Result of one `LowerTopology::compute()` call.
#[derive(Debug, Clone, PartialEq)]
pub struct LowerResult {
    /// 10D topology vector (state). Body[0:5] + Mind willing[10:15] combined.
    pub topology_10d: [f32; LOWER_10D],
    /// Derived observables from the 10D state.
    pub observables: LowerObservables,
    /// Grounding signal (gradient toward ground center). Drives ground_up
    /// nudge in C-S5/C-S6 daemons per Preamble G5 + G10.
    pub grounding_signal_10d: [f32; LOWER_10D],
}

/// Stateful lower-topology engine — tracks rolling history for velocity
/// computation. Mirrors Python `LowerTopology` class structure.
#[derive(Debug, Clone)]
pub struct LowerTopology {
    variant: LowerVariant,
    grounding_strength: f32,
    reference: [f32; LOWER_10D],
    magnitude_history: Vec<f32>,
    last_state: Option<[f32; LOWER_10D]>,
}

impl LowerTopology {
    /// Construct with explicit variant + grounding strength.
    pub fn new(variant: LowerVariant, grounding_strength: f32) -> Self {
        Self {
            variant,
            grounding_strength,
            reference: GROUND_REFERENCE,
            magnitude_history: Vec::with_capacity(TOPOLOGY_VOLUME_HISTORY_SIZE as usize),
            last_state: None,
        }
    }

    /// Inner variant with default grounding strength
    /// (`GROUND_UP_DEFAULT_STRENGTH = 0.1`).
    pub fn inner_default() -> Self {
        Self::new(LowerVariant::Inner, GROUND_UP_DEFAULT_STRENGTH as f32)
    }

    /// Outer variant with default grounding strength.
    pub fn outer_default() -> Self {
        Self::new(LowerVariant::Outer, GROUND_UP_DEFAULT_STRENGTH as f32)
    }

    /// Compute the 10D lower topology + observables + grounding signal.
    ///
    /// Inputs:
    /// - `body_5d` — daemon body tensor (inner or outer per variant)
    /// - `mind_willing_5d` — daemon mind[10..15] willing dims per Preamble G10
    /// - `whole_10d` — last computed whole topology (None on first tick)
    ///
    /// Per `lower_topology.py:64-167`. Outer variant in C-S3 uses
    /// `anchor_factor = 1.0` (Option B per PLAN §10.5.1 — C3 no-file fallback).
    pub fn compute(
        &mut self,
        body_5d: &[f32; BODY_5D],
        mind_willing_5d: &[f32; BODY_5D],
        whole_10d: Option<&[f32; WHOLE_10D]>,
    ) -> LowerResult {
        // Combine into 10D state — body[0..5] || mind_willing[0..5]
        let mut state: [f32; LOWER_10D] = [0.0; LOWER_10D];
        state[..BODY_5D].copy_from_slice(body_5d);
        state[BODY_5D..].copy_from_slice(mind_willing_5d);

        // Outer variant in C-S3 applies anchor-factor = 1.0 default per PLAN §10.5.1.
        // The Python outer-variant modulates state[5,7,9] by anchor data; C3 with
        // anchor_factor=1.0 leaves them unchanged. C-S7 may switch to Option C
        // (Python sidecar) if OBS-c-s3-topology-byte-identical flags drift.
        // (No code change here — anchor_factor=1.0 is identity.)

        // Magnitude (L2 norm).
        let magnitude = l2_norm_5d_pair(&state);

        // Track magnitude history.
        self.magnitude_history.push(magnitude);
        if self.magnitude_history.len() > TOPOLOGY_VOLUME_HISTORY_SIZE as usize {
            self.magnitude_history.remove(0);
        }

        // Velocity (delta of magnitude history).
        let velocity = if self.magnitude_history.len() >= 2 {
            self.magnitude_history[self.magnitude_history.len() - 1]
                - self.magnitude_history[self.magnitude_history.len() - 2]
        } else {
            0.0
        };

        // Direction = +1 contracting (velocity < 0), -1 expanding, 0 stable.
        let direction = if velocity < 0.0 {
            1.0
        } else if velocity > 0.0 {
            -1.0
        } else {
            0.0
        };

        // Coherence = cosine_sim with reference [0.5]*10.
        let coherence = cosine_sim_10d(&state, &self.reference);

        // Polarity = cosine_sim with whole_10d (or 0.0 if not provided).
        let polarity = whole_10d.map(|w| cosine_sim_10d(&state, w)).unwrap_or(0.0);

        // Round to 6 decimals (matches Python observables construction)
        let observables = LowerObservables {
            coherence: round6(coherence),
            magnitude: round6(magnitude),
            velocity: round6(velocity),
            direction: round6(direction),
            polarity: round6(polarity),
        };

        // Grounding signal: gradient (ground_center - state) × strength.
        let ground_center = self.compute_ground_center(whole_10d);
        let grounding_signal_10d: [f32; LOWER_10D] =
            std::array::from_fn(|i| (ground_center[i] - state[i]) * self.grounding_strength);

        self.last_state = Some(state);

        LowerResult {
            topology_10d: state,
            observables,
            grounding_signal_10d,
        }
    }

    /// Compute material equilibrium point. Per `lower_topology.py:169-204`.
    fn compute_ground_center(&self, whole_10d: Option<&[f32; WHOLE_10D]>) -> [f32; LOWER_10D] {
        let mut center = self.reference;
        if let Some(w) = whole_10d {
            // curvature = whole[1] (if available)
            let curvature = w[1];
            let curvature_factor = (curvature.abs() / std::f32::consts::PI).min(1.0) * 0.1;
            match self.variant {
                LowerVariant::Outer => {
                    // Dense: ground center biased toward last_state (body dims only)
                    if let Some(ref last) = self.last_state {
                        for (i, c) in center.iter_mut().enumerate().take(BODY_5D) {
                            *c = self.reference[i] * 0.7 + last[i] * 0.3 + curvature_factor * 0.1;
                        }
                    }
                }
                LowerVariant::Inner => {
                    // Ethereal: stays close to reference, slight curvature offset
                    for c in center.iter_mut() {
                        *c += curvature_factor * 0.05;
                    }
                }
            }
        }
        center
    }
}

/// Basic-topology inputs to [`compute_whole_10d`]. Per `topology.py:241-309`
/// signature. Substrate populates these from body-part observables when
/// available (Phase D); C-S3 supplies zeros (substrate has no body-part
/// observable derivation today).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BasicTopology {
    /// Sum of pairwise distances between body parts. Zero in C-S3.
    pub volume: f32,
    /// Rate of volume change (per `_compute_curvature`). Zero in C-S3.
    pub curvature: f32,
    /// Mean of all pairwise distances. Zero in C-S3.
    pub mean_distance: f32,
    /// Number of body-part clusters at threshold 0.3. Zero in C-S3.
    pub cluster_count: u32,
    /// Last cross-layer mirror from extended topology. Zero in C-S3.
    pub cross_layer_mirror: f32,
}

/// Compute the 10-D WHOLE topology per `topology.py:241-338`.
///
/// Per Preamble G4 layout (the third 10D in the 30D output):
///
/// ```text
/// [0] volume
/// [1] curvature
/// [2] density            = 1 / max(0.01, mean_distance)  if mean_distance > 0 else 0
/// [3] mean_distance
/// [4] cross_layer_mirror
/// [5] cluster_count
/// [6] grounding_tension  = |inner_mag − outer_mag| × anchor_factor (=1.0 in C-S3)
/// [7] matter_spirit_ratio= mean_lower_mag / max(0.01, mean_spirit_mag)
/// [8] willing_coherence  = cosine_sim(inner_mind_willing, outer_mind_willing)
/// [9] field_polarity     = curvature  (Python reuses volume curvature here)
/// ```
///
/// Anchor-factor handling per PLAN §10.5.1 Option B: C-S3 substrate uses
/// `anchor_factor = 1.0` constant. Matches Python's fallback when
/// `data/anchor_state.json` is absent.
pub fn compute_whole_10d(
    basic: &BasicTopology,
    inner_lower: &LowerResult,
    outer_lower: &LowerResult,
    inner_mind_willing: &[f32; BODY_5D],
    outer_mind_willing: &[f32; BODY_5D],
    spirit_magnitudes: &[f32; 2],
) -> [f32; WHOLE_10D] {
    let volume = basic.volume;
    let curvature = basic.curvature;
    let mean_dist = basic.mean_distance;
    let density = if mean_dist > 0.0 {
        1.0 / mean_dist.max(0.01)
    } else {
        0.0
    };
    let cluster_count = basic.cluster_count as f32;
    let cross_mirror = basic.cross_layer_mirror;

    // [6] Grounding tension — anchor_factor = 1.0 in C-S3 (Option B per PLAN §10.5.1)
    let inner_mag = inner_lower.observables.magnitude;
    let outer_mag = outer_lower.observables.magnitude;
    let base_tension = (inner_mag - outer_mag).abs();
    let anchor_factor: f32 = 1.0;
    let grounding_tension = base_tension * anchor_factor;

    // [7] Matter-spirit ratio
    let mean_lower_mag = (inner_mag + outer_mag) / 2.0;
    let mean_spirit_mag = (spirit_magnitudes[0] + spirit_magnitudes[1]) / 2.0;
    let matter_spirit_ratio = mean_lower_mag / mean_spirit_mag.max(0.01);

    // [8] Willing coherence — cosine sim between inner + outer mind willing
    let willing_coherence = cosine_sim_5d(inner_mind_willing, outer_mind_willing);

    // [9] Field polarity = curvature (Python reuses, line 322)
    let field_polarity = curvature;

    [
        round6(volume),
        round6(curvature),
        round6(density),
        round6(mean_dist),
        round6(cross_mirror),
        cluster_count, // Python casts to float; no rounding needed (integral)
        round6(grounding_tension),
        round6(matter_spirit_ratio),
        round6(willing_coherence),
        round6(field_polarity),
    ]
}

/// Assemble 30D topology vector per Preamble G4 + SPEC §9.A.
///
/// `[0:10]` outer_lower + `[10:20]` inner_lower + `[20:30]` whole.
pub fn assemble_topology_30d(
    outer_lower: &LowerResult,
    inner_lower: &LowerResult,
    whole_10d: &[f32; WHOLE_10D],
) -> [f32; TOPOLOGY_30D] {
    let mut out = [0.0f32; TOPOLOGY_30D];
    out[0..10].copy_from_slice(&outer_lower.topology_10d);
    out[10..20].copy_from_slice(&inner_lower.topology_10d);
    out[20..30].copy_from_slice(whole_10d);
    out
}

// ── Math helpers (byte-identical to Python utility functions) ─────────────

/// L2 norm of a 10D vector. Matches Python `_l2_norm` in lower_topology.py:272.
fn l2_norm_5d_pair(v: &[f32; LOWER_10D]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L2 norm of an arbitrary slice — used for spirit-magnitude computation.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity between two 10D vectors. Matches Python `_cosine_sim`
/// in lower_topology.py:262-269.
fn cosine_sim_10d(a: &[f32; LOWER_10D], b: &[f32; LOWER_10D]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let mag_a = l2_norm_5d_pair(a);
    let mag_b = l2_norm_5d_pair(b);
    if mag_a < MIN_MAGNITUDE || mag_b < MIN_MAGNITUDE {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// Cosine similarity between two 5D vectors — used for willing_coherence.
fn cosine_sim_5d(a: &[f32; BODY_5D], b: &[f32; BODY_5D]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let mag_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < MIN_MAGNITUDE || mag_b < MIN_MAGNITUDE {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// Round to 6 decimal places (matches Python `round(_, 6)` semantics on f32
/// after f64 widening — close-enough for parity within 1e-6 tolerance).
fn round6(v: f32) -> f32 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── LowerTopology ────────────────────────────────────────────────────

    #[test]
    fn lower_compute_zero_input_returns_zero_topology() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.0; 5], &[0.0; 5], None);
        assert_eq!(r.topology_10d, [0.0; 10]);
        assert_eq!(r.observables.magnitude, 0.0);
        assert_eq!(r.observables.velocity, 0.0);
        assert_eq!(r.observables.direction, 0.0);
        // coherence with [0; 10] vs [0.5; 10] = 0/0 → 0
        assert_eq!(r.observables.coherence, 0.0);
        // No whole_10d → polarity = 0
        assert_eq!(r.observables.polarity, 0.0);
    }

    #[test]
    fn lower_compute_zero_input_has_grounding_signal_toward_reference() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.0; 5], &[0.0; 5], None);
        // Inner variant w/ no whole → ground_center = reference [0.5]*10
        // signal = (0.5 - 0) × 0.1 = 0.05 each
        for v in r.grounding_signal_10d.iter() {
            assert!(approx(*v, 0.05, 1e-6));
        }
    }

    #[test]
    fn lower_compute_balanced_input_coherence_is_one() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.5; 5], &[0.5; 5], None);
        // state = [0.5; 10] = reference → cosine_sim = 1.0
        assert!(approx(r.observables.coherence, 1.0, 1e-5));
        // magnitude = sqrt(10 * 0.25) = sqrt(2.5) ≈ 1.58114
        assert!(approx(r.observables.magnitude, 1.581_139, 1e-4));
    }

    #[test]
    fn lower_compute_balanced_input_grounding_signal_zero() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.5; 5], &[0.5; 5], None);
        // state == reference → grounding_signal = (0.5 - 0.5) × 0.1 = 0
        for v in r.grounding_signal_10d.iter() {
            assert!(approx(*v, 0.0, 1e-6));
        }
    }

    #[test]
    fn lower_compute_velocity_zero_on_first_tick() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.5; 5], &[0.5; 5], None);
        assert_eq!(r.observables.velocity, 0.0);
    }

    #[test]
    fn lower_compute_velocity_tracks_magnitude_change() {
        let mut lt = LowerTopology::inner_default();
        let _ = lt.compute(&[0.5; 5], &[0.5; 5], None); // mag ≈ 1.58
        let r = lt.compute(&[0.0; 5], &[0.0; 5], None); // mag = 0 → velocity = -1.58
        assert!(r.observables.velocity < 0.0);
        // direction = +1 contracting (velocity < 0)
        assert_eq!(r.observables.direction, 1.0);
    }

    #[test]
    fn lower_outer_default_uses_outer_variant() {
        let lt = LowerTopology::outer_default();
        assert_eq!(lt.variant, LowerVariant::Outer);
    }

    #[test]
    fn lower_inner_default_uses_inner_variant() {
        let lt = LowerTopology::inner_default();
        assert_eq!(lt.variant, LowerVariant::Inner);
    }

    // ── compute_whole_10d ────────────────────────────────────────────────

    #[test]
    fn whole_10d_zero_inputs_returns_all_zeros() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let basic = BasicTopology::default();
        let whole = compute_whole_10d(&basic, &inner, &outer, &[0.0; 5], &[0.0; 5], &[0.0, 0.0]);
        // All zeros: volume=0, curv=0, density=0 (mean_dist=0 path), mean_dist=0,
        // cross=0, cluster=0, ground_tens=0, ratio=0/0.01=0, coh=0 (zero vectors), polarity=0
        for (i, v) in whole.iter().enumerate() {
            assert_eq!(*v, 0.0, "whole[{i}] should be 0.0");
        }
    }

    #[test]
    fn whole_10d_density_is_inverse_of_mean_distance_floor_at_001() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let basic = BasicTopology {
            mean_distance: 0.5,
            ..Default::default()
        };
        let whole = compute_whole_10d(&basic, &inner, &outer, &[0.0; 5], &[0.0; 5], &[0.0, 0.0]);
        // density[2] = 1/0.5 = 2.0
        assert!(approx(whole[2], 2.0, 1e-5));
        // mean_distance[3] = 0.5
        assert!(approx(whole[3], 0.5, 1e-5));
    }

    #[test]
    fn whole_10d_density_floor_at_001_for_tiny_mean_distance() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let basic = BasicTopology {
            mean_distance: 0.001, // < 0.01 floor
            ..Default::default()
        };
        let whole = compute_whole_10d(&basic, &inner, &outer, &[0.0; 5], &[0.0; 5], &[0.0, 0.0]);
        // density = 1.0 / max(0.01, 0.001) = 1.0 / 0.01 = 100.0
        assert!(approx(whole[2], 100.0, 1e-5));
    }

    #[test]
    fn whole_10d_grounding_tension_is_zero_when_inner_outer_mags_equal() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.5; 5], &[0.5; 5], None);
        let outer = outer_lt.compute(&[0.5; 5], &[0.5; 5], None);
        let whole = compute_whole_10d(
            &BasicTopology::default(),
            &inner,
            &outer,
            &[0.5; 5],
            &[0.5; 5],
            &[0.5, 0.5],
        );
        // inner_mag == outer_mag → base_tension = 0 → grounding_tension = 0
        assert!(approx(whole[6], 0.0, 1e-5));
    }

    #[test]
    fn whole_10d_willing_coherence_is_one_for_aligned_willings() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.5; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.5; 5], None);
        let whole = compute_whole_10d(
            &BasicTopology::default(),
            &inner,
            &outer,
            &[0.5; 5],
            &[0.5; 5],
            &[0.0, 0.0],
        );
        // willing_coherence = cosine_sim([0.5;5], [0.5;5]) = 1.0
        assert!(approx(whole[8], 1.0, 1e-5));
    }

    #[test]
    fn whole_10d_field_polarity_equals_curvature() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let basic = BasicTopology {
            curvature: 0.42,
            ..Default::default()
        };
        let whole = compute_whole_10d(&basic, &inner, &outer, &[0.0; 5], &[0.0; 5], &[0.0, 0.0]);
        // field_polarity[9] reuses curvature[1] per Python topology.py:322
        assert!(approx(whole[1], 0.42, 1e-6));
        assert!(approx(whole[9], 0.42, 1e-6));
        assert_eq!(whole[1], whole[9]);
    }

    #[test]
    fn whole_10d_matter_spirit_ratio_zero_when_lower_zero() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let whole = compute_whole_10d(
            &BasicTopology::default(),
            &inner,
            &outer,
            &[0.0; 5],
            &[0.0; 5],
            &[1.0, 1.0],
        );
        // mean_lower_mag = 0 → ratio = 0 / 1.0 = 0
        assert!(approx(whole[7], 0.0, 1e-5));
    }

    // ── assemble_topology_30d ─────────────────────────────────────────────

    #[test]
    fn assemble_layout_outer_inner_whole_per_g4() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        // Distinguishable values: inner_lower all 0.1, outer_lower all 0.2
        // We can't directly inject those — fake by computing then overriding
        let mut inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        inner.topology_10d = [0.1; 10];
        let mut outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        outer.topology_10d = [0.2; 10];
        let whole = [0.3; 10];
        let result = assemble_topology_30d(&outer, &inner, &whole);
        // [0:10] outer
        for v in result.iter().take(10) {
            assert_eq!(*v, 0.2);
        }
        // [10:20] inner
        for v in result.iter().take(20).skip(10) {
            assert_eq!(*v, 0.1);
        }
        // [20:30] whole
        for v in result.iter().skip(20) {
            assert_eq!(*v, 0.3);
        }
    }

    #[test]
    fn assemble_total_30_dims() {
        let mut inner_lt = LowerTopology::inner_default();
        let mut outer_lt = LowerTopology::outer_default();
        let inner = inner_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let outer = outer_lt.compute(&[0.0; 5], &[0.0; 5], None);
        let whole = [0.0; 10];
        let result = assemble_topology_30d(&outer, &inner, &whole);
        assert_eq!(result.len(), 30);
    }

    // ── Math helpers ──────────────────────────────────────────────────────

    #[test]
    fn l2_norm_zero_vector_is_zero() {
        assert_eq!(l2_norm(&[0.0; 5]), 0.0);
    }

    #[test]
    fn l2_norm_unit_vector_is_one() {
        assert!(approx(l2_norm(&[1.0, 0.0, 0.0, 0.0, 0.0]), 1.0, 1e-6));
    }

    #[test]
    fn l2_norm_45d_spirit_balanced() {
        // [0.5; 45] → sqrt(45 * 0.25) ≈ 3.354
        let v = [0.5_f32; 45];
        assert!(approx(l2_norm(&v), 3.354_102, 1e-3));
    }

    #[test]
    fn cosine_sim_5d_aligned_returns_one() {
        assert!(approx(cosine_sim_5d(&[0.5; 5], &[0.5; 5]), 1.0, 1e-5));
    }

    #[test]
    fn cosine_sim_5d_zero_input_returns_zero() {
        assert_eq!(cosine_sim_5d(&[0.0; 5], &[0.5; 5]), 0.0);
        assert_eq!(cosine_sim_5d(&[0.5; 5], &[0.0; 5]), 0.0);
    }

    #[test]
    fn round6_truncates_to_six_decimals() {
        // 0.1234567 → 0.123457 (rounds half-up)
        assert!(approx(round6(0.123_456_7), 0.123_457, 1e-7));
        // 0.1 stays 0.1 (representable)
        assert!(approx(round6(0.1), 0.1, 1e-7));
    }
}
