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
//! - `LowerTopology` ports `titan_hcl/logic/lower_topology.py:64-204`
//! - `compute_whole_10d` ports `titan_hcl/logic/topology.py:241-338`
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

        // Coherence = `1 - variance/0.25` per Python `middle_path.layer_coherence`.
        // SPEC §G4 + §G11 + D-SPEC-84 — cosine-vs-uniform was port-time drift.
        let coherence = layer_coherence(&state);

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
    anchor_factor: f32,
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

    // [6] Grounding tension — anchor_factor passed in by caller. Read from
    // data/anchor_state.json via AnchorReader per
    // rFP_phase_c_substrate_observable_closure.md §2.2; caller computes
    // per-tick or via 60s cache. Default 1.0 (no anchor activity → full
    // tension) preserves topology.py:288-302 no-file fallback semantics.
    let inner_mag = inner_lower.observables.magnitude;
    let outer_mag = outer_lower.observables.magnitude;
    let base_tension = (inner_mag - outer_mag).abs();
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

// ── TopologyEngine — basic + extended scalar derivation per SPEC §G4 ──────────
//
// rFP_phase_c_substrate_observable_closure.md §2.1: ports Python
// `titan_hcl/logic/topology.py::TopologyEngine.compute()` + the
// `compute_extended` cross_layer_mirror scalar (lines 384-392) into Rust so
// `BasicTopology` is populated with real values instead of zeros at
// `tick_loop.rs:153`.
//
// State carried across ticks: `volume_history` for curvature rate-of-change
// computation. Threshold for cluster_count is SPEC-locked at
// `TOPOLOGY_CLUSTER_THRESHOLD` (`titan-core::constants:38`).
//
// Chunk A (this commit): struct scaffolding + constructor + state holders.
// Chunks B-E will fill in the compute methods. Chunk F wires it into the body
// cycle.

use std::collections::VecDeque;
use titan_core::constants::TOPOLOGY_CLUSTER_THRESHOLD;

/// One layer's 5D observable signature — coherence/magnitude/velocity/direction/polarity.
/// Matches Python `topology.py:23-24 OBSERVABLE_KEYS` field order. The 6 layers
/// (inner_body/mind/spirit + outer_body/mind/spirit) each produce one of these
/// per body cycle. `TopologyEngine::compute` consumes 6 of them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerObservable {
    /// `coherence` ∈ [0, 1].
    pub coherence: f32,
    /// `magnitude` ∈ [0, 1].
    pub magnitude: f32,
    /// `velocity` ∈ [0, 1] (rate of change between cycles).
    pub velocity: f32,
    /// `direction` ∈ [-1, 1] (sign of recent change).
    pub direction: f32,
    /// `polarity` ∈ [-1, 1] (longer-running positivity bias).
    pub polarity: f32,
}

impl LayerObservable {
    /// View as a 5D `[f32; 5]` slice in `OBSERVABLE_KEYS` order — used for
    /// pairwise-distance + cosine_sim computations.
    pub fn as_array(self) -> [f32; 5] {
        [
            self.coherence,
            self.magnitude,
            self.velocity,
            self.direction,
            self.polarity,
        ]
    }
}

/// Cadence for the willing_coherence diagnostic logger (per
/// rFP_phase_c_substrate_observable_closure.md §2.3). At a body cycle of
/// ~1.149s, 1000 ticks ≈ 19 minutes between diagnostic emits — low-noise.
pub const WILLING_DIAGNOSTIC_TICK_CADENCE: u64 = 1000;

/// Engine that computes the 5 `BasicTopology` fields (volume, curvature,
/// mean_distance, cluster_count, cross_layer_mirror) from per-tick substrate
/// inputs. Stateful across ticks via `volume_history` + `prev_magnitudes`.
#[derive(Debug, Clone)]
pub struct TopologyEngine {
    /// Rolling-window of recent volumes for curvature rate-of-change.
    /// Max length = `TOPOLOGY_VOLUME_HISTORY_SIZE` (20 per SPEC constants).
    volume_history: VecDeque<f32>,
    /// Cluster threshold for single-linkage edge-count. Locked at
    /// `TOPOLOGY_CLUSTER_THRESHOLD` = 0.3 by default.
    cluster_threshold: f32,
    /// Previous tick's normalized magnitude per layer (inner_body, inner_mind,
    /// inner_spirit, outer_body, outer_mind, outer_spirit). Used to derive
    /// `velocity` + `direction` observable fields. `None` on first tick.
    prev_magnitudes: Option<[f32; 6]>,
    /// Body-cycle tick counter — monotonically incremented on each `compute()`
    /// call. Drives the willing_coherence diagnostic cadence per
    /// rFP_phase_c_substrate_observable_closure.md §2.3.
    tick_count: u64,
}

impl TopologyEngine {
    /// Construct with SPEC-locked defaults. `volume_history` starts empty;
    /// curvature returns 0.0 until the second `compute()` call.
    pub fn new() -> Self {
        Self {
            volume_history: VecDeque::with_capacity(TOPOLOGY_VOLUME_HISTORY_SIZE as usize),
            cluster_threshold: TOPOLOGY_CLUSTER_THRESHOLD as f32,
            prev_magnitudes: None,
            tick_count: 0,
        }
    }

    /// Returns true when the current tick should emit the willing_coherence
    /// diagnostic log (every `WILLING_DIAGNOSTIC_TICK_CADENCE` ticks, starting
    /// at tick 1). Caller (substrate body_tick) gates the log emit on this.
    pub fn should_log_willing_diagnostic(&self) -> bool {
        self.tick_count > 0
            && self
                .tick_count
                .is_multiple_of(WILLING_DIAGNOSTIC_TICK_CADENCE)
    }

    /// Current body-cycle tick count (incremented each `compute()` call).
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Derive 6 `LayerObservable` (one per trinity layer) from the 6 daemon
    /// tensors. Each tensor produces a 5D observable matching the Python
    /// `OBSERVABLE_KEYS = (coherence, magnitude, velocity, direction, polarity)`
    /// contract (topology.py:23-24):
    ///
    /// - `coherence`  = `1 - variance(tensor) / 0.25` clamped [0, 1]
    ///   (per Python `middle_path.layer_coherence` — SPEC §G4 + D-SPEC-84)
    /// - `magnitude`  = l2_norm(tensor) / sqrt(N) clamped [0, 1] (normalized)
    /// - `velocity`   = |magnitude - prev_magnitude[layer]| clamped [0, 1]
    /// - `direction`  = sign(magnitude - prev_magnitude[layer]) ∈ {-1, 0, +1}
    /// - `polarity`   = clamp((mean(tensor) - 0.5) * 2, -1, 1)
    ///
    /// Updates `prev_magnitudes` for next call. On first call (prev_magnitudes
    /// = None), velocity + direction default to 0.
    pub fn derive_layer_observables(
        &mut self,
        inner_body_5d: &[f32; BODY_5D],
        inner_mind_15d: &[f32; MIND_15D],
        inner_spirit_45d: &[f32; SPIRIT_45D],
        outer_body_5d: &[f32; BODY_5D],
        outer_mind_15d: &[f32; MIND_15D],
        outer_spirit_45d: &[f32; SPIRIT_45D],
    ) -> [LayerObservable; 6] {
        // Compute current normalized magnitudes.
        let m_ib = norm_magnitude(inner_body_5d);
        let m_im = norm_magnitude(inner_mind_15d);
        let m_is = norm_magnitude(inner_spirit_45d);
        let m_ob = norm_magnitude(outer_body_5d);
        let m_om = norm_magnitude(outer_mind_15d);
        let m_os = norm_magnitude(outer_spirit_45d);
        let cur_mags = [m_ib, m_im, m_is, m_ob, m_om, m_os];

        // velocity + direction need prev magnitudes; default to 0 on first tick.
        let prev = self.prev_magnitudes.unwrap_or([0.0; 6]);
        let has_prev = self.prev_magnitudes.is_some();

        let derive = |idx: usize, tensor: &[f32]| -> LayerObservable {
            let cur = cur_mags[idx];
            let prev_m = prev[idx];
            let dm = cur - prev_m;
            let (vel, dir) = if has_prev {
                (dm.abs().clamp(0.0, 1.0), dm.signum_or_zero())
            } else {
                (0.0, 0.0)
            };
            LayerObservable {
                coherence: layer_coherence(tensor),
                magnitude: cur,
                velocity: vel,
                direction: dir,
                polarity: tensor_polarity(tensor),
            }
        };

        let obs = [
            derive(0, inner_body_5d),
            derive(1, inner_mind_15d),
            derive(2, inner_spirit_45d),
            derive(3, outer_body_5d),
            derive(4, outer_mind_15d),
            derive(5, outer_spirit_45d),
        ];

        self.prev_magnitudes = Some(cur_mags);
        obs
    }

    /// Convenience: produce a full `BasicTopology` from the 6 daemon tensors
    /// (canonical entry point used by `SubstrateState::body_tick`). Wires
    /// chunks B+C+D+E together: derives layer observables, computes volume,
    /// curvature, cluster_count, cross_layer_mirror.
    pub fn compute(
        &mut self,
        inner_body_5d: &[f32; BODY_5D],
        inner_mind_15d: &[f32; MIND_15D],
        inner_spirit_45d: &[f32; SPIRIT_45D],
        outer_body_5d: &[f32; BODY_5D],
        outer_mind_15d: &[f32; MIND_15D],
        outer_spirit_45d: &[f32; SPIRIT_45D],
    ) -> BasicTopology {
        self.tick_count = self.tick_count.saturating_add(1);
        let observables = self.derive_layer_observables(
            inner_body_5d,
            inner_mind_15d,
            inner_spirit_45d,
            outer_body_5d,
            outer_mind_15d,
            outer_spirit_45d,
        );
        let (volume, mean_distance, distances) = self.compute_volume_and_mean(&observables);
        let curvature = self.compute_curvature();
        let cluster_count = self.compute_cluster_count(&distances);

        // Build inner_65d / outer_65d for cross_layer_mirror (chunk E).
        let mut inner_65d = [0.0f32; 65];
        let mut outer_65d = [0.0f32; 65];
        inner_65d[..5].copy_from_slice(inner_body_5d);
        inner_65d[5..20].copy_from_slice(inner_mind_15d);
        inner_65d[20..65].copy_from_slice(inner_spirit_45d);
        outer_65d[..5].copy_from_slice(outer_body_5d);
        outer_65d[5..20].copy_from_slice(outer_mind_15d);
        outer_65d[20..65].copy_from_slice(outer_spirit_45d);
        let cross_layer_mirror = self.compute_cross_layer_mirror(&inner_65d, &outer_65d);

        BasicTopology {
            volume,
            curvature,
            mean_distance,
            cluster_count,
            cross_layer_mirror,
        }
    }

    /// Window size for volume_history (test/inspection accessor).
    pub fn volume_history_len(&self) -> usize {
        self.volume_history.len()
    }

    /// Most recent computed volume, if any (test/inspection accessor).
    pub fn last_volume(&self) -> Option<f32> {
        self.volume_history.back().copied()
    }
}

impl Default for TopologyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyEngine {
    // ── Chunk B: volume + mean_distance (ports topology.py:67-87) ─────────

    /// Compute (volume, mean_distance, all_15_distances) from 6 layer-observable
    /// 5D vectors. Volume = sum of all 15 pairwise L2 distances. mean_distance =
    /// volume / 15. Mirrors `topology.py:67-87`.
    ///
    /// The layer order is fixed (matches `topology.py:67 sorted(vectors.keys())`):
    /// inner_body, inner_mind, inner_spirit, outer_body, outer_mind, outer_spirit.
    /// 6 layers → C(6,2) = 15 unordered pairs.
    pub fn compute_volume_and_mean(
        &mut self,
        observables: &[LayerObservable; 6],
    ) -> (f32, f32, [f32; 15]) {
        let arrs: [[f32; 5]; 6] = std::array::from_fn(|i| observables[i].as_array());
        let mut distances = [0.0f32; 15];
        let mut idx = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                distances[idx] = round6(l2_distance_5d(&arrs[i], &arrs[j]));
                idx += 1;
            }
        }
        let volume: f32 = distances.iter().sum();
        let mean_distance = if !distances.is_empty() {
            volume / distances.len() as f32
        } else {
            0.0
        };

        // Append to rolling window for curvature (chunk C).
        self.volume_history.push_back(volume);
        if self.volume_history.len() > TOPOLOGY_VOLUME_HISTORY_SIZE as usize {
            self.volume_history.pop_front();
        }

        (round6(volume), round6(mean_distance), distances)
    }

    // ── Chunk C: curvature (ports topology.py:98-115 _compute_curvature) ──

    /// Rate of volume change. Positive = contracting (prev > curr), negative =
    /// expanding. Returns 0.0 when fewer than 2 volumes recorded OR when prev
    /// volume is below `MIN_MAGNITUDE` (1e-10). Mirrors `topology.py:98-115`.
    pub fn compute_curvature(&self) -> f32 {
        if self.volume_history.len() < 2 {
            return 0.0;
        }
        let n = self.volume_history.len();
        let prev = self.volume_history[n - 2];
        let curr = self.volume_history[n - 1];
        if prev < MIN_MAGNITUDE {
            return 0.0;
        }
        round6((prev - curr) / prev)
    }

    // ── Chunk D: cluster_count (ports topology.py:117-162 _find_clusters) ─

    /// Count of connected components of size ≥ 2 in the single-linkage
    /// adjacency graph at `cluster_threshold`. Each node = one layer; edge
    /// exists when pairwise distance ≤ threshold. Mirrors `topology.py:117-162`
    /// `_find_clusters` (returns cluster count only — full cluster lists not
    /// part of the SPEC §G4 WHOLE-10D output).
    ///
    /// `distances` MUST be the 15-element pairwise array returned from
    /// `compute_volume_and_mean` (canonical layer-order: ib-im-is-ob-om-os).
    pub fn compute_cluster_count(&self, distances: &[f32; 15]) -> u32 {
        // Adjacency by union-find over the 6 layer-nodes.
        let mut parent: [usize; 6] = [0, 1, 2, 3, 4, 5];

        fn find(parent: &mut [usize; 6], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize; 6], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // Walk 15 pairs in the same iteration order as compute_volume_and_mean.
        let mut idx = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                if distances[idx] <= self.cluster_threshold {
                    union(&mut parent, i, j);
                }
                idx += 1;
            }
        }

        // Count component sizes; cluster_count = number of components with ≥2 nodes.
        let mut sizes = [0u32; 6];
        for node in 0..6 {
            let root = find(&mut parent, node);
            sizes[root] += 1;
        }
        sizes.iter().filter(|&&s| s >= 2).count() as u32
    }

    // ── Chunk E: cross_layer_mirror (ports topology.py:384-392) ───────────

    /// `cosine_sim(inner_65d, outer_65d)` per `topology.py:384-392`. Returns
    /// 0.0 if either magnitude falls below `MIN_MAGNITUDE` (1e-10). The 65D
    /// vectors are `body_5d ++ mind_15d ++ spirit_45d` per Preamble G1.
    pub fn compute_cross_layer_mirror(&self, inner_65d: &[f32; 65], outer_65d: &[f32; 65]) -> f32 {
        let mut dot = 0.0f32;
        let mut mag_i_sq = 0.0f32;
        let mut mag_o_sq = 0.0f32;
        for k in 0..65 {
            let a = inner_65d[k];
            let b = outer_65d[k];
            dot += a * b;
            mag_i_sq += a * a;
            mag_o_sq += b * b;
        }
        let mag_i = mag_i_sq.sqrt();
        let mag_o = mag_o_sq.sqrt();
        if mag_i < MIN_MAGNITUDE || mag_o < MIN_MAGNITUDE {
            return 0.0;
        }
        round6(dot / (mag_i * mag_o))
    }
}

/// L2 distance between two 5D vectors (used for pairwise distance matrix).
fn l2_distance_5d(a: &[f32; 5], b: &[f32; 5]) -> f32 {
    let mut sum = 0.0f32;
    for k in 0..5 {
        let d = a[k] - b[k];
        sum += d * d;
    }
    sum.sqrt()
}

/// `l2_norm(tensor) / sqrt(N)` — normalized magnitude in roughly [0, 1] when
/// tensor values are in [0, 1]. Used by `derive_layer_observables` for the
/// magnitude observable field.
fn norm_magnitude(tensor: &[f32]) -> f32 {
    let n = tensor.len() as f32;
    if n < 1.0 {
        return 0.0;
    }
    let sum_sq: f32 = tensor.iter().map(|v| v * v).sum();
    (sum_sq.sqrt() / n.sqrt()).clamp(0.0, 1.0)
}

/// Per-layer coherence — `1 - variance / 0.25` clamped [0, 1].
///
/// SPEC §G4 + §G11 + D-SPEC-84: canonical formula mirrors Python
/// `titan_hcl/logic/middle_path.py:51 layer_coherence`. Measures how
/// aligned the dimensions are with each other (low variance → high
/// coherence). Position-independent: `[0.1, 0.1, 0.1]` and `[0.9, 0.9, 0.9]`
/// are both perfectly coherent (1.0). The previous cosine-vs-uniform-0.5
/// formula was a port-time drift that stalled the fleet 2026-05-13 → 2026-05-18
/// (4 of 6 sphere clocks stuck at radius=1.0 because real-world tensors
/// rarely cross the 0.80 cosine threshold while keeping reasonable variance).
///
/// Single in-process source of truth — used by `derive_layer_observables`
/// (per-layer 5D/15D/45D), `LowerTopology::compute` (combined 10D), and
/// `tick_loop::body_tick` (inner_spirit_45d + outer_spirit_45d sphere clock
/// coherence input). Single name keeps SPEC §G4 lock-in trivially auditable.
pub fn layer_coherence(tensor: &[f32]) -> f32 {
    let n = tensor.len();
    if n < 2 {
        return 1.0; // matches middle_path.py:62-63
    }
    let n_f = n as f32;
    let mean = tensor.iter().sum::<f32>() / n_f;
    let variance = tensor.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_f;
    (1.0 - variance / 0.25).clamp(0.0, 1.0)
}

/// Polarity = (mean(tensor) - 0.5) * 2 clamped to [-1, +1]. Captures
/// whether the layer's average value sits above (+) or below (-) the
/// neutral 0.5 center. Returns 0.0 when the tensor has zero magnitude
/// (matches Python "zero observables → zero topology" parity — empty
/// observables produce a zero-vector after `obs.get(k, 0.0)` defaulting,
/// so the basic compute output is all zeros, not a -1 polarity bias).
fn tensor_polarity(tensor: &[f32]) -> f32 {
    if tensor.is_empty() {
        return 0.0;
    }
    let mag_sq: f32 = tensor.iter().map(|v| v * v).sum();
    if mag_sq < MIN_MAGNITUDE {
        return 0.0;
    }
    let mean: f32 = tensor.iter().sum::<f32>() / tensor.len() as f32;
    ((mean - 0.5) * 2.0).clamp(-1.0, 1.0)
}

/// Trait shim for f32 — strict sign (returns 0.0 at 0.0 instead of f32's
/// default ±1.0 for ±0.0). Matches Python `1 if x > 0 else -1 if x < 0 else 0`.
trait SignumOrZero {
    fn signum_or_zero(self) -> Self;
}
impl SignumOrZero for f32 {
    fn signum_or_zero(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── LowerTopology ────────────────────────────────────────────────────

    #[test]
    fn lower_compute_zero_input_returns_uniform_coherent_topology() {
        let mut lt = LowerTopology::inner_default();
        let r = lt.compute(&[0.0; 5], &[0.0; 5], None);
        assert_eq!(r.topology_10d, [0.0; 10]);
        assert_eq!(r.observables.magnitude, 0.0);
        assert_eq!(r.observables.velocity, 0.0);
        assert_eq!(r.observables.direction, 0.0);
        // Post-D-SPEC-84: coherence = `1 - variance/0.25` per middle_path.py:51.
        // [0;10] is position-independent uniform (variance=0) → coherence=1.0.
        // (Pre-D-SPEC-84 cosine-vs-uniform returned 0 for zero-magnitude — that
        // was the port-time drift that stalled the fleet 2026-05-13→2026-05-18.)
        assert_eq!(r.observables.coherence, 1.0);
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
        let whole = compute_whole_10d(
            &basic,
            &inner,
            &outer,
            &[0.0; 5],
            &[0.0; 5],
            &[0.0, 0.0],
            1.0,
        );
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
        let whole = compute_whole_10d(
            &basic,
            &inner,
            &outer,
            &[0.0; 5],
            &[0.0; 5],
            &[0.0, 0.0],
            1.0,
        );
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
        let whole = compute_whole_10d(
            &basic,
            &inner,
            &outer,
            &[0.0; 5],
            &[0.0; 5],
            &[0.0, 0.0],
            1.0,
        );
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
            1.0,
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
            1.0,
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
        let whole = compute_whole_10d(
            &basic,
            &inner,
            &outer,
            &[0.0; 5],
            &[0.0; 5],
            &[0.0, 0.0],
            1.0,
        );
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
            1.0,
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

    // ── layer_coherence parity with middle_path.py:51 (SPEC §G4 + D-SPEC-84) ──

    #[test]
    fn layer_coherence_uniform_is_one() {
        // Position-independent: all-equal → variance=0 → coherence=1.0
        assert!(approx(layer_coherence(&[0.5; 5]), 1.0, 1e-6));
        assert!(approx(layer_coherence(&[0.1; 5]), 1.0, 1e-6));
        assert!(approx(layer_coherence(&[0.9; 5]), 1.0, 1e-6));
        assert!(approx(layer_coherence(&[0.7; 45]), 1.0, 1e-6));
    }

    #[test]
    fn layer_coherence_max_variance_is_zero() {
        // [0,0,1,1] — mean=0.5, variance = ((0.5)²·2 + (0.5)²·2)/4 = 0.25
        // (the theoretical max for [0,1]-bounded values) → coherence =
        // 1 - 0.25/0.25 = 0.0. Matches middle_path.py:51 doctrine "0.0
        // when variance is maximal (half at 0, half at 1)".
        assert!(approx(layer_coherence(&[0.0, 0.0, 1.0, 1.0]), 0.0, 1e-6));
    }

    #[test]
    fn layer_coherence_short_tensor_returns_one() {
        // Matches middle_path.py:62-63 short-circuit for len<2.
        assert!(approx(layer_coherence(&[]), 1.0, 1e-6));
        assert!(approx(layer_coherence(&[0.5]), 1.0, 1e-6));
    }

    #[test]
    fn layer_coherence_python_parity_case() {
        // Specific case verified against middle_path.py:51 in Python:
        // tensor = [0.5, 0.6, 0.4, 0.55, 0.45]
        // mean = 0.5
        // variance = ((0)² + (0.1)² + (-0.1)² + (0.05)² + (-0.05)²) / 5
        //          = (0 + 0.01 + 0.01 + 0.0025 + 0.0025) / 5 = 0.005
        // coherence = 1 - 0.005/0.25 = 0.98
        let tensor = [0.5_f32, 0.6, 0.4, 0.55, 0.45];
        assert!(approx(layer_coherence(&tensor), 0.98, 1e-5));
    }

    #[test]
    fn layer_coherence_15d_mind_realistic() {
        // Realistic inner_mind 15D with willing dims populated (post-D-SPEC-81):
        // SENS [0..5] ≈ 0.5, ASSESS [5..10] ≈ 0.5, WILLING [10..15] varied.
        let mut tensor = [0.5_f32; 15];
        tensor[10] = 0.503; // IMPULSE
        tensor[11] = 0.549; // EMPATHY
        tensor[12] = 0.575; // CREATIVITY
        tensor[13] = 0.633; // VIGILANCE
        tensor[14] = 0.448; // CURIOSITY
        let coh = layer_coherence(&tensor);
        // Real production data on T1 — coherence should comfortably exceed 0.80
        // (cosine-vs-uniform formula gave 0.806; variance-based gives ~0.99+).
        assert!(
            coh > 0.95,
            "expected >0.95 for realistic mind 15D, got {}",
            coh
        );
    }

    #[test]
    fn layer_coherence_unaffected_by_position() {
        // Same dims, shuffled — coherence is position-independent
        let a = [0.1_f32, 0.5, 0.9, 0.3, 0.7];
        let b = [0.7_f32, 0.3, 0.5, 0.9, 0.1];
        assert!(approx(layer_coherence(&a), layer_coherence(&b), 1e-6));
    }

    #[test]
    fn round6_truncates_to_six_decimals() {
        // 0.1234567 → 0.123457 (rounds half-up)
        assert!(approx(round6(0.123_456_7), 0.123_457, 1e-7));
        // 0.1 stays 0.1 (representable)
        assert!(approx(round6(0.1), 0.1, 1e-7));
    }

    // ── TopologyEngine chunks B-E ────────────────────────────────────────

    fn layer_obs(c: f32, m: f32, v: f32, d: f32, p: f32) -> LayerObservable {
        LayerObservable {
            coherence: c,
            magnitude: m,
            velocity: v,
            direction: d,
            polarity: p,
        }
    }

    /// 6 identical layer-observables → all pairwise distances = 0 → volume = 0.
    #[test]
    fn topology_engine_volume_identical_layers_is_zero() {
        let mut eng = TopologyEngine::new();
        let obs = [layer_obs(0.5, 0.5, 0.5, 0.0, 0.0); 6];
        let (vol, mean, dists) = eng.compute_volume_and_mean(&obs);
        assert_eq!(vol, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(dists, [0.0; 15]);
    }

    /// 6 distinct unit-separated layers → volume = sum of 15 nonzero distances.
    #[test]
    fn topology_engine_volume_distinct_layers_is_positive() {
        let mut eng = TopologyEngine::new();
        // Construct 6 maximally-separated observables along different axes.
        let obs = [
            layer_obs(1.0, 0.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 1.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 1.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 1.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 1.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let (vol, mean, dists) = eng.compute_volume_and_mean(&obs);
        // 10 of 15 distances = sqrt(2) (axis-vs-axis pairs)
        // 5 of 15 distances = 1.0 (origin vs each axis)
        // volume = 10*sqrt(2) + 5*1 ≈ 19.1421
        assert!(approx(vol, 10.0 * 2.0f32.sqrt() + 5.0, 1e-3));
        assert!(approx(mean, vol / 15.0, 1e-6));
        assert_eq!(dists.len(), 15);
        assert!(dists.iter().all(|&d| d > 0.0));
    }

    /// Volume history accumulates and capped at TOPOLOGY_VOLUME_HISTORY_SIZE.
    #[test]
    fn topology_engine_volume_history_caps_at_window_size() {
        let mut eng = TopologyEngine::new();
        let obs = [layer_obs(1.0, 0.0, 0.0, 0.0, 0.0); 6];
        let cap = TOPOLOGY_VOLUME_HISTORY_SIZE as usize;
        for _ in 0..(cap + 5) {
            eng.compute_volume_and_mean(&obs);
        }
        assert_eq!(eng.volume_history_len(), cap);
    }

    /// volume_history records the actual computed volume.
    #[test]
    fn topology_engine_volume_history_records_last_volume() {
        let mut eng = TopologyEngine::new();
        let obs = [
            layer_obs(1.0, 0.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 1.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 1.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 1.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 1.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let (vol, _, _) = eng.compute_volume_and_mean(&obs);
        assert_eq!(eng.last_volume(), Some(vol));
    }

    // ── Chunk C: curvature ────────────────────────────────────────────────

    #[test]
    fn topology_engine_curvature_empty_history_is_zero() {
        let eng = TopologyEngine::new();
        assert_eq!(eng.compute_curvature(), 0.0);
    }

    #[test]
    fn topology_engine_curvature_single_volume_is_zero() {
        let mut eng = TopologyEngine::new();
        eng.compute_volume_and_mean(&[layer_obs(1.0, 0.0, 0.0, 0.0, 0.0); 6]);
        assert_eq!(eng.compute_curvature(), 0.0);
    }

    /// Two volumes: contracting (smaller curr) → positive curvature.
    /// Expanding (larger curr) → negative curvature. Matches `topology.py:113-115`.
    #[test]
    fn topology_engine_curvature_contracting_is_positive() {
        let mut eng = TopologyEngine::new();
        // Tick 1: 6 maximally-separated layers → volume ≈ 19.14
        let big = [
            layer_obs(1.0, 0.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 1.0, 0.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 1.0, 0.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 1.0, 0.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 1.0),
            layer_obs(0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        eng.compute_volume_and_mean(&big);
        // Tick 2: identical → volume = 0 → contracting
        let small = [layer_obs(0.5, 0.5, 0.5, 0.0, 0.0); 6];
        eng.compute_volume_and_mean(&small);
        let curv = eng.compute_curvature();
        // (prev - curr)/prev with prev ≈ 19.14, curr ≈ 0 → ~1.0
        assert!(curv > 0.99);
    }

    // ── Chunk D: cluster_count ────────────────────────────────────────────

    /// All-identical layers (distances all 0) → 1 component of size 6.
    #[test]
    fn topology_engine_cluster_count_all_close_is_one() {
        let mut eng = TopologyEngine::new();
        let (_, _, dists) = eng.compute_volume_and_mean(&[layer_obs(0.5, 0.5, 0.0, 0.0, 0.0); 6]);
        let cc = eng.compute_cluster_count(&dists);
        assert_eq!(cc, 1);
    }

    /// All-distant layers (distances > threshold) → 0 clusters of size ≥ 2.
    #[test]
    fn topology_engine_cluster_count_all_far_is_zero() {
        let eng = TopologyEngine::new();
        let dists = [1.0; 15]; // all 1.0 > threshold 0.3
        let cc = eng.compute_cluster_count(&dists);
        assert_eq!(cc, 0);
    }

    /// Two pairs close + 2 isolated: cluster_count = 2 (two pairs).
    #[test]
    fn topology_engine_cluster_count_two_pairs() {
        let eng = TopologyEngine::new();
        // Pair 0-1 close, pair 2-3 close, 4 and 5 isolated.
        // Pair indices in 15-distance array (i,j) iteration order:
        //   (0,1)=0  (0,2)=1  (0,3)=2  (0,4)=3  (0,5)=4
        //   (1,2)=5  (1,3)=6  (1,4)=7  (1,5)=8
        //   (2,3)=9  (2,4)=10 (2,5)=11
        //   (3,4)=12 (3,5)=13
        //   (4,5)=14
        let mut dists = [1.0; 15]; // start everything far
        dists[0] = 0.1; // (0,1) close
        dists[9] = 0.1; // (2,3) close
        let cc = eng.compute_cluster_count(&dists);
        assert_eq!(cc, 2);
    }

    /// Threshold value (0.3) verified — matches `TOPOLOGY_CLUSTER_THRESHOLD`.
    #[test]
    fn topology_engine_cluster_count_threshold_boundary() {
        let eng = TopologyEngine::new();
        let mut dists = [1.0; 15];
        dists[0] = 0.3; // exactly at threshold (≤) → should connect
        let cc_at = eng.compute_cluster_count(&dists);
        assert_eq!(cc_at, 1);

        dists[0] = 0.31; // just above → should NOT connect
        let cc_above = eng.compute_cluster_count(&dists);
        assert_eq!(cc_above, 0);
    }

    // ── Chunk E: cross_layer_mirror ───────────────────────────────────────

    #[test]
    fn topology_engine_cross_mirror_identical_is_one() {
        let eng = TopologyEngine::new();
        let inner: [f32; 65] = std::array::from_fn(|i| (i as f32) * 0.01);
        let outer = inner;
        let m = eng.compute_cross_layer_mirror(&inner, &outer);
        assert!(approx(m, 1.0, 1e-5));
    }

    #[test]
    fn topology_engine_cross_mirror_zero_input_is_zero() {
        let eng = TopologyEngine::new();
        let zero = [0.0f32; 65];
        let nonzero: [f32; 65] = std::array::from_fn(|i| (i as f32) * 0.01);
        assert_eq!(eng.compute_cross_layer_mirror(&zero, &nonzero), 0.0);
        assert_eq!(eng.compute_cross_layer_mirror(&nonzero, &zero), 0.0);
    }

    #[test]
    fn topology_engine_cross_mirror_orthogonal_is_zero() {
        let eng = TopologyEngine::new();
        // First half nonzero in inner, second half nonzero in outer → dot = 0.
        let mut inner = [0.0f32; 65];
        let mut outer = [0.0f32; 65];
        for i in 0..32 {
            inner[i] = 1.0;
        }
        for i in 32..65 {
            outer[i] = 1.0;
        }
        let m = eng.compute_cross_layer_mirror(&inner, &outer);
        assert!(approx(m, 0.0, 1e-5));
    }

    // ── Chunk H: willing_coherence diagnostic cadence ────────────────────

    #[test]
    fn topology_engine_willing_diagnostic_does_not_fire_on_tick_zero() {
        let eng = TopologyEngine::new();
        assert!(!eng.should_log_willing_diagnostic());
        assert_eq!(eng.tick_count(), 0);
    }

    #[test]
    fn topology_engine_willing_diagnostic_fires_at_cadence() {
        let mut eng = TopologyEngine::new();
        let zero = [0.0f32; 5];
        let zero15 = [0.0f32; 15];
        let zero45 = [0.0f32; 45];
        for i in 1..=WILLING_DIAGNOSTIC_TICK_CADENCE {
            let _ = eng.compute(&zero, &zero15, &zero45, &zero, &zero15, &zero45);
            if i == WILLING_DIAGNOSTIC_TICK_CADENCE {
                assert!(
                    eng.should_log_willing_diagnostic(),
                    "diagnostic should fire at tick {WILLING_DIAGNOSTIC_TICK_CADENCE}"
                );
            } else if !i.is_multiple_of(WILLING_DIAGNOSTIC_TICK_CADENCE) {
                assert!(
                    !eng.should_log_willing_diagnostic(),
                    "diagnostic should NOT fire at tick {i}"
                );
            }
        }
        assert_eq!(eng.tick_count(), WILLING_DIAGNOSTIC_TICK_CADENCE);
    }

    // ── l2_distance_5d helper ────────────────────────────────────────────

    #[test]
    fn l2_distance_5d_basic() {
        assert!(approx(l2_distance_5d(&[0.0; 5], &[0.0; 5]), 0.0, 1e-7));
        assert!(approx(
            l2_distance_5d(&[1.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0, 0.0]),
            2.0f32.sqrt(),
            1e-5
        ));
    }
}
