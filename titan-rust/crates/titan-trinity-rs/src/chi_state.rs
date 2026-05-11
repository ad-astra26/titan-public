//! chi_state — Chi state computation + serialization for `chi_state.bin`.
//!
//! Per SPEC §7.1 chi_state.bin: 6 × float32 LE = 24 bytes payload.
//! Field order: total, spirit, mind, body, coherence, urgency.
//!
//! Per SPEC §7.1 row 13 (ratified by Maker 2026-05-06): owner =
//! `titan-trinity-rs`. The substrate computes this slot per body cycle
//! (~0.87 Hz) from substrate-scoped inputs alone.
//!
//! # Substrate-scoped chi vs Python `LifeForceEngine`
//!
//! Python's `titan_plugin/logic/life_force.py::LifeForceEngine` computes a
//! richer chi from L2-only signals (vocabulary, sovereignty, sol_balance,
//! anchor freshness, expression rate, etc.) — those signals are owned by
//! Python L2 modules and are not visible to the substrate. The substrate's
//! `chi_state.bin` is the substrate-scoped subset using only:
//!   - 6 daemon tensors (inner+outer × body/mind/spirit) read from shm
//!   - sphere_clocks state (coherence-driven contraction velocities)
//!   - neuromod_state.bin (6 modulator levels) read from shm
//!
//! Layer weights are fixed at the Python `LifeForceEngine::compute_weights`
//! "mature" terminus (0.40 / 0.35 / 0.25) because the substrate does not
//! own a developmental-age clock — that's a Python L2 concept (π-clusters,
//! ~5min each). The rFP §4.3 wording "read developmental_age from
//! sphere_clocks Journey 2D state" conflated two concepts (Journey 2D =
//! `[phase, velocity]` per `unified-spirit::self_assembly::extract_journey_2d`,
//! NOT a dev-age signal). Higher-order chi computation that DOES need
//! dev-age continues to live in Python L2 (LifeForceEngine.evaluate); it
//! reads chi_state.bin as a substrate-scoped baseline + augments with its
//! own L2 inputs.

use titan_core::constants::{
    CHI_STATE_FIELD_COUNT, CHI_STATE_PAYLOAD_BYTES, NEUROMOD_FIELD_COUNT,
};

use crate::sphere_clocks::SphereClockSet;
use crate::topology::{l2_norm, BODY_5D, MIND_15D, MIND_WILLING_RANGE, SPIRIT_45D};

const _: () = assert!(CHI_STATE_FIELD_COUNT == 6);
const _: () = assert!(CHI_STATE_PAYLOAD_BYTES == 24);

/// Layer weight constants — fixed at Python `LifeForceEngine::compute_weights`
/// "mature" terminus. Substrate has no developmental-age clock.
const W_SPIRIT: f32 = 0.40;
const W_MIND: f32 = 0.35;
const W_BODY: f32 = 0.25;
// Sanity: weights sum to 1.0 (spirit-heavy mature trinity per Python's terminus).
const _: () = assert!((W_SPIRIT + W_MIND + W_BODY - 1.0).abs() < 1e-6);

/// Trinity-magnitude normalization scale: `BODY_5D=5` so theoretical max
/// L2 norm of [1.0; 5] is √5 ≈ 2.236. Normalize by √5 to get [0,1] range
/// for the body axis. Spirit (45D) max is √45, mind willing (5D) max is √5.
const BODY_NORM: f32 = 2.236_068; // √5
const SPIRIT_NORM: f32 = 6.708_204; // √45
const MIND_NORM: f32 = BODY_NORM; // willing dims (5D subset of 15D)

/// Inputs for [`compute_chi`]. All values come from substrate-readable
/// sources (daemon tensors, sphere clocks state, neuromod_state.bin).
#[derive(Debug, Clone, Copy)]
pub struct ChiInputs<'a> {
    /// Inner-body daemon tensor.
    pub inner_body_5d: &'a [f32; BODY_5D],
    /// Inner-mind daemon tensor.
    pub inner_mind_15d: &'a [f32; MIND_15D],
    /// Inner-spirit daemon tensor.
    pub inner_spirit_45d: &'a [f32; SPIRIT_45D],
    /// Outer-body daemon tensor.
    pub outer_body_5d: &'a [f32; BODY_5D],
    /// Outer-mind daemon tensor.
    pub outer_mind_15d: &'a [f32; MIND_15D],
    /// Outer-spirit daemon tensor.
    pub outer_spirit_45d: &'a [f32; SPIRIT_45D],
    /// Current sphere-clocks state (post-tick) — contraction velocity per
    /// clock encodes coherence pressure.
    pub sphere_clocks: &'a SphereClockSet,
    /// 6 neuromodulator levels read from `neuromod_state.bin` (DA, NE, ACh,
    /// 5HT, GABA, ADO per Python L2 ordering). Each in [0, 1].
    pub neuromod_6: &'a [f32; NEUROMOD_FIELD_COUNT as usize],
}

/// 6-field chi state per SPEC §7.1 chi_state.bin layout.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChiState {
    /// Total chi: weighted sum of layer chi values per fixed mature weights
    /// (W_SPIRIT × spirit + W_MIND × mind + W_BODY × body), clamped [0, 1].
    pub total: f32,
    /// Spirit-component chi: spirit-tensor magnitude × spirit sphere clock
    /// coherence pressure (avg of inner + outer spirit clocks). [0, 1].
    pub spirit: f32,
    /// Mind-component chi: mind willing magnitude × mind sphere clock
    /// coherence pressure. [0, 1].
    pub mind: f32,
    /// Body-component chi: body magnitude × body sphere clock coherence
    /// pressure. [0, 1].
    pub body: f32,
    /// Cross-component coherence: average of body↔mind, mind↔spirit, body↔spirit
    /// cosine similarities computed at the per-trinity (inner/outer) average.
    /// [0, 1] (negative cosines clamped to 0).
    pub coherence: f32,
    /// Urgency: peak neuromodulator level — captures somatic pressure that
    /// the substrate experiences regardless of rich Python interpretation. [0, 1].
    pub urgency: f32,
}

impl ChiState {
    /// All-zero chi state — used as initial state before the first body
    /// cycle has populated real values. Once `body_cycle::run_substrate_body_cycle`
    /// is running, every cycle computes a fresh `ChiState` via [`compute_chi`].
    pub fn zero() -> Self {
        Self::default()
    }

    /// Compute substrate-scoped chi from substrate-readable inputs.
    /// Pure function — no I/O. Caller assembles [`ChiInputs`] and writes
    /// the returned state via [`ChiState::serialize`].
    pub fn compute(inputs: &ChiInputs<'_>) -> Self {
        // ── Layer magnitudes (normalized to [0, 1]) ─────────────────
        let body_inner_mag = l2_norm(inputs.inner_body_5d) / BODY_NORM;
        let body_outer_mag = l2_norm(inputs.outer_body_5d) / BODY_NORM;
        let body_mag = ((body_inner_mag + body_outer_mag) * 0.5).clamp(0.0, 1.0);

        // Mind willing dims [10..15] feed the magnitude — matches lower
        // topology compute (Preamble G10).
        let inner_mind_willing: [f32; BODY_5D] = std::array::from_fn(|i| {
            inputs.inner_mind_15d[MIND_WILLING_RANGE.start + i]
        });
        let outer_mind_willing: [f32; BODY_5D] = std::array::from_fn(|i| {
            inputs.outer_mind_15d[MIND_WILLING_RANGE.start + i]
        });
        let mind_inner_mag = l2_norm(&inner_mind_willing) / MIND_NORM;
        let mind_outer_mag = l2_norm(&outer_mind_willing) / MIND_NORM;
        let mind_mag = ((mind_inner_mag + mind_outer_mag) * 0.5).clamp(0.0, 1.0);

        let spirit_inner_mag = l2_norm(inputs.inner_spirit_45d) / SPIRIT_NORM;
        let spirit_outer_mag = l2_norm(inputs.outer_spirit_45d) / SPIRIT_NORM;
        let spirit_mag = ((spirit_inner_mag + spirit_outer_mag) * 0.5).clamp(0.0, 1.0);

        // ── Sphere-clock coherence pressure per layer ────────────────
        // contraction_velocity ∈ [0, 1] (already clamped in tick).
        // Average of inner + outer clock for that layer.
        let body_coh = (inputs.sphere_clocks.inner_body.contraction_velocity
            + inputs.sphere_clocks.outer_body.contraction_velocity)
            * 0.5;
        let mind_coh = (inputs.sphere_clocks.inner_mind.contraction_velocity
            + inputs.sphere_clocks.outer_mind.contraction_velocity)
            * 0.5;
        let spirit_coh = (inputs.sphere_clocks.inner_spirit.contraction_velocity
            + inputs.sphere_clocks.outer_spirit.contraction_velocity)
            * 0.5;

        // ── Layer chi = magnitude × coherence ─────────────────────────
        let body = (body_mag * body_coh).clamp(0.0, 1.0);
        let mind = (mind_mag * mind_coh).clamp(0.0, 1.0);
        let spirit = (spirit_mag * spirit_coh).clamp(0.0, 1.0);

        // ── Cross-component coherence: 3×3 trinity matrix off-diagonal ─
        // Compute cosine(body, mind), cosine(mind, spirit), cosine(body, spirit)
        // at the per-trinity averaged tensor; average + clamp to [0, 1].
        let body_avg = avg_pair(inputs.inner_body_5d, inputs.outer_body_5d);
        let mind_willing_avg = avg_pair(&inner_mind_willing, &outer_mind_willing);
        // Spirit truncated to first 5 dims for cross-cosine — full 45D is
        // dimensionally incompatible with body/mind. Spirit is its own
        // axis; we sample the first 5 for the matrix off-diagonal.
        let spirit_head_inner: [f32; BODY_5D] =
            std::array::from_fn(|i| inputs.inner_spirit_45d[i]);
        let spirit_head_outer: [f32; BODY_5D] =
            std::array::from_fn(|i| inputs.outer_spirit_45d[i]);
        let spirit_avg = avg_pair(&spirit_head_inner, &spirit_head_outer);

        let cos_bm = cosine_similarity(&body_avg, &mind_willing_avg);
        let cos_ms = cosine_similarity(&mind_willing_avg, &spirit_avg);
        let cos_bs = cosine_similarity(&body_avg, &spirit_avg);
        let coherence = ((cos_bm + cos_ms + cos_bs) / 3.0).clamp(0.0, 1.0);

        // ── Urgency: peak neuromod level ───────────────────────────────
        // Captures somatic pressure at substrate scope. Python L2's richer
        // chi may interpret this with hormonal context.
        let urgency = inputs
            .neuromod_6
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .clamp(0.0, 1.0);

        // ── Total: weighted sum at fixed mature weights ────────────────
        let total = (W_SPIRIT * spirit + W_MIND * mind + W_BODY * body).clamp(0.0, 1.0);

        Self {
            total,
            spirit,
            mind,
            body,
            coherence,
            urgency,
        }
    }

    /// Serialize to 24-byte float32 LE payload per SPEC §7.1.
    pub fn serialize(&self) -> [u8; CHI_STATE_PAYLOAD_BYTES as usize] {
        let mut out = [0u8; CHI_STATE_PAYLOAD_BYTES as usize];
        let fields = [
            self.total,
            self.spirit,
            self.mind,
            self.body,
            self.coherence,
            self.urgency,
        ];
        for (i, v) in fields.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Deserialize from a 24-byte payload per SPEC §7.1. Used by parity tests
    /// + downstream consumers that prefer typed access over byte slicing.
    pub fn from_bytes(bytes: &[u8; CHI_STATE_PAYLOAD_BYTES as usize]) -> Self {
        let read_f32 =
            |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 bytes"));
        Self {
            total: read_f32(0),
            spirit: read_f32(4),
            mind: read_f32(8),
            body: read_f32(12),
            coherence: read_f32(16),
            urgency: read_f32(20),
        }
    }
}

/// Element-wise mean of two equal-size arrays.
fn avg_pair<const N: usize>(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    std::array::from_fn(|i| (a[i] + b[i]) * 0.5)
}

/// Cosine similarity of two equal-size vectors. Returns 0.0 for either
/// zero-magnitude input. Result clamped to [-1.0, 1.0] for numerical stability.
fn cosine_similarity<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a = l2_norm(a);
    let mag_b = l2_norm(b);
    if mag_a < 1e-10 || mag_b < 1e-10 {
        0.0
    } else {
        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn zero_chi_serializes_to_24_zero_bytes() {
        let bytes = ChiState::zero().serialize();
        assert_eq!(bytes.len(), 24);
        for v in bytes.iter() {
            assert_eq!(*v, 0u8);
        }
    }

    #[test]
    fn serialize_layout_matches_spec_71_field_order() {
        let chi = ChiState {
            total: 1.0,
            spirit: 2.0,
            mind: 3.0,
            body: 4.0,
            coherence: 5.0,
            urgency: 6.0,
        };
        let bytes = chi.serialize();
        let read_f32 =
            |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        assert_eq!(read_f32(0), 1.0);
        assert_eq!(read_f32(4), 2.0);
        assert_eq!(read_f32(8), 3.0);
        assert_eq!(read_f32(12), 4.0);
        assert_eq!(read_f32(16), 5.0);
        assert_eq!(read_f32(20), 6.0);
    }

    #[test]
    fn from_bytes_round_trips_serialize() {
        let chi = ChiState {
            total: 0.5,
            spirit: 0.6,
            mind: 0.7,
            body: 0.8,
            coherence: 0.9,
            urgency: 0.4,
        };
        let bytes = chi.serialize();
        let recovered = ChiState::from_bytes(&bytes);
        assert!(approx(recovered.total, chi.total, 1e-6));
        assert!(approx(recovered.spirit, chi.spirit, 1e-6));
        assert!(approx(recovered.mind, chi.mind, 1e-6));
        assert!(approx(recovered.body, chi.body, 1e-6));
        assert!(approx(recovered.coherence, chi.coherence, 1e-6));
        assert!(approx(recovered.urgency, chi.urgency, 1e-6));
    }

    #[test]
    fn compute_zero_inputs_yields_zero_chi() {
        let sc = SphereClockSet::new();
        let neuromod = [0.0_f32; 6];
        let inner_body = [0.0_f32; 5];
        let inner_mind = [0.0_f32; 15];
        let inner_spirit = [0.0_f32; 45];
        let outer_body = [0.0_f32; 5];
        let outer_mind = [0.0_f32; 15];
        let outer_spirit = [0.0_f32; 45];
        let inputs = ChiInputs {
            inner_body_5d: &inner_body,
            inner_mind_15d: &inner_mind,
            inner_spirit_45d: &inner_spirit,
            outer_body_5d: &outer_body,
            outer_mind_15d: &outer_mind,
            outer_spirit_45d: &outer_spirit,
            sphere_clocks: &sc,
            neuromod_6: &neuromod,
        };
        let chi = ChiState::compute(&inputs);
        assert_eq!(chi.total, 0.0);
        assert_eq!(chi.spirit, 0.0);
        assert_eq!(chi.mind, 0.0);
        assert_eq!(chi.body, 0.0);
        assert_eq!(chi.coherence, 0.0);
        assert_eq!(chi.urgency, 0.0);
    }

    #[test]
    fn compute_neuromod_drives_urgency() {
        let sc = SphereClockSet::new();
        let neuromod = [0.1, 0.2, 0.95, 0.3, 0.0, 0.5];
        let inner_body = [0.0_f32; 5];
        let inner_mind = [0.0_f32; 15];
        let inner_spirit = [0.0_f32; 45];
        let outer_body = [0.0_f32; 5];
        let outer_mind = [0.0_f32; 15];
        let outer_spirit = [0.0_f32; 45];
        let inputs = ChiInputs {
            inner_body_5d: &inner_body,
            inner_mind_15d: &inner_mind,
            inner_spirit_45d: &inner_spirit,
            outer_body_5d: &outer_body,
            outer_mind_15d: &outer_mind,
            outer_spirit_45d: &outer_spirit,
            sphere_clocks: &sc,
            neuromod_6: &neuromod,
        };
        let chi = ChiState::compute(&inputs);
        assert!(
            approx(chi.urgency, 0.95, 1e-6),
            "urgency = max(neuromod) = 0.95; got {}",
            chi.urgency
        );
    }

    #[test]
    fn compute_total_is_weighted_sum_of_layers() {
        // Construct inputs that yield specific layer chi values, then
        // verify total = W_SPIRIT × spirit + W_MIND × mind + W_BODY × body.
        let mut sc = SphereClockSet::new();
        // Force contraction velocities to 1.0 so layer chi = magnitude.
        for clk in sc.iter_mut() {
            clk.contraction_velocity = 1.0;
        }
        // Body magnitude: [1/√5; 5] → l2 = 1.0 → body_mag = 1.0/√5 / √5 ...
        // Use direct approach: [0.5; 5] → l2 = √(5×0.25) = √1.25 ≈ 1.118 → /√5 = 0.5
        let inner_body = [0.5_f32; 5];
        let outer_body = [0.5_f32; 5];
        let mut inner_mind = [0.0_f32; 15];
        let mut outer_mind = [0.0_f32; 15];
        for i in 10..15 {
            inner_mind[i] = 0.5;
            outer_mind[i] = 0.5;
        }
        // Spirit magnitude: [0.5; 45] → l2 = √(45×0.25) = √11.25 ≈ 3.354 → /√45 = 0.5
        let inner_spirit = [0.5_f32; 45];
        let outer_spirit = [0.5_f32; 45];
        let neuromod = [0.0_f32; 6];

        let inputs = ChiInputs {
            inner_body_5d: &inner_body,
            inner_mind_15d: &inner_mind,
            inner_spirit_45d: &inner_spirit,
            outer_body_5d: &outer_body,
            outer_mind_15d: &outer_mind,
            outer_spirit_45d: &outer_spirit,
            sphere_clocks: &sc,
            neuromod_6: &neuromod,
        };
        let chi = ChiState::compute(&inputs);
        // Each layer mag ≈ 0.5; coh = 1.0 → layer chi ≈ 0.5.
        assert!(approx(chi.body, 0.5, 1e-2));
        assert!(approx(chi.mind, 0.5, 1e-2));
        assert!(approx(chi.spirit, 0.5, 1e-2));
        // Total = 0.40 × 0.5 + 0.35 × 0.5 + 0.25 × 0.5 = 0.5
        assert!(approx(chi.total, 0.5, 1e-2));
    }

    #[test]
    fn compute_produces_non_zero_when_inputs_are_active() {
        // Realistic-ish state with mixed values; chi should be non-zero
        // and within [0, 1] range.
        let mut sc = SphereClockSet::new();
        for clk in sc.iter_mut() {
            clk.contraction_velocity = 0.5;
        }
        let inner_body = [0.3, 0.4, 0.5, 0.6, 0.7];
        let outer_body = [0.5; 5];
        let mut inner_mind = [0.0_f32; 15];
        for i in 10..15 {
            inner_mind[i] = 0.4;
        }
        let mut outer_mind = [0.0_f32; 15];
        for i in 10..15 {
            outer_mind[i] = 0.6;
        }
        let inner_spirit = [0.5_f32; 45];
        let outer_spirit = [0.4_f32; 45];
        let neuromod = [0.3, 0.5, 0.4, 0.2, 0.1, 0.6];

        let inputs = ChiInputs {
            inner_body_5d: &inner_body,
            inner_mind_15d: &inner_mind,
            inner_spirit_45d: &inner_spirit,
            outer_body_5d: &outer_body,
            outer_mind_15d: &outer_mind,
            outer_spirit_45d: &outer_spirit,
            sphere_clocks: &sc,
            neuromod_6: &neuromod,
        };
        let chi = ChiState::compute(&inputs);
        assert!(chi.total > 0.0 && chi.total <= 1.0);
        assert!(chi.spirit > 0.0 && chi.spirit <= 1.0);
        assert!(chi.mind > 0.0 && chi.mind <= 1.0);
        assert!(chi.body > 0.0 && chi.body <= 1.0);
        assert!(chi.coherence >= 0.0 && chi.coherence <= 1.0);
        assert_eq!(chi.urgency, 0.6); // max of neuromod
    }

    #[test]
    fn compute_round_trips_through_serialize() {
        let mut sc = SphereClockSet::new();
        for clk in sc.iter_mut() {
            clk.contraction_velocity = 0.7;
        }
        let inner_body = [0.4_f32; 5];
        let outer_body = [0.6_f32; 5];
        let inner_mind = [0.5_f32; 15];
        let outer_mind = [0.5_f32; 15];
        let inner_spirit = [0.5_f32; 45];
        let outer_spirit = [0.5_f32; 45];
        let neuromod = [0.2, 0.3, 0.4, 0.1, 0.0, 0.5];

        let inputs = ChiInputs {
            inner_body_5d: &inner_body,
            inner_mind_15d: &inner_mind,
            inner_spirit_45d: &inner_spirit,
            outer_body_5d: &outer_body,
            outer_mind_15d: &outer_mind,
            outer_spirit_45d: &outer_spirit,
            sphere_clocks: &sc,
            neuromod_6: &neuromod,
        };
        let chi = ChiState::compute(&inputs);
        let bytes = chi.serialize();
        let recovered = ChiState::from_bytes(&bytes);
        assert!(approx(recovered.total, chi.total, 1e-6));
        assert!(approx(recovered.spirit, chi.spirit, 1e-6));
        assert!(approx(recovered.mind, chi.mind, 1e-6));
        assert!(approx(recovered.body, chi.body, 1e-6));
        assert!(approx(recovered.coherence, chi.coherence, 1e-6));
        assert!(approx(recovered.urgency, chi.urgency, 1e-6));
    }

    #[test]
    fn cosine_similarity_handles_zero_vectors() {
        let zero = [0.0_f32; 5];
        let nonzero = [0.5_f32; 5];
        assert_eq!(cosine_similarity(&zero, &nonzero), 0.0);
        assert_eq!(cosine_similarity(&nonzero, &zero), 0.0);
        assert_eq!(cosine_similarity(&zero, &zero), 0.0);
    }

    #[test]
    fn cosine_similarity_aligned_vectors_return_one() {
        let a = [0.5_f32; 5];
        let b = [0.7_f32; 5];
        assert!(approx(cosine_similarity(&a, &b), 1.0, 1e-5));
    }

    #[test]
    fn weights_sum_to_unity() {
        assert!((W_SPIRIT + W_MIND + W_BODY - 1.0).abs() < 1e-6);
    }

}
