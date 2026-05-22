//! tick_loop — Substrate per-body-cycle orchestration.
//!
//! Per SPEC §10.G step 5 + PLAN §10.1 substrate body cycle (~1.149 s = 9
//! spirit Schumann ticks). Reads 6 daemon slots → computes lower + whole
//! topology → ticks 6 sphere clocks → updates chi placeholder → returns
//! slot-ready outputs.
//!
//! # Pure compute — NO I/O in this module
//!
//! Slot reads (`titan-state::Slot::read_*`) and writes (`Slot::write_with_seqlock`)
//! happen in C3-6 boot integration; this module is the pure-compute heart of
//! the substrate body tick that the integration layer wraps. Keeping I/O out
//! makes the body-cycle math testable without shm + reusable in C-S4
//! unified-spirit's similar-shaped tick.
//!
//! # Telemetry write-then-publish ordering (SPEC §10.E)
//!
//! C-S3's `BodyTickOutputs` returns the *bytes to write*. The integration
//! layer must:
//!   1. SeqLock-write topology_30d.bin (24-byte header + 120-byte payload)
//!   2. SeqLock-write sphere_clocks.bin (24-byte header + 168-byte payload)
//!   3. SeqLock-write chi_state.bin (24-byte header + 24-byte payload)
//!   4. THEN publish `TRINITY_SUBSTRATE_TOPOLOGY_UPDATED` + per-pulse
//!      `SPHERE_PULSE` events on main bus.
//!
//! Reverse order = race per SPEC §10.E. arch_map static-checks call ordering
//! in the integration layer.

use crate::anchor::AnchorReader;
use crate::chi_state::ChiState;
use crate::sphere_clocks::{PulseEvent, SphereClockSet};
use crate::topology::TopologyEngine;
use crate::topology::{
    assemble_topology_30d, compute_whole_10d, l2_norm, layer_coherence, LowerResult, LowerTopology,
    BODY_5D, MIND_15D, MIND_WILLING_RANGE, SPIRIT_45D, TOPOLOGY_30D,
};

/// Inputs to one substrate body tick — all 6 daemon tensors per SPEC §9.A
/// trinity-rs row "Shm reads: ... all 6 daemon slots".
///
/// In C-S3 these come back zero-initialized (kernel pre-allocates daemon
/// slots in C-S2 but daemons writing them ship in C-S5/C-S6). The substrate
/// is wired correctly; output values become meaningful when daemons populate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTickInputs {
    /// Inner-body daemon output (5D float32).
    pub inner_body_5d: [f32; BODY_5D],
    /// Inner-mind daemon output (15D float32; willing dims [10..15] feed lower topology).
    pub inner_mind_15d: [f32; MIND_15D],
    /// Inner-spirit daemon output (45D float32; magnitude feeds whole-10D matter_spirit_ratio).
    pub inner_spirit_45d: [f32; SPIRIT_45D],
    /// Outer-body daemon output (5D float32).
    pub outer_body_5d: [f32; BODY_5D],
    /// Outer-mind daemon output (15D float32; willing dims feed lower topology).
    pub outer_mind_15d: [f32; MIND_15D],
    /// Outer-spirit daemon output (45D float32).
    pub outer_spirit_45d: [f32; SPIRIT_45D],
    /// Wall-clock seconds since previous body cycle. Substrate measures this
    /// from its `epoch_t0`; default 1.149 s when initialized.
    pub dt_s: f32,
}

impl Default for BodyTickInputs {
    fn default() -> Self {
        Self {
            inner_body_5d: [0.0; BODY_5D],
            inner_mind_15d: [0.0; MIND_15D],
            inner_spirit_45d: [0.0; SPIRIT_45D],
            outer_body_5d: [0.0; BODY_5D],
            outer_mind_15d: [0.0; MIND_15D],
            outer_spirit_45d: [0.0; SPIRIT_45D],
            dt_s: 1.149,
        }
    }
}

/// Outputs of one substrate body tick — slot-ready bytes + bus event list.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyTickOutputs {
    /// 30D topology vector ready for SeqLock-write to `topology_30d.bin`
    /// (per Preamble G4 + SPEC §7.1; payload = 30 × float32 LE = 120 bytes).
    pub topology_30d: [f32; TOPOLOGY_30D],
    /// 168-byte sphere_clocks.bin payload per SPEC §7.1.
    pub sphere_clocks_payload: [u8; 168],
    /// 24-byte chi_state.bin payload per SPEC §7.1 (zero in C-S3).
    pub chi_state_payload: [u8; 24],
    /// Pulses fired this tick — substrate's main bus publisher emits one
    /// `SPHERE_PULSE` per entry (P0 per SPEC §8.6).
    pub pulses: Vec<PulseEvent>,
    /// Inner-lower observable: magnitude — surfaced for telemetry / diagnostics.
    pub inner_magnitude: f32,
    /// Outer-lower observable: magnitude.
    pub outer_magnitude: f32,
}

/// Substrate body-tick state — mutable across ticks. Holds the 2 lower
/// topology engines + 6 sphere clocks + chi placeholder + last-whole cache.
#[derive(Debug, Clone)]
pub struct SubstrateState {
    /// Inner-trinity lower topology engine.
    pub inner_lower: LowerTopology,
    /// Outer-trinity lower topology engine.
    pub outer_lower: LowerTopology,
    /// 6 sphere clocks per Preamble G11.
    pub sphere_clocks: SphereClockSet,
    /// Placeholder chi state — real values come C-S4.
    pub chi_state: ChiState,
    /// Last whole_10d output, used as `whole_10d` input to next tick's
    /// lower-topology compute (drives polarity observable).
    pub last_whole_10d: Option<[f32; 10]>,
    /// Topology engine — produces the 5 BasicTopology fields (volume,
    /// curvature, mean_distance, cluster_count, cross_layer_mirror) per
    /// rFP_phase_c_substrate_observable_closure.md §2.1. Stateful across
    /// ticks via volume_history + prev_magnitudes.
    pub topology_engine: TopologyEngine,
    /// On-chain anchor freshness reader. Re-reads `data/anchor_state.json`
    /// at 60s cadence; produces `anchor_factor` in [0.5, 1.0] that
    /// modulates whole-topology grounding_tension per
    /// rFP_phase_c_substrate_observable_closure.md §2.2.
    pub anchor_reader: AnchorReader,
}

impl SubstrateState {
    /// Construct with default tunings — used at substrate boot.
    /// `data_dir` is the substrate's data directory (parent of
    /// `anchor_state.json`); typically `data/` under the Titan repo root.
    pub fn new(data_dir: std::path::PathBuf) -> Self {
        Self {
            inner_lower: LowerTopology::inner_default(),
            outer_lower: LowerTopology::outer_default(),
            sphere_clocks: SphereClockSet::new(),
            chi_state: ChiState::zero(),
            last_whole_10d: None,
            topology_engine: TopologyEngine::new(),
            anchor_reader: AnchorReader::new(data_dir),
        }
    }

    /// Run one body cycle. Pure compute — caller is responsible for slot I/O
    /// (per module docs §10.E telemetry write-then-publish ordering).
    pub fn body_tick(&mut self, inputs: &BodyTickInputs) -> BodyTickOutputs {
        // Step 1: extract mind willing dims [10..15] per Preamble G10
        let inner_mind_willing: [f32; BODY_5D] =
            std::array::from_fn(|i| inputs.inner_mind_15d[MIND_WILLING_RANGE.start + i]);
        let outer_mind_willing: [f32; BODY_5D] =
            std::array::from_fn(|i| inputs.outer_mind_15d[MIND_WILLING_RANGE.start + i]);

        // Step 2: compute lower topologies
        let inner: LowerResult = self.inner_lower.compute(
            &inputs.inner_body_5d,
            &inner_mind_willing,
            self.last_whole_10d.as_ref(),
        );
        let outer: LowerResult = self.outer_lower.compute(
            &inputs.outer_body_5d,
            &outer_mind_willing,
            self.last_whole_10d.as_ref(),
        );

        // Step 3: spirit magnitudes feed whole-10D matter_spirit_ratio
        let spirit_magnitudes = [
            l2_norm(&inputs.inner_spirit_45d),
            l2_norm(&inputs.outer_spirit_45d),
        ];

        // Step 4: compute basic topology from 6 daemon tensors via
        // TopologyEngine per rFP_phase_c_substrate_observable_closure.md §2.1.
        // Closes the C-S3 -> C-S4 inter-PLAN orphan where BasicTopology was
        // hardcoded to zero; volume / curvature / mean_distance / cluster_count
        // / cross_layer_mirror now flow from real layer-observable derivation.
        let basic = self.topology_engine.compute(
            &inputs.inner_body_5d,
            &inputs.inner_mind_15d,
            &inputs.inner_spirit_45d,
            &inputs.outer_body_5d,
            &inputs.outer_mind_15d,
            &inputs.outer_spirit_45d,
        );
        let anchor_factor = self.anchor_reader.factor();
        let whole_10d = compute_whole_10d(
            &basic,
            &inner,
            &outer,
            &inner_mind_willing,
            &outer_mind_willing,
            &spirit_magnitudes,
            anchor_factor,
        );

        // rFP_phase_c_substrate_observable_closure §2.3 — willing_coherence
        // diagnostic: every WILLING_DIAGNOSTIC_TICK_CADENCE ticks (~19 min
        // body cycles), emit the inner/outer willing dims + cosine_sim
        // numerator + magnitudes so live = 0 episodes can be traced to the
        // upstream producer (outer-mind-rs / inner-mind-rs) rather than the
        // computation itself. INFO level so journalctl captures by default.
        if self.topology_engine.should_log_willing_diagnostic() {
            let dot: f32 = inner_mind_willing
                .iter()
                .zip(outer_mind_willing.iter())
                .map(|(a, b)| a * b)
                .sum();
            let mag_i: f32 = inner_mind_willing.iter().map(|v| v * v).sum::<f32>().sqrt();
            let mag_o: f32 = outer_mind_willing.iter().map(|v| v * v).sum::<f32>().sqrt();
            let willing_coh = whole_10d[8]; // position [8] in WHOLE-10D per Preamble G4
            tracing::info!(
                event = "WILLING_COHERENCE_DIAGNOSTIC",
                tick = self.topology_engine.tick_count(),
                inner_mind_willing = ?inner_mind_willing,
                outer_mind_willing = ?outer_mind_willing,
                dot_product = dot,
                mag_inner = mag_i,
                mag_outer = mag_o,
                willing_coherence = willing_coh,
                "willing_coherence diagnostic (rFP §2.3) — live=0 traces to mag<MIN_MAGNITUDE on either side"
            );
        }

        // Step 5: assemble 30D output per Preamble G4 layout
        let topology_30d = assemble_topology_30d(&outer, &inner, &whole_10d);

        // Cache for next tick
        self.last_whole_10d = Some(whole_10d);

        // Step 6: tick all 6 sphere clocks. Coherence input per SPEC §G4 +
        // §G11 + D-SPEC-84 — canonical `layer_coherence` (1 - variance/0.25)
        // applied to the layer-relevant tensor:
        //   - inner_body / inner_mind clocks use combined inner-lower 10D coherence
        //   - outer_body / outer_mind clocks use combined outer-lower 10D coherence
        //   - inner_spirit / outer_spirit clocks use direct 45D spirit coherence
        // (Spirit has no lower-topology by design — it IS the higher form per
        // §G3. The 45D tensor's variance-coherence directly drives the clock.)
        let inner_coh = inner.observables.coherence;
        let outer_coh = outer.observables.coherence;
        let inner_spirit_coh = layer_coherence(&inputs.inner_spirit_45d);
        let outer_spirit_coh = layer_coherence(&inputs.outer_spirit_45d);

        let mut pulses: Vec<PulseEvent> = Vec::new();
        if let Some(p) = self.sphere_clocks.inner_body.tick(inner_coh, inputs.dt_s) {
            pulses.push(p);
        }
        if let Some(p) = self.sphere_clocks.inner_mind.tick(inner_coh, inputs.dt_s) {
            pulses.push(p);
        }
        if let Some(p) = self
            .sphere_clocks
            .inner_spirit
            .tick(inner_spirit_coh, inputs.dt_s)
        {
            pulses.push(p);
        }
        if let Some(p) = self.sphere_clocks.outer_body.tick(outer_coh, inputs.dt_s) {
            pulses.push(p);
        }
        if let Some(p) = self.sphere_clocks.outer_mind.tick(outer_coh, inputs.dt_s) {
            pulses.push(p);
        }
        if let Some(p) = self
            .sphere_clocks
            .outer_spirit
            .tick(outer_spirit_coh, inputs.dt_s)
        {
            pulses.push(p);
        }

        // Step 7: serialize sphere clocks + chi placeholder
        let sphere_clocks_payload = self.sphere_clocks.serialize();
        let chi_state_payload = self.chi_state.serialize();

        BodyTickOutputs {
            topology_30d,
            sphere_clocks_payload,
            chi_state_payload,
            pulses,
            inner_magnitude: inner.observables.magnitude,
            outer_magnitude: outer.observables.magnitude,
        }
    }
}

/// Default constructs an in-memory-only substrate state (no anchor file).
/// Used in legacy tests; production code calls `SubstrateState::new(data_dir)`.
impl Default for SubstrateState {
    fn default() -> Self {
        Self::new(std::path::PathBuf::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn zero_inputs() -> BodyTickInputs {
        BodyTickInputs {
            inner_body_5d: [0.0; 5],
            inner_mind_15d: [0.0; 15],
            inner_spirit_45d: [0.0; 45],
            outer_body_5d: [0.0; 5],
            outer_mind_15d: [0.0; 15],
            outer_spirit_45d: [0.0; 45],
            dt_s: 1.149,
        }
    }

    #[test]
    fn body_tick_zero_inputs_topology_30d_lower_blocks_zero_whole_minimal() {
        // Under rFP_phase_c_substrate_observable_closure: zero tensors yield
        // zero lower-topology values (parity with Python `compute()` returning
        // zeros when observables are empty/zero) BUT degenerate cluster_count
        // = 1 because all 6 zero-vectors are identical → adjacency complete →
        // one connected component of size 6. This is SPEC-correct behavior:
        // the C-S3 stub's "zero in zero out" expectation was incorrect for
        // a real TopologyEngine.
        let mut s = SubstrateState::default();
        let out = s.body_tick(&zero_inputs());
        // [0:20] outer_lower + inner_lower — pure zero (no observables → no signal)
        for (i, v) in out.topology_30d[..20].iter().enumerate() {
            assert_eq!(*v, 0.0, "lower-block topology_30d[{i}] should be 0.0");
        }
        // [20] volume = 0 (all distances zero)
        assert_eq!(out.topology_30d[20], 0.0);
        // [21] curvature = 0 (first tick, no history)
        assert_eq!(out.topology_30d[21], 0.0);
        // [22] density = 0 (mean_distance = 0 → density floor at 0)
        assert_eq!(out.topology_30d[22], 0.0);
        // [23] mean_distance = 0
        assert_eq!(out.topology_30d[23], 0.0);
        // [24] cross_layer_mirror = 0 (both 65D vectors zero magnitude)
        assert_eq!(out.topology_30d[24], 0.0);
        // [25] cluster_count = 1 (degenerate: all-zero observables are identical → one cluster)
        assert_eq!(out.topology_30d[25], 1.0);
        // [26..30] grounding_tension / matter_spirit_ratio / willing_coherence / field_polarity
        // are all 0 with zero inputs (matter_spirit_ratio = 0/0.5 clamp = 0).
        for (offset, v) in out.topology_30d[26..30].iter().enumerate() {
            assert_eq!(
                *v,
                0.0,
                "whole-block topology_30d[{}] should be 0.0",
                26 + offset
            );
        }
    }

    #[test]
    fn body_tick_topology_30d_layout_outer_inner_whole_per_g4() {
        let mut s = SubstrateState::default();
        let mut inputs = zero_inputs();
        // Distinguishable inputs per trinity
        inputs.inner_body_5d = [0.3; 5];
        inputs.inner_mind_15d[10] = 0.3;
        inputs.inner_mind_15d[11] = 0.3;
        inputs.inner_mind_15d[12] = 0.3;
        inputs.inner_mind_15d[13] = 0.3;
        inputs.inner_mind_15d[14] = 0.3;
        inputs.outer_body_5d = [0.7; 5];
        inputs.outer_mind_15d[10] = 0.7;
        inputs.outer_mind_15d[11] = 0.7;
        inputs.outer_mind_15d[12] = 0.7;
        inputs.outer_mind_15d[13] = 0.7;
        inputs.outer_mind_15d[14] = 0.7;
        let out = s.body_tick(&inputs);
        // [0:10] outer_lower = outer_body || outer_mind willing = [0.7]*10
        for i in 0..10 {
            assert!(
                approx(out.topology_30d[i], 0.7, 1e-5),
                "outer_lower[{i}] = {}",
                out.topology_30d[i]
            );
        }
        // [10:20] inner_lower = inner_body || inner_mind willing = [0.3]*10
        for i in 10..20 {
            assert!(
                approx(out.topology_30d[i], 0.3, 1e-5),
                "inner_lower[{i}] = {}",
                out.topology_30d[i]
            );
        }
        // [20:30] whole — verify dim-9 (field_polarity = curvature = 0 in C-S3)
        assert_eq!(out.topology_30d[29], 0.0);
    }

    #[test]
    fn body_tick_inner_outer_magnitudes_surfaced_for_telemetry() {
        let mut s = SubstrateState::default();
        let mut inputs = zero_inputs();
        inputs.inner_body_5d = [0.5; 5];
        inputs.inner_mind_15d[10..15].copy_from_slice(&[0.5; 5]);
        let out = s.body_tick(&inputs);
        // Inner magnitude = sqrt(10 × 0.25) ≈ 1.581
        assert!(
            approx(out.inner_magnitude, 1.581, 1e-2),
            "got {}",
            out.inner_magnitude
        );
        // Outer is all zero
        assert_eq!(out.outer_magnitude, 0.0);
    }

    #[test]
    fn body_tick_no_pulse_on_first_tick_with_zero_coherence() {
        let mut s = SubstrateState::default();
        let out = s.body_tick(&zero_inputs());
        // Zero coherence → min velocity = 0.0075 / tick → no pulse on first tick
        assert!(out.pulses.is_empty());
    }

    #[test]
    fn body_tick_six_sphere_clocks_advance_each_call() {
        let mut s = SubstrateState::default();
        let inputs = zero_inputs();
        let _ = s.body_tick(&inputs);
        for clk in s.sphere_clocks.iter() {
            assert_eq!(clk.total_ticks, 1);
        }
        let _ = s.body_tick(&inputs);
        for clk in s.sphere_clocks.iter() {
            assert_eq!(clk.total_ticks, 2);
        }
    }

    #[test]
    fn body_tick_balanced_inputs_eventually_produce_pulses() {
        let mut s = SubstrateState::default();
        let mut inputs = zero_inputs();
        // Drive inner-lower coherence high: state == reference [0.5]*10
        inputs.inner_body_5d = [0.5; 5];
        inputs.inner_mind_15d[10..15].copy_from_slice(&[0.5; 5]);
        inputs.outer_body_5d = [0.5; 5];
        inputs.outer_mind_15d[10..15].copy_from_slice(&[0.5; 5]);

        // Run enough ticks (~25 at full speed) for first pulses to fire
        let mut total_pulses = 0;
        for _ in 0..30 {
            let out = s.body_tick(&inputs);
            total_pulses += out.pulses.len();
        }
        assert!(
            total_pulses > 0,
            "expected at least one pulse over 30 balanced ticks"
        );
    }

    #[test]
    fn body_tick_sphere_clocks_payload_is_168_bytes() {
        let mut s = SubstrateState::default();
        let out = s.body_tick(&zero_inputs());
        assert_eq!(out.sphere_clocks_payload.len(), 168);
    }

    #[test]
    fn body_tick_chi_payload_is_zero_24_bytes_in_c_s3() {
        let mut s = SubstrateState::default();
        let out = s.body_tick(&zero_inputs());
        assert_eq!(out.chi_state_payload.len(), 24);
        for v in out.chi_state_payload.iter() {
            assert_eq!(*v, 0u8);
        }
    }

    #[test]
    fn body_tick_caches_last_whole_for_next_polarity_observable() {
        let mut s = SubstrateState::default();
        assert!(s.last_whole_10d.is_none());
        let mut inputs = zero_inputs();
        inputs.inner_body_5d = [0.5; 5];
        let _ = s.body_tick(&inputs);
        assert!(s.last_whole_10d.is_some());
    }

    // ── spirit clock coherence input parity (SPEC §G4 + D-SPEC-84) ─────────

    #[test]
    fn spirit_clock_coherence_uniform_45d_returns_one() {
        // [0.5; 45] → variance=0 → layer_coherence=1.0 (≥ balanced threshold)
        assert!(approx(layer_coherence(&[0.5_f32; 45]), 1.0, 1e-6));
    }

    #[test]
    fn spirit_clock_coherence_zero_45d_returns_one() {
        // [0.0; 45] → variance=0 → layer_coherence=1.0 (uniform, even at zero)
        // Note: this is a SEMANTIC fix vs pre-D-SPEC-84 cosine standin which
        // returned 0.0 for [0.0;45]. Uniform-zero IS coherent per
        // middle_path.py:62-66 position-independence rule.
        assert!(approx(layer_coherence(&[0.0_f32; 45]), 1.0, 1e-6));
    }
}
