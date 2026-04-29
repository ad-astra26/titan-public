"""Tests for EMOT-CGN v3 — emergent emotion substrate (rFP §19–§21).

Covers:
- emot_bundle_protocol: native-first schema, torn-read protection,
  per-Titan paths, Plug A/Plug C helpers.
- emot_thin_encoder: pure assembly, valence/arousal/novelty derivation,
  per-Titan determinism.
- emot_region_clusterer: HDBSCAN region discovery, NOISE acceptance,
  signature-stable IDs across reloads.
- emot_kin_protocol: payload roundtrip, validation, MSL binding rules.
- CGNConsumerClient Plug B: note_incoming_cross_insight EMA + filtering.

Run with: python -m pytest tests/test_emot_cgn_v3.py -v -p no:anchorpy
"""
import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from titan_plugin.logic.emot_bundle_protocol import (
    BUNDLE_SIZE, CORE_SIZE, RESERVED_TAIL, BUNDLE_SCHEMA_VERSION,
    FELT_TENSOR_DIM, TRAJECTORY_DIM, SPACE_TOPOLOGY_DIM, NEUROMOD_DIM,
    HORMONE_DIM, NS_URGENCY_DIM, CGN_BETA_DIM, MSL_ACT_DIM, PI_PHASE_DIM,
    TRUNK_DIM, INNER_DIM, OUTER_DIM,
    ENCODER_THIN_ASSEMBLY, REGION_UNCLUSTERED, REGION_NOISE, GRAD_SHADOW,
    LEGACY_PRIMITIVES, NS_PROGRAMS, CGN_CONSUMERS, MSL_CONCEPTS,
    BundleWriter, BundleReader,
    read_emotion_valence_normalized, read_full_emotion_context,
    _fnv1a_32,
)
from titan_plugin.logic.emot_thin_encoder import ThinEmotEncoder
from titan_plugin.logic.emot_region_clusterer import (
    RegionClusterer, STATE_DIM, NATIVE_CORE_DIM, SIDE_CHANNEL_DIM,
    assemble_state_vec, _signature_from_centroid,
)
from titan_plugin.logic.emot_kin_protocol import (
    build_kin_emot_state_payload, parse_kin_emot_state,
    compute_msl_activations, KIN_EMOT_STATE_MSG_TYPE,
)


# ── Bundle protocol ───────────────────────────────────────────────────

class TestBundleProtocol:
    """Schema invariants + roundtrip + safety."""

    def test_dim_constants(self):
        assert FELT_TENSOR_DIM == 130
        assert TRAJECTORY_DIM == 2
        assert SPACE_TOPOLOGY_DIM == 30
        assert NEUROMOD_DIM == 6
        native_total = (FELT_TENSOR_DIM + TRAJECTORY_DIM
                        + SPACE_TOPOLOGY_DIM + NEUROMOD_DIM)
        assert native_total == 168, "native consciousness = 168D"
        assert TRUNK_DIM == INNER_DIM == OUTER_DIM == 32  # L5 reserved
        assert len(LEGACY_PRIMITIVES) == 8
        assert len(NS_PROGRAMS) == 11 == HORMONE_DIM == NS_URGENCY_DIM
        assert len(CGN_CONSUMERS) == 8 == CGN_BETA_DIM
        assert len(MSL_CONCEPTS) == 6 == MSL_ACT_DIM

    def test_bundle_size_math(self):
        assert CORE_SIZE + RESERVED_TAIL == BUNDLE_SIZE
        assert BUNDLE_SIZE == 2048
        assert RESERVED_TAIL > 0, "must have room to grow"

    def test_roundtrip_native_fields(self, tmp_path):
        path = str(tmp_path / "bundle.bin")
        w = BundleWriter(path=path, titan_id="T1")
        felt = list(np.linspace(0.1, 0.9, 130).astype(np.float32))
        w.write(
            encoder_id=ENCODER_THIN_ASSEMBLY,
            felt_tensor_130d=felt, trajectory_2d=[0.15, -0.22],
            space_topology_30d=[0.5] * 30,
            neuromod_state_6d=[0.6, 0.7, 0.4, 0.5, 0.5, 0.5],
            hormone_levels_11d=[0.1] * 11, ns_urgencies_11d=[0.2] * 11,
            cgn_beta_states_8d=[0.3] * 8, msl_activations_6d=[0.4] * 6,
            pi_phase_6d=[0.5] * 6,
            region_id=REGION_UNCLUSTERED, legacy_idx=6,
            graduation_status=GRAD_SHADOW, regions_emerged=0,
            valence=0.3, arousal=-0.1, novelty=0.7,
            region_confidence=0.5, region_residence_s=42.0,
            region_signature=0xABCDEF,
        )
        r = BundleReader(path=path, expected_titan_id="T1")
        d = r.read()
        assert d is not None
        assert np.allclose(d["felt_tensor"][:10], felt[:10], atol=1e-5)
        assert abs(d["trajectory"][0] - 0.15) < 1e-5
        assert abs(d["neuromod_state"][2] - 0.4) < 1e-5
        assert d["legacy_label"] == "WONDER"  # idx 6
        assert d["region_id"] == REGION_UNCLUSTERED
        assert d["region_signature"] == 0xABCDEF
        assert abs(d["valence"] - 0.3) < 1e-5

    def test_l5_reserved_zero_with_thin_encoder(self, tmp_path):
        path = str(tmp_path / "bundle.bin")
        w = BundleWriter(path=path, titan_id="T1")
        w.write(encoder_id=ENCODER_THIN_ASSEMBLY,
                felt_tensor_130d=None, trajectory_2d=None,
                space_topology_30d=None, neuromod_state_6d=None,
                hormone_levels_11d=None, ns_urgencies_11d=None,
                cgn_beta_states_8d=None, msl_activations_6d=None,
                pi_phase_6d=None)
        d = BundleReader(path=path, expected_titan_id="T1").read()
        assert all(v == 0.0 for v in d["z_trunk"])
        assert all(v == 0.0 for v in d["z_inner"])
        assert all(v == 0.0 for v in d["z_outer"])

    def test_titan_id_mismatch_rejected(self, tmp_path):
        path = str(tmp_path / "bundle.bin")
        BundleWriter(path=path, titan_id="T1").write(
            encoder_id=ENCODER_THIN_ASSEMBLY,
            felt_tensor_130d=[0.5] * 130, trajectory_2d=[0] * 2,
            space_topology_30d=[0.5] * 30,
            neuromod_state_6d=[0.5] * 6,
            hormone_levels_11d=[0] * 11, ns_urgencies_11d=[0] * 11,
            cgn_beta_states_8d=[0] * 8, msl_activations_6d=[0] * 6,
            pi_phase_6d=[0] * 6)
        assert BundleReader(path=path, expected_titan_id="T1").read() is not None
        assert BundleReader(path=path, expected_titan_id="T2").read() is None
        assert BundleReader(path=path, expected_titan_id="T3").read() is None

    def test_has_new_version_counter(self, tmp_path):
        path = str(tmp_path / "bundle.bin")
        w = BundleWriter(path=path, titan_id="T1")
        r = BundleReader(path=path, expected_titan_id="T1")
        w.write(encoder_id=0, felt_tensor_130d=[0.5] * 130,
                trajectory_2d=[0] * 2, space_topology_30d=[0.5] * 30,
                neuromod_state_6d=[0.5] * 6,
                hormone_levels_11d=[0] * 11, ns_urgencies_11d=[0] * 11,
                cgn_beta_states_8d=[0] * 8, msl_activations_6d=[0] * 6,
                pi_phase_6d=[0] * 6)
        d = r.read()
        assert d is not None
        assert not r.has_new()  # just read
        w.write(encoder_id=0, felt_tensor_130d=[0.6] * 130,
                trajectory_2d=[0] * 2, space_topology_30d=[0.5] * 30,
                neuromod_state_6d=[0.5] * 6,
                hormone_levels_11d=[0] * 11, ns_urgencies_11d=[0] * 11,
                cgn_beta_states_8d=[0] * 8, msl_activations_6d=[0] * 6,
                pi_phase_6d=[0] * 6)
        assert r.has_new()

    def test_fnv1a_stable(self):
        assert _fnv1a_32("T1") == _fnv1a_32("T1")
        assert _fnv1a_32("T1") != _fnv1a_32("T2")


# ── Plug A/C helpers ──────────────────────────────────────────────────

class TestPlugHelpers:
    def test_valence_normalization_unavailable(self):
        assert read_emotion_valence_normalized(
            titan_id="NONEXISTENT", default=0.5) == 0.5

    def test_valence_remapping(self, tmp_path):
        path = str(tmp_path / "b.bin")
        w = BundleWriter(path=path, titan_id="T1")
        for valence, expected_slot in [(0.8, 0.9), (-0.4, 0.3), (0.0, 0.5)]:
            w.write(encoder_id=0, felt_tensor_130d=[0.5] * 130,
                    trajectory_2d=[0] * 2, space_topology_30d=[0.5] * 30,
                    neuromod_state_6d=[0.5] * 6,
                    hormone_levels_11d=[0] * 11, ns_urgencies_11d=[0] * 11,
                    cgn_beta_states_8d=[0] * 8, msl_activations_6d=[0] * 6,
                    pi_phase_6d=[0] * 6,
                    valence=valence, arousal=0, novelty=0.5,
                    region_id=-2, legacy_idx=0, region_confidence=0,
                    region_residence_s=0, region_signature=0,
                    graduation_status=0, regions_emerged=0)
            r = BundleReader(path=path, expected_titan_id="T1")
            slot = read_emotion_valence_normalized(reader=r, default=0.5)
            assert abs(slot - expected_slot) < 1e-5, \
                f"v={valence} → {slot}, expected {expected_slot}"

    def test_full_context_fields(self, tmp_path):
        path = str(tmp_path / "b.bin")
        BundleWriter(path=path, titan_id="T1").write(
            encoder_id=1, felt_tensor_130d=[0.5] * 130,
            trajectory_2d=[0] * 2, space_topology_30d=[0.5] * 30,
            neuromod_state_6d=[0.5] * 6,
            hormone_levels_11d=[0] * 11, ns_urgencies_11d=[0] * 11,
            cgn_beta_states_8d=[0] * 8, msl_activations_6d=[0] * 6,
            pi_phase_6d=[0] * 6,
            valence=0.4, arousal=-0.2, novelty=0.9,
            region_id=5, legacy_idx=3, regions_emerged=7,
            region_confidence=0.85, region_residence_s=120.0,
            region_signature=0xCAFEBABE, graduation_status=1)
        ctx = read_full_emotion_context(
            reader=BundleReader(path=path, expected_titan_id="T1"))
        assert ctx is not None
        assert ctx["region_id"] == 5
        assert ctx["legacy_label"] == "PEACE"  # idx 3
        assert abs(ctx["valence"] - 0.4) < 1e-5
        assert ctx["regions_emerged"] == 7
        assert ctx["region_signature"] == 0xCAFEBABE
        assert ctx["encoder_id"] == 1
        assert ctx["graduation_status"] == 1

    def test_full_context_none_when_unavailable(self):
        assert read_full_emotion_context(titan_id="NONEXISTENT") is None


# ── Thin encoder ──────────────────────────────────────────────────────

class TestThinEncoder:
    def test_shapes(self):
        enc = ThinEmotEncoder(titan_id="T1")
        out = enc.encode(felt_tensor_130d=[0.5] * 130)
        assert out["felt_tensor_130d"].shape == (130,)
        assert out["trajectory_2d"].shape == (2,)
        assert out["space_topology_30d"].shape == (30,)
        assert out["neuromod_state_6d"].shape == (6,)

    def test_lossless_passthrough(self):
        enc = ThinEmotEncoder(titan_id="T1")
        felt = list(np.linspace(0, 1, 130).astype(np.float32))
        out = enc.encode(felt_tensor_130d=felt,
                         trajectory_2d=[0.1, 0.2])
        assert np.allclose(out["felt_tensor_130d"], felt)
        assert abs(out["trajectory_2d"][0] - 0.1) < 1e-5

    def test_valence_reward_ema(self):
        enc = ThinEmotEncoder(titan_id="T1")
        # Warmup
        for _ in range(40):
            out = enc.encode(felt_tensor_130d=[0.5] * 130,
                             last_terminal_reward=0.9)
        # After 40 high-reward ticks, valence should be strongly positive
        assert out["valence"] > 0.2

    def test_arousal_from_neuromods(self):
        enc = ThinEmotEncoder(titan_id="T1")
        out_high = enc.encode(neuromod_state_6d=[0.9, 0.5, 0.9, 0.5, 0.5, 0.5])
        out_low = enc.encode(neuromod_state_6d=[0.1, 0.5, 0.1, 0.5, 0.5, 0.5])
        assert out_high["arousal"] > 0.5
        assert out_low["arousal"] < -0.5

    def test_novelty_warmup(self):
        enc = ThinEmotEncoder(titan_id="T1")
        # First call returns neutral (warmup)
        out = enc.encode(felt_tensor_130d=[0.5] * 130)
        assert out["novelty"] == 0.5

    def test_novelty_tier1_dynamic_range(self):
        """Tier-1 fix (2026-04-22): novelty must produce real dynamic range
        during the pre-emergence phase. Prior cosine-on-cumulative-mean
        implementation flat-lined near 0 (<0.01) regardless of input,
        starving observers of signal during the 14-day soak window."""
        import numpy as np
        enc = ThinEmotEncoder(titan_id="T1")
        rng = np.random.default_rng(42)

        # Warm up with ~30 stable-ish observations near 0.5 mean
        for _ in range(30):
            felt = list(rng.normal(0.5, 0.05, 130).astype(np.float32))
            enc.encode(felt_tensor_130d=felt)

        # Baseline state — novelty should be in moderate range, NOT pinned near 0
        baseline_samples = []
        for _ in range(20):
            felt = list(rng.normal(0.5, 0.05, 130).astype(np.float32))
            out = enc.encode(felt_tensor_130d=felt)
            baseline_samples.append(out["novelty"])
        baseline_mean = sum(baseline_samples) / len(baseline_samples)

        # Novelty shouldn't flatline — baseline variability should register
        # SOMEWHERE in [0.1, 0.9] rather than stuck at ~0.
        assert 0.05 < baseline_mean < 0.95, (
            f"baseline novelty flatlined at {baseline_mean:.4f} — "
            f"Tier-1 EMA approach should produce real variation")

        # Big deviation should SPIKE novelty
        spike_felt = list(rng.normal(0.9, 0.05, 130).astype(np.float32))
        out_spike = enc.encode(felt_tensor_130d=spike_felt)
        spike_novelty = out_spike["novelty"]
        assert spike_novelty > baseline_mean + 0.15, (
            f"unusual state should spike novelty above baseline "
            f"({baseline_mean:.3f}); got {spike_novelty:.3f}")

    def test_novelty_bounded(self):
        """Novelty must always stay in [0, 1]. Extreme input shouldn't overflow."""
        import numpy as np
        enc = ThinEmotEncoder(titan_id="T1")
        # Warm up
        for _ in range(15):
            enc.encode(felt_tensor_130d=[0.5] * 130)
        # Extreme inputs — very high and very low
        for v in [0.0, 1.0, 0.999, 0.001]:
            out = enc.encode(felt_tensor_130d=[v] * 130)
            assert 0.0 <= out["novelty"] <= 1.0, (
                f"novelty {out['novelty']} out of [0,1] for input {v}")

    def test_novelty_still_state_low(self):
        """Titan in a very stable trajectory should converge to low novelty,
        not hover at 0.5 forever."""
        import numpy as np
        enc = ThinEmotEncoder(titan_id="T1")
        rng = np.random.default_rng(7)
        felt_stable = [0.5] * 130  # exactly baseline, no noise

        novelties = []
        for _ in range(50):
            out = enc.encode(felt_tensor_130d=felt_stable)
            novelties.append(out["novelty"])

        # After many identical observations, novelty should drop below
        # warmup neutral. 0.3 is a reasonable "settled-quiet" threshold.
        late_avg = sum(novelties[-10:]) / 10
        assert late_avg < 0.4, (
            f"static state should produce low novelty after warmup; "
            f"got late avg {late_avg:.3f}")


# ── Region clusterer ──────────────────────────────────────────────────

class TestRegionClusterer:
    def test_state_dim(self):
        # Schema v2 (2026-04-21): pi_phase 4D → 6D; STATE_DIM 208 → 210.
        assert STATE_DIM == 210
        assert NATIVE_CORE_DIM == 168
        assert SIDE_CHANNEL_DIM == 42

    def test_assemble_state_vec(self):
        encoded = {
            "felt_tensor_130d": np.full(130, 0.5, dtype=np.float32),
            "trajectory_2d": np.array([0.1, 0.2], dtype=np.float32),
            "space_topology_30d": np.full(30, 0.3, dtype=np.float32),
            "neuromod_state_6d": np.full(6, 0.4, dtype=np.float32),
        }
        sv = assemble_state_vec(encoded)
        assert sv.shape == (STATE_DIM,)
        assert sv[0] == 0.5   # felt
        assert sv[130] == 0.1  # traj[0]
        assert sv[132] == 0.3  # space[0]

    def test_signature_stability(self):
        c1 = np.array([0.1] * STATE_DIM, dtype=np.float32)
        c2 = c1 + 0.01  # below bucket=0.05 → same sig
        c3 = c1 + 0.5   # above bucket → different sig
        assert _signature_from_centroid(c1) == _signature_from_centroid(c2)
        assert _signature_from_centroid(c1) != _signature_from_centroid(c3)

    def test_unclustered_until_density(self, tmp_path):
        rc = RegionClusterer(save_dir=str(tmp_path), min_cluster_size=5)
        rid, conf, res, sig = rc.observe(np.zeros(STATE_DIM,dtype=np.float32))
        assert rid == REGION_UNCLUSTERED
        assert sig == 0

    def test_discovers_two_clusters(self, tmp_path):
        rc = RegionClusterer(save_dir=str(tmp_path), min_cluster_size=5)
        rng = np.random.default_rng(1)
        for _ in range(60):
            rc.observe(rng.normal(0.2, 0.01, STATE_DIM).astype(np.float32))
        for _ in range(60):
            rc.observe(rng.normal(0.8, 0.01, STATE_DIM).astype(np.float32))
        n = rc.recluster()
        assert n == 2, f"expected 2 regions, got {n}"

    def test_noise_for_far_points(self, tmp_path):
        # Need ≥2 clusters for HDBSCAN to not label everything noise.
        rc = RegionClusterer(save_dir=str(tmp_path), min_cluster_size=5)
        rng = np.random.default_rng(1)
        for _ in range(60):
            rc.observe(rng.normal(0.2, 0.01, STATE_DIM).astype(np.float32))
        for _ in range(60):
            rc.observe(rng.normal(0.8, 0.01, STATE_DIM).astype(np.float32))
        assert rc.recluster() >= 1, "need at least one cluster"
        # Probe far from any centroid
        rid, _, _, _ = rc.observe(np.full(STATE_DIM,5.0, dtype=np.float32))
        assert rid == REGION_NOISE

    def test_stable_region_id_across_reload(self, tmp_path):
        sd = str(tmp_path)
        rc1 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        rng = np.random.default_rng(1)
        for _ in range(60):
            rc1.observe(rng.normal(0.2, 0.01, STATE_DIM).astype(np.float32))
        for _ in range(60):
            rc1.observe(rng.normal(0.8, 0.01, STATE_DIM).astype(np.float32))
        rc1.recluster()
        rid_a_before, _, _, sig_a = rc1.observe(
            np.full(STATE_DIM,0.2, dtype=np.float32))
        rc1.save_state()

        rc2 = RegionClusterer(save_dir=sd, min_cluster_size=5)
        assert rc2.regions_count() == 2
        rid_a_after, _, _, sig_a2 = rc2.observe(
            np.full(STATE_DIM,0.2, dtype=np.float32))
        assert rid_a_before == rid_a_after
        assert sig_a == sig_a2


# ── Kin protocol + MSL binding ────────────────────────────────────────

class TestKinProtocol:
    def test_roundtrip(self):
        p = build_kin_emot_state_payload(
            titan_src="T2", region_id=3, region_signature=0xCAFEBABE,
            region_confidence=0.8, region_residence_s=120.0,
            regions_emerged=4, valence=0.6, arousal=0.2, novelty=0.4,
            legacy_idx=0, encoder_id=0)
        parsed = parse_kin_emot_state(p, expected_self_id="T1")
        assert parsed is not None
        assert parsed["titan_src"] == "T2"
        assert parsed["region_signature"] == 0xCAFEBABE
        assert abs(parsed["valence"] - 0.6) < 1e-5
        assert parsed["age_s"] < 1.0

    def test_own_emission_filtered(self):
        p = build_kin_emot_state_payload(
            titan_src="T1", region_id=0, region_signature=1,
            region_confidence=0.5, region_residence_s=0, regions_emerged=1,
            valence=0, arousal=0, novelty=0.5, legacy_idx=0, encoder_id=0)
        assert parse_kin_emot_state(p, expected_self_id="T1") is None

    def test_stale_rejected(self):
        p = build_kin_emot_state_payload(
            titan_src="T2", region_id=0, region_signature=1,
            region_confidence=0.5, region_residence_s=0, regions_emerged=1,
            valence=0, arousal=0, novelty=0.5, legacy_idx=0, encoder_id=0,
            ts_ms=int((time.time() - 1000) * 1000))
        assert parse_kin_emot_state(
            p, expected_self_id="T1", freshness_s=300) is None

    def test_malformed_rejected(self):
        assert parse_kin_emot_state({}, expected_self_id="T1") is None
        assert parse_kin_emot_state(None, expected_self_id="T1") is None

    def test_msl_self_only(self):
        msl = compute_msl_activations(
            self_region_confidence=0.7, self_region_signature=0xAAA,
            self_valence=0.4, peer_state=None)
        assert len(msl) == 6
        I, YOU, ME, WE, YES, NO = msl
        assert I == 0.7 and YOU == 0.0 and WE == 0.0
        assert ME == 0.4
        assert abs(YES + NO - 1.0) < 1e-5

    def test_msl_same_region(self):
        msl = compute_msl_activations(
            self_region_confidence=0.8, self_region_signature=0xAAA,
            self_valence=0.3,
            peer_state={"region_confidence": 0.7, "region_signature": 0xAAA})
        _, _, _, WE, _, _ = msl
        assert WE == 1.0

    def test_msl_different_region(self):
        msl = compute_msl_activations(
            self_region_confidence=0.8, self_region_signature=0xAAA,
            self_valence=-0.5,
            peer_state={"region_confidence": 0.7, "region_signature": 0xBBB})
        _, YOU, _, WE, YES, NO = msl
        assert WE == 0.0
        assert YOU == 0.7
        assert NO > YES  # negative valence

    def test_msl_zero_signature_no_we(self):
        """Two Titans with region_signature=0 (both unclustered) should
        NOT activate WE — it'd be a spurious 'we're both in no-region.'"""
        msl = compute_msl_activations(
            self_region_confidence=0.0, self_region_signature=0,
            self_valence=0,
            peer_state={"region_confidence": 0.0, "region_signature": 0})
        assert msl[3] == 0.0  # WE


# ── Plug B library method (CGNConsumerClient) ─────────────────────────

class TestPlugB:
    def test_cross_insight_ema(self):
        from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
        c = CGNConsumerClient(consumer_name="language")
        assert c._emot_insight_reward_ema == 0.5

        c.note_incoming_cross_insight({
            "origin_consumer": "emotional",
            "terminal_reward": 0.9,
        })
        assert abs(c._emot_insight_reward_ema - 0.62) < 1e-5
        assert c._emot_insight_count == 1

        c.note_incoming_cross_insight({
            "origin_consumer": "emotional",
            "terminal_reward": 0.1,
        })
        assert abs(c._emot_insight_reward_ema - 0.464) < 1e-5

    def test_own_emission_ignored(self):
        from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
        c = CGNConsumerClient(consumer_name="language")
        c.note_incoming_cross_insight({
            "origin_consumer": "language",
            "terminal_reward": 0.9,
        })
        assert c._emot_insight_reward_ema == 0.5  # unchanged

    def test_non_emotional_origin_ignored(self):
        from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
        c = CGNConsumerClient(consumer_name="knowledge")
        c.note_incoming_cross_insight({
            "origin_consumer": "meta",
            "terminal_reward": 0.9,
        })
        assert c._emot_insight_reward_ema == 0.5  # unchanged
