"""Integration test for Phase A producer wiring (rFP §23.6+).

Catches the "schema defined, producer forgot" class of bug that slipped
past the original v3 session: bundle slots were allocated in the
protocol, encoder/writer/reader/clusterer all worked, but 6 of the 9
field groups were silently receiving zeros because no producer was
sending them.

These tests assert the full BundleWriter → BundleReader → assemble_state_vec
→ RegionClusterer path carries data for EVERY field group when realistic
inputs are provided. Failure = a group was dropped/truncated somewhere
in the pipeline. Pairs with A4's runtime dead-dim detector (which
catches the same class of bug live, at 15-min recluster cadence).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from titan_plugin.logic.emot_bundle_protocol import (
    BundleReader, BundleWriter, BUNDLE_SCHEMA_VERSION,
    ENCODER_THIN_ASSEMBLY,
    FELT_TENSOR_DIM, TRAJECTORY_DIM, SPACE_TOPOLOGY_DIM, NEUROMOD_DIM,
    HORMONE_DIM, NS_URGENCY_DIM, CGN_BETA_DIM, MSL_ACT_DIM, PI_PHASE_DIM,
    REGION_UNCLUSTERED, GRAD_SHADOW,
)
from titan_plugin.logic.emot_region_clusterer import (
    RegionClusterer, STATE_DIM, assemble_state_vec,
)
from titan_plugin.logic.emot_thin_encoder import ThinEmotEncoder


# Realistic input fixtures — non-zero, non-constant per group so that
# each group has positive std and the wiring check is meaningful.
def _realistic_inputs():
    rng = np.random.default_rng(42)
    return {
        "felt_tensor_130d":      rng.uniform(0.1, 0.9, FELT_TENSOR_DIM).tolist(),
        "trajectory_2d":         [0.37, 0.64],
        "space_topology_30d":    rng.uniform(0.2, 0.8, SPACE_TOPOLOGY_DIM).tolist(),
        "neuromod_state_6d":     [0.7, 0.6, 0.55, 0.5, 0.45, 0.4],
        "hormone_levels_11d":    rng.uniform(0.3, 1.5, HORMONE_DIM).tolist(),
        "ns_urgencies_11d":      rng.uniform(0.0, 0.7, NS_URGENCY_DIM).tolist(),
        "cgn_beta_states_8d":    rng.uniform(0.2, 0.5, CGN_BETA_DIM).tolist(),
        "msl_activations_6d":    [0.6, 0.3, 0.8, 0.4, 0.5, 0.1],
        "pi_phase_6d":           [0.12, 0.47, 0.89, 0.21, 0.65, 0.34],
    }


class TestBundleWriterAllGroupsPopulated:
    """Prove every bundle field group round-trips cleanly when the
    producer actually sends data. Schema allocates slots → writer
    packs → reader unpacks → downstream sees non-zero variance.
    """

    def test_all_field_groups_roundtrip_nonzero(self, tmp_path):
        """The wiring-gap regression guard.

        Fails if any group is silently dropped between writer and
        reader. One assert per group — failure message names the
        offending group directly.
        """
        inputs = _realistic_inputs()
        path = str(tmp_path / "b.bin")
        BundleWriter(path=path, titan_id="T1").write(
            encoder_id=ENCODER_THIN_ASSEMBLY,
            **inputs,
            region_id=REGION_UNCLUSTERED, legacy_idx=0,
            graduation_status=GRAD_SHADOW, regions_emerged=0,
            valence=0.2, arousal=0.3, novelty=0.4,
            region_confidence=0.5, region_residence_s=10.0,
            region_signature=0xDEADBEEF)
        d = BundleReader(path=path, expected_titan_id="T1").read()
        assert d is not None, "bundle read failed — schema mismatch?"

        # Map input key → reader output key (reader strips _NNd suffix).
        reader_keys = {
            "felt_tensor_130d":   "felt_tensor",
            "trajectory_2d":      "trajectory",
            "space_topology_30d": "space_topology",
            "neuromod_state_6d":  "neuromod_state",
            "hormone_levels_11d": "hormone_levels",
            "ns_urgencies_11d":   "ns_urgencies",
            "cgn_beta_states_8d": "cgn_beta_states",
            "msl_activations_6d": "msl_activations",
            "pi_phase_6d":        "pi_phase",
        }
        for in_key, out_key in reader_keys.items():
            arr = np.asarray(d[out_key], dtype=np.float32)
            nonzero = int((arr != 0).sum())
            assert nonzero > 0, (
                f"bundle group '{out_key}' came back ALL ZEROS — "
                f"producer-wiring regression in the path for '{in_key}'. "
                f"Check emot_bundle_protocol.BundleWriter / .BundleReader / "
                f"consumer handler in emot_cgn_worker.")
            assert float(arr.std()) > 1e-6, (
                f"bundle group '{out_key}' has zero variance even "
                f"though producer sent varying inputs. "
                f"Likely a slicing/truncation bug in the writer.")

    def test_schema_version_advertised(self, tmp_path):
        """Schema version surfaces through reader — required so consumers
        know which layout they're looking at (v2 has 6D pi_phase vs v1's 4D)."""
        path = str(tmp_path / "b.bin")
        BundleWriter(path=path, titan_id="T1").write(
            encoder_id=ENCODER_THIN_ASSEMBLY, **_realistic_inputs(),
            region_id=-2, legacy_idx=0, regions_emerged=0,
            graduation_status=0, valence=0, arousal=0, novelty=0.5,
            region_confidence=0, region_residence_s=0, region_signature=0)
        d = BundleReader(path=path, expected_titan_id="T1").read()
        assert d["schema_version"] == BUNDLE_SCHEMA_VERSION
        assert BUNDLE_SCHEMA_VERSION == 2, (
            "schema v2 expected — pi_phase widened 4D→6D in 2026-04-21 "
            "Trinity-symmetry commit; if you're changing this, also bump "
            "the dependent consts in emot_region_clusterer")


class TestThinEncoderPassesThrough:
    """Encoder is pure assembly — every input key must survive unchanged
    into the output dict. Regressions here produce silent zeros
    downstream (RegionClusterer won't see the data even if the producer
    sent it).
    """

    def test_encoder_preserves_all_groups(self):
        enc = ThinEmotEncoder(titan_id="T1")
        inputs = _realistic_inputs()
        out = enc.encode(
            last_terminal_reward=0.6,
            **inputs,
        )
        # Compare each input key — each must appear in `out` with
        # matching dim count and non-zero variance.
        for key in inputs:
            assert key in out, (
                f"encoder dropped key '{key}' — output dict missing it")
            arr = np.asarray(out[key], dtype=np.float32)
            assert arr.size == len(inputs[key]), (
                f"encoder truncated '{key}': "
                f"in={len(inputs[key])} → out={arr.size}")
            assert float(arr.std()) > 1e-6 or len(inputs[key]) <= 1, (
                f"encoder flattened variance in '{key}'")


class TestAssembleStateVecAllGroups:
    """Regression guard for assemble_state_vec — the 210D state vector
    HDBSCAN sees must include all 9 groups in the right order with
    correct per-group dim counts.
    """

    def test_state_vec_has_all_groups_at_correct_offsets(self):
        encoded = {k: np.asarray(v, dtype=np.float32)
                   for k, v in _realistic_inputs().items()}
        sv = assemble_state_vec(encoded)
        assert sv.shape == (STATE_DIM,) == (210,), \
            f"STATE_DIM drift — got {sv.shape}, expected 210"
        # Each group range should match the corresponding input.
        offset = 0
        for key, dim in [
            ("felt_tensor_130d",   FELT_TENSOR_DIM),
            ("trajectory_2d",      TRAJECTORY_DIM),
            ("space_topology_30d", SPACE_TOPOLOGY_DIM),
            ("neuromod_state_6d",  NEUROMOD_DIM),
            ("hormone_levels_11d", HORMONE_DIM),
            ("ns_urgencies_11d",   NS_URGENCY_DIM),
            ("cgn_beta_states_8d", CGN_BETA_DIM),
            ("msl_activations_6d", MSL_ACT_DIM),
            ("pi_phase_6d",        PI_PHASE_DIM),
        ]:
            slice_ = sv[offset:offset + dim]
            expected = np.asarray(encoded[key], dtype=np.float32)[:dim]
            assert np.allclose(slice_, expected, atol=1e-5), (
                f"group '{key}' placed at wrong offset {offset} or "
                f"data corrupted in assembly")
            offset += dim
        assert offset == STATE_DIM, \
            f"total assembled dim {offset} != STATE_DIM {STATE_DIM}"


class TestDeadDimDetectorCatchesRegression:
    """A4 dead-dim detector fires WARN + logs to recluster_telemetry.jsonl
    when any group has zero variance across the buffer. Here we simulate
    a regression (one group sent all zeros while others vary) and
    assert the detector catches it.
    """

    def test_detector_fires_for_zero_variance_group(self, tmp_path):
        rc = RegionClusterer(save_dir=str(tmp_path), min_cluster_size=5)
        rng = np.random.default_rng(0)
        # 2 clusters so HDBSCAN finds structure, but zero out pi_phase
        # slot to simulate a pre-A2.4 wiring gap.
        PI_START = STATE_DIM - PI_PHASE_DIM
        for _ in range(60):
            v = rng.normal(0.3, 0.05, STATE_DIM).astype(np.float32)
            v[PI_START:] = 0.0
            rc.observe(v)
        for _ in range(60):
            v = rng.normal(0.7, 0.05, STATE_DIM).astype(np.float32)
            v[PI_START:] = 0.0
            rc.observe(v)
        rc.recluster()

        # Telemetry file should exist with one entry.
        tel_path = os.path.join(str(tmp_path), "recluster_telemetry.jsonl")
        assert os.path.exists(tel_path)
        import json
        with open(tel_path) as f:
            entry = json.loads(f.readline().strip())
        assert "pi_phase" in entry["dead_groups"], (
            "dead-dim detector failed to flag pi_phase as zero-variance")
        assert entry["per_group"]["pi_phase"]["std"] == 0.0
        # healthy groups should NOT appear
        assert "felt_tensor" not in entry["dead_groups"]
