"""Tests for emotion_cluster.EmotionClusterer (rFP_emot_cgn_v2)."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from titan_plugin.logic.emotion_cluster import (
    EMOT_PRIMITIVES,
    EMOT_PRIMITIVE_INDEX,
    FEATURE_DIM,
    NUM_PRIMITIVES,
    EmotionCluster,
    EmotionClusterer,
    _neutral_centroid,
    _seed_centroids,
)


# ── Module-level constants ──────────────────────────────────────────

def test_num_primitives_is_8():
    assert NUM_PRIMITIVES == 8
    assert len(EMOT_PRIMITIVES) == 8


def test_feature_dim_is_150():
    assert FEATURE_DIM == 150


def test_primitives_include_love():
    """LOVE was Maker's addition 2026-04-19; must be present."""
    assert "LOVE" in EMOT_PRIMITIVES


def test_primitive_index_mapping_consistent():
    for i, p in enumerate(EMOT_PRIMITIVES):
        assert EMOT_PRIMITIVE_INDEX[p] == i


def test_neutral_centroid_shape_150d():
    c = _neutral_centroid()
    assert c.shape == (FEATURE_DIM,)
    # felt tensor slice at neutral 0.5
    assert np.allclose(c[:130], 0.5)


def test_seed_centroids_shape():
    s = _seed_centroids()
    assert s.shape == (NUM_PRIMITIVES, FEATURE_DIM)


def test_seed_centroids_anchors_distinct():
    s = _seed_centroids()
    # FLOW, IMPASSE_TENSION, RESOLUTION (indices 0,1,2) should differ
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(s[i], s[j])


# ── EmotionClusterer construction ─────────────────────────────────

def test_clusterer_construction_creates_8_clusters():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        assert len(cl._clusters) == 8
        for p in EMOT_PRIMITIVES:
            assert p in cl._clusters
            assert isinstance(cl._clusters[p], EmotionCluster)


def test_clusterer_anchors_marked_emerged():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        assert cl._clusters["FLOW"].is_emerged
        assert cl._clusters["IMPASSE_TENSION"].is_emerged
        assert cl._clusters["RESOLUTION"].is_emerged


def test_clusterer_emergent_slots_not_emerged_at_start():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        for p in ["PEACE", "CURIOSITY", "GRIEF", "WONDER", "LOVE"]:
            assert not cl._clusters[p].is_emerged


def test_clusterer_save_dir_created():
    with tempfile.TemporaryDirectory() as tmp:
        subdir = os.path.join(tmp, "emot_sub")
        EmotionClusterer(save_dir=subdir)
        assert os.path.isdir(subdir)


# ── Assignment / observation ──────────────────────────────────────

def test_assign_returns_valid_primitive():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        p, d, c = cl.assign(v)
        assert p in EMOT_PRIMITIVES
        assert d >= 0.0
        assert 0.0 <= c <= 1.0


def test_assign_failsafe_on_short_vector():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        p, d, c = cl.assign(np.array([0.5, 0.5]))
        assert p == "FLOW"  # documented fallback


def test_assign_failsafe_on_none():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        p, d, c = cl.assign(None)
        assert p == "FLOW"


def test_observe_updates_cluster_count():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        start = cl._clusters["FLOW"].n_observations + sum(
            cl._clusters[p].n_observations for p in EMOT_PRIMITIVES)
        cl.observe(v)
        end = sum(cl._clusters[p].n_observations for p in EMOT_PRIMITIVES)
        assert end == start + 1


def test_observe_updates_recent_assignments_deque():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        for _ in range(10):
            cl.observe(v)
        # Deque maxlen 8
        assert len(cl._recent_assignments) <= 8


def test_cluster_history_onehot_sums_to_1():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        for _ in range(5):
            cl.observe(v)
        hist = cl.get_cluster_history_onehot()
        assert hist.shape == (NUM_PRIMITIVES,)
        assert abs(float(hist.sum()) - 1.0) < 1e-3


def test_emergence_threshold_fires():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        # Force vector close to PEACE centroid (index 3)
        target = cl._clusters["PEACE"].centroid.copy()
        assert not cl._clusters["PEACE"].is_emerged
        for _ in range(150):  # > threshold of 100
            cl.observe(target)
        # PEACE should have emerged (or at least some cluster did)
        any_new_emerged = any(cl._clusters[p].is_emerged
                              for p in ["PEACE", "CURIOSITY", "GRIEF",
                                        "WONDER", "LOVE"])
        assert any_new_emerged


# ── Recentering ───────────────────────────────────────────────────

def test_maybe_recenter_skips_without_data():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        # No observations yet — should skip
        assert not cl.maybe_recenter(force=True)


def test_maybe_recenter_with_force_and_data():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        # Observe 100 times — ~20 will be stored (every 5th)
        for _ in range(100):
            cl.observe(v)
        # Not enough observations per cluster → likely skips even with force
        result = cl.maybe_recenter(force=True)
        # May be True or False depending on distribution, but doesn't crash
        assert isinstance(result, bool)


# ── Seeding from dream clusters ───────────────────────────────────

def test_seed_from_dream_clusters_empty():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        assert cl.seed_from_dream_clusters([]) == 0


def test_seed_from_dream_clusters_valid_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        dreams = [
            {"tensor": [0.3] * 130},
            {"tensor": [0.7] * 130},
            {"tensor": [0.4] * 130},
        ]
        seeded = cl.seed_from_dream_clusters(dreams)
        assert seeded >= 1
        assert seeded <= 5  # only 5 emergent slots


def test_seed_from_dream_clusters_malformed_tensor_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        # Too-short tensor should be skipped
        seeded = cl.seed_from_dream_clusters([{"tensor": [0.5] * 10}])
        assert seeded == 0


# ── Persistence ───────────────────────────────────────────────────

def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cl1 = EmotionClusterer(save_dir=tmp)
        v = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        for _ in range(10):
            cl1.observe(v)
        cl1.save_state()
        assert os.path.exists(os.path.join(tmp, "clusters_state.json"))
        # Reload
        cl2 = EmotionClusterer(save_dir=tmp)
        total2 = sum(cl2._clusters[p].n_observations for p in EMOT_PRIMITIVES)
        assert total2 == 10


def test_set_label_renames_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        ok = cl.set_label("LOVE", "heart-warm")
        assert ok
        assert cl._clusters["LOVE"].label == "heart-warm"


def test_set_label_rejects_unknown_primitive():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        assert not cl.set_label("UNKNOWN", "x")


def test_snapshot_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        cl = EmotionClusterer(save_dir=tmp)
        cl._snapshot()
        assert os.path.exists(os.path.join(tmp, "cluster_snapshots.jsonl"))
