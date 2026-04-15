"""Tests for Dreaming clustering rewrite — rFP #3 Phase 2.

Covers the `_distill_experiences` clustering path: element-wise aggregation
with temporal-coherence clustering, proper `felt_tensor` list emission (the
root fix), and edge cases around mixed-dim / empty / no-significant buffers.
"""
import pytest

from titan_plugin.logic.dreaming import DreamingEngine


def _mk_engine(threshold: float = 0.85, distill_threshold: float = 0.005):
    """Build a DreamingEngine with a controllable cluster merge threshold.

    `cluster_merge_threshold` is loaded from DNA at construction time, so we
    pass it via the `dna` dict — matching production loading convention.
    """
    eng = DreamingEngine(dna={"cluster_merge_threshold": threshold})
    eng._distill_threshold = distill_threshold
    return eng


class TestDistillClusteringFelt130D:
    """rFP #3 Phase 2 — clustering + felt_tensor-as-list."""

    def test_felt_tensor_is_list_not_scalar(self):
        """THE ROOT FIX: emitted insight.felt_tensor is a list, not scalar."""
        import random
        random.seed(1)
        eng = _mk_engine()
        buffer = [
            {"full_130dt": [0.5 + random.uniform(-0.2, 0.2) for _ in range(130)],
             "ts": float(i)}
            for i in range(10)
        ]
        insights = eng._distill_experiences(buffer)
        assert len(insights) >= 1
        for ins in insights:
            assert isinstance(ins["felt_tensor"], list), \
                "felt_tensor must be a list, not a scalar"
            assert len(ins["felt_tensor"]) == 130
            # Explicit: no scalar tensor_mean leak
            assert "tensor_mean" not in ins

    def test_no_significant_snapshots_emits_nothing(self):
        """Flat buffer → variance below threshold → no insights."""
        eng = _mk_engine()
        buffer = [{"full_130dt": [0.5] * 130, "ts": float(i)} for i in range(5)]
        insights = eng._distill_experiences(buffer)
        assert insights == []

    def test_similar_snapshots_merge_into_one_cluster(self):
        """High-variance snapshots sharing direction → 1 cluster → 1 insight."""
        eng = _mk_engine(threshold=0.5)
        buffer = []
        for i in range(10):
            t = [0.9] * 65 + [0.1] * 65  # high variance, same direction
            t[i % 130] += 0.01  # slight uniqueness, still dominantly same shape
            buffer.append({"full_130dt": t, "ts": float(i)})
        insights = eng._distill_experiences(buffer)
        assert len(insights) == 1
        assert insights[0]["num_samples"] == 10
        assert insights[0]["cluster_idx"] == 0

    def test_two_distinct_experiences_produce_two_clusters(self):
        """Early joy + late fear pattern → 2 clusters, 2 insights, distinct centroids."""
        eng = _mk_engine(threshold=0.5)
        buffer = []
        # Cluster A — first 5 snapshots, high in first half
        for i in range(5):
            buffer.append({"full_130dt": [0.95] * 65 + [0.05] * 65, "ts": float(i)})
        # Cluster B — last 5 snapshots, opposite pattern
        for i in range(5):
            buffer.append({"full_130dt": [0.05] * 65 + [0.95] * 65, "ts": float(i + 5)})
        insights = eng._distill_experiences(buffer)
        assert len(insights) == 2
        # Cluster 0 should have high first-half values
        assert sum(insights[0]["felt_tensor"][:65]) / 65 > 0.8
        assert sum(insights[0]["felt_tensor"][65:]) / 65 < 0.2
        # Cluster 1 should have high second-half values
        assert sum(insights[1]["felt_tensor"][65:]) / 65 > 0.8
        assert sum(insights[1]["felt_tensor"][:65]) / 65 < 0.2

    def test_empty_buffer_returns_empty(self):
        """No buffer input → no insights."""
        eng = _mk_engine()
        assert eng._distill_experiences([]) == []

    def test_fallback_to_65d_works(self):
        """No full_130dt, but full_65dt present → emits 65D tensor."""
        eng = _mk_engine()
        buffer = []
        for i in range(5):
            buffer.append({"full_65dt": [0.9] * 32 + [0.1] * 33, "ts": float(i)})
        insights = eng._distill_experiences(buffer)
        # Variance of [0.9]*32 + [0.1]*33 is well above 0.005 → insights expected
        assert len(insights) >= 1
        for ins in insights:
            assert ins["dim"] == 65
            assert len(ins["felt_tensor"]) == 65
            assert isinstance(ins["felt_tensor"], list)

    def test_mixed_dim_buffer_handles_gracefully(self):
        """Buffer with BOTH 130D and 65D (transition-window) → cross-dim never merges.

        Cosine similarity returns 0.0 on dim mismatch, so snapshots of
        different dims always start a new cluster. No crash, graceful
        degradation — each insight's felt_tensor matches its own dim.
        """
        eng = _mk_engine(threshold=0.5)
        buffer = []
        # First 3 snapshots: 130D
        for i in range(3):
            buffer.append({"full_130dt": [0.9] * 65 + [0.1] * 65, "ts": float(i)})
        # Next 3 snapshots: 65D only (simulating pre-rFP-#1 payload)
        for i in range(3):
            buffer.append({"full_65dt": [0.9] * 32 + [0.1] * 33, "ts": float(i + 3)})

        insights = eng._distill_experiences(buffer)
        # Each insight's felt_tensor length must equal its reported dim
        # (no mixed-dim corruption across the cluster boundary).
        for ins in insights:
            assert len(ins["felt_tensor"]) == ins["dim"]
            assert ins["dim"] in (65, 130)

    def test_dna_none_uses_defaults(self):
        """DreamingEngine(dna=None) → clustering threshold defaults to 0.85."""
        eng = DreamingEngine(dna=None)
        assert eng._cluster_merge_threshold == 0.85
        assert eng._felt_tensor_expected_dim == 130
