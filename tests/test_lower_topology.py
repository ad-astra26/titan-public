"""Tests for LowerTopology — 10DT grounding space topology."""
import pytest
from titan_plugin.logic.lower_topology import LowerTopology, _cosine_sim, _l2_norm


class TestLowerTopology:
    def test_init_inner_variant(self):
        lt = LowerTopology(variant="inner")
        assert lt.variant == "inner"
        assert lt.grounding_strength == 0.1

    def test_init_outer_variant(self):
        lt = LowerTopology(variant="outer")
        assert lt.variant == "outer"

    def test_compute_returns_10dt(self):
        lt = LowerTopology(variant="inner")
        body = [0.5, 0.4, 0.6, 0.3, 0.7]
        mind_willing = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = lt.compute(body, mind_willing)
        assert len(result["topology_10d"]) == 10
        assert result["topology_10d"] == body + mind_willing

    def test_observables_keys(self):
        lt = LowerTopology(variant="inner")
        lt.compute([0.5] * 5, [0.5] * 5)
        obs = lt.get_observables()
        assert "coherence" in obs
        assert "magnitude" in obs
        assert "velocity" in obs
        assert "direction" in obs
        assert "polarity" in obs

    def test_grounding_signal_length(self):
        lt = LowerTopology(variant="inner")
        lt.compute([0.5] * 5, [0.5] * 5)
        signal = lt.get_grounding_signal()
        assert len(signal) == 10

    def test_coherence_range(self):
        lt = LowerTopology(variant="inner")
        lt.compute([0.5] * 5, [0.5] * 5)
        obs = lt.get_observables()
        assert -1.0 <= obs["coherence"] <= 1.0

    def test_velocity_tracking(self):
        lt = LowerTopology(variant="inner")
        # First compute: no velocity (no history)
        lt.compute([0.3] * 5, [0.3] * 5)
        obs1 = lt.get_observables()
        assert obs1["velocity"] == 0.0
        # Second compute: velocity should be non-zero if state changes
        lt.compute([0.7] * 5, [0.7] * 5)
        obs2 = lt.get_observables()
        assert obs2["velocity"] != 0.0

    def test_ground_center_computation(self):
        lt = LowerTopology(variant="inner")
        whole = [0.5, 0.0, 0.5, 0.5, 0.5, 3, 0.1, 0.5, 0.5, 0.5]
        result = lt.compute([0.5] * 5, [0.5] * 5, whole_10d=whole)
        # Should compute without error
        assert result["topology_10d"] is not None

    def test_balanced_state_high_coherence(self):
        lt = LowerTopology(variant="inner")
        # Balanced state (0.5 everywhere) should have high coherence with reference
        lt.compute([0.5] * 5, [0.5] * 5)
        obs = lt.get_observables()
        assert obs["coherence"] > 0.9  # Very close to reference

    def test_extreme_state_low_coherence(self):
        lt = LowerTopology(variant="inner")
        # Extreme state should have lower coherence
        lt.compute([1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0])
        obs = lt.get_observables()
        assert obs["coherence"] < 0.9

    def test_history_rolling_window(self):
        lt = LowerTopology(variant="inner")
        for i in range(30):
            lt.compute([0.5] * 5, [0.5] * 5)
        assert len(lt._history) == 20  # HISTORY_SIZE cap

    def test_stats_complete(self):
        lt = LowerTopology(variant="outer")
        lt.compute([0.5] * 5, [0.5] * 5)
        stats = lt.get_stats()
        assert stats["variant"] == "outer"
        assert stats["compute_count"] == 1
        assert len(stats["topology_10d"]) == 10
        assert "observables" in stats
        assert "grounding_signal_magnitude" in stats


class TestHelpers:
    def test_cosine_sim_identical(self):
        assert abs(_cosine_sim([1, 0], [1, 0]) - 1.0) < 1e-6

    def test_cosine_sim_orthogonal(self):
        assert abs(_cosine_sim([1, 0], [0, 1])) < 1e-6

    def test_l2_norm(self):
        assert abs(_l2_norm([3.0, 4.0]) - 5.0) < 1e-6
