"""Tests for GroundUpEnricher — GROUND_UP enrichment mechanism."""
import pytest
from titan_plugin.logic.ground_up import GroundUpEnricher


class TestGroundUpEnricher:
    def test_init_default_strength(self):
        g = GroundUpEnricher()
        assert g.strength == 0.1
        assert g.damping == 0.95

    def test_nudge_length(self):
        g = GroundUpEnricher()
        nudge = g.compute_nudge([0.1] * 10, [0.5] * 5, [0.5] * 5)
        assert len(nudge["body_nudge"]) == 5
        assert len(nudge["mind_nudge"]) == 5
        assert nudge["total_magnitude"] >= 0.0

    def test_apply_modifies_body(self):
        g = GroundUpEnricher(strength=0.5)
        body = [0.3, 0.3, 0.3, 0.3, 0.3]
        mind = [0.5] * 15
        # Strong positive signal should push body values up
        signal = [0.2] * 5 + [0.0] * 5
        new_body, new_mind = g.apply(body, mind, signal, dt=1.0)
        # Body should have changed
        assert new_body != body

    def test_apply_modifies_mind_willing_only(self):
        g = GroundUpEnricher(strength=0.5)
        body = [0.5] * 5
        mind = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.5, 0.5, 0.5, 0.5, 0.5]  # dims 10-14 = 0.5
        signal = [0.0] * 5 + [0.2] * 5  # Only mind signal
        new_body, new_mind = g.apply(body, mind, signal, dt=1.0)
        # Mind willing [10:15] should change
        assert new_mind[10:15] != mind[10:15]

    def test_apply_leaves_mind_sensory_unchanged(self):
        g = GroundUpEnricher(strength=0.5)
        body = [0.5] * 5
        mind = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.5, 0.5, 0.5, 0.5, 0.5]
        signal = [0.0] * 5 + [0.5] * 5
        new_body, new_mind = g.apply(body, mind, signal, dt=1.0)
        # Sensory [0:5] unchanged
        assert new_mind[0:5] == mind[0:5]

    def test_apply_leaves_mind_cognitive_unchanged(self):
        g = GroundUpEnricher(strength=0.5)
        body = [0.5] * 5
        mind = [0.5] * 15
        signal = [0.0] * 5 + [0.5] * 5
        new_body, new_mind = g.apply(body, mind, signal, dt=1.0)
        # Cognitive [5:10] unchanged
        assert new_mind[5:10] == mind[5:10]

    def test_balanced_state_minimal_nudge(self):
        g = GroundUpEnricher()
        # When grounding signal is zero, nudge should be minimal
        nudge = g.compute_nudge([0.0] * 10, [0.5] * 5, [0.5] * 5)
        assert nudge["total_magnitude"] < 0.01

    def test_damping_reduces_overshoot(self):
        g = GroundUpEnricher(damping=0.95)
        # First: large signal
        n1 = g.compute_nudge([0.5] * 10, [0.5] * 5, [0.5] * 5)
        # Second: zero signal — should decay, not snap to zero
        n2 = g.compute_nudge([0.0] * 10, [0.5] * 5, [0.5] * 5)
        # Nudge should be reduced but not zero (damping preserves some)
        assert n2["total_magnitude"] > 0.0
        assert n2["total_magnitude"] < n1["total_magnitude"]

    def test_stats_complete(self):
        g = GroundUpEnricher()
        g.apply([0.5] * 5, [0.5] * 15, [0.1] * 10, dt=1.0)
        stats = g.get_stats()
        assert stats["total_applications"] == 1
        assert "cumulative_body_delta" in stats
        assert "cumulative_mind_delta" in stats
        assert len(stats["last_body_nudge"]) == 5
        assert len(stats["last_mind_nudge"]) == 5

    def test_output_clamped_to_valid_range(self):
        g = GroundUpEnricher(strength=1.0)
        # Even with extreme signal, output should be [0, 1]
        body = [0.99, 0.99, 0.99, 0.99, 0.99]
        mind = [0.5] * 15
        signal = [1.0] * 10  # Max push
        new_body, new_mind = g.apply(body, mind, signal, dt=1.0)
        for v in new_body:
            assert 0.0 <= v <= 1.0
        for v in new_mind:
            assert 0.0 <= v <= 1.0
