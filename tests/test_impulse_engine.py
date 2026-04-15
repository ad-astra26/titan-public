"""Tests for Step 7.1 — ImpulseEngine."""
import time
import pytest


def _make_centered():
    """All-centered tensor (no deficit)."""
    return [0.5, 0.5, 0.5, 0.5, 0.5]


def _make_deficit(dim: int, value: float):
    """Tensor with a deficit on a specific dimension."""
    t = [0.5, 0.5, 0.5, 0.5, 0.5]
    t[dim] = value
    return t


class TestImpulseEngineThreshold:
    """Threshold detection and adaptive learning."""

    def test_no_impulse_below_threshold(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.3)
        # Small deficit (0.1) — below threshold
        body = _make_deficit(0, 0.4)  # deficit = 0.1
        result = engine.observe(body, _make_centered(), _make_centered())
        assert result is None

    def test_impulse_above_threshold(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.3, cooldown_seconds=0)
        # Large deficit (0.4) — above threshold
        body = _make_deficit(0, 0.1)  # deficit = 0.4
        result = engine.observe(body, _make_centered(), _make_centered())
        assert result is not None
        assert result["posture"] == "rest"  # body[0] maps to rest
        assert result["impulse_id"] == 1
        assert result["urgency"] > 0

    def test_cooldown_prevents_rapid_fire(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=300)
        body = _make_deficit(0, 0.1)  # deficit = 0.4
        # First impulse fires
        r1 = engine.observe(body, _make_centered(), _make_centered())
        assert r1 is not None
        # Second impulse blocked by cooldown
        r2 = engine.observe(body, _make_centered(), _make_centered())
        assert r2 is None

    def test_cooldown_expired_allows_fire(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=0)
        body = _make_deficit(0, 0.1)
        mind = _make_deficit(0, 0.1)  # different posture (research)
        r1 = engine.observe(body, _make_centered(), _make_centered())
        assert r1 is not None
        # Different posture should fire (no cooldown, different posture)
        r2 = engine.observe(_make_centered(), mind, _make_centered())
        assert r2 is not None
        assert r2["impulse_id"] == 2


class TestImpulseEngineOutcomeLearning:
    """Threshold adapts based on action outcomes."""

    def test_success_decreases_threshold(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.3, alpha=0.01, beta=0.015)
        initial = engine.threshold
        # Simulate successful outcome (loss decreased)
        before = {"body": [0.2] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        after = {"body": [0.45] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        outcome = engine.record_outcome(1, before, after)
        assert outcome["success"] is True
        assert engine.threshold < initial

    def test_failure_increases_threshold(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.3, alpha=0.01, beta=0.015)
        initial = engine.threshold
        # Simulate failed outcome (loss increased)
        before = {"body": [0.45] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        after = {"body": [0.2] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        outcome = engine.record_outcome(1, before, after)
        assert outcome["success"] is False
        assert engine.threshold > initial

    def test_threshold_respects_floor(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.11, floor=0.1, alpha=0.05)
        before = {"body": [0.2] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        after = {"body": [0.5] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        engine.record_outcome(1, before, after)
        assert engine.threshold >= 0.1

    def test_threshold_respects_ceil(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.69, ceil=0.7, beta=0.05)
        before = {"body": [0.5] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        after = {"body": [0.1] * 5, "mind": [0.5] * 5, "spirit": [0.5] * 5}
        engine.record_outcome(1, before, after)
        assert engine.threshold <= 0.7


class TestImpulseEnginePostureMapping:
    """Deficit → posture mapping correctness."""

    def test_body_energy_maps_to_rest(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=0)
        body = _make_deficit(0, 0.1)  # interoception deficit
        result = engine.observe(body, _make_centered(), _make_centered())
        assert result is not None
        assert result["posture"] == "rest"

    def test_mind_vision_maps_to_research(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=0)
        mind = _make_deficit(0, 0.1)  # vision deficit
        result = engine.observe(_make_centered(), mind, _make_centered())
        assert result is not None
        assert result["posture"] == "research"

    def test_mind_social_maps_to_socialize(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=0)
        mind = _make_deficit(1, 0.1)  # hearing deficit
        result = engine.observe(_make_centered(), mind, _make_centered())
        assert result is not None
        assert result["posture"] == "socialize"

    def test_mind_mood_maps_to_create(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine(threshold=0.2, cooldown_seconds=0)
        mind = _make_deficit(4, 0.1)  # touch/mood deficit
        result = engine.observe(_make_centered(), mind, _make_centered())
        assert result is not None
        assert result["posture"] == "create"


class TestImpulseEngineStats:
    """Stats and introspection."""

    def test_stats_structure(self):
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        engine = ImpulseEngine()
        stats = engine.get_stats()
        assert "threshold" in stats
        assert "impulse_count" in stats
        assert "cooldown_seconds" in stats
        assert "outcome_count" in stats
        assert "success_rate" in stats
