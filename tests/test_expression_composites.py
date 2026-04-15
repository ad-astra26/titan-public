"""Tests for ExpressionComposites — SPEAK, ART, MUSIC, SOCIAL."""
import pytest
from titan_plugin.logic.expression_composites import (
    ExpressionComposite, ExpressionManager,
    create_speak, create_art, create_music, create_social,
)


class TestExpressionComposite:
    def test_speak_creation(self):
        speak = create_speak()
        assert speak.name == "SPEAK"
        assert "CREATIVITY" in speak.hormone_weights
        assert "REFLECTION" in speak.hormone_weights
        assert "EMPATHY" in speak.hormone_weights

    def test_evaluate_below_threshold(self):
        speak = create_speak()
        result = speak.evaluate({"CREATIVITY": 0.1, "REFLECTION": 0.1, "EMPATHY": 0.1})
        assert result["should_fire"] is False
        assert result["urge"] < speak.threshold

    def test_evaluate_above_threshold(self):
        speak = create_speak()
        result = speak.evaluate({"CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0})
        assert result["should_fire"] is True
        assert result["urge"] > speak.threshold

    def test_vocabulary_gate(self):
        speak = create_speak()
        # High hormones but low vocabulary → don't fire
        result = speak.evaluate(
            {"CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0},
            vocabulary_confidence=0.1)
        assert result["should_fire"] is False

    def test_cooldown(self):
        speak = create_speak()
        speak.evaluate({"CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0})
        speak.fire()
        # Immediately after fire → cooldown blocks
        result = speak.evaluate({"CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0})
        assert result["should_fire"] is False

    def test_fire_increments_count(self):
        speak = create_speak()
        assert speak._fire_count == 0
        speak.evaluate({"CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0})
        speak.fire()
        assert speak._fire_count == 1

    def test_adapt_threshold(self):
        speak = create_speak()
        initial = speak.threshold
        speak.adapt_threshold(reward=0.8)
        assert speak.threshold < initial  # Positive reward → lower threshold

    def test_dominant_hormone(self):
        speak = create_speak()
        result = speak.evaluate({"CREATIVITY": 0.1, "REFLECTION": 3.0, "EMPATHY": 0.1})
        assert result["dominant_hormone"] == "REFLECTION"

    def test_art_no_vocabulary_needed(self):
        art = create_art()
        result = art.evaluate(
            {"CREATIVITY": 2.0, "INSPIRATION": 2.0, "IMPULSE": 2.0},
            vocabulary_confidence=0.0)  # Zero vocab is fine for art
        assert result["should_fire"] is True

    def test_stats(self):
        speak = create_speak()
        speak.evaluate({"CREATIVITY": 1.0, "REFLECTION": 1.0, "EMPATHY": 1.0})
        stats = speak.get_stats()
        assert stats["name"] == "SPEAK"
        assert stats["total_evaluations"] == 1


class TestExpressionManager:
    def test_register_composites(self):
        mgr = ExpressionManager()
        mgr.register(create_speak())
        mgr.register(create_art())
        assert len(mgr.composites) == 2

    def test_evaluate_all(self):
        mgr = ExpressionManager()
        mgr.register(create_speak())
        mgr.register(create_art())
        # Both should fire with high hormones
        fired = mgr.evaluate_all({
            "CREATIVITY": 2.0, "REFLECTION": 2.0, "EMPATHY": 2.0,
            "INSPIRATION": 2.0, "IMPULSE": 2.0,
        })
        assert len(fired) >= 1  # At least SPEAK should fire

    def test_stats(self):
        mgr = ExpressionManager()
        mgr.register(create_speak())
        stats = mgr.get_stats()
        assert stats["total_composites"] == 1
        assert "SPEAK" in stats["composites"]
