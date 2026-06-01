"""Tests for ExpressionComposites — SPEAK, ART, MUSIC, SOCIAL."""
import pytest
from titan_hcl.logic.expression_composites import (
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

    def test_hormonal_depletion_eventually_pauses(self):
        """Per design (expression_composites.py:124-127): no fixed cooldown.
        Timing is emergent from hormonal rebuild rate — repeated fires
        deplete hormones until urge drops below threshold (natural pause).

        2026-05-10: replaced obsolete `test_cooldown` (which assumed a
        post-fire timer that no longer exists; failing on titan-v6 baseline
        before this session's pre-D8 audit closure work). The new test
        exercises the actual depletion-driven semantics."""
        speak = create_speak()
        # Start near threshold so depletion measurably moves urge below it.
        hormones = {"CREATIVITY": 1.0, "REFLECTION": 1.0, "EMPATHY": 1.0}
        fires = 0
        for _ in range(20):
            result = speak.evaluate(hormones)
            if not result["should_fire"]:
                break
            # Caller is responsible for applying consumption per the fire()
            # docstring: "consumption dict tells the caller which hormones
            # to deplete and by how much."
            consumption = speak.fire()["consumption"]
            for h, c in consumption.items():
                hormones[h] = max(0.0, hormones[h] - c)
            fires += 1
        # Final evaluate confirms the natural pause kicked in.
        assert speak.evaluate(hormones)["should_fire"] is False
        # Sanity: at least one fire happened and the loop terminated by
        # depletion rather than running out of iterations.
        assert 0 < fires < 20

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

    def test_evaluate_all_exposes_per_hormone_consumption(self):
        """2026-06-01: fired dicts carry the per-hormone consumption so a
        cross-process caller (expression_worker, hormonal_system=None) can
        publish HORMONE_CONSUME and deplete the hormone owner's levels. This
        is the data that restores the severed consumption→refractory loop."""
        from titan_hcl.logic.expression_composites import create_social
        mgr = ExpressionManager()
        mgr.register(create_social())
        fired = mgr.evaluate_all(
            {"EMPATHY": 2.0, "CURIOSITY": 2.0, "IMPULSE": 2.0},
            vocabulary_confidence=1.0)
        assert len(fired) == 1
        consumption = fired[0]["consumption"]
        # SOCIAL weights {EMPATHY:0.5, CURIOSITY:0.3, IMPULSE:0.2} × rate 0.55
        assert set(consumption) == {"EMPATHY", "CURIOSITY", "IMPULSE"}
        assert all(v > 0 for v in consumption.values())

    def test_cross_process_consumption_pauses_social(self):
        """End-to-end of the restored loop: applying each fire's consumption
        dict to a separate hormone store (mimicking hormonal_worker.consume
        over the bus) drives the SOCIAL urge below threshold — proving the
        runaway stops once depletion is actually applied cross-process."""
        from titan_hcl.logic.expression_composites import create_social
        from titan_hcl.logic.hormonal_pressure import HormonalPressure
        social = create_social()
        # Separate hormone store = the hormonal_worker's owned levels.
        store = {n: HormonalPressure(name=n, base_secretion_rate=0.0,
                                     stimulus_sensitivity=1.0, decay_rate=0.0,
                                     fire_threshold=0.5, refractory_strength=0.8,
                                     refractory_decay=0.02)
                 for n in ("EMPATHY", "CURIOSITY", "IMPULSE")}
        for h in store.values():
            h.level = 2.0
        fires = 0
        for _ in range(40):
            levels = {n: h.level for n, h in store.items()}
            if not social.evaluate(levels)["should_fire"]:
                break
            consumption = social.fire()["consumption"]
            for n, amt in consumption.items():          # hormonal_worker side
                store[n].consume(amt)
            fires += 1
        levels = {n: h.level for n, h in store.items()}
        assert social.evaluate(levels)["should_fire"] is False
        assert 0 < fires < 40   # paused by depletion, not by iteration cap

    def test_stats(self):
        mgr = ExpressionManager()
        mgr.register(create_speak())
        stats = mgr.get_stats()
        assert stats["total_composites"] == 1
        assert "SPEAK" in stats["composites"]
