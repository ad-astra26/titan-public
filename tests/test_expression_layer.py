"""Tests for the Expression Translation Layer (Phase E)."""
import pytest
from titan_plugin.logic.expression_translator import (
    ExpressionTranslator, FeedbackRouter, MIN_CONFIDENCE,
)


@pytest.fixture
def translator():
    return ExpressionTranslator(all_helpers=[
        "web_search", "art_generate", "audio_generate",
        "social_post", "infra_inspect", "code_knowledge", "coding_sandbox",
    ])


class TestExpressionTranslator:
    def test_translate_returns_none_low_confidence(self, translator):
        """Without enough observations, should return None (LLM fallback)."""
        result = translator.translate(
            program="CURIOSITY", intensity=1.0, posture="explore",
            available_helpers=["web_search", "art_generate"])
        assert result is None  # No observations yet

    def test_translate_returns_helper_after_learning(self, translator):
        """After enough observations, should return learned helper."""
        # Record enough outcomes for web_search
        for _ in range(MIN_CONFIDENCE + 1):
            translator.record_outcome("CURIOSITY", "web_search", score=0.9)

        result = translator.translate(
            program="CURIOSITY", intensity=1.0, posture="explore",
            available_helpers=["web_search", "art_generate"])
        assert result is not None
        assert result["helper"] == "web_search"
        assert result["confidence"] > 0

    def test_high_score_preferred(self, translator):
        """Higher-scoring helper should be preferred."""
        for _ in range(5):
            translator.record_outcome("CREATIVITY", "art_generate", score=0.9)
            translator.record_outcome("CREATIVITY", "audio_generate", score=0.4)

        result = translator.translate(
            program="CREATIVITY", intensity=1.0, posture="create",
            available_helpers=["art_generate", "audio_generate"])
        assert result["helper"] == "art_generate"

    def test_available_helpers_filtered(self, translator):
        """Should only select from available helpers."""
        for _ in range(5):
            translator.record_outcome("CURIOSITY", "web_search", score=0.9)

        result = translator.translate(
            program="CURIOSITY", intensity=1.0, posture="explore",
            available_helpers=["art_generate"])  # web_search NOT available
        # web_search is learned but not available — should fall back
        assert result is None or result["helper"] != "web_search"

    def test_record_outcome_updates_score(self, translator):
        """Outcomes should move scores toward actual quality."""
        translator.record_outcome("FOCUS", "infra_inspect", score=0.9)
        s1 = translator._scores["FOCUS"]["infra_inspect"]

        translator.record_outcome("FOCUS", "infra_inspect", score=0.9)
        s2 = translator._scores["FOCUS"]["infra_inspect"]

        assert s2 > s1  # Score should increase toward 0.9

    def test_sovereignty_tracking(self, translator):
        assert translator.sovereignty_ratio == 0.0

        translator.record_action_type(was_learned=True)
        translator.record_action_type(was_learned=True)
        translator.record_action_type(was_learned=False)

        assert abs(translator.sovereignty_ratio - 2/3) < 0.01

    def test_stats_include_sovereignty(self, translator):
        translator.record_action_type(was_learned=True)
        stats = translator.get_stats()
        assert "sovereignty_ratio" in stats
        assert "learned_actions" in stats
        assert stats["learned_actions"] == 1

    def test_save_load_roundtrip(self, translator, tmp_path):
        translator.record_outcome("CURIOSITY", "web_search", score=0.85)
        translator.record_action_type(was_learned=True)
        path = str(tmp_path / "expression.json")
        translator.save(path)

        t2 = ExpressionTranslator()
        t2.load(path)
        assert t2._scores["CURIOSITY"]["web_search"] == \
               translator._scores["CURIOSITY"]["web_search"]
        assert t2._learned_actions == 1


class TestFeedbackRouter:
    def test_positive_outcome_reduces_pressure(self):
        from titan_plugin.logic.hormonal_pressure import HormonalSystem
        hs = HormonalSystem(["CREATIVITY"])
        hs.get_hormone("CREATIVITY").level = 1.0
        hs.get_hormone("CREATIVITY").refractory = 0.0
        translator = ExpressionTranslator()
        router = FeedbackRouter(hs, translator)

        router.route({
            "triggering_program": "CREATIVITY",
            "helper": "art_generate",
            "score": 0.85,
            "success": True,
        })
        # Pressure should drop significantly
        assert hs.get_hormone("CREATIVITY").level < 0.5

    def test_negative_outcome_partial_relief(self):
        from titan_plugin.logic.hormonal_pressure import HormonalSystem
        hs = HormonalSystem(["CURIOSITY"])
        hs.get_hormone("CURIOSITY").level = 1.0
        hs.get_hormone("CURIOSITY").refractory = 0.0
        translator = ExpressionTranslator()
        router = FeedbackRouter(hs, translator)

        router.route({
            "triggering_program": "CURIOSITY",
            "helper": "web_search",
            "score": 0.2,
            "success": False,
        })
        # Partial relief only
        assert 0.5 < hs.get_hormone("CURIOSITY").level < 1.0

    def test_translator_learns_from_feedback(self):
        from titan_plugin.logic.hormonal_pressure import HormonalSystem
        hs = HormonalSystem(["CREATIVITY"])
        translator = ExpressionTranslator()
        router = FeedbackRouter(hs, translator)

        old_score = translator._scores.get("CREATIVITY", {}).get("art_generate", 0.5)
        router.route({
            "triggering_program": "CREATIVITY",
            "helper": "art_generate",
            "score": 0.95,
            "success": True,
        })
        new_score = translator._scores["CREATIVITY"]["art_generate"]
        assert new_score > old_score

    def test_no_crash_on_missing_program(self):
        from titan_plugin.logic.hormonal_pressure import HormonalSystem
        hs = HormonalSystem(["FOCUS"])
        translator = ExpressionTranslator()
        router = FeedbackRouter(hs, translator)
        # Should not crash with unknown program
        router.route({"triggering_program": "UNKNOWN", "helper": "x",
                       "score": 0.5, "success": True})
