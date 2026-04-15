"""Tests for Word Selector and Composition Engine — felt-state language."""
import pytest

from titan_plugin.logic.word_selector import WordSelector, _cosine_sim
from titan_plugin.logic.composition_engine import CompositionEngine


def _word(name, word_type, confidence=0.5, felt_tensor=None, hormone_pattern=None):
    return {
        "word": name,
        "word_type": word_type,
        "confidence": confidence,
        "felt_tensor": felt_tensor or [0.5] * 10,
        "hormone_pattern": hormone_pattern or {},
    }


SAMPLE_VOCAB = [
    _word("warm", "adjective", 0.8, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    _word("cold", "adjective", 0.7, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    _word("curious", "adjective", 0.9, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          {"CURIOSITY": 0.4}),
    _word("explore", "verb", 0.8, [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          {"CURIOSITY": 0.3}),
    _word("create", "verb", 0.6, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          {"CREATIVITY": 0.4}),
    _word("feel", "verb", 0.9, [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    _word("I", "pronoun", 0.5, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    _word("energy", "noun", 0.7, [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    _word("alive", "adjective", 0.8, [0.6, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    _word("rest", "verb", 0.5, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
]


class TestWordSelector:

    def test_select_by_type(self):
        ws = WordSelector()
        result = ws.select("adjective", [0.5] * 10, SAMPLE_VOCAB)
        assert result is not None
        word, sim, conf = result
        assert word in ("warm", "cold", "curious", "alive")

    def test_select_cosine_similarity(self):
        ws = WordSelector()
        # State similar to "curious" tensor: [0, 0, 1, 0, ...]
        state = [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = ws.select("adjective", state, SAMPLE_VOCAB)
        assert result is not None
        assert result[0] == "curious"

    def test_select_excludes_used(self):
        ws = WordSelector()
        state = [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = ws.select("adjective", state, SAMPLE_VOCAB, exclude={"curious"})
        assert result is not None
        assert result[0] != "curious"

    def test_select_by_hormone(self):
        ws = WordSelector()
        results = ws.select_by_hormone("CURIOSITY", SAMPLE_VOCAB, top_k=3)
        assert len(results) >= 1
        words = [r[0] for r in results]
        assert "curious" in words or "explore" in words

    def test_select_empty_vocab(self):
        ws = WordSelector()
        result = ws.select("adjective", [0.5] * 10, [])
        assert result is None

    def test_select_any_type(self):
        ws = WordSelector()
        result = ws.select_any([0.5] * 10, SAMPLE_VOCAB)
        assert result is not None

    def test_select_respects_min_confidence(self):
        ws = WordSelector()
        low_conf_vocab = [_word("test", "adjective", 0.01)]
        result = ws.select("adjective", [0.5] * 10, low_conf_vocab, min_confidence=0.1)
        assert result is None


class TestCompositionEngine:

    def test_compose_level_1(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=1)
        assert result["level"] == 1
        assert result["sentence"] != ""
        assert len(result["words_used"]) == 1

    def test_compose_level_3(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=3)
        assert result["level"] <= 3
        assert "I" in result["sentence"] or result["level"] <= 2

    def test_compose_level_5(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=5)
        assert result["level"] <= 5
        assert len(result["words_used"]) >= 1

    def test_compose_returns_confidence(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=3)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_compose_empty_vocabulary(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, [], max_level=3)
        assert result["sentence"] == ""
        assert result["level"] == 0

    def test_compose_with_intent(self):
        ce = CompositionEngine()
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB,
                           intent="express_feeling", max_level=5)
        assert result["intent"] == "express_feeling"

    def test_compose_fills_all_slots(self):
        ce = CompositionEngine()
        # With enough vocabulary, all slots should be fillable
        result = ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=3)
        assert "___" not in result["sentence"]

    def test_stats_tracking(self):
        ce = CompositionEngine()
        ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=3)
        ce.compose([0.5] * 10, SAMPLE_VOCAB, max_level=3)
        stats = ce.get_stats()
        assert stats["total_compositions"] == 2

    def test_words_match_felt_state(self):
        ce = CompositionEngine()
        # State that strongly resembles "curious"
        curious_state = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = ce.compose(curious_state, SAMPLE_VOCAB, max_level=1)
        # At level 1, should pick word most similar to state
        assert result["words_used"][0] in ("curious", "explore")
