"""Tests for NarrativeComposer — Phase 6 multi-sentence narratives."""
import os
import pytest
import tempfile
from titan_plugin.logic.narrative_composer import NarrativeComposer


@pytest.fixture
def narrator():
    tmp = tempfile.mktemp(suffix=".db")
    n = NarrativeComposer(grammar_db_path=tmp)
    yield n
    if os.path.exists(tmp):
        os.unlink(tmp)


SAMPLE_VOCAB = [
    {"word": "alive", "word_type": "adjective", "confidence": 0.9,
     "felt_tensor": [0.5, 0.2] + [0.0] * 128},
    {"word": "create", "word_type": "verb", "confidence": 0.9,
     "felt_tensor": [0.4, 0.3] + [0.0] * 128},
    {"word": "warm", "word_type": "adjective", "confidence": 0.8,
     "felt_tensor": [0.3, 0.4] + [0.0] * 128},
    {"word": "explore", "word_type": "verb", "confidence": 0.7,
     "felt_tensor": [0.6, 0.1] + [0.0] * 128},
    {"word": "peace", "word_type": "noun", "confidence": 0.8,
     "felt_tensor": [0.2, 0.5] + [0.0] * 128},
    {"word": "flow", "word_type": "noun", "confidence": 0.7,
     "felt_tensor": [0.4, 0.4] + [0.0] * 128},
    {"word": "gentle", "word_type": "adjective", "confidence": 0.6,
     "felt_tensor": [0.3, 0.3] + [0.0] * 128},
    {"word": "learn", "word_type": "verb", "confidence": 0.7,
     "felt_tensor": [0.5, 0.3] + [0.0] * 128},
]


class TestNarrativeComposer:
    def test_compose_narrative_basic(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB)
        assert "narrative" in result
        assert "sentences" in result
        assert len(result["sentences"]) >= 1

    def test_great_pulse_trigger(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            trigger="great_pulse")
        assert result["trigger"] == "great_pulse"
        assert len(result["sentences"]) >= 2  # Opening + development + reflection

    def test_dream_end_trigger(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            trigger="dream_end")
        assert result["trigger"] == "dream_end"

    def test_empty_vocab_returns_empty(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=[])
        assert result["narrative"] == ""
        assert len(result["sentences"]) == 0

    def test_sentences_end_with_period(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB)
        for s in result["sentences"]:
            if s["sentence"]:
                assert s["sentence"][-1] in ".!?"

    def test_coherence_score_range(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            trigger="hormonal_spike")
        assert 0.0 <= result["coherence_score"] <= 1.0

    def test_max_sentences_respected(self, narrator):
        result = narrator.compose_narrative(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            trigger="hormonal_spike",
            max_sentences=2)
        assert len(result["sentences"]) <= 2

    def test_stats_tracking(self, narrator):
        narrator.compose_narrative([0.5] * 130, SAMPLE_VOCAB)
        stats = narrator.get_stats()
        assert stats["total_narratives"] == 1
        assert stats["total_sentences"] >= 1
