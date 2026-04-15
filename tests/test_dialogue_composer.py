"""Tests for DialogueComposer — Phase 5 composed responses."""
import os
import pytest
import tempfile
from titan_plugin.logic.dialogue_composer import DialogueComposer


@pytest.fixture
def composer():
    tmp = tempfile.mktemp(suffix=".db")
    c = DialogueComposer(grammar_db_path=tmp)
    yield c
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
    {"word": "joy", "word_type": "noun", "confidence": 0.8,
     "felt_tensor": [0.5, 0.5] + [0.0] * 128},
]


class TestDialogueComposer:
    def test_compose_response_basic(self, composer):
        result = composer.compose_response(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB)
        # Should return a dict with response key
        assert "response" in result
        assert "composed" in result
        assert "intent" in result

    def test_compose_with_hormone_shifts(self, composer):
        result = composer.compose_response(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            hormone_shifts={"EMPATHY": 0.5})
        assert result["intent"] == "empathize"

    def test_curiosity_shift_asks_question(self, composer):
        result = composer.compose_response(
            felt_state=[0.5] * 130,
            vocabulary=SAMPLE_VOCAB,
            hormone_shifts={"CURIOSITY": 0.5})
        assert result["intent"] == "ask_question"

    def test_fallback_on_empty_vocab(self, composer):
        result = composer.compose_response(
            felt_state=[0.5] * 130,
            vocabulary=[])
        assert result["composed"] is False

    def test_stats_tracking(self, composer):
        composer.compose_response([0.5] * 130, SAMPLE_VOCAB)
        composer.compose_response([0.5] * 130, [])
        stats = composer.get_stats()
        assert stats["total_responses"] == 2

    def test_confidence_threshold(self, composer):
        composer._confidence_threshold = 99.0  # Unreachably high
        result = composer.compose_response([0.5] * 130, SAMPLE_VOCAB)
        assert result["composed"] is False
