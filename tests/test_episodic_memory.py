"""Tests for Episodic Memory — autobiographical life events."""
import pytest
from titan_plugin.logic.episodic_memory import EpisodicMemory


def _fresh(tmp_path, name):
    return EpisodicMemory(db_path=str(tmp_path / f"{name}.db"))


class TestEpisodicMemory:

    def test_record_episode(self, tmp_path):
        mem = _fresh(tmp_path, "ep1")
        row_id = mem.record_episode("word_learned", "Learned 'warm'",
                                     felt_state=[0.5] * 10, significance=0.7)
        assert row_id is not None
        assert row_id >= 1

    def test_significance_gating(self, tmp_path):
        mem = _fresh(tmp_path, "ep2")
        row_id = mem.record_episode("word_learned", "Low significance",
                                     significance=0.1)
        assert row_id is None  # Below threshold
        assert mem.count() == 0

    def test_recall_by_time(self, tmp_path):
        mem = _fresh(tmp_path, "ep3")
        mem.record_episode("word_learned", "Word A", epoch_id=100, significance=0.5)
        mem.record_episode("conversation", "Chat B", epoch_id=200, significance=0.5)
        mem.record_episode("word_learned", "Word C", epoch_id=300, significance=0.5)

        results = mem.recall_by_time(50, 250)
        assert len(results) == 2
        assert results[0]["epoch_id"] == 100
        assert results[1]["epoch_id"] == 200

    def test_recall_by_feeling(self, tmp_path):
        mem = _fresh(tmp_path, "ep4")
        mem.record_episode("word_learned", "Warm feeling",
                          felt_state=[1.0, 0.0, 0.0], significance=0.6)
        mem.record_episode("word_learned", "Cold feeling",
                          felt_state=[0.0, 0.0, 1.0], significance=0.6)

        results = mem.recall_by_feeling([0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert "Warm" in results[0]["description"]

    def test_recall_by_type(self, tmp_path):
        mem = _fresh(tmp_path, "ep5")
        mem.record_episode("word_learned", "Word A", significance=0.5)
        mem.record_episode("conversation", "Chat B", significance=0.5)
        mem.record_episode("word_learned", "Word C", significance=0.5)

        results = mem.recall_by_type("word_learned")
        assert len(results) == 2

    def test_autobiography(self, tmp_path):
        mem = _fresh(tmp_path, "ep6")
        mem.record_episode("word_learned", "Minor", significance=0.4)
        mem.record_episode("great_pulse", "GREAT PULSE!", significance=1.0)
        mem.record_episode("conversation", "Nice chat", significance=0.6)

        auto = mem.get_autobiography(limit=2)
        assert len(auto) == 2
        assert auto[0]["significance"] == pytest.approx(1.0)
        assert auto[0]["event_type"] == "great_pulse"

    def test_count_by_type(self, tmp_path):
        mem = _fresh(tmp_path, "ep7")
        mem.record_episode("word_learned", "A", significance=0.5)
        mem.record_episode("word_learned", "B", significance=0.5)
        mem.record_episode("conversation", "C", significance=0.5)

        by_type = mem.count_by_type()
        assert by_type["word_learned"] == 2
        assert by_type["conversation"] == 1

    def test_stats(self, tmp_path):
        mem = _fresh(tmp_path, "ep8")
        mem.record_episode("word_learned", "Test", significance=0.5)
        stats = mem.get_stats()
        assert stats["total"] == 1
        assert "word_learned" in stats["by_type"]
