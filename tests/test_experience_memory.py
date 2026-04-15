"""Tests for Experience Memory (ex_mem) — action outcome learning."""
import pytest
from titan_plugin.logic.experience_memory import ExperienceMemory


def _fresh(tmp_path, name):
    return ExperienceMemory(db_path=str(tmp_path / f"{name}.db"))


class TestRecordAndRetrieve:

    def test_record_returns_id(self, tmp_path):
        mem = _fresh(tmp_path, "rec1")
        row_id = mem.record_experience("art_generate", outcome_score=0.8, success=True)
        assert row_id >= 1

    def test_count_after_records(self, tmp_path):
        mem = _fresh(tmp_path, "rec2")
        mem.record_experience("art_generate")
        mem.record_experience("web_search")
        assert mem.count() == 2

    def test_similar_count_increments(self, tmp_path):
        mem = _fresh(tmp_path, "rec3")
        mem.record_experience("art_generate")
        mem.record_experience("art_generate")
        mem.record_experience("art_generate")
        assert mem.get_experience_count("art_generate") == 3


class TestRecall:

    def test_recall_similar_by_task(self, tmp_path):
        mem = _fresh(tmp_path, "rcl1")
        mem.record_experience("art_generate", outcome_score=0.8, success=True)
        mem.record_experience("web_search", outcome_score=0.5)

        results = mem.recall_similar("art_generate")
        assert len(results) == 1
        assert results[0]["task_type"] == "art_generate"

    def test_recall_similar_cosine(self, tmp_path):
        mem = _fresh(tmp_path, "rcl2")
        mem.record_experience("art_generate",
                              inner_before=[1.0, 0.0, 0.0],
                              outcome_score=0.9, success=True)
        mem.record_experience("art_generate",
                              inner_before=[0.0, 0.0, 1.0],
                              outcome_score=0.3, success=False)

        results = mem.recall_similar("art_generate",
                                     current_inner=[0.9, 0.1, 0.0],
                                     top_k=2)
        assert len(results) == 2
        # First result should be more similar to [1,0,0]
        assert results[0]["outcome_score"] == pytest.approx(0.9)

    def test_recall_empty(self, tmp_path):
        mem = _fresh(tmp_path, "rcl3")
        assert mem.recall_similar("nonexistent") == []


class TestSuccessRate:

    def test_success_rate(self, tmp_path):
        mem = _fresh(tmp_path, "sr1")
        mem.record_experience("art_generate", success=True)
        mem.record_experience("art_generate", success=True)
        mem.record_experience("art_generate", success=False)
        assert mem.get_success_rate("art_generate") == pytest.approx(2/3, abs=0.01)

    def test_success_rate_empty(self, tmp_path):
        mem = _fresh(tmp_path, "sr2")
        assert mem.get_success_rate("art_generate") == 0.0


class TestBestConditions:

    def test_best_conditions(self, tmp_path):
        mem = _fresh(tmp_path, "bc1")
        mem.record_experience("art_generate",
                              inner_before=[0.1, 0.2, 0.3],
                              outcome_score=0.3)
        mem.record_experience("art_generate",
                              inner_before=[0.9, 0.8, 0.7],
                              outcome_score=0.95)

        best = mem.get_best_conditions("art_generate")
        assert best is not None
        assert best["score"] == pytest.approx(0.95)
        assert best["inner_state"][0] == pytest.approx(0.9)

    def test_best_conditions_empty(self, tmp_path):
        mem = _fresh(tmp_path, "bc2")
        assert mem.get_best_conditions("nonexistent") is None


class TestHormonalDelta:

    def test_hormonal_delta_stored(self, tmp_path):
        mem = _fresh(tmp_path, "hd1")
        mem.record_experience("art_generate",
                              hormonal_delta={"INSPIRATION": 0.5, "CREATIVITY": -0.3},
                              outcome_score=0.8, success=True)
        results = mem.recall_similar("art_generate")
        assert results[0]["hormonal_delta"]["INSPIRATION"] == pytest.approx(0.5)

    def test_intent_hormones_stored(self, tmp_path):
        mem = _fresh(tmp_path, "hd2")
        mem.record_experience("art_generate",
                              intent_hormones={"CREATIVITY": 1.2, "IMPULSE": 0.8})
        results = mem.recall_similar("art_generate")
        assert results[0]["intent_hormones"]["CREATIVITY"] == pytest.approx(1.2)


class TestStats:

    def test_stats_by_type(self, tmp_path):
        mem = _fresh(tmp_path, "st1")
        mem.record_experience("art_generate", outcome_score=0.8, success=True)
        mem.record_experience("art_generate", outcome_score=0.6, success=False)
        mem.record_experience("web_search", outcome_score=0.9, success=True)

        stats = mem.get_stats()
        assert stats["total"] == 3
        assert "art_generate" in stats["by_type"]
        assert stats["by_type"]["art_generate"]["count"] == 2
        assert stats["by_type"]["web_search"]["success_rate"] == pytest.approx(1.0)
