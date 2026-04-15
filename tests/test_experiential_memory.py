"""Tests for Experiential Memory (e_mem) — dream-distilled insight store."""
import math
import os
import pytest

from titan_plugin.logic.experiential_memory import ExperientialMemory, _cosine_sim


def _fresh(tmp_path, name, dev_age=100):
    return ExperientialMemory(
        db_path=str(tmp_path / f"{name}.db"),
        developmental_age_fn=lambda: dev_age,
    )


def _make_insight(sig=0.5, tensor=None, epoch_id=1, hormones=None):
    return {
        "significance": sig,
        "felt_tensor": tensor or [0.1] * 10,
        "epoch_id": epoch_id,
        "hormones": hormones or {},
    }


class TestCosineSimHelper:

    def test_identical_vectors(self):
        assert _cosine_sim([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_sim([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_sim([1, 2], [1, 2, 3]) == 0.0


class TestStoreAndRetrieve:

    def test_store_returns_id(self, tmp_path):
        mem = _fresh(tmp_path, "store1")
        row_id = mem.store_insight(_make_insight(sig=0.3), dream_cycle=1)
        assert row_id >= 1

    def test_count_after_store(self, tmp_path):
        mem = _fresh(tmp_path, "store2")
        assert mem.count() == 0
        mem.store_insight(_make_insight(), dream_cycle=1)
        mem.store_insight(_make_insight(), dream_cycle=1)
        assert mem.count() == 2


class TestRecall:

    def test_recall_by_state_returns_similar(self, tmp_path):
        mem = _fresh(tmp_path, "recall1")
        # Store two insights with different tensors
        mem.store_insight(_make_insight(tensor=[1.0, 0.0, 0.0]), dream_cycle=1)
        mem.store_insight(_make_insight(tensor=[0.0, 0.0, 1.0]), dream_cycle=1)

        # Query with state similar to first
        results = mem.recall_by_state([0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["felt_tensor"][0] == pytest.approx(1.0)

    def test_recall_increments_count(self, tmp_path):
        mem = _fresh(tmp_path, "recall2")
        mem.store_insight(_make_insight(tensor=[1.0, 0.0]), dream_cycle=1)

        results = mem.recall_by_state([1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["recall_count"] == 0  # Was 0 before this recall

        # Second recall should show incremented count
        results2 = mem.recall_by_state([1.0, 0.0], top_k=1)
        assert results2[0]["recall_count"] == 1  # Incremented from first recall

    def test_recall_by_recency(self, tmp_path):
        mem = _fresh(tmp_path, "recall3")
        mem.store_insight(_make_insight(sig=0.1), dream_cycle=1)
        mem.store_insight(_make_insight(sig=0.9), dream_cycle=2)

        results = mem.recall_by_recency(limit=1)
        assert len(results) == 1
        assert results[0]["significance"] == pytest.approx(0.9)

    def test_recall_empty_store(self, tmp_path):
        mem = _fresh(tmp_path, "recall4")
        assert mem.recall_by_state([1.0, 0.0], top_k=3) == []
        assert mem.recall_by_recency(limit=5) == []


class TestBookmarking:

    def test_auto_bookmark_high_significance(self, tmp_path):
        mem = _fresh(tmp_path, "bm1")
        mem.store_insight(_make_insight(sig=0.85), dream_cycle=1)
        assert mem.count_bookmarked() == 1

    def test_auto_bookmark_identity_hormones(self, tmp_path):
        mem = _fresh(tmp_path, "bm2")
        hormones = {"INSPIRATION": 0.6, "CREATIVITY": 0.7, "REFLECTION": 0.1}
        mem.store_insight(_make_insight(sig=0.3, hormones=hormones), dream_cycle=1)
        assert mem.count_bookmarked() == 1

    def test_no_auto_bookmark_low_significance(self, tmp_path):
        mem = _fresh(tmp_path, "bm3")
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=1)
        assert mem.count_bookmarked() == 0

    def test_intentional_bookmark(self, tmp_path):
        mem = _fresh(tmp_path, "bm4")
        row_id = mem.store_insight(_make_insight(sig=0.3), dream_cycle=1)
        assert mem.count_bookmarked() == 0

        mem.bookmark_insight(row_id, reason_tensor=[0.5] * 10)
        assert mem.count_bookmarked() == 1


class TestRetention:

    def test_retention_window_grows_with_age(self, tmp_path):
        mem_young = _fresh(tmp_path, "ret1", dev_age=50)
        mem_old = _fresh(tmp_path, "ret2", dev_age=1000)
        assert mem_old.compute_retention_window() > mem_young.compute_retention_window()

    def test_prune_ordinary_dreams(self, tmp_path):
        mem = _fresh(tmp_path, "ret3", dev_age=100)
        # Store old ordinary dream
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=1)
        # Store recent dream
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=9999)
        assert mem.count() == 2

        pruned = mem.prune_stale()
        assert pruned == 1
        assert mem.count() == 1  # Only recent one remains

    def test_prune_keeps_recalled_longer(self, tmp_path):
        mem = _fresh(tmp_path, "ret4", dev_age=100)
        # Store old dream that was recalled
        row_id = mem.store_insight(_make_insight(sig=0.3, tensor=[1.0, 0.0]), dream_cycle=1)
        mem.recall_by_state([1.0, 0.0], top_k=1)  # Increment recall_count

        # Store a very recent dream to set max_cycle high
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=600)

        # Prune: ordinary cutoff = 600 - 480 = 120, recalled cutoff = 600 - 960 = -360
        # Dream at cycle 1 is recalled → uses 2× window → cutoff -360 → NOT pruned
        pruned = mem.prune_stale()
        assert pruned == 0  # Recalled dream survives

    def test_prune_never_removes_bookmarked(self, tmp_path):
        mem = _fresh(tmp_path, "ret5", dev_age=100)
        # Store old bookmarked dream
        mem.store_insight(_make_insight(sig=0.9), dream_cycle=1)  # Auto-bookmarked
        # Store recent dream to set max_cycle high
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=9999)

        pruned = mem.prune_stale()
        # Bookmarked dream should survive, only ordinary at cycle 1 would be pruned
        # but it's bookmarked so it survives
        assert mem.count_bookmarked() == 1


class TestQualityMetrics:

    def test_recall_ratio(self, tmp_path):
        mem = _fresh(tmp_path, "qm1")
        mem.store_insight(_make_insight(sig=0.3, tensor=[1.0, 0.0]), dream_cycle=1)
        mem.store_insight(_make_insight(sig=0.3, tensor=[0.0, 1.0]), dream_cycle=1)
        assert mem.get_recall_ratio() == pytest.approx(0.0)

        mem.recall_by_state([1.0, 0.0], top_k=1)
        assert mem.get_recall_ratio() == pytest.approx(0.5)

    def test_dream_quality(self, tmp_path):
        mem = _fresh(tmp_path, "qm2")
        mem.store_insight(_make_insight(sig=0.3), dream_cycle=1)
        mem.store_insight(_make_insight(sig=0.7), dream_cycle=1)
        assert mem.get_dream_quality(last_n_cycles=5) == pytest.approx(0.5)

    def test_dream_quality_empty(self, tmp_path):
        mem = _fresh(tmp_path, "qm3")
        assert mem.get_dream_quality() == 0.0

    def test_stats_complete(self, tmp_path):
        mem = _fresh(tmp_path, "qm4")
        mem.store_insight(_make_insight(sig=0.9), dream_cycle=1)
        stats = mem.get_stats()
        assert "total" in stats
        assert "bookmarked" in stats
        assert "recall_ratio" in stats
        assert "dream_quality" in stats
        assert "retention_window" in stats
        assert stats["total"] == 1
        assert stats["bookmarked"] == 1  # Auto-bookmarked (sig=0.9)


class TestDefensiveGuards:
    """rFP #3 Phase 1 — defensive guards against scalar felt_tensor corruption."""

    def test_cosine_sim_returns_zero_on_scalar_inputs(self):
        """Non-list input → 0.0, no crash."""
        assert _cosine_sim(0.5, [1.0, 2.0]) == 0.0
        assert _cosine_sim([1.0, 2.0], "not a list") == 0.0
        assert _cosine_sim(None, [1.0]) == 0.0
        assert _cosine_sim([1.0], 0.3656) == 0.0

    def test_store_rejects_non_list_felt_tensor(self, tmp_path, caplog):
        """Scalar felt_tensor → stored as empty list, WARNING logged."""
        import logging
        mem = _fresh(tmp_path, "guard1")
        with caplog.at_level(logging.WARNING, logger="titan_plugin.logic.experiential_memory"):
            row_id = mem.store_insight(
                {"significance": 0.3, "felt_tensor": 0.3656, "epoch_id": 1},
                dream_cycle=1,
            )
        assert row_id >= 1
        assert any("non-list felt_tensor rejected" in r.message for r in caplog.records)

    def test_store_warns_on_unexpected_dim(self, tmp_path, caplog):
        """List of dim != 65 or 130 → stored, but WARNING emitted."""
        import logging
        mem = _fresh(tmp_path, "guard2")
        with caplog.at_level(logging.WARNING, logger="titan_plugin.logic.experiential_memory"):
            mem.store_insight(
                {"significance": 0.3, "felt_tensor": [0.1] * 42, "epoch_id": 1},
                dream_cycle=1,
            )
        assert any("unexpected felt_tensor dim=42" in r.message for r in caplog.records)

    def test_recall_skips_scalar_polluted_legacy_rows(self, tmp_path):
        """Inject scalar row directly (simulating legacy) — recall ignores it silently."""
        mem = _fresh(tmp_path, "guard3")
        with mem._lock:
            mem._conn.execute(
                "INSERT INTO experiential_memory "
                "(significance, felt_tensor, epoch_id, dream_cycle, "
                " bookmarked, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (0.5, "0.3656", 100, 1, 0, 0.0),
            )
            mem._conn.commit()
        mem.store_insight({"significance": 0.4, "felt_tensor": [0.5] * 130}, dream_cycle=1)
        results = mem.recall_by_state([0.5] * 130, top_k=5)
        assert len(results) == 1
        assert isinstance(results[0]["felt_tensor"], list)
        assert len(results[0]["felt_tensor"]) == 130

    def test_store_accepts_130d_list(self, tmp_path):
        """Happy path: 130-float list stores + recalls correctly."""
        mem = _fresh(tmp_path, "guard4")
        mem.store_insight(
            {"significance": 0.5, "felt_tensor": [0.5] * 130}, dream_cycle=1
        )
        results = mem.recall_by_state([0.5] * 130, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0]["felt_tensor"], list)
        assert len(results[0]["felt_tensor"]) == 130
        assert results[0]["similarity"] == pytest.approx(1.0)


class TestE2EDreamCycleRoundtrip:
    """rFP #3 Phase 2 — end-to-end integration tests.

    Would have caught the scalar-tensor bug class BEFORE it shipped, because
    recall_by_state on a scalar-stored row would have crashed the round-trip.
    """

    def test_e2e_dream_cycle_to_recall_roundtrip_130d(self, tmp_path):
        """Full pipeline at 130D: distill → store → recall round-trip."""
        import random
        from titan_plugin.logic.dreaming import DreamingEngine

        random.seed(42)
        eng = DreamingEngine(dna={"cluster_merge_threshold": 0.5})
        eng._distill_threshold = 0.005

        buffer = []
        for i in range(5):
            t = [0.95] * 65 + [0.05] * 65
            t[i] += random.uniform(-0.02, 0.02)
            buffer.append({"full_130dt": t, "ts": float(i)})
        for i in range(5):
            t = [0.05] * 65 + [0.95] * 65
            t[65 + i] += random.uniform(-0.02, 0.02)
            buffer.append({"full_130dt": t, "ts": float(i + 5)})

        insights = eng._distill_experiences(buffer)
        assert len(insights) >= 2, f"Expected ≥2 clusters, got {len(insights)}"
        for ins in insights:
            assert isinstance(ins["felt_tensor"], list)
            assert len(ins["felt_tensor"]) == 130

        mem = _fresh(tmp_path, "e2e1")
        for ins in insights:
            row_id = mem.store_insight(ins, dream_cycle=1)
            assert row_id >= 1

        query = [0.95] * 65 + [0.05] * 65
        results = mem.recall_by_state(query, top_k=3)
        assert len(results) >= 1
        top_sim = results[0]["similarity"]
        assert top_sim > 0.5, f"Expected top recall similarity > 0.5, got {top_sim}"

    def test_e2e_scalar_polluted_row_silently_invisible_to_recall(self, tmp_path):
        """Legacy scalar-polluted rows exist in DB but never appear in recall."""
        import sqlite3
        mem = _fresh(tmp_path, "e2e2")

        with mem._lock:
            mem._conn.execute(
                "INSERT INTO experiential_memory "
                "(significance, felt_tensor, epoch_id, dream_cycle, "
                " bookmarked, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (0.5, "0.3656", 100, 1, 0, 0.0),
            )
            mem._conn.commit()

        mem.store_insight({"felt_tensor": [0.5] * 130, "significance": 0.3},
                          dream_cycle=1)

        conn = sqlite3.connect(mem._db_path)
        total = conn.execute("SELECT COUNT(*) FROM experiential_memory").fetchone()[0]
        conn.close()
        assert total == 2

        results = mem.recall_by_state([0.5] * 130, top_k=10)
        assert len(results) == 1
        assert isinstance(results[0]["felt_tensor"], list)
