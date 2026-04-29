"""Tests for titan_plugin.logic.meta_teacher_memory — Phase B of
rFP_meta_teacher_v2_content_awareness_memory.md.

Covers:
  - canonical_topic_key — outer vs inner-only chain topic_keys
  - TeacherMemory.load / add_critique / retrieve_similar
  - importance_weight formula corners (recency, adoption, quality_delta, SNP)
  - still_needs_push detection (count + quality_delta epsilon)
  - record_adoption retroactive update
  - Hot deque eviction respects memory_buffer_hot_size
  - Cold-tier journal persistence + round-trip load
  - archive_inactive moves 90d+ idle entries
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from titan_plugin.logic.meta_teacher_memory import (
    TeacherMemory,
    canonical_topic_key,
)


def _cfg(**overrides):
    base = {
        "teaching_memory_enabled": True,
        "memory_buffer_hot_size": 50,
        "retrieval_top_k": 3,
        "retrieval_similarity_threshold": 0.2,   # lower for test matches
        "still_needs_push_critique_threshold": 3,
        "still_needs_push_quality_delta_epsilon": 0.05,
        "cold_tier_archival_days": 90,
    }
    base.update(overrides)
    return base


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mem(tmp_data_dir):
    m = TeacherMemory(_cfg(), data_dir=tmp_data_dir)
    m.load()
    return m


# ── canonical_topic_key ────────────────────────────────────────────────────

class TestCanonicalTopicKey:
    def test_topic_plus_person(self):
        tk = canonical_topic_key(
            {"primary_person": "@jkacrpto", "current_topic": "Sovereignty"})
        assert tk == "sovereignty|person=@jkacrpto"

    def test_topic_only(self):
        tk = canonical_topic_key({"current_topic": "AI Development"})
        assert tk == "ai development"

    def test_person_only(self):
        tk = canonical_topic_key({"primary_person": "@abc"})
        assert tk == "person=@abc"

    def test_handle_without_at_sign(self):
        tk = canonical_topic_key({"primary_person": "jkacrpto"})
        assert tk == "person=@jkacrpto"

    def test_no_outer_inner_only_fallback(self):
        tk = canonical_topic_key(
            None, primitives_used=["FORMULATE"], domain="meta")
        assert tk == "inner::FORMULATE::meta"

    def test_ultimate_fallback_when_no_prims(self):
        tk = canonical_topic_key(None, primitives_used=[], domain="")
        assert tk == "inner::unknown::general"


# ── add_critique / retrieve_similar ────────────────────────────────────────

class TestAddCritique:
    def test_disabled_add_returns_none(self, tmp_data_dir):
        m = TeacherMemory(_cfg(teaching_memory_enabled=False),
                           data_dir=tmp_data_dir)
        assert m.add_critique({"ts": 1, "quality_score": 0.5}) is None

    def test_add_populates_hot_and_cold(self, mem):
        entry = {
            "ts": time.time(), "chain_id": 1, "domain": "meta",
            "quality_score": 0.4, "critique_text": "shallow", "suggested_primitives": [],
        }
        tk = mem.add_critique(entry, outer_summary={"current_topic": "sovereignty"})
        assert tk == "sovereignty"
        assert len(mem._hot) == 1
        assert "sovereignty" in mem._cold
        cold = mem._cold["sovereignty"]
        assert cold["critique_count"] == 1
        assert len(cold["quality_trajectory"]) == 1
        assert cold["quality_trajectory"][0]["chain_quality"] == 0.4
        assert cold["still_needs_push"] is False  # only 1 critique

    def test_hot_deque_evicts_beyond_max(self, tmp_data_dir):
        m = TeacherMemory(_cfg(memory_buffer_hot_size=5), data_dir=tmp_data_dir)
        m.load()
        for i in range(10):
            m.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": 0.5},
                outer_summary={"current_topic": f"t{i}"})
        assert len(m._hot) == 5
        # Oldest evicted
        topic_keys = [h["topic_key"] for h in m._hot]
        assert topic_keys[0] == "t5"

    def test_still_needs_push_fires_after_threshold(self, mem):
        # 3 critiques with quality_delta = 0 → still_needs_push True
        for i in range(3):
            mem.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": 0.5},
                outer_summary={"current_topic": "stuck_topic"})
        assert mem._cold["stuck_topic"]["still_needs_push"] is True

    def test_still_needs_push_not_when_quality_improving(self, mem):
        # 3 critiques with rising quality
        for i, q in enumerate((0.3, 0.5, 0.8)):
            mem.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": q},
                outer_summary={"current_topic": "improving"})
        cold = mem._cold["improving"]
        assert cold["still_needs_push"] is False
        assert cold["quality_delta"] > 0

    def test_persist_and_reload(self, tmp_data_dir):
        m1 = TeacherMemory(_cfg(), data_dir=tmp_data_dir)
        m1.load()
        for i in range(3):
            m1.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": 0.5},
                outer_summary={"current_topic": "persist_me"})
        # Fresh instance — reload from disk
        m2 = TeacherMemory(_cfg(), data_dir=tmp_data_dir)
        m2.load()
        assert "persist_me" in m2._cold
        assert m2._cold["persist_me"]["critique_count"] == 3


class TestRetrieveSimilar:
    def test_disabled_returns_empty(self, tmp_data_dir):
        m = TeacherMemory(_cfg(teaching_memory_enabled=False),
                           data_dir=tmp_data_dir)
        assert m.retrieve_similar("anything", {}) == []

    def test_cold_exact_match_even_without_embedding(self, mem):
        # Force embedding unavailable to prove cold-exact path works alone.
        mem._embed_index._available = False
        mem._embed_index._init_attempted = True
        mem.add_critique(
            {"ts": time.time(), "chain_id": 1, "quality_score": 0.5},
            outer_summary={"current_topic": "x"})
        hits = mem.retrieve_similar("x", {"current_topic": "x"})
        assert len(hits) == 1
        assert hits[0]["source"] == "cold"
        assert hits[0]["topic_key"] == "x"
        assert hits[0]["similarity"] == 1.0

    def test_returns_empty_on_unknown_topic(self, mem):
        # Even with embedding off, unknown topic_key → nothing.
        mem._embed_index._available = False
        mem._embed_index._init_attempted = True
        mem.add_critique(
            {"ts": time.time(), "chain_id": 1, "quality_score": 0.5},
            outer_summary={"current_topic": "a"})
        hits = mem.retrieve_similar("z", {"current_topic": "z"})
        assert hits == []


class TestImportanceWeight:
    def test_baseline_no_cold(self, mem):
        assert mem._importance_weight({}) == 1.0

    def test_recent_high_adoption_quality_still_needs_push(self, mem):
        cold = {
            "last_seen": time.time(),
            "adoption_trajectory": [
                {"ts": 1, "adopted_bool": True},
                {"ts": 2, "adopted_bool": True},
                {"ts": 3, "adopted_bool": True},
            ],
            "quality_delta": 0.2,
            "still_needs_push": True,
        }
        w = mem._importance_weight(cold)
        # 1.0 + 0.3*~1.0 + 0.5*1.0 + 0.3*1 + 0.4*1 = ~2.5
        assert 2.3 <= w <= 3.0

    def test_clamped_to_0_5_floor(self, mem):
        cold = {
            "last_seen": time.time() - 365 * 86400,  # very old → recency_boost=0
            "adoption_trajectory": [{"adopted_bool": False}] * 5,
            "quality_delta": -0.5,
            "still_needs_push": False,
        }
        w = mem._importance_weight(cold)
        # 1.0 + 0 + 0 + (-0.3) + 0 = 0.7. No adoptions → rate=0.
        assert w >= 0.5
        assert w <= 0.9

    def test_clamped_to_3_0_ceiling(self, mem):
        # Artificial: pretend the formula adds more than 3 — it can't, but
        # the clamp still holds for edge floats.
        cold = {
            "last_seen": time.time(),
            "adoption_trajectory": [{"adopted_bool": True}] * 10,
            "quality_delta": 1.0,
            "still_needs_push": True,
        }
        assert mem._importance_weight(cold) <= 3.0


class TestRecordAdoption:
    def test_updates_existing_cold_entry(self, mem):
        mem.add_critique(
            {"ts": time.time(), "chain_id": 1, "quality_score": 0.5},
            outer_summary={"current_topic": "adopt_test"})
        ok = mem.record_adoption("adopt_test", adopted=True,
                                  suggested_primitives=["INTROSPECT"])
        assert ok is True
        cold = mem._cold["adopt_test"]
        assert len(cold["adoption_trajectory"]) == 1
        assert cold["adoption_trajectory"][0]["adopted_bool"] is True
        assert cold["adoption_trajectory"][0]["suggested_list"] == ["INTROSPECT"]

    def test_nonexistent_topic_returns_false(self, mem):
        assert mem.record_adoption("never_seen", adopted=True) is False

    def test_disabled_memory_returns_false(self, tmp_data_dir):
        m = TeacherMemory(_cfg(teaching_memory_enabled=False),
                           data_dir=tmp_data_dir)
        assert m.record_adoption("x", adopted=True) is False


class TestStillNeedsPushList:
    def test_ordered_by_critique_count_desc(self, mem):
        # Three topics, only 2 stuck
        for i, (topic, q) in enumerate([
            ("stuck_a", 0.5), ("stuck_a", 0.5), ("stuck_a", 0.5), ("stuck_a", 0.5),
            ("stuck_b", 0.5), ("stuck_b", 0.5), ("stuck_b", 0.5),
            ("rising", 0.3), ("rising", 0.6), ("rising", 0.9),
        ]):
            mem.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": q},
                outer_summary={"current_topic": topic})
        snp = mem.still_needs_push_list()
        # Order: stuck_a (4 critiques), stuck_b (3). rising excluded.
        assert [r["topic_key"] for r in snp] == ["stuck_a", "stuck_b"]
        assert snp[0]["critique_count"] == 4
        assert snp[1]["critique_count"] == 3

    def test_hash_changes_when_list_changes(self, mem):
        h1 = mem.still_needs_push_hash()
        for i in range(3):
            mem.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": 0.5},
                outer_summary={"current_topic": "stuck_c"})
        h2 = mem.still_needs_push_hash()
        assert h1 != h2

    def test_hash_stable_across_calls(self, mem):
        for i in range(3):
            mem.add_critique(
                {"ts": time.time() + i, "chain_id": i, "quality_score": 0.5},
                outer_summary={"current_topic": "stable_topic"})
        h1 = mem.still_needs_push_hash()
        h2 = mem.still_needs_push_hash()
        assert h1 == h2


class TestArchival:
    def test_archive_moves_old_entries_out(self, tmp_data_dir):
        m = TeacherMemory(_cfg(cold_tier_archival_days=30),
                          data_dir=tmp_data_dir)
        m.load()
        very_old_ts = time.time() - 365 * 86400  # 1 year ago
        fresh_ts = time.time()
        # Directly seed cold tier with one very-old, one fresh
        m._cold["old_topic"] = {
            "topic_key": "old_topic", "first_seen": very_old_ts,
            "last_seen": very_old_ts, "critique_count": 5,
            "quality_trajectory": [{"ts": very_old_ts, "chain_quality": 0.5}],
            "adoption_trajectory": [], "quality_delta": 0.0,
            "still_needs_push": False, "last_voice_applied": 2,
            "summary_cache": "x",
        }
        m._cold["fresh_topic"] = {
            "topic_key": "fresh_topic", "first_seen": fresh_ts,
            "last_seen": fresh_ts, "critique_count": 1,
            "quality_trajectory": [{"ts": fresh_ts, "chain_quality": 0.9}],
            "adoption_trajectory": [], "quality_delta": 0.0,
            "still_needs_push": False, "last_voice_applied": 2,
            "summary_cache": "y",
        }
        n = m.archive_inactive(now=fresh_ts)
        assert n == 1
        assert "old_topic" not in m._cold
        assert "fresh_topic" in m._cold
        # Archive file exists
        import gzip
        archive = os.path.join(
            tmp_data_dir, "meta_teacher", "teaching_journal.archive.jsonl.gz")
        assert os.path.exists(archive)
        with gzip.open(archive, "rt") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 1
        assert rows[0]["topic_key"] == "old_topic"


class TestSnapshot:
    def test_snapshot_shape(self, mem):
        mem.add_critique(
            {"ts": time.time(), "chain_id": 1, "quality_score": 0.5},
            outer_summary={"current_topic": "s"})
        s = mem.snapshot()
        assert s["enabled"] is True
        assert s["hot_tier_size"] == 1
        assert s["cold_tier_topics"] == 1
        assert s["critiques_absorbed"] == 1
        assert "retrieval_hit_rate" in s
        assert "similarity_threshold" in s


class TestJournalRobustness:
    def test_corrupt_line_skipped_on_load(self, tmp_data_dir):
        mt_dir = os.path.join(tmp_data_dir, "meta_teacher")
        os.makedirs(mt_dir, exist_ok=True)
        path = os.path.join(mt_dir, "teaching_journal.jsonl")
        with open(path, "w") as f:
            f.write('{"topic_key": "good", "critique_count": 1, "last_seen": 1, '
                    '"quality_trajectory": [], "adoption_trajectory": [], '
                    '"quality_delta": 0.0, "still_needs_push": false}\n')
            f.write('{not valid json\n')
            f.write('{"topic_key": "also_good", "critique_count": 2, "last_seen": 2, '
                    '"quality_trajectory": [], "adoption_trajectory": [], '
                    '"quality_delta": 0.0, "still_needs_push": false}\n')
        m = TeacherMemory(_cfg(), data_dir=tmp_data_dir)
        m.load()
        assert "good" in m._cold
        assert "also_good" in m._cold
        assert len(m._cold) == 2

    def test_load_is_idempotent(self, mem):
        mem.add_critique(
            {"ts": time.time(), "chain_id": 1, "quality_score": 0.5},
            outer_summary={"current_topic": "idem"})
        count_before = len(mem._cold)
        mem.load()  # No-op second call
        assert len(mem._cold) == count_before
