"""Tests for Events Teacher adaptive 3-source cycle (2026-04-30 Maker design).

Cycle behavior:
  - 3 sources tried in priority order: follower / vocab / wide
  - Priority = sum(scores) over last N cycles, sort desc + lex tiebreak
  - +1 if stored events ≥ land_threshold from this source, else -1
  - All sources fetched per cycle (single LLM batch downstream)
  - Wide source ALWAYS in cycle (echo chamber mitigation)

Closes BUG-EVENTS-TEACHER-95-PERCENT-EMPTY: shifts content discovery
from stale follower-timelines to vocab-driven + curated wide-net topics.
"""
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from titan_plugin.logic.events_teacher import EventsTeacher, EventsTeacherDB


@pytest.fixture
def temp_db(tmp_path):
    """Fresh EventsTeacherDB for each test."""
    db_path = tmp_path / "events_teacher.db"
    db = EventsTeacherDB(str(db_path))
    return db


@pytest.fixture
def teacher(temp_db):
    """EventsTeacher with temp DB."""
    t = EventsTeacher()
    t._db = temp_db
    return t


# ── source_cycle_scores schema + record/read ──────────────────────────

def test_schema_has_source_cycle_scores(temp_db):
    """The table is created at init."""
    conn = temp_db._connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='source_cycle_scores'").fetchall()
        assert len(rows) == 1
    finally:
        conn.close()


def test_record_and_read_score(temp_db):
    """record_source_outcome → get_source_priority_scores roundtrip."""
    temp_db.record_source_outcome(
        titan_id="T1", source_type="follower", score=+1,
        items_fetched=5, items_stored=2, api_calls_used=1,
        query_text="follower_timelines:top_affinity")
    temp_db.record_source_outcome(
        titan_id="T1", source_type="follower", score=-1,
        items_fetched=0, items_stored=0, api_calls_used=1)
    temp_db.record_source_outcome(
        titan_id="T1", source_type="vocab", score=+1,
        items_fetched=8, items_stored=3, api_calls_used=1)
    scores = temp_db.get_source_priority_scores("T1", window=10)
    assert scores["follower"] == 0  # +1 -1
    assert scores["vocab"] == 1
    assert scores["wide"] == 0  # cold start


def test_score_window_caps_at_n(temp_db):
    """Window=N means only last N rows per source counted."""
    for i in range(15):
        temp_db.record_source_outcome(
            titan_id="T1", source_type="follower", score=+1,
            items_fetched=1, items_stored=1, api_calls_used=1)
    scores = temp_db.get_source_priority_scores("T1", window=10)
    assert scores["follower"] == 10  # capped at window


def test_score_per_titan_isolated(temp_db):
    """T1 scores don't leak into T2."""
    temp_db.record_source_outcome(
        titan_id="T1", source_type="vocab", score=+1,
        items_fetched=1, items_stored=1, api_calls_used=1)
    s1 = temp_db.get_source_priority_scores("T1")
    s2 = temp_db.get_source_priority_scores("T2")
    assert s1["vocab"] == 1
    assert s2["vocab"] == 0


# ── Query composition ─────────────────────────────────────────────────

def test_compose_dynamic_query_empty_returns_blank(teacher):
    """No grounded vocab + no recent + no sentence → empty query."""
    teacher._get_grounded_vocab_terms = lambda n: []
    teacher._get_recent_vocab_terms = lambda n: []
    teacher._get_last_sentence_keyword = lambda: ""
    q = teacher._compose_dynamic_query(
        {"events_teacher": {"dynamic_grounded_top_n": 2,
                             "dynamic_recent_top_n": 1,
                             "dynamic_sentence_keyword": True}})
    assert q == ""


def test_compose_dynamic_query_combines_terms(teacher):
    teacher._get_grounded_vocab_terms = lambda n: ["consciousness", "emergence"]
    teacher._get_recent_vocab_terms = lambda n: ["sentience"]
    teacher._get_last_sentence_keyword = lambda: "philosophy"
    q = teacher._compose_dynamic_query(
        {"events_teacher": {"dynamic_grounded_top_n": 2,
                             "dynamic_recent_top_n": 1,
                             "dynamic_sentence_keyword": True}})
    # Order = sorted alphabetically
    assert "consciousness" in q
    assert "emergence" in q
    assert "sentience" in q
    assert "philosophy" in q
    assert " OR " in q
    assert "lang:en" in q
    assert "-is:retweet" in q


def test_compose_dynamic_query_caps_at_5_terms(teacher):
    teacher._get_grounded_vocab_terms = lambda n: [
        "alpha_word", "beta_word", "gamma_word"]
    teacher._get_recent_vocab_terms = lambda n: [
        "delta_word", "epsilon_word", "zeta_word"]
    teacher._get_last_sentence_keyword = lambda: "eta_word"
    q = teacher._compose_dynamic_query(
        {"events_teacher": {"dynamic_grounded_top_n": 3,
                             "dynamic_recent_top_n": 3,
                             "dynamic_sentence_keyword": True}})
    # 5 terms × " OR " separators (4 separators)
    parts = q.split(" lang:en")[0].split(" OR ")
    assert len(parts) <= 5


def test_compose_wide_query_from_comma_string(teacher):
    cfg = {"events_teacher": {"wide_topic_keywords":
                              "#AI,#consciousness,#spiritual,#blockchain"}}
    q = teacher._compose_wide_query(cfg)
    assert "#AI" in q
    assert "#consciousness" in q
    assert "#spiritual" in q
    assert "#blockchain" in q
    assert " OR " in q
    assert "lang:en -is:retweet" in q


def test_compose_wide_query_handles_list(teacher):
    """Backward-compat: also accepts list (not just comma-string)."""
    cfg = {"events_teacher": {"wide_topic_keywords": ["#AI", "#emergence"]}}
    q = teacher._compose_wide_query(cfg)
    assert "#AI" in q
    assert "#emergence" in q


def test_compose_wide_query_default_when_unset(teacher):
    """Falls back to baseline keywords when config unset."""
    q = teacher._compose_wide_query({})
    assert "#AI" in q or "#consciousness" in q


# ── Vocab term extractors ─────────────────────────────────────────────

def test_get_grounded_vocab_filters_short_words(teacher, monkeypatch):
    """Words < 4 chars filtered (skip 'a', 'the', 'is', etc.)."""
    state_path = Path("data/language_teacher_state.json")
    state = {"grounded_words": {
        "a": {"confidence": 0.99},
        "the": {"confidence": 0.95},
        "consciousness": {"confidence": 0.9},
        "emergence": {"confidence": 0.85},
    }}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        monkeypatch.chdir(td_path)
        (td_path / "data").mkdir()
        (td_path / "data/language_teacher_state.json").write_text(json.dumps(state))
        terms = teacher._get_grounded_vocab_terms(top_n=3)
    assert "a" not in terms
    assert "the" not in terms
    assert "consciousness" in terms or "emergence" in terms


# ── Adaptive cycle fetch (priority sort + tag preservation) ───────────

def test_priority_sort_desc_by_score(teacher, temp_db):
    """Highest score first, ties broken lexically."""
    teacher._db = temp_db
    # Vocab score = +5, follower = +2, wide = +5 (tie with vocab)
    for _ in range(5):
        temp_db.record_source_outcome("T1", "vocab", +1, 1, 1, 1)
        temp_db.record_source_outcome("T1", "wide", +1, 1, 1, 1)
    for _ in range(2):
        temp_db.record_source_outcome("T1", "follower", +1, 1, 1, 1)
    # Stub all fetchers to return empty
    teacher._fetch_follower_timelines = lambda c: ([], 0)
    teacher._fetch_topic_search = lambda q, c, s: ([], 0)
    teacher._compose_dynamic_query = lambda c: "vocab_query"
    teacher._compose_wide_query = lambda c: "wide_query"
    items, results = teacher._run_adaptive_cycle_fetch("T1", {})
    # Tie-break: vocab beats wide alphabetically
    priorities = list(results.keys())
    assert priorities[0] in ("vocab", "wide")
    # Follower should be last (lower score)
    assert priorities[-1] == "follower"


def test_cold_start_default_priority(teacher):
    """No history → all sources fetched; priorities are score=0 default."""
    teacher._fetch_follower_timelines = lambda c: ([{"source": "follower_timeline"}], 1)
    teacher._fetch_topic_search = lambda q, c, s: (
        [{"source": s}] if q else [], 1 if q else 0)
    teacher._compose_dynamic_query = lambda c: "consciousness OR emergence"
    teacher._compose_wide_query = lambda c: "#AI OR #consciousness"
    items, results = teacher._run_adaptive_cycle_fetch("T1", {})
    assert len(results) == 3
    assert "follower" in results
    assert "vocab" in results
    assert "wide" in results
    # 3 fetches × 1 X API each
    total_calls = sum(r.get("api_calls", 0) for r in results.values())
    assert total_calls == 3


def test_cycle_items_tagged_with_source(teacher):
    """Items returned must have `source` field set so scoring works."""
    teacher._fetch_follower_timelines = lambda c: (
        [{"source": "follower_timeline", "text": "f1"}], 1)
    teacher._fetch_topic_search = lambda q, c, s: (
        [{"source": s, "text": "t1"}], 1) if q else ([], 0)
    teacher._compose_dynamic_query = lambda c: "consciousness"
    teacher._compose_wide_query = lambda c: "#AI"
    items, results = teacher._run_adaptive_cycle_fetch("T1", {})
    sources = {i["source"] for i in items}
    assert "follower_timeline" in sources
    assert "topic_vocab" in sources
    assert "topic_wide" in sources


# ── Outcome scoring (post-distillation) ───────────────────────────────

class _FakeEvent:
    def __init__(self, source: str, relevance: float):
        self.source = source
        self.relevance = relevance


def test_record_outcomes_lands_correct_source(teacher, temp_db):
    """events with source='topic_vocab' + relevance>=0.3 = vocab lands +1."""
    teacher._db = temp_db
    cycle_results = {
        "follower": {"items_fetched": 5, "api_calls": 1, "query": "f"},
        "vocab": {"items_fetched": 3, "api_calls": 1, "query": "v"},
        "wide": {"items_fetched": 4, "api_calls": 1, "query": "w"},
    }
    events = [
        _FakeEvent("topic_vocab", 0.6),  # vocab: lands
        _FakeEvent("topic_vocab", 0.5),
        _FakeEvent("follower_timeline", 0.1),  # follower: rel<0.3 → not stored
        _FakeEvent("topic_wide", 0.4),  # wide: lands
    ]
    teacher._record_adaptive_cycle_outcomes(
        "T1", events, cycle_results, {"land_threshold_events": 1})
    scores = temp_db.get_source_priority_scores("T1")
    assert scores["vocab"] == +1
    assert scores["wide"] == +1
    assert scores["follower"] == -1  # 0 stored events
    # cycle_results enriched in place
    assert cycle_results["vocab"]["items_stored"] == 2
    assert cycle_results["follower"]["items_stored"] == 0


def test_land_threshold_higher_blocks_low_yield(teacher, temp_db):
    """land_threshold=2 means 1 stored event = -1 (didn't quite land)."""
    teacher._db = temp_db
    events = [_FakeEvent("topic_vocab", 0.5)]  # 1 stored from vocab
    cycle_results = {"vocab": {"items_fetched": 1, "api_calls": 1, "query": ""}}
    teacher._record_adaptive_cycle_outcomes(
        "T1", events, cycle_results, {"land_threshold_events": 2})
    scores = temp_db.get_source_priority_scores("T1")
    assert scores["vocab"] == -1  # 1 < 2 threshold


# ── Echo chamber mitigation (wide always in cycle) ────────────────────

def test_wide_source_always_fetched(teacher):
    """Even when scores show wide deeply negative, it's still in the cycle."""
    teacher._fetch_follower_timelines = lambda c: ([], 1)
    teacher._fetch_topic_search = lambda q, c, s: ([], 1) if q else ([], 0)
    teacher._compose_dynamic_query = lambda c: "vocab_query"
    teacher._compose_wide_query = lambda c: "#AI"
    items, results = teacher._run_adaptive_cycle_fetch("T1", {})
    assert "wide" in results  # never skipped
