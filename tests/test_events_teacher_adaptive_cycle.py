"""Tests for Events Teacher adaptive 3-source cycle.

Original design (2026-04-30): 3 sources (follower/vocab/wide) tried in
priority order = sum(scores over last N cycles); +1 if stored ≥ land_threshold
else −1; ALL fetched every cycle (order only).

C3 conversion-anchored rewrite (rFP_x_post_onchain_provenance_honesty Part C,
Maker-ratified 2026-06-03): the score now changes fetch VOLUME, not just order.
  - REWARD = post CONVERSION (joined fe_id → felt_experiences.source) over a
    trailing window: +2 if converted ≥1, else −1.
  - GATE: fetch a source only if score ≥ gate_threshold(0) OR exploration-due.
  - SCALE: converter → full topic_search_count; explore-probe → scale_floor.
  - Exploration floor: every source re-probed ≥1× per explore_cycles windows
    (replaces the old unconditional 'wide always fetched' — echo-chamber
    mitigation is now the floor, not a blanket fetch).
"""
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from titan_hcl.logic.events_teacher import (
    EventsTeacher, EventsTeacherDB, DistilledEvent, _resolve_fe_id)


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


# ── B4 — author_id capture (felt_experiences) ─────────────────────────

def test_store_felt_experience_persists_author_identity(temp_db):
    """B4: store_felt_experience round-trips author_id + author_followers
    (captured at search time → followable by the auto-follow policy)."""
    ev = DistilledEvent(
        source="topic_wide", author="newvoice", topic="ai",
        sentiment=0.2, arousal=0.1, relevance=0.9,
        concept_signals=[], felt_summary="s", contagion_type=None,
        raw_text="t", timestamp=time.time(),
        author_id="1234567890", author_followers=4242)
    temp_db.store_felt_experience("T1", ev, mode="observation", window_id=1)
    conn = temp_db._connect()
    try:
        row = conn.execute(
            "SELECT author_id, author_followers FROM felt_experiences "
            "WHERE author='newvoice'").fetchone()
    finally:
        conn.close()
    assert row["author_id"] == "1234567890"
    assert row["author_followers"] == 4242


# ── C3 — fe_id resolver (Gap ①: heterogeneous metadata) ───────────────

def test_resolve_fe_id_bare_int():
    assert _resolve_fe_id('{"source_id": "1607", "fe_id": "1607"}') == 1607


def test_resolve_fe_id_fea_prefix():
    """outer_rumination stores 'feA:891' — strip the pool namespace."""
    assert _resolve_fe_id('{"source_id": "feA:891"}') == 891


def test_resolve_fe_id_amplify_and_outer_keys():
    assert _resolve_fe_id('{"amplify_source_id": "1876"}') == 1876
    assert _resolve_fe_id('{"outer_id": 1612, "source_id": "1612"}') == 1612


def test_resolve_fe_id_non_fe_namespaces_return_none():
    """mentionC:/livetest: source_ids are NOT felt experiences."""
    assert _resolve_fe_id('{"source_id": "mentionC:2061027167832568021"}') is None
    assert _resolve_fe_id('{"source_id": "livetest:2060722172016890084"}') is None


def test_resolve_fe_id_missing_or_malformed():
    assert _resolve_fe_id("") is None
    assert _resolve_fe_id(None) is None
    assert _resolve_fe_id("{not json") is None
    assert _resolve_fe_id('{"archetype": "amplify"}') is None
    # dict (already-parsed) also accepted
    assert _resolve_fe_id({"fe_id": 42}) == 42


# ── C3 — conversion-score join (REWARD) ───────────────────────────────

def _seed_social_x(path: str):
    """Minimal social_x.db with the actions columns the join reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE actions (id INTEGER PRIMARY KEY, action_type TEXT, "
        "status TEXT, post_type TEXT, created_at REAL, posted_at REAL, "
        "metadata TEXT)")
    conn.commit()
    conn.close()


def _add_action(path, post_type, metadata, status="verified", age_s=3600.0):
    conn = sqlite3.connect(path)
    now = time.time()
    conn.execute(
        "INSERT INTO actions (action_type, status, post_type, created_at, "
        "posted_at, metadata) VALUES ('post', ?, ?, ?, ?, ?)",
        (status, post_type, now - age_s, now - age_s, json.dumps(metadata)))
    conn.commit()
    conn.close()


def _add_fe(db: EventsTeacherDB, fe_id: int, source: str, age_s: float = 3600.0):
    """Insert a felt_experiences row with a fixed id + source."""
    conn = db._connect()
    conn.execute(
        "INSERT INTO felt_experiences (id, titan_id, source, author, topic, "
        "felt_summary, created_at) VALUES (?, 'T1', ?, 'a', 't', 's', ?)",
        (fe_id, source, time.time() - age_s))
    conn.commit()
    conn.close()


def test_conversion_score_converter_vs_waste(temp_db, tmp_path):
    """vocab converts (+2); wide stores but never converts (−1)."""
    sx = str(tmp_path / "social_x.db")
    _seed_social_x(sx)
    # felt experiences: vocab fe #10 (will convert), wide fe #20 (won't)
    _add_fe(temp_db, 10, "topic_vocab")
    _add_fe(temp_db, 20, "topic_wide")
    _add_fe(temp_db, 30, "follower_timeline")
    # a posted world_mirror converts the vocab fe; nothing cites the wide fe
    _add_action(sx, "world_mirror", {"source_id": "10", "fe_id": "10"})
    scores = temp_db.get_source_conversion_scores("T1", window_s=172800.0,
                                                   social_db_path=sx)
    assert scores["topic_vocab"]["converted"] == 1
    assert scores["topic_vocab"]["score"] == 2
    assert scores["topic_wide"]["converted"] == 0
    assert scores["topic_wide"]["stored"] == 1
    assert scores["topic_wide"]["score"] == -1        # stored-but-unconverted
    assert scores["follower_timeline"]["score"] == -1  # nothing converted


def test_conversion_score_window_excludes_old_posts(temp_db, tmp_path):
    """A conversion older than the window does not count."""
    sx = str(tmp_path / "social_x.db")
    _seed_social_x(sx)
    _add_fe(temp_db, 11, "topic_vocab", age_s=10 * 86400)
    _add_action(sx, "outer_rumination", {"source_id": "feA:11"},
                age_s=10 * 86400)   # 10 days old
    scores = temp_db.get_source_conversion_scores("T1", window_s=172800.0,
                                                   social_db_path=sx)
    assert scores["topic_vocab"]["converted"] == 0
    assert scores["topic_vocab"]["score"] == -1


def test_conversion_score_ignores_non_fe_and_failed(temp_db, tmp_path):
    """mention-sourced + failed posts never credit a paid cycle source."""
    sx = str(tmp_path / "social_x.db")
    _seed_social_x(sx)
    _add_fe(temp_db, 12, "topic_vocab")
    _add_action(sx, "outer_rumination",
                {"source_id": "mentionC:99"})            # non-fe → ignored
    _add_action(sx, "world_mirror", {"source_id": "12"},
                status="failed")                          # failed → ignored
    scores = temp_db.get_source_conversion_scores("T1", window_s=172800.0,
                                                   social_db_path=sx)
    assert scores["topic_vocab"]["converted"] == 0


# ── C3 — GATE + SCALE + exploration floor ─────────────────────────────

def _fetch_stub(teacher, captured):
    """Stub fetchers; capture the `count` passed to topic searches."""
    teacher._fetch_follower_timelines = lambda c: (
        [{"source": "follower_timeline"}], 1)

    def _topic(q, c, s, count=None):
        captured[s] = count
        return ([{"source": s}], 1)
    teacher._fetch_topic_search = _topic
    teacher._compose_dynamic_query = lambda c: "vocab_query"
    teacher._compose_wide_query = lambda c: "#AI"


def test_cold_start_explore_floor_fetches_all(teacher, temp_db):
    """No history → every source is exploration-due → all 3 fetched."""
    teacher._db = temp_db
    cap = {}
    _fetch_stub(teacher, cap)
    items, results = teacher._run_adaptive_cycle_fetch("T1", {}, window_number=1)
    assert set(results.keys()) == {"follower", "vocab", "wide"}
    assert sum(r.get("api_calls", 0) for r in results.values()) == 3
    assert all(not r["gated"] for r in results.values())


def test_gate_skips_nonconverter_with_recent_probe(teacher, temp_db):
    """wide scores −1 AND was probed recently → GATED (call eliminated)."""
    teacher._db = temp_db
    # conversion: vocab/follower convert (+2), wide does not (−1)
    teacher._db.get_source_conversion_scores = lambda titan_id, window_s=0: {
        "follower_timeline": {"converted": 2, "stored": 5, "score": 2},
        "topic_vocab": {"converted": 1, "stored": 4, "score": 2},
        "topic_wide": {"converted": 0, "stored": 6, "score": -1},
    }
    # all probed at window 9; current window 10 → wide NOT explore-due (E=3)
    teacher._db.get_last_fetched_windows = lambda titan_id: {
        "follower": 9, "vocab": 9, "wide": 9}
    cap = {}
    _fetch_stub(teacher, cap)
    items, results = teacher._run_adaptive_cycle_fetch("T1", {}, window_number=10)
    assert results["wide"]["gated"] is True
    assert results["wide"]["api_calls"] == 0           # call eliminated
    assert results["vocab"]["gated"] is False
    assert results["follower"]["gated"] is False
    # 2 paid calls this cycle instead of 3
    assert sum(r.get("api_calls", 0) for r in results.values()) == 2


def test_scale_converter_full_explore_probe_floor(teacher, temp_db):
    """Converter fetched at full count; an explore-due gated source at floor."""
    teacher._db = temp_db
    teacher._db.get_source_conversion_scores = lambda titan_id, window_s=0: {
        "follower_timeline": {"converted": 0, "stored": 0, "score": -1},
        "topic_vocab": {"converted": 3, "stored": 5, "score": 2},   # converter
        "topic_wide": {"converted": 0, "stored": 6, "score": -1},   # waste
    }
    # wide last probed at window 1; current 10 → explore-due (>=E) → probed at floor
    teacher._db.get_last_fetched_windows = lambda titan_id: {
        "follower": 9, "vocab": 9, "wide": 1}
    cap = {}
    _fetch_stub(teacher, cap)
    cfg = {"events_teacher": {"topic_search_count": 10},
           "social_x": {"auto_cycle": {"scale_floor": 3, "explore_cycles": 3}}}
    items, results = teacher._run_adaptive_cycle_fetch("T1", cfg, window_number=10)
    assert cap["topic_vocab"] == 10        # converter → full
    assert cap["topic_wide"] == 3          # explore probe → floor
    assert results["wide"]["explore_due"] is True


def test_gate_threshold_zero_passes_cold(teacher, temp_db):
    """gate_threshold=0: a score-0 source would pass; −1 needs the floor."""
    teacher._db = temp_db
    teacher._db.get_source_conversion_scores = lambda titan_id, window_s=0: {
        "follower_timeline": {"converted": 0, "stored": 0, "score": -1},
        "topic_vocab": {"converted": 0, "stored": 0, "score": -1},
        "topic_wide": {"converted": 0, "stored": 0, "score": -1},
    }
    # everyone probed last window → none explore-due → all −1 < 0 → all gated
    teacher._db.get_last_fetched_windows = lambda titan_id: {
        "follower": 9, "vocab": 9, "wide": 9}
    cap = {}
    _fetch_stub(teacher, cap)
    items, results = teacher._run_adaptive_cycle_fetch("T1", {}, window_number=10)
    assert all(r["gated"] for r in results.values())   # 0 paid calls
    assert sum(r.get("api_calls", 0) for r in results.values()) == 0


# ── C3 — outcome recording (conversion score + window_number) ─────────

class _FakeEvent:
    def __init__(self, source: str, relevance: float):
        self.source = source
        self.relevance = relevance


def test_record_outcomes_persists_conversion_score(teacher, temp_db):
    """The recorded score is the conversion score carried in cycle_results."""
    teacher._db = temp_db
    teacher._window_count = 7
    cycle_results = {
        "vocab": {"items_fetched": 3, "api_calls": 1, "query": "v", "score": 2},
        "wide": {"items_fetched": 0, "api_calls": 0, "query": "", "score": -1},
    }
    events = [_FakeEvent("topic_vocab", 0.6), _FakeEvent("topic_vocab", 0.2)]
    teacher._record_adaptive_cycle_outcomes("T1", events, cycle_results, {})
    # priority-sum reads back the persisted +2 / −1
    scores = temp_db.get_source_priority_scores("T1")
    assert scores["vocab"] == 2
    assert scores["wide"] == -1
    # this-cycle stored count (rel≥0.3) recorded for audit
    assert cycle_results["vocab"]["items_stored"] == 1
    # window_number persisted; only the fetched (vocab) row counts as a probe
    last = temp_db.get_last_fetched_windows("T1")
    assert last.get("vocab") == 7
    assert "wide" not in last       # items_fetched=0 → not a probe


def test_last_fetched_windows_tracks_only_probes(temp_db):
    """get_last_fetched_windows ignores 0-fetch (gated) rows."""
    temp_db.record_source_outcome("T1", "vocab", 2, items_fetched=5,
                                  items_stored=3, api_calls_used=1,
                                  window_number=4)
    temp_db.record_source_outcome("T1", "vocab", -1, items_fetched=0,
                                  items_stored=0, api_calls_used=0,
                                  window_number=8)   # gated, later window
    last = temp_db.get_last_fetched_windows("T1")
    assert last["vocab"] == 4    # the gated window-8 row is NOT a probe
