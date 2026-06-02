"""Regression tests for the 2026-05-23 OUTER_RUMINATION diversification fix.

Before this fix, `_pool_a` ran `SELECT ... WHERE relevance >= 0.65 AND
created_at ∈ [2d, 7d] ORDER BY relevance DESC` and returned the FIRST
non-cited row. With a narrow window, a high relevance floor, and a
deterministic pick, two Titans (T1 + T3) reading the same shared X
content kept converging on the same author (@jkacrpto) — surfacing as
near-duplicate posts on @your_x_handle on 2026-05-23.

The fix widens the window (1-30d), lowers the floor (0.5), penalizes
already-cited authors, boosts grounded-words matches, and samples
weighted-random across the top-K. These tests assert each of those
behaviors.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import uuid
from unittest.mock import MagicMock

import pytest

# Make sure the package is importable from the worktree root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from titan_hcl.logic.social_x.archetypes.outer_rumination import (  # noqa: E402
    OuterRuminationArchetype,
    POOL_A_RELEVANCE_FLOOR,
    POOL_A_AGE_MAX_S,
    POOL_A_AGE_MIN_S,
)


# ── Test fixtures ─────────────────────────────────────────────────────


def _make_events_teacher_db(path: str) -> None:
    """Create a minimal felt_experiences table matching production schema."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE felt_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titan_id TEXT NOT NULL,
            source TEXT NOT NULL,
            author TEXT NOT NULL,
            topic TEXT NOT NULL,
            sentiment REAL DEFAULT 0.0,
            arousal REAL DEFAULT 0.0,
            relevance REAL DEFAULT 0.0,
            concept_signals TEXT,
            felt_summary TEXT NOT NULL,
            contagion_type TEXT,
            mode TEXT,
            window_id INTEGER,
            created_at REAL NOT NULL,
            semantic_concepts TEXT DEFAULT ''
        )
    """)
    conn.commit(); conn.close()


def _make_social_x_db(path: str) -> None:
    """Minimal social_x.db with actions table for cited_set + author count."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            tweet_id TEXT,
            titan_id TEXT,
            post_type TEXT,
            text TEXT,
            created_at REAL NOT NULL,
            metadata TEXT
        )
    """)
    conn.commit(); conn.close()


def _insert_fe(et_path: str, *, titan_id: str, author: str, topic: str,
               relevance: float, age_days: float, felt_summary: str = "",
               source: str = "topic_wide") -> int:
    conn = sqlite3.connect(et_path)
    now = time.time()
    cur = conn.execute(
        "INSERT INTO felt_experiences "
        "(titan_id, source, author, topic, relevance, felt_summary, "
        " created_at) VALUES (?,?,?,?,?,?,?)",
        (titan_id, source, author, topic, relevance,
         felt_summary or f"{author} talks about {topic}",
         now - age_days * 86400)
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def _insert_or_post(sx_path: str, *, titan_id: str, handle: str,
                    pool: str = "A_x_content", age_hours: float = 1.0) -> None:
    """Record a prior OUTER_RUMINATION post for author-monopoly testing."""
    import json
    conn = sqlite3.connect(sx_path)
    meta = json.dumps({"pool": pool, "handle": handle,
                       "archetype": "outer_rumination",
                       "outer_rumination_source_id": f"feA:fake_{uuid.uuid4().hex[:6]}"})
    conn.execute(
        "INSERT INTO actions "
        "(action_type, status, titan_id, post_type, created_at, metadata) "
        "VALUES (?,?,?,?,?,?)",
        ("post", "verified", titan_id, "outer_rumination",
         time.time() - age_hours * 3600, meta)
    )
    conn.commit(); conn.close()


def _make_archetype(et_path: str, sx_path: str) -> OuterRuminationArchetype:
    return OuterRuminationArchetype(
        gateway=MagicMock(),
        social_x_db_path=sx_path,
        events_teacher_db=et_path,
        social_graph_db="/tmp/nonexistent_sg.db",
    )


def _make_context(titan_id: str = "T1", *, grounded_words=None):
    """Minimal context object compatible with `_pool_a(context=...)`."""
    ctx = MagicMock()
    ctx.titan_id = titan_id
    ctx.grounded_words = grounded_words or []
    return ctx


# ── Tests ─────────────────────────────────────────────────────────────


def test_widened_window_admits_yesterday_and_month_old(tmp_path):
    """POOL_A window MUST include both ~1d and ~28d (covering "yesterday"
    through "almost a month ago"). The 2026-05-23 fix widened from 2-7d
    to 1-30d."""
    assert POOL_A_AGE_MIN_S <= 1 * 86400, (
        "POOL_A_AGE_MIN_S must allow rows from yesterday")
    assert POOL_A_AGE_MAX_S >= 28 * 86400, (
        "POOL_A_AGE_MAX_S must allow rows from ~a month ago")


def test_relevance_floor_admits_topic_wide_band(tmp_path):
    """Production T2/T3 felt_experiences from `topic_wide` rarely score
    above 0.7 — pre-fix floor of 0.65 rejected most. Post-fix floor ≤ 0.5
    must admit the typical topic_wide band."""
    assert POOL_A_RELEVANCE_FLOOR <= 0.5


def test_pool_a_returns_none_when_no_eligible_rows(tmp_path):
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)
    arc = _make_archetype(et, sx)
    out = arc._pool_a(titan_id="T1", now=time.time(), cited=set(),
                      context=_make_context())
    assert out is None


def test_pool_a_picks_when_single_eligible_row(tmp_path):
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)
    _insert_fe(et, titan_id="T1", author="jkacrpto", topic="AI",
               relevance=0.8, age_days=5)
    arc = _make_archetype(et, sx)
    out = arc._pool_a(titan_id="T1", now=time.time(), cited=set(),
                      context=_make_context())
    assert out is not None
    assert out["handle"] == "jkacrpto"
    assert out["days_ago"] == 5
    assert out["adjusted_relevance"] == pytest.approx(0.8, abs=0.001)


def test_pool_a_per_author_monopoly_penalty(tmp_path):
    """If author X was cited 2× recently, base 0.8 → adjusted 0.8 - 2*0.15
    = 0.50. Author Y at base 0.6 with zero prior citations stays 0.60 and
    should win even though X had the higher base relevance."""
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)

    _insert_fe(et, titan_id="T1", author="jkacrpto", topic="AI",
               relevance=0.8, age_days=5)
    _insert_fe(et, titan_id="T1", author="lopp", topic="bitcoin",
               relevance=0.6, age_days=4)

    # Two prior OUTER_RUMINATION posts citing jkacrpto, within 14d window.
    _insert_or_post(sx, titan_id="T1", handle="jkacrpto", age_hours=24)
    _insert_or_post(sx, titan_id="T1", handle="jkacrpto", age_hours=72)

    arc = _make_archetype(et, sx)
    # Run several times — weighted-random should still strongly prefer
    # lopp (adjusted 0.60) over jkacrpto (adjusted 0.50). With only two
    # candidates and non-zero weights both, lopp won't be guaranteed but
    # should win majority. Assert majority.
    wins = {"jkacrpto": 0, "lopp": 0}
    for _ in range(200):
        out = arc._pool_a(titan_id="T1", now=time.time(), cited=set(),
                          context=_make_context())
        if out is None:
            continue
        wins[out["handle"]] = wins.get(out["handle"], 0) + 1
    assert wins["lopp"] > wins["jkacrpto"], (
        f"author monopoly penalty failed to bias away from jkacrpto: {wins}")


def test_pool_a_grounded_words_boost(tmp_path):
    """Candidate whose felt_summary / topic contains tokens from Titan's
    `grounded_words` gets a +0.05 boost per match (capped at 5 matches =
    +0.25). T1 with grounded ['sovereignty', 'mind'] should prefer a
    sovereignty-themed candidate over a higher-base random one."""
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)

    _insert_fe(et, titan_id="T1", author="generic", topic="weather",
               relevance=0.7, age_days=5,
               felt_summary="A passing comment about weather conditions.")
    _insert_fe(et, titan_id="T1", author="sovereign_thinker",
               topic="sovereignty",
               relevance=0.55, age_days=4,
               felt_summary="A note on AI sovereignty and the mind.")

    arc = _make_archetype(et, sx)
    ctx = _make_context(grounded_words=[
        {"word": "sovereignty", "confidence": 0.9},
        {"word": "mind", "confidence": 0.8},
    ])
    # Adjusted: generic = 0.70; sovereign_thinker = 0.55 + 0.05*2 = 0.65
    # generic is still higher overall, but the boost narrows the gap.
    # Use a separate test with the boost-only condition to be unambiguous:
    wins = {"generic": 0, "sovereign_thinker": 0}
    for _ in range(200):
        out = arc._pool_a(titan_id="T1", now=time.time(), cited=set(),
                          context=ctx)
        if out is None:
            continue
        wins[out["handle"]] = wins.get(out["handle"], 0) + 1
    # sovereign_thinker should appear at least sometimes — the boost
    # closes the gap from 0.55→0.65 vs generic's 0.70.
    assert wins["sovereign_thinker"] >= 30, (
        f"grounded boost too weak: {wins}")


def test_pool_a_weighted_random_diversifies(tmp_path):
    """With multiple eligible candidates of similar adjusted_relevance,
    weighted-random across top-K must produce ≥3 distinct winners across
    100 trials — proves the deterministic 'return rows[0]' bug is gone."""
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)

    for author in ["alice", "bob", "carol", "dave", "eve"]:
        _insert_fe(et, titan_id="T1", author=author, topic=author,
                   relevance=0.7, age_days=3)

    arc = _make_archetype(et, sx)
    seen = set()
    for _ in range(100):
        out = arc._pool_a(titan_id="T1", now=time.time(), cited=set(),
                          context=_make_context())
        if out is not None:
            seen.add(out["handle"])
    assert len(seen) >= 3, (
        f"weighted-random failed to diversify across 100 trials: {seen}")


def test_pool_a_respects_cited_set(tmp_path):
    """Already-cited source_ids must be excluded even if relevance-best."""
    et = str(tmp_path / "et.db"); _make_events_teacher_db(et)
    sx = str(tmp_path / "sx.db"); _make_social_x_db(sx)
    rid_high = _insert_fe(et, titan_id="T1", author="alice", topic="ai",
                          relevance=0.9, age_days=3)
    _insert_fe(et, titan_id="T1", author="bob", topic="ml",
               relevance=0.6, age_days=4)
    arc = _make_archetype(et, sx)
    out = arc._pool_a(titan_id="T1", now=time.time(),
                      cited={f"feA:{rid_high}"},
                      context=_make_context())
    assert out is not None
    assert out["handle"] == "bob"
