"""
KP-8 — Smart routing learner tests.

Covers RoutingLearner behaviour per rFP_knowledge_pipeline_v2.md §3.4:
  * record_outcome updates counters + latency/quality EMAs
  * record_quality updates only the quality EMA
  * record_usage boosts reputation without changing success-rate
  * learned_chain returns None on cold start (no row ≥ min_samples)
  * learned_chain reorders warm backends by reputation descending
  * learned_chain demotes below-threshold warm backends entirely
  * cold backends stay at the tail of the reordered list in static order
  * learned_chain returns None when learner is disabled
  * window pruning drops stale rows
  * persistence round-trips across instantiations
  * snapshot() shape for /v4/search-pipeline/learning
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time

import pytest

from titan_plugin.logic.knowledge_router import QueryType
from titan_plugin.logic.knowledge_routing_learner import (
    DEFAULT_MIN_SAMPLES,
    EVENT_KIND_CHAIN_REORDERED,
    BackendStats,
    RoutingLearner,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def learner(tmp_path):
    db = os.path.join(tmp_path, "routing.db")
    return RoutingLearner(db_path=db, min_samples=3, demote_threshold=0.10)


def _feed(learner: RoutingLearner, qt: QueryType, backend: str,
           n_success: int, n_fail: int,
           latency_ms: float = 100.0, quality: float = 0.6):
    for _ in range(n_success):
        learner.record_outcome(qt, backend, success=True,
                                latency_ms=latency_ms, quality=quality)
    for _ in range(n_fail):
        learner.record_outcome(qt, backend, success=False,
                                latency_ms=latency_ms, quality=0.0)


# ── Counter + EMA updates ─────────────────────────────────────────────

def test_record_outcome_increments_counters(learner):
    _feed(learner, QueryType.DICTIONARY, "wiktionary", n_success=5, n_fail=2)
    snap = learner.snapshot()
    rows = snap["by_query_type"]["dictionary"]
    assert len(rows) == 1
    r = rows[0]
    assert r["backend"] == "wiktionary"
    assert r["n_attempts"] == 7
    assert r["n_success"] == 5
    # snapshot rounds success_rate to 4 decimals, so tolerance matches
    assert abs(r["success_rate"] - 5 / 7) < 1e-3
    assert r["warm"] is True   # min_samples=3, 7 >= 3


def test_record_quality_updates_ema_only(learner):
    # Start with some attempts at middling quality
    _feed(learner, QueryType.CONCEPTUAL, "searxng_ddg_brave_wiki",
          n_success=5, n_fail=0, quality=0.3)
    snap1 = learner.snapshot()
    q1 = snap1["by_query_type"]["conceptual"][0]["avg_quality"]
    attempts1 = snap1["by_query_type"]["conceptual"][0]["n_attempts"]

    # Pure quality update — counters must NOT move
    for _ in range(5):
        learner.record_quality(QueryType.CONCEPTUAL,
                                "searxng_ddg_brave_wiki", 0.9)
    snap2 = learner.snapshot()
    q2 = snap2["by_query_type"]["conceptual"][0]["avg_quality"]
    attempts2 = snap2["by_query_type"]["conceptual"][0]["n_attempts"]

    assert attempts1 == attempts2   # counters unchanged
    assert q2 > q1                   # quality EMA moved up


def test_record_usage_boosts_reputation(learner):
    _feed(learner, QueryType.DICTIONARY, "wiktionary",
          n_success=10, n_fail=0, quality=0.5)
    rep_before = learner.snapshot()["by_query_type"]["dictionary"][0]["reputation"]
    for _ in range(20):
        learner.record_usage(QueryType.DICTIONARY, "wiktionary")
    rep_after = learner.snapshot()["by_query_type"]["dictionary"][0]["reputation"]
    assert rep_after > rep_before


# ── learned_chain — cold / warm / demotion ────────────────────────────

def test_learned_chain_cold_start_returns_none(learner):
    # Zero observations → static_chain returned as None (use static)
    assert learner.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary", "wikipedia_direct"]) is None


def test_learned_chain_below_min_samples_stays_cold(learner):
    # min_samples=3, feed only 2 → still cold
    _feed(learner, QueryType.DICTIONARY, "wiktionary",
          n_success=2, n_fail=0)
    assert learner.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary"]) is None


def test_learned_chain_reorders_by_reputation(learner):
    # Wiktionary: low success (3/10 = 0.3 success_rate, rep ≈ 0.21)
    _feed(learner, QueryType.DICTIONARY, "wiktionary",
          n_success=3, n_fail=7, quality=0.4)
    # Free dictionary: high success (8/10 = 0.8 success_rate, rep ≈ 0.56 + 0.3*q)
    _feed(learner, QueryType.DICTIONARY, "free_dictionary",
          n_success=8, n_fail=2, quality=0.7)
    chain = learner.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary", "wikipedia_direct"])
    assert chain is not None
    # free_dictionary promoted ahead of wiktionary
    assert chain.index("free_dictionary") < chain.index("wiktionary")
    # wikipedia_direct was cold (no stats) → stays at the tail
    assert chain[-1] == "wikipedia_direct"


def test_learned_chain_demotes_below_threshold(learner):
    # demote_threshold=0.10; very low success + very low quality → drop
    _feed(learner, QueryType.DICTIONARY, "wiktionary",
          n_success=0, n_fail=10, quality=0.0)
    _feed(learner, QueryType.DICTIONARY, "free_dictionary",
          n_success=9, n_fail=1, quality=0.7)
    chain = learner.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary"])
    assert chain is not None
    assert "wiktionary" not in chain
    assert chain == ["free_dictionary"]


def test_learned_chain_no_change_returns_none(learner):
    # Both backends warm + already in reputation order → no reorder needed
    _feed(learner, QueryType.DICTIONARY, "wiktionary",
          n_success=9, n_fail=1, quality=0.8)   # high rep
    _feed(learner, QueryType.DICTIONARY, "free_dictionary",
          n_success=5, n_fail=5, quality=0.5)   # lower rep
    # Static chain is already [wiktionary, free_dictionary] (rep order)
    chain = learner.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary"])
    assert chain is None   # no change → None so dispatcher keeps static


def test_learned_chain_disabled_returns_none(tmp_path):
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                        min_samples=3, enabled=False)
    _feed(l, QueryType.DICTIONARY, "wiktionary", 10, 0, quality=0.9)
    # Even with warm stats, disabled learner refuses to reorder
    assert l.learned_chain(QueryType.DICTIONARY, ["wiktionary"]) is None


# ── Window pruning + persistence ──────────────────────────────────────

def test_window_prune_drops_stale_rows(tmp_path):
    db = os.path.join(tmp_path, "r.db")
    l = RoutingLearner(db_path=db, min_samples=1, window_days=1)
    _feed(l, QueryType.DICTIONARY, "wiktionary", 3, 0)
    # Rewind last_update_ts to "2 days ago" (beyond the 1-day window)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE routing_stats SET last_update_ts = ?",
            (time.time() - 2 * 86400,))
    # Snapshot (which prunes) → row gone
    snap = l.snapshot()
    assert snap["by_query_type"] == {}


def test_persistence_round_trip(tmp_path):
    db = os.path.join(tmp_path, "r.db")
    l1 = RoutingLearner(db_path=db, min_samples=1)
    _feed(l1, QueryType.DICTIONARY, "wiktionary", 5, 1, quality=0.7)
    l1.record_usage(QueryType.DICTIONARY, "wiktionary")

    # New instance reads the same DB
    l2 = RoutingLearner(db_path=db, min_samples=1)
    snap = l2.snapshot()
    row = snap["by_query_type"]["dictionary"][0]
    assert row["n_attempts"] == 6
    assert row["n_success"] == 5
    assert row["n_usage"] == 1
    assert row["avg_quality"] > 0


# ── Snapshot shape ────────────────────────────────────────────────────

def test_snapshot_config_fields(learner):
    snap = learner.snapshot()
    assert "ts" in snap
    cfg = snap["config"]
    assert cfg["enabled"] is True
    assert cfg["window_days"] == 7
    assert cfg["min_samples"] == 3
    assert cfg["demote_threshold"] == pytest.approx(0.10)


def test_backend_stats_reputation_formula():
    # Explicit sanity on the reputation math — success_weight 0.7,
    # quality_weight 0.3, usage_bonus 0.1 * tanh(usage/10)
    s = BackendStats(query_type="dictionary", backend="wiktionary",
                      n_attempts=10, n_success=10, n_usage=0,
                      avg_quality=0.5)
    # Pure: 0.7*1.0 + 0.3*0.5 + 0.1*tanh(0) = 0.85
    assert abs(s.reputation() - 0.85) < 1e-4
    # With usage → bonus
    s.n_usage = 10
    # 0.85 + 0.1 * tanh(1) ≈ 0.85 + 0.0762 = 0.9262 (capped at 1.0)
    assert s.reputation() > 0.85
    assert s.reputation() <= 1.0


def test_record_usage_creates_row_if_missing(learner):
    # No prior attempts — record_usage should still create a row
    learner.record_usage(QueryType.NEWS, "news_api")
    snap = learner.snapshot()
    rows = snap["by_query_type"]["news"]
    assert len(rows) == 1
    assert rows[0]["n_usage"] == 1
    assert rows[0]["n_attempts"] == 0


# ── on_event callback (KP-8.1) ────────────────────────────────────────

def test_on_event_fires_on_chain_reorder(tmp_path):
    events = []
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                       min_samples=3, demote_threshold=0.10,
                       on_event=lambda k, c: events.append((k, c)))
    # Warm two backends, free_dictionary much better than wiktionary
    _feed(l, QueryType.DICTIONARY, "wiktionary",
          n_success=3, n_fail=7, quality=0.4)
    _feed(l, QueryType.DICTIONARY, "free_dictionary",
          n_success=8, n_fail=2, quality=0.7)
    chain = l.learned_chain(
        QueryType.DICTIONARY,
        ["wiktionary", "free_dictionary", "wikipedia_direct"])
    assert chain is not None
    assert len(events) == 1
    kind, ctx = events[0]
    assert kind == EVENT_KIND_CHAIN_REORDERED
    assert ctx["query_type"] == "dictionary"
    assert ctx["static_chain"] == ["wiktionary", "free_dictionary",
                                     "wikipedia_direct"]
    assert ctx["reordered_chain"] == chain
    assert ctx["n_warm"] == 2


def test_on_event_includes_demoted_backends(tmp_path):
    events = []
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                       min_samples=3, demote_threshold=0.10,
                       on_event=lambda k, c: events.append((k, c)))
    # wiktionary — below demote threshold (rep ≈ 0)
    _feed(l, QueryType.DICTIONARY, "wiktionary",
          n_success=0, n_fail=10, quality=0.0)
    _feed(l, QueryType.DICTIONARY, "free_dictionary",
          n_success=9, n_fail=1, quality=0.7)
    l.learned_chain(QueryType.DICTIONARY,
                     ["wiktionary", "free_dictionary"])
    assert len(events) == 1
    _, ctx = events[0]
    assert "wiktionary" in ctx["demoted"]
    assert "free_dictionary" not in ctx["demoted"]


def test_on_event_not_fired_on_cold_start(tmp_path):
    events = []
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                       min_samples=3,
                       on_event=lambda k, c: events.append((k, c)))
    # Fewer than min_samples → cold → returns None, no event
    _feed(l, QueryType.DICTIONARY, "wiktionary", n_success=2, n_fail=0)
    assert l.learned_chain(QueryType.DICTIONARY,
                            ["wiktionary", "free_dictionary"]) is None
    assert events == []


def test_on_event_not_fired_when_chain_unchanged(tmp_path):
    events = []
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                       min_samples=3,
                       on_event=lambda k, c: events.append((k, c)))
    # Warm both, already in reputation order → no reorder → no event
    _feed(l, QueryType.DICTIONARY, "wiktionary",
          n_success=9, n_fail=1, quality=0.8)
    _feed(l, QueryType.DICTIONARY, "free_dictionary",
          n_success=5, n_fail=5, quality=0.5)
    l.learned_chain(QueryType.DICTIONARY,
                     ["wiktionary", "free_dictionary"])
    assert events == []


def test_on_event_callback_exception_does_not_break_learner(tmp_path):
    def exploding_cb(_k, _c):
        raise RuntimeError("boom")
    l = RoutingLearner(db_path=os.path.join(tmp_path, "r.db"),
                       min_samples=3, on_event=exploding_cb)
    _feed(l, QueryType.DICTIONARY, "wiktionary",
          n_success=3, n_fail=7, quality=0.4)
    _feed(l, QueryType.DICTIONARY, "free_dictionary",
          n_success=8, n_fail=2, quality=0.7)
    # Must not raise despite the callback exploding
    chain = l.learned_chain(
        QueryType.DICTIONARY, ["wiktionary", "free_dictionary"])
    assert chain is not None
