"""DK.5 (RFP §7.D-knowledge) — research-path-as-skill + the Axis-1 volatility
classifier. The research SKILL (which source answered a goal-class) is kept +
reinforced (research_recipes), surviving the volatile data's decay; a matured
recipe crystallizes the (goal_class, 'research') macro. Covers GD-DK5 + the
discern classifier.

Run: python -m pytest tests/test_research_skill.py -v -p no:anchorpy
"""
from __future__ import annotations

import os
import tempfile
import contextlib

import duckdb
import pytest

from titan_hcl.synthesis.research_volatility import classify_volatility, is_volatile
from titan_hcl.synthesis.reasoning_store import ReasoningStore
from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.kuzu_spine_schema import bootstrap_spine_schema


# ── Axis-1 volatility classifier ──────────────────────────────────────

def test_volatile_temporal_markers():
    for q in ("what is the current SOL price",
              "latest news on Solana",
              "how many validators are there now",
              "today's weather in Berlin"):
        assert classify_volatility(q) == "volatile", q


def test_durable_conceptual():
    for q in ("what does sovereignty mean to you",
              "how does Solana consensus work",
              "explain proof of history"):
        assert classify_volatility(q) == "durable", q


def test_domain_override_beats_lexical():
    # volatile domain → volatile even without a lexical marker
    assert classify_volatility("Solana overview", "market") == "volatile"
    # evergreen domain → durable even if a marker appears
    assert classify_volatility("the current theory", "philosophy") == "durable"


def test_is_volatile_helper_and_empty():
    assert is_volatile("current price of SOL") is True
    assert is_volatile("what is a transformer") is False
    assert classify_volatility("") == "durable"   # nothing to judge → keep


# ── DK.5 research_recipes reinforcement + maturity ────────────────────

@contextlib.contextmanager
def _store():
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "d.kuzu"))
        bootstrap_spine_schema(g)
        conn = duckdb.connect(os.path.join(tmp, "synth.duckdb"))
        s = ReasoningStore(conn, faiss_path=os.path.join(tmp, "rv.faiss"),
                           graph=g, embedder=lambda _t: None, writer=None)
        try:
            yield s
        finally:
            g.close()


def test_recipe_reinforces_and_matures():
    with _store() as s:
        # First success → count 1, not matured (mature_at=2).
        cnt, matured = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=10.0)
        assert cnt == 1 and matured is False
        # Second success (same goal+source) → count 2, JUST matured.
        cnt, matured = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=20.0)
        assert cnt == 2 and matured is True
        # Third → keeps reinforcing (living skill), no longer "just matured".
        cnt, matured = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=30.0)
        assert cnt == 3 and matured is False


def test_distinct_sources_counted_separately():
    with _store() as s:
        s.record_research_recipe(goal_class="g", source="src_a", epoch=1.0)
        cnt_b, _ = s.record_research_recipe(goal_class="g", source="src_b", epoch=2.0)
        assert cnt_b == 1   # a different source starts its own count
        row = s._db.execute(
            "SELECT COUNT(*) FROM research_recipes WHERE goal_class='g'").fetchone()
        assert int(row[0]) == 2   # two recipes for the goal-class


def test_empty_goal_or_source_is_noop():
    with _store() as s:
        assert s.record_research_recipe(goal_class="", source="x", epoch=1.0) == (0, False)
        assert s.record_research_recipe(goal_class="g", source="", epoch=1.0) == (0, False)


def test_last_used_epoch_updates():
    with _store() as s:
        s.record_research_recipe(goal_class="g", source="s", epoch=5.0)
        s.record_research_recipe(goal_class="g", source="s", epoch=99.0)
        row = s._db.execute(
            "SELECT success_count, last_used_epoch FROM research_recipes "
            "WHERE goal_class='g' AND source='s'").fetchone()
        assert int(row[0]) == 2 and float(row[1]) == 99.0
