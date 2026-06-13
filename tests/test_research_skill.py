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

from titan_hcl.synthesis.research_volatility import (
    classify_volatility, is_volatile,
    age_epochs, is_stale, freshness_weight, DEFAULT_VOLATILE_LIFETIME_EPOCHS)
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
        # First success → count 1, no crystallize (mature_at=2).
        cnt, signal = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=10.0)
        assert cnt == 1 and signal == ""
        # Second success (same goal+source) → count 2, INITIAL crystallize.
        cnt, signal = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=20.0)
        assert cnt == 2 and signal == "initial"
        # Third → keeps reinforcing (living skill), not yet a re-version.
        cnt, signal = s.record_research_recipe(
            goal_class="solana-price", source="https://api.x/sol", epoch=30.0)
        assert cnt == 3 and signal == ""


def test_recipe_reversions_on_further_growth():
    """M5 — after the initial crystallize (count 2), a recipe RE-VERSIONS once it
    grows another macro_compose_min (default 2) → count 4 fires 'reversion'."""
    with _store() as s:
        kw = dict(goal_class="g", source="src", )
        assert s.record_research_recipe(epoch=1.0, **kw) == (1, "")
        assert s.record_research_recipe(epoch=2.0, **kw) == (2, "initial")
        assert s.record_research_recipe(epoch=3.0, **kw) == (3, "")
        assert s.record_research_recipe(epoch=4.0, **kw) == (4, "reversion")
        assert s.record_research_recipe(epoch=5.0, **kw) == (5, "")
        assert s.record_research_recipe(epoch=6.0, **kw) == (6, "reversion")


def test_macro_lineage_versions():
    """M5 — research_macro_lineage tracks the macro version chain so successors are
    minted research::{gc}::v{n+1} with the prior label as composed_from."""
    with _store() as s:
        # No macro yet.
        assert s.research_macro_lineage("g") == ("", 1)
        # Mint the base macro.
        s.write_macro(reasoning_id="research::g", goal_class="g", action="research",
                      signature=[], b_i=2.0, c=1.0, time_cost=1.0, use_count=2,
                      source="src")
        assert s.research_macro_lineage("g") == ("research::g", 2)
        # Mint v2.
        s.write_macro(reasoning_id="research::g::v2", goal_class="g",
                      action="research", signature=[], b_i=4.0, c=1.0,
                      time_cost=1.0, use_count=4, composed_from=["research::g"],
                      source="src")
        assert s.research_macro_lineage("g") == ("research::g::v2", 3)


def test_dk5_crystallize_then_reversion_endtoend():
    """The exact synthesis daemon sequence (M5/M6): repeated confirms of one
    (goal_class, source) → 'initial' mints research::g (carrying source), further
    growth → 'reversion' mints research::g::v2 composed_from [research::g]."""
    with _store() as s:
        gc, src = "solana-price", "https://api.x/sol"

        def _confirm(epoch):
            cnt, signal = s.record_research_recipe(
                goal_class=gc, source=src, epoch=epoch,
                mature_at=2, macro_compose_min=2)
            if signal == "initial":
                s.write_macro(reasoning_id=f"research::{gc}", goal_class=gc,
                              action="research", signature=[], b_i=float(cnt),
                              c=1.0, time_cost=1.0, use_count=cnt, source=src)
            elif signal == "reversion":
                prior, nextv = s.research_macro_lineage(gc)
                s.write_macro(reasoning_id=f"research::{gc}::v{nextv}",
                              goal_class=gc, action="research", signature=[],
                              b_i=float(cnt), c=1.0, time_cost=1.0, use_count=cnt,
                              composed_from=[prior] if prior else None, source=src)
            return signal

        signals = [_confirm(float(i)) for i in range(1, 5)]
        assert signals == ["", "initial", "", "reversion"]

        # Base macro exists, carries the matured source (M6).
        base = s._db.execute(
            "SELECT action, source FROM reasoning_records "
            "WHERE reasoning_id=? AND kind='macro_strategy'",
            [f"research::{gc}"]).fetchone()
        assert base is not None and base[0] == "research" and base[1] == src
        # Successor macro exists, carries source + composed_from the base (M5).
        succ = s._db.execute(
            "SELECT source FROM reasoning_records WHERE reasoning_id=?",
            [f"research::{gc}::v2"]).fetchone()
        assert succ is not None and succ[0] == src
        # COMPOSED_FROM lineage edge: v2 → base (Kuzu spine).
        qr = s._graph._conn.execute(
            "MATCH (a:Reasoning {reasoning_id:$v})-[:REASONING_COMPOSED_FROM]->"
            "(b:Reasoning {reasoning_id:$base}) RETURN COUNT(a)",
            {"v": f"research::{gc}::v2", "base": f"research::{gc}"})
        assert qr.has_next() and int(qr.get_next()[0]) == 1


def test_write_macro_persists_source():
    """M6 — a crystallized research macro carries its matured source (deref-able
    'research via <source>') in reasoning_records.source."""
    with _store() as s:
        ok = s.write_macro(reasoning_id="research::g", goal_class="g",
                           action="research", signature=[], b_i=2.0, c=1.0,
                           time_cost=1.0, use_count=2,
                           source="https://api.x/sol")
        assert ok
        row = s._db.execute(
            "SELECT source FROM reasoning_records WHERE reasoning_id='research::g'"
        ).fetchone()
        assert row is not None and row[0] == "https://api.x/sol"


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
        assert s.record_research_recipe(goal_class="", source="x", epoch=1.0) == (0, "")
        assert s.record_research_recipe(goal_class="g", source="", epoch=1.0) == (0, "")


# ── M0/M1 — emergent-epoch volatility helpers (the E.2-shared durability core) ──

def test_age_epochs_and_grandfather():
    # age = now - created
    assert age_epochs(100.0, 517.0) == 417.0
    # legacy / unstamped (created<=0) → grandfather (age 0, never stale)
    assert age_epochs(0.0, 999999.0) == 0.0
    # clock-backwards never yields negative
    assert age_epochs(500.0, 100.0) == 0.0


def test_is_stale_at_lifetime():
    lt = DEFAULT_VOLATILE_LIFETIME_EPOCHS  # 417
    assert is_stale(1000.0, 1000.0 + lt) is True       # exactly at half-life → stale
    assert is_stale(1000.0, 1000.0 + lt - 1) is False  # one short → fresh
    assert is_stale(0.0, 10_000_000.0) is False        # grandfathered → never stale


def test_freshness_weight_ramps_to_zero():
    lt = 400.0
    assert freshness_weight(100.0, 100.0, lt) == 1.0          # birth → 1.0
    assert freshness_weight(100.0, 300.0, lt) == pytest.approx(0.5)  # half-life/2
    assert freshness_weight(100.0, 500.0, lt) == 0.0          # at lifetime → 0
    assert freshness_weight(0.0, 9_999.0, lt) == 1.0          # grandfathered → 1.0


def test_last_used_epoch_updates():
    with _store() as s:
        s.record_research_recipe(goal_class="g", source="s", epoch=5.0)
        s.record_research_recipe(goal_class="g", source="s", epoch=99.0)
        row = s._db.execute(
            "SELECT success_count, last_used_epoch FROM research_recipes "
            "WHERE goal_class='g' AND source='s'").fetchone()
        assert int(row[0]) == 2 and float(row[1]) == 99.0
