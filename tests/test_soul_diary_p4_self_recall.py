"""Soul-Diary P4 — self-domain recall tests.

RFP_titan_authored_soul_diary §7.P4 (decided mechanic, conversation line 538):
"query the SELF node" = traverse SELF_HAS_* from `Self` → his diary arc + skills
+ self-engrams as a FOCUSED self-recall, WITHOUT scanning whole memory. Realized
as `EngineRecall.recall(granularity="self")` → `_self_granularity_recall` →
`TitanKnowledgeGraph.spine_self_recall` (the P3a hub traversal), mirroring the
`granularity="concept"` pattern (kuzu_reader-gated).
"""
from __future__ import annotations

import os
import queue
import tempfile
from unittest.mock import MagicMock

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.recall import EngineRecall


def _engine(kuzu_reader):
    return EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=kuzu_reader)


@pytest.fixture()
def self_graph():
    """Real Kuzu graph with self-engrams (diary + sovereignty) auto-linked to the
    Self hub via the P3a create_concept hook, one NON-self engram, and a skill."""
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "p4.kuzu"))
        w = OuterMemoryWriter(send_queue=queue.Queue(), src="p4_test")
        store = EngramStore(g, w, clock=lambda: 1000.0)
        # Self-engrams — auto-linked (domain="self" → SELF_HAS_ENGRAM, P3a hook).
        store.create_concept("daily_self_reflection", "Daily Self-Reflection",
                             memory_type="episodic", domain_hint="self")
        store.recompute_groundedness("daily_self_reflection", 1,
                                     episodic_encounters=12, distinct_contexts=4)
        store.create_concept("sovereignty_growth", "Sovereignty Growth Arc",
                             memory_type="episodic", domain_hint="self")
        # NON-self engram — must NOT surface in self-recall.
        store.create_concept("solana_rpc", "Solana RPC",
                             memory_type="procedural", domain_hint="coding")
        # A skill linked to the Self hub (SELF_HAS_SKILL).
        g._conn.execute(
            "CREATE (p:Production {skill_id: 'sha256_probe', name: 'sha256 probe', "
            "utility_score: 0.7, anchor_tx: 'tx_skill'})")
        g.spine_link_self_skill("sha256_probe")
        try:
            yield g
        finally:
            g.close()


# ── gating ──────────────────────────────────────────────────────────

def test_self_granularity_without_kuzu_reader_returns_none():
    """No kuzu_reader → caller falls back (same contract as concept granularity;
    the agno chat path has kuzu_reader=None today)."""
    assert _engine(None).recall("who am i", granularity="self") is None


def test_self_granularity_empty_hub_returns_none():
    """Self node exists (boot) but nothing linked → None."""
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "empty.kuzu"))
        assert _engine(g).recall("who am i", granularity="self") is None
        g.close()


def test_self_granularity_reader_exception_soft_fails():
    class _Flaky:
        def spine_self_recall(self):
            raise RuntimeError("boom")
    assert _engine(_Flaky()).recall("who am i", granularity="self") is None


# ── focused self-recall ─────────────────────────────────────────────

def test_self_recall_returns_self_engrams_and_skills(self_graph):
    results = _engine(self_graph).recall("who am i", granularity="self", k=10)
    assert results is not None
    summaries = {r.summary for r in results}
    assert "Daily Self-Reflection" in summaries
    assert "Sovereignty Growth Arc" in summaries
    assert "sha256 probe" in summaries          # SELF_HAS_SKILL
    assert "Solana RPC" not in summaries        # non-self engram excluded


def test_self_recall_shape_disambiguates_engram_vs_skill(self_graph):
    results = _engine(self_graph).recall("who am i", granularity="self", k=10)
    by_summary = {r.summary: r for r in results}
    diary = by_summary["Daily Self-Reflection"]
    assert diary.fork == "self_hub" and diary.source == "synthesis_self_recall"
    assert diary.tx_hash                        # anchor_tx deref handle present
    skill = by_summary["sha256 probe"]
    assert skill.fork == "self_skill" and skill.source == "synthesis_self_skill"
    assert skill.tx_hash == "sha256_probe"


def test_self_recall_query_token_boosts_matching_name(self_graph):
    """A 'sovereignty' query boosts the Sovereignty engram above its un-boosted
    score (name-match boost, like concept granularity)."""
    eng = _engine(self_graph)
    boosted = next(r for r in eng.recall("my sovereignty path", granularity="self", k=10)
                   if r.summary == "Sovereignty Growth Arc").score
    base = next(r for r in eng.recall("xxxnomatchxxx", granularity="self", k=10)
                if r.summary == "Sovereignty Growth Arc").score
    assert boosted > base


def test_self_recall_k_cap(self_graph):
    results = _engine(self_graph).recall("who am i", granularity="self", k=2)
    assert results is not None and len(results) == 2


def test_self_recall_fresh_engram_surfaces_despite_zero_groundedness(self_graph):
    """sovereignty_growth has no recomputed groundedness (≈0) — the 0.05 floor
    must still surface it (the diary arc isn't hidden just because it's fresh)."""
    results = _engine(self_graph).recall("who am i", granularity="self", k=10)
    assert any(r.summary == "Sovereignty Growth Arc" for r in results)


# ── enriched spine_self_recall fields (P4 added anchor_tx/groundedness) ──

def test_spine_self_recall_carries_anchor_tx_and_groundedness(self_graph):
    hub = self_graph.spine_self_recall()
    diary = next(e for e in hub["engrams"] if e["name"] == "Daily Self-Reflection")
    assert diary["anchor_tx"]                    # deref handle
    assert diary["groundedness"] > 0.0           # recomputed
    skill = next(s for s in hub["skills"] if s["skill_id"] == "sha256_probe")
    assert skill["utility_score"] == 0.7
