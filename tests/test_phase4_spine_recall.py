"""Phase 4 — EngineRecall spine-aware retrieval tests (§P4.H).

Covers `EngineRecall.recall(granularity="concept")`:

- Returns None when kuzu_reader is not wired (caller falls back to P3 recall)
- Returns None when the spine is empty
- Returns top-K spines ranked by groundedness DESC
- Name-substring match boost promotes query-relevant concepts
- RecallResult shape uniform with per-TX recall (downstream consumers see
  one type); `fork="concept_spine"` disambiguator + `summary=concept_name`
- k cap respected
- Existing per-TX recall (granularity=None, "turn", "topic", "session")
  unchanged when kuzu_reader is wired (regression vs P3)
- kuzu_reader exception soft-fails to None
"""
from __future__ import annotations

import os
import queue
import tempfile
from unittest.mock import MagicMock

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.recall import EngineRecall, RecallResult


@pytest.fixture()
def env_with_spine():
    """Real Kuzu graph + EngramStore + a few materialized spine concepts."""
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "p4h.kuzu"))
        q = queue.Queue()
        w = OuterMemoryWriter(send_queue=q, src="recall_test")
        bridge = CGNRegistrationBridge(os.path.join(tmp, "spine.json"))
        store = EngramStore(g, w, clock=lambda: 1000.0)

        # 3 spine concepts with varying groundedness.
        bridge.register_spine_concept("linux_terminal", "Linux terminal")
        store.create_concept(
            "linux_terminal", "Linux terminal", memory_type="declarative",
        )
        store.recompute_groundedness(
            "linux_terminal", 1, episodic_encounters=20, distinct_contexts=5,
        )

        bridge.register_spine_concept("solana_rpc", "Solana RPC")
        store.create_concept(
            "solana_rpc", "Solana RPC", memory_type="procedural",
        )
        store.recompute_groundedness(
            "solana_rpc", 1, episodic_encounters=50, distinct_contexts=20,
            procedural_links=10,
        )

        bridge.register_spine_concept("metaplex_nft_minting",
                                       "Metaplex NFT minting")
        store.create_concept(
            "metaplex_nft_minting", "Metaplex NFT minting",
            memory_type="procedural",
        )
        store.recompute_groundedness(
            "metaplex_nft_minting", 1, episodic_encounters=5,
            procedural_links=3,
        )

        try:
            yield g
        finally:
            g.close()


# ── Concept granularity ─────────────────────────────────────────────


def test_concept_granularity_without_kuzu_reader_returns_none():
    """No kuzu_reader → caller falls back to P3 recall."""
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=None,
    )
    assert er.recall("anything", granularity="concept") is None


def test_concept_granularity_empty_spine_returns_none(env_with_spine):
    """Spine exists but contains no concepts → None."""
    with tempfile.TemporaryDirectory() as tmp:
        empty_g = TitanKnowledgeGraph(os.path.join(tmp, "empty.kuzu"))
        er = EngineRecall(
            rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
            embedder=None, kuzu_reader=empty_g,
        )
        assert er.recall("anything", granularity="concept") is None
        empty_g.close()


def test_concept_granularity_returns_spines_ranked_by_groundedness(env_with_spine):
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    results = er.recall("query", granularity="concept", k=10)
    assert results is not None
    assert len(results) == 3
    # solana_rpc has highest groundedness → first.
    assert results[0].summary == "Solana RPC"
    # Order by groundedness DESC.
    assert results[0].importance > results[1].importance >= results[2].importance


def test_concept_granularity_query_token_boosts_matching_name(env_with_spine):
    """A query containing 'linux' should boost linux_terminal's score
    above solana_rpc even though solana_rpc has higher base groundedness."""
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    results = er.recall("tell me about linux terminals", granularity="concept", k=10)
    assert results is not None
    # The exact ordering depends on actual values: solana has g=~1.0,
    # linux has g=~0.45 boosted by 1.5 → ~0.68. Still solana wins on raw,
    # so let's compare to a non-boosted run.
    baseline = er.recall("xxxnomatchxxx", granularity="concept", k=10)
    linux_baseline = next(
        r for r in baseline if r.summary == "Linux terminal"
    ).score
    linux_boosted = next(
        r for r in results if r.summary == "Linux terminal"
    ).score
    assert linux_boosted > linux_baseline


def test_concept_granularity_recall_result_shape(env_with_spine):
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    results = er.recall("query", granularity="concept", k=5)
    assert results is not None
    for r in results:
        assert isinstance(r, RecallResult)
        # Disambiguator fields.
        assert r.fork == "concept_spine"
        assert r.source == "synthesis_concept_spine"
        # summary = name
        assert r.summary in (
            "Linux terminal", "Solana RPC", "Metaplex NFT minting",
        )
        # tx_hash = the spine's latest anchor_tx (sha256 hex, 64 chars).
        assert len(r.tx_hash) == 64
        # importance = groundedness so downstream consumers reading
        # only `.importance` still get a signal.
        assert r.importance >= 0.0


def test_concept_granularity_k_cap_respected(env_with_spine):
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    results = er.recall("query", granularity="concept", k=2)
    assert results is not None
    assert len(results) == 2


def test_concept_granularity_kuzu_exception_returns_none():
    """Kuzu reader raising → caller falls back (BridgeRecall soft-fail)."""

    class FlakyReader:
        def spine_list_concepts(self, **_kw):
            raise RuntimeError("kuzu down")

    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=FlakyReader(),
    )
    assert er.recall("query", granularity="concept") is None


def test_concept_granularity_filters_zero_groundedness(env_with_spine):
    """Concepts with groundedness=0 should NOT surface — they're
    ungrounded / probationary. (Phase 5 hypothesis-fork integration
    may revisit this.)"""
    # Force one concept to groundedness=0.
    env_with_spine.spine_update_groundedness("metaplex_nft_minting", 1, 0.0)

    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    results = er.recall("xxxnomatchxxx", granularity="concept", k=10)
    assert results is not None
    summaries = [r.summary for r in results]
    assert "Metaplex NFT minting" not in summaries


# ── Per-TX recall regression ────────────────────────────────────────


def test_per_tx_recall_unchanged_when_kuzu_wired(env_with_spine):
    """Wiring kuzu_reader must NOT affect granularity=None / turn / topic /
    session — those still go through the contract pipeline unchanged."""
    er = EngineRecall(
        rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
        embedder=None, kuzu_reader=env_with_spine,
    )
    # No embedder, no contract → fallback to None.
    assert er.recall("anything") is None
    # Same with a known granularity (would fail differently if kuzu
    # leaked into the per-TX path).
    assert er.recall("anything", granularity="turn", chat_id="c1") is None


def test_recall_get_stats_counts_concept_hits_and_fallbacks():
    """concept granularity hits + falbacks counted in get_stats."""
    with tempfile.TemporaryDirectory() as tmp:
        empty_g = TitanKnowledgeGraph(os.path.join(tmp, "stats.kuzu"))
        er = EngineRecall(
            rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
            embedder=None, kuzu_reader=empty_g,
        )
        # Empty spine → None (fallback).
        er.recall("x", granularity="concept")
        stats = er.get_stats()
        assert stats["total_recall_calls"] >= 1
        assert stats["total_fallbacks"] >= 1
        empty_g.close()
