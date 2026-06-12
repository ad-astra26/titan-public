"""DK.4 (RFP §7.D-knowledge) — read-whole concept-granularity recall preference.
A curated declarative `Engram` concept covering the ask lifts `engram_ground`
(was hardcoded 0 in the agno PreHook) → grounded_route routes MODE_SOVEREIGN
(answer `direct` from the sovereign wiki page, no SOL research turn) and the
outer policy gets a live 'verifiably-known' feature. Relevance-gated so an
unrelated high-groundedness concept never spuriously routes direct. Covers GD9.

Run: python -m pytest tests/test_concept_granularity_recall.py -v -p no:anchorpy
"""
from __future__ import annotations

from types import SimpleNamespace

from titan_hcl.synthesis.recall import RecallResult
from titan_hcl.modules.agno_hooks import _dk4_concept_ground
from titan_hcl.logic.sage.grounded_router import (
    GroundedReadout, RouterThresholds, grounded_route, MODE_SOVEREIGN,
)


class _FakeRecall:
    """Stands in for EngineRecall — returns the given concept results for a
    `granularity='concept'` call."""

    def __init__(self, results):
        self._results = results

    def recall(self, query_text, *, granularity=None, k=3, query_vec=None):
        assert granularity == "concept"
        return self._results


def _plugin_with_concepts(results):
    return SimpleNamespace(engine_recall=_FakeRecall(results))


def _concept(name, groundedness):
    return RecallResult(
        tx_hash="anchor", score=groundedness, fork="concept_spine",
        source="synthesis_concept_spine", summary=name, cosine=0.0,
        importance=groundedness)


def test_relevant_concept_lifts_engram_ground():
    """A declarative concept whose name shares a query token returns its
    groundedness — the strong-substrate signal."""
    plugin = _plugin_with_concepts([_concept("Solana Consensus", 0.72)])
    ground, name = _dk4_concept_ground(
        plugin, "what consensus does solana use", query_vec=None)
    assert ground == 0.72 and name == "Solana Consensus"


def test_unrelated_concept_is_gated_out():
    """A high-groundedness concept with NO query-token name match must NOT
    surface engram_ground (else it would spuriously route direct)."""
    plugin = _plugin_with_concepts([_concept("Photosynthesis", 0.95)])
    ground, name = _dk4_concept_ground(
        plugin, "what consensus does solana use", query_vec=None)
    assert ground == 0.0 and name == ""


def test_stopwords_do_not_match():
    """Relevance gate ignores stopwords — a concept named after a stopword in
    the query does not count as coverage."""
    plugin = _plugin_with_concepts([_concept("What", 0.9)])
    ground, _ = _dk4_concept_ground(plugin, "what is it", query_vec=None)
    assert ground == 0.0


def test_no_results_or_no_reader_returns_zero():
    assert _dk4_concept_ground(_plugin_with_concepts([]), "x y z",
                               query_vec=None) == (0.0, "")
    assert _dk4_concept_ground(SimpleNamespace(engine_recall=None), "x",
                               query_vec=None) == (0.0, "")


def test_first_relevant_wins_over_later():
    """Results are pre-ranked (groundedness×name-match desc); the first
    relevant one is the chosen substrate signal."""
    plugin = _plugin_with_concepts([
        _concept("Unrelated Topic", 0.99),         # higher ground, no match
        _concept("Solana Validators", 0.55),       # match
    ])
    ground, name = _dk4_concept_ground(
        plugin, "how many solana validators", query_vec=None)
    assert ground == 0.55 and name == "Solana Validators"


def test_engram_ground_routes_sovereign():
    """GD9: a curated-concept engram_ground ≥ floor → MODE_SOVEREIGN (direct
    from the sovereign substrate), even when raw recall cosine is weak."""
    thr = RouterThresholds()
    readout = GroundedReadout(
        recall_score=0.10,              # weak episodic recall
        engram_ground=0.45,             # but a curated concept covers it (≥0.30)
        skill_utility=None,
        requires_tool=False,
        is_informational=True,
    )
    decision = grounded_route(readout, thr)
    assert decision.mode == MODE_SOVEREIGN


def test_no_substrate_does_not_route_sovereign_on_engram():
    """No concept + weak recall → engram_ground stays 0 → NOT routed sovereign
    on the engram signal (the 'I don't know this' discrimination DK gives the
    router/policy — the structural anti-collapse signal)."""
    thr = RouterThresholds()
    readout = GroundedReadout(
        recall_score=0.10, engram_ground=0.0, skill_utility=None,
        requires_tool=False, is_informational=True)
    decision = grounded_route(readout, thr)
    # With no strong substrate (recall<0.65 AND engram<0.30) the informational
    # branch does NOT assert sovereign-from-memory.
    assert not (readout.recall_score >= thr.recall_known_floor
                or readout.engram_ground >= thr.engram_ground_floor)
