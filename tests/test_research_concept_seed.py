"""DK.1 (RFP §7.D-knowledge) — the sovereign LLM-Wiki research→declarative-concept
seed. A confirmed `acquired:research` finding becomes/refines an anchored
`memory_type='declarative'` Engram concept, continuously (not dream-gated), with
the LLM as librarian-never-author. Pure-seeder tests with fakes (the bus wiring
is exercised by the worker handlers); covers GD6 + GD10.

Run: python -m pytest tests/test_research_concept_seed.py -v -p no:anchorpy
"""
from __future__ import annotations

from titan_hcl.synthesis.engram_store import Engram
from titan_hcl.synthesis.research_wiki import (
    fallback_concept_name,
    make_research_name_fn,
    seed_research_concept,
)
from titan_hcl.synthesis.consolidation import LLMProposal


class _FakeCGNBridge:
    def __init__(self):
        self.registered = []

    def register_spine_concept(self, concept_id, name, seed_consumer="synthesis_engine"):
        self.registered.append((concept_id, name, seed_consumer))
        return True


class _FakeEngramStore:
    """Records create/bump/groundedness calls; `_existing` simulates the dedup
    probe (None → create path; a row → bump path)."""

    def __init__(self, existing=None):
        self._existing = existing
        self.created = []
        self.bumped = []
        self.grounded = []
        self._version = 1 if existing is None else int(existing["version"]) + 1

    def latest_concept(self, concept_id):
        return self._existing

    def create_concept(self, *, concept_id, name, memory_type,
                       derivation_evidence=None, composed_from=None,
                       domain_hint=""):
        self.created.append(dict(
            concept_id=concept_id, name=name, memory_type=memory_type,
            derivation_evidence=list(derivation_evidence or []),
            domain_hint=domain_hint))
        return Engram(concept_id, 1, name, memory_type, 0.1, "anchor_tx_v1", 0.0)

    def bump_version(self, *, concept_id, derivation_evidence=None,
                     composed_from=None, domain_hint=""):
        self.bumped.append(dict(
            concept_id=concept_id,
            derivation_evidence=list(derivation_evidence or []),
            domain_hint=domain_hint))
        return Engram(concept_id, self._version,
                      self._existing["name"], "declarative", 0.2,
                      "anchor_tx_v%d" % self._version, 0.0)

    def recompute_groundedness(self, concept_id, version, *,
                              episodic_encounters=0, distinct_contexts=0,
                              procedural_links=0, felt_coverage=0.0,
                              oracle_evidence=0):
        self.grounded.append(dict(
            concept_id=concept_id, version=version,
            episodic_encounters=episodic_encounters,
            distinct_contexts=distinct_contexts,
            felt_coverage=felt_coverage))
        return 0.3


def _name_fn(_content):
    return ("solana_consensus", "Solana Consensus", "blockchain")


def test_solo_research_finding_seeds_declarative_concept():
    """GD6: a solo confirmed research finding becomes a declarative Engram
    (anchored), even with no sibling cluster."""
    store = _FakeEngramStore(existing=None)
    cgn = _FakeCGNBridge()
    cv = seed_research_concept(
        engram_store=store, cgn_bridge=cgn,
        tx_hash="0xresearchtx", content="Solana uses Proof of History.",
        name_fn=_name_fn)

    assert cv is not None and cv.concept_id == "solana_consensus"
    assert len(store.created) == 1
    created = store.created[0]
    # memory_type IS the idea_type — a research fact is declarative.
    assert created["memory_type"] == "declarative"
    # GD10 sovereignty: derivation_evidence cites the verified research tx (a
    # deref target → INV-OML-10), NOT LLM parametric knowledge.
    assert created["derivation_evidence"] == ["0xresearchtx"]
    assert created["domain_hint"] == "blockchain"
    # CGN registration happened (concept_id known to the grounding authority).
    assert cgn.registered and cgn.registered[0][0] == "solana_consensus"
    # groundedness recomputed so DK.4 recall can surface the fresh concept.
    assert store.grounded and store.grounded[0]["episodic_encounters"] == 1


def test_repeat_finding_refines_via_bump_not_overwrite():
    """INV-OML-5 mutate-not-update: a 2nd finding for an existing concept_id
    bumps a successor version, never a fresh create (refinement, the
    compounding 'wiki page' growing)."""
    existing = {"concept_id": "solana_consensus", "version": 1,
                "name": "Solana Consensus", "memory_type": "declarative",
                "anchor_tx": "anchor_tx_v1"}
    store = _FakeEngramStore(existing=existing)
    cv = seed_research_concept(
        engram_store=store, cgn_bridge=_FakeCGNBridge(),
        tx_hash="0xresearchtx2", content="Solana consensus combines PoH and PoS.",
        name_fn=_name_fn)

    assert cv is not None and cv.version == 2
    assert not store.created                  # never a fresh create
    assert len(store.bumped) == 1
    assert store.bumped[0]["derivation_evidence"] == ["0xresearchtx2"]


def test_empty_tx_hash_refused_no_deref_target():
    """INV-OML-10: no anchored evidence → no concept (a wiki hit must always
    deref to a real chain record; never seed an unverifiable assertion)."""
    store = _FakeEngramStore(existing=None)
    cv = seed_research_concept(
        engram_store=store, cgn_bridge=_FakeCGNBridge(),
        tx_hash="", content="some fact", name_fn=_name_fn)
    assert cv is None
    assert not store.created and not store.bumped


def test_llm_reject_falls_back_to_deterministic_name():
    """The librarian only NAMES; a naming failure must NOT drop verified
    knowledge. A rejecting proposer → deterministic fallback id, fact still
    persists."""
    def _reject_propose(_cluster):
        return LLMProposal(action="reject", reason="noise")

    name_fn = make_research_name_fn(_reject_propose)
    store = _FakeEngramStore(existing=None)
    cv = seed_research_concept(
        engram_store=store, cgn_bridge=_FakeCGNBridge(),
        tx_hash="0xtx", content="What consensus does Solana use",
        name_fn=name_fn)
    assert cv is not None
    # Deterministic slug from salient tokens (stopwords dropped).
    assert store.created[0]["concept_id"].startswith("research_")
    assert "consensus" in store.created[0]["concept_id"]


def test_llm_name_used_when_proposer_succeeds():
    """The librarian names the concept over verified content (the Karpathy
    'curate' discipline), forced to declarative regardless of LLM memory_type."""
    def _propose(_cluster):
        return LLMProposal(action="new_concept", concept_id="photosynthesis",
                           proposed_name="Photosynthesis", memory_type="meta",
                           domain_hint="biology")

    name_fn = make_research_name_fn(_propose)
    store = _FakeEngramStore(existing=None)
    cv = seed_research_concept(
        engram_store=store, cgn_bridge=_FakeCGNBridge(),
        tx_hash="0xtx", content="Plants convert light to energy.",
        name_fn=name_fn)
    assert cv is not None and cv.concept_id == "photosynthesis"
    assert store.created[0]["memory_type"] == "declarative"  # forced
    assert store.created[0]["domain_hint"] == "biology"


def test_fallback_concept_name_deterministic():
    cid1, name1 = fallback_concept_name("What consensus does Solana use?")
    cid2, name2 = fallback_concept_name("What consensus does Solana use?")
    assert cid1 == cid2 and name1 == name2
    assert cid1.startswith("research_") and "consensus" in cid1
    # Empty content → stable generic, never raises.
    assert fallback_concept_name("") == ("research_finding", "Research Finding")
