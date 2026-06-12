"""DK.2 (RFP §7.D-knowledge) — concept-of-concepts (the "wiki index"). The
consolidation/curation survey composes related declarative `Engram` concepts
(grouped by domain_hint) into a SUMMARY `Engram` whose composed_from = the child
concepts. Idempotent via the `summary::` id convention (existing → bump); summary
concepts are excluded from the base set (no infinite re-summarization); the LLM
only names (never authors). Covers GD7 + the GD10 librarian gate.

Run: python -m pytest tests/test_concept_of_concepts.py -v -p no:anchorpy
"""
from __future__ import annotations

from titan_hcl.synthesis.engram_store import Engram
from titan_hcl.synthesis.research_wiki import compose_concept_summaries


class _FakeCGNBridge:
    def __init__(self):
        self.registered = []

    def register_spine_concept(self, concept_id, name, seed_consumer="synthesis_engine"):
        self.registered.append((concept_id, name))
        return True


class _FakeEngramStore:
    def __init__(self, concepts, existing_summaries=None):
        self._concepts = concepts
        self._existing = existing_summaries or {}
        self.created = []
        self.bumped = []

    def list_declarative_concepts(self, limit=200):
        return list(self._concepts)

    def latest_concept(self, concept_id):
        return self._existing.get(concept_id)

    def create_concept(self, *, concept_id, name, memory_type,
                       composed_from=None, derivation_evidence=None, domain_hint=""):
        self.created.append(dict(
            concept_id=concept_id, name=name, memory_type=memory_type,
            composed_from=list(composed_from or []), domain_hint=domain_hint))
        return Engram(concept_id, 1, name, memory_type, 0.4, "atx", 0.0)

    def bump_version(self, *, concept_id, composed_from=None,
                     derivation_evidence=None, domain_hint=""):
        self.bumped.append(dict(
            concept_id=concept_id, composed_from=list(composed_from or [])))
        return Engram(concept_id, 2, "x", "declarative", 0.4, "atx2", 0.0)


def _c(cid, ver, domain, ground=0.5, anchor="tx"):
    return {"concept_id": cid, "version": ver, "name": cid.title(),
            "groundedness": ground, "anchor_tx": anchor, "domain_hint": domain}


def test_two_base_concepts_in_domain_compose_summary():
    """GD7: ≥2 declarative concepts sharing a domain → a summary Engram
    composed_from them."""
    store = _FakeEngramStore([
        _c("solana_consensus", 1, "blockchain", anchor="tx_a"),
        _c("solana_validators", 1, "blockchain", anchor="tx_b"),
    ])
    n = compose_concept_summaries(engram_store=store, cgn_bridge=_FakeCGNBridge())
    assert n == 1 and len(store.created) == 1
    s = store.created[0]
    assert s["concept_id"] == "summary::blockchain"
    assert s["memory_type"] == "declarative"
    assert set(s["composed_from"]) == {("solana_consensus", 1), ("solana_validators", 1)}
    assert s["domain_hint"] == "blockchain"


def test_existing_summary_is_bumped_not_recreated():
    """INV-OML-5: a domain that already has a summary refreshes it via bump
    (children may have grown), never a duplicate create."""
    store = _FakeEngramStore(
        concepts=[_c("a", 1, "bio"), _c("b", 1, "bio"), _c("d", 2, "bio")],
        existing_summaries={"summary::bio": {"concept_id": "summary::bio",
                                             "version": 1, "name": "Bio"}})
    n = compose_concept_summaries(engram_store=store, cgn_bridge=_FakeCGNBridge())
    assert n == 1 and not store.created and len(store.bumped) == 1
    assert store.bumped[0]["concept_id"] == "summary::bio"
    assert len(store.bumped[0]["composed_from"]) == 3


def test_summary_concepts_excluded_from_base_no_recursion():
    """A `summary::` concept is itself declarative but must NOT be treated as a
    base concept (else summaries-of-summaries spiral)."""
    store = _FakeEngramStore([
        _c("summary::blockchain", 1, "blockchain"),  # an existing summary
        _c("solana_consensus", 1, "blockchain"),
    ])
    # Only ONE real base concept in the domain (the summary is skipped) → <2 → no compose.
    n = compose_concept_summaries(engram_store=store, cgn_bridge=_FakeCGNBridge())
    assert n == 0 and not store.created and not store.bumped


def test_single_concept_domain_skipped():
    store = _FakeEngramStore([_c("lonely", 1, "niche")])
    assert compose_concept_summaries(
        engram_store=store, cgn_bridge=_FakeCGNBridge()) == 0


def test_empty_domain_hint_skipped():
    """A concept with no domain_hint can't be grouped → never composed."""
    store = _FakeEngramStore([_c("x", 1, ""), _c("y", 1, "")])
    assert compose_concept_summaries(
        engram_store=store, cgn_bridge=_FakeCGNBridge()) == 0


def test_max_summaries_per_pass_caps_work():
    concepts = []
    for dom in ("d1", "d2", "d3", "d4"):
        concepts += [_c(f"{dom}_a", 1, dom), _c(f"{dom}_b", 1, dom)]
    store = _FakeEngramStore(concepts)
    n = compose_concept_summaries(
        engram_store=store, cgn_bridge=_FakeCGNBridge(), max_summaries_per_pass=2)
    assert n == 2 and len(store.created) == 2


def test_llm_librarian_names_summary_when_provided():
    """The LLM may NAME the summary (curation); it never authors the facts (the
    children are already-verified concepts)."""
    def _name_fn(_text):
        return ("ignored_id", "Blockchain Fundamentals", "blockchain")

    store = _FakeEngramStore([_c("a", 1, "blockchain"), _c("b", 1, "blockchain")])
    compose_concept_summaries(
        engram_store=store, cgn_bridge=_FakeCGNBridge(), name_fn=_name_fn)
    assert store.created[0]["name"] == "Blockchain Fundamentals"
    # id stays the deterministic summary:: convention, NOT the LLM's id.
    assert store.created[0]["concept_id"] == "summary::blockchain"
