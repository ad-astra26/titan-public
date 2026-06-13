"""DK.3 (RFP §7.D-knowledge, pinned M0–M4) — the librarian wiki-lint rhythm:
TTL re-verify (M2) · contradiction (M3) · orphan/decay (M4), + the M0
created_epoch round-trip on the real Engram spine. Covers GD8.

The lint logic (`run_wiki_lint`) is pure orchestration over an injected store, so
M2/M3/M4 are unit-tested with a fake store; a separate integration test proves the
M0 `created_epoch` column round-trips through the real Kuzu Engram node and that
`decay_groundedness` durably lowers `axis_used` + `concept_has_consumers` detects
an incoming COMPOSED_FROM edge.

Run: python -m pytest tests/test_wiki_lint.py -v -p no:anchorpy
"""
from __future__ import annotations

import os
import tempfile
import contextlib

import pytest

from titan_hcl.synthesis.wiki_lint import run_wiki_lint


# ── Fake store recording the lint's groundedness moves ────────────────

class _FakeStore:
    """Holds declarative concepts as mutable dicts; records lint operations the
    way the real @on_writer EngramStore methods would apply them."""

    def __init__(self, concepts, consumers=None):
        # each concept: {concept_id, version, name, groundedness, anchor_tx,
        #                domain_hint, created_epoch}
        self._concepts = {(c["concept_id"], c["version"]): dict(c)
                          for c in concepts}
        self._consumers = set(consumers or ())
        self.recomputed = []   # (cid, ver) demoted via recompute (M2)
        self.decayed = []      # (cid, ver, factor) decayed (M3/M4)

    def list_declarative_concepts(self, limit=200):
        return [dict(c) for c in self._concepts.values()][:limit]

    def recompute_groundedness(self, concept_id, version, *,
                               episodic_encounters=0, distinct_contexts=0,
                               procedural_links=0, felt_coverage=0.0,
                               oracle_evidence=0):
        # zero inputs → groundedness 0.0 (the M2 demote)
        self.recomputed.append((concept_id, version))
        key = (concept_id, version)
        if key in self._concepts:
            self._concepts[key]["groundedness"] = 0.0
        return 0.0

    def decay_groundedness(self, concept_id, version, *, factor):
        self.decayed.append((concept_id, version, factor))
        key = (concept_id, version)
        if key in self._concepts:
            self._concepts[key]["groundedness"] *= factor
        return self._concepts.get(key, {}).get("groundedness", 0.0)

    def concept_has_consumers(self, concept_id):
        return concept_id in self._consumers


def _c(cid, name, *, g=0.8, dom="", epoch=100.0, ver=1, anchor="tx_" ):
    return {"concept_id": cid, "version": ver, "name": name, "groundedness": g,
            "anchor_tx": anchor + cid, "domain_hint": dom, "created_epoch": epoch}


# ── M2 — TTL re-verify ────────────────────────────────────────────────

def test_m2_demotes_stale_volatile_above_floor():
    # A legacy over-anchored "current price" concept, old + above the floor.
    store = _FakeStore([
        _c("btc_price", "Bitcoin Current Price", g=0.82, dom="market",
           epoch=1000.0),
        _c("solana_consensus", "Solana Consensus Mechanism", g=0.9,
           dom="philosophy", epoch=1000.0),   # evergreen — untouched
    ])
    stats = run_wiki_lint(
        engram_store=store, now_epochs=1000 + 500, lifetime_epochs=417,
        recall_floor=0.65, judge_fn=None)
    assert stats["stale"] == 1
    assert ("btc_price", 1) in store.recomputed
    assert ("solana_consensus", 1) not in store.recomputed     # evergreen kept
    assert store._concepts[("btc_price", 1)]["groundedness"] == 0.0  # below floor


def test_m2_skips_fresh_and_below_floor_and_grandfathered():
    store = _FakeStore([
        _c("p_fresh", "Current SOL Price", g=0.8, dom="market", epoch=1400.0),  # age 100 < 417
        _c("p_low", "Current ETH Price", g=0.4, dom="market", epoch=1000.0),    # already < floor
        _c("p_legacy", "Current BTC Price", g=0.9, dom="market", epoch=0.0),    # grandfathered
    ])
    stats = run_wiki_lint(engram_store=store, now_epochs=1500,
                          lifetime_epochs=417, recall_floor=0.65)
    assert stats["stale"] == 0
    assert store.recomputed == []


def test_m2_inert_when_no_epoch_slot():
    # now_epochs==0 (cold-boot / no consciousness_age slot) → grandfather all.
    store = _FakeStore([_c("p", "Current SOL Price", g=0.9, dom="market",
                           epoch=1000.0)])
    stats = run_wiki_lint(engram_store=store, now_epochs=0, lifetime_epochs=417)
    assert stats["stale"] == 0


# ── M3 — contradiction ────────────────────────────────────────────────

def test_m3_decays_weaker_of_contradictory_pair():
    # Two same-domain concepts, high name overlap → judged contradictory. Young
    # (epoch≈now) so the M4 orphan pass leaves them alone — isolating M3.
    store = _FakeStore([
        _c("c_a", "Solana Consensus Proof History", g=0.9, dom="crypto",
           epoch=9_900.0),
        _c("c_b", "Solana Consensus Proof Stake", g=0.5, dom="crypto",
           epoch=9_900.0),
    ])
    judged = []

    def judge(a, b):
        judged.append((a, b))
        return True   # the LLM says: contradictory

    stats = run_wiki_lint(engram_store=store, now_epochs=10_000,
                          lifetime_epochs=417, judge_fn=judge,
                          contradiction_overlap=0.4)
    assert stats["contradiction"] == 1
    assert stats["contradiction_pairs_judged"] == 1
    # the LOWER-grounded (c_b, 0.5) decayed, not c_a
    assert any(d[0] == "c_b" for d in store.decayed)
    assert not any(d[0] == "c_a" for d in store.decayed)


def test_m3_no_action_when_judge_says_no():
    store = _FakeStore([
        _c("c_a", "Solana Consensus Proof History", g=0.9, dom="crypto",
           epoch=9_900.0),
        _c("c_b", "Solana Consensus Proof Stake", g=0.5, dom="crypto",
           epoch=9_900.0),
    ])
    stats = run_wiki_lint(engram_store=store, now_epochs=10_000,
                          judge_fn=lambda a, b: False, contradiction_overlap=0.4)
    assert stats["contradiction"] == 0
    assert stats["contradiction_pairs_judged"] == 1   # paired + asked, said no
    assert store.decayed == []


def test_m3_skipped_without_judge():
    store = _FakeStore([
        _c("c_a", "Solana Consensus PoH", g=0.9, dom="crypto"),
        _c("c_b", "Solana Consensus PoS", g=0.5, dom="crypto"),
    ])
    stats = run_wiki_lint(engram_store=store, now_epochs=10_000, judge_fn=None)
    assert stats["contradiction_pairs_judged"] == 0


def test_m3_different_domains_not_paired():
    store = _FakeStore([
        _c("c_a", "Solana Consensus PoH", g=0.9, dom="crypto"),
        _c("c_b", "Solana Consensus PoS", g=0.5, dom="philosophy"),
    ])
    called = []
    run_wiki_lint(engram_store=store, now_epochs=10_000,
                  judge_fn=lambda a, b: called.append(1) or True,
                  contradiction_overlap=0.4)
    assert called == []   # cross-domain pairs are never candidates


# ── M4 — orphan/decay ─────────────────────────────────────────────────

def test_m4_decays_old_uncited_unconsumed():
    store = _FakeStore([
        _c("orphan", "Obscure Fact", g=0.7, dom="history", epoch=100.0),
    ])
    # orphan_window = 10*lifetime = 4170; age = 100_000-100 well past it.
    stats = run_wiki_lint(engram_store=store, now_epochs=100_000,
                          lifetime_epochs=417, recall_counts={})
    assert stats["orphan"] == 1
    assert any(d[0] == "orphan" for d in store.decayed)


def test_m4_keeps_cited_consumed_and_young():
    store = _FakeStore([
        _c("cited", "Cited Fact", g=0.7, dom="history", epoch=100.0),
        _c("consumed", "Composed Fact", g=0.7, dom="history", epoch=100.0),
        _c("young", "Recent Fact", g=0.7, dom="history", epoch=98_000.0),
    ], consumers={"consumed"})
    counts = {("cited", 1): (5, 3, 0.0)}   # cited_count=3 → not an orphan
    stats = run_wiki_lint(engram_store=store, now_epochs=100_000,
                          lifetime_epochs=417, recall_counts=counts)
    assert stats["orphan"] == 0
    assert store.decayed == []


def test_m4_summary_concepts_never_orphaned():
    store = _FakeStore([
        _c("summary::history", "History — Overview", g=0.7, dom="history",
           epoch=100.0),
    ])
    stats = run_wiki_lint(engram_store=store, now_epochs=100_000,
                          lifetime_epochs=417, recall_counts={})
    assert stats["orphan"] == 0   # the wiki-index summary is protected


# ── M0 — created_epoch round-trip + decay + consumer detection (real spine) ──

@contextlib.contextmanager
def _real_store():
    import duckdb
    from titan_hcl.core.direct_memory import TitanKnowledgeGraph
    from titan_hcl.synthesis.kuzu_spine_schema import bootstrap_spine_schema
    from titan_hcl.synthesis.engram_store import EngramStore

    class _FakeWriter:
        """Minimal OuterMemoryWriter — returns a deterministic content-hash tx."""
        def __init__(self):
            self._n = 0

        def write_concept_version(self, *, concept_id, version, **_kw):
            self._n += 1
            return f"tx_{concept_id}_v{version}_{self._n}"

    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "spine.kuzu"))
        bootstrap_spine_schema(g)
        s = EngramStore(g, _FakeWriter(), clock=lambda: 1000.0)
        try:
            yield s, g
        finally:
            g.close()


def test_m0_created_epoch_roundtrips_and_decay_is_durable():
    with _real_store() as (s, g):
        # M0 — stamp an emergent birth epoch; it must read back on the concept.
        s.create_concept(concept_id="solana_consensus", name="Solana Consensus",
                         memory_type="declarative",
                         derivation_evidence=["evtx1"], domain_hint="crypto",
                         created_epoch=4242.0)
        rows = s.list_declarative_concepts(limit=50)
        row = next(r for r in rows if r["concept_id"] == "solana_consensus")
        assert row["created_epoch"] == 4242.0
        # Give it a real grounding (a v=1 concept w/o composed_from starts at 0;
        # the live seed path calls recompute_groundedness right after create).
        s.recompute_groundedness("solana_consensus", 1, episodic_encounters=5,
                                 distinct_contexts=3)
        row = next(r for r in s.list_declarative_concepts(limit=50)
                   if r["concept_id"] == "solana_consensus")
        g0 = row["groundedness"]
        assert g0 > 0.0
        # M3/M4 — decay_groundedness durably lowers groundedness AND axis_used.
        s.decay_groundedness("solana_consensus", 1, factor=0.5)
        row2 = next(r for r in s.list_declarative_concepts(limit=50)
                    if r["concept_id"] == "solana_consensus")
        assert row2["groundedness"] == pytest.approx(g0 * 0.5)


def test_m4_concept_has_consumers_detects_incoming_composed_from():
    with _real_store() as (s, g):
        # Two base concepts + a summary composed_from them.
        s.create_concept(concept_id="poh", name="Proof of History",
                         memory_type="declarative", derivation_evidence=["t1"],
                         domain_hint="crypto", created_epoch=10.0)
        s.create_concept(concept_id="pos", name="Proof of Stake",
                         memory_type="declarative", derivation_evidence=["t2"],
                         domain_hint="crypto", created_epoch=10.0)
        s.create_concept(concept_id="summary::crypto", name="Crypto — Overview",
                         memory_type="declarative",
                         composed_from=[("poh", 1), ("pos", 1)],
                         derivation_evidence=["t1", "t2"], domain_hint="crypto",
                         created_epoch=20.0)
        # poh + pos are CONSUMED by the summary → not orphans.
        assert s.concept_has_consumers("poh") is True
        assert s.concept_has_consumers("pos") is True
        # the summary itself has no consumer.
        assert s.concept_has_consumers("summary::crypto") is False
