"""Soul-Diary P3a — the SELF node (self-knowledge hub) tests.

RFP_titan_authored_soul_diary §7.P3a / INV-SD-16: the `Self` singleton + the
`SELF_HAS_ENGRAM` (diary + self-engrams) + `SELF_HAS_SKILL` (→`Production`) hub,
so "what have I learned / what can I do?" resolves in one hop. The outward-
expression rels (`SELF_HAS_EXPRESSION`) + the `Persona` node are a DEFERRED step
(persona mechanic undesigned — RFP frontmatter scope_decisions).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import EngramStore


class _FakeWriter:
    """Minimal OuterMemoryWriter stand-in — returns a deterministic tx hash."""

    def write_concept_version(self, **kw) -> str:
        return f"tx_{kw['concept_id']}_v{kw['version']}"


@pytest.fixture()
def graph():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_self.kuzu")
        g = TitanKnowledgeGraph(path)
        try:
            yield g
        finally:
            g.close()


def _mk_engram(graph, cid, name, domain_hint, ts=100.0, ver=1):
    return graph.spine_create_concept_node(
        concept_id=cid, version=ver, name=name, memory_type="episodic",
        groundedness=0.2, anchor_tx=f"tx_{cid}", created_at=ts,
        domain_hint=domain_hint)


def _mk_skill(graph, skill_id, name):
    graph._conn.execute(
        "CREATE (p:Production {skill_id: $s, name: $n, utility_score: 0.5, "
        "anchor_tx: $t})", {"s": skill_id, "n": name, "t": f"tx_{skill_id}"})


# ── Self singleton ──────────────────────────────────────────────────

def test_self_node_not_auto_created_in_generic_graph(graph):
    """The generic TitanKnowledgeGraph.__init__ must NOT create the Self node —
    spine DATA writes are synthesis-owned (G21/INV-Syn-7); a fresh graph (e.g. the
    memory worker's knowledge_graph.kuzu) gets the empty Self TABLE only, no node.
    The hub is bootstrapped in synthesis_worker's boot / lazily via the link hooks."""
    qr = graph._conn.execute("MATCH (s:Self) RETURN COUNT(s)")
    assert qr.get_next()[0] == 0


def test_ensure_self_node_creates_it(graph):
    """spine_ensure_self_node (called by synthesis boot / the link hooks) creates
    the singleton."""
    assert graph.spine_ensure_self_node() is True
    qr = graph._conn.execute("MATCH (s:Self) RETURN COUNT(s)")
    assert qr.get_next()[0] == 1


def test_ensure_self_node_idempotent(graph):
    """Re-ensuring never duplicates the singleton."""
    assert graph.spine_ensure_self_node() is True
    assert graph.spine_ensure_self_node() is True
    qr = graph._conn.execute("MATCH (s:Self) RETURN COUNT(s)")
    assert qr.get_next()[0] == 1


# ── SELF_HAS_ENGRAM ─────────────────────────────────────────────────

def test_link_self_engram_round_trips(graph):
    _mk_engram(graph, "soul_diary_20260610", "Soul-Diary Reflection", "self")
    assert graph.spine_link_self_engram("soul_diary_20260610", 1) is True
    recall = graph.spine_self_recall()
    ids = [e["concept_id"] for e in recall["engrams"]]
    assert "soul_diary_20260610" in ids


def test_link_self_engram_idempotent_single_edge(graph):
    _mk_engram(graph, "self_refactor", "Self-Refactor", "self")
    graph.spine_link_self_engram("self_refactor", 1)
    graph.spine_link_self_engram("self_refactor", 1)   # link twice
    recall = graph.spine_self_recall()
    assert sum(e["concept_id"] == "self_refactor" for e in recall["engrams"]) == 1


def test_link_missing_engram_returns_false(graph):
    assert graph.spine_link_self_engram("ghost", 1) is False


# ── SELF_HAS_SKILL ──────────────────────────────────────────────────

def test_link_self_skill_round_trips(graph):
    _mk_skill(graph, "sha256_probe", "sha256 probe")
    assert graph.spine_link_self_skill("sha256_probe") is True
    recall = graph.spine_self_recall()
    assert any(s["skill_id"] == "sha256_probe" for s in recall["skills"])


def test_link_missing_skill_returns_false(graph):
    assert graph.spine_link_self_skill("ghost_skill") is False


# ── Backfill (the boot completeness pass) ───────────────────────────

def test_backfill_links_only_self_engrams_plus_all_skills(graph):
    _mk_engram(graph, "self_a", "Self A", "self", ts=300.0)
    _mk_engram(graph, "self_b", "Self B", "self", ts=200.0)
    _mk_engram(graph, "philosophy_x", "Philosophy X", "philosophy", ts=100.0)
    _mk_skill(graph, "skill_a", "Skill A")
    out = graph.spine_backfill_self_links()
    assert out["engrams"] == 2 and out["skills"] == 1
    recall = graph.spine_self_recall()
    eids = {e["concept_id"] for e in recall["engrams"]}
    assert eids == {"self_a", "self_b"}                 # NOT philosophy_x
    assert recall["engrams"][0]["concept_id"] == "self_a"   # newest-first (300>200)
    assert {s["skill_id"] for s in recall["skills"]} == {"skill_a"}


def test_backfill_idempotent(graph):
    _mk_engram(graph, "self_a", "Self A", "self")
    graph.spine_backfill_self_links()
    graph.spine_backfill_self_links()                   # run again
    recall = graph.spine_self_recall()
    assert sum(e["concept_id"] == "self_a" for e in recall["engrams"]) == 1


# ── create_concept hook (the live consolidation path) ───────────────

def test_create_concept_self_domain_links_to_hub(graph):
    """A domain="self" engram created via EngramStore.create_concept is linked
    to the Self hub automatically (the §7.P3a hook)."""
    store = EngramStore(graph, _FakeWriter(), clock=lambda: 1000.0)
    store.create_concept(concept_id="daily_self_reflection",
                         name="Daily Self-Reflection", memory_type="episodic",
                         domain_hint="self")
    recall = graph.spine_self_recall()
    assert any(e["concept_id"] == "daily_self_reflection"
               for e in recall["engrams"])


def test_create_concept_non_self_domain_not_linked(graph):
    """A non-self engram is NOT linked to the Self hub (keeps it clean)."""
    store = EngramStore(graph, _FakeWriter(), clock=lambda: 1000.0)
    store.create_concept(concept_id="solana_rpc", name="Solana RPC",
                         memory_type="procedural", domain_hint="coding")
    recall = graph.spine_self_recall()
    assert all(e["concept_id"] != "solana_rpc" for e in recall["engrams"])
