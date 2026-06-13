"""Tests for MakerStore — Titan's persistent, sovereign model of his Maker
(RFP_missions_and_the_maker_model §7.1).

Covers the DuckDB scalar path (graph=None — the convention; record/recall/dynamic-
placement/provenance/confidence/significance) AND the real Kuzu schema + spine helpers
(a temp TitanKnowledgeGraph: Maker/MakerFact tables + Self→Maker→MakerFact edges). No
torch, no network.
"""
import os
import tempfile

import duckdb
import pytest

from titan_hcl.synthesis.maker_store import MakerStore, _CONF_CEIL
from titan_hcl.core.direct_memory import TitanKnowledgeGraph


def _store(tmp_path, graph=None):
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    return MakerStore(conn, graph=graph, writer=None, clock=lambda: 1000.0)


# ── DuckDB store logic (graph=None) ──────────────────────────────────

def test_record_and_recall(tmp_path):
    s = _store(tmp_path)
    fid = s.record_fact(category="occupation", value="software architect",
                        provenance="maker-told", confidence=0.9)
    assert fid == "maker:occupation:v1"
    facts = s.recall()
    assert len(facts) == 1
    f = facts[0]
    assert f["category"] == "occupation"
    assert f["value"] == "software architect"
    assert f["provenance"] == "maker-told"
    assert f["version"] == 1
    assert f["research_urgency"] == "none"


def test_confidence_never_absolute(tmp_path):
    # INV-MIS-EPISTEMIC-HONESTY: nothing about the Maker is 100% truth, not even
    # maker-told. A confidence of 1.0 is capped below certainty.
    s = _store(tmp_path)
    s.record_fact(category="name", value="Jirka", confidence=1.0)
    f = s.recall("name")[0]
    assert f["confidence"] <= _CONF_CEIL
    assert f["confidence"] < 1.0


def test_significance_set_on_create(tmp_path):
    s = _store(tmp_path)
    s.record_fact(category="hobby", value="climbing", confidence=0.9)
    f = s.recall("hobby")[0]
    assert 1.0 <= f["significance"] <= 100.0
    # higher confidence → higher initial significance
    s.record_fact(category="city", value="Berlin", confidence=0.3)
    assert s.recall("city")[0]["significance"] < f["significance"]


def test_reinforce_same_value_bumps_confidence_no_new_version(tmp_path):
    s = _store(tmp_path)
    f1 = s.record_fact(category="occupation", value="architect", confidence=0.6)
    f2 = s.record_fact(category="occupation", value="architect", confidence=0.9)
    assert f1 == f2 == "maker:occupation:v1"      # same node, reinforced
    facts = s.recall("occupation")
    assert len(facts) == 1                          # NOT a new version
    assert facts[0]["confidence"] >= 0.9 - 1e-9     # bumped to the higher
    assert facts[0]["version"] == 1


def test_version_on_contradiction_supersedes(tmp_path):
    s = _store(tmp_path)
    s.record_fact(category="occupation", value="software architect", confidence=0.8)
    f2 = s.record_fact(category="occupation", value="principal architect",
                       confidence=0.85)
    assert f2 == "maker:occupation:v2"
    current = s.recall("occupation")
    assert len(current) == 1                         # only the un-superseded one
    assert current[0]["value"] == "principal architect"
    assert current[0]["version"] == 2


def test_recall_ranks_by_significance_and_filters(tmp_path):
    s = _store(tmp_path)
    s.record_fact(category="a", value="x", confidence=0.3)   # low significance
    s.record_fact(category="b", value="y", confidence=0.95)  # high significance
    allf = s.recall()
    assert [f["category"] for f in allf] == ["b", "a"]       # ranked desc
    assert len(s.recall("a")) == 1                            # category filter


def test_empty_inputs_are_noops(tmp_path):
    s = _store(tmp_path)
    assert s.record_fact(category="", value="x") == ""
    assert s.record_fact(category="c", value="  ") == ""
    assert s.recall() == []


def test_unknown_provenance_defaults_to_maker_told(tmp_path):
    s = _store(tmp_path)
    s.record_fact(category="x", value="y", provenance="hearsay")
    assert s.recall("x")[0]["provenance"] == "maker-told"


# ── Real Kuzu schema + spine helpers ─────────────────────────────────

@pytest.fixture()
def graph():
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "maker_test.kuzu"))
        try:
            yield g
        finally:
            g.close()


def test_schema_creates_maker_tables(graph):
    # bootstrap_spine_schema (run at graph open) created the new node/rel tables.
    names = set()
    qr = graph._conn.execute("CALL SHOW_TABLES() RETURN *")
    while qr.has_next():
        row = qr.get_next()
        for cell in row:
            if isinstance(cell, str):
                names.add(cell)
    assert "Maker" in names
    assert "MakerFact" in names


def test_spine_ensure_maker_node_and_link(graph):
    assert graph.spine_ensure_maker_node() is True
    # idempotent
    assert graph.spine_ensure_maker_node() is True
    # exactly one Maker, linked under Self
    qr = graph._conn.execute("MATCH (m:Maker) RETURN COUNT(m)")
    assert qr.get_next()[0] == 1
    qr = graph._conn.execute(
        "MATCH (s:Self)-[:SELF_HAS_MAKER]->(m:Maker) RETURN COUNT(m)")
    assert qr.get_next()[0] == 1


def test_spine_create_fact_node_and_edge(graph):
    ok = graph.spine_create_maker_fact_node(
        fact_id="maker:occupation:v1", category="occupation",
        value="software architect", provenance="maker-told", confidence=0.9,
        significance=63.0, research_urgency="none", version=1,
        created_at=1000.0, updated_at=1000.0)
    assert ok is True
    # idempotent (PK) → False on replay
    assert graph.spine_create_maker_fact_node(
        fact_id="maker:occupation:v1", category="occupation",
        value="software architect", provenance="maker-told", confidence=0.9,
        significance=63.0, research_urgency="none", version=1,
        created_at=1000.0, updated_at=1000.0) is False
    assert graph.spine_link_maker_fact("maker:occupation:v1") is True
    qr = graph._conn.execute(
        "MATCH (m:Maker)-[:MAKER_HAS_FACT]->(f:MakerFact {fact_id:'maker:occupation:v1'}) "
        "RETURN COUNT(f)")
    assert qr.get_next()[0] == 1


def test_spine_supersede(graph):
    graph.spine_create_maker_fact_node(
        fact_id="maker:city:v1", category="city", value="Berlin",
        provenance="maker-told", confidence=0.8, significance=50.0,
        research_urgency="none", version=1, created_at=1000.0, updated_at=1000.0)
    assert graph.spine_supersede_maker_fact("maker:city:v1") is True
    qr = graph._conn.execute(
        "MATCH (f:MakerFact {fact_id:'maker:city:v1'}) RETURN f.superseded")
    assert qr.get_next()[0] == 1


def test_end_to_end_with_graph(tmp_path, graph):
    # MakerStore writing BOTH DuckDB scalars + the Kuzu node/edge.
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s = MakerStore(conn, graph=graph, writer=None, clock=lambda: 1000.0)
    fid = s.record_fact(category="occupation", value="architect", confidence=0.9)
    assert fid == "maker:occupation:v1"
    assert s.recall("occupation")[0]["value"] == "architect"   # DuckDB
    qr = graph._conn.execute(
        "MATCH (m:Maker)-[:MAKER_HAS_FACT]->(f:MakerFact {fact_id:$f}) RETURN COUNT(f)",
        {"f": fid})
    assert qr.get_next()[0] == 1                                 # Kuzu
