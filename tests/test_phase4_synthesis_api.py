"""Phase 4 — Observatory /v6/synthesis/concepts/* handler tests (§P4.I).

Covers `titan_hcl/api/synthesis_concept_handlers.py`:
- list endpoint: pagination, memory_type filter, ordering by groundedness DESC
- get-one endpoint: returns all versions + composition edges
- heatmap endpoint: 4x10 grid bucketed by (memory_type, groundedness decile)
- soft-fail: missing Kuzu file → empty response with kuzu="missing"

Each test sets up a real Kuzu graph in tmp dir + seeds spine concepts via
ConceptStore, then opens the handler read-only against the same path. This
exercises the production read-only cross-process pattern.
"""
from __future__ import annotations

import os
import queue
import tempfile

import pytest

from titan_hcl.api import synthesis_concept_handlers as handlers
from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.concept_store import ConceptStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


@pytest.fixture()
def seeded_kuzu(monkeypatch):
    """Real Kuzu graph in tmp + 4 spine concepts spanning all memory_types.
    Sets TITAN_DATA_DIR so the handlers resolve the same Kuzu path."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TITAN_DATA_DIR", tmp)
        handlers._reset_cache_for_tests()
        kuzu_path = os.path.join(tmp, "synthesis_spine.kuzu")

        # Build the graph (read-write — we're the "writer" here).
        g = TitanKnowledgeGraph(kuzu_path)
        q = queue.Queue()
        w = OuterMemoryWriter(send_queue=q, src="api_test")
        store = ConceptStore(g, w, clock=lambda: 1000.0)

        store.create_concept("linux_terminal", "Linux terminal",
                             memory_type="declarative")
        store.recompute_groundedness("linux_terminal", 1,
                                     episodic_encounters=20)
        store.create_concept("solana_rpc", "Solana RPC",
                             memory_type="procedural")
        store.recompute_groundedness("solana_rpc", 1,
                                     episodic_encounters=50,
                                     procedural_links=10)
        store.create_concept("first_login", "First login",
                             memory_type="episodic")
        store.create_concept(
            "linux_basics", "Linux basics", memory_type="declarative",
            composed_from=[("linux_terminal", 1)],
        )
        store.bump_version("linux_terminal")  # produces v=2

        # IMPORTANT: close the writer handle before opening the read-only
        # handle in the handlers. Kuzu 0.11 supports concurrent read-only
        # with an active writer ONLY across processes, not within the same
        # process. Tests model the cross-process case by closing first.
        g.close()

        try:
            yield kuzu_path
        finally:
            handlers._reset_cache_for_tests()


# ── List endpoint ───────────────────────────────────────────────────


def test_list_returns_all_concepts_ordered_by_groundedness(seeded_kuzu):
    resp = handlers.get_synthesis_concepts(limit=20)
    assert resp["ok"] is True
    assert resp["total"] == 4
    concepts = resp["concepts"]
    # Ordered groundedness DESC.
    scores = [c["groundedness"] for c in concepts]
    assert scores == sorted(scores, reverse=True)


def test_list_memory_type_filter(seeded_kuzu):
    resp = handlers.get_synthesis_concepts(memory_type="declarative")
    assert resp["ok"] is True
    # declarative: linux_terminal v=2, linux_basics v=1
    assert resp["total"] == 2
    for c in resp["concepts"]:
        assert c["memory_type"] == "declarative"


def test_list_pagination(seeded_kuzu):
    resp1 = handlers.get_synthesis_concepts(limit=2, offset=0)
    resp2 = handlers.get_synthesis_concepts(limit=2, offset=2)
    assert resp1["total"] == 4 and resp2["total"] == 4
    ids1 = {c["concept_id"] for c in resp1["concepts"]}
    ids2 = {c["concept_id"] for c in resp2["concepts"]}
    assert not (ids1 & ids2)  # disjoint pages


def test_list_clamps_excessive_limit(seeded_kuzu):
    """limit=99999 must not blow up; clamped to 500 page cap."""
    resp = handlers.get_synthesis_concepts(limit=99999)
    assert resp["ok"] is True
    assert len(resp["concepts"]) <= 500


# ── Get-one endpoint ───────────────────────────────────────────────


def test_get_one_returns_all_versions(seeded_kuzu):
    resp = handlers.get_synthesis_concept("linux_terminal")
    assert resp["ok"] is True
    assert resp["exists"] is True
    versions = resp["versions"]
    assert len(versions) == 2
    # Ordered by version ASC.
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2
    assert resp["latest_version"] == 2


def test_get_one_returns_composition_edges(seeded_kuzu):
    """linux_basics v1 has COMPOSED_FROM → linux_terminal v1."""
    resp = handlers.get_synthesis_concept("linux_basics")
    assert resp["ok"] is True
    assert resp["exists"] is True
    composed_from = resp["composed_from"]
    assert any(
        e["concept_id"] == "linux_terminal" and e["version"] == 1
        for e in composed_from
    )


def test_get_one_missing_concept_returns_empty_versions(seeded_kuzu):
    resp = handlers.get_synthesis_concept("never_existed")
    assert resp["ok"] is True
    assert resp["exists"] is False
    assert resp["versions"] == []


def test_get_one_empty_concept_id_rejected(seeded_kuzu):
    resp = handlers.get_synthesis_concept("")
    assert resp["ok"] is False
    assert "empty_concept_id" in resp.get("error", "")


# ── Heatmap endpoint ───────────────────────────────────────────────


def test_heatmap_buckets_by_memory_type_and_decile(seeded_kuzu):
    resp = handlers.get_synthesis_concepts_heatmap()
    assert resp["ok"] is True
    heatmap = resp["heatmap"]
    # 4 memory types × 10 deciles.
    assert set(heatmap.keys()) == {
        "declarative", "procedural", "episodic", "meta",
    }
    for row in heatmap.values():
        assert len(row) == 10
    # Total of all cells matches total concept count.
    grand_total = sum(sum(row) for row in heatmap.values())
    assert grand_total == resp["total"] == 4


def test_heatmap_solana_rpc_lands_in_mid_decile(seeded_kuzu):
    """solana_rpc has the highest groundedness in the seed set
    (epi=50 + proc=10 → g ≈ 0.54 with default weights). Lands in the
    procedural row's decile 5+."""
    resp = handlers.get_synthesis_concepts_heatmap()
    proc_row = resp["heatmap"]["procedural"]
    # At least one procedural concept in decile 5+ (solana_rpc).
    assert sum(proc_row[5:]) >= 1


# ── Soft-fail paths ─────────────────────────────────────────────────


def test_missing_kuzu_returns_empty_response(monkeypatch):
    """No Kuzu file → handler returns ok=True with empty list (the
    frontend renders an empty state, not an error toast)."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TITAN_DATA_DIR", tmp)
        handlers._reset_cache_for_tests()
        resp = handlers.get_synthesis_concepts()
        assert resp["ok"] is True
        assert resp["concepts"] == []
        assert resp.get("kuzu") == "missing"

        resp2 = handlers.get_synthesis_concept("anything")
        assert resp2["ok"] is True
        assert resp2["versions"] == []

        resp3 = handlers.get_synthesis_concepts_heatmap()
        assert resp3["ok"] is True
        # Empty heatmap = all zeros.
        for row in resp3["heatmap"].values():
            assert row == [0] * 10
