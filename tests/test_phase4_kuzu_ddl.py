"""Phase 4 — Kuzu Engram-spine DDL tests (P4.A; spine node renamed Concept→Engram, RFP §7.B).

Covers ARCHITECTURE_synthesis_engine.md §6.1/§6.2 + PLAN_synthesis_engine_Phase4.md §P4.A:
- All 4 node tables (Engram / Production / ActionChain / HypothesisFork) exist.
- All 5 rel tables (COMPOSED_FROM / COMPOSED_INTO / USES_SKILL / COMPILED_FROM
  / EXPLORES) exist.
- Bootstrap is idempotent.
- Composite PRIMARY KEY(concept_id, version) allows two rows with same
  concept_id but different versions.
- spine_* helper methods round-trip correctly.
- Existing Kuzu tables (Person/Topic/Trinity entities) are unaffected
  (regression).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.kuzu_spine_schema import (
    _NODE_TABLES,
    _REL_TABLES,
    _table_exists,
    bootstrap_spine_schema,
)


@pytest.fixture()
def graph():
    """Fresh TitanKnowledgeGraph in a tmp dir; closed after each test."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_spine.kuzu")
        g = TitanKnowledgeGraph(path)
        try:
            yield g
        finally:
            g.close()


# ─── Schema presence ────────────────────────────────────────────────

def test_all_spine_node_tables_exist_after_init(graph):
    """All 4 spine node tables present after TitanKnowledgeGraph init."""
    for table_name, _ in _NODE_TABLES:
        assert _table_exists(graph._conn, table_name), (
            f"node table {table_name!r} not created by TitanKnowledgeGraph init"
        )


def test_all_spine_rel_tables_exist_after_init(graph):
    """All 5 spine rel tables present after TitanKnowledgeGraph init."""
    for rel_name, _, _ in _REL_TABLES:
        assert _table_exists(graph._conn, rel_name), (
            f"rel table {rel_name!r} not created by TitanKnowledgeGraph init"
        )


def test_bootstrap_is_idempotent(graph):
    """Re-running bootstrap_spine_schema does not raise + creates nothing."""
    second = bootstrap_spine_schema(graph)
    assert second["errors"] == [], (
        f"unexpected errors on re-bootstrap: {second['errors']}"
    )
    # Nothing should be created on re-bootstrap (all tables already exist).
    assert second["created_nodes"] == []
    assert second["created_rels"] == []


# ─── Composite primary key (§10 versioning) ─────────────────────────

def test_composite_pk_two_versions_coexist(graph):
    """Two Concept rows with same concept_id but different versions must
    both persist — this is the load-bearing invariant for §10 versioned
    concepts (v(n) stays immutable; v(n+1) is a new row, not an update)."""
    assert graph.spine_create_concept_node(
        concept_id="metaplex_nft_minting", version=1,
        name="Metaplex NFT minting", memory_type="procedural",
        groundedness=0.2, anchor_tx="tx_v1_hash", created_at=1000.0,
    )
    assert graph.spine_create_concept_node(
        concept_id="metaplex_nft_minting", version=2,
        name="Metaplex NFT minting", memory_type="procedural",
        groundedness=0.4, anchor_tx="tx_v2_hash", created_at=2000.0,
    )

    v1 = graph.spine_get_concept_version("metaplex_nft_minting", 1)
    v2 = graph.spine_get_concept_version("metaplex_nft_minting", 2)
    assert v1 is not None and v1["anchor_tx"] == "tx_v1_hash"
    assert v2 is not None and v2["anchor_tx"] == "tx_v2_hash"
    # Latest must return v2 (highest version).
    latest = graph.spine_get_latest_concept("metaplex_nft_minting")
    assert latest is not None and latest["version"] == 2


def test_duplicate_concept_version_insert_is_idempotent_false(graph):
    """Inserting the same (concept_id, version) twice: first returns True,
    second returns False without raising — safe for replay scenarios."""
    assert graph.spine_create_concept_node(
        concept_id="linux_terminal", version=1, name="Linux terminal",
        memory_type="declarative", groundedness=0.1, anchor_tx="tx_a",
        created_at=100.0,
    ) is True
    assert graph.spine_create_concept_node(
        concept_id="linux_terminal", version=1, name="Linux terminal",
        memory_type="declarative", groundedness=0.1, anchor_tx="tx_a",
        created_at=100.0,
    ) is False


# ─── Helper round-trips ─────────────────────────────────────────────

def test_spine_helpers_round_trip(graph):
    """Create + get + update_groundedness + count, all consistent."""
    assert graph.spine_count_concepts() == 0

    graph.spine_create_concept_node(
        concept_id="cosmetic_business_website", version=1,
        name="Cosmetic business website", memory_type="declarative",
        groundedness=0.0, anchor_tx="tx_init", created_at=500.0,
    )
    graph.spine_create_concept_node(
        concept_id="cosmetic_business_website", version=2,
        name="Cosmetic business website", memory_type="declarative",
        groundedness=0.1, anchor_tx="tx_v2", created_at=600.0,
    )
    graph.spine_create_concept_node(
        concept_id="solana_rpc", version=1,
        name="Solana RPC", memory_type="procedural",
        groundedness=0.5, anchor_tx="tx_rpc", created_at=300.0,
    )

    assert graph.spine_count_concepts() == 3

    # update_groundedness must succeed on existing row + fail on missing row.
    assert graph.spine_update_groundedness(
        "cosmetic_business_website", 2, 0.35,
    ) is True
    refreshed = graph.spine_get_concept_version(
        "cosmetic_business_website", 2,
    )
    assert refreshed is not None
    assert abs(refreshed["groundedness"] - 0.35) < 1e-9

    assert graph.spine_update_groundedness("ghost_concept", 1, 0.5) is False
    assert graph.spine_update_groundedness(
        "cosmetic_business_website", 99, 0.5,
    ) is False


def test_composition_edges_round_trip(graph):
    """COMPOSED_FROM + COMPOSED_INTO edges round-trip; counts + neighbor
    lookup are correct; missing endpoints return False; spreading-activation
    neighbor lookup walks both directions."""
    # Build a small spine: cosmetic_business_website v1 composed_from
    # [domain_dns v1, ssl_cert v1] and composed_into [biz_starter_kit v1].
    rows = [
        ("cosmetic_business_website", 1, "Cosmetic business website",
         "declarative"),
        ("domain_dns", 1, "Domain + DNS", "declarative"),
        ("ssl_cert", 1, "SSL certificate", "declarative"),
        ("biz_starter_kit", 1, "Business starter kit", "meta"),
    ]
    for cid, ver, name, mt in rows:
        graph.spine_create_concept_node(
            concept_id=cid, version=ver, name=name, memory_type=mt,
            groundedness=0.2, anchor_tx=f"tx_{cid}", created_at=100.0,
        )

    # cosmetic_business_website v1 -[COMPOSED_FROM]-> domain_dns v1
    assert graph.spine_add_composition_edge(
        "cosmetic_business_website", 1, "domain_dns", 1, direction="from",
    ) is True
    # cosmetic_business_website v1 -[COMPOSED_FROM]-> ssl_cert v1
    assert graph.spine_add_composition_edge(
        "cosmetic_business_website", 1, "ssl_cert", 1, direction="from",
    ) is True
    # cosmetic_business_website v1 -[COMPOSED_INTO]-> biz_starter_kit v1
    assert graph.spine_add_composition_edge(
        "cosmetic_business_website", 1, "biz_starter_kit", 1,
        direction="into",
    ) is True

    # Missing endpoint must return False (no raise).
    assert graph.spine_add_composition_edge(
        "ghost", 1, "cosmetic_business_website", 1, direction="from",
    ) is False
    assert graph.spine_add_composition_edge(
        "cosmetic_business_website", 1, "ghost", 1, direction="into",
    ) is False

    assert graph.spine_count_composition_edges("from") == 2
    assert graph.spine_count_composition_edges("into") == 1

    # Spreading-activation neighbor lookup picks up BOTH rel directions
    # (per spine_concept_neighbors contract for §P4.F).
    neighbors = graph.spine_concept_neighbors(
        "cosmetic_business_website", limit=20,
    )
    assert len(neighbors) == 3
    assert ("domain_dns", 1) in neighbors
    assert ("ssl_cert", 1) in neighbors
    assert ("biz_starter_kit", 1) in neighbors

    # Bad direction string must not corrupt the graph.
    assert graph.spine_add_composition_edge(
        "domain_dns", 1, "ssl_cert", 1, direction="lateral",
    ) is False


def test_list_concepts_returns_latest_per_id_ordered_by_groundedness(graph):
    """spine_list_concepts collapses to latest version per concept_id and
    sorts by groundedness DESC — used by the Observatory /v6/synthesis/engrams
    endpoint and the groundedness heatmap."""
    graph.spine_create_concept_node(
        concept_id="a_concept", version=1, name="A v1",
        memory_type="declarative", groundedness=0.1,
        anchor_tx="tx", created_at=100.0,
    )
    graph.spine_create_concept_node(
        concept_id="a_concept", version=2, name="A v2",
        memory_type="declarative", groundedness=0.9,
        anchor_tx="tx", created_at=200.0,
    )
    graph.spine_create_concept_node(
        concept_id="b_concept", version=1, name="B v1",
        memory_type="declarative", groundedness=0.5,
        anchor_tx="tx", created_at=150.0,
    )

    listing = graph.spine_list_concepts(limit=10)
    assert len(listing) == 2
    # Latest version per concept_id (a_concept must be v2).
    a_row = next(r for r in listing if r["concept_id"] == "a_concept")
    b_row = next(r for r in listing if r["concept_id"] == "b_concept")
    assert a_row["version"] == 2
    assert b_row["version"] == 1
    # Order: groundedness DESC → a (0.9) before b (0.5).
    assert listing[0]["concept_id"] == "a_concept"
    assert listing[1]["concept_id"] == "b_concept"


# ─── Regression: existing Kuzu tables unaffected ───────────────────

def test_existing_kuzu_entities_unaffected_by_spine_bootstrap(graph):
    """Adding a Person entity + a Topic entity after spine bootstrap must
    succeed — the spine schema is purely additive, not a replacement."""
    # Person via the existing add_entity API.
    graph.add_entity("alice", "person", source_node=0,
                     attributes={"user_id": "u1"})
    graph.add_entity("agriculture", "topic", source_node=0)

    stats = graph.get_stats()
    assert stats.get("Person", 0) >= 1, f"Person count missing: {stats}"
    assert stats.get("Topic", 0) >= 1, f"Topic count missing: {stats}"

    # Spine bootstrap re-run must still be a no-op.
    second = bootstrap_spine_schema(graph)
    assert second["created_nodes"] == []
    assert second["created_rels"] == []
