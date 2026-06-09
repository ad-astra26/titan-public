"""EEL Pillar A / Phase A1 — research provenance.

`RFP_emergent_experience_learning.md` §7.A1 / INV-EEL-6: a researched answer's
persisted memory carries `acquired:research` tag(s) + its source, so promotion
is provenanced and the fact is auditable. Mechanic (this session, 2026-06-09):
`memory_nodes` gains `tags` (JSON list) + `acquired_source` columns;
`add_to_mempool(tags=, source=)` threads them onto the node (and stamps them
onto an existing node on the semantic-dedup path); they survive reload.

Scope-fence: A1 is provenance ONLY — no scoring, no promotion change (that is A2).
"""
from __future__ import annotations

import json

import pytest

from titan_hcl.core.direct_memory import TitanDuckDB
from titan_hcl.core.memory import TieredMemoryGraph


# ── TitanDuckDB layer: schema + migration + insert round-trip ────────────────

def test_provenance_columns_present(tmp_path):
    db = TitanDuckDB(str(tmp_path / "titan_memory.duckdb"))
    try:
        cols = [c[1] for c in db._conn.execute(
            "PRAGMA table_info('memory_nodes')").fetchall()]
        assert "tags" in cols, f"`tags` column missing; have {cols}"
        assert "acquired_source" in cols, f"`acquired_source` missing; have {cols}"
    finally:
        db._conn.close()


def test_migration_idempotent_on_reopen(tmp_path):
    """ADD COLUMN IF NOT EXISTS must silent-skip on an existing DB (the
    `memory_type` / `timechain_tx_hash` precedent)."""
    db_path = str(tmp_path / "titan_memory.duckdb")
    TitanDuckDB(db_path)._conn.close()
    db2 = TitanDuckDB(db_path)  # _init_schema runs again → must be a no-op
    try:
        cols = [c[1] for c in db2._conn.execute(
            "PRAGMA table_info('memory_nodes')").fetchall()]
        assert "tags" in cols and "acquired_source" in cols
    finally:
        db2._conn.close()


def test_insert_node_persists_tags_json_and_source(tmp_path):
    db = TitanDuckDB(str(tmp_path / "titan_memory.duckdb"))
    try:
        db.insert_node({
            "id": 1, "user_prompt": "Jupiter TVL?", "agent_response": "~$X",
            "status": "mempool", "tags": ["acquired:research"],
            "acquired_source": "searxng:jupiter tvl",
        })
        rows = db.get_all_nodes()
        assert len(rows) == 1
        row = rows[0]
        # insert_node JSON-encodes a list `tags` (mirrors neuromod_context)
        assert json.loads(row["tags"]) == ["acquired:research"]
        assert row["acquired_source"] == "searxng:jupiter tvl"
    finally:
        db._conn.close()


# ── TieredMemoryGraph: add_to_mempool threading + dedup stamp + reload ────────

@pytest.mark.asyncio
async def test_add_to_mempool_threads_provenance(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    await mem.add_to_mempool(
        "Jupiter TVL?", "I didn't have that — I looked it up: ~$X", "user_a",
        tags=["acquired:research"], source="searxng:jupiter tvl")
    pool = await mem.fetch_mempool()
    assert len(pool) == 1
    node = pool[0]
    assert node["tags"] == ["acquired:research"]
    assert node["acquired_source"] == "searxng:jupiter tvl"


@pytest.mark.asyncio
async def test_plain_turn_has_no_provenance(tmp_path):
    """A normal (non-research) turn carries no acquired-research provenance."""
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    await mem.add_to_mempool("hi", "hello there", "user_a")
    pool = await mem.fetch_mempool()
    assert pool[0]["tags"] == []
    assert pool[0]["acquired_source"] is None


@pytest.mark.asyncio
async def test_provenance_survives_reload(tmp_path):
    """The restart-survival contract (the persistence half of EEL-G1): a
    researched fact's provenance must reload as a list, not a JSON string."""
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    await mem.add_to_mempool(
        "Jupiter TVL?", "~$X", "user_a",
        tags=["acquired:research"], source="searxng:x")
    # Simulate boot: drop the in-memory store + re-load from the same DuckDB.
    mem._node_store.clear()
    mem._load_node_store()
    nodes = [v for v in mem._node_store.values()
             if v.get("type") == "MemoryNode"]
    assert len(nodes) == 1
    n = nodes[0]
    assert n["tags"] == ["acquired:research"], (
        f"tags must decode to a list on reload, got {n['tags']!r}")
    assert n["acquired_source"] == "searxng:x"


@pytest.mark.asyncio
async def test_dedup_stamps_provenance_on_existing_node(tmp_path):
    """A research answer that semantically dedups into an existing node must
    still carry its provenance (Maker-confirmed 2026-06-09). The similarity is
    monkeypatched so the dedup branch fires deterministically (no embedder)."""
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    await mem.add_to_mempool("Jupiter TVL?", "hmm, not sure", "user_a")
    pool = await mem.fetch_mempool()
    existing_id = pool[0]["id"]
    assert pool[0]["tags"] == []  # plain node, no provenance yet

    # Force the dedup→reinforce branch onto the existing node.
    mem.find_similar_mempool_node = lambda *a, **k: existing_id
    await mem.add_to_mempool(
        "Jupiter TVL?", "I looked it up: ~$X", "user_a",
        tags=["acquired:research"], source="searxng:x")

    pool2 = await mem.fetch_mempool()
    assert len(pool2) == 1, "must dedup (reinforce), not create a 2nd node"
    assert pool2[0]["tags"] == ["acquired:research"]
    assert pool2[0]["acquired_source"] == "searxng:x"
