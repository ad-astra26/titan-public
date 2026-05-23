"""Integration test for the synthesis-aware _cognee_search wiring.

D-SPEC-123 / SPEC v1.56.0 §25. Validates:
  - TieredMemoryGraph accepts bus_emit + degrades gracefully when None
  - When watermark stale/missing → cosine-only fallback (no regression)
  - When watermark fresh + activation_state populated → composite re-rank
    runs + MEMORY_RETRIEVAL_USED emitted per returned item
  - Emit failures don't crash _cognee_search (degrade silently)
"""
from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest


@pytest.fixture
def isolated_shm(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    monkeypatch.setenv("TITAN_SHM_ROOT", str(shm_dir))
    yield shm_dir


@pytest.fixture
def isolated_data_dir(monkeypatch, tmp_path):
    """Point TITAN_DATA_DIR at a temp dir so the test owns its DuckDB."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("TITAN_DATA_DIR", str(data))
    # Also reset the bridge_recall singleton so each test gets a fresh one
    # pointing at the right TITAN_SHM_ROOT.
    import titan_hcl.synthesis.bridge_recall as br_mod
    monkeypatch.setattr(br_mod, "_bridge_recall_singleton", None)
    yield str(data)


def _make_memory_with_two_persistent_nodes(data_dir, bus_emit=None):
    """Construct TieredMemoryGraph + insert 2 persistent nodes with
    embeddings the test can FAISS-search against."""
    from titan_hcl.core.memory import TieredMemoryGraph
    cfg = {"data_dir": data_dir, "api": {"port": 7777, "internal_key": ""}}
    mem = TieredMemoryGraph(config=cfg, bus_emit=bus_emit)
    # Insert two persistent nodes directly into the in-mem store + DuckDB
    # + FAISS index, bypassing the chat flow.
    now = time.time()
    for node_id, text in [(1, "the quick brown fox"),
                          (2, "lorem ipsum dolor sit amet")]:
        node = {
            "id": node_id, "type": "MemoryNode",
            "user_prompt": text, "agent_response": text,
            "source_id": "test", "status": "persistent",
            "base_weight": 1.0, "anchor_bonus": 0.0,
            "reinforcement_count": 0, "emotional_intensity": 0,
            "mempool_weight": 1.0, "mempool_reinforcements": 0,
            "effective_weight": 0.5,
            "created_at": now, "last_accessed": now, "last_reinforced": now,
            "embedding_id": node_id, "cognified": True,
            "neuromod_context": "{}",
        }
        mem._node_store[node_id] = node
        mem._duckdb.insert_node(node)
        emb = mem._embed_text(text)
        if emb is not None:
            mem._vectors.add(emb, node_id)
    mem._next_id = 3
    return mem


# ─────────────────────────────────────────────────────────────────────────
# Backward-compat: no bus_emit → cosine-only, NO crash
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cognee_search_without_bus_emit_returns_cosine_only(
    isolated_data_dir,
):
    mem = _make_memory_with_two_persistent_nodes(isolated_data_dir,
                                                  bus_emit=None)
    results = await mem._cognee_search("quick brown fox", top_k=5)
    assert len(results) == 2
    # The query is closer to node 1 — FAISS should rank it first.
    assert results[0]["id"] == 1


# ─────────────────────────────────────────────────────────────────────────
# With bus_emit but no fresh watermark → cosine-only, NO crash, no emit
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cognee_search_degrades_when_watermark_missing(
    isolated_data_dir, isolated_shm,
):
    """No synthesis_worker booted → watermark slot missing → activation
    lookup returns {} → composite_score falls through to cosine-only.
    Emit still fires (it's a separate concern — the worker subscribes
    even if the watermark is stale; future booted worker picks up the
    backlog from the bus)."""
    emitted = []
    mem = _make_memory_with_two_persistent_nodes(
        isolated_data_dir,
        bus_emit=lambda msg_type, payload: emitted.append((msg_type, payload)),
    )
    results = await mem._cognee_search("quick brown fox", top_k=5)
    assert len(results) == 2
    assert results[0]["id"] == 1   # cosine-only ranking preserved
    # Stale watermark → BridgeRecall.activation_lookup() returns {} →
    # composite_score still runs (just with all cold-start) → emit still
    # fires per returned item.
    assert len(emitted) == 2
    assert all(t == "MEMORY_RETRIEVAL_USED" for t, _ in emitted)
    item_ids = {p["item_id"] for _, p in emitted}
    assert item_ids == {"mem:1", "mem:2"}


# ─────────────────────────────────────────────────────────────────────────
# With fresh watermark + activation_state → re-rank changes order
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cognee_search_with_fresh_watermark_reranks(
    isolated_data_dir, isolated_shm,
):
    """Pre-populate activation_state with a very high B_i for node 2 +
    publish a fresh watermark → composite_score re-ranks node 2 above
    node 1, even though node 1 has higher cosine."""
    # Build memory + nodes + FAISS index FIRST so the DuckDB write below
    # adds to an already-initialized schema.
    emitted = []
    mem = _make_memory_with_two_persistent_nodes(
        isolated_data_dir,
        bus_emit=lambda msg_type, payload: emitted.append((msg_type, payload)),
    )
    # Inject very-high activation for node 2 so composite_score promotes
    # it ahead of node 1. activation_state lives in synthesis.duckdb now
    # (post-relocation 2026-05-23). Use ActivationStore as the writer to
    # ensure the schema lands the way the worker would create it.
    from titan_hcl.modules.synthesis_worker import ActivationStore
    now = time.time()
    synth_db_path = os.path.join(isolated_data_dir, "synthesis.duckdb")
    store = ActivationStore(synth_db_path)
    try:
        store._conn.execute(
            "INSERT OR REPLACE INTO activation_state "
            "(item_id, base_level, last_access, access_count, "
            " first_access, last_recompute) VALUES "
            "(?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)",
            ("mem:1", -2.0, now, 1, now - 1000, now,
             "mem:2", 10.0, now, 100, now - 1000, now),
        )
    finally:
        store.close()
    # Publish a fresh watermark from a writer that points at the same
    # SHM root + uses the same titan_id default.
    from titan_hcl.modules.synthesis_worker import SynthStatusWriter
    writer = SynthStatusWriter(titan_id="T1")
    try:
        writer.publish(last_consistent_event_ts=now,
                       last_recompute_ts=now,
                       items_tracked=2,
                       recompute_count_increment=1)
    finally:
        writer.close()

    results = await mem._cognee_search("quick brown fox", top_k=5)
    assert len(results) == 2
    # With B_i for mem:2 dominating (+10 raw, +1 z-score vs -1 for mem:1),
    # mem:2 should now lead despite mem:1 having higher cosine.
    assert results[0]["id"] == 2, (
        f"composite re-rank should put node 2 (high B_i) first, got: "
        f"{[r['id'] for r in results]}")
    # MEMORY_RETRIEVAL_USED emitted for both
    item_ids = {p["item_id"] for _, p in emitted}
    assert item_ids == {"mem:1", "mem:2"}


# ─────────────────────────────────────────────────────────────────────────
# Defensive: emit callback raises → search still completes
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cognee_search_survives_emit_failure(
    isolated_data_dir, isolated_shm,
):
    """If bus_emit raises (broker dead, queue full, etc.) — _cognee_search
    must still return results."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated broker outage")
    mem = _make_memory_with_two_persistent_nodes(
        isolated_data_dir, bus_emit=boom)
    results = await mem._cognee_search("quick brown fox", top_k=5)
    assert len(results) == 2  # still returns results


# ─────────────────────────────────────────────────────────────────────────
# Empty FAISS index → empty result + no emit
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cognee_search_empty_index_returns_empty(
    isolated_data_dir, isolated_shm,
):
    from titan_hcl.core.memory import TieredMemoryGraph
    emitted = []
    mem = TieredMemoryGraph(
        config={"data_dir": isolated_data_dir,
                "api": {"port": 7777, "internal_key": ""}},
        bus_emit=lambda mt, p: emitted.append((mt, p)),
    )
    results = await mem._cognee_search("anything", top_k=5)
    assert results == []
    assert emitted == []   # no emits for empty result set
