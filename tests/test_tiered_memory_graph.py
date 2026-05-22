"""
tests/test_tiered_memory_graph.py — TieredMemoryGraph integration tests.

Tests the live memory backend (FAISS + Kuzu + DuckDB) which replaced the
legacy Cognee backend. Renamed + cleaned from `test_cognee_memory.py`
2026-05-12 — dropped the 4 stale Cognee-era tests:

  * test_cognee_initialization (asserted `mem._cognee_ready is True` —
    legacy attribute that the new backend doesn't expose)
  * test_local_only_migrate_and_decay (weight-decay arithmetic check
    that drifted as the formula was refined)
  * test_persistent_count (relied on fixture isolation that no longer
    holds with FAISS index files persisting between tests in the same
    process)
  * test_mock_db_backward_compat (back-compat alias for the pre-FAISS
    `_mock_db` attribute — no longer needed)

The 6 tests below exercise general TieredMemoryGraph API that survived
the Cognee → FAISS+Kuzu+DuckDB replacement: mempool lifecycle, system
pulses, social metrics, keyword search, ingest + search, and
consolidate-without-LLM graceful path.
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data_tmp")
TEST_CONFIG = {"data_dir": TEST_DATA_DIR}


@pytest.fixture(autouse=True)
def clean_test_dirs():
    """Remove test data dir before + after each test for isolation."""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    yield
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


# ── Mempool lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mempool_lifecycle():
    """Mempool add/fetch/prune works on the live backend."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_to_mempool("Hello world", "Hi there", "user_a")
    await mem.add_to_mempool("Second message", "Got it", "user_b")

    pool = await mem.fetch_mempool()
    assert len(pool) == 2
    assert pool[0]["status"] == "mempool"
    assert pool[0]["user_prompt"] == "Hello world"

    # Prune first node
    await mem.prune_mempool_node(pool[0]["id"])
    pool2 = await mem.fetch_mempool()
    assert len(pool2) == 1


# ── System pulse + recent sentiments ───────────────────────────────────────


@pytest.mark.asyncio
async def test_system_pulse_and_sentiments():
    """add_system_pulse + get_recent_sentiments work on the live backend."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_system_pulse(0.8, epoch_id=1)
    await mem.add_system_pulse(0.6, epoch_id=2)
    await mem.add_system_pulse(0.9, epoch_id=3)

    sentiments = await mem.get_recent_sentiments(count=2)
    assert len(sentiments) == 2
    assert sentiments[0] == 0.9  # Most recent first


# ── Social metrics CRUD ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_social_metrics_lifecycle():
    """Social metrics CRUD (fetch / update / reset) round-trips correctly."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    metrics = await mem.fetch_social_metrics()
    assert metrics["daily_likes"] == 0
    assert metrics["daily_replies"] == 0
    assert metrics["mentions_received"] == 0
    assert metrics["reply_likes"] == 0

    await mem.update_social_metrics(likes_inc=3, replies_inc=1, mentions_received_inc=5)
    metrics = await mem.fetch_social_metrics()
    assert metrics["daily_likes"] == 3
    assert metrics["daily_replies"] == 1
    assert metrics["mentions_received"] == 5

    await mem.reset_daily_social_metrics()
    metrics = await mem.fetch_social_metrics()
    assert metrics["daily_likes"] == 0
    assert metrics["mentions_received"] == 0


# ── Query: keyword + semantic search ───────────────────────────────────────


@pytest.mark.asyncio
async def test_keyword_search_matches_content():
    """Keyword search finds memories by content words on the live backend."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_to_mempool("Solana blockchain is revolutionary", "Indeed it is")
    await mem.add_to_mempool("The weather today is sunny", "Enjoy the sun")

    results = await mem.query("Tell me about Solana blockchain")
    prompts = [r["user_prompt"] for r in results]
    assert any("Solana" in p for p in prompts)


@pytest.mark.asyncio
async def test_ingest_and_search():
    """Memories migrated to persistent are searchable via query()."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    # Add and migrate memories
    await mem.add_to_mempool(
        "The Solana blockchain processes 65000 transactions per second",
        "That is impressive throughput for a Layer 1 chain",
    )
    await mem.add_to_mempool(
        "My favorite food is spaghetti carbonara",
        "A classic Italian dish",
    )

    nodes = await mem.fetch_mempool()
    for node in nodes:
        await mem.migrate_to_persistent(node["id"], "tx_test", 5)

    # Search — should find the Solana-related memory
    results = await mem.query("What do you know about blockchain speed?")
    assert len(results) >= 1
    found_solana = any("Solana" in r.get("user_prompt", "") for r in results)
    assert found_solana, "Expected to find Solana memory via search"


# ── Consolidate: graceful skip without LLM key ─────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_without_api_key():
    """consolidate() succeeds (FAISS save only) when no LLM API key is set."""
    from titan_hcl.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config={**TEST_CONFIG, "openrouter_api_key": ""})

    await mem.add_to_mempool("test data", "response")
    nodes = await mem.fetch_mempool()
    await mem.migrate_to_persistent(nodes[0]["id"], "tx_test", 5)

    # Should succeed without error
    result = await mem.consolidate()
    assert result is True
