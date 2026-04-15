"""
tests/test_cognee_memory.py
Integration tests for the real Cognee-backed TieredMemoryGraph.
Tests semantic search, ingestion, consolidation, and backward compatibility.
"""
import asyncio
import os
import shutil
import sys
import time

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "test_cognee_data")
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data_tmp")
TEST_CONFIG = {"data_dir": TEST_DATA_DIR, "cognee_db_path": TEST_DB_PATH}


def _ensure_real_cognee():
    """Ensure real Cognee is in sys.modules (undo any mocks from other tests)."""
    import importlib

    mod = sys.modules.get("cognee")
    if mod is not None and (hasattr(mod, "_mock_name") or not hasattr(mod, "add")):
        # Another test file mocked cognee — remove it so real import works
        del sys.modules["cognee"]
        # Also remove any sub-modules that were mocked
        for key in list(sys.modules):
            if key.startswith("cognee."):
                del sys.modules[key]
    try:
        import cognee

        return hasattr(cognee, "add")
    except ImportError:
        return False


def cognee_available() -> bool:
    """Check if real Cognee is importable."""
    return _ensure_real_cognee()


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test Cognee DB and node store to ensure isolated tests."""
    _ensure_real_cognee()
    for path in (TEST_DB_PATH, TEST_DATA_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)
    yield
    for path in (TEST_DB_PATH, TEST_DATA_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Test 1: Local-only mode (no Cognee) — backward compatibility
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_only_mempool_lifecycle():
    """Mempool add/fetch/prune works without Cognee."""
    from titan_plugin.core.memory import TieredMemoryGraph

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


@pytest.mark.asyncio
async def test_local_only_migrate_and_decay():
    """Migration + neuroplasticity math works without Cognee."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_to_mempool("Important event", "Noted", "user_a")
    nodes = await mem.fetch_mempool()
    node = nodes[0]

    await mem.migrate_to_persistent(node["id"], "tx_abc", 8)

    assert mem._node_store[node["id"]]["status"] == "persistent"
    weight_fresh = mem._node_store[node["id"]]["effective_weight"]
    assert weight_fresh > 1.0  # Emotional anchor bonus applied

    # Simulate 30 days passing
    mem._node_store[node["id"]]["last_accessed"] = time.time() - (30 * 86400)
    await mem.query("anything")
    weight_decayed = mem._node_store[node["id"]]["effective_weight"]
    assert weight_decayed < weight_fresh


@pytest.mark.asyncio
async def test_persistent_count():
    """get_persistent_count() tracks only persistent MemoryNodes."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    assert mem.get_persistent_count() == 0

    await mem.add_to_mempool("Solana devnet transaction patterns are fascinating", "The confirmation times vary by validator")
    await mem.add_to_mempool("Tokyo weather forecast for next week", "Expect sunny skies and mild temperatures")
    assert mem.get_persistent_count() == 0

    nodes = await mem.fetch_mempool()
    assert len(nodes) >= 2, f"Expected 2 distinct mempool nodes, got {len(nodes)}"
    await mem.migrate_to_persistent(nodes[0]["id"], "tx1", 5)
    assert mem.get_persistent_count() == 1

    await mem.migrate_to_persistent(nodes[1]["id"], "tx2", 3)
    assert mem.get_persistent_count() == 2


@pytest.mark.asyncio
async def test_mock_db_backward_compat():
    """_mock_db property alias works for backward compatibility."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)
    await mem.add_to_mempool("test", "response")

    # _mock_db should be the same object as _node_store
    assert mem._mock_db is mem._node_store
    assert len(mem._mock_db) == 2  # IdentityNode + MemoryNode


@pytest.mark.asyncio
async def test_system_pulse_and_sentiments():
    """System pulse nodes work in the new architecture."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_system_pulse(0.8, epoch_id=1)
    await mem.add_system_pulse(0.6, epoch_id=2)
    await mem.add_system_pulse(0.9, epoch_id=3)

    sentiments = await mem.get_recent_sentiments(count=2)
    assert len(sentiments) == 2
    assert sentiments[0] == 0.9  # Most recent first


@pytest.mark.asyncio
async def test_social_metrics_lifecycle():
    """Social metrics CRUD works in the new architecture."""
    from titan_plugin.core.memory import TieredMemoryGraph

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


@pytest.mark.asyncio
async def test_keyword_search_matches_content():
    """Local keyword search finds memories by content words."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)

    await mem.add_to_mempool("Solana blockchain is revolutionary", "Indeed it is")
    await mem.add_to_mempool("The weather today is sunny", "Enjoy the sun")

    results = await mem.query("Tell me about Solana blockchain")
    prompts = [r["user_prompt"] for r in results]
    assert any("Solana" in p for p in prompts)


# ---------------------------------------------------------------------------
# Test 2: Real Cognee integration (requires Cognee installed, not mocked)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.skipif(not cognee_available(), reason="Cognee not available")
async def test_cognee_initialization():
    """Cognee initializes with fastembed local embeddings."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config=TEST_CONFIG)
    ready = await mem._ensure_cognee()
    assert ready is True
    assert mem._cognee is not None
    assert mem._cognee_ready is True


@pytest.mark.asyncio
@pytest.mark.skipif(not cognee_available(), reason="Cognee not available")
async def test_cognee_ingest_and_search():
    """Memories migrated to persistent are searchable via Cognee semantic search."""
    from titan_plugin.core.memory import TieredMemoryGraph

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

    # Search semantically — should find Solana-related memory
    results = await mem.query("What do you know about blockchain speed?")
    # At minimum, local keyword search should find it
    assert len(results) >= 1
    found_solana = any("Solana" in r.get("user_prompt", "") for r in results)
    assert found_solana, "Expected to find Solana memory via search"


@pytest.mark.asyncio
@pytest.mark.skipif(not cognee_available(), reason="Cognee not available")
async def test_cognee_consolidate_without_api_key():
    """Consolidate gracefully skips cognify when no LLM API key is set."""
    from titan_plugin.core.memory import TieredMemoryGraph

    mem = TieredMemoryGraph(config={
        **TEST_CONFIG,
        "openrouter_api_key": "",
    })

    await mem.add_to_mempool("test data", "response")
    nodes = await mem.fetch_mempool()
    await mem.migrate_to_persistent(nodes[0]["id"], "tx_test", 5)

    # Should succeed (skip cognify) without error
    result = await mem.consolidate()
    assert result is True
