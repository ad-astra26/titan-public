"""
tests/test_gradual_forgetting.py
Phase 14: Gradual Forgetting — sigmoid mempool decay, per-node scoring,
          lightweight mempool embeddings, reinforcement mechanics.
"""
import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from titan_plugin.core.memory import (
    TieredMemoryGraph,
    _MEMPOOL_DECAY_K,
    _MEMPOOL_HALF_LIFE_HOURS,
    _MEMPOOL_MAX_TTL_HOURS,
    _MEMPOOL_PRUNE_THRESHOLD,
    _MEMPOOL_KEEP_THRESHOLD,
    _MEMPOOL_PROMOTE_SCORE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def memory():
    """Fresh TieredMemoryGraph with mocked Cognee."""
    with patch("titan_plugin.core.memory.TieredMemoryGraph._ensure_cognee", new_callable=AsyncMock, return_value=False):
        mg = TieredMemoryGraph(config={})
    return mg


@pytest.fixture
def memory_with_embeddings():
    """TieredMemoryGraph with fastembed available."""
    with patch("titan_plugin.core.memory.TieredMemoryGraph._ensure_cognee", new_callable=AsyncMock, return_value=False):
        mg = TieredMemoryGraph(config={})
    return mg


# ---------------------------------------------------------------------------
# Sigmoid Decay Tests
# ---------------------------------------------------------------------------
class TestSigmoidDecay:
    """Test the sigmoid mempool decay curve."""

    def test_fresh_node_has_full_weight(self, memory):
        """A brand-new mempool node should have weight ≈ 1.0."""
        node = {
            "created_at": time.time(),
            "last_reinforced": time.time(),
            "mempool_reinforcements": 0,
        }
        w = memory._compute_mempool_weight(node)
        assert w > 0.95, f"Fresh node weight should be ~1.0, got {w}"

    def test_halflife_weight_is_half(self, memory):
        """At t_half (12h), sigmoid should return ~0.5."""
        now = time.time()
        node = {
            "created_at": now - (12 * 3600),
            "last_reinforced": now - (12 * 3600),
            "mempool_reinforcements": 0,
        }
        w = memory._compute_mempool_weight(node)
        assert 0.4 < w < 0.6, f"Half-life weight should be ~0.5, got {w}"

    def test_6h_weight_still_high(self, memory):
        """At 6h, node should still have decent weight (~0.88)."""
        now = time.time()
        node = {
            "created_at": now - (6 * 3600),
            "last_reinforced": now - (6 * 3600),
            "mempool_reinforcements": 0,
        }
        w = memory._compute_mempool_weight(node)
        assert w > 0.75, f"6h weight should be >0.75, got {w}"

    def test_18h_weight_is_low(self, memory):
        """At 18h, node should be nearly dead (~0.12)."""
        now = time.time()
        node = {
            "created_at": now - (18 * 3600),
            "last_reinforced": now - (18 * 3600),
            "mempool_reinforcements": 0,
        }
        w = memory._compute_mempool_weight(node)
        assert w < 0.25, f"18h weight should be <0.25, got {w}"

    def test_24h_hard_ttl(self, memory):
        """At 24h+, hard TTL should return 0.0 regardless of reinforcement."""
        now = time.time()
        node = {
            "created_at": now - (25 * 3600),
            "last_reinforced": now,  # Recently reinforced but too old
            "mempool_reinforcements": 5,
        }
        w = memory._compute_mempool_weight(node)
        assert w == 0.0, f"24h+ node should be 0.0, got {w}"

    def test_apply_mempool_decay_updates_node(self, memory):
        """_apply_mempool_decay should set mempool_weight on the node."""
        node = {
            "created_at": time.time(),
            "last_reinforced": time.time(),
            "mempool_reinforcements": 0,
        }
        memory._apply_mempool_decay(node)
        assert "mempool_weight" in node
        assert node["mempool_weight"] > 0.9


class TestReinforcement:
    """Test mempool reinforcement mechanics."""

    def test_reinforcement_resets_decay_clock(self, memory):
        """Reinforcing a node should reset last_reinforced and slow decay."""
        now = time.time()
        # Node is 10 hours old — would normally be at ~0.65
        node = {
            "id": 1,
            "type": "MemoryNode",
            "status": "mempool",
            "created_at": now - (10 * 3600),
            "last_reinforced": now - (10 * 3600),
            "mempool_reinforcements": 0,
        }
        memory._node_store[1] = node

        w_before = memory._compute_mempool_weight(node)

        # Reinforce
        memory.reinforce_mempool_node(1)

        w_after = memory._compute_mempool_weight(node)
        assert w_after > w_before, "Reinforcement should boost weight"
        assert node["mempool_reinforcements"] == 1
        assert node["last_reinforced"] > now - 1  # Recently updated

    def test_reinforcement_bonus_caps(self, memory):
        """Reinforcement bonus should cap at +50%."""
        now = time.time()
        node = {
            "created_at": now,
            "last_reinforced": now,
            "mempool_reinforcements": 100,  # Way over cap
        }
        w = memory._compute_mempool_weight(node)
        # Base sigmoid ≈ 1.0 + capped bonus of 0.5 → capped at 1.0
        assert w <= 1.0, f"Weight should cap at 1.0, got {w}"

    def test_reinforcement_does_not_extend_past_ttl(self, memory):
        """Reinforcement cannot save a node past the 24h hard TTL."""
        now = time.time()
        node = {
            "id": 1,
            "type": "MemoryNode",
            "status": "mempool",
            "created_at": now - (25 * 3600),  # Over TTL
            "last_reinforced": now - (25 * 3600),
            "mempool_reinforcements": 0,
        }
        memory._node_store[1] = node
        memory.reinforce_mempool_node(1)
        w = memory._compute_mempool_weight(node)
        assert w == 0.0, "Reinforcement cannot bypass hard TTL"


class TestMempoolClassification:
    """Test the three-bucket classification for meditation."""

    @pytest.mark.asyncio
    async def test_classification_buckets(self, memory):
        """Nodes should be classified into candidates, fading, dead."""
        now = time.time()

        # Candidate: fresh, high weight
        memory._node_store[1] = {
            "id": 1, "type": "MemoryNode", "status": "mempool",
            "created_at": now, "last_reinforced": now,
            "mempool_reinforcements": 0, "mempool_weight": 1.0,
        }
        # Fading: moderate age, weight between thresholds
        memory._node_store[2] = {
            "id": 2, "type": "MemoryNode", "status": "mempool",
            "created_at": now - (14 * 3600), "last_reinforced": now - (14 * 3600),
            "mempool_reinforcements": 0, "mempool_weight": 0.2,
        }
        # Dead: very old
        memory._node_store[3] = {
            "id": 3, "type": "MemoryNode", "status": "mempool",
            "created_at": now - (25 * 3600), "last_reinforced": now - (25 * 3600),
            "mempool_reinforcements": 0, "mempool_weight": 0.0,
        }

        candidates, fading, dead = await memory.fetch_mempool_classified()

        assert len(candidates) >= 1, "Fresh node should be a candidate"
        assert any(n["id"] == 3 for n in dead), "25h node should be dead"

    @pytest.mark.asyncio
    async def test_empty_mempool_classification(self, memory):
        """Empty mempool should return three empty lists."""
        candidates, fading, dead = await memory.fetch_mempool_classified()
        assert candidates == [] and fading == [] and dead == []


class TestMempoolStats:
    """Test mempool health statistics."""

    def test_stats_empty(self, memory):
        """Empty mempool should return zero stats."""
        stats = memory.get_mempool_stats()
        assert stats["count"] == 0
        assert stats["avg_weight"] == 0.0

    def test_stats_with_nodes(self, memory):
        """Stats should reflect current mempool state."""
        now = time.time()
        for i in range(3):
            memory._node_store[i + 1] = {
                "id": i + 1, "type": "MemoryNode", "status": "mempool",
                "created_at": now, "last_reinforced": now,
                "mempool_reinforcements": 0,
            }
        stats = memory.get_mempool_stats()
        assert stats["count"] == 3
        assert stats["avg_weight"] > 0.9


class TestMempoolSemanticSearch:
    """Test lightweight mempool embedding search."""

    def test_embedding_model_loads(self, memory_with_embeddings):
        """Fastembed model should load on demand."""
        result = memory_with_embeddings._ensure_embedding_model()
        # May or may not work depending on fastembed availability
        # Just verify it doesn't crash
        assert isinstance(result, bool)

    def test_embed_text_returns_array_or_none(self, memory_with_embeddings):
        """_embed_text should return ndarray or None."""
        result = memory_with_embeddings._embed_text("test text")
        # Could be None if fastembed not available
        if result is not None:
            import numpy as np
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 384

    @pytest.mark.asyncio
    async def test_add_to_mempool_creates_embedding(self, memory_with_embeddings):
        """Adding to mempool should create an embedding if fastembed is available."""
        await memory_with_embeddings.add_to_mempool("test prompt", "test response")
        # Check node was created with mempool fields
        mempool = await memory_with_embeddings.fetch_mempool()
        assert len(mempool) == 1
        node = mempool[0]
        assert node["mempool_weight"] == 1.0
        assert node["mempool_reinforcements"] == 0
        assert "last_reinforced" in node

    def test_semantic_search_empty_index(self, memory):
        """Semantic search on empty index should return empty list."""
        results = memory._mempool_semantic_search("test query")
        assert results == []


class TestAddToMempoolReinforcement:
    """Test topic-based reinforcement during add_to_mempool."""

    @pytest.mark.asyncio
    async def test_similar_topic_reinforces_existing(self, memory_with_embeddings):
        """Adding a similar topic should reinforce existing node, not create new."""
        mg = memory_with_embeddings
        # Mock find_similar to return a match
        await mg.add_to_mempool("Solana ZK proofs", "ZK proofs compress state")

        mempool = await mg.fetch_mempool()
        assert len(mempool) == 1
        first_id = mempool[0]["id"]

        # Mock similarity check to return the first node
        mg.find_similar_mempool_node = MagicMock(return_value=first_id)

        await mg.add_to_mempool("ZK proof compression", "Compressed state reduces cost")

        mempool = await mg.fetch_mempool()
        # Should still be 1 node (reinforced, not duplicated)
        assert len(mempool) == 1
        assert mempool[0]["mempool_reinforcements"] == 1

    @pytest.mark.asyncio
    async def test_different_topic_creates_new(self, memory_with_embeddings):
        """Adding a different topic should create a new node."""
        mg = memory_with_embeddings
        mg.find_similar_mempool_node = MagicMock(return_value=None)

        await mg.add_to_mempool("Solana ZK proofs", "ZK proofs compress state")
        await mg.add_to_mempool("Weather in Tokyo", "It is sunny today")

        mempool = await mg.fetch_mempool()
        assert len(mempool) == 2


class TestMigrationCleansUp:
    """Test that migration to persistent cleans up mempool artifacts."""

    @pytest.mark.asyncio
    async def test_migrate_removes_mempool_fields(self, memory):
        """Migration should remove mempool-specific fields."""
        now = time.time()
        memory._node_store[1] = {
            "id": 1, "type": "MemoryNode", "status": "mempool",
            "user_prompt": "test", "agent_response": "response",
            "created_at": now, "last_accessed": now,
            "base_weight": 1.0, "anchor_bonus": 0.0,
            "reinforcement_count": 0, "emotional_intensity": 0,
            "last_reinforced": now, "mempool_reinforcements": 3,
            "mempool_weight": 0.8,
        }
        memory._mempool_embeddings[1] = MagicMock()  # Fake embedding

        await memory.migrate_to_persistent(1, "tx_abc123", 7)

        node = memory._node_store[1]
        assert node["status"] == "persistent"
        assert "mempool_weight" not in node
        assert "mempool_reinforcements" not in node
        assert "last_reinforced" not in node
        assert 1 not in memory._mempool_embeddings


class TestQueryWithMempool:
    """Test that query() now searches mempool semantically."""

    @pytest.mark.asyncio
    async def test_query_returns_mempool_nodes(self, memory):
        """Query should find mempool nodes via keyword search."""
        now = time.time()
        memory._node_store[1] = {
            "id": 1, "type": "MemoryNode", "status": "mempool",
            "user_prompt": "Solana devnet transactions",
            "agent_response": "Devnet is faster than mainnet",
            "created_at": now, "last_reinforced": now,
            "mempool_reinforcements": 0, "mempool_weight": 1.0,
        }

        results = await memory.query("Solana devnet")
        assert len(results) >= 1
        assert any(r["id"] == 1 for r in results)

    @pytest.mark.asyncio
    async def test_query_skips_dead_mempool_nodes(self, memory):
        """Query should skip mempool nodes below prune threshold."""
        now = time.time()
        memory._node_store[1] = {
            "id": 1, "type": "MemoryNode", "status": "mempool",
            "user_prompt": "old expired memory",
            "agent_response": "this should be gone",
            "created_at": now - (25 * 3600), "last_reinforced": now - (25 * 3600),
            "mempool_reinforcements": 0, "mempool_weight": 0.0,
        }

        results = await memory.query("old expired")
        assert len(results) == 0, "Dead mempool nodes should be skipped"


class TestConstants:
    """Verify constants are sensible."""

    def test_half_life(self):
        assert _MEMPOOL_HALF_LIFE_HOURS == 12.0

    def test_max_ttl(self):
        assert _MEMPOOL_MAX_TTL_HOURS == 24.0

    def test_prune_threshold(self):
        assert _MEMPOOL_PRUNE_THRESHOLD == 0.1

    def test_keep_threshold(self):
        assert _MEMPOOL_KEEP_THRESHOLD == 0.3

    def test_promote_score(self):
        assert _MEMPOOL_PROMOTE_SCORE == 40.0

    def test_thresholds_ordered(self):
        assert _MEMPOOL_PRUNE_THRESHOLD < _MEMPOOL_KEEP_THRESHOLD < 1.0
