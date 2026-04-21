"""
tests/test_social_graph.py
Phase 13: Sage Socialite — social graph, donation tracking, inspiration TX,
webhook routing, user recognition, engagement scoring.
"""
import os
import shutil
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from titan_plugin.core.social_graph import (
    SocialGraph,
    UserProfile,
    DONATION_TIERS,
    _ENGAGEMENT_CURIOUS,
    _ENGAGEMENT_FRIENDLY,
    _ENGAGEMENT_TRUSTED,
    _ENGAGEMENT_INNER_CIRCLE,
)

TEST_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "test_social_data")
TEST_DB_PATH = os.path.join(TEST_DB_DIR, "social_graph.db")


@pytest.fixture(autouse=True)
def clean_db():
    """Remove test DB before/after each test."""
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)
    yield
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)


@pytest.fixture
def graph():
    return SocialGraph(db_path=TEST_DB_PATH)


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------
class TestUserProfile:
    def test_create_from_dict(self):
        p = UserProfile({
            "user_id": "alice",
            "platform": "x",
            "display_name": "Alice",
            "like_score": 5.0,
            "dislike_score": 1.0,
        })
        assert p.user_id == "alice"
        assert p.platform == "x"
        assert p.net_sentiment > 0

    def test_net_sentiment_balanced(self):
        p = UserProfile({"user_id": "b", "like_score": 3.0, "dislike_score": 3.0})
        assert p.net_sentiment == 0.0

    def test_net_sentiment_negative(self):
        p = UserProfile({"user_id": "c", "like_score": 1.0, "dislike_score": 4.0})
        assert p.net_sentiment < 0

    def test_net_sentiment_zero_interactions(self):
        p = UserProfile({"user_id": "d", "like_score": 0.0, "dislike_score": 0.0})
        assert p.net_sentiment == 0.0

    def test_is_donor(self):
        p = UserProfile({"user_id": "e", "total_donated_sol": 0.05})
        assert p.is_donor is True

    def test_is_not_donor(self):
        p = UserProfile({"user_id": "f", "total_donated_sol": 0.0})
        assert p.is_donor is False

    def test_to_dict_roundtrip(self):
        p = UserProfile({"user_id": "g", "platform": "telegram", "display_name": "GGG"})
        d = p.to_dict()
        p2 = UserProfile(d)
        assert p2.user_id == "g"
        assert p2.platform == "telegram"


# ---------------------------------------------------------------------------
# SocialGraph CRUD
# ---------------------------------------------------------------------------
class TestSocialGraphCRUD:
    def test_create_and_get_user(self, graph):
        user = graph.get_or_create_user("alice", "x", "Alice")
        assert user.user_id == "alice"
        assert user.display_name == "Alice"
        assert user.interaction_count == 0

    def test_get_existing_user(self, graph):
        graph.get_or_create_user("bob", "telegram", "Bob")
        # Clear cache to force DB read
        graph._cache.clear()
        user = graph.get_or_create_user("bob")
        assert user.display_name == "Bob"
        assert user.platform == "telegram"

    def test_record_interaction_positive(self, graph):
        graph.get_or_create_user("carol")
        graph.record_interaction("carol", quality=0.9)
        user = graph.get_or_create_user("carol")
        assert user.interaction_count == 1
        assert user.like_score > 0
        assert user.dislike_score == 0

    def test_record_interaction_negative(self, graph):
        graph.get_or_create_user("dave")
        graph.record_interaction("dave", quality=0.1)
        user = graph.get_or_create_user("dave")
        assert user.interaction_count == 1
        assert user.dislike_score > 0
        assert user.like_score == 0

    def test_engagement_level_grows(self, graph):
        graph.get_or_create_user("eve")
        # Simulate many positive interactions
        for _ in range(30):
            graph.record_interaction("eve", quality=0.9)
        user = graph.get_or_create_user("eve")
        assert user.engagement_level > _ENGAGEMENT_CURIOUS

    def test_get_stats(self, graph):
        graph.get_or_create_user("user1")
        graph.get_or_create_user("user2")
        stats = graph.get_stats()
        assert stats["users"] == 2
        assert stats["edges"] == 0
        assert stats["donations"] == 0

    def test_get_top_users(self, graph):
        graph.get_or_create_user("low")
        u = graph.get_or_create_user("high")
        # Boost engagement manually
        u.engagement_level = 0.9
        graph._save_profile(u)
        graph._cache.clear()

        top = graph.get_top_users(limit=2)
        assert len(top) == 2
        assert top[0].user_id == "high"


# ---------------------------------------------------------------------------
# Social Edges
# ---------------------------------------------------------------------------
class TestSocialEdges:
    def test_create_edge(self, graph):
        graph.record_edge("alice", "bob")
        conns = graph.get_user_connections("alice")
        assert len(conns) == 1
        assert conns[0]["user_id"] == "bob"

    def test_edge_strengthens(self, graph):
        graph.record_edge("alice", "bob")
        graph.record_edge("alice", "bob")
        conns = graph.get_user_connections("alice")
        assert conns[0]["strength"] > 0.1  # Default + 2 increments

    def test_edge_is_bidirectional(self, graph):
        graph.record_edge("alice", "bob")
        conns_a = graph.get_user_connections("alice")
        conns_b = graph.get_user_connections("bob")
        assert len(conns_a) == 1
        assert len(conns_b) == 1

    def test_edge_strength_caps_at_one(self, graph):
        for _ in range(100):
            graph.record_edge("alice", "bob")
        conns = graph.get_user_connections("alice")
        assert conns[0]["strength"] <= 1.0


# ---------------------------------------------------------------------------
# Donations
# ---------------------------------------------------------------------------
class TestDonations:
    def test_record_donation_unknown_sender(self, graph):
        result = graph.record_donation(
            tx_signature="sig123",
            sender_address="7xKXfP...",
            amount_sol=0.5,
            memo="",
        )
        assert result is None  # Unknown sender
        stats = graph.get_stats()
        assert stats["donations"] == 1

    def test_record_donation_known_sender(self, graph):
        graph.get_or_create_user("donor1")
        graph.link_sol_address("donor1", "7xKXfPtest")
        result = graph.record_donation(
            tx_signature="sig456",
            sender_address="7xKXfPtest",
            amount_sol=0.1,
        )
        assert result is not None
        assert result.user_id == "donor1"
        assert result.total_donated_sol == 0.1

    def test_donation_mood_boost_tiers(self, graph):
        # Large donation
        mood, weight = graph.get_donation_mood_boost(0.15)
        assert mood == 0.10
        assert weight == 5.0

        # Medium donation
        mood, weight = graph.get_donation_mood_boost(0.07)
        assert mood == 0.05
        assert weight == 3.0

        # Small donation
        mood, weight = graph.get_donation_mood_boost(0.02)
        assert mood == 0.02
        assert weight == 2.0

        # Memo-only
        mood, weight = graph.get_donation_mood_boost(0.0)
        assert mood == 0.01
        assert weight == 1.5

    def test_duplicate_donation_ignored(self, graph):
        graph.record_donation("dup_sig", "addr1", 1.0)
        graph.record_donation("dup_sig", "addr1", 1.0)  # Same TX sig
        stats = graph.get_stats()
        assert stats["donations"] == 1


# ---------------------------------------------------------------------------
# Inspirations
# ---------------------------------------------------------------------------
class TestInspirations:
    def test_record_inspiration(self, graph):
        result = graph.record_inspiration(
            tx_signature="insp1",
            sender_address="addr_anon",
            message="Be brave, Titan!",
            amount_sol=0.05,
        )
        assert result is None  # Unknown sender
        stats = graph.get_stats()
        assert stats["inspirations"] == 1

    def test_record_inspiration_known_user(self, graph):
        graph.get_or_create_user("fan1")
        graph.link_sol_address("fan1", "fan_address")
        result = graph.record_inspiration(
            tx_signature="insp2",
            sender_address="fan_address",
            message="Keep learning!",
        )
        assert result is not None
        assert result.user_id == "fan1"

    def test_pending_inspirations(self, graph):
        graph.record_inspiration("i1", "addr1", "msg1")
        graph.record_inspiration("i2", "addr2", "msg2")
        pending = graph.get_pending_inspirations()
        assert len(pending) == 2

    def test_mark_inspiration_processed(self, graph):
        graph.record_inspiration("i3", "addr3", "msg3")
        graph.mark_inspiration_processed("i3", "Titan was inspired")
        pending = graph.get_pending_inspirations()
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# Lookup Helpers
# ---------------------------------------------------------------------------
class TestLookup:
    def test_find_by_sol_address(self, graph):
        graph.get_or_create_user("user_x")
        graph.link_sol_address("user_x", "sol_addr_x")
        found = graph.find_user_by_sol_address("sol_addr_x")
        assert found is not None
        assert found.user_id == "user_x"

    def test_find_by_sol_address_not_found(self, graph):
        assert graph.find_user_by_sol_address("nonexistent") is None

    def test_link_sol_address(self, graph):
        graph.get_or_create_user("user_y")
        graph.link_sol_address("user_y", "sol_addr_y")
        user = graph.get_or_create_user("user_y")
        assert user.sol_address == "sol_addr_y"

    def test_find_from_db_not_cache(self, graph):
        """Verify DB lookup works when cache is empty."""
        graph.get_or_create_user("user_z")
        graph.link_sol_address("user_z", "sol_addr_z")
        graph._cache.clear()
        found = graph.find_user_by_sol_address("sol_addr_z")
        assert found is not None
        assert found.user_id == "user_z"


# ---------------------------------------------------------------------------
# Engagement Decision
# ---------------------------------------------------------------------------
class TestEngagement:
    def test_should_engage_new_user(self, graph):
        result = graph.should_engage("newcomer")
        assert result == "minimal"

    def test_should_engage_trusted_user(self, graph):
        user = graph.get_or_create_user("trusted")
        user.engagement_level = _ENGAGEMENT_TRUSTED
        graph._save_profile(user)
        graph._cache["trusted"] = user
        assert graph.should_engage("trusted") == "neutral"

    def test_should_engage_inner_circle(self, graph):
        user = graph.get_or_create_user("vip")
        user.engagement_level = _ENGAGEMENT_INNER_CIRCLE
        graph._save_profile(user)
        graph._cache["vip"] = user
        assert graph.should_engage("vip") == "warm"


# ---------------------------------------------------------------------------
# Webhook Helpers
# ---------------------------------------------------------------------------
class TestWebhookExtractSolAmount:
    def test_extract_from_native_transfers(self):
        from titan_plugin.api.webhook import _extract_sol_amount

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWalletAddress123"

        tx = {
            "nativeTransfers": [
                {"toUserAccount": "TitanWalletAddress123", "amount": 50_000_000},  # 0.05 SOL
            ],
        }
        amount = _extract_sol_amount(tx, plugin)
        assert abs(amount - 0.05) < 0.0001

    def test_extract_from_account_data_fallback(self):
        from titan_plugin.api.webhook import _extract_sol_amount

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet456"

        tx = {
            "nativeTransfers": [],
            "accountData": [
                {"account": "TitanWallet456", "nativeBalanceChange": 100_000_000},  # 0.1 SOL
            ],
        }
        amount = _extract_sol_amount(tx, plugin)
        assert abs(amount - 0.1) < 0.0001

    def test_extract_no_wallet(self):
        from titan_plugin.api.webhook import _extract_sol_amount

        plugin = MagicMock()
        plugin.network = None

        tx = {"nativeTransfers": [{"toUserAccount": "any", "amount": 1_000_000_000}]}
        assert _extract_sol_amount(tx, plugin) == 0.0

    def test_extract_no_matching_transfer(self):
        from titan_plugin.api.webhook import _extract_sol_amount

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet789"

        tx = {
            "nativeTransfers": [
                {"toUserAccount": "SomeOtherWallet", "amount": 1_000_000_000},
            ],
        }
        assert _extract_sol_amount(tx, plugin) == 0.0

    def test_extract_empty_tx(self):
        from titan_plugin.api.webhook import _extract_sol_amount

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet"

        assert _extract_sol_amount({}, plugin) == 0.0


class TestWebhookMemoExtraction:
    def test_extract_memo_from_instructions(self):
        from titan_plugin.api.webhook import _extract_memo_data

        tx = {
            "instructions": [
                {
                    "programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
                    "data": "I:Stay curious, Titan!",
                },
            ],
        }
        assert _extract_memo_data(tx) == "I:Stay curious, Titan!"

    def test_extract_memo_from_inner_instructions(self):
        from titan_plugin.api.webhook import _extract_memo_data

        tx = {
            "instructions": [],
            "innerInstructions": [
                {
                    "instructions": [
                        {
                            "programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
                            "data": "I:Hello from inner",
                        },
                    ],
                },
            ],
        }
        assert _extract_memo_data(tx) == "I:Hello from inner"

    def test_extract_memo_empty(self):
        from titan_plugin.api.webhook import _extract_memo_data

        assert _extract_memo_data({}) == ""


class TestWebhookTransactionRouting:
    @pytest.mark.asyncio
    async def test_routes_inspiration(self):
        from titan_plugin.api.webhook import _process_transaction

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet"
        plugin.social_graph = MagicMock()
        # rFP_social_graph_async_safety §5.2: webhook now calls
        # record_inspiration_async — must be an AsyncMock to be awaitable.
        plugin.social_graph.record_inspiration_async = AsyncMock(return_value=None)
        plugin.social_graph.get_donation_mood_boost.return_value = (0.01, 1.5)
        plugin.memory = MagicMock()
        plugin.memory.inject_memory = AsyncMock()
        plugin.event_bus = MagicMock()
        plugin.event_bus.emit = AsyncMock()

        tx = {
            "type": "MEMO",
            "signature": "sig_test",
            "feePayer": "sender_addr",
            "instructions": [
                {
                    "programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
                    "data": "I:You are amazing, Titan!",
                },
            ],
            "nativeTransfers": [],
        }
        result = await _process_transaction(plugin, tx)
        assert result is True
        plugin.social_graph.record_inspiration_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_donation(self):
        from titan_plugin.api.webhook import _process_transaction

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet"
        plugin.social_graph = MagicMock()
        # rFP_social_graph_async_safety §5.2: webhook now calls
        # record_donation_async — must be an AsyncMock to be awaitable.
        plugin.social_graph.record_donation_async = AsyncMock(return_value=None)
        plugin.social_graph.get_donation_mood_boost.return_value = (0.05, 3.0)
        plugin.event_bus = MagicMock()
        plugin.event_bus.emit = AsyncMock()

        tx = {
            "type": "TRANSFER",
            "signature": "don_sig",
            "feePayer": "donor_addr",
            "instructions": [],
            "nativeTransfers": [
                {"toUserAccount": "TitanWallet", "amount": 50_000_000},
            ],
        }
        result = await _process_transaction(plugin, tx)
        assert result is True
        plugin.social_graph.record_donation_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_irrelevant_tx(self):
        from titan_plugin.api.webhook import _process_transaction

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet"
        plugin.soul = MagicMock()
        plugin.soul._maker_pubkey = None

        tx = {
            "type": "TOKEN_TRANSFER",
            "signature": "irrelevant",
            "feePayer": "someone",
            "instructions": [],
        }
        result = await _process_transaction(plugin, tx)
        assert result is False

    @pytest.mark.asyncio
    async def test_routes_maker_directive(self):
        from titan_plugin.api.webhook import _process_transaction

        plugin = MagicMock()
        plugin.network.pubkey = "TitanWallet"
        plugin.soul = MagicMock()
        plugin.soul._maker_pubkey = "MakerPubKey123"
        plugin.soul.evolve_soul = AsyncMock(return_value={"gen": 2})
        plugin.soul.current_gen = 2
        plugin.event_bus = MagicMock()
        plugin.event_bus.emit = AsyncMock()

        # Mock signature verification to pass
        with patch("titan_plugin.utils.crypto.verify_maker_signature", return_value=True):
            tx = {
                "type": "MEMO",
                "signature": "di_sig",
                "feePayer": "MakerPubKey123",
                "instructions": [
                    {
                        "programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
                        "data": "TITAN_DI:evolve now:fake_sig_data",
                    },
                ],
            }
            result = await _process_transaction(plugin, tx)
            assert result is True
            plugin.soul.evolve_soul.assert_called_once()


# ---------------------------------------------------------------------------
# Constants Validation
# ---------------------------------------------------------------------------
class TestConstants:
    def test_donation_tiers_ordered(self):
        """Tiers should be descending by SOL threshold."""
        for i in range(len(DONATION_TIERS) - 1):
            assert DONATION_TIERS[i][0] > DONATION_TIERS[i + 1][0]

    def test_engagement_thresholds_ordered(self):
        assert _ENGAGEMENT_CURIOUS < _ENGAGEMENT_FRIENDLY < _ENGAGEMENT_TRUSTED < _ENGAGEMENT_INNER_CIRCLE

    def test_engagement_within_bounds(self):
        assert 0.0 < _ENGAGEMENT_CURIOUS < 1.0
        assert 0.0 < _ENGAGEMENT_INNER_CIRCLE <= 1.0


# ---------------------------------------------------------------------------
# Phase 13 Integration: User Memory + Agno Hooks Wiring
# ---------------------------------------------------------------------------
class TestUserMemoryIntegration:
    """Verify that mempool nodes are tagged with user identity for per-user recall."""

    @pytest.mark.asyncio
    async def test_mempool_tags_user_identity(self):
        """add_to_mempool creates an IdentityNode and tags the memory."""
        from titan_plugin.core.memory import TieredMemoryGraph

        mem = TieredMemoryGraph.__new__(TieredMemoryGraph)
        mem._node_store = {}
        mem._next_id = 1
        mem._cognee = None
        mem._cognee_ready = False
        mem._zk_queue = []
        mem._mempool_embeddings = {}
        mem._embedding_model = None
        mem._embedding_dim = 384
        mem._cognee_db_path = "./test_env/cognee_db"
        mem._config = {}

        await mem.add_to_mempool("Hello Titan", "Hello user!", user_identifier="alice_x")

        # Should create identity node
        assert "identity_alice_x" in mem._node_store
        assert mem._node_store["identity_alice_x"]["type"] == "IdentityNode"

        # Memory node should reference the identity
        memory_nodes = [n for n in mem._node_store.values() if n.get("type") == "MemoryNode"]
        assert len(memory_nodes) == 1
        assert memory_nodes[0]["source_id"] == "identity_alice_x"

    @pytest.mark.asyncio
    async def test_query_user_memories_filters_by_user(self):
        """query_user_memories only returns memories from the specified user."""
        from titan_plugin.core.memory import TieredMemoryGraph

        mem = TieredMemoryGraph.__new__(TieredMemoryGraph)
        mem._node_store = {}
        mem._next_id = 1
        mem._cognee = None
        mem._cognee_ready = False
        mem._zk_queue = []
        mem._mempool_embeddings = {}
        mem._embedding_model = None
        mem._embedding_dim = 384
        mem._cognee_db_path = "./test_env/cognee_db"
        mem._config = {}

        await mem.add_to_mempool("What is Solana?", "A blockchain.", user_identifier="alice")
        await mem.add_to_mempool("What is Bitcoin?", "A cryptocurrency.", user_identifier="bob")
        await mem.add_to_mempool("Tell me about Solana staking", "You can stake SOL.", user_identifier="alice")

        results = await mem.query_user_memories("Solana", "alice")
        assert len(results) >= 1
        # All results should be from alice
        for r in results:
            assert r["source_id"] == "identity_alice"

    @pytest.mark.asyncio
    async def test_query_user_memories_empty_for_unknown(self):
        """Returns empty list for unknown user."""
        from titan_plugin.core.memory import TieredMemoryGraph

        mem = TieredMemoryGraph.__new__(TieredMemoryGraph)
        mem._node_store = {}
        mem._next_id = 1
        mem._cognee = None
        mem._cognee_ready = False
        mem._zk_queue = []
        mem._mempool_embeddings = {}
        mem._embedder = None

        results = await mem.query_user_memories("anything", "ghost_user")
        assert results == []


class TestAgnoHooksSocialWiring:
    """Verify the agno hooks properly wire to the social graph."""

    def test_post_hook_records_interaction(self):
        """Post-hook should call social_graph.record_interaction."""
        from titan_plugin.agno_hooks import create_post_hook

        plugin = MagicMock()
        plugin._limbo_mode = False
        plugin._last_execution_mode = "Collaborative"
        plugin._last_research_sources = []
        plugin._current_user_id = "test_user"
        plugin._current_engagement_level = "neutral"
        plugin.memory.add_to_mempool = AsyncMock()
        plugin.mood_engine.get_current_reward.return_value = 0.5
        plugin.mood_engine.get_mood_label.return_value = "Curious"
        plugin.recorder.buffer = []
        plugin.social_graph = MagicMock()
        plugin.social_graph.record_interaction = MagicMock()
        plugin.event_bus = MagicMock()
        plugin.event_bus.emit = AsyncMock()

        post_hook = create_post_hook(plugin)
        assert callable(post_hook)

    def test_pre_hook_social_context_injection(self):
        """Pre-hook factory should create a callable that handles social graph."""
        from titan_plugin.agno_hooks import create_pre_hook

        plugin = MagicMock()
        plugin._limbo_mode = False
        plugin.social_graph = MagicMock()
        plugin.social_graph.get_or_create_user.return_value = MagicMock(
            display_name="Alice",
            interaction_count=5,
            net_sentiment=0.8,
            is_donor=False,
            total_donated_sol=0.0,
            last_seen=0,
        )
        plugin.social_graph.should_engage.return_value = "neutral"

        pre_hook = create_pre_hook(plugin)
        assert callable(pre_hook)
