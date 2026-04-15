"""
tests/test_zk_compression.py
ZK Compression test suite for Phase 8 — Light Protocol integration.

Offline tests (batch root, instruction builders, decoders, dual-mode fallback)
run without any RPC connection. Live tests (Photon client, integrity loop)
require a Helius devnet endpoint and skip gracefully if unavailable.
"""
import asyncio
import hashlib
import json
import os
import struct
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HELIUS_RPC_URL = os.environ.get(
    "HELIUS_RPC_URL",
    "https://devnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY",
)


@pytest.fixture
def test_env(tmp_path):
    """Provide an isolated temp directory for ZK queue persistence tests."""
    return tmp_path


# ---------------------------------------------------------------------------
# TestBatchRootComputation (offline)
# ---------------------------------------------------------------------------

class TestBatchRootComputation:
    """Test client-side Merkle root computation."""

    def test_single_hash_root(self):
        """Root of a single hash is the hash itself."""
        from titan_plugin.utils.solana_client import compute_batch_root

        h = hashlib.sha256(b"memory_1").digest()
        root = compute_batch_root([h])
        assert root == h

    def test_deterministic(self):
        """Same hashes always produce the same root."""
        from titan_plugin.utils.solana_client import compute_batch_root

        hashes = [hashlib.sha256(f"mem_{i}".encode()).digest() for i in range(5)]
        root1 = compute_batch_root(hashes)
        root2 = compute_batch_root(hashes)
        assert root1 == root2

    def test_different_hashes_different_roots(self):
        """Different input hashes produce different roots."""
        from titan_plugin.utils.solana_client import compute_batch_root

        hashes_a = [hashlib.sha256(b"alpha").digest()]
        hashes_b = [hashlib.sha256(b"beta").digest()]
        assert compute_batch_root(hashes_a) != compute_batch_root(hashes_b)

    def test_order_matters(self):
        """[A,B] produces a different root than [B,A]."""
        from titan_plugin.utils.solana_client import compute_batch_root

        a = hashlib.sha256(b"first").digest()
        b = hashlib.sha256(b"second").digest()
        assert compute_batch_root([a, b]) != compute_batch_root([b, a])

    def test_empty_returns_zeros(self):
        """Empty list returns 32 zero bytes."""
        from titan_plugin.utils.solana_client import compute_batch_root

        root = compute_batch_root([])
        assert root == b"\x00" * 32

    def test_two_hash_root(self):
        """Two hashes produce SHA-256(left || right)."""
        from titan_plugin.utils.solana_client import compute_batch_root

        a = hashlib.sha256(b"A").digest()
        b = hashlib.sha256(b"B").digest()
        expected = hashlib.sha256(a + b).digest()
        assert compute_batch_root([a, b]) == expected

    def test_three_hash_odd_promotion(self):
        """Three hashes: pair first two, promote third."""
        from titan_plugin.utils.solana_client import compute_batch_root

        a = hashlib.sha256(b"A").digest()
        b = hashlib.sha256(b"B").digest()
        c = hashlib.sha256(b"C").digest()
        # Level 1: [SHA256(a||b), c]
        ab = hashlib.sha256(a + b).digest()
        # Level 2: SHA256(ab || c)
        expected = hashlib.sha256(ab + c).digest()
        assert compute_batch_root([a, b, c]) == expected


# ---------------------------------------------------------------------------
# TestCompressedInstructionBuilders (offline)
# ---------------------------------------------------------------------------

class TestCompressedInstructionBuilders:
    """Test instruction data packing (offline, no RPC needed)."""

    def test_compress_memory_batch_data_layout(self):
        """Verify discriminator and field packing for compress_memory_batch."""
        from titan_plugin.utils.solana_client import (
            build_compress_memory_batch_instruction,
            _VAULT_IX_COMPRESS_MEMORY_BATCH,
            is_available,
        )

        if not is_available():
            pytest.skip("Solana SDK not installed")

        from solders.pubkey import Pubkey

        authority = Pubkey.new_unique()
        batch_root = hashlib.sha256(b"test_root").digest()
        ix = build_compress_memory_batch_instruction(
            authority_pubkey=authority,
            batch_root=batch_root,
            node_count=42,
            epoch_id=100,
            sovereignty_score=5000,
        )
        assert ix is not None
        data = bytes(ix.data)

        # Check discriminator
        assert data[:8] == _VAULT_IX_COMPRESS_MEMORY_BATCH

        # Check proof = None (byte 0)
        assert data[8] == 0

        # Check batch_root
        offset = 9  # 8 disc + 1 proof option
        assert data[offset:offset + 32] == batch_root

        # Check node_count
        offset += 32
        assert struct.unpack_from("<H", data, offset)[0] == 42

        # Check epoch_id
        offset += 2
        assert struct.unpack_from("<Q", data, offset)[0] == 100

        # Check sovereignty_score
        offset += 8
        assert struct.unpack_from("<H", data, offset)[0] == 5000

    def test_append_epoch_snapshot_data_layout(self):
        """Verify discriminator and field packing for append_epoch_snapshot."""
        from titan_plugin.utils.solana_client import (
            build_append_epoch_snapshot_instruction,
            _VAULT_IX_APPEND_EPOCH_SNAPSHOT,
            is_available,
        )

        if not is_available():
            pytest.skip("Solana SDK not installed")

        from solders.pubkey import Pubkey

        authority = Pubkey.new_unique()
        state_root = hashlib.sha256(b"state").digest()
        shadow_hash = hashlib.sha256(b"shadow").digest()

        ix = build_append_epoch_snapshot_instruction(
            authority_pubkey=authority,
            state_root=state_root,
            memory_count=1000,
            sovereignty_score=7500,
            shadow_url_hash=shadow_hash,
        )
        assert ix is not None
        data = bytes(ix.data)

        # Check discriminator
        assert data[:8] == _VAULT_IX_APPEND_EPOCH_SNAPSHOT

        # Check proof = None
        assert data[8] == 0

        # Check state_root
        offset = 9
        assert data[offset:offset + 32] == state_root

        # Check memory_count
        offset += 32
        assert struct.unpack_from("<Q", data, offset)[0] == 1000

        # Check sovereignty_score
        offset += 8
        assert struct.unpack_from("<H", data, offset)[0] == 7500

        # Check shadow_url_hash
        offset += 2
        assert data[offset:offset + 32] == shadow_hash

    def test_decode_compressed_memory_batch_round_trip(self):
        """Encode and decode a CompressedMemoryBatch."""
        from titan_plugin.utils.solana_client import decode_compressed_memory_batch

        authority = b"\x01" * 32
        batch_root = hashlib.sha256(b"memories").digest()

        # Build raw data matching the Borsh layout
        data = (
            authority
            + struct.pack("<Q", 7)      # epoch_id
            + struct.pack("<q", 1710000000)  # timestamp
            + struct.pack("<H", 8000)   # sovereignty_score
            + batch_root
            + struct.pack("<H", 15)     # node_count
        )

        decoded = decode_compressed_memory_batch(data)
        assert decoded is not None
        assert decoded["type"] == "CompressedMemoryBatch"
        assert decoded["epoch_id"] == 7
        assert decoded["timestamp"] == 1710000000
        assert decoded["sovereignty_score"] == 8000
        assert decoded["node_count"] == 15
        assert decoded["batch_root"] == batch_root.hex()

    def test_decode_compressed_epoch_snapshot_round_trip(self):
        """Encode and decode a CompressedEpochSnapshot."""
        from titan_plugin.utils.solana_client import decode_compressed_epoch_snapshot

        authority = b"\x02" * 32
        state_root = hashlib.sha256(b"state").digest()
        shadow_hash = hashlib.sha256(b"shadow").digest()

        data = (
            authority
            + struct.pack("<Q", 10)         # epoch_number
            + state_root
            + struct.pack("<Q", 500)        # memory_count
            + struct.pack("<H", 9500)       # sovereignty_score
            + shadow_hash
            + struct.pack("<q", 1710100000) # timestamp
        )

        decoded = decode_compressed_epoch_snapshot(data)
        assert decoded is not None
        assert decoded["type"] == "CompressedEpochSnapshot"
        assert decoded["epoch_number"] == 10
        assert decoded["memory_count"] == 500
        assert decoded["sovereignty_score"] == 9500
        assert decoded["timestamp"] == 1710100000

    def test_decode_too_short_returns_none(self):
        """Short data returns None, no crash."""
        from titan_plugin.utils.solana_client import (
            decode_compressed_memory_batch,
            decode_compressed_epoch_snapshot,
        )

        assert decode_compressed_memory_batch(b"\x00" * 10) is None
        assert decode_compressed_epoch_snapshot(b"\x00" * 10) is None
        assert decode_compressed_memory_batch(None) is None
        assert decode_compressed_epoch_snapshot(None) is None


# ---------------------------------------------------------------------------
# TestDualModeFallback (mocked)
# ---------------------------------------------------------------------------

class TestDualModeFallback:
    """Test the 3-tier degradation strategy."""

    @pytest.mark.asyncio
    async def test_queue_persistence(self, test_env):
        """ZK queue survives restart via file-backed persistence."""
        from titan_plugin.core.memory import TieredMemoryGraph

        config = {"cognee_db_path": str(test_env / "cognee_db")}

        # Create graph and queue some hashes
        graph1 = TieredMemoryGraph(config=config)
        h1 = hashlib.sha256(b"mem1").digest()
        h2 = hashlib.sha256(b"mem2").digest()
        graph1._queue_for_compression(h1)
        graph1._queue_for_compression(h2)
        assert len(graph1._zk_queue) == 2

        # Create new graph instance — should recover from disk
        graph2 = TieredMemoryGraph(config=config)
        assert len(graph2._zk_queue) == 2
        assert graph2._zk_queue[0] == h1
        assert graph2._zk_queue[1] == h2

    @pytest.mark.asyncio
    async def test_drain_clears_queue(self, test_env):
        """drain_zk_queue returns hashes and clears the queue."""
        from titan_plugin.core.memory import TieredMemoryGraph

        config = {"cognee_db_path": str(test_env / "cognee_db")}
        graph = TieredMemoryGraph(config=config)

        h1 = hashlib.sha256(b"test1").digest()
        h2 = hashlib.sha256(b"test2").digest()
        graph._queue_for_compression(h1)
        graph._queue_for_compression(h2)

        drained = graph.drain_zk_queue()
        assert len(drained) == 2
        assert len(graph._zk_queue) == 0

        # Disk should also be empty after drain
        graph2 = TieredMemoryGraph(config=config)
        assert len(graph2._zk_queue) == 0

    @pytest.mark.asyncio
    async def test_migrate_queues_hash(self, test_env):
        """migrate_to_persistent queues memory hash for ZK compression."""
        from titan_plugin.core.memory import TieredMemoryGraph

        config = {"cognee_db_path": str(test_env / "cognee_db")}
        graph = TieredMemoryGraph(config=config)

        # Add to mempool
        await graph.add_to_mempool("hello", "world")
        mempool = await graph.fetch_mempool()
        assert len(mempool) == 1
        node_id = mempool[0]["id"]

        # Migrate — should queue a hash
        await graph.migrate_to_persistent(node_id, "tx_sig_123", 5)
        assert len(graph._zk_queue) == 1
        assert len(graph._zk_queue[0]) == 32  # SHA-256

    @pytest.mark.asyncio
    async def test_tier3_offline_requeue(self, test_env):
        """When SDK unavailable, _zk_batch_compress re-queues hashes."""
        from titan_plugin.logic.meditation import MeditationEpoch
        from titan_plugin.core.memory import TieredMemoryGraph

        config = {"cognee_db_path": str(test_env / "cognee_db")}
        memory = TieredMemoryGraph(config=config)
        h1 = hashlib.sha256(b"offline_test").digest()
        memory._queue_for_compression(h1)

        network = MagicMock()
        network.pubkey = None  # No wallet loaded

        meditation = MeditationEpoch(memory, network)
        meditation._photon = None

        await meditation._zk_batch_compress()

        # Hash should be re-queued since SDK/wallet not available
        assert len(memory._zk_queue) == 1


# ---------------------------------------------------------------------------
# TestPhotonClient (live devnet — skips if RPC unavailable)
# ---------------------------------------------------------------------------

class TestPhotonClient:
    """Test Photon JSON-RPC client against live devnet."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Photon responds to getIndexerHealth."""
        from titan_plugin.utils.photon_client import PhotonClient

        client = PhotonClient(HELIUS_RPC_URL)
        try:
            result = await client.health_check()
        except Exception:
            pytest.skip("Helius RPC not reachable")
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_account(self):
        """Fetching a non-existent compressed account returns None gracefully."""
        from titan_plugin.utils.photon_client import PhotonClient

        client = PhotonClient(HELIUS_RPC_URL)
        try:
            ok = await client.health_check()
            if not ok:
                pytest.skip("Photon not healthy")
        except Exception:
            pytest.skip("Helius RPC not reachable")

        result = await client.fetch_compressed_account(
            "1111111111111111111111111111111111111111111"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rpc_error_handling(self):
        """Bad URL returns None, no crash."""
        from titan_plugin.utils.photon_client import PhotonClient

        client = PhotonClient("http://localhost:1/invalid")
        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_accounts_by_owner(self):
        """Fetch compressed accounts by owner returns list (possibly empty)."""
        from titan_plugin.utils.photon_client import PhotonClient

        client = PhotonClient(HELIUS_RPC_URL)
        try:
            ok = await client.health_check()
            if not ok:
                pytest.skip("Photon not healthy")
        except Exception:
            pytest.skip("Helius RPC not reachable")

        # Use a real devnet pubkey — may return empty list
        result = await client.fetch_compressed_accounts_by_owner(
            "8LBHvVcskwpDJsDEVYMhNCRMDi3NV4eHnynhLUo5XrrS"
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestConfigIntegration (offline)
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify helius_rpc_url flows through config pipeline."""

    def test_config_has_helius_key(self):
        """config.toml contains helius_rpc_url under [network]."""
        import os

        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib  # type: ignore

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "titan_plugin",
            "config.toml",
        )
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        assert "network" in config
        assert "helius_rpc_url" in config["network"]

    def test_no_photon_when_empty(self):
        """When helius_rpc_url is empty, _photon should not be wired."""
        # The default config has helius_rpc_url = ""
        import os

        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib  # type: ignore

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "titan_plugin",
            "config.toml",
        )
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        helius_rpc = config.get("network", {}).get("helius_rpc_url", "")
        # Default should be empty string
        assert helius_rpc == ""

    def test_light_protocol_constants(self):
        """Light Protocol program IDs are defined."""
        from titan_plugin.utils.solana_client import (
            LIGHT_SYSTEM_PROGRAM_ID,
            LIGHT_REGISTRY_PROGRAM_ID,
            LIGHT_COMPRESSION_PROGRAM_ID,
        )

        assert LIGHT_SYSTEM_PROGRAM_ID == "SySTEM1eSU2p4BGQfQpimFEWWSC1XDFeun3Nqzz3rT7"
        assert LIGHT_REGISTRY_PROGRAM_ID == "Lighton6oQpVkeewmo2mcPTQQp7kYHr4fWpAgJyEmDX"
        assert LIGHT_COMPRESSION_PROGRAM_ID == "compr6CUsB5m2jS4Y3831ztGSTnDpnKJTKS95d64XVq"

    def test_new_discriminators_defined(self):
        """New instruction discriminators match IDL output."""
        from titan_plugin.utils.solana_client import (
            _VAULT_IX_COMPRESS_MEMORY_BATCH,
            _VAULT_IX_APPEND_EPOCH_SNAPSHOT,
        )

        assert _VAULT_IX_COMPRESS_MEMORY_BATCH == bytes([105, 76, 210, 140, 189, 129, 57, 135])
        assert _VAULT_IX_APPEND_EPOCH_SNAPSHOT == bytes([213, 217, 65, 120, 202, 70, 5, 131])
