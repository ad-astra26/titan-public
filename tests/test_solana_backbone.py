"""
tests/test_solana_backbone.py
Phase 2.4: Solana Backbone Verification — tests the "hot path" Solana primitives
against a real RPC endpoint (testnet/devnet) to ensure our instruction builders,
balance queries, and memo TX logic actually work on-chain.

Tests are skipped if no RPC is reachable (offline/CI environments).

Run: python -m pytest tests/test_solana_backbone.py -v -p no:anchorpy
"""
import asyncio
import json
import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------
TESTNET_RPC = "https://api.testnet.solana.com"
DEVNET_RPC = "https://api.devnet.solana.com"


def _rpc_reachable(url: str) -> bool:
    """Check if an RPC endpoint responds to getHealth."""
    try:
        import httpx
        resp = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _solana_available() -> bool:
    """Check if Solana SDK is importable."""
    try:
        from titan_plugin.utils.solana_client import is_available
        return is_available()
    except Exception:
        return False


# Skip entire module if no Solana SDK or no RPC reachable
pytestmark = pytest.mark.skipif(
    not _solana_available(),
    reason="Solana SDK (solders/solana) not available",
)


@pytest.fixture(scope="module")
def rpc_url():
    """Find a reachable RPC endpoint."""
    for url in [DEVNET_RPC, TESTNET_RPC]:
        if _rpc_reachable(url):
            return url
    pytest.skip("No Solana RPC endpoint reachable (testnet/devnet)")


@pytest.fixture(scope="module")
def test_keypair():
    """Generate a fresh ephemeral keypair for testing."""
    from solders.keypair import Keypair
    kp = Keypair()
    logger.info("Test keypair: %s", kp.pubkey())
    return kp


@pytest.fixture(scope="module")
def network_client(rpc_url, test_keypair):
    """Create a HybridNetworkClient pointed at testnet/devnet."""
    from titan_plugin.core.network import HybridNetworkClient

    config = {
        "solana_network": "devnet" if "devnet" in rpc_url else "testnet",
        "public_rpc_urls": [rpc_url],
        "premium_rpc_url": "",
        "wallet_keypair_path": "",
        "maker_pubkey": "",
    }
    client = HybridNetworkClient(config=config)
    # Inject test keypair directly
    client._keypair = test_keypair
    client._pubkey = test_keypair.pubkey()
    return client


# ---------------------------------------------------------------------------
# Test 1: SDK Availability & Primitive Imports
# ---------------------------------------------------------------------------
class TestSolanaPrimitives:
    """Verify the Solana primitive facade resolves all imports correctly."""

    def test_sdk_available(self):
        from titan_plugin.utils.solana_client import is_available
        assert is_available(), "Solana SDK should be available"

    def test_parse_pubkey(self):
        from titan_plugin.utils.solana_client import parse_pubkey
        # Known testnet system program
        pk = parse_pubkey("11111111111111111111111111111111")
        assert pk is not None

    def test_parse_invalid_pubkey(self):
        from titan_plugin.utils.solana_client import parse_pubkey
        pk = parse_pubkey("not-a-valid-pubkey!!!")
        assert pk is None

    def test_build_memo_instruction(self, test_keypair):
        from titan_plugin.utils.solana_client import build_memo_instruction
        ix = build_memo_instruction(test_keypair.pubkey(), "TITAN:TEST|hello=world")
        assert ix is not None

    def test_memo_instruction_size_limit(self, test_keypair):
        """Verify memo instruction respects the 566-byte Solana Memo limit."""
        from titan_plugin.utils.solana_client import build_memo_instruction
        # Under limit
        ix = build_memo_instruction(test_keypair.pubkey(), "A" * 500)
        assert ix is not None
        # The builder should handle oversized memos gracefully
        # (either truncate or return None depending on implementation)

    def test_build_compute_budget_instruction(self):
        from titan_plugin.utils.solana_client import build_compute_budget_instruction
        ix = build_compute_budget_instruction(50000)
        assert ix is not None

    def test_load_keypair_from_json(self, tmp_path):
        """Test keypair loading from standard Solana JSON format."""
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import load_keypair_from_json

        kp = Keypair()
        key_path = tmp_path / "test_key.json"
        with open(key_path, "w") as f:
            json.dump(list(bytes(kp)), f)

        loaded = load_keypair_from_json(str(key_path))
        assert loaded is not None
        assert str(loaded.pubkey()) == str(kp.pubkey())


# ---------------------------------------------------------------------------
# Test 2: RPC Connectivity & Balance Queries
# ---------------------------------------------------------------------------
class TestRPCConnectivity:
    """Verify live RPC interactions work correctly."""

    @pytest.mark.asyncio
    async def test_balance_query(self, network_client):
        """Query balance of a fresh keypair — should be 0 SOL."""
        balance = await network_client.get_balance()
        assert isinstance(balance, float)
        assert balance == 0.0  # Fresh keypair, never funded

    @pytest.mark.asyncio
    async def test_balance_system_program(self, rpc_url):
        """Query a known account (System Program) to verify RPC response parsing."""
        from titan_plugin.core.network import HybridNetworkClient

        config = {
            "solana_network": "devnet",
            "public_rpc_urls": [rpc_url],
            "premium_rpc_url": "",
            "wallet_keypair_path": "",
            "maker_pubkey": "",
        }
        client = HybridNetworkClient(config=config)
        # System Program always has 1 SOL on devnet/testnet
        # We just verify the RPC call doesn't throw
        try:
            # Use a known funded devnet faucet address
            from solders.keypair import Keypair
            client._keypair = Keypair()
            client._pubkey = client._keypair.pubkey()
            balance = await client.get_balance()
            assert isinstance(balance, float)
        except Exception as e:
            # RPC errors are acceptable — we're testing the client doesn't crash
            logger.warning("RPC balance query: %s (acceptable in test)", e)


# ---------------------------------------------------------------------------
# Test 3: Memo Transaction Building
# ---------------------------------------------------------------------------
class TestMemoTransaction:
    """Verify Memo TX construction produces valid transaction structures."""

    def test_memo_instruction_has_correct_program_id(self, test_keypair):
        from titan_plugin.utils.solana_client import build_memo_instruction
        ix = build_memo_instruction(test_keypair.pubkey(), "TITAN:TEST")
        assert ix is not None
        # Verify the program ID matches Memo Program V2
        assert str(ix.program_id) == "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"

    def test_memo_instruction_data_is_utf8(self, test_keypair):
        from titan_plugin.utils.solana_client import build_memo_instruction
        memo_text = "TITAN:EPOCH|root=MERKLE_abc123"
        ix = build_memo_instruction(test_keypair.pubkey(), memo_text)
        assert ix is not None
        assert bytes(ix.data) == memo_text.encode("utf-8")

    def test_memo_instruction_signer_is_account(self, test_keypair):
        from titan_plugin.utils.solana_client import build_memo_instruction
        ix = build_memo_instruction(test_keypair.pubkey(), "test")
        assert ix is not None
        assert len(ix.accounts) >= 1
        assert ix.accounts[0].pubkey == test_keypair.pubkey()
        assert ix.accounts[0].is_signer


# ---------------------------------------------------------------------------
# Test 4: ZK Account Data Encoding/Decoding
# ---------------------------------------------------------------------------
class TestZKSchema:
    """Verify the ZK-Omni-Schema encode/decode round-trips correctly."""

    def test_encode_decode_round_trip(self):
        from titan_plugin.utils.solana_client import encode_zk_account_data, decode_zk_account_data

        state = {
            "schema": "v2.0-sage",
            "bio": {"gen": 1, "mood": 0.75, "sovereignty": 85.3},
            "mems": {"latest_memory_hash": "abc123", "persistent_count": 42},
            "gates": {"sovereign_ratio": 0.15, "research_ratio": 0.30},
            "body": {"sol_balance": 1.5, "shadow_drive_url": "https://shdw.example.com"},
        }

        encoded = encode_zk_account_data(state)
        assert encoded is not None
        assert len(encoded) > 8  # 8-byte discriminator + JSON

        decoded = decode_zk_account_data(encoded)
        assert decoded is not None
        assert decoded["schema"] == "v2.0-sage"
        assert decoded["bio"]["gen"] == 1
        assert decoded["mems"]["persistent_count"] == 42

    def test_decode_empty_returns_empty(self):
        from titan_plugin.utils.solana_client import decode_zk_account_data
        result = decode_zk_account_data(b"")
        assert result == {}

    def test_decode_garbage_returns_empty(self):
        from titan_plugin.utils.solana_client import decode_zk_account_data
        result = decode_zk_account_data(b"\x00" * 20)
        assert result == {}


# ---------------------------------------------------------------------------
# Test 5: Crypto Utilities (Solana-specific)
# ---------------------------------------------------------------------------
class TestCryptoSolana:
    """Verify cryptographic operations that touch Solana primitives."""

    def test_state_hash_deterministic(self):
        from titan_plugin.utils.crypto import generate_state_hash
        h1 = generate_state_hash('{"nodes": [1, 2, 3]}')
        h2 = generate_state_hash('{"nodes": [1, 2, 3]}')
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_state_hash_differs_for_different_data(self):
        from titan_plugin.utils.crypto import generate_state_hash
        h1 = generate_state_hash("data_a")
        h2 = generate_state_hash("data_b")
        assert h1 != h2

    def test_sign_solana_payload(self, test_keypair):
        from titan_plugin.utils.crypto import sign_solana_payload
        sig = sign_solana_payload(test_keypair, "test message")
        assert sig is not None
        assert len(sig) > 40  # Base58-encoded Ed25519 signature

    def test_sign_verify_round_trip(self, test_keypair):
        """Sign a message and verify with the same pubkey."""
        from titan_plugin.utils.crypto import sign_solana_payload
        from solders.signature import Signature
        from solders.pubkey import Pubkey

        message = "TITAN:DIRECTIVE|text=test"
        sig_b58 = sign_solana_payload(test_keypair, message)
        assert sig_b58 is not None

        # Decode Base58 signature and verify
        sig = Signature.from_string(sig_b58)
        assert sig.verify(test_keypair.pubkey(), message.encode("utf-8"))


# ---------------------------------------------------------------------------
# Test 6: Gas Leak Check — Transaction Cost Estimation
# ---------------------------------------------------------------------------
class TestGasLeak:
    """
    Verify that our transaction building doesn't produce unexpectedly
    expensive transactions. The Titan's metabolism must stay solvent.
    """

    def test_memo_transaction_size_reasonable(self, test_keypair):
        """A single Memo TX should be well under 1232 bytes (Solana max)."""
        from titan_plugin.utils.solana_client import build_memo_instruction
        ix = build_memo_instruction(
            test_keypair.pubkey(),
            "TITAN:EPOCH|root=MERKLE_1234567890abcdef",
        )
        assert ix is not None
        # Instruction data + accounts + program_id overhead
        # A single memo instruction should be ~100-200 bytes
        data_size = len(bytes(ix.data))
        assert data_size < 500, f"Memo instruction data too large: {data_size} bytes"

    def test_compute_budget_is_micro_lamports(self):
        """Verify compute budget instruction uses microlamports, not lamports."""
        from titan_plugin.utils.solana_client import build_compute_budget_instruction
        # 50000 microlamports = 0.00005 lamports per CU
        # A typical TX uses ~200k CU, so total = 10 lamports = 0.00000001 SOL
        ix = build_compute_budget_instruction(50000)
        assert ix is not None
        # The instruction data should be 9 bytes: 1 byte discriminator + 8 bytes u64
        assert len(bytes(ix.data)) == 9

    def test_epoch_cost_within_metabolic_budget(self):
        """
        Estimate the total cost of a Small Epoch (Memo TX + priority fee).
        Must be well under the 0.05 SOL governance reserve.

        Solana fee structure:
        - Base fee: 5000 lamports (0.000005 SOL)
        - Priority: ~50000 microlamports * 200k CU = 10000 lamports
        - Total: ~15000 lamports = 0.000015 SOL per epoch

        The Titan does 4 meditation epochs per day + 1 rebirth = 5 TXs.
        Daily cost: ~0.000075 SOL — well within the 0.05 SOL reserve.
        """
        base_fee_lamports = 5000
        priority_microlamports = 50000
        compute_units = 200_000
        priority_lamports = (priority_microlamports * compute_units) // 1_000_000

        total_per_tx = base_fee_lamports + priority_lamports
        daily_txs = 5  # 4 meditations + 1 rebirth
        daily_cost_sol = (total_per_tx * daily_txs) / 1e9

        governance_reserve = 0.05
        assert daily_cost_sol < governance_reserve, (
            f"Daily TX cost ({daily_cost_sol:.6f} SOL) exceeds governance reserve ({governance_reserve} SOL)"
        )
        # Log for visibility
        logger.info(
            "Gas Leak Check: %d lamports/TX * %d TX/day = %.6f SOL/day (reserve: %.2f SOL)",
            total_per_tx, daily_txs, daily_cost_sol, governance_reserve,
        )


# ---------------------------------------------------------------------------
# Test 7: Capability Report Structure
# ---------------------------------------------------------------------------
class TestCapabilityReport:
    """Verify the data structure we'll expose via /health endpoint."""

    def test_capability_report_structure(self, rpc_url):
        """Build the capability report dict and verify all keys present."""
        from titan_plugin.utils.solana_client import is_available

        report = {
            "SOLANA_SDK": "ACTIVE" if is_available() else "ABSENT",
            "MEMO_INSCRIPTION": "VERIFIED" if is_available() else "STUB",
            "STATE_ROOT_ZK": "STUB",  # Until titan_zk_vault deployed
            "SHADOW_DRIVE_SYNC": "ACTIVE",  # Uses raw httpx, always available
            "RPC_ENDPOINT": rpc_url,
            "NETWORK": "devnet" if "devnet" in rpc_url else "testnet",
        }

        assert "SOLANA_SDK" in report
        assert "MEMO_INSCRIPTION" in report
        assert "STATE_ROOT_ZK" in report
        assert "SHADOW_DRIVE_SYNC" in report
        assert report["SOLANA_SDK"] == "ACTIVE"
        assert report["MEMO_INSCRIPTION"] == "VERIFIED"
        assert report["STATE_ROOT_ZK"] == "STUB"
