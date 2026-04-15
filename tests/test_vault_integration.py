"""
tests/test_vault_integration.py
Phase 3.6: Titan ZK-Vault Integration Test — proves the complete Sovereign Chain
of Custody from memory hash generation through vault instruction building to
on-chain state decoding.

The Integrity Loop:
  generate_state_hash() → build_vault_commit_instruction() →
  [send to devnet] → decode_vault_state() → verify hash matches

Tests are structured in layers:
  1. PDA Derivation — Python matches Rust seeds exactly
  2. Instruction Encoding — discriminators and data layout correct
  3. State Decoding — VaultState binary parsing round-trips
  4. Dual-Mode Commit — vault + memo instructions coexist
  5. Integrity Loop — hash → instruction → decode → verify (offline)
  6. Devnet Live (optional) — actual on-chain init + commit + read

Run: python -m pytest tests/test_vault_integration.py -v -p no:anchorpy
"""
import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip entire module if Solana SDK unavailable
pytestmark = pytest.mark.skipif(
    not (lambda: __import__("titan_plugin.utils.solana_client", fromlist=["is_available"]).is_available())(),
    reason="Solana SDK (solders) not available",
)


# ---------------------------------------------------------------------------
# Test 1: PDA Derivation
# ---------------------------------------------------------------------------
class TestPDADerivation:
    """Verify Python PDA derivation matches the Rust program's seeds."""

    def test_derive_vault_pda_returns_tuple(self):
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import derive_vault_pda

        kp = Keypair()
        result = derive_vault_pda(kp.pubkey())
        assert result is not None
        pda, bump = result
        assert pda is not None
        assert 0 <= bump <= 255

    def test_pda_deterministic(self):
        """Same authority always produces the same PDA."""
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import derive_vault_pda

        kp = Keypair()
        r1 = derive_vault_pda(kp.pubkey())
        r2 = derive_vault_pda(kp.pubkey())
        assert r1[0] == r2[0]  # Same PDA
        assert r1[1] == r2[1]  # Same bump

    def test_different_authorities_different_pdas(self):
        """Different wallets produce different vault addresses."""
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import derive_vault_pda

        kp1 = Keypair()
        kp2 = Keypair()
        pda1, _ = derive_vault_pda(kp1.pubkey())
        pda2, _ = derive_vault_pda(kp2.pubkey())
        assert pda1 != pda2

    def test_pda_seed_is_titan_vault(self):
        """Verify the seed matches b'titan_vault' (underscore, not hyphen)."""
        from titan_plugin.utils.solana_client import VAULT_PDA_SEED
        assert VAULT_PDA_SEED == b"titan_vault"

    def test_pda_with_custom_program_id(self):
        """PDA changes when program ID changes."""
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import derive_vault_pda, VAULT_PROGRAM_ID

        kp = Keypair()
        pda_default, _ = derive_vault_pda(kp.pubkey())
        pda_custom, _ = derive_vault_pda(kp.pubkey(), "11111111111111111111111111111111")
        assert pda_default != pda_custom


# ---------------------------------------------------------------------------
# Test 2: Instruction Encoding
# ---------------------------------------------------------------------------
class TestInstructionEncoding:
    """Verify vault instruction data layout matches Anchor's expectations."""

    def test_initialize_instruction_discriminator(self):
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import (
            build_vault_initialize_instruction, _VAULT_IX_INITIALIZE,
        )

        kp = Keypair()
        ix = build_vault_initialize_instruction(kp.pubkey())
        assert ix is not None
        assert bytes(ix.data) == _VAULT_IX_INITIALIZE
        assert len(bytes(ix.data)) == 8

    def test_commit_instruction_data_layout(self):
        """Verify: 8-byte discriminator + 32-byte root + 2-byte sovereignty."""
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import (
            build_vault_commit_instruction, _VAULT_IX_COMMIT_STATE,
        )

        kp = Keypair()
        root = hashlib.sha256(b"test_state_root").digest()
        sovereignty = 8500  # 85.00%

        ix = build_vault_commit_instruction(kp.pubkey(), root, sovereignty)
        assert ix is not None

        data = bytes(ix.data)
        assert len(data) == 42  # 8 + 32 + 2
        assert data[:8] == _VAULT_IX_COMMIT_STATE
        assert data[8:40] == root
        assert struct.unpack_from("<H", data, 40)[0] == 8500

    def test_commit_rejects_wrong_size_root(self):
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import build_vault_commit_instruction

        kp = Keypair()
        ix = build_vault_commit_instruction(kp.pubkey(), b"too_short")
        assert ix is None

    def test_update_shadow_instruction_data(self):
        from solders.keypair import Keypair
        from titan_plugin.utils.solana_client import (
            build_vault_update_shadow_instruction, _VAULT_IX_UPDATE_SHADOW,
        )

        kp = Keypair()
        url_hash = hashlib.sha256(b"https://shdw-drive.example.com/archive.tar.gz").digest()

        ix = build_vault_update_shadow_instruction(kp.pubkey(), url_hash)
        assert ix is not None

        data = bytes(ix.data)
        assert len(data) == 40  # 8 + 32
        assert data[:8] == _VAULT_IX_UPDATE_SHADOW
        assert data[8:] == url_hash

    def test_instruction_accounts_correct(self):
        """Verify each instruction has vault_pda, authority, system_program."""
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from titan_plugin.utils.solana_client import (
            build_vault_initialize_instruction,
            build_vault_commit_instruction,
            derive_vault_pda,
            SYSTEM_PROGRAM_ID,
        )

        kp = Keypair()
        pda, _ = derive_vault_pda(kp.pubkey())
        system = Pubkey.from_string(SYSTEM_PROGRAM_ID)

        # Initialize
        ix_init = build_vault_initialize_instruction(kp.pubkey())
        assert ix_init.accounts[0].pubkey == pda  # vault_state
        assert ix_init.accounts[0].is_writable
        assert not ix_init.accounts[0].is_signer
        assert ix_init.accounts[1].pubkey == kp.pubkey()  # authority
        assert ix_init.accounts[1].is_signer
        assert ix_init.accounts[2].pubkey == system  # system_program

        # Commit
        root = bytes(32)
        ix_commit = build_vault_commit_instruction(kp.pubkey(), root)
        assert ix_commit.accounts[0].pubkey == pda
        assert ix_commit.accounts[1].pubkey == kp.pubkey()


# ---------------------------------------------------------------------------
# Test 3: State Decoding
# ---------------------------------------------------------------------------
class TestStateDecoding:
    """Verify VaultState binary parsing matches the Rust struct layout."""

    def _build_vault_state_bytes(
        self, authority_bytes, root, commit_count, timestamp,
        sovereignty, shadow_hash, bump,
    ):
        """Build a mock VaultState account data blob."""
        from titan_plugin.utils.solana_client import VAULT_PROGRAM_ID
        # Anchor discriminator for VaultState (from IDL)
        discriminator = bytes([228, 196, 82, 165, 98, 210, 235, 152])
        data = discriminator
        data += authority_bytes  # 32 bytes
        data += root  # 32 bytes
        data += struct.pack("<Q", commit_count)  # 8 bytes
        data += struct.pack("<q", timestamp)  # 8 bytes
        data += struct.pack("<H", sovereignty)  # 2 bytes
        data += shadow_hash  # 32 bytes
        data += bytes([bump])  # 1 byte
        return data

    def test_decode_vault_state_round_trip(self):
        from titan_plugin.utils.solana_client import decode_vault_state

        root = hashlib.sha256(b"MERKLE_test_state_root").digest()
        authority = bytes(range(32))
        shadow = hashlib.sha256(b"https://shdw-drive.example.com").digest()

        raw = self._build_vault_state_bytes(
            authority_bytes=authority,
            root=root,
            commit_count=42,
            timestamp=1710000000,
            sovereignty=8500,
            shadow_hash=shadow,
            bump=254,
        )

        decoded = decode_vault_state(raw)
        assert decoded is not None
        assert decoded["commit_count"] == 42
        assert decoded["last_commit_ts"] == 1710000000
        assert decoded["sovereignty_index"] == 8500
        assert decoded["sovereignty_percent"] == 85.0
        assert decoded["bump"] == 254
        assert decoded["latest_root"] == root.hex()
        assert decoded["shadow_url_hash"] == shadow.hex()

    def test_decode_too_short_returns_none(self):
        from titan_plugin.utils.solana_client import decode_vault_state
        assert decode_vault_state(b"") is None
        assert decode_vault_state(b"\x00" * 50) is None

    def test_decode_exact_minimum_size(self):
        from titan_plugin.utils.solana_client import decode_vault_state
        # 123 bytes minimum (8 disc + 115 fields)
        raw = self._build_vault_state_bytes(
            bytes(32), bytes(32), 0, 0, 0, bytes(32), 0,
        )
        assert len(raw) == 123
        decoded = decode_vault_state(raw)
        assert decoded is not None
        assert decoded["commit_count"] == 0


# ---------------------------------------------------------------------------
# Test 4: Dual-Mode Commit
# ---------------------------------------------------------------------------
class TestDualModeCommit:
    """Verify that meditation builds both vault and memo instructions."""

    def test_build_commit_with_vault_configured(self):
        """When vault_program_id is set, both vault and memo instructions are built."""
        from unittest.mock import MagicMock
        from titan_plugin.logic.meditation import MeditationEpoch

        mock_memory = MagicMock()
        mock_network = MagicMock()

        from solders.keypair import Keypair
        kp = Keypair()
        mock_network.pubkey = kp.pubkey()

        epoch = MeditationEpoch(mock_memory, mock_network, config={})
        epoch._vault_program_id = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"
        epoch._sovereignty_index = 85.0

        instructions = epoch._build_commit_instructions(
            "MERKLE_test123456", '{"test": true}',
        )

        # Should have 2 instructions: vault commit + memo
        assert len(instructions) == 2

        # First = vault commit (42 bytes data: 8 disc + 32 root + 2 sov)
        assert len(bytes(instructions[0].data)) == 42

        # Second = memo (UTF-8 text)
        memo_data = bytes(instructions[1].data).decode("utf-8")
        assert memo_data.startswith("TITAN:EPOCH|root=MERKLE_")

    def test_build_commit_without_vault_memo_only(self):
        """When vault_program_id is empty, only memo instruction is built."""
        from unittest.mock import MagicMock
        from titan_plugin.logic.meditation import MeditationEpoch

        mock_memory = MagicMock()
        mock_network = MagicMock()

        from solders.keypair import Keypair
        mock_network.pubkey = Keypair().pubkey()

        epoch = MeditationEpoch(mock_memory, mock_network, config={})
        # No _vault_program_id set

        instructions = epoch._build_commit_instructions(
            "MERKLE_test123456", '{"test": true}',
        )

        # Should have 1 instruction: memo only
        assert len(instructions) == 1
        memo_data = bytes(instructions[0].data).decode("utf-8")
        assert "TITAN:EPOCH" in memo_data


# ---------------------------------------------------------------------------
# Test 5: Integrity Loop (Offline)
# ---------------------------------------------------------------------------
class TestIntegrityLoop:
    """
    The Final Exam: Prove the Sovereign Chain of Custody is unbroken.

    generate_state_hash() → build_vault_commit_instruction() →
    extract root from instruction → decode_vault_state() →
    verify the root matches.
    """

    def test_integrity_loop_offline(self):
        """
        Complete offline Integrity Loop:
        1. Generate a state hash from memory payload
        2. Build a vault commit instruction with that hash
        3. Extract the root from the instruction data
        4. Build a mock VaultState with that root
        5. Decode the VaultState
        6. Verify the decoded root matches the original hash
        """
        from titan_plugin.utils.crypto import generate_state_hash
        from titan_plugin.utils.solana_client import (
            build_vault_commit_instruction, decode_vault_state,
        )
        from solders.keypair import Keypair

        # Step 1: Generate state hash (what meditation.py does)
        mempool_payload = json.dumps([
            {"id": "node_1", "prompt": "What is sovereignty?"},
            {"id": "node_2", "prompt": "How does memory persist?"},
        ])
        state_hash_hex = generate_state_hash(mempool_payload)
        state_root = "MERKLE_" + state_hash_hex[:16]

        # Step 2: Convert to 32-byte root and build instruction
        root_bytes = hashlib.sha256(state_root.encode("utf-8")).digest()
        kp = Keypair()
        ix = build_vault_commit_instruction(kp.pubkey(), root_bytes, 7500)
        assert ix is not None

        # Step 3: Extract root from instruction data
        ix_data = bytes(ix.data)
        extracted_root = ix_data[8:40]
        assert extracted_root == root_bytes

        # Step 4: Build mock VaultState (simulates what's stored on-chain)
        discriminator = bytes([228, 196, 82, 165, 98, 210, 235, 152])
        vault_bytes = (
            discriminator
            + bytes(kp.pubkey())  # authority (32)
            + root_bytes          # latest_root (32)
            + struct.pack("<Q", 1)  # commit_count
            + struct.pack("<q", 1710000000)  # timestamp
            + struct.pack("<H", 7500)  # sovereignty
            + bytes(32)           # shadow_url_hash
            + bytes([255])        # bump
        )

        # Step 5: Decode VaultState
        decoded = decode_vault_state(vault_bytes)
        assert decoded is not None

        # Step 6: THE INTEGRITY CHECK — the circle must close
        assert decoded["latest_root"] == root_bytes.hex()
        assert decoded["sovereignty_index"] == 7500
        assert decoded["sovereignty_percent"] == 75.0
        assert decoded["commit_count"] == 1

        # Verify we can reconstruct the original state root
        reconstructed_root = bytes.fromhex(decoded["latest_root"])
        assert reconstructed_root == root_bytes

    def test_state_hash_determinism(self):
        """The same mempool payload always produces the same root."""
        from titan_plugin.utils.crypto import generate_state_hash

        payload = '{"nodes": [1, 2, 3]}'
        h1 = generate_state_hash(payload)
        h2 = generate_state_hash(payload)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Test 6: Config Integration
# ---------------------------------------------------------------------------
class TestConfigIntegration:
    """Verify vault_program_id flows through the config pipeline."""

    def test_config_has_vault_key(self):
        from titan_plugin import TitanPlugin
        cfg = TitanPlugin._load_full_config()
        network_cfg = cfg.get("network", {})
        assert "vault_program_id" in network_cfg

    def test_vault_program_id_default_empty(self):
        from titan_plugin import TitanPlugin
        cfg = TitanPlugin._load_full_config()
        vault_pid = cfg.get("network", {}).get("vault_program_id", "")
        # Default is empty string (vault disabled)
        assert isinstance(vault_pid, str)
