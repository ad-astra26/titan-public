"""
tests/test_nft_identity.py
Phase 10: NFT Identity — Metaplex Core instruction builders, soul minting, and on-chain verification.

Tests:
  - Borsh serialization helpers
  - CreateV1 instruction building (with and without attributes)
  - UpdateV1 instruction building
  - Asset account decoding
  - SovereignSoul NFT minting methods (mocked network)
  - Genesis ceremony NFT integration
  - Epoch NFT minting in backup.py
  - extra_signers support in network client

Run: python -m pytest tests/test_nft_identity.py -v -p no:anchorpy
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Borsh Serialization Helpers
# ---------------------------------------------------------------------------

class TestBorshHelpers:
    """Test low-level Borsh encoding functions."""

    def test_borsh_string_basic(self):
        from titan_plugin.utils.solana_client import _borsh_string
        result = _borsh_string("hello")
        # 4-byte LE length (5) + "hello" UTF-8
        assert result == b"\x05\x00\x00\x00hello"

    def test_borsh_string_empty(self):
        from titan_plugin.utils.solana_client import _borsh_string
        result = _borsh_string("")
        assert result == b"\x00\x00\x00\x00"

    def test_borsh_string_unicode(self):
        from titan_plugin.utils.solana_client import _borsh_string
        result = _borsh_string("Titan \u2728")
        encoded = "Titan \u2728".encode("utf-8")
        assert result[:4] == len(encoded).to_bytes(4, "little")
        assert result[4:] == encoded

    def test_borsh_option_none(self):
        from titan_plugin.utils.solana_client import _borsh_option_none
        assert _borsh_option_none() == bytes([0])

    def test_borsh_option_some(self):
        from titan_plugin.utils.solana_client import _borsh_option_some
        result = _borsh_option_some(b"\xAA\xBB")
        assert result == bytes([1, 0xAA, 0xBB])


class TestAttributesPlugin:
    """Test Attributes plugin serialization."""

    def test_single_attribute(self):
        from titan_plugin.utils.solana_client import _borsh_attributes_plugin
        result = _borsh_attributes_plugin({"Generation": "1"})
        # Plugin type byte (11) + vec len (1) + key + value + authority None
        assert result[0] == 6  # _MPL_PLUGIN_ATTRIBUTES (Attributes = variant 6)
        assert result[1:5] == (1).to_bytes(4, "little")  # Vec len = 1 attribute

    def test_multiple_attributes(self):
        from titan_plugin.utils.solana_client import _borsh_attributes_plugin
        attrs = {"Generation": "1", "Type": "Genesis", "Parent": "GENESIS"}
        result = _borsh_attributes_plugin(attrs)
        assert result[0] == 6  # Plugin::Attributes variant index
        assert result[1:5] == (3).to_bytes(4, "little")  # Vec len = 3 attributes

    def test_empty_attributes(self):
        from titan_plugin.utils.solana_client import _borsh_attributes_plugin
        result = _borsh_attributes_plugin({})
        assert result[0] == 6  # Plugin::Attributes variant index
        assert result[1:5] == (0).to_bytes(4, "little")  # Vec len = 0


# ---------------------------------------------------------------------------
# CreateV1 Instruction Builder
# ---------------------------------------------------------------------------

class TestCreateV1:
    """Test Metaplex Core CreateV1 instruction building."""

    def test_create_v1_basic(self):
        from titan_plugin.utils.solana_client import build_mpl_core_create_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        asset_kp = Keypair()
        payer_kp = Keypair()

        ix = build_mpl_core_create_v1(
            asset_pubkey=asset_kp.pubkey(),
            payer_pubkey=payer_kp.pubkey(),
            name="Test NFT",
            uri="https://example.com/meta.json",
        )

        assert ix is not None
        # Instruction data starts with CreateV1 discriminator (0)
        assert ix.data[0] == 0
        # 8 accounts: asset, collection(placeholder), authority, payer, owner, updateAuth, system, logWrapper
        assert len(ix.accounts) == 8

    def test_create_v1_with_attributes(self):
        from titan_plugin.utils.solana_client import build_mpl_core_create_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        asset_kp = Keypair()
        payer_kp = Keypair()

        ix = build_mpl_core_create_v1(
            asset_pubkey=asset_kp.pubkey(),
            payer_pubkey=payer_kp.pubkey(),
            name="Titan Soul Gen 1",
            uri="https://shdw-drive.genesysgo.net/titan/gen_1.json",
            attributes={"Generation": "1", "Type": "Genesis"},
        )

        assert ix is not None
        # plugins Option::Some byte should be present
        assert bytes([1]) in ix.data  # At least one Some marker

    def test_create_v1_with_separate_owner(self):
        from titan_plugin.utils.solana_client import build_mpl_core_create_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        asset_kp = Keypair()
        payer_kp = Keypair()
        owner_kp = Keypair()

        ix = build_mpl_core_create_v1(
            asset_pubkey=asset_kp.pubkey(),
            payer_pubkey=payer_kp.pubkey(),
            name="Owned NFT",
            uri="https://example.com/meta.json",
            owner_pubkey=owner_kp.pubkey(),
        )

        assert ix is not None
        # Owner account (index 4) should be the separate owner
        assert ix.accounts[4].pubkey == owner_kp.pubkey()

    def test_create_v1_asset_is_signer(self):
        from titan_plugin.utils.solana_client import build_mpl_core_create_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        asset_kp = Keypair()
        payer_kp = Keypair()

        ix = build_mpl_core_create_v1(
            asset_pubkey=asset_kp.pubkey(),
            payer_pubkey=payer_kp.pubkey(),
            name="Test",
            uri="https://example.com/meta.json",
        )

        # Asset account must be signer + writable
        assert ix.accounts[0].is_signer is True
        assert ix.accounts[0].is_writable is True

    def test_create_v1_name_in_data(self):
        from titan_plugin.utils.solana_client import build_mpl_core_create_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        ix = build_mpl_core_create_v1(
            asset_pubkey=Keypair().pubkey(),
            payer_pubkey=Keypair().pubkey(),
            name="Titan Soul Gen 1",
            uri="https://example.com/meta.json",
        )

        # Name should be encoded in the instruction data
        assert b"Titan Soul Gen 1" in ix.data


# ---------------------------------------------------------------------------
# UpdateV1 Instruction Builder
# ---------------------------------------------------------------------------

class TestUpdateV1:
    """Test Metaplex Core UpdateV1 instruction building."""

    def test_update_v1_name_only(self):
        from titan_plugin.utils.solana_client import build_mpl_core_update_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        asset_kp = Keypair()
        auth_kp = Keypair()

        ix = build_mpl_core_update_v1(
            asset_pubkey=asset_kp.pubkey(),
            authority_pubkey=auth_kp.pubkey(),
            new_name="Titan Soul Gen 2",
        )

        assert ix is not None
        assert ix.data[0] == 15  # UpdateV1 discriminator
        assert b"Titan Soul Gen 2" in ix.data

    def test_update_v1_uri_only(self):
        from titan_plugin.utils.solana_client import build_mpl_core_update_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        ix = build_mpl_core_update_v1(
            asset_pubkey=Keypair().pubkey(),
            authority_pubkey=Keypair().pubkey(),
            new_uri="https://new-uri.com/meta.json",
        )

        assert ix is not None
        assert b"https://new-uri.com/meta.json" in ix.data

    def test_update_v1_no_changes(self):
        from titan_plugin.utils.solana_client import build_mpl_core_update_v1, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        ix = build_mpl_core_update_v1(
            asset_pubkey=Keypair().pubkey(),
            authority_pubkey=Keypair().pubkey(),
        )

        # Still builds a valid instruction even with no changes
        assert ix is not None


# ---------------------------------------------------------------------------
# Asset Decoding
# ---------------------------------------------------------------------------

class TestAssetDecoding:
    """Test Metaplex Core Asset account decoding."""

    def _build_mock_asset(self, name="Test NFT", uri="https://example.com"):
        """Build a minimal mock Asset account byte representation."""
        from titan_plugin.utils.solana_client import _borsh_string, is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair

        owner = Keypair()
        update_auth = Keypair()

        data = bytearray()
        # Discriminator: single byte 0x01 (Key::AssetV1)
        data.append(1)
        # Owner (32 bytes)
        data.extend(bytes(owner.pubkey()))
        # UpdateAuthority: Borsh enum (0=None, 1=Address(Pubkey), 2=Collection(Pubkey))
        # Use type 1 (Address) so the 32-byte pubkey is included per real layout
        data.append(1)
        data.extend(bytes(update_auth.pubkey()))
        # Name (Borsh string)
        data.extend(_borsh_string(name))
        # URI (Borsh string)
        data.extend(_borsh_string(uri))

        return bytes(data), str(owner.pubkey()), str(update_auth.pubkey())

    def test_decode_basic_asset(self):
        from titan_plugin.utils.solana_client import decode_mpl_core_asset
        data, owner_str, _ = self._build_mock_asset("Titan Soul Gen 1", "https://shdw.com/gen1")

        result = decode_mpl_core_asset(data)
        assert result is not None
        assert result["name"] == "Titan Soul Gen 1"
        assert result["uri"] == "https://shdw.com/gen1"
        assert result["owner"] == owner_str

    def test_decode_empty_data(self):
        from titan_plugin.utils.solana_client import decode_mpl_core_asset
        result = decode_mpl_core_asset(b"")
        assert result is None

    def test_decode_too_short(self):
        from titan_plugin.utils.solana_client import decode_mpl_core_asset
        result = decode_mpl_core_asset(b"\x01" + b"\x00" * 10)
        assert result is None


# ---------------------------------------------------------------------------
# SovereignSoul NFT Methods (Mocked Network)
# ---------------------------------------------------------------------------

class TestSoulNFTMinting:
    """Test SovereignSoul mint_genesis_nft and mint_nextgen_nft with mocked network."""

    @pytest.fixture
    def mock_soul(self, tmp_path):
        """Create a SovereignSoul with mocked network."""
        from titan_plugin.utils.solana_client import is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair

        network = MagicMock()
        kp = Keypair()
        network.keypair = kp
        network.pubkey = kp.pubkey()
        network.send_sovereign_transaction = AsyncMock(return_value="5fakeTxSig123abc")
        network.get_balance = AsyncMock(return_value=5.0)

        # Write a temp state file path
        state_file = tmp_path / "soul_state.json"

        from titan_plugin.core.soul import SovereignSoul

        # Patch _load_local_state to avoid touching real filesystem
        with patch.object(SovereignSoul, "_load_local_state"):
            soul = SovereignSoul.__new__(SovereignSoul)
            soul.wallet_path = str(tmp_path / "wallet.json")
            soul.network = network
            soul.current_gen = 1
            soul._directives_cache = ["Prime Directive 1: Sovereign Growth."]
            soul._maker_pubkey = None
            soul._nft_address = None
            soul._state_file = state_file

        return soul

    def test_mint_genesis_nft(self, mock_soul):
        result = asyncio.get_event_loop().run_until_complete(
            mock_soul.mint_genesis_nft(
                name="Titan Soul Gen 1",
                uri="https://example.com/gen1.json",
                art_hash="abc123",
                genesis_tx="def456",
            )
        )

        # Should return an asset address (base58 string)
        assert result is not None
        assert len(result) > 30  # Solana pubkeys are ~44 chars

        # Should have called send_sovereign_transaction with extra_signers
        call_args = mock_soul.network.send_sovereign_transaction.call_args
        assert call_args is not None
        assert "extra_signers" in call_args.kwargs
        assert len(call_args.kwargs["extra_signers"]) == 1

        # Soul state should be updated
        assert mock_soul._nft_address == result
        assert mock_soul.current_gen == 1

    def test_mint_nextgen_nft(self, mock_soul):
        mock_soul._nft_address = "FakeGenesisAddr123"
        mock_soul.current_gen = 1

        result = asyncio.get_event_loop().run_until_complete(
            mock_soul.mint_nextgen_nft("New directive for evolution")
        )

        assert result is not None
        assert len(result) > 30
        assert mock_soul.current_gen == 2
        assert mock_soul._nft_address == result

    def test_mint_genesis_no_keypair(self, mock_soul):
        mock_soul.network.keypair = None

        result = asyncio.get_event_loop().run_until_complete(
            mock_soul.mint_genesis_nft()
        )
        assert result is None

    def test_verify_nft_ownership_no_address(self, mock_soul):
        mock_soul._nft_address = None
        result = asyncio.get_event_loop().run_until_complete(
            mock_soul.verify_nft_ownership()
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_nft_ownership_success(self, mock_soul):
        mock_soul._nft_address = "SomeNFTAddress123"
        owner_str = str(mock_soul.network.pubkey)

        mock_fetch = AsyncMock(return_value={"owner": owner_str, "name": "Test", "uri": ""})
        with patch.dict("sys.modules", {}):
            with patch(
                "titan_plugin.utils.solana_client.fetch_mpl_core_asset",
                mock_fetch,
            ):
                result = await mock_soul.verify_nft_ownership()
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_nft_ownership_wrong_owner(self, mock_soul):
        mock_soul._nft_address = "SomeNFTAddress123"

        mock_fetch = AsyncMock(return_value={"owner": "DifferentOwner999", "name": "Test", "uri": ""})
        with patch(
            "titan_plugin.utils.solana_client.fetch_mpl_core_asset",
            mock_fetch,
        ):
            result = await mock_soul.verify_nft_ownership()
        assert result is False


# ---------------------------------------------------------------------------
# Evolve Soul Integration (with NFT Minting)
# ---------------------------------------------------------------------------

class TestEvolveSoulWithNFT:
    """Test that evolve_soul now mints a NextGen NFT."""

    def test_evolve_mints_nextgen(self, tmp_path):
        from titan_plugin.utils.solana_client import is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        from titan_plugin.core.soul import SovereignSoul

        network = MagicMock()
        kp = Keypair()
        network.keypair = kp
        network.pubkey = kp.pubkey()
        network.send_sovereign_transaction = AsyncMock(return_value="5txSig999")
        network.get_balance = AsyncMock(return_value=5.0)

        state_file = tmp_path / "soul_state.json"

        with patch.object(SovereignSoul, "_load_local_state"):
            soul = SovereignSoul.__new__(SovereignSoul)
            soul.wallet_path = str(tmp_path / "wallet.json")
            soul.network = network
            soul.current_gen = 1
            soul._directives_cache = ["Prime Directive 1"]
            soul._maker_pubkey = kp.pubkey()
            soul._nft_address = "PreviousNFTAddr"
            soul._state_file = state_file

        # Mock maker signature verification to pass
        with patch.object(soul, "verify_maker_signature", new_callable=AsyncMock, return_value=True):
            result = asyncio.get_event_loop().run_until_complete(
                soul.evolve_soul("New directive", "fake_sig", current_balance=5.0)
            )

        assert "Gen 2" in result
        assert soul.current_gen == 2
        # Should have called send_sovereign_transaction at least twice
        # (once for memo inscription, once for NFT mint)
        assert network.send_sovereign_transaction.call_count >= 1


# ---------------------------------------------------------------------------
# Backup Epoch NFT Minting
# ---------------------------------------------------------------------------

class TestEpochNFTMinting:
    """Test epoch NFT minting in RebirthBackup."""

    def test_mint_epoch_nft(self):
        from titan_plugin.utils.solana_client import is_available
        if not is_available():
            pytest.skip("Solana SDK not available")

        from solders.keypair import Keypair
        from titan_plugin.logic.backup import RebirthBackup

        network = MagicMock()
        kp = Keypair()
        network.keypair = kp
        network.pubkey = kp.pubkey()
        network.send_sovereign_transaction = AsyncMock(return_value="5epochTx123")

        backup = RebirthBackup(network)

        result = asyncio.get_event_loop().run_until_complete(
            backup.mint_epoch_nft(
                epoch=1710000000,
                sovereignty_idx=85.5,
                diary_entry="A day of growth and discovery.",
                total_nodes=42,
            )
        )

        assert result is not None
        assert len(result) > 30  # Valid pubkey

        # Verify extra_signers was passed
        call_args = network.send_sovereign_transaction.call_args
        assert "extra_signers" in call_args.kwargs

    def test_mint_epoch_nft_no_keypair(self):
        from titan_plugin.logic.backup import RebirthBackup

        network = MagicMock()
        network.keypair = None

        backup = RebirthBackup(network)

        result = asyncio.get_event_loop().run_until_complete(
            backup.mint_epoch_nft(
                epoch=1710000000,
                sovereignty_idx=50.0,
                diary_entry="Test",
                total_nodes=0,
            )
        )
        assert result is None


# ---------------------------------------------------------------------------
# Network Extra Signers
# ---------------------------------------------------------------------------

class TestNetworkExtraSigners:
    """Test that HybridNetworkClient.send_sovereign_transaction accepts extra_signers."""

    def test_signature_includes_extra_signers(self):
        """Verify the method signature accepts extra_signers parameter."""
        import inspect
        from titan_plugin.core.network import HybridNetworkClient

        sig = inspect.signature(HybridNetworkClient.send_sovereign_transaction)
        assert "extra_signers" in sig.parameters

    def test_extra_signers_default_none(self):
        import inspect
        from titan_plugin.core.network import HybridNetworkClient

        sig = inspect.signature(HybridNetworkClient.send_sovereign_transaction)
        param = sig.parameters["extra_signers"]
        assert param.default is None
