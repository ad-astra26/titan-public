"""
Tests for scripts/init_vault_pda.py — idempotent vault PDA initializer.

Mocks the Solana RPC client + the keypair load so we can exercise:
  * Idempotency (PDA already exists → exit 0, no TX)
  * Mainnet guard (refuses without explicit override flag)
  * Dry-run + missing --confirm (no TX submitted)
  * Bad keypair file (clean error, exit 1)
  * Successful init path (TX submitted + confirmed + verified)
  * Failed confirmation (timeout / on-chain error → exit 1)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root + scripts to sys.path so we can import the script as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))


@pytest.fixture
def fresh_keypair_file(tmp_path: Path) -> Path:
    """Generate a real Solders keypair JSON file so _load_keypair succeeds."""
    from solders.keypair import Keypair
    kp = Keypair()
    path = tmp_path / "id.json"
    path.write_text(json.dumps(list(bytes(kp))))
    return path


@pytest.fixture
def script_module():
    """Load the init_vault_pda module fresh per test (avoids state bleed)."""
    import importlib
    if "init_vault_pda" in sys.modules:
        del sys.modules["init_vault_pda"]
    return importlib.import_module("init_vault_pda")


def _patch_client(monkeypatch, *, balance: int = 5_000_000_000,
                  pda_exists_before: bool = False, pda_exists_after: bool = True,
                  send_returns_sig: str = "SIG_init_abc123"):
    """Patch solana.rpc.api.Client used inside init_vault_pda.main()."""
    fake_client = MagicMock()

    fake_client.get_balance.return_value = MagicMock(value=balance)

    # get_account_info: first call returns pre-init, second call returns post-init.
    pre = MagicMock(value=None) if not pda_exists_before else MagicMock(
        value=MagicMock(data=b"\x00" * 123)
    )
    post = MagicMock(value=MagicMock(data=b"\x00" * 123)) if pda_exists_after else MagicMock(value=None)
    fake_client.get_account_info.side_effect = [pre, post, post, post]

    # Use a real Solders Hash so Message.new_with_blockhash accepts it.
    from solders.hash import Hash as _SoldersHash
    _real_blockhash = _SoldersHash.from_string("11111111111111111111111111111111")
    fake_client.get_latest_blockhash.return_value = MagicMock(
        value=MagicMock(blockhash=_real_blockhash, last_valid_block_height=12345)
    )
    fake_client.send_raw_transaction.return_value = MagicMock(value=send_returns_sig)
    return fake_client


# ---------------------------------------------------------------------------
# Sanity guards
# ---------------------------------------------------------------------------

def test_mainnet_refused_without_explicit_flag(script_module, fresh_keypair_file):
    """RPC URL containing 'mainnet' must require --i-know-this-is-mainnet."""
    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.mainnet-beta.solana.com",
    ])
    assert rc == 2


def test_missing_keypair_file_exits_1(script_module, tmp_path):
    rc = script_module.main([
        "--keypair", str(tmp_path / "does_not_exist.json"),
        "--rpc-url", "https://api.devnet.solana.com",
    ])
    assert rc == 1


def test_malformed_keypair_file_exits_1(script_module, tmp_path):
    bad = tmp_path / "id.json"
    bad.write_text(json.dumps({"not": "a list"}))
    rc = script_module.main([
        "--keypair", str(bad),
        "--rpc-url", "https://api.devnet.solana.com",
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_pda_already_exists_exits_0_without_tx(script_module, fresh_keypair_file, monkeypatch):
    """Already-initialized PDA → exit 0, no TX submitted."""
    fake_client = _patch_client(monkeypatch, pda_exists_before=True)
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
    ])
    assert rc == 0
    fake_client.send_raw_transaction.assert_not_called()


# ---------------------------------------------------------------------------
# Dry-run / no --confirm
# ---------------------------------------------------------------------------

def test_dry_run_does_not_submit_tx(script_module, fresh_keypair_file, monkeypatch):
    fake_client = _patch_client(monkeypatch, pda_exists_before=False)
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
        "--dry-run",
    ])
    assert rc == 0
    fake_client.send_raw_transaction.assert_not_called()


def test_no_confirm_flag_does_not_submit_tx(script_module, fresh_keypair_file, monkeypatch):
    """Without --confirm, even non-dry-run runs as preview-only."""
    fake_client = _patch_client(monkeypatch, pda_exists_before=False)
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
    ])
    assert rc == 0
    fake_client.send_raw_transaction.assert_not_called()


# ---------------------------------------------------------------------------
# Successful init
# ---------------------------------------------------------------------------

def test_confirm_submits_tx_and_verifies(script_module, fresh_keypair_file, monkeypatch):
    """--confirm path: submits TX, polls for confirmation, verifies post-init."""
    fake_client = _patch_client(
        monkeypatch,
        pda_exists_before=False,
        pda_exists_after=True,
        send_returns_sig="SIG_init_real",
    )
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)

    # Mock _confirm_signature to return True without actually polling.
    monkeypatch.setattr(script_module, "_confirm_signature",
                        lambda client, sig, timeout_s=60.0: True)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
        "--confirm",
    ])
    assert rc == 0
    fake_client.send_raw_transaction.assert_called_once()


def test_confirm_path_fails_when_post_init_pda_missing(
    script_module, fresh_keypair_file, monkeypatch,
):
    """If the PDA still doesn't exist post-confirmation, surface as error."""
    fake_client = _patch_client(
        monkeypatch,
        pda_exists_before=False,
        pda_exists_after=False,
    )
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)
    monkeypatch.setattr(script_module, "_confirm_signature",
                        lambda client, sig, timeout_s=60.0: True)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
        "--confirm",
    ])
    assert rc == 1


def test_confirm_signature_timeout_returns_1(script_module, fresh_keypair_file, monkeypatch):
    """Confirmation timeout must propagate as exit 1."""
    fake_client = _patch_client(monkeypatch, pda_exists_before=False)
    monkeypatch.setattr("solana.rpc.api.Client", lambda url: fake_client)
    monkeypatch.setattr(script_module, "_confirm_signature",
                        lambda client, sig, timeout_s=60.0: False)

    rc = script_module.main([
        "--keypair", str(fresh_keypair_file),
        "--rpc-url", "https://api.devnet.solana.com",
        "--confirm",
    ])
    assert rc == 1
