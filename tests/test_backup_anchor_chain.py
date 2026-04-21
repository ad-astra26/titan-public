"""Tests for rFP_backup_worker Phase 8.1 — anchor chain (prev_anchor_hash + chain file).

Covers:
  1. Chain file write/read roundtrip (append-only, atomic via tmp+rename)
  2. verify_chain: genesis (prev=""), valid chain, break detection
  3. Memo text v=2 format: verified by inspecting built string
  4. Chain-tip hash feeds next prev correctly
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from titan_plugin.logic.backup import RebirthBackup
from titan_plugin.logic.backup_chain import (
    read_chain,
    verify_chain,
    verify_chain_file,
)


# ────────────────────────────────────────────────────────────────────────────
# Chain file I/O
# ────────────────────────────────────────────────────────────────────────────

def test_read_chain_absent_returns_empty(tmp_path):
    assert read_chain("T1", base_dir=str(tmp_path)) == []


def test_read_chain_valid_file(tmp_path):
    p = tmp_path / "backup_anchor_chain_T1.json"
    p.write_text(json.dumps({
        "version": 1, "titan_id": "T1",
        "anchors": [
            {"backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
             "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0},
        ],
    }))
    got = read_chain("T1", base_dir=str(tmp_path))
    assert len(got) == 1
    assert got[0]["archive_hash"] == "a" * 64


# ────────────────────────────────────────────────────────────────────────────
# Verification
# ────────────────────────────────────────────────────────────────────────────

def test_verify_empty_chain_ok():
    r = verify_chain([])
    assert r == {"ok": True, "length": 0, "break_index": None, "break_reason": None}


def test_verify_valid_two_entry_chain():
    anchors = [
        {"archive_hash": "A" * 64, "prev_anchor_hash": ""},
        {"archive_hash": "B" * 64, "prev_anchor_hash": "A" * 64},
    ]
    r = verify_chain(anchors)
    assert r["ok"] is True
    assert r["length"] == 2
    assert r["break_index"] is None


def test_verify_valid_long_chain():
    prev = ""
    anchors = []
    for i in range(10):
        h = f"{i:02d}" * 32
        anchors.append({"archive_hash": h, "prev_anchor_hash": prev})
        prev = h
    assert verify_chain(anchors)["ok"] is True


def test_verify_detects_break_in_middle():
    anchors = [
        {"archive_hash": "A" * 64, "prev_anchor_hash": ""},
        {"archive_hash": "B" * 64, "prev_anchor_hash": "A" * 64},
        {"archive_hash": "C" * 64, "prev_anchor_hash": "X" * 64},  # BROKEN
        {"archive_hash": "D" * 64, "prev_anchor_hash": "C" * 64},
    ]
    r = verify_chain(anchors)
    assert r["ok"] is False
    assert r["break_index"] == 2
    assert "does not match" in r["break_reason"]


def test_verify_detects_non_empty_genesis_prev():
    anchors = [{"archive_hash": "A" * 64, "prev_anchor_hash": "X" * 64}]
    r = verify_chain(anchors)
    assert r["ok"] is False
    assert r["break_index"] == 0


def test_verify_chain_file_roundtrip(tmp_path):
    p = tmp_path / "backup_anchor_chain_T1.json"
    p.write_text(json.dumps({
        "version": 1, "titan_id": "T1",
        "anchors": [
            {"archive_hash": "a" * 64, "prev_anchor_hash": "",
             "backup_id": 0, "tx": "t1"},
            {"archive_hash": "b" * 64, "prev_anchor_hash": "a" * 64,
             "backup_id": 1, "tx": "t2"},
        ],
    }))
    r = verify_chain_file("T1", base_dir=str(tmp_path))
    assert r["ok"] is True
    assert r["length"] == 2


# ────────────────────────────────────────────────────────────────────────────
# RebirthBackup._append_chain_entry + _chain_tip_hash
# ────────────────────────────────────────────────────────────────────────────

def _make_backup(tmp_path, titan_id: str = "T1") -> RebirthBackup:
    # Minimal construct — no network, no config, just chain file I/O
    mock_network = MagicMock()
    b = RebirthBackup.__new__(RebirthBackup)
    b.network = mock_network
    b._titan_id = titan_id
    b._arweave_store = None
    b._full_config = {}
    b._last_personality_date = ""
    b._last_soul_date = ""
    b._meditation_count = 0
    b._meditation_count_since_nft = 0
    # Override chain path to tmp
    b._anchor_chain_path = lambda: str(tmp_path / f"backup_anchor_chain_{titan_id}.json")
    return b


def test_append_chain_entry_creates_file_and_appends(tmp_path):
    b = _make_backup(tmp_path)
    assert b._read_chain() == []
    b._append_chain_entry({
        "backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
        "tx": "sig1", "ts": 1, "backup_type": "personality", "size_mb": 25.0,
    })
    chain = b._read_chain()
    assert len(chain) == 1
    assert chain[0]["archive_hash"] == "a" * 64

    b._append_chain_entry({
        "backup_id": 1, "archive_hash": "b" * 64, "prev_anchor_hash": "a" * 64,
        "tx": "sig2", "ts": 2, "backup_type": "personality", "size_mb": 25.1,
    })
    chain = b._read_chain()
    assert len(chain) == 2
    # Verify full chain integrity
    assert verify_chain(chain)["ok"] is True


def test_chain_tip_hash_empty_initially(tmp_path):
    b = _make_backup(tmp_path)
    assert b._chain_tip_hash() == ""


def test_chain_tip_hash_returns_latest(tmp_path):
    b = _make_backup(tmp_path)
    b._append_chain_entry({
        "archive_hash": "a" * 64, "prev_anchor_hash": "",
        "tx": "s1", "ts": 1, "backup_type": "personality", "size_mb": 1.0,
        "backup_id": 0,
    })
    assert b._chain_tip_hash() == "a" * 64
    b._append_chain_entry({
        "archive_hash": "b" * 64, "prev_anchor_hash": "a" * 64,
        "tx": "s2", "ts": 2, "backup_type": "personality", "size_mb": 1.0,
        "backup_id": 1,
    })
    assert b._chain_tip_hash() == "b" * 64


def test_append_atomic_via_tmp_rename(tmp_path):
    """Ensure write is via .tmp + os.replace (no partial file on crash)."""
    b = _make_backup(tmp_path)
    b._append_chain_entry({
        "archive_hash": "a" * 64, "prev_anchor_hash": "",
        "tx": "s1", "ts": 1, "backup_type": "personality", "size_mb": 1.0,
        "backup_id": 0,
    })
    # No .tmp lingering
    tmp_siblings = list(tmp_path.glob("*.tmp"))
    assert not tmp_siblings


# ────────────────────────────────────────────────────────────────────────────
# anchor_backup_hash memo format v=2 + chain append on success
# ────────────────────────────────────────────────────────────────────────────

def test_anchor_emits_v2_memo_with_prev_genesis(tmp_path):
    """First anchor: prev=genesis."""
    b = _make_backup(tmp_path)
    b.network = MagicMock()
    b.network.keypair = MagicMock()
    b.network.pubkey = MagicMock()
    b.network.send_sovereign_transaction = AsyncMock(return_value="sig_first")

    captured_memo = {}

    def _build_memo(pk, text):
        captured_memo["text"] = text
        return MagicMock()

    with patch("titan_plugin.utils.solana_client.build_memo_instruction", side_effect=_build_memo), \
         patch("titan_plugin.utils.solana_client.is_available", return_value=True):
        sig = asyncio.run(b.anchor_backup_hash(
            archive_hash="abc123" * 10 + "defg",
            size_mb=25.0, backup_type="personality"))

    assert sig == "sig_first"
    assert "v=2" in captured_memo["text"]
    assert "prev=genesis" in captured_memo["text"]
    # And the chain was appended
    chain = b._read_chain()
    assert len(chain) == 1
    assert chain[0]["tx"] == "sig_first"
    assert chain[0]["prev_anchor_hash"] == ""


def test_anchor_second_entry_references_first(tmp_path):
    """Second anchor: prev = first's archive_hash[:16]."""
    b = _make_backup(tmp_path)
    b.network = MagicMock()
    b.network.keypair = MagicMock()
    b.network.pubkey = MagicMock()
    b.network.send_sovereign_transaction = AsyncMock(side_effect=["sig_1", "sig_2"])

    memos = []

    def _build(pk, text):
        memos.append(text)
        return MagicMock()

    with patch("titan_plugin.utils.solana_client.build_memo_instruction", side_effect=_build), \
         patch("titan_plugin.utils.solana_client.is_available", return_value=True):
        h1 = "a" * 64
        asyncio.run(b.anchor_backup_hash(h1, 25.0, "personality"))
        h2 = "b" * 64
        asyncio.run(b.anchor_backup_hash(h2, 25.0, "personality"))

    # Memo 0: prev=genesis; memo 1: prev=h1[:16]
    assert "prev=genesis" in memos[0]
    assert f"prev={h1[:16]}" in memos[1]
    # Chain has 2 entries, integrity intact
    chain = b._read_chain()
    assert verify_chain(chain)["ok"] is True
    assert chain[1]["prev_anchor_hash"] == h1


def test_anchor_no_chain_append_when_tx_fails(tmp_path):
    """If send_sovereign_transaction returns None, chain stays empty."""
    b = _make_backup(tmp_path)
    b.network = MagicMock()
    b.network.keypair = MagicMock()
    b.network.pubkey = MagicMock()
    b.network.send_sovereign_transaction = AsyncMock(return_value=None)

    with patch("titan_plugin.utils.solana_client.build_memo_instruction", return_value=MagicMock()), \
         patch("titan_plugin.utils.solana_client.is_available", return_value=True):
        sig = asyncio.run(b.anchor_backup_hash("a" * 64, 25.0, "personality"))

    assert sig is None
    assert b._read_chain() == [], "chain must not be appended if on-chain write failed"
