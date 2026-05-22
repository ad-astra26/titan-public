"""Tests for SPEC §24.7 — ZK Vault Merkle commit per backup event.

Per rFP §5.2 test coverage:
  - Deterministic event_merkle_root (sha256 over canonical concat)
  - v=2 memo format string (event_id, root[:32], prev[:16] / "genesis")
  - prev-chain linkage round-trip via parse_zk_memo
  - Length bound: memo stays within Solana memo cap

Plus a thin integration test that RebirthBackup.commit_event_merkle_to_zk_vault
delegates to send_sovereign_transaction with the canonical memo (mocked Solana).
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from titan_hcl.logic.backup import RebirthBackup
from titan_hcl.logic.backup_zk_commit import (
    ZK_COMMIT_MEMO_MAX_BYTES,
    ZK_COMMIT_MEMO_VERSION,
    build_zk_memo,
    compute_event_merkle_root,
    parse_zk_memo,
    zero_merkle_root,
)


# ── compute_event_merkle_root ────────────────────────────────────────────


def test_event_merkle_is_sha256_of_concat():
    p = "a" * 64
    t = "b" * 64
    s = "c" * 64
    expected = hashlib.sha256(
        bytes.fromhex(p) + bytes.fromhex(t) + bytes.fromhex(s)
    ).hexdigest()
    assert compute_event_merkle_root(p, t, s) == expected


def test_event_merkle_deterministic():
    p = "1" * 64
    t = "2" * 64
    s = "3" * 64
    r1 = compute_event_merkle_root(p, t, s)
    r2 = compute_event_merkle_root(p, t, s)
    assert r1 == r2


def test_event_merkle_no_soul_uses_zero32():
    """Non-soul events (most days) substitute zero32 for soul slot."""
    p = "a" * 64
    t = "b" * 64
    explicit = compute_event_merkle_root(p, t, zero_merkle_root())
    implicit = compute_event_merkle_root(p, t)
    assert explicit == implicit


def test_event_merkle_distinguishes_soul_vs_no_soul():
    """If soul slot is present, root must differ from the zero-soul root."""
    p = "a" * 64
    t = "b" * 64
    s = "c" * 64
    assert compute_event_merkle_root(p, t, s) != compute_event_merkle_root(p, t)


@pytest.mark.parametrize("bad_input", [
    "abc",         # too short
    "g" * 64,      # invalid hex
    None,          # not a string
    "a" * 63,      # off by one
    "a" * 65,      # off by one
])
def test_event_merkle_rejects_malformed_input(bad_input):
    valid = "a" * 64
    with pytest.raises(ValueError):
        compute_event_merkle_root(bad_input, valid)


def test_zero_merkle_root_is_64_zeros():
    assert zero_merkle_root() == "0" * 64
    assert len(zero_merkle_root()) == 64


# ── build_zk_memo ────────────────────────────────────────────────────────


def test_memo_format_matches_spec_24_7():
    """v=2;event_id={id};root={root[:32]};prev={prev[:16]}"""
    event_id = "abc123-event-uuid"
    root = "f" * 64
    prev = "e" * 64
    memo = build_zk_memo(event_id, root, prev)
    assert memo == (
        f"v=2;event_id={event_id};"
        f"root={'f' * 32};"
        f"prev={'e' * 16}"
    )


def test_memo_first_event_uses_genesis_prev():
    """First event in chain → prev='genesis' literal (matches Phase 8 convention)."""
    memo = build_zk_memo("first_ev", "0" * 64, prev_event_merkle_root=None)
    assert "prev=genesis" in memo


def test_memo_empty_event_id_rejected():
    with pytest.raises(ValueError, match="event_id must be a non-empty"):
        build_zk_memo("", "a" * 64)


def test_memo_short_root_rejected():
    with pytest.raises(ValueError, match="must be 64 hex chars"):
        build_zk_memo("evt", "abc")


def test_memo_stays_within_solana_cap():
    """Memo string must fit comfortably in Solana memo (cap 200 bytes per
    SPEC §24.7 ZK_COMMIT_MEMO_MAX_BYTES; full Solana cap is 566 — we want
    headroom)."""
    memo = build_zk_memo(
        event_id="ev_" + "x" * 32,  # 35 chars — generous event_id
        event_merkle_root="a" * 64,
        prev_event_merkle_root="b" * 64,
    )
    assert len(memo.encode("utf-8")) <= ZK_COMMIT_MEMO_MAX_BYTES


def test_memo_rejects_oversize_event_id():
    huge_id = "x" * 200
    with pytest.raises(ValueError, match="exceeds max"):
        build_zk_memo(huge_id, "a" * 64, "b" * 64)


# ── parse_zk_memo round-trip ─────────────────────────────────────────────


def test_parse_zk_memo_round_trip():
    original_id = "evt_uuid_4_xyz"
    original_root = "f" * 64
    original_prev = "e" * 64
    memo = build_zk_memo(original_id, original_root, original_prev)
    parsed = parse_zk_memo(memo)
    assert parsed["event_id"] == original_id
    assert parsed["root"] == original_root[:32]
    assert parsed["prev"] == original_prev[:16]


def test_parse_zk_memo_first_event_genesis():
    memo = build_zk_memo("first_ev", "a" * 64, None)
    parsed = parse_zk_memo(memo)
    assert parsed["prev"] == "genesis"


@pytest.mark.parametrize("bad_memo", [
    "v=1;event_id=x;root=...",                                          # wrong version
    "v=2;event_id=x;root=zz" + "0" * 30 + ";prev=" + "0" * 16,         # invalid hex root
    "v=2;event_id=;root=" + "a" * 32 + ";prev=" + "b" * 16,            # empty event_id
    "v=2;event_id=x;prev=" + "b" * 16,                                  # missing root
    "garbage",
    "",
])
def test_parse_zk_memo_rejects_malformed(bad_memo):
    assert parse_zk_memo(bad_memo) is None


def test_zk_commit_memo_version_constant():
    assert ZK_COMMIT_MEMO_VERSION == 2


# ── RebirthBackup integration (mocked Solana) ─────────────────────────────


@pytest.mark.asyncio
async def test_commit_event_merkle_to_zk_vault_builds_correct_memo():
    """RebirthBackup.commit_event_merkle_to_zk_vault delegates to
    send_sovereign_transaction with the canonical v=2 memo + returns
    the resulting tx_id."""
    # Build a fake network client
    fake_network = MagicMock()
    fake_network.send_sovereign_transaction = AsyncMock(return_value="fake_tx_sig_123")
    fake_network.keypair = object()  # not None
    fake_network.pubkey = MagicMock()

    rb = RebirthBackup(network_client=fake_network, titan_id="T1",
                       arweave_store=None, full_config={})

    # Patch the solana_client module so we don't need a real Solana client
    captured_memo = []

    def _fake_build_memo_ix(pubkey, text):
        captured_memo.append(text)
        return "fake_memo_ix"

    with patch("titan_hcl.utils.solana_client.is_available", return_value=True), \
         patch("titan_hcl.utils.solana_client.build_memo_instruction",
               side_effect=_fake_build_memo_ix):
        sig = await rb.commit_event_merkle_to_zk_vault(
            event_id="ev_abc",
            event_merkle_root="a" * 64,
            prev_event_merkle_root="b" * 64,
        )

    assert sig == "fake_tx_sig_123"
    assert len(captured_memo) == 1
    memo = captured_memo[0]
    assert memo.startswith("v=2;event_id=ev_abc;")
    assert f"root={'a' * 32}" in memo
    assert f"prev={'b' * 16}" in memo
    fake_network.send_sovereign_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_commit_event_merkle_returns_none_when_no_network():
    """No network client → return None, don't crash."""
    rb = RebirthBackup(network_client=None, titan_id="T1",
                       arweave_store=None, full_config={})
    result = await rb.commit_event_merkle_to_zk_vault(
        event_id="ev_x", event_merkle_root="a" * 64,
        prev_event_merkle_root=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_commit_event_merkle_returns_none_when_keypair_missing():
    fake_network = MagicMock()
    fake_network.send_sovereign_transaction = AsyncMock()
    fake_network.keypair = None  # no signing key

    rb = RebirthBackup(network_client=fake_network, titan_id="T1",
                       arweave_store=None, full_config={})
    with patch("titan_hcl.utils.solana_client.is_available", return_value=True):
        result = await rb.commit_event_merkle_to_zk_vault(
            event_id="ev_x", event_merkle_root="a" * 64,
            prev_event_merkle_root=None,
        )
    assert result is None
    fake_network.send_sovereign_transaction.assert_not_called()


@pytest.mark.asyncio
async def test_commit_event_merkle_handles_send_failure():
    """If send_sovereign_transaction raises, return None gracefully
    (caller emits BACKUP_EVENT_FAILED + retries next meditation)."""
    fake_network = MagicMock()
    fake_network.send_sovereign_transaction = AsyncMock(
        side_effect=RuntimeError("Solana RPC unreachable"))
    fake_network.keypair = object()
    fake_network.pubkey = MagicMock()

    rb = RebirthBackup(network_client=fake_network, titan_id="T1",
                       arweave_store=None, full_config={})

    with patch("titan_hcl.utils.solana_client.is_available", return_value=True), \
         patch("titan_hcl.utils.solana_client.build_memo_instruction",
               return_value="fake_ix"):
        result = await rb.commit_event_merkle_to_zk_vault(
            event_id="ev_x", event_merkle_root="a" * 64,
            prev_event_merkle_root=None,
        )
    assert result is None
