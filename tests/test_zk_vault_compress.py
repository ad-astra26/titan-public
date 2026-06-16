"""
Offline tests for the ZK-Vault Light-Protocol compressed-account client +
the backup-event audit emit (End-state 1, PLAN_zk_vault_proof_completion).

Verifies the exact light-sdk 0.23 v1 remaining-accounts ordering, the
output-only no-proof framing, the instruction layout, the snapshot decoder,
and the in-process emit/verify/observable plumbing — all without touching the
chain. Run in its own pytest process (project convention).
"""
import asyncio
import json
import os
import struct
import tempfile

import pytest

from titan_hcl.utils import solana_client as sc

pytestmark = pytest.mark.skipif(not sc.is_available(), reason="solders not installed")

PROGRAM_ID = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"
AUTH = "YOUR_DEPLOYER_PUBKEY"  # any valid pubkey


# ── Light v1 remaining-accounts ordering (the make-or-break) ────────────────

def test_cpi_authority_pda_is_canonical():
    pda = sc.derive_light_cpi_authority(PROGRAM_ID)
    assert str(pda) == "J9jz43pWgXDPQJrRefw3FCfqXfi9a7TxCeCcdYZRxS3k"


def test_light_v1_remaining_accounts_order_and_flags():
    accts = sc.build_light_v1_remaining_accounts(PROGRAM_ID)
    assert accts is not None and len(accts) == 9, "8 system accounts + 1 state tree"
    expected = [
        sc.LIGHT_SYSTEM_PROGRAM_ID,
        "J9jz43pWgXDPQJrRefw3FCfqXfi9a7TxCeCcdYZRxS3k",  # cpi authority PDA
        sc.LIGHT_REGISTERED_PROGRAM_PDA,
        sc.LIGHT_NOOP_PROGRAM_ID,
        sc.LIGHT_ACCOUNT_COMPRESSION_AUTHORITY,
        sc.LIGHT_ACCOUNT_COMPRESSION_PROGRAM_ID,
        PROGRAM_ID,
        sc.SYSTEM_PROGRAM_ID,
        sc.LIGHT_V1_STATE_TREE,
    ]
    for i, (meta, exp) in enumerate(zip(accts, expected)):
        assert str(meta.pubkey) == exp, f"idx {i}: {meta.pubkey} != {exp}"
        assert meta.is_signer is False, f"idx {i} must not be signer"
    # Only the state tree (idx 8) is writable.
    assert accts[8].is_writable is True
    assert all(accts[i].is_writable is False for i in range(8))


def test_compression_authority_is_sdk_value_not_docs_value():
    # The 0.23-SDK value, NOT the zkcompression.com docs page (HZH7qS…).
    assert sc.LIGHT_ACCOUNT_COMPRESSION_AUTHORITY == "HwXnGK3tPkkVY6P439H2p68AxpeuWXd5PcrAxFpbmfbA"


# ── append_epoch_snapshot instruction layout (output-only, no proof) ────────

def test_append_epoch_snapshot_instruction_layout():
    from solders.pubkey import Pubkey
    state_root = bytes(range(32))
    shadow = bytes([7]) * 32
    ix = sc.build_append_epoch_snapshot_instruction(
        authority_pubkey=Pubkey.from_string(AUTH),
        state_root=state_root,
        memory_count=219450,
        sovereignty_score=2823,
        shadow_url_hash=shadow,
        program_id_str=PROGRAM_ID,
    )
    assert ix is not None
    # named [vault_pda, authority] + 9 Light remaining accounts
    assert len(ix.accounts) == 11
    assert ix.accounts[1].is_signer is True   # authority signs
    assert ix.accounts[0].is_signer is False  # vault pda
    # data: 8 disc + 1 (Borsh Option::None) + 32 state_root + 8 memcount
    #       + 2 sov + 32 shadow + 1 output_tree_index
    data = bytes(ix.data)
    assert data[:8] == sc._VAULT_IX_APPEND_EPOCH_SNAPSHOT
    assert data[8] == 0, "output-only ⇒ Borsh Option::None proof (single 0x00)"
    off = 9
    assert data[off:off + 32] == state_root
    off += 32
    assert struct.unpack_from("<Q", data, off)[0] == 219450
    off += 8
    assert struct.unpack_from("<H", data, off)[0] == 2823
    off += 2
    assert data[off:off + 32] == shadow
    off += 32
    assert data[off] == 0  # output_tree_index = 0


def test_compress_memory_batch_instruction_has_remaining_accounts():
    from solders.pubkey import Pubkey
    ix = sc.build_compress_memory_batch_instruction(
        authority_pubkey=Pubkey.from_string(AUTH),
        batch_root=bytes(range(32)),
        node_count=5,
        epoch_id=42,
        sovereignty_score=2823,
        program_id_str=PROGRAM_ID,
    )
    assert ix is not None
    assert len(ix.accounts) == 11  # was 2 before End-state 1
    assert bytes(ix.data)[8] == 0  # no proof


# ── snapshot decoder round-trip ─────────────────────────────────────────────

def test_decode_compressed_epoch_snapshot_roundtrip():
    auth = bytes([1]) * 32
    state_root = bytes(range(32))
    shadow = bytes([9]) * 32
    raw = (
        auth
        + struct.pack("<Q", 7)        # epoch_number
        + state_root
        + struct.pack("<Q", 219450)   # memory_count
        + struct.pack("<H", 2823)     # sovereignty_score
        + shadow
        + struct.pack("<q", 1700000000)
    )
    dec = sc.decode_compressed_epoch_snapshot(raw)
    assert dec is not None
    assert dec["state_root"] == state_root.hex()
    assert dec["memory_count"] == 219450
    assert dec["sovereignty_score"] == 2823
    assert dec["epoch_number"] == 7


# ── emit/verify/observable plumbing (mock network) ──────────────────────────

class _FakeNetwork:
    def __init__(self, pubkey, sig="5KQtestSig" + "a" * 80, program_id=PROGRAM_ID):
        from solders.pubkey import Pubkey
        self.pubkey = Pubkey.from_string(pubkey)
        self._vault_program_id = program_id
        self._sig = sig
        self.sent = []

    async def send_sovereign_transaction(self, ixs, priority="MEDIUM"):
        self.sent.append((ixs, priority))
        return self._sig


def test_emit_epoch_snapshot_builds_sends_and_writes_state(tmp_path, monkeypatch):
    from titan_hcl.logic import zk_vault_state as zk
    monkeypatch.chdir(tmp_path)  # so data/ files land in tmp
    net = _FakeNetwork(AUTH)
    state_root_hex = bytes(range(32)).hex()

    res = asyncio.get_event_loop().run_until_complete(
        zk.emit_epoch_snapshot(
            net, state_root_hex=state_root_hex, sovereignty_bp=2823,
            archive_hash="abc123", arweave_url="https://arweave.net/TX",
            titan_id="T2", memory_count=208128, photon=None,
        )
    )
    assert res["tx"] == net._sig
    assert res["error"] is None
    assert len(net.sent) == 1
    # observable state file written
    st = zk.read_zk_audit_state("T2")
    assert st is not None
    assert st["continuous_enabled"] is True
    assert st["last_tx"] == net._sig
    assert st["last_memory_count"] == 208128
    assert st["last_state_root"] == state_root_hex
    # snapshot history written too
    assert os.path.exists(zk.vault_snapshots_path("T2"))


def test_emit_epoch_snapshot_no_network_is_graceful():
    from titan_hcl.logic import zk_vault_state as zk
    res = asyncio.get_event_loop().run_until_complete(
        zk.emit_epoch_snapshot(
            None, state_root_hex="00" * 32, sovereignty_bp=0,
            archive_hash="x", arweave_url="", titan_id="T2", memory_count=1,
        )
    )
    assert res["tx"] is None and res["error"] == "no_network"


def test_read_timechain_block_count_missing_db_returns_none(tmp_path):
    from titan_hcl.logic import zk_vault_state as zk
    assert zk.read_timechain_block_count(str(tmp_path)) is None
