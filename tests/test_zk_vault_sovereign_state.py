"""
Offline tests for the ZK-Vault SovereignState client (End-state 2 — the running
canonical SNARK-per-write compressed account, RFP_zk_vault_snark_per_write).

Locks down — WITHOUT touching the chain — the parts that would otherwise only be
caught by a failed (real-SOL) devnet tx:
  • the pure-Python Keccak-256 + light-sdk v1 `derive_address` (byte-exact vs the
    crate's own test vectors — `light-sdk-types-0.23/src/address.rs` test mod),
  • the anchor instruction discriminators (sha256("global:<name>")[:8]),
  • the create/update instruction byte-layouts,
  • the §7.B remaining-accounts orderings (CREATE 11 / UPDATE 10),
  • the SovereignState serialize↔decode roundtrip.

Run in its own pytest process: `python -m pytest tests/test_zk_vault_sovereign_state.py -v -p no:anchorpy`
"""
import hashlib
import struct

import pytest

from titan_hcl.utils import solana_client as sc
from titan_hcl.utils._keccak import keccak256

pytestmark = pytest.mark.skipif(not sc.is_available(), reason="solders not installed")

PROGRAM_ID = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"
AUTH = "YOUR_DEPLOYER_PUBKEY"


# ── Keccak-256 (NOT hashlib.sha3_256) ───────────────────────────────────────

def test_keccak256_empty_vector():
    # canonical Ethereum keccak256("") — proves pre-NIST padding, not sha3.
    assert keccak256(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    assert keccak256(b"") != hashlib.sha3_256(b"").digest()


# ── light-sdk v1 derive_address — byte-exact vs crate test vectors ───────────

def _derive_seed(seeds, prog):
    h = bytearray(keccak256(prog + b"".join(seeds)))
    h[0] = 0
    return bytes(h)


def _derive_addr(seeds, tree, prog):
    s = _derive_seed(seeds, prog)
    h = bytearray(keccak256(tree + s + bytes([0xFF])))
    h[0] = 0
    return bytes(h)


def test_derive_address_matches_crate_vectors():
    from based58 import b58decode
    prog = b58decode(b"7yucc7fL3JGbyMwg4neUaenNSdySS39hbAk89Ao3t1Hz")
    # vector 1 (address.rs test_derive_address, seeds foo/bar, tree=[0;32])
    assert _derive_addr([b"foo", b"bar"], bytes(32), prog) == bytes([
        0, 141, 60, 24, 250, 156, 15, 250, 237, 196, 171, 243, 182, 10, 8, 66,
        147, 57, 27, 209, 222, 86, 109, 234, 161, 219, 142, 43, 121, 104, 16, 63])
    # vector 2 (seeds ayy/lmao)
    assert _derive_addr([b"ayy", b"lmao"], bytes(32), prog) == bytes([
        0, 104, 207, 102, 176, 61, 126, 178, 11, 174, 213, 195, 17, 36, 71, 95,
        0, 231, 179, 87, 218, 195, 114, 84, 47, 97, 176, 93, 106, 175, 72, 115])


def test_derive_light_v1_address_deterministic_and_field_safe():
    from solders.pubkey import Pubkey
    auth = Pubkey.from_string(AUTH)
    a1 = sc.derive_light_v1_address(auth, program_id_str=PROGRAM_ID)
    a2 = sc.derive_light_v1_address(auth, program_id_str=PROGRAM_ID)
    assert a1 is not None and len(a1) == 32
    assert a1 == a2, "address must be deterministic per (authority, tree, program)"
    assert a1[0] == 0, "truncated to <bn254 field size (first byte zeroed)"
    # different authority → different address
    other = sc.derive_light_v1_address(
        Pubkey.from_string(PROGRAM_ID), program_id_str=PROGRAM_ID)
    assert other != a1


# ── anchor discriminators ───────────────────────────────────────────────────

def test_instruction_discriminators():
    def disc(name):
        return hashlib.sha256(("global:" + name).encode()).digest()[:8]
    assert sc._VAULT_IX_CREATE_SOVEREIGN_STATE == disc("create_sovereign_state")
    assert sc._VAULT_IX_UPDATE_SOVEREIGN_STATE == disc("update_sovereign_state")


# ── CREATE instruction: layout + §7.B account ordering ───────────────────────

def test_build_create_sovereign_state_layout_and_accounts():
    from solders.pubkey import Pubkey
    auth = Pubkey.from_string(AUTH)
    ix = sc.build_create_sovereign_state_instruction(
        auth,
        proof_bytes=bytes(range(128)),
        address_merkle_tree_index=1,
        address_queue_index=2,
        address_root_index=1187,
        state_root=b"\x11" * 32,
        memory_count=219450,
        sovereignty_score=2823,
        shadow_url_hash=b"\x22" * 32,
        output_tree_index=0,
        program_id_str=PROGRAM_ID,
    )
    assert ix is not None
    # data: 8 disc + 1 Some + 128 proof + 4 PackedAddressTreeInfo + 1 out_idx
    #       + 32 state_root + 8 mem + 2 sov + 32 shadow = 216
    # (epoch_number is read on-chain from vault.commit_count — NOT in the data)
    d = bytes(ix.data)
    assert d[:8] == sc._VAULT_IX_CREATE_SOVEREIGN_STATE
    assert d[8] == 1, "Option::Some tag"
    assert d[9:9 + 128] == bytes(range(128))
    off = 9 + 128
    assert d[off] == 1 and d[off + 1] == 2, "addr mt/queue packed indices"
    assert struct.unpack_from("<H", d, off + 2)[0] == 1187
    assert d[off + 4] == 0, "output_tree_index"
    off += 5
    assert d[off:off + 32] == b"\x11" * 32
    assert struct.unpack_from("<Q", d, off + 32)[0] == 219450
    assert struct.unpack_from("<H", d, off + 40)[0] == 2823
    assert d[off + 42:off + 74] == b"\x22" * 32
    assert len(d) == 216
    # accounts: [vault, authority] + 8 system + [out_tree, addr_tree, addr_queue]
    accts = ix.accounts
    assert len(accts) == 13
    assert accts[1].is_signer and accts[1].is_writable  # authority
    assert str(accts[2].pubkey) == sc.LIGHT_SYSTEM_PROGRAM_ID  # system[0] = remaining[0]
    assert str(accts[10].pubkey) == sc.LIGHT_V1_STATE_TREE     # packed 0 = remaining[8]
    assert str(accts[11].pubkey) == sc.LIGHT_V1_ADDRESS_TREE   # packed 1 = remaining[9]
    assert str(accts[12].pubkey) == sc.LIGHT_V1_ADDRESS_QUEUE  # packed 2 = remaining[10]
    assert accts[10].is_writable and accts[11].is_writable and accts[12].is_writable


def test_create_rejects_bad_proof():
    from solders.pubkey import Pubkey
    auth = Pubkey.from_string(AUTH)
    ix = sc.build_create_sovereign_state_instruction(
        auth, proof_bytes=b"\x00" * 64,  # wrong length
        address_merkle_tree_index=1, address_queue_index=2, address_root_index=0,
        state_root=b"\x11" * 32, memory_count=1,
        sovereignty_score=1, shadow_url_hash=b"\x22" * 32, program_id_str=PROGRAM_ID)
    assert ix is None


# ── UPDATE instruction: layout + §7.B account ordering ───────────────────────

def _sample_old_state():
    return {
        "authority": ("aa" * 32),
        "epoch_number": 108,
        "state_root": ("bb" * 32),
        "memory_count": 219000,
        "sovereignty_score": 2800,
        "shadow_url_hash": ("cc" * 32),
        "timestamp": 1700000000,
    }


def test_build_update_sovereign_state_layout_and_accounts():
    from solders.pubkey import Pubkey
    auth = Pubkey.from_string(AUTH)
    ix = sc.build_update_sovereign_state_instruction(
        auth,
        proof_bytes=bytes(range(128)),
        old_state=_sample_old_state(),
        address=b"\x33" * 32,
        leaf_index=412,
        root_index=1187,
        prove_by_index=False,
        merkle_tree_index=0,
        queue_index=1,
        output_state_tree_index=0,
        state_root=b"\x44" * 32,
        memory_count=219450,
        sovereignty_score=2823,
        shadow_url_hash=b"\x55" * 32,
        program_id_str=PROGRAM_ID,
    )
    assert ix is not None
    d = bytes(ix.data)
    assert d[:8] == sc._VAULT_IX_UPDATE_SOVEREIGN_STATE
    assert d[8] == 1 and d[9:9 + 128] == bytes(range(128))
    off = 9 + 128
    # CompressedAccountMeta.tree_info = PackedStateTreeInfo
    assert struct.unpack_from("<H", d, off)[0] == 1187      # root_index
    assert d[off + 2] == 0                                  # prove_by_index
    assert d[off + 3] == 0                                  # merkle_tree_pubkey_index
    assert d[off + 4] == 1                                  # queue_pubkey_index
    assert struct.unpack_from("<I", d, off + 5)[0] == 412   # leaf_index
    off += 9
    assert d[off:off + 32] == b"\x33" * 32                  # address
    assert d[off + 32] == 0                                 # output_state_tree_index
    off += 33
    # old_state SovereignState (122 bytes) then new fields
    old_blob = d[off:off + 122]
    assert old_blob == sc._serialize_sovereign_state(_sample_old_state())
    off += 122
    assert d[off:off + 32] == b"\x44" * 32                  # new state_root
    assert struct.unpack_from("<Q", d, off + 32)[0] == 219450  # memory_count
    assert struct.unpack_from("<H", d, off + 40)[0] == 2823    # sovereignty
    assert d[off + 42:off + 74] == b"\x55" * 32                # shadow_url_hash
    # total = 8 + 129 + 42 + 122 + 74 = 375 (epoch_number read on-chain, not in data)
    assert len(d) == 375
    # accounts: [vault, authority] + 8 system + [state_tree, nullifier_queue]
    accts = ix.accounts
    assert len(accts) == 12
    assert str(accts[10].pubkey) == sc.LIGHT_V1_STATE_TREE        # packed 0 = remaining[8]
    assert str(accts[11].pubkey) == sc.LIGHT_V1_NULLIFIER_QUEUE   # packed 1 = remaining[9]
    assert accts[10].is_writable and accts[11].is_writable


# ── SovereignState serialize ↔ decode roundtrip ─────────────────────────────

# ── emit_sovereign_state orchestration (selector: genesis vs update) ─────────

import asyncio
import base64


class _FakeNetwork:
    def __init__(self, pubkey, sig="5KQ" + "z" * 85, program_id=PROGRAM_ID):
        from solders.pubkey import Pubkey
        self.pubkey = Pubkey.from_string(pubkey)
        self._vault_program_id = program_id
        self._sig = sig
        self.sent = []

    async def send_sovereign_transaction(self, ixs, priority="MEDIUM"):
        self.sent.append((ixs, priority))
        return self._sig


def _fake_proof_value(root_index=7):
    return {
        "compressedProof": {
            "a": base64.b64encode(b"\x01" * 32).decode(),
            "b": base64.b64encode(b"\x02" * 64).decode(),
            "c": base64.b64encode(b"\x03" * 32).decode(),
        },
        "rootIndex": root_index,
        "addressRootIndex": root_index,
    }


class _FakePhoton:
    """Genesis when `existing is None`; update when it returns an account dict."""
    def __init__(self, existing=None):
        self.existing = existing

    async def get_compressed_account(self, address_b58):
        return self.existing

    async def get_new_address_proof(self, address_b58, tree_b58):
        return _fake_proof_value()

    async def get_validity_proof(self, hashes):
        return _fake_proof_value()


def test_emit_sovereign_state_genesis_creates(tmp_path, monkeypatch):
    from titan_hcl.logic import zk_vault_state as zk
    monkeypatch.chdir(tmp_path)
    net = _FakeNetwork(AUTH)
    photon = _FakePhoton(existing=None)  # no account → GENESIS
    res = asyncio.new_event_loop().run_until_complete(
        zk.emit_sovereign_state(
            net, state_root_hex=bytes(range(32)).hex(), sovereignty_bp=2823,
            arweave_url="https://arweave.net/TX", archive_hash="abc",
            titan_id="T2", memory_count=208128, photon=photon,
        )
    )
    assert res["op"] == "create"
    assert res["tx"] == net._sig
    assert len(net.sent) == 1
    sent_ix = net.sent[0][0][0]
    assert bytes(sent_ix.data)[:8] == sc._VAULT_IX_CREATE_SOVEREIGN_STATE
    assert net.sent[0][1] == "HIGH"


def test_emit_sovereign_state_update_consumes_existing(tmp_path, monkeypatch):
    from titan_hcl.logic import zk_vault_state as zk
    monkeypatch.chdir(tmp_path)
    net = _FakeNetwork(AUTH)
    old = {
        "authority": "aa" * 32, "epoch_number": 5, "state_root": "bb" * 32,
        "memory_count": 1000, "sovereignty_score": 2000,
        "shadow_url_hash": "cc" * 32, "timestamp": 1700000000,
    }
    existing = {
        "data": {"data": base64.b64encode(sc._serialize_sovereign_state(old)).decode()},
        "hash": "HzznVxith6L5pp5w8Aht5oV2vXpQ8Qd1qbWYHGAytestHASH",
        "leafIndex": 412,
    }
    photon = _FakePhoton(existing=existing)  # account exists → UPDATE
    res = asyncio.new_event_loop().run_until_complete(
        zk.emit_sovereign_state(
            net, state_root_hex=bytes(range(32, 64)).hex(), sovereignty_bp=2900,
            arweave_url="https://arweave.net/TX2", archive_hash="def",
            titan_id="T2", memory_count=208200, photon=photon,
        )
    )
    assert res["op"] == "update"
    assert res["tx"] == net._sig
    sent_ix = net.sent[0][0][0]
    assert bytes(sent_ix.data)[:8] == sc._VAULT_IX_UPDATE_SOVEREIGN_STATE


def test_emit_sovereign_state_requires_photon(tmp_path, monkeypatch):
    from titan_hcl.logic import zk_vault_state as zk
    monkeypatch.chdir(tmp_path)
    res = asyncio.new_event_loop().run_until_complete(
        zk.emit_sovereign_state(
            _FakeNetwork(AUTH), state_root_hex=bytes(range(32)).hex(),
            sovereignty_bp=1, arweave_url="", archive_hash="", titan_id="T2",
            memory_count=1, photon=None,
        )
    )
    assert res["error"] == "no_photon" and res["tx"] is None


def test_sovereign_state_roundtrip():
    s = _sample_old_state()
    blob = sc._serialize_sovereign_state(s)
    assert blob is not None and len(blob) == 122
    dec = sc.decode_sovereign_state(blob)
    assert dec["type"] == "SovereignState"
    assert dec["authority"] == s["authority"]
    assert dec["epoch_number"] == s["epoch_number"]
    assert dec["state_root"] == s["state_root"]
    assert dec["memory_count"] == s["memory_count"]
    assert dec["sovereignty_score"] == s["sovereignty_score"]
    assert dec["shadow_url_hash"] == s["shadow_url_hash"]
    assert dec["timestamp"] == s["timestamp"]
