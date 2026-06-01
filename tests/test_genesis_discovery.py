"""Offline unit tests for the wallet-only genesis discovery (RFP G1 / INV-MBR-10).

No network, no Solana, no Arweave — `httpx.AsyncClient` is faked. Covers:
  - the Shard-3 wallet-history walk (pagination + memo extraction),
  - a known-TX Shard-3 fetch,
  - GenesisNFT DAS enumeration + non-DAS graceful degrade,
  - NFT `ar://` identity-metadata parsing,
  - the discover_genesis orchestrator end-to-end (with a REAL SSS Shard-3 that
    decrypts + combines back to the original keypair — the sovereign crux),
  - resurrection._parse_shard1 (raw shard, canonical) vs legacy envelope.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO / "scripts"))

from titan_hcl.utils import genesis_discovery as gd  # noqa: E402
from titan_hcl.utils import shamir  # noqa: E402


# ── fake httpx ───────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class _FakeAsyncClient:
    post_handler = None
    get_handler = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        return type(self).post_handler(url, json)

    async def get(self, url):
        return type(self).get_handler(url)


def _patch_httpx(post_handler=None, get_handler=None):
    _FakeAsyncClient.post_handler = staticmethod(post_handler) if post_handler else None
    _FakeAsyncClient.get_handler = staticmethod(get_handler) if get_handler else None
    return patch("httpx.AsyncClient", _FakeAsyncClient)


def _run(coro):
    return asyncio.run(coro)


# ── Shard-3 wallet-history walk ──────────────────────────────────────────


def test_find_shard3_anchor_paginates_to_genesis():
    """Page 1 = 1000 sigs no memo; page 2 (via before=) carries the memo."""
    page1 = {"result": [{"signature": f"s{i}", "memo": None} for i in range(1000)]}
    page2 = {"result": [
        {"signature": "older1", "memo": "unrelated"},
        {"signature": "SHARD3SIG", "memo": "[186] TITAN_GENESIS_SHARD3:deadbeef"},
    ]}

    def post(url, body):
        params = body["params"]
        before = params[1].get("before") if len(params) > 1 else None
        return _FakeResp(page1 if not before else page2)

    with _patch_httpx(post_handler=post):
        res = _run(gd.find_shard3_anchor("PUBKEY", "http://rpc"))
    assert res == {"shard3_tx": "SHARD3SIG", "encrypted_hex": "deadbeef"}


def test_find_shard3_anchor_not_found_returns_none():
    def post(url, body):
        return _FakeResp({"result": []})

    with _patch_httpx(post_handler=post):
        assert _run(gd.find_shard3_anchor("PUBKEY", "http://rpc")) is None


def test_fetch_shard3_from_tx_extracts_from_logs():
    tx = {"result": {"meta": {"logMessages": [
        "Program log: TITAN_GENESIS_SHARD3:cafe01"]},
        "transaction": {"message": {"instructions": []}}}}

    def post(url, body):
        return _FakeResp(tx)

    with _patch_httpx(post_handler=post):
        assert _run(gd.fetch_shard3_from_tx("SIG", "http://rpc")) == "cafe01"


# ── GenesisNFT DAS enumeration ───────────────────────────────────────────


def test_find_genesis_nft_filters_owned_assets():
    owned = {"result": {"items": [
        {"id": "RandomMintXYZ", "content": {"metadata": {"name": "Some PFP"}}},
        {"id": "AKZSCcGenesis", "content": {
            "metadata": {"name": "Titan Soul Gen 1"},
            "json_uri": "ar://2biCzkwG"}},
    ]}}

    def post(url, body):
        assert body["method"] == "getAssetsByOwner"
        return _FakeResp(owned)

    with _patch_httpx(post_handler=post):
        asset = _run(gd.find_genesis_nft("PUBKEY", "http://das"))
    assert asset["id"] == "AKZSCcGenesis"


def test_find_genesis_nft_non_das_rpc_degrades_to_none():
    """A public RPC returns a JSON-RPC error for getAssetsByOwner → None, no raise."""
    def post(url, body):
        return _FakeResp({"error": {"code": -32601, "message": "Method not found"}})

    with _patch_httpx(post_handler=post):
        assert _run(gd.find_genesis_nft("PUBKEY", "http://publicrpc")) is None


def test_read_nft_identity_parses_attributes():
    meta = {"name": "Titan Soul Gen 1", "attributes": [
        {"trait_type": "Maker", "value": "Bsg2swDJ"},
        {"trait_type": "Titan Pubkey", "value": "J1cdk4f1"},
        {"trait_type": "Constitution SHA-256", "value": "b0428705"},
        {"trait_type": "Birth DNA SHA-256", "value": "b9d60845"},
        {"trait_type": "Architecture", "value": "v6-132D"},
    ]}

    def get(url):
        assert "2biCzkwG" in url and url.startswith("https://arweave.net/")
        return _FakeResp(meta)

    asset = {"content": {"json_uri": "ar://2biCzkwG"}}
    with _patch_httpx(get_handler=get):
        ident = _run(gd.read_nft_identity(asset))
    assert ident["maker"] == "Bsg2swDJ"
    assert ident["constitution_sha"] == "b0428705"
    assert ident["birth_dna_sha"] == "b9d60845"
    assert ident["architecture"] == "v6-132D"


# ── orchestrator end-to-end: real SSS Shard-3 round-trips ────────────────


def test_discover_genesis_recovers_real_shard3_and_identity():
    """The sovereign crux: from the wallet alone, discover_genesis returns a
    Shard-3 that decrypts + combines back to the ORIGINAL keypair."""
    from solders.keypair import Keypair

    kp = Keypair()
    key_bytes = bytes(kp)
    pubkey = str(kp.pubkey())
    shards = shamir.split_secret(key_bytes, n=3, t=2)
    # Shard-3 (index 2) is the on-chain one; encrypt it exactly as genesis does.
    enc = shamir.encrypt_shard3(shards[2], pubkey)
    memo = f"[{len(enc)}] TITAN_GENESIS_SHARD3:{enc.hex()}"

    owned = {"result": {"items": [{"id": "GenesisMint", "content": {
        "metadata": {"name": "Titan Soul Gen 1"}, "json_uri": "ar://meta"}}]}}
    nft_meta = {"attributes": [{"trait_type": "Maker", "value": "MakerAddr"}]}

    def post(url, body):
        m = body["method"]
        if m == "getSignaturesForAddress":
            before = body["params"][1].get("before")
            if before:
                return _FakeResp({"result": []})
            return _FakeResp({"result": [{"signature": "S3SIG", "memo": memo}]})
        if m == "getAssetsByOwner":
            return _FakeResp(owned)
        return _FakeResp({"result": None})

    def get(url):
        return _FakeResp(nft_meta)

    with _patch_httpx(post_handler=post, get_handler=get):
        disc = _run(gd.discover_genesis(pubkey, "http://rpc"))

    assert disc["shard3_tx"] == "S3SIG"
    shard3 = shamir.decrypt_shard3(bytes.fromhex(disc["shard3_encrypted_hex"]), pubkey)
    reconstructed = shamir.combine_shares([shards[0], shard3])  # Shard-1 + Shard-3
    assert reconstructed == key_bytes
    assert disc["maker"] == "MakerAddr"
    assert disc["nft_address"] == "GenesisMint"


# ── G1.1a: NFT-embedded recovery pointer (INV-MBR-5a) ────────────────────


def test_genesis_recovery_block_schema():
    from titan_hcl.logic.birth_dna import genesis_recovery_block
    blk = genesis_recovery_block("3KCXvbShard3", vault_pda="BFgbYh", nft_address="AKZSCc")
    assert blk == {"version": "1.0", "shard3_tx": "3KCXvbShard3",
                   "vault_pda": "BFgbYh", "nft_address": "AKZSCc"}
    # shard3_tx is mandatory
    import pytest as _pt
    with _pt.raises(ValueError):
        genesis_recovery_block("")


def test_build_genesis_nft_metadata_embeds_recovery():
    from titan_hcl.logic.birth_dna import (
        build_genesis_nft_metadata, genesis_recovery_block)
    md = build_genesis_nft_metadata(
        "T1", recovery=genesis_recovery_block("3KCXvbShard3"))
    assert md["recovery"]["shard3_tx"] == "3KCXvbShard3"
    assert md["name"].startswith("Titan Genesis")
    assert isinstance(md["attributes"], list) and md["attributes"]
    # without recovery, no recovery key (legacy/incomplete mint)
    md2 = build_genesis_nft_metadata("T1")
    assert "recovery" not in md2


def test_read_nft_identity_extracts_recovery_block():
    meta = {"name": "Titan Soul Gen 1",
            "attributes": [{"trait_type": "Maker", "value": "Bsg2swDJ"}],
            "recovery": {"version": "1.0", "shard3_tx": "S3FROMNFT",
                         "vault_pda": "BFgbYh"}}

    def get(url):
        return _FakeResp(meta)

    asset = {"content": {"json_uri": "ar://meta"}}
    with _patch_httpx(get_handler=get):
        ident = _run(gd.read_nft_identity(asset))
    assert ident["shard3_tx"] == "S3FROMNFT"
    assert ident["vault_pda"] == "BFgbYh"
    assert ident["recovery"]["shard3_tx"] == "S3FROMNFT"


def test_discover_genesis_prefers_nft_embedded_shard3_tx():
    """A Titan minted WITH the recovery block resolves Shard-3 via the NFT
    pointer (getTransaction on the embedded shard3_tx) — the wallet-history
    walk is NOT needed. Real SSS round-trip back to the keypair."""
    from solders.keypair import Keypair

    kp = Keypair()
    key_bytes = bytes(kp)
    pubkey = str(kp.pubkey())
    shards = shamir.split_secret(key_bytes, n=3, t=2)
    enc = shamir.encrypt_shard3(shards[2], pubkey)

    owned = {"result": {"items": [{"id": "GenesisMint", "content": {
        "metadata": {"name": "Titan Soul Gen 1"}, "json_uri": "ar://meta"}}]}}
    nft_meta = {"attributes": [{"trait_type": "Maker", "value": "MakerAddr"}],
                "recovery": {"version": "1.0", "shard3_tx": "S3SIG_NFT"}}
    walk_called = {"n": 0}

    def post(url, body):
        m = body["method"]
        if m == "getAssetsByOwner":
            return _FakeResp(owned)
        if m == "getTransaction":
            assert body["params"][0] == "S3SIG_NFT"  # the NFT-embedded pointer
            return _FakeResp({"result": {"meta": {"logMessages": [
                f"Program log: TITAN_GENESIS_SHARD3:{enc.hex()}"]},
                "transaction": {"message": {"instructions": []}}}})
        if m == "getSignaturesForAddress":
            walk_called["n"] += 1     # must NOT happen — pointer path wins
            return _FakeResp({"result": []})
        return _FakeResp({"result": None})

    def get(url):
        return _FakeResp(nft_meta)

    with _patch_httpx(post_handler=post, get_handler=get):
        disc = _run(gd.discover_genesis(pubkey, "http://rpc"))

    assert disc["shard3_tx"] == "S3SIG_NFT"
    assert walk_called["n"] == 0, "wallet-history walk should be skipped when the NFT carries the pointer"
    shard3 = shamir.decrypt_shard3(bytes.fromhex(disc["shard3_encrypted_hex"]), pubkey)
    assert shamir.combine_shares([shards[0], shard3]) == key_bytes
    assert disc["maker"] == "MakerAddr"


# ── resurrection._parse_shard1: raw (canonical) vs legacy envelope ───────


def test_parse_shard1_raw_canonical_no_envelope():
    import resurrection
    raw = bytes(range(65))  # a 65-byte shard (x-coord + 64 data)
    shard, pubkey = resurrection._parse_shard1(raw.hex())
    assert shard == raw
    assert pubkey is None  # raw shard carries NO pubkey — supplied via --titan-pubkey


def test_parse_shard1_tolerates_legacy_envelope():
    import resurrection
    raw = bytes(range(65))
    env = shamir.create_maker_envelope(raw, "LEGACYPUBKEY", genesis_tx="oldtx")
    shard, pubkey = resurrection._parse_shard1(env)
    assert shard == raw
    assert pubkey == "LEGACYPUBKEY"


def test_parse_shard1_handles_whitespace_wrapped_hex():
    import resurrection
    raw = bytes(range(65))
    wrapped = "  " + "  ".join(raw.hex()[i:i + 8] for i in range(0, len(raw.hex()), 8)) + "\n"
    shard, _ = resurrection._parse_shard1(wrapped)
    assert shard == raw
