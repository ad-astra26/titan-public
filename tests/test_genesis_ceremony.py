"""Offline tests for the production genesis ceremony (RFP_genesis_ceremony_production
Phase I / §8 gates G1-G2). All on-chain effects are excluded — these cover the
pure logic: shard math, the shard3_tx≠genesis_tx separation (INV-MBR-3), the
funding gate (INV-GEN-FUND), record completeness, and the --network branch.
"""
import base64
import importlib
import os
import sys

import pytest

# genesis_ceremony lives in scripts/ — import it directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
gc = importlib.import_module("genesis_ceremony")

from titan_hcl.utils.shamir import (  # noqa: E402
    split_secret, verify_all_combinations, combine_shares, encrypt_shard3,
)


# ── Shard math (G1) ───────────────────────────────────────────────────────────

def test_shamir_split_verifies_all_combinations():
    key = bytes(range(64))
    shards = split_secret(key, n=3, t=2)
    assert len(shards) == 3
    assert all(len(s) == 65 for s in shards)  # 1 x-coord byte + 64 share bytes
    # The ceremony aborts unless EVERY 2-of-3 combo reconstructs (INV-MBR-1).
    assert verify_all_combinations(key, shards, t=2) is True


def test_shamir_every_pair_reconstructs():
    key = os.urandom(64)
    shards = split_secret(key, n=3, t=2)
    for combo in ((0, 1), (0, 2), (1, 2)):
        assert combine_shares([shards[i] for i in combo]) == key


# ── shard3_tx ≠ genesis_tx, memo formats (G1 / INV-MBR-3) ─────────────────────

def test_shard3_memo_pipe_format_roundtrips():
    enc = encrypt_shard3(bytes(range(65)), "9xQkTestPubkey1111111111111111111111111111")
    memo = gc.build_shard3_memo(enc)
    assert memo.startswith("TITAN|SHARD3|v=2.0|data=")
    assert len(memo) <= 566
    payload = memo.split("data=", 1)[1]
    assert base64.b64decode(payload) == enc  # discover_genesis decodes this


def test_three_memos_have_distinct_prefixes():
    enc = encrypt_shard3(bytes(range(65)), "9xQkTestPubkey1111111111111111111111111111")
    shard3 = gc.build_shard3_memo(enc)
    art = gc.build_art_memo("9xQkPUBKEYabcdef0000", "deadbeefcafe0000")
    genesis = gc.build_genesis_memo("9xQkPUBKEYabcdef0000", "MakerABCD", "NFTaddr12345678", "deadbeefcafe0000")
    # INV-MBR-3: the Shard-3 anchor and the identity memo are NEVER the same TX —
    # which starts with the prefix being distinct so they can't collide.
    assert shard3.startswith("TITAN|SHARD3")
    assert art.startswith("TITAN:ART")
    assert genesis.startswith("TITAN:GENESIS")
    assert genesis != art
    assert genesis.split("|")[0] != shard3.split("|")[0]


def test_oversize_shard3_falls_back_to_hash_memo():
    huge = os.urandom(600)
    memo = gc.build_shard3_memo(huge)
    assert memo.startswith("TITAN_GENESIS_SHARD3_HASH:")
    assert len(memo) <= 566


# ── Funding gate (G2 / INV-GEN-FUND) ──────────────────────────────────────────

def test_funding_satisfied_threshold():
    assert gc.funding_satisfied(0.0, 0.05) is False
    assert gc.funding_satisfied(0.049, 0.05) is False
    assert gc.funding_satisfied(0.05, 0.05) is True
    assert gc.funding_satisfied(1.5, 0.05) is True


def test_wait_for_funding_blocks_then_proceeds():
    balances = iter([0.0, 0.0, 0.06])
    final = gc.wait_for_funding(
        "9xQkTest", "devnet", 0.05,
        balance_fn=lambda: next(balances),
        interactive=False, poll_interval=0)
    assert final == 0.06


def test_wait_for_funding_raises_if_never_funded():
    with pytest.raises(TimeoutError):
        gc.wait_for_funding(
            "9xQkTest", "devnet", 0.05,
            balance_fn=lambda: 0.0,
            interactive=False, poll_interval=0, max_polls=3)


# ── Network branch (G... / INV-GEN-REUSE) ─────────────────────────────────────

def test_resolve_network_config_devnet():
    cfg = gc.resolve_network_config("devnet")
    assert cfg["solana_network"] == "devnet"
    assert cfg["public_rpc_urls"] == [gc.DEVNET_RPC]
    assert cfg["premium_rpc_url"] == ""


def test_resolve_network_config_mainnet_defaults():
    cfg = gc.resolve_network_config("mainnet")
    # mainnet keeps config.toml's cluster (or defaults to mainnet-beta), never devnet.
    assert cfg.get("solana_network", "mainnet-beta") != "devnet"


def test_primary_rpc_url_branches():
    assert gc.primary_rpc_url("devnet", {}) == gc.DEVNET_RPC
    assert gc.primary_rpc_url("mainnet", {"premium_rpc_url": "https://x"}) == "https://x"
    assert gc.primary_rpc_url("mainnet", {"public_rpc_urls": ["https://pub"]}) == "https://pub"
    assert gc.primary_rpc_url("mainnet", {}) == gc.DEFAULT_MAINNET_RPC


def test_estimate_funding_sol():
    assert gc.estimate_funding_sol("devnet") == 0.05
    assert gc.estimate_funding_sol("mainnet") == 0.10


# ── Record completeness (G1 / acceptance) ─────────────────────────────────────

def test_genesis_record_schema_complete_and_separated():
    rec = gc.build_genesis_record(
        titan_pubkey="9xQk", titan_id="T4", name="T4-test", maker="MakerABCD",
        network="devnet", shard3_tx="SHARD3sig", genesis_tx="GENESISsig",
        art_tx="ARTsig", vault_pda="VAULTpda", vault_tx="VAULTtx",
        nft_address="NFTaddr", nft_tx="NFTtx", nft_uri="store://x",
        shard2_hex="aa", shard3_encrypted_hex="bb", constitution_sha="csha",
        birth_dna_sha="dsha", genesis_art_hash="arthash",
        ceremony_timestamp=123, version="3.0")
    # Every canonical pointer present (T1's record shape).
    assert set(rec.keys()) == set(gc.GENESIS_RECORD_KEYS)
    # INV-MBR-3: shard3_tx and genesis_tx are DISTINCT fields with distinct values.
    assert rec["shard3_tx"] == "SHARD3sig"
    assert rec["genesis_tx"] == "GENESISsig"
    assert rec["shard3_tx"] != rec["genesis_tx"]
    assert rec["ceremony_timestamp"] == 123


def test_genesis_record_defaults_fill_missing_keys():
    rec = gc.build_genesis_record(titan_pubkey="9xQk")
    assert set(rec.keys()) == set(gc.GENESIS_RECORD_KEYS)
    assert rec["version"] == "3.0"
    assert rec["ceremony_timestamp"] == 0
    assert rec["nft_address"] == ""
