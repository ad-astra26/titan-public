"""5J-6 tests for the sovereign v=3 backup-chain memo (5J-1 pure logic).

Acceptance gate 7 (RFP §3B.3): v=3 round-trip deterministic; Mode A URL
round-trip with the HKDF-derived key recovers the exact tx_id over 1000
iterations with no nonce collision; Mode B preserves the URL byte-identical;
genesis anchor parse; mixed-mode chain decodable; on-chain event type (typ=B|I);
malformed rejection.

Pure-logic — no Solana, no network, no disk. Run:
    python -m pytest tests/test_backup_memo_v3.py -p no:anchorpy -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan_hcl.logic import backup_memo_v3 as v3  # noqa: E402

# Realistic 43-char Arweave/Irys tx ids (from T1's live manifest).
TX_PERSONALITY = "EFMlEVvRPKrkW3zBIFdox5ClKgkETRzdNx10n9oZTUc"
TX_TIMECHAIN = "PQXjo0VP41y3MvxQjq3kYq-SB7pVS3ytRY0m2Dxl3GQ"
# Day-1 eternal genesis anchor.
TX_DAY1 = "5StBnZIfus1mbuYJ520Ct2a4OomNUuOm_3VGZmeNGQw"

ARC = "0d00d59ba8dc495796992d0155f4667130b0d8e3da0836447b84e94dab378e33"
MRKL = "d841b3444ec9d5aa1a2ae2b54dd2c68cece909a285eb875aa97cebcf02fe0e68"
EVENT_ID = "560d18cfb8634f5180cb12e635cdbe21"
SEED = bytes(range(32))  # deterministic 32-byte Ed25519 seed


# ── key derivation ───────────────────────────────────────────────────────────


def test_url_key_deterministic_and_32_bytes():
    k1 = v3.derive_backup_url_key(SEED)
    k2 = v3.derive_backup_url_key(SEED + b"\x00" * 32)  # 64-byte keypair, same seed prefix
    assert isinstance(k1, bytes) and len(k1) == 32
    assert k1 == k2, "same seed prefix → same url key (recovery property)"


def test_url_key_differs_per_seed():
    assert v3.derive_backup_url_key(SEED) != v3.derive_backup_url_key(bytes(range(1, 33)))


def test_url_key_rejects_short_seed():
    with pytest.raises(ValueError):
        v3.derive_backup_url_key(b"too-short")


# ── tier mapping ─────────────────────────────────────────────────────────────


def test_tier_component_roundtrip():
    for comp, tier in (("personality", "PT"), ("timechain", "TC"), ("soul", "SL")):
        assert v3.component_to_tier(comp) == tier
        assert v3.tier_to_component(tier) == comp
    with pytest.raises(ValueError):
        v3.component_to_tier("bogus")
    with pytest.raises(ValueError):
        v3.tier_to_component("XX")


# ── Mode A: encrypt/decrypt URL round-trip + no nonce reuse ──────────────────


def test_mode_a_url_roundtrip_and_no_nonce_collision():
    key = v3.derive_backup_url_key(SEED)
    blobs = set()
    for _ in range(1000):
        blob = v3.encrypt_url_mode_a(TX_PERSONALITY, key)
        assert blob not in blobs, "nonce reuse → identical blob detected"
        blobs.add(blob)
        assert v3.decrypt_url_mode_a(blob, key) == TX_PERSONALITY
    assert len(blobs) == 1000


def test_mode_a_wrong_key_fails():
    good = v3.derive_backup_url_key(SEED)
    bad = v3.derive_backup_url_key(bytes(range(1, 33)))
    blob = v3.encrypt_url_mode_a(TX_TIMECHAIN, good)
    with pytest.raises(Exception):  # cryptography InvalidTag
        v3.decrypt_url_mode_a(blob, bad)


# ── full memo build/parse round-trip ─────────────────────────────────────────


def test_v3_memo_roundtrip_mode_a_baseline():
    key = v3.derive_backup_url_key(SEED)
    memo = v3.build_v3_memo(
        event_id=EVENT_ID, ts=1748000000, event_type="baseline", tier="PT",
        archive_hash=ARC, merkle_root=MRKL, arweave_tx=TX_PERSONALITY,
        mode="A", prev_sig=None, url_key=key,
    )
    assert memo.startswith("v=3;evt=" + EVENT_ID + ";")
    assert ";typ=B;" in memo
    assert ";prev=genesis" in memo  # prev_sig None → genesis
    parsed = v3.parse_v3_memo(memo)
    assert parsed is not None
    assert parsed["event_id"] == EVENT_ID
    assert parsed["ts"] == 1748000000
    assert parsed["type"] == "baseline"
    assert parsed["tier"] == "PT" and parsed["component"] == "personality"
    assert parsed["arc"] == ARC[:32] and parsed["mrkl"] == MRKL[:32]
    assert parsed["mode"] == "A"
    assert parsed["prev"] == "genesis"
    assert v3.read_url(parsed, key) == TX_PERSONALITY


def test_v3_memo_roundtrip_mode_b_incremental_byte_identical_url():
    memo = v3.build_v3_memo(
        event_id=EVENT_ID, ts=1748000001, event_type="incremental", tier="TC",
        archive_hash=ARC, merkle_root=MRKL, arweave_tx=TX_TIMECHAIN,
        mode="B", prev_sig="5xPrevSig9aBcDeFgHiJkLmNoP",
    )
    assert ";typ=I;" in memo
    parsed = v3.parse_v3_memo(memo)
    assert parsed["type"] == "incremental"
    assert parsed["mode"] == "B"
    assert parsed["url"] == "raw:" + TX_TIMECHAIN
    # Mode B needs no key to resolve, and is byte-identical.
    assert v3.read_url(parsed) == TX_TIMECHAIN


def test_prev_sig_truncated_to_16():
    memo = v3.build_v3_memo(
        event_id=EVENT_ID, ts=1, event_type="baseline", tier="SL",
        archive_hash=ARC, merkle_root=MRKL, arweave_tx=TX_DAY1, mode="B",
        prev_sig="A" * 88,  # full base58 sig length
    )
    parsed = v3.parse_v3_memo(memo)
    assert parsed["prev"] == "A" * 16


def test_genesis_day1_anchor_parses():
    memo = v3.build_v3_memo(
        event_id="genesis-day1", ts=1744000140, event_type="baseline", tier="PT",
        archive_hash=ARC, merkle_root=MRKL, arweave_tx=TX_DAY1,
        mode="B", prev_sig=None,
    )
    parsed = v3.parse_v3_memo(memo)
    assert parsed["prev"] == "genesis"
    assert parsed["type"] == "baseline"
    assert v3.read_url(parsed) == TX_DAY1


def test_mixed_mode_chain_decodable():
    key = v3.derive_backup_url_key(SEED)
    a = v3.build_v3_memo(event_id="e1", ts=1, event_type="baseline", tier="PT",
                         archive_hash=ARC, merkle_root=MRKL,
                         arweave_tx=TX_PERSONALITY, mode="A",
                         prev_sig=None, url_key=key)
    b = v3.build_v3_memo(event_id="e2", ts=2, event_type="incremental", tier="TC",
                         archive_hash=ARC, merkle_root=MRKL,
                         arweave_tx=TX_TIMECHAIN, mode="B",
                         prev_sig="2bQ9xKpAemvntdef")  # valid base58 (no O/0/I/l)
    pa, pb = v3.parse_v3_memo(a), v3.parse_v3_memo(b)
    assert pa["mode"] == "A" and pb["mode"] == "B"
    assert pa["type"] == "baseline" and pb["type"] == "incremental"
    assert v3.read_url(pa, key) == TX_PERSONALITY
    assert v3.read_url(pb) == TX_TIMECHAIN


# ── validation + malformed rejection ─────────────────────────────────────────


def test_build_rejects_bad_inputs():
    key = v3.derive_backup_url_key(SEED)
    base = dict(event_id="e", ts=1, event_type="baseline", tier="PT",
                archive_hash=ARC, merkle_root=MRKL, arweave_tx=TX_DAY1)
    with pytest.raises(ValueError):  # bad event_type
        v3.build_v3_memo(**{**base, "event_type": "ZZZ", "mode": "B"})
    with pytest.raises(ValueError):  # bad tier
        v3.build_v3_memo(**{**base, "tier": "ZZ", "mode": "B"})
    with pytest.raises(ValueError):  # bad mode
        v3.build_v3_memo(**{**base, "mode": "C"})
    with pytest.raises(ValueError):  # mode A without key
        v3.build_v3_memo(**{**base, "mode": "A"})
    with pytest.raises(ValueError):  # short hash
        v3.build_v3_memo(**{**base, "archive_hash": "abc", "mode": "B"})
    with pytest.raises(ValueError):  # event_id with delimiter
        v3.build_v3_memo(**{**base, "event_id": "e;x", "mode": "B"})


def test_parse_rejects_malformed():
    assert v3.parse_v3_memo("v=2;event_id=x;root=" + "a" * 32 + ";prev=genesis") is None
    assert v3.parse_v3_memo("not a memo") is None
    assert v3.parse_v3_memo("") is None
    assert v3.parse_v3_memo(None) is None  # type: ignore[arg-type]
    # v=3 missing typ= (old shape) must be rejected by the strict parser
    assert v3.parse_v3_memo(
        "v=3;evt=e;ts=1;tier=PT;arc=" + "a" * 32 + ";mrkl=" + "b" * 32
        + ";url=raw:x;prev=genesis") is None


def test_memo_within_solana_cap():
    key = v3.derive_backup_url_key(SEED)
    memo = v3.build_v3_memo(event_id="f" * 36, ts=1748000000, event_type="baseline",
                            tier="PT", archive_hash=ARC, merkle_root=MRKL,
                            arweave_tx=TX_PERSONALITY, mode="A",
                            prev_sig="z" * 16, url_key=key)
    assert len(memo.encode("utf-8")) <= v3.V3_MEMO_MAX_BYTES
