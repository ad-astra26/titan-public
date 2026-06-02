"""5J-5 backfill — offline pure-logic tests (no network, no mainnet writes)."""
import sys
from pathlib import Path

# APPEND (never insert-0) so `import titan_hcl` still resolves to the PACKAGE, not
# scripts/titan_hcl.py (the agent entry) — same hazard the engine guards against.
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from backup_chain_backfill_v3 import (  # noqa: E402
    normalize_record, build_chain_plan, existing_arcs, GENESIS_ANCHOR_TX,
    load_manifest_events,
)

_HASH = "f518812982628dd71c9f30aca14e2c2a8ef6b4dddad04387718ee672ba2b90c0"
_HASH2 = "d841b3444ec9d5aa1a2ae2b54dd2c68cece909a285eb875aa97cebcf02fe0e68"
_HASH3 = "0d00d59ba8dc495796992d0155f4667130b0d8e3da0836447b84e94dab378e33"


def test_normalize_legacy_record_builds_component():
    rec = {"_file": "personality_1.json", "uploaded_at": 1777709224,
           "arweave_tx": "ARWEAVE_TX_1", "archive_hash": _HASH,
           "backup_type": "personality"}
    e = normalize_record(rec)
    assert "_skip_reason" not in e
    assert e["components"] == [{"tier": "PT", "tx_id": "ARWEAVE_TX_1", "arc": _HASH}]
    assert e["ts"] == 1777709224


def test_normalize_legacy_timechain_maps_to_TC():
    rec = {"_file": "tc.json", "uploaded_at": 100, "arweave_tx": "TX",
           "archive_hash": _HASH, "backup_type": "timechain"}
    assert normalize_record(rec)["components"][0]["tier"] == "TC"


def test_normalize_unified_v2_event_is_skipped_not_fabricated():
    rec = {"_file": "ev.json", "uploaded_at": 200, "event_id": "abc",
           "personality_tx": "PT_TX", "timechain_tx": "TC_TX",
           "event_merkle_root": _HASH}  # no per-component sha256
    e = normalize_record(rec)
    assert "_skip_reason" in e and "per-component sha256" in e["_skip_reason"]


def test_normalize_missing_ts_skipped():
    assert "_skip_reason" in normalize_record({"_file": "x", "archive_hash": _HASH,
                                               "arweave_tx": "T"})


def test_build_chain_plan_sorts_oldest_first_and_separates_skips():
    recs = [
        {"_file": "b", "uploaded_at": 300, "arweave_tx": "T2", "archive_hash": _HASH,
         "backup_type": "personality"},
        {"_file": "a", "uploaded_at": 100, "arweave_tx": "T1", "archive_hash": _HASH,
         "backup_type": "personality"},
        {"_file": "skip", "uploaded_at": 200, "personality_tx": "P",
         "event_merkle_root": _HASH},  # unified-v2 → skipped
    ]
    events, skipped = build_chain_plan(recs)
    assert [e["ts"] for e in events] == [100, 300]   # oldest first
    assert len(skipped) == 1 and skipped[0]["_file"] == "skip"


def test_existing_arcs_parses_v3_memo_for_idempotency():
    from titan_hcl.logic.backup_memo_v3 import build_v3_memo
    memo = build_v3_memo(event_id="e1", ts=100, event_type="incremental", tier="PT",
                         archive_hash=_HASH, merkle_root=_HASH, arweave_tx="ARW",
                         mode="B", iv_b64="q83vEjRWeJC2yJ3k")  # Mode B → plaintext url + iv
    arcs = existing_arcs([memo, "not a v3 memo", "v=2;legacy"])
    assert _HASH[:32] in arcs  # arc fragment is the first 32 chars (HASH_FRAGMENT_LEN)


def test_genesis_anchor_constant():
    assert GENESIS_ANCHOR_TX == "5StBnZIfus1mbuYJ520Ct2a4OomNUuOm_3VGZmeNGQw"


# ── --from-manifest loader (the v=2→v=3 backfill source) ────────────────────

def _write_manifest(tmp_path, events):
    import json
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"titan_id": "T1", "events": events}))
    return str(p)


def test_load_manifest_events_baseline_and_increment(tmp_path):
    """Baseline → typ baseline; per-component arc = manifest merkle_root; mrkl is the
    recomputed event merkle (matches the producer + restore verifier exactly)."""
    from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root
    man = [
        {"event_id": "ev_base", "ts_unix": 100, "type": "baseline",
         "personality": {"tx_id": "PT_TX", "merkle_root": _HASH3},
         "timechain": {"tx_id": "TC_TX", "merkle_root": _HASH2}, "soul": None},
        {"event_id": "ev_inc", "ts_unix": 200, "type": "incremental",
         "personality": {"tx_id": "PT2", "merkle_root": _HASH},
         "timechain": {"tx_id": "TC2", "merkle_root": _HASH2}, "soul": None},
    ]
    events, skipped = load_manifest_events(_write_manifest(tmp_path, man))
    assert skipped == []
    assert [e["event_type"] for e in events] == ["baseline", "incremental"]
    base = events[0]
    assert base["components"] == [
        {"tier": "PT", "tx_id": "PT_TX", "arc": _HASH3},
        {"tier": "TC", "tx_id": "TC_TX", "arc": _HASH2}]
    # mrkl is the FULL recomputed root; build_v3_memo truncates [:32] at post time.
    assert base["mrkl"] == compute_event_merkle_root(_HASH3, _HASH2, None)


def test_load_manifest_events_soul_component_included(tmp_path):
    man = [{"event_id": "ev_soul", "ts_unix": 50, "type": "incremental",
            "personality": {"tx_id": "P", "merkle_root": _HASH},
            "timechain": {"tx_id": "T", "merkle_root": _HASH2},
            "soul": {"tx_id": "S", "merkle_root": _HASH3}}]
    events, _ = load_manifest_events(_write_manifest(tmp_path, man))
    tiers = [c["tier"] for c in events[0]["components"]]
    assert tiers == ["PT", "TC", "SL"]


def test_load_manifest_events_skips_missing_component_sha_not_fabricated(tmp_path):
    man = [{"event_id": "bad", "ts_unix": 10, "type": "incremental",
            "personality": {"tx_id": "P"},  # no merkle_root → cannot build a faithful memo
            "timechain": {"tx_id": "T", "merkle_root": _HASH2}, "soul": None}]
    events, skipped = load_manifest_events(_write_manifest(tmp_path, man))
    assert events == [] and len(skipped) == 1
    assert "merkle_root" in skipped[0]["_skip_reason"]


def test_load_manifest_events_sorts_oldest_first(tmp_path):
    man = [
        {"event_id": "late", "ts_unix": 900, "type": "incremental",
         "personality": {"tx_id": "P", "merkle_root": _HASH},
         "timechain": {"tx_id": "T", "merkle_root": _HASH2}, "soul": None},
        {"event_id": "early", "ts_unix": 100, "type": "baseline",
         "personality": {"tx_id": "P", "merkle_root": _HASH3},
         "timechain": {"tx_id": "T", "merkle_root": _HASH2}, "soul": None},
    ]
    events, _ = load_manifest_events(_write_manifest(tmp_path, man))
    assert [e["event_id"] for e in events] == ["early", "late"]
