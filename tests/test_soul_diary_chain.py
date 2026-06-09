"""Tests for the Soul-Diary hash-chain ledger (RFP §1.0 ⑤ / INV-SD-10).

Verifies the tamper-evident chain: entry_hash = sha256(text),
cumulative_hash = sha256(prev ‖ entry_hash); chaining across days; the G7
verify check (intact + breaks-on-tamper); idempotency; ref updates.
"""
import hashlib
import json
import os

from titan_hcl.core import soul_diary_chain as sdc


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_append_chains_correctly(tmp_path):
    p = str(tmp_path / "chain.json")
    r1 = sdc.append_entry("2026-06-08", "day one entry", path=p)
    r2 = sdc.append_entry("2026-06-09", "day two entry", path=p)

    assert r1["entry_hash"] == _sha("day one entry")
    assert r1["prev_cumulative"] == ""
    assert r1["cumulative_hash"] == _sha("" + r1["entry_hash"])

    assert r2["entry_hash"] == _sha("day two entry")
    assert r2["prev_cumulative"] == r1["cumulative_hash"]
    assert r2["cumulative_hash"] == _sha(r1["cumulative_hash"] + r2["entry_hash"])

    assert sdc.last_cumulative(path=p) == r2["cumulative_hash"]


def test_verify_chain_intact_and_tamper(tmp_path):
    p = str(tmp_path / "chain.json")
    sdc.append_entry("2026-06-08", "alpha", path=p)
    sdc.append_entry("2026-06-09", "beta", path=p)
    sdc.append_entry("2026-06-10", "gamma", path=p)
    assert sdc.verify_chain(path=p) is True

    # Tamper: alter a past row's entry_hash → the chain must break.
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["entries"][1]["entry_hash"] = _sha("TAMPERED")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    assert sdc.verify_chain(path=p) is False


def test_idempotent_same_date(tmp_path):
    p = str(tmp_path / "chain.json")
    r1 = sdc.append_entry("2026-06-09", "first text", path=p)
    r2 = sdc.append_entry("2026-06-09", "DIFFERENT text same day", path=p)
    assert r2 == r1  # not re-appended; original returned
    assert len(sdc.load_chain(path=p)) == 1


def test_empty_and_missing(tmp_path):
    p = str(tmp_path / "nope.json")
    assert sdc.load_chain(path=p) == []
    assert sdc.last_cumulative(path=p) == ""
    assert sdc.verify_chain(path=p) is True  # vacuously intact


def test_update_refs(tmp_path):
    p = str(tmp_path / "chain.json")
    sdc.append_entry("2026-06-09", "entry", path=p)
    assert sdc.update_refs("2026-06-09", nft_addr="ASSET123",
                           timechain_block="blk_abc", arweave_uri="ar://xyz",
                           path=p) is True
    row = sdc.load_chain(path=p)[0]
    assert row["nft_addr"] == "ASSET123"
    assert row["timechain_block"] == "blk_abc"
    assert row["arweave_uri"] == "ar://xyz"
    # refs do not change the hashes → chain still verifies
    assert sdc.verify_chain(path=p) is True
    assert sdc.update_refs("2026-01-01", nft_addr="x", path=p) is False  # no such date


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "chain.json")
    sdc.append_entry("2026-06-09", "entry", path=p)
    assert os.path.exists(p)
    assert not os.path.exists(p + ".tmp")
