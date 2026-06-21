"""devnet orphan-sweep retention — `RebirthBackup._sweep_orphan_devnet_bundles`.

Root-caused 2026-06-21: arweave_devnet bundles referenced by no manifest accumulate
unbounded on the devnet box (156/Titan = 96G, filled the disk). The sweep deletes
ONLY unreachable orphans, devnet-only. These tests pin the FOUR mainnet guards (any
one must make it a no-op) + the happy path + the empty-keep-set abort.
"""
import json
import os

import pytest

from titan_hcl.logic.backup import RebirthBackup


def _sweeper():
    """Bare instance — the method only uses os.path (cwd-relative) + _rebase_params,
    which itself only checks for data/genesis_record.json. No heavy __init__."""
    return object.__new__(RebirthBackup)


def _make_cache(root, referenced, orphans, manifest_refs=True):
    """Create data/arweave_devnet bundles + a manifest referencing `referenced`."""
    cache = root / "data" / "arweave_devnet"
    cache.mkdir(parents=True)
    for tx in referenced + orphans:
        (cache / f"{tx}.data").write_bytes(b"x" * 1024)
        (cache / f"{tx}.tags.json").write_text("{}")
    if manifest_refs:
        tc = root / "data" / "timechain"
        tc.mkdir(parents=True)
        (tc / "arweave_manifest.json").write_text(json.dumps(
            {"latest": {"tx_id": referenced[0]} if referenced else {},
             "entries": [{"tx_id": t} for t in referenced]}))


def test_happy_path_deletes_orphans_keeps_referenced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = ["devnet_aaa1", "devnet_bbb2"]
    orph = ["devnet_ccc3", "devnet_ddd4", "devnet_eee5"]
    _make_cache(tmp_path, ref, orph)
    n = _sweeper()._sweep_orphan_devnet_bundles()
    assert n == 3
    cache = tmp_path / "data" / "arweave_devnet"
    for t in ref:
        assert (cache / f"{t}.data").exists()        # referenced KEPT
    for t in orph:
        assert not (cache / f"{t}.data").exists()     # orphan DELETED
        assert not (cache / f"{t}.tags.json").exists()


def test_guard1_mainnet_genesis_record_present_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_cache(tmp_path, ["devnet_keep"], ["devnet_orphan"])
    (tmp_path / "data" / "genesis_record.json").write_text("{}")   # ← mainnet marker
    n = _sweeper()._sweep_orphan_devnet_bundles()
    assert n == 0
    # NOTHING deleted on mainnet — even the orphan survives.
    assert (tmp_path / "data" / "arweave_devnet" / "devnet_orphan.data").exists()


def test_guard3_no_cache_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    assert _sweeper()._sweep_orphan_devnet_bundles() == 0


def test_guard4_empty_keepset_aborts_without_deleting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # bundles exist but NO json references any devnet_ tx → keep-set empty → ABORT.
    _make_cache(tmp_path, [], ["devnet_orphan1", "devnet_orphan2"], manifest_refs=False)
    n = _sweeper()._sweep_orphan_devnet_bundles()
    assert n == 0
    # defensive: refuse to delete blind — orphans survive a failed scan.
    assert (tmp_path / "data" / "arweave_devnet" / "devnet_orphan1.data").exists()


def test_keepset_scans_nested_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = ["devnet_abc123"]
    orph = ["devnet_dead8f"]
    _make_cache(tmp_path, ref, orph, manifest_refs=False)
    # reference lives in a DEEP json (soul_diary style) — recursive scan must find it
    deep = tmp_path / "data" / "sub" / "dir"
    deep.mkdir(parents=True)
    (deep / "soul_diary_chain.json").write_text(json.dumps({"x": ref[0]}))
    n = _sweeper()._sweep_orphan_devnet_bundles()
    assert n == 1
    assert (tmp_path / "data" / "arweave_devnet" / f"{ref[0]}.data").exists()
    assert not (tmp_path / "data" / "arweave_devnet" / f"{orph[0]}.data").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
