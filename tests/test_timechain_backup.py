"""
tests/test_timechain_backup.py — TimeChainBackup tarball path verification.

Per rFP_backup_worker Phase 0 (2026-04-13): JSON write path retired, tarball
path is the single unified upload path. These tests verify the tarball
creation + restore roundtrip + upload-with-mock flow.

Run:
    source test_env/bin/activate
    python -m pytest tests/test_timechain_backup.py -v -p no:anchorpy
"""
import asyncio
import os
from pathlib import Path

import pytest

from titan_plugin.logic.timechain_backup import TimeChainBackup


class _MockArweave:
    """Minimal ArweaveStore mock: just records calls + returns a tx id."""

    def __init__(self, tx_id: str = "MOCK_TX_12345", fail: bool = False):
        self._tx_id = tx_id
        self._fail = fail
        self.calls: list[dict] = []

    async def upload_file_bytes(self, data: bytes, tags: dict, content_type: str):
        self.calls.append({
            "method": "upload_file_bytes",
            "size": len(data), "tags": tags, "content_type": content_type,
        })
        return None if self._fail else self._tx_id

    async def upload_json(self, snapshot: dict, tags: dict):
        self.calls.append({"method": "upload_json", "tags": tags})
        return None if self._fail else self._tx_id


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def tc_dir(tmp_path, monkeypatch):
    """Create a minimal TimeChain data dir + isolated manifest path.

    CRITICAL: MANIFEST_PATH is module-level hardcoded (BUG-4 per rFP audit).
    Without monkeypatch, tests would write to production manifest. Phase 1 of
    rFP fixes this by making MANIFEST_PATH per-Titan; until then, tests must
    monkeypatch MANIFEST_PATH + CWD to avoid pollution.
    """
    import titan_plugin.logic.timechain_backup as tcb_mod
    isolated_manifest = str(tmp_path / "arweave_manifest_test.json")
    monkeypatch.setattr(tcb_mod, "MANIFEST_PATH", isolated_manifest)
    monkeypatch.chdir(tmp_path)

    from titan_plugin.logic.timechain import TimeChain
    tc = TimeChain(data_dir=str(tmp_path), titan_id="T1")
    if not tc.has_genesis:
        tc.create_genesis({"prime_directives": ["Sovereign Integrity"], "titan_id": "T1"})
    return tmp_path


# ── Tarball creation ─────────────────────────────────────────────────

def test_create_snapshot_tarball_has_metadata(tc_dir):
    backup = TimeChainBackup(data_dir=str(tc_dir), titan_id="T1")
    tarball, metadata = backup.create_snapshot_tarball()
    assert isinstance(tarball, bytes)
    assert len(tarball) > 0
    # Required metadata fields per rFP
    for key in ("version", "titan_id", "genesis_hash", "merkle_root",
                "total_blocks", "tarball_sha256", "tarball_size_bytes",
                "compression", "timestamp"):
        assert key in metadata, f"missing metadata key: {key}"
    assert metadata["titan_id"] == "T1"
    assert metadata["compression"] in ("zstd-19", "gzip-9")


def test_create_snapshot_tarball_sha256_matches(tc_dir):
    import hashlib
    backup = TimeChainBackup(data_dir=str(tc_dir), titan_id="T1")
    tarball, metadata = backup.create_snapshot_tarball()
    assert hashlib.sha256(tarball).hexdigest() == metadata["tarball_sha256"]


def test_create_snapshot_tarball_no_genesis_returns_empty(tmp_path):
    """Tarball create on empty dir should return (b'', {}) per current contract."""
    backup = TimeChainBackup(data_dir=str(tmp_path), titan_id="T1")
    tarball, metadata = backup.create_snapshot_tarball()
    assert tarball == b""
    assert metadata == {}


# ── Upload with mock ─────────────────────────────────────────────────

def test_snapshot_to_arweave_records_manifest(tc_dir):
    mock = _MockArweave(tx_id="TARBALL_TX_123")
    backup = TimeChainBackup(data_dir=str(tc_dir), titan_id="T1", arweave_store=mock)
    tx = asyncio.run(backup.snapshot_to_arweave())
    assert tx == "TARBALL_TX_123"
    assert len(mock.calls) == 1
    # Should use tarball path — upload_file_bytes, not upload_json
    assert mock.calls[0]["method"] == "upload_file_bytes"
    # Manifest should have one entry
    status = backup.get_backup_status()
    assert status["total_snapshots"] == 1
    assert status["latest_tx"] == "TARBALL_TX_123"


def test_snapshot_to_arweave_no_store_returns_none(tc_dir):
    backup = TimeChainBackup(data_dir=str(tc_dir), titan_id="T1", arweave_store=None)
    tx = asyncio.run(backup.snapshot_to_arweave())
    assert tx is None


def test_snapshot_to_arweave_failure_no_manifest_entry(tc_dir):
    mock = _MockArweave(fail=True)
    backup = TimeChainBackup(data_dir=str(tc_dir), titan_id="T1", arweave_store=mock)
    tx = asyncio.run(backup.snapshot_to_arweave())
    assert tx is None
    status = backup.get_backup_status()
    assert status["total_snapshots"] == 0


# ── JSON path retirement verified ────────────────────────────────────

def test_snapshot_to_arweave_json_no_longer_exists():
    """Phase 0 retirement: JSON write path must be gone."""
    backup = TimeChainBackup(data_dir="data/timechain", titan_id="T1")
    assert not hasattr(backup, "snapshot_to_arweave_json"), \
        "snapshot_to_arweave_json should be retired per rFP Phase 0"
    assert not hasattr(backup, "create_snapshot"), \
        "create_snapshot (JSON path) should be retired per rFP Phase 0"


def test_restore_from_json_preserved_for_backward_compat():
    """JSON RESTORE path kept for scripts/resurrect_timechain.py historical recovery."""
    backup = TimeChainBackup(data_dir="data/timechain", titan_id="T1")
    assert hasattr(backup, "_restore_from_json"), \
        "_restore_from_json must be preserved for historical devnet snapshot recovery"
    assert hasattr(backup, "_restore_from_tarball"), \
        "_restore_from_tarball is the primary restore path"


# ── Per-Titan manifest path (rFP Phase 1 BUG-4 fix) ──────────────────

def test_legacy_MANIFEST_PATH_constant_unchanged():
    """Module-level MANIFEST_PATH stays for legacy callers (resurrect_timechain.py)
    and test monkeypatching. Per-Titan logic is in _manifest_path()."""
    from titan_plugin.logic.timechain_backup import MANIFEST_PATH
    assert MANIFEST_PATH == "data/timechain/arweave_manifest.json"


def test_per_titan_manifest_path_T1(tmp_path, monkeypatch):
    """_manifest_path() returns arweave_manifest_T1.json when no monkeypatch."""
    monkeypatch.chdir(tmp_path)
    from titan_plugin.logic.timechain_backup import _manifest_path
    assert _manifest_path("T1").endswith("arweave_manifest_T1.json")
    assert _manifest_path("T2").endswith("arweave_manifest_T2.json")
    assert _manifest_path("T3").endswith("arweave_manifest_T3.json")


def test_per_titan_manifest_migration_from_legacy(tmp_path, monkeypatch):
    """If legacy manifest exists and per-Titan does not, first call migrates."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "timechain").mkdir(parents=True)
    legacy = tmp_path / "data" / "timechain" / "arweave_manifest.json"
    legacy.write_text('{"titan_id": "T1", "snapshots": [{"tx_id": "HIST"}], "total_snapshots": 1}')

    from titan_plugin.logic.timechain_backup import _manifest_path
    new_path = _manifest_path("T1")
    # Migration side-effect: new per-Titan file now exists with same content
    import os
    assert os.path.exists(new_path)
    assert os.path.exists(str(legacy) + ".bak_pre_per_titan_20260413")
    import json
    assert json.load(open(new_path))["snapshots"][0]["tx_id"] == "HIST"


def test_per_titan_manifest_no_cross_contamination(tmp_path, monkeypatch):
    """T1 writes must not appear in T2's manifest."""
    import titan_plugin.logic.timechain_backup as tcb_mod
    # Reset module-level override so real per-Titan logic is used
    monkeypatch.setattr(tcb_mod, "MANIFEST_PATH", "data/timechain/arweave_manifest.json")
    monkeypatch.chdir(tmp_path)
    from titan_plugin.logic.timechain import TimeChain
    # Create two Titans sharing a data_dir but different titan_id manifests
    tc = TimeChain(data_dir=str(tmp_path), titan_id="T1")
    tc.create_genesis({"prime_directives": ["Sovereign Integrity"], "titan_id": "T1"})

    # T1's backup
    mock1 = _MockArweave(tx_id="T1_TX_A")
    b1 = TimeChainBackup(data_dir=str(tmp_path), titan_id="T1", arweave_store=mock1)
    asyncio.run(b1.snapshot_to_arweave())

    # T2's backup (shared data_dir in test; real deploys have separate VPS)
    mock2 = _MockArweave(tx_id="T2_TX_B")
    b2 = TimeChainBackup(data_dir=str(tmp_path), titan_id="T2", arweave_store=mock2)
    asyncio.run(b2.snapshot_to_arweave())

    # T1's manifest should ONLY have T1_TX_A; T2's only T2_TX_B
    s1 = b1.get_backup_status()
    s2 = b2.get_backup_status()
    assert s1["latest_tx"] == "T1_TX_A"
    assert s2["latest_tx"] == "T2_TX_B"
    # Independent manifest files
    assert s1["manifest_path"] != s2["manifest_path"]
