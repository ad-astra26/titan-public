"""
tests/test_timechain_backup.py — TimeChainBackup restore + manifest verification.

The TimeChain backup WRITE path (create_snapshot_tarball / snapshot_to_arweave
/ BackupCascade) was removed in the backup write-path reconciliation
(RFP_backup_redesign_spine, 2026-06-10): TimeChain now ships as the `timechain`
tier of the unified daily incremental via the ChainProvider. TimeChainBackup is
now the read/verify half — restore + genesis-verify + manifest. These tests
guard that the write path stays retired and the restore/manifest surface stays
intact (used by scripts/resurrect_timechain.py).

Run:
    source test_env/bin/activate
    python -m pytest tests/test_timechain_backup.py -v -p no:anchorpy
"""
from titan_hcl.logic.timechain_backup import TimeChainBackup


# ── Write-path retirement verified (this reconciliation's regression guard) ──

def test_write_path_retired():
    """The legacy TimeChain backup WRITE path must be gone — it was unified into
    the BackupOrchestrator/BackupWorker (timechain tier via ChainProvider)."""
    backup = TimeChainBackup(data_dir="data/timechain", titan_id="T1")
    for gone in ("snapshot_to_arweave", "create_snapshot_tarball",
                 "_record_arweave_anchor",
                 "snapshot_to_arweave_json", "create_snapshot"):
        assert not hasattr(backup, gone), (
            f"{gone} should be retired — TimeChain backup writes live in the "
            "unified BackupOrchestrator, not here")


def test_restore_surface_preserved_for_resurrection():
    """The restore/verify surface MUST survive — scripts/resurrect_timechain.py
    (the resurrection path) + the /v4/timechain/backup-status endpoint use it."""
    backup = TimeChainBackup(data_dir="data/timechain", titan_id="T1")
    for kept in ("restore_from_arweave", "_restore_from_tarball",
                 "_restore_from_json", "verify_genesis_integrity",
                 "get_backup_status", "get_latest_arweave_tx"):
        assert hasattr(backup, kept), (
            f"{kept} must be preserved for resurrection / status")


# ── Per-Titan manifest path (rFP Phase 1 BUG-4 fix) ──────────────────

def test_legacy_MANIFEST_PATH_constant_unchanged():
    """Module-level MANIFEST_PATH stays for legacy callers (resurrect_timechain.py)
    and test monkeypatching. Per-Titan logic is in _manifest_path()."""
    from titan_hcl.logic.timechain_backup import MANIFEST_PATH
    assert MANIFEST_PATH == "data/timechain/arweave_manifest.json"


def test_per_titan_manifest_path_T1(tmp_path, monkeypatch):
    """_manifest_path() returns arweave_manifest_T1.json when no monkeypatch."""
    monkeypatch.chdir(tmp_path)
    from titan_hcl.logic.timechain_backup import _manifest_path
    assert _manifest_path("T1").endswith("arweave_manifest_T1.json")
    assert _manifest_path("T2").endswith("arweave_manifest_T2.json")
    assert _manifest_path("T3").endswith("arweave_manifest_T3.json")


def test_per_titan_manifest_migration_from_legacy(tmp_path, monkeypatch):
    """If legacy manifest exists and per-Titan does not, first call migrates."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "timechain").mkdir(parents=True)
    legacy = tmp_path / "data" / "timechain" / "arweave_manifest.json"
    legacy.write_text('{"titan_id": "T1", "snapshots": [{"tx_id": "HIST"}], "total_snapshots": 1}')

    from titan_hcl.logic.timechain_backup import _manifest_path
    new_path = _manifest_path("T1")
    # Migration side-effect: new per-Titan file now exists with same content
    import os
    assert os.path.exists(new_path)
    assert os.path.exists(str(legacy) + ".bak_pre_per_titan_20260413")
    import json
    assert json.load(open(new_path))["snapshots"][0]["tx_id"] == "HIST"
