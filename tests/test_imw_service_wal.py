"""Tests for IMW ServiceWAL — append, checkpoint, replay on boot."""
from pathlib import Path

from titan_plugin.persistence.service_wal import ServiceWAL


def test_append_and_iterate(tmp_path):
    wal = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    wal.append_request("r1", "INSERT INTO x VALUES (?)", [1])
    wal.append_request("r2", "INSERT INTO x VALUES (?)", [2])
    wal.close()

    # Reopen (simulates daemon restart before checkpoint)
    wal2 = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    recs = list(wal2.iter_uncommitted())
    assert len(recs) == 2
    assert recs[0][1]["req_id"] == "r1"
    assert recs[1][1]["params"] == [2]
    wal2.close()


def test_checkpoint_skips_replay(tmp_path):
    wal = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    off1 = wal.append_request("r1", "x", [1])
    off2 = wal.append_request("r2", "x", [2])
    # Checkpoint after second record
    wal.checkpoint(off2 + 100)  # any offset past r2 works
    wal.close()

    wal2 = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    recs = list(wal2.iter_uncommitted())
    # Nothing to replay — checkpoint covered everything
    assert len(recs) == 0
    wal2.close()


def test_target_db_preserved(tmp_path):
    wal = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    wal.append_request("r1", "x", [1], target_db="shadow")
    wal.append_request("r2", "x", [2], target_db="primary")
    wal.close()
    wal2 = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    recs = list(wal2.iter_uncommitted())
    assert recs[0][1]["target_db"] == "shadow"
    assert recs[1][1]["target_db"] == "primary"
    wal2.close()


def test_size_mb(tmp_path):
    wal = ServiceWAL(str(tmp_path / "wal"), max_mb=10)
    assert wal.size_mb() == 0.0
    for i in range(20):
        wal.append_request(f"r{i}", "x", [i])
    assert wal.size_mb() > 0
    wal.close()
