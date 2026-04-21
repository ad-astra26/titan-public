"""Tests for titan_plugin/logic/offhost_mirror.py — rFP_backup_worker Phase 9.1.

No live SSH/rsync involved. Covers:
  1. Construction: disabled by default, honors config keys
  2. _build_rsync_cmd: correct flags + ssh_user@host:path + include/exclude
  3. _parse_rsync_stats: extracts files + bytes from rsync output
  4. pull_one: success path (subprocess.run stubbed)
  5. pull_one: nonzero exit returns ok=False with error fragment
  6. pull_one: timeout returns ok=False
  7. pull_all: respects enabled flag, runs concurrently, aggregates ok
  8. cleanup: removes older-than-retention files but preserves 2 newest
  9. status: reports per-host file counts, sizes, newest ages
"""
import asyncio
import os
import subprocess
import time
from unittest import mock

import pytest

from titan_plugin.logic.offhost_mirror import OffhostMirror


def _cfg(enabled=True, **overrides):
    base = {
        "backup": {
            "mirror": {
                "enabled": enabled,
                "ssh_user": "antigravity",
                "t2_host": "10.135.0.6",
                "t2_backup_path": "/home/antigravity/projects/titan/data/backups",
                "t3_host": "10.135.0.6",
                "t3_backup_path": "/home/antigravity/projects/titan3/data/backups",
                "retention_days": 7,
                "local_base": "data/backups/mirror",
                **overrides,
            }
        }
    }
    return base


# ────────────────────────────────────────────────────────────────────────────
# 1. Construction
# ────────────────────────────────────────────────────────────────────────────

def test_disabled_by_default():
    m = OffhostMirror({})
    assert m.enabled is False
    assert m.hosts == []


def test_reads_hosts_from_config():
    m = OffhostMirror(_cfg())
    assert m.enabled is True
    tids = [t[0] for t in m.hosts]
    assert tids == ["T2", "T3"]


def test_skips_host_with_missing_fields():
    c = _cfg(enabled=True, t3_host="", t3_backup_path="")
    m = OffhostMirror(c)
    tids = [t[0] for t in m.hosts]
    assert tids == ["T2"]


# ────────────────────────────────────────────────────────────────────────────
# 2. rsync command construction
# ────────────────────────────────────────────────────────────────────────────

def test_build_rsync_cmd_has_right_flags(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    cmd = m._build_rsync_cmd(
        "10.135.0.6", "/remote/data/backups", str(tmp_path / "mirror" / "T2"))
    assert cmd[0] == "rsync"
    assert "-avP" in cmd
    assert "--append-verify" in cmd
    assert "-e" in cmd
    # SSH user@host:path syntax
    assert any("antigravity@10.135.0.6:" in c for c in cmd)
    # include patterns
    assert any("personality_*.tar.gz" in c for c in cmd)
    assert any("personality_*.tar.gz.enc" in c for c in cmd)
    assert any("timechain_*.tar.zst" in c for c in cmd)
    # exclude fall-through
    assert "--exclude=*" in cmd


# ────────────────────────────────────────────────────────────────────────────
# 3. rsync output parsing
# ────────────────────────────────────────────────────────────────────────────

def test_parse_rsync_stats_handles_thousand_separator():
    out = """sending incremental file list
Number of files: 5
Number of regular files transferred: 3
Total file size: 1,234,567 bytes
total size is 1,234,567 speedup is 1.00
"""
    got = OffhostMirror._parse_rsync_stats(out)
    assert got["files_transferred"] == 3
    assert got["bytes_total"] == 1234567


def test_parse_rsync_stats_empty_output():
    got = OffhostMirror._parse_rsync_stats("")
    assert got == {"files_transferred": 0, "bytes_total": 0}


# ────────────────────────────────────────────────────────────────────────────
# 4. pull_one happy + error paths
# ────────────────────────────────────────────────────────────────────────────

def _fake_proc(returncode=0, stdout="", stderr=""):
    p = mock.MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_pull_one_success(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    stdout = "Number of regular files transferred: 2\ntotal size is 100 speedup is 1\n"
    with mock.patch("subprocess.run", return_value=_fake_proc(0, stdout)):
        r = m.pull_one("T2", "10.135.0.6", "/remote/path")
    assert r["ok"] is True
    assert r["titan_id"] == "T2"
    assert r["files_transferred"] == 2
    assert r["bytes_total"] == 100
    # Local dir created
    assert os.path.isdir(os.path.join(str(tmp_path / "mirror"), "T2"))


def test_pull_one_nonzero_returncode(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    with mock.patch("subprocess.run", return_value=_fake_proc(23, "", "Permission denied")):
        r = m.pull_one("T2", "10.135.0.6", "/remote/path")
    assert r["ok"] is False
    assert r["returncode"] == 23
    assert "Permission denied" in r["error"]


def test_pull_one_timeout(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror"),
                            rsync_timeout_sec=1))
    with mock.patch("subprocess.run",
                     side_effect=subprocess.TimeoutExpired(cmd=["rsync"], timeout=1)):
        r = m.pull_one("T2", "10.135.0.6", "/remote/path")
    assert r["ok"] is False
    assert "timeout" in r["error"]


# ────────────────────────────────────────────────────────────────────────────
# 5. pull_all
# ────────────────────────────────────────────────────────────────────────────

def test_pull_all_disabled_returns_disabled():
    m = OffhostMirror(_cfg(enabled=False))
    r = asyncio.run(m.pull_all())
    assert r["ok"] is True
    assert r["mode"] == "disabled"


def test_pull_all_no_hosts():
    m = OffhostMirror(_cfg(enabled=True, t2_host="", t2_backup_path="",
                            t3_host="", t3_backup_path=""))
    r = asyncio.run(m.pull_all())
    assert r["ok"] is True
    assert r["mode"] == "no_hosts"


def test_pull_all_success_aggregates_both_hosts(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    stdout = "Number of regular files transferred: 1\ntotal size is 50 speedup is 1\n"
    with mock.patch("subprocess.run", return_value=_fake_proc(0, stdout)):
        r = asyncio.run(m.pull_all())
    assert r["ok"] is True
    assert len(r["results"]) == 2
    assert all(x["ok"] for x in r["results"])


def test_pull_all_partial_failure_propagates_not_ok(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    call_count = {"n": 0}

    def _rsync_side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _fake_proc(0, "Number of regular files transferred: 1\ntotal size is 10 speedup is 1\n")
        return _fake_proc(23, "", "failed")

    with mock.patch("subprocess.run", side_effect=_rsync_side_effect):
        r = asyncio.run(m.pull_all())
    assert r["ok"] is False
    # One ok, one failure
    statuses = [x["ok"] for x in r["results"]]
    assert True in statuses and False in statuses


# ────────────────────────────────────────────────────────────────────────────
# 6. Retention cleanup
# ────────────────────────────────────────────────────────────────────────────

def test_cleanup_removes_old_files_preserves_two_newest(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror"),
                            retention_days=7))
    local = os.path.join(str(tmp_path / "mirror"), "T2")
    os.makedirs(local, exist_ok=True)
    now = time.time()
    # 5 files of same pattern, varying ages (newest first)
    ages_days = [0, 1, 5, 14, 30]
    files = []
    for i, age_d in enumerate(ages_days):
        f = os.path.join(local, f"personality_2026_{i}_aaa.tar.gz")
        with open(f, "wb") as fh:
            fh.write(b"x")
        mtime = now - age_d * 86400
        os.utime(f, (mtime, mtime))
        files.append(f)

    deleted = m.cleanup("T2")
    remaining = [f for f in files if os.path.exists(f)]
    # Retention=7d → files at days 14 + 30 should be deleted → 2 removed
    # But two newest are always preserved regardless; files at days 0 + 1 are preserved by both rules
    # File at day 5 is preserved by retention. Files at 14 + 30 are beyond retention and not in top-2
    assert deleted == 2
    assert len(remaining) == 3


def test_cleanup_noop_when_dir_missing(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "nope")))
    assert m.cleanup("T2") == 0


# ────────────────────────────────────────────────────────────────────────────
# 7. Status
# ────────────────────────────────────────────────────────────────────────────

def test_status_enabled_with_files(tmp_path):
    m = OffhostMirror(_cfg(local_base=str(tmp_path / "mirror")))
    local = os.path.join(str(tmp_path / "mirror"), "T2")
    os.makedirs(local, exist_ok=True)
    p = os.path.join(local, "personality_20260501_abc.tar.gz")
    with open(p, "wb") as f:
        f.write(b"x" * (2 * 1024 * 1024))  # 2 MB
    s = m.status()
    assert s["enabled"] is True
    t2 = next(h for h in s["hosts"] if h["titan_id"] == "T2")
    assert t2["files"] == 1
    assert t2["size_mb"] >= 2.0
    assert t2["newest_age_h"] is not None


def test_status_disabled():
    m = OffhostMirror(_cfg(enabled=False))
    s = m.status()
    assert s["enabled"] is False
