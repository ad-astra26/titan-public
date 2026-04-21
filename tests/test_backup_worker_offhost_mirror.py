"""Tests for rFP_backup_worker Phase 9.2 — BackupWorker → OffhostMirror wiring.

Covers the `_run_offhost_mirror` helper that fires after a successful
meditation cascade. Does not spawn a real subprocess; stubs the mirror
and pump the state dict directly.
"""
import asyncio
import os
import sys
from unittest import mock

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from titan_plugin.modules.backup_worker import _run_offhost_mirror


def _state(full_config, send_queue=None):
    return {
        "backup": mock.MagicMock(),
        "mode": "mainnet_arweave",
        "titan_id": "T1",
        "loop": asyncio.new_event_loop(),
        "send_queue": send_queue if send_queue is not None else mock.MagicMock(),
        "name": "backup",
        "full_config": full_config,
    }


def _cfg(enabled=True, hosts=True):
    base = {
        "backup": {
            "mirror": {
                "enabled": enabled,
                "ssh_user": "antigravity",
                "t2_host": "10.135.0.6" if hosts else "",
                "t2_backup_path": "/rp/t2/backups" if hosts else "",
                "t3_host": "10.135.0.6" if hosts else "",
                "t3_backup_path": "/rp/t3/backups" if hosts else "",
                "retention_days": 7,
                "local_base": "/tmp/mirror_test_nonexistent",
            }
        }
    }
    return base


# ────────────────────────────────────────────────────────────────────────────
# Disabled / no-hosts → no-op
# ────────────────────────────────────────────────────────────────────────────

def test_disabled_returns_immediately(tmp_path):
    cfg = _cfg(enabled=False)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    sq = mock.MagicMock()
    state = _state(cfg, send_queue=sq)
    _run_offhost_mirror(state, med_count=42)
    # No bus emissions when disabled
    sq.put.assert_not_called() if hasattr(sq, "put") else None


def test_no_hosts_configured_returns_immediately(tmp_path):
    cfg = _cfg(enabled=True, hosts=False)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    sq = mock.MagicMock()
    state = _state(cfg, send_queue=sq)
    _run_offhost_mirror(state, med_count=1)
    # No emit when no hosts
    # (send_queue is not an actual Queue; we just confirm no errors)


# ────────────────────────────────────────────────────────────────────────────
# Happy path — all mirror pulls succeed
# ────────────────────────────────────────────────────────────────────────────

def test_happy_path_emits_complete(tmp_path):
    cfg = _cfg(enabled=True, hosts=True)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    sent_events = []

    def _capture_send(queue, ev, src, dst, payload):
        sent_events.append((ev, src, dst, payload))

    # Stub OffhostMirror to return success without real rsync
    with mock.patch(
        "titan_plugin.logic.offhost_mirror.OffhostMirror.pull_all",
        new=mock.AsyncMock(return_value={
            "ok": True,
            "results": [
                {"ok": True, "titan_id": "T2", "host": "10.135.0.6",
                 "duration_s": 1.2, "files_transferred": 1, "bytes_total": 500},
                {"ok": True, "titan_id": "T3", "host": "10.135.0.6",
                 "duration_s": 0.8, "files_transferred": 0, "bytes_total": 0},
            ],
            "completed_at": 1778000000,
        })), \
         mock.patch("titan_plugin.modules.backup_worker._send", side_effect=_capture_send), \
         mock.patch("titan_plugin.modules.backup_worker._write_i7_telemetry"):
        state = _state(cfg)
        _run_offhost_mirror(state, med_count=7)

    assert len(sent_events) == 1
    ev, src, dst, payload = sent_events[0]
    assert ev == "OFFHOST_MIRROR_COMPLETE"
    assert dst == "all"
    assert payload["meditation_count"] == 7
    assert len(payload["hosts"]) == 2


# ────────────────────────────────────────────────────────────────────────────
# Partial failure → OFFHOST_MIRROR_FAILED with errors list
# ────────────────────────────────────────────────────────────────────────────

def test_partial_failure_emits_failed(tmp_path):
    cfg = _cfg(enabled=True, hosts=True)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    sent_events = []

    def _capture_send(queue, ev, src, dst, payload):
        sent_events.append((ev, src, dst, payload))

    with mock.patch(
        "titan_plugin.logic.offhost_mirror.OffhostMirror.pull_all",
        new=mock.AsyncMock(return_value={
            "ok": False,
            "results": [
                {"ok": True, "titan_id": "T2", "duration_s": 1.0,
                 "files_transferred": 1, "bytes_total": 100},
                {"ok": False, "titan_id": "T3", "host": "10.135.0.6",
                 "duration_s": 0.5, "error": "ssh: connection refused"},
            ],
        })), \
         mock.patch("titan_plugin.modules.backup_worker._send", side_effect=_capture_send), \
         mock.patch("titan_plugin.modules.backup_worker._write_i7_telemetry"):
        state = _state(cfg)
        _run_offhost_mirror(state, med_count=9)

    assert len(sent_events) == 1
    ev, _, _, payload = sent_events[0]
    assert ev == "OFFHOST_MIRROR_FAILED"
    assert any("connection refused" in str(e.get("error", "")) for e in payload["errors"])


# ────────────────────────────────────────────────────────────────────────────
# Mirror crash → outer exception → emits FAILED without raising
# ────────────────────────────────────────────────────────────────────────────

def test_mirror_crash_emits_failed_not_raised(tmp_path):
    cfg = _cfg(enabled=True, hosts=True)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    sent_events = []

    def _capture_send(queue, ev, src, dst, payload):
        sent_events.append((ev, src, dst, payload))

    with mock.patch(
        "titan_plugin.logic.offhost_mirror.OffhostMirror.pull_all",
        new=mock.AsyncMock(side_effect=RuntimeError("rsync exploded"))), \
         mock.patch("titan_plugin.modules.backup_worker._send", side_effect=_capture_send), \
         mock.patch("titan_plugin.modules.backup_worker._write_i7_telemetry"):
        state = _state(cfg)
        # Must NOT raise — a mirror failure can never fail the backup
        _run_offhost_mirror(state, med_count=12)

    assert len(sent_events) == 1
    assert sent_events[0][0] == "OFFHOST_MIRROR_FAILED"
    assert "rsync exploded" in sent_events[0][3]["error"]


# ────────────────────────────────────────────────────────────────────────────
# Cleanup_all is invoked on success
# ────────────────────────────────────────────────────────────────────────────

def test_cleanup_called_on_success(tmp_path):
    cfg = _cfg(enabled=True, hosts=True)
    cfg["backup"]["mirror"]["local_base"] = str(tmp_path / "m")
    cleanup_called = {"n": 0}

    def _fake_cleanup(self):
        cleanup_called["n"] += 1
        return {"T2": 0, "T3": 0}

    with mock.patch(
        "titan_plugin.logic.offhost_mirror.OffhostMirror.pull_all",
        new=mock.AsyncMock(return_value={"ok": True, "results": []})), \
         mock.patch(
             "titan_plugin.logic.offhost_mirror.OffhostMirror.cleanup_all",
             new=_fake_cleanup), \
         mock.patch("titan_plugin.modules.backup_worker._send"), \
         mock.patch("titan_plugin.modules.backup_worker._write_i7_telemetry"):
        state = _state(cfg)
        _run_offhost_mirror(state, med_count=1)

    assert cleanup_called["n"] == 1
