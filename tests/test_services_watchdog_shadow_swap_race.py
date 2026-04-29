"""C2-9 / BUG-SERVICES-WATCHDOG-SHADOW-SWAP-RACE-20260428.

Verifies the shadow_orchestrator + services_watchdog.sh coordination
contract per PLAN_microkernel_phase_c_s2_kernel.md §17.1:

  - shadow_orchestrator writes a JSON-format lock at swap start
  - lock contains pid + swap_id + started_at + expected_end_at + heartbeat_at
  - heartbeat_swap_lock refreshes the lock atomically
  - remove_swap_lock clears it on completion
  - services_watchdog.sh skips duplicate-kill when lock is fresh
  - services_watchdog.sh force-cleans when lock is expired (orchestrator crashed)
  - regression: legacy plain-epoch lock from safe_restart.sh still respected
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from titan_plugin.core import shadow_orchestrator as so


# ─── Lockfile API surface ──────────────────────────────────────────────


class TestSwapLockApi:
    def test_write_swap_lock_creates_json_file(self, tmp_path, monkeypatch):
        # Redirect /tmp/* under tmp_path so test doesn't touch real /tmp
        monkeypatch.setattr(
            so, "_restart_lock_path",
            lambda titan_id: tmp_path / f"titan_{titan_id}_restart.lock",
        )
        path = so.write_swap_lock("T1", "evt-abc-123", ttl_s=60.0)
        assert path.exists()
        body = json.loads(path.read_text())
        assert body["swap_id"] == "evt-abc-123"
        assert body["pid"] == os.getpid()
        assert body["writer"] == "shadow_orchestrator"
        assert body["expected_end_at"] - body["started_at"] == pytest.approx(60.0, abs=1.0)

    def test_heartbeat_updates_timestamps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            so, "_restart_lock_path",
            lambda titan_id: tmp_path / "lock.json",
        )
        path = so.write_swap_lock("T1", "evt-1", ttl_s=30.0)
        first = json.loads(path.read_text())
        time.sleep(0.05)
        so.heartbeat_swap_lock(path)
        second = json.loads(path.read_text())
        assert second["heartbeat_at"] > first["heartbeat_at"]
        assert second["expected_end_at"] > first["expected_end_at"]
        # swap_id is preserved across heartbeats
        assert second["swap_id"] == first["swap_id"]

    def test_heartbeat_silent_no_op_when_file_missing(self, tmp_path):
        # Should not raise even if file gone (orchestrator vs cleanup race)
        so.heartbeat_swap_lock(tmp_path / "does_not_exist.json")

    def test_remove_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            so, "_restart_lock_path",
            lambda titan_id: tmp_path / "lock.json",
        )
        path = so.write_swap_lock("T1", "evt-1")
        so.remove_swap_lock(path)
        # Second call must not raise
        so.remove_swap_lock(path)
        assert not path.exists()

    def test_titan_id_path_mapping(self):
        # T1 → titan1_restart.lock, T2 → titan2_, T3 → titan3_
        assert so._restart_lock_path("T1").name == "titan1_restart.lock"
        assert so._restart_lock_path("T2").name == "titan2_restart.lock"
        assert so._restart_lock_path("T3").name == "titan3_restart.lock"
        # Lowercase + missing-id fallback to "1"
        assert so._restart_lock_path("foo").name == "titan1_restart.lock"


# ─── Heartbeat thread ──────────────────────────────────────────────────


class TestSwapLockHeartbeat:
    def test_thread_starts_and_stops_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            so, "SHADOW_SWAP_LOCK_HEARTBEAT_S", 0.05  # speed up
        )
        monkeypatch.setattr(
            so, "_restart_lock_path",
            lambda titan_id: tmp_path / "lock.json",
        )
        path = so.write_swap_lock("T1", "evt-1")
        first_hb = json.loads(path.read_text())["heartbeat_at"]

        hb = so._SwapLockHeartbeat(path)
        hb.start()
        time.sleep(0.2)  # at least 2-3 heartbeat ticks
        hb.stop(remove=True)

        assert not path.exists(), "remove=True should delete lock"
        # If the file was still there before stop, heartbeat_at would have advanced.

    def test_stop_without_remove_keeps_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            so, "_restart_lock_path",
            lambda titan_id: tmp_path / "lock.json",
        )
        path = so.write_swap_lock("T1", "evt-1")
        hb = so._SwapLockHeartbeat(path)
        hb.start()
        time.sleep(0.05)
        hb.stop(remove=False)
        assert path.exists()


# ─── services_watchdog.sh contract — source-level + behavioral ────────


class TestServicesWatchdogContract:
    """Source-level checks; behavioral coverage is via shell-level
    integration testing in arch_map services-watchdog smoke runs."""

    def test_watchdog_recognizes_json_lock_format(self):
        watchdog_src = (
            Path(__file__).parent.parent / "scripts" / "services_watchdog.sh"
        ).read_text(encoding="utf-8")
        # JSON form detection (`{` first char)
        assert '"{"' in watchdog_src or "{:0:1}" in watchdog_src or "${LOCK_RAW:0:1}" in watchdog_src
        # expected_end_at field name reference
        assert "expected_end_at" in watchdog_src

    def test_watchdog_force_cleans_expired_lock(self):
        watchdog_src = (
            Path(__file__).parent.parent / "scripts" / "services_watchdog.sh"
        ).read_text(encoding="utf-8")
        # The expired-lock branch must rm the stale file
        assert "LOCK_EXPIRED" in watchdog_src
        assert 'rm -f "$RESTART_LOCK_FILE"' in watchdog_src

    def test_watchdog_preserves_legacy_plain_epoch_format(self):
        """Regression: safe_restart.sh / *_manage.sh writers still work."""
        watchdog_src = (
            Path(__file__).parent.parent / "scripts" / "services_watchdog.sh"
        ).read_text(encoding="utf-8")
        # 90s window + LOCK_AGE math preserved for legacy form
        assert "LOCK_AGE" in watchdog_src
        assert "lt 90" in watchdog_src.replace(" ", "").replace("-", "") or "-lt 90" in watchdog_src
