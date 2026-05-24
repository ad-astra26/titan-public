"""Tests for ModuleReadyShmWriter + Reader (D-SPEC-123 follow-up Option B).

Cross-process liveness check via SHM watermark — replaces the tactical
guardian=None tolerance in proxies/_start_safe.py.
"""
from __future__ import annotations

import pytest

from titan_hcl.core.module_ready_shm import (
    ModuleReadyShmReader,
    ModuleReadyShmWriter,
    get_module_ready_reader,
)


@pytest.fixture
def isolated_shm(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    monkeypatch.setenv("TITAN_SHM_ROOT", str(shm_dir))
    # Reset singleton so tests don't leak state across each other.
    import titan_hcl.core.module_ready_shm as mrs
    monkeypatch.setattr(mrs, "_reader_singleton", None)
    yield shm_dir


# ─────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────

def test_writer_publish_creates_slot_file(isolated_shm):
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({"memory": "running", "synthesis": "starting"})
        slot_path = isolated_shm / "module_ready.bin"
        assert slot_path.exists()
        assert slot_path.stat().st_size > 100   # header + buffers + payload
    finally:
        w.close()


def test_writer_publish_overwrites_previous(isolated_shm):
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({"memory": "starting"})
        w.publish({"memory": "running", "synthesis": "running"})
    finally:
        w.close()
    # Reader should see the LATEST snapshot
    r = ModuleReadyShmReader(titan_id="test")
    try:
        snap = r.read_snapshot()
        assert snap == {"memory": "running", "synthesis": "running"}
    finally:
        r.close()


def test_writer_handles_large_snapshot_under_cap(isolated_shm):
    """200 modules should fit comfortably under 16KB cap."""
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        big = {f"module_{i}": "running" for i in range(200)}
        w.publish(big)
    finally:
        w.close()
    r = ModuleReadyShmReader(titan_id="test")
    try:
        snap = r.read_snapshot()
        assert len(snap) == 200
    finally:
        r.close()


# ─────────────────────────────────────────────────────────────────────────
# Reader
# ─────────────────────────────────────────────────────────────────────────

def test_reader_returns_empty_when_slot_missing(isolated_shm):
    """No writer ever published — reader returns {} not raises."""
    r = ModuleReadyShmReader(titan_id="test")
    try:
        assert r.read_snapshot() == {}
        assert r.is_running("memory") is False
        assert r.get_state("memory") is None
    finally:
        r.close()


def test_is_running_true_for_alive_states(isolated_shm):
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({
            "running_mod": "running",
            "starting_mod": "starting",
            "unhealthy_mod": "unhealthy",
            "stopped_mod": "stopped",
            "crashed_mod": "crashed",
            "disabled_mod": "disabled",
        })
    finally:
        w.close()

    r = ModuleReadyShmReader(titan_id="test")
    try:
        # Alive states → is_running True
        assert r.is_running("running_mod") is True
        assert r.is_running("starting_mod") is True
        assert r.is_running("unhealthy_mod") is True
        # Dead states → False
        assert r.is_running("stopped_mod") is False
        assert r.is_running("crashed_mod") is False
        assert r.is_running("disabled_mod") is False
        # Absent → False
        assert r.is_running("never_existed") is False
    finally:
        r.close()


def test_is_started_alias_matches_is_running(isolated_shm):
    """is_started is an alias for is_running — supports the getattr
    lookup at _start_safe.py:80 which prefers is_started if present."""
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({"memory": "running"})
    finally:
        w.close()
    r = ModuleReadyShmReader(titan_id="test")
    try:
        assert r.is_started("memory") is r.is_running("memory")
    finally:
        r.close()


def test_get_state_returns_raw_state_string(isolated_shm):
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({"memory": "unhealthy"})
    finally:
        w.close()
    r = ModuleReadyShmReader(titan_id="test")
    try:
        assert r.get_state("memory") == "unhealthy"
    finally:
        r.close()


# ─────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────

def test_get_module_ready_reader_singleton(isolated_shm):
    a = get_module_ready_reader(titan_id="test")
    b = get_module_ready_reader(titan_id="test")
    assert a is b


# ─────────────────────────────────────────────────────────────────────────
# End-to-end: _start_safe + SHM reader integration
# ─────────────────────────────────────────────────────────────────────────

def test_start_safe_uses_shm_when_guardian_none(isolated_shm):
    """ensure_started_async_safe with guardian=None consults SHM."""
    from titan_hcl.proxies._start_safe import ensure_started_async_safe

    # First — no writer yet, SHM missing. Falls back to optimistic-True.
    assert ensure_started_async_safe(
        guardian=None, module="memory", proxy_id=1) is True

    # Now publish — module is running. Path still returns True (correct).
    w = ModuleReadyShmWriter(titan_id="test")
    try:
        w.publish({"memory": "running"})
    finally:
        w.close()
    assert ensure_started_async_safe(
        guardian=None, module="memory", proxy_id=2) is True
