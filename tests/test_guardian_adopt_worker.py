"""
Tests for Phase B.2.1 Guardian.adopt_worker + cleanup distinction.

Covers:
- adopt_worker(name, real_pid, spec) registers without spawning
- Adopted ModuleInfo: state=RUNNING, adopted=True, no .process, fresh clocks
- Cleanup of adopted worker uses os.kill SIGTERM, then SIGKILL fallback
- Adopting unknown name returns False
- Adopting non-existent PID returns False
- adopt_worker rejects already-running fresh module (no double-claim)
- BUS_WORKER_ADOPT_REQUEST dispatch sends BUS_WORKER_ADOPT_ACK with status
- get_status exposes adopted + start_method + adopt_ts fields

Adopted-worker cleanup path uses os.kill (worker process is externally
spawned; we don't own multiprocessing.Process). Pre-B.2.1 path uses
process.kill() and pgid-based killpg.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin.bus import (
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    make_msg,
)
from titan_plugin.guardian import (
    Guardian,
    ModuleInfo,
    ModuleSpec,
    ModuleState,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _dummy_entry(*args, **kwargs):
    """Spec entry_fn — never invoked under adopt_worker."""
    return None


def _make_guardian():
    bus = DivineBus()
    return Guardian(bus)


def _spawn_sleeper() -> subprocess.Popen:
    """Spawn a real subprocess that sleeps; returns Popen handle.

    Used to test adoption against a live external PID. The sleeper prints
    its PID then sleeps 60s — easily killed by SIGTERM/SIGKILL in tests.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── _pid_alive helper ──────────────────────────────────────────────────


def test_pid_alive_returns_true_for_self():
    g = _make_guardian()
    assert g._pid_alive(os.getpid()) is True


def test_pid_alive_returns_false_for_dead_pid():
    g = _make_guardian()
    proc = _spawn_sleeper()
    proc.kill()
    proc.wait(timeout=2.0)
    # PID may be reused; in practice 1 second after kill it's still gone.
    # Use a known-dead PID via wait().
    assert g._pid_alive(proc.pid) is False


# ── adopt_worker happy path ─────────────────────────────────────────────


def test_adopt_worker_registers_without_spawning():
    """Adopting a live-PID worker registers ModuleInfo with adopted=True."""
    g = _make_guardian()
    g.register(ModuleSpec(name="adopt_test", entry_fn=_dummy_entry,
                          layer="L3", start_method="spawn"))
    proc = _spawn_sleeper()
    try:
        ok = g.adopt_worker("adopt_test", proc.pid)
        assert ok is True
        info = g._modules["adopt_test"]
        assert info.adopted is True
        assert info.pid == proc.pid
        assert info.state == ModuleState.RUNNING
        assert info.process is None  # we don't own it
        assert info.start_time > 0
        assert info.last_heartbeat > 0
        assert info.ready_time > 0
        assert info.adopt_ts > 0
    finally:
        proc.kill()
        proc.wait(timeout=2.0)


def test_adopt_worker_uses_passed_spec_when_unregistered():
    """If spec is provided + no prior register, adopt_worker still succeeds."""
    g = _make_guardian()
    spec = ModuleSpec(name="adhoc", entry_fn=_dummy_entry,
                     layer="L3", start_method="spawn")
    proc = _spawn_sleeper()
    try:
        ok = g.adopt_worker("adhoc", proc.pid, spec=spec)
        assert ok is True
        assert g._modules["adhoc"].spec.start_method == "spawn"
    finally:
        proc.kill()
        proc.wait(timeout=2.0)


def test_adopt_worker_unknown_name_returns_false():
    """No spec registered + no spec passed = reject."""
    g = _make_guardian()
    assert g.adopt_worker("never_registered", 1) is False


def test_adopt_worker_dead_pid_returns_false():
    """Spec exists but PID isn't alive = reject."""
    g = _make_guardian()
    g.register(ModuleSpec(name="ghost", entry_fn=_dummy_entry, layer="L3"))
    proc = _spawn_sleeper()
    proc.kill()
    proc.wait(timeout=2.0)
    assert g.adopt_worker("ghost", proc.pid) is False


def test_adopt_worker_rejects_already_running():
    """If a fresh worker is already RUNNING (not adopted), reject re-claim."""
    g = _make_guardian()
    g.register(ModuleSpec(name="busy", entry_fn=_dummy_entry, layer="L3"))
    info = g._modules["busy"]
    info.state = ModuleState.RUNNING
    info.adopted = False
    info.pid = 999_999  # arbitrary
    proc = _spawn_sleeper()
    try:
        # Attempt to adopt with a different live PID — must be rejected
        # because a fresh worker is already claiming this slot.
        assert g.adopt_worker("busy", proc.pid) is False
    finally:
        proc.kill()
        proc.wait(timeout=2.0)


# ── Cleanup distinction (adopted vs owned) ──────────────────────────────


def test_cleanup_adopted_uses_os_kill_sigterm():
    """Adopted worker cleanup goes through _kill_adopted_process (os.kill SIGTERM)."""
    g = _make_guardian()
    g.register(ModuleSpec(name="cleanup_a", entry_fn=_dummy_entry, layer="L3"))
    proc = _spawn_sleeper()
    try:
        assert g.adopt_worker("cleanup_a", proc.pid) is True
        # Trigger cleanup — should send SIGTERM, wait briefly, then proc exits
        g._cleanup_module("cleanup_a")
        # Give a moment for the OS to deliver SIGTERM and process to die
        time.sleep(0.5)
        # Process should be dead now
        proc.wait(timeout=3.0)
        assert proc.returncode is not None
        # ModuleInfo state cleared
        info = g._modules["cleanup_a"]
        assert info.state == ModuleState.STOPPED
        assert info.pid is None
        assert info.adopted is False  # cleared by _finalize
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


def test_cleanup_adopted_dead_pid_is_noop():
    """If adopted worker already exited, cleanup is graceful (ProcessLookupError)."""
    g = _make_guardian()
    g.register(ModuleSpec(name="zombie", entry_fn=_dummy_entry, layer="L3"))
    info = g._modules["zombie"]
    info.adopted = True
    info.pid = 1  # init — kill(1, 0) returns PermissionError = "alive"
    info.state = ModuleState.RUNNING

    # Mock os.kill so we don't actually try to SIGTERM init.
    with patch("titan_plugin.guardian.os.kill") as m_kill:
        m_kill.side_effect = ProcessLookupError()
        g._kill_adopted_process(info, "zombie")
    # ProcessLookupError → returns without escalating to SIGKILL


def test_cleanup_adopted_escalates_to_sigkill_after_grace():
    """If process doesn't die within 2s of SIGTERM, escalates to SIGKILL."""
    g = _make_guardian()
    g.register(ModuleSpec(name="stubborn", entry_fn=_dummy_entry, layer="L3"))
    info = g._modules["stubborn"]
    info.adopted = True
    info.pid = 12345  # arbitrary
    info.state = ModuleState.RUNNING

    call_log = []

    def fake_kill(pid, sig):
        call_log.append((pid, sig))
        if sig == 0:
            return None  # always "alive" — never exits gracefully
        if sig == signal.SIGKILL:
            raise ProcessLookupError()  # finally gone after SIGKILL

    with patch("titan_plugin.guardian.os.kill", side_effect=fake_kill), \
         patch("titan_plugin.guardian.time.sleep"), \
         patch("titan_plugin.guardian.time.time", side_effect=[0, 0.5, 1.0, 1.5, 2.5]):
        g._kill_adopted_process(info, "stubborn")
    # Should have called SIGTERM, then SIGKILL after grace
    sigs = [c[1] for c in call_log]
    assert signal.SIGTERM in sigs
    assert signal.SIGKILL in sigs


# ── BUS_WORKER_ADOPT_REQUEST dispatch ────────────────────────────────────


def test_adopt_request_dispatch_publishes_ack_adopted():
    """Guardian dispatches BUS_WORKER_ADOPT_REQUEST → publishes BUS_WORKER_ADOPT_ACK."""
    g = _make_guardian()
    g.register(ModuleSpec(name="dispatch_ok", entry_fn=_dummy_entry,
                          layer="L3", start_method="spawn"))
    proc = _spawn_sleeper()

    # Capture published messages
    published = []
    g.bus.publish = MagicMock(side_effect=lambda m: published.append(m))

    # Inject the request directly into Guardian's queue (bypass real bus path)
    req = make_msg(
        BUS_WORKER_ADOPT_REQUEST,
        "dispatch_ok",
        "kernel",
        {"name": "dispatch_ok", "pid": proc.pid, "start_method": "spawn"},
        rid="rid-test-1",
    )
    g._guardian_queue.put(req)

    try:
        g._process_guardian_messages()
    finally:
        proc.kill()
        proc.wait(timeout=2.0)

    # Find the ACK
    acks = [m for m in published if m.get("type") == BUS_WORKER_ADOPT_ACK]
    assert len(acks) == 1, f"Expected 1 ACK, got {published}"
    ack = acks[0]
    assert ack["rid"] == "rid-test-1"
    assert ack["payload"]["status"] == "adopted"
    assert ack["payload"]["name"] == "dispatch_ok"
    assert ack["payload"]["shadow_pid"] == os.getpid()
    # ModuleInfo updated
    assert g._modules["dispatch_ok"].adopted is True


def test_adopt_request_dispatch_unknown_name_rejects():
    """Unknown name → ACK status=rejected with reason=unknown_name."""
    g = _make_guardian()
    proc = _spawn_sleeper()
    try:
        published = []
        g.bus.publish = MagicMock(side_effect=lambda m: published.append(m))
        req = make_msg(
            BUS_WORKER_ADOPT_REQUEST,
            "ghost",
            "kernel",
            {"name": "ghost", "pid": proc.pid, "start_method": "spawn"},
            rid="rid-2",
        )
        g._guardian_queue.put(req)
        g._process_guardian_messages()

        acks = [m for m in published if m.get("type") == BUS_WORKER_ADOPT_ACK]
        assert len(acks) == 1
        assert acks[0]["payload"]["status"] == "rejected"
        assert acks[0]["payload"]["reason"] == "unknown_name"
    finally:
        proc.kill()
        proc.wait(timeout=2.0)


def test_adopt_request_dispatch_malformed_payload_ignored():
    """Malformed payload (missing pid) → no ACK published."""
    g = _make_guardian()
    g.register(ModuleSpec(name="ok", entry_fn=_dummy_entry, layer="L3"))
    published = []
    g.bus.publish = MagicMock(side_effect=lambda m: published.append(m))
    req = make_msg(
        BUS_WORKER_ADOPT_REQUEST,
        "ok",
        "kernel",
        {"name": "ok"},  # no pid
        rid="rid-3",
    )
    g._guardian_queue.put(req)
    g._process_guardian_messages()

    acks = [m for m in published if m.get("type") == BUS_WORKER_ADOPT_ACK]
    assert len(acks) == 0


# ── get_status exposes B.2.1 fields ─────────────────────────────────────


def test_get_status_exposes_adopted_and_start_method():
    """Status dict per module includes adopted, adopt_ts, start_method."""
    g = _make_guardian()
    g.register(ModuleSpec(name="m1", entry_fn=_dummy_entry,
                          layer="L3", start_method="spawn"))
    g.register(ModuleSpec(name="m2", entry_fn=_dummy_entry,
                          layer="L3", start_method="fork"))
    proc = _spawn_sleeper()
    try:
        g.adopt_worker("m1", proc.pid)
        status = g.get_status()
        assert status["m1"]["adopted"] is True
        assert status["m1"]["adopt_ts"] > 0
        assert status["m1"]["start_method"] == "spawn"
        # Non-adopted module
        assert status["m2"]["adopted"] is False
        assert status["m2"]["adopt_ts"] == 0.0
        assert status["m2"]["start_method"] == "fork"
    finally:
        proc.kill()
        proc.wait(timeout=2.0)
