"""Tests for Option B — Guardian.restart_async() executor offload.

Pre-fix: monitor_tick called self.restart() synchronously, blocking up
to 30s per worker on SAVE_NOW wait. Multi-restart bursts cascaded into
180s+ of monitor_tick blockage → heartbeat starvation → false timeouts
→ MORE restarts. Option B kicks restart() onto a separate executor
thread so monitor_tick continues processing the guardian queue.

See BUG-GUARDIAN-STOP-SAVE-NOW-HEARTBEAT-CASCADE-20260502.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import Guardian, ModuleSpec


def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L3", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


# ── restart_async semantics ───────────────────────────────────────────────


def test_restart_async_returns_future_for_known_module():
    """restart_async() submits work to executor and returns a Future."""
    g = Guardian(DivineBus())
    g.register(_spec("test_worker"))
    # Mock restart so we don't try to actually spawn anything
    g.restart = MagicMock(return_value=True)

    fut = g.restart_async("test_worker", reason="unit_test")
    assert fut is not None, "restart_async should return a Future for known module"
    # Wait for completion
    assert fut.result(timeout=2.0) is True
    g.restart.assert_called_once_with("test_worker", reason="unit_test")
    g.stop_all()


def test_restart_async_idempotent_returns_none_when_in_flight():
    """Submitting restart_async() for an already-restarting module returns
    None — prevents queue buildup of duplicate restart submissions."""
    g = Guardian(DivineBus())
    g.register(_spec("slow_worker"))

    # Use an Event to keep the first restart pending in-flight
    blocker = threading.Event()
    call_count = [0]

    def slow_restart(name, reason="x"):
        call_count[0] += 1
        blocker.wait(timeout=2.0)
        return True

    g.restart = slow_restart

    fut1 = g.restart_async("slow_worker", reason="first")
    assert fut1 is not None
    # Wait briefly for executor to pick it up + add to in-flight set
    time.sleep(0.05)

    # Second submit should return None — already in flight
    fut2 = g.restart_async("slow_worker", reason="second")
    assert fut2 is None, "second submit should be a no-op"

    blocker.set()
    fut1.result(timeout=2.0)
    assert call_count[0] == 1, "restart should run exactly once for in-flight dedup"

    # After the first completes, a NEW submit should succeed (no longer in flight)
    fut3 = g.restart_async("slow_worker", reason="third")
    assert fut3 is not None
    fut3.result(timeout=2.0)
    assert call_count[0] == 2

    g.stop_all()


def test_restart_async_clears_in_flight_on_exception():
    """If restart() raises, the in-flight set must still be cleared so
    subsequent restart_async() calls work."""
    g = Guardian(DivineBus())
    g.register(_spec("crashy"))

    def buggy_restart(name, reason="x"):
        raise RuntimeError("simulated crash inside restart()")

    g.restart = buggy_restart

    fut = g.restart_async("crashy", reason="first")
    # Wait for the future to complete (with exception)
    with pytest.raises(RuntimeError):
        fut.result(timeout=2.0)

    # After exception, in-flight set should be cleared
    assert "crashy" not in g._restarts_in_flight

    # Subsequent submit must work
    g.restart = MagicMock(return_value=True)
    fut2 = g.restart_async("crashy", reason="second")
    assert fut2 is not None
    fut2.result(timeout=2.0)

    g.stop_all()


def test_concurrent_restart_async_for_different_modules():
    """4 different modules can restart concurrently via the executor
    (max_workers=4). Each runs in its own thread."""
    g = Guardian(DivineBus())
    for n in ["a", "b", "c", "d"]:
        g.register(_spec(n))

    thread_ids = set()
    enter_event = threading.Event()
    enter_count = [0]
    release_event = threading.Event()

    def slow_concurrent_restart(name, reason="x"):
        thread_ids.add(threading.get_ident())
        enter_count[0] += 1
        if enter_count[0] >= 4:
            enter_event.set()
        # Block until released
        release_event.wait(timeout=5.0)
        return True

    g.restart = slow_concurrent_restart

    futures = [g.restart_async(n, reason="parallel") for n in ["a", "b", "c", "d"]]
    assert all(f is not None for f in futures)

    # All 4 should enter restart concurrently (within 4-worker pool)
    assert enter_event.wait(timeout=3.0), "expected 4 concurrent restarts within 3s"
    assert len(thread_ids) >= 2, "expected concurrent execution across threads"

    release_event.set()
    for f in futures:
        f.result(timeout=3.0)

    g.stop_all()


# ── Integration with monitor_tick ─────────────────────────────────────────


def test_monitor_tick_does_not_call_restart_synchronously():
    """Inspect monitor_tick source — it must NOT call self.restart()
    synchronously (which would block the tick on SAVE_NOW), but instead emit a
    restart REQUEST asynchronously. Post Phase-11 split (§11.I.1) monitor_tick
    lives on Supervisor and publishes MODULE_RESTART_REQUEST via
    `publish_module_restart_request` rather than calling restart() inline."""
    import inspect
    from titan_hcl.supervisor import Supervisor

    src = inspect.getsource(Supervisor.monitor_tick)
    # No SYNCHRONOUS restart inside the tick (the original starvation bug).
    assert "self.restart(" not in src, \
        "monitor_tick must NOT call self.restart() synchronously (Option B)"
    # The fault paths emit an async restart REQUEST instead.
    assert "publish_module_restart_request" in src, \
        "monitor_tick fault paths must emit an async restart-request"
    # RSS-overrun + heartbeat-timeout faults are detected here and routed to
    # the async request path (reason strings carried on the request).
    assert 'f"rss_' in src, "RSS-overrun fault must be detected in monitor_tick"
    assert "heartbeat_timeout" in src, \
        "heartbeat-timeout fault must be detected in monitor_tick"


def test_bus_peer_died_handler_uses_restart_async():
    """The BUS_PEER_DIED handler in _process_guardian_messages must also
    use restart_async (it runs inside monitor_tick → same starvation risk)."""
    import inspect
    from titan_hcl import guardian_hcl as g_mod

    src = inspect.getsource(g_mod.Guardian._process_guardian_messages)
    assert 'restart_async(name, reason="broker_peer_dead")' in src, \
        "BUS_PEER_DIED handler must use restart_async (Option B)"


# ── stop_all shuts down executor cleanly ──────────────────────────────────


def test_stop_all_shuts_down_restart_executor():
    """stop_all() must shut down the restart executor so threads don't
    leak across kernel restarts."""
    g = Guardian(DivineBus())
    g.register(_spec("x"))
    g.restart = MagicMock(return_value=True)

    g.restart_async("x", reason="test")
    time.sleep(0.05)

    g.stop_all()
    # Executor should be shut down — submitting after raises RuntimeError
    with pytest.raises(RuntimeError):
        g._restart_executor.submit(lambda: None)
