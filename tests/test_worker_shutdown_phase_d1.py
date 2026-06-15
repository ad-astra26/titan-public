"""Phase D.1 (RFP_supervision_lifecycle §7.D) — bus-independent worker self-save.

Verifies worker_shutdown's registry + run-once semantics + the floor SIGTERM
handler. These are the unit-level guarantees behind "a worker persists its
state on SIGTERM without the bus" — the foundation that makes graceful shutdown
survive a dead/slow bus (no hang, no data loss → no SIGKILL).
"""
import importlib
import signal

import pytest


@pytest.fixture(autouse=True)
def _fresh_module():
    """Each test gets a clean process-global registry (run-once guard resets)."""
    import titan_hcl.core.worker_shutdown as ws
    importlib.reload(ws)
    yield ws
    # Restore default SIGTERM disposition so a test's installed handler can't
    # leak into the pytest process's later behavior.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (ValueError, OSError):
        pass


def test_register_and_run_invokes_each_callback_once(_fresh_module):
    ws = _fresh_module
    calls = []
    ws.register_shutdown_save("a", lambda: calls.append("a"))
    ws.register_shutdown_save("b", lambda: calls.append("b"))

    ran = ws.run_shutdown_saves()

    assert ran == 2
    assert calls == ["a", "b"]


def test_run_is_idempotent_run_once(_fresh_module):
    ws = _fresh_module
    calls = []
    ws.register_shutdown_save("a", lambda: calls.append("a"))

    first = ws.run_shutdown_saves()
    second = ws.run_shutdown_saves()  # finally may fire after an orderly save

    assert first == 1
    assert second == 0  # run-once guard — no double-save
    assert calls == ["a"]


def test_one_failing_callback_does_not_starve_the_rest(_fresh_module):
    ws = _fresh_module
    calls = []

    def boom():
        raise RuntimeError("save failed")

    ws.register_shutdown_save("ok1", lambda: calls.append("ok1"))
    ws.register_shutdown_save("boom", boom)
    ws.register_shutdown_save("ok2", lambda: calls.append("ok2"))

    ran = ws.run_shutdown_saves()

    # boom is counted only on success; the two good savers still ran.
    assert ran == 2
    assert calls == ["ok1", "ok2"]


def test_no_registrations_is_a_noop(_fresh_module):
    ws = _fresh_module
    assert ws.run_shutdown_saves() == 0


def test_floor_handler_converts_sigterm_to_keyboardinterrupt(_fresh_module):
    ws = _fresh_module
    assert ws.install_worker_sigterm_handler() is True
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    # The handler must raise KeyboardInterrupt so a blocking recv loop unwinds
    # into the wrapper's finally (where saves run) instead of dying mid-loop.
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


def test_floor_handler_actually_interrupts_a_blocking_wait(_fresh_module):
    """End-to-end: a real SIGTERM raises KeyboardInterrupt in the main thread,
    exactly as it would interrupt a worker blocked in recv_queue.get()."""
    import os
    import threading
    import time

    ws = _fresh_module
    ws.install_worker_sigterm_handler()

    def _send_sigterm_soon():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=_send_sigterm_soon, daemon=True)
    t.start()
    with pytest.raises(KeyboardInterrupt):
        # Stand-in for the worker's blocking recv_queue.get(timeout=…) loop.
        for _ in range(100):
            time.sleep(0.1)
    t.join(timeout=1.0)
