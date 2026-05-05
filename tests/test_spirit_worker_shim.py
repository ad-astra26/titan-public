"""Tests for the C-S5 chunk C5-8 spirit_worker.py reduction.

Verifies the flag-gated dispatch:
  - microkernel.l0_rust_enabled = false (default) → legacy path below
    runs unchanged (byte-identical to today per SPEC §3.0).
  - microkernel.l0_rust_enabled = true → spirit_worker becomes a thin
    shim that only handles lifecycle messages.

These tests do NOT spawn a real subprocess — they exercise the shim
helper function directly with in-memory queue.Queue() instances.
Subprocess integration (under guardian_HCL) is covered by the broader
session-close gate per PLAN §6.

See: titan-docs/PLAN_microkernel_phase_c_l0_l1_rust.md §10.5 chunk C5-8 +
     titan-docs/PLAN_microkernel_phase_c_s5_inner_trinity.md §4.8.
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from unittest.mock import patch

import pytest

from titan_plugin import bus
from titan_plugin.modules import spirit_worker


def _drain(q: Queue, max_msgs: int = 100) -> list[dict]:
    """Drain a queue without blocking. Returns up to max_msgs items."""
    out = []
    for _ in range(max_msgs):
        try:
            out.append(q.get_nowait())
        except Exception:
            break
    return out


def test_shim_loop_function_is_callable():
    """The C5-8 shim entry point exists with the expected signature."""
    assert hasattr(spirit_worker, "_spirit_worker_shim_loop")
    assert callable(spirit_worker._spirit_worker_shim_loop)


def test_flag_off_skips_shim_legacy_path_runs():
    """When config.microkernel.l0_rust_enabled is false (default), the
    spirit_worker_main entry must NOT call _spirit_worker_shim_loop.
    Verified by patching the shim, running spirit_worker_main in a
    background thread, and confirming the shim is not entered before
    we shut the worker down."""
    config = {
        "microkernel": {"l0_rust_enabled": False},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()
    shim_called = {"yes": False}

    def fake_shim(*args, **kwargs):
        shim_called["yes"] = True

    with patch.object(spirit_worker, "_spirit_worker_shim_loop", side_effect=fake_shim):
        # Legacy path may crash on heavy imports OR enter its main loop;
        # either way, the shim should NOT have been called. Run in a thread
        # so a hung loop doesn't hang the test.
        # Pre-load MODULE_SHUTDOWN so if the legacy main loop is reached
        # it exits cleanly.
        recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})

        def _run():
            try:
                spirit_worker.spirit_worker_main(recv_q, send_q, "spirit", config)
            except Exception:
                pass

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout=3.0)
        # Whether the legacy path crashed or completed cleanly, the shim
        # must NOT have been entered. (We don't assert on whether the
        # thread is alive since the legacy path may legitimately be
        # blocked on heavy initialization that won't complete in CI —
        # the daemon=True flag ensures it dies with the test process.)

    assert shim_called["yes"] is False, (
        "Legacy path should NOT call the shim when flag is off"
    )


def test_flag_on_dispatches_to_shim():
    """When config.microkernel.l0_rust_enabled is true, spirit_worker_main
    must early-return after calling _spirit_worker_shim_loop and not
    touch any of the heavy init below."""
    config = {
        "microkernel": {"l0_rust_enabled": True},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()
    shim_called = {"yes": False, "args": None}

    def fake_shim(rq, sq, n, c):
        shim_called["yes"] = True
        shim_called["args"] = (rq, sq, n, c)

    with patch.object(spirit_worker, "_spirit_worker_shim_loop", side_effect=fake_shim):
        spirit_worker.spirit_worker_main(recv_q, send_q, "spirit", config)

    assert shim_called["yes"] is True
    rq, sq, n, c = shim_called["args"]
    assert rq is recv_q
    assert sq is send_q
    assert n == "spirit"
    assert c is config


def test_shim_sends_module_ready_with_shim_mode_flag():
    """The shim sends MODULE_READY on boot with shim_mode=True payload —
    distinguishes itself from legacy MODULE_READY for observability."""
    config = {
        "microkernel": {"l0_rust_enabled": True},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    # Run shim in background; signal shutdown after small delay.
    th = threading.Thread(
        target=spirit_worker._spirit_worker_shim_loop,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.3)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
    th.join(timeout=2.0)
    assert not th.is_alive(), "Shim must exit cleanly on MODULE_SHUTDOWN"

    msgs = _drain(send_q)
    ready = [m for m in msgs if m.get("type") == bus.MODULE_READY]
    assert len(ready) >= 1
    assert ready[0]["src"] == "spirit"
    assert ready[0]["dst"] == "guardian"
    assert ready[0]["payload"]["shim_mode"] is True
    assert ready[0]["payload"]["titan_id"] == "T1"


def test_shim_handles_module_shutdown_cleanly():
    """MODULE_SHUTDOWN must cause the shim loop to return promptly."""
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    th = threading.Thread(
        target=spirit_worker._spirit_worker_shim_loop,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.1)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
    th.join(timeout=1.5)
    assert not th.is_alive()


def test_shim_ignores_other_message_types():
    """The shim must IGNORE every message except MODULE_SHUTDOWN +
    SWAP_HANDOFF / ADOPTION_REQUEST. Verified by feeding diverse traffic
    + confirming the shim doesn't echo anything except heartbeats / ready."""
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    th = threading.Thread(
        target=spirit_worker._spirit_worker_shim_loop,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.1)
    # Feed messages that the legacy path would have responded to —
    # the shim must ignore them all.
    for msg_type in (
        getattr(bus, "QUERY", "QUERY"),
        getattr(bus, "BODY_STATE", "BODY_STATE"),
        getattr(bus, "MIND_STATE", "MIND_STATE"),
        getattr(bus, "KERNEL_EPOCH_TICK", "KERNEL_EPOCH_TICK"),
    ):
        recv_q.put({"type": msg_type, "src": "test", "payload": {}, "ts": time.time()})
    time.sleep(0.4)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
    th.join(timeout=2.0)
    assert not th.is_alive()

    msgs = _drain(send_q)
    # Only MODULE_READY (+ possibly MODULE_HEARTBEAT) should appear in send_q.
    allowed_types = {bus.MODULE_READY, bus.MODULE_HEARTBEAT}
    for m in msgs:
        assert m.get("type") in allowed_types, (
            f"Shim leaked a non-lifecycle message: {m.get('type')}"
        )
    # MODULE_READY must be present at boot
    assert any(m.get("type") == bus.MODULE_READY for m in msgs)


def test_shim_no_legacy_imports_required():
    """The shim must not import any of the heavy legacy modules
    (spirit_loop, NeuralNervousSystem, HormonalSystem, etc.). This is a
    structural check: the shim helper must be importable and runnable
    even if those legacy imports would fail. Done by patching the legacy
    spirit_loop import to raise + confirming the shim still works."""
    import sys
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    # Save + remove cached spirit_loop so an import attempt would re-execute.
    cached_spirit_loop = sys.modules.pop(
        "titan_plugin.modules.spirit_loop", None)

    class _BlowUpOnImport:
        def __getattr__(self, name):
            raise ImportError("spirit_loop must not be imported by shim")
    sys.modules["titan_plugin.modules.spirit_loop"] = _BlowUpOnImport()
    try:
        th = threading.Thread(
            target=spirit_worker._spirit_worker_shim_loop,
            args=(recv_q, send_q, "spirit", config),
            daemon=True,
        )
        th.start()
        time.sleep(0.1)
        recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
        th.join(timeout=1.5)
        assert not th.is_alive(), "Shim must work without spirit_loop importable"
    finally:
        # Restore
        if cached_spirit_loop is not None:
            sys.modules["titan_plugin.modules.spirit_loop"] = cached_spirit_loop
        else:
            sys.modules.pop("titan_plugin.modules.spirit_loop", None)
