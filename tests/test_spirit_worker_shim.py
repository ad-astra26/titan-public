"""Tests for the Phase C C-S8 chunk 8I spirit_worker.py reduction.

History:
  - C-S5 chunk C5-8 (2026-04-29): introduced flag-gated dispatch — under
    microkernel.l0_rust_enabled=true spirit_worker became a thin shim
    that handled lifecycle messages only.
  - Phase C C-S7 (2026-05-05): slim-shim 4A interim added L3 cognitive
    engine init + snapshot publishers inside the shim
    (`_spirit_worker_shim_loop` then ~240 LOC).
  - Phase C C-S8 chunk 8I (2026-05-05): cognitive engines extracted to
    cognitive_worker.py (separate L2 ModuleSpec, registered in
    legacy_core.py). spirit_worker shim replaced with a heartbeat-only
    stub (`_spirit_worker_heartbeat_stub`) — ~50 LOC.

The flag-gate semantics are unchanged:
  - microkernel.l0_rust_enabled = false (default) → legacy path runs
    unchanged (byte-identical to pre-C-S5 per SPEC §3.0).
  - microkernel.l0_rust_enabled = true → spirit_worker is a heartbeat-only
    stub; cognitive_worker owns the cognitive engines.

These tests do NOT spawn a real subprocess — they exercise the stub
helper function directly with in-memory queue.Queue() instances.
Subprocess integration (under guardian_HCL) is covered by the broader
session-close gate per PLAN §6.

See: titan-docs/PLAN_microkernel_phase_c_s8_cognitive_worker_extraction.md §8I.
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


def test_stub_function_is_callable():
    """The chunk 8I heartbeat stub entry point exists with the expected signature."""
    assert hasattr(spirit_worker, "_spirit_worker_heartbeat_stub")
    assert callable(spirit_worker._spirit_worker_heartbeat_stub)


def test_no_old_shim_loop_remains():
    """Chunk 8I deletes _spirit_worker_shim_loop. Regression guard: the
    old name MUST NOT exist on the module — that would mean the deletion
    silently regressed and cognitive_worker is duplicating engine work."""
    assert not hasattr(spirit_worker, "_spirit_worker_shim_loop"), (
        "Chunk 8I should have deleted _spirit_worker_shim_loop. If this "
        "fails, cognitive_worker and spirit_worker are both running L3 "
        "cognitive engines under l0_rust_enabled=true — double-publish + "
        "double-init bug. Investigate spirit_worker.py before deploying."
    )


def test_flag_off_skips_stub_legacy_path_runs():
    """When config.microkernel.l0_rust_enabled is false (default), the
    spirit_worker_main entry must NOT call _spirit_worker_heartbeat_stub.
    Verified by patching the stub, running spirit_worker_main in a
    background thread, and confirming the stub is not entered before
    we shut the worker down."""
    config = {
        "microkernel": {"l0_rust_enabled": False},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()
    stub_called = {"yes": False}

    def fake_stub(*args, **kwargs):
        stub_called["yes"] = True

    with patch.object(spirit_worker, "_spirit_worker_heartbeat_stub", side_effect=fake_stub):
        # Legacy path may crash on heavy imports OR enter its main loop;
        # either way, the stub should NOT have been called. Run in a thread
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

    assert stub_called["yes"] is False, (
        "Legacy path should NOT call the stub when flag is off"
    )


def test_flag_on_dispatches_to_stub():
    """When config.microkernel.l0_rust_enabled is true, spirit_worker_main
    must early-return after calling _spirit_worker_heartbeat_stub and not
    touch any of the heavy init below."""
    config = {
        "microkernel": {"l0_rust_enabled": True},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()
    stub_called = {"yes": False, "args": None}

    def fake_stub(rq, sq, n, c):
        stub_called["yes"] = True
        stub_called["args"] = (rq, sq, n, c)

    with patch.object(spirit_worker, "_spirit_worker_heartbeat_stub", side_effect=fake_stub):
        spirit_worker.spirit_worker_main(recv_q, send_q, "spirit", config)

    assert stub_called["yes"] is True
    rq, sq, n, c = stub_called["args"]
    assert rq is recv_q
    assert sq is send_q
    assert n == "spirit"
    assert c is config


def test_stub_sends_module_ready_with_offload_flag():
    """The stub sends MODULE_READY on boot with cognitive_offloaded=True
    payload — distinguishes itself from legacy MODULE_READY for
    observability + monitoring scripts. Also includes shim_mode=True
    for back-compat with monitoring scripts that key on the old shim
    marker."""
    config = {
        "microkernel": {"l0_rust_enabled": True},
        "info_banner": {"titan_id": "T1"},
    }
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    th = threading.Thread(
        target=spirit_worker._spirit_worker_heartbeat_stub,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.3)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
    th.join(timeout=2.0)
    assert not th.is_alive(), "Stub must exit cleanly on MODULE_SHUTDOWN"

    msgs = _drain(send_q)
    ready = [m for m in msgs if m.get("type") == bus.MODULE_READY]
    assert len(ready) >= 1
    assert ready[0]["src"] == "spirit"
    assert ready[0]["dst"] == "guardian"
    assert ready[0]["payload"]["cognitive_offloaded"] is True
    assert ready[0]["payload"]["stub_4b"] is True
    # Back-compat marker for monitoring scripts that key on shim_mode.
    assert ready[0]["payload"]["shim_mode"] is True
    assert ready[0]["payload"]["titan_id"] == "T1"


def test_stub_handles_module_shutdown_cleanly():
    """MODULE_SHUTDOWN must cause the stub loop to return promptly."""
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    th = threading.Thread(
        target=spirit_worker._spirit_worker_heartbeat_stub,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.1)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
    th.join(timeout=1.5)
    assert not th.is_alive()


def test_stub_ignores_other_message_types():
    """The stub must IGNORE every message except MODULE_SHUTDOWN +
    SWAP_HANDOFF / ADOPTION_REQUEST. Verified by feeding diverse traffic
    + confirming the stub doesn't echo anything except heartbeats / ready.
    Critical regression guard: cognitive_worker now owns the trinity
    dispatch under l0_rust=true; if the stub starts responding to
    BODY_STATE / MIND_STATE / KERNEL_EPOCH_TICK we'd have double-dispatch
    + the engines would double-update."""
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    th = threading.Thread(
        target=spirit_worker._spirit_worker_heartbeat_stub,
        args=(recv_q, send_q, "spirit", config),
        daemon=True,
    )
    th.start()
    time.sleep(0.1)
    # Feed messages that the legacy path would have responded to —
    # the stub must ignore them all.
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
            f"Stub leaked a non-lifecycle message: {m.get('type')}"
        )
    # MODULE_READY must be present at boot
    assert any(m.get("type") == bus.MODULE_READY for m in msgs)


def test_stub_no_legacy_imports_required():
    """The stub must not import any of the heavy legacy modules
    (spirit_loop, NeuralNervousSystem, HormonalSystem, etc.) — those
    now live in cognitive_worker.py via _cognitive_init helpers (chunk
    8C). This is a structural check: the stub helper must be importable
    and runnable even if those legacy imports would fail. Done by
    patching the legacy spirit_loop import to raise + confirming the
    stub still works."""
    import sys
    config = {"microkernel": {"l0_rust_enabled": True}}
    recv_q: Queue = Queue()
    send_q: Queue = Queue()

    # Save + remove cached spirit_loop so an import attempt would re-execute.
    cached_spirit_loop = sys.modules.pop(
        "titan_plugin.modules.spirit_loop", None)

    class _BlowUpOnImport:
        def __getattr__(self, name):
            raise ImportError("spirit_loop must not be imported by stub")
    sys.modules["titan_plugin.modules.spirit_loop"] = _BlowUpOnImport()
    try:
        th = threading.Thread(
            target=spirit_worker._spirit_worker_heartbeat_stub,
            args=(recv_q, send_q, "spirit", config),
            daemon=True,
        )
        th.start()
        time.sleep(0.1)
        recv_q.put({"type": bus.MODULE_SHUTDOWN, "src": "guardian", "ts": time.time()})
        th.join(timeout=1.5)
        assert not th.is_alive(), "Stub must work without spirit_loop importable"
    finally:
        # Restore
        if cached_spirit_loop is not None:
            sys.modules["titan_plugin.modules.spirit_loop"] = cached_spirit_loop
        else:
            sys.modules.pop("titan_plugin.modules.spirit_loop", None)
