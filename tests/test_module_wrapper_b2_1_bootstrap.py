"""Phase B.2.1 chunk C1 — Guardian._module_wrapper centralization tests.

Verifies the per-process B.2.1 bootstrap path:

  1. set_active_swap_state / get_active_swap_state work as a process global
  2. start_supervision_thread for spawn-mode starts a daemon that:
        • ticks supervision_check
        • detects bus_client.reconnect_count increase + is_connected=True
          while _swap_pending and calls request_adoption
        • exits cleanly when state._adopted=True
        • survives transient errors in supervision_check
  3. start_supervision_thread for fork-mode returns a no-op daemon
  4. _module_wrapper invokes setup_worker_bus, registers SwapHandlerState
     when bus_client is non-None, falls through to legacy mp.Queue mode
     gracefully when env vars are absent
  5. _module_wrapper passes start_method through to SwapHandlerState
  6. _module_wrapper finally-block stops bus_client + clears active state

Tests use mocks for setup_worker_bus + bus_client so they don't depend on
a live broker. Real-subprocess swap behaviour is covered by chunk C4 and
test_b2_1_supervision_transfer_e2e.py.
"""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin.core import worker_swap_handler
from titan_plugin.core.worker_lifecycle import WatcherState
from titan_plugin.core.worker_swap_handler import (
    SwapHandlerState,
    get_active_swap_state,
    set_active_swap_state,
    start_supervision_thread,
)
from titan_plugin.guardian import _module_wrapper


# ── Active-state helpers (process global) ────────────────────────────────


def test_set_get_active_swap_state_round_trip():
    set_active_swap_state(None)
    assert get_active_swap_state() is None

    state = SwapHandlerState(
        name="w1", start_method="spawn",
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )
    try:
        set_active_swap_state(state)
        assert get_active_swap_state() is state
    finally:
        set_active_swap_state(None)


def test_set_active_swap_state_clears_to_none():
    state = SwapHandlerState(
        name="w1", start_method="spawn",
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )
    set_active_swap_state(state)
    assert get_active_swap_state() is state

    set_active_swap_state(None)
    assert get_active_swap_state() is None


# ── Supervision daemon thread ────────────────────────────────────────────


def _make_state(
    *, start_method: str = "spawn",
    is_connected: bool = False,
    reconnect_count: int = 0,
    swap_pending: bool = False,
    adopted: bool = False,
) -> SwapHandlerState:
    client = MagicMock()
    client.is_connected = is_connected
    client.reconnect_count = reconnect_count
    state = SwapHandlerState(
        name="w_test",
        start_method=start_method,
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=client,
    )
    state._swap_pending = swap_pending
    state._adopted = adopted
    return state


def test_start_supervision_thread_fork_mode_is_noop():
    state = _make_state(start_method="fork")
    t = start_supervision_thread(state, interval=0.05)
    t.join(timeout=1.0)
    # Fork-mode daemon is no-op; thread exits immediately.
    assert not t.is_alive()
    assert state._swap_pending is False  # untouched


def test_start_supervision_thread_spawn_mode_is_daemon():
    state = _make_state(start_method="spawn")
    stop = threading.Event()
    t = start_supervision_thread(state, interval=0.05, stop_event=stop)
    try:
        assert t.is_alive()
        assert t.daemon is True
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not t.is_alive()


def test_supervision_thread_calls_request_adoption_on_reconnect():
    state = _make_state(
        start_method="spawn",
        is_connected=False,
        reconnect_count=0,
        swap_pending=True,
    )
    stop = threading.Event()
    with patch.object(worker_swap_handler, "request_adoption") as mock_req:
        t = start_supervision_thread(state, interval=0.02, stop_event=stop)
        try:
            time.sleep(0.1)
            # Simulate broker reconnect: count++ AND is_connected→True
            state.bus_client.reconnect_count = 1
            state.bus_client.is_connected = True
            time.sleep(0.15)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert mock_req.called, "request_adoption should fire on reconnect bump"
        called_with_state = mock_req.call_args[0][0]
        assert called_with_state is state


def test_supervision_thread_no_adoption_when_not_swap_pending():
    state = _make_state(
        start_method="spawn",
        is_connected=False,
        reconnect_count=0,
        swap_pending=False,  # not in swap window
    )
    stop = threading.Event()
    with patch.object(worker_swap_handler, "request_adoption") as mock_req:
        t = start_supervision_thread(state, interval=0.02, stop_event=stop)
        try:
            time.sleep(0.05)
            state.bus_client.reconnect_count = 1
            state.bus_client.is_connected = True
            time.sleep(0.1)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert not mock_req.called, (
            "request_adoption must NOT fire outside swap window"
        )


def test_supervision_thread_exits_on_adopted():
    state = _make_state(start_method="spawn", swap_pending=True)
    stop = threading.Event()
    t = start_supervision_thread(state, interval=0.02, stop_event=stop)
    try:
        time.sleep(0.05)
        # Simulate ADOPT_ACK — daemon should exit on next tick
        state._adopted = True
        time.sleep(0.1)
        assert not t.is_alive(), "Daemon should exit when _adopted=True"
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_supervision_thread_survives_tick_errors():
    state = _make_state(start_method="spawn", swap_pending=True)
    stop = threading.Event()

    call_count = {"n": 0}

    def _flaky_supervision_check(s: SwapHandlerState, *, now: Any = None) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient broker hiccup")

    with patch.object(worker_swap_handler, "supervision_check",
                      side_effect=_flaky_supervision_check):
        t = start_supervision_thread(state, interval=0.02, stop_event=stop)
        try:
            time.sleep(0.15)
            assert t.is_alive(), "Daemon must survive transient tick error"
            assert call_count["n"] >= 2, "Should retry after the error"
        finally:
            stop.set()
            t.join(timeout=2.0)


# ── _module_wrapper centralization ───────────────────────────────────────


class _RecordingEntry:
    """Captures the exact (recv, send, name, config) entry_fn was called with."""

    def __init__(self) -> None:
        self.called_with: tuple | None = None

    def __call__(self, recv_q, send_q, name, config) -> None:
        self.called_with = (recv_q, send_q, name, config)


def test_module_wrapper_passes_start_method_to_swap_state():
    entry = _RecordingEntry()
    fake_recv, fake_send = object(), object()
    fake_client = MagicMock()
    fake_client.is_connected = False
    fake_client.reconnect_count = 0
    captured = {}

    def fake_setup(name, recv_q, send_q):
        return fake_recv, fake_send, fake_client

    def fake_start_thread(state, *, interval=5.0, stop_event=None):
        captured["state"] = state
        # Return a pre-stopped daemon so wrapper exits cleanly
        ev = threading.Event()
        ev.set()
        t = threading.Thread(target=lambda: ev.wait(0), daemon=True)
        t.start()
        return t

    set_active_swap_state(None)
    with patch("titan_plugin.core.worker_bus_bootstrap.setup_worker_bus",
               side_effect=fake_setup), \
         patch.object(worker_swap_handler, "start_supervision_thread",
                      side_effect=fake_start_thread):
        _module_wrapper(entry, "w_test", object(), object(), {"k": 1},
                        start_method="spawn")
    assert captured["state"].start_method == "spawn"
    assert captured["state"].name == "w_test"
    # entry_fn must receive the rebound queues from setup_worker_bus
    assert entry.called_with == (fake_recv, fake_send, "w_test", {"k": 1})


def test_module_wrapper_legacy_mode_no_swap_state():
    entry = _RecordingEntry()
    orig_recv, orig_send = object(), object()

    def fake_setup(name, recv_q, send_q):
        # Legacy mode — env vars absent → return original queues + None client
        return recv_q, send_q, None

    set_active_swap_state(None)
    with patch("titan_plugin.core.worker_bus_bootstrap.setup_worker_bus",
               side_effect=fake_setup):
        _module_wrapper(entry, "w_legacy", orig_recv, orig_send, {},
                        start_method="fork")
    # Legacy: no swap state registered; original queues passed through
    assert get_active_swap_state() is None
    assert entry.called_with[0] is orig_recv
    assert entry.called_with[1] is orig_send


def test_module_wrapper_clears_active_state_on_exit():
    entry = _RecordingEntry()
    fake_client = MagicMock()
    fake_client.is_connected = False
    fake_client.reconnect_count = 0

    def fake_setup(name, recv_q, send_q):
        return object(), object(), fake_client

    def fake_start_thread(state, **kw):
        ev = threading.Event(); ev.set()
        t = threading.Thread(target=lambda: None, daemon=True); t.start()
        return t

    with patch("titan_plugin.core.worker_bus_bootstrap.setup_worker_bus",
               side_effect=fake_setup), \
         patch.object(worker_swap_handler, "start_supervision_thread",
                      side_effect=fake_start_thread):
        _module_wrapper(entry, "w_clean", object(), object(), {},
                        start_method="spawn")
    assert get_active_swap_state() is None, (
        "_module_wrapper finally-block must clear active swap state"
    )
    fake_client.stop.assert_called_once()


def test_module_wrapper_handles_setup_failure_gracefully():
    entry = _RecordingEntry()
    orig_recv, orig_send = object(), object()

    def boom(name, recv_q, send_q):
        raise RuntimeError("simulated setup_worker_bus failure")

    with patch("titan_plugin.core.worker_bus_bootstrap.setup_worker_bus",
               side_effect=boom):
        # Must NOT crash; falls through to entry_fn with original queues
        _module_wrapper(entry, "w_fail", orig_recv, orig_send, {},
                        start_method="spawn")
    assert entry.called_with == (orig_recv, orig_send, "w_fail", {})
    assert get_active_swap_state() is None


def test_module_wrapper_default_start_method_is_fork():
    """Backward compat: pre-B.2.1 callers (none today, but tests etc.) get fork."""
    entry = _RecordingEntry()

    def fake_setup(name, recv_q, send_q):
        return recv_q, send_q, None  # legacy mode → no swap state path

    with patch("titan_plugin.core.worker_bus_bootstrap.setup_worker_bus",
               side_effect=fake_setup):
        _module_wrapper(entry, "w_default", object(), object(), {})
        # No exception = pass; default kw works
    assert entry.called_with is not None


# ── Guardian spawn-site arity ────────────────────────────────────────────


def test_guardian_spawn_site_passes_start_method():
    """Drift guard: Guardian's ctx.Process(args=...) must include start_method.

    AST-walks guardian.py to find the args= literal in the spawn site
    and verifies info.spec.start_method appears as the 6th element.
    Catches any future regression that drops the parameter.
    """
    import ast
    import inspect

    import titan_plugin.guardian as guardian_mod

    src = inspect.getsource(guardian_mod)
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "Process"):
            continue
        # find args=(...) keyword
        for kw in node.keywords:
            if kw.arg != "args":
                continue
            if not isinstance(kw.value, ast.Tuple):
                continue
            elts = kw.value.elts
            if len(elts) != 6:
                continue
            # 6th element must be info.spec.start_method
            sixth = elts[5]
            if (isinstance(sixth, ast.Attribute)
                    and sixth.attr == "start_method"):
                found = True
                break
        if found:
            break

    assert found, (
        "Guardian's ctx.Process(...) call must pass info.spec.start_method "
        "as the 6th element of args=(...). _module_wrapper requires it."
    )
