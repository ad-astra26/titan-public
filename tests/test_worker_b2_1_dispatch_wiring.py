"""Phase B.2.1 chunk C2 — worker entry_fn dispatch wiring tests.

Two layers of verification:

  1. **Behavioral** — `maybe_dispatch_swap_msg` correctly routes the three
     B.2.1 supervision-transfer message types to the right handler against
     the active SwapHandlerState, and returns False (no-op) for unrelated
     messages or when no state is registered.

  2. **AST drift guard** — every `*_worker_main` entry_fn in
     titan_plugin/modules/ contains a call to
     `worker_swap_handler.maybe_dispatch_swap_msg(...)` inside its body.
     Catches future regressions where someone forgets to wire B.2.1 into
     a new worker.

Wiring contract (per worker, after recv_queue.get → msg, after _b1_reporter
dispatch, before worker-specific elif chain):

    from titan_plugin.core import worker_swap_handler as _swap
    if _swap.maybe_dispatch_swap_msg(msg):
        continue
"""
from __future__ import annotations

import ast
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin import bus
from titan_plugin.core import worker_swap_handler
from titan_plugin.core.worker_lifecycle import WatcherState
from titan_plugin.core.worker_swap_handler import (
    SwapHandlerState,
    maybe_dispatch_swap_msg,
    set_active_swap_state,
)


def _make_state(start_method: str = "spawn") -> SwapHandlerState:
    return SwapHandlerState(
        name="w_test",
        start_method=start_method,
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )


# ── maybe_dispatch_swap_msg behaviour ────────────────────────────────────


def test_dispatch_handoff_routes_to_on_bus_handoff():
    state = _make_state()
    set_active_swap_state(state)
    try:
        msg = {"type": bus.BUS_HANDOFF, "payload": {"event_id": "e1"}}
        with patch.object(worker_swap_handler, "on_bus_handoff") as h:
            assert maybe_dispatch_swap_msg(msg) is True
        h.assert_called_once_with(state, msg)
    finally:
        set_active_swap_state(None)


def test_dispatch_adopt_ack_routes_to_on_bus_adopt_ack():
    state = _make_state()
    set_active_swap_state(state)
    try:
        msg = {"type": bus.BUS_WORKER_ADOPT_ACK,
               "payload": {"status": "adopted"}, "rid": "r1"}
        with patch.object(worker_swap_handler, "on_bus_adopt_ack") as h:
            assert maybe_dispatch_swap_msg(msg) is True
        h.assert_called_once_with(state, msg)
    finally:
        set_active_swap_state(None)


def test_dispatch_canceled_routes_to_on_bus_handoff_canceled():
    state = _make_state()
    set_active_swap_state(state)
    try:
        msg = {"type": bus.BUS_HANDOFF_CANCELED, "payload": {}}
        with patch.object(worker_swap_handler, "on_bus_handoff_canceled") as h:
            assert maybe_dispatch_swap_msg(msg) is True
        h.assert_called_once_with(state, msg)
    finally:
        set_active_swap_state(None)


def test_dispatch_other_messages_returns_false():
    state = _make_state()
    set_active_swap_state(state)
    try:
        for mtype in ("SPHERE_PULSE", "MODULE_SHUTDOWN", "QUERY",
                      "BUS_HANDOFF_ACK", "MODULE_READY", ""):
            msg = {"type": mtype}
            assert maybe_dispatch_swap_msg(msg) is False, mtype
    finally:
        set_active_swap_state(None)


def test_dispatch_no_state_returns_false():
    set_active_swap_state(None)
    msg = {"type": bus.BUS_HANDOFF, "payload": {}}
    # No state registered → must not crash, must return False
    assert maybe_dispatch_swap_msg(msg) is False


def test_dispatch_missing_type_field_safe():
    state = _make_state()
    set_active_swap_state(state)
    try:
        # Malformed message — no "type" key. Must not crash.
        assert maybe_dispatch_swap_msg({}) is False
        assert maybe_dispatch_swap_msg({"payload": {}}) is False
    finally:
        set_active_swap_state(None)


# ── AST drift guard: every *_worker_main has the dispatch ────────────────


WORKERS_DIR = (
    Path(__file__).parent.parent / "titan_plugin" / "modules"
)

EXPECTED_WORKERS = {
    "body_worker.py", "mind_worker.py", "llm_worker.py", "cgn_worker.py",
    "meta_teacher_worker.py", "media_worker.py", "language_worker.py",
    "timechain_worker.py", "rl_worker.py", "memory_worker.py",
    "emot_cgn_worker.py", "backup_worker.py", "knowledge_worker.py",
    "spirit_worker.py", "warning_monitor_worker.py",
}


def _entry_fn_calls_swap_dispatch(tree: ast.AST, fn_name: str) -> bool:
    """Walk ast.FunctionDef body looking for a Call to maybe_dispatch_swap_msg.

    Matches any of:
      worker_swap_handler.maybe_dispatch_swap_msg(...)
      _swap.maybe_dispatch_swap_msg(...)
      maybe_dispatch_swap_msg(...)         (direct import alias)
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == fn_name):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if isinstance(f, ast.Attribute) and f.attr == "maybe_dispatch_swap_msg":
                return True
            if isinstance(f, ast.Name) and f.id == "maybe_dispatch_swap_msg":
                return True
    return False


@pytest.mark.parametrize("worker_file", sorted(EXPECTED_WORKERS))
def test_every_worker_has_b2_1_dispatch(worker_file: str):
    """Each *_worker_main MUST call maybe_dispatch_swap_msg in its body.

    Uniform dispatch is the contract that B.2.1 supervision-transfer rests on:
    if a worker doesn't dispatch BUS_HANDOFF / BUS_WORKER_ADOPT_ACK /
    BUS_HANDOFF_CANCELED through the swap handler, it can't outlive a swap.
    """
    path = WORKERS_DIR / worker_file
    assert path.exists(), f"Expected worker file missing: {worker_file}"
    tree = ast.parse(path.read_text())
    fn_name = worker_file.replace(".py", "_main")
    assert _entry_fn_calls_swap_dispatch(tree, fn_name), (
        f"{worker_file}::{fn_name} is missing the B.2.1 dispatch call. "
        f"Add `if _swap.maybe_dispatch_swap_msg(msg): continue` after the "
        f"_b1_reporter dispatch block."
    )


def test_worker_inventory_has_no_stragglers():
    """Sanity check: no *_worker.py files exist that aren't in EXPECTED_WORKERS.

    Catches the case where a NEW worker module is added to the modules
    directory but the test inventory + the wiring is forgotten.
    """
    found = {p.name for p in WORKERS_DIR.glob("*_worker.py")}
    extra = found - EXPECTED_WORKERS
    missing = EXPECTED_WORKERS - found
    assert not extra, (
        f"Worker files exist that are not in EXPECTED_WORKERS — they may "
        f"be unwired: {sorted(extra)}. Add to EXPECTED_WORKERS + wire "
        f"B.2.1 dispatch in each."
    )
    assert not missing, f"Expected workers not found: {sorted(missing)}"
