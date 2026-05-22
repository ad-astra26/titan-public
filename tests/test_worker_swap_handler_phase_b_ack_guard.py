"""Regression test for BUG-PHASE-B-OLD-WORKER-SELF-SIGTERM-ON-NAME-ALIASED-ACK-20260514.

When a Phase B reload's BUS_WORKER_ADOPT_REQUEST is rejected (e.g., NEW pid was
SIGKILL'd before Guardian could process the request), the rejection ACK
fans out via the broker to ALL subscribers with the same `name` (the
multi-name semantics from SPEC §8.2 v1.3.0). The OLD owner of the name —
which NEVER requested adoption — used to interpret this ACK as its own
rejection and self-SIGTERM, causing a cascade of false-positive worker
deaths after a botched Phase B reload.

Fix: `worker_swap_handler.on_bus_adopt_ack` now ignores ACKs when
`state._adopt_rid is None` (worker never requested adoption). Only the
worker that emitted the matching ADOPTION_REQUEST processes the ACK.
"""
from __future__ import annotations

import signal
import threading
from unittest.mock import patch, MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.core.worker_swap_handler import (
    SwapHandlerState,
    on_bus_adopt_ack,
)


def _make_state(name: str = "test_worker", start_method: str = "spawn",
                adopt_rid: str | None = None) -> SwapHandlerState:
    """Build a SwapHandlerState for testing. Uses spawn-mode by default so
    the _is_spawn_mode guard doesn't short-circuit before our code path."""
    state = SwapHandlerState(
        name=name,
        start_method=start_method,
        watcher_state=MagicMock(),
        bus_client=MagicMock(),
    )
    state._adopt_rid = adopt_rid
    return state


def test_ack_with_no_pending_adopt_request_is_ignored():
    """OLD owner of a name receives a name-aliased ACK fanout — MUST NOT
    self-SIGTERM. Pre-fix the worker self-SIGTERM'd on `status=rejected`
    because the rid-mismatch guard was bypassed when `_adopt_rid is None`."""
    state = _make_state(adopt_rid=None)
    msg = {
        "type": bus.BUS_WORKER_ADOPT_ACK,
        "rid": "some-rid-from-different-worker",
        "payload": {
            "status": "rejected",
            "reason": "pid_not_alive",
        },
    }
    with patch("os.kill") as kill_mock:
        on_bus_adopt_ack(state, msg)
        kill_mock.assert_not_called(), (
            "OLD worker must NOT self-SIGTERM on a name-aliased ACK fanout"
        )


def test_ack_with_matching_rid_processes_rejection():
    """The worker that emitted the original request MUST process its own
    rejection ACK (self-SIGTERM). Sanity-check the happy path."""
    state = _make_state(adopt_rid="my-rid-abc")
    msg = {
        "type": bus.BUS_WORKER_ADOPT_ACK,
        "rid": "my-rid-abc",
        "payload": {
            "status": "rejected",
            "reason": "test_rejection",
        },
    }
    with patch("os.kill") as kill_mock:
        on_bus_adopt_ack(state, msg)
        kill_mock.assert_called_once()
        args, kwargs = kill_mock.call_args
        # signal.SIGTERM to own pid
        assert args[1] == signal.SIGTERM


def test_ack_with_mismatched_rid_when_request_pending_is_ignored():
    """Worker DID request adoption (rid set) but receives an ACK for a
    DIFFERENT rid — must ignore (not self-SIGTERM)."""
    state = _make_state(adopt_rid="my-rid-abc")
    msg = {
        "type": bus.BUS_WORKER_ADOPT_ACK,
        "rid": "someone-elses-rid-xyz",
        "payload": {
            "status": "rejected",
            "reason": "test_rejection",
        },
    }
    with patch("os.kill") as kill_mock:
        on_bus_adopt_ack(state, msg)
        kill_mock.assert_not_called()


def test_ack_with_matching_rid_processes_adoption():
    """Worker receives ACK status="adopted" with matching rid — exits
    swap-pending state normally (no SIGTERM, _adopted flag set)."""
    state = _make_state(adopt_rid="my-rid-abc")
    state._swap_pending = True
    msg = {
        "type": bus.BUS_WORKER_ADOPT_ACK,
        "rid": "my-rid-abc",
        "payload": {
            "status": "adopted",
            "shadow_pid": 12345,
        },
    }
    # Patch resume_parent_watcher so the test doesn't try to manipulate
    # a real watcher_state thread.
    with patch("titan_hcl.core.worker_swap_handler.resume_parent_watcher"):
        with patch("os.kill") as kill_mock:
            on_bus_adopt_ack(state, msg)
            kill_mock.assert_not_called()
            assert state._adopted is True
            assert state._swap_pending is False


def test_fork_mode_ack_is_spurious_always_ignored():
    """Sanity: fork-mode workers never request adoption — ACK is always
    spurious, regardless of `_adopt_rid` setting."""
    state = _make_state(start_method="fork", adopt_rid=None)
    msg = {
        "type": bus.BUS_WORKER_ADOPT_ACK,
        "rid": "any-rid",
        "payload": {"status": "rejected", "reason": "any"},
    }
    with patch("os.kill") as kill_mock:
        on_bus_adopt_ack(state, msg)
        kill_mock.assert_not_called()
