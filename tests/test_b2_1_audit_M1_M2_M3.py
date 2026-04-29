"""Phase B.2.1 audit fixes M1 + M2 + M3 (2026-04-27 PM).

After the comprehensive audit identified the workers-killed-during-hibernate
bug as the last load-bearing failure, these three changes resolve it:

M1: orchestrator's _phase_hibernate splits ack collection by start_method —
    HIBERNATE_ACK from fork-mode workers, BUS_HANDOFF_ACK from spawn-mode
    workers. straggler_kill ONLY fork-mode missing.

M2: b2_1_adoption_timeout_s default 15s → 30s.

M3: supervision daemon uses min(interval, 1.0) tick during _swap_pending
    so adoption latency is bounded by ~1s instead of ~5s.

Plus: BUS_HANDOFF_ACK dst changed from "kernel" to "shadow_swap" so the
orchestrator's inbox subscription collects it.
"""
from __future__ import annotations

import inspect
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin import bus
from titan_plugin.core import worker_swap_handler
from titan_plugin.core.worker_lifecycle import WatcherState
from titan_plugin.core.worker_swap_handler import (
    SwapHandlerState,
    on_bus_handoff,
    start_supervision_thread,
)


# ── M1: BUS_HANDOFF_ACK dst is "shadow_swap" ──────────────────────────────


def test_bus_worker_adopt_request_dst_is_guardian():
    """request_adoption publishes BUS_WORKER_ADOPT_REQUEST with dst='guardian'.

    M1.5 fix (2026-04-27 PM): Guardian subscribes to bus as 'guardian'
    (guardian.py:156). Previously dst='kernel' had no in-process subscriber
    so adoption requests were silently dropped — workers reconnected to
    shadow's broker but Guardian never registered them as adopted=True.
    """
    from titan_plugin.core.worker_swap_handler import request_adoption
    state = SwapHandlerState(
        name="body",
        start_method="spawn",
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )
    state.bus_client.publish = MagicMock(return_value=True)
    state._swap_pending = True  # required for request_adoption to fire

    request_adoption(state)

    published = [c.args[0] for c in state.bus_client.publish.call_args_list
                 if c.args and c.args[0].get("type") == "BUS_WORKER_ADOPT_REQUEST"]
    assert len(published) == 1
    req = published[0]
    assert req["dst"] == "guardian", (
        f"BUS_WORKER_ADOPT_REQUEST dst must be 'guardian' so shadow Guardian's "
        f"_process_guardian_messages can handle it; got {req['dst']!r}"
    )
    assert req["src"] == "body"
    assert req["payload"]["name"] == "body"
    assert req["payload"]["start_method"] == "spawn"


def test_bus_handoff_ack_dst_is_shadow_swap():
    """on_bus_handoff publishes BUS_HANDOFF_ACK with dst='shadow_swap'.

    Before M1: dst='kernel' (silently dropped — no in-process subscriber).
    After M1: dst='shadow_swap' (matches orchestrator's inbox subscription
    so _drain_messages can collect it alongside HIBERNATE_ACK).
    """
    state = SwapHandlerState(
        name="body",
        start_method="spawn",
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )
    state.bus_client.publish = MagicMock(return_value=True)

    msg = {
        "type": bus.BUS_HANDOFF,
        "src": "kernel",
        "dst": "all",
        "payload": {"event_id": "evt_test"},
    }
    with patch.object(worker_swap_handler, "clear_parent_death_signal",
                      return_value=True):
        on_bus_handoff(state, msg)

    # Find the published BUS_HANDOFF_ACK
    published = [c.args[0] for c in state.bus_client.publish.call_args_list
                 if c.args and c.args[0].get("type") == "BUS_HANDOFF_ACK"]
    assert len(published) == 1, f"Expected 1 BUS_HANDOFF_ACK; got {len(published)}"
    ack = published[0]
    assert ack["dst"] == "shadow_swap", (
        f"BUS_HANDOFF_ACK dst must be 'shadow_swap' for orchestrator "
        f"collection; got {ack['dst']!r}"
    )
    assert ack["src"] == "body"
    assert ack["payload"]["start_method"] == "spawn"


# ── M1: orchestrator splits ack by start_method ────────────────────────────


def test_phase_hibernate_drains_both_ack_types():
    """_phase_hibernate must call _drain_messages with BOTH HIBERNATE_ACK
    and BUS_HANDOFF_ACK in the message-type set."""
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod._phase_hibernate)
    # Look for the _drain_messages call
    drain_idx = src.find("_drain_messages(")
    assert drain_idx > 0
    chunk = src[drain_idx:drain_idx + 200]
    assert "HIBERNATE_ACK" in chunk and "BUS_HANDOFF_ACK" in chunk, (
        "_phase_hibernate must drain BOTH ack types; got chunk:\n" + chunk
    )


def test_phase_hibernate_classifies_missing_per_start_method():
    """_phase_hibernate must compute spawn_missing + fork_missing separately
    based on each worker's ModuleSpec.start_method."""
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod._phase_hibernate)
    assert "spawn_expected" in src and "fork_expected" in src, (
        "_phase_hibernate must split expected workers by start_method"
    )
    assert "fork_missing" in src and "spawn_missing" in src, (
        "_phase_hibernate must compute fork_missing + spawn_missing separately"
    )


def test_phase_hibernate_only_kills_fork_mode_stragglers():
    """fast_kill must be called ONLY for fork-mode missing workers.
    Spawn-mode workers in B.2.1 swap_pending are alive and awaiting
    adoption — killing them defeats the whole graduation design."""
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod._phase_hibernate)
    fast_kill_idx = src.find("fast_kill(")
    assert fast_kill_idx > 0, "_phase_hibernate must still fast_kill stragglers"
    # The 200 chars before fast_kill should iterate over fork_missing, NOT missing
    pre_chunk = src[max(0, fast_kill_idx - 300):fast_kill_idx]
    assert "fork_missing" in pre_chunk, (
        "fast_kill must iterate over fork_missing (not the combined "
        "missing set), so spawn-mode workers stay alive"
    )


def test_phase_hibernate_emits_split_ack_event():
    """The hibernate_acks_collected event should include hibernate_acked
    and handoff_acked fields for audit traceability."""
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod._phase_hibernate)
    event_idx = src.find('result.event(\n        "hibernate_acks_collected"')
    if event_idx < 0:
        event_idx = src.find('result.event("hibernate_acks_collected"')
    assert event_idx >= 0, "_phase_hibernate must emit hibernate_acks_collected event"
    chunk = src[event_idx:event_idx + 600]
    assert "hibernate_acked" in chunk and "handoff_acked" in chunk, (
        "hibernate_acks_collected event must include both hibernate_acked + "
        "handoff_acked for audit traceability"
    )


# ── M2: adoption_wait timeout default 30s ──────────────────────────────────


def test_orchestrate_swap_adoption_timeout_default_is_30s():
    """orchestrate_shadow_swap's b2_1_adoption_timeout_s default = 30.0."""
    from titan_plugin.core.shadow_orchestrator import orchestrate_shadow_swap
    sig = inspect.signature(orchestrate_shadow_swap)
    assert sig.parameters["b2_1_adoption_timeout_s"].default == 30.0, (
        "M2: b2_1_adoption_timeout_s default must be 30.0 (was 15.0; "
        "adoption typically completes in 5-7s but 15s was tight under load)"
    )


# ── M3: supervision daemon faster tick during _swap_pending ────────────────


def test_supervision_daemon_uses_min_interval_in_swap_pending():
    """When _swap_pending=True, supervision daemon ticks at min(interval, 1.0)s
    not the lazy steady-state interval. With production interval=5.0s, fast
    interval becomes 1.0s. With test interval=0.02s, fast stays 0.02s."""
    state = SwapHandlerState(
        name="w_test",
        start_method="spawn",
        watcher_state=WatcherState(stop_event=threading.Event()),
        bus_client=MagicMock(),
    )
    state.bus_client.is_connected = False
    state.bus_client.reconnect_count = 0
    state._swap_pending = True

    # Fast tick (interval=0.05) — supervision_check called many times in 0.3s
    tick_count = {"n": 0}

    def _counting_supervision_check(s, *, now=None):
        tick_count["n"] += 1

    stop = threading.Event()
    with patch.object(worker_swap_handler, "supervision_check",
                      side_effect=_counting_supervision_check):
        t = start_supervision_thread(state, interval=0.05, stop_event=stop)
        time.sleep(0.4)
        stop.set()
        t.join(timeout=2.0)
    # 0.4s / 0.05s = ~8 ticks (>= 4 generously)
    assert tick_count["n"] >= 4, (
        f"Expected at least 4 ticks at interval=0.05s in 0.4s; "
        f"got {tick_count['n']}. M3 may have broken fast tick."
    )


def test_supervision_daemon_M3_fast_interval_is_min():
    """AST guard: M3 implementation must use min(interval, 1.0) — not a
    hardcoded 1.0 — so test fixtures with smaller intervals still work."""
    import inspect
    src = inspect.getsource(worker_swap_handler.start_supervision_thread)
    assert "min(interval, 1.0)" in src or "min(1.0, interval)" in src, (
        "M3: fast_interval must be min(interval, 1.0) so tests with "
        "tighter intervals aren't slowed down by the optimization"
    )


def test_supervision_daemon_falls_back_to_lazy_when_not_swap_pending():
    """When _swap_pending=False, the daemon should NOT use the fast tick —
    steady-state observers can stay at 5s."""
    import inspect
    src = inspect.getsource(worker_swap_handler.start_supervision_thread)
    # The _effective_interval branch should gate on state._swap_pending
    assert "_swap_pending" in src and "_effective_interval" in src, (
        "M3: effective interval must depend on state._swap_pending"
    )
