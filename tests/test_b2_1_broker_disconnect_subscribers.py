"""Phase B.2.1 (2026-04-27 PM) — broker disconnect_subscribers + orchestrator wiring.

Closes the architectural gap where graduated workers acked BUS_HANDOFF +
entered swap_pending, but never reconnected to shadow's broker because
their existing FD connections to the OLD kernel's broker persisted
(TCP-like) until old kernel exited. Adoption_wait timed out with adopted=[].

Fix: BusSocketServer.disconnect_subscribers(names) actively closes
specific subscriber connections. Orchestrator calls it after shadow_health_ok
+ before _phase_b2_1_wait_adoption, targeting all spawn-mode expected
workers. Workers' BusSocketClient detects EOF → reconnects → lands on
shadow's broker (now bound on same socket path) → supervision daemon
observes reconnect_count++ → calls request_adoption.
"""
from __future__ import annotations

import inspect
import threading
from unittest.mock import MagicMock

import pytest

from titan_plugin.core.bus_socket import BrokerSubscriber, BusSocketServer


# ── BusSocketServer.disconnect_subscribers ──────────────────────────────


def _make_fake_subscriber(name: str) -> BrokerSubscriber:
    """Build a subscriber object with a mock conn so close() doesn't bomb."""
    conn = MagicMock()
    sub = BrokerSubscriber.__new__(BrokerSubscriber)
    sub.name = name
    sub.conn = conn
    sub.closed = False
    sub.has_data_event = threading.Event()
    sub.lock = threading.Lock()
    sub.subscribed_topics = set()
    sub.last_pong_ts = 0.0
    sub.last_warning_ts = 0.0
    sub.drop_count_60s = 0
    sub.recv_count_60s = 0
    sub.last_window_reset_ts = 0.0
    sub.coalesce_index = {}
    sub.ring = MagicMock()
    return sub


def test_disconnect_subscribers_closes_named_connections():
    server = BusSocketServer(titan_id="T_test", authkey=b"x" * 32)
    sub_a = _make_fake_subscriber("body")
    sub_b = _make_fake_subscriber("mind")
    sub_c = _make_fake_subscriber("imw")  # NOT in target list — must stay
    server._subscribers = {"body": sub_a, "mind": sub_b, "imw": sub_c}

    purged = server.disconnect_subscribers(["body", "mind"], reason="test_handoff")
    assert sorted(purged) == ["body", "mind"]
    assert sub_a.closed is True
    assert sub_b.closed is True
    assert sub_c.closed is False, "untouched subs must NOT be closed"
    sub_a.conn.close.assert_called()
    sub_b.conn.close.assert_called()
    sub_c.conn.close.assert_not_called()


def test_disconnect_subscribers_skips_unknown_names_silently():
    server = BusSocketServer(titan_id="T_test", authkey=b"x" * 32)
    sub_a = _make_fake_subscriber("body")
    server._subscribers = {"body": sub_a}

    purged = server.disconnect_subscribers(
        ["body", "nonexistent_worker"], reason="test",
    )
    assert purged == ["body"], "unknown names are silent no-ops"
    assert sub_a.closed is True


def test_disconnect_subscribers_empty_list_is_safe():
    server = BusSocketServer(titan_id="T_test", authkey=b"x" * 32)
    sub_a = _make_fake_subscriber("body")
    server._subscribers = {"body": sub_a}

    purged = server.disconnect_subscribers([], reason="test")
    assert purged == []
    assert sub_a.closed is False


def test_disconnect_subscribers_removes_from_registry():
    """After disconnect, the subscriber must be GONE from server._subscribers
    so subsequent queries don't try to deliver to a dead connection."""
    server = BusSocketServer(titan_id="T_test", authkey=b"x" * 32)
    sub_a = _make_fake_subscriber("body")
    server._subscribers = {"body": sub_a}

    server.disconnect_subscribers(["body"], reason="test")
    assert "body" not in server._subscribers


# ── Orchestrator wiring: disconnect before adoption_wait ────────────────


def test_orchestrator_calls_disconnect_subscribers_before_adoption_wait():
    """AST guard: orchestrate_shadow_swap MUST call disconnect_subscribers
    on the broker BETWEEN shadow_health_ok and _phase_b2_1_wait_adoption.

    Without this, workers stay connected to old kernel's broker and never
    reconnect to shadow → adoption_wait times out with adopted=[].
    """
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod.orchestrate_shadow_swap)

    disconnect_idx = src.find("disconnect_subscribers")
    adoption_idx = src.find("_phase_b2_1_wait_adoption")
    assert disconnect_idx >= 0, (
        "orchestrate_shadow_swap MUST call broker.disconnect_subscribers "
        "to force workers off old kernel's broker"
    )
    assert adoption_idx >= 0, "orchestrate_shadow_swap calls _phase_b2_1_wait_adoption"
    assert disconnect_idx < adoption_idx, (
        "disconnect_subscribers MUST be called BEFORE _phase_b2_1_wait_adoption "
        f"(found disconnect at offset {disconnect_idx}, adoption at {adoption_idx})"
    )


def test_orchestrator_disconnect_targets_spawn_mode_only():
    """The orchestrator's disconnect call must filter to spawn-mode workers.

    Fork-mode workers were already straggler-killed (HIBERNATE flow) and
    don't have live broker connections. Disconnecting them is a no-op
    but clutters the audit log.
    """
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod.orchestrate_shadow_swap)

    # Look for the spawn-mode filter near the disconnect call
    chunk_start = src.find("disconnect_subscribers")
    assert chunk_start > 0
    # The orchestrator's filter logic should set up spawn-mode names somewhere
    # within +/- 600 chars of the disconnect call.
    window = src[max(0, chunk_start - 200):chunk_start + 600]
    assert 'start_method' in window and '"spawn"' in window, (
        "orchestrator must filter disconnect targets to spawn-mode workers"
    )


def test_orchestrator_disconnect_uses_b2_1_handoff_reason():
    """Audit trail: the disconnect reason should mark this as a B.2.1 handoff
    (not a generic close) so log forensics can distinguish swap-time
    disconnects from other purge events (pong_timeout, server_stop, etc.)."""
    import titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod.orchestrate_shadow_swap)
    chunk_start = src.find("disconnect_subscribers")
    chunk = src[chunk_start:chunk_start + 500]
    assert "b2_1_handoff" in chunk, (
        "disconnect_subscribers should be called with reason='b2_1_handoff' "
        "for audit traceability"
    )
