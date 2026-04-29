"""
Tests for Phase B.2 shadow_orchestrator integration (C8).

Covers:
- HealthCriteria has the 2 new B.2 fields with safe defaults (gate disabled)
- _check_bus_broker_criteria returns None when both gates disabled
- _check_bus_broker_criteria reports broker stats absence as a failure
- _check_bus_broker_criteria passes when subscriber count + drop rate within bounds
- _check_bus_broker_criteria fails when subscriber count below threshold
- _check_bus_broker_criteria fails when worst drop rate exceeds threshold
- BUS_HANDOFF emitted in Phase 2 (between SYSTEM_UPGRADE_STARTING and HIBERNATE)

These are unit-level: we don't spin up a real shadow swap (that's deploy-time
integration). The goal here is to lock the contract — criteria fields exist,
check stages produce correct verdicts, BUS_HANDOFF is in the orchestrator's
publish stream.
"""
from __future__ import annotations

import dataclasses

import pytest

from titan_plugin.core.shadow_orchestrator import (
    HealthCriteria,
    _check_bus_broker_criteria,
)


# ── HealthCriteria has the new fields ──────────────────────────────────────


def test_health_criteria_has_b2_fields():
    fields = {f.name for f in dataclasses.fields(HealthCriteria)}
    assert "min_connected_bus_workers" in fields
    assert "max_bus_drop_rate_60s_pct" in fields


def test_health_criteria_b2_defaults_disable_gate():
    """Defaults must keep the gate disabled so Phase B.1-only kernels still pass."""
    c = HealthCriteria()
    assert c.min_connected_bus_workers == 0
    assert c.max_bus_drop_rate_60s_pct == 100.0


# ── _check_bus_broker_criteria behavior ────────────────────────────────────


def test_bus_check_disabled_returns_none():
    """Both gates at defaults → returns None (skip the check entirely)."""
    state = {"data": {"bus_broker": {"subscriber_count": 0, "subscribers": []}}}
    out = _check_bus_broker_criteria(state, HealthCriteria())
    assert out is None


def test_bus_check_active_but_no_broker_stats_fails():
    """Gate active but kernel reports no broker → diagnostic config-mismatch error."""
    state = {"data": {}}  # no bus_broker key
    crit = HealthCriteria(min_connected_bus_workers=3)
    out = _check_bus_broker_criteria(state, crit)
    assert out is not None
    assert out["pass"] is False
    assert "absent" in out.get("error", "")


def test_bus_check_passes_with_enough_subscribers_and_clean_drop_rate():
    state = {
        "data": {
            "bus_broker": {
                "subscriber_count": 14,
                "subscribers": [
                    {"name": f"w{i}", "recv_count_60s": 100, "drop_count_60s": 1}
                    for i in range(14)
                ],
            }
        }
    }
    crit = HealthCriteria(min_connected_bus_workers=14,
                          max_bus_drop_rate_60s_pct=5.0)
    out = _check_bus_broker_criteria(state, crit)
    assert out["pass"] is True
    assert out["subscriber_count"] == 14
    assert out["worst_drop_rate_pct"] == 1.0  # 1/100


def test_bus_check_fails_when_subscriber_count_short():
    state = {
        "data": {
            "bus_broker": {
                "subscriber_count": 10,
                "subscribers": [],
            }
        }
    }
    crit = HealthCriteria(min_connected_bus_workers=14)
    out = _check_bus_broker_criteria(state, crit)
    assert out["pass"] is False
    assert out["subscriber_count"] == 10
    assert out["min_required"] == 14


def test_bus_check_fails_on_high_drop_rate():
    state = {
        "data": {
            "bus_broker": {
                "subscriber_count": 14,
                "subscribers": [
                    {"name": "w_slow", "recv_count_60s": 100, "drop_count_60s": 30},
                    {"name": "w_clean", "recv_count_60s": 100, "drop_count_60s": 1},
                ],
            }
        }
    }
    crit = HealthCriteria(min_connected_bus_workers=2,
                          max_bus_drop_rate_60s_pct=5.0)
    out = _check_bus_broker_criteria(state, crit)
    assert out["pass"] is False
    assert out["worst_drop_rate_pct"] == 30.0
    assert out["worst_subscriber"] == "w_slow"


def test_bus_check_handles_zero_recv_safely():
    """Subscriber with zero recv_count_60s must not divide-by-zero."""
    state = {
        "data": {
            "bus_broker": {
                "subscriber_count": 1,
                "subscribers": [
                    {"name": "w", "recv_count_60s": 0, "drop_count_60s": 0},
                ],
            }
        }
    }
    crit = HealthCriteria(min_connected_bus_workers=1,
                          max_bus_drop_rate_60s_pct=5.0)
    out = _check_bus_broker_criteria(state, crit)
    # 0/1=0 since we max(recv, 1) — graceful
    assert out["pass"] is True


# ── BUS_HANDOFF emission verification ─────────────────────────────────────


def test_bus_handoff_emitted_in_phase_hibernate():
    """The orchestrator's _phase_hibernate publishes BUS_HANDOFF between
    SYSTEM_UPGRADE_STARTING and HIBERNATE. We capture publish() calls
    and assert ordering.

    This is a self-contained verification: we don't run a full swap,
    just exercise _phase_hibernate with mock bus + kernel.
    """
    from titan_plugin import bus as bus_module
    from titan_plugin.core import shadow_orchestrator as so

    # Capture publish() calls in order
    published: list[dict] = []

    class MockBus:
        def publish(self, msg):
            published.append(msg)
            return 1

    class MockKernel:
        # Minimal stand-in — _phase_hibernate uses kernel only for
        # _drain_messages which we'll short-circuit by returning empty acks
        pass

    # _drain_messages reads from the inbox queue; we provide an empty queue
    import queue as _q
    inbox = _q.Queue()

    # Build a SwapResult with the necessary fields
    result = so.SwapResult(
        event_id="test-event-id",
        reason="C8 unit test",
    )
    result.shadow_port = 7779
    result.phase = "readiness_wait"

    # Mock kernel so _drain_messages no-ops cleanly. We patch the function
    # to short-circuit (we don't care about HIBERNATE_ACK collection here).
    original_drain = so._drain_messages
    so._drain_messages = lambda *a, **kw: []
    try:
        # _phase_hibernate signature: (bus_obj, inbox, result, *, expected_workers, kernel)
        # Pass at least one worker so the per-worker HIBERNATE loop fires
        # (Fix A 2026-04-27: replaced single dst="all" broadcast with a
        # per-worker dst loop so reply_only modules like IMW receive it).
        ok = so._phase_hibernate(
            MockBus(), inbox, result,
            expected_workers=["spirit"],
            kernel=MockKernel(),
        )
    finally:
        so._drain_messages = original_drain

    # Verify publish ordering: SYSTEM_UPGRADE_STARTING, then BUS_HANDOFF,
    # then HIBERNATE (3 messages, in this order)
    types = [m.get("type") for m in published]
    assert "SYSTEM_UPGRADE_STARTING" in types, f"got {types}"
    assert "BUS_HANDOFF" in types, f"got {types}"
    assert "HIBERNATE" in types, f"got {types}"
    starting_idx = types.index("SYSTEM_UPGRADE_STARTING")
    handoff_idx = types.index("BUS_HANDOFF")
    hibernate_idx = types.index("HIBERNATE")
    assert starting_idx < handoff_idx < hibernate_idx, \
        f"order wrong: {types}"

    # Verify BUS_HANDOFF payload shape
    handoff = published[handoff_idx]
    assert handoff["src"] == "kernel"
    assert handoff["dst"] == "all"
    assert handoff["payload"]["reason"] == "shadow_swap"
    assert handoff["payload"]["event_id"] == "test-event-id"
    assert "expected_downtime_ms" in handoff["payload"]
