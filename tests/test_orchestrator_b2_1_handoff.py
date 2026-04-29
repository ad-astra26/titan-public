"""
Tests for Phase B.2.1 orchestrator extensions.

Covers:
- HealthCriteria.min_adopted_workers default 0 (disabled gate)
- _check_adopted_workers_criteria returns None when min=0 (back-compat)
- _check_adopted_workers_criteria pass when adopted count meets threshold
- _check_adopted_workers_criteria fail when below threshold
- _check_adopted_workers_criteria fail when /v4/state.guardian missing
- _phase_b2_1_wait_adoption returns True with no spawn-mode workers
- _phase_b2_1_wait_adoption returns True when expected adopted set seen
- _phase_b2_1_wait_adoption returns False on timeout
- _unwind_b2_1_handoff publishes BUS_HANDOFF_CANCELED when broker present
- _unwind_b2_1_handoff is no-op when no socket broker
- orchestrate_shadow_swap accepts b2_1_forced + b2_1_adoption_timeout_s
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin import bus
from titan_plugin.core.shadow_orchestrator import (
    HealthCriteria,
    SwapResult,
    _check_adopted_workers_criteria,
    _phase_b2_1_wait_adoption,
    _unwind_b2_1_handoff,
    orchestrate_shadow_swap,
)
from titan_plugin.guardian import Guardian, ModuleSpec, ModuleState


def _noop_entry(*args, **kwargs):  # pragma: no cover
    return None


# ── HealthCriteria.min_adopted_workers default + criteria check ─────────


def test_min_adopted_workers_default_zero_disabled():
    c = HealthCriteria()
    assert c.min_adopted_workers == 0


def test_check_adopted_workers_criteria_disabled_returns_none():
    c = HealthCriteria(min_adopted_workers=0)
    assert _check_adopted_workers_criteria({}, c) is None


def test_check_adopted_workers_criteria_pass_when_count_sufficient():
    c = HealthCriteria(min_adopted_workers=2)
    state = {"data": {"guardian": {
        "body":   {"adopted": True,  "state": "running"},
        "mind":   {"adopted": True,  "state": "running"},
        "memory": {"adopted": False, "state": "running"},
    }}}
    diag = _check_adopted_workers_criteria(state, c)
    assert diag is not None
    assert diag["pass"] is True
    assert diag["adopted_count"] == 2
    assert diag["min_required"] == 2
    assert diag["adopted_modules"] == ["body", "mind"]


def test_check_adopted_workers_criteria_fail_when_below_threshold():
    c = HealthCriteria(min_adopted_workers=3)
    state = {"data": {"guardian": {
        "body": {"adopted": True, "state": "running"},
    }}}
    diag = _check_adopted_workers_criteria(state, c)
    assert diag is not None
    assert diag["pass"] is False
    assert diag["adopted_count"] == 1


def test_check_adopted_workers_criteria_fail_when_guardian_missing():
    c = HealthCriteria(min_adopted_workers=1)
    state = {"data": {"v4": True}}  # no guardian key
    diag = _check_adopted_workers_criteria(state, c)
    assert diag is not None
    assert diag["pass"] is False
    assert diag["error"] == "guardian_status_absent"


# ── _phase_b2_1_wait_adoption ───────────────────────────────────────────


def _make_kernel_with_specs(specs: list[ModuleSpec]):
    """Build a minimal kernel stub holding a Guardian with registered specs.

    M4 (2026-04-27 PM): _phase_b2_1_wait_adoption now skips workers with
    info.pid is None (autostart=False or never started). Test fixtures
    set info.pid to a fake PID so the worker is treated as 'live' and
    expected to adopt — preserving the existing test semantics.
    """
    div = bus.DivineBus(maxsize=100)
    g = Guardian(div)
    for s in specs:
        g.register(s)
        # Set fake PID so M4's "live worker" filter includes this module
        info = g._modules.get(s.name)
        if info is not None:
            info.pid = 99999  # fake but non-None
    kernel = MagicMock()
    kernel.guardian = g
    kernel.bus = div
    return kernel


def test_phase_b2_1_wait_adoption_no_spawn_workers_returns_true():
    """With no spawn-mode workers in ModuleSpec set, adoption phase no-ops."""
    kernel = _make_kernel_with_specs([
        ModuleSpec(name="body", entry_fn=_noop_entry, layer="L1", start_method="fork"),
        ModuleSpec(name="mind", entry_fn=_noop_entry, layer="L1", start_method="fork"),
    ])
    result = SwapResult(event_id="evt-test", reason="t")
    ok = _phase_b2_1_wait_adoption(
        kernel, expected_workers=["body", "mind"], shadow_port=7779,
        result=result, timeout=1.0,
    )
    assert ok is True
    # Should NOT have polled HTTP; logged the no-spawn-mode noop
    events = [e["msg"] for e in result.audit]
    assert "b2_1_no_spawn_mode_workers" in events


def test_phase_b2_1_wait_adoption_returns_true_when_expected_adopted():
    """Polls succeed → adopted set covers spawn-mode expected → returns True."""
    kernel = _make_kernel_with_specs([
        ModuleSpec(name="backup", entry_fn=_noop_entry, layer="L3",
                   start_method="spawn"),
    ])
    result = SwapResult(event_id="evt-test-2", reason="t")

    fake_state = {"data": {"guardian": {
        "backup": {"adopted": True, "state": "running"},
    }}}
    with patch(
        "titan_plugin.core.shadow_orchestrator._fetch_state_json",
        return_value=fake_state,
    ):
        ok = _phase_b2_1_wait_adoption(
            kernel, expected_workers=["backup"], shadow_port=7779,
            result=result, timeout=2.0,
        )
    assert ok is True
    events = [e["msg"] for e in result.audit]
    assert "b2_1_adoption_acks_collected" in events


def test_phase_b2_1_wait_adoption_returns_false_on_timeout():
    """Adopted set never covers spawn-mode expected → returns False after timeout."""
    kernel = _make_kernel_with_specs([
        ModuleSpec(name="backup", entry_fn=_noop_entry, layer="L3",
                   start_method="spawn"),
    ])
    result = SwapResult(event_id="evt-test-3", reason="t")

    # Shadow's state always shows adopted=False
    fake_state = {"data": {"guardian": {
        "backup": {"adopted": False, "state": "running"},
    }}}
    with patch(
        "titan_plugin.core.shadow_orchestrator._fetch_state_json",
        return_value=fake_state,
    ):
        ok = _phase_b2_1_wait_adoption(
            kernel, expected_workers=["backup"], shadow_port=7779,
            result=result, timeout=0.5,  # quick timeout for tests
        )
    assert ok is False
    events = [e["msg"] for e in result.audit]
    assert "b2_1_adoption_timeout" in events


# ── _unwind_b2_1_handoff ─────────────────────────────────────────────────


def test_unwind_b2_1_handoff_publishes_canceled_when_broker_present():
    """With broker active, publishes BUS_HANDOFF_CANCELED + logs event."""
    bus_obj = MagicMock()
    bus_obj.has_socket_broker = True
    result = SwapResult(event_id="evt-unwind", reason="t")

    _unwind_b2_1_handoff(kernel=None, bus_obj=bus_obj, result=result,
                          reason="shadow_boot_failed")
    bus_obj.publish.assert_called_once()
    sent = bus_obj.publish.call_args[0][0]
    assert sent["type"] == bus.BUS_HANDOFF_CANCELED
    assert sent["payload"]["reason"] == "shadow_boot_failed"
    events = [e["msg"] for e in result.audit]
    assert "b2_1_handoff_canceled" in events


def test_unwind_b2_1_handoff_noop_without_broker():
    """No socket broker → no publish, no event."""
    bus_obj = MagicMock()
    bus_obj.has_socket_broker = False
    result = SwapResult(event_id="evt-unwind-2", reason="t")

    _unwind_b2_1_handoff(kernel=None, bus_obj=bus_obj, result=result,
                          reason="shadow_boot_failed")
    bus_obj.publish.assert_not_called()
    events = [e["msg"] for e in result.audit]
    assert "b2_1_handoff_canceled" not in events


# ── orchestrate_shadow_swap kwargs surface ──────────────────────────────


def test_orchestrate_shadow_swap_accepts_b2_1_kwargs():
    """The function signature must accept b2_1_forced + b2_1_adoption_timeout_s.

    We don't run the full swap — just verify the kwargs are accepted and
    dispatched into the function without TypeError.
    """
    import inspect
    sig = inspect.signature(orchestrate_shadow_swap)
    assert "b2_1_forced" in sig.parameters
    assert sig.parameters["b2_1_forced"].default is False
    assert "b2_1_adoption_timeout_s" in sig.parameters
    # M2 (2026-04-27 PM audit): bumped 15s → 30s. 5-7s typical adoption
    # latency leaves plenty of headroom; 15s was tight under load.
    assert sig.parameters["b2_1_adoption_timeout_s"].default == 30.0
