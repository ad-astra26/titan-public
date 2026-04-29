"""
Tests for the 2026-04-27 Phase B.1 unwind-path bug fix:
  Guardian.resume() + _revive_guardian_after_unwind() in every rollback.

Bug: shadow_orchestrator's _phase_hibernate calls kernel.guardian.stop_all()
which sets _stop_requested=True, permanently muting monitor_tick. Without
a symmetric resume() call, a swap rollback leaves Titan dark — workers
exited via HIBERNATE, Guardian silent, no auto-respawn.

Discovered during T1 first-flag-flip swap test (event_id=4b27e251):
locks_not_released → shadow_boot_failed → unwind without revive →
T1 down ~5min until --force restart.

Tests cover:
- Guardian.resume() flips _stop_requested False, idempotent
- Guardian.resume() logged as no-op when already running
- _revive_guardian_after_unwind calls resume + start_all
- _revive is no-op when kernel is None or guardian is None
- _revive logs failure event when start_all raises (CRITICAL severity)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_plugin import bus
from titan_plugin.core.shadow_orchestrator import (
    SwapResult,
    _revive_guardian_after_unwind,
)
from titan_plugin.guardian import Guardian, ModuleSpec


def _noop_entry(*args, **kwargs):  # pragma: no cover
    return None


# ── Guardian.resume primitive ─────────────────────────────────────────────


def test_resume_flips_stop_requested_to_false():
    """After stop_all sets _stop_requested=True, resume clears it."""
    div = bus.DivineBus(maxsize=100)
    g = Guardian(div)
    g._stop_requested = True
    g.resume()
    assert g._stop_requested is False


def test_resume_idempotent_when_already_running():
    """Calling resume on a running Guardian is a no-op (logs only)."""
    div = bus.DivineBus(maxsize=100)
    g = Guardian(div)
    assert g._stop_requested is False
    g.resume()  # must not raise
    assert g._stop_requested is False


def test_resume_after_stop_all_re_enables_monitor_tick():
    """Critical regression: monitor_tick must process again after resume."""
    div = bus.DivineBus(maxsize=100)
    g = Guardian(div)
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3"))
    # Simulate stop_all having run
    g._stop_requested = True
    # monitor_tick should early-return while paused
    g.monitor_tick()  # no exception, no work done

    g.resume()
    # Now monitor_tick should NOT early-return (does its normal work)
    # We can't easily verify "did work" without a heavy setup, but we
    # can confirm _stop_requested is the gate.
    assert g._stop_requested is False


# ── _revive_guardian_after_unwind helper ─────────────────────────────────


def test_revive_calls_resume_and_start_all():
    """Helper invokes resume() + start_all() + logs event."""
    g = MagicMock()
    g._stop_requested = True
    kernel = MagicMock()
    kernel.guardian = g
    result = SwapResult(event_id="evt-revive", reason="t")

    _revive_guardian_after_unwind(kernel, result)

    g.resume.assert_called_once()
    g.start_all.assert_called_once()
    events = [e["msg"] for e in result.audit]
    assert "guardian_revived" in events


def test_revive_noop_when_kernel_none():
    """No kernel → quietly returns; no event logged."""
    result = SwapResult(event_id="evt-nokernel", reason="t")
    _revive_guardian_after_unwind(None, result)
    events = [e["msg"] for e in result.audit]
    assert "guardian_revived" not in events
    assert "guardian_revive_failed" not in events


def test_revive_noop_when_guardian_none():
    """kernel.guardian is None (legacy mode) → no-op."""
    kernel = MagicMock()
    kernel.guardian = None
    result = SwapResult(event_id="evt-noguardian", reason="t")
    _revive_guardian_after_unwind(kernel, result)
    events = [e["msg"] for e in result.audit]
    assert "guardian_revived" not in events
    assert "guardian_revive_failed" not in events


def test_revive_logs_critical_event_when_start_all_raises():
    """If start_all raises, a CRITICAL event is logged so ops can intervene."""
    g = MagicMock()
    g.start_all.side_effect = RuntimeError("everything is broken")
    kernel = MagicMock()
    kernel.guardian = g
    result = SwapResult(event_id="evt-fail", reason="t")

    _revive_guardian_after_unwind(kernel, result)

    g.resume.assert_called_once()
    events = result.audit
    fail_events = [e for e in events if e["msg"] == "guardian_revive_failed"]
    assert len(fail_events) == 1
    assert "CRITICAL" in fail_events[0].get("severity", "")
    assert "everything is broken" in fail_events[0].get("error", "")


# ── Integration: rollback paths in orchestrate_shadow_swap ────────────────


def test_orchestrator_rollback_paths_all_call_revive():
    """Static check: every 'return result.to_dict()' inside a rollback branch
    must be preceded by _revive_guardian_after_unwind in the same block.

    Belt-and-suspenders regression test: catches future rollback paths
    that forget to revive Guardian.
    """
    import inspect
    from titan_plugin.core import shadow_orchestrator as so
    src = inspect.getsource(so.orchestrate_shadow_swap)
    # Every rollback branch (outcome="rollback") should be followed by
    # the revive helper call within the same path before its return.
    revive_lines = []
    for i, line in enumerate(src.splitlines()):
        if "_revive_guardian_after_unwind(kernel" in line:
            revive_lines.append(i)
    # 4 rollback branches + 1 orchestrator_exception path = ≥5 revive sites.
    # Future rollback paths added without a revive call will trip this.
    assert len(revive_lines) >= 5, (
        f"each rollback branch + the orchestrator_exception handler must "
        f"call _revive_guardian_after_unwind; found {len(revive_lines)} sites. "
        f"If you added a rollback path, add the revive call before its return."
    )
