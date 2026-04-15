"""
Unit tests for DiskHealthMonitor — state machine + hysteresis + edge-
detected event emission. These run fast (no real disk, no threads) by
driving _compute_state directly and calling _transition with mocked hooks.
"""
import pytest
from unittest.mock import MagicMock, patch

from titan_plugin.core.disk_health import (
    DiskHealthMonitor,
    DiskState,
    DEFAULT_THRESHOLDS,
    assert_disk_bootable,
    _GB,
)


# --------------------------------------------------------------------
# State machine — boundary transitions
# --------------------------------------------------------------------

def _mk(initial=DiskState.HEALTHY):
    """Build a monitor without starting the thread and set initial state."""
    m = DiskHealthMonitor(path="/tmp", publish_fn=MagicMock(), shutdown_fn=MagicMock())
    m._state = initial
    return m


def test_healthy_stays_healthy_with_plenty_of_space():
    m = _mk()
    assert m._compute_state(10 * _GB) == DiskState.HEALTHY


def test_healthy_to_warning_on_threshold_cross():
    m = _mk()
    # Just below warning_enter (5 GB)
    assert m._compute_state(4 * _GB) == DiskState.WARNING


def test_warning_to_critical_on_threshold_cross():
    m = _mk(DiskState.WARNING)
    # Just below critical_enter (2 GB)
    assert m._compute_state(int(1.5 * _GB)) == DiskState.CRITICAL


def test_critical_to_emergency_on_threshold_cross():
    m = _mk(DiskState.CRITICAL)
    # Just below emergency_enter (0.5 GB)
    assert m._compute_state(int(0.4 * _GB)) == DiskState.EMERGENCY


def test_emergency_takes_priority_from_any_state():
    for start in DiskState:
        m = _mk(start)
        assert m._compute_state(int(0.1 * _GB)) == DiskState.EMERGENCY, (
            f"EMERGENCY should trigger from {start} at 0.1 GB free"
        )


# --------------------------------------------------------------------
# Hysteresis — downgrades require extra headroom (no flapping)
# --------------------------------------------------------------------

def test_hysteresis_warning_does_not_immediately_return_to_healthy():
    m = _mk(DiskState.WARNING)
    # Free rises just above warning_enter (5 GB) but below warning_exit (5.5 GB)
    # Must stay in WARNING
    assert m._compute_state(int(5.1 * _GB)) == DiskState.WARNING


def test_hysteresis_warning_returns_to_healthy_above_exit_threshold():
    m = _mk(DiskState.WARNING)
    # Free above warning_exit (5.5 GB)
    assert m._compute_state(int(6 * _GB)) == DiskState.HEALTHY


def test_hysteresis_critical_does_not_immediately_return_to_warning():
    m = _mk(DiskState.CRITICAL)
    # Free rises just above critical_enter (2 GB) but below critical_exit (2.2 GB)
    assert m._compute_state(int(2.1 * _GB)) == DiskState.CRITICAL


def test_hysteresis_emergency_does_not_immediately_return_to_critical():
    m = _mk(DiskState.EMERGENCY)
    # Free rises just above emergency_enter (0.5 GB) but below exit (0.55 GB)
    assert m._compute_state(int(0.52 * _GB)) == DiskState.EMERGENCY


def test_recovery_full_path_emergency_to_healthy():
    """Once free rises well above all boundaries, state downgrades to HEALTHY."""
    m = _mk(DiskState.EMERGENCY)
    assert m._compute_state(10 * _GB) == DiskState.HEALTHY


# --------------------------------------------------------------------
# Transition hooks — publish + shutdown edge-triggered
# --------------------------------------------------------------------

def test_transition_publishes_event():
    m = _mk(DiskState.HEALTHY)
    m._transition(DiskState.HEALTHY, DiskState.WARNING, 4 * _GB)
    m._publish_fn.assert_called_once()
    args, _ = m._publish_fn.call_args
    assert args[0] == DiskState.WARNING
    assert args[1] == 4 * _GB


def test_emergency_entry_triggers_shutdown():
    m = _mk(DiskState.CRITICAL)
    m._transition(DiskState.CRITICAL, DiskState.EMERGENCY, int(0.3 * _GB))
    m._shutdown_fn.assert_called_once()
    # Shutdown reason should name disk + free space
    reason = m._shutdown_fn.call_args[0][0]
    assert "disk_emergency" in reason
    assert "GB" in reason


def test_emergency_to_emergency_does_not_retrigger_shutdown():
    """If monitor re-observes EMERGENCY (shouldn't happen via _transition since
    it's only called on state changes), shutdown must not be re-called.
    Guard: _transition checks old != new — we verify here that logic stays
    intact."""
    m = _mk(DiskState.EMERGENCY)
    # Direct call with old == new would be a bug in caller; the monitor
    # itself only invokes _transition when state changes (see _loop).
    # So here we only verify that the state field is used correctly:
    # transitioning critical -> emergency triggers; staying in emergency does not.
    m._transition(DiskState.CRITICAL, DiskState.EMERGENCY, int(0.3 * _GB))
    assert m._shutdown_fn.call_count == 1
    m._shutdown_fn.reset_mock()
    # Now simulate another downgrade — no shutdown because old == EMERGENCY
    m._transition(DiskState.EMERGENCY, DiskState.EMERGENCY, int(0.2 * _GB))
    m._shutdown_fn.assert_not_called()


def test_warning_transition_does_not_shutdown():
    m = _mk(DiskState.HEALTHY)
    m._transition(DiskState.HEALTHY, DiskState.WARNING, 4 * _GB)
    m._shutdown_fn.assert_not_called()


def test_critical_transition_does_not_shutdown():
    m = _mk(DiskState.WARNING)
    m._transition(DiskState.WARNING, DiskState.CRITICAL, int(1.5 * _GB))
    m._shutdown_fn.assert_not_called()


# --------------------------------------------------------------------
# Monitor resilience — poll errors don't crash the thread
# --------------------------------------------------------------------

def test_publish_fn_exception_does_not_crash_transition():
    """If publish_fn throws (e.g. bus unreachable), the monitor must
    survive and still call shutdown_fn on EMERGENCY."""
    m = _mk(DiskState.CRITICAL)
    m._publish_fn = MagicMock(side_effect=RuntimeError("bus unreachable"))
    # Should not raise
    m._transition(DiskState.CRITICAL, DiskState.EMERGENCY, int(0.3 * _GB))
    # shutdown_fn still invoked despite publish error
    m._shutdown_fn.assert_called_once()


def test_shutdown_fn_exception_does_not_crash_transition():
    m = _mk(DiskState.CRITICAL)
    m._shutdown_fn = MagicMock(side_effect=RuntimeError("guardian down"))
    m._transition(DiskState.CRITICAL, DiskState.EMERGENCY, int(0.3 * _GB))
    # No exception propagates


# --------------------------------------------------------------------
# Boot-time sanity check
# --------------------------------------------------------------------

def test_assert_disk_bootable_passes_on_healthy_disk():
    """Happy path — generous free space, no exception."""
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = MagicMock(free=10 * _GB)
        assert_disk_bootable("/tmp")  # no exception


def test_assert_disk_bootable_raises_on_critical_disk():
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = MagicMock(free=int(0.1 * _GB))
        with pytest.raises(SystemExit) as exc_info:
            assert_disk_bootable("/tmp")
        assert exc_info.value.code == 2


def test_assert_disk_bootable_skips_on_unreadable_path():
    """If shutil.disk_usage itself fails (permissions, ENOENT), the boot
    check is skipped — we must never block boot on our own helper failing."""
    with patch("shutil.disk_usage", side_effect=OSError("no such file")):
        assert_disk_bootable("/definitely/does/not/exist")  # no exception


# --------------------------------------------------------------------
# Snapshot API — for /v4/health endpoint consumers
# --------------------------------------------------------------------

def test_snapshot_format():
    m = _mk(DiskState.WARNING)
    m._last_free_bytes = 3 * _GB
    snap = m.snapshot()
    assert snap["state"] == "warning"
    assert snap["free_bytes"] == 3 * _GB
    assert snap["free_gb"] == 3.0
    assert snap["path"] == "/tmp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
