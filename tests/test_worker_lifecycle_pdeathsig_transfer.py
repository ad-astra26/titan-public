"""
Tests for Phase B.2.1 worker_lifecycle PDEATHSIG clear/reset + watcher pause/resume.

Covers:
- clear_parent_death_signal() returns True on Linux (smoke + libc-mock fallback)
- After clear, watcher in relaxed-mode tolerates getppid()==1 unless bus dead
- reset_parent_death_signal() re-arms (alias for install_parent_death_signal)
- pause_parent_watcher / resume_parent_watcher state transitions
- WatcherState bus tracking helpers
- on_bus_handoff for spawn-mode strips PDEATHSIG + pauses watcher + acks
- on_bus_handoff for fork-mode is a no-op (improved-B.1 path)
- on_bus_adopt_ack flips state to relaxed mode
- on_bus_adopt_ack with status="rejected" self-SIGTERMs (mocked)
- on_bus_handoff_canceled re-arms PDEATHSIG + restores strict watcher
- supervision_check self-SIGTERMs at 30s threshold while swap-pending+disconnected

Tests use real prctl when available (Linux) and mock libc otherwise. The
self-SIGTERM tests mock os.kill so the test runner doesn't get killed.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin.core import worker_lifecycle, worker_swap_handler
from titan_plugin.core.worker_lifecycle import (
    WatcherState,
    clear_parent_death_signal,
    install_full_protection,
    install_parent_death_signal,
    pause_parent_watcher,
    reset_parent_death_signal,
    resume_parent_watcher,
    start_parent_watcher,
)
from titan_plugin.core.worker_swap_handler import (
    SwapHandlerState,
    on_bus_adopt_ack,
    on_bus_handoff,
    on_bus_handoff_canceled,
    request_adoption,
    supervision_check,
)


# ── PDEATHSIG clear / reset primitives ─────────────────────────────────────


@pytest.mark.skipif(not os.path.exists("/lib/x86_64-linux-gnu/libc.so.6")
                    and not os.path.exists("/lib/aarch64-linux-gnu/libc.so.6")
                    and not os.path.exists("/lib64/libc.so.6"),
                    reason="Linux libc.so.6 unavailable")
def test_clear_parent_death_signal_returns_true_on_linux():
    """Smoke: prctl call succeeds on Linux."""
    # First arm so we have something to clear
    install_parent_death_signal(sig=signal.SIGTERM)
    assert clear_parent_death_signal() is True


def test_clear_parent_death_signal_libc_unavailable():
    """When libc.so.6 fails to load, returns False without raising."""
    with patch("ctypes.CDLL", side_effect=OSError("no libc")):
        assert clear_parent_death_signal() is False


def test_clear_parent_death_signal_prctl_rejected():
    """When prctl returns non-zero, function returns False without raising."""
    fake_libc = MagicMock()
    fake_libc.prctl.return_value = -1
    with patch("ctypes.CDLL", return_value=fake_libc), \
         patch("ctypes.get_errno", return_value=22):
        assert clear_parent_death_signal() is False


def test_reset_parent_death_signal_aliases_install():
    """reset_parent_death_signal must call install_parent_death_signal."""
    with patch.object(worker_lifecycle, "install_parent_death_signal", return_value=True) as m:
        assert reset_parent_death_signal(sig=signal.SIGTERM) is True
        m.assert_called_once_with(sig=signal.SIGTERM)


# ── WatcherState pause / resume / bus tracking ─────────────────────────────


def test_watcher_state_bus_tracking_helpers():
    """mark_bus_unreachable sets timestamp once; mark_bus_healthy clears."""
    stop = threading.Event()
    state = WatcherState(stop_event=stop)
    assert state._bus_unreachable_since is None

    state.mark_bus_unreachable()
    first_ts = state._bus_unreachable_since
    assert first_ts is not None

    # Idempotent — does not move the timestamp forward (we want elapsed
    # since FIRST disconnect)
    time.sleep(0.01)
    state.mark_bus_unreachable()
    assert state._bus_unreachable_since == first_ts

    state.mark_bus_healthy()
    assert state._bus_unreachable_since is None


def test_pause_resume_strict_then_relaxed():
    """pause sets _paused True; resume(relaxed=True) sets relaxed mode."""
    stop = threading.Event()
    state = WatcherState(stop_event=stop)
    pause_parent_watcher(state)
    assert state._paused is True
    assert state._b2_1_relaxed_mode is False

    resume_parent_watcher(state, relaxed=True)
    assert state._paused is False
    assert state._b2_1_relaxed_mode is True

    resume_parent_watcher(state, relaxed=False)
    assert state._b2_1_relaxed_mode is False


def test_install_full_protection_returns_watcher_state():
    """install_full_protection includes watcher_state for B.2.1 callers."""
    out = install_full_protection(watcher_interval=10.0)
    assert "watcher_state" in out
    assert isinstance(out["watcher_state"], WatcherState)
    assert out["watcher_state"].is_alive()
    # Tear down to avoid thread leak between cases
    out["watcher_state"].stop_event.set()


# ── on_bus_handoff dispatch by start_method ────────────────────────────────


def _make_state(start_method: str, *, bus_client=None):
    """Test helper — minimal SwapHandlerState with a real WatcherState (paused)."""
    stop = threading.Event()
    ws = WatcherState(stop_event=stop, _supervision_timeout_s=30.0)
    return SwapHandlerState(
        name="test_worker",
        start_method=start_method,
        watcher_state=ws,
        bus_client=bus_client or MagicMock(is_connected=True, publish=MagicMock(return_value=True)),
    )


def test_on_bus_handoff_spawn_mode_strips_and_acks():
    """spawn-mode workers: clear PDEATHSIG + pause watcher + ack + swap_pending=True."""
    state = _make_state("spawn")
    msg = {"type": "BUS_HANDOFF", "payload": {"event_id": "evt-1"}}

    with patch.object(worker_swap_handler, "clear_parent_death_signal", return_value=True) as m_clear:
        on_bus_handoff(state, msg)

    m_clear.assert_called_once()
    assert state.watcher_state._paused is True
    assert state._swap_pending is True
    assert state._handoff_event_id == "evt-1"
    # publish was called with a BUS_HANDOFF_ACK
    state.bus_client.publish.assert_called_once()
    sent_msg = state.bus_client.publish.call_args[0][0]
    assert sent_msg["type"] == "BUS_HANDOFF_ACK"
    assert sent_msg["payload"]["event_id"] == "evt-1"


def test_on_bus_handoff_fork_mode_is_noop():
    """fork-mode workers: ignore HANDOFF (improved-B.1 path)."""
    state = _make_state("fork")
    msg = {"type": "BUS_HANDOFF", "payload": {"event_id": "evt-1"}}

    with patch.object(worker_swap_handler, "clear_parent_death_signal") as m_clear:
        on_bus_handoff(state, msg)

    m_clear.assert_not_called()
    assert state.watcher_state._paused is False
    assert state._swap_pending is False
    state.bus_client.publish.assert_not_called()


def test_on_bus_handoff_clear_failure_falls_back():
    """If clear_parent_death_signal fails, worker stays armed (no swap_pending, no ack)."""
    state = _make_state("spawn")
    msg = {"type": "BUS_HANDOFF", "payload": {"event_id": "evt-1"}}

    with patch.object(worker_swap_handler, "clear_parent_death_signal", return_value=False):
        on_bus_handoff(state, msg)

    # Did NOT pause watcher, did NOT mark swap_pending, did NOT ack
    assert state.watcher_state._paused is False
    assert state._swap_pending is False
    state.bus_client.publish.assert_not_called()


# ── on_bus_adopt_ack ──────────────────────────────────────────────────────


def test_on_bus_adopt_ack_status_adopted_relaxes_watcher():
    """status=adopted: exit swap_pending + resume watcher in relaxed mode."""
    state = _make_state("spawn")
    state._swap_pending = True
    state._adopt_rid = "rid-abc"

    msg = {
        "type": "BUS_WORKER_ADOPT_ACK",
        "rid": "rid-abc",
        "payload": {"status": "adopted", "shadow_pid": 12345},
    }
    on_bus_adopt_ack(state, msg)

    assert state._adopted is True
    assert state._swap_pending is False
    assert state.watcher_state._b2_1_relaxed_mode is True
    assert state.watcher_state._paused is False


def test_on_bus_adopt_ack_status_rejected_self_sigterms():
    """status=rejected: log + os.kill(pid, SIGTERM)."""
    state = _make_state("spawn")
    state._swap_pending = True
    state._adopt_rid = "rid-abc"

    msg = {
        "type": "BUS_WORKER_ADOPT_ACK",
        "rid": "rid-abc",
        "payload": {"status": "rejected", "reason": "unknown_name"},
    }
    with patch("os.kill") as m_kill:
        on_bus_adopt_ack(state, msg)
    m_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


def test_on_bus_adopt_ack_rid_mismatch_ignored():
    """Stale ACK with different rid is ignored (no state change, no kill)."""
    state = _make_state("spawn")
    state._swap_pending = True
    state._adopt_rid = "rid-correct"

    msg = {
        "type": "BUS_WORKER_ADOPT_ACK",
        "rid": "rid-stale",
        "payload": {"status": "adopted"},
    }
    with patch("os.kill") as m_kill:
        on_bus_adopt_ack(state, msg)
    m_kill.assert_not_called()
    assert state._adopted is False
    assert state._swap_pending is True  # unchanged


# ── on_bus_handoff_canceled ──────────────────────────────────────────────


def test_on_bus_handoff_canceled_rearms_strict():
    """CANCEL re-arms PDEATHSIG + restores strict watcher (P-2c unwind)."""
    state = _make_state("spawn")
    state._swap_pending = True
    state.watcher_state._paused = True
    state.watcher_state._b2_1_relaxed_mode = False  # never relaxed; we cancelled before adoption

    with patch.object(worker_swap_handler, "install_parent_death_signal", return_value=True) as m_install:
        on_bus_handoff_canceled(state, {"type": "BUS_HANDOFF_CANCELED", "payload": {}})

    m_install.assert_called_once_with(sig=signal.SIGTERM)
    assert state._swap_pending is False
    assert state.watcher_state._paused is False
    assert state.watcher_state._b2_1_relaxed_mode is False


# ── supervision_check ─────────────────────────────────────────────────────


def test_supervision_check_no_swap_pending_is_noop():
    """When not swap-pending, supervision_check must not signal."""
    state = _make_state("spawn")
    state._swap_pending = False
    with patch("os.kill") as m_kill:
        supervision_check(state, now=time.time())
    m_kill.assert_not_called()


def test_supervision_check_connected_clears_unreachable():
    """While connected, supervision_check clears any prior unreachable timestamp."""
    state = _make_state("spawn", bus_client=MagicMock(is_connected=True))
    state._swap_pending = True
    state._bus_unreachable_since = time.time() - 100.0  # already past threshold
    state.watcher_state._bus_unreachable_since = state._bus_unreachable_since

    with patch("os.kill") as m_kill:
        supervision_check(state, now=time.time())

    m_kill.assert_not_called()
    assert state._bus_unreachable_since is None
    assert state.watcher_state._bus_unreachable_since is None


def test_supervision_check_disconnected_first_seen_marks_timestamp():
    """First disconnected observation marks timestamp; doesn't kill."""
    state = _make_state("spawn", bus_client=MagicMock(is_connected=False))
    state._swap_pending = True
    state._bus_unreachable_since = None

    now = time.time()
    with patch("os.kill") as m_kill:
        supervision_check(state, now=now)

    m_kill.assert_not_called()
    assert state._bus_unreachable_since == now


def test_supervision_check_disconnected_under_threshold_skip_kill():
    """Disconnected for less than threshold: does not kill."""
    state = _make_state("spawn", bus_client=MagicMock(is_connected=False))
    state._swap_pending = True
    base = time.time()
    state._bus_unreachable_since = base
    state.watcher_state._supervision_timeout_s = 30.0

    with patch("os.kill") as m_kill:
        supervision_check(state, now=base + 5.0)
    m_kill.assert_not_called()


def test_supervision_check_disconnected_over_threshold_self_sigterms():
    """Disconnected ≥ threshold while swap-pending: self-SIGTERM."""
    state = _make_state("spawn", bus_client=MagicMock(is_connected=False))
    state._swap_pending = True
    base = time.time()
    state._bus_unreachable_since = base
    state.watcher_state._supervision_timeout_s = 30.0

    with patch("os.kill") as m_kill:
        supervision_check(state, now=base + 31.0)
    m_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


def test_supervision_check_fork_mode_ignored():
    """Fork-mode workers never enter the supervision-via-bus path."""
    state = _make_state("fork", bus_client=MagicMock(is_connected=False))
    state._swap_pending = True  # arbitrary; fork-mode never sets this
    state._bus_unreachable_since = time.time() - 100.0

    with patch("os.kill") as m_kill:
        supervision_check(state, now=time.time())
    m_kill.assert_not_called()


# ── request_adoption ──────────────────────────────────────────────────────


def test_request_adoption_publishes_rid_routed_request():
    """Sends BUS_WORKER_ADOPT_REQUEST with name/pid/start_method/boot_ts + rid."""
    state = _make_state("spawn")
    state._swap_pending = True

    request_adoption(state)

    state.bus_client.publish.assert_called_once()
    sent = state.bus_client.publish.call_args[0][0]
    assert sent["type"] == "BUS_WORKER_ADOPT_REQUEST"
    assert sent["payload"]["name"] == "test_worker"
    assert sent["payload"]["pid"] == os.getpid()
    assert sent["payload"]["start_method"] == "spawn"
    assert "boot_ts" in sent["payload"]
    assert sent["rid"] is not None
    assert state._adopt_rid == sent["rid"]


def test_request_adoption_skipped_when_not_swap_pending():
    """If we're not swap-pending, request_adoption must not publish."""
    state = _make_state("spawn")
    state._swap_pending = False
    request_adoption(state)
    state.bus_client.publish.assert_not_called()


def test_request_adoption_skipped_for_fork_mode():
    """Fork-mode workers never request adoption."""
    state = _make_state("fork")
    state._swap_pending = True  # paranoia; can't actually happen
    request_adoption(state)
    state.bus_client.publish.assert_not_called()
