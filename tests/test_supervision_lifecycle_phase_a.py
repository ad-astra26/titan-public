"""RFP_supervision_lifecycle §7.A — restart-window 60s + boot-grace + sustained-uptime-reset.

Proves the over-eager-disable fixes that ended the disable→full-restart vicious
circle (synthesis OOM-flap incident 2026-06-15):

  1. BOOT-GRACE — a transient rss_* fault inside a module's boot window is NOT
     counted toward escalation and does NOT trigger kill→respawn (let it finish
     booting). heartbeat_timeout + post-boot rss faults are still counted.
  2. 60s WINDOW — flaps spread over minutes age out (no false crash-loop), while
     a genuine tight loop within 60s still disables.
  3. SUSTAINED-UPTIME-RESET — a module stable for >300s since its last restart
     has its restart history cleared (was wired to config but never applied).

Drives the supervised fault path `Guardian.restart(name, reason)` directly with
`start`/`stop` mocked, manipulating ModuleInfo timestamps relative to now.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleState


def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L3", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


def _guardian(name: str = "m"):
    """A Guardian with one registered module; start/stop mocked so restart()
    exercises only the counting/escalation logic, never a real spawn."""
    g = Guardian(DivineBus())
    g.register(_spec(name))
    g.start = MagicMock(return_value=True)
    g.stop = MagicMock(return_value=None)
    info = g._modules[name]
    info.state = ModuleState.RUNNING
    return g, info


# ── 1. BOOT-GRACE ─────────────────────────────────────────────────────────

def test_boot_grace_exempts_rss_fault_during_boot():
    """An rss_* fault within the boot window: not counted, not restarted."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now            # just started → inside boot-grace
    info.last_restart = 0.0

    ok = g.restart("m", reason="rss_713mb")

    assert ok is False, "boot-grace rss fault should return False (no restart)"
    assert len(info.restart_timestamps) == 0, "boot-grace fault must NOT be counted"
    g.start.assert_not_called()       # let it finish booting, don't kill→respawn
    assert info.state is not ModuleState.DISABLED
    g.stop_all()


def test_boot_grace_does_not_exempt_heartbeat_timeout():
    """A heartbeat_timeout during boot-grace is NOT exempt — a genuinely hung
    boot must still be caught (boot-grace 45s < heartbeat_timeout 90s)."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now            # inside boot-grace
    info.last_restart = 0.0          # no backoff

    ok = g.restart("m", reason="heartbeat_timeout")

    assert ok is True, "heartbeat_timeout must proceed to restart even in boot-grace"
    assert len(info.restart_timestamps) == 1, "non-rss fault is counted"
    g.start.assert_called_once()
    g.stop_all()


def test_rss_fault_after_boot_grace_is_counted():
    """Once past the boot window, an rss_* fault is counted + restarts normally."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now - 200.0    # past the 180s boot+settle grace
    info.last_restart = 0.0

    ok = g.restart("m", reason="rss_713mb")

    assert ok is True
    assert len(info.restart_timestamps) == 1, "post-boot rss fault IS counted"
    g.start.assert_called_once()
    g.stop_all()


# ── 2. 60s WINDOW ─────────────────────────────────────────────────────────

def test_60s_window_ages_out_spread_flaps_no_disable():
    """5 flaps spread over ~200s (today's synthesis pattern) do NOT disable —
    the 60s window prunes the old ones below the escalation threshold."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now - 300.0    # past boot-grace
    info.last_restart = now - 5.0    # recent → no uptime-reset
    # 5 timestamps spread across 200s; only the most recent is within 60s.
    info.restart_timestamps.extend([now - 200, now - 160, now - 120, now - 80, now - 40])

    g.restart("m", reason="rss_700mb")

    assert info.state is not ModuleState.DISABLED, \
        "spread-out flaps must age out of the 60s window — no false crash-loop"
    g.stop_all()


def test_60s_window_disables_tight_crash_loop():
    """A genuine tight loop — 5 faults all within 60s — still escalates→DISABLED."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now - 300.0    # past boot-grace
    info.last_restart = now - 5.0    # recent → no uptime-reset
    info.restart_timestamps.extend([now - 50, now - 40, now - 30, now - 20, now - 10])

    g.restart("m", reason="rss_700mb")

    assert info.state is ModuleState.DISABLED, \
        "a real tight crash-loop within 60s must still disable"
    g.stop_all()


# ── 3. SUSTAINED-UPTIME-RESET ─────────────────────────────────────────────

def test_sustained_uptime_reset_clears_history_prevents_disable():
    """5 recent faults WOULD disable, but >300s sustained uptime since the last
    restart clears the history first → not disabled (the previously-dead reset)."""
    g, info = _guardian()
    now = time.time()
    info.start_time = now - 500.0    # past boot-grace
    info.last_restart = now - 400.0  # stable >300s → triggers the reset
    info.restart_timestamps.extend([now - 50, now - 40, now - 30, now - 20, now - 10])

    g.restart("m", reason="rss_700mb")

    assert info.state is not ModuleState.DISABLED, \
        "sustained-uptime-reset must clear stale history before the window check"
    g.stop_all()
