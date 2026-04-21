"""Tests for Guardian's CPU-aware heartbeat (added 2026-04-21).

Background. On 2026-04-21 we observed cascading media-module restart loops
on the shared T2/T3 VPS during iter-3 ARC training: arc_competition.py
ran at 100% CPU for 75 min, load avg hit ~14 on a 4-vCPU box, and the
media module's heartbeat thread was preempted >180s repeatedly. Guardian
read this as "deadlocked" and restarted media every 3 min, which made
the cascade worse.

Fix: when wallclock heartbeat times out, sample /proc/<pid>/stat CPU
time. If CPU grew ≥ MIN_CPU_DELTA_FOR_ALIVE since last sample, treat
the module as alive-but-starved and defer restart for up to
MAX_STARVED_CYCLES wallclock heartbeat windows; then force-restart
(bounded grace prevents runaway hang on a truly stuck module).

These tests cover the foundation: the CPU-time sampler. End-to-end
behavior (defer-restart on starvation) is exercised in a slower
integration test.
"""

from __future__ import annotations

import os
import time

from titan_plugin.guardian import (
    MAX_STARVED_CYCLES,
    MIN_CPU_DELTA_FOR_ALIVE,
    Guardian,
)


def test_get_cpu_time_seconds_grows_under_load():
    """A busy loop in this very process should show CPU-time growth."""
    pid = os.getpid()
    t0 = Guardian._get_cpu_time_seconds(pid)
    assert t0 > 0.0, "Process has consumed some CPU since boot"
    # Burn ~100ms of CPU time
    end = time.time() + 0.1
    n = 0
    while time.time() < end:
        n += 1
    t1 = Guardian._get_cpu_time_seconds(pid)
    delta = t1 - t0
    # At minimum we should see ~50ms of growth — slack lets the test pass
    # under heavy host load. The point is that it's nonzero + monotonic.
    assert delta >= 0.05, f"expected ≥50ms CPU growth, got {delta * 1000:.0f}ms"


def test_get_cpu_time_seconds_returns_zero_for_dead_pid():
    """Reading /proc/<dead-pid>/stat should fail gracefully and return 0.0."""
    # PID 999999 is almost certainly not in use; if by mischance it is,
    # this test will spuriously fail — but the failure surfaces the helper's
    # error path either way, which is the goal.
    assert Guardian._get_cpu_time_seconds(999999) == 0.0


def test_get_cpu_time_seconds_handles_zero_pid():
    """PID 0 isn't a real process — should return 0.0 cleanly."""
    assert Guardian._get_cpu_time_seconds(0) == 0.0


def test_constants_have_sensible_defaults():
    """MIN_CPU_DELTA_FOR_ALIVE and MAX_STARVED_CYCLES bounds.

    Too low MIN_CPU_DELTA → easy to spoof aliveness with idle wakeups.
    Too high MAX_STARVED_CYCLES → defer restart of a truly stuck module too long.
    """
    assert MIN_CPU_DELTA_FOR_ALIVE >= 0.5, "Threshold should require real CPU work"
    assert MIN_CPU_DELTA_FOR_ALIVE <= 5.0, "Threshold shouldn't gate normal modules"
    assert 1 <= MAX_STARVED_CYCLES <= 10, "Grace window should be bounded"
