"""
Regression test for 11F probe-wait queue-drain.

Live-discovered 2026-05-27 T3 deploy: during `start_all()`, the supervision
loop's `_process_guardian_messages` hasn't yet started (scripts/guardian_hcl.py
only enters the supervision tick AFTER start_all returns). Workers' MODULE_READY
broadcasts therefore piled up in the guardian queue without ever updating
`info.state`. Every un-migrated worker (which is most of the fleet pre-11I)
timed out at the full probe budget (~30s × 16 mandatory = ~8 minutes).

Fix: `_wait_for_module_running` drains `_process_guardian_messages` each poll
iteration so the legacy MODULE_READY → info.state path actually fires within
the probe-wait window.

This test pins the drain so a future refactor doesn't silently regress
the back-compat boot path.
"""
from __future__ import annotations

import threading
import time

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.orchestrator import ModuleSpec, ModuleState, Orchestrator


def _dummy_entry(*_a, **_kw) -> None:
    pass


def test_wait_for_module_running_drains_guardian_queue_inline():
    """When MODULE_READY is published while start_all is mid-walk (the
    supervision tick hasn't started yet), `_wait_for_module_running` must
    inline-drain it so info.state transitions to RUNNING within the
    probe-wait window."""
    bus = DivineBus()
    orch = Orchestrator(bus, config={
        "boot_stagger_delay_s": 0.0,
        "probe_wait_timeout_s": 2.0,
        "phase_11_pipeline_enabled": True,
    })
    orch.register(ModuleSpec(name="x", entry_fn=_dummy_entry, layer="L2"))

    # Simulate a worker that publishes MODULE_READY shortly after spawn.
    # (Real workers do this from their entry_fn after init.)
    from titan_hcl.bus import MODULE_READY, make_msg

    def fake_worker():
        time.sleep(0.2)
        bus.publish(make_msg(
            MODULE_READY, src="x", dst="guardian", payload={},
        ))

    threading.Thread(target=fake_worker, daemon=True).start()

    t0 = time.time()
    ok = orch._wait_for_module_running("x")
    elapsed = time.time() - t0

    assert ok is True, (
        "MODULE_READY arrived but _wait_for_module_running returned False — "
        "the inline drain regressed; legacy boot path no longer works.")
    # Should return promptly after MODULE_READY arrives, not at the full
    # timeout. 1.5s leaves comfortable headroom.
    assert elapsed < 1.5, (
        f"Probe-wait drained successfully but took {elapsed:.2f}s — slower "
        f"than expected. The inline drain may be missing.")
    assert orch._modules["x"].state == ModuleState.RUNNING


def test_wait_for_module_running_still_times_out_for_silent_worker():
    """If MODULE_READY never arrives, the probe-wait still bounds at
    the timeout (no infinite block from the drain change)."""
    bus = DivineBus()
    orch = Orchestrator(bus, config={
        "boot_stagger_delay_s": 0.0,
        "probe_wait_timeout_s": 0.3,
        "phase_11_pipeline_enabled": True,
    })
    orch.register(ModuleSpec(name="silent", entry_fn=_dummy_entry, layer="L2"))

    t0 = time.time()
    ok = orch._wait_for_module_running("silent")
    elapsed = time.time() - t0

    assert ok is False
    assert 0.2 < elapsed < 1.0, (
        f"Timeout should fire near 0.3s; got {elapsed:.2f}s — drain may be "
        f"introducing blocking calls.")
