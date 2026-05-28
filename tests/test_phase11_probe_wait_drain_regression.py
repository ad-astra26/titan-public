"""
Regression test for 11F/§11.I.2 probe-driven readiness.

Phase 11 §11.I.2 (locked D1/D2): the legacy MODULE_READY bus path is DELETED.
`_wait_for_module_running` is now SHM-only:

  1. It polls the worker's `module_<name>_state.bin` slot.
  2. When it observes `state=booted` it publishes MODULE_PROBE_REQUEST
     (non-blocking) to drive the worker through probing→running.
  3. On observed `state=running` it mirrors RUNNING into the in-process
     ModuleInfo (state + ready_time) and returns True.

These tests pin the SHM contract so a future refactor can't silently
reintroduce the MODULE_READY bus dependency.
"""
from __future__ import annotations

import threading
import time

from titan_hcl.bus import DivineBus
from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
from titan_hcl.core.state_registry import resolve_titan_id
from titan_hcl.orchestrator import ModuleSpec, ModuleState, Orchestrator


def _dummy_entry(*_a, **_kw) -> None:
    pass


def test_wait_for_module_running_reaches_running_via_shm():
    """When the worker's SHM slot transitions booted→running (the probe
    response), `_wait_for_module_running` returns True and mirrors RUNNING
    into the in-process ModuleInfo — no MODULE_READY bus event involved."""
    bus = DivineBus()
    orch = Orchestrator(bus, config={
        "boot_stagger_delay_s": 0.0,
        "probe_wait_timeout_s": 3.0,
        "phase_11_pipeline_enabled": True,
    })
    orch.register(ModuleSpec(name="x", entry_fn=_dummy_entry, layer="L2"))

    titan_id = resolve_titan_id()
    writer = ModuleStateWriter(
        module_name="x", layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id=titan_id)
    writer.write_state("starting")
    writer.write_state("booted")

    # Simulate the worker's probe handler driving the slot to running shortly
    # after the orchestrator dispatches MODULE_PROBE_REQUEST.
    def fake_probe_response():
        time.sleep(0.3)
        writer.write_state("running")

    threading.Thread(target=fake_probe_response, daemon=True).start()

    try:
        t0 = time.time()
        ok = orch._wait_for_module_running("x")
        elapsed = time.time() - t0

        assert ok is True, (
            "SHM slot reached state=running but _wait_for_module_running "
            "returned False — the SHM readiness path regressed.")
        assert elapsed < 2.5, (
            f"Returned True but took {elapsed:.2f}s — slower than expected.")
        assert orch._modules["x"].state == ModuleState.RUNNING
        assert orch._modules["x"].ready_time > 0.0
    finally:
        writer.close()


def test_wait_for_module_running_times_out_for_silent_worker():
    """If the slot never reaches running (worker stuck booted), the probe-wait
    bounds at the timeout and returns False (no infinite block)."""
    bus = DivineBus()
    orch = Orchestrator(bus, config={
        "boot_stagger_delay_s": 0.0,
        "probe_wait_timeout_s": 0.5,
        "phase_11_pipeline_enabled": True,
    })
    orch.register(ModuleSpec(name="silent", entry_fn=_dummy_entry, layer="L2"))

    titan_id = resolve_titan_id()
    writer = ModuleStateWriter(
        module_name="silent", layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id=titan_id)
    writer.write_state("starting")
    writer.write_state("booted")  # never advances to running

    try:
        t0 = time.time()
        ok = orch._wait_for_module_running("silent")
        elapsed = time.time() - t0

        assert ok is False
        assert 0.4 < elapsed < 1.5, (
            f"Timeout should fire near 0.5s; got {elapsed:.2f}s.")
    finally:
        writer.close()
