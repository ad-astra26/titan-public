"""
Tests for Microkernel v2 Phase A §A.5 — layer-aware fault log levels.

When a module faults (heartbeat-timeout / crash):
  - L1 (Trinity daemon) logs at ERROR level (architecturally unexpected)
  - L2/L3 log at WARNING level (normal operational churn)

The fault-detection + layer-aware log live in `Supervisor.monitor_tick`
(`titan_hcl/supervisor/core.py:366` — `ERROR if layer=="L1" else WARNING`).
Per the Phase-11 split + D-SPEC-141, monitor_tick reads the AUTHORITATIVE
per-module SHM slot (`ModuleStateReaderBank.read` → `ModuleStateEntry`), NOT the
legacy in-memory `info.last_heartbeat`. These tests therefore inject a synthetic
SHM entry (state="running", an ALIVE pid + a stale heartbeat → genuine
`heartbeat_timeout` fault) via a mocked reader bank, and assert the log level.

Uses caplog to inspect log records without spawning real processes.
Reference: titan-docs/PLAN_microkernel_phase_a.md §4.1.1 (log-level policy)
"""
from __future__ import annotations

import logging
import os
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.core.module_state import BootPriority, ModuleStateEntry
from titan_hcl.supervisor import Supervisor  # Phase 11 §11.I.1 supervisor split
from titan_hcl.guardian_hcl import (
    Guardian, ModuleInfo, ModuleSpec, ModuleState,
)


def _noop_entry(recv_q, send_q, name, config):  # pragma: no cover
    pass


@pytest.fixture
def guardian():
    bus = DivineBus(maxsize=100)
    g = Guardian(bus)
    yield g


def _seed_running_module(guardian, name, layer, hb_age_s=999):
    """Register a module marked RUNNING + install a mocked SHM reader bank whose
    slot reports an ALIVE pid (this test process) with a stale heartbeat — so
    `Supervisor.monitor_tick` detects a genuine `heartbeat_timeout` fault and
    emits the layer-aware log. `restart_on_crash=True` so monitor_tick reaches
    the fault-log (a non-restart module is skipped before logging)."""
    guardian.register(ModuleSpec(
        name=name, entry_fn=_noop_entry, layer=layer,
        heartbeat_timeout=30.0, restart_on_crash=True,
    ))
    info = guardian._modules[name]
    info.state = ModuleState.RUNNING
    # last_cpu_time=0 → the CPU-aware grace treats it as "not CPU-starved" so
    # the stale heartbeat resolves to a real heartbeat_timeout fault, not a
    # deferred starved-grace cycle.
    info.last_cpu_time = 0.0

    entry = ModuleStateEntry(
        name=name, layer=layer, boot_priority=BootPriority.MANDATORY,
        state="running", pid=os.getpid(),  # alive → passes liveness → hb path
        last_heartbeat=time.time() - hb_age_s,
    )
    bank = MagicMock()
    bank.read = lambda n: entry if n == name else None
    # monitor_tick reads via orch._ensure_module_state_reader_bank() (orch=guardian)
    guardian._ensure_module_state_reader_bank = lambda: bank
    return info


def _heartbeat_fault_records(caplog):
    return [r for r in caplog.records
            if "heartbeat_timeout" in r.getMessage()]


def test_l1_heartbeat_timeout_logs_at_error(guardian, caplog):
    """L1 module (Trinity daemon) fault logs at ERROR."""
    _seed_running_module(guardian, name="body", layer="L1")
    with caplog.at_level(logging.WARNING):
        Supervisor(guardian.bus, guardian).monitor_tick()
    recs = _heartbeat_fault_records(caplog)
    assert any(r.levelno >= logging.ERROR and "[L1]" in r.getMessage() for r in recs), (
        f"Expected L1 ERROR-level heartbeat fault log; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")


def test_l2_heartbeat_timeout_logs_at_warning(guardian, caplog):
    """L2 module fault logs at WARNING (not elevated)."""
    _seed_running_module(guardian, name="memory", layer="L2")
    with caplog.at_level(logging.WARNING):
        Supervisor(guardian.bus, guardian).monitor_tick()
    recs = _heartbeat_fault_records(caplog)
    assert any(r.levelno == logging.WARNING and "[L2]" in r.getMessage() for r in recs), (
        f"Expected L2 WARNING-level heartbeat fault log; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")


def test_l3_heartbeat_timeout_logs_at_warning(guardian, caplog):
    """L3 module fault logs at WARNING (not elevated)."""
    _seed_running_module(guardian, name="llm", layer="L3")
    with caplog.at_level(logging.WARNING):
        Supervisor(guardian.bus, guardian).monitor_tick()
    recs = _heartbeat_fault_records(caplog)
    assert any(r.levelno == logging.WARNING and "[L3]" in r.getMessage() for r in recs), (
        f"Expected L3 WARNING-level heartbeat fault log; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")


def test_register_log_includes_layer(guardian, caplog):
    """Guardian.register() log line includes the layer tag."""
    with caplog.at_level(logging.INFO, logger="titan_hcl.orchestrator"):
        guardian.register(ModuleSpec(
            name="spirit", entry_fn=_noop_entry, layer="L1"))
    reg_records = [r for r in caplog.records
                   if "Registered module" in r.getMessage()]
    assert any("[L1]" in r.getMessage() for r in reg_records), (
        f"Expected [L1] in register log; got: "
        f"{[r.getMessage() for r in caplog.records]}")
