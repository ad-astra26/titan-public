"""
Tests for Microkernel v2 Phase A §A.5 — layer-aware crash log levels.

When a module crashes or heartbeat-times-out:
  - L1 (Trinity daemon) crashes log at ERROR level (architecturally unexpected)
  - L2/L3 crashes log at WARNING level (normal operational churn)

Uses caplog to inspect log records without spawning real processes.

Reference: titan-docs/PLAN_microkernel_phase_a.md §4.1.1 (log-level policy)
"""
from __future__ import annotations

import logging
import time

import pytest

from titan_plugin.bus import DivineBus
from titan_plugin.guardian import (
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
    """
    Register a module and forcibly mark it RUNNING with a stale heartbeat,
    so monitor_tick() will consider it timed-out.
    """
    guardian.register(ModuleSpec(
        name=name, entry_fn=_noop_entry, layer=layer,
        heartbeat_timeout=30.0, restart_on_crash=False,
    ))
    info = guardian._modules[name]
    info.state = ModuleState.RUNNING
    info.last_heartbeat = time.time() - hb_age_s
    info.start_time = time.time() - hb_age_s
    # Simulate a PID that is NOT actually alive — monitor_tick() will
    # treat this as a dead process.
    info.pid = 999999  # very unlikely to exist
    return info


def test_l1_heartbeat_timeout_logs_at_error(guardian, caplog):
    """L1 module (Trinity daemon) crash logs at ERROR."""
    _seed_running_module(guardian, name="body", layer="L1")
    with caplog.at_level(logging.WARNING, logger="titan_plugin.guardian"):
        guardian.monitor_tick()
    # The heartbeat-timeout log entry should be ERROR level with L1 tag
    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "heartbeat timeout" in r.getMessage()
    ]
    # At least one ERROR entry containing "[L1]"
    assert any("[L1]" in r.getMessage() for r in error_records), (
        f"Expected L1 ERROR-level heartbeat log; got: "
        f"{[r.getMessage() for r in caplog.records]}")


def test_l2_heartbeat_timeout_logs_at_warning(guardian, caplog):
    """L2 module crash logs at WARNING (not elevated)."""
    _seed_running_module(guardian, name="memory", layer="L2")
    with caplog.at_level(logging.WARNING, logger="titan_plugin.guardian"):
        guardian.monitor_tick()
    warn_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "heartbeat timeout" in r.getMessage()
    ]
    assert any("[L2]" in r.getMessage() for r in warn_records), (
        f"Expected L2 WARNING-level heartbeat log; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")


def test_l3_heartbeat_timeout_logs_at_warning(guardian, caplog):
    """L3 module crash logs at WARNING (not elevated)."""
    _seed_running_module(guardian, name="llm", layer="L3")
    with caplog.at_level(logging.WARNING, logger="titan_plugin.guardian"):
        guardian.monitor_tick()
    warn_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "heartbeat timeout" in r.getMessage()
    ]
    assert any("[L3]" in r.getMessage() for r in warn_records), (
        f"Expected L3 WARNING-level heartbeat log; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")


def test_register_log_includes_layer(guardian, caplog):
    """Guardian.register() log line includes the layer tag."""
    with caplog.at_level(logging.INFO, logger="titan_plugin.guardian"):
        guardian.register(ModuleSpec(
            name="spirit", entry_fn=_noop_entry, layer="L1"))
    reg_records = [r for r in caplog.records
                   if "Registered module" in r.getMessage()]
    assert any("[L1]" in r.getMessage() for r in reg_records), (
        f"Expected [L1] in register log; got: "
        f"{[r.getMessage() for r in caplog.records]}")
