"""RFP_supervision_lifecycle §7.C — non-blocking enable + fast auto-re-enable + eta.

Proves a DISABLED module recovers WITHOUT a full restart:
  1. enable() is NON-BLOCKING — it resets DISABLED+counters synchronously then
     submits start() to the background executor and returns immediately, instead
     of blocking up to SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S on start()'s
     dep-activation (the 60s enable-RPC-timeout that forced full restarts).
  2. auto-re-enable cooldown is 180s (was 600).
  3. get_status exposes reenable_eta_s so "re-enabling in Ns" is observable.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleState
from titan_hcl.orchestrator import core as orch_core


def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L2", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


def _disabled_guardian(name: str = "m"):
    g = Guardian(DivineBus())
    g.register(_spec(name))
    info = g._modules[name]
    info.state = ModuleState.DISABLED
    info.disabled_at = time.time()
    info.restart_count = 3
    info.restart_timestamps.extend([time.time(), time.time(), time.time()])
    return g, info


# ── 1. NON-BLOCKING enable ────────────────────────────────────────────────

def test_enable_is_non_blocking():
    """enable() must return immediately even when start() blocks (the dep-
    activation wait that caused the 60s RPC timeout)."""
    g, info = _disabled_guardian()
    started = threading.Event()
    release = threading.Event()

    def blocking_start(name, **kw):
        started.set()
        release.wait(timeout=5.0)
        return True

    g.start = blocking_start
    t0 = time.time()
    ok = g.enable("m")
    elapsed = time.time() - t0

    assert ok is True, "enable should return True (initiated)"
    assert elapsed < 1.0, f"enable() blocked {elapsed:.2f}s — must be non-blocking"
    assert started.wait(timeout=2.0), "start() must run in the background executor"
    release.set()
    g.stop_all()


def test_enable_resets_disabled_state_and_counters():
    g, info = _disabled_guardian()
    g.start = MagicMock(return_value=True)

    ok = g.enable("m")

    assert ok is True
    assert info.state == ModuleState.STOPPED, "DISABLED must be cleared"
    assert info.restart_count == 0
    assert len(info.restart_timestamps) == 0
    assert info.disabled_at == 0.0, "reenable-eta marker cleared"
    # start() submitted to the executor (runs async)
    for _ in range(20):
        if g.start.called:
            break
        time.sleep(0.02)
    g.start.assert_called_with("m")
    g.stop_all()


def test_enable_unknown_returns_false():
    g = Guardian(DivineBus())
    assert g.enable("does_not_exist") is False
    g.stop_all()


def test_enable_already_enabled_is_noop():
    g, info = _disabled_guardian()
    info.state = ModuleState.RUNNING  # not disabled
    g.start = MagicMock(return_value=True)
    ok = g.enable("m")
    assert ok is True
    time.sleep(0.1)
    g.start.assert_not_called()
    g.stop_all()


# ── 2. fast auto-re-enable cooldown ───────────────────────────────────────

def test_reenable_cooldown_is_180():
    assert orch_core.REENABLE_COOLDOWN_S == 180.0
    # the supervisor's auto-re-enable path loads the same value
    from titan_hcl.supervisor import core as sup_core
    sup_core._load_constants()
    assert sup_core._REENABLE_COOLDOWN_S == 180.0


# ── 3. reenable_eta_s in status ───────────────────────────────────────────

def test_get_status_exposes_reenable_eta():
    g, info = _disabled_guardian()
    info.disabled_at = time.time() - 60.0  # disabled 60s ago → ~120s eta

    st = g.get_status()["m"]
    assert st["reenable_eta_s"] is not None
    assert 100.0 <= st["reenable_eta_s"] <= 130.0, \
        f"expected ~120s eta (180-60); got {st['reenable_eta_s']}"

    # a non-disabled module exposes None
    info.state = ModuleState.RUNNING
    info.disabled_at = 0.0
    assert g.get_status()["m"]["reenable_eta_s"] is None
    g.stop_all()
