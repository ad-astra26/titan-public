"""Phase B.2.1 Path B' (2026-04-27 PM) — shadow Guardian skip-spawn + relaxed health.

Validates the architectural fix that lets graduated workers truly outlive
a kernel swap:

  1. Guardian.start_all() reads TITAN_B2_1_ADOPTION_PENDING from env.
     When set: spawn-mode autostart modules are SKIPPED. Adoption phase
     covers them via BUS_WORKER_ADOPT_REQUEST → Guardian.adopt_worker.
     Fork-mode workers + non-graduated specials still start normally.

  2. _phase_shadow_boot accepts b2_1_active=True. When True:
       • TITAN_B2_1_ADOPTION_PENDING=1 is set in shadow's env
       • HealthCriteria(min_modules_running=5) — relaxed gate so the 9
         graduated workers' absence (until adoption) doesn't fail boot

  3. orchestrate_shadow_swap computes b2_1_active BEFORE shadow_boot and
     passes it down. (Same value used by adoption_wait phase 3.6.)

These were the two missing pieces that caused the first true-outlive E2E
test on T1 (16:25 UTC) to fail: shadow respawned the 9 graduated workers
fresh, creating duplicates that fought for shm/locks.
"""
from __future__ import annotations

import ast
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from titan_plugin import bus as bus_mod
from titan_plugin.guardian import Guardian, ModuleSpec


def _noop_entry(*args, **kwargs):  # pragma: no cover
    return None


@pytest.fixture(autouse=True)
def _clear_env():
    """Each test starts with TITAN_B2_1_ADOPTION_PENDING unset."""
    prior = os.environ.pop("TITAN_B2_1_ADOPTION_PENDING", None)
    yield
    if prior is None:
        os.environ.pop("TITAN_B2_1_ADOPTION_PENDING", None)
    else:
        os.environ["TITAN_B2_1_ADOPTION_PENDING"] = prior


# ── Guardian.start_all skip-spawn behaviour ──────────────────────────────


def test_start_all_no_env_starts_everything_autostart():
    """Without TITAN_B2_1_ADOPTION_PENDING: legacy behaviour, all autostart
    modules start regardless of start_method."""
    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    g.register(ModuleSpec(name="m_fork", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="fork"))
    g.register(ModuleSpec(name="m_spawn", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="spawn"))
    g.register(ModuleSpec(name="m_off", layer="L3", entry_fn=_noop_entry,
                           autostart=False, start_method="spawn"))

    started = []
    with patch.object(g, "start", side_effect=lambda n: started.append(n) or True):
        g.start_all()
    # autostart=True regardless of method → both started; m_off stays off
    assert sorted(started) == ["m_fork", "m_spawn"]


def test_start_all_with_env_skips_spawn_mode_only():
    """With TITAN_B2_1_ADOPTION_PENDING=1: skip spawn-mode autostart only."""
    os.environ["TITAN_B2_1_ADOPTION_PENDING"] = "1"
    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    g.register(ModuleSpec(name="m_fork", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="fork"))
    g.register(ModuleSpec(name="m_spawn", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="spawn"))
    g.register(ModuleSpec(name="m_fork_2", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="fork"))

    started = []
    with patch.object(g, "start", side_effect=lambda n: started.append(n) or True):
        g.start_all()
    # spawn skipped; both fork started
    assert sorted(started) == ["m_fork", "m_fork_2"]


def test_start_all_env_zero_does_not_skip():
    """TITAN_B2_1_ADOPTION_PENDING=0 (or any non-'1' value) is treated as off."""
    os.environ["TITAN_B2_1_ADOPTION_PENDING"] = "0"
    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    g.register(ModuleSpec(name="m_spawn", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="spawn"))
    started = []
    with patch.object(g, "start", side_effect=lambda n: started.append(n) or True):
        g.start_all()
    assert started == ["m_spawn"]


def test_start_all_skipping_does_not_skip_explicit_start():
    """The skip is at start_all() boot time only — explicit Guardian.start(name)
    still works (e.g., orchestrator fallback after adoption window closes)."""
    os.environ["TITAN_B2_1_ADOPTION_PENDING"] = "1"
    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    g.register(ModuleSpec(name="m_spawn", layer="L3", entry_fn=_noop_entry,
                           autostart=True, start_method="spawn"))
    started_by_all = []
    started_explicit = []
    with patch.object(g, "start",
                      side_effect=lambda n: (started_explicit.append(n) or True)):
        g.start_all()
        # start_all skipped; explicit start still works
        g.start("m_spawn")
    assert started_explicit == ["m_spawn"]


# ── _phase_shadow_boot env + relaxed health ──────────────────────────────


def test_phase_shadow_boot_b2_1_sets_env_and_relaxes_health():
    """When b2_1_active=True:
       (a) subprocess.Popen receives env TITAN_B2_1_ADOPTION_PENDING=1
       (b) _wait_for_health is called with relaxed HealthCriteria
           (min_modules_running=5)
    """
    from titan_plugin.core import shadow_orchestrator as so

    captured_env: dict = {}
    captured_criteria = []

    class _MockProc:
        pid = 99999
        def terminate(self):  # pragma: no cover
            pass

    def _fake_popen(cmd, **kw):
        captured_env.update(kw.get("env") or {})
        return _MockProc()

    def _fake_wait_for_health(port, timeout, criteria=None):
        captured_criteria.append(criteria)
        return True, {"checks": {}}

    fake_kernel = type("K", (), {})()
    fake_kernel.bus = bus_mod.DivineBus(maxsize=10)
    fake_kernel.guardian = None
    result = so.SwapResult(event_id="test_eid", reason="t")

    with patch.object(so.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(so, "_wait_for_health",
                      side_effect=_fake_wait_for_health), \
         patch("titan_plugin.core.shadow_data_dir.copy_data_dir",
               return_value=(True, "hardlink")), \
         patch("titan_plugin.core.shadow_data_dir.cleanup_shadow_dir"), \
         patch.object(so, "read_active_port", return_value=7777), \
         patch.object(so, "pick_shadow_port", return_value=7779):
        proc = so._phase_shadow_boot(
            fake_kernel, result,
            snapshot_path="/tmp/snap.msgpack",
            b2_1_active=True,
        )

    assert proc is not None
    assert captured_env.get("TITAN_B2_1_ADOPTION_PENDING") == "1", (
        "shadow subprocess must inherit the adoption-pending flag"
    )
    assert len(captured_criteria) == 1
    crit = captured_criteria[0]
    assert crit is not None, "relaxed criteria must be passed (not None)"
    assert crit.min_modules_running == 3, (
        f"b2_1_active relaxes min_modules_running to 3; got {crit.min_modules_running}"
    )
    # Graduated workers (memory, timechain, emot_cgn, rl, body, mind, etc.)
    # are excluded from critical — they're checked in adoption_wait phase.
    assert "memory" not in crit.critical_modules
    assert "timechain" not in crit.critical_modules
    assert "body" not in crit.critical_modules
    assert "mind" not in crit.critical_modules
    # Spirit (fork-mode) + imw + api still must be in shadow at boot time.
    assert "spirit" in crit.critical_modules
    assert "imw" in crit.critical_modules
    assert "api" in crit.critical_modules


def test_phase_shadow_boot_legacy_no_env_no_relaxation():
    """When b2_1_active=False (default): no env set, default HealthCriteria."""
    from titan_plugin.core import shadow_orchestrator as so

    captured_env: dict = {}
    captured_criteria = []

    class _MockProc:
        pid = 99999
        def terminate(self):  # pragma: no cover
            pass

    def _fake_popen(cmd, **kw):
        captured_env.update(kw.get("env") or {})
        return _MockProc()

    def _fake_wait_for_health(port, timeout, criteria=None):
        captured_criteria.append(criteria)
        return True, {"checks": {}}

    fake_kernel = type("K", (), {})()
    fake_kernel.bus = bus_mod.DivineBus(maxsize=10)
    fake_kernel.guardian = None
    result = so.SwapResult(event_id="test_eid", reason="t")

    with patch.object(so.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(so, "_wait_for_health",
                      side_effect=_fake_wait_for_health), \
         patch("titan_plugin.core.shadow_data_dir.copy_data_dir",
               return_value=(True, "hardlink")), \
         patch("titan_plugin.core.shadow_data_dir.cleanup_shadow_dir"), \
         patch.object(so, "read_active_port", return_value=7777), \
         patch.object(so, "pick_shadow_port", return_value=7779):
        so._phase_shadow_boot(
            fake_kernel, result,
            snapshot_path="/tmp/snap.msgpack",
            # b2_1_active default False
        )

    assert "TITAN_B2_1_ADOPTION_PENDING" not in captured_env
    assert captured_criteria == [None], (
        "legacy path: criteria=None means default HealthCriteria"
    )


# ── orchestrate_shadow_swap passes b2_1_active to shadow_boot ────────────


def test_orchestrator_passes_b2_1_active_to_shadow_boot():
    """AST guard: orchestrate_shadow_swap must compute b2_1_active BEFORE
    calling _phase_shadow_boot and pass it as kwarg. Without this, the
    skip-spawn + relaxed-health path is dead code."""
    import inspect, titan_plugin.core.shadow_orchestrator as so_mod
    src = inspect.getsource(so_mod.orchestrate_shadow_swap)
    # Find the _phase_shadow_boot call. It must contain b2_1_active=
    boot_call_idx = src.find("_phase_shadow_boot(")
    assert boot_call_idx >= 0, "orchestrate_shadow_swap must call _phase_shadow_boot"
    # Look at the next ~250 chars (the call's argument list)
    call_chunk = src[boot_call_idx:boot_call_idx + 400]
    assert "b2_1_active=" in call_chunk, (
        "_phase_shadow_boot call must pass b2_1_active= kwarg — without it, "
        "shadow respawns graduated workers + min_modules_running fails"
    )
