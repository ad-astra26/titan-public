"""
Phase 11 §11.I.7 / §11.I.8 / D-SPEC-141 — orchestrator boot pipeline (11F).

Covers:
  1. `_partition_autostart_by_boot_priority` — MANDATORY / OPTIONAL_POST_BOOT /
     LAZY classification including autostart=True + boot_priority=lazy WARN
     skip and unknown-value fallback.
  2. `start_all` — Phase A walks the MANDATORY bucket synchronously and
     blocks on each module reaching RUNNING; schedules Phase B in a
     background daemon thread.
  3. `start_all` back-compat — `phase_11_pipeline_enabled=False` falls
     back to the Phase 9 9L flat staggered walk.
  4. `_wait_for_module_running` — returns True when info.state hits
     RUNNING via the in-process path; False on timeout.
  5. `TitanHclStateWriter` — fleet_ready writes are SHM-publishable and
     round-trip through `TitanHclStateReader`; first-true latches
     `fleet_ready_at` wall-clock.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.core.titan_hcl_state import (
    TitanHclStateEntry,
    TitanHclStateReader,
    TitanHclStateWriter,
)
from titan_hcl.orchestrator import ModuleSpec, ModuleState, Orchestrator


def _dummy_entry(*_a, **_kw) -> None:
    pass


def _spec(
    name: str,
    *,
    layer: str = "L3",
    autostart: bool = False,
    boot_priority: str = "mandatory",
) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer=layer, entry_fn=_dummy_entry,
        autostart=autostart, restart_on_crash=False,
        boot_priority=boot_priority,
    )


def _make_orch(**cfg_overrides) -> Orchestrator:
    cfg = {
        "boot_stagger_delay_s": 0.0,
        "post_boot_stagger_delay_s": 0.0,
        "probe_wait_timeout_s": 1.0,
        "phase_11_pipeline_enabled": True,
        **cfg_overrides,
    }
    return Orchestrator(DivineBus(), config=cfg)


# ── 1. Partition by boot_priority ────────────────────────────────────


def test_partition_classifies_mandatory_post_boot_lazy():
    o = _make_orch()
    o.register(_spec("a", boot_priority="mandatory"))
    o.register(_spec("b", boot_priority="post_boot"))
    o.register(_spec("c", boot_priority="mandatory"))
    o.register(_spec("d", boot_priority="post_boot"))
    mandatory, post_boot, lazy = (
        o._partition_autostart_by_boot_priority(["a", "b", "c", "d"]))
    assert mandatory == ["a", "c"]
    assert post_boot == ["b", "d"]
    assert lazy == []


def test_partition_warns_and_skips_lazy_autostart(caplog):
    """autostart=True + boot_priority=lazy is a config mistake — partition
    logs WARN and excludes the module from both autostart buckets."""
    o = _make_orch()
    o.register(_spec("real_lazy", autostart=True, boot_priority="lazy"))
    with caplog.at_level("WARNING", logger="titan_hcl.orchestrator"):
        mandatory, post_boot, lazy = (
            o._partition_autostart_by_boot_priority(["real_lazy"]))
    assert mandatory == []
    assert post_boot == []
    assert lazy == ["real_lazy"]
    assert any(
        "real_lazy" in r.getMessage() and "boot_priority=lazy" in r.getMessage()
        for r in caplog.records
    )


def test_partition_unknown_boot_priority_defaults_to_mandatory(caplog):
    o = _make_orch()
    o.register(_spec("typo", boot_priority="MAINDATORY"))
    with caplog.at_level("WARNING", logger="titan_hcl.orchestrator"):
        mandatory, post_boot, lazy = (
            o._partition_autostart_by_boot_priority(["typo"]))
    assert mandatory == ["typo"]
    assert post_boot == []
    assert lazy == []


# ── 2. start_all — Phase A + Phase B routing ─────────────────────────


def test_start_all_calls_phase_a_modules_synchronously_in_dep_order():
    """Phase A spawns MANDATORY modules in topo+layer order with the
    configured stagger. Verified via a mocked start() that records the
    call order."""
    o = _make_orch()
    o.register(_spec("a", autostart=True, layer="L2", boot_priority="mandatory"))
    o.register(_spec("b", autostart=True, layer="L2", boot_priority="mandatory"))
    o.register(_spec("c", autostart=True, layer="L3", boot_priority="post_boot"))
    # Mock start so it both records + advances info.state to RUNNING (so
    # the probe-wait helper returns immediately without polling SHM).
    call_order: list[str] = []

    def fake_start(name: str, activate_deps: bool = True) -> bool:
        call_order.append(name)
        o._modules[name].state = ModuleState.RUNNING
        return True

    o.start = fake_start  # type: ignore[assignment]
    o.start_all()
    # Phase A (a, b) runs synchronously; Phase B (c) runs in background.
    # Wait briefly for the background thread to also fire.
    time.sleep(0.5)
    assert call_order[0:2] == ["a", "b"], (
        f"Phase A should fire a + b in order, got {call_order}")
    assert "c" in call_order, "Phase B should eventually fire c"


def test_start_all_legacy_flat_walk_when_phase_11_disabled():
    """When `phase_11_pipeline_enabled=False`, falls back to the
    Phase 9 9L flat staggered walk — no partitioning, no Phase B
    background thread, no SHM publication."""
    o = _make_orch(phase_11_pipeline_enabled=False)
    o.register(_spec("a", autostart=True, boot_priority="mandatory"))
    o.register(_spec("b", autostart=True, boot_priority="post_boot"))

    call_order: list[str] = []

    def fake_start(name: str, activate_deps: bool = True) -> bool:
        call_order.append(name)
        o._modules[name].state = ModuleState.RUNNING
        return True

    o.start = fake_start  # type: ignore[assignment]
    o.start_all()
    # Flat walk: both modules fired in one pass (alphabetic within same layer).
    assert sorted(call_order) == ["a", "b"]


def test_start_all_writes_fleet_ready_to_titan_hcl_state(tmp_path,
                                                         monkeypatch):
    """End-to-end SHM write: Phase A completion publishes
    fleet_ready=true to titan_hcl_state.bin so kernel-rs +
    guardian_hcl + observatory can read it."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test")
    # G21 single-writer gate (per orchestrator._ensure_titan_hcl_state_writer):
    # only the canonical orchestrator process may publish titan_hcl_state.bin.
    # Tests exercising the canonical path must opt in.
    monkeypatch.setenv("TITAN_HCL_STATE_WRITER_CANONICAL", "1")

    from titan_hcl.core.module_state import BootPriority, ModuleStateWriter

    o = _make_orch()
    o.register(_spec("a", autostart=True, boot_priority="mandatory"))

    _writers: list = []

    def fake_start(name: str, activate_deps: bool = True) -> bool:
        # Mimic the worker lifecycle: write starting→booted→running to the
        # module's OWN SHM slot — the canonical readiness signal the wave-wait
        # + probe poller read (§11.I.2 locked D1/D2). Setting info.state alone
        # is NOT a readiness signal post-Phase-11.
        w = ModuleStateWriter(
            module_name=name, layer="L3",
            boot_priority=BootPriority.MANDATORY, titan_id="test")
        w.write_state("starting")
        w.write_state("booted")
        w.write_state("running")
        _writers.append(w)
        o._modules[name].state = ModuleState.RUNNING
        return True

    o.start = fake_start  # type: ignore[assignment]
    try:
        o.start_all()
        # Phase B is empty → fleet_optional_ready latches immediately.
        reader = TitanHclStateReader(titan_id="test")
        entry = reader.read()
        assert entry is not None
        assert entry.fleet_ready is True
        assert entry.fleet_optional_ready is True
        assert entry.mandatory_total == 1
        assert entry.mandatory_ready == 1
        reader.close()
    finally:
        o.stop_all(reason="test")
        for w in _writers:
            w.close()


def test_start_all_publishes_phase_a_done_before_phase_b_finishes(
        tmp_path, monkeypatch):
    """After Phase A: fleet_ready=true + boot_phase="phase_a_done";
    fleet_optional_ready remains False until Phase B finishes. Verifies
    the SHM contract for kernel-rs (it gates on fleet_ready, not on
    fleet_optional_ready)."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test")
    monkeypatch.setenv("TITAN_HCL_STATE_WRITER_CANONICAL", "1")

    o = _make_orch(post_boot_stagger_delay_s=10.0)  # never finishes in test
    o.register(_spec("a", autostart=True, boot_priority="mandatory"))
    o.register(_spec("b", autostart=True, boot_priority="post_boot"))

    def fake_start(name: str, activate_deps: bool = True) -> bool:
        # Mandatory reaches RUNNING fast; post_boot never gets there
        # within this test (we just care that fleet_ready latches early).
        if name == "a":
            o._modules[name].state = ModuleState.RUNNING
        return True

    o.start = fake_start  # type: ignore[assignment]
    # Set probe wait short so we don't block on the post-boot module.
    o._probe_wait_timeout_s = 0.2
    o.start_all()
    reader = TitanHclStateReader(titan_id="test")
    entry = reader.read()
    assert entry is not None
    assert entry.fleet_ready is True
    assert entry.boot_phase in ("phase_a_done", "booting_b"), entry.boot_phase
    # fleet_optional_ready not yet true (Phase B background thread still
    # in its long stagger sleep).
    assert entry.fleet_optional_ready is False
    reader.close()


# ── 3. _wait_for_module_running — in-process path ────────────────────


def test_wait_for_module_running_returns_true_when_state_running():
    """SHM-attested readiness (§11.I.2 locked D1/D2): when the module's own
    slot reads state=running, the helper returns True. (in-process info.state
    is mirrored FROM the slot — the slot is the authority, not the reverse.)"""
    from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
    from titan_hcl.core.state_registry import resolve_titan_id
    o = _make_orch(probe_wait_timeout_s=2.0)
    o.register(_spec("x"))
    writer = ModuleStateWriter(
        module_name="x", layer="L3",
        boot_priority=BootPriority.MANDATORY, titan_id=resolve_titan_id())
    writer.write_state("running")
    try:
        assert o._wait_for_module_running("x") is True
    finally:
        writer.close()


def test_wait_for_module_running_returns_false_on_timeout():
    o = _make_orch(probe_wait_timeout_s=0.2)
    o.register(_spec("y"))
    # y stays STOPPED forever; wait returns False after the timeout.
    t0 = time.time()
    ok = o._wait_for_module_running("y")
    elapsed = time.time() - t0
    assert ok is False
    assert 0.15 < elapsed < 1.0, (
        f"Probe-wait timeout should fire ≈0.2s; got {elapsed:.2f}s")


def test_wait_for_module_running_returns_false_for_unknown_module():
    o = _make_orch(probe_wait_timeout_s=0.5)
    assert o._wait_for_module_running("nonexistent") is False


def test_wait_for_module_running_picks_up_transition_mid_wait():
    """When the module's SHM slot transitions booted→running mid-wait (the
    probe response), the helper returns True promptly (≪ timeout). Readiness
    is SHM-attested per locked D1/D2 (§11.I.2)."""
    from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
    from titan_hcl.core.state_registry import resolve_titan_id

    o = _make_orch(probe_wait_timeout_s=2.0)
    o.register(_spec("z"))

    writer = ModuleStateWriter(
        module_name="z", layer="L3",
        boot_priority=BootPriority.MANDATORY, titan_id=resolve_titan_id())
    writer.write_state("booted")

    def flip_state_async():
        time.sleep(0.2)
        writer.write_state("running")

    threading.Thread(target=flip_state_async, daemon=True).start()
    try:
        t0 = time.time()
        ok = o._wait_for_module_running("z")
        elapsed = time.time() - t0
        assert ok is True
        assert elapsed < 1.5, (
            f"Should return promptly after slot flips to running; got {elapsed:.2f}s")
        assert o._modules["z"].state == ModuleState.RUNNING
    finally:
        writer.close()


# ── 4. TitanHclStateWriter / Reader contract ─────────────────────────


def test_titan_hcl_state_writer_writes_initial_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test")
    writer = TitanHclStateWriter(titan_id="test")
    try:
        reader = TitanHclStateReader(titan_id="test")
        entry = reader.read()
        assert entry is not None
        assert entry.fleet_ready is False
        assert entry.fleet_optional_ready is False
        assert entry.boot_phase == "booting_a"
        reader.close()
    finally:
        writer.close()


def test_titan_hcl_state_writer_round_trip_fleet_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test")
    writer = TitanHclStateWriter(titan_id="test")
    try:
        t0 = time.time()
        writer.update(fleet_ready=True, boot_phase="phase_a_done",
                      mandatory_total=3, mandatory_ready=3)
        reader = TitanHclStateReader(titan_id="test")
        entry = reader.read()
        assert entry is not None
        assert entry.fleet_ready is True
        assert entry.boot_phase == "phase_a_done"
        assert entry.mandatory_total == 3
        assert entry.mandatory_ready == 3
        # First-true latches the timestamp.
        assert entry.fleet_ready_at >= t0
        reader.close()
    finally:
        writer.close()


def test_titan_hcl_state_writer_first_true_latches_ts(tmp_path, monkeypatch):
    """Once fleet_ready latches True, subsequent updates that pass
    fleet_ready=True keep the original `fleet_ready_at` timestamp."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test")
    writer = TitanHclStateWriter(titan_id="test")
    try:
        writer.update(fleet_ready=True, boot_phase="phase_a_done")
        first_ts = writer.entry.fleet_ready_at
        assert first_ts > 0
        time.sleep(0.05)
        writer.update(fleet_ready=True, mandatory_ready=5)
        assert writer.entry.fleet_ready_at == first_ts, (
            "fleet_ready_at must latch on first True transition, "
            "not on every subsequent re-affirmation")
    finally:
        writer.close()


def test_titan_hcl_state_entry_wire_round_trip():
    """Wire encoding round-trips byte-identically for the dataclass."""
    original = TitanHclStateEntry(
        fleet_ready=True,
        fleet_optional_ready=True,
        boot_phase="quiescent",
        mandatory_total=13,
        mandatory_ready=13,
        post_boot_total=28,
        post_boot_ready=27,
        lazy_total=1,
        boot_started_at=1234567890.0,
        fleet_ready_at=1234567920.0,
        fleet_optional_ready_at=1234568000.0,
    )
    d = original.as_wire_dict()
    restored = TitanHclStateEntry.from_wire_dict(d)
    assert restored.fleet_ready == original.fleet_ready
    assert restored.fleet_optional_ready == original.fleet_optional_ready
    assert restored.boot_phase == original.boot_phase
    assert restored.mandatory_total == original.mandatory_total
    assert restored.mandatory_ready == original.mandatory_ready
    assert restored.post_boot_total == original.post_boot_total
    assert restored.post_boot_ready == original.post_boot_ready
    assert restored.lazy_total == original.lazy_total
    assert restored.boot_started_at == original.boot_started_at
    assert restored.fleet_ready_at == original.fleet_ready_at
    assert restored.fleet_optional_ready_at == original.fleet_optional_ready_at
