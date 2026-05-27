"""
Phase 11 §11.I.1 / D-SPEC-141 — orchestrator/supervisor role split (11E.b.1).

Covers:
  1. Canonical imports — `Orchestrator` from `titan_hcl.orchestrator`,
     `Supervisor` from `titan_hcl.supervisor`.
  2. Back-compat — `Guardian` re-export from `titan_hcl.guardian_hcl` is
     the SAME class as `Orchestrator` (alias, not subclass) so all 30+
     existing callsites + tests keep working unchanged.
  3. Locked D5 routing — `Supervisor.publish_module_restart_request(name,
     reason)` emits a bus event of type `MODULE_RESTART_REQUEST` with the
     correct payload + destination, rather than calling
     `Orchestrator.restart_async` directly.
  4. Forward methods (is_running / is_started / get_status / get_layer /
     layer_stats / monitor_tick) delegate to the wrapped Orchestrator.
"""
from __future__ import annotations

import logging

import pytest

from titan_hcl.bus import DivineBus, MODULE_RESTART_REQUEST
from titan_hcl.orchestrator import (
    Guardian as GuardianFromOrchestrator,
    ModuleSpec,
    ModuleState,
    Orchestrator,
)
from titan_hcl.supervisor import Supervisor


def _dummy_entry(*_a, **_kw) -> None:
    pass


def _spec(name: str, layer: str = "L3") -> ModuleSpec:
    return ModuleSpec(
        name=name, layer=layer, entry_fn=_dummy_entry,
        autostart=False, restart_on_crash=False,
    )


# ── 1. Canonical imports ────────────────────────────────────────────


def test_orchestrator_importable_from_canonical_path():
    """`from titan_hcl.orchestrator import Orchestrator` resolves."""
    assert Orchestrator is not None
    assert isinstance(Orchestrator, type)


def test_supervisor_importable_from_canonical_path():
    """`from titan_hcl.supervisor import Supervisor` resolves."""
    assert Supervisor is not None
    assert isinstance(Supervisor, type)


# ── 2. Back-compat alias ────────────────────────────────────────────


def test_guardian_is_orchestrator_alias_in_orchestrator_pkg():
    """Phase 11 §11.I.1 — `Guardian` is the literal class object as
    `Orchestrator`, not a subclass — keeps `isinstance(x, Guardian)` /
    `Guardian.method` parity for the 30+ legacy callsites."""
    assert GuardianFromOrchestrator is Orchestrator


def test_guardian_alias_holds_in_guardian_hcl_pkg():
    """`from titan_hcl.guardian_hcl import Guardian` returns the same
    class object as `Orchestrator`."""
    from titan_hcl.guardian_hcl import Guardian as GuardianFromGuardianHcl
    assert GuardianFromGuardianHcl is Orchestrator


def test_guardian_alias_holds_via_guardian_hcl_core_path():
    """Legacy `from titan_hcl.guardian_hcl.core import Guardian` still works."""
    from titan_hcl.guardian_hcl.core import Guardian as GuardianViaCore
    assert GuardianViaCore is Orchestrator


def test_dep_activation_mixin_back_compat_alias():
    """`GuardianDepActivationMixin` is preserved as an alias for the
    renamed `OrchestratorDepActivationMixin`."""
    from titan_hcl.guardian_hcl import (
        GuardianDepActivationMixin,
        OrchestratorDepActivationMixin,
    )
    assert GuardianDepActivationMixin is OrchestratorDepActivationMixin


def test_module_registry_dataclasses_re_exported():
    """ModuleSpec / ModuleState / ModuleInfo / ReloadState resolve under
    both the canonical (`titan_hcl.orchestrator.module_registry`) and
    the back-compat (`titan_hcl.guardian_hcl.module_registry`) paths."""
    from titan_hcl.orchestrator.module_registry import (
        ModuleSpec as ModuleSpecCanonical,
        ModuleState as ModuleStateCanonical,
    )
    from titan_hcl.guardian_hcl.module_registry import (
        ModuleSpec as ModuleSpecLegacy,
        ModuleState as ModuleStateLegacy,
    )
    assert ModuleSpecCanonical is ModuleSpecLegacy
    assert ModuleStateCanonical is ModuleStateLegacy


# ── 3. Locked D5 routing — Supervisor publishes MODULE_RESTART_REQUEST ─


def test_supervisor_publish_module_restart_request_emits_bus_event():
    """Locked D5: instead of calling orchestrator.restart_async directly,
    the Supervisor publishes MODULE_RESTART_REQUEST(name, reason) so the
    orchestrator's lifecycle-request subscriber re-enters via the bus."""
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("agno_worker"))
    supervisor = Supervisor(bus, orchestrator)

    captured: list[dict] = []
    original_publish = bus.publish

    def spy(msg, *args, **kwargs):
        if isinstance(msg, dict) and msg.get("type") == MODULE_RESTART_REQUEST:
            captured.append(msg)
        return original_publish(msg, *args, **kwargs)

    bus.publish = spy  # type: ignore[method-assign]

    supervisor.publish_module_restart_request(
        "agno_worker", reason="heartbeat_timeout")

    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == MODULE_RESTART_REQUEST
    assert msg["src"] == "supervisor"
    assert msg["dst"] == "guardian_hcl_lifecycle"
    assert msg["payload"]["name"] == "agno_worker"
    assert msg["payload"]["reason"] == "heartbeat_timeout"


def test_supervisor_publish_module_restart_request_passes_extra_kwargs():
    """`extra` kwargs propagate into the payload (e.g. save_first=False
    for fast-restart paths)."""
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("cgn"))
    supervisor = Supervisor(bus, orchestrator)

    captured: list[dict] = []
    original_publish = bus.publish

    def spy(msg, *args, **kwargs):
        if isinstance(msg, dict) and msg.get("type") == MODULE_RESTART_REQUEST:
            captured.append(msg)
        return original_publish(msg, *args, **kwargs)

    bus.publish = spy  # type: ignore[method-assign]

    supervisor.publish_module_restart_request(
        "cgn", reason="rss_overflow", save_first=False, save_timeout=5.0)

    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["name"] == "cgn"
    assert payload["reason"] == "rss_overflow"
    assert payload["save_first"] is False
    assert payload["save_timeout"] == 5.0


def test_supervisor_publish_routing_does_not_short_circuit_orchestrator():
    """The Supervisor's publish path does NOT touch orchestrator state
    directly — the orchestrator's lifecycle thread is the sole executor."""
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("memory"))
    supervisor = Supervisor(bus, orchestrator)

    pre_state = orchestrator._modules["memory"].state
    supervisor.publish_module_restart_request("memory", reason="rss")
    post_state = orchestrator._modules["memory"].state

    # Supervisor only publishes — it does not call restart()/restart_async.
    # State remains untouched until the orchestrator's MODULE_RESTART_REQUEST
    # subscriber thread picks the message up.
    assert pre_state == post_state


# ── 4. Forwarded status surface ─────────────────────────────────────


def test_supervisor_forwards_is_running():
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("memory"))
    supervisor = Supervisor(bus, orchestrator)

    assert supervisor.is_running("memory") is False
    orchestrator._modules["memory"].state = ModuleState.RUNNING
    assert supervisor.is_running("memory") is True


def test_supervisor_forwards_is_started():
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("memory"))
    supervisor = Supervisor(bus, orchestrator)

    assert supervisor.is_started("memory") is False
    orchestrator._modules["memory"].state = ModuleState.STARTING
    assert supervisor.is_started("memory") is True


def test_supervisor_forwards_get_status_and_layer():
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("memory", layer="L2"))
    supervisor = Supervisor(bus, orchestrator)

    status = supervisor.get_status()
    assert "memory" in status
    assert status["memory"]["layer"] == "L2"
    assert supervisor.get_layer("memory") == "L2"
    assert supervisor.get_layer("unknown") is None


def test_supervisor_forwards_layer_stats():
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    orchestrator.register(_spec("memory", layer="L2"))
    orchestrator.register(_spec("agno", layer="L2"))
    orchestrator.register(_spec("api", layer="L3"))
    supervisor = Supervisor(bus, orchestrator)

    stats = supervisor.layer_stats()
    assert stats["L2"]["total"] == 2
    assert stats["L3"]["total"] == 1


def test_supervisor_monitor_tick_delegates_to_orchestrator():
    """11E.b.1 surface — monitor_tick forwards to orchestrator. The
    full SHM-poll body lands in 11E.b.2 alongside kernel-rs peer-spawn."""
    bus = DivineBus()
    orchestrator = Orchestrator(bus)
    supervisor = Supervisor(bus, orchestrator)

    call_count = [0]
    original_tick = orchestrator.monitor_tick

    def counted_tick():
        call_count[0] += 1
        return original_tick()

    orchestrator.monitor_tick = counted_tick  # type: ignore[method-assign]
    supervisor.monitor_tick()
    assert call_count[0] == 1
