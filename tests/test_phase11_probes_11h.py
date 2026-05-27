"""
Phase 11 §11.I.3 / D-SPEC-141 / v1.65.0 — per-module probes (Chunk 11H).

Covers:
  1. PROBE_REGISTRY exhaustiveness — 10 entries per RFP §3H.2.
  2. Each probe returns ProbeResult.ok_() under the 2s budget (shell
     implementation per the 11H plan; bodies fleshed out in 11I).
  3. End-to-end via worker-side handler `handle_module_probe_request`:
     each probe runs cleanly, writes state=RUNNING to SHM, and
     publishes MODULE_PROBE_RESPONSE to the send queue.
  4. Catalog wiring — `build_catalog` attaches probe_fn to each of
     the 10 heaviest workers' ModuleSpec.
"""
from __future__ import annotations

import queue as _queue_mod
import time
from typing import Any

import pytest

from titan_hcl.bus import DivineBus, MODULE_PROBE_REQUEST
from titan_hcl.core.module_state import (
    BootPriority,
    ModuleStateWriter,
    ProbeResult,
)
from titan_hcl.core.probe_dispatcher import (
    PROBE_TIMEOUT_S,
    handle_module_probe_request,
)
from titan_hcl.orchestrator import Orchestrator
from titan_hcl.probes import (
    PROBE_REGISTRY,
    agno_worker_probe,
    cgn_probe,
    cognitive_worker_probe,
    expression_worker_probe,
    meditation_probe,
    memory_probe,
    observatory_probe,
    output_verifier_probe,
    social_worker_probe,
    synthesis_probe,
)


# ── 1. Roster coverage ────────────────────────────────────────────────


CANONICAL_ROSTER = {
    "agno_worker",
    "cognitive_worker",
    "memory",
    "cgn",
    "synthesis",
    "observatory",
    "social_worker",
    "expression_worker",
    "meditation",
    "output_verifier",
}


def test_probe_registry_has_ten_canonical_entries():
    """RFP §3H.2 enumerates 10 heaviest workers — PROBE_REGISTRY must
    cover exactly those, no more no less."""
    assert set(PROBE_REGISTRY.keys()) == CANONICAL_ROSTER, (
        f"Registry mismatch: {set(PROBE_REGISTRY.keys()) ^ CANONICAL_ROSTER}")


def test_each_probe_importable_under_canonical_name():
    """Each module name in the roster has a top-level export under
    `titan_hcl.probes` with the convention `<name>_probe` (or
    `<short_name>_probe` for cgn/memory/observatory/synthesis/meditation)."""
    expected_callables = {
        "agno_worker": agno_worker_probe,
        "cognitive_worker": cognitive_worker_probe,
        "memory": memory_probe,
        "cgn": cgn_probe,
        "synthesis": synthesis_probe,
        "observatory": observatory_probe,
        "social_worker": social_worker_probe,
        "expression_worker": expression_worker_probe,
        "meditation": meditation_probe,
        "output_verifier": output_verifier_probe,
    }
    for name, fn in expected_callables.items():
        assert PROBE_REGISTRY[name] is fn, (
            f"Registry entry for {name!r} should be the top-level export")


# ── 2. Each probe returns ok ≤ budget ────────────────────────────────


# Probes with REAL bodies (Chunk 11N onwards) — readiness sentinels live on
# the worker module and must be flipped True before the probe returns ok.
# Shells return ok unconditionally so they don't need anything.
_REAL_PROBE_SENTINELS: dict[str, list[tuple[str, str]]] = {
    "agno_worker": [
        ("titan_hcl.modules.agno_worker", "_AGENT_READY"),
        ("titan_hcl.modules.agno_worker", "_OVG_READY"),
    ],
}


def _set_worker_sentinels(name: str, value: bool) -> None:
    import importlib
    for mod_path, attr in _REAL_PROBE_SENTINELS.get(name, []):
        mod = importlib.import_module(mod_path)
        setattr(mod, attr, value)


@pytest.mark.parametrize("name,probe_fn", list(PROBE_REGISTRY.items()))
def test_probe_returns_ok_under_budget(name: str, probe_fn):
    """Per SPEC §11.I.3 budget ≤2s. Probes with real bodies (per the
    11N migration) need their module-level readiness sentinels flipped
    True; shells return ok unconditionally."""
    _set_worker_sentinels(name, True)
    try:
        t0 = time.perf_counter()
        result = probe_fn(None)
        elapsed_s = time.perf_counter() - t0
        assert isinstance(result, ProbeResult), (
            f"Probe {name!r} must return a ProbeResult; got {type(result).__name__}")
        assert result.ok is True, (
            f"Probe {name!r} should return ok=True with sentinels True; got {result}")
        assert elapsed_s < PROBE_TIMEOUT_S, (
            f"Probe {name!r} took {elapsed_s:.3f}s > {PROBE_TIMEOUT_S}s budget")
        # The probe must record a real latency (used by orchestrator-side
        # histograms even on the shell path).
        assert result.latency_ms >= 0.0
    finally:
        _set_worker_sentinels(name, False)


# ── 3. End-to-end via worker-side handler ────────────────────────────


@pytest.mark.parametrize("name,probe_fn", list(PROBE_REGISTRY.items()))
def test_probe_through_worker_side_handler(
        name: str, probe_fn, tmp_path, monkeypatch):
    """When the worker recv-loop invokes `handle_module_probe_request`
    with the probe_fn from PROBE_REGISTRY, the handler writes state=running
    to the SHM slot AND emits MODULE_PROBE_RESPONSE to the send queue."""
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    monkeypatch.setenv("TITAN_ID", "test_h")
    _set_worker_sentinels(name, True)
    writer = ModuleStateWriter(
        module_name=name, layer="L2",
        boot_priority=BootPriority.MANDATORY, titan_id="test_h", pid=12345,
    )
    try:
        send_q = _queue_mod.Queue()
        msg = {
            "type": MODULE_PROBE_REQUEST,
            "src": "titan_hcl",
            "dst": name,
            "rid": "test-probe-id",
            "payload": {"name": name, "probe_id": "test-probe-id"},
        }
        result = handle_module_probe_request(
            msg, probe_fn=probe_fn, send_queue=send_q,
            module_name=name, state_writer=writer,
        )
        assert result.ok is True, f"Probe {name!r} via handler failed: {result}"
        # Response message published to send_queue.
        assert not send_q.empty()
        resp = send_q.get_nowait()
        assert resp["type"] == "MODULE_PROBE_RESPONSE"
        assert resp["src"] == name
        assert resp["dst"] == "titan_hcl"
        assert resp["rid"] == "test-probe-id"
        assert resp["payload"]["probe_id"] == "test-probe-id"
        assert resp["payload"]["result"]["ok"] is True
    finally:
        writer.close()
        _set_worker_sentinels(name, False)


def test_agno_probe_fails_when_sentinels_cold():
    """Chunk 11N — agno probe with cold sentinels returns ok=False +
    a ModuleError envelope explaining which sentinel is still False, so
    /v6/errors makes the cause obvious."""
    from titan_hcl.modules import agno_worker as _a
    _a._AGENT_READY = False
    _a._OVG_READY = False
    result = agno_worker_probe(None)
    assert result.ok is False
    assert result.error_envelope is not None
    assert "_AGENT_READY" in result.error_envelope.message
    assert "_OVG_READY" in result.error_envelope.message


def test_agno_probe_fails_when_only_agent_ready():
    """Half-warm worker — Agent built but OVG didn't warm — still fails."""
    from titan_hcl.modules import agno_worker as _a
    _a._AGENT_READY = True
    _a._OVG_READY = False
    try:
        result = agno_worker_probe(None)
        assert result.ok is False
        assert result.error_envelope is not None
        assert "_OVG_READY" in result.error_envelope.message
        assert "_AGENT_READY" not in result.error_envelope.message
    finally:
        _a._AGENT_READY = False
        _a._OVG_READY = False


# ── 4. Catalog wiring (build_catalog attaches probe_fn) ──────────────


def _build_catalog_with_all_flags_on() -> Orchestrator:
    """Mirror of the 11G fixture — every flag-gated worker enabled so the
    10 heaviest are all in the registry."""
    bus = DivineBus()
    orch = Orchestrator(bus)
    cfg = {
        "microkernel": {
            "a8_output_verifier_subprocess_enabled": True,
            "a8_reflex_subprocess_enabled": True,
            "a8_agency_subprocess_enabled": True,
            "a8_sage_scholar_gatekeeper_subprocess_enabled": True,
            "outer_interface_worker_enabled": True,
            "spawn_graduated_workers_enabled": False,
            "l0_rust_enabled": True,
            "social_worker_enabled": True,
        },
        "persistence": {"enabled": True},
        "memory_and_storage": {"data_dir": "./data"},
        "inference": {},
        "stealth_sage": {},
        "expressive": {},
        "studio": {},
        "info_banner": {},
        "outer_interface": {},
        "self_exploration": {},
        "action_decoder": {},
        "action_narrator": {},
        "kin": {},
    }
    from titan_hcl.module_catalog import build_catalog
    build_catalog(bus, orch, cfg, titan_id="test")
    return orch


def test_catalog_wires_probe_fn_for_all_heaviest_workers():
    """Every module in PROBE_REGISTRY must end up with `spec.probe_fn`
    set on its ModuleInfo after build_catalog runs."""
    orch = _build_catalog_with_all_flags_on()
    for name, expected_fn in PROBE_REGISTRY.items():
        assert name in orch._modules, (
            f"Probe target {name!r} not registered — flag-gated off in "
            f"the test fixture?")
        info = orch._modules[name]
        assert info.spec.probe_fn is expected_fn, (
            f"Probe wiring missed for {name!r}: spec.probe_fn = "
            f"{info.spec.probe_fn!r}, expected {expected_fn!r}")


def test_modules_outside_roster_keep_default_probe_fn_none():
    """Modules NOT in PROBE_REGISTRY must keep `probe_fn=None` so the
    worker-side handler returns ProbeResult.ok_() per §11.I.2 trivial-pass."""
    orch = _build_catalog_with_all_flags_on()
    for name, info in orch._modules.items():
        if name in PROBE_REGISTRY:
            continue
        assert info.spec.probe_fn is None, (
            f"Module {name!r} outside the §3H.2 roster has unexpectedly "
            f"wired probe_fn = {info.spec.probe_fn!r}")
