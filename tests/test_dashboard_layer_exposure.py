"""
Tests for Microkernel v2 Phase A §A.5 — layer exposure in dashboard endpoints.

Covers:
  - /v3/guardian returns layer_stats + per-module layer
  - /v4/layers returns layer_stats + modules_by_layer

Uses FastAPI TestClient against an in-process Guardian fixture (no real
subprocess spawn).

Reference: titan-docs/PLAN_microkernel_phase_a.md §4.1.3
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_hcl.api.dashboard import router
from titan_hcl.bus import DivineBus
from titan_hcl.guardian import Guardian, ModuleSpec, ModuleState


def _noop_entry(recv_q, send_q, name, config):  # pragma: no cover
    pass


class _FakePlugin:
    """Minimal plugin stub exposing only the attributes the endpoints touch."""

    def __init__(self, guardian: Guardian):
        self.guardian = guardian
        self.bus = guardian.bus


@pytest.fixture
def client():
    bus = DivineBus(maxsize=100)
    guardian = Guardian(bus)
    # Seed a mix across all 4 layers.
    guardian.register(ModuleSpec(name="body", entry_fn=_noop_entry, layer="L1"))
    guardian.register(ModuleSpec(name="spirit", entry_fn=_noop_entry, layer="L1"))
    guardian.register(ModuleSpec(name="memory", entry_fn=_noop_entry, layer="L2"))
    guardian.register(ModuleSpec(name="cgn", entry_fn=_noop_entry, layer="L2"))
    guardian.register(ModuleSpec(name="timechain", entry_fn=_noop_entry, layer="L2"))
    guardian.register(ModuleSpec(name="llm", entry_fn=_noop_entry, layer="L3"))
    guardian.register(ModuleSpec(name="knowledge", entry_fn=_noop_entry, layer="L3"))
    # Simulate some runtime states
    guardian._modules["body"].state = ModuleState.RUNNING
    guardian._modules["memory"].state = ModuleState.RUNNING
    guardian._modules["llm"].state = ModuleState.CRASHED

    app = FastAPI()
    plugin = _FakePlugin(guardian)
    # Post-S5-amendment: dashboard reads via `app.state.titan_state`
    # (StateAccessor pattern). Some other api/ modules still use
    # the legacy `titan_hcl` name; set both for fixture neutrality.
    app.state.titan_hcl = plugin
    app.state.titan_state = plugin
    app.include_router(router)
    # Phase E: mount the v6 roof + legacy /v3,/v4→/v6 redirects so
    # deprecated paths resolve via 301/308 to the live v6 handler.
    from titan_hcl.api.v6 import router as _v6_router
    from titan_hcl.api.v6_deprecation import router as _v6_dep_router
    app.include_router(_v6_router)
    app.include_router(_v6_dep_router)
    with TestClient(app) as c:
        yield c


def test_v3_guardian_includes_layer_stats(client):
    resp = client.get("/v3/guardian")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    data = body["data"]
    assert "layer_stats" in data
    stats = data["layer_stats"]
    assert stats["L1"]["total"] == 2
    assert stats["L2"]["total"] == 3
    assert stats["L3"]["total"] == 2
    assert stats["L0"]["total"] == 0


def test_v3_guardian_modules_include_layer_per_module(client):
    resp = client.get("/v3/guardian")
    data = resp.json()["data"]
    modules = data["modules"]
    assert modules["body"]["layer"] == "L1"
    assert modules["cgn"]["layer"] == "L2"
    assert modules["llm"]["layer"] == "L3"


def test_v4_layers_returns_modules_by_layer(client):
    resp = client.get("/v4/layers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    data = body["data"]
    assert set(data["modules_by_layer"].keys()) == {"L0", "L1", "L2", "L3"}
    assert data["modules_by_layer"]["L1"] == ["body", "spirit"]
    assert data["modules_by_layer"]["L2"] == ["cgn", "memory", "timechain"]
    assert data["modules_by_layer"]["L3"] == ["knowledge", "llm"]
    assert data["modules_by_layer"]["L0"] == []


def test_v4_layers_stats_include_running_and_crashed(client):
    resp = client.get("/v4/layers")
    data = resp.json()["data"]
    stats = data["layer_stats"]
    assert stats["L1"]["running"] == 1  # body is RUNNING
    assert stats["L2"]["running"] == 1  # memory is RUNNING
    assert stats["L3"]["crashed"] == 1  # llm is CRASHED
