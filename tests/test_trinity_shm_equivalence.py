"""
CRITICAL GATE: /v4/trinity-shm must return byte-identical Trinity state
whether served from shm (microkernel.shm_trinity_enabled=true) or from
the legacy state_register path (flag=false).

If this test fails, shm_trinity_enabled MUST NOT be flipped on any Titan.

Strategy:
  1. Build a deterministic state_register populated with known 130D felt
     + 30D topology + consciousness curvature/density values.
  2. Instantiate the endpoint against that state_register with flag=false →
     captures legacy JSON.
  3. Write the same state to a test shm registry via StateRegistryWriter.
  4. Instantiate the endpoint with flag=true → captures shm JSON.
  5. Assert numerical equivalence (within float32 round-trip tolerance).

Reference: titan-docs/PLAN_microkernel_phase_a.md §5.5.2
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_plugin.api.dashboard import router
from titan_plugin.core.state_registry import (
    TRINITY_STATE,
    RegistryBank,
    StateRegistryWriter,
)


class _FakeStateRegister:
    """Minimal state_register exposing the 3 methods the endpoint calls."""

    def __init__(self, felt_130, topo_30, curvature, density):
        self._felt = list(felt_130)
        self._topo = list(topo_30)
        self._curvature = curvature
        self._density = density

    def get_full_130dt(self):
        return list(self._felt)

    def get_full_30d_topology(self):
        return list(self._topo)

    def snapshot(self):
        return {
            "consciousness": {
                "curvature": self._curvature,
                "density": self._density,
            },
        }

    def age_seconds(self):
        return 0.5


class _FakePlugin:
    def __init__(self, state_register, registry_bank=None):
        self.state_register = state_register
        self._registry_bank = registry_bank


@pytest.fixture
def deterministic_state():
    """Build a fully-specified state (no zeros or 0.5 defaults everywhere)."""
    rng = np.random.default_rng(seed=20260424)
    felt_130 = (rng.random(130, dtype=np.float32)).tolist()
    # Topology is signed-centered (uses 0.0 pad in OuterState), so include
    # negative values to exercise full float32 range.
    topo_30 = ((rng.random(30, dtype=np.float32) * 2.0 - 1.0)).tolist()
    curvature = 2.3456
    density = 0.0789
    return felt_130, topo_30, curvature, density


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _make_client(plugin):
    app = FastAPI()
    # Post-S5-amendment: dashboard reads via `app.state.titan_state`
    # (StateAccessor pattern); legacy api/ modules still use
    # `titan_plugin`. Set both for fixture neutrality.
    app.state.titan_plugin = plugin
    app.state.titan_state = plugin
    app.include_router(router)
    return TestClient(app)


def test_legacy_path_returns_expected_layout(deterministic_state):
    felt_130, topo_30, curv, dens = deterministic_state
    reg = _FakeStateRegister(felt_130, topo_30, curv, dens)
    plugin = _FakePlugin(reg, registry_bank=None)  # no bank → legacy only
    with _make_client(plugin) as client:
        resp = client.get("/v4/trinity-shm")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["source"] == "legacy"
    assert len(data["full_130dt"]) == 130
    assert len(data["full_30d_topology"]) == 30
    assert len(data["journey"]) == 2
    # Values should round-trip (JSON carries floats fine within precision)
    np.testing.assert_allclose(data["full_130dt"], felt_130, rtol=1e-6)
    np.testing.assert_allclose(data["full_30d_topology"], topo_30, rtol=1e-6)
    np.testing.assert_allclose(data["journey"], [curv, dens], rtol=1e-6)


def test_shm_path_returns_same_layout_when_flag_on(deterministic_state, shm_root):
    """The critical equivalence gate."""
    felt_130, topo_30, curv, dens = deterministic_state

    # Bank with flag ON
    bank = RegistryBank(
        titan_id="TEST",
        config={"microkernel": {"shm_trinity_enabled": True}},
    )
    bank.shm_root = shm_root

    # Write the same state to shm via the writer.
    arr_162 = np.asarray(
        list(felt_130) + list(topo_30) + [curv, dens],
        dtype=np.float32,
    )
    writer = StateRegistryWriter(TRINITY_STATE, shm_root)
    writer.write(arr_162)

    reg = _FakeStateRegister(felt_130, topo_30, curv, dens)
    plugin = _FakePlugin(reg, registry_bank=bank)
    with _make_client(plugin) as client:
        resp = client.get("/v4/trinity-shm")
    writer.close()

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["source"] == "shm", "flag is on but endpoint fell back to legacy"
    assert len(data["full_130dt"]) == 130
    assert len(data["full_30d_topology"]) == 30
    assert len(data["journey"]) == 2
    assert data["seq"] is not None and data["seq"] > 0
    # shm path stores as float32 → reads back float32 → JSON serialization
    # loses nothing. Equivalence to original within float32 precision.
    np.testing.assert_allclose(data["full_130dt"], felt_130, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data["full_30d_topology"], topo_30, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data["journey"], [curv, dens], rtol=1e-5, atol=1e-6)


def test_shm_and_legacy_paths_produce_numerically_equivalent_output(
        deterministic_state, shm_root):
    """
    CRITICAL GATE: endpoint output must be numerically equivalent whether
    served from shm (flag=on) or legacy (flag=off).

    This is THE test that determines whether shm_trinity_enabled=true is
    safe to flip on a production Titan.
    """
    felt_130, topo_30, curv, dens = deterministic_state

    # ── Flag OFF: legacy path ────────────────────────────────────────
    bank_off = RegistryBank(
        titan_id="TEST_OFF",
        config={"microkernel": {"shm_trinity_enabled": False}},
    )
    bank_off.shm_root = shm_root
    reg_off = _FakeStateRegister(felt_130, topo_30, curv, dens)
    plugin_off = _FakePlugin(reg_off, registry_bank=bank_off)
    with _make_client(plugin_off) as client_off:
        resp_off = client_off.get("/v4/trinity-shm")
    data_off = resp_off.json()["data"]
    assert data_off["source"] == "legacy"

    # ── Flag ON: shm path (same underlying state) ────────────────────
    bank_on = RegistryBank(
        titan_id="TEST_ON",
        config={"microkernel": {"shm_trinity_enabled": True}},
    )
    bank_on.shm_root = shm_root
    arr_162 = np.asarray(
        list(felt_130) + list(topo_30) + [curv, dens],
        dtype=np.float32,
    )
    writer = StateRegistryWriter(TRINITY_STATE, shm_root)
    writer.write(arr_162)

    reg_on = _FakeStateRegister(felt_130, topo_30, curv, dens)
    plugin_on = _FakePlugin(reg_on, registry_bank=bank_on)
    with _make_client(plugin_on) as client_on:
        resp_on = client_on.get("/v4/trinity-shm")
    writer.close()
    data_on = resp_on.json()["data"]
    assert data_on["source"] == "shm"

    # ── Numerical equivalence across both paths ─────────────────────
    np.testing.assert_allclose(
        data_on["full_130dt"], data_off["full_130dt"],
        rtol=1e-5, atol=1e-6,
        err_msg="full_130dt diverges between shm and legacy paths",
    )
    np.testing.assert_allclose(
        data_on["full_30d_topology"], data_off["full_30d_topology"],
        rtol=1e-5, atol=1e-6,
        err_msg="full_30d_topology diverges between shm and legacy paths",
    )
    np.testing.assert_allclose(
        data_on["journey"], data_off["journey"],
        rtol=1e-5, atol=1e-6,
        err_msg="journey diverges between shm and legacy paths",
    )


def test_shm_fallback_when_file_missing_even_with_flag_on(
        deterministic_state, shm_root):
    """Flag on but no shm file → endpoint falls back to legacy, no error."""
    felt_130, topo_30, curv, dens = deterministic_state
    bank = RegistryBank(
        titan_id="TEST_MISSING",
        config={"microkernel": {"shm_trinity_enabled": True}},
    )
    bank.shm_root = shm_root
    # DO NOT create the file via writer.

    reg = _FakeStateRegister(felt_130, topo_30, curv, dens)
    plugin = _FakePlugin(reg, registry_bank=bank)
    with _make_client(plugin) as client:
        resp = client.get("/v4/trinity-shm")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["source"] == "legacy"  # fallback engaged
