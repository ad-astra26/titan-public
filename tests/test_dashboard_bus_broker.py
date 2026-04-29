"""
Tests for Phase B.2.1 /v4/state.bus_broker exposure + kernel_rpc proxy.

Covers:
- TitanKernel.bus_broker_stats() returns None when broker is None
- TitanKernel.bus_broker_stats() returns broker.stats() when broker is set
- TitanKernel.bus_broker_stats() returns None on broker.stats() exception
- KERNEL_RPC_EXPOSED_METHODS includes both "bus_broker_stats" and
  "kernel.bus_broker_stats" so api_subprocess can proxy through
- /v4/state.bus_broker == None when no broker (mocked FakePlugin)
- /v4/state.bus_broker == broker.stats() dict when broker is mocked
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_plugin.api.dashboard import router
from titan_plugin.bus import DivineBus
from titan_plugin.core.kernel import KERNEL_RPC_EXPOSED_METHODS
from titan_plugin.guardian import Guardian


# ── KERNEL_RPC exposure ─────────────────────────────────────────────────


def test_kernel_rpc_exposes_bus_broker_stats():
    """KERNEL_RPC_EXPOSED_METHODS contains both top-level and kernel-prefixed paths."""
    assert "bus_broker_stats" in KERNEL_RPC_EXPOSED_METHODS
    assert "kernel.bus_broker_stats" in KERNEL_RPC_EXPOSED_METHODS


# ── TitanKernel.bus_broker_stats ────────────────────────────────────────


class _FakeKernel:
    """Minimal kernel stub with the bus_broker_stats method we want to test."""

    def __init__(self, broker=None):
        self._bus_broker = broker

    # Re-import the real method onto our stub for fidelity
    from titan_plugin.core.kernel import TitanKernel
    bus_broker_stats = TitanKernel.bus_broker_stats


def test_bus_broker_stats_returns_none_when_no_broker():
    kernel = _FakeKernel(broker=None)
    assert kernel.bus_broker_stats() is None


def test_bus_broker_stats_returns_dict_when_broker_set():
    fake_broker = MagicMock()
    fake_broker.stats.return_value = {
        "sock_path": "/tmp/titan_bus_T1.sock",
        "subscriber_count": 3,
        "subscribers": [{"name": "body_worker"}],
    }
    kernel = _FakeKernel(broker=fake_broker)
    out = kernel.bus_broker_stats()
    assert out is not None
    assert out["subscriber_count"] == 3
    assert out["sock_path"].endswith(".sock")


def test_bus_broker_stats_returns_none_on_broker_exception():
    fake_broker = MagicMock()
    fake_broker.stats.side_effect = RuntimeError("broker locked")
    kernel = _FakeKernel(broker=fake_broker)
    assert kernel.bus_broker_stats() is None


# ── /v4/state endpoint integration ──────────────────────────────────────


class _FakeSpiritProxy:
    def get_v4_state(self):
        return {"epoch": 12345, "consciousness": "active"}


class _FakePluginWithBroker:
    """Plugin stub returning bus_broker_stats from kernel."""

    def __init__(self, broker=None):
        self.bus = DivineBus(maxsize=100)
        self.guardian = Guardian(self.bus)
        self.spirit = _FakeSpiritProxy()
        self.kernel = _FakeKernel(broker=broker)

    def get_v3_status(self):  # required marker for /v4/state
        return {}


def _make_client(plugin):
    app = FastAPI()
    # dashboard.py reads two app.state attributes:
    #   - titan_state for typed sub-accessors (guardian.get_status, spirit, etc.)
    #   - titan_plugin for the kernel_rpc proxy that exposes kernel methods
    #     like kernel.bus_broker_stats (added in Phase B.2.1 hot-fix).
    # Our _FakePluginWithBroker stub plays both roles in tests — its .kernel
    # attribute is reached via either the StateAccessor (legacy) or the
    # kernel_rpc proxy (api_subprocess mode).
    app.state.titan_state = plugin
    app.state.titan_plugin = plugin
    app.include_router(router)
    return TestClient(app)


def test_v4_state_bus_broker_none_when_disabled():
    """No broker set → /v4/state.bus_broker == None."""
    plugin = _FakePluginWithBroker(broker=None)
    client = _make_client(plugin)
    resp = client.get("/v4/state")
    assert resp.status_code == 200
    body = resp.json()
    # _ok wraps response in {"ok": True, "data": {...}} or similar
    payload = body.get("data", body)
    assert payload.get("bus_broker") is None
    assert payload.get("v4") is True


def test_v4_state_bus_broker_dict_when_running():
    """Broker mocked → /v4/state.bus_broker contains stats dict."""
    fake_broker = MagicMock()
    fake_broker.stats.return_value = {
        "sock_path": "/tmp/titan_bus_T1.sock",
        "subscriber_count": 2,
        "subscribers": [
            {"name": "backup_worker", "ring_size": 0, "drop_count_60s": 0},
            {"name": "body_worker", "ring_size": 5, "drop_count_60s": 0},
        ],
    }
    plugin = _FakePluginWithBroker(broker=fake_broker)
    client = _make_client(plugin)
    resp = client.get("/v4/state")
    assert resp.status_code == 200
    body = resp.json()
    payload = body.get("data", body)
    bb = payload.get("bus_broker")
    assert bb is not None
    assert bb["subscriber_count"] == 2
    assert len(bb["subscribers"]) == 2
    assert bb["subscribers"][0]["name"] == "backup_worker"
