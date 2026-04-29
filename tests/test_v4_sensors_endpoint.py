"""Unit tests for the /v4/sensors endpoint.

rFP_phase1_sensory_wiring §3: exposes raw producer outputs
(system_sensor, network_monitor, timechain_v2 stats) alongside the current
blended outer_body composite.

Uses FastAPI TestClient + a stub plugin providing .network, .bus, and
.outer_state.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_plugin.api.dashboard import router as dashboard_router
from titan_plugin.logic import timechain_v2 as tcv2
from titan_plugin.utils import system_sensor as ss


@pytest.fixture
def app_client():
    # Clear producer state so tests are deterministic
    ss._reset_for_testing()
    tcv2._reset_rich_signal_state_for_testing()

    app = FastAPI()
    app.include_router(dashboard_router)

    plugin = MagicMock()

    # Network stub — single RPC URL
    plugin.network = MagicMock()
    plugin.network.rpc_urls = ["http://fake-rpc"]

    # Bus stub with stats dict
    plugin.bus = MagicMock()
    plugin.bus.stats = {
        "published": 1000,
        "dropped": 5,
        "routed": 2000,
        "modules": ["a", "b", "c", "d", "e"],
    }

    # State register stub with rich outer_body
    plugin.outer_state = MagicMock()
    plugin.outer_state.outer_body = [0.23, 0.85, 0.73, 0.05, 0.52]

    app.state.titan_plugin = plugin

    with TestClient(app) as client:
        yield client, plugin

    ss._reset_for_testing()
    tcv2._reset_rich_signal_state_for_testing()


def test_endpoint_returns_ok_shape(app_client):
    client, _ = app_client
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        resp = client.get("/v4/sensors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "data" in body


def test_endpoint_exposes_all_producers(app_client):
    client, _ = app_client
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]

    # All 4 producer sections present
    assert "system_sensor" in data
    assert "network_monitor" in data
    assert "tx_latency" in data
    assert "block_delta" in data
    # Plus composite + labels
    assert "outer_body" in data
    assert "outer_body_dims" in data


def test_outer_body_dims_are_v6_semantic_labels(app_client):
    client, _ = app_client
    resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    assert data["outer_body_dims"] == [
        "interoception", "proprioception", "somatosensation",
        "entropy", "thermal",
    ]


def test_outer_body_pulled_from_state_register(app_client):
    """Fallback path: when coordinator cache has no 130D state_vector,
    endpoint reads state_register directly. Most test fixtures land here
    because MagicMock plugin has no real coordinator snapshot."""
    client, _ = app_client
    with patch(
        "titan_plugin.api.dashboard._get_cached_coordinator_async",
        return_value={},  # no consciousness/state_vector → fall through
    ):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    assert data["outer_body"] == [0.23, 0.85, 0.73, 0.05, 0.52]
    assert data["outer_body_source"] == "state_register_fallback"


def test_outer_body_pulled_from_coordinator_state_vector(app_client):
    """Primary path: coordinator cache provides a 130D state_vector and
    outer_body is sliced from dims [65:70]. Mirrors the /v3/trinity source
    of truth so both endpoints return identical composites."""
    client, plugin = app_client
    # Set state_register outer_body to a DIFFERENT value so we can assert
    # that the coordinator path wins.
    plugin.outer_state.outer_body = [0.99, 0.99, 0.99, 0.99, 0.99]
    fake_state_vector = [0.0] * 132
    fake_state_vector[65:70] = [0.41, 0.88, 0.65, 0.20, 0.47]
    with patch(
        "titan_plugin.api.dashboard._get_cached_coordinator_async",
        return_value={"consciousness": {"state_vector": fake_state_vector}},
    ):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    assert data["outer_body"] == [0.41, 0.88, 0.65, 0.20, 0.47]
    assert data["outer_body_source"] == "coordinator_state_vector"


def test_outer_body_falls_back_when_state_vector_too_short(app_client):
    """If coordinator returns a too-short state_vector (malformed), the
    code must fall back to state_register rather than slicing past end."""
    client, _ = app_client
    with patch(
        "titan_plugin.api.dashboard._get_cached_coordinator_async",
        return_value={"consciousness": {"state_vector": [0.5] * 30}},
    ):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    assert data["outer_body"] == [0.23, 0.85, 0.73, 0.05, 0.52]
    assert data["outer_body_source"] == "state_register_fallback"


def test_system_sensor_fields_present(app_client):
    client, plugin = app_client
    with patch("os.getloadavg", return_value=(0.5, 0.5, 0.5)), \
         patch("os.cpu_count", return_value=2), \
         patch("os.path.isdir", return_value=False):  # no thermal
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    sys_stats = data["system_sensor"]
    assert sys_stats is not None
    assert set(sys_stats.keys()) == {
        "cpu_load", "cpu_thermal", "circadian_phase", "cpu_spike_rate"
    }
    for v in sys_stats.values():
        assert 0.0 <= v <= 1.0


def test_network_monitor_fields_present(app_client):
    client, _ = app_client
    with patch("urllib.request.urlopen", side_effect=OSError("no net")):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    net_stats = data["network_monitor"]
    assert net_stats is not None
    assert set(net_stats.keys()) == {
        "peer_entropy", "ping_variance", "bus_drop_rate", "bus_module_diversity"
    }


def test_tx_latency_shape(app_client):
    client, _ = app_client
    with patch("urllib.request.urlopen", side_effect=OSError("no net")):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    tx = data["tx_latency"]
    assert tx is not None
    assert set(tx.keys()) == {"samples", "median_s", "p95_s", "normalized"}


def test_tx_latency_reflects_recorded_samples(app_client):
    client, _ = app_client
    # Record a few TX latencies; endpoint should show them
    tcv2.record_tx_latency(1.5)
    tcv2.record_tx_latency(2.0)
    tcv2.record_tx_latency(3.0)
    with patch("urllib.request.urlopen", side_effect=OSError("no net")):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    assert data["tx_latency"]["samples"] == 3
    assert data["tx_latency"]["median_s"] == 2.0


def test_block_delta_shape(app_client):
    client, _ = app_client
    with patch("urllib.request.urlopen", side_effect=OSError("no net")):
        resp = client.get("/v4/sensors")
    data = resp.json()["data"]
    bd = data["block_delta"]
    assert bd is not None
    assert set(bd.keys()) == {
        "samples", "latest_height", "blocks_per_min", "normalized"
    }


def test_endpoint_survives_state_register_missing(app_client):
    """If plugin has no outer_state AND no coordinator data, endpoint
    returns default fallback."""
    client, plugin = app_client
    plugin.outer_state = None
    plugin.state_register = None
    # MagicMock's attribute-missing returns another MagicMock — simulate real
    # absence by reassigning. The endpoint's getattr(..., None) path handles it.
    del plugin.outer_state
    del plugin.state_register
    with patch("urllib.request.urlopen", side_effect=OSError("no net")), \
         patch(
             "titan_plugin.api.dashboard._get_cached_coordinator_async",
             return_value={},
         ):
        resp = client.get("/v4/sensors")
    assert resp.status_code == 200
    data = resp.json()["data"]
    # Final fallback path triggers: outer_body defaults
    assert data["outer_body"] == [0.5] * 5
    assert data["outer_body_source"] == "default_fallback"


def test_endpoint_never_500s_on_producer_failures(app_client):
    """Every producer wrapped in try/except. If one crashes, endpoint still returns 200."""
    client, _ = app_client
    # Patch one producer to raise an exception internally
    with patch("titan_plugin.utils.system_sensor.get_all_stats",
               side_effect=RuntimeError("sensor boom")), \
         patch("urllib.request.urlopen", side_effect=OSError("no net")):
        resp = client.get("/v4/sensors")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["system_sensor"] is None
    assert "system_sensor_error" in data
    # Other producers unaffected
    assert data["network_monitor"] is not None
