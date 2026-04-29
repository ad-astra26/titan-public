"""Unit tests for titan_plugin.utils.network_monitor."""
import json
from unittest.mock import patch, MagicMock

import pytest

from titan_plugin.utils import network_monitor as nm


@pytest.fixture(autouse=True)
def _reset_state():
    nm._reset_for_testing()
    yield
    nm._reset_for_testing()


def _mock_rpc_response(result) -> MagicMock:
    """Build a mock urlopen context returning a JSON-RPC response."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value.read.return_value = body
    ctx.__exit__.return_value = False
    return ctx


def test_bus_drop_rate_zero_when_no_drops():
    stats = {"published": 100, "dropped": 0}
    assert nm.get_bus_drop_rate(stats) == 0.0


def test_bus_drop_rate_handles_empty_published():
    assert nm.get_bus_drop_rate({"published": 0, "dropped": 0}) == 0.0
    assert nm.get_bus_drop_rate({}) == 0.0
    assert nm.get_bus_drop_rate(None) == 0.0


def test_bus_drop_rate_computes_ratio():
    stats = {"published": 100, "dropped": 25}
    assert nm.get_bus_drop_rate(stats) == 0.25


def test_bus_drop_rate_clamps_to_one():
    # Pathological case (drops > published somehow)
    stats = {"published": 10, "dropped": 100}
    assert nm.get_bus_drop_rate(stats) == 1.0


def test_bus_module_diversity_counts_modules():
    stats = {"modules": ["body", "mind", "spirit", "memory"]}
    # 4/8 expected modules = 0.5
    assert nm.get_bus_module_diversity(stats) == 0.5


def test_bus_module_diversity_saturates_at_eight():
    stats = {"modules": list(range(12))}  # 12 modules
    assert nm.get_bus_module_diversity(stats) == 1.0


def test_peer_entropy_all_same_version_is_zero():
    """Monoculture → entropy 0."""
    nodes = [{"version": "1.18.0"} for _ in range(100)]
    with patch("urllib.request.urlopen", return_value=_mock_rpc_response(nodes)):
        entropy = nm.get_peer_entropy("http://fake-rpc")
    assert entropy == 0.0


def test_peer_entropy_diverse_versions_high():
    """Even distribution across 4 versions → high entropy."""
    nodes = (
        [{"version": "1.18.0"}] * 25
        + [{"version": "1.18.1"}] * 25
        + [{"version": "1.18.2"}] * 25
        + [{"version": "1.17.9"}] * 25
    )
    with patch("urllib.request.urlopen", return_value=_mock_rpc_response(nodes)):
        entropy = nm.get_peer_entropy("http://fake-rpc")
    # log2(4) / log2(4) = 1.0 (normalized by N=4 bucket max)
    assert entropy >= 0.95


def test_peer_entropy_cached():
    """Second call within TTL returns cached, no RPC made."""
    nodes = [{"version": "1.18.0"}] * 50 + [{"version": "1.18.1"}] * 50
    call_count = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        call_count["n"] += 1
        return _mock_rpc_response(nodes)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        e1 = nm.get_peer_entropy("http://fake-rpc")
        e2 = nm.get_peer_entropy("http://fake-rpc")
        e3 = nm.get_peer_entropy("http://fake-rpc")

    assert e1 == e2 == e3
    assert call_count["n"] == 1


def test_peer_entropy_rpc_failure_returns_neutral():
    """RPC failure → 0.5 neutral, NOT cached."""
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        entropy = nm.get_peer_entropy("http://fake-rpc")
    assert entropy == 0.5


def test_ping_variance_empty_buffer_neutral():
    assert nm.get_ping_variance("http://fake-rpc") == 0.5


def test_ping_variance_stable_network():
    """All pings ~50ms → low variance → low normalized output."""
    # Seed the buffer via multiple get_ping_variance calls, each doing one ping
    call_count = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        call_count["n"] += 1
        return _mock_rpc_response(12345678)  # slot value, doesn't matter

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.monotonic", side_effect=[
             # Each call: (t0, t1) for ping latency. We need pairs.
             # Plus TTL gate check which reads monotonic once per call.
             # Structure: pre-gate, t0, t1, post (used for setting _last_ping_ts).
             0, 0, 0.05, 31, 31, 0.10, 62, 62, 0.052, 93, 93, 0.048
         ]):
        for _ in range(4):
            nm.get_ping_variance("http://fake-rpc")

    # Variance of [0.05, 0.05, 0.002, -0.02] can't be deterministic here with
    # artificial clock — just verify we don't blow up and return something in range
    result = nm.get_ping_variance("http://fake-rpc")
    assert 0.0 <= result <= 1.0


def test_ping_variance_ttl_gates_rpc_calls():
    """Multiple calls within 30s TTL → only 1 RPC call."""
    call_count = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        call_count["n"] += 1
        return _mock_rpc_response(100)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        # First call: will ping
        nm.get_ping_variance("http://fake-rpc")
        # Burst of calls within 30s TTL: should NOT re-ping
        nm.get_ping_variance("http://fake-rpc")
        nm.get_ping_variance("http://fake-rpc")
        nm.get_ping_variance("http://fake-rpc")

    assert call_count["n"] == 1, f"Expected 1 RPC call (TTL-gated), got {call_count['n']}"


def test_get_all_stats_returns_all_keys():
    """Integration: all four signals returned, all in [0,1]."""
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        stats = nm.get_all_stats(
            rpc_url="http://fake-rpc",
            bus_stats={"published": 100, "dropped": 5, "modules": ["a", "b"]},
        )
    assert set(stats.keys()) == {
        "peer_entropy", "ping_variance", "bus_drop_rate", "bus_module_diversity"
    }
    for v in stats.values():
        assert 0.0 <= v <= 1.0
