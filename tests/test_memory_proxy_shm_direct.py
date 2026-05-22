"""
Tests for the SHM-direct migration of ``titan_hcl.proxies.memory_proxy``.

Phase C Session 2 of rFP_phase_c_async_shm_consumer_migration §4.C.13.

Validates that the 3 migrated state-lookup methods read from SHM (not
bus.request) and return schema-compatible dicts. Includes the G19
zero-bus-request assertion that proves the deadlock surface
(memory_proxy.get_growth_metrics — the post-Session-1 sidecar blocker)
is closed.

Run: ``python -m pytest tests/test_memory_proxy_shm_direct.py -v -p no:anchorpy``
"""
from __future__ import annotations

import logging

import pytest

from titan_hcl.logic.memory_state_publisher import MemoryStatePublisher
from titan_hcl.proxies.memory_proxy import MemoryProxy

from tests.test_memory_state_publisher import _StubMemory  # noqa: E402


# ── Stubs ─────────────────────────────────────────────────────────────


class _StubBus:
    def __init__(self):
        self.request_call_count = 0

    def subscribe(self, name, reply_only=False):
        return _StubReplyQueue()

    def request(self, *args, **kwargs):
        self.request_call_count += 1
        return None


class _StubReplyQueue:
    pass


class _StubGuardian:
    pass


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def populated_shm(shm_root):
    """Run publisher once so proxy reads find fresh data."""
    pub = MemoryStatePublisher(titan_id="T_TEST")
    pub.publish(_StubMemory(persistent_count=42))
    return shm_root


@pytest.fixture()
def proxy_with_data(populated_shm):
    bus = _StubBus()
    return MemoryProxy(bus=bus, guardian=_StubGuardian()), bus


@pytest.fixture()
def proxy_cold(shm_root):
    """Proxy with no producer published yet (cold-boot)."""
    bus = _StubBus()
    return MemoryProxy(bus=bus, guardian=_StubGuardian()), bus


# ── 1. Init logs ──────────────────────────────────────────────────────


def test_init_logs(shm_root, caplog):
    caplog.set_level(logging.INFO, logger="titan_hcl.proxies.memory_proxy")
    proxy = MemoryProxy(bus=_StubBus(), guardian=_StubGuardian())
    assert any("initialized" in r.message and "SHM reader" in r.message
               for r in caplog.records)


# ── 2. get_persistent_count ───────────────────────────────────────────


def test_get_persistent_count_after_publish(proxy_with_data):
    proxy, bus = proxy_with_data
    n = proxy.get_persistent_count()
    assert n == 42
    assert bus.request_call_count == 0  # G19


def test_get_persistent_count_cold_boot_returns_zero(proxy_cold):
    proxy, bus = proxy_cold
    n = proxy.get_persistent_count()
    assert n == 0
    assert bus.request_call_count == 0  # G19


# ── 3. get_memory_status ──────────────────────────────────────────────


def test_get_memory_status_after_publish(proxy_with_data):
    proxy, bus = proxy_with_data
    status = proxy.get_memory_status()
    assert status["persistent_count"] == 42
    assert status["cognee_ready"] is True
    assert status["mempool_size"] == 3
    assert "backend_ready" in status  # API-compat field
    assert bus.request_call_count == 0  # G19


def test_get_memory_status_cold_boot(proxy_cold):
    proxy, bus = proxy_cold
    status = proxy.get_memory_status()
    assert status["persistent_count"] == 0
    assert status["cognee_ready"] is False
    assert status["mempool_size"] == 0
    assert bus.request_call_count == 0  # G19


# ── 4. get_growth_metrics (THE deadlock-causing method) ───────────────


def test_get_growth_metrics_default_saturation_after_publish(proxy_with_data):
    """Default node_saturation_24h=30 → fast path, returns
    publisher-precomputed values."""
    proxy, bus = proxy_with_data
    metrics = proxy.get_growth_metrics()
    assert "learning_velocity" in metrics
    assert "directive_alignment" in metrics
    assert metrics["directive_alignment"] == pytest.approx(0.6)
    # Raw counts also exposed
    assert metrics["total_persistent"] == 8
    assert metrics["high_quality_count"] == 4
    # G19 — THE assertion: the deadlock-causing call now NEVER touches bus
    assert bus.request_call_count == 0


def test_get_growth_metrics_non_default_saturation_recomputes(proxy_with_data):
    """Non-default saturation → slow path; learning_velocity recomputed
    locally from raw effective_nodes_24h. directive_alignment unchanged
    (saturation-independent)."""
    proxy, bus = proxy_with_data
    metrics_default = proxy.get_growth_metrics(node_saturation_24h=30)
    metrics_lower = proxy.get_growth_metrics(node_saturation_24h=5)
    metrics_higher = proxy.get_growth_metrics(node_saturation_24h=100)
    # Lower saturation → easier to reach 1.0; higher → harder
    assert metrics_lower["learning_velocity"] >= metrics_default["learning_velocity"]
    assert metrics_higher["learning_velocity"] <= metrics_default["learning_velocity"]
    # directive_alignment same across all
    assert (metrics_lower["directive_alignment"] ==
            metrics_default["directive_alignment"] ==
            metrics_higher["directive_alignment"])
    assert bus.request_call_count == 0


def test_get_growth_metrics_cold_boot_returns_neutral_defaults(proxy_cold):
    proxy, bus = proxy_cold
    metrics = proxy.get_growth_metrics()
    assert metrics["learning_velocity"] == 0.5
    assert metrics["directive_alignment"] == 0.5
    assert bus.request_call_count == 0


# ── 5. G19 zero-bus-request assertion (the deadlock-surface proof) ────


def test_zero_bus_request_calls_during_state_lookup(proxy_with_data):
    """G19 enforcement: memory_proxy state-lookup methods MUST NOT
    issue bus.request. This is the architectural assertion that the
    second-layer T3 sidecar deadlock surface (memory_proxy.get_growth_metrics
    — proven blocking by py-spy 2026-05-07 post-Session-1) is closed."""
    proxy, bus = proxy_with_data
    proxy.get_persistent_count()
    proxy.get_memory_status()
    proxy.get_growth_metrics()
    proxy.get_growth_metrics(node_saturation_24h=10)
    proxy.get_growth_metrics(node_saturation_24h=50)
    assert bus.request_call_count == 0, (
        f"G19 VIOLATION: memory_proxy state-lookup made "
        f"{bus.request_call_count} bus.request calls. The post-Session-1 "
        f"py-spy proof showed get_growth_metrics blocking sidecars on "
        f"sock.sendall — Session 2 SHM migration MUST eliminate them.")


# ── 6. Diagnostics ────────────────────────────────────────────────────


def test_diagnostics_lists_migration_status(proxy_cold):
    proxy, _ = proxy_cold
    diag = proxy.get_diagnostics()
    assert "session2_migrated_methods" in diag
    assert "session3_pending_methods" in diag
    migrated = set(diag["session2_migrated_methods"])
    assert {"get_persistent_count", "get_memory_status",
            "get_growth_metrics"} == migrated
    pending = set(diag["session3_pending_methods"])
    # Pending list contains the parameterized queries + work RPCs
    for m in ("query", "fetch_mempool", "get_top_memories",
              "get_topology", "get_knowledge_graph", "run_meditation"):
        assert m in pending


# ── 7. Cold-boot fallback INFO log fires (throttled to first per cause) ─


def test_cold_boot_logs_first_fallback_then_throttles(proxy_cold, caplog):
    caplog.set_level(logging.INFO, logger="titan_hcl.proxies.memory_proxy")
    proxy, _ = proxy_cold
    proxy.get_memory_status()  # triggers fallback
    # Repeat — should NOT add new first-fallback logs (throttle)
    proxy.get_memory_status()
    proxy.get_memory_status()
    fallback_logs = [r for r in caplog.records
                     if "FIRST FALLBACK" in r.message]
    assert len(fallback_logs) == 1, \
        f"expected exactly 1 FIRST FALLBACK log (throttled), got {len(fallback_logs)}"
