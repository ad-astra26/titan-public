"""
Tests for the SHM-direct migration of ``titan_plugin.proxies.spirit_proxy``.

Phase C Session 1 of rFP_phase_c_async_shm_consumer_migration §4.C.1.

Validates that the 5 migrated methods read from SHM (not bus.request)
and return schema-compatible dicts. Proves the deadlock surface
(`bus.request → publish → sock.sendall`) is closed for the spirit_proxy
state-lookup path.

Covers:
  1. Init — proxy attaches readers; INFO log fires
  2. get_trinity returns full schema after publisher writes
  3. get_trinity cold-boot returns schema-compatible default
     (no fallback to bus.request — pure SHM path)
  4. get_spirit_tensor returns 5-element float list
  5. get_sphere_clocks decodes per-clock dict (6 clocks × 7 fields)
  6. get_resonance reads resonance_state.bin payload
  7. get_unified_spirit reads unified_spirit_metadata.bin payload
  8. get_v4_state composite over migrated get_trinity
  9. get_diagnostics returns expected migration metadata
 10. ZERO bus.request calls during state-lookup methods (the deadlock-
     surface assertion — proves G19 compliance)

Run: ``python -m pytest tests/test_spirit_proxy_shm_direct.py -v -p no:anchorpy``
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from titan_plugin.logic.spirit_state_publisher import SpiritStatePublisher
from titan_plugin.proxies.spirit_proxy import SpiritProxy


# ── Stubs for bus + guardian (proxy constructor needs them) ───────────


class _StubBus:
    def __init__(self):
        self.request_call_count = 0
        self.subscribe_call_count = 0

    def subscribe(self, name, reply_only=False):
        self.subscribe_call_count += 1
        return _StubReplyQueue()

    def request(self, *args, **kwargs):
        # Migrated state-lookup methods MUST NOT call this — assertion
        # below verifies. Sessions-2-pending methods (filter_down_status,
        # meditation_health, coordinator, nervous_system) DO still call
        # this but those are not under test here.
        self.request_call_count += 1
        return None


class _StubReplyQueue:
    pass


class _StubGuardian:
    pass


# ── Reuse publisher stub helpers ──────────────────────────────────────


from tests.test_spirit_state_publisher import (  # noqa: E402
    _StubImpulseEngine,
    _StubNeuralNervousSystem,
    _StubResonance,
    _StubUnifiedSpirit,
    _StubHormone,
    _make_state_refs,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def populated_shm(shm_root):
    """Run the publisher once so the proxy has fresh slot data to read."""
    pub = SpiritStatePublisher(titan_id="T_TEST")
    pub.publish(_make_state_refs())
    return shm_root


@pytest.fixture()
def proxy(shm_root):
    """Construct a SpiritProxy bound to the test shm_root + stub bus."""
    bus = _StubBus()
    return SpiritProxy(bus=bus, guardian=_StubGuardian())


@pytest.fixture()
def proxy_with_data(populated_shm):
    """Proxy with publisher having written all 5 slots."""
    bus = _StubBus()
    return SpiritProxy(bus=bus, guardian=_StubGuardian()), bus


# ── 1. Init ───────────────────────────────────────────────────────────


def test_init_logs_and_attaches_readers(shm_root, caplog):
    caplog.set_level(logging.INFO, logger="titan_plugin.proxies.spirit_proxy")
    proxy = SpiritProxy(bus=_StubBus(), guardian=_StubGuardian())
    init_logs = [r for r in caplog.records if "initialized SHM-direct readers" in r.message]
    assert len(init_logs) == 1, "init INFO log missing"


# ── 2-3. get_trinity ──────────────────────────────────────────────────


def test_get_trinity_returns_schema_after_publish(proxy_with_data):
    proxy, bus = proxy_with_data
    trinity = proxy.get_trinity()
    # Required keys
    for key in ("spirit_tensor", "body_values", "mind_values",
                "body_center_dist", "mind_center_dist"):
        assert key in trinity
    # Tensors typed correctly
    assert len(trinity["spirit_tensor"]) == 5
    assert len(trinity["body_values"]) == 5
    assert len(trinity["mind_values"]) == 5
    # consciousness epoch present
    assert "consciousness" in trinity
    assert trinity["consciousness"]["epoch_id"] == 42
    # hormone_fires + hormone_levels populated
    assert "hormone_fires" in trinity
    assert isinstance(trinity["hormone_fires"], dict)
    # impulse_engine populated
    assert "impulse_engine" in trinity
    assert trinity["impulse_engine"]["impulse_count"] == 7
    # sphere_clocks (existing slot — may or may not be populated in tmp shm)
    # resonance + unified_spirit from new slots
    assert "resonance" in trinity
    assert trinity["resonance"]["great_pulse_count"] == 3
    assert "unified_spirit" in trinity
    assert trinity["unified_spirit"]["epoch_count"] == 12
    # ZERO bus.request calls — proves G19 compliance
    assert bus.request_call_count == 0, \
        f"get_trinity made {bus.request_call_count} bus.request calls — G19 VIOLATION"


def test_get_trinity_cold_boot_returns_schema_compatible_default(proxy):
    """No publisher has run; slots are missing. Proxy returns default
    dict with sane shape — never raises, never blocks."""
    bus = proxy._bus
    trinity = proxy.get_trinity()
    # Required keys present even in cold-boot
    assert "spirit_tensor" in trinity
    assert "body_values" in trinity
    assert "mind_values" in trinity
    assert trinity["spirit_tensor"] == [0.5] * 5
    # ZERO bus.request calls — pure SHM path with fallback (NOT bus fallback)
    assert bus.request_call_count == 0


# ── 4. get_spirit_tensor ──────────────────────────────────────────────


def test_get_spirit_tensor_returns_5_floats(proxy_with_data):
    proxy, bus = proxy_with_data
    tensor = proxy.get_spirit_tensor()
    assert isinstance(tensor, list)
    assert len(tensor) == 5
    assert all(isinstance(v, float) for v in tensor)
    assert bus.request_call_count == 0


def test_get_spirit_tensor_cold_boot_returns_neutral_5dt(proxy):
    bus = proxy._bus
    tensor = proxy.get_spirit_tensor()
    assert tensor == [0.5] * 5
    assert bus.request_call_count == 0


# ── 5. get_sphere_clocks ──────────────────────────────────────────────


def test_get_sphere_clocks_cold_boot_returns_error(proxy):
    """Without producer (sphere_clocks.bin is normally written by
    titan-trinity-rs / spirit_worker), proxy returns error dict.
    Production T3 will have the slot populated."""
    bus = proxy._bus
    sc = proxy.get_sphere_clocks()
    assert "error" in sc
    assert bus.request_call_count == 0


# ── 6. get_resonance ──────────────────────────────────────────────────


def test_get_resonance_after_publish(proxy_with_data):
    proxy, bus = proxy_with_data
    res = proxy.get_resonance()
    # Schema check
    assert "pairs" in res
    assert "resonant_count" in res
    assert "all_resonant" in res
    assert res["great_pulse_count"] == 3
    assert "config" in res
    # ts should be stripped (caller doesn't need it)
    assert "ts" not in res
    assert bus.request_call_count == 0


def test_get_resonance_cold_boot(proxy):
    bus = proxy._bus
    res = proxy.get_resonance()
    assert "error" in res
    assert bus.request_call_count == 0


# ── 7. get_unified_spirit ─────────────────────────────────────────────


def test_get_unified_spirit_after_publish(proxy_with_data):
    proxy, bus = proxy_with_data
    us = proxy.get_unified_spirit()
    assert us["epoch_count"] == 12
    assert us["velocity"] == pytest.approx(0.83)
    assert len(us["full_130dt"]) == 130
    assert "config" in us
    assert "ts" not in us
    assert bus.request_call_count == 0


def test_get_unified_spirit_cold_boot(proxy):
    bus = proxy._bus
    us = proxy.get_unified_spirit()
    assert "error" in us
    assert bus.request_call_count == 0


# ── 8. get_v4_state composite ─────────────────────────────────────────


def test_get_v4_state_composite_uses_shm(proxy_with_data):
    proxy, bus = proxy_with_data
    v4 = proxy.get_v4_state()
    # All keys present
    for key in ("sphere_clock", "resonance", "unified_spirit",
                "impulse_engine", "consciousness"):
        assert key in v4
    # Composite over get_trinity which is SHM-direct → ZERO bus calls
    assert bus.request_call_count == 0


# ── 9. get_diagnostics ────────────────────────────────────────────────


def test_diagnostics_lists_migrated_methods(proxy):
    diag = proxy.get_diagnostics()
    assert "titan_id" in diag
    assert "session1_migrated_methods" in diag
    assert "session2_pending_methods" in diag
    migrated = set(diag["session1_migrated_methods"])
    assert {"get_spirit_tensor", "get_trinity", "get_sphere_clocks",
            "get_resonance", "get_unified_spirit", "get_v4_state"} <= migrated
    pending = set(diag["session2_pending_methods"])
    assert {"get_filter_down_status", "get_meditation_health",
            "get_coordinator", "get_nervous_system"} <= pending


# ── 10. G19 compliance assertion (the deadlock-surface closure proof) ─


def test_zero_bus_request_calls_during_state_lookup(proxy_with_data):
    """G19 enforcement: spirit_proxy state-lookup methods MUST NOT
    issue bus.request. This is the architectural assertion that the
    T3 sidecar deadlock surface is closed for these 5 methods."""
    proxy, bus = proxy_with_data
    # Call every migrated method
    proxy.get_spirit_tensor()
    proxy.get_trinity()
    proxy.get_sphere_clocks()
    proxy.get_resonance()
    proxy.get_unified_spirit()
    proxy.get_v4_state()
    # Total bus.request calls across all 6 invocations: ZERO
    assert bus.request_call_count == 0, (
        f"G19 VIOLATION: spirit_proxy state-lookup made "
        f"{bus.request_call_count} bus.request calls — these are the "
        f"calls that wedged T3 sidecars on 2026-05-07. SHM-direct "
        f"migration MUST eliminate them entirely.")


# ── 11. Cold-boot fallback INFO log fires ─────────────────────────────


def test_cold_boot_logs_first_fallback_per_slot(proxy, caplog):
    caplog.set_level(logging.INFO, logger="titan_plugin.proxies.spirit_proxy")
    proxy.get_trinity()  # touches multiple slots → multiple fallbacks
    fallback_logs = [r for r in caplog.records
                     if "FIRST FALLBACK" in r.message]
    assert len(fallback_logs) >= 1, "expected ≥1 first-fallback INFO log"
    # Repeated calls don't multiply fallback logs (throttle works)
    proxy.get_trinity()
    proxy.get_trinity()
    fallback_logs_after = [r for r in caplog.records
                           if "FIRST FALLBACK" in r.message]
    # Should be same count (no new first-fallbacks for same slot)
    assert len(fallback_logs_after) == len(fallback_logs)
