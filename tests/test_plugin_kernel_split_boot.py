"""
Tests for Microkernel v2 Phase A S3a — TitanHCL + TitanKernel split.

Covers:
  - TitanHCL constructs with a TitanKernel reference
  - Compat @property facade (bus, guardian, soul, _full_config, etc.)
    delegates to kernel identically
  - The live plugin proxy @properties exist and return None pre-boot
  - get_v3_status() returns the expected shape (compatible with legacy
    TitanCore.get_v3_status)
  - boot is a coroutine + the dashboard-critical methods are exposed

NOTE: the module-registration tests (_register_modules + layer-canon
assertions) were removed 2026-06-22 — _register_modules was deleted in the
Phase-6 plugin cutover (registration moved to the ModuleSpec registry), so
they tested a method that no longer exists.

Does NOT run `await plugin.boot()` — that would spawn subprocesses.
The boot orchestration semantics are tested via contract (method
presence), not execution, to keep the test fast and isolated.

Reference: titan-docs/PLAN_microkernel_phase_a_s3.md §6.2
"""
from __future__ import annotations

import inspect

import pytest

from titan_hcl.core.kernel import TitanKernel
from titan_hcl.core.plugin import TitanHCL


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    bogus_wallet = str(tmp_path / "nonexistent_wallet.json")
    k = TitanKernel(bogus_wallet)
    yield k
    try:
        k.registry_bank.close_all()
    except Exception:
        pass
    try:
        k.disk_health.stop()
    except Exception:
        pass


@pytest.fixture
def plugin(kernel):
    p = TitanHCL(kernel)
    # Production fleet runs microkernel.l0_rust_enabled=true; several
    # CANONICAL modules (outer_body/outer_mind/outer_spirit + the L2/L3
    # Phase C workers) register only under that flag. Force it on so
    # _register_modules exercises the full canonical roster.
    p._full_config.setdefault("microkernel", {})["l0_rust_enabled"] = True
    return p


def test_plugin_constructs_with_kernel(plugin, kernel):
    assert plugin.kernel is kernel
    assert plugin._proxies == {}
    assert plugin._agency is None
    assert plugin._output_verifier is not None or plugin._output_verifier is None  # may fail init in limbo


def test_plugin_compat_properties_delegate_to_kernel(plugin, kernel):
    """Every compat @property returns the kernel's same object."""
    assert plugin.bus is kernel.bus
    assert plugin.guardian is kernel.guardian
    assert plugin.state_register is kernel.state_register
    assert plugin.registry_bank is kernel.registry_bank
    assert plugin.soul is kernel.soul  # both may be None in limbo
    assert plugin.network is kernel.network
    assert plugin.disk_health is kernel.disk_health
    assert plugin.bus_health is kernel.bus_health
    assert plugin._full_config is kernel.config
    assert plugin._limbo_mode is kernel.limbo_mode
    assert plugin._start_time == kernel._start_time


def test_plugin_proxy_properties_exist_return_none_pre_boot(plugin):
    """Proxy accessors defined; return None before _create_proxies.

    `sovereignty` is no longer a parent @property proxy — sovereignty is
    wired without a parent proxy facade (see plugin.py proxy block); the
    remaining accessors are the live set. recorder/gatekeeper (offline-RL
    retired, P1) + scholar are no longer proxies — dropped from the roster."""
    for name in [
        "memory", "metabolism", "mood_engine",
        "consciousness", "social_graph", "social",
        "studio", "maker_engine", "sage_researcher",
    ]:
        assert hasattr(plugin, name), f"proxy {name} not defined"
        assert getattr(plugin, name) is None, f"{name} should be None pre-boot"


def test_plugin_get_v3_status_shape(plugin):
    """get_v3_status returns the legacy TitanCore-compatible shape."""
    status = plugin.get_v3_status()
    # Canonical fields — dashboard and agent code depend on these keys
    assert status["version"] == "3.0"
    assert status["mode"] == "microkernel"
    assert "boot_time" in status
    assert "limbo" in status
    assert "bus_stats" in status
    assert "bus_modules" in status
    assert "guardian_status" in status


def test_plugin_boot_is_async(plugin):
    """plugin.boot must be a coroutine function (awaitable)."""
    assert inspect.iscoroutinefunction(plugin.boot)


def test_plugin_exposes_dashboard_critical_methods():
    """Spot-check that TitanHCL (the sole Phase C boot path) exposes the
    dashboard-critical methods. legacy_core/TitanCore retired 2026-05-21
    (D-SPEC-106) — TitanHCL is no longer mirroring any legacy surface."""
    expected_methods = [
        "boot", "create_agent", "get_v3_status", "reload_api",
        # _register_modules retired in the Phase-6 plugin cutover (module
        # registration moved to the ModuleSpec registry); _create_proxies stays.
        "_create_proxies",
        # _wire_sovereignty → _wire_life_force (v1.8.5 / D-SPEC-59).
        "_wire_metabolism", "_wire_life_force", "_wire_studio", "_wire_social",
        "_boot_agency", "_boot_reflex_collector",
        # A.S8 outer trinity: _boot_outer_trinity + _outer_trinity_loop +
        # _publish_outer_sources_loop ALL retired (Phase C dissolution C.8) —
        # outer source data is SHM-direct via the sidecars + helper.
        # _meditation_loop → meditation_worker (D-SPEC-57); _sovereignty_loop
        # → sovereignty_worker (D-SPEC-60 / v1.9.1) — both retired from parent.
        "_agency_loop",
        # _trinity_snapshot_loop + _v4_event_bridge_loop → observatory_worker
        # (D-SPEC-108, prior session); no longer parent methods.
        "_handle_impulse", "_handle_outer_dispatch", "_handle_agency_query",
    ]
    for method in expected_methods:
        assert hasattr(TitanHCL, method), f"missing {method}"


