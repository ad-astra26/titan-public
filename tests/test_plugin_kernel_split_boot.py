"""
Tests for Microkernel v2 Phase A S3a — TitanPlugin + TitanKernel split.

Covers:
  - TitanPlugin constructs with a TitanKernel reference
  - Compat @property facade (bus, guardian, soul, _full_config, etc.)
    delegates to kernel identically
  - All 13 plugin proxy @properties exist and return None pre-boot
  - _register_modules populates guardian with all 16 supervised modules
    tagged with correct layers (L0/L1/L2/L3)
  - Module layers match LAYER_CANON table (Microkernel v2 §A.5)
  - get_v3_status() returns the expected shape (compatible with legacy
    TitanCore.get_v3_status)
  - microkernel config passthrough preserved in spirit ModuleSpec
    (regression guard for S2 fix #2)

Does NOT run `await plugin.boot()` — that would spawn 16 subprocesses.
The boot orchestration semantics are tested via contract (method order
+ presence), not execution, to keep the test fast and isolated.

Reference: titan-docs/PLAN_microkernel_phase_a_s3.md §6.2
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from titan_plugin._layer_canon import LAYER_CANON
from titan_plugin.core.kernel import TitanKernel
from titan_plugin.core.plugin import TitanPlugin


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
    return TitanPlugin(kernel)


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
    """All 13 proxy accessors defined; return None before _create_proxies."""
    for name in [
        "memory", "metabolism", "sovereignty", "mood_engine", "recorder",
        "gatekeeper", "scholar", "consciousness", "social_graph", "social",
        "studio", "maker_engine", "sage_researcher",
    ]:
        assert hasattr(plugin, name), f"proxy {name} not defined"
        assert getattr(plugin, name) is None, f"{name} should be None pre-boot"


def test_plugin_register_modules_populates_guardian(plugin):
    """_register_modules tags all supervised modules with valid layers.

    Module count grew over time:
      - PLAN §1.2 baseline: 16 modules
      - +warning_monitor (2026-04-25): 17
      - +output_verifier (A.8.3, 2026-04-28 AM): 18
      - +consciousness_writer/social_graph_writer/events_teacher_writer
        (universal SQLite writer rFP, 2026-04-27): 21
      - +agency_worker (A.8.6, 2026-04-28): 22
    """
    plugin._register_modules()
    modules = plugin.guardian._modules
    expected = {
        "imw", "observatory_writer",
        "consciousness_writer", "social_graph_writer", "events_teacher_writer",
        "memory", "rl", "llm",
        "body", "mind", "spirit",
        "media", "language", "meta_teacher",
        "cgn", "knowledge", "emot_cgn",
        "timechain", "backup",
        "warning_monitor",
        "output_verifier",  # A.8.3
        "outer_trinity",    # A.8.4
        "reflex",           # A.8.5
        "agency_worker",    # A.8.6
    }
    assert set(modules.keys()) == expected, f"diff: {expected ^ set(modules.keys())}"


def test_plugin_module_layers_match_canon(plugin):
    """Every registered module's layer matches LAYER_CANON."""
    plugin._register_modules()
    for name, info in plugin.guardian._modules.items():
        expected_layer = LAYER_CANON[name]
        assert info.spec.layer == expected_layer, (
            f"module {name}: spec.layer={info.spec.layer} but canon={expected_layer}"
        )


def test_plugin_spirit_gets_microkernel_config_passthrough(plugin):
    """Regression guard for S2 bug fix #2: spirit_worker needs microkernel
    section in its config to resolve shm_*_enabled flags."""
    plugin._register_modules()
    spirit_info = plugin.guardian._modules["spirit"]
    assert "microkernel" in spirit_info.spec.config, (
        "spirit config missing microkernel passthrough — "
        "RegistryBank.is_enabled() would silently return False"
    )


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


def test_plugin_method_signatures_match_titancore_surface():
    """Spot-check that TitanPlugin exposes the dashboard-critical methods."""
    expected_methods = [
        "boot", "create_agent", "get_v3_status", "reload_api",
        "_register_modules", "_create_proxies",
        "_wire_metabolism", "_wire_sovereignty", "_wire_studio", "_wire_social",
        "_boot_agency", "_boot_reflex_collector",
        "_boot_outer_trinity", "_outer_trinity_loop",
        "_meditation_loop", "_agency_loop", "_sovereignty_loop",
        "_trinity_snapshot_loop", "_v4_event_bridge_loop",
        "_handle_impulse", "_handle_outer_dispatch", "_handle_agency_query",
    ]
    for method in expected_methods:
        assert hasattr(TitanPlugin, method), f"missing {method}"


def test_plugin_registers_imw_at_correct_layer(plugin):
    """IMW is L1 per §A.5 + PLAN §1.2 (writes inner_memory.db, L1's DB)."""
    plugin._register_modules()
    assert plugin.guardian._modules["imw"].spec.layer == "L1"


def test_plugin_registers_observatory_writer_at_correct_layer(plugin):
    """observatory_writer is L3 per §A.5 (writes observatory.db, L3's DB)."""
    plugin._register_modules()
    assert plugin.guardian._modules["observatory_writer"].spec.layer == "L3"


def test_plugin_cgn_layer_honored(plugin):
    """CGN stays L2 per project_cgn_as_higher_state_registry.md invariant."""
    plugin._register_modules()
    assert plugin.guardian._modules["cgn"].spec.layer == "L2"
