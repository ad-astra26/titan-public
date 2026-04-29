"""
Tests for Microkernel v2 Phase A §A.4 (S5) — flag-aware API path selection.

Validates that the api_process_separation_enabled flag correctly routes
plugin.boot() to either:
  - Legacy in-process uvicorn path (flag off, byte-identical pre-S5)
  - api_subprocess Guardian-spawned path (flag on, S5 architecture)

Tests use stubs to verify the BRANCHING decisions in plugin.boot()
without actually spawning subprocesses (full integration is covered by
the deploy verification on T1/T2/T3 — separate from this pytest suite).

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s5.md §5.1 + §5.2
  - titan_plugin/core/plugin.py:_register_api_subprocess_module
  - titan_plugin/core/plugin.py:boot (Phase 5 flag check)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_register_api_subprocess_skipped_when_flag_off():
    """When api_process_separation_enabled=False, no api module registered."""
    from titan_plugin.core.plugin import TitanPlugin

    plugin = MagicMock(spec=TitanPlugin)
    plugin._full_config = {"microkernel": {"api_process_separation_enabled": False}}
    plugin.guardian = MagicMock()

    # Call the helper directly via the unbound class method
    TitanPlugin._register_api_subprocess_module(plugin)

    # No guardian.register call expected
    plugin.guardian.register.assert_not_called()


def test_register_api_subprocess_runs_when_flag_on():
    """When api_process_separation_enabled=True, guardian.register called
    with ModuleSpec(name='api', layer='L3', autostart=True, ...)."""
    from titan_plugin.core.plugin import TitanPlugin
    from titan_plugin.guardian import ModuleSpec

    plugin = MagicMock(spec=TitanPlugin)
    plugin._full_config = {
        "microkernel": {"api_process_separation_enabled": True},
        "api": {"host": "0.0.0.0", "port": 7777},
    }
    plugin.guardian = MagicMock()

    TitanPlugin._register_api_subprocess_module(plugin)

    # guardian.register should have been called exactly once
    plugin.guardian.register.assert_called_once()
    call_args = plugin.guardian.register.call_args[0]
    spec = call_args[0]
    assert isinstance(spec, ModuleSpec)
    assert spec.name == "api"
    assert spec.layer == "L3"
    assert spec.autostart is True
    assert spec.lazy is False
    assert spec.rss_limit_mb == 300
    assert spec.heartbeat_timeout == 60.0
    # entry_fn must be the api_subprocess_main function
    from titan_plugin.api.api_subprocess import api_subprocess_main
    assert spec.entry_fn is api_subprocess_main
    # Sub-config carries [api] + [microkernel] sections
    assert "api" in spec.config
    assert "microkernel" in spec.config


def test_kernel_rpc_skipped_when_flag_off():
    """TitanKernel._start_kernel_rpc() is a no-op when flag is off."""
    from titan_plugin.core.kernel import TitanKernel

    k = MagicMock(spec=TitanKernel)
    k._config = {"microkernel": {"api_process_separation_enabled": False}}
    k._plugin_ref = "fake_plugin"

    # Should not raise; should not start anything
    TitanKernel._start_kernel_rpc(k)
    # _rpc_server should not have been assigned
    assert not hasattr(k, "_rpc_server") or k._rpc_server is None or \
        isinstance(k._rpc_server, MagicMock)  # MagicMock auto-creates attrs


def test_kernel_rpc_skipped_when_no_plugin_ref():
    """When flag is on but _plugin_ref isn't set yet, log warning + bail.
    Prevents crashing on misconfigured boot order."""
    from titan_plugin.core.kernel import TitanKernel

    k = MagicMock(spec=TitanKernel)
    k._config = {"microkernel": {"api_process_separation_enabled": True}}
    k._plugin_ref = None  # not set yet
    k.titan_id = "TEST"

    # Should not raise; should log warning + return
    TitanKernel._start_kernel_rpc(k)


def test_exposed_methods_includes_critical_paths():
    """Smoke check that the EXPOSED_METHODS list covers the most-used
    plugin paths from the audit (§1.3)."""
    from titan_plugin.core.kernel import KERNEL_RPC_EXPOSED_METHODS

    must_have = {
        "guardian.get_status",
        "guardian.start",
        "guardian.layer_stats",
        "soul",
        "soul.current_gen",
        "bus.publish",
        "bus.request",
        "_full_config",
        "_is_meditating",
        "_start_time",
        "memory",
        "metabolism",
        "reload_api",
        "get_v3_status",
    }
    missing = must_have - KERNEL_RPC_EXPOSED_METHODS
    assert not missing, f"EXPOSED_METHODS missing critical paths: {missing}"


def test_create_app_accepts_proxy_or_plugin():
    """create_app docstring states it accepts either real plugin or
    _RPCRemoteRef proxy. Behavior identical because app.state.titan_plugin
    is opaque to FastAPI — endpoint code does request.app.state.titan_plugin.X
    and the proxy intercepts."""
    from titan_plugin.api import create_app
    from titan_plugin.api.events import EventBus
    from titan_plugin.core.kernel_rpc import _RPCRemoteRef

    fake_client = MagicMock()
    proxy = _RPCRemoteRef(fake_client, ())

    event_bus = EventBus()
    app = create_app(proxy, event_bus, config={"cors_origins": []})
    assert app.state.titan_plugin is proxy
    assert app.state.event_bus is event_bus
