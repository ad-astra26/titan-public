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
  - titan_hcl/core/plugin.py:_register_api_subprocess_module
  - titan_hcl/core/plugin.py:boot (Phase 5 flag check)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# NOTE (2026-06-02, session/20260602_apireload_drainfix): the two
# `test_register_api_subprocess_*` tests were REMOVED here. They exercised
# `TitanHCL._register_api_subprocess_module`, a pre-Phase-6 helper that
# registered the api as a Guardian-supervised L2 module. Phase 6 (D-SPEC-135)
# carved the api into a standalone kernel-rs-spawned L3 peer (SPEC §11.B.4 /
# INV-PROC-5) — that helper was deleted (see the `# Pre-Phase-6` markers at
# `titan_hcl/core/plugin.py:309,1872`), so the tests had been failing with
# AttributeError ever since. The api lifecycle is now owned by
# `kernel_supervisor.rs` (spawn/health-gate/drain/respawn); its socket-activation
# listen path is covered by `test_adopt_listen_fd_*` below + the kernel-rs
# `spawn.rs` env-contract tests + the T1 zero-drop E2E.


def test_kernel_rpc_skipped_when_flag_off():
    """TitanKernel._start_kernel_rpc() is a no-op when flag is off."""
    from titan_hcl.core.kernel import TitanKernel

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
    from titan_hcl.core.kernel import TitanKernel

    k = MagicMock(spec=TitanKernel)
    k._config = {"microkernel": {"api_process_separation_enabled": True}}
    k._plugin_ref = None  # not set yet
    k.titan_id = "TEST"

    # Should not raise; should log warning + return
    TitanKernel._start_kernel_rpc(k)


def test_exposed_methods_includes_critical_paths():
    """Smoke check that the EXPOSED_METHODS list covers the most-used
    plugin paths from the audit (§1.3)."""
    from titan_hcl.core.kernel import KERNEL_RPC_EXPOSED_METHODS

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


def test_adopt_listen_fd_wraps_inherited_socket():
    """SPEC §11.B.5 / rFP_kernel_zero_downtime_api_reload — kernel socket
    activation: _adopt_listen_fd wraps a kernel-handed, already-bound+listening
    fd WITHOUT rebinding. In production the kernel binds the port once and hands
    every api child a dup of that fd (dup2→fd3); OLD + NEW serve on dups of the
    SAME socket (one accept queue → zero-drop handover). Here a stand-in
    'kernel' socket + an os.dup model that handoff."""
    import os
    import socket
    from titan_hcl.api.api_subprocess import _adopt_listen_fd

    # Ephemeral high port; not the real api port (avoids clashing with a live api).
    port = 17790
    kernel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    kernel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    kernel_sock.bind(("0.0.0.0", port))
    kernel_sock.listen(8)
    try:
        # The api child inherits a dup of the kernel's listening fd.
        child_fd = os.dup(kernel_sock.fileno())
        adopted = _adopt_listen_fd("0.0.0.0", child_fd)
        try:
            # Same port, already listening — no rebind happened.
            assert adopted.getsockname()[1] == port
            assert adopted.family == socket.AF_INET
            # Non-blocking (asyncio create_server adopts it directly).
            assert adopted.getblocking() is False
        finally:
            adopted.close()  # owns + closes child_fd
    finally:
        kernel_sock.close()


def test_adopt_listen_fd_ipv6_family():
    """_adopt_listen_fd selects AF_INET6 for an IPv6 host literal."""
    import os
    import socket
    from titan_hcl.api.api_subprocess import _adopt_listen_fd

    kernel_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    kernel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    kernel_sock.bind(("::1", 17791))
    kernel_sock.listen(8)
    try:
        child_fd = os.dup(kernel_sock.fileno())
        s = _adopt_listen_fd("::1", child_fd)
        try:
            assert s.family == socket.AF_INET6
        finally:
            s.close()
    finally:
        kernel_sock.close()


def test_create_app_accepts_proxy_or_plugin():
    """create_app docstring states it accepts either real plugin or
    _RPCRemoteRef proxy. Behavior identical because app.state.titan_hcl
    is opaque to FastAPI — endpoint code does request.app.state.titan_hcl.X
    and the proxy intercepts."""
    from titan_hcl.api import create_app
    from titan_hcl.api.events import EventBus
    from titan_hcl.core.kernel_rpc import _RPCRemoteRef

    fake_client = MagicMock()
    proxy = _RPCRemoteRef(fake_client, ())

    event_bus = EventBus()
    app = create_app(proxy, event_bus, config={"cors_origins": []})
    assert app.state.titan_hcl is proxy
    assert app.state.event_bus is event_bus
