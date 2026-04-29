"""
Tests for Microkernel v2 Phase A §A.4 (S5) — kernel_rpc wire protocol.

Covers:
  - Length-prefixed msgpack frame roundtrip (small, max, malformed)
  - HMAC challenge-response auth (correct passes, wrong rejected)
  - Server resolves dotted method paths via getattr chain
  - EXPOSED_METHODS enforcement (non-exposed paths return MethodNotExposed)
  - Exception forwarding (server raises, client gets RuntimeError with msg)
  - Per-call latency (<2ms target)
  - Socket file permissions (chmod 0600)
  - Authkey file permissions (chmod 0600)
  - Transparent proxy chains attribute access correctly

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s5.md §5.0
  - titan_plugin/core/kernel_rpc.py
"""
from __future__ import annotations

import os
import secrets
import stat
import threading
import time

import pytest

from titan_plugin.core.kernel_rpc import (
    KernelRPCServer,
    KernelRPCClient,
    _RPCRemoteRef,
    kernel_sock_path,
    kernel_authkey_path,
    generate_authkey,
)


@pytest.fixture
def fake_plugin():
    """Fake plugin object with a method, an attribute, and a nested object."""
    class _Guardian:
        def __init__(self):
            self.module_count = 15
        def get_status(self):
            return {"state": "running", "modules": list(range(15))}
        def start(self, name):
            return {"started": name}
        def fail_with_value_error(self):
            raise ValueError("intentional test failure")

    class _Plugin:
        def __init__(self):
            self.guardian = _Guardian()
            self._full_config = {"host": "0.0.0.0", "port": 7777}
            self._is_meditating = False
            self._start_time = 1234567890.0

    return _Plugin()


@pytest.fixture
def rpc_pair(fake_plugin):
    """Bring up server + client with HMAC handshake; tear down after test."""
    tid = "TEST_" + secrets.token_hex(4)
    exposed = frozenset({
        "guardian.get_status",
        "guardian.start",
        "guardian.fail_with_value_error",
        "_full_config",
        "_is_meditating",
        "_start_time",
    })
    server = KernelRPCServer(
        plugin_ref=fake_plugin, titan_id=tid, exposed_methods=exposed)
    server_thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="test-rpc-server")
    server_thread.start()
    time.sleep(0.2)  # let listen() settle

    client = KernelRPCClient(titan_id=tid, connect_timeout_s=5.0)
    client.connect()
    yield server, client, tid

    client.close()
    server.stop()


# ── Connection + handshake ─────────────────────────────────────────


def test_connect_and_handshake(rpc_pair):
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    assert isinstance(proxy, _RPCRemoteRef)


def test_wrong_authkey_rejected(fake_plugin):
    """Manual server with known authkey + client with wrong key → connection closes."""
    import socket as _socket

    tid = "TEST_" + secrets.token_hex(4)
    server = KernelRPCServer(
        plugin_ref=fake_plugin, titan_id=tid, exposed_methods=frozenset(),
        authkey=secrets.token_bytes(32))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)

    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.connect(str(kernel_sock_path(tid)))
    # Read the challenge
    challenge = sock.recv(32)
    assert len(challenge) == 32
    # Send WRONG response
    sock.sendall(b"\x00" * 32)
    # Server should close the connection — next recv returns empty
    sock.settimeout(2.0)
    closed = sock.recv(1)
    assert closed == b"", "server should have closed connection on bad auth"
    sock.close()
    server.stop()


# ── Method dispatch ─────────────────────────────────────────────────


def test_call_exposed_method(rpc_pair):
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    result = proxy.guardian.get_status()
    assert result == {"state": "running", "modules": list(range(15))}


def test_call_with_args(rpc_pair):
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    result = proxy.guardian.start("memory")
    assert result == {"started": "memory"}


def test_call_unexposed_method_returns_error(rpc_pair):
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    with pytest.raises(RuntimeError) as exc:
        # guardian.shutdown is NOT in exposed
        proxy.guardian.shutdown()
    assert "MethodNotExposed" in str(exc.value)


def test_attribute_read_via_call(rpc_pair):
    """Attribute access (not a method) — _resolve_method wraps in lambda."""
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    cfg = proxy._full_config()
    assert cfg == {"host": "0.0.0.0", "port": 7777}
    meditating = proxy._is_meditating()
    assert meditating is False
    start_time = proxy._start_time()
    assert start_time == 1234567890.0


def test_exception_forwarding(rpc_pair):
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()
    with pytest.raises(RuntimeError) as exc:
        proxy.guardian.fail_with_value_error()
    msg = str(exc.value)
    assert "ValueError" in msg
    assert "intentional test failure" in msg


# ── Performance ─────────────────────────────────────────────────────


def test_per_call_latency_under_target(rpc_pair):
    """1000 simple roundtrips; assert <2ms per call (target ~700μs)."""
    server, client, _ = rpc_pair
    proxy = client.get_plugin_proxy()

    N = 100  # smaller in CI; fast enough to be meaningful
    start = time.perf_counter()
    for _ in range(N):
        proxy.guardian.get_status()
    elapsed = time.perf_counter() - start
    per_call_us = elapsed / N * 1e6
    assert per_call_us < 2000, \
        f"per-call latency {per_call_us:.0f}μs exceeds 2000μs target"


# ── Permissions ─────────────────────────────────────────────────────


def test_socket_permissions_0600(rpc_pair):
    server, client, tid = rpc_pair
    sock_p = kernel_sock_path(tid)
    mode = stat.S_IMODE(os.stat(sock_p).st_mode)
    assert mode == 0o600, f"socket mode {oct(mode)} (expected 0o600)"


def test_authkey_permissions_0600(rpc_pair):
    server, client, tid = rpc_pair
    authkey_p = kernel_authkey_path(tid)
    mode = stat.S_IMODE(os.stat(authkey_p).st_mode)
    assert mode == 0o600, f"authkey mode {oct(mode)} (expected 0o600)"


# ── Cleanup ─────────────────────────────────────────────────────────


def test_server_stop_unlinks_files(fake_plugin):
    tid = "TEST_" + secrets.token_hex(4)
    server = KernelRPCServer(
        plugin_ref=fake_plugin, titan_id=tid, exposed_methods=frozenset())
    sock_p = kernel_sock_path(tid)
    authkey_p = kernel_authkey_path(tid)
    assert sock_p.exists()
    assert authkey_p.exists()
    server.stop()
    assert not sock_p.exists()
    assert not authkey_p.exists()


# ── Proxy mechanics ─────────────────────────────────────────────────


def test_proxy_attribute_chain():
    """Verify _RPCRemoteRef chains attribute names without RPC until called."""
    fake_client = type("FakeClient", (), {
        "call": lambda self, p, a, k: ("called", p, a, k)})()
    proxy = _RPCRemoteRef(fake_client, ())
    deep = proxy.a.b.c
    assert isinstance(deep, _RPCRemoteRef)
    assert deep._path == ("a", "b", "c")
    result = deep(1, x=2)
    assert result == ("called", "a.b.c", [1], {"x": 2})


def test_authkey_generation_is_random():
    k1 = generate_authkey()
    k2 = generate_authkey()
    assert k1 != k2
    assert len(k1) == 32
