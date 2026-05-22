"""SPEC §8.1 + §11.B.3.1 (D-SPEC-93, v1.32.0) — MODULE_SHUTDOWN pid-targeting.

The bus broker routes by `dst=module_name`. During reload's dual-pid
name-aliased subscription window (NEW + OLD pids both subscribed under
the same name between Step 3 spawn-NEW and Step 7 atomic-swap), a
MODULE_SHUTDOWN frame published to `dst=module_name` would fan-out to
BOTH pids absent pid-targeting. The worker-side bus recv path
(`BusSocketClient._handle_inbound`) drops MODULE_SHUTDOWN frames whose
`payload.target_pid` is set and does not match `os.getpid()`.

These tests exercise the filter at the delivery boundary in isolation
— no real socket, no broker, no Guardian. Just instantiate BusSocketClient
(without `.start()` so no connection), call `_handle_inbound(msg)` with
crafted frames, observe `_inbound` deque state.
"""
from __future__ import annotations

import os

import pytest

from titan_hcl import bus
from titan_hcl.core.bus_socket import BusSocketClient


@pytest.fixture
def client() -> BusSocketClient:
    """Minimal client — no socket, no thread. We only need the inbound
    deque + _handle_inbound method."""
    return BusSocketClient(
        titan_id="testT",
        authkey=b"k" * 32,
        name="knowledge",
    )


def _inbound_types(client: BusSocketClient) -> list[str]:
    with client._inbound_lock:
        return [m.get("type") for m in client._inbound]


def test_module_shutdown_without_target_pid_delivers_to_all(client):
    """target_pid absent or null → all subscribers honor (legacy behavior)."""
    msg = {
        "type": bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {"reason": "fleet_wide_save"},
    }
    client._handle_inbound(msg)
    assert _inbound_types(client) == [bus.MODULE_SHUTDOWN]


def test_module_shutdown_target_pid_null_explicit_delivers(client):
    """Explicit target_pid=None → same as absent."""
    msg = {
        "type": bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {"reason": "graceful", "target_pid": None},
    }
    client._handle_inbound(msg)
    assert _inbound_types(client) == [bus.MODULE_SHUTDOWN]


def test_module_shutdown_target_pid_matches_delivers(client):
    """target_pid == os.getpid() → frame reaches inbound."""
    msg = {
        "type": bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {
            "reason": "reload",
            "target_pid": os.getpid(),
            "swap_id": "test-swap-id",
        },
    }
    client._handle_inbound(msg)
    assert _inbound_types(client) == [bus.MODULE_SHUTDOWN]


def test_module_shutdown_target_pid_mismatch_dropped(client):
    """target_pid != os.getpid() → frame dropped at delivery boundary."""
    msg = {
        "type": bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {
            "reason": "reload",
            "target_pid": os.getpid() + 99999,  # an unrelated pid
            "swap_id": "test-swap-id",
        },
    }
    client._handle_inbound(msg)
    assert _inbound_types(client) == [], (
        "MODULE_SHUTDOWN with mismatching target_pid must NOT reach inbound deque")


def test_filter_only_applies_to_module_shutdown(client):
    """Other message types with payload.target_pid are NOT filtered —
    the contract is specifically for MODULE_SHUTDOWN per SPEC §11.B.3.1.
    Future expansion to other types would require an explicit SPEC update."""
    msg = {
        "type": "SAVE_NOW",
        "src": "guardian",
        "dst": "knowledge",
        "payload": {
            "target_pid": os.getpid() + 99999,  # would be filtered if MODULE_SHUTDOWN
        },
    }
    client._handle_inbound(msg)
    assert _inbound_types(client) == ["SAVE_NOW"]


def test_module_shutdown_in_batch_filtered(client):
    """BUS_BATCH unwraps and recurses through _handle_inbound — pid filter
    applies to MODULE_SHUTDOWN frames inside a batch envelope."""
    msg = {
        "type": "BUS_BATCH",
        "msgs": [
            {
                "type": bus.MODULE_SHUTDOWN,
                "src": "guardian",
                "dst": "knowledge",
                "payload": {
                    "reason": "reload",
                    "target_pid": os.getpid() + 99999,  # mismatch → drop
                },
            },
            {
                "type": bus.MODULE_SHUTDOWN,
                "src": "guardian",
                "dst": "knowledge",
                "payload": {
                    "reason": "graceful",  # no target_pid → deliver
                },
            },
            {
                "type": "SAVE_NOW",
                "src": "guardian",
                "dst": "knowledge",
                "payload": {},
            },
        ],
    }
    client._handle_inbound(msg)
    # Only the no-target_pid MODULE_SHUTDOWN + SAVE_NOW survive.
    assert _inbound_types(client) == [bus.MODULE_SHUTDOWN, "SAVE_NOW"]


def test_dual_pid_name_aliased_subscription_only_target_receives():
    """The reload step-6 scenario: two BusSocketClients share the bus
    name 'knowledge' (NEW + OLD pids during reload). Guardian publishes
    MODULE_SHUTDOWN(target_pid=A); only client-A's inbound receives.

    Two clients instantiated in the same Python process share os.getpid()
    so we simulate the pid-aliasing by treating one client's _handle_inbound
    as the "old pid" path: set target_pid=os.getpid() for it (matches);
    and the other's target_pid as something the filter rejects. In
    production these are two different OS pids — the contract is the
    same: per-client filter."""
    new_client = BusSocketClient(
        titan_id="testT", authkey=b"k" * 32, name="knowledge")
    old_client = BusSocketClient(
        titan_id="testT", authkey=b"k" * 32, name="knowledge")

    # Frame intended for OLD pid only — under production this would be
    # `target_pid=old_pid`; under this in-proc test we use os.getpid()
    # for the client that should accept and (os.getpid()+1) for the one
    # that should drop.
    fake_old_pid = os.getpid()
    fake_new_pid = os.getpid() + 1

    msg_to_old = {
        "type": bus.MODULE_SHUTDOWN,
        "src": "guardian",
        "dst": "knowledge",
        "payload": {
            "reason": "reload",
            "target_pid": fake_old_pid,
            "swap_id": "test-swap-id",
        },
    }

    # Both clients receive the broadcast (bus broker fanout simulation).
    # OLD pid client's filter sees target_pid==os.getpid() → deliver.
    old_client._handle_inbound(msg_to_old)
    assert _inbound_types(old_client) == [bus.MODULE_SHUTDOWN]

    # Now simulate the NEW pid client by patching os.getpid for THIS
    # client's filter pass. The frame payload says target_pid=fake_old_pid
    # but the "new" worker's os.getpid() is fake_new_pid → filter drops.
    import unittest.mock as _mock
    with _mock.patch("titan_hcl.core.bus_socket.os.getpid",
                     return_value=fake_new_pid):
        new_client._handle_inbound(msg_to_old)
    assert _inbound_types(new_client) == [], (
        "NEW pid client (os.getpid() != target_pid) must drop the frame")
