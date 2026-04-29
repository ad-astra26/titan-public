"""
Tests for DivineBus dual-mode integration (Phase B.2 C6).

Covers:
- Flag default false: byte-identical legacy behavior (no broker attached)
- attach_broker() makes DivineBus.publish() fan out to broker subscribers
  ALONG WITH in-process subscribers — both code paths active simultaneously
- detach_broker() cleanly reverts behavior
- has_socket_broker property
- Broker exception during fan-out doesn't break in-process delivery
- All BUS_* constants exist in bus.py and are registered in MSG_SPECS
- bus_specs.audit_against_bus_constants returns clean (drift detection)
- titan_params.toml has bus_ipc_socket_enabled = false default
"""
from __future__ import annotations

import queue
import threading
import time

import pytest

from titan_plugin import bus as bus_module
from titan_plugin.bus import DivineBus
from titan_plugin.bus_specs import (
    audit_against_bus_constants,
    get_spec,
)


# ── Constants are defined ──────────────────────────────────────────────────


def test_b2_bus_constants_exist():
    """All B.2 message constants must be defined in bus.py."""
    for name in ("BUS_SUBSCRIBE", "BUS_UNSUBSCRIBE", "BUS_PING", "BUS_PONG",
                 "BUS_SLOW_CONSUMER", "BUS_HANDOFF"):
        assert hasattr(bus_module, name), f"bus.{name} missing"
        assert getattr(bus_module, name) == name, f"bus.{name} value drift"


def test_msg_specs_includes_b2_constants_as_p0():
    """All BUS_* control-plane messages are P0 (never droppable)."""
    for name in ("BUS_SUBSCRIBE", "BUS_UNSUBSCRIBE", "BUS_PING", "BUS_PONG",
                 "BUS_SLOW_CONSUMER", "BUS_HANDOFF"):
        spec = get_spec(name)
        assert spec.priority == 0, f"{name} should be P0, got {spec.priority}"


def test_audit_against_bus_constants_clean_with_b2():
    """After C6, the spec table includes BUS_* and bus.py defines them.
    Drift detection must return zero issues."""
    issues = audit_against_bus_constants()
    assert issues == [], f"drift detected: {issues}"


def test_bus_ipc_socket_enabled_flag_defaults_false():
    """Verify titan_params.toml registers the flag with default false."""
    from titan_plugin.config_loader import load_titan_config
    cfg = load_titan_config()
    micro = cfg.get("microkernel", {})
    assert "bus_ipc_socket_enabled" in micro, \
        "microkernel.bus_ipc_socket_enabled flag not registered"
    assert micro["bus_ipc_socket_enabled"] is False, \
        f"default should be false, got {micro['bus_ipc_socket_enabled']}"


# ── DivineBus dual-mode behavior ───────────────────────────────────────────


def test_divinebus_default_has_no_broker():
    bus = DivineBus()
    assert bus.has_socket_broker is False


def test_divinebus_publish_unaffected_when_no_broker_attached():
    """Legacy path: no broker → publish behaves exactly as before."""
    bus = DivineBus()
    q = bus.subscribe("legacy_sub")
    delivered = bus.publish({
        "type": "TEST", "src": "k", "dst": "legacy_sub",
        "payload": {"v": 1},
    })
    assert delivered == 1
    msg = q.get(timeout=1.0)
    assert msg["payload"]["v"] == 1


class _FakeBroker:
    """Minimal stand-in for BusSocketServer used in attach/detach tests
    (avoids spinning up a real socket in unit tests)."""
    def __init__(self):
        self.sock_path = "/tmp/fake_bus.sock"
        self.received: list[dict] = []
        self.raise_next = False

    def publish(self, msg: dict) -> None:
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("simulated broker failure")
        self.received.append(msg)


def test_attach_broker_routes_publishes_to_both_paths():
    """When broker is attached, publish() delivers to BOTH the in-process
    subscriber AND the broker (cross-process subscribers will pick it up)."""
    bus = DivineBus()
    q = bus.subscribe("local_sub")
    fake_broker = _FakeBroker()
    bus.attach_broker(fake_broker)
    assert bus.has_socket_broker is True

    bus.publish({"type": "BOTH", "src": "k", "dst": "local_sub",
                 "payload": {"v": 7}})
    # In-process subscriber received it
    msg = q.get(timeout=1.0)
    assert msg["payload"]["v"] == 7
    # Broker also received it
    assert len(fake_broker.received) == 1
    assert fake_broker.received[0]["payload"]["v"] == 7


def test_detach_broker_reverts_to_legacy_only():
    bus = DivineBus()
    q = bus.subscribe("local_sub")
    fake_broker = _FakeBroker()
    bus.attach_broker(fake_broker)
    bus.publish({"type": "P1", "src": "k", "dst": "local_sub", "payload": {}})
    assert len(fake_broker.received) == 1
    # Drain in-process queue
    q.get(timeout=1.0)
    # Detach
    bus.detach_broker()
    assert bus.has_socket_broker is False
    bus.publish({"type": "P2", "src": "k", "dst": "local_sub", "payload": {}})
    # In-process still works
    msg = q.get(timeout=1.0)
    assert msg["type"] == "P2"
    # Broker did NOT receive P2
    assert len(fake_broker.received) == 1
    assert fake_broker.received[0]["type"] == "P1"


def test_broker_exception_does_not_break_in_process_delivery():
    """If the broker.publish raises (transient socket issue, etc.),
    in-process subscribers still receive their message — broker fan-out
    is best-effort."""
    bus = DivineBus()
    q = bus.subscribe("local_sub")
    fake_broker = _FakeBroker()
    fake_broker.raise_next = True
    bus.attach_broker(fake_broker)
    # Should not raise even though broker.publish does
    delivered = bus.publish({"type": "RESILIENT", "src": "k",
                             "dst": "local_sub", "payload": {}})
    assert delivered == 1
    msg = q.get(timeout=1.0)
    assert msg["type"] == "RESILIENT"


def test_broadcast_publish_routes_to_broker_too():
    """dst='all' — both in-process broadcast AND broker fan-out happen."""
    bus = DivineBus()
    q1 = bus.subscribe("a")
    q2 = bus.subscribe("b")
    fake_broker = _FakeBroker()
    bus.attach_broker(fake_broker)
    bus.publish({"type": "BCAST", "src": "k", "dst": "all", "payload": {}})
    assert q1.get(timeout=1.0)["type"] == "BCAST"
    assert q2.get(timeout=1.0)["type"] == "BCAST"
    assert len(fake_broker.received) == 1
    assert fake_broker.received[0]["type"] == "BCAST"


# ── End-to-end via real BusSocketServer ────────────────────────────────────


def test_dual_mode_with_real_broker_e2e(tmp_path):
    """Real socket-mode test: in-process subscriber receives via ThreadQueue,
    cross-process-style BusSocketClient receives via socket. Both off the
    same DivineBus.publish() call."""
    from titan_plugin.core.bus_socket import BusSocketClient, BusSocketServer

    sock = tmp_path / "bus.sock"
    authkey = b"x" * 32

    # Real broker
    broker = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    broker.start()

    bus = DivineBus()
    inproc_q = bus.subscribe("inproc_sub")
    bus.attach_broker(broker)

    # External client
    client = BusSocketClient(titan_id="testT", authkey=authkey,
                             name="external_sub", sock_path=sock)
    client.start()
    assert client.wait_until_connected(timeout=3.0)
    # Wait for broker registration
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with broker._subs_lock:
            if "external_sub" in broker._subscribers:
                break
        time.sleep(0.02)
    ext_q = client.inbound_queue()

    try:
        # Publish a broadcast — both subscribers receive
        bus.publish({"type": "DUAL", "src": "k", "dst": "all",
                     "payload": {"v": 42}})
        m_inproc = inproc_q.get(timeout=2.0)
        assert m_inproc["payload"]["v"] == 42
        # Filter for DUAL on external (may have BUS_PING etc in flight)
        deadline = time.time() + 3.0
        m_ext = None
        while time.time() < deadline:
            try:
                cand = ext_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if cand.get("type") == "DUAL":
                m_ext = cand
                break
        assert m_ext is not None, "external subscriber did not receive DUAL"
        assert m_ext["payload"]["v"] == 42
    finally:
        client.stop()
        bus.detach_broker()
        broker.stop(timeout=2.0)
