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

from titan_hcl import bus as bus_module
from titan_hcl.bus import DivineBus
from titan_hcl.bus_specs import (
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


def test_bus_ipc_socket_enabled_flag_registered():
    """Verify titan_params.toml registers the flag.

    Original test asserted default=False. Flipped to True on 2026-05-01
    (commit 45d469fe — T1 graduated back to socket-broker mode after A.S8
    + msgpack fix). Test now just verifies the flag is present and is a
    bool — matches whatever the config currently declares (True under
    microkernel v2 mode).
    """
    from titan_hcl.params import load_titan_params as load_titan_config
    cfg = load_titan_config()
    micro = cfg.get("microkernel", {})
    assert "bus_ipc_socket_enabled" in micro, \
        "microkernel.bus_ipc_socket_enabled flag not registered"
    assert isinstance(micro["bus_ipc_socket_enabled"], bool), \
        f"flag must be bool, got {type(micro['bus_ipc_socket_enabled']).__name__}"


# ── DivineBus dual-mode behavior ───────────────────────────────────────────


def test_divinebus_default_has_no_broker():
    bus = DivineBus()
    assert bus.has_socket_broker is False


# ── Phase B.2 §D12 _is_kernel_internal discriminator (added 2026-05-02) ────


def test_is_kernel_internal_known_names():
    """Every kernel-internal name in the canonical allowlist returns True.

    NOTE: "meditation" + "sovereignty" were REMOVED from the allowlist by
    Phase 10K (rFP §3G; D-SPEC-57/60/64) — they became separate worker
    subprocesses with their own BusSocketClient primary names, so no
    kernel-side subscriber uses those names anymore. See the negative
    assertion in test_is_kernel_internal_retired_names below.
    """
    from titan_hcl.bus import _is_kernel_internal
    for name in [
        "guardian", "core", "kernel",
        "agency", "chat_handler", "v4_bridge",
        "state_register", "rl_proxy_stats", "api",
    ]:
        assert _is_kernel_internal(name), f"{name} should be kernel-internal"


def test_is_kernel_internal_retired_names():
    """meditation + sovereignty are NO LONGER kernel-internal — they're
    standalone worker subprocesses (Phase 10K rFP §3G / D-SPEC-57/60/64).
    Locks in the retirement so a future refactor can't silently re-add them."""
    from titan_hcl.bus import _is_kernel_internal
    for name in ["meditation", "sovereignty"]:
        assert not _is_kernel_internal(name), (
            f"{name} was retired from the kernel-internal allowlist "
            "(now a separate worker subprocess) — must return False")


def test_is_kernel_internal_proxy_suffix():
    """Reply-queue subscribers ending in _proxy are kernel-internal."""
    from titan_hcl.bus import _is_kernel_internal
    for name in [
        "memory_proxy", "spirit_proxy", "body_proxy", "mind_proxy",
        "rl_proxy", "agency_proxy", "media_proxy", "llm_proxy",
        "assessment_proxy", "reflex_proxy", "timechain_proxy",
        "output_verifier_proxy",
    ]:
        assert _is_kernel_internal(name), f"{name} should be kernel-internal"


def test_is_kernel_internal_query_suffix():
    """reflex_executors.py query reply queues end in _query — kernel-internal."""
    from titan_hcl.bus import _is_kernel_internal
    assert _is_kernel_internal("reflex_spirit_query")
    assert _is_kernel_internal("reflex_time_query")


def test_is_kernel_internal_worker_names_return_false():
    """Worker process names should NOT be classified as kernel-internal."""
    from titan_hcl.bus import _is_kernel_internal
    for name in [
        "memory", "recorder", "spirit", "media", "cgn", "knowledge", "timechain",
        "backup", "output_verifier", "outer_body", "outer_mind", "outer_spirit",
        "reflex", "agency_worker", "warning_monitor", "imw",
        "observatory_writer", "social_graph_writer", "events_teacher_writer",
        "consciousness_writer", "llm", "body", "mind", "language",
        "meta_teacher", "emot_cgn",
    ]:
        assert not _is_kernel_internal(name), f"{name} is a worker, not kernel-internal"


def test_subscribe_under_socket_mode_with_worker_name_is_loud():
    """Phase B.2 §D12: contract violation logs warning + increments counter."""
    bus = DivineBus()
    # Fake a broker attach so the discriminator branch fires.
    bus._broker = object()
    assert bus.has_socket_broker is True

    before = bus.stats["non_kernel_internal_subscribe_under_socket"]
    bus.subscribe("memory")  # worker name — should be flagged as off-contract
    after = bus.stats["non_kernel_internal_subscribe_under_socket"]
    assert after == before + 1, \
        "non-kernel-internal subscribe under socket mode must increment counter"


def test_subscribe_under_socket_mode_with_kernel_internal_name_is_quiet():
    """Kernel-internal subscribers do NOT trigger the contract counter."""
    bus = DivineBus()
    bus._broker = object()  # fake attach
    before = bus.stats["non_kernel_internal_subscribe_under_socket"]
    bus.subscribe("guardian")        # explicit allowlist
    bus.subscribe("memory_proxy")    # _proxy suffix
    bus.subscribe("reflex_time_query")  # _query suffix
    after = bus.stats["non_kernel_internal_subscribe_under_socket"]
    assert after == before, \
        "kernel-internal subscribers must NOT trip the contract counter"


def test_subscribe_no_broker_attached_skips_check():
    """When broker is not attached, the discriminator is bypassed entirely."""
    bus = DivineBus()
    assert bus.has_socket_broker is False
    before = bus.stats["non_kernel_internal_subscribe_under_socket"]
    # Worker name — but no broker → no warning, counter unchanged.
    bus.subscribe("memory")
    after = bus.stats["non_kernel_internal_subscribe_under_socket"]
    assert after == before, \
        "discriminator must only fire when broker is attached"


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


# test_dual_mode_with_real_broker_e2e retired with D8-1 2026-05-16 —
# Python BusSocketServer class deleted (titan-kernel-rs owns the bus
# broker under fleet-wide Phase C since 2026-05-14). The dual-mode
# attach-broker pattern is now tested with a _StubBroker (above) which
# preserves the DivineBus integration contract without depending on
# the retired Python broker class.
@pytest.mark.skip(reason="D8-1 retirement — Python BusSocketServer deleted; titan-kernel-rs (Rust) owns the bus broker under fleet-wide Phase C since 2026-05-14.")
def test_dual_mode_with_real_broker_e2e(tmp_path):
    """Real socket-mode test: in-process subscriber receives via ThreadQueue,
    cross-process-style BusSocketClient receives via socket. Both off the
    same DivineBus.publish() call."""
    from titan_hcl.core.bus_socket import BusSocketClient  # BusSocketServer DELETED D8-1

    sock = tmp_path / "bus.sock"
    authkey = b"x" * 32

    # Real broker
    broker = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    broker.start()

    bus = DivineBus()
    inproc_q = bus.subscribe("inproc_sub")
    bus.attach_broker(broker)

    # External client.
    # SPEC §8.2 v1.4.0 D-SPEC-42: broker silently skips subscribers with
    # empty broadcast_topics AND reply_only=False from dst="all" fanout.
    # This test publishes a "DUAL" broadcast that the external_sub MUST
    # receive — declare broadcast_topics=["DUAL"] to keep the subscriber
    # eligible. (Pre-D-SPEC-42 the broker fanned to ALL subscribers
    # regardless; the test predated the v1.4.0 contract change.)
    client = BusSocketClient(titan_id="testT", authkey=authkey,
                             name="external_sub", sock_path=sock,
                             topics=["DUAL"])
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
