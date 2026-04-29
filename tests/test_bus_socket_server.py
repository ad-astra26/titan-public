"""
Tests for the BusSocketServer broker (Phase B.2 C4).

Uses a minimal raw-socket client helper to exercise the server in isolation.
The full BusSocketClient + SocketQueue land in C5 — they sit on top of this.

Coverage targets (per PLAN §7 chunk C4):
- Bind + chmod 0600 on socket file
- Accept + handshake (correct authkey passes, wrong rejected)
- BUS_SUBSCRIBE registers subscriber name + topics
- Broker routes published messages to matching subscribers (by dst, topic)
- Coalesce on (src, type) — multiple BODY_STATE land in single ring slot
- Priority: P0 reserves slots; P3 drops newest under hard pressure
- BUS_PING sent every PING_INTERVAL_S; broker disconnects on PONG timeout
- Accept rate limit
- Send batching when >=5 msgs queued for one subscriber
- Server stop unbinds socket + unlinks file
"""
from __future__ import annotations

import os
import socket
import stat
import time

import msgpack
import pytest

from titan_plugin.core.bus_socket import (
    PING_INTERVAL_S,
    PING_TIMEOUT_S,
    SEND_BATCH_THRESHOLD,
    BoundedRing,
    BrokerSubscriber,
    BusSocketServer,
    bus_sock_path,
)
from titan_plugin.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    compute_hmac,
    recv_exact,
    recv_frame,
    send_frame,
)


# ── BoundedRing unit tests ─────────────────────────────────────────────────


def test_boundedring_basic_append_and_pop():
    r = BoundedRing(capacity=10, p0_reserve=2)
    assert r.is_empty()
    assert r.append_main({"id": 1}) is True
    assert r.append_main({"id": 2}) is True
    assert len(r) == 2
    out = r.pop_for_send(max_msgs=10)
    assert [m["id"] for m in out] == [1, 2]
    assert r.is_empty()


def test_boundedring_p0_reserve_separate_from_main():
    r = BoundedRing(capacity=10, p0_reserve=2)
    # Fill main fully
    for i in range(8):
        assert r.append_main({"id": i}) is True
    assert r.main_is_full()
    # P0 still has room
    assert r.append_p0({"id": "p0_a"}) is True
    assert r.append_p0({"id": "p0_b"}) is True
    out = r.pop_for_send(max_msgs=99)
    # P0 drains FIRST
    assert out[0]["id"] == "p0_a"
    assert out[1]["id"] == "p0_b"
    assert [m["id"] for m in out[2:]] == list(range(8))


def test_boundedring_main_eviction_returns_false():
    r = BoundedRing(capacity=4, p0_reserve=1)  # main maxlen=3
    assert r.append_main({"id": 1}) is True
    assert r.append_main({"id": 2}) is True
    assert r.append_main({"id": 3}) is True
    assert r.append_main({"id": 4}) is False  # evicts 1
    out = r.pop_for_send(max_msgs=10)
    assert [m["id"] for m in out] == [2, 3, 4]


def test_boundedring_invalid_p0_reserve_raises():
    with pytest.raises(ValueError):
        BoundedRing(capacity=10, p0_reserve=10)
    with pytest.raises(ValueError):
        BoundedRing(capacity=10, p0_reserve=99)


# ── BusSocketServer fixtures + raw client helper ──────────────────────────


@pytest.fixture
def authkey() -> bytes:
    return b"k" * 32


@pytest.fixture
def server(tmp_path, authkey) -> BusSocketServer:
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s.start()
    try:
        yield s
    finally:
        s.stop(timeout=2.0)


def _connect_and_handshake(sock_path, authkey: bytes, *,
                           subscribe_as: str | None = None,
                           topics: list[str] | None = None) -> socket.socket:
    """Helper — connect, complete handshake, optionally send BUS_SUBSCRIBE."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect(str(sock_path))
    challenge = recv_exact(s, CHALLENGE_SIZE)
    response = compute_hmac(authkey, challenge)
    s.sendall(response)
    if subscribe_as is not None:
        sub_msg = {
            "type": "BUS_SUBSCRIBE",
            "src": subscribe_as,
            "dst": "broker",
            "payload": {"name": subscribe_as, "topics": topics or []},
        }
        send_frame(s, msgpack.packb(sub_msg))
    return s


def _wait_for_subscriber(server: BusSocketServer, name: str,
                         timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with server._subs_lock:
            if name in server._subscribers:
                return True
        time.sleep(0.02)
    return False


def _drain_one(s: socket.socket, timeout: float = 2.0) -> dict | None:
    """Receive one frame, decode it. Returns None on timeout/EOF."""
    s.settimeout(timeout)
    try:
        payload = recv_frame(s)
    except (ConnectionError, OSError, socket.timeout):
        return None
    return msgpack.unpackb(payload, raw=False)


def _drain_until(s: socket.socket, predicate, timeout: float = 5.0) -> dict | None:
    """Drain frames until one matches predicate (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        msg = _drain_one(s, timeout=remaining)
        if msg is None:
            return None
        # Could be a BUS_BATCH wrapper
        if msg.get("type") == "BUS_BATCH":
            for inner in msg.get("msgs", []):
                if predicate(inner):
                    return inner
            continue
        if predicate(msg):
            return msg
    return None


# ── Bind + permissions ─────────────────────────────────────────────────────


def test_socket_file_chmod_0600(server):
    mode = os.stat(server.sock_path).st_mode
    perm_bits = stat.S_IMODE(mode)
    assert perm_bits == 0o600


def test_server_stop_unlinks_socket(tmp_path, authkey):
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="x", authkey=authkey, sock_path=sock)
    s.start()
    assert sock.exists()
    s.stop(timeout=2.0)
    assert not sock.exists()


def test_server_unlinks_stale_socket_file(tmp_path, authkey):
    """A leftover socket file from a prior crashed kernel must be unlinked
    cleanly so a new kernel can bind."""
    sock = tmp_path / "bus.sock"
    sock.touch()  # stale file
    s = BusSocketServer(titan_id="x", authkey=authkey, sock_path=sock)
    s.start()
    try:
        assert sock.exists()  # rebound
    finally:
        s.stop(timeout=2.0)


# ── Handshake ──────────────────────────────────────────────────────────────


def test_correct_authkey_handshake(server, authkey):
    s = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w1")
    try:
        assert _wait_for_subscriber(server, "w1")
    finally:
        s.close()


def test_wrong_authkey_closes_connection(server):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect(str(server.sock_path))
    challenge = recv_exact(s, CHALLENGE_SIZE)
    bad_response = compute_hmac(b"WRONG_KEY" * 4, challenge)  # 32 bytes
    s.sendall(bad_response)
    # Server should close connection — recv returns 0 bytes
    try:
        s.settimeout(2.0)
        leftover = s.recv(1024)
        assert leftover == b""
    except (ConnectionError, OSError):
        pass  # also acceptable
    finally:
        s.close()


# ── Subscribe + dispatch ───────────────────────────────────────────────────


def test_publish_routed_to_subscriber_by_dst(server, authkey):
    s = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w1")
    try:
        assert _wait_for_subscriber(server, "w1")
        # Kernel-side publish targeting w1
        server.publish({"type": "TEST_MSG", "src": "kernel", "dst": "w1",
                        "payload": {"x": 42}})
        msg = _drain_until(s, lambda m: m.get("type") == "TEST_MSG")
        assert msg is not None
        assert msg["payload"]["x"] == 42
    finally:
        s.close()


def test_publish_broadcast_dst_all(server, authkey):
    a = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w1")
    b = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w2")
    try:
        assert _wait_for_subscriber(server, "w1")
        assert _wait_for_subscriber(server, "w2")
        server.publish({"type": "BCAST", "src": "kernel", "dst": "all",
                        "payload": {"k": 1}})
        ma = _drain_until(a, lambda m: m.get("type") == "BCAST")
        mb = _drain_until(b, lambda m: m.get("type") == "BCAST")
        assert ma is not None and mb is not None
        assert ma["payload"]["k"] == 1
        assert mb["payload"]["k"] == 1
    finally:
        a.close()
        b.close()


def test_worker_publish_routed_to_other_subscriber(server, authkey):
    """When one worker publishes via the broker, another subscriber receives it."""
    a = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w1")
    b = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w2")
    try:
        assert _wait_for_subscriber(server, "w1")
        assert _wait_for_subscriber(server, "w2")
        # w1 publishes a message destined for w2
        send_frame(a, msgpack.packb({"type": "WORK_OUT", "src": "w1", "dst": "w2",
                                     "payload": {"v": 99}}))
        msg = _drain_until(b, lambda m: m.get("type") == "WORK_OUT")
        assert msg is not None
        assert msg["payload"]["v"] == 99
        # w1 must NOT receive its own publish
        assert _drain_until(a, lambda m: m.get("type") == "WORK_OUT", timeout=0.5) is None
    finally:
        a.close()
        b.close()


# ── Coalesce ───────────────────────────────────────────────────────────────


def test_coalesce_body_state_keeps_only_freshest_when_consumer_is_slow(server, authkey):
    """Drop a stream of BODY_STATE while consumer is paused; verify only the
    final (freshest) state is delivered.

    This is the GRACEFUL design proof: stale state never reaches consumers.
    """
    s = _connect_and_handshake(server.sock_path, authkey, subscribe_as="body_consumer")
    try:
        assert _wait_for_subscriber(server, "body_consumer")
        # Pause the consumer briefly via a sleep BEFORE we read frames.
        # Send 50 BODY_STATE messages while consumer isn't reading
        for i in range(50):
            server.publish({"type": "BODY_STATE", "src": "body_worker",
                            "dst": "body_consumer",
                            "payload": {"seq": i}})
        # Now drain. With coalesce on (src, type), only the LAST one should be
        # in the ring (the 49 prior were overwritten in place).
        msg = _drain_until(s, lambda m: m.get("type") == "BODY_STATE", timeout=3.0)
        assert msg is not None
        # Freshness — must be the latest seq we sent
        assert msg["payload"]["seq"] == 49
        # No more BODY_STATE in the pipe (those 49 are gone, never sent)
        followup = _drain_until(s, lambda m: m.get("type") == "BODY_STATE", timeout=0.5)
        assert followup is None
    finally:
        s.close()


# ── BUS_PING heartbeat ─────────────────────────────────────────────────────


def test_broker_sends_periodic_ping(server, authkey):
    """Broker sends BUS_PING every PING_INTERVAL_S — confirms subscriber liveness."""
    s = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w_ping")
    try:
        assert _wait_for_subscriber(server, "w_ping")
        # Wait for first ping (allow PING_INTERVAL_S + slack)
        msg = _drain_until(s, lambda m: m.get("type") == "BUS_PING",
                           timeout=PING_INTERVAL_S + 2.0)
        assert msg is not None
        assert msg["src"] == "broker"
    finally:
        s.close()


# ── Server stop ────────────────────────────────────────────────────────────


def test_server_stop_closes_connections(tmp_path, authkey):
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="x", authkey=authkey, sock_path=sock)
    s.start()
    try:
        c = _connect_and_handshake(sock, authkey, subscribe_as="w1")
        assert _wait_for_subscriber(s, "w1")
    finally:
        s.stop(timeout=2.0)
    # After stop, the connection should be closed; reads return EOF
    c.settimeout(2.0)
    try:
        r = c.recv(1024)
        assert r == b""
    except (ConnectionError, OSError):
        pass  # also acceptable
    c.close()


# ── stats ──────────────────────────────────────────────────────────────────


def test_stats_reports_subscribers(server, authkey):
    a = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w1",
                               topics=["BODY_STATE", "MIND_STATE"])
    try:
        assert _wait_for_subscriber(server, "w1")
        st = server.stats()
        assert st["subscriber_count"] == 1
        assert st["subscribers"][0]["name"] == "w1"
        assert sorted(st["subscribers"][0]["topics"]) == ["BODY_STATE", "MIND_STATE"]
    finally:
        a.close()


# ── Broker stays alive when one client misbehaves ─────────────────────────


def test_broker_survives_client_send_garbage(server, authkey):
    """Send a malformed (non-msgpack) frame after handshake — broker should
    drop just THAT connection, not crash."""
    s = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w_bad")
    try:
        assert _wait_for_subscriber(server, "w_bad")
        send_frame(s, b"\xc1" * 16)  # invalid msgpack
        # Broker should close this conn but stay running for others
        time.sleep(0.5)
        # New connection still works
        s2 = _connect_and_handshake(server.sock_path, authkey, subscribe_as="w_good")
        assert _wait_for_subscriber(server, "w_good")
        s2.close()
    finally:
        s.close()
