"""
End-to-end in-process test for bus_socket — server + multiple clients
exercising the full B.2 promise:

  • Multiple workers connect via Unix socket
  • Coalesce works under load (state stays fresh, never accumulates)
  • Broker swap: kill broker, start a new one on the SAME path → all
    clients auto-reconnect, re-subscribe, message flow resumes WITHOUT
    any client restart. This is the architectural promise of B.2.

These tests run server + clients in the same Python process for speed; the
real cross-process test (workers in separate Python processes) lives in C7.
"""
from __future__ import annotations

import queue
import time

import pytest

from titan_plugin.core.bus_socket import (
    BusSocketClient,
    BusSocketServer,
)


@pytest.fixture
def authkey() -> bytes:
    return b"k" * 32


def _start_pair(tmp_path, authkey, name="w1"):
    sock = tmp_path / "bus.sock"
    server = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    server.start()
    client = BusSocketClient(
        titan_id="testT", authkey=authkey, name=name, sock_path=sock,
    )
    client.start()
    assert client.wait_until_connected(timeout=3.0)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with server._subs_lock:
            if name in server._subscribers:
                return server, client
        time.sleep(0.02)
    raise AssertionError(f"broker never registered {name}")


def _drain_for_type(q, mtype, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            m = q.get(timeout=max(0.05, deadline - time.time()))
        except queue.Empty:
            continue
        if m.get("type") == mtype:
            return m
    return None


# ── Multi-client routing ───────────────────────────────────────────────────


def test_three_workers_concurrent_publish_and_receive(tmp_path, authkey):
    sock = tmp_path / "bus.sock"
    server = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    server.start()
    clients = []
    queues = []
    try:
        for name in ("w1", "w2", "w3"):
            c = BusSocketClient(titan_id="testT", authkey=authkey, name=name,
                                sock_path=sock)
            c.start()
            assert c.wait_until_connected(timeout=3.0)
            clients.append(c)
            queues.append(c.inbound_queue())
        # Wait for all to register
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with server._subs_lock:
                if all(n in server._subscribers for n in ("w1", "w2", "w3")):
                    break
            time.sleep(0.05)

        # w1 → w2
        clients[0].publish({"type": "P12", "src": "w1", "dst": "w2",
                            "payload": {"v": 1}})
        m = _drain_for_type(queues[1], "P12", timeout=3.0)
        assert m is not None and m["payload"]["v"] == 1

        # w2 → w3
        clients[1].publish({"type": "P23", "src": "w2", "dst": "w3",
                            "payload": {"v": 2}})
        m = _drain_for_type(queues[2], "P23", timeout=3.0)
        assert m is not None and m["payload"]["v"] == 2

        # broadcast from kernel-side
        server.publish({"type": "BCAST", "src": "kernel", "dst": "all",
                        "payload": {"k": 9}})
        for q in queues:
            m = _drain_for_type(q, "BCAST", timeout=3.0)
            assert m is not None and m["payload"]["k"] == 9
    finally:
        for c in clients:
            c.stop()
        server.stop(timeout=2.0)


# ── E2E coalesce: client sees only freshest BODY_STATE ─────────────────────


def test_e2e_coalesce_through_real_socket(tmp_path, authkey):
    """The graceful-design proof at the C5 boundary: a slow consumer
    receives only the freshest BODY_STATE, even though 100 were sent."""
    server, client = _start_pair(tmp_path, authkey, name="body_consumer")
    q = client.inbound_queue()
    try:
        # Publish 100 BODY_STATE — coalesce on (src, type) means broker
        # collapses them into a single ring slot
        for i in range(100):
            server.publish({
                "type": "BODY_STATE", "src": "body_worker",
                "dst": "body_consumer", "payload": {"seq": i},
            })
        # Drain BODY_STATE messages
        seen = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                m = q.get(timeout=0.3)
            except queue.Empty:
                break
            if m.get("type") == "BODY_STATE":
                seen.append(m["payload"]["seq"])
        # We must see at least one BODY_STATE; the LAST one we see must be
        # the freshest (seq=99) because coalesce mutates in place
        assert seen, "no BODY_STATE delivered"
        assert seen[-1] == 99, f"latest seq should be 99, got seen={seen}"
        # And we should see far fewer than 100 delivered (coalesce worked)
        assert len(seen) < 100, f"coalesce did not work; got {len(seen)} unique"
    finally:
        client.stop()
        server.stop(timeout=2.0)


# ── E2E broker swap (the B.2 promise) ──────────────────────────────────────


def test_broker_swap_invisible_to_clients(tmp_path, authkey):
    """SIMULATES KERNEL SWAP: kill broker, restart on same socket path,
    client reconnects + resubscribes automatically; message flow resumes.

    This is the load-bearing test for Phase B.2. If this works, workers
    truly outlive kernel swaps and Schumann ticks never pause."""
    sock = tmp_path / "bus.sock"
    # Phase A: original kernel
    s1 = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s1.start()
    c = BusSocketClient(titan_id="testT", authkey=authkey, name="persistent_w",
                        sock_path=sock)
    c.start()
    q = c.inbound_queue()
    try:
        assert c.wait_until_connected(timeout=3.0)
        # Wait for broker to process BUS_SUBSCRIBE (promote anon-N to persistent_w)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with s1._subs_lock:
                if "persistent_w" in s1._subscribers:
                    break
            time.sleep(0.02)
        # Phase A flow
        s1.publish({"type": "PRE_SWAP", "src": "k1", "dst": "persistent_w",
                    "payload": {"v": 1}})
        m = _drain_for_type(q, "PRE_SWAP", timeout=3.0)
        assert m is not None
        # Capture connection state pre-swap
        pre_reconnects = c.reconnect_count

        # Kernel swap moment: kill broker
        s1.stop(timeout=2.0)
        time.sleep(0.1)

        # Phase B: new kernel binds same path
        s2 = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
        s2.start()
        try:
            # Wait for client auto-reconnect
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if c.is_connected and c.reconnect_count > pre_reconnects:
                    break
                time.sleep(0.05)
            assert c.is_connected, "client did not reconnect"
            assert c.reconnect_count > pre_reconnects, "reconnect counter did not advance"

            # Wait for re-subscription to land
            deadline = time.time() + 2.0
            while time.time() < deadline:
                with s2._subs_lock:
                    if "persistent_w" in s2._subscribers:
                        break
                time.sleep(0.05)

            # Phase B flow
            s2.publish({"type": "POST_SWAP", "src": "k2", "dst": "persistent_w",
                        "payload": {"v": 2}})
            m = _drain_for_type(q, "POST_SWAP", timeout=3.0)
            assert m is not None
            assert m["payload"]["v"] == 2
        finally:
            s2.stop(timeout=2.0)
    finally:
        c.stop()


def test_three_clients_all_survive_broker_swap(tmp_path, authkey):
    """The full fleet swap: 3 clients all reconnect + resubscribe + flow resumes."""
    sock = tmp_path / "bus.sock"
    s1 = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s1.start()
    clients = []
    qs = []
    try:
        for name in ("body", "mind", "spirit"):
            c = BusSocketClient(titan_id="testT", authkey=authkey, name=name,
                                sock_path=sock)
            c.start()
            assert c.wait_until_connected(timeout=3.0)
            clients.append(c)
            qs.append(c.inbound_queue())
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with s1._subs_lock:
                if all(n in s1._subscribers for n in ("body", "mind", "spirit")):
                    break
            time.sleep(0.05)

        pre_rc = [c.reconnect_count for c in clients]
        s1.stop(timeout=2.0)
        time.sleep(0.2)
        s2 = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
        s2.start()
        try:
            # All 3 must reconnect — generous budget for suite-load (3 clients
            # reconnecting simultaneously hit accept_rate_limit + backoff jitter)
            deadline = time.time() + 12.0
            while time.time() < deadline:
                if all(c.is_connected and c.reconnect_count > pre for c, pre
                       in zip(clients, pre_rc)):
                    break
                time.sleep(0.1)
            for i, c in enumerate(clients):
                assert c.is_connected, f"client {i} did not reconnect"
                assert c.reconnect_count > pre_rc[i]
            # Wait for re-subscriptions to land on broker
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with s2._subs_lock:
                    if all(n in s2._subscribers for n in ("body", "mind", "spirit")):
                        break
                time.sleep(0.1)
            with s2._subs_lock:
                assert all(n in s2._subscribers for n in ("body", "mind", "spirit")), \
                    f"re-subscriptions incomplete: {list(s2._subscribers.keys())}"
            # Broadcast must reach all 3
            s2.publish({"type": "POST_RESUME", "src": "k2", "dst": "all",
                        "payload": {}})
            for i, q in enumerate(qs):
                m = _drain_for_type(q, "POST_RESUME", timeout=5.0)
                assert m is not None, f"client {i} ({clients[i].name}) did not receive POST_RESUME"
        finally:
            s2.stop(timeout=2.0)
    finally:
        for c in clients:
            c.stop()
