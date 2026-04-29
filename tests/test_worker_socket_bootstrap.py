"""
Tests for worker_bus_bootstrap (Phase B.2 C7).

The worker bootstrap is the env-driven helper that lets every Guardian-spawned
worker pick mp.Queue (legacy) vs SocketQueue (B.2) based on env vars set
by the kernel when bus_ipc_socket_enabled=true. We verify:

- Mode resolution from env vars (in-process; no subprocess needed)
  - Legacy mode when no env vars set
  - Legacy fallback when keypair file missing
  - Socket mode when all env vars present + keypair valid
- End-to-end with a REAL FORKED Python subprocess: the worker process
  bootstraps via env, connects to broker, exchanges a message, exits clean
- Reconnect: kill broker, restart on same path → forked worker reconnects
  WITHOUT being killed. The full B.2 architectural promise across a real
  process boundary.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time

import msgpack
import pytest

from titan_plugin.core.bus_socket import BusSocketServer
from titan_plugin.core.worker_bus_bootstrap import (
    ENV_BUS_KEYPAIR_PATH,
    ENV_BUS_SOCKET_PATH,
    ENV_BUS_TITAN_ID,
    setup_worker_bus,
)


# ── In-process env-driven resolution ───────────────────────────────────────


def _make_keypair_file(tmp_path) -> str:
    """Write a valid 64-int Solana-style keypair JSON to tmp_path."""
    p = tmp_path / "keypair.json"
    payload = list(range(64))  # any 64 bytes; HKDF accepts arbitrary bytes
    p.write_text(json.dumps(payload))
    return str(p)


def test_legacy_mode_when_no_env_vars():
    """No env vars set → workers get back the original mp.Queue handles."""
    recv_q = "MOCK_RECV"
    send_q = "MOCK_SEND"
    out_recv, out_send, client = setup_worker_bus(
        "w1", recv_q, send_q, env={})
    assert out_recv is recv_q
    assert out_send is send_q
    assert client is None


def test_legacy_mode_when_keypair_missing(tmp_path):
    """Env vars set but keypair file doesn't exist → graceful fallback to legacy."""
    env = {
        ENV_BUS_SOCKET_PATH: str(tmp_path / "nonexistent.sock"),
        ENV_BUS_TITAN_ID: "testT",
        ENV_BUS_KEYPAIR_PATH: str(tmp_path / "missing_keypair.json"),
    }
    recv_q = "MOCK_RECV"
    send_q = "MOCK_SEND"
    out_recv, out_send, client = setup_worker_bus(
        "w1", recv_q, send_q, env=env)
    assert out_recv is recv_q
    assert out_send is send_q
    assert client is None


def test_legacy_mode_when_keypair_malformed(tmp_path):
    """Env vars set, file exists but is invalid → graceful fallback."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    env = {
        ENV_BUS_SOCKET_PATH: "/tmp/whatever.sock",
        ENV_BUS_TITAN_ID: "testT",
        ENV_BUS_KEYPAIR_PATH: str(bad),
    }
    out_recv, out_send, client = setup_worker_bus("w1", "RECV", "SEND", env=env)
    assert out_recv == "RECV"
    assert out_send == "SEND"
    assert client is None


def test_socket_mode_when_env_complete(tmp_path):
    """All env vars set + valid keypair + broker running → SocketQueue path."""
    keypair = _make_keypair_file(tmp_path)
    sock = tmp_path / "bus.sock"
    # Need broker for client to actually connect
    from titan_plugin.core.bus_authkey import derive_bus_authkey
    secret = bytes(range(64))
    authkey = derive_bus_authkey(secret, "testT")
    broker = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    broker.start()
    env = {
        ENV_BUS_SOCKET_PATH: str(sock),
        ENV_BUS_TITAN_ID: "testT",
        ENV_BUS_KEYPAIR_PATH: keypair,
    }
    try:
        recv, send, client = setup_worker_bus("w1", "MP_RECV", "MP_SEND", env=env)
        try:
            # Different objects from the legacy queue handles
            assert recv != "MP_RECV"
            assert client is not None
            assert client.wait_until_connected(timeout=3.0)
            # recv and send share the SAME SocketQueue (single-client design)
            assert recv is send
        finally:
            client.stop()
    finally:
        broker.stop(timeout=2.0)


# ── Real-fork worker subprocess test ───────────────────────────────────────


def _worker_subprocess_main(sock_path: str, keypair_path: str, titan_id: str,
                            result_path: str, name: str = "fork_worker") -> None:
    """Standalone fork-worker entry. Bootstraps via env + writes received
    messages to a JSON file the test reads.

    Lives at module scope so multiprocessing fork can pickle it (well, actually
    we use fork so pickling isn't needed — but multiprocessing prefers it
    importable for safety)."""
    import json as _json
    from titan_plugin.core.worker_bus_bootstrap import setup_worker_bus
    env = {
        ENV_BUS_SOCKET_PATH: sock_path,
        ENV_BUS_TITAN_ID: titan_id,
        ENV_BUS_KEYPAIR_PATH: keypair_path,
    }
    # Set os.environ so setup_worker_bus default-arg path also works for
    # workers that don't pass env=
    os.environ.update(env)
    recv_q, send_q, client = setup_worker_bus(name, None, None, env=env)
    if client is None:
        # Bootstrap fell back to legacy — that's a test failure
        with open(result_path, "w") as f:
            _json.dump({"error": "fell_back_to_legacy"}, f)
        return
    received: list[dict] = []
    try:
        if not client.wait_until_connected(timeout=5.0):
            with open(result_path, "w") as f:
                _json.dump({"error": "did_not_connect"}, f)
            return
        # Drain up to 3 messages or 4s; ignore BUS_PING etc
        deadline = time.time() + 4.0
        while time.time() < deadline and len(received) < 3:
            try:
                msg = recv_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if msg.get("type") == "TEST_TO_WORKER":
                received.append(msg)
        # Emit a publish back to the kernel-side test
        send_q.put_nowait({"type": "FROM_WORKER", "src": name,
                           "dst": "test_observer",
                           "payload": {"got": len(received)}})
        # Tell test we're done
        with open(result_path, "w") as f:
            _json.dump({"ok": True, "got": len(received),
                        "reconnects": client.reconnect_count}, f)
    finally:
        client.stop()


def test_real_fork_worker_connects_and_exchanges(tmp_path):
    """Spawn a real Python subprocess; that subprocess bootstraps via env,
    connects to broker, receives messages we publish, sends back its own
    publish, exits cleanly."""
    keypair = _make_keypair_file(tmp_path)
    sock = tmp_path / "bus.sock"
    result = tmp_path / "result.json"
    titan_id = "testT"
    # Boot broker
    from titan_plugin.core.bus_authkey import derive_bus_authkey
    secret = bytes(range(64))
    authkey = derive_bus_authkey(secret, titan_id)
    broker = BusSocketServer(titan_id=titan_id, authkey=authkey, sock_path=sock)
    broker.start()

    # Subscribe ourselves so we can observe FROM_WORKER on the broker side
    from titan_plugin.core.bus_socket import BusSocketClient
    observer = BusSocketClient(titan_id=titan_id, authkey=authkey,
                               name="test_observer", sock_path=sock)
    observer.start()
    obs_q = observer.inbound_queue()
    assert observer.wait_until_connected(timeout=3.0)
    # Wait for observer registration on broker
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with broker._subs_lock:
            if "test_observer" in broker._subscribers:
                break
        time.sleep(0.02)

    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_worker_subprocess_main,
        args=(str(sock), keypair, titan_id, str(result)),
        name="bus-test-worker",
        daemon=True,
    )
    try:
        proc.start()
        # Wait for worker to register
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with broker._subs_lock:
                if "fork_worker" in broker._subscribers:
                    break
            time.sleep(0.05)
        with broker._subs_lock:
            assert "fork_worker" in broker._subscribers, \
                "subprocess worker never registered on broker"
        # Publish 3 messages to the worker
        for i in range(3):
            broker.publish({"type": "TEST_TO_WORKER", "src": "kernel",
                            "dst": "fork_worker", "payload": {"i": i}})
        # Worker should send FROM_WORKER back
        deadline = time.time() + 5.0
        from_worker = None
        while time.time() < deadline:
            try:
                m = obs_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if m.get("type") == "FROM_WORKER":
                from_worker = m
                break
        assert from_worker is not None, "no FROM_WORKER msg received"
        assert from_worker["payload"]["got"] == 3
        # Subprocess should exit cleanly
        proc.join(timeout=8.0)
        assert not proc.is_alive(), "worker subprocess did not exit"
        assert proc.exitcode == 0
        # Verify result file
        result_data = json.loads(result.read_text())
        assert result_data.get("ok") is True
        assert result_data["got"] == 3
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3.0)
        observer.stop()
        broker.stop(timeout=2.0)


def _reconnecting_worker_main(sock_path: str, keypair_path: str, titan_id: str,
                              ready_path: str, result_path: str,
                              shutdown_path: str) -> None:
    """Worker that survives a broker swap. Loops reading messages until
    a SHUTDOWN message arrives; reports total received + reconnect count."""
    import json as _json
    from titan_plugin.core.worker_bus_bootstrap import setup_worker_bus
    env = {
        ENV_BUS_SOCKET_PATH: sock_path,
        ENV_BUS_TITAN_ID: titan_id,
        ENV_BUS_KEYPAIR_PATH: keypair_path,
    }
    os.environ.update(env)
    recv_q, send_q, client = setup_worker_bus("survivor", None, None, env=env)
    if client is None:
        with open(result_path, "w") as f:
            _json.dump({"error": "no_client"}, f)
        return
    if not client.wait_until_connected(timeout=5.0):
        with open(result_path, "w") as f:
            _json.dump({"error": "no_connect"}, f)
        return
    # Tell test "I'm ready"
    with open(ready_path, "w") as f:
        f.write("ready")
    received_count = 0
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            msg = recv_q.get(timeout=0.5)
        except queue.Empty:
            # Check shutdown signal
            if os.path.exists(shutdown_path):
                break
            continue
        if msg.get("type") == "PING_WORK":
            received_count += 1
        elif msg.get("type") == "DONE":
            break
    with open(result_path, "w") as f:
        _json.dump({"ok": True, "received": received_count,
                    "reconnects": client.reconnect_count}, f)
    client.stop()


def test_real_fork_worker_survives_broker_swap(tmp_path):
    """The B.2 architectural promise across a real process boundary:

    1. Boot broker; fork worker; worker connects + reports ready
    2. Send N messages — worker receives them
    3. STOP THE BROKER (simulating kernel swap)
    4. Start a NEW broker on the same socket path
    5. Worker auto-reconnects; resubscribe happens automatically
    6. Send M more messages — worker receives them
    7. Send DONE — worker exits with received=N+M and reconnects>=1

    If this works, B.2 delivers on its promise: kernels can be replaced
    under live workers without restarting them.
    """
    keypair = _make_keypair_file(tmp_path)
    sock = tmp_path / "bus.sock"
    ready = tmp_path / "ready.flag"
    result = tmp_path / "result.json"
    shutdown = tmp_path / "shutdown.flag"
    titan_id = "testT"
    from titan_plugin.core.bus_authkey import derive_bus_authkey
    secret = bytes(range(64))
    authkey = derive_bus_authkey(secret, titan_id)

    s1 = BusSocketServer(titan_id=titan_id, authkey=authkey, sock_path=sock)
    s1.start()

    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_reconnecting_worker_main,
        args=(str(sock), keypair, titan_id, str(ready), str(result),
              str(shutdown)),
        name="bus-survivor",
        daemon=True,
    )
    try:
        proc.start()
        # Wait for worker ready signal
        deadline = time.time() + 8.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.1)
        assert ready.exists(), "worker never reached ready state"

        # Phase A — send 3 messages while s1 is alive
        for i in range(3):
            s1.publish({"type": "PING_WORK", "src": "k1", "dst": "survivor",
                        "payload": {"i": i}})
        time.sleep(0.5)

        # SWAP — kill s1, start s2 on same path
        s1.stop(timeout=2.0)
        time.sleep(0.2)
        s2 = BusSocketServer(titan_id=titan_id, authkey=authkey, sock_path=sock)
        s2.start()
        try:
            # Wait for worker to reconnect (visible on broker side)
            deadline = time.time() + 8.0
            while time.time() < deadline:
                with s2._subs_lock:
                    if "survivor" in s2._subscribers:
                        break
                time.sleep(0.1)
            with s2._subs_lock:
                assert "survivor" in s2._subscribers, \
                    "worker did not reconnect to new broker"

            # Phase B — send 2 more messages
            for i in range(3, 5):
                s2.publish({"type": "PING_WORK", "src": "k2", "dst": "survivor",
                            "payload": {"i": i}})
            time.sleep(0.5)
            s2.publish({"type": "DONE", "src": "k2", "dst": "survivor",
                        "payload": {}})
            shutdown.write_text("go")  # backup signal in case DONE is dropped
        finally:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3.0)
            s2.stop(timeout=2.0)

        result_data = json.loads(result.read_text())
        assert result_data.get("ok") is True
        # Worker received both phases — proves uninterrupted operation
        assert result_data["received"] >= 5, f"only got {result_data['received']} of 5"
        # Reconnect counter advanced — proves the swap actually happened
        assert result_data["reconnects"] >= 1, \
            f"reconnect count did not advance: {result_data['reconnects']}"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3.0)
        try:
            s1.stop(timeout=1.0)
        except Exception:
            pass
