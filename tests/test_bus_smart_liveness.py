"""Tests for the Phase B.2 §D9 smart-liveness algorithm (2026-05-02).

Covers:
- _pid_alive helper (Tier 2 OS oracle)
- SO_PEERCRED capture at accept time
- Stats counter init + split (peer_dead vs pong_timeout)
- BUS_PEER_DIED bus message constant + spec entry
- Guardian._process_guardian_messages BUS_PEER_DIED handler
  (named → restart, was_anon → log, unknown → log)

Live tier-1-to-4 dispatch is exercised in the existing
`test_broker_sends_periodic_ping` + integration tests; the smart-liveness
behavior is verified per-tier by manipulating subscriber state directly.

See `BUG-BUS-PING-PONG-TIGHT-TIMEOUT-VS-HEAVY-WORKER-INIT-20260502`
for the bug this closes.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from titan_plugin import bus
from titan_plugin.bus_specs import MSG_SPECS
from titan_plugin.core.bus_socket import (
    ANON_SUBSCRIBE_GRACE_S,
    PING_INTERVAL_S,
    PING_TIMEOUT_S,
    BrokerSubscriber,
    BoundedRing,
    BusSocketServer,
)


# ── _pid_alive helper ─────────────────────────────────────────────────────


def test_pid_alive_returns_true_for_self():
    """Current process is alive."""
    assert BusSocketServer._pid_alive(os.getpid()) is True


def test_pid_alive_returns_false_for_dead_pid():
    """A definitely-not-running PID returns False."""
    # PID 0 is a special placeholder; PID 999999 is essentially never used
    # on a fresh system. Try a few high values to find one that isn't alive.
    for pid in (999999, 999998, 999997):
        try:
            os.kill(pid, 0)
            continue  # actually alive (rare); try next
        except (ProcessLookupError, PermissionError, OSError):
            assert BusSocketServer._pid_alive(pid) is False
            return
    pytest.skip("Couldn't find an unused high PID — system saturated")


def test_pid_alive_returns_true_when_pid_is_none():
    """None peer_pid (e.g., non-Linux) degrades safely to True."""
    assert BusSocketServer._pid_alive(None) is True


# ── BUS_PEER_DIED constant + spec ─────────────────────────────────────────


def test_bus_peer_died_constant_defined():
    """The bus message constant is exposed."""
    assert hasattr(bus, "BUS_PEER_DIED")
    assert bus.BUS_PEER_DIED == "BUS_PEER_DIED"


def test_bus_peer_died_spec_priority_zero():
    """Phase B.2 §D9 — kernel-critical, never drop."""
    spec = MSG_SPECS.get("BUS_PEER_DIED")
    assert spec is not None, "BUS_PEER_DIED must have a bus_specs entry"
    assert spec.priority == 0


# ── BrokerSubscriber gained peer_pid + last_recv_ts fields ────────────────


def test_broker_subscriber_has_peer_pid_field():
    """Phase B.2 §D9 — peer_pid populated from SO_PEERCRED at accept."""
    sub = BrokerSubscriber(
        name="test", conn=MagicMock(), addr="anon-1", ring=BoundedRing(),
    )
    assert hasattr(sub, "peer_pid")
    assert sub.peer_pid is None  # default until populated


def test_broker_subscriber_has_last_recv_ts_field():
    """last_recv_ts tracks any-frame recency for tier-1 silence horizon."""
    sub = BrokerSubscriber(
        name="test", conn=MagicMock(), addr="anon-1", ring=BoundedRing(),
    )
    assert hasattr(sub, "last_recv_ts")
    # Default factory: time.time() — should be close to now
    assert abs(sub.last_recv_ts - time.time()) < 1.0


# ── Stats counters ────────────────────────────────────────────────────────


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


def test_broker_stats_init_with_split_counters(server):
    """Phase B.2 §D9 — peer_dead + pong_timeout counters initialized to 0."""
    stats = server.stats()
    assert stats["peer_dead_purges_total"] == 0
    assert stats["pong_timeout_purges_total"] == 0


def test_broker_stats_includes_peer_pid_per_subscriber(tmp_path, authkey):
    """Phase B.2 §D9 — per-subscriber peer_pid surfaces in stats() for arch_map."""
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s.start()
    try:
        # Connect a real Unix socket so SO_PEERCRED gives a real pid
        from tests.test_bus_socket_server import _connect_and_handshake
        client = _connect_and_handshake(sock, authkey, subscribe_as="probe")
        # Wait for subscriber to appear
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with s._subs_lock:
                if "probe" in s._subscribers:
                    break
            time.sleep(0.02)
        stats = s.stats()
        probe_subs = [x for x in stats["subscribers"] if x["name"] == "probe"]
        assert len(probe_subs) == 1
        assert "peer_pid" in probe_subs[0]
        # peer_pid should be the test process itself (we connected from same proc)
        assert probe_subs[0]["peer_pid"] == os.getpid()
        client.close()
    finally:
        s.stop(timeout=2.0)


# ── Tier 2 — peer-dead detection + BUS_PEER_DIED publish ──────────────────


def test_purge_peer_dead_publishes_bus_peer_died(tmp_path, authkey):
    """Phase B.2 §D9 — when broker purges with reason='peer_dead', it
    publishes BUS_PEER_DIED via on_inbound_publish so Guardian can react."""
    sock = tmp_path / "bus.sock"
    captured: list[dict] = []

    def fake_cb(msg: dict) -> None:
        captured.append(msg)

    s = BusSocketServer(
        titan_id="testT", authkey=authkey, sock_path=sock,
        on_inbound_publish=fake_cb,
    )
    s.start()
    try:
        # Build a fake subscriber + manually invoke _purge_subscriber with
        # log_reason="peer_dead" — bypasses the live ping loop. Tests that
        # the publish happens unconditionally on this reason.
        fake_conn = MagicMock()
        sub = BrokerSubscriber(
            name="memory", conn=fake_conn, addr="anon-7",
            ring=BoundedRing(), peer_pid=98765,
        )
        with s._subs_lock:
            s._subscribers["memory"] = sub
        s._purge_subscriber(sub, log_reason="peer_dead", silent_for=22.5)

        # Find the BUS_PEER_DIED message in captured cb events
        peer_died_msgs = [m for m in captured if m.get("type") == "BUS_PEER_DIED"]
        assert len(peer_died_msgs) == 1, f"expected 1 BUS_PEER_DIED, got {captured}"
        msg = peer_died_msgs[0]
        assert msg["src"] == "broker"
        assert msg["dst"] == "guardian"
        payload = msg["payload"]
        assert payload["name"] == "memory"
        assert payload["pid"] == 98765
        assert payload["was_anon"] is False
        assert payload["silent_for_s"] == 22.5
    finally:
        s.stop(timeout=2.0)


def test_purge_peer_dead_increments_counter_only(tmp_path, authkey):
    """Phase B.2 §D9 — peer_dead reason increments peer_dead_purges_total
    but NOT pong_timeout_purges_total."""
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s.start()
    try:
        fake_conn = MagicMock()
        sub = BrokerSubscriber(
            name="rl", conn=fake_conn, addr="anon-1",
            ring=BoundedRing(), peer_pid=11111,
        )
        with s._subs_lock:
            s._subscribers["rl"] = sub
        before = s.stats()
        s._purge_subscriber(sub, log_reason="peer_dead", silent_for=10.0)
        after = s.stats()
        assert after["peer_dead_purges_total"] == before["peer_dead_purges_total"] + 1
        assert after["pong_timeout_purges_total"] == before["pong_timeout_purges_total"]
    finally:
        s.stop(timeout=2.0)


def test_purge_pong_timeout_increments_separate_counter(tmp_path, authkey):
    """Phase B.2 §D9 — pong_timeout reason increments its own counter."""
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(titan_id="testT", authkey=authkey, sock_path=sock)
    s.start()
    try:
        fake_conn = MagicMock()
        sub = BrokerSubscriber(
            name="cgn", conn=fake_conn, addr="anon-2",
            ring=BoundedRing(), peer_pid=22222,
        )
        with s._subs_lock:
            s._subscribers["cgn"] = sub
        before = s.stats()
        s._purge_subscriber(sub, log_reason="pong_timeout", silent_for=15.0)
        after = s.stats()
        assert after["pong_timeout_purges_total"] == before["pong_timeout_purges_total"] + 1
        assert after["peer_dead_purges_total"] == before["peer_dead_purges_total"]
    finally:
        s.stop(timeout=2.0)


def test_purge_peer_dead_no_publish_when_no_callback(tmp_path, authkey):
    """When on_inbound_publish=None (test fixtures), BUS_PEER_DIED is
    skipped — broker remains usable in isolation."""
    sock = tmp_path / "bus.sock"
    s = BusSocketServer(
        titan_id="testT", authkey=authkey, sock_path=sock,
        on_inbound_publish=None,
    )
    s.start()
    try:
        fake_conn = MagicMock()
        sub = BrokerSubscriber(
            name="spirit", conn=fake_conn, addr="anon-3",
            ring=BoundedRing(), peer_pid=33333,
        )
        with s._subs_lock:
            s._subscribers["spirit"] = sub
        # Should not raise
        s._purge_subscriber(sub, log_reason="peer_dead", silent_for=99.9)
        # Counter still increments
        assert s.stats()["peer_dead_purges_total"] >= 1
    finally:
        s.stop(timeout=2.0)


# ── Constants are present + sane ──────────────────────────────────────────


def test_anon_subscribe_grace_is_substantially_higher_than_ping_timeout():
    """ANON_SUBSCRIBE_GRACE_S must give heavy-init workers room. Bound at
    120s (2026-05-02 — bumped from 60s after Stage 1 deployed; absorbs
    GIL-pressure variance for the slowest pre-BUS_SUBSCRIBE code path while
    still aggressive enough to close truly-stuck anon connections."""
    assert ANON_SUBSCRIBE_GRACE_S >= 60.0, \
        "anon grace too tight; will kill heavy-init workers again"
    assert ANON_SUBSCRIBE_GRACE_S <= 300.0, \
        "anon grace too loose; would let stuck connections accumulate"
    # Tier-3 only fires past PING_TIMEOUT_S, so anon grace must be > that
    assert ANON_SUBSCRIBE_GRACE_S > PING_TIMEOUT_S


# ── Guardian BUS_PEER_DIED handler ────────────────────────────────────────


def test_guardian_handles_bus_peer_died_named_worker_triggers_restart():
    """Guardian receives BUS_PEER_DIED for a known + restartable module → restart()."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    g = Guardian(DivineBus())
    g.register(ModuleSpec(
        name="memory", layer="L2", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    ))
    # Mock restart so we don't actually try to spawn
    g.restart = MagicMock(return_value=True)

    # Inject a BUS_PEER_DIED into the guardian queue
    msg = {
        "type": bus.BUS_PEER_DIED, "src": "broker", "dst": "guardian",
        "payload": {"name": "memory", "pid": 12345,
                    "was_anon": False, "silent_for_s": 18.5},
    }
    g._guardian_queue.put_nowait(msg)
    g._process_guardian_messages()

    g.restart.assert_called_once()
    call_kwargs = g.restart.call_args
    assert call_kwargs.args[0] == "memory"
    assert call_kwargs.kwargs.get("reason") == "broker_peer_dead"


def test_guardian_handles_bus_peer_died_was_anon_logs_only():
    """Guardian receives BUS_PEER_DIED with was_anon=True → no restart;
    polling-path will discover the actual module within ~1s."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian

    g = Guardian(DivineBus())
    g.restart = MagicMock(return_value=True)

    msg = {
        "type": bus.BUS_PEER_DIED, "src": "broker", "dst": "guardian",
        "payload": {"name": "anon-17", "pid": 54321,
                    "was_anon": True, "silent_for_s": 30.0},
    }
    g._guardian_queue.put_nowait(msg)
    g._process_guardian_messages()

    g.restart.assert_not_called()


def test_guardian_handles_bus_peer_died_unknown_module_logs_only():
    """Guardian receives BUS_PEER_DIED for a module it doesn't know → log + no-op."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian

    g = Guardian(DivineBus())
    g.restart = MagicMock(return_value=True)

    msg = {
        "type": bus.BUS_PEER_DIED, "src": "broker", "dst": "guardian",
        "payload": {"name": "nonexistent_worker", "pid": 99999,
                    "was_anon": False, "silent_for_s": 20.0},
    }
    g._guardian_queue.put_nowait(msg)
    g._process_guardian_messages()

    g.restart.assert_not_called()


def test_guardian_skips_restart_when_restart_on_crash_false():
    """Module with restart_on_crash=False stays stopped on BUS_PEER_DIED."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    g = Guardian(DivineBus())
    g.register(ModuleSpec(
        name="oneshot", layer="L3", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=False,
    ))
    g.restart = MagicMock(return_value=True)

    msg = {
        "type": bus.BUS_PEER_DIED, "src": "broker", "dst": "guardian",
        "payload": {"name": "oneshot", "pid": 11111,
                    "was_anon": False, "silent_for_s": 5.0},
    }
    g._guardian_queue.put_nowait(msg)
    g._process_guardian_messages()

    g.restart.assert_not_called()
