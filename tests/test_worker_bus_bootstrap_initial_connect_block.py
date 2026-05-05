"""
Tests for Phase B.2 IPC §D8 outbound buffer in BusSocketClient.

Background (2026-04-30 follow-up after BUG-BUS-IPC-IN-PROCESS-ORPHAN-QUEUE
gate fix):

`BusSocketClient.start()` returns immediately; the connection thread takes
~50-150ms (jitter + handshake) to establish the socket. Light-init workers
(outer_trinity, reflex, warning_monitor, output_verifier) hit
`send_queue.put(MODULE_READY)` within 1-10ms — `_raw_send` returned False
(sock is None) and the message was silently dropped. Guardian never
transitioned worker to RUNNING.

Initial fix attempt (Option A) added `wait_until_connected(timeout=5.0)`
in `setup_worker_bus`, but that put 50-150ms boot latency on every worker
— against microkernel v2's design intent. Final fix (Option B): outbound
buffer in BusSocketClient. publish() while disconnected enqueues into a
bounded deque; _connection_loop flushes on connect. Zero added boot
latency. Same mechanism survives kernel-swap reconnects (matches §D8
intent for ANY disconnect, not just initial connect).

These tests pin the contract:
- publish() while sock is None → buffers, returns True
- _flush_outbound_buffer drains FIFO over reconnected socket
- Bounded buffer (deque maxlen=256) — overflow drops oldest
- Mid-flush connection break re-buffers unsent tail
- Buffer survives kernel-swap reconnect cycles

B.3 cleanup §11.4.a does NOT delete the buffer — kernel swaps still
need it. Only the legacy mp.Queue fallback path retires.
"""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from titan_plugin.core.bus_socket import BusSocketClient


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_client(name: str = "outer_trinity") -> BusSocketClient:
    """Build a client without starting the connection thread."""
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_bus.sock",
        topics=None,
    )


# ── Buffer behavior under disconnect ───────────────────────────────────


def test_publish_buffers_when_sock_is_none():
    """When sock is None (pre-connect or mid-reconnect), publish() must
    buffer the message rather than dropping it. Returns True so caller
    treats as success — fire-and-forget semantics same as in-process bus."""
    client = _make_client()
    assert client._sock is None  # not started

    msg = {"type": "MODULE_READY", "src": "outer_trinity", "dst": "guardian", "payload": {}}
    result = client.publish(msg)

    assert result is True, "publish must return True so worker doesn't treat as failure"
    assert len(client._outbound_buffer) == 1, "message must be buffered"
    assert client._outbound_buffer[0] is msg


def test_publish_buffer_preserves_fifo_order():
    """Buffer drains in FIFO — first publish before connect = first sent."""
    client = _make_client()
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(5)]
    for m in msgs:
        client.publish(m)

    assert list(client._outbound_buffer) == msgs


def test_publish_buffer_bounded_at_256():
    """Buffer maxlen=256; overflow drops oldest. Worker boot bursts are
    well under 256 (MODULE_READY + first heartbeat + a few state
    publishes). Any worker hitting this cap has a runaway publish bug."""
    client = _make_client()
    for i in range(300):
        client.publish({"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}})

    assert len(client._outbound_buffer) == 256, "deque maxlen=256"
    # Oldest dropped: MSG_0..MSG_43 evicted; buffer holds MSG_44..MSG_299
    assert client._outbound_buffer[0]["type"] == "MSG_44"
    assert client._outbound_buffer[-1]["type"] == "MSG_299"


# ── Flush after connect ────────────────────────────────────────────────


def test_flush_drains_buffer_over_live_socket():
    """After connect, _flush_outbound_buffer drains buffered messages
    over the now-live socket via send_frame."""
    client = _make_client()
    # Pre-populate buffer
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(3)]
    for m in msgs:
        client.publish(m)
    assert len(client._outbound_buffer) == 3

    # Inject fake live socket
    fake_sock = MagicMock()
    client._sock = fake_sock

    with patch("titan_plugin.core.bus_socket.send_frame") as mock_send:
        client._flush_outbound_buffer()

    assert mock_send.call_count == 3, "all 3 buffered messages sent"
    assert len(client._outbound_buffer) == 0, "buffer cleared after flush"


def test_flush_with_empty_buffer_is_noop():
    """Flush on empty buffer must not call send_frame at all (cheap path)."""
    client = _make_client()
    client._sock = MagicMock()

    with patch("titan_plugin.core.bus_socket.send_frame") as mock_send:
        client._flush_outbound_buffer()

    mock_send.assert_not_called()


def test_flush_when_sock_is_none_is_noop():
    """If somehow flush is called while sock is None (race), it returns
    cleanly without touching the buffer."""
    client = _make_client()
    assert client._sock is None
    client.publish({"type": "MSG", "src": "x", "dst": "y", "payload": {}})
    assert len(client._outbound_buffer) == 1

    client._flush_outbound_buffer()  # no-op

    assert len(client._outbound_buffer) == 1, "buffer untouched when sock None"


def test_flush_rebuffers_unsent_tail_on_connection_break():
    """If the socket breaks mid-flush, the unsent tail must be re-buffered
    (prepended to the front) so next reconnect retries them in order."""
    client = _make_client()
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(5)]
    for m in msgs:
        client.publish(m)

    fake_sock = MagicMock()
    client._sock = fake_sock

    # send_frame succeeds for MSG_0+MSG_1, fails on MSG_2 with ConnectionError
    call_count = [0]
    def send_side_effect(sock, payload):
        call_count[0] += 1
        if call_count[0] == 3:
            raise ConnectionError("simulated mid-flush break")

    with patch("titan_plugin.core.bus_socket.send_frame", side_effect=send_side_effect):
        client._flush_outbound_buffer()

    # Tail (MSG_2, MSG_3, MSG_4) must be re-buffered in order.
    remaining = [m["type"] for m in client._outbound_buffer]
    assert remaining == ["MSG_2", "MSG_3", "MSG_4"], (
        f"unsent tail must be re-buffered FIFO; got {remaining}"
    )


# ── _raw_send buffers on mid-send connection break ─────────────────────


def test_raw_send_buffers_on_connection_error_mid_send():
    """If sock is non-None but send_frame raises ConnectionError (broker
    died between sock-check and write), buffer the message so next
    reconnect retries. Caller still sees True (eventual delivery)."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock

    msg = {"type": "MODULE_READY", "src": "x", "dst": "guardian", "payload": {}}
    with patch(
        "titan_plugin.core.bus_socket.send_frame",
        side_effect=ConnectionError("broker dropped"),
    ):
        result = client._raw_send(msg)

    assert result is True, "buffered = success from caller's view"
    assert len(client._outbound_buffer) == 1
    assert client._outbound_buffer[0] is msg


# ── B.3 seam ────────────────────────────────────────────────────────────


def test_b3_cleanup_seam_documented():
    """Pin the comment that documents the B.3 deletion seam — buffer
    survives B.3 (kernel swaps need it); only the legacy mp.Queue
    fallback path retires."""
    import inspect

    from titan_plugin.core import bus_socket as bs

    src = inspect.getsource(bs.BusSocketClient)
    assert "B.3 cleanup" in src or "Phase B.2 IPC §D8" in src, (
        "The B.3 cleanup notes in BusSocketClient must remain. They "
        "tell a future maintainer the buffer is permanent (kernel swap "
        "uses it) while the mp.Queue fallback retires. Don't remove "
        "without updating B.3 cleanup PLAN."
    )
