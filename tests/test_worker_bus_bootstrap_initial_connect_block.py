"""
Tests for the BusSocketClient outbound buffer.

Originally added 2026-04-30 for Phase B.2 IPC §D8 (BUG-BUS-IPC-WORKER-READY-RACE)
to pin the 256-frame bounded buffer that survived the initial-connect window.

Rewritten 2026-05-14 for SPEC §8.0.ter (rFP_bus_socket_outbound_writer_thread.md
v1.6.0): the outbound buffer is now the canonical write path for ALL publishes,
not just the disconnect window. `_raw_send` packs the frame synchronously
(fail-fast validation per rFP_bus_payload_contracts §4) then appends the
**packed bytes** to the buffer. The dedicated writer thread is the sole
`send_frame()` caller. Buffer is unbounded (§8.0 P0 never-drop); high-water
warning fires at OUTBOUND_BUFFER_HIGH_WATER per Chunk 4 of the rFP.

This test file pins the storage contract (bytes in buffer) + drain behavior.
The writer-thread/non-blocking guarantees are pinned by the new test files
in Chunk 3: `test_bus_socket_writer_thread.py` and
`test_bus_socket_deadlock_impossible.py`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack

from titan_hcl.core.bus_socket import (
    BusSocketClient,
    OUTBOUND_BUFFER_HIGH_WATER,
)


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


def _unpack(frame: bytes) -> dict:
    """Decode a buffered frame back to its dict form for assertions."""
    return msgpack.unpackb(frame, raw=False, strict_map_key=False)


# ── Buffer behavior under disconnect ───────────────────────────────────


def test_publish_buffers_when_sock_is_none():
    """When sock is None (pre-connect or mid-reconnect), publish() must
    buffer the message rather than dropping it. Returns True so caller
    treats as success — fire-and-forget semantics same as in-process bus.

    Per SPEC §8.0.ter, publish() ALWAYS enqueues (regardless of sock state)
    — the writer thread is the sole socket-toucher."""
    client = _make_client()
    assert client._sock is None  # not started

    msg = {"type": "MODULE_READY", "src": "outer_trinity", "dst": "guardian", "payload": {}}
    result = client.publish(msg)

    assert result is True, "publish must return True so worker doesn't treat as failure"
    assert len(client._outbound_buffer) == 1, "message must be buffered"
    assert _unpack(client._outbound_buffer[0])["type"] == "MODULE_READY"


def test_publish_buffer_preserves_fifo_order():
    """Buffer drains in FIFO — first publish before connect = first sent.
    Order preserved across pack → enqueue → drain cycle."""
    client = _make_client()
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(5)]
    for m in msgs:
        client.publish(m)

    buffered_types = [_unpack(b)["type"] for b in client._outbound_buffer]
    expected_types = [m["type"] for m in msgs]
    assert buffered_types == expected_types


def test_publish_buffer_is_unbounded_per_spec_8_0_ter():
    """Buffer is UNBOUNDED per SPEC §8.0 P0 never-drop guarantee — every
    enqueued frame is preserved until drained. Backpressure surfaces as
    rate-limited high-water WARN (see Chunk 4), not as a silent drop.

    Pre-§8.0.ter the buffer was deque(maxlen=256) which would silently
    drop oldest frames on overflow — a P0 violation per §8.0.
    """
    client = _make_client()
    # Push well past the old maxlen=256 to prove no drops.
    n = 1500
    for i in range(n):
        client.publish({"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}})

    assert len(client._outbound_buffer) == n, (
        f"buffer must keep all {n} frames; got {len(client._outbound_buffer)}"
    )
    # First + last present in FIFO order — no eviction.
    assert _unpack(client._outbound_buffer[0])["type"] == "MSG_0"
    assert _unpack(client._outbound_buffer[-1])["type"] == f"MSG_{n - 1}"


def test_publish_high_water_warn_fires_after_threshold(caplog):
    """Once depth crosses OUTBOUND_BUFFER_HIGH_WATER (1000), a rate-limited
    WARN per the SPEC §8.0.ter spec is emitted. This is operator-visible
    backpressure signal — the producing module is named in the log line.
    """
    import logging
    client = _make_client(name="testworker")
    caplog.set_level(logging.WARNING, logger="titan_hcl.core.bus_socket")
    for i in range(OUTBOUND_BUFFER_HIGH_WATER + 5):
        client.publish({"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}})

    matches = [
        r for r in caplog.records
        if "outbound buffer high water" in r.getMessage()
           and "testworker" in r.getMessage()
    ]
    assert matches, "expected at least one high-water WARN naming the client"


# ── Flush after connect ────────────────────────────────────────────────


def test_flush_drains_buffer_over_live_socket():
    """After connect, _flush_outbound_buffer drains buffered messages
    over the now-live socket via send_frame.

    Post-§8.0.ter the buffer holds pre-packed bytes; flush is pure I/O
    (no re-pack)."""
    client = _make_client()
    # Pre-populate buffer through the canonical publish() path.
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(3)]
    for m in msgs:
        client.publish(m)
    assert len(client._outbound_buffer) == 3

    # Inject fake live socket.
    fake_sock = MagicMock()
    client._sock = fake_sock

    with patch("titan_hcl.core.bus_socket.send_frame") as mock_send:
        client._flush_outbound_buffer()

    assert mock_send.call_count == 3, "all 3 buffered messages sent"
    assert len(client._outbound_buffer) == 0, "buffer cleared after flush"

    # send_frame called with pre-packed bytes (verify via decode round-trip).
    sent_types = [
        _unpack(call.args[1])["type"]
        for call in mock_send.call_args_list
    ]
    assert sent_types == [m["type"] for m in msgs], "FIFO preserved through drain"


def test_flush_with_empty_buffer_is_noop():
    """Flush on empty buffer must not call send_frame at all (cheap path)."""
    client = _make_client()
    client._sock = MagicMock()

    with patch("titan_hcl.core.bus_socket.send_frame") as mock_send:
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
    (prepended to the front) so next reconnect retries them in order.

    Per §8.0.ter the buffer holds packed bytes; the re-prepend logic
    operates on bytes, not dicts, but FIFO order is identical."""
    client = _make_client()
    msgs = [{"type": f"MSG_{i}", "src": "x", "dst": "y", "payload": {}} for i in range(5)]
    for m in msgs:
        client.publish(m)

    fake_sock = MagicMock()
    client._sock = fake_sock

    # send_frame succeeds for MSG_0+MSG_1, fails on MSG_2 with ConnectionError.
    call_count = [0]
    def send_side_effect(sock, payload):
        call_count[0] += 1
        if call_count[0] == 3:
            raise ConnectionError("simulated mid-flush break")

    with patch("titan_hcl.core.bus_socket.send_frame", side_effect=send_side_effect):
        client._flush_outbound_buffer()

    # Tail (MSG_2, MSG_3, MSG_4) must be re-buffered in order.
    remaining = [_unpack(b)["type"] for b in client._outbound_buffer]
    assert remaining == ["MSG_2", "MSG_3", "MSG_4"], (
        f"unsent tail must be re-buffered FIFO; got {remaining}"
    )


# ── _raw_send buffers on every publish (post-§8.0.ter — no mid-send branch) ──


def test_raw_send_always_enqueues_never_calls_send_frame():
    """Post-§8.0.ter, `_raw_send` MUST NOT call `send_frame` directly under
    any circumstance — that's the writer thread's sole responsibility.

    Pre-§8.0.ter, `_raw_send` checked sock and called send_frame inline
    when sock was non-None. Now it always enqueues + signals the writer."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock  # even with a live sock, _raw_send must not write

    msg = {"type": "MODULE_READY", "src": "x", "dst": "guardian", "payload": {}}
    with patch("titan_hcl.core.bus_socket.send_frame") as mock_send:
        result = client._raw_send(msg)

    assert result is True, "enqueue succeeded"
    assert mock_send.call_count == 0, (
        "SPEC §8.0.ter: _raw_send MUST NOT touch the socket. "
        "Writer thread is the sole send_frame caller."
    )
    assert len(client._outbound_buffer) == 1
    assert _unpack(client._outbound_buffer[0])["type"] == "MODULE_READY"


def test_raw_send_signals_outbound_event():
    """Every successful enqueue must signal _outbound_event so the writer
    thread wakes promptly (microsecond wake latency)."""
    client = _make_client()
    client._outbound_event.clear()  # start cleared
    assert not client._outbound_event.is_set()

    client.publish({"type": "MSG", "src": "x", "dst": "y", "payload": {}})

    assert client._outbound_event.is_set(), (
        "publish() must set _outbound_event so the writer thread wakes"
    )


def test_raw_send_returns_false_and_does_not_enqueue_on_validation_failure():
    """If `_packb_safe` raises (unencodable payload / contract violation),
    `_raw_send` must return False AND not enqueue. This is the fail-fast
    accountability point per rFP_bus_payload_contracts §4: producer learns
    about the violation; bad frame never enters the buffer."""
    client = _make_client()
    msg = {"type": "TEST", "src": "x", "dst": "y", "payload": {}}

    with patch(
        "titan_hcl.core.bus_socket._packb_safe",
        side_effect=TypeError("unencodable"),
    ):
        result = client._raw_send(msg)

    assert result is False, "validation failure → False return"
    assert len(client._outbound_buffer) == 0, (
        "bad frame must NOT be enqueued; writer would just drop it later"
    )
