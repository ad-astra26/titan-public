"""
SPEC §8.0.ter writer-thread contract tests (rFP_bus_socket_outbound_writer_thread.md).

These tests pin the behavioral guarantees of the BusSocketClient writer thread:

  1. publish() never blocks the caller's thread, even when the broker stalls.
  2. The writer thread preserves FIFO order across many publishes.
  3. flush() correctly waits for the buffer to drain.

The deadlock-impossible regression test is in
`test_bus_socket_deadlock_impossible.py`.

These tests use unit-level fakes (patched send_frame) rather than a real
BusSocketServer because the contract under test is about the CLIENT-side
publish path, independent of broker implementation. End-to-end coverage
already exists in `test_bus_socket_e2e_inproc.py`.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import msgpack

from titan_hcl.core.bus_socket import BusSocketClient


def _make_client(name: str = "test_writer") -> BusSocketClient:
    """Build a client without starting any threads — we'll inject a fake
    socket and drive the writer thread manually in some tests, or call
    start() to exercise the real path."""
    return BusSocketClient(
        titan_id="T1",
        authkey=b"\x00" * 32,
        name=name,
        sock_path="/tmp/test_bus_writer.sock",
        topics=None,
    )


def _unpack(frame: bytes) -> dict:
    return msgpack.unpackb(frame, raw=False, strict_map_key=False)


# ── Test 1: publish never blocks under backpressure ────────────────────


def test_publish_never_blocks_under_backpressure():
    """SPEC §8.0.ter performance budget: publish() returns in ≤10 µs p99
    warm cache regardless of broker drain speed.

    This test simulates a deliberately slow broker (50 ms per `send_frame`)
    and times 1000 publishes. Pre-§8.0.ter wall-time would be ~50 s
    (50 ms × 1000 serial sends). Post-§8.0.ter the publisher returns
    after enqueue; only the writer thread sees the 50 ms latency.

    Asserts: 1000 publishes complete in well under 1 s of publisher wall-time.

    Cleanup: uses a release_event so the slow_send returns immediately
    on test exit — prevents the writer thread from orphaning into the
    next test (1000 × 50ms = 50s drain otherwise).
    """
    client = _make_client()
    # Inject a fake live socket so writer thread sees sock as available.
    fake_sock = MagicMock()
    client._sock = fake_sock

    release_event = threading.Event()

    # Patch send_frame to sleep 50 ms OR until release_event is set.
    # During the test body the event stays clear; in finally we set it
    # so the writer can drain remaining frames fast and exit cleanly.
    def slow_send(sock, payload):
        # wait() returns True if event set; either way we proceed.
        release_event.wait(timeout=0.050)

    with patch("titan_hcl.core.bus_socket.send_frame", side_effect=slow_send):
        # Start writer thread; it'll pick up frames as they arrive and
        # plod through them at 20 frames/sec while we publish at full speed.
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            n = 1000
            t0 = time.monotonic()
            for i in range(n):
                client.publish({"type": f"MSG_{i}", "src": "x", "dst": "y",
                                "payload": {}})
            elapsed = time.monotonic() - t0
        finally:
            # Release the slow_send so writer can finish quickly. Then
            # signal stop and join. This prevents the writer from
            # orphaning into the next test in the suite.
            release_event.set()
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=5.0)

    # Publisher wall-time is the contract under test. Generous 1.0 s cap:
    # if publish() blocks on send_frame, this would be ~50 s.
    assert elapsed < 1.0, (
        f"SPEC §8.0.ter: 1000 publishes against 50ms-stall broker took "
        f"{elapsed:.3f}s on publisher's thread — must be < 1s "
        f"(publish() returns after enqueue, NOT after send_frame)."
    )


# ── Test 2: writer thread preserves FIFO order ─────────────────────────


def test_writer_thread_preserves_fifo_order():
    """Writer thread drains the outbound buffer in FIFO order. 10000
    sequentially-numbered messages must arrive at send_frame in the same
    order they were published."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock

    received_ids: list[int] = []
    received_lock = threading.Lock()

    def capture_send(sock, payload):
        decoded = _unpack(payload)
        with received_lock:
            received_ids.append(int(decoded["payload"]["id"]))

    with patch("titan_hcl.core.bus_socket.send_frame", side_effect=capture_send):
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            n = 10000
            for i in range(n):
                client.publish({
                    "type": "FIFO_TEST", "src": "x", "dst": "y",
                    "payload": {"id": i},
                })
            # Wait for writer to drain everything.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with received_lock:
                    if len(received_ids) >= n:
                        break
                time.sleep(0.01)
        finally:
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=2.0)

    with received_lock:
        observed = list(received_ids)

    assert len(observed) == n, (
        f"writer dropped frames: got {len(observed)} of {n}"
    )
    # FIFO: observed must equal range(n) exactly.
    assert observed == list(range(n)), (
        f"FIFO violation. First mismatch at index "
        f"{next((i for i, v in enumerate(observed) if v != i), 'n/a')}"
    )


# ── Test 3: flush() returns when drained ───────────────────────────────


def test_flush_returns_true_when_buffer_drained():
    """flush(timeout) returns True once writer thread drains the buffer."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock

    with patch("titan_hcl.core.bus_socket.send_frame") as mock_send:
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            n = 50
            for i in range(n):
                client.publish({
                    "type": "FLUSH_TEST", "src": "x", "dst": "y",
                    "payload": {"id": i},
                })
            # flush should return True within the default 5s budget.
            ok = client.flush(timeout=5.0)
        finally:
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=2.0)

    assert ok is True, "flush must return True when buffer drained"
    assert mock_send.call_count == n, (
        f"all {n} frames should have been sent by flush deadline; "
        f"got {mock_send.call_count}"
    )


def test_flush_returns_false_on_timeout_when_writer_stalls():
    """If the writer thread is stuck on a slow send_frame, flush() with
    a small timeout returns False (caller learns the buffer didn't drain).
    The buffer keeps draining in the background regardless."""
    client = _make_client()
    fake_sock = MagicMock()
    client._sock = fake_sock

    release_event = threading.Event()

    # Each send_frame takes 200 ms OR until release_event set (for cleanup).
    def slow_send(sock, payload):
        release_event.wait(timeout=0.200)

    with patch("titan_hcl.core.bus_socket.send_frame", side_effect=slow_send):
        client._writer_thread = threading.Thread(
            target=client._writer_loop, daemon=True,
            name=f"bus-writer-{client.name}")
        client._writer_thread.start()
        try:
            for i in range(5):
                client.publish({
                    "type": "STALL_TEST", "src": "x", "dst": "y",
                    "payload": {"id": i},
                })
            # 100 ms timeout, slow_send takes 200 ms each → cannot drain.
            ok = client.flush(timeout=0.1)
        finally:
            release_event.set()
            client._stop_event.set()
            client._outbound_event.set()
            client._writer_thread.join(timeout=2.0)

    assert ok is False, (
        "flush(timeout=0.1) MUST return False when writer can't drain "
        "5 × 200ms frames in time"
    )


# ── Edge case: flush with empty buffer returns immediately ─────────────


def test_flush_returns_true_immediately_when_buffer_empty():
    """flush() with no queued frames returns True on the first iteration
    — no spurious wait."""
    client = _make_client()
    t0 = time.monotonic()
    ok = client.flush(timeout=5.0)
    elapsed = time.monotonic() - t0
    assert ok is True
    # Should return in microseconds, not seconds. Generous 100 ms cap
    # to absorb scheduling jitter.
    assert elapsed < 0.1, (
        f"flush on empty buffer returned True but took {elapsed:.3f}s — "
        f"expected microseconds"
    )
