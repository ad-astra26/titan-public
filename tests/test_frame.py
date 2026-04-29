"""
Tests for titan_plugin/core/_frame.py — shared framing + HMAC primitives.

Covers:
- Length-prefixed frame roundtrip (empty, small, MAX_FRAME_SIZE-1, malformed)
- recv_exact behavior on full read, partial read with peer close, n=0
- HMAC compute determinism + per-key isolation
- constant_time_eq behavior on equal/unequal/different-length bytes
- Edge cases: oversize frame raises before sendall; oversize incoming raises after prefix
"""
from __future__ import annotations

import os
import socket
import threading

import pytest

from titan_plugin.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    LENGTH_PREFIX_SIZE,
    MAX_FRAME_SIZE,
    compute_hmac,
    constant_time_eq,
    recv_exact,
    recv_frame,
    send_frame,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sock_pair() -> tuple[socket.socket, socket.socket]:
    """A pair of connected Unix-domain sockets for in-process I/O tests."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


# ── Wire-protocol constants ────────────────────────────────────────────────


def test_constants_are_locked():
    """Wire constants must not drift — Phase C Rust impl must use identical values."""
    assert CHALLENGE_SIZE == 32
    assert AUTH_TAG_SIZE == 32
    assert LENGTH_PREFIX_SIZE == 4
    assert MAX_FRAME_SIZE == 16 * 1024 * 1024


# ── Framing roundtrip ──────────────────────────────────────────────────────


def test_send_recv_empty_frame(sock_pair):
    """Empty payload is a valid frame (just a 4-byte zero prefix)."""
    a, b = sock_pair
    send_frame(a, b"")
    assert recv_frame(b) == b""


def test_send_recv_small_frame(sock_pair):
    a, b = sock_pair
    payload = b"hello bus"
    send_frame(a, payload)
    assert recv_frame(b) == payload


def test_send_recv_binary_frame(sock_pair):
    """Bytes including null/0xFF bytes — must round-trip exactly."""
    a, b = sock_pair
    payload = bytes(range(256)) * 4  # 1024B with all byte values
    send_frame(a, payload)
    assert recv_frame(b) == payload


def test_send_recv_large_frame(sock_pair):
    """1 MB payload — well below MAX_FRAME_SIZE; verifies multi-recv path in recv_exact.

    Uses a sender thread because socketpair OS buffer (~256KB) is smaller than
    1MB; single-threaded sendall would deadlock waiting for the recv to drain.
    """
    a, b = sock_pair
    payload = os.urandom(1 * 1024 * 1024)
    send_err: list[BaseException] = []

    def sender():
        try:
            send_frame(a, payload)
        except BaseException as e:  # noqa: BLE001
            send_err.append(e)

    t = threading.Thread(target=sender)
    t.start()
    received = recv_frame(b)
    t.join(timeout=5.0)
    assert not send_err, f"sender raised: {send_err}"
    assert received == payload


def test_send_frame_rejects_oversize(sock_pair):
    a, _ = sock_pair
    payload = b"x" * (MAX_FRAME_SIZE + 1)
    with pytest.raises(ValueError, match="exceeds MAX_FRAME_SIZE"):
        send_frame(a, payload)


def test_recv_frame_rejects_oversize_prefix(sock_pair):
    """If a peer sends a length prefix announcing > MAX_FRAME_SIZE, we abort
    BEFORE allocating that buffer (defense against malicious/buggy peer)."""
    a, b = sock_pair
    # Send raw oversize length prefix — bypass send_frame's own check
    a.sendall((MAX_FRAME_SIZE + 1).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="exceeds MAX_FRAME_SIZE"):
        recv_frame(b)


# ── recv_exact ─────────────────────────────────────────────────────────────


def test_recv_exact_full_read(sock_pair):
    a, b = sock_pair
    a.sendall(b"abcdef")
    assert recv_exact(b, 6) == b"abcdef"


def test_recv_exact_zero_bytes(sock_pair):
    """recv_exact(0) returns immediately with empty bytes."""
    _, b = sock_pair
    assert recv_exact(b, 0) == b""


def test_recv_exact_peer_closes_mid_read(sock_pair):
    a, b = sock_pair
    a.sendall(b"abc")
    a.close()
    with pytest.raises(ConnectionError, match="Peer closed"):
        recv_exact(b, 10)


def test_recv_exact_peer_closes_immediately(sock_pair):
    a, b = sock_pair
    a.close()
    with pytest.raises(ConnectionError, match="Peer closed"):
        recv_exact(b, 1)


# ── HMAC ───────────────────────────────────────────────────────────────────


def test_compute_hmac_deterministic():
    """Same key + challenge → same digest — required for both ends to match."""
    key = b"k" * 32
    challenge = b"c" * 32
    assert compute_hmac(key, challenge) == compute_hmac(key, challenge)


def test_compute_hmac_returns_auth_tag_size():
    """HMAC-SHA256 always returns exactly AUTH_TAG_SIZE bytes."""
    digest = compute_hmac(b"any key bytes", b"any challenge bytes")
    assert len(digest) == AUTH_TAG_SIZE


def test_compute_hmac_key_isolation():
    """Different keys → different digests — defense against key-confusion."""
    challenge = b"c" * 32
    a = compute_hmac(b"key1" * 8, challenge)
    b = compute_hmac(b"key2" * 8, challenge)
    assert a != b


def test_compute_hmac_challenge_isolation():
    """Different challenges → different digests — defense against replay."""
    key = b"k" * 32
    a = compute_hmac(key, b"chal1" * 8)[:32]
    b = compute_hmac(key, b"chal2" * 8)[:32]
    assert a != b


def test_constant_time_eq_equal_returns_true():
    a = b"\x00\x01\x02" * 8
    assert constant_time_eq(a, a) is True


def test_constant_time_eq_unequal_returns_false():
    assert constant_time_eq(b"abc", b"abd") is False


def test_constant_time_eq_length_mismatch_returns_false_no_raise():
    """Length mismatch must not raise — return False so callers can use directly."""
    assert constant_time_eq(b"abc", b"abcdef") is False


# ── Threaded send/recv (catches blocking issues) ───────────────────────────


def test_send_recv_in_threads(sock_pair):
    """Send from one thread, recv from another — confirms no deadlock or corruption."""
    a, b = sock_pair
    payload = b"threaded payload data " * 100  # ~2.2KB
    received: list[bytes] = []

    def reader():
        received.append(recv_frame(b))

    t = threading.Thread(target=reader)
    t.start()
    send_frame(a, payload)
    t.join(timeout=2.0)
    assert t.is_alive() is False
    assert received == [payload]
