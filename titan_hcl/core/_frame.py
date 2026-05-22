"""
_frame — Shared length-prefixed framing + HMAC primitives for Microkernel v2 IPC.

Used by:
  - kernel_rpc.py (Phase A.5 S5) — request/response RPC, api↔kernel
  - bus_socket.py (Phase B.2)    — pub/sub bus broker, kernel↔workers

The wire format is intentionally lean and Phase-C-portable: every helper here
maps directly to a Rust equivalent (RustCrypto `hmac` + `sha2` + `tokio` Unix
sockets + `rmp-serde` for msgpack). The fixed-vector parity test in
tests/test_frame_parity.py locks the protocol against silent drift between
implementations.

Frame format (every message, every direction):
  [4 bytes: little-endian uint32 length]
  [N bytes: msgpack-encoded payload]

HMAC challenge-response handshake (used at connect time by both kernel_rpc
and bus_socket):
  1. Server sends 32-byte random challenge (secrets.token_bytes(CHALLENGE_SIZE))
  2. Client sends HMAC-SHA256(authkey, challenge) (32 bytes)
  3. Server compares with constant_time_eq; closes connection on mismatch

Single-direction sendall(prefix + payload) → no torn-write window. recv_exact
loops until n bytes are read OR peer closes (returns ConnectionError).
"""
from __future__ import annotations

import hashlib
import hmac
import socket
import struct

# ── Wire-protocol constants (locked; do not change without bumping protocol) ─

CHALLENGE_SIZE = 32         # bytes — server's random nonce per connection
AUTH_TAG_SIZE = 32          # bytes — HMAC-SHA256 output (constant by hash choice)
LENGTH_PREFIX_SIZE = 4      # bytes — uint32 little-endian frame length prefix
MAX_FRAME_SIZE = 16 * 1024 * 1024  # 16 MB hard ceiling per frame

# ── Framing ────────────────────────────────────────────────────────────────


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed frame in a single sendall (no torn-write).

    Raises ValueError if payload exceeds MAX_FRAME_SIZE.
    Raises socket.error / BrokenPipeError on transport failure.
    """
    if len(payload) > MAX_FRAME_SIZE:
        raise ValueError(
            f"Frame {len(payload)}B exceeds MAX_FRAME_SIZE {MAX_FRAME_SIZE}B"
        )
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock; raises ConnectionError if peer closes early.

    Loops over recv() until buffer is full. Bytes are returned as immutable bytes
    (not bytearray) so callers can use them as dict keys or in HMAC compare.
    """
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Peer closed connection after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    """Read a length-prefixed frame. Returns the payload (length prefix stripped).

    Raises ValueError if the prefix announces a frame > MAX_FRAME_SIZE (defensive
    against a malicious or buggy peer).
    Raises ConnectionError if peer closes mid-frame.
    """
    prefix = recv_exact(sock, LENGTH_PREFIX_SIZE)
    n = struct.unpack("<I", prefix)[0]
    if n == 0:
        return b""
    if n > MAX_FRAME_SIZE:
        raise ValueError(
            f"Incoming frame {n}B exceeds MAX_FRAME_SIZE {MAX_FRAME_SIZE}B"
        )
    return recv_exact(sock, n)


# ── HMAC challenge-response ────────────────────────────────────────────────


def compute_hmac(authkey: bytes, challenge: bytes) -> bytes:
    """HMAC-SHA256 of challenge with shared authkey. Returns AUTH_TAG_SIZE bytes."""
    return hmac.new(authkey, challenge, hashlib.sha256).digest()


def constant_time_eq(a: bytes, b: bytes) -> bool:
    """Constant-time bytes equality — defensive against timing oracles in HMAC verify.

    Wraps hmac.compare_digest so callers don't need to import hmac just for this.
    Returns False (never raises) on length mismatch.
    """
    return hmac.compare_digest(a, b)


# ── Backwards-compat aliases ────────────────────────────────────────────────
# kernel_rpc.py historically used leading-underscore names. Aliases preserved
# so external imports (if any) continue to work; new code should use public names.

_send_frame = send_frame
_recv_exact = recv_exact
_recv_frame = recv_frame
_compute_hmac = compute_hmac
