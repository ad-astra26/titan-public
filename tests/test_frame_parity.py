"""
Phase C Rust parity vectors for titan_plugin/core/_frame.py.

This file locks the wire protocol against silent drift between Python (Phase A/B)
and Rust (Phase C). Every test below pins fixed inputs to fixed outputs. When
the Rust kernel is implemented, its `tests/parity.rs` MUST produce byte-identical
results for the same inputs.

If a test in this file fails, the protocol changed — either revert the change
or bump a versioned salt/constant and update both Python and Rust simultaneously.

Why this matters: B.2's promise is "kernel can be replaced under live workers".
If a future kernel build subtly differs in framing or HMAC, workers will
disconnect with HMAC failures and never reconnect — visible to observability
but very hard to root-cause without these vectors.
"""
from __future__ import annotations

import struct

from titan_plugin.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    LENGTH_PREFIX_SIZE,
    MAX_FRAME_SIZE,
    compute_hmac,
)


# ── Locked constants (immutable; bump only with simultaneous Rust update) ──


def test_parity_constants_v1():
    """Wire protocol v1 constants. Phase C Rust impl MUST use these exact values."""
    assert CHALLENGE_SIZE == 32
    assert AUTH_TAG_SIZE == 32
    assert LENGTH_PREFIX_SIZE == 4
    assert MAX_FRAME_SIZE == 16777216  # 16 MB exactly


# ── HMAC-SHA256 fixed vectors ──────────────────────────────────────────────
# Generated with Python's hmac.new(key, msg, hashlib.sha256).digest().
# Verified against RFC 4231 § Test Case 2 + custom Titan vectors.


def test_parity_hmac_rfc_test_case_2():
    """RFC 4231 § Test Case 2 — HMAC-SHA256("Jefe", "what do ya want for nothing?")."""
    key = b"Jefe"
    msg = b"what do ya want for nothing?"
    expected = bytes.fromhex(
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    )
    assert compute_hmac(key, msg) == expected


def test_parity_hmac_rfc_test_case_1():
    """RFC 4231 § Test Case 1 — HMAC-SHA256(0x0b * 20, "Hi There")."""
    key = bytes.fromhex("0b" * 20)
    msg = b"Hi There"
    expected = bytes.fromhex(
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    )
    assert compute_hmac(key, msg) == expected


def test_parity_hmac_titan_zero_key_zero_challenge():
    """Edge case: 32-byte zero key + 32-byte zero challenge."""
    key = b"\x00" * 32
    challenge = b"\x00" * 32
    expected = bytes.fromhex(
        "33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a"
    )
    assert compute_hmac(key, challenge) == expected


def test_parity_hmac_titan_canonical():
    """A Titan-canonical vector — uses ASCII bytes that are easy to inspect.

    Both Python and Rust impls must produce this exact digest. If either side
    diverges, this test fails immediately and the divergence is visible.
    """
    key = b"titan-bus-authkey-vector-32bytes"  # 32 bytes
    challenge = b"abcdefghijklmnopqrstuvwxyz012345"  # 32 bytes
    assert len(key) == 32
    assert len(challenge) == 32
    digest = compute_hmac(key, challenge)
    expected = bytes.fromhex(
        "93de31ec6b7d38ee281759e50e47532536c8a806c173327c3f30d289a7dabe7a"
    )
    assert digest == expected
    assert len(digest) == AUTH_TAG_SIZE


# ── Length-prefix encoding fixed vectors ───────────────────────────────────
# Length prefix is 4-byte little-endian uint32. Phase C Rust impl uses
# u32::to_le_bytes() which produces identical bytes.


def test_parity_length_prefix_zero():
    """Empty frame — 4 zero bytes."""
    assert struct.pack("<I", 0) == b"\x00\x00\x00\x00"


def test_parity_length_prefix_one():
    assert struct.pack("<I", 1) == b"\x01\x00\x00\x00"


def test_parity_length_prefix_256():
    """256 = 0x100 — verifies multi-byte little-endian ordering."""
    assert struct.pack("<I", 256) == b"\x00\x01\x00\x00"


def test_parity_length_prefix_max():
    """MAX_FRAME_SIZE = 16 MB = 0x01000000 in little-endian."""
    assert struct.pack("<I", MAX_FRAME_SIZE) == b"\x00\x00\x00\x01"


# ── Documentation hint for Phase C implementer ─────────────────────────────


def test_parity_documentation_present():
    """Reminder for Phase C: this file documents the wire protocol.

    Rust impl reference (planned for Phase C):
      - hmac::Hmac<sha2::Sha256>            — HMAC-SHA256
      - u32::to_le_bytes() / from_le_bytes() — length prefix
      - tokio::net::UnixListener            — socket transport
      - rmp_serde                           — msgpack encode/decode

    All vectors above must produce byte-identical output in Rust.
    """
    # This test exists to surface the documentation in `pytest -v` output.
    assert True
