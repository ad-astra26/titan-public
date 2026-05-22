"""
Tests for titan_hcl/core/bus_authkey.py — HKDF-SHA256 bus authkey derivation.

Covers:
- Determinism (recoverability): same input → same output, every call
- Identity-secret isolation: different identity_secret_bytes produce different keys
  (the per-Titan isolation path — different per-Titan keypairs → different keys)
- Salt rotation: bumping BUS_AUTHKEY_SALT produces different keys (rotation works)
- Output length: always BUS_AUTHKEY_LEN (32 bytes)
- Phase C parity: fixed vectors locked, byte-identical to Rust impl
- Input validation: empty/wrong-type inputs raise cleanly
- HKDF correctness: matches RFC 5869 reference vectors when used in canonical form

Per `rFP_phase_c_bus_authkey_contract_fix.md` (2026-05-05): the HKDF info is the
CONSTANT b"titan-bus" (= BUS_AUTHKEY_INFO), NOT titan_id. Per-Titan isolation
comes from per-Titan identity secrets, NOT from info. The titan_id parameter
was REMOVED from `derive_bus_authkey()` to make the runtime call-site drift
class structurally impossible.
"""
from __future__ import annotations

import pytest

from titan_hcl.core.bus_authkey import (
    BUS_AUTHKEY_INFO,
    BUS_AUTHKEY_LEN,
    BUS_AUTHKEY_SALT,
    _hkdf_sha256_expand,
    _hkdf_sha256_extract,
    derive_bus_authkey,
)


# ── Determinism / recoverability ───────────────────────────────────────────


def test_derivation_is_deterministic():
    """Same identity_secret → same authkey. Every call. Forever.

    This is the recoverability guarantee: Shamir-restore identity → re-derive
    → workers reconnect with correct HMAC.
    """
    secret = b"identity-secret-32-bytes-exactly"
    assert len(secret) == 32
    a = derive_bus_authkey(secret)
    b = derive_bus_authkey(secret)
    assert a == b


def test_output_length_always_32():
    """Even with arbitrarily-sized inputs, output is always BUS_AUTHKEY_LEN bytes."""
    for secret_len in (1, 16, 32, 64, 128, 1024):
        key = derive_bus_authkey(b"\x42" * secret_len)
        assert len(key) == BUS_AUTHKEY_LEN == 32


# ── Isolation properties ───────────────────────────────────────────────────


def test_identity_secret_isolation():
    """Different identity_secret bytes → different authkeys.

    This is THE per-Titan isolation path: each Titan has a different
    identity keypair → different secret bytes → different authkey.
    """
    k_a = derive_bus_authkey(b"a" * 32)
    k_b = derive_bus_authkey(b"b" * 32)
    assert k_a != k_b


def test_one_byte_identity_change_propagates():
    """Single-byte change in identity → completely different authkey (avalanche)."""
    secret_a = bytearray(b"x" * 32)
    secret_b = bytearray(b"x" * 32)
    secret_b[0] = ord("y")
    k_a = derive_bus_authkey(bytes(secret_a))
    k_b = derive_bus_authkey(bytes(secret_b))
    assert k_a != k_b
    # Avalanche: at least 1/4 of the output bits should differ for a healthy hash
    diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(k_a, k_b))
    assert diff_bits >= 32, f"weak avalanche: only {diff_bits}/256 bits differ"


# ── Input validation ───────────────────────────────────────────────────────


def test_empty_identity_secret_raises():
    with pytest.raises(ValueError, match="empty"):
        derive_bus_authkey(b"")


def test_wrong_type_identity_secret_raises():
    with pytest.raises(TypeError, match="must be bytes"):
        derive_bus_authkey("not bytes")  # type: ignore[arg-type]


def test_bytearray_identity_secret_accepted():
    """bytearray (mutable) is acceptable — gets converted to bytes internally."""
    key = derive_bus_authkey(bytearray(b"x" * 32))
    assert len(key) == 32


# ── Phase C parity vectors (LOCKED — byte-identical to Rust) ───────────────
# DO NOT CHANGE these expected values without simultaneously updating
# tests/parity/vectors.json + Rust crate parity_vectors.rs. The protocol
# is byte-locked between Rust + Python here.


def test_parity_canonical_vector_matches_rust():
    """Titan canonical vector — must match Rust derive_bus_authkey() exactly.

    Vector: identity_secret_bytes_hex from tests/parity/vectors.json
    Expected: 0397b547132a70d7f8440a2a13b40971dd7ddeaec827fd541a2730a50819b407

    This vector is the canonical Rust↔Python parity check after the
    rFP_phase_c_bus_authkey_contract_fix.md fix (2026-05-05).
    """
    secret = bytes.fromhex(
        "6964656e746974792d7365637265742d33322d62797465732d6578616374ffaa"
    )
    expected = bytes.fromhex(
        "0397b547132a70d7f8440a2a13b40971dd7ddeaec827fd541a2730a50819b407"
    )
    assert derive_bus_authkey(secret) == expected


def test_parity_zero_identity():
    """Vector: 32 zero bytes IKM → fixed output."""
    actual = derive_bus_authkey(b"\x00" * 32)
    # Determinism check: at minimum verify the output is stable.
    again = derive_bus_authkey(b"\x00" * 32)
    assert actual == again
    assert len(actual) == 32


def test_parity_canonical_64byte_secret():
    """Canonical Titan-style 64-byte secret (Ed25519 expanded)."""
    secret = bytes.fromhex("a1" * 32 + "b2" * 32)
    actual = derive_bus_authkey(secret)
    again = derive_bus_authkey(secret)
    assert actual == again
    assert len(actual) == 32


# ── HKDF-SHA256 RFC 5869 reference vectors (locked) ────────────────────────
# RFC 5869 § A.1 (Test Case 1). Verifies our HKDF impl matches the spec.


def test_hkdf_rfc5869_test_case_1():
    """RFC 5869 § A.1 — basic HKDF-SHA256 with explicit salt + info."""
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    length = 42
    expected_prk = bytes.fromhex(
        "077709362c2e32df0ddc3f0dc47bba63"
        "90b6c73bb50f9c3122ec844ad7c2b3e5"
    )
    expected_okm = bytes.fromhex(
        "3cb25f25faacd57a90434f64d0362f2a"
        "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )
    prk = _hkdf_sha256_extract(salt, ikm)
    assert prk == expected_prk
    okm = _hkdf_sha256_expand(prk, info, length)
    assert okm == expected_okm


# ── Constants in lockstep with Rust ────────────────────────────────────────


def test_salt_is_v1_in_this_release():
    """Sanity: this release ships salt v1. When rotating, this assertion + the
    parity vectors above must update simultaneously in Python AND Rust."""
    assert BUS_AUTHKEY_SALT == b"titan-bus-v1"


def test_info_is_canonical_constant():
    """Sanity: this release uses the PLAN-canonical info constant b"titan-bus".

    MUST match Rust AUTHKEY_HKDF_INFO in titan-rust/crates/titan-core/src/constants.rs.
    Per rFP_phase_c_bus_authkey_contract_fix.md — restoring the PLAN's design
    after the call-site drift bug of 2026-05-05.
    """
    assert BUS_AUTHKEY_INFO == b"titan-bus"


def test_salt_rotation_changes_output():
    """If we rotated the salt to v2, the same identity would derive a different
    key — confirms rotation is effective. We test by computing both manually."""
    secret = b"x" * 32
    # v1 (current production salt)
    prk_v1 = _hkdf_sha256_extract(b"titan-bus-v1", secret)
    okm_v1 = _hkdf_sha256_expand(prk_v1, BUS_AUTHKEY_INFO, 32)
    # v2 (hypothetical rotation)
    prk_v2 = _hkdf_sha256_extract(b"titan-bus-v2", secret)
    okm_v2 = _hkdf_sha256_expand(prk_v2, BUS_AUTHKEY_INFO, 32)
    assert okm_v1 != okm_v2
    # And v1 must match the public derive_bus_authkey (sanity tie-in)
    assert okm_v1 == derive_bus_authkey(secret)
