"""
Tests for titan_plugin/core/bus_authkey.py — HKDF-SHA256 bus authkey derivation.

Covers:
- Determinism (recoverability): same inputs → same output, every call
- Per-titan isolation: T1 vs T2 vs T3 produce different keys
- Identity-secret isolation: different identity_secret_bytes produce different keys
- Salt rotation: bumping BUS_AUTHKEY_SALT produces different keys (rotation works)
- Output length: always BUS_AUTHKEY_LEN (32 bytes)
- Phase C parity: fixed vectors locked for future Rust impl
- Input validation: empty/wrong-type inputs raise cleanly
- HKDF correctness: matches RFC 5869 reference vectors when used in canonical form
"""
from __future__ import annotations

import pytest

from titan_plugin.core.bus_authkey import (
    BUS_AUTHKEY_LEN,
    BUS_AUTHKEY_SALT,
    _hkdf_sha256_expand,
    _hkdf_sha256_extract,
    derive_bus_authkey,
)


# ── Determinism / recoverability ───────────────────────────────────────────


def test_derivation_is_deterministic():
    """Same identity_secret + same titan_id → same authkey. Every call. Forever.

    This is the recoverability guarantee: Shamir-restore identity → re-derive
    → workers reconnect with correct HMAC.
    """
    secret = b"identity-secret-32-bytes-exactly"
    titan_id = "titan_T1"
    assert len(secret) == 32
    a = derive_bus_authkey(secret, titan_id)
    b = derive_bus_authkey(secret, titan_id)
    assert a == b


def test_output_length_always_32():
    """Even with arbitrarily-sized inputs, output is always BUS_AUTHKEY_LEN bytes."""
    for secret_len in (1, 16, 32, 64, 128, 1024):
        key = derive_bus_authkey(b"\x42" * secret_len, "titan_T1")
        assert len(key) == BUS_AUTHKEY_LEN == 32


# ── Isolation properties ───────────────────────────────────────────────────


def test_per_titan_isolation():
    """T1, T2, T3 with the SAME identity must produce DIFFERENT authkeys
    (defense-in-depth — even if identity were ever shared, no cross-titan reuse)."""
    secret = b"identity-secret-32-bytes-exactly"
    k1 = derive_bus_authkey(secret, "titan_T1")
    k2 = derive_bus_authkey(secret, "titan_T2")
    k3 = derive_bus_authkey(secret, "titan_T3")
    assert k1 != k2
    assert k2 != k3
    assert k1 != k3


def test_identity_secret_isolation():
    """Different identity_secret bytes → different authkeys (the primary isolator)."""
    titan_id = "titan_T1"
    k_a = derive_bus_authkey(b"a" * 32, titan_id)
    k_b = derive_bus_authkey(b"b" * 32, titan_id)
    assert k_a != k_b


def test_one_byte_identity_change_propagates():
    """Single-byte change in identity → completely different authkey (avalanche)."""
    secret_a = bytearray(b"x" * 32)
    secret_b = bytearray(b"x" * 32)
    secret_b[0] = ord("y")
    k_a = derive_bus_authkey(bytes(secret_a), "titan_T1")
    k_b = derive_bus_authkey(bytes(secret_b), "titan_T1")
    assert k_a != k_b
    # Avalanche: at least 1/4 of the output bits should differ for a healthy hash
    diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(k_a, k_b))
    assert diff_bits >= 32, f"weak avalanche: only {diff_bits}/256 bits differ"


# ── Input validation ───────────────────────────────────────────────────────


def test_empty_identity_secret_raises():
    with pytest.raises(ValueError, match="empty"):
        derive_bus_authkey(b"", "titan_T1")


def test_wrong_type_identity_secret_raises():
    with pytest.raises(TypeError, match="must be bytes"):
        derive_bus_authkey("not bytes", "titan_T1")  # type: ignore[arg-type]


def test_empty_titan_id_raises():
    with pytest.raises(ValueError, match="non-empty"):
        derive_bus_authkey(b"x" * 32, "")


def test_bytearray_identity_secret_accepted():
    """bytearray (mutable) is acceptable — gets converted to bytes internally."""
    key = derive_bus_authkey(bytearray(b"x" * 32), "titan_T1")
    assert len(key) == 32


# ── Phase C parity vectors (LOCKED) ────────────────────────────────────────
# DO NOT CHANGE these expected values. Future Rust implementation MUST produce
# byte-identical output for these inputs. If a test fails after a code change,
# the protocol drifted — revert OR bump BUS_AUTHKEY_SALT to v2 simultaneously
# in Python AND Rust AND update vectors in BOTH places.


def test_parity_zero_identity_t1():
    """Vector: 32 zero bytes IKM, titan_T1 → fixed output."""
    expected = bytes.fromhex(
        "ca99e153b73f31d8e18191bb66f231a0b9985f49e1c5a0ce49b8544b368b3746"
    )
    assert derive_bus_authkey(b"\x00" * 32, "titan_T1") == expected


def test_parity_zero_identity_t2():
    """Same zero IKM, different titan_id → different output."""
    expected = bytes.fromhex(
        "ff6804d3b454631fda01e1b9a9ee2ea5f3c91bf1ac55af4bc68e89da62aa0ae3"
    )
    assert derive_bus_authkey(b"\x00" * 32, "titan_T2") == expected


def test_parity_canonical_64byte_secret_t1():
    """Canonical Titan-style 64-byte secret (Ed25519 expanded) with titan_T1."""
    secret = bytes.fromhex("a1" * 32 + "b2" * 32)
    expected = bytes.fromhex(
        "a364abf4900ab2e18a7262327574281ecfd7c1a48c07d0dacd5b5f09db2bedff"
    )
    assert derive_bus_authkey(secret, "titan_T1") == expected


def test_parity_canonical_64byte_secret_t3():
    """Same 64-byte secret with titan_T3 — different output via per-titan info."""
    secret = bytes.fromhex("a1" * 32 + "b2" * 32)
    expected = bytes.fromhex(
        "c771708f3b206071af26a420e6adac7a77d5306d57a5700731ba739c1005e923"
    )
    assert derive_bus_authkey(secret, "titan_T3") == expected


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


# ── Salt rotation works ────────────────────────────────────────────────────


def test_salt_is_v1_in_this_release():
    """Sanity: this release ships salt v1. When rotating, this assertion + the
    parity vectors above must update simultaneously."""
    assert BUS_AUTHKEY_SALT == b"titan-bus-v1"


def test_salt_rotation_changes_output():
    """If we rotated the salt to v2, the same identity would derive a different
    key — confirms rotation is effective. We test by computing both manually."""
    secret = b"x" * 32
    titan_id = "titan_T1"
    # v1 (current production salt)
    prk_v1 = _hkdf_sha256_extract(b"titan-bus-v1", secret)
    okm_v1 = _hkdf_sha256_expand(prk_v1, titan_id.encode(), 32)
    # v2 (hypothetical rotation)
    prk_v2 = _hkdf_sha256_extract(b"titan-bus-v2", secret)
    okm_v2 = _hkdf_sha256_expand(prk_v2, titan_id.encode(), 32)
    assert okm_v1 != okm_v2
    # And v1 must match the public derive_bus_authkey (sanity tie-in)
    assert okm_v1 == derive_bus_authkey(secret, titan_id)
