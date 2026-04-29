"""
bus_authkey — HKDF-SHA256 derivation of the bus authkey from Titan's identity.

Microkernel v2 Phase B.2 §D1 (Maker-locked 2026-04-27): the bus authkey is
NOT a separate persistent secret on disk. It is derived deterministically at
every kernel boot from Titan's existing Ed25519 identity keypair (the soul) via
HKDF-SHA256. Properties:

  • Recoverable — Shamir-restore the identity keypair → same secret_bytes →
    same authkey. No second secret to lose.
  • Secure in-flight — never crosses the wire. HMAC challenge-response per
    connect (32B random challenge, replay-proof). Key derived locally on both
    ends from material both ends already possess.
  • Per-titan isolated — T1, T2, T3 have different identity keys → different
    authkeys → no cross-titan listener can ever HMAC-validate.
  • Rotation — bump BUS_AUTHKEY_SALT to b"titan-bus-v2" + restart Titan.
    No migration drama; one-line ops change.
  • Phase C portable — RustCrypto `hkdf` crate produces byte-identical output
    for the same inputs. Locked by tests/test_bus_authkey.py fixed vectors.

Distinct from kernel_rpc's per-boot ephemeral authkey:
  - kernel_rpc.generate_authkey() — random 32 bytes via secrets.token_bytes,
    written to /tmp/titan_kernel_<id>.authkey (chmod 0600), single client
    (api_subprocess) on localhost. Different threat model; ephemerality is
    correct there.
  - bus_authkey.derive_bus_authkey() — derived deterministically; survives
    kernel swaps because workers must reconnect to a kernel restart with the
    same key (B.2's whole point).

This module implements HKDF-SHA256 (RFC 5869) directly using stdlib hmac +
hashlib. We do NOT depend on `cryptography` or other external libs — keeps
L0 lean (Maker priority for mobile portability) and avoids resolving an
import that doesn't exist before titan_main is ready.
"""
from __future__ import annotations

import hashlib
import hmac

# ── Locked constants — change only with simultaneous Rust + parity test bump ─

BUS_AUTHKEY_SALT = b"titan-bus-v1"   # version-bumpable for rotation without identity touch
BUS_AUTHKEY_LEN = 32                 # 256-bit HMAC key (matches AUTH_TAG_SIZE in _frame)


# ── HKDF-SHA256 (RFC 5869) ─────────────────────────────────────────────────


def _hkdf_sha256_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract: PRK = HMAC-SHA256(salt, IKM). Output is 32 bytes."""
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_sha256_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Expand: T(N) = HMAC-SHA256(PRK, T(N-1) || info || N).

    T(0) = empty. Output is concatenation of T(1)..T(ceil(L/HashLen)), truncated to L.
    For Titan's L=32 case, this is a single block (HashLen=32), so the loop
    iterates once — kept generic for clarity / Rust parity.
    """
    if length > 255 * 32:
        raise ValueError("HKDF-Expand length exceeds 255*HashLen")
    out = bytearray()
    t_prev = b""
    counter = 1
    while len(out) < length:
        t_prev = hmac.new(
            prk, t_prev + info + bytes([counter]), hashlib.sha256
        ).digest()
        out.extend(t_prev)
        counter += 1
    return bytes(out[:length])


# ── Public API ─────────────────────────────────────────────────────────────


def derive_bus_authkey(identity_secret_bytes: bytes, titan_id: str) -> bytes:
    """Derive the bus HMAC authkey from Titan's identity keypair via HKDF-SHA256.

    Inputs:
      identity_secret_bytes — Ed25519 secret_bytes (typically 32 or 64 bytes
        depending on how the keypair is stored; HKDF accepts arbitrary length).
        Must be the IKM, not a public key.
      titan_id — string identity ("titan_T1", "titan_T2", "titan_T3"...) used
        as HKDF info. Encoded UTF-8 NFC; canonical form is plain ASCII so
        no normalization issue.

    Returns BUS_AUTHKEY_LEN bytes (32). Deterministic for fixed inputs +
    fixed BUS_AUTHKEY_SALT. Recoverable across kernel swaps. Survives Shamir
    restore as long as the identity keypair is recovered.
    """
    if not isinstance(identity_secret_bytes, (bytes, bytearray)):
        raise TypeError(
            f"identity_secret_bytes must be bytes, got {type(identity_secret_bytes).__name__}"
        )
    if len(identity_secret_bytes) == 0:
        raise ValueError("identity_secret_bytes is empty — cannot derive authkey")
    if not titan_id:
        raise ValueError("titan_id must be non-empty")
    info = titan_id.encode("utf-8")
    prk = _hkdf_sha256_extract(salt=BUS_AUTHKEY_SALT, ikm=bytes(identity_secret_bytes))
    return _hkdf_sha256_expand(prk=prk, info=info, length=BUS_AUTHKEY_LEN)
