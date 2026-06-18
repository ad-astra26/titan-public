"""Hashed identity helping-signals — RFP_verifiable_autobiographical_presence_memory §7.F (F.2).

A salted, one-way HMAC over a stable identifier (Privy DID `claims.sub`, or the client
IP) used ONLY to merge "the same human across handles" during presence recall. The hash
is an INTERNAL helping-signal — it is NEVER rendered into the prompt context or any
output, NEVER displayed, and is not reversible (privacy: "a hashed helping-signal, never
a displayed fact"). The salt is per-Titan and secret, so hashes are non-reversible and
cannot be correlated across Titans.

Pure functions — the caller (the chat edge) resolves the salt and passes the raw value;
only the hash travels onward onto the bus / into any store.
"""
from __future__ import annotations

import hashlib
import hmac

# Domain-separation tags so a salt derived from one secret can't collide across uses.
_SALT_DERIVE_TAG = b"titan-presence-identity-salt-v1"
_HASH_TAG = b"titan-presence-identity-v1"


def derive_salt(secret: str) -> str:
    """Derive a stable per-Titan salt from an existing per-Titan secret (e.g. the
    `api.internal_key`). Deterministic within a Titan, secret across Titans. Empty
    secret → '' (the caller then produces empty hashes — a no-op helping-signal)."""
    if not secret:
        return ""
    return hmac.new(_SALT_DERIVE_TAG, secret.encode("utf-8"), hashlib.sha256).hexdigest()


def identity_hash(value: str, salt: str) -> str:
    """One-way salted hash of a stable identifier. Empty `value` or empty `salt`
    → '' (no signal). 32 hex chars (128-bit) — ample for a merge key, compact on
    the bus / in the Person node."""
    if not value or not salt:
        return ""
    return hmac.new(
        salt.encode("utf-8"),
        _HASH_TAG + value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]


def client_ip_from_xff(xff_header: str, direct_ip: str = "") -> str:
    """Resolve the real client IP behind nginx. Titan sits behind its OWN trusted
    nginx, so the LEFT-MOST `X-Forwarded-For` hop is the originating client; absent
    a XFF header, fall back to the direct socket peer. Trusting XFF is safe ONLY
    because the proxy is ours (do not use this for an untrusted ingress)."""
    if xff_header:
        first = xff_header.split(",")[0].strip()
        if first:
            return first
    return (direct_ip or "").strip()


__all__ = ("derive_salt", "identity_hash", "client_ip_from_xff")
