"""Per-Titan user_id hashing for conversation-fork TX tagging.

Closes the Phase 2 chat-pipeline drift from `ARCHITECTURE_synthesis_engine.md
§7` (PLAN_synthesis_engine_Phase2.md §B closure, 2026-05-25). Arch §7
mandates conversation-fork TX tags include `"user:<hash>"`; the chat
pipeline previously tagged TXs with `["verified_output", channel]` only.
This module provides the canonical per-Titan-salted hash used to mint
that tag (and the `content.user_id_hash` field per arch §7 normative
content shape).

Privacy + sovereignty axiom (rFP §20 Q6 + `feedback_sovereignty_axiom_no_infra_dependency`):
the salt is **per-Titan**, persisted in `~/.titan/secrets.toml [synthesis]
user_id_hash_salt` (chmod 600). Same user chatting with T1 vs T3 produces
DIFFERENT `user:<hash>` tags — no cross-Titan correlation is possible
without each Titan's private salt. The salt never leaves the Titan's
filesystem; it is regenerated only by explicit Maker rotation.

Hash construction: `sha256(salt_bytes || b":" || user_id.encode("utf-8"))`
truncated to 16 hex chars (64-bit). 64-bit truncation is sufficient for
the per-user-bundle key space (10^6 users → ~10^-7 collision via birthday
bound; bundles are per-Titan-scoped so the bound is tight). The output
form is `"user:<16-hex>"` to match the arch §7 tag literal.

INV-Syn-3 holds — this module is callable from any process. The salt is
read-only from this module's perspective (only the first-call generator
writes, atomically via `config_loader.update_secret`). All subsequent
loads cache the salt in-process so the hot path on chat-TX construction
is a single sha256 (~microsecond) with no I/O.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
from typing import Optional
from titan_hcl.params import load_titan_params

logger = logging.getLogger(__name__)

# Hash truncation: 16 hex chars = 64 bits. Per arch §20 Q6 the trade is
# privacy (shorter = harder to fingerprint back to user_id without salt)
# vs collision (longer = lower). 64 bits balances both for typical
# per-Titan user populations.
_HASH_LEN_HEX = 16

# Tag literal prefix — arch §7 normative.
_USER_TAG_PREFIX = "user:"

# Process-local cache of the salt bytes. First call resolves from
# secrets.toml or generates atomically; subsequent calls hit the cache.
# Reset via `clear_cache()` (tests / rotation).
_salt_cache: Optional[bytes] = None
_salt_lock = threading.Lock()


def _generate_salt() -> str:
    """Generate a fresh 32-byte salt as hex (64 chars). 32 bytes = 256
    bits of entropy — far more than the 64-bit hash truncation needs.
    Persists at first call ever and remains constant for the Titan's
    lifetime."""
    return secrets.token_hex(32)


def get_user_id_hash_salt() -> bytes:
    """Return the per-Titan salt as bytes.

    Resolution order:
      1. In-process cache.
      2. `~/.titan/secrets.toml [synthesis] user_id_hash_salt`.
      3. Generate fresh (secrets.token_hex(32)) + persist atomically via
         `config_loader.update_secret`, then return.

    Step 3 is the one-time bootstrap on a fresh Titan; subsequent boots
    hit step 2 on first call.
    """
    global _salt_cache
    if _salt_cache is not None:
        return _salt_cache

    with _salt_lock:
        # Double-check under lock — another thread may have populated.
        if _salt_cache is not None:
            return _salt_cache

        from titan_hcl.params import update_secret

        cfg = load_titan_params() or {}
        synth_section = cfg.get("synthesis") or {}
        salt_hex = synth_section.get("user_id_hash_salt", "")

        if not isinstance(salt_hex, str) or len(salt_hex) < 32:
            # Bootstrap: generate + persist. Persistence is best-effort —
            # any failure (missing tomli_w dep, file perms, disk, etc.) is
            # caught here so OVG's build_timechain_payload NEVER raises
            # because of a salt-persistence problem. In-process salt still
            # mints valid bundles for THIS session; only continuity across
            # restart is lost. Maker-visible WARNING surfaces the issue.
            salt_hex = _generate_salt()
            ok = False
            try:
                ok = update_secret(
                    "synthesis", "user_id_hash_salt", salt_hex)
            except Exception as persist_err:
                logger.warning(
                    "[user_id_hash] update_secret raised %s: %s — "
                    "falling back to process-local salt (will regenerate "
                    "next restart, breaking bundle continuity)",
                    type(persist_err).__name__, persist_err)
            if not ok:
                logger.warning(
                    "[user_id_hash] failed to persist user_id_hash_salt "
                    "to ~/.titan/secrets.toml — using process-local salt "
                    "(will REGENERATE next restart, breaking bundle "
                    "continuity). Check tomli_w installed + secrets file "
                    "permissions.")
            else:
                logger.info(
                    "[user_id_hash] generated + persisted new per-Titan "
                    "user_id_hash_salt (32B hex) in "
                    "~/.titan/secrets.toml [synthesis]")

        try:
            _salt_cache = bytes.fromhex(salt_hex)
        except ValueError:
            # FAIL-CLOSED (AUDIT §5.3): an EXISTING-but-corrupt salt must NOT be
            # silently rotated + persisted — that permanently orphans every
            # prior user:<hash> bundle key (a sovereignty/continuity foot-gun,
            # and a transient corruption would reset the keyspace forever).
            # Alarm loudly and use a DETERMINISTIC, NON-persisted process-local
            # salt this run so a human can restore the real salt in
            # ~/.titan/secrets.toml without the keyspace having been rewritten.
            logger.error(
                "[user_id_hash] secrets.toml [synthesis].user_id_hash_salt is "
                "CORRUPT (not valid hex) — NOT auto-rotating (that would orphan "
                "every prior user bundle). Using a non-persisted process-local "
                "salt this run; restore the real salt to recover continuity.")
            _salt_cache = hashlib.sha256(
                ("corrupt-salt-fallback:" + salt_hex).encode("utf-8")).digest()

        return _salt_cache


def hash_user_id(user_id: str) -> str:
    """Return the canonical `"user:<16-hex>"` tag value for a user_id.

    Empty / None / "anonymous" → returns empty string (caller should skip
    adding the tag). The `"anonymous"` short-circuit prevents minting a
    `user:<hash>` tag for genuinely anonymous traffic where the bundle
    would aggregate every anon visitor under one key — not useful, and
    arguably misleading per arch §7 ("the hashed user IDENTIFIER").

    Args:
        user_id: Raw user identifier (Privy sub, "maker", channel-synth
                 like "pitch-visitor-<sha>", etc.). Treated as utf-8.

    Returns:
        Tag string `"user:<16-hex>"` ready to drop into TX tags list, OR
        empty string for anonymous / missing user_id (caller must check
        truthiness before appending).
    """
    if not user_id or user_id == "anonymous":
        return ""

    salt = get_user_id_hash_salt()
    h = hashlib.sha256()
    h.update(salt)
    h.update(b":")
    h.update(user_id.encode("utf-8", errors="replace"))
    digest_hex = h.hexdigest()[:_HASH_LEN_HEX]
    return f"{_USER_TAG_PREFIX}{digest_hex}"


def hash_user_id_raw(user_id: str) -> str:
    """Return ONLY the hex digest (no `user:` prefix) for use in
    content fields like `content.user_id_hash` per arch §7 normative
    content shape. Empty user_id → empty string."""
    tag = hash_user_id(user_id)
    if not tag:
        return ""
    return tag[len(_USER_TAG_PREFIX):]


def clear_cache() -> None:
    """Clear the process-local salt cache. Called by tests + any future
    salt-rotation flow. Next call to `get_user_id_hash_salt()` re-reads
    from `~/.titan/secrets.toml`."""
    global _salt_cache
    with _salt_lock:
        _salt_cache = None


__all__ = [
    "hash_user_id",
    "hash_user_id_raw",
    "get_user_id_hash_salt",
    "clear_cache",
]
