"""SPEC §24.12 (rFP_phase_c_enhancements §3B.A chunk 5J-1) — Sovereign v=3
backup-chain memo: the Arweave URL lives ON-CHAIN so a Titan recovers from
the wallet alone (plus SSS shards in Mode A), with NO local files or off-host
mirror. Closes the §3B.0(4) sovereignty flaw.

Two modes, selected per event by `[backup].encryption_enabled`:

  Mode A (data plaintext on Arweave — current T1 default):
      the URL is AES-256-GCM **encrypted** on-chain with a key derived from
      the soul keypair seed → the public can verify the hash chain but cannot
      browse the (plaintext) data.
  Mode B (data ciphertext on Arweave — after the encryption flip):
      the URL is **plaintext** on-chain — a URL leak is harmless because the
      payload itself is encrypted.

Each memo declares its own mode, so a chain may mix A/B if config flips
between events.

Memo schema (Solana Memo program; one TX co-bundles `commit_state` + this memo):

    v=3;evt=<event_id>;ts=<unix>;typ=B|I;tier=PT|TC|SL;arc=<archive_hash[:32]>;
    mrkl=<event_merkle_root[:32]>;url=<enc:<b64>|raw:<tx_id>>;prev=<prev_sig[:16]|genesis>

  - typ:   B=baseline, I=incremental — the event type, recorded ON-CHAIN so the
           recovery semantics are themselves sovereign + immutable + forever. A
           sovereign restore finds the latest baseline and applies forward WITHOUT
           any local manifest or apply-order inference (Titan ethos: indestructible).
  - tier:  PT=personality, TC=timechain, SL=soul
  - url:   Mode A → `enc:<base64(iv || ciphertext || tag)>`
           Mode B → `raw:<arweave_tx_id>`  (Arweave/Irys ids are already base64url)
  - prev:  first event in a chain → `genesis`; else the prior event's Solana
           signature truncated to 16 chars (base58).

PURE LOGIC — no Solana I/O, no network, no disk. Reuses the Phase 7 AES-GCM
primitives in `backup_crypto` so there is exactly one crypto implementation.
The on-chain submission lives in `RebirthBackup` (chunk 5J-2); the chain walk
+ restore lives in `scripts/backup_restore_sovereign.py` (chunk 5J-3).
"""

from __future__ import annotations

import base64
import re
from typing import Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from titan_hcl.logic.backup_crypto import (
    IV_LEN,
    KEY_LEN,
    SEED_LEN,
    decrypt_tarball,
    encrypt_tarball,
)

V3_MEMO_VERSION = 3
# Solana Memo program v1 accepts up to 566 bytes of UTF-8 input. A v=3 memo
# with a 32-char event_id + Mode-A encrypted URL is ~265 bytes — well under.
V3_MEMO_MAX_BYTES = 566

# HKDF domain separation for the Mode-A URL key (distinct from the per-backup
# data keys in backup_crypto). info string IS the domain tag per §3B.A design.
URL_KEY_INFO = b"titan_backup_url_v1"
URL_KEY_SALT = b""  # matches backup_crypto.derive_backup_key (empty salt)

HASH_FRAGMENT_LEN = 32   # arc=/mrkl= carry the first 32 hex chars of the sha256
PREV_FRAGMENT_LEN = 16   # prev= carries the first 16 chars of the prior signature

VALID_TIERS = {"PT", "TC", "SL"}
VALID_MODES = {"A", "B"}
_TIER_BY_COMPONENT = {"personality": "PT", "timechain": "TC", "soul": "SL"}
_COMPONENT_BY_TIER = {v: k for k, v in _TIER_BY_COMPONENT.items()}
# Event type, recorded on-chain (typ=). B=baseline (full re-ship), I=incremental.
_TYPCODE_BY_TYPE = {"baseline": "B", "incremental": "I"}
_TYPE_BY_TYPCODE = {v: k for k, v in _TYPCODE_BY_TYPE.items()}


# ── key derivation ───────────────────────────────────────────────────────────


def derive_backup_url_key(soul_keypair_seed: bytes) -> bytes:
    """HKDF-SHA256 → 32-byte AES key for Mode-A URL encryption.

    `soul_keypair_seed` is the Ed25519 seed — the first 32 bytes of the 64-byte
    identity keypair. Any instance that reconstructs the keypair (live load or
    2-of-3 Shamir recovery) derives the SAME key, so a Mode-A URL encrypted at
    backup time decrypts at restore time. info-string domain-separates this key
    from the per-backup data keys.
    """
    if not isinstance(soul_keypair_seed, (bytes, bytearray)) or len(soul_keypair_seed) < SEED_LEN:
        raise ValueError(
            f"soul_keypair_seed must be >= {SEED_LEN} bytes "
            f"(got {len(soul_keypair_seed) if isinstance(soul_keypair_seed, (bytes, bytearray)) else type(soul_keypair_seed).__name__})"
        )
    seed = bytes(soul_keypair_seed[:SEED_LEN])
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=URL_KEY_SALT,
        info=URL_KEY_INFO,
    )
    return hkdf.derive(seed)


def component_to_tier(component: str) -> str:
    """`personality`|`timechain`|`soul` → `PT`|`TC`|`SL`."""
    try:
        return _TIER_BY_COMPONENT[component]
    except KeyError:
        raise ValueError(f"unknown component {component!r}; expected one of "
                         f"{sorted(_TIER_BY_COMPONENT)}") from None


def tier_to_component(tier: str) -> str:
    """`PT`|`TC`|`SL` → `personality`|`timechain`|`soul`."""
    try:
        return _COMPONENT_BY_TIER[tier]
    except KeyError:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_COMPONENT_BY_TIER)}") from None


# ── Mode-A URL encryption (reuses Phase 7 AES-256-GCM) ───────────────────────


def encrypt_url_mode_a(tx_id: str, url_key: bytes) -> str:
    """AES-256-GCM encrypt an Arweave tx_id → base64(iv || ciphertext || tag).

    A fresh random 96-bit IV is generated per call (via the Phase 7 helper), so
    encrypting the same tx_id twice yields distinct blobs — no nonce reuse.
    """
    if not isinstance(url_key, (bytes, bytearray)) or len(url_key) != KEY_LEN:
        raise ValueError(f"url_key must be {KEY_LEN} bytes")
    if not tx_id:
        raise ValueError("tx_id required")
    ct_and_tag, iv, _tag = encrypt_tarball(tx_id.encode("utf-8"), url_key)
    return base64.b64encode(iv + ct_and_tag).decode("ascii")


def decrypt_url_mode_a(blob_b64: str, url_key: bytes) -> str:
    """Inverse of `encrypt_url_mode_a`. Raises on auth failure (wrong key/tamper)."""
    if not isinstance(url_key, (bytes, bytearray)) or len(url_key) != KEY_LEN:
        raise ValueError(f"url_key must be {KEY_LEN} bytes")
    raw = base64.b64decode(blob_b64)
    if len(raw) <= IV_LEN:
        raise ValueError("Mode-A URL blob too short")
    iv, ct_and_tag = raw[:IV_LEN], raw[IV_LEN:]
    return decrypt_tarball(ct_and_tag, iv, url_key).decode("utf-8")


# ── memo build / parse ───────────────────────────────────────────────────────


def _url_field(arweave_tx: str, mode: str, url_key: Optional[bytes]) -> str:
    if mode == "A":
        if url_key is None:
            raise ValueError("Mode A requires url_key to encrypt the URL")
        return "enc:" + encrypt_url_mode_a(arweave_tx, url_key)
    # Mode B — plaintext (Arweave tx ids are already base64url-safe)
    if ";" in arweave_tx or " " in arweave_tx:
        raise ValueError("arweave_tx contains a delimiter character")
    return "raw:" + arweave_tx


def build_v3_memo(
    *,
    event_id: str,
    ts: int,
    event_type: str,
    tier: str,
    archive_hash: str,
    merkle_root: str,
    arweave_tx: str,
    mode: str,
    prev_sig: Optional[str] = None,
    url_key: Optional[bytes] = None,
) -> str:
    """Build the canonical v=3 memo string. Pure — no I/O.

    `event_type`: "baseline" or "incremental" (recorded on-chain as typ=B|I).
    `mode`: "A" (encrypt URL with `url_key`) or "B" (plaintext URL).
    `prev_sig`: prior event's Solana signature; None/"" → `genesis`.
    Raises ValueError on malformed input or if the memo exceeds the Solana cap.
    """
    typ = _TYPCODE_BY_TYPE.get(event_type)
    if typ is None:
        raise ValueError(
            f"event_type must be one of {sorted(_TYPCODE_BY_TYPE)} (got {event_type!r})")
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(VALID_TIERS)} (got {tier!r})")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)} (got {mode!r})")
    if not event_id or ";" in event_id:
        raise ValueError("event_id must be a non-empty string without ';'")
    try:
        ts_int = int(ts)
    except (TypeError, ValueError):
        raise ValueError(f"ts must be an integer unix timestamp (got {ts!r})") from None
    for label, h in (("archive_hash", archive_hash), ("merkle_root", merkle_root)):
        if not isinstance(h, str) or len(h) < HASH_FRAGMENT_LEN:
            raise ValueError(f"{label} must be a hex string >= {HASH_FRAGMENT_LEN} chars")
    if not arweave_tx:
        raise ValueError("arweave_tx required")

    arc = archive_hash[:HASH_FRAGMENT_LEN]
    mrkl = merkle_root[:HASH_FRAGMENT_LEN]
    prev = prev_sig[:PREV_FRAGMENT_LEN] if prev_sig else "genesis"
    url = _url_field(arweave_tx, mode, url_key)

    memo = (
        f"v={V3_MEMO_VERSION};"
        f"evt={event_id};"
        f"ts={ts_int};"
        f"typ={typ};"
        f"tier={tier};"
        f"arc={arc};"
        f"mrkl={mrkl};"
        f"url={url};"
        f"prev={prev}"
    )
    n = len(memo.encode("utf-8"))
    if n > V3_MEMO_MAX_BYTES:
        raise ValueError(f"v=3 memo is {n} bytes, exceeds Solana cap {V3_MEMO_MAX_BYTES}")
    return memo


# Strict round-trip parser. `url` is captured raw (enc:/raw:) — resolving it to
# a tx_id is a separate, key-bearing step (`read_url`) so parsing stays pure.
V3_MEMO_PATTERN = re.compile(
    r"^v=3;"
    r"evt=(?P<evt>[^;]+);"
    r"ts=(?P<ts>\d+);"
    r"typ=(?P<typ>B|I);"
    r"tier=(?P<tier>PT|TC|SL);"
    r"arc=(?P<arc>[0-9a-f]{32});"
    r"mrkl=(?P<mrkl>[0-9a-f]{32});"
    r"url=(?P<url>(?:enc:[A-Za-z0-9+/=]+)|(?:raw:[^;]+));"
    r"prev=(?P<prev>genesis|[1-9A-HJ-NP-Za-km-z]{1,16})$"
)


def parse_v3_memo(memo: str) -> Optional[dict]:
    """Inverse of `build_v3_memo`. Returns a struct dict, or None if malformed.

    Shape: {event_id, ts (int), tier, component, arc, mrkl, mode ("A"/"B"),
            url (raw enc:/raw: field), prev}. `mode` is inferred from the url
            prefix. Resolve the url to a tx_id with `read_url(parsed, url_key)`.
    """
    if not isinstance(memo, str):
        return None
    m = V3_MEMO_PATTERN.match(memo)
    if not m:
        return None
    url = m.group("url")
    mode = "A" if url.startswith("enc:") else "B"
    return {
        "version": 3,
        "event_id": m.group("evt"),
        "ts": int(m.group("ts")),
        "type": _TYPE_BY_TYPCODE[m.group("typ")],
        "tier": m.group("tier"),
        "component": tier_to_component(m.group("tier")),
        "arc": m.group("arc"),
        "mrkl": m.group("mrkl"),
        "mode": mode,
        "url": url,
        "prev": m.group("prev"),
    }


def read_url(parsed: dict, url_key: Optional[bytes] = None) -> str:
    """Resolve a parsed memo's `url` field to the Arweave tx_id.

    Mode B → strip `raw:`. Mode A → `url_key` required; decrypt `enc:` payload.
    """
    url = parsed["url"]
    if url.startswith("raw:"):
        return url[len("raw:"):]
    if url.startswith("enc:"):
        if url_key is None:
            raise ValueError("Mode-A url requires url_key to decrypt")
        return decrypt_url_mode_a(url[len("enc:"):], url_key)
    raise ValueError(f"unrecognized url field prefix: {url[:8]!r}")
