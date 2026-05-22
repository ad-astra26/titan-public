"""SPEC §24.7 — ZK Vault Merkle commit per backup event.

For every unified-manifest event (baseline + incremental on the Arweave plane):
  1. Compute `event_merkle_root = sha256(
       personality_merkle_root || timechain_merkle_root || soul_merkle_root_or_zero
     )` over the canonical 96-byte concatenation
  2. Build Solana memo (v=2 format):
       `v=2;event_id={id};root={event_merkle_root[:32]};prev={prev_event_merkle_root[:16]}`
  3. Submit via existing `network.send_sovereign_transaction([memo_ix], priority="LOW")`
  4. Return Solana tx_id → caller stores in manifest event's `zk_commit_tx` field

Restore-time verification (§24.7 + §3.1 step 8): walk the manifest events,
fetch each Arweave tarball, compute its sha256, recompose the event_merkle_root,
and verify it matches the on-chain `commit_state` payload at `zk_commit_tx`.
Mismatch at ANY level → REJECT tarball, halt restore, emit BACKUP_MERKLE_MISMATCH.

This module hosts the PURE LOGIC helpers (deterministic, side-effect free,
Solana-call-free) so we can unit-test the protocol without spinning up a
Solana client. The on-chain submission lives in RebirthBackup.commit_event_
merkle_to_zk_vault() which uses these helpers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

# Maker decision 2026-05-15 Q1: SPEC v1.9.0 §24.7 — single memo version
ZK_COMMIT_MEMO_VERSION = 2
# Solana memo program v1 caps at 566 bytes input; we stay well under
ZK_COMMIT_MEMO_MAX_BYTES = 200


def zero_merkle_root() -> str:
    """64 hex chars of zero — used as soul_merkle_root for non-soul events
    (Sunday-only soul tier per SPEC §24.4.C). Distinct from a real all-zero
    hash because nothing legitimate sha256-hashes to zero in practice."""
    return "00" * 32


def compute_event_merkle_root(
    personality_merkle_root: str,
    timechain_merkle_root: str,
    soul_merkle_root: Optional[str] = None,
) -> str:
    """SPEC §24.7 step 1.

    event_merkle_root = sha256(personality || timechain || (soul or zero32))
    over the canonical concatenation of the three 32-byte (hex-encoded) hashes.

    Each input is a hex string (64 chars). Bytes are decoded from hex,
    concatenated (96 bytes), then sha256'd. Output is hex-encoded (64 chars).

    Raises ValueError on malformed input (not 64 hex chars).
    """
    soul = soul_merkle_root if soul_merkle_root is not None else zero_merkle_root()
    for label, h in (("personality", personality_merkle_root),
                     ("timechain", timechain_merkle_root),
                     ("soul", soul)):
        if not isinstance(h, str) or len(h) != 64:
            raise ValueError(
                f"{label}_merkle_root must be 64 hex chars (32 bytes), "
                f"got len={len(h) if isinstance(h, str) else type(h).__name__}"
            )
        try:
            bytes.fromhex(h)
        except ValueError as e:
            raise ValueError(
                f"{label}_merkle_root is not valid hex: {e}"
            ) from None

    combined = (
        bytes.fromhex(personality_merkle_root)
        + bytes.fromhex(timechain_merkle_root)
        + bytes.fromhex(soul)
    )
    assert len(combined) == 96, "canonical concat must be 96 bytes"
    return hashlib.sha256(combined).hexdigest()


def build_zk_memo(
    event_id: str,
    event_merkle_root: str,
    prev_event_merkle_root: Optional[str] = None,
) -> str:
    """SPEC §24.7 step 2 — build the v=2 memo string.

    Format: `v=2;event_id={id};root={root[:32]};prev={prev[:16]}`

    First event in chain → prev = "genesis" (matches existing v=2 anchor_backup_hash
    convention at backup.py:1611 to stay coherent with the pre-§24 chain).
    """
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    if len(event_merkle_root) != 64:
        raise ValueError(
            f"event_merkle_root must be 64 hex chars, got len={len(event_merkle_root)}"
        )
    prev_fragment = (
        prev_event_merkle_root[:16] if prev_event_merkle_root else "genesis"
    )
    memo = (
        f"v={ZK_COMMIT_MEMO_VERSION};"
        f"event_id={event_id};"
        f"root={event_merkle_root[:32]};"
        f"prev={prev_fragment}"
    )
    if len(memo.encode("utf-8")) > ZK_COMMIT_MEMO_MAX_BYTES:
        raise ValueError(
            f"Memo length {len(memo.encode('utf-8'))} exceeds max "
            f"{ZK_COMMIT_MEMO_MAX_BYTES} — shorten event_id or hash fragments"
        )
    return memo


# Regex used by verifier (Phase 6 restore path) — exported here for unit
# testing the round-trip parse contract.
ZK_MEMO_V2_PATTERN = re.compile(
    r"^v=2;event_id=(?P<event_id>[^;]+);"
    r"root=(?P<root>[0-9a-f]{32});"
    r"prev=(?P<prev>[0-9a-f]{16}|genesis)$"
)


def parse_zk_memo(memo: str) -> Optional[dict]:
    """Inverse of build_zk_memo. Returns parsed dict or None if malformed.

    Returned shape: {"event_id": str, "root": str (32 hex), "prev": str
                     (16 hex or 'genesis')}. Used by Phase 6 restore + by
                     `arch_map backup verify --restore-sim` to walk the
                     on-chain ZK Vault hash chain.
    """
    m = ZK_MEMO_V2_PATTERN.match(memo)
    if not m:
        return None
    return {
        "event_id": m.group("event_id"),
        "root": m.group("root"),
        "prev": m.group("prev"),
    }
