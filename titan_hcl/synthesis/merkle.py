"""Merkle-root helpers for Phase 5 hypothesis-fork graduation + tombstone proofs.

Per `ARCHITECTURE_synthesis_engine.md §11.2` (proof middleware):

  > Merkle (default workhorse): integrity + inclusion. Cheap; sufficient for
  > all internal consolidation where the verifier is Titan or a chain-trusting
  > observer. Energy-consistent — ZK proving on CPU would blow the no-GPU-farm
  > budget, so Merkle-default is principled, not merely simpler.

Why a fresh module rather than reusing `timechain_v2`'s Merkle batching: the
v2 chain Merkle is over `Transaction` objects + carries batch-sealing
semantics. Phase 5's Merkle is over fork-exploration TX hashes (already-hex
strings) and is consumed only by the synthesis engine + the tombstone TX
verifier. Keeping the implementations separate keeps each one's contract
narrow and means a future change to the chain's batching cannot accidentally
silent-rewrite the meaning of a fork's `exploration_root`.

The construction is the canonical pairwise SHA-256 binary tree with leaf
duplication on odd counts (Bitcoin-style). Pure functions; no state.
"""
from __future__ import annotations

import hashlib
from typing import Iterable


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_root_hex(leaves: Iterable[str]) -> str:
    """Pairwise SHA-256 Merkle root over hex-encoded leaves.

    Each leaf is parsed as hex bytes; pairs are concatenated and hashed; odd
    counts duplicate the last node before pairing. Returns the root as a
    lowercase 64-char hex string.

    Edge cases:
      - **Empty input** → returns the SHA-256 of the empty byte string (so a
        fork with zero exploration TXs still produces a deterministic root —
        proves "the set was empty," not "the proof is missing").
      - **Single leaf** → returns SHA-256(leaf || leaf) per the standard
        odd-count duplication rule, NOT the leaf itself; this keeps the
        single-leaf case schema-consistent with multi-leaf roots.
    """
    leaf_bytes: list[bytes] = []
    for h in leaves:
        leaf_bytes.append(bytes.fromhex(h))
    if not leaf_bytes:
        return _h(b"").hex()

    # Single-leaf duplication so the root is always SHA-256(a||b), never the
    # bare leaf — keeps the verifier's tree-walk uniform.
    level: list[bytes] = leaf_bytes
    if len(level) == 1:
        return _h(level[0] + level[0]).hex()

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt: list[bytes] = []
        for i in range(0, len(level), 2):
            nxt.append(_h(level[i] + level[i + 1]))
        level = nxt
    return level[0].hex()


__all__ = ("merkle_root_hex",)
