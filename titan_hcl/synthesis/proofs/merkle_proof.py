"""Phase 6 — `MerkleProofStrategy` (§P6.G — default everywhere).

Wraps the Phase-5 ``synthesis/merkle.py:merkle_root_hex`` (pairwise
SHA-256 binary tree with leaf duplication on odd counts) under the
`ProofStrategyPlug` protocol.

Per arch §11.2:
  > Merkle (default workhorse): integrity + inclusion. Cheap; sufficient
  > for all internal consolidation where the verifier is Titan or a
  > chain-trusting observer.

Per INV-Syn-14: Merkle is the default proof strategy everywhere — ZK
fires only on the (privacy-domain whitelist UNION per-fork flag) union.

The strategy accepts payloads in two shapes:

- **bytes** — the caller has already serialized; the strategy treats the
  bytes as a single leaf and returns the root of a 1-leaf tree
  (SHA-256(leaf || leaf) per merkle_root_hex's odd-count duplication).
- **iterable of leaves** — multiple leaves to roll up. The caller passes
  ``[leaf1_bytes, leaf2_bytes, ...]`` (or ``[hex_str, hex_str, ...]``);
  the strategy normalizes to hex + computes the root.

``verify(proof, payload)`` recomputes the root from the supplied payload
and compares against ``proof.commitment``. Without the payload, Merkle
verification is impossible by design (that's why we need ZK for the
"skeptic cannot see the data" case — INV-Syn-14).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Iterable, Optional, Union

from titan_hcl.synthesis.merkle import merkle_root_hex
from titan_hcl.synthesis.plugs import Proof

logger = logging.getLogger(__name__)


MerkleInput = Union[bytes, Iterable[Union[bytes, str]]]


def _normalize_leaves(payload: MerkleInput) -> list[str]:
    """Convert assorted payload shapes into a list of hex-encoded leaves."""
    if isinstance(payload, (bytes, bytearray)):
        return [hashlib.sha256(bytes(payload)).hexdigest()]
    leaves: list[str] = []
    for leaf in payload:
        if isinstance(leaf, (bytes, bytearray)):
            leaves.append(hashlib.sha256(bytes(leaf)).hexdigest())
        elif isinstance(leaf, str):
            # If the caller passes pre-hashed hex, keep it; otherwise hash the string.
            stripped = leaf.strip()
            if len(stripped) == 64 and all(c in "0123456789abcdefABCDEF" for c in stripped):
                leaves.append(stripped.lower())
            else:
                leaves.append(hashlib.sha256(stripped.encode("utf-8")).hexdigest())
        else:
            raise TypeError(f"Unsupported leaf type {type(leaf).__name__}")
    return leaves


class MerkleProofStrategy:
    """ProofStrategyPlug — Merkle root commitments.

    Cost is always 0.0 SOL (pure off-chain hashing — energy-consistent
    per INV-7 + arch §11.2 no-GPU-farm budget).
    """

    strategy: str = "merkle"

    def commit(self, payload: MerkleInput) -> Proof:
        """Compute the Merkle root over the payload.

        ``payload_ref`` is left None — the caller (synthesis_worker)
        decides whether to CAS-store the leaves separately. For the
        Phase-6 hypothesis-fork graduation / tombstone path, the leaves
        are the fork's exploration TX hashes which already live on the
        chain (no CAS needed); for arbitrary attestations the caller
        passes a CAS hash to ``commit(...)`` via the iterable shape.
        """
        leaves = _normalize_leaves(payload)
        root_hex = merkle_root_hex(leaves)
        return Proof(
            strategy="merkle",
            commitment=bytes.fromhex(root_hex),
            payload_ref=None,
            cost=0.0,
        )

    def verify(self, proof: Proof, payload: Optional[MerkleInput] = None) -> bool:
        """Re-walk the leaves and compare roots. Merkle verification REQUIRES
        the payload (or its leaves) — without it returns False (the proof
        is unverifiable by design)."""
        if payload is None:
            return False
        if proof.strategy != "merkle":
            return False
        leaves = _normalize_leaves(payload)
        recomputed = merkle_root_hex(leaves)
        try:
            return proof.commitment == bytes.fromhex(recomputed)
        except ValueError:
            return False


__all__ = ("MerkleProofStrategy",)
