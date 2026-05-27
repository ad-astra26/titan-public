"""Phase 6 — `ZKProofStrategy` (§P6.G — targeted, INV-Syn-14 trigger).

Wraps the existing on-chain `titan_zk_vault` Anchor program (via the
``backup_zk_commit.py``-style Solana memo submission path) under the
`ProofStrategyPlug` protocol.

Per arch §11.2 + INV-Syn-14:
  > ZK (targeted, opt-in per fork, cost-gated): prove a *property of
  > hidden data to a skeptic who cannot see it* — only for **privacy**
  > or **external compute-integrity**.

The v1 implementation provides **on-chain commitment + audit** (the
ZK Vault's existing surface). Actual zero-knowledge proving (zk-SNARK
/ Groth16 / etc.) is an upgrade path "Phase 6+"; the plug protocol is
shaped now so the upgrade is a body-swap, not an interface change.

INV-Syn-14 selection (which payloads even reach this strategy) is
enforced by ``ProofStrategyRegistry.select(...)``; this module
implements the strategy mechanics once the selection has already
fired.

Pays through INV-Syn-13 under ``oracle_id="zk_prover"``. Cost is the
Solana memo fee (typically ~0.000005 SOL priority + ~5000 lamport
base = ~5e-6 SOL). Reported per-call via the injected ``commit_fn``.

Dependencies are injected at construction (``commit_fn`` /
``verify_fn``) so the plug is unit-testable without spinning up a
Solana client. Default callables raise ``NotImplementedError`` — the
synthesis_worker wires the real ZK Vault submitter at boot.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable, Optional, Union

from titan_hcl.synthesis.plugs import Proof

logger = logging.getLogger(__name__)


# ``commit_fn(payload_bytes) -> (commitment_bytes, payload_ref, cost_sol)``
# - commitment_bytes: the verifier-key commitment as bytes (typically a
#   32-byte SHA-256 digest the Vault stored on-chain).
# - payload_ref: the Solana tx id (or ZK Vault row id) the verifier
#   can re-fetch later.
# - cost_sol: the actual Solana fee paid for the commit (caller uses
#   it for INV-Syn-13 spend accounting).
ZKCommitFn = Callable[[bytes], tuple[bytes, str, float]]

# ``verify_fn(commitment_bytes, payload_ref) -> bool``
# - Verifies a previously-issued proof. Returns True iff the on-chain
#   commit at payload_ref matches the supplied commitment.
ZKVerifyFn = Callable[[bytes, str], bool]

DEFAULT_PER_CALL_COST_SOL: float = 5e-6


def _default_commit_fn(payload: bytes) -> tuple[bytes, str, float]:
    """Default no-op commit_fn — raises so a misconfigured worker
    surfaces the problem instead of silently using a stub commit.
    The real wiring lives in P6.I / synthesis_worker boot."""
    raise NotImplementedError(
        "ZKProofStrategy.commit() requires commit_fn to be injected at "
        "construction — see synthesis_worker boot path (P6.I)"
    )


def _default_verify_fn(commitment: bytes, payload_ref: str) -> bool:
    raise NotImplementedError(
        "ZKProofStrategy.verify() requires verify_fn to be injected at "
        "construction — see synthesis_worker boot path (P6.I)"
    )


class ZKProofStrategy:
    """ProofStrategyPlug — ZK Vault on-chain commit.

    Selected ONLY by ``ProofStrategyRegistry`` when INV-Syn-14 fires
    (privacy-domain whitelist UNION per-fork ``proof_strategy="zk"``
    flag). Never invoked directly from outside the registry.
    """

    strategy: str = "zk"

    def __init__(
        self,
        *,
        commit_fn: ZKCommitFn = _default_commit_fn,
        verify_fn: ZKVerifyFn = _default_verify_fn,
        nominal_per_call_cost_sol: float = DEFAULT_PER_CALL_COST_SOL,
    ):
        self._commit_fn = commit_fn
        self._verify_fn = verify_fn
        self._nominal_cost = float(nominal_per_call_cost_sol)

    def commit(self, payload: Union[bytes, bytearray, str]) -> Proof:
        """Submit a commitment to the ZK Vault.

        The strategy hashes the payload to a 32-byte digest (the
        commitment that lands on-chain), then delegates the on-chain
        write to ``commit_fn``. The Vault's Solana tx_id is stored
        in ``payload_ref`` so verify() can re-fetch.
        """
        if isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
        else:
            raise TypeError(f"Unsupported payload type {type(payload).__name__}")

        # Pre-hash to keep the on-chain memo bounded + leak-resistant —
        # only the digest goes on-chain, not the full payload.
        digest = hashlib.sha256(payload_bytes).digest()

        try:
            commitment, payload_ref, actual_cost = self._commit_fn(digest)
        except NotImplementedError:
            # Surface clearly — the synthesis_worker forgot to inject
            # commit_fn. Returning a degraded proof would lie about the
            # ZK guarantee.
            raise
        except Exception:
            logger.exception("[zk_proof] commit_fn raised; returning failed proof")
            return Proof(
                strategy="zk",
                commitment=b"",
                payload_ref="commit_failed",
                cost=0.0,
            )
        return Proof(
            strategy="zk",
            commitment=commitment,
            payload_ref=payload_ref,
            cost=float(actual_cost) if actual_cost is not None else self._nominal_cost,
        )

    def verify(self, proof: Proof, payload: Optional[Union[bytes, str]] = None) -> bool:
        """Verify by re-fetching the on-chain commit at payload_ref.

        Unlike Merkle, ZK verification does NOT need the original
        payload (that's the privacy property — the skeptic can verify
        the commit without seeing the data). The verifier fn checks
        the on-chain commitment at payload_ref equals proof.commitment.
        """
        if proof.strategy != "zk":
            return False
        if not proof.payload_ref or proof.payload_ref == "commit_failed":
            return False
        try:
            return bool(self._verify_fn(proof.commitment, proof.payload_ref))
        except NotImplementedError:
            raise
        except Exception:
            logger.exception("[zk_proof] verify_fn raised")
            return False


__all__ = ("ZKProofStrategy", "DEFAULT_PER_CALL_COST_SOL", "ZKCommitFn", "ZKVerifyFn")
