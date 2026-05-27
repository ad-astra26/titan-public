"""Phase 6 — `ProofStrategyRegistry` (§P6.G — INV-Syn-14 enforcement).

Selects between ``MerkleProofStrategy`` (default) and ``ZKProofStrategy``
(targeted) for each proof request, enforcing INV-Syn-14 verbatim.

Per SPEC §25.1 INV-Syn-14:
  > Merkle is the default proof strategy everywhere. ZK fires **only**
  > on the union of:
  >   (i) Claim-domain whitelist for privacy
  >       (OracleClaim.domain ∈ {"private_user_data", "user_pii",
  >        "private_transaction"}) auto-promotes proof strategy to "zk".
  >   (ii) Caller-explicit per-fork flag
  >        (HypothesisFork.proof_strategy ∈ {"merkle", "zk"},
  >        default "merkle").

The registry exposes two surfaces:

- ``select(claim_domain, fork_proof_strategy)`` — pure selection logic;
  returns either the Merkle or ZK strategy instance.
- ``commit(payload, claim_domain, fork_proof_strategy)`` — convenience
  helper that selects + commits in one call.

Privacy domains are loaded from ``titan_params.toml [synthesis.oracle.zk]
privacy_domains`` at construction (via the helper from §P6.A
``oracle_gate.zk_privacy_domains``).
"""
from __future__ import annotations

import logging
from typing import Optional

from titan_hcl.synthesis.plugs import Proof
from titan_hcl.synthesis.proofs.merkle_proof import MerkleProofStrategy
from titan_hcl.synthesis.proofs.zk_proof import ZKProofStrategy

logger = logging.getLogger(__name__)


class ProofStrategyRegistry:
    """INV-Syn-14 enforcement layer over the two concrete strategies.

    The registry is constructed once at synthesis_worker boot with the
    privacy-domain whitelist + the two strategy instances. Callers
    (Phase-5 graduation paths, Phase-6 OracleRouter for batch proofs)
    invoke ``select(...)`` or ``commit(...)`` and never touch the
    strategy classes directly — that's what makes INV-Syn-14 hard
    enforcement possible.
    """

    def __init__(
        self,
        *,
        merkle: MerkleProofStrategy,
        zk: ZKProofStrategy,
        privacy_domains: frozenset[str] = frozenset(),
    ):
        self._merkle = merkle
        self._zk = zk
        self._privacy_domains = frozenset(privacy_domains)

    @property
    def privacy_domains(self) -> frozenset[str]:
        return self._privacy_domains

    @property
    def merkle(self) -> MerkleProofStrategy:
        return self._merkle

    @property
    def zk(self) -> ZKProofStrategy:
        return self._zk

    def select(
        self,
        *,
        claim_domain: Optional[str] = None,
        fork_proof_strategy: Optional[str] = None,
    ) -> MerkleProofStrategy | ZKProofStrategy:
        """Pure INV-Syn-14 selection.

        ``claim_domain`` — the OracleClaim domain (or any other
        domain-typed string the caller has). None disables the
        privacy-whitelist path.

        ``fork_proof_strategy`` — the HypothesisFork.proof_strategy
        column ("merkle" | "zk"); None defaults to "merkle".

        Returns the Merkle strategy unless the union (privacy domain
        OR explicit "zk") fires.
        """
        # (i) Claim-domain whitelist
        if claim_domain is not None and claim_domain in self._privacy_domains:
            logger.debug(
                "[ProofStrategyRegistry] INV-Syn-14 → ZK by privacy-domain whitelist "
                "(claim_domain=%s)", claim_domain,
            )
            return self._zk
        # (ii) Caller-explicit per-fork flag
        if fork_proof_strategy == "zk":
            logger.debug(
                "[ProofStrategyRegistry] INV-Syn-14 → ZK by explicit per-fork flag"
            )
            return self._zk
        # Default Merkle everywhere else.
        return self._merkle

    def commit(
        self,
        payload,
        *,
        claim_domain: Optional[str] = None,
        fork_proof_strategy: Optional[str] = None,
    ) -> Proof:
        """Select strategy + commit in one call. Convenience helper for
        call sites that don't need to introspect the chosen strategy.
        """
        strategy = self.select(
            claim_domain=claim_domain,
            fork_proof_strategy=fork_proof_strategy,
        )
        return strategy.commit(payload)


__all__ = ("ProofStrategyRegistry",)
