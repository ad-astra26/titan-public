"""Phase 6 — concrete `ProofStrategyPlug` implementations (D-SPEC-PHASE6).

Two strategies per SPEC §25.5 + arch §11.2 + INV-Syn-14:

- ``merkle_proof.MerkleProofStrategy`` — default everywhere; wraps the
  Phase-5 ``synthesis/merkle.py`` pairwise-SHA-256 binary-tree root.
  Free (no on-chain commit; verifier re-walks the leaves).
- ``zk_proof.ZKProofStrategy`` — opt-in only on (privacy-domain claim
  whitelist UNION caller-explicit per-fork flag) per INV-Syn-14; wraps
  the existing ``titan_zk_vault`` Anchor program via
  ``backup_zk_commit.py``-style Solana memo submission. Metered
  (``zk_prover`` per INV-Syn-13).

The selection logic lives in ``registry.ProofStrategyRegistry``.
"""
