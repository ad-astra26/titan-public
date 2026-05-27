"""Phase 6 — ProofStrategyPlug + ProofStrategyRegistry tests (§P6.G; INV-Syn-14).

Covers:
- MerkleProofStrategy: bytes payload, multi-leaf iterable, pre-hashed-hex,
  string payload, round-trip verify(), wrong-payload verify-False,
  empty-input edge case
- ZKProofStrategy: commit_fn delegation, verify_fn delegation, cost
  reporting, default unconfigured commit_fn raises (surfaces misconfig),
  commit_fn exception → failed proof, payload-bytes vs payload-str
- ProofStrategyRegistry: INV-Syn-14 HARD enforcement —
    * Merkle is default everywhere
    * Privacy domain whitelist → ZK auto-promote (3 domains parametrized)
    * Per-fork "zk" flag → ZK
    * Per-fork "merkle" flag → Merkle (explicit override)
    * Both triggers together → ZK
    * Neither trigger → Merkle (the default)
- commit() convenience routes to right strategy by selection
"""
from __future__ import annotations

import hashlib

import pytest

from titan_hcl.synthesis.merkle import merkle_root_hex
from titan_hcl.synthesis.plugs import Proof
from titan_hcl.synthesis.proofs.merkle_proof import MerkleProofStrategy, _normalize_leaves
from titan_hcl.synthesis.proofs.registry import ProofStrategyRegistry
from titan_hcl.synthesis.proofs.zk_proof import ZKProofStrategy


# ─────────────────────────────────────────────────────────────────────────
# MerkleProofStrategy
# ─────────────────────────────────────────────────────────────────────────


def test_merkle_strategy_id():
    s = MerkleProofStrategy()
    assert s.strategy == "merkle"


def test_merkle_commit_bytes_single_leaf():
    s = MerkleProofStrategy()
    payload = b"hello world"
    proof = s.commit(payload)
    assert proof.strategy == "merkle"
    assert proof.cost == 0.0
    assert proof.payload_ref is None
    # The leaf is sha256(b"hello world"); 1-leaf root = sha256(leaf || leaf).
    leaf = hashlib.sha256(payload).hexdigest()
    expected = merkle_root_hex([leaf])
    assert proof.commitment == bytes.fromhex(expected)


def test_merkle_commit_iterable_of_bytes():
    s = MerkleProofStrategy()
    leaves = [b"a", b"b", b"c"]
    proof = s.commit(leaves)
    expected_hex_leaves = [hashlib.sha256(l).hexdigest() for l in leaves]
    assert proof.commitment == bytes.fromhex(merkle_root_hex(expected_hex_leaves))


def test_merkle_commit_iterable_of_pre_hashed_hex():
    """Pre-hashed hex strings of length 64 are used as-is."""
    s = MerkleProofStrategy()
    leaves_hex = [
        "a" * 64,
        "b" * 64,
    ]
    proof = s.commit(leaves_hex)
    expected = merkle_root_hex(leaves_hex)
    assert proof.commitment == bytes.fromhex(expected)


def test_merkle_commit_iterable_of_strings_hashes_them():
    s = MerkleProofStrategy()
    strings = ["alpha", "beta"]
    proof = s.commit(strings)
    expected_hex_leaves = [hashlib.sha256(s.encode()).hexdigest() for s in strings]
    assert proof.commitment == bytes.fromhex(merkle_root_hex(expected_hex_leaves))


def test_merkle_verify_round_trip_bytes():
    s = MerkleProofStrategy()
    payload = b"some content"
    proof = s.commit(payload)
    assert s.verify(proof, payload) is True


def test_merkle_verify_round_trip_iterable():
    s = MerkleProofStrategy()
    leaves = [b"x", b"y", b"z"]
    proof = s.commit(leaves)
    assert s.verify(proof, leaves) is True


def test_merkle_verify_wrong_payload_is_false():
    s = MerkleProofStrategy()
    proof = s.commit(b"original")
    assert s.verify(proof, b"tampered") is False


def test_merkle_verify_missing_payload_is_false():
    s = MerkleProofStrategy()
    proof = s.commit(b"x")
    assert s.verify(proof, None) is False


def test_merkle_verify_wrong_strategy_in_proof_is_false():
    s = MerkleProofStrategy()
    fake_proof = Proof(strategy="zk", commitment=b"\x00" * 32)
    assert s.verify(fake_proof, b"x") is False


def test_normalize_leaves_rejects_unknown_types():
    with pytest.raises(TypeError):
        _normalize_leaves([12345])  # not bytes / str


# ─────────────────────────────────────────────────────────────────────────
# ZKProofStrategy
# ─────────────────────────────────────────────────────────────────────────


def test_zk_strategy_id():
    s = ZKProofStrategy()
    assert s.strategy == "zk"


def test_zk_commit_default_fn_raises_clearly():
    """Misconfigured worker (forgot to inject commit_fn) MUST surface
    immediately, not silently fall through to a degraded proof."""
    s = ZKProofStrategy()
    with pytest.raises(NotImplementedError, match="commit_fn"):
        s.commit(b"x")


def test_zk_verify_default_fn_raises_clearly():
    s = ZKProofStrategy()
    # Construct a "fake" proof with a non-empty payload_ref so the
    # default verify_fn is actually invoked.
    proof = Proof(strategy="zk", commitment=b"\xab" * 32, payload_ref="tx_id_123")
    with pytest.raises(NotImplementedError, match="verify_fn"):
        s.verify(proof)


def test_zk_commit_delegates_to_injected_commit_fn():
    calls = []

    def fake_commit(digest: bytes):
        calls.append(digest)
        return (b"\xaa" * 32, "solana_tx_abc", 0.000005)

    s = ZKProofStrategy(commit_fn=fake_commit)
    payload = b"private user data"
    proof = s.commit(payload)
    # The digest passed to commit_fn is sha256(payload).
    assert calls[0] == hashlib.sha256(payload).digest()
    assert proof.strategy == "zk"
    assert proof.commitment == b"\xaa" * 32
    assert proof.payload_ref == "solana_tx_abc"
    assert proof.cost == 0.000005


def test_zk_commit_string_payload_encodes_to_utf8_then_hashes():
    captured = {}

    def fake_commit(digest):
        captured["digest"] = digest
        return (b"\x01" * 32, "tx", 0.0)

    s = ZKProofStrategy(commit_fn=fake_commit)
    s.commit("café")
    assert captured["digest"] == hashlib.sha256("café".encode("utf-8")).digest()


def test_zk_commit_rejects_unsupported_payload_type():
    s = ZKProofStrategy(commit_fn=lambda d: (b"", "", 0.0))
    with pytest.raises(TypeError):
        s.commit(12345)


def test_zk_commit_fn_exception_returns_failed_proof():
    def angry_commit(digest):
        raise RuntimeError("solana down")

    s = ZKProofStrategy(commit_fn=angry_commit)
    proof = s.commit(b"x")
    assert proof.strategy == "zk"
    assert proof.commitment == b""
    assert proof.payload_ref == "commit_failed"
    assert proof.cost == 0.0


def test_zk_verify_delegates_to_injected_verify_fn():
    def fake_verify(commitment, payload_ref):
        return commitment == b"\xaa" * 32 and payload_ref == "tx_abc"

    s = ZKProofStrategy(commit_fn=lambda d: (b"", "", 0.0), verify_fn=fake_verify)
    good = Proof(strategy="zk", commitment=b"\xaa" * 32, payload_ref="tx_abc")
    bad = Proof(strategy="zk", commitment=b"\xbb" * 32, payload_ref="tx_abc")
    assert s.verify(good) is True
    assert s.verify(bad) is False


def test_zk_verify_failed_commit_proof_returns_false():
    s = ZKProofStrategy(verify_fn=lambda c, r: True)  # would otherwise return True
    failed = Proof(strategy="zk", commitment=b"", payload_ref="commit_failed")
    assert s.verify(failed) is False


def test_zk_verify_wrong_strategy_is_false():
    s = ZKProofStrategy(verify_fn=lambda c, r: True)
    merkle_proof = Proof(strategy="merkle", commitment=b"\x00" * 32, payload_ref="x")
    assert s.verify(merkle_proof) is False


# ─────────────────────────────────────────────────────────────────────────
# ProofStrategyRegistry — INV-Syn-14 HARD enforcement
# ─────────────────────────────────────────────────────────────────────────


def _registry(privacy_domains=None) -> ProofStrategyRegistry:
    return ProofStrategyRegistry(
        merkle=MerkleProofStrategy(),
        zk=ZKProofStrategy(commit_fn=lambda d: (b"\x00" * 32, "fake_tx", 0.0)),
        privacy_domains=frozenset(privacy_domains or ()),
    )


def test_registry_default_is_merkle_when_no_triggers_fire():
    """INV-Syn-14: Merkle is the default everywhere."""
    reg = _registry()
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy="merkle")
    assert s is reg.merkle


def test_registry_no_args_returns_merkle():
    reg = _registry()
    s = reg.select()
    assert s is reg.merkle


@pytest.mark.parametrize(
    "domain", ["private_user_data", "user_pii", "private_transaction"]
)
def test_inv_syn_14_privacy_domain_whitelist_auto_promotes_to_zk_hard(domain):
    """INV-Syn-14 (i): claim_domain in privacy whitelist → ZK."""
    reg = _registry(
        privacy_domains={"private_user_data", "user_pii", "private_transaction"}
    )
    s = reg.select(claim_domain=domain, fork_proof_strategy="merkle")
    assert s is reg.zk, (
        f"INV-Syn-14 (i) violation: privacy domain {domain!r} did not auto-promote to ZK"
    )


def test_inv_syn_14_non_privacy_domain_stays_merkle():
    reg = _registry(privacy_domains={"private_user_data"})
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy="merkle")
    assert s is reg.merkle


def test_inv_syn_14_per_fork_zk_flag_triggers_zk_hard():
    """INV-Syn-14 (ii): per-fork proof_strategy="zk" → ZK."""
    reg = _registry()
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy="zk")
    assert s is reg.zk


def test_inv_syn_14_per_fork_explicit_merkle_stays_merkle():
    reg = _registry()
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy="merkle")
    assert s is reg.merkle


def test_inv_syn_14_default_fork_proof_strategy_none_is_merkle():
    """Backwards-compat: existing forks without proof_strategy column → Merkle."""
    reg = _registry()
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy=None)
    assert s is reg.merkle


def test_inv_syn_14_both_triggers_yield_zk():
    """Privacy domain AND per-fork zk → ZK (union still ZK)."""
    reg = _registry(privacy_domains={"private_user_data"})
    s = reg.select(claim_domain="private_user_data", fork_proof_strategy="zk")
    assert s is reg.zk


def test_inv_syn_14_unrecognized_fork_proof_strategy_falls_back_to_merkle():
    """Misspelled / unknown values must NOT silently promote to ZK."""
    reg = _registry()
    s = reg.select(claim_domain="code_correctness", fork_proof_strategy="zk_typo")
    assert s is reg.merkle


# ─────────────────────────────────────────────────────────────────────────
# commit() convenience
# ─────────────────────────────────────────────────────────────────────────


def test_registry_commit_routes_to_merkle_by_default():
    reg = _registry()
    proof = reg.commit(b"x")
    assert proof.strategy == "merkle"


def test_registry_commit_routes_to_zk_under_inv_syn_14():
    reg = _registry(privacy_domains={"private_user_data"})
    proof = reg.commit(b"x", claim_domain="private_user_data")
    assert proof.strategy == "zk"
    assert proof.payload_ref == "fake_tx"


def test_registry_exposes_privacy_domains():
    reg = _registry(privacy_domains={"private_user_data", "user_pii"})
    assert reg.privacy_domains == frozenset({"private_user_data", "user_pii"})


def test_registry_exposes_strategy_instances():
    reg = _registry()
    assert isinstance(reg.merkle, MerkleProofStrategy)
    assert isinstance(reg.zk, ZKProofStrategy)
