"""Phase 6 — `web_api` TruthOraclePlug tests (§P6.D).

Covers `titan_hcl/synthesis/oracles/web_api_oracle.py`. Both `search_fn`
and `judge_fn` are injected, so tests don't touch the real web search +
dispatch infra (that's exercised end-to-end in §P6.M live runtime gate).
"""
from __future__ import annotations

import hashlib

import pytest

from titan_hcl.synthesis.oracles.web_api_oracle import (
    DEFAULT_PER_CALL_COST_SOL,
    SUPPORTED_DOMAINS,
    WebApiOracle,
    _default_judge_fn,
)
from titan_hcl.synthesis.plugs import OracleClaim


# ─────────────────────────────────────────────────────────────────────────
# Protocol surface
# ─────────────────────────────────────────────────────────────────────────


def test_oracle_id_and_cost_class():
    o = WebApiOracle(search_fn=lambda q: [], judge_fn=lambda c, e: "unknown")
    assert o.oracle_id == "web_api"
    assert o.cost_class == "metered"


def test_supported_domains():
    assert SUPPORTED_DOMAINS == frozenset({"web_fact", "wiki_fact"})


def test_can_handle_surface():
    o = WebApiOracle(search_fn=lambda q: [], judge_fn=lambda c, e: "unknown")
    assert o.can_handle("web_fact") is True
    assert o.can_handle("wiki_fact") is True
    assert o.can_handle("code_correctness") is False
    assert o.can_handle("solana_tx_confirmed") is False


def test_verify_unsupported_domain_returns_unknown():
    o = WebApiOracle(search_fn=lambda q: ["irrelevant"], judge_fn=lambda c, e: "true")
    v = o.verify(OracleClaim(domain="other", payload={"claim": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "domain_unsupported"


# ─────────────────────────────────────────────────────────────────────────
# Verdict flow — happy path
# ─────────────────────────────────────────────────────────────────────────


def test_verify_true_when_judge_says_true():
    o = WebApiOracle(
        search_fn=lambda q: ["evidence snippet 1", "evidence snippet 2"],
        judge_fn=lambda c, e: "true",
    )
    v = o.verify(
        OracleClaim(domain="web_fact", payload={"claim": "the sky is blue"})
    )
    assert v.verdict == "true"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL
    assert v.latency_ms >= 0
    # evidence_ref is sha256 of joined snippets
    expected_hash = hashlib.sha256(
        "evidence snippet 1\n\nevidence snippet 2".encode("utf-8")
    ).hexdigest()
    assert v.evidence_ref == expected_hash


def test_verify_false_when_judge_says_false():
    o = WebApiOracle(
        search_fn=lambda q: ["contradicting evidence"],
        judge_fn=lambda c, e: "false",
    )
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "X"}))
    assert v.verdict == "false"


def test_verify_unknown_when_judge_says_unknown():
    o = WebApiOracle(
        search_fn=lambda q: ["ambiguous evidence"],
        judge_fn=lambda c, e: "unknown",
    )
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "X"}))
    assert v.verdict == "unknown"


def test_judge_returns_garbage_collapses_to_unknown():
    """Defensive: a misbehaving LLM-judge returning 'maybe' / '' / None
    must not propagate into the verdict — collapse to 'unknown'."""
    o = WebApiOracle(
        search_fn=lambda q: ["e"],
        judge_fn=lambda c, e: "maybe-yes-no",
    )
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "X"}))
    assert v.verdict == "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────────────────


def test_missing_claim_payload_is_unknown():
    o = WebApiOracle(search_fn=lambda q: ["x"], judge_fn=lambda c, e: "true")
    v = o.verify(OracleClaim(domain="web_fact", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_claim"


def test_empty_claim_string_is_unknown():
    o = WebApiOracle(search_fn=lambda q: ["x"], judge_fn=lambda c, e: "true")
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "  "}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_claim"


def test_search_raises_yields_unknown_search_unreachable():
    def boom(q):
        raise RuntimeError("network down")

    o = WebApiOracle(search_fn=boom, judge_fn=lambda c, e: "true")
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "search_unreachable"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL


def test_search_returns_empty_is_unknown_no_evidence():
    o = WebApiOracle(search_fn=lambda q: [], judge_fn=lambda c, e: "true")
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "no_evidence"


def test_judge_raises_yields_unknown_judge_exception():
    def angry_judge(c, e):
        raise ValueError("judge crashed")

    o = WebApiOracle(search_fn=lambda q: ["evidence"], judge_fn=angry_judge)
    v = o.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "judge_exception"


# ─────────────────────────────────────────────────────────────────────────
# Default judge heuristic — deterministic substring-coverage
# ─────────────────────────────────────────────────────────────────────────


def test_default_judge_returns_true_on_high_coverage():
    claim = "the python programming language is interpreted"
    evidence = "Python is an interpreted high-level programming language."
    assert _default_judge_fn(claim, evidence) == "true"


def test_default_judge_returns_unknown_on_low_coverage():
    claim = "the moon is made of cheese"
    evidence = "Astronomy: stars and galaxies form the cosmos."
    assert _default_judge_fn(claim, evidence) == "unknown"


def test_default_judge_returns_false_on_contradiction():
    claim = "vaccines cause autism in children"
    evidence = (
        "Multiple studies show that vaccines do not cause autism. The original "
        "Wakefield paper is false and was retracted. Wrong claims persist."
    )
    assert _default_judge_fn(claim, evidence) == "false"


def test_default_judge_empty_inputs_yield_unknown():
    assert _default_judge_fn("", "evidence") == "unknown"
    assert _default_judge_fn("claim", "") == "unknown"
    assert _default_judge_fn("", "") == "unknown"


def test_default_judge_short_token_only_claim_yields_unknown():
    """Claim with no tokens ≥3 chars (e.g. 'a b c') → unknown."""
    assert _default_judge_fn("a b c d", "some evidence here") == "unknown"


# ─────────────────────────────────────────────────────────────────────────
# evidence_ref stability
# ─────────────────────────────────────────────────────────────────────────


def test_evidence_ref_stable_across_identical_snippets():
    o1 = WebApiOracle(
        search_fn=lambda q: ["snippet A", "snippet B"],
        judge_fn=lambda c, e: "true",
    )
    o2 = WebApiOracle(
        search_fn=lambda q: ["snippet A", "snippet B"],
        judge_fn=lambda c, e: "false",
    )
    v1 = o1.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    v2 = o2.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    # Same evidence → same ref (the verdict differs, but the evidence
    # the chain anchors against is content-addressed).
    assert v1.evidence_ref == v2.evidence_ref
    assert v1.verdict != v2.verdict


def test_evidence_ref_differs_for_different_snippets():
    o1 = WebApiOracle(
        search_fn=lambda q: ["snippet A"],
        judge_fn=lambda c, e: "true",
    )
    o2 = WebApiOracle(
        search_fn=lambda q: ["snippet B"],
        judge_fn=lambda c, e: "true",
    )
    v1 = o1.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    v2 = o2.verify(OracleClaim(domain="web_fact", payload={"claim": "x"}))
    assert v1.evidence_ref != v2.evidence_ref
