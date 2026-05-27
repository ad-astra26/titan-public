"""Phase 6 — `coding_sandbox` TruthOraclePlug tests (§P6.B).

Covers `titan_hcl/synthesis/oracles/coding_sandbox_oracle.py` against
SPEC §25.3 + arch §11.1 + the TruthOraclePlug protocol in `plugs.py`:

- can_handle covers {code_correctness, math_correctness} and nothing else
- cost_class == "free" (INV-Syn-13: always admits)
- code_correctness verdict mapping (expected_stdout match → true; mismatch → false; AST rejection → unknown; runtime exception → false)
- assertion mode (assert pass → true; assert fail → false)
- math_correctness lowering (expression evaluates + matches expected → true; tolerance honored)
- evidence_ref is content-addressed (same payload → same ref; different payload → different ref)
- latency_ms + ts populated
- defensive: unsupported-domain claim returns clean "unknown" not exception
- defensive: empty code returns "unknown"

These tests RUN the real sandbox subprocess (cheap math). They're tagged
as integration-flavored — but the subprocess work is bounded to <2s per
test, so the suite stays fast.
"""
from __future__ import annotations

import time

import pytest

from titan_hcl.synthesis.oracles.coding_sandbox_oracle import (
    SUPPORTED_DOMAINS,
    CodingSandboxOracle,
)
from titan_hcl.synthesis.plugs import OracleClaim


@pytest.fixture
def oracle():
    return CodingSandboxOracle()


# ─────────────────────────────────────────────────────────────────────────
# Protocol surface
# ─────────────────────────────────────────────────────────────────────────


def test_oracle_id_and_cost_class(oracle):
    assert oracle.oracle_id == "coding_sandbox"
    assert oracle.cost_class == "free"


def test_can_handle_supported_domains(oracle):
    assert oracle.can_handle("code_correctness") is True
    assert oracle.can_handle("math_correctness") is True
    assert SUPPORTED_DOMAINS == frozenset({"code_correctness", "math_correctness"})


def test_can_handle_rejects_other_domains(oracle):
    for d in ("solana_tx_confirmed", "web_fact", "x_event_real", "random"):
        assert oracle.can_handle(d) is False


def test_verify_rejects_unsupported_domain_gracefully(oracle):
    """Defensive: even if the router mis-dispatches, verify() must not raise."""
    v = oracle.verify(
        OracleClaim(domain="not_a_real_domain", payload={"code": "print(1)"})
    )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "domain_unsupported"
    assert v.oracle_id == "coding_sandbox"
    assert v.cost == 0.0


# ─────────────────────────────────────────────────────────────────────────
# code_correctness — expected_stdout mode
# ─────────────────────────────────────────────────────────────────────────


def test_code_correctness_expected_stdout_matches(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "print(2 + 2)",
                "expected_stdout": "4",
            },
        )
    )
    assert v.verdict == "true"
    assert v.oracle_id == "coding_sandbox"
    assert v.cost == 0.0
    assert v.latency_ms >= 0
    assert v.ts > 0
    assert len(v.evidence_ref) == 64  # sha256 hex


def test_code_correctness_expected_stdout_mismatch(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "print(2 + 2)",
                "expected_stdout": "5",
            },
        )
    )
    assert v.verdict == "false"
    assert v.cost == 0.0


def test_code_correctness_no_expectation_returns_true_when_runs(oracle):
    """No assertion + no expected_stdout = "did it execute without raising?" """
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "x = 1 + 1\nprint('hi')",
            },
        )
    )
    assert v.verdict == "true"


def test_code_correctness_runtime_exception_returns_false(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "raise ValueError('boom')",
            },
        )
    )
    assert v.verdict == "false"


# ─────────────────────────────────────────────────────────────────────────
# code_correctness — assertion mode
# ─────────────────────────────────────────────────────────────────────────


def test_code_correctness_assertion_passes(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "import math\nresult = math.sqrt(4)",
                "assertion": "result == 2.0",
            },
        )
    )
    assert v.verdict == "true"


def test_code_correctness_assertion_fails(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "result = 1 + 1",
                "assertion": "result == 3",
            },
        )
    )
    assert v.verdict == "false"


# ─────────────────────────────────────────────────────────────────────────
# AST validation — blocked imports → unknown
# ─────────────────────────────────────────────────────────────────────────


def test_blocked_import_yields_unknown_verdict(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "import os\nprint(os.environ.get('HOME'))",
                "expected_stdout": "/anything",
            },
        )
    )
    assert v.verdict == "unknown"
    assert v.cost == 0.0


def test_syntax_error_yields_unknown_verdict(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={
                "language": "python",
                "code": "def broken(:",  # syntax error
                "expected_stdout": "anything",
            },
        )
    )
    assert v.verdict == "unknown"


def test_empty_code_yields_unknown(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="code_correctness",
            payload={"language": "python", "code": "   \n  "},
        )
    )
    assert v.verdict == "unknown"


# ─────────────────────────────────────────────────────────────────────────
# math_correctness — expression mode
# ─────────────────────────────────────────────────────────────────────────


def test_math_correctness_integer_match(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={"expression": "2 + 2", "expected": 4},
        )
    )
    assert v.verdict == "true"


def test_math_correctness_mismatch(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={"expression": "2 + 2", "expected": 5},
        )
    )
    assert v.verdict == "false"


def test_math_correctness_with_tolerance(oracle):
    """1/3 ≈ 0.3333... — exact stdout mismatch but tolerance covers it."""
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={
                "expression": "1 / 3",
                "expected": 0.333,
                "tolerance": 0.001,
            },
        )
    )
    assert v.verdict == "true"


def test_math_correctness_tolerance_too_tight_returns_false(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={
                "expression": "1 / 3",
                "expected": 0.333,
                "tolerance": 0.0,
            },
        )
    )
    assert v.verdict == "false"


def test_math_correctness_empty_expression_unknown(oracle):
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={"expression": "", "expected": 0},
        )
    )
    assert v.verdict == "unknown"


# ─────────────────────────────────────────────────────────────────────────
# evidence_ref content-addressing
# ─────────────────────────────────────────────────────────────────────────


def test_evidence_ref_same_for_identical_payload(oracle):
    payload = {
        "language": "python",
        "code": "print(7)",
        "expected_stdout": "7",
    }
    v1 = oracle.verify(OracleClaim(domain="code_correctness", payload=payload))
    v2 = oracle.verify(OracleClaim(domain="code_correctness", payload=dict(payload)))
    assert v1.evidence_ref == v2.evidence_ref


def test_evidence_ref_differs_for_different_expected(oracle):
    p1 = {"language": "python", "code": "print(7)", "expected_stdout": "7"}
    p2 = {"language": "python", "code": "print(7)", "expected_stdout": "8"}
    v1 = oracle.verify(OracleClaim(domain="code_correctness", payload=p1))
    v2 = oracle.verify(OracleClaim(domain="code_correctness", payload=p2))
    assert v1.evidence_ref != v2.evidence_ref


# ─────────────────────────────────────────────────────────────────────────
# Latency budget — meta-check, not a hard performance gate
# ─────────────────────────────────────────────────────────────────────────


def test_short_math_claim_completes_quickly(oracle):
    """Sanity: simple math should finish well under the sandbox 30s timeout."""
    t0 = time.perf_counter()
    v = oracle.verify(
        OracleClaim(
            domain="math_correctness",
            payload={"expression": "2 ** 10", "expected": 1024},
        )
    )
    elapsed = time.perf_counter() - t0
    assert v.verdict == "true"
    assert elapsed < 10.0  # generous; typical run is <1s
    assert v.latency_ms < 10_000
