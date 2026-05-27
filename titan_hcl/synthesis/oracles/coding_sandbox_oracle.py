"""Phase 6 — `coding_sandbox` TruthOraclePlug (§P6.B; SPEC §25.3 + §25.5).

Wraps the existing `CodingSandboxHelper`
(`titan_hcl/logic/agency/helpers/coding_sandbox.py`) as a
`TruthOraclePlug` for deterministic code + math claims.

Per SPEC §25.3 day-one set + arch §11.1: free oracle (always admits per
INV-Syn-13). Claim domains served:

- **code_correctness** — "does this Python code produce X?" The claim
  payload carries `{language: "python", code: <src>, expected_stdout?:
  <str>, assertion?: <str>}`. Verdict:
    - `"true"` if `expected_stdout` matches stdout (or `assertion`
      evaluates to truthy when explicit).
    - `"false"` if stdout disagrees / assertion fails / runtime exception.
    - `"unknown"` if AST validation rejects the source (un-verifiable
      shape — caller might re-issue with valid code).

- **math_correctness** — "does this expression equal X?" The claim
  payload carries `{expression: <str>, expected: <number-or-str>,
  tolerance?: <float>}`. Internally rewritten as a single-line print
  + tolerance-aware comparison. Same verdict semantics as
  `code_correctness`.

The wrapper is **synchronous** (the `TruthOraclePlug` protocol is sync).
The underlying `CodingSandboxHelper._run_code()` is itself sync — the
helper's `async execute()` wrapper exists for the FastAPI hot path.
The synthesis_worker process calls `verify()` synchronously from its
recv loop; no asyncio bridge needed here.

`evidence_ref` carries a content-addressed digest of the (code,
expected) tuple so two callers verifying the same claim get the same
ref — making the verdict re-discoverable on the chain by stable hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

from titan_hcl.logic.agency.helpers.coding_sandbox import (
    CodingSandboxHelper,
    validate_code,
)
from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict

logger = logging.getLogger(__name__)


SUPPORTED_DOMAINS = frozenset({"code_correctness", "math_correctness"})


def _evidence_ref(payload: dict[str, Any]) -> str:
    """Stable hash of the claim payload — re-discovery key for the chain.

    Hashing the FULL canonical payload (not just `code`) so a claim
    with a different `expected_stdout` is a distinct verdict from the
    same code with another expectation — both are independently
    verifiable + auditable.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _math_payload_to_code(payload: dict[str, Any]) -> tuple[str, Optional[str], Optional[str]]:
    """Lower a math_correctness claim to a code_correctness shape.

    Returns ``(generated_code, expected_stdout, assertion)``. The
    generated code is a minimal print(expression). Tolerance handling
    lives in the verdict comparator below — we always compare stdout as
    a string, then if both sides parse as floats we apply `tolerance`.
    """
    expr = str(payload.get("expression", ""))
    expected = payload.get("expected")
    if expr == "":
        return ("", None, None)
    # Minimal expression eval via print — sandbox AST already rejects
    # `import os` / `subprocess` / etc., so this stays bounded.
    code = f"print({expr})"
    expected_stdout: Optional[str]
    if expected is None:
        expected_stdout = None
    else:
        expected_stdout = str(expected)
    return (code, expected_stdout, None)


def _stdout_matches(actual: str, expected: str, tolerance: float) -> bool:
    """Compare run output to expectation; numeric-tolerant when both parse as float."""
    actual = actual.strip()
    expected = expected.strip()
    if actual == expected:
        return True
    if tolerance > 0.0:
        try:
            return abs(float(actual) - float(expected)) <= tolerance
        except (ValueError, TypeError):
            return False
    return False


class CodingSandboxOracle:
    """TruthOraclePlug wrapping `CodingSandboxHelper`.

    Instantiate once per synthesis_worker boot; pass the same instance
    to the `ToolPlug` wrapper in P6.I so one underlying helper serves
    both surfaces (sandbox-as-oracle + sandbox-as-tool — arch §11.1
    "doubles as a ToolPlug"; SPEC §25.3 "same instance as the
    truth-oracle, two thin wrappers").
    """

    oracle_id: str = "coding_sandbox"
    cost_class: str = "free"  # INV-Syn-13: always admits

    def __init__(self, helper: Optional[CodingSandboxHelper] = None):
        self._helper = helper or CodingSandboxHelper()

    @property
    def helper(self) -> CodingSandboxHelper:
        """Expose the underlying helper so the P6.I `coding_sandbox_tool`
        wrapper can share the same instance (one resource, two surfaces)."""
        return self._helper

    def can_handle(self, domain: str) -> bool:
        return domain in SUPPORTED_DOMAINS

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        t0 = time.perf_counter()
        ts_now = time.time()

        if not self.can_handle(claim.domain):
            # Defensive — the OracleRouter (P6.F) should not dispatch
            # claims this plug rejects, but if it does, return a clean
            # "unknown" so the router doesn't crash mid-routing.
            return OracleVerdict(
                oracle_id=self.oracle_id,
                verdict="unknown",
                evidence_ref="domain_unsupported",
                cost=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                ts=ts_now,
            )

        payload = claim.payload or {}
        evidence = _evidence_ref(payload)

        if claim.domain == "math_correctness":
            code, expected_stdout, assertion = _math_payload_to_code(payload)
            tolerance = float(payload.get("tolerance", 0.0))
        else:  # code_correctness
            code = str(payload.get("code", ""))
            expected_stdout = payload.get("expected_stdout")
            assertion = payload.get("assertion")
            tolerance = float(payload.get("tolerance", 0.0))

        if not code.strip():
            return OracleVerdict(
                oracle_id=self.oracle_id,
                verdict="unknown",
                evidence_ref=evidence,
                cost=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                ts=ts_now,
            )

        # AST validation FIRST — blocked imports → "unknown" (un-verifiable;
        # caller may re-issue with safe code).
        if assertion:
            # Wrap the assertion into a single executable program; the
            # helper's validator runs against this composed source.
            full_source = f"{code}\nassert ({assertion})\nprint('ASSERTION_OK')"
        else:
            full_source = code
        ast_ok, ast_msg = validate_code(full_source)
        if not ast_ok:
            logger.debug("[coding_sandbox_oracle] AST rejected: %s", ast_msg)
            return OracleVerdict(
                oracle_id=self.oracle_id,
                verdict="unknown",
                evidence_ref=evidence,
                cost=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                ts=ts_now,
            )

        # Run the validated source in the sandbox subprocess. _run_code
        # is the sync primitive (CodingSandboxHelper.execute wraps it in
        # asyncio.to_thread for FastAPI; we're already sync here).
        try:
            run = self._helper._run_code(full_source)
        except Exception:  # pragma: no cover — _run_code is defensive itself
            logger.exception("[coding_sandbox_oracle] sandbox run raised")
            return OracleVerdict(
                oracle_id=self.oracle_id,
                verdict="unknown",
                evidence_ref=evidence,
                cost=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                ts=ts_now,
            )

        success = bool(run.get("success", False))
        stdout = str(run.get("result", "") or "")

        # Verdict logic — assertion mode FIRST (when present), then stdout match.
        if assertion is not None:
            # The sandbox prints 'ASSERTION_OK' only if assert () passed.
            verdict = "true" if (success and "ASSERTION_OK" in stdout) else "false"
        elif expected_stdout is not None:
            verdict = (
                "true"
                if (success and _stdout_matches(stdout, str(expected_stdout), tolerance))
                else "false"
            )
        else:
            # No assertion + no expected_stdout — caller wants "does this
            # compile + run without raising?" → success → true; raise → false.
            verdict = "true" if success else "false"

        return OracleVerdict(
            oracle_id=self.oracle_id,
            verdict=verdict,
            evidence_ref=evidence,
            cost=0.0,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ts=ts_now,
        )


__all__ = ("CodingSandboxOracle", "SUPPORTED_DOMAINS")
