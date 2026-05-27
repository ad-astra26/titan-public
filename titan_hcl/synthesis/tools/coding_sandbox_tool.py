"""Phase 6 — `coding_sandbox` ToolPlug (§P6.I).

Wraps the same `CodingSandboxHelper` instance the P6.B oracle uses
(SPEC §25.3 "same instance as the truth-oracle, two thin wrappers").
Pre-P6 the sandbox existed but was unused by the outer self
(arch §11.3 gap); this ToolPlug + the agno_tools wiring closes that.

Capabilities: ``["code_execution", "computation", "data_analysis"]``.
"""
from __future__ import annotations

import logging
from typing import Optional

from titan_hcl.logic.agency.helpers.coding_sandbox import (
    CodingSandboxHelper,
    validate_code,
)
from titan_hcl.synthesis.oracles.coding_sandbox_oracle import (
    CodingSandboxOracle,
)
from titan_hcl.synthesis.plugs import OracleClaim, ToolCall, TruthOraclePlug
from titan_hcl.synthesis.tools.base import ToolPlugBase

logger = logging.getLogger(__name__)


class CodingSandboxTool(ToolPlugBase):
    tool_id: str = "coding_sandbox"
    _capabilities = ("code_execution", "computation", "data_analysis")

    def __init__(
        self,
        *,
        writer,
        router=None,
        helper: Optional[CodingSandboxHelper] = None,
        oracle: Optional[CodingSandboxOracle] = None,
    ):
        super().__init__(writer=writer, router=router)
        self._helper = helper or CodingSandboxHelper()
        # If an oracle wrapper was supplied (P6.B), share its helper so
        # both surfaces hit the same subprocess pool. Else build one over
        # the same helper instance.
        self._oracle = oracle if oracle is not None else CodingSandboxOracle(helper=self._helper)

    def oracle(self) -> Optional[TruthOraclePlug]:
        """Doubling-as-oracle surface (arch §11.1)."""
        return self._oracle

    def _execute(self, call: ToolCall) -> dict:
        code = str(call.args.get("code", ""))
        if not code.strip():
            return {"success": False, "result_summary": "no code provided"}
        ok, msg = validate_code(code)
        if not ok:
            return {"success": False, "result_summary": f"AST rejected: {msg}"}
        result = self._helper._run_code(code)
        return {
            "success": bool(result.get("success", False)),
            "result_summary": str(result.get("result", "") or "")[:512],
            "result_full_payload": str(result.get("result", "") or ""),
            "exception": result.get("error"),
        }

    def _companion_claim_for(self, call: ToolCall, raw: dict) -> Optional[OracleClaim]:
        """If the caller asked the tool to also verify (expected_stdout
        or assertion present in args), emit a companion code_correctness
        claim so the verdict rides the tool-call TX per INV-Syn-12.
        """
        expected = call.args.get("expected_stdout")
        assertion = call.args.get("assertion")
        if expected is None and assertion is None:
            return None
        payload = {
            "language": "python",
            "code": call.args.get("code", ""),
        }
        if expected is not None:
            payload["expected_stdout"] = expected
        if assertion is not None:
            payload["assertion"] = assertion
        return OracleClaim(domain="code_correctness", payload=payload, importance=0.5)


__all__ = ("CodingSandboxTool",)
