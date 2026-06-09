"""Phase 6 — `ToolPlug` base class (§P6.I).

Common machinery for the four day-one ToolPlugs:

- Procedural-fork TX emission via OuterMemoryWriter.write_tool_call
- INV-Syn-15 scored_by integration with OracleRouter companion verdict
  path (when an oracle wrapper is available for the tool)
- Result-payload content-addressing (CAS hash for ToolResult.result_full_hash)

Subclasses implement ``_execute(call: ToolCall) -> dict`` returning a
``dict`` carrying ``{"success": bool, "result_summary": str,
"result_full_payload": str (optional), "latency_ms": int (optional),
"exception": str (optional)}``. The base ``invoke()`` wraps that with
common bookkeeping (timing if not supplied, TX emission, optional
oracle companion verdict trigger).

Every ToolPlug invocation IS a tool-call procedural TX — INV-12 single
action surface; no parallel paths.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from titan_hcl.synthesis.plugs import (
    OracleClaim,
    ToolCall,
    ToolResult,
    TruthOraclePlug,
)

if TYPE_CHECKING:
    from titan_hcl.synthesis.oracle_router import OracleRouter
    from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter

logger = logging.getLogger(__name__)


class ToolPlugBase:
    """Base class for the four Phase-6 ToolPlugs.

    Subclasses set ``tool_id`` + ``_capabilities`` class attrs and
    implement ``_execute(call) -> dict``. The base handles the rest.

    Construction wires the OuterMemoryWriter (for procedural TX emit)
    and optionally an OracleRouter (for companion verdict trigger when
    the tool's outcome maps to a claim domain). Both are injected so
    unit tests can stub independently.
    """

    tool_id: str = "tool_base"
    _capabilities: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        writer: "OuterMemoryWriter",
        router: Optional["OracleRouter"] = None,
        skill_outcome_sink=None,
        companion_verdict_sink=None,
    ):
        self._writer = writer
        self._router = router
        # Operator-closure C2 (W7): callable
        # `(parent_tool_call_tx, oracle_id, verdict, evidence_ref, ts) -> None`.
        # For a tool that IS its own oracle (oracle() not None) running in a
        # process WITHOUT the OracleRouter (chat-time tools in agno), the
        # execution result is itself the verdict — this sink ships that
        # pre-computed verdict to the synthesis_worker's router for the
        # dream-boundary OracleVerdictBatch flush (coverage), with no re-exec.
        self._companion_verdict_sink = companion_verdict_sink
        # Phase 9 (P9.D): callable `(skill_id: str, success: bool) -> None`.
        # Invoked when a delegated skill's tool call resolves, so the
        # synthesis_worker can run the P8 utility loop (increment_success/
        # failure) + the SkillFailureTracker (repair-fork-on-failure, §9.3).
        # Also closes the P8 gap: increment_success/failure had no live caller.
        self._skill_outcome_sink = skill_outcome_sink

    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    def oracle(self) -> Optional[TruthOraclePlug]:
        """Subclasses override to expose a doubling-as-oracle surface.
        Default: pure tool, no oracle surface. Per arch §11.1 / §11.3,
        ``coding_sandbox`` and ``x_research`` are the doublers; the
        other two (events_teacher, knowledge) are pure tools."""
        return None

    def invoke(self, call: ToolCall) -> ToolResult:
        """Common invocation wrapper: execute → emit procedural TX →
        optionally trigger companion oracle verdict → return ToolResult.
        """
        t0 = time.perf_counter()
        try:
            raw = self._execute(call) or {}
        except Exception as exc:
            logger.exception(
                "[%s] _execute raised on tool_id=%s", self.__class__.__name__, self.tool_id,
            )
            raw = {
                "success": False,
                "result_summary": f"{type(exc).__name__}: {exc}"[:512],
                "exception": str(exc),
            }
        latency_ms = int((time.perf_counter() - t0) * 1000)

        success = bool(raw.get("success", False))
        result_summary = str(raw.get("result_summary", ""))
        result_full = raw.get("result_full_payload")
        exception = raw.get("exception")
        result_full_hash: Optional[str] = None
        if isinstance(result_full, (bytes, bytearray, str)):
            payload_bytes = (
                result_full if isinstance(result_full, (bytes, bytearray))
                else str(result_full).encode("utf-8")
            )
            result_full_hash = hashlib.sha256(bytes(payload_bytes)).hexdigest()

        # Operator-closure C2 (W7 / INV-Syn-15): a tool that IS its own truth
        # oracle (oracle() not None — e.g. coding_sandbox) verified the outcome
        # by executing it. That execution result is an oracle verdict, so the
        # tool-call TX is scored_by="oracle" immediately (no separate plug run,
        # no assertion required). Pure tools (events_teacher/knowledge) stay
        # scored_by=None (Phase 8 LLM-judge may score them later).
        _is_self_oracle = False
        try:
            _is_self_oracle = self.oracle() is not None
        except Exception:
            _is_self_oracle = False
        _scored_by = "oracle" if _is_self_oracle else None

        # Anchor the procedural TX BEFORE the companion oracle path — even
        # if the oracle trigger fails, the tool-call TX is on the chain.
        try:
            parent_tx = self._writer.write_tool_call(
                tool_id=self.tool_id,
                args=dict(call.args or {}),
                success=success,
                result_summary=result_summary,
                result_full_hash=result_full_hash,
                latency_ms=raw.get("latency_ms", latency_ms),
                scored_by=_scored_by,
                parent_chat_tx=call.parent_chat_tx,
                parent_goal=call.parent_goal,
                parent_skill_id=call.parent_skill_id,
            )
        except Exception:
            logger.exception("[%s] write_tool_call failed", self.__class__.__name__)
            parent_tx = None

        # Operator-closure C2 (W7): buffer the self-oracle's pre-computed verdict
        # for the dream-boundary OracleVerdictBatch flush (→ §A.6 coverage). In
        # the synthesis_worker process the router is in-hand; in agno (chat-time)
        # the verdict ships over the bus via companion_verdict_sink. No re-exec.
        if _is_self_oracle and parent_tx is not None:
            _verdict_str = "true" if success else "false"
            _ev_ref = result_full_hash or (result_summary[:64] if result_summary else "executed")
            try:
                if self._router is not None:
                    self._router.record_companion_verdict(
                        parent_tool_call_tx=parent_tx, oracle_id=self.tool_id,
                        verdict=_verdict_str, evidence_ref=_ev_ref,
                        latency_ms=latency_ms, fork="procedural",
                        # EEL B1 — name the goal+tool so the verdict can become a
                        # (outcome, task-shape) skill-score event at flush.
                        parent_goal=call.parent_goal, tool_id=self.tool_id,
                    )
                elif self._companion_verdict_sink is not None:
                    self._companion_verdict_sink(
                        parent_tool_call_tx=parent_tx, oracle_id=self.tool_id,
                        verdict=_verdict_str, evidence_ref=_ev_ref,
                        latency_ms=latency_ms,
                        parent_goal=call.parent_goal, tool_id=self.tool_id,
                    )
            except Exception:
                logger.exception(
                    "[%s] self-oracle companion verdict record failed",
                    self.__class__.__name__)

        # Phase 9 P9.D — feed the skill-outcome loop when this tool call was a
        # delegated compiled skill. Closes the P8 utility loop (increment_*) +
        # drives the SkillFailureTracker (repair-fork-on-failure, §9.3). Soft.
        if call.parent_skill_id and self._skill_outcome_sink is not None:
            try:
                self._skill_outcome_sink(call.parent_skill_id, success)
            except Exception:
                logger.exception(
                    "[%s] skill_outcome_sink raised", self.__class__.__name__,
                )

        # Trigger companion verdict if the tool exposes an oracle surface AND
        # the subclass supplies a `_companion_claim_for(call, raw)` mapping.
        if self._router is not None and parent_tx is not None:
            companion_claim = self._companion_claim_for(call, raw)
            if companion_claim is not None:
                try:
                    self._router.verify(
                        companion_claim,
                        parent_tool_call_tx=parent_tx,
                        parent_tool_call_fork="procedural",
                        # EEL B1 — name the goal+tool for the skill-score capture.
                        parent_goal=call.parent_goal, tool_id=self.tool_id,
                    )
                except Exception:
                    logger.exception(
                        "[%s] companion oracle verify raised", self.__class__.__name__,
                    )

        return ToolResult(
            tool_id=self.tool_id,
            success=success,
            result_summary=result_summary,
            result_full_hash=result_full_hash,
            latency_ms=raw.get("latency_ms", latency_ms),
            exception=exception,
        )

    # ── overridable hooks ──────────────────────────────────────────────

    def _execute(self, call: ToolCall) -> dict:
        raise NotImplementedError("ToolPlugBase subclasses must implement _execute")

    def _companion_claim_for(self, call: ToolCall, raw: dict) -> Optional[OracleClaim]:
        """Subclasses override to map a tool invocation to an OracleClaim
        for the companion verdict path. Return None to suppress (default).

        The base wires the router call if the subclass returns a claim;
        the verdict rides the parent tool-call TX's fork (`procedural`).
        """
        return None


__all__ = ("ToolPlugBase",)
