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
    ):
        self._writer = writer
        self._router = router

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

        # Anchor the procedural TX BEFORE the companion oracle path — even
        # if the oracle trigger fails, the tool-call TX is on the chain
        # and scored_by stays "none" (Phase 8 may score it later).
        try:
            parent_tx = self._writer.write_tool_call(
                tool_id=self.tool_id,
                args=dict(call.args or {}),
                success=success,
                result_summary=result_summary,
                result_full_hash=result_full_hash,
                latency_ms=raw.get("latency_ms", latency_ms),
                scored_by=None,  # set "oracle" if the companion path lands a verdict
                parent_chat_tx=call.parent_chat_tx,
                parent_goal=call.parent_goal,
                parent_skill_id=call.parent_skill_id,
            )
        except Exception:
            logger.exception("[%s] write_tool_call failed", self.__class__.__name__)
            parent_tx = None

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
