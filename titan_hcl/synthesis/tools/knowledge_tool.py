"""Phase 6 — `knowledge` ToolPlug (§P6.I).

Wraps the existing knowledge_router + StealthSage research path as a
`ToolPlug`. Pure tool — does NOT double as an oracle (web_api / P6.D
covers the web-fact verification surface).

Capabilities: ``["web_search", "doc_search"]``.

The knowledge_router invocation surface is async in production; the
ToolPlug takes a synchronous ``invoke_fn`` callable injected at
construction (synthesis_worker wires this to the sage_researcher /
knowledge_dispatcher path; unit tests stub).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from titan_hcl.synthesis.plugs import ToolCall
from titan_hcl.synthesis.tools.base import ToolPlugBase

logger = logging.getLogger(__name__)


# ``invoke_fn(query: str, mode: str) -> dict`` where mode ∈ {"web_search", "doc_search"}
# Returns: {"success": bool, "result_summary": str, "result_full_payload"?: str}
KnowledgeInvokeFn = Callable[[str, str], dict]


def _default_invoke_fn(query: str, mode: str) -> dict:
    return {
        "success": False,
        "result_summary": (
            f"knowledge tool unconfigured (mode={mode!r}); "
            "synthesis_worker must inject invoke_fn at boot"
        ),
    }


SUPPORTED_MODES = frozenset({"web_search", "doc_search"})


class KnowledgeTool(ToolPlugBase):
    tool_id: str = "knowledge"
    _capabilities = ("web_search", "doc_search")

    def __init__(
        self,
        *,
        writer,
        router=None,
        invoke_fn: KnowledgeInvokeFn = _default_invoke_fn,
        skill_outcome_sink=None,
    ):
        super().__init__(writer=writer, router=router,
                         skill_outcome_sink=skill_outcome_sink)
        self._invoke_fn = invoke_fn

    def _execute(self, call: ToolCall) -> dict:
        query = str(call.args.get("query", "")).strip()
        if not query:
            return {"success": False, "result_summary": "missing 'query' arg"}
        mode = str(call.args.get("mode", "web_search")).strip()
        if mode not in SUPPORTED_MODES:
            return {
                "success": False,
                "result_summary": (
                    f"unsupported mode {mode!r} (want one of {sorted(SUPPORTED_MODES)})"
                ),
            }
        try:
            result = self._invoke_fn(query, mode)
        except Exception as exc:
            logger.exception("[knowledge_tool] invoke_fn raised")
            return {
                "success": False,
                "result_summary": f"invoke_fn raised: {exc}",
                "exception": str(exc),
            }
        if not isinstance(result, dict):
            return {"success": False, "result_summary": "invoke_fn returned non-dict"}
        return {
            "success": bool(result.get("success", False)),
            "result_summary": str(result.get("result_summary", ""))[:512],
            "result_full_payload": result.get("result_full_payload"),
            "exception": result.get("exception"),
        }


__all__ = ("KnowledgeTool", "SUPPORTED_MODES")
