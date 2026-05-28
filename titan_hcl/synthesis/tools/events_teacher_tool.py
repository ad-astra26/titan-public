"""Phase 6 — `events_teacher` ToolPlug (§P6.I).

Wraps the existing events_teacher X-event distillation surface as a
`ToolPlug`. Per arch §11.3 + SPEC §25.3 day-one set, events_teacher
is a pure tool (does NOT double as an oracle — x_oracle / P6.E covers
the X social-truth verification surface).

Capabilities: ``["distill_event", "fetch_recent_events"]``.

The events_teacher itself runs in its own worker; this ToolPlug is the
synthesis-engine-facing surface that routes invocations through
``OuterMemoryWriter.write_tool_call`` (procedural fork TX) per INV-12.

The events_teacher invocation is async + bus-mediated in production —
the ToolPlug takes a synchronous ``invoke_fn`` callable injected at
construction. Synthesis_worker boot wires this to a thin bus-RPC
client; unit tests inject a callable returning the canned distillation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from titan_hcl.synthesis.plugs import ToolCall
from titan_hcl.synthesis.tools.base import ToolPlugBase

logger = logging.getLogger(__name__)


# ``invoke_fn(action: str, payload: dict) -> dict`` —
# Returns: {"success": bool, "result_summary": str, "result_full_payload"?: str}
EventsTeacherInvokeFn = Callable[[str, dict], dict]


def _default_invoke_fn(action: str, payload: dict) -> dict:
    """Default no-op invoker — surfaces clearly when synthesis_worker
    forgot to inject the real bus-RPC client."""
    return {
        "success": False,
        "result_summary": (
            f"events_teacher tool unconfigured (action={action!r}); "
            "synthesis_worker must inject invoke_fn at boot"
        ),
    }


SUPPORTED_ACTIONS = frozenset({"distill_event", "fetch_recent_events"})


class EventsTeacherTool(ToolPlugBase):
    tool_id: str = "events_teacher"
    _capabilities = ("distill_event", "fetch_recent_events")

    def __init__(
        self,
        *,
        writer,
        router=None,
        invoke_fn: EventsTeacherInvokeFn = _default_invoke_fn,
    ):
        super().__init__(writer=writer, router=router)
        self._invoke_fn = invoke_fn

    def _execute(self, call: ToolCall) -> dict:
        action = str(call.args.get("action", "")).strip()
        if not action:
            return {"success": False, "result_summary": "missing 'action' arg"}
        if action not in SUPPORTED_ACTIONS:
            return {
                "success": False,
                "result_summary": (
                    f"unsupported action {action!r} (want one of "
                    f"{sorted(SUPPORTED_ACTIONS)})"
                ),
            }
        try:
            result = self._invoke_fn(action, dict(call.args))
        except Exception as exc:
            logger.exception("[events_teacher_tool] invoke_fn raised")
            return {
                "success": False,
                "result_summary": f"invoke_fn raised: {exc}",
                "exception": str(exc),
            }
        # Normalize the invoke_fn result into our standard dict shape.
        if not isinstance(result, dict):
            return {"success": False, "result_summary": "invoke_fn returned non-dict"}
        return {
            "success": bool(result.get("success", False)),
            "result_summary": str(result.get("result_summary", ""))[:512],
            "result_full_payload": result.get("result_full_payload"),
            "exception": result.get("exception"),
        }


__all__ = ("EventsTeacherTool", "SUPPORTED_ACTIONS")
