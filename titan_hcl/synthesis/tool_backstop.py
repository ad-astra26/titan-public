"""titan_hcl.synthesis.tool_backstop — deterministic tool-invocation control plane.

2026-06-01 (Maker-greenlit). Closes the "synthesis oracle coverage = 0" gap:
the heavy agno chat context steers the LLM into *narrating* fake tool execution
instead of emitting real `tool_calls`, so `coding_sandbox` was never actually
invoked. The hooks ARE the Trinity — they decide and act; the LLM narrates.

Pipeline (shared by the PreHook force path and the OVG PostHook backstop):

    regex gate (tool_intent)        ~0ms, $0 — skips ~95% pure-conversation turns
        │ compute signal
        ▼
    fast-model router (tool_router) only on gated turns — emits exact Python
        │ (falls back to regex/response code extraction if the router is empty)
        ▼
    coding_sandbox ToolPlug.invoke   real execution → anchors tool_call_tx
        │                            (scored_by="oracle") → coverage moves
        ▼
    BackstopResult                   verdict text the caller injects / appends

Every path is soft-fail: a router hiccup, missing plug, or sandbox error never
breaks the chat — it just yields `executed=False` and the chat proceeds.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from titan_hcl.synthesis.tool_intent import detect_tool_intent, extract_executable
from titan_hcl.synthesis.tool_router import ToolRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackstopResult:
    fired: bool = False            # the regex gate detected tool intent
    executed: bool = False         # the sandbox actually ran
    success: bool = False          # the run succeeded (verdict "true")
    verdict: str = ""              # "true" | "false" | ""
    result_summary: str = ""       # sandbox stdout / error, truncated
    code: str = ""                 # the code that was run
    reason: str = ""               # why it didn't execute, when applicable

    def verdict_block(self, *, corrective: bool = False) -> str:
        """Render the verified-result block injected into the LLM context
        (pre) or appended to the response (post).

        `corrective=True` frames it as a backstop catch (PostHook), naming the
        tool so the UX can explain the extra latency."""
        if not self.executed:
            return ""
        status = "PASS ✅" if self.success else "FAIL ❌"
        header = (
            "### Tool Backstop — coding_sandbox\n"
            "(The model did not call the tool; Titan executed it deterministically.)\n"
            if corrective else
            "### Verified Result — coding_sandbox\n"
        )
        return (
            f"{header}"
            f"Executed in the sandbox:\n```python\n{self.code}\n```\n"
            f"Output: {self.result_summary or '(no stdout)'}\n"
            f"Verdict: {status}\n\n"
        )


def _cfg(plugin: Any) -> dict:
    full = getattr(plugin, "_full_config", {}) or {}
    return ((full.get("synthesis", {}) or {}).get("tool_backstop", {}) or {})


def _get_router(plugin: Any, cfg: dict) -> Optional[ToolRouter]:
    """Lazily build + cache the fast-model router on the plugin."""
    router = getattr(plugin, "_tool_router_cache", None)
    if router is not None:
        return router
    provider = getattr(plugin, "_inference_provider", None)
    if provider is None:
        return None
    try:
        router = ToolRouter(
            provider,
            model_class=str(cfg.get("router_model_class", "fast")),
            timeout=float(cfg.get("router_timeout_s", 12.0)),
        )
        plugin._tool_router_cache = router
        return router
    except Exception as e:  # noqa: BLE001
        logger.debug("[ToolBackstop] router build failed: %s", e)
        return None


def _invoke_sandbox(plug: Any, code: str):
    """Run the coding_sandbox ToolPlug synchronously (called via to_thread).
    The plug's invoke() anchors the tool_call_tx with scored_by='oracle'."""
    from titan_hcl.synthesis.plugs import ToolCall
    return plug.invoke(ToolCall(tool_id="coding_sandbox", args={"code": code}))


async def run_tool_backstop(
    plugin: Any,
    *,
    prompt: str,
    response: str = "",
    phase: str,
) -> BackstopResult:
    """Run the gate → router → execute pipeline for one chat turn.

    phase="pre"  — called from the PreHook (response="" — extract from prompt).
    phase="post" — called from the OVG PostHook (response set — the LLM's text;
                   used both as a fallback code source and to detect a
                   hallucinated-execution claim the prompt signal may have missed).

    Returns a BackstopResult; the caller decides how to inject/append the
    verdict block and emit any UX signal.
    """
    cfg = _cfg(plugin)
    if not cfg.get("enabled", True):
        return BackstopResult(reason="disabled")

    # 1) cheap regex gate — exits on ~95% of (pure-conversation) turns.
    intent = detect_tool_intent(prompt, response)
    if not intent.requires_tool:
        return BackstopResult(fired=False, reason="no_intent")

    # 2) fast-model router — reliable code on gated turns. The router also
    #    double-confirms need (suppresses a regex false-positive). Fall back to
    #    regex / response extraction when the router is unavailable or empty.
    code = ""
    router = _get_router(plugin, cfg)
    if router is not None:
        try:
            decision = await router.route(prompt or response)
            if decision.needs_tool:
                code = decision.code
            elif not response:
                # Router says no tool needed and we have no response to mine —
                # trust it over the regex gate (avoids spurious sandbox runs).
                return BackstopResult(fired=True, executed=False,
                                      reason="router_declined")
        except Exception as e:  # noqa: BLE001
            logger.debug("[ToolBackstop] router error (soft): %s", e)
    if not code:
        code = intent.code or (extract_executable(response) if response else "")
    if not code.strip():
        return BackstopResult(fired=True, executed=False, reason="no_code")

    # 3) execute through the already-wired coding_sandbox ToolPlug.
    plugs = getattr(plugin, "synthesis_tool_plugs", {}) or {}
    plug = plugs.get("coding_sandbox")
    if plug is None:
        logger.warning("[ToolBackstop] coding_sandbox plug not wired — "
                       "cannot execute (phase=%s)", phase)
        return BackstopResult(fired=True, executed=False, reason="plug_not_wired",
                              code=code)
    try:
        result = await asyncio.to_thread(_invoke_sandbox, plug, code)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ToolBackstop] sandbox invoke failed (soft): %s", e)
        return BackstopResult(fired=True, executed=False, reason=f"exec_error:{e}",
                              code=code)

    success = bool(getattr(result, "success", False))
    summary = str(getattr(result, "result_summary", "") or "")[:512]
    logger.info(
        "[ToolBackstop] phase=%s EXECUTED coding_sandbox success=%s "
        "code_len=%d summary=%r", phase, success, len(code), summary[:120],
    )
    return BackstopResult(
        fired=True, executed=True, success=success,
        verdict="true" if success else "false",
        result_summary=summary, code=code,
    )
