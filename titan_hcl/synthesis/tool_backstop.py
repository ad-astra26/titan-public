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
            "(The model narrated a tool call without emitting one; Titan "
            "re-executed it deterministically to verify the result.)\n"
            if corrective else
            "### Verified Result — coding_sandbox\n"
        )
        return (
            f"{header}"
            f"Executed in the sandbox:\n```python\n{self.code}\n```\n"
            f"Output: {self.result_summary or '(no stdout)'}\n"
            f"Verdict: {status}\n\n"
        )

    def activity(self, *, phase: str) -> Optional[dict]:
        """Structured tool-activity descriptor for the CHAT_RESPONSE payload so
        the frontend / comma channel can show 'Titan verified this via its
        sandbox' (and explain the extra latency). None when nothing executed."""
        if not self.executed:
            return None
        return {
            "tool": "coding_sandbox",
            "phase": phase,                      # "pre" (forced) | "post" (salvage)
            "executed": True,
            "success": self.success,
            "verdict": self.verdict,             # "true" | "false"
            "summary": (self.result_summary or "")[:200],
            "salvaged": phase == "post",         # model missed the call → we ran it
        }


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


def _e1_recipe_replay(plugin: Any, prompt: str, cfg: dict) -> str:
    """§7.E (E.1) — symbolic recipe replay BEFORE the LLM router.

    If a high-confidence compute composite matched THIS prompt (stashed by the agno
    decide via `OuterCompositeReader.match`) and carries a templatized recipe, bind
    this prompt's numeric params and return runnable code → the router is skipped
    (zero LLM derivation; the cheap reuse). Any miss / below-floor / non-numeric /
    ambiguity ⇒ '' ⇒ the caller falls through to the router (the LLM fallback). The
    oracle re-verify (the synchronous verdict at invoke) is the final safety net.

    SAFE: substitutes ONLY sanitized numeric literals (recipe_template.bind); a
    literal (non-templatized) recipe is NOT replayed without captured params (→
    fallback) — we never run a recipe whose params we cannot verify match."""
    try:
        ecfg = (cfg.get("e_compose", {}) or {})
        if not ecfg.get("enabled", True):
            return ""
        floor = float(ecfg.get("floor", 0.85))
        match = getattr(plugin, "_last_composite_match", None)
        if not match or float(match.get("score", 0.0) or 0.0) < floor:
            return ""
        if str(match.get("action", "")) not in ("tool", "skill_delegate"):
            return ""  # compute lane only; E.3 (research) is a separate path
        recipe_raw = str(match.get("recipe_json", "") or "")
        if not recipe_raw:
            return ""
        import json
        recipe = json.loads(recipe_raw)
        if str(recipe.get("tool_id", "")) != "coding_sandbox":
            return ""
        if "code_template" not in recipe:
            return ""  # literal recipe → no verifiable param match → LLM fallback
        from titan_hcl.synthesis.recipe_template import (
            bind, extract_numeric_params)
        code = bind(recipe, extract_numeric_params(prompt or ""))
        if code:
            logger.info(
                "[ToolBackstop][E.1] composite recipe replay (templated, score=%.3f) "
                "for goal_class=%s — LLM router skipped",
                float(match.get("score", 0.0) or 0.0), match.get("goal_class", ""))
            return code
        return ""
    except Exception as e:  # noqa: BLE001 — replay must never break the backstop
        logger.debug("[ToolBackstop][E.1] recipe replay skipped (soft): %s", e)
        return ""


def _invoke_sandbox(plug: Any, code: str, parent_goal: Optional[str] = None,
                    decision=None):
    """Run the coding_sandbox ToolPlug synchronously (called via to_thread).
    The plug's invoke() anchors the tool_call_tx with scored_by='oracle'.

    EEL B1 (D-SPEC-153 / INV-Syn-29) — `parent_goal` names the goal this
    AUTONOMOUS tool-use serves so the oracle-verified verdict forms an
    OUTCOME-keyed (oracle_id, goal_class) skill. WITHOUT it the companion
    verdict carries parent_goal=None and the OracleRouter flush DROPS it
    (`if not e.parent_goal: continue`, oracle_router.py:570) → no positive
    skill ever forms from the backstop path (the dominant autonomous path;
    the 2026-06-09 soak found 0 promoted despite oracle coverage=1.0). The
    agent-explicit agno tool (agno_tools.py) already threads this from the
    `goal` buffer; the backstop sources it from the turn's prompt — the
    same user request, so both paths key the same goal_class."""
    from titan_hcl.synthesis.plugs import ToolCall
    _feats, _action = (decision if decision else (None, None))
    return plug.invoke(ToolCall(tool_id="coding_sandbox", args={"code": code},
                                parent_goal=parent_goal,
                                decision_features=_feats, decision_action=_action))


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

    # 1) Gate. When the OuterMetaPolicy is enabled it is the ROUTER — the PRE
    #    backstop must honor its choice (autonomy/emergence, §24.7): fire tool on
    #    a pre-LLM turn ONLY if the policy chose tool. The regex `requires_tool`
    #    no longer FORCES tool over the policy's non-tool choice — that hijacked
    #    routing + credit (every regex-matched conversational turn became a
    #    tool-use crediting `tool`) and re-collapsed the policy to always-tool
    #    (live-verified 2026-06-12). The POST phase is VERIFICATION (OVG salvage
    #    of numeric claims in the response — correctness), so it still fires on a
    #    regex/router signal regardless of the policy; the on-policy credit in
    #    synthesis_worker ensures such a verification does NOT credit `tool` when
    #    the policy chose a non-tool action. When OML is OFF (no decision), the
    #    legacy regex gate stands (back-compat).
    intent = detect_tool_intent(prompt, response)
    _dec = getattr(plugin, "_last_outer_decision", None)
    _oml_enabled = _dec is not None and len(_dec) == 2
    _policy_wants_tool = False
    if _oml_enabled:
        try:
            from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS
            _policy_wants_tool = (int(_dec[1]) == OUTER_ACTIONS.index("tool"))
        except Exception:  # noqa: BLE001
            _policy_wants_tool = False
    if phase == "pre" and _oml_enabled:
        # policy is the router on a live pre-LLM turn — fire iff it chose tool
        if not _policy_wants_tool:
            return BackstopResult(fired=False, reason="policy_non_tool")
    elif not intent.requires_tool and not _policy_wants_tool:
        return BackstopResult(fired=False, reason="no_intent")

    # §7.E (E.1) — symbolic recipe REPLAY first (pre-phase only): a matched compute
    #    composite's templatized recipe, bound to this prompt's params, skips the LLM
    #    router entirely (the cheap reuse). '' ⇒ fall through to the router below.
    code = _e1_recipe_replay(plugin, prompt, cfg) if phase == "pre" else ""

    # 2) fast-model router — reliable code on gated turns. The router also
    #    double-confirms need (suppresses a regex false-positive). Fall back to
    #    regex / response extraction when the router is unavailable or empty.
    #    SKIPPED when E.1 already produced code (the reuse win).
    router = _get_router(plugin, cfg) if not code else None
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
    # EEL B1 — the goal this autonomous tool-use serves (→ goal_class). The
    # prompt IS the user's request; falls back to the response text only if the
    # prompt is empty (post-phase salvage). None → unchanged drop behaviour.
    parent_goal = (prompt or "").strip() or (response or "").strip() or None
    # v1.1 — the OuterMetaPolicy decision that chose to fire this tool (set on the
    # plugin by the agno DECIDE block when the policy is enabled; None otherwise).
    # Carried so synthesis's verdict-time C1 capture trains the policy.
    decision = getattr(plugin, "_last_outer_decision", None)
    try:
        result = await asyncio.to_thread(_invoke_sandbox, plug, code, parent_goal,
                                         decision)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ToolBackstop] sandbox invoke failed (soft): %s", e)
        return BackstopResult(fired=True, executed=False, reason=f"exec_error:{e}",
                              code=code)

    success = bool(getattr(result, "success", False))
    summary = str(getattr(result, "result_summary", "") or "")[:512]
    # Log the ACTUAL code + result (not just lengths) so the kernel journal is a
    # faithful audit trail — proves the sandbox really executed the emitted code
    # (the fast-model router's code), not that the LLM invented the answer.
    logger.info(
        "[ToolBackstop] phase=%s EXECUTED coding_sandbox success=%s code_len=%d\n"
        "  CODE: %s\n  RESULT: %s",
        phase, success, len(code),
        (code[:400] + ("…" if len(code) > 400 else "")).replace("\n", "\\n"),
        summary[:256].replace("\n", "\\n"),
    )
    return BackstopResult(
        fired=True, executed=True, success=success,
        verdict="true" if success else "false",
        result_summary=summary, code=code,
    )
