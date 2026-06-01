"""titan_hcl.synthesis.tool_router — lean fast-model tool-decision router.

Part of the 2026-06-01 tool-backstop fix (Maker-greenlit). The main agno chat
model, buried under the heavy narration context ("the Trinity decides, the LLM
narrates"), role-plays tool execution instead of emitting real `tool_calls`.
Rather than coercing that model, the deterministic control plane decides:

    cheap regex gate (tool_intent)  →  this fast-model router  →  plug execution

The router is invoked ONLY when the regex gate detects a compute/verify signal
(≈5% of turns), so the fast-model call's cost + latency never touch the ~95% of
pure-conversation turns. It does NOT use OpenAI tool-calling (which the heavy
path can't reliably trigger); instead it asks a small model to emit the exact
Python to run as structured JSON. We then execute that through the
`coding_sandbox` ToolPlug ourselves — deterministic, coverage-anchoring, and an
honest verdict (the truth oracle runs the model's own code).

The generated code runs in the AST-gated 256MB/30s sandbox (validate_code
blocks os/sys/subprocess/network), so model-authored code is safe to execute.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lean system prompt — NO persona, NO inner state. Just the routing job.
_ROUTER_SYSTEM = (
    "You are a deterministic tool router for a coding sandbox. Given a user "
    "message, decide whether answering it correctly requires RUNNING code or a "
    "computation (math, algorithm check, verifying a code snippet, numeric "
    "claim). You do NOT answer the question. You ONLY emit Python to run.\n\n"
    "Reply with a single JSON object and nothing else:\n"
    '  {\"needs_tool\": true, \"code\": \"<python that PRINTS the answer>\"}\n'
    '  {\"needs_tool\": false, \"code\": null}\n\n'
    "Rules for the code:\n"
    "- It MUST print() the result so the answer is observable.\n"
    "- If the user asserts a specific value (e.g. 'equals 285', 'is 144'), add "
    "an `assert result == <value>` so the run fails when the claim is false.\n"
    "- If the user gave a code snippet to verify, include it verbatim and call "
    "it with the stated arguments.\n"
    "- Use ONLY the standard library. No imports of os/sys/subprocess/network.\n"
    "- Keep it short and self-contained. No explanations, no markdown fences."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class RouteDecision:
    needs_tool: bool = False
    code: str = ""
    raw: str = ""  # raw model text, for telemetry/debug


def _parse(text: str) -> RouteDecision:
    """Parse the router's JSON reply. Tolerant of stray prose / fences."""
    if not text:
        return RouteDecision(raw=text)
    m = _JSON_RE.search(text)
    if not m:
        return RouteDecision(raw=text)
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # Fast models sometimes emit single quotes / trailing commas — one
        # forgiving retry before giving up.
        try:
            obj = json.loads(m.group(0).replace("'", '"').rstrip(", \n}") + "}")
        except Exception:
            return RouteDecision(raw=text)
    needs = bool(obj.get("needs_tool", False))
    code = obj.get("code") or ""
    code = code if isinstance(code, str) else ""
    return RouteDecision(needs_tool=needs and bool(code.strip()),
                         code=code.strip(), raw=text)


class ToolRouter:
    """Fast-model code router. Reuses the worker's inference provider; calls the
    'fast' model class (gemma3:4b by default). Soft-fails to a no-tool decision
    so the chat path never breaks on a router hiccup."""

    def __init__(self, provider, model_class: str = "fast", timeout: float = 12.0):
        self._provider = provider
        self._model_class = model_class
        self._timeout = float(timeout)

    async def route(self, prompt: str) -> RouteDecision:
        if not prompt or self._provider is None:
            return RouteDecision()
        try:
            model = None
            try:
                model = self._provider.resolve_model_class(self._model_class)
            except Exception:
                model = None
            text = await self._provider.chat(
                [
                    {"role": "system", "content": _ROUTER_SYSTEM},
                    {"role": "user", "content": prompt[:4000]},
                ],
                model=model,
                temperature=0.0,
                max_tokens=512,
                timeout=self._timeout,
            )
            decision = _parse(text or "")
            logger.info(
                "[ToolRouter] needs_tool=%s code_len=%d model=%s",
                decision.needs_tool, len(decision.code), model,
            )
            return decision
        except Exception as e:
            logger.warning("[ToolRouter] route failed (soft): %s", e)
            return RouteDecision()
