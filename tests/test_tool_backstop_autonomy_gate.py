"""Tool-backstop autonomy gate (§24.7, 2026-06-12).

When the OuterMetaPolicy is enabled it is the ROUTER: the PRE backstop must fire
tool ONLY if the policy chose tool — the regex `requires_tool` no longer FORCES
tool over the policy's non-tool choice (that hijacked routing + credit and
re-collapsed the policy to always-tool, live-verified). The POST phase stays a
correctness verification (fires on a regex/router signal regardless of policy).
When OML is off, the legacy regex gate stands (back-compat)."""
import asyncio

from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS
from titan_hcl.synthesis.tool_backstop import run_tool_backstop

TOOL = OUTER_ACTIONS.index("tool")
DIRECT = OUTER_ACTIONS.index("direct")
COMPUTE_PROMPT = "Compute the factorial of 12 and print the result."
CHAT_PROMPT = "Tell me about your sense of sovereignty."


class _Plugin:
    def __init__(self, decision=None):
        self._full_config = {"synthesis": {"tool_backstop": {"enabled": True}}}
        self._last_outer_decision = decision  # (features, action) or None


def _run(plugin, prompt, phase, response=""):
    return asyncio.run(
        run_tool_backstop(plugin, prompt=prompt, response=response, phase=phase))


def test_pre_oml_nontool_does_not_fire_even_on_compute_prompt():
    """The autonomy fix: policy chose `direct` → PRE backstop does NOT fire tool,
    even though the regex would detect compute intent. (No routing hijack.)"""
    p = _Plugin(decision=([0.0] * 30, DIRECT))
    r = _run(p, COMPUTE_PROMPT, "pre")
    assert r.fired is False
    assert r.reason == "policy_non_tool"


def test_pre_oml_tool_proceeds_past_gate():
    """Policy chose tool → PRE backstop proceeds past the gate (not short-circuited
    as policy_non_tool / no_intent)."""
    p = _Plugin(decision=([0.0] * 30, TOOL))
    r = _run(p, CHAT_PROMPT, "pre")
    assert r.reason not in ("policy_non_tool", "no_intent")


def test_post_phase_is_verification_not_policy_gated():
    """POST is correctness verification: a non-compute response + non-tool policy
    → no_intent (nothing to verify), NOT the policy gate."""
    p = _Plugin(decision=([0.0] * 30, DIRECT))
    r = _run(p, CHAT_PROMPT, "post", response="Sovereignty means self-governance.")
    assert r.fired is False
    assert r.reason == "no_intent"   # not policy_non_tool — POST isn't policy-gated


def test_oml_off_falls_back_to_regex_gate():
    """No decision (OML off) → legacy regex gate (back-compat): a pure-chat prompt
    short-circuits as no_intent, not policy_non_tool."""
    p = _Plugin(decision=None)
    r = _run(p, CHAT_PROMPT, "pre")
    assert r.fired is False
    assert r.reason == "no_intent"
