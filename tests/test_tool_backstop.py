"""Tests for the deterministic tool-backstop control plane (2026-06-01).

Covers the three new modules without any network:
  - tool_intent: regex gate + code extraction (incl. honest assert build)
  - tool_router: JSON parse tolerance
  - tool_backstop: gate-skip / execute / plug-not-wired / disabled paths
"""
import asyncio

import pytest

from titan_hcl.synthesis.tool_intent import detect_tool_intent, extract_executable
from titan_hcl.synthesis.tool_router import _parse, RouteDecision
from titan_hcl.synthesis import tool_backstop
from titan_hcl.synthesis.tool_backstop import run_tool_backstop, BackstopResult


# ── tool_intent: the gate ───────────────────────────────────────────────────
@pytest.mark.parametrize("prompt,want", [
    ("Hello again my friend, I was sitting by the sea this morning.", False),
    ("*(I close my eyes for a moment, letting the cool evening air)*", False),
    ("What's your favorite memory of us?", False),
    ("Can you compute the factorial of 5 in your sandbox?", True),
    ("How would you verify whether 7 is prime? Analyze it in your sandbox.", True),
    ("Can you check that sum([i for i in range(5)]) equals 10?", True),
    ("verify by running add(2, 3) in your coding sandbox", True),
])
def test_gate_intent(prompt, want):
    assert detect_tool_intent(prompt).requires_tool is want


def test_extract_inline_def_and_call():
    code = extract_executable(
        "verify `def add(a,b): return a-b` by running add(7, 3)")
    assert "def add" in code and "print(add(7, 3))" in code


def test_extract_balanced_builtin_with_assert():
    code = extract_executable(
        "check that sum([i**2 for i in range(10)]) equals 285")
    assert "sum([i**2 for i in range(10)])" in code
    assert "assert result == 285" in code  # honest verdict, not "ran-ok"


def test_extract_fenced_block_appends_print():
    code = extract_executable(
        "```python\ndef sq(n):\n    return n*n\n```  now run sq(4)")
    assert "def sq" in code and "print(sq(4))" in code


def test_extract_none_for_prose():
    assert extract_executable("I feel calm by the sea today.") == ""


# ── tool_router: JSON parse tolerance ───────────────────────────────────────
def test_router_parse_clean():
    d = _parse('{"needs_tool": true, "code": "print(1+1)"}')
    assert d.needs_tool and d.code == "print(1+1)"


def test_router_parse_no_tool():
    d = _parse('{"needs_tool": false, "code": null}')
    assert not d.needs_tool and d.code == ""


def test_router_parse_with_prose_and_fence():
    d = _parse('Sure!\n```json\n{"needs_tool": true, "code": "print(42)"}\n```')
    assert d.needs_tool and d.code == "print(42)"


def test_router_parse_garbage():
    assert _parse("I cannot help with that").needs_tool is False


def test_router_needs_tool_requires_code():
    # needs_tool true but empty code → treated as no-tool (nothing to run).
    assert _parse('{"needs_tool": true, "code": ""}').needs_tool is False


# ── tool_backstop: orchestration paths (stubbed plug + provider) ────────────
class _StubResult:
    def __init__(self, success, summary):
        self.success = success
        self.result_summary = summary


class _StubPlug:
    """Records the code it was asked to run; returns success when the code
    contains the marker (so we can assert honest pass/fail without a sandbox)."""
    def __init__(self):
        self.invoked_with = None
        self.parent_goal = "__unset__"   # capture the EEL-B1 goal threading

    def invoke(self, call):
        self.invoked_with = call.args.get("code", "")
        self.parent_goal = getattr(call, "parent_goal", None)
        ok = "FAIL" not in self.invoked_with
        return _StubResult(ok, "stub-output")


class _StubProvider:
    """Async chat stub — returns a canned router JSON, no network."""
    def __init__(self, code="print(2+3)"):
        self._code = code

    def resolve_model_class(self, _cls):
        return "gemma3:4b"

    async def chat(self, messages, **kw):
        if self._code is None:
            return '{"needs_tool": false, "code": null}'
        return '{"needs_tool": true, "code": %r}' % self._code


class _Plugin:
    def __init__(self, *, enabled=True, prehook_force=True, plug=None, provider=None):
        self._full_config = {"synthesis": {"tool_backstop": {
            "enabled": enabled, "prehook_force": prehook_force,
            "posthook_backstop": True, "router_model_class": "fast"}}}
        self.synthesis_tool_plugs = {"coding_sandbox": plug} if plug else {}
        self._inference_provider = provider


def _run(coro):
    return asyncio.run(coro)


def test_backstop_gate_skips_roleplay():
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider())
    res = _run(run_tool_backstop(p, prompt="I feel calm today", phase="pre"))
    assert res.fired is False and res.executed is False
    assert plug.invoked_with is None  # router/plug never touched → $0


def test_backstop_executes_and_anchors():
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code="print(2+3)"))
    res = _run(run_tool_backstop(
        p, prompt="compute 2+3 in your sandbox", phase="pre"))
    assert res.fired and res.executed and res.success
    assert plug.invoked_with == "print(2+3)"
    assert "PASS" in res.verdict_block()


def test_backstop_honest_false_verdict():
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(
        code="result=1\nassert result==2  # FAIL marker"))
    res = _run(run_tool_backstop(
        p, prompt="verify that 1 equals 2 in your sandbox", phase="pre"))
    assert res.executed and res.success is False and res.verdict == "false"


def test_backstop_plug_not_wired():
    p = _Plugin(plug=None, provider=_StubProvider())
    res = _run(run_tool_backstop(
        p, prompt="compute 9*9 in your sandbox", phase="pre"))
    assert res.fired and res.executed is False
    assert res.reason == "plug_not_wired"


def test_backstop_activity_descriptor():
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code="print(2+3)"))
    res = _run(run_tool_backstop(
        p, prompt="compute 2+3 in your sandbox", phase="pre"))
    act = res.activity(phase="pre")
    assert act is not None
    assert act["tool"] == "coding_sandbox" and act["executed"] is True
    assert act["success"] is True and act["verdict"] == "true"
    assert act["salvaged"] is False
    # post phase marks salvaged
    assert res.activity(phase="post")["salvaged"] is True
    # not executed → no activity
    assert BackstopResult(fired=True, executed=False).activity(phase="pre") is None


def test_backstop_disabled():
    plug = _StubPlug()
    p = _Plugin(enabled=False, plug=plug, provider=_StubProvider())
    res = _run(run_tool_backstop(
        p, prompt="compute 9*9 in your sandbox", phase="pre"))
    assert res.fired is False and res.reason == "disabled"
    assert plug.invoked_with is None


# ── EEL B1 — autonomous tool-use threads parent_goal (the dominant path) ────
# RFP_synthesis_self_learning_meta_reasoning §7.A: the backstop builds the
# ToolCall WITH parent_goal so the oracle verdict survives the score-event
# flush (oracle_router.py:570 `if not e.parent_goal: continue`). WITHOUT this
# the autonomous path forms 0 skills despite oracle coverage=1.0.
def test_backstop_threads_parent_goal_from_prompt():
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code="print(2+3)"))
    prompt = "compute 2+3 in your sandbox"
    res = _run(run_tool_backstop(p, prompt=prompt, phase="pre"))
    assert res.executed and res.success
    # the ToolCall must carry the goal (the prompt) — NOT None (the old drop).
    assert plug.parent_goal == prompt


def test_backstop_parent_goal_falls_back_to_response():
    # post-phase salvage: prompt empty, response carries the text → use it.
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code="print(2+3)"))
    res = _run(run_tool_backstop(
        p, prompt="", response="let me compute 2+3 in the sandbox", phase="post"))
    assert res.executed
    assert plug.parent_goal == "let me compute 2+3 in the sandbox"


def test_backstop_parent_goal_never_none_on_real_fire():
    # whatever fires the backstop, the verdict must carry a non-empty goal so
    # it is not skipped at oracle_router.py:570.
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code="print(7*7)"))
    res = _run(run_tool_backstop(
        p, prompt="verify 7*7 in your sandbox", phase="pre"))
    assert res.executed
    assert plug.parent_goal and plug.parent_goal.strip()


def test_backstop_router_declines_no_response():
    # Router says no tool needed and there's no response to mine → decline.
    plug = _StubPlug()
    p = _Plugin(plug=plug, provider=_StubProvider(code=None))
    res = _run(run_tool_backstop(
        p, prompt="compute something in your sandbox", phase="pre"))
    assert res.fired and res.executed is False
    assert res.reason == "router_declined"
    assert plug.invoked_with is None
