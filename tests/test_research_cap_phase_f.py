"""Phase F (RFP_research_resilience_and_knowledge_unification §7.F) — offline.

Verifies the unified agent research-tool path + the per-turn invocation cap:
  1. The cap helpers (_research_cap_reset / _research_cap_allow) bound calls per
     turn and reset cleanly (fail-open).
  2. The RESEARCH reflex (reflex_executors.execute_research) routes through the
     shared _dispatch_knowledge_research helper and respects the cap.
  3. The agno research TOOL (agno_tools.create_tools → research) routes through
     the SAME helper, respects the SHARED cap, and returns a steering message
     when capped.

Run isolated (TorchRL mmap Bus): python -m pytest tests/test_research_cap_phase_f.py -v -p no:anchorpy
"""
import asyncio

import pytest

import titan_hcl.modules.agno_hooks as agno_hooks
from titan_hcl.modules.agno_hooks import (
    _RESEARCH_TURN_CAP, _research_cap_allow, _research_cap_reset)


class _StubMemory:
    def __init__(self):
        self.topics = []

    def add_research_topic(self, t):
        self.topics.append(t)


class _StubPlugin:
    """Minimal plugin: truthy sage_researcher (the cap/route guards), a memory
    stub, and the source-extractor the tool calls on a hit."""
    def __init__(self):
        self.sage_researcher = object()
        self.memory = _StubMemory()
        self.synthesis_tool_plugs = {}      # → legacy (dispatch) path, not KnowledgeTool
        self._last_research_sources = []

    def _extract_sources_from_findings(self, findings):
        return []


# ── 1. Cap helpers ──────────────────────────────────────────────────────────

def test_cap_allows_exactly_N_then_blocks():
    p = _StubPlugin()
    _research_cap_reset(p)
    seq = [_research_cap_allow(p) for _ in range(_RESEARCH_TURN_CAP + 2)]
    assert seq == [True] * _RESEARCH_TURN_CAP + [False, False]


def test_cap_reset_reopens_the_turn():
    p = _StubPlugin()
    _research_cap_reset(p)
    for _ in range(_RESEARCH_TURN_CAP):
        _research_cap_allow(p)
    assert _research_cap_allow(p) is False
    _research_cap_reset(p)                       # next chat turn
    assert _research_cap_allow(p) is True


def test_cap_fail_open_on_unsettable_plugin():
    """A plugin that can't hold the counter attr must never block research."""
    class _Frozen:
        __slots__ = ()                           # can't set _research_call_count
    assert _research_cap_allow(_Frozen()) is True


# ── 2. RESEARCH reflex routes through dispatch + respects cap ────────────────

def _build_research_reflex(plugin):
    from titan_hcl.logic.reflexes import ReflexCollector, ReflexType
    from titan_hcl.logic.reflex_executors import register_reflex_executors
    collector = ReflexCollector({"fire_threshold": 0.15, "session_cooldown": 1.0})
    register_reflex_executors(collector, plugin)
    return collector._executors[ReflexType.RESEARCH]


def test_reflex_routes_through_dispatch_and_caps(monkeypatch):
    calls = {"n": 0}

    async def _fake_dispatch(plugin, gap):
        calls["n"] += 1
        return "[SAGE_RESEARCH_FINDINGS]: %s answer" % gap

    monkeypatch.setattr(agno_hooks, "_dispatch_knowledge_research", _fake_dispatch)

    p = _StubPlugin()
    _research_cap_reset(p)
    reflex = _build_research_reflex(p)

    async def _run():
        r1 = await reflex({"message": "ephemeral"})
        r2 = await reflex({"message": "weather berlin"})
        r3 = await reflex({"message": "third one"})   # over cap
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(_run())
    assert r1.get("success") and "answer" in r1["findings"]   # routed via dispatch
    assert r2.get("success")
    assert r3.get("error") == "research cap reached"
    assert calls["n"] == _RESEARCH_TURN_CAP                    # dispatch NOT called the 3rd time
    assert p.memory.topics == ["ephemeral", "weather berlin"]


# ── 3. agno research TOOL routes through dispatch + shares the cap ───────────

def _get_research_tool(plugin):
    from titan_hcl.modules.agno_tools import create_tools
    tools = create_tools(plugin)
    for t in tools:
        fn = getattr(t, "entrypoint", None) or getattr(t, "__wrapped__", None) or t
        if getattr(fn, "__name__", "") == "research" or getattr(t, "__name__", "") == "research":
            return fn if callable(fn) else t
    # fallback: agno may wrap as Function with .name
    for t in tools:
        if getattr(t, "name", "") == "research":
            return getattr(t, "entrypoint", t)
    raise AssertionError("research tool not found in create_tools output")


def test_tool_routes_through_dispatch_and_caps(monkeypatch):
    calls = {"n": 0}

    async def _fake_dispatch(plugin, gap):
        calls["n"] += 1
        return "[SAGE_RESEARCH_FINDINGS]: %s answer" % gap

    monkeypatch.setattr(agno_hooks, "_dispatch_knowledge_research", _fake_dispatch)

    p = _StubPlugin()
    _research_cap_reset(p)
    research = _get_research_tool(p)

    async def _run():
        a = await research("define ephemeral")
        b = await research("weather berlin")
        c = await research("third call")              # over cap → steering msg
        return a, b, c

    a, b, c = asyncio.run(_run())
    assert "answer" in a and "answer" in b
    assert "Research limit reached" in c
    assert calls["n"] == _RESEARCH_TURN_CAP


def test_tool_and_reflex_share_one_turn_budget(monkeypatch):
    """The cap is COMBINED: a tool call + a reflex call exhaust the same budget."""
    async def _fake_dispatch(plugin, gap):
        return "[SAGE_RESEARCH_FINDINGS]: ok"

    monkeypatch.setattr(agno_hooks, "_dispatch_knowledge_research", _fake_dispatch)

    p = _StubPlugin()
    _research_cap_reset(p)
    research = _get_research_tool(p)
    reflex = _build_research_reflex(p)

    async def _run():
        t1 = await research("q1")                     # 1
        x1 = await reflex({"message": "q2"})          # 2 → cap reached
        t2 = await research("q3")                      # 3 → blocked
        return t1, x1, t2

    t1, x1, t2 = asyncio.run(_run())
    assert "ok" in t1
    assert x1.get("success")
    assert "Research limit reached" in t2             # shared budget exhausted
