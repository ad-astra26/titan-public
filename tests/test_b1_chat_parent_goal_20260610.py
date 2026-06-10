"""EEL B1 regression — the chat-path tool invocation MUST thread the turn's goal
as ``ToolCall.parent_goal`` (D-SPEC-153 / INV-Syn-29).

Root cause found by the 2026-06-09 soak: `agno_tools._invoke_tool_plug_sync`
built `ToolCall(...)` WITHOUT `parent_goal`, so every oracle-verified chat tool-use
carried `parent_goal=None` → the OracleRouter flush dropped it
(`if not e.parent_goal: continue`) → 0 positive skills formed despite oracle
coverage = 1.0. The fix sources the goal from the `goal` working-memory buffer
(written by the pre-LLM goal hook, INV-Syn-17) before agent.arun.

These tests pin BOTH halves:
  1. the chat-path ToolCall now carries the goal-buffer text as parent_goal, and
  2. the OracleRouter flush only forwards a score event when parent_goal is present
     (the guard the fix unblocks) — and the forwarded (oracle_id, goal_class) is right.
"""
import asyncio
import types

import pytest

from titan_hcl.modules import agno_tools
from titan_hcl.synthesis.oracle_router import OracleRouter


# ── 1. The fix: chat-path ToolCall threads the goal-buffer text ──────────────

class _FakeBufferCache:
    """Minimal BufferCache stub — `.get(chat_id, buffer_name)` returns a row dict."""
    def __init__(self, rows):
        self._rows = rows  # {(chat_id, buffer_name): {"content": ...}}

    def get(self, chat_id, buffer_name):
        return self._rows.get((chat_id, buffer_name))


class _CapturingPlug:
    """Stub ToolPlug — records the ToolCall it was invoked with."""
    def __init__(self):
        self.last_call = None

    def invoke(self, call):
        self.last_call = call
        return types.SimpleNamespace(
            success=True, result_summary="5040", exception=None)


def _find_tool(tools, name):
    for t in tools:
        if getattr(t, "__name__", None) == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[getattr(t,'__name__',t) for t in tools]}")


def test_chat_tool_threads_goal_buffer_as_parent_goal():
    plug = _CapturingPlug()
    goal_text = "Can you help me figure out the factorial of 7? Run it in your sandbox."
    plugin = types.SimpleNamespace(
        synthesis_tool_plugs={"coding_sandbox": plug},
        synthesis_buffer_cache=_FakeBufferCache(
            {("@jake:synth_rec_jake", "goal"): {"content": goal_text}}),
        _current_user_id="@jake",
        _current_session_id="synth_rec_jake",
    )
    tools = agno_tools.create_tools(plugin)
    coding_sandbox = _find_tool(tools, "coding_sandbox")

    out = asyncio.run(coding_sandbox(code="print(5040)"))
    assert out.startswith("OK")
    assert plug.last_call is not None, "plug was never invoked"
    # THE FIX: parent_goal carries the goal-buffer text (was None pre-fix → dropped).
    assert plug.last_call.parent_goal == goal_text


def test_chat_tool_parent_goal_none_when_goal_buffer_empty():
    """No goal buffered → parent_goal stays None (graceful, no regression/raise)."""
    plug = _CapturingPlug()
    plugin = types.SimpleNamespace(
        synthesis_tool_plugs={"coding_sandbox": plug},
        synthesis_buffer_cache=_FakeBufferCache({}),  # empty
        _current_user_id="@x", _current_session_id="s",
    )
    tools = agno_tools.create_tools(plugin)
    coding_sandbox = _find_tool(tools, "coding_sandbox")
    asyncio.run(coding_sandbox(code="print(1)"))
    assert plug.last_call.parent_goal is None


def test_chat_tool_no_buffer_cache_is_soft():
    """No buffer cache at all (early boot) → parent_goal None, tool still runs."""
    plug = _CapturingPlug()
    plugin = types.SimpleNamespace(
        synthesis_tool_plugs={"coding_sandbox": plug},
        synthesis_buffer_cache=None,
        _current_user_id="@x", _current_session_id="s",
    )
    tools = agno_tools.create_tools(plugin)
    coding_sandbox = _find_tool(tools, "coding_sandbox")
    asyncio.run(coding_sandbox(code="print(1)"))
    assert plug.last_call.parent_goal is None


# ── 2. The guard the fix unblocks: flush forwards iff parent_goal present ─────

class _RecordingWriter:
    """Captures write_oracle_verdict_batch; returns a fake anchor tx."""
    def write_oracle_verdict_batch(self, *, fork, merkle_root, entries):
        return "tx_" + merkle_root[:8]


def _router_with_sink():
    captured = []
    # flush_companion_batches only touches self._writer + the score-event sink;
    # gate/spend_store/balance_provider are unused on this path → dummy stubs.
    r = OracleRouter(
        gate=object(), spend_store=object(),
        outer_memory_writer=_RecordingWriter(),
        balance_provider=lambda: 1.0,
    )
    r.set_score_event_sink(lambda **kw: captured.append(kw))
    return r, captured


def test_flush_forwards_score_event_only_with_parent_goal():
    # WITH parent_goal → the sink fires with the right outcome key.
    r, captured = _router_with_sink()
    r.record_companion_verdict(
        parent_tool_call_tx="txA", oracle_id="coding_sandbox", verdict="true",
        evidence_ref="e", latency_ms=1, fork="procedural",
        parent_goal="compute the factorial of 7", tool_id="coding_sandbox")
    r.flush_companion_batches()
    assert len(captured) == 1
    assert captured[0]["oracle_id"] == "coding_sandbox"
    assert captured[0]["parent_goal"] == "compute the factorial of 7"
    assert captured[0]["tool_id"] == "coding_sandbox"


def test_flush_skips_score_event_without_parent_goal():
    # WITHOUT parent_goal (the pre-fix chat path) → dropped, no event (the bug).
    r, captured = _router_with_sink()
    r.record_companion_verdict(
        parent_tool_call_tx="txB", oracle_id="coding_sandbox", verdict="true",
        evidence_ref="e", latency_ms=1, fork="procedural",
        parent_goal="", tool_id="coding_sandbox")
    r.flush_companion_batches()
    assert captured == []
