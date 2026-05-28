"""Phase 9 P9.D — ToolPlugBase skill-outcome sink wiring.

The invoke() wrapper feeds (skill_id, success) to the sink ONLY when the tool
call was a delegated compiled skill (parent_skill_id set). Closes the P8
increment_* gap + drives the SkillFailureTracker (repair-fork-on-failure)."""

from titan_hcl.synthesis.tools.base import ToolPlugBase
from titan_hcl.synthesis.plugs import ToolCall


class _FakeWriter:
    def write_tool_call(self, **kw):
        return "parent_tx_hash"


class _Tool(ToolPlugBase):
    tool_id = "test_tool"

    def __init__(self, *, outcome, **kw):
        super().__init__(**kw)
        self._outcome = outcome

    def _execute(self, call):
        return {"success": self._outcome, "result_summary": "ok"}


def test_sink_called_on_delegated_skill_success():
    seen = []
    t = _Tool(outcome=True, writer=_FakeWriter(),
              skill_outcome_sink=lambda sid, ok: seen.append((sid, ok)))
    t.invoke(ToolCall(tool_id="test_tool", args={}, parent_skill_id="skill_7"))
    assert seen == [("skill_7", True)]


def test_sink_called_on_delegated_skill_failure():
    seen = []
    t = _Tool(outcome=False, writer=_FakeWriter(),
              skill_outcome_sink=lambda sid, ok: seen.append((sid, ok)))
    t.invoke(ToolCall(tool_id="test_tool", args={}, parent_skill_id="skill_7"))
    assert seen == [("skill_7", False)]


def test_sink_not_called_without_parent_skill_id():
    seen = []
    t = _Tool(outcome=True, writer=_FakeWriter(),
              skill_outcome_sink=lambda sid, ok: seen.append((sid, ok)))
    t.invoke(ToolCall(tool_id="test_tool", args={}))
    assert seen == []


def test_sink_failure_is_soft():
    def _boom(sid, ok):
        raise RuntimeError("tracker down")
    t = _Tool(outcome=True, writer=_FakeWriter(), skill_outcome_sink=_boom)
    # must not raise
    res = t.invoke(ToolCall(tool_id="test_tool", args={}, parent_skill_id="s"))
    assert res.success is True


def test_no_sink_is_fine():
    t = _Tool(outcome=True, writer=_FakeWriter())
    res = t.invoke(ToolCall(tool_id="test_tool", args={}, parent_skill_id="s"))
    assert res.success is True
