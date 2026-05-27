"""Phase 6 — ToolPlug × 4 tests (§P6.I; INV-12).

Covers:
- ToolPlugBase: TX emission, router companion-verdict trigger, exception
  defense (subclass _execute raises → ToolResult.success=False, TX still
  anchored), oracle() default returns None
- CodingSandboxTool: real sandbox execution, capability list, oracle()
  returns CodingSandboxOracle sharing the same helper instance,
  _companion_claim_for fires only when expected_stdout/assertion present
- EventsTeacherTool: action dispatch, unsupported action, invoke_fn
  exception defense, oracle() returns None (pure tool)
- KnowledgeTool: mode dispatch, missing query / mode validation
- XResearchTool: 4 capability handlers with mocked gateway, oracle()
  returns XOracle sharing same gateway, defensive paths
- Procedural TX shape: every invocation produces a write_tool_call TX
  with scored_by="none" by default; tags include tool_id + scored_by
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue

import pytest

from titan_hcl.synthesis.oracle_gate import OracleGate, OracleGateConfig
from titan_hcl.synthesis.oracle_router import OracleRouter, OracleSpendStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.plugs import (
    OracleVerdict,
    ToolCall,
    ToolResult,
    TruthOraclePlug,
)
from titan_hcl.synthesis.tools.base import ToolPlugBase
from titan_hcl.synthesis.tools.coding_sandbox_tool import CodingSandboxTool
from titan_hcl.synthesis.tools.events_teacher_tool import EventsTeacherTool
from titan_hcl.synthesis.tools.knowledge_tool import KnowledgeTool
from titan_hcl.synthesis.tools.x_research_tool import XResearchTool


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def writer():
    q = Queue()
    w = OuterMemoryWriter(q, src="test_p6i")
    w._test_queue = q
    return w


# ─────────────────────────────────────────────────────────────────────────
# Fake gateway shared by XResearchTool tests
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FakeGateway:
    search_response: dict = field(default_factory=lambda: {"tweets": []})
    recent_response: dict = field(default_factory=lambda: {"tweets": []})
    post_response: dict = field(default_factory=lambda: {"status": "ok", "tweet_id": "x123"})
    search_calls: list = field(default_factory=list)
    recent_calls: list = field(default_factory=list)
    post_calls: list = field(default_factory=list)

    def search_tweets(self, query, query_type="Latest", count=20, *, api_key=""):
        self.search_calls.append({"query": query, "query_type": query_type, "count": count})
        return self.search_response

    def fetch_recent_tweets(self, user_name, count=10, *, api_key=""):
        self.recent_calls.append({"user_name": user_name, "count": count})
        return self.recent_response

    def post(self, *, text, api_key=""):
        self.post_calls.append({"text": text})
        return self.post_response


# ─────────────────────────────────────────────────────────────────────────
# ToolPlugBase — common machinery
# ─────────────────────────────────────────────────────────────────────────


class _DummyTool(ToolPlugBase):
    tool_id = "dummy"
    _capabilities = ("a", "b")

    def __init__(self, *, writer, router=None, behavior="ok"):
        super().__init__(writer=writer, router=router)
        self._behavior = behavior

    def _execute(self, call):
        if self._behavior == "raise":
            raise RuntimeError("kaboom")
        if self._behavior == "fail":
            return {"success": False, "result_summary": "execute returned failure"}
        return {
            "success": True,
            "result_summary": "ok",
            "result_full_payload": "FULL_PAYLOAD",
        }


def test_base_invoke_emits_procedural_tool_call_tx(writer):
    t = _DummyTool(writer=writer)
    r = t.invoke(ToolCall(tool_id="dummy", args={"x": 1}))
    assert isinstance(r, ToolResult)
    assert r.tool_id == "dummy"
    assert r.success is True
    assert r.result_full_hash is not None  # CAS hashed full payload

    msg = writer._test_queue.get_nowait()
    payload = msg["payload"]
    assert payload["fork"] == "procedural"
    assert payload["thought_type"] == "tool_call"
    assert "tool_call" in payload["tags"]
    assert "tool:dummy" in payload["tags"]
    assert "scored_by:none" in payload["tags"]
    assert payload["content"]["tool_id"] == "dummy"
    assert payload["content"]["scored_by"] is None


def test_base_invoke_exception_collapsed_to_failed_result(writer):
    t = _DummyTool(writer=writer, behavior="raise")
    r = t.invoke(ToolCall(tool_id="dummy", args={}))
    assert r.success is False
    assert r.exception is not None
    assert "kaboom" in r.exception
    # TX still anchored on procedural fork
    msg = writer._test_queue.get_nowait()
    assert msg["payload"]["thought_type"] == "tool_call"
    assert msg["payload"]["content"]["success"] is False


def test_base_invoke_fail_result_propagates(writer):
    t = _DummyTool(writer=writer, behavior="fail")
    r = t.invoke(ToolCall(tool_id="dummy", args={}))
    assert r.success is False
    assert r.result_summary == "execute returned failure"


def test_base_capabilities_list(writer):
    t = _DummyTool(writer=writer)
    assert t.capabilities() == ["a", "b"]


def test_base_oracle_default_none(writer):
    t = _DummyTool(writer=writer)
    assert t.oracle() is None


def test_base_invoke_companion_path_fires_when_subclass_supplies_claim(writer, tmp_path):
    """When subclass overrides _companion_claim_for, the router is called
    with parent_tool_call_tx set so the verdict rides the tool-call TX
    per INV-Syn-12 batched companion routing."""
    import duckdb

    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    spend = OracleSpendStore(conn)
    gate = OracleGate(OracleGateConfig(daily_sol_budget={}))
    router = OracleRouter(
        gate=gate, spend_store=spend, outer_memory_writer=writer,
        balance_provider=lambda: 10.0,
    )

    # Register a fake plug to handle the companion claim domain.
    class FakeCompanionPlug:
        oracle_id = "fake_companion"
        cost_class = "free"

        def can_handle(self, d):
            return d == "code_correctness"

        def verify(self, claim):
            return OracleVerdict(
                oracle_id="fake_companion",
                verdict="true",
                evidence_ref="ev",
                cost=0.0,
                latency_ms=1,
                ts=time.time(),
            )

    router.register(FakeCompanionPlug())

    class CompanionTool(_DummyTool):
        def _companion_claim_for(self, call, raw):
            from titan_hcl.synthesis.plugs import OracleClaim
            return OracleClaim(domain="code_correctness", payload={"code": "x"})

    t = CompanionTool(writer=writer, router=router)
    t.invoke(ToolCall(tool_id="dummy", args={}))

    # Router buffered companion verdict (not yet flushed → no batch TX yet)
    assert router.companion_buffer_size() == 1
    # Tool-call TX was written
    msg = writer._test_queue.get_nowait()
    assert msg["payload"]["thought_type"] == "tool_call"

    conn.close()


# ─────────────────────────────────────────────────────────────────────────
# CodingSandboxTool
# ─────────────────────────────────────────────────────────────────────────


def test_coding_sandbox_tool_executes_real_sandbox(writer):
    t = CodingSandboxTool(writer=writer)
    r = t.invoke(ToolCall(tool_id="coding_sandbox", args={"code": "print(2 + 2)"}))
    assert r.success is True
    assert "4" in r.result_summary


def test_coding_sandbox_tool_missing_code_arg_fails(writer):
    t = CodingSandboxTool(writer=writer)
    r = t.invoke(ToolCall(tool_id="coding_sandbox", args={}))
    assert r.success is False
    assert "no code provided" in r.result_summary


def test_coding_sandbox_tool_ast_rejection_fails(writer):
    t = CodingSandboxTool(writer=writer)
    r = t.invoke(
        ToolCall(tool_id="coding_sandbox", args={"code": "import os\nprint('x')"})
    )
    assert r.success is False
    assert "AST rejected" in r.result_summary


def test_coding_sandbox_tool_doubles_as_oracle(writer):
    t = CodingSandboxTool(writer=writer)
    oracle = t.oracle()
    assert oracle is not None
    assert oracle.oracle_id == "coding_sandbox"
    # Shares same helper instance — arch §11.3 "two thin wrappers"
    assert oracle._helper is t._helper


def test_coding_sandbox_tool_companion_claim_fires_only_when_expectation_present(writer):
    t = CodingSandboxTool(writer=writer)
    # No expectation → None
    assert t._companion_claim_for(
        ToolCall(tool_id="coding_sandbox", args={"code": "print(1)"}),
        raw={},
    ) is None
    # expected_stdout → claim
    claim = t._companion_claim_for(
        ToolCall(
            tool_id="coding_sandbox",
            args={"code": "print(2 + 2)", "expected_stdout": "4"},
        ),
        raw={},
    )
    assert claim is not None
    assert claim.domain == "code_correctness"
    assert claim.payload["expected_stdout"] == "4"


def test_coding_sandbox_tool_capabilities(writer):
    t = CodingSandboxTool(writer=writer)
    assert "code_execution" in t.capabilities()


# ─────────────────────────────────────────────────────────────────────────
# EventsTeacherTool
# ─────────────────────────────────────────────────────────────────────────


def test_events_teacher_tool_dispatches_to_invoke_fn(writer):
    calls = []

    def fake_invoke(action, payload):
        calls.append((action, payload))
        return {"success": True, "result_summary": f"action={action} done"}

    t = EventsTeacherTool(writer=writer, invoke_fn=fake_invoke)
    r = t.invoke(
        ToolCall(
            tool_id="events_teacher",
            args={"action": "distill_event", "event_id": "x1"},
        )
    )
    assert r.success is True
    assert calls[0][0] == "distill_event"


def test_events_teacher_tool_unsupported_action_fails(writer):
    t = EventsTeacherTool(writer=writer)
    r = t.invoke(
        ToolCall(tool_id="events_teacher", args={"action": "bogus_action"})
    )
    assert r.success is False
    assert "unsupported action" in r.result_summary


def test_events_teacher_tool_missing_action_fails(writer):
    t = EventsTeacherTool(writer=writer)
    r = t.invoke(ToolCall(tool_id="events_teacher", args={}))
    assert r.success is False
    assert "missing 'action'" in r.result_summary


def test_events_teacher_tool_invoke_fn_raises_collapsed_to_failed(writer):
    def bad(action, payload):
        raise RuntimeError("teacher down")

    t = EventsTeacherTool(writer=writer, invoke_fn=bad)
    r = t.invoke(
        ToolCall(tool_id="events_teacher", args={"action": "distill_event"})
    )
    assert r.success is False
    assert "teacher down" in r.result_summary


def test_events_teacher_tool_default_invoke_fn_yields_unconfigured(writer):
    t = EventsTeacherTool(writer=writer)
    r = t.invoke(
        ToolCall(tool_id="events_teacher", args={"action": "distill_event"})
    )
    assert r.success is False
    assert "unconfigured" in r.result_summary


def test_events_teacher_tool_oracle_is_none(writer):
    t = EventsTeacherTool(writer=writer)
    assert t.oracle() is None


# ─────────────────────────────────────────────────────────────────────────
# KnowledgeTool
# ─────────────────────────────────────────────────────────────────────────


def test_knowledge_tool_web_search_dispatches(writer):
    calls = []

    def fake(query, mode):
        calls.append((query, mode))
        return {"success": True, "result_summary": "found"}

    t = KnowledgeTool(writer=writer, invoke_fn=fake)
    r = t.invoke(
        ToolCall(tool_id="knowledge", args={"query": "x", "mode": "web_search"})
    )
    assert r.success is True
    assert calls[0] == ("x", "web_search")


def test_knowledge_tool_default_mode_is_web_search(writer):
    calls = []

    def fake(query, mode):
        calls.append((query, mode))
        return {"success": True, "result_summary": "ok"}

    t = KnowledgeTool(writer=writer, invoke_fn=fake)
    t.invoke(ToolCall(tool_id="knowledge", args={"query": "x"}))
    assert calls[0][1] == "web_search"


def test_knowledge_tool_missing_query_fails(writer):
    t = KnowledgeTool(writer=writer)
    r = t.invoke(ToolCall(tool_id="knowledge", args={}))
    assert r.success is False
    assert "missing 'query'" in r.result_summary


def test_knowledge_tool_invalid_mode_fails(writer):
    t = KnowledgeTool(writer=writer)
    r = t.invoke(
        ToolCall(tool_id="knowledge", args={"query": "x", "mode": "bogus"})
    )
    assert r.success is False
    assert "unsupported mode" in r.result_summary


def test_knowledge_tool_oracle_is_none(writer):
    t = KnowledgeTool(writer=writer)
    assert t.oracle() is None


# ─────────────────────────────────────────────────────────────────────────
# XResearchTool
# ─────────────────────────────────────────────────────────────────────────


def test_x_research_tool_post_via_gateway(writer):
    gw = FakeGateway()
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(
        ToolCall(
            tool_id="x_research",
            args={"capability": "post", "text": "hello world"},
        )
    )
    assert r.success is True
    assert "tweet_id=x123" in r.result_summary
    assert gw.post_calls[0]["text"] == "hello world"


def test_x_research_tool_post_missing_text_fails(writer):
    gw = FakeGateway()
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(ToolCall(tool_id="x_research", args={"capability": "post"}))
    assert r.success is False
    assert "missing 'text'" in r.result_summary


def test_x_research_tool_fetch_topic_via_gateway(writer):
    gw = FakeGateway(search_response={"tweets": [{"id": "1"}, {"id": "2"}]})
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(
        ToolCall(
            tool_id="x_research",
            args={"capability": "fetch_topic", "topic": "AI"},
        )
    )
    assert r.success is True
    assert "2 tweets" in r.result_summary


def test_x_research_tool_fetch_account_via_gateway(writer):
    gw = FakeGateway(recent_response={"tweets": [{"id": "1"}]})
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(
        ToolCall(
            tool_id="x_research",
            args={"capability": "fetch_account", "handle": "@alice"},
        )
    )
    assert r.success is True
    assert "@alice" in r.result_summary
    assert gw.recent_calls[0]["user_name"] == "alice"


def test_x_research_tool_fetch_thread_via_gateway(writer):
    gw = FakeGateway(search_response={"tweets": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}]})
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(
        ToolCall(
            tool_id="x_research",
            args={"capability": "fetch_thread", "thread_root_id": "root123"},
        )
    )
    assert r.success is True
    assert "3 replies" in r.result_summary


def test_x_research_tool_unsupported_capability_fails(writer):
    t = XResearchTool(writer=writer, gateway=FakeGateway())
    r = t.invoke(
        ToolCall(tool_id="x_research", args={"capability": "delete_account"})
    )
    assert r.success is False
    assert "unsupported capability" in r.result_summary


def test_x_research_tool_doubles_as_oracle(writer):
    gw = FakeGateway()
    t = XResearchTool(writer=writer, gateway=gw)
    oracle = t.oracle()
    assert oracle is not None
    assert oracle.oracle_id == "x_api"
    # Shares same gateway instance — arch §11.3 "one process, two surfaces"
    assert oracle._gateway is gw


def test_x_research_tool_gateway_error_propagates_as_failed(writer):
    gw = FakeGateway(search_response={"status": "error", "message": "rate limited"})
    t = XResearchTool(writer=writer, gateway=gw)
    r = t.invoke(
        ToolCall(
            tool_id="x_research",
            args={"capability": "fetch_topic", "topic": "x"},
        )
    )
    assert r.success is False
    assert "gateway error" in r.result_summary


def test_x_research_tool_capabilities_list(writer):
    t = XResearchTool(writer=writer, gateway=FakeGateway())
    caps = t.capabilities()
    assert set(caps) == {"post", "fetch_thread", "fetch_topic", "fetch_account"}
