"""Phase C — oracle loop closure (W6/W7).

Locks the operator-closure additions:
  * a self-oracle tool (oracle() != None) scores its tool-call TX scored_by=
    "oracle" + ships a pre-computed companion verdict (router in-process, or
    companion_verdict_sink cross-process) — no plug re-execution.
  * OracleRouter.record_companion_verdict buffers → flush_companion_batches
    anchors an OracleVerdictBatch referencing the parent tool-call TX (coverage).
"""
import pytest

from titan_hcl.synthesis.plugs import ToolCall
from titan_hcl.synthesis.tools.base import ToolPlugBase


class _FakeWriter:
    def __init__(self):
        self.tool_calls = []
        self.batches = []

    def write_tool_call(self, **kw):
        self.tool_calls.append(kw)
        return "parenttx_" + str(len(self.tool_calls))

    def write_oracle_verdict_batch(self, **kw):
        self.batches.append(kw)
        return "batchtx_" + str(len(self.batches))


class _SelfOracleTool(ToolPlugBase):
    tool_id = "selforacle"
    _capabilities = ("x",)

    def _execute(self, call):
        return {"success": bool(call.args.get("ok", True)), "result_summary": "ran"}

    def oracle(self):
        return object()  # non-None → this tool IS its own truth oracle


class _PureTool(ToolPlugBase):
    tool_id = "pure"

    def _execute(self, call):
        return {"success": True, "result_summary": "ran"}


def test_self_oracle_sets_scored_by_oracle_and_uses_sink():
    w = _FakeWriter()
    captured = []

    def sink(**kw):
        captured.append(kw)

    tool = _SelfOracleTool(writer=w, companion_verdict_sink=sink)
    tool.invoke(ToolCall(tool_id="selforacle", args={"ok": True}))

    assert w.tool_calls[0]["scored_by"] == "oracle"
    assert len(captured) == 1
    assert captured[0]["verdict"] == "true"
    assert captured[0]["parent_tool_call_tx"] == "parenttx_1"
    assert captured[0]["oracle_id"] == "selforacle"


def test_self_oracle_failure_verdict_false():
    w = _FakeWriter()
    captured = []
    tool = _SelfOracleTool(writer=w, companion_verdict_sink=lambda **kw: captured.append(kw))
    tool.invoke(ToolCall(tool_id="selforacle", args={"ok": False}))
    assert captured[0]["verdict"] == "false"


def test_pure_tool_stays_unscored_no_sink_call():
    w = _FakeWriter()
    captured = []
    tool = _PureTool(writer=w, companion_verdict_sink=lambda **kw: captured.append(kw))
    tool.invoke(ToolCall(tool_id="pure", args={}))
    assert w.tool_calls[0]["scored_by"] is None
    assert captured == []   # pure tools don't ship a self-oracle verdict


def test_record_companion_verdict_then_flush_anchors_batch(tmp_path):
    """record_companion_verdict buffers a PRE-COMPUTED verdict (no plug run) →
    flush_companion_batches anchors an OracleVerdictBatch referencing the parent
    tool-call TX (the §A.6 coverage signal). Uses the real OracleRouter."""
    import duckdb
    from titan_hcl.synthesis.oracle_router import OracleRouter, OracleSpendStore
    from titan_hcl.synthesis.oracle_gate import OracleGate, OracleGateConfig

    w = _FakeWriter()
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    router = OracleRouter(
        gate=OracleGate(OracleGateConfig(
            balance_sol_baseline=1.0, admit_threshold=0.15,
            default_daily_sol_budget=0.1, daily_sol_budget={})),
        spend_store=OracleSpendStore(conn),
        outer_memory_writer=w,
        balance_provider=lambda: 1.0,
    )
    # No registered plug needed — record_companion_verdict never runs a plug.
    router.record_companion_verdict(
        parent_tool_call_tx="parenttx_99", oracle_id="coding_sandbox",
        verdict="true", evidence_ref="ran_ok", fork="procedural",
    )
    anchored = router.flush_companion_batches()
    conn.close()
    assert anchored.get("procedural"), "companion batch must anchor on the procedural fork"
    assert w.batches, "OracleVerdictBatch TX must be written"
    entries = w.batches[0]["entries"]
    assert entries[0]["parent_tool_call_tx"] == "parenttx_99"
    assert entries[0]["verdict"] == "true"
