"""Phase 6 amendment — coding_sandbox ToolPlug wired locally in agno.

The synthesis ToolPlugs are instantiated in the process that INVOKES them
(arch §11.3 / SPEC §25.5). The chat-time `coding_sandbox` tool therefore
builds its plug in agno_worker (not synthesis_worker — a worker cannot
populate another process's plugin attr). These guard that wiring so the
tool can never silently regress to "not wired" + that the procedural TX
still anchors via the canonical OuterMemoryWriter path (INV-4).
"""
from titan_hcl.modules.agno_worker import _build_local_tool_plugs
from titan_hcl.synthesis.plugs import ToolCall


class _CaptureQueue:
    """Stand-in send_queue: captures the TIMECHAIN_COMMIT the plug emits."""

    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def test_agno_builds_coding_sandbox_plug():
    plugs = _build_local_tool_plugs(_CaptureQueue())
    assert "coding_sandbox" in plugs, "agno must wire coding_sandbox locally"
    assert plugs["coding_sandbox"].tool_id == "coding_sandbox"


def test_agno_plug_is_torch_free():
    import sys
    _build_local_tool_plugs(_CaptureQueue())
    assert "torch" not in sys.modules, \
        "wiring coding_sandbox must not drag torch into agno (§3J)"


def test_coding_sandbox_executes_and_anchors():
    q = _CaptureQueue()
    plug = _build_local_tool_plugs(q)["coding_sandbox"]
    result = plug.invoke(ToolCall(tool_id="coding_sandbox", args={"code": "print(6 * 7)"}))
    assert result.success, f"sandbox should run trivial code: {result.result_summary}"
    assert "42" in result.result_summary
    # The procedural fork TX must anchor via the canonical write path (INV-4).
    assert any(m.get("type") == "TIMECHAIN_COMMIT" or "timechain" in str(m.get("dst", "")).lower()
               for m in q.items), "tool-call must anchor a procedural TX"
