"""Phase B (§7.B B.4) — live progress: OUR-logic phase emission (safe metadata,
NOT OVG/PreHook-gated) streamed over the chat-stream channel. Source = our decision
logic; agno is the transport."""
from titan_hcl.modules.agno_hooks import _emit_stream_progress, _phase_for_mode


class _FakeQ:
    def __init__(self):
        self.items = []

    def put_nowait(self, msg):
        self.items.append(msg)


class _Plugin:
    pass


def test_phase_for_mode():
    assert _phase_for_mode("MODE_RESEARCH") == "researching"
    assert _phase_for_mode("MODE_TOOL_ORACLE") == "running-tool"
    assert _phase_for_mode("MODE_SKILL_DELEGATE") == "using-skill"
    assert _phase_for_mode("MODE_SOVEREIGN") == "reasoning"   # direct
    assert _phase_for_mode("MODE_SHADOW") == "reasoning"      # IDK
    assert _phase_for_mode("anything-else") == "reasoning"
    assert _phase_for_mode(None) == "reasoning"


def test_emit_progress_publishes_phase_frame():
    q = _FakeQ()
    p = _Plugin()
    p._stream_progress_ctx = {"send_queue": q, "src": "s", "rid": "r1",
                              "name": "agno", "request_id": "req1"}
    _emit_stream_progress(p, "researching", detail="combinatorics")
    assert len(q.items) == 1
    payload = q.items[0]["payload"]
    assert payload["phase"] == "researching"
    assert payload["detail"] == "combinatorics"
    assert payload["chunk"] == "" and payload["done"] is False   # metadata, not content
    assert payload["request_id"] == "req1"


def test_emit_progress_noop_without_ctx():
    q = _FakeQ()
    p = _Plugin()  # no _stream_progress_ctx → non-streaming turn
    _emit_stream_progress(p, "thinking")
    assert q.items == []   # nothing published, no crash
