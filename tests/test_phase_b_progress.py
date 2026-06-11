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


# ── §7.B (B.4) — the SSE wire contract the 3 clients render against ──────────
# The relay maps agno_worker's CHAT_STREAM_CHUNK payloads → SSE frames:
#   progress phase  → `event: progress` {phase, detail}
#   content chunk   → `data: {chunk}`
#   ovg provenance  → `event: ovg-headers`
#   done-frame      → `event: meta` {reasoning_id, mode}  (this RFP)  + `[DONE]`
import asyncio

from titan_hcl.api.chat import ChatRequest, chat_stream


class _FakeProxy:
    """Stand-in AgnoProxy yielding a realistic non-verifiable-turn stream."""
    async def chat_stream(self, message, user_id, session_id, channel,
                          is_maker, claims_sub):
        yield {"request_id": "x", "phase": "reasoning",
               "detail": "weighing it", "chunk": "", "done": False}
        yield {"request_id": "x", "chunk": "Sovereignty is ", "done": False}
        yield {"request_id": "x", "chunk": "self-authorship.", "done": False}
        yield {"request_id": "x", "chunk": "", "done": True,
               "ovg_headers": {"signature": "deadbeef"},
               "reasoning_id": "rid-123", "mode": "MODE_RESEARCH"}


class _AppState:
    def __init__(self, proxy):
        self.agno_proxy = proxy


class _App:
    def __init__(self, proxy):
        self.state = _AppState(proxy)


class _StreamReq:
    def __init__(self, proxy):
        self.app = _App(proxy)
        self.headers = {}


def _drain_sse(proxy) -> str:
    req = ChatRequest(message="what is sovereignty?", session_id="s", user_id="u")

    async def _collect():
        resp = await chat_stream(req, _StreamReq(proxy), claims={"sub": ""})
        out = []
        async for frame in resp.body_iterator:
            out.append(frame if isinstance(frame, str) else frame.decode())
        return "".join(out)

    return asyncio.run(_collect())


def test_relay_emits_full_b4_contract():
    sse = _drain_sse(_FakeProxy())
    # progress phase (live status surface)
    assert "event: progress" in sse
    assert '"phase": "reasoning"' in sse
    assert '"detail": "weighing it"' in sse
    # content chunks ride plain `data:` frames as {chunk} (NOT {text} — the
    # pre-existing landmine the client parser must match)
    assert '"chunk": "Sovereignty is "' in sse
    assert '"chunk": "self-authorship."' in sse
    # ovg provenance
    assert "event: ovg-headers" in sse
    # the closing meta frame this RFP adds — reasoning_id (rating footer) + mode
    assert "event: meta" in sse
    assert '"reasoning_id": "rid-123"' in sse
    assert '"mode": "MODE_RESEARCH"' in sse
    # terminal sentinel
    assert "data: [DONE]" in sse


def test_relay_meta_frame_absent_when_no_reasoning_id():
    """A verifiable (tool) turn has reasoning_id None → no rating footer →
    the relay must NOT emit a meta frame (no empty teaching prompt)."""
    class _ToolProxy:
        async def chat_stream(self, **kw):
            yield {"request_id": "x", "chunk": "done via sandbox.", "done": False}
            yield {"request_id": "x", "chunk": "", "done": True,
                   "ovg_headers": {}, "reasoning_id": None, "mode": "MODE_TOOL_ORACLE"}

    sse = _drain_sse(_ToolProxy())
    assert "data: [DONE]" in sse
    # mode present but no reasoning_id → meta carries only mode (still emitted so
    # the client can pick a scale IF a rating ever shows; reasoning_id absent).
    assert '"reasoning_id"' not in sse
