"""Phase 9 — POST /v6/synthesis/feedback (INV-Syn-24 Tier-2 producer)."""

import asyncio

import titan_hcl.api.synthesis_metrics_handlers as H


class _Bus:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _State:
    def __init__(self):
        self.bus = _Bus()


class _App:
    def __init__(self):
        self.state = type("S", (), {"titan_state": _State()})()


class _Req:
    def __init__(self, body):
        self._body = body
        self.app = _App()

    async def json(self):
        return self._body


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_valid_positive_feedback_publishes():
    req = _Req({"tool_call_tx": "tx_1", "verdict": "positive", "skill_id": "s1"})
    out = _run(H.post_v6_synthesis_feedback(req))
    assert out["ok"] is True and out["accepted"] is True
    pub = req.app.state.titan_state.bus.published
    assert len(pub) == 1


def test_negative_feedback_ok():
    req = _Req({"tool_call_tx": "tx_2", "verdict": "negative"})
    out = _run(H.post_v6_synthesis_feedback(req))
    assert out["ok"] is True


def test_missing_tool_call_tx_rejected():
    req = _Req({"verdict": "positive"})
    out = _run(H.post_v6_synthesis_feedback(req))
    assert out["ok"] is False
    assert out["error"] == "tool_call_tx_required"


def test_bad_verdict_rejected():
    req = _Req({"tool_call_tx": "tx", "verdict": "meh"})
    out = _run(H.post_v6_synthesis_feedback(req))
    assert out["ok"] is False
    assert "verdict" in out["error"]


def test_no_titan_state_soft_error():
    req = _Req({"tool_call_tx": "tx", "verdict": "positive"})
    req.app.state.titan_state = None
    out = _run(H.post_v6_synthesis_feedback(req))
    assert out["ok"] is False
