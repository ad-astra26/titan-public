"""Phase B (§7.B B.3) — the turn_feedback endpoint: per-lane reward mapping +
Maker-vs-user (bigger delta routes via source weight + a graphed MakerAssessment
bond; ordinary user is reward-only) + ReasoningStore.record_maker_assessment."""
import asyncio
import hashlib

import duckdb
import numpy as np

from titan_hcl import bus as _bus
from titan_hcl.api.synthesis_metrics_handlers import post_v6_synthesis_turn_feedback
from titan_hcl.synthesis.reasoning_store import EMBEDDING_DIM, ReasoningStore


class _FakeBus:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _NS:
    pass


class _Req:
    def __init__(self, body, headers=None, bus=None):
        self._body = body
        self.headers = headers or {}
        ts = _NS(); ts.bus = bus
        st = _NS(); st.titan_state = ts
        self.app = _NS(); self.app.state = st

    async def json(self):
        return self._body


def _call(body, headers=None):
    bus = _FakeBus()
    out = asyncio.run(post_v6_synthesis_turn_feedback(_Req(body, headers, bus)))
    return out, bus


def test_stars5_mapping():
    assert _call({"reasoning_id": "r1", "score": 5, "scale": "stars5"})[0]["reward"] == 1.0
    assert _call({"reasoning_id": "r1", "score": 1, "scale": "stars5"})[0]["reward"] == -1.0
    assert _call({"reasoning_id": "r1", "score": 3, "scale": "stars5"})[0]["reward"] == 0.0
    assert _call({"reasoning_id": "r1", "score": 4, "scale": "stars5"})[0]["reward"] == 0.5


def test_research3_mapping():
    assert _call({"reasoning_id": "r2", "score": 3, "scale": "research3"})[0]["reward"] == 1.0
    assert _call({"reasoning_id": "r2", "score": 2, "scale": "research3"})[0]["reward"] == 0.0
    assert _call({"reasoning_id": "r2", "score": 1, "scale": "research3"})[0]["reward"] == -1.0


def test_user_reward_only_no_bond():
    out, bus = _call({"reasoning_id": "r3", "score": 4, "scale": "stars5"})
    assert out["source"] == "user"
    types = [m["type"] for m in bus.msgs]
    assert _bus.SELF_LEARN_REWARD in types
    assert _bus.MAKER_ASSESSMENT_RECORD not in types        # ordinary user → no bond
    rw = next(m for m in bus.msgs if m["type"] == _bus.SELF_LEARN_REWARD)
    assert rw["payload"]["source"] == "user"
    assert rw["payload"]["parent_tool_call_tx"] == "r3"


def test_maker_routes_maker_source_and_bond():
    out, bus = _call({"reasoning_id": "r4", "score": 5, "scale": "stars5"},
                     headers={"X-Titan-User-Id": "maker"})
    assert out["source"] == "maker"
    types = [m["type"] for m in bus.msgs]
    assert _bus.SELF_LEARN_REWARD in types and _bus.MAKER_ASSESSMENT_RECORD in types
    rw = next(m for m in bus.msgs if m["type"] == _bus.SELF_LEARN_REWARD)
    assert rw["payload"]["source"] == "maker"   # self_learning applies maker_reward_weight


def test_validation_errors():
    assert _call({"score": 5, "scale": "stars5"})[0]["ok"] is False        # no reasoning_id
    assert _call({"reasoning_id": "r", "score": 9, "scale": "stars5"})[0]["ok"] is False
    assert _call({"reasoning_id": "r", "score": 5, "scale": "bogus"})[0]["ok"] is False
    assert _call({"reasoning_id": "r", "score": None, "scale": "stars5"})[0]["ok"] is False


def _fake_embed(text):
    h = hashlib.sha256((text or "").encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_record_maker_assessment_persists(tmp_path):
    conn = duckdb.connect(str(tmp_path / "s.duckdb"))
    s = ReasoningStore(conn, faiss_path=str(tmp_path / "rv.faiss"),
                       graph=None, embedder=_fake_embed, writer=None)
    ok = s.record_maker_assessment(reasoning_id="r5", score=5.0, scale="stars5",
                                   reward=1.0, turn_summary="great answer")
    assert ok is True
    row = conn.execute(
        "SELECT score, reward, scale FROM maker_assessments WHERE reasoning_id='r5'"
    ).fetchone()
    assert row == (5.0, 1.0, "stars5")
