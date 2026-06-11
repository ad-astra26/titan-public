"""Phase B (§7.B C1′) — ReasoningStore.record_turn graphs a NON-verifiable turn
record (kind='turn', reward=NULL pending) under SELF → LEARNING → REASONING,
deref-able + SC-searchable. The turn-judge (B.2) / a user-Maker rating (B.3)
scores it later, keyed by the same reasoning_id. graph=None (Kuzu path soft)."""
import hashlib

import duckdb
import numpy as np

from titan_hcl.synthesis.reasoning_store import EMBEDDING_DIM, ReasoningStore


def _fake_embed(text):
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _store(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    return ReasoningStore(conn, faiss_path=str(tmp_path / "rv.faiss"),
                          graph=None, embedder=_fake_embed, writer=None)


def _feat():
    return [0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_record_turn_persists_with_pending_reward(tmp_path):
    s = _store(tmp_path)
    ok = s.record_turn(reasoning_id="rid_1", goal_class="philosophy_of_mind-query",
                       action="direct", features=_feat(),
                       signature_text="What does sovereignty mean to you?")
    assert ok is True
    assert s.count() == 1
    rec = s.get_record("rid_1")
    assert rec is not None and rec["kind"] == "turn" and rec["action"] == "direct"
    # reward is NULL (pending) — get_record coalesces NULL→0.0, so check the raw DB.
    raw = s._db.execute(
        "SELECT reward FROM reasoning_records WHERE reasoning_id='rid_1'").fetchone()
    assert raw[0] is None  # pending until a judge / user reward fills it


def test_record_turn_is_searchable(tmp_path):
    s = _store(tmp_path)
    s.record_turn(reasoning_id="rid_2", goal_class="general-query", action="direct",
                  features=_feat(), signature_text="describe the sea at dawn")
    hits = s.search("describe the sea at dawn", k=3)
    assert any(h.get("reasoning_id") == "rid_2" for h in hits)


def test_record_turn_idempotent(tmp_path):
    s = _store(tmp_path)
    s.record_turn(reasoning_id="rid_3", goal_class="g", action="research",
                  features=_feat(), signature_text="x")
    s.record_turn(reasoning_id="rid_3", goal_class="g", action="research",
                  features=_feat(), signature_text="x")
    assert s.count() == 1  # ON CONFLICT (reasoning_id) DO NOTHING
