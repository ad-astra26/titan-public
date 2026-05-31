"""Phase B — executor wired into the live path (W1/W2/B1).

Covers the two seams that make chat recall actually fire:
  * FaissReader plugged into the RuleEvaluator SEARCH op returns tx_hashes
    (the contract the op always specified — proves the wiring end-to-end).
  * TxContentDeref dereferences a SEARCH-returned tx_hash back into content
    (so the operator can assemble injectable context).
"""
import sqlite3

import numpy as np
import pytest

from titan_hcl.logic.timechain_v2 import RuleEvaluator
from titan_hcl.synthesis import tx_index_builder as tib
from titan_hcl.synthesis.synthesis_vector_index import FaissReader, SynthesisVectorStore


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _stub_embedder():
    def embed(text: str):
        return _unit(abs(hash(text)) % (2**31))
    return embed


def test_faiss_reader_through_search_op_returns_tx_hashes(tmp_path):
    """The RuleEvaluator SEARCH op, given the Phase-A FaissReader as its
    faiss_reader, returns [{tx_hash, score, fork}] — the dead W2 path, live."""
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("conversation", "f1" * 32, _unit(1))
    store.add_vector("conversation", "f2" * 32, _unit(2))
    store.save()

    reader = FaissReader(data_dir=str(tmp_path))
    ev = RuleEvaluator(orchestrator=None, faiss_reader=reader, index_db=None)

    rule = {
        "op": "SEARCH", "fork": "conversation",
        "query_embedding": "$qe", "limit": 5, "min_similarity": 0.0,
        "store": "$hits",
    }
    out = ev._exec_search(rule, {"event": "retrieval_request"}, {"$qe": list(_unit(1))})
    assert out, "SEARCH must return hits now that faiss_reader is wired (W2 closed)"
    assert out[0]["tx_hash"] == "f1" * 32
    assert out[0]["fork"] == "conversation"
    assert "score" in out[0]


def test_search_op_no_faiss_reader_still_empty():
    """Regression: without a faiss_reader the op soft-fails to [] (unchanged)."""
    ev = RuleEvaluator(orchestrator=None, faiss_reader=None, index_db=None)
    rule = {"op": "SEARCH", "fork": "conversation", "query_embedding": "$qe", "limit": 5}
    assert ev._exec_search(rule, {}, {"$qe": list(_unit(1))}) == []


@pytest.fixture
def content_map(monkeypatch):
    store: dict = {}

    def fake_read(data_dir, fork_id, offset):
        return store.get((int(fork_id), int(offset)))

    monkeypatch.setattr(tib, "read_block_content_at", fake_read)
    return store


def test_tx_content_deref_roundtrips_snippet(tmp_path, content_map):
    db_dir = tmp_path / "timechain"
    db_dir.mkdir()
    conn = sqlite3.connect(str(db_dir / "index.db"))
    conn.execute("CREATE TABLE block_index (block_hash BLOB PRIMARY KEY, fork_id INTEGER, file_offset INTEGER)")
    conn.execute("INSERT INTO block_index VALUES (?,?,?)", (bytes.fromhex("ab" * 32), 5, 0))
    conn.commit(); conn.close()
    content_map[(5, 0)] = {"user_msg": "how to mint nft", "agent_response": "use metaplex"}

    deref = tib.TxContentDeref(data_dir=str(tmp_path))
    snip = deref.snippet("ab" * 32, "conversation")
    assert snip and "metaplex" in snip
    # Unknown hash → None (operator drops it from context).
    assert deref.snippet("ff" * 32, "conversation") is None
    deref.close()


def test_tx_content_deref_no_index_db_returns_none(tmp_path):
    deref = tib.TxContentDeref(data_dir=str(tmp_path))
    assert deref.snippet("ab" * 32) is None
