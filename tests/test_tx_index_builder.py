"""Phase A — TxIndexBuilder (A1+A2+A4): chain → tx_hash-native FAISS population.

Uses a synthetic read-only index.db (fork_registry + block_index) and a
monkeypatched content reader, so the scan / watermark / idempotency / fork
resolution / embedding logic is exercised without real chain `.bin` files.
"""
import os
import sqlite3

import numpy as np
import pytest

from titan_hcl.synthesis import tx_index_builder as tib
from titan_hcl.synthesis.synthesis_vector_index import FaissReader, SynthesisVectorStore


def _stub_embedder():
    def embed(text: str) -> np.ndarray:
        v = np.random.default_rng(abs(hash(text)) % (2**31)).standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)
    return embed


def _make_index_db(path, rows):
    """rows: list of (fork_id, fork_name, block_hash_hex, height, offset)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE fork_registry (fork_id INTEGER PRIMARY KEY, fork_name TEXT);"
        "CREATE TABLE block_index (block_hash BLOB PRIMARY KEY, fork_id INTEGER, "
        "block_height INTEGER, file_offset INTEGER);"
    )
    seen_forks = {}
    for fork_id, fork_name, bh_hex, height, offset in rows:
        seen_forks[fork_id] = fork_name
        conn.execute(
            "INSERT OR REPLACE INTO block_index VALUES (?,?,?,?)",
            (bytes.fromhex(bh_hex), fork_id, height, offset),
        )
    for fid, fname in seen_forks.items():
        conn.execute("INSERT OR REPLACE INTO fork_registry VALUES (?,?)", (fid, fname))
    conn.commit()
    conn.close()


@pytest.fixture
def content_map(monkeypatch):
    """Patch read_block_content_at to serve canned content by (fork_id, offset)."""
    store: dict[tuple, dict] = {}

    def fake_read(data_dir, fork_id, offset):
        return store.get((int(fork_id), int(offset)))

    monkeypatch.setattr(tib, "read_block_content_at", fake_read)
    return store


def _builder(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    db_dir = tmp_path / "timechain"
    db_dir.mkdir(exist_ok=True)
    return store, str(db_dir / "index.db")


def test_indexes_conversation_declarative_procedural(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [
        (5, "conversation", "aa" * 32, 1, 0),
        (1, "declarative", "bb" * 32, 1, 0),
        (2, "procedural", "cc" * 32, 1, 0),
        (3, "episodic", "dd" * 32, 1, 0),   # NOT indexed
    ])
    content_map[(5, 0)] = {"user_msg": "how do I mint an NFT", "agent_response": "use metaplex"}
    content_map[(1, 0)] = {"name": "metaplex_nft_minting", "concept_id": "metaplex_nft_minting"}
    content_map[(2, 0)] = {"tool_id": "coding_sandbox", "result_summary": "ran ok"}
    content_map[(3, 0)] = {"text": "inner dream image"}

    b = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path))
    s = b.run()
    assert s["indexed"] == 3      # episodic skipped (not an indexed fork)
    assert store.stats() == {"conversation": 1, "declarative": 1, "procedural": 1}
    # The block_hash becomes the retrievable key.
    hits = FaissReader(data_dir=str(tmp_path)).knn(
        "conversation", store.get_vector("conversation", "aa" * 32), k=1, min_similarity=0.5)
    assert hits[0]["tx_hash"] == "aa" * 32


def test_idempotent_across_runs(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [(1, "declarative", "ab" * 32, 1, 0)])
    content_map[(1, 0)] = {"name": "linux_terminal"}
    b = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path))
    assert b.run()["indexed"] == 1
    # Second run over the same data → watermark + has() dedup → nothing new.
    b2 = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path))
    s2 = b2.run()
    assert s2["indexed"] == 0
    assert store.stats()["declarative"] == 1


def test_watermark_advances_and_persists(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [(1, "declarative", "a0" * 32, 5, 0)])
    content_map[(1, 0)] = {"name": "c5"}
    tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run()
    wm_file = tmp_path / tib.WATERMARK_NAME
    assert wm_file.exists()
    # New, higher block appears; a fresh builder resumes from the watermark.
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO block_index VALUES (?,?,?,?)",
                 (bytes.fromhex("a1" * 32), 1, 9, 10))
    conn.commit(); conn.close()
    content_map[(1, 10)] = {"name": "c9"}
    s = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run()
    assert s["indexed"] == 1
    assert store.stats()["declarative"] == 2


def test_dry_run_counts_without_writing(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [(2, "procedural", "ac" * 32, 1, 0)])
    content_map[(2, 0)] = {"tool_id": "x", "result_summary": "y"}
    s = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run(dry_run=True)
    assert s["indexed"] == 1
    assert store.stats()["procedural"] == 0          # nothing written
    assert not (tmp_path / tib.WATERMARK_NAME).exists()


def test_no_content_is_skipped_not_indexed(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [(1, "declarative", "ad" * 32, 1, 0)])
    # content_map has no entry → read returns None → empty text → no_content.
    s = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run()
    assert s["indexed"] == 0 and s["no_content"] == 1
    assert store.stats()["declarative"] == 0


def test_from_scratch_ignores_watermark(tmp_path, content_map):
    store, db_path = _builder(tmp_path)
    _make_index_db(db_path, [(1, "declarative", "ae" * 32, 3, 0)])
    content_map[(1, 0)] = {"name": "c"}
    tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run()
    # A backfill (from_scratch) over a fresh store re-reads from height 0.
    store2 = SynthesisVectorStore(data_dir=str(tmp_path / "fresh"), embedder=_stub_embedder())
    s = tib.TxIndexBuilder(store=store2, data_dir=str(tmp_path)).run(from_scratch=True)
    assert s["indexed"] == 1


def test_embeddable_text_fork_shapes():
    assert "metaplex" in tib._embeddable_text("conversation", {"user_msg": "metaplex?", "agent_response": "yes"})
    assert "coding_sandbox" in tib._embeddable_text("procedural", {"tool_id": "coding_sandbox", "result_summary": "ok"})
    assert "linux" in tib._embeddable_text("declarative", {"name": "linux", "concept_id": "linux"})
    # Unknown shape → bounded JSON fallback (never empty for non-empty content).
    assert tib._embeddable_text("declarative", {"weird_key": "value123"}) != ""
    assert tib._embeddable_text("declarative", {}) == ""


def test_no_index_db_is_noop(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    s = tib.TxIndexBuilder(store=store, data_dir=str(tmp_path)).run()
    assert s["indexed"] == 0
    assert "no index_db" in s.get("note", "")
