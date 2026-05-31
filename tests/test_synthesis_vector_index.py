"""Phase A (operator closure) — tx_hash-native FAISS substrate.

Gate A unit coverage: round-trip tx_hash<->vector, knn returns tx_hashes (not
node_ids), per-fork shard isolation, idempotent add, cross-process reader sees
the writer's atomic save, fork="auto" merge, and reconstruct-by-tx_hash (the W4
consolidation path). The "first SEARCH hit ever" is the knn assertion below.
"""
import os

import numpy as np
import pytest

from titan_hcl.synthesis.synthesis_vector_index import (
    EMBEDDING_DIM,
    INDEXED_FORKS,
    FaissReader,
    SynthesisVectorStore,
)


def _unit(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic pseudo-random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _stub_embedder():
    """Map text -> a deterministic unit vector keyed by hash(text)."""
    def embed(text: str) -> np.ndarray:
        return _unit(abs(hash(text)) % (2**31))
    return embed


def test_roundtrip_knn_returns_tx_hashes(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    v1, v2, v3 = _unit(1), _unit(2), _unit(3)
    assert store.add_vector("declarative", "txAAA", v1)
    assert store.add_vector("declarative", "txBBB", v2)
    assert store.add_vector("declarative", "txCCC", v3)
    store.save()

    # Query with v1 → txAAA is the top hit (cosine == 1.0). The hit is a
    # tx_hash, never a node_id — the whole point of the spine.
    hits = store.knn("declarative", v1, k=3, min_similarity=0.0)
    assert hits, "first SEARCH hit ever must be non-empty"
    assert hits[0]["tx_hash"] == "txAAA"
    assert hits[0]["fork"] == "declarative"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert {h["tx_hash"] for h in hits} == {"txAAA", "txBBB", "txCCC"}


def test_min_similarity_threshold(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("declarative", "txAAA", _unit(1))
    store.add_vector("declarative", "txZZZ", _unit(999))
    store.save()
    # A high threshold keeps only the (near-)exact match.
    hits = store.knn("declarative", _unit(1), k=5, min_similarity=0.99)
    assert [h["tx_hash"] for h in hits] == ["txAAA"]


def test_idempotent_add(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    assert store.add_vector("procedural", "txDUP", _unit(5)) is True
    assert store.add_vector("procedural", "txDUP", _unit(6)) is False  # already present
    assert store.has("procedural", "txDUP")
    assert store.stats()["procedural"] == 1


def test_add_text_uses_embedder(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    assert store.add_text("conversation", "txHELLO", "hello world")
    assert not store.add_text("conversation", "txHELLO", "hello world")  # idempotent
    # Unindexed fork → no-op.
    assert not store.add_text("episodic", "txINNER", "image speak sound")
    store.save()
    assert store.stats()["conversation"] == 1


def test_per_fork_shard_isolation(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("conversation", "txCONV", _unit(10))
    store.add_vector("procedural", "txPROC", _unit(20))
    store.save()
    # A query on conversation never returns the procedural tx, and vice versa.
    conv = store.knn("conversation", _unit(10), k=5, min_similarity=0.5)
    assert [h["tx_hash"] for h in conv] == ["txCONV"]
    proc = store.knn("procedural", _unit(10), k=5, min_similarity=0.5)
    assert proc == []  # _unit(10) is txCONV's vector, absent from procedural


def test_fork_auto_merges_shards(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("conversation", "txCONV", _unit(10))
    store.add_vector("declarative", "txDECL", _unit(10))   # same vector, other fork
    store.add_vector("procedural", "txPROC", _unit(20))
    store.save()
    hits = store.knn("auto", _unit(10), k=10, min_similarity=0.9)
    got = {h["tx_hash"] for h in hits}
    assert "txCONV" in got and "txDECL" in got
    assert "txPROC" not in got  # _unit(20) is orthogonal-ish, below 0.9


def test_reconstruct_by_tx_hash(tmp_path):
    """W4: ConsolidationPass fetches a stored vector by tx_hash to cluster by cosine."""
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    v = _unit(42)
    store.add_vector("declarative", "txR", v)
    store.save()
    got = store.get_vector("declarative", "txR")
    assert got is not None
    # Stored vector is L2-normalized; cosine with the original ~= 1.
    assert float(np.dot(got, v)) == pytest.approx(1.0, abs=1e-4)
    assert store.get_vector("declarative", "txMISSING") is None


def test_cross_process_reader_sees_atomic_save(tmp_path):
    """FaissReader (consumer process) reads the writer's shards + the knn shape
    the RuleEvaluator SEARCH op consumes."""
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("declarative", "txAAA", _unit(1))
    store.save()

    reader = FaissReader(data_dir=str(tmp_path))
    hits = reader.knn("declarative", _unit(1), k=3, min_similarity=0.0)
    assert hits[0]["tx_hash"] == "txAAA"
    assert reader.ntotal("declarative") == 1

    # Writer appends + saves again → reader reloads on mtime and sees it.
    store.add_vector("declarative", "txBBB", _unit(2))
    store.save()
    hits2 = reader.knn("declarative", _unit(2), k=3, min_similarity=0.9)
    assert [h["tx_hash"] for h in hits2] == ["txBBB"]


def test_missing_shard_returns_empty(tmp_path):
    reader = FaissReader(data_dir=str(tmp_path))  # nothing written yet
    assert reader.knn("declarative", _unit(1), k=5, min_similarity=0.0) == []
    assert reader.ntotal("conversation") == 0


def test_shard_files_named_per_fork(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    store.add_vector("conversation", "txC", _unit(1))
    store.save()
    assert os.path.exists(tmp_path / "synthesis_vectors_conversation.faiss")
    assert os.path.exists(tmp_path / "synthesis_vectors_conversation.faiss.idmap.json")
    # Unwritten forks have no file.
    assert not os.path.exists(tmp_path / "synthesis_vectors_meta.faiss")


def test_add_texts_batch_embeds_and_dedups(tmp_path):
    calls = {"n": 0}

    def batch_embed(texts):
        calls["n"] += 1            # count fastembed calls — must be ONE per batch
        return [_unit(abs(hash(t)) % (2**31)) for t in texts]

    store = SynthesisVectorStore(
        data_dir=str(tmp_path), embedder=_stub_embedder(), batch_embedder=batch_embed)
    added = store.add_texts("declarative", [
        ("txA", "alpha"), ("txB", "beta"), ("txC", "gamma")])
    assert added == 3
    assert calls["n"] == 1         # all 3 embedded in ONE batch call
    # Re-adding (with one new) dedups the present ones, embeds only the new.
    added2 = store.add_texts("declarative", [("txA", "alpha"), ("txD", "delta")])
    assert added2 == 1
    assert calls["n"] == 2         # second batch had only the 1 new text
    store.save()
    assert store.stats()["declarative"] == 4


def test_add_texts_falls_back_per_text_without_batch_embedder(tmp_path):
    store = SynthesisVectorStore(data_dir=str(tmp_path), embedder=_stub_embedder())
    assert store.add_texts("procedural", [("txX", "x"), ("txY", "y")]) == 2
    assert store.stats()["procedural"] == 2


def test_indexed_forks_constant():
    assert INDEXED_FORKS == ("conversation", "declarative", "procedural")
    assert "episodic" not in INDEXED_FORKS and "meta" not in INDEXED_FORKS
