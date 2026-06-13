"""§7.E (E.2) — the self-verifying prompt→solution cache (store + lock-free reader).

Covers the literal-serve discipline: a hit is served ONLY when durable + params
identical + semantically-near; volatile / param-differ / far all fall through (None →
E.1). Durable facts never TTL. Plus mutate-not-update (a conflicting answer → successor).

A template-normalizing fake embedder simulates a real sentence embedder (numeric
variants of one prompt cluster) so the param-identity logic is exercised independently
of cosine. The shared research_volatility durability seam decides literal-vs-reverify.

Run: python -m pytest tests/test_prompt_signature_cache.py -v -p no:anchorpy
"""
import hashlib

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.prompt_signature import (
    EMBEDDING_DIM, PromptSignatureReader, PromptSignatureStore, prompt_template,
    signature_id_for,
)

faiss = pytest.importorskip("faiss")


def _tmpl_embed(text: str):
    # embed the PARAM-NORMALIZED template → numeric variants of one prompt cluster
    # (a faithful stand-in for a real embedder, which puts "order 8"/"order 10" near).
    key = prompt_template(text)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _store(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    return PromptSignatureStore(
        conn, faiss_path=str(tmp_path / "prompt_signature_vectors.faiss"),
        snapshot_path=str(tmp_path / "prompt_signature_snapshot.json"),
        embedder=_tmpl_embed, writer=None, graph=None)


def _reader(tmp_path):
    return PromptSignatureReader(
        str(tmp_path / "prompt_signature_vectors.faiss"),
        str(tmp_path / "prompt_signature_snapshot.json"))


# ── store ───────────────────────────────────────────────────────────────────
def test_write_and_count(tmp_path):
    s = _store(tmp_path)
    ok = s.write_signature(prompt="order my 8 climbing routes", literal_answer="40320",
                           solved_by="tx_perm", durability="durable", created_epoch=100.0)
    assert ok is True and s.count() == 1


def test_idempotent_same_answer(tmp_path):
    s = _store(tmp_path)
    for _ in range(2):
        s.write_signature(prompt="order 8 routes", literal_answer="40320",
                          solved_by="tx", durability="durable", created_epoch=1.0)
    assert s.count() == 1  # identical re-verify is a no-op


def test_mutate_not_update_on_conflicting_answer(tmp_path):
    s = _store(tmp_path)
    s.write_signature(prompt="order 8 routes", literal_answer="40320",
                      solved_by="tx1", durability="durable", created_epoch=1.0)
    s.write_signature(prompt="order 8 routes", literal_answer="WRONG",
                      solved_by="tx2", durability="durable", created_epoch=2.0)
    assert s.count() == 2  # successor minted, prior intact (INV-OML-5)


# ── reader (the literal-serve discipline) ────────────────────────────────────
def test_literal_hit_durable_identical_params(tmp_path):
    s = _store(tmp_path)
    s.write_signature(prompt="order my 8 climbing routes", literal_answer="40320",
                      solved_by="tx_perm", durability="durable", created_epoch=100.0)
    r = _reader(tmp_path)
    hit = r.lookup("order my 8 climbing routes",
                   _tmpl_embed("order my 8 climbing routes"))
    assert hit is not None
    assert hit["literal_answer"] == "40320"
    assert hit["solved_by"] == "tx_perm"


def test_param_differ_falls_through(tmp_path):
    s = _store(tmp_path)
    s.write_signature(prompt="order my 8 climbing routes", literal_answer="40320",
                      solved_by="tx", durability="durable", created_epoch=100.0)
    r = _reader(tmp_path)
    # same template (clusters), different number → params differ → None (→ E.1 replay)
    assert r.lookup("order my 10 climbing routes",
                    _tmpl_embed("order my 10 climbing routes")) is None


def test_volatile_never_literal(tmp_path):
    s = _store(tmp_path)
    s.write_signature(prompt="current sol price", literal_answer="$71",
                      solved_by="tx", durability="volatile", created_epoch=100.0)
    r = _reader(tmp_path)
    assert r.lookup("current sol price",
                    _tmpl_embed("current sol price")) is None  # volatile → re-fetch (E.3), never cached value


def test_durable_never_ttls(tmp_path):
    # a DURABLE fact is evergreen — it is NEVER TTL-expired (the volatile 417-epoch
    # lifetime is the volatile class's decay, applied by DK.3/E.3, NOT to durable
    # literal serves — else evergreen answers would wrongly vanish after ~1hr).
    s = _store(tmp_path)
    s.write_signature(prompt="what consensus does solana use", literal_answer="PoH+PoS",
                      solved_by="tx", durability="durable", created_epoch=1.0)
    r = _reader(tmp_path)
    assert r.lookup("what consensus does solana use",
                    _tmpl_embed("what consensus does solana use")) is not None


def test_different_prompt_below_floor(tmp_path):
    s = _store(tmp_path)
    s.write_signature(prompt="order 8 routes", literal_answer="40320",
                      solved_by="tx", durability="durable", created_epoch=1.0)
    r = _reader(tmp_path)
    assert r.lookup("what is the capital of france",
                    _tmpl_embed("what is the capital of france")) is None


def test_cold_start_empty(tmp_path):
    r = _reader(tmp_path)
    assert r.lookup("anything", _tmpl_embed("anything")) is None


def test_signature_id_stable_and_param_insensitive_to_case():
    assert signature_id_for("Order 8 Routes") == signature_id_for("order 8 routes")
