"""OML Phase C piece 7b — OuterCompositeReader (the parametric retrieval prior).

Lock-free SC-search of the prompt against macro composites: the faiss FILE
(read-only-safe cross-process) + the `reasoning_snapshot.json` macro map (the
SPEC-canonical snapshot pattern — DuckDB holds the exclusive lock even read_only).
Verifies a clean match → (score, action_norm), and every miss → (0.0, 0.0)."""
import json

import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS, OuterCompositeReader

faiss = pytest.importorskip("faiss")
_DIM = 384


def _norm(v):
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _build(tmp_path, macros):
    rng = np.random.RandomState(0)
    vecs = np.array([_norm(rng.randn(_DIM)) for _ in range(3)], dtype=np.float32)
    idx = faiss.IndexFlatL2(_DIM)
    idx.add(vecs)
    fpath = str(tmp_path / "reasoning_vectors.faiss")
    faiss.write_index(idx, fpath)
    spath = str(tmp_path / "reasoning_snapshot.json")
    with open(spath, "w") as f:
        json.dump({"version": 1, "macros": macros}, f)
    return fpath, spath, vecs


def test_prior_matches_top_macro(tmp_path):
    fpath, spath, vecs = _build(
        tmp_path, [{"embedding_id": 0, "action": "skill_delegate", "goal_class": "g0"},
                   {"embedding_id": 1, "action": "tool", "goal_class": "g1"}])
    r = OuterCompositeReader(fpath, spath)
    score, action_norm = r.prior(vecs[0], now=1000.0)  # identical to macro 0
    assert score > 0.99                                  # near-perfect cosine
    assert abs(action_norm - OUTER_ACTIONS.index("skill_delegate") / 4.0) < 1e-6


def test_prior_returns_macro_action_for_second_macro(tmp_path):
    fpath, spath, vecs = _build(
        tmp_path, [{"embedding_id": 1, "action": "tool", "goal_class": "g1"}])
    r = OuterCompositeReader(fpath, spath)
    score, action_norm = r.prior(vecs[1], now=1000.0)   # identical to macro 1
    assert score > 0.99
    assert abs(action_norm - OUTER_ACTIONS.index("tool") / 4.0) < 1e-6


def test_cold_start_missing_files(tmp_path):
    r = OuterCompositeReader(str(tmp_path / "nope.faiss"), str(tmp_path / "nope.json"))
    assert r.prior(np.zeros(_DIM, dtype=np.float32), now=1.0) == (0.0, 0.0)


def test_no_macros_in_snapshot(tmp_path):
    fpath, spath, vecs = _build(tmp_path, [])      # tool_use only, no macros
    r = OuterCompositeReader(fpath, spath)
    assert r.prior(vecs[0], now=1.0) == (0.0, 0.0)


def test_none_prompt_vec(tmp_path):
    fpath, spath, _ = _build(
        tmp_path, [{"embedding_id": 0, "action": "tool", "goal_class": "g"}])
    r = OuterCompositeReader(fpath, spath)
    assert r.prior(None, now=1.0) == (0.0, 0.0)


def test_dim_mismatch_safe(tmp_path):
    fpath, spath, _ = _build(
        tmp_path, [{"embedding_id": 0, "action": "tool", "goal_class": "g"}])
    r = OuterCompositeReader(fpath, spath)
    assert r.prior(np.zeros(16, dtype=np.float32), now=1.0) == (0.0, 0.0)  # wrong dim
