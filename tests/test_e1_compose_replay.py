"""§7.E (E.1) — compose-from-composite hot-path replay.

Two layers: (1) OuterCompositeReader.match() exposes a matched MACRO composite's
replay recipe / research source / goal_class lock-free (faiss FILE + snapshot);
(2) _e1_recipe_replay binds THIS prompt's numeric params into the templated recipe
and returns runnable code (skip the LLM router), or '' (fall through) on any
miss / below-floor / non-tool / literal / non-numeric. The oracle re-verify is the
live final net; these prove the deref + safe-bind glue.

Run: python -m pytest tests/test_e1_compose_replay.py -v -p no:anchorpy
"""
import json

import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import OuterCompositeReader
from titan_hcl.synthesis.tool_backstop import _e1_recipe_replay

faiss = pytest.importorskip("faiss")
_DIM = 384


def _norm(v):
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _templated_recipe():
    return json.dumps({"tool_id": "coding_sandbox",
                       "code_template": "math.factorial({p0})",
                       "param_kinds": ["number"], "captured_params": ["8"]},
                      separators=(",", ":"))


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


# ── OuterCompositeReader.match() ────────────────────────────────────────────
def test_match_returns_recipe_and_meta(tmp_path):
    fpath, spath, vecs = _build(tmp_path, [
        {"embedding_id": 0, "action": "tool", "goal_class": "combinatorics",
         "reasoning_id": "macro_perm", "recipe_json": _templated_recipe(), "source": ""}])
    r = OuterCompositeReader(fpath, spath)
    m = r.match(vecs[0], now=1000.0)
    assert m is not None
    assert m["score"] > 0.99
    assert m["action"] == "tool"
    assert m["goal_class"] == "combinatorics"
    assert "code_template" in m["recipe_json"]
    assert m["reasoning_id"] == "macro_perm"


def test_match_none_on_cold_start(tmp_path):
    r = OuterCompositeReader(str(tmp_path / "no.faiss"), str(tmp_path / "no.json"))
    assert r.match(np.zeros(_DIM, dtype=np.float32), now=1.0) is None


def test_match_none_when_no_macros(tmp_path):
    fpath, spath, vecs = _build(tmp_path, [])
    assert OuterCompositeReader(fpath, spath).match(vecs[0], now=1.0) is None


# ── _e1_recipe_replay glue (a fake plugin carrying the stashed match) ───────
class _Plugin:
    def __init__(self, match, e_compose=None):
        self._last_composite_match = match
        self._full_config = {"synthesis": {"tool_backstop": {
            "e_compose": e_compose if e_compose is not None else {"enabled": True, "floor": 0.85}}}}


def _cfg(plugin):
    return plugin._full_config["synthesis"]["tool_backstop"]


def _match(score=0.95, action="tool", recipe=None):
    return {"score": score, "action": action, "goal_class": "combinatorics",
            "recipe_json": recipe if recipe is not None else _templated_recipe(),
            "source": "", "reasoning_id": "macro_perm"}


def test_replay_binds_new_param():
    p = _Plugin(_match())
    code = _e1_recipe_replay(p, "how many orderings of 10 routes?", _cfg(p))
    assert code == "math.factorial(10)"          # 8 → 10 bound, router skipped


def test_replay_identical_param():
    p = _Plugin(_match())
    code = _e1_recipe_replay(p, "order my 8 climbing routes", _cfg(p))
    assert code == "math.factorial(8)"


def test_replay_below_floor_falls_through():
    p = _Plugin(_match(score=0.50))
    assert _e1_recipe_replay(p, "order 10 routes", _cfg(p)) == ""


def test_replay_non_tool_action_falls_through():
    p = _Plugin(_match(action="research"))
    assert _e1_recipe_replay(p, "order 10 routes", _cfg(p)) == ""


def test_replay_no_match_falls_through():
    p = _Plugin(None)
    assert _e1_recipe_replay(p, "order 10 routes", _cfg(p)) == ""


def test_replay_literal_recipe_falls_through():
    # a non-templatized (literal) recipe is never blind-replayed → LLM fallback
    lit = json.dumps({"tool_id": "coding_sandbox", "args": {"code": "x=1"}})
    p = _Plugin(_match(recipe=lit))
    assert _e1_recipe_replay(p, "anything", _cfg(p)) == ""


def test_replay_no_numeric_param_falls_through():
    # template needs 1 numeric param; a prompt with none → bind fails → fallthrough
    p = _Plugin(_match())
    assert _e1_recipe_replay(p, "order my climbing routes", _cfg(p)) == ""


def test_replay_disabled_flag():
    p = _Plugin(_match(), e_compose={"enabled": False})
    assert _e1_recipe_replay(p, "order 10 routes", _cfg(p)) == ""
