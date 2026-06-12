"""§24.12 Track 2 — OuterCompositeReader matches oracle-VERIFIED tool_use priors
(not only the rare hand-distilled macro_strategy), reward-weighted, with
macro_strategy taking precedence on a shared embedding_id.

This is the emergent, oracle-scored tool-intent path: Titan's own verified
tool-use experience (kind='tool_use', reward>0) becomes a retrieval prior matched
by embedding similarity. Without it, the prior consulted only macro_strategy
(typically a near-empty library) → composite_match_score pinned at 0.0 live.
"""
import json
import os
import tempfile

import faiss
import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import (
    NUM_OUTER_ACTIONS, OUTER_ACTIONS, OuterCompositeReader)

_TOOL = OUTER_ACTIONS.index("tool")
_RESEARCH = OUTER_ACTIONS.index("research")
_DIM = 8


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-9)


def _build(tmp, vectors, snapshot):
    """Write a faiss index (position == embedding_id) + a reasoning snapshot."""
    fp = os.path.join(tmp, "reasoning_vectors.faiss")
    sp = os.path.join(tmp, "reasoning_snapshot.json")
    idx = faiss.IndexFlatL2(_DIM)
    idx.add(np.vstack([_unit(v) for v in vectors]).astype(np.float32))
    faiss.write_index(idx, fp)
    with open(sp, "w") as f:
        json.dump(snapshot, f)
    return fp, sp


def test_verified_tool_use_prior_fires():
    """A compute prompt near a verified tool_use record (reward>0) → the prior
    returns a strong score + the tool action (was 0.0 with macro_strategy-only)."""
    with tempfile.TemporaryDirectory() as tmp:
        v_compute = _unit([1, 1, 0, 0, 0, 0, 0, 0])
        v_conv = _unit([0, 0, 0, 0, 0, 0, 1, 1])   # orthogonal to compute
        fp, sp = _build(tmp, [v_compute, v_conv], {
            "macros": [],
            "verified_priors": [
                {"embedding_id": 0, "action": "tool", "goal_class": "general-compute",
                 "reward": 0.8}],
        })
        r = OuterCompositeReader(fp, sp)
        score, a_norm = r.prior(_unit([1, 0.9, 0, 0, 0, 0, 0, 0]).tolist())  # ~ v_compute
        assert score > 0.6, f"expected strong verified match, got {score}"
        # action_norm encodes the tool action
        assert abs(a_norm - _TOOL / (NUM_OUTER_ACTIONS - 1)) < 1e-6
        # a conversational prompt has no near verified prior → ~0 signal
        s_conv, _ = r.prior(_unit([0, 0, 0, 0, 0, 0, 1, 0.9]).tolist())
        assert s_conv < 0.2, f"conversational should not match the compute prior: {s_conv}"


def test_reward_weighting_picks_best():
    """Between a CLOSER low-reward prior and a FARTHER high-reward prior, the
    reader picks the max of cos×reward (reward-weighted), not merely the nearest."""
    with tempfile.TemporaryDirectory() as tmp:
        near = _unit([1, 0, 0, 0, 0, 0, 0, 0])
        far = _unit([0.6, 0.8, 0, 0, 0, 0, 0, 0])
        fp, sp = _build(tmp, [near, far], {
            "macros": [],
            "verified_priors": [
                {"embedding_id": 0, "action": "tool", "reward": 0.1},      # close, weak
                {"embedding_id": 1, "action": "research", "reward": 1.0}],  # farther, strong
        })
        r = OuterCompositeReader(fp, sp)
        # query closest to `near`; near gives cos≈1×0.1=0.1, far gives cos×1.0 (>0.1)
        score, a_norm = r.prior(near.tolist())
        assert abs(a_norm - _RESEARCH / (NUM_OUTER_ACTIONS - 1)) < 1e-6, \
            "high-reward farther prior should win on cos×reward"
        assert score > 0.1


def test_macro_strategy_takes_precedence_on_shared_eid():
    """A macro_strategy overlays a verified prior on the SAME embedding_id (the
    refined distillate wins, full weight)."""
    with tempfile.TemporaryDirectory() as tmp:
        v = _unit([1, 1, 0, 0, 0, 0, 0, 0])
        fp, sp = _build(tmp, [v], {
            "verified_priors": [{"embedding_id": 0, "action": "tool", "reward": 0.5}],
            "macros": [{"embedding_id": 0, "action": "research", "goal_class": "x"}],
        })
        r = OuterCompositeReader(fp, sp)
        score, a_norm = r.prior(v.tolist())
        assert abs(a_norm - _RESEARCH / (NUM_OUTER_ACTIONS - 1)) < 1e-6, "macro should win"
        assert score > 0.9, f"macro weight=1.0 → score≈cos, got {score}"


def test_empty_priors_returns_zero():
    """No macros + no verified_priors → (0.0, 0.0) (unchanged miss behaviour)."""
    with tempfile.TemporaryDirectory() as tmp:
        v = _unit([1, 0, 0, 0, 0, 0, 0, 0])
        fp, sp = _build(tmp, [v], {"macros": [], "verified_priors": []})
        r = OuterCompositeReader(fp, sp)
        assert r.prior(v.tolist()) == (0.0, 0.0)
