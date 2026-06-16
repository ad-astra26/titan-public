"""RFP_synthesis_reuse_and_routing_revival — offline gates.

Break F: SnapshotProceduralReader surfaces a delegate-gated skill cross-process
         from skills_snapshot.json + skills_vectors.faiss (no DuckDB lock).
Break D: the idle structural pass re-teaches feature-discrimination via balanced
         synthetic lanes so a COLLAPSED restored policy recovers (does not stay
         pinned to `direct`).
Break A: OuterFeatures carries skill_matched → the structural ladder reaches
         skill_delegate when a skill matches.

Run isolated: pytest tests/test_synthesis_reuse_revival_20260616.py -p no:anchorpy
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest


# ── Break F — SnapshotProceduralReader ───────────────────────────────────────

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _write_skill_snapshot_and_faiss(data_dir):
    """A verified, delegatable skill (emb id 0) + an unverified one (emb id 1)."""
    faiss = pytest.importorskip("faiss")
    dim = 8
    vec_deploy = _unit([1, 1, 0, 0, 0, 0, 0, 0])   # "deploy solana nft"
    vec_other = _unit([0, 0, 0, 0, 1, 1, 0, 0])    # unrelated
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack([vec_deploy, vec_other]).astype(np.float32))  # ids 0,1
    faiss.write_index(index, os.path.join(data_dir, "skills_vectors.faiss"))
    snap = {
        "version": 2, "ts": 1.0, "count": 2,
        "skills": [
            {"skill_id": "skill_deploy", "oracle_id": "coding_sandbox",
             "goal_class": "deploy-nft", "name": "deploy solana nft",
             "nl_description": "deploy a solana nft via coding_sandbox",
             "promoted": True, "verified_at": 123.0, "utility_score": 0.8,
             "success_count": 5, "failure_count": 0, "task_shapes": 1,
             "embedding_id": 0},
            # unverified + zero-utility → must be filtered by read_for_match
            {"skill_id": "skill_unproven", "oracle_id": "coding_sandbox",
             "goal_class": "x", "name": "unproven thing",
             "nl_description": "not yet verified", "promoted": False,
             "verified_at": None, "utility_score": 0.0,
             "success_count": 0, "failure_count": 0, "task_shapes": 1,
             "embedding_id": 1},
        ],
    }
    with open(os.path.join(data_dir, "skills_snapshot.json"), "w") as fh:
        json.dump(snap, fh)
    return vec_deploy


def test_snapshot_reader_surfaces_delegatable_skill(tmp_path):
    from titan_hcl.synthesis.snapshot_procedural_reader import SnapshotProceduralReader
    data_dir = str(tmp_path)
    qvec = _write_skill_snapshot_and_faiss(data_dir)

    def _embedder(text):
        # the query embeds near the deploy skill's vector
        return qvec

    reader = SnapshotProceduralReader(data_dir, _embedder)
    results = reader.recall("deploy a solana nft", k=3)
    assert results, "expected a procedural match"
    top = results[0]
    assert top["skill_id"] == "skill_deploy"
    assert top["match_score"] >= 0.65, top["match_score"]
    assert reader.should_delegate(top) is True
    # the unverified/zero-utility skill must never surface
    assert all(r["skill_id"] != "skill_unproven" for r in results)


def test_snapshot_reader_no_match_below_floor(tmp_path):
    from titan_hcl.synthesis.snapshot_procedural_reader import SnapshotProceduralReader
    data_dir = str(tmp_path)
    _write_skill_snapshot_and_faiss(data_dir)

    def _embedder(text):
        return _unit([0, 0, 1, 0, 0, 0, 0, 0])  # orthogonal to both skills

    reader = SnapshotProceduralReader(data_dir, _embedder)
    results = reader.recall("what is the weather", k=3)
    # an orthogonal query yields low cosine → match_score below the delegate floor
    top = results[0] if results else None
    assert not reader.should_delegate(top)


def test_snapshot_reader_missing_files_is_empty(tmp_path):
    from titan_hcl.synthesis.snapshot_procedural_reader import SnapshotProceduralReader
    reader = SnapshotProceduralReader(str(tmp_path), lambda t: _unit([1, 0]))
    assert reader.recall("anything") == []


# ── Break A — OuterFeatures.skill_matched reaches the ladder ──────────────────

def test_structural_target_reaches_skill_delegate_when_matched():
    from titan_hcl.synthesis.outer_meta_policy import (
        OuterFeatures, OUTER_ACTIONS, structural_target_action)
    feats = OuterFeatures(recall_top_cosine=0.2, skill_utility=0.8,
                          skill_matched=True, requires_tool=False)
    vec = feats.to_vector()
    assert structural_target_action(vec) == OUTER_ACTIONS.index("skill_delegate")
    # the same context with skill_matched=False must NOT route skill_delegate
    feats_nomatch = OuterFeatures(recall_top_cosine=0.2, skill_utility=0.8,
                                  skill_matched=False, requires_tool=False)
    assert structural_target_action(feats_nomatch.to_vector()) != \
        OUTER_ACTIONS.index("skill_delegate")


# ── Break D — anti-collapse recovery via synthetic lanes ──────────────────────

class _StubStore:
    """Minimal store for _structural_explore: no recent contexts (the collapse
    case), no-op persistence + logging."""
    def distinct_recent_contexts(self, n):
        return []
    def save_policy_flat(self, *a, **k):
        pass
    def log_explore(self, *a, **k):
        pass


class _StubWriter:
    def write(self, *a, **k):
        pass


def _collapse_policy_to_direct(policy):
    """Drive the policy to the live-collapse state: reward `direct` densely on
    every shape, mirroring the quality-judge stream that pins T1 to direct."""
    from titan_hcl.synthesis.outer_meta_policy import (
        OuterFeatures, OUTER_ACTIONS)
    direct = OUTER_ACTIONS.index("direct")
    rng = np.random.default_rng(7)
    for _ in range(400):
        feats = OuterFeatures(
            recall_top_cosine=float(rng.uniform(0, 1)),
            skill_utility=float(rng.uniform(0, 1)),
            skill_matched=bool(rng.uniform() > 0.5),
            requires_tool=bool(rng.uniform() > 0.5),
            has_code_signal=bool(rng.uniform() > 0.5))
        policy.learn(feats.to_vector(), direct, 1.0, baseline_alpha=0.05)


def test_idle_synthetic_pass_recovers_collapsed_policy():
    from titan_hcl.synthesis.outer_meta_policy import (
        OuterFeatures, OuterMetaPolicy, OUTER_ACTIONS)
    from titan_hcl.modules.self_learning_worker import (
        _structural_explore, _DEFAULTS)

    np.random.seed(0)
    policy = OuterMetaPolicy()
    _collapse_policy_to_direct(policy)

    # a clearly tool-shaped context (requires_tool + code) — should route `tool`.
    tool_feats = OuterFeatures(recall_top_cosine=0.1, requires_tool=True,
                               has_code_signal=True)
    skill_feats = OuterFeatures(recall_top_cosine=0.1, skill_utility=0.8,
                                skill_matched=True)
    tool_idx = OUTER_ACTIONS.index("tool")
    skill_idx = OUTER_ACTIONS.index("skill_delegate")

    collapsed_tool = policy.exploit_action(tool_feats.to_vector())
    # sanity: the collapse pinned even a tool-shaped context to direct
    assert collapsed_tool == OUTER_ACTIONS.index("direct")

    cfg = dict(_DEFAULTS)
    cfg["structural_synthetic_passes"] = 40   # several maintenance ticks' worth
    store = _StubStore()
    for _ in range(8):                        # a handful of idle ticks
        _structural_explore(cfg, store, policy, _StubWriter(), None, "test")

    assert policy.exploit_action(tool_feats.to_vector()) == tool_idx
    assert policy.exploit_action(skill_feats.to_vector()) == skill_idx
