"""Tests for the self_learning_worker reward loop — RFP §7.A.3.

The load-bearing detail: the reward is ASYNC (the oracle verdict arrives at the
dream-flush, NOT at turn time), so the decision and reward are two events joined
on `parent_tool_call_tx`. We test the store + join + macro-distillation logic
directly (no subprocess spin).
"""
import json

import numpy as np
import pytest

from titan_hcl.bus import SELF_LEARN_MACRO_READY
from titan_hcl.modules.self_learning_worker import (
    _SelfLearningStore,
    _cfg,
    _handle_reward,
    _maybe_distill_macro,
)
from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS,
    OuterFeatures,
    OuterMetaPolicy,
)


class _Q:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


def _store(tmp_path):
    return _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))


def _feat(**kw):
    return OuterFeatures(**kw).to_vector().tolist()


def test_decision_stash_then_reward_join_trains(tmp_path):
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    tool = OUTER_ACTIONS.index("tool")
    feats = _feat(has_code_signal=True)
    store.stash_decision(tx="tx_abc", features=feats, action=tool,
                         goal_class="combinatorics", turn_id="t1")

    w_before = float(np.linalg.norm(policy.to_flat()))
    updates_before = policy.total_updates

    trained = _handle_reward(
        {"parent_tool_call_tx": "tx_abc", "reward": 1.0}, store, policy,
        None, _cfg({}), _Q(), "self_learning")

    assert trained is True
    assert policy.total_updates == updates_before + 1
    assert float(np.linalg.norm(policy.to_flat())) != w_before  # weights moved (G3)
    # the decision was consumed (one-shot join)
    assert store.pop_decision("tx_abc") is None
    # the reward tuple was recorded
    assert len(store.recent_reward_tuples(10)) == 1


def test_reward_without_matching_decision_is_noop(tmp_path):
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    updates_before = policy.total_updates
    trained = _handle_reward(
        {"parent_tool_call_tx": "nonexistent", "reward": 1.0}, store, policy,
        None, _cfg({}), _Q(), "self_learning")
    assert trained is False
    assert policy.total_updates == updates_before  # no spurious training


def test_negative_reward_trains(tmp_path):
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    idk = OUTER_ACTIONS.index("IDK")
    store.stash_decision(tx="tx_neg", features=_feat(requires_tool=True), action=idk,
                         goal_class="compute", turn_id="t2")
    trained = _handle_reward(
        {"parent_tool_call_tx": "tx_neg", "reward": -1.0}, store, policy,
        None, _cfg({}), _Q(), "self_learning")
    assert trained is True
    assert policy.reward_baseline < 0.0  # EMA tracked the −1


def test_macro_distilled_after_enough_wins(tmp_path):
    store = _store(tmp_path)
    cfg = _cfg({})
    q = _Q()
    tool = OUTER_ACTIONS.index("tool")
    feats = _feat(has_code_signal=True)
    # record macro_min_wins verified wins for (combinatorics, tool)
    for _ in range(int(cfg["macro_min_wins"])):
        store.record_reward_tuple(features=feats, action=tool, reward=1.0,
                                  goal_class="combinatorics")
    _maybe_distill_macro("combinatorics", tool, store, cfg, q, "self_learning")
    macros = [m for m in q.items if m["type"] == SELF_LEARN_MACRO_READY]
    assert len(macros) == 1
    p = macros[0]["payload"]
    assert p["goal_class"] == "combinatorics"
    assert p["verified"] is True
    assert len(p["signature"]) == len(feats)
    # idempotent — second call does not re-emit
    _maybe_distill_macro("combinatorics", tool, store, cfg, q, "self_learning")
    assert len([m for m in q.items if m["type"] == SELF_LEARN_MACRO_READY]) == 1


def test_macro_not_distilled_below_threshold(tmp_path):
    store = _store(tmp_path)
    cfg = _cfg({})
    q = _Q()
    tool = OUTER_ACTIONS.index("tool")
    store.record_reward_tuple(features=_feat(has_code_signal=True), action=tool,
                              reward=1.0, goal_class="combinatorics")
    _maybe_distill_macro("combinatorics", tool, store, cfg, q, "self_learning")
    assert not [m for m in q.items if m["type"] == SELF_LEARN_MACRO_READY]


def test_policy_flat_roundtrip_through_store(tmp_path):
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    feats = _feat(skill_matched=True)
    for _ in range(20):
        policy.learn(feats, OUTER_ACTIONS.index("skill_delegate"), 1.0)
    store.save_policy_flat(policy.to_flat().tolist(), policy.total_updates,
                           policy.reward_baseline)
    loaded = store.load_policy_flat()
    assert loaded is not None
    flat_list, updates, baseline = loaded
    restored = OuterMetaPolicy.from_flat(np.asarray(flat_list, dtype=np.float32))
    np.testing.assert_allclose(restored.forward(np.asarray(feats, dtype=np.float32)),
                               policy.forward(np.asarray(feats, dtype=np.float32)),
                               rtol=1e-5, atol=1e-5)
    assert updates == policy.total_updates


def test_prune_pending(tmp_path):
    store = _store(tmp_path)
    store.stash_decision(tx="old", features=_feat(), action=0,
                         goal_class="x", turn_id="t")
    # ttl=-1 → everything is "older than now+1" → pruned
    pruned = store.prune_pending(-1.0)
    assert pruned == 1
    assert store.pop_decision("old") is None
