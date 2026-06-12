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
    # Phase B (§7.B): the join no longer POPS the decision — it MARKS it rewarded
    # (kept so a higher-authority reward can correct it; TTL-pruned otherwise).
    peeked = store.peek_decision("tx_abc")
    assert peeked is not None and peeked[4] == "llm_judge"  # applied_source set
    # the reward tuple was recorded
    assert len(store.recent_reward_tuples(10)) == 1


def test_direct_reward_trains_without_stash(tmp_path):
    # v1.1 (INV-OML-12): the C1 capture emits (features, action, reward) DIRECTLY
    # — no decision stash, no tx join. The worker trains on it immediately.
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    tool = OUTER_ACTIONS.index("tool")
    feats = _feat(has_code_signal=True)
    updates_before = policy.total_updates
    trained = _handle_reward(
        {"features": feats, "action": tool, "reward": 1.0,
         "goal_class": "combinatorics"},
        store, policy, None, _cfg({}), _Q(), "self_learning")
    assert trained is True
    assert policy.total_updates == updates_before + 1
    # recorded as a reward tuple (drives macro distillation)
    assert len(store.recent_reward_tuples(10)) == 1
    # no pending-decision stash was needed
    assert store.pop_decision("anything") is None


def test_direct_reward_distills_macro(tmp_path):
    # enough direct verified wins of one (goal_class, action) → macro emitted.
    # Phase-C piece 6: the reactive `_maybe_distill_macro` is now the FALLBACK,
    # superseded by the explore-tick deliberative path when `outer_meta_enabled`
    # (the default). This test covers the reactive fallback → flag it off.
    store = _store(tmp_path)
    cfg = _cfg({"synthesis": {"self_learning": {"outer_meta_enabled": False}}})
    q = _Q()
    policy = OuterMetaPolicy(lr=0.05)
    tool = OUTER_ACTIONS.index("tool")
    feats = _feat(has_code_signal=True)
    for _ in range(int(cfg["macro_min_wins"])):
        _handle_reward(
            {"features": feats, "action": tool, "reward": 1.0,
             "goal_class": "combinatorics"},
            store, policy, None, cfg, q, "self_learning")
    macros = [m for m in q.items if m["type"] == SELF_LEARN_MACRO_READY]
    assert len(macros) == 1
    assert macros[0]["payload"]["goal_class"] == "combinatorics"


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


# ── Phase B (§7.B) — source weights + corrective-delta ───────────────────────

def test_maker_weight_larger_than_user(tmp_path):
    # Same raw rating (+1), but Maker carries a larger weight → larger effective
    # reward (the "bigger delta for Maker" mechanic). applied_reward = raw×weight.
    def _effective(source, db):
        store = _SelfLearningStore(path=str(tmp_path / db))
        policy = OuterMetaPolicy(lr=0.05)
        direct = OUTER_ACTIONS.index("direct")
        store.stash_decision(tx="tx", features=_feat(), action=direct,
                             goal_class="philosophy", turn_id="t")
        _handle_reward({"parent_tool_call_tx": "tx", "reward": 1.0, "source": source},
                       store, policy, None, _cfg({}), _Q(), "self_learning")
        return store.peek_decision("tx")[3]  # applied_reward = raw × source weight
    assert _effective("user", "u.duckdb") == pytest.approx(1.0)
    assert _effective("maker", "m.duckdb") == pytest.approx(2.0)


def test_corrective_delta_user_after_judge(tmp_path):
    # Judge says good (+1); the user later disagrees (★1 → −1). The higher-
    # authority user applies a CORRECTIVE DELTA over the judge's prior, retrains
    # once, and the decision is re-marked as user-sourced + net-negative.
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    direct = OUTER_ACTIONS.index("direct")
    store.stash_decision(tx="tx", features=_feat(), action=direct,
                         goal_class="philosophy", turn_id="t")
    _handle_reward({"parent_tool_call_tx": "tx", "reward": 1.0, "source": "llm_judge"},
                   store, policy, None, _cfg({}), _Q(), "self_learning")
    upd_after_judge = policy.total_updates
    assert store.peek_decision("tx")[4] == "llm_judge"
    trained = _handle_reward(
        {"parent_tool_call_tx": "tx", "reward": -1.0, "source": "user"},
        store, policy, None, _cfg({}), _Q(), "self_learning")
    assert trained is True
    assert policy.total_updates == upd_after_judge + 1   # the correction retrained
    peeked = store.peek_decision("tx")
    assert peeked[4] == "user" and peeked[3] < 0.0       # re-marked user, net negative


def test_lower_authority_reward_after_higher_is_ignored(tmp_path):
    # The user rates first (+1); the judge runs later (lower authority) → ignored,
    # no double-train, the decision stays user-sourced.
    store = _store(tmp_path)
    policy = OuterMetaPolicy(lr=0.05)
    direct = OUTER_ACTIONS.index("direct")
    store.stash_decision(tx="tx", features=_feat(), action=direct,
                         goal_class="philosophy", turn_id="t")
    _handle_reward({"parent_tool_call_tx": "tx", "reward": 1.0, "source": "user"},
                   store, policy, None, _cfg({}), _Q(), "self_learning")
    upd = policy.total_updates
    trained = _handle_reward(
        {"parent_tool_call_tx": "tx", "reward": -1.0, "source": "llm_judge"},
        store, policy, None, _cfg({}), _Q(), "self_learning")
    assert trained is False
    assert policy.total_updates == upd                   # no training
    assert store.peek_decision("tx")[4] == "user"        # still user-sourced


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
