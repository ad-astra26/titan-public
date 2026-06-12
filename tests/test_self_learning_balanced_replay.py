"""Balanced experience-replay for the idle EXPLORE tick (deadlock fix, step 1).

`recent_reward_tuples` is strict time-order, so on the live tool-heavy stream a
replay batch is ~all `tool` → it just re-reinforces the always-tool collapse.
`balanced_reward_tuples` draws round-robin PER ACTION so the minority
direct/research/IDK samples train at parity. Pinned here against a tool-heavy
store where the minority actions are buried beneath the recent window."""
import numpy as np

from titan_hcl.modules.self_learning_worker import _SelfLearningStore
from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS, OUTER_POLICY_INPUT_DIM

TOOL = OUTER_ACTIONS.index("tool")
DIRECT = OUTER_ACTIONS.index("direct")
IDK = OUTER_ACTIONS.index("IDK")


def _feats():
    return [0.0] * OUTER_POLICY_INPUT_DIM


def test_balanced_replay_surfaces_minority_actions(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "sl.duckdb"))
    try:
        # minority non-tool samples inserted FIRST (so they fall outside the
        # recent window), then a dense `tool` stream that dominates recency.
        for _ in range(3):
            store.record_reward_tuple(features=_feats(), action=DIRECT, reward=0.5, goal_class="g")
        for _ in range(2):
            store.record_reward_tuple(features=_feats(), action=IDK, reward=0.2, goal_class="u")
        for _ in range(90):
            store.record_reward_tuple(features=_feats(), action=TOOL, reward=1.0, goal_class="c")

        recent = store.recent_reward_tuples(16)
        recent_actions = {a for _, a, _ in recent}
        # time-order replay is monopolised by `tool` — the bug
        assert recent_actions == {TOOL}

        balanced = store.balanced_reward_tuples(16)
        bal_actions = {a for _, a, _ in balanced}
        # the minority lanes are now represented — the fix
        assert DIRECT in bal_actions
        assert IDK in bal_actions
        assert TOOL in bal_actions
        # no single action monopolises the balanced batch
        from collections import Counter
        counts = Counter(a for _, a, _ in balanced)
        assert counts[TOOL] <= 4   # per_action cap = 16//5 + 1 = 4
    finally:
        store.close()


def test_balanced_replay_empty_store_is_safe(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "empty.duckdb"))
    try:
        assert store.balanced_reward_tuples(16) == []
    finally:
        store.close()


def test_clear_reward_tuples_wipes_collapsed_history(tmp_path):
    """Therapeutic cold-start cleaning (§24.7): clear_reward_tuples wipes the
    replay history so a cold-start is a clean baseline (no collapsed-era replay)."""
    store = _SelfLearningStore(path=str(tmp_path / "clear.duckdb"))
    try:
        for _ in range(30):
            store.record_reward_tuple(features=_feats(), action=TOOL, reward=1.0, goal_class="c")
        assert len(store.recent_reward_tuples(50)) == 30
        n = store.clear_reward_tuples()
        assert n == 30
        assert store.recent_reward_tuples(50) == []
        assert store.balanced_reward_tuples(16) == []
        assert store.clear_reward_tuples() == 0   # safe on an empty store
    finally:
        store.close()


def test_distinct_recent_contexts_spans_goal_classes(tmp_path):
    store = _SelfLearningStore(path=str(tmp_path / "ctx.duckdb"))
    try:
        for gc in ("conversational", "computable", "unknowable"):
            for _ in range(4):
                store.record_reward_tuple(features=_feats(), action=TOOL, reward=0.0, goal_class=gc)
        ctxs = store.distinct_recent_contexts(24)
        classes = {gc for _, gc in ctxs}
        assert classes == {"conversational", "computable", "unknowable"}
    finally:
        store.close()


def test_structural_explore_moves_policy_and_logs(tmp_path):
    """The step-2 plumbing end-to-end: _structural_explore pulls recalled
    contexts, Boltzmann-explores, scores structurally, learns, persists, logs."""
    import numpy as np
    from titan_hcl.modules.self_learning_worker import _DEFAULTS, _structural_explore
    from titan_hcl.synthesis.outer_meta_policy import (
        OUTER_ACTIONS, OuterMetaPolicy, OUTER_POLICY_INPUT_DIM)

    store = _SelfLearningStore(path=str(tmp_path / "expl.duckdb"))
    try:
        # an unknowable context (recall=0) seeded under its goal_class
        unk = [0.0] * OUTER_POLICY_INPUT_DIM
        unk[0] = 1.0  # bias; recall_top_cosine (idx 1) stays 0 → "doesn't know"
        for _ in range(6):
            store.record_reward_tuple(features=unk, action=OUTER_ACTIONS.index("tool"),
                                      reward=0.0, goal_class="unknowable")
        np.random.seed(3)                  # select_action draws from global np.random
        policy = OuterMetaPolicy()
        cfg = dict(_DEFAULTS)
        cfg["explore_structural_batch"] = 32
        before = int(policy.total_updates)
        # life=None → research affordable; many ticks to learn
        for _ in range(200):
            _structural_explore(cfg, store, policy, None, None, "test")
        assert int(policy.total_updates) > before          # it trained
        vec = np.asarray(unk, dtype=np.float32)
        # affordable + no recall → research is the structural target it learns
        assert int(policy.exploit_action(vec)) == OUTER_ACTIONS.index("research")
        # explore_log carries the structural pass
        n = store._conn.execute(
            "SELECT COUNT(*) FROM explore_log WHERE kind='structural'").fetchone()[0]
        assert int(n) > 0
    finally:
        store.close()
