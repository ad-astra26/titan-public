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
