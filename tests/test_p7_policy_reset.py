"""P7 — clean-baseline reset (uncollapse). reset_policy_artifacts clears the
collapsed routing-policy state (π weights, IQL nets, level, replay buffer) so the
next boot cold-starts on the IQL loop, while KEEPING the verified macros."""
import os
import tempfile

import numpy as np

from titan_hcl.synthesis.mastery_level import MasteryLevel
from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM, OuterMetaPolicy


def _store(tmp):
    from titan_hcl.modules.self_learning_worker import _SelfLearningStore
    return _SelfLearningStore(path=os.path.join(tmp, "sl.duckdb"))


def _feat():
    v = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
    v[0] = 1.0
    return v.tolist()


def test_reset_clears_policy_state_but_keeps_macros():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        # populate collapsed-era state
        p = OuterMetaPolicy()
        p.init_iql()
        store.save_policy_flat(p.to_flat().tolist(), 197000, 0.0)
        store.save_iql_flat(p.iql_to_flat().tolist(), 600)
        ml = MasteryLevel(n_grades=10, grade_lo=-5, grade_hi=5)
        ml.update(2.0, 0.4, 1)
        store.save_mastery_state(ml.to_dict())
        for _ in range(50):
            store.record_reward_tuple(features=_feat(), action=0, reward=1.0, goal_class="g")
        store.mark_macro_emitted("g", 0)  # a verified macro (kept)

        assert store.load_policy_flat() is not None
        assert store.load_iql_flat() is not None
        assert store.load_mastery_state() is not None
        assert len(store.recent_reward_tuples(100)) == 50
        assert store.chunk_count() == 1

        cleared = store.reset_policy_artifacts()

        # all collapsed scaffolding gone → next boot cold-starts
        assert store.load_policy_flat() is None
        assert store.load_iql_flat() is None
        assert store.load_mastery_state() is None
        assert len(store.recent_reward_tuples(100)) == 0
        # macros KEPT (verified mastered routines, not collapsed scaffolding)
        assert store.chunk_count() == 1
        # reported counts
        assert cleared["policy_state"] == 1
        assert cleared["reward_tuples"] == 50
        assert cleared["policy_iql_state"] == 1
        assert cleared["mastery_level_state"] == 1


def test_reset_is_idempotent_on_empty():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        cleared = store.reset_policy_artifacts()  # nothing to clear
        assert cleared["policy_state"] == 0
        assert store.load_policy_flat() is None
