"""P4 — level-driven reward shaping (ARCHITECTURE_mastery_leveling.md §4).

Pins: only POSITIVE `direct` rewards decay with level (floor>0, INV-MC-4); other
actions + negative rewards are untouched; ungrounded direct is damped MORE;
mastered (graduated) goal_classes get eased damping; the buffered reward_tuples
stay RAW (the shaping is on training transitions only — INV-ML-3)."""
import os
import tempfile

import numpy as np
import pytest

from titan_hcl.modules.self_learning_worker import (
    _DIRECT_ACTION_IDX,
    _direct_damping,
    _shape_transitions_for_level,
    _cfg,
)
from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS, OUTER_POLICY_INPUT_DIM


def _state(recall=1.0):
    s = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
    s[0] = 1.0
    s[1] = recall  # recall_top_cosine
    return s.tolist()


def _store(tmp):
    from titan_hcl.modules.self_learning_worker import _SelfLearningStore
    return _SelfLearningStore(path=os.path.join(tmp, "sl.duckdb"))


def test_direct_damping_decays_with_level_and_respects_floor():
    d0 = _direct_damping(0.0, floor=0.3, slope=0.07, graduated=False, graduated_floor=0.6)
    d5 = _direct_damping(5.0, floor=0.3, slope=0.07, graduated=False, graduated_floor=0.6)
    d20 = _direct_damping(20.0, floor=0.3, slope=0.07, graduated=False, graduated_floor=0.6)
    assert d0 == pytest.approx(1.0)
    assert d5 < d0
    assert d20 == pytest.approx(0.3)  # floored, never below (INV-MC-4)


def test_graduated_class_eases_damping():
    d = _direct_damping(20.0, floor=0.3, slope=0.07, graduated=False, graduated_floor=0.6)
    dg = _direct_damping(20.0, floor=0.3, slope=0.07, graduated=True, graduated_floor=0.6)
    assert dg > d and dg == pytest.approx(0.6)  # mastered → coasting eased to the graduated floor


def test_only_positive_direct_is_shaped():
    cfg = _cfg({})
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        direct = _DIRECT_ACTION_IDX
        tool = OUTER_ACTIONS.index("tool")
        trans = [
            {"state": _state(), "action": direct, "reward": 1.0, "goal_class": "g"},   # damped
            {"state": _state(), "action": direct, "reward": -1.0, "goal_class": "g"},  # untouched (penalty)
            {"state": _state(), "action": tool, "reward": 1.0, "goal_class": "g"},      # untouched (not direct)
        ]
        n = _shape_transitions_for_level(trans, level=10.0, cfg=cfg, store=store)
        assert n == 1
        assert trans[0]["reward"] < 1.0          # positive direct decayed
        assert trans[1]["reward"] == -1.0        # negative direct untouched (never ease a penalty)
        assert trans[2]["reward"] == 1.0         # tool untouched at full reward


def test_ungrounded_direct_damped_more():
    cfg = _cfg({})
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        direct = _DIRECT_ACTION_IDX
        grounded = [{"state": _state(recall=0.9), "action": direct, "reward": 1.0, "goal_class": "g"}]
        ungrounded = [{"state": _state(recall=0.1), "action": direct, "reward": 1.0, "goal_class": "g"}]
        _shape_transitions_for_level(grounded, level=8.0, cfg=cfg, store=store)
        _shape_transitions_for_level(ungrounded, level=8.0, cfg=cfg, store=store)
        assert ungrounded[0]["reward"] < grounded[0]["reward"], (
            "ungrounded direct (low recall) must be damped MORE")


def test_graduated_goal_class_eased_in_shaping():
    cfg = _cfg({})
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        store.mark_macro_emitted("mastered", _DIRECT_ACTION_IDX)  # graduate this class
        direct = _DIRECT_ACTION_IDX
        frontier = [{"state": _state(), "action": direct, "reward": 1.0, "goal_class": "frontier"}]
        mastered = [{"state": _state(), "action": direct, "reward": 1.0, "goal_class": "mastered"}]
        _shape_transitions_for_level(frontier, level=20.0, cfg=cfg, store=store)
        _shape_transitions_for_level(mastered, level=20.0, cfg=cfg, store=store)
        assert mastered[0]["reward"] > frontier[0]["reward"], (
            "a mastered/graduated class should have eased (higher) direct reward")
