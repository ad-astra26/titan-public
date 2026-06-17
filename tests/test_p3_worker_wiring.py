"""P3 — worker wiring: scale-free competence signals, graduation map, and the
MasteryLevel update/publish path in the idle explore tick."""

import os
import tempfile

import numpy as np
import pytest

from titan_hcl.synthesis.mastery_level import (
    MASTERY_LEVEL_FLAT_DIM,
    MasteryLevel,
    mastery_flat_to_readout,
)
from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM, OuterMetaPolicy


def _feat(seed=0):
    v = np.full(OUTER_POLICY_INPUT_DIM, float(seed % 5) / 5.0, dtype=np.float32)
    v[0] = 1.0
    return v.tolist()


def _store(tmp):
    from titan_hcl.modules.self_learning_worker import _SelfLearningStore
    return _SelfLearningStore(path=os.path.join(tmp, "sl.duckdb"))


def test_success_rate_is_scale_free_fraction():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        # 3 wins, 1 loss → 0.75 regardless of reward magnitude
        for r in (1.0, 5.0, 0.2, -1.0):
            store.record_reward_tuple(features=_feat(), action=1, reward=r, goal_class="g")
        assert store.success_rate(100) == pytest.approx(0.75)
        assert store.success_rate(100) <= 1.0  # dimensionless


def test_chunk_count_and_graduation_map():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        assert store.chunk_count() == 0
        assert store.is_graduated("compute") is False
        store.mark_macro_emitted("compute", 1, version=1, wins_at_emit=5)
        store.mark_macro_emitted("lookup", 0, version=1, wins_at_emit=5)
        assert store.chunk_count() == 2
        assert store.is_graduated("compute") is True
        assert store.is_graduated("translate") is False


def test_frontier_excludes_graduated_classes():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        for gc in ("compute", "lookup", "translate"):
            store.record_reward_tuple(features=_feat(), action=1, reward=1.0, goal_class=gc)
        store.mark_macro_emitted("compute", 1)  # graduate compute
        frontier = set(store.frontier_goal_classes())
        assert "compute" not in frontier          # mastered → off the frontier
        assert {"lookup", "translate"} <= frontier  # un-chunked → on the frontier


def test_mastery_state_roundtrip_through_store():
    with tempfile.TemporaryDirectory() as d:
        store = _store(d)
        assert store.load_mastery_state() is None
        ml = MasteryLevel(n_grades=10, grade_lo=-5, grade_hi=5)
        ml.update(2.0, 0.8, 1)
        store.save_mastery_state(ml.to_dict())
        loaded = store.load_mastery_state()
        ml2 = MasteryLevel(n_grades=10, grade_lo=-5, grade_hi=5)
        assert ml2.load_dict(loaded) is True
        assert ml2.readout()["grade"] == ml.readout()["grade"]


def test_update_mastery_level_publishes_competence_gated_level():
    from titan_hcl.modules.self_learning_worker import (
        _SelfLearningStore, _build_routing_transitions, _cfg, _update_mastery_level,
    )

    class _CapWriter:
        def __init__(self):
            self.last = None

        def write(self, arr):
            self.last = np.asarray(arr, dtype=np.float32)
            return 0

    np.random.seed(3)
    with tempfile.TemporaryDirectory() as d:
        store = _SelfLearningStore(path=os.path.join(d, "sl.duckdb"))
        # a verified-win stream on action 'tool' for goal_class compute
        feat = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
        feat[0] = 1.0
        for _ in range(120):
            a = int(np.random.randint(5))
            store.record_reward_tuple(features=feat.tolist(), action=a,
                                      reward=(1.0 if a == 1 else 0.0), goal_class="compute")
        store.mark_macro_emitted("compute", 1)  # one chunk
        policy = OuterMetaPolicy()
        rows = store.iql_transitions(2000)
        trans = _build_routing_transitions(rows)
        for _ in range(6):
            policy.train_iql(trans, steps=40, batch_size=32)
        cfg = _cfg({})
        ml = MasteryLevel(
            n_grades=int(cfg["level_n_grades"]),
            grade_lo=float(cfg["level_grade_lo"]), grade_hi=float(cfg["level_grade_hi"]))
        writer = _CapWriter()
        for _ in range(20):
            _update_mastery_level(cfg, store, policy, trans, ml, writer)
        # SHM vector published + decodable
        assert writer.last is not None and writer.last.shape[0] == MASTERY_LEVEL_FLAT_DIM
        readout = mastery_flat_to_readout(writer.last)
        assert readout["competence"] > 0.0      # genuine verified competence accrued
        assert readout["n_chunks"] == 1         # the chunk surfaced (graduation)
        assert readout["level"] >= 0.0
        # persisted
        assert store.load_mastery_state() is not None
