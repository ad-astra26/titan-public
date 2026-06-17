"""P2 — worker-side wiring: trajectory builder, IQL store persistence, and the
buffer-only _handle_reward branch (full-IQL pure-offline path)."""

import os
import tempfile

import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM, OuterMetaPolicy


def _feat(seed):
    v = np.full(OUTER_POLICY_INPUT_DIM, float(seed % 7) / 7.0, dtype=np.float32)
    v[0] = 1.0
    return v.tolist()


def test_build_routing_transitions_per_goal_class_terminal_tail():
    from titan_hcl.modules.self_learning_worker import _build_routing_transitions
    # rows pre-sorted by ts ASC; two goal_classes interleaved.
    rows = [
        (_feat(1), 0, 0.0, "lookup", 1.0),
        (_feat(2), 1, 1.0, "compute", 2.0),
        (_feat(3), 0, 0.5, "lookup", 3.0),
        (_feat(4), 1, 1.0, "compute", 4.0),
        (_feat(5), 0, 0.7, "lookup", 5.0),
    ]
    trans = _build_routing_transitions(rows)
    assert len(trans) == 5
    by_action = {}
    # lookup trajectory: 3 entries → first two non-terminal, last terminal.
    lookup = [t for t in trans if t["state"][1] in
              (_feat(1)[1], _feat(3)[1], _feat(5)[1])]
    # Simpler structural check: exactly one terminal per goal_class (2 total).
    terminals = [t for t in trans if t["terminal"]]
    assert len(terminals) == 2, "one terminal tail per goal_class"
    nonterminals = [t for t in trans if not t["terminal"]]
    assert len(nonterminals) == 3
    for t in nonterminals:
        assert t["next_state"] is not None
    for t in terminals:
        assert t["next_state"] is None


def test_build_routing_transitions_drops_bad_dim():
    from titan_hcl.modules.self_learning_worker import _build_routing_transitions
    rows = [
        (_feat(1), 0, 0.0, "g", 1.0),
        ([0.0, 1.0], 1, 1.0, "g", 2.0),  # wrong dim → dropped
        (_feat(2), 0, 0.5, "g", 3.0),
    ]
    trans = _build_routing_transitions(rows)
    assert len(trans) == 2


def test_store_iql_flat_roundtrip():
    from titan_hcl.modules.self_learning_worker import _SelfLearningStore
    with tempfile.TemporaryDirectory() as d:
        store = _SelfLearningStore(path=os.path.join(d, "sl.duckdb"))
        assert store.load_iql_flat() is None
        p = OuterMetaPolicy()
        p.init_iql()
        flat = p.iql_to_flat().tolist()
        store.save_iql_flat(flat, p.total_iql_updates)
        loaded = store.load_iql_flat()
        assert loaded is not None
        assert np.allclose(np.asarray(loaded, dtype=np.float32),
                           np.asarray(flat, dtype=np.float32))


def test_store_iql_transitions_orders_and_carries_goal_class():
    from titan_hcl.modules.self_learning_worker import _SelfLearningStore
    with tempfile.TemporaryDirectory() as d:
        store = _SelfLearningStore(path=os.path.join(d, "sl.duckdb"))
        store.record_reward_tuple(features=_feat(1), action=0, reward=0.0, goal_class="a")
        store.record_reward_tuple(features=_feat(2), action=1, reward=1.0, goal_class="b")
        store.record_reward_tuple(features=_feat(3), action=2, reward=0.5, goal_class="a")
        rows = store.iql_transitions(100)
        assert len(rows) == 3
        # ordered by ts ASC; each row is (features, action, reward, goal_class, ts)
        assert [r[3] for r in rows] == ["a", "b", "a"]
        assert rows[0][1] == 0 and rows[1][1] == 1
        # feed straight into the trajectory builder
        from titan_hcl.modules.self_learning_worker import _build_routing_transitions
        trans = _build_routing_transitions(rows)
        assert len(trans) == 3


def test_iql_consolidation_endtoend_from_store():
    """Record a balanced reward stream where goal_class 'compute' is rewarded on
    action 'tool'; the IQL pass over store transitions must learn Q(tool)>others
    on that pattern."""
    from titan_hcl.modules.self_learning_worker import (
        _SelfLearningStore, _build_routing_transitions,
    )
    np.random.seed(7)
    with tempfile.TemporaryDirectory() as d:
        store = _SelfLearningStore(path=os.path.join(d, "sl.duckdb"))
        feat = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
        feat[0] = 1.0
        feat[5] = 0.9  # a discriminating feature for the compute pattern
        for _ in range(120):
            a = int(np.random.randint(5))
            r = 1.0 if a == 1 else 0.0  # action 1 == "tool"
            store.record_reward_tuple(features=feat.tolist(), action=a,
                                      reward=r, goal_class="compute")
        rows = store.iql_transitions(2000)
        trans = _build_routing_transitions(rows)
        p = OuterMetaPolicy()
        for _ in range(8):
            p.train_iql(trans, steps=50, batch_size=32)
        q_all, _ = p._mlp2_forward(feat.reshape(1, -1), p._qw1, p._qb1, p._qw2, p._qb2)
        q_all = q_all.reshape(-1)
        assert q_all[1] == pytest.approx(q_all.max(), abs=1e-5), (
            f"IQL must learn Q(tool) is the max on the compute pattern: {q_all}")
