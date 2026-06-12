"""D.4b-worker (RFP §7.D-strategy) — auto-detect macro-of-macros reuse. The
worker is numpy-only with no 384-D embedder (D7), so the reuse signal is cosine
over the 30-D mean-feature signatures of its OWN already-emitted macros: a
verified deliberation that operates in the same feature-region as ≥2 emitted
macros emits `composed_from` those child macro labels → REASONING_COMPOSED_FROM
edges (parent-of-children provenance). Covers GD4-worker.

Run: python -m pytest tests/test_macro_of_macros.py -v -p no:anchorpy
"""
from __future__ import annotations

import pytest

from titan_hcl.modules.self_learning_worker import _SelfLearningStore
from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_POLICY_INPUT_DIM, action_index_to_name,
)


@pytest.fixture
def store():
    return _SelfLearningStore(path=":memory:")


def _seed(store, goal_class, action, vec, n=6):
    for _ in range(n):
        store.record_reward_tuple(
            features=vec, action=action, reward=1.0, goal_class=goal_class)
    store.mark_macro_emitted(goal_class, action, version=1, wins_at_emit=n)


def _near_vec():
    return [0.3] * OUTER_POLICY_INPUT_DIM


def _far_vec():
    v = [0.0] * OUTER_POLICY_INPUT_DIM
    v[0] = 1.0   # one-hot → cosine ~0.18 vs all-0.3
    return v


def test_related_returns_feature_near_emitted_macros(store):
    """A/B share G's feature-region (cosine 1.0); C is far (cosine ~0.18) → only
    A,B are related; G itself is excluded."""
    _seed(store, "G", 0, _near_vec())
    _seed(store, "A", 0, _near_vec())
    _seed(store, "B", 0, _near_vec())
    _seed(store, "C", 0, _far_vec())

    related = store.related_emitted_macros(
        _near_vec(), exclude_goal_class="G", exclude_action=0,
        floor=0.85, limit=4)
    gcs = {gc for gc, _a, _c in related}
    assert gcs == {"A", "B"}                 # near in, far + self out
    assert all(cos >= 0.85 for _gc, _a, cos in related)


def test_far_only_yields_no_children(store):
    """Only a far macro present → nothing ≥ floor → not a macro-of-macros."""
    _seed(store, "G", 0, _near_vec())
    _seed(store, "C", 0, _far_vec())
    related = store.related_emitted_macros(
        _near_vec(), exclude_goal_class="G", exclude_action=0, floor=0.85)
    assert related == []


def test_child_labels_match_canonical_macro_ids(store):
    """The composed_from children use the canonical v1 macro label
    `macro::{gc}::{action_name}` — the spine reasoning_ids the synthesis handler
    links REASONING_COMPOSED_FROM to."""
    _seed(store, "G", 0, _near_vec())
    _seed(store, "A", 0, _near_vec())
    _seed(store, "B", 0, _near_vec())
    related = store.related_emitted_macros(
        _near_vec(), exclude_goal_class="G", exclude_action=0, floor=0.85)
    children = [f"macro::{gc}::{action_index_to_name(a)}" for gc, a, _c in related]
    assert len(children) >= 2
    for c in children:
        assert c.startswith("macro::") and "::" in c[len("macro::"):]
    assert f"macro::A::{action_index_to_name(0)}" in children


def test_self_excluded_even_if_present(store):
    """The target class is never its own child (no self-loop edge)."""
    _seed(store, "G", 0, _near_vec())
    _seed(store, "A", 0, _near_vec())
    related = store.related_emitted_macros(
        _near_vec(), exclude_goal_class="G", exclude_action=0, floor=0.85)
    assert all(gc != "G" for gc, _a, _c in related)


def test_zero_signature_returns_empty(store):
    _seed(store, "A", 0, _near_vec())
    assert store.related_emitted_macros(
        [0.0] * OUTER_POLICY_INPUT_DIM, exclude_goal_class="G",
        exclude_action=0, floor=0.85) == []
