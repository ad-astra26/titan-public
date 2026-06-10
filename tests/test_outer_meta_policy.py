"""Tests for the OuterMetaPolicy — the learned outer decision operator.

RFP_synthesis_self_learning_meta_reasoning §7.A.3:
forward / train_step / save-load / exploit-vs-explore + SHM flat round-trip.
No torch, no network, no SHM file (pure numpy).
"""
import os
import tempfile

import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import (
    NUM_OUTER_ACTIONS,
    OUTER_ACTIONS,
    OUTER_POLICY_FLAT_DIM,
    OUTER_POLICY_INPUT_DIM,
    OuterFeatures,
    OuterMetaPolicy,
    action_index_to_mode,
    action_index_to_name,
)
from titan_hcl.logic.sage.grounded_router import (
    MODE_RESEARCH,
    MODE_SHADOW,
    MODE_SKILL_DELEGATE,
    MODE_SOVEREIGN,
    MODE_TOOL_ORACLE,
)


def _feat(**kw) -> np.ndarray:
    return OuterFeatures(**kw).to_vector()


def test_feature_vector_dim():
    v = OuterFeatures().to_vector()
    assert v.shape == (OUTER_POLICY_INPUT_DIM,)
    assert v.dtype == np.float32
    assert v[0] == 1.0  # bias slot


def test_feature_vector_clips_and_normalizes():
    v = OuterFeatures(
        recall_top_cosine=2.0,      # clip → 1.0
        recall_count=999,           # norm → 1.0 (capped at 10/10)
        skill_utility=-5.0,         # clip → 0.0
        skill_matched=True,
        requires_tool=True,
        msl_i_confidence=float("nan"),  # NaN → 0.0
    ).to_vector()
    assert v[1] == 1.0
    assert v[2] == 1.0
    assert v[3] == 0.0
    assert v[4] == 1.0
    assert v[6] == 1.0
    assert v[8] == 0.0


def test_forward_shape():
    p = OuterMetaPolicy()
    out = p.forward(_feat(requires_tool=True))
    assert out.shape == (NUM_OUTER_ACTIONS,)


def test_exploit_action_valid_index():
    p = OuterMetaPolicy()
    a = p.exploit_action(_feat(has_code_signal=True))
    assert 0 <= a < NUM_OUTER_ACTIONS


def test_explore_low_temperature_concentrates():
    # at near-zero temperature, sampling collapses to argmax. Seed a clear
    # winner first so the test is deterministic regardless of random init ties.
    p = OuterMetaPolicy()
    tool = OUTER_ACTIONS.index("tool")
    p.seed_prior(tool, strength=10.0)
    x = _feat(requires_tool=True, has_code_signal=True)
    assert p.exploit_action(x) == tool
    draws = [p.select_action(x, temperature=0.1) for _ in range(200)]
    # the dominant action takes ~all of the low-temperature samples
    assert draws.count(tool) > 180


def test_learn_shifts_toward_rewarded_action():
    # repeatedly reward the "tool" action on a computational feature → its
    # probability must rise measurably (the decision LEARNS, §8 G3).
    p = OuterMetaPolicy(lr=0.05)
    x = _feat(requires_tool=False, has_code_signal=True)  # §1.3: regex says no
    tool = OUTER_ACTIONS.index("tool")
    p0 = float(p.action_probs(x)[tool])
    for _ in range(150):
        p.learn(x, tool, reward=1.0)
    p1 = float(p.action_probs(x)[tool])
    assert p1 > p0 + 0.05
    assert p.total_updates == 150
    # the EMA baseline tracked the +1 rewards
    assert p.reward_baseline > 0.5


def test_learn_punishes_negative_reward():
    p = OuterMetaPolicy(lr=0.05)
    x = _feat(requires_tool=True)
    idk = OUTER_ACTIONS.index("IDK")
    p0 = float(p.action_probs(x)[idk])
    for _ in range(150):
        p.learn(x, idk, reward=-1.0)
    p1 = float(p.action_probs(x)[idk])
    assert p1 < p0  # a punished action loses probability mass


def test_save_load_roundtrip(tmp_path):
    p = OuterMetaPolicy(lr=0.05)
    x = _feat(has_code_signal=True)
    for _ in range(20):
        p.learn(x, OUTER_ACTIONS.index("tool"), reward=1.0)
    path = str(tmp_path / "policy.json")
    p.save(path)
    q = OuterMetaPolicy()
    assert q.load(path) is True
    assert q.total_updates == p.total_updates
    assert abs(q.reward_baseline - p.reward_baseline) < 1e-6
    np.testing.assert_allclose(q.forward(x), p.forward(x), rtol=1e-5, atol=1e-5)


def test_load_missing_file_returns_false(tmp_path):
    p = OuterMetaPolicy()
    assert p.load(str(tmp_path / "nope.json")) is False


def test_to_flat_from_flat_roundtrip():
    p = OuterMetaPolicy(lr=0.05)
    x = _feat(skill_matched=True, skill_utility=0.9)
    for _ in range(30):
        p.learn(x, OUTER_ACTIONS.index("skill_delegate"), reward=1.0)
    flat = p.to_flat()
    assert flat.shape == (OUTER_POLICY_FLAT_DIM,)
    assert flat.dtype == np.float32
    q = OuterMetaPolicy.from_flat(flat)
    np.testing.assert_allclose(q.forward(x), p.forward(x), rtol=1e-5, atol=1e-5)
    assert q.total_updates == p.total_updates
    assert abs(q.reward_baseline - p.reward_baseline) < 1e-5


def test_from_flat_rejects_bad_dim():
    with pytest.raises(ValueError):
        OuterMetaPolicy.from_flat(np.zeros(7, dtype=np.float32))


def test_action_to_mode_mapping():
    assert action_index_to_mode(OUTER_ACTIONS.index("direct")) == MODE_SOVEREIGN
    assert action_index_to_mode(OUTER_ACTIONS.index("tool")) == MODE_TOOL_ORACLE
    assert action_index_to_mode(OUTER_ACTIONS.index("skill_delegate")) == MODE_SKILL_DELEGATE
    assert action_index_to_mode(OUTER_ACTIONS.index("research")) == MODE_RESEARCH
    assert action_index_to_mode(OUTER_ACTIONS.index("IDK")) == MODE_SHADOW
    assert action_index_to_name(1) == "tool"


def test_seed_prior_biases_action():
    p = OuterMetaPolicy()
    x = _feat()
    tool = OUTER_ACTIONS.index("tool")
    before = float(p.action_probs(x)[tool])
    p.seed_prior(tool, strength=3.0)
    after = float(p.action_probs(x)[tool])
    assert after > before
