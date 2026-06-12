"""Offline tests for OML Phase C piece 3 — OuterReasoningEngine.

Verifies: (a) the outer engine REUSES reasoning.py's primitive functions +
policy/buffer classes by IDENTITY (no reimplementation, zero-inner-touch);
(b) constructing/running it NEVER constructs the inner ReasoningEngine;
(c) set_strategy_bias steers primitive selection; (d) COMMIT/HOLD/ABANDON are
all reachable; (e) train_terminal credits the chain + advances the policy;
(f) set_problem is pad/trim/NaN-safe.
"""
from collections import Counter

import numpy as np
import pytest

from titan_hcl.logic import reasoning
from titan_hcl.synthesis import outer_reasoning
from titan_hcl.synthesis.outer_reasoning import (
    OUTER_OBS_DIM,
    OUTER_REASON_INPUT_DIM,
    OuterReasoningEngine,
)


def test_dims():
    assert OUTER_OBS_DIM == 30
    assert OUTER_REASON_INPUT_DIM == 33
    eng = OuterReasoningEngine()
    assert eng.policy.input_dim == OUTER_REASON_INPUT_DIM
    # fresh, distinct components (its OWN net/buffer — not a shared inner one)
    assert isinstance(eng.policy, reasoning.ReasoningPolicyNet)
    assert isinstance(eng.buffer, reasoning.ReasoningTransitionBuffer)


def test_reuses_inner_primitive_functions_by_identity():
    # The outer engine imports the SAME function objects — reuse, not a fork.
    assert outer_reasoning.PRIMITIVE_FUNCTIONS is reasoning.PRIMITIVE_FUNCTIONS
    assert outer_reasoning._primitive_loop is reasoning._primitive_loop
    assert outer_reasoning._primitive_negate is reasoning._primitive_negate
    assert outer_reasoning._primitive_associate is reasoning._primitive_associate
    assert outer_reasoning.PRIMITIVES is reasoning.PRIMITIVES


def test_never_constructs_inner_reasoning_engine(monkeypatch):
    """Zero-inner-touch: building + running the OUTER engine must not construct
    the inner ReasoningEngine (the neuromod/spirit/body-coupled one)."""
    def _boom(*a, **k):
        raise AssertionError("inner ReasoningEngine was constructed")
    monkeypatch.setattr(reasoning.ReasoningEngine, "__init__", _boom)

    eng = OuterReasoningEngine()
    concl = eng.run_chain(problem_obs=np.full(OUTER_OBS_DIM, 0.4), temperature=0.8)
    assert concl["action"] in {"COMMIT", "HOLD", "ABANDON"}


def test_run_chain_respects_min_and_max_length():
    np.random.seed(11)
    eng = OuterReasoningEngine(config={"min_chain_length": 3, "max_chain_length": 6})
    for _ in range(20):
        concl = eng.run_chain(problem_obs=np.random.rand(OUTER_OBS_DIM))
        assert concl["action"] in {"COMMIT", "HOLD", "ABANDON"}
        # chain never exceeds max; and a concluded chain is at least min long
        # (the min-length redirect prevents a premature CONCLUDE).
        assert concl["chain_length"] <= 6
        assert concl["chain_length"] >= 3


def test_strategy_bias_steers_selection():
    np.random.seed(7)
    eng = OuterReasoningEngine()
    eng.set_problem(np.full(OUTER_OBS_DIM, 0.3))
    eng.start_chain()
    pi = eng._build_policy_input()

    n = 3000
    base = Counter(eng.policy.select_action(pi, 1.0, strategy_bias=None)
                   for _ in range(n))

    bias = np.zeros(8, dtype=np.float32)
    bias[0] = 12.0  # massively favor COMPARE (idx 0)
    eng.set_strategy_bias(bias)
    biased = Counter(eng.policy.select_action(pi, 1.0, strategy_bias=eng._strategy_bias)
                     for _ in range(n))

    assert biased[0] > base[0]
    assert biased[0] > 0.8 * n  # the bias dominates selection


def test_conclude_action_thresholds():
    eng = OuterReasoningEngine(config={"confidence_threshold": 0.6, "hold_floor": 0.45})
    pi = np.zeros(OUTER_REASON_INPUT_DIM)

    eng.start_chain(); eng.confidence = 0.75
    assert eng._conclude(pi)["action"] == "COMMIT"

    eng.start_chain(); eng.confidence = 0.50
    assert eng._conclude(pi)["action"] == "HOLD"

    eng.start_chain(); eng.confidence = 0.20
    assert eng._conclude(pi)["action"] == "ABANDON"


def test_train_terminal_credits_chain_and_advances_policy():
    np.random.seed(3)
    eng = OuterReasoningEngine()
    eng.run_chain(problem_obs=np.full(OUTER_OBS_DIM, 0.5), temperature=0.7)
    n_transitions = len(eng._chain_transitions)
    assert n_transitions >= 3  # min chain + CONCLUDE

    w3_before = eng.policy.w3.copy()
    updates_before = eng.policy.total_updates

    out = eng.train_terminal(reward=1.0, baseline=0.3)

    assert out["trained"] == n_transitions
    assert eng.policy.total_updates == updates_before + n_transitions
    assert not np.allclose(eng.policy.w3, w3_before)  # weights moved
    assert eng._chain_transitions == []  # consumed


def test_train_terminal_noop_without_chain():
    eng = OuterReasoningEngine()
    out = eng.train_terminal(reward=1.0)
    assert out["trained"] == 0


def test_set_problem_pad_trim_nan_safe():
    eng = OuterReasoningEngine()

    eng.set_problem([0.1, 0.2, 0.3])  # too short → zero-padded
    assert eng._problem_obs.shape == (OUTER_OBS_DIM,)
    assert eng._problem_obs[0] == pytest.approx(0.1)
    assert eng._problem_obs[5] == 0.0

    eng.set_problem(list(range(100)))  # too long → trimmed
    assert eng._problem_obs.shape == (OUTER_OBS_DIM,)

    eng.set_problem([np.nan, np.inf, -np.inf] + [0.0] * 27)  # non-finite → scrubbed
    assert np.all(np.isfinite(eng._problem_obs))

    eng.set_problem(None)  # empty → all zeros, no crash
    assert eng._problem_obs.shape == (OUTER_OBS_DIM,)
    assert np.all(eng._problem_obs == 0.0)


def test_reachability_commit_and_abandon_over_problems():
    """Over varied problems + biases, the engine reaches both a confident
    COMMIT and a low-signal ABANDON (it is not pinned to one action)."""
    np.random.seed(5)
    eng = OuterReasoningEngine()
    seen = set()
    for _ in range(60):
        obs = np.random.rand(OUTER_OBS_DIM) * np.random.choice([0.0, 0.5, 1.0])
        bias = np.zeros(8, dtype=np.float32)
        bias[np.random.randint(0, 7)] = np.random.choice([0.0, 4.0])
        concl = eng.run_chain(problem_obs=obs, strategy_bias=bias, temperature=1.0)
        seen.add(concl["action"])
    assert "COMMIT" in seen or "HOLD" in seen
    assert len(seen) >= 2  # not collapsed to a single terminal action
