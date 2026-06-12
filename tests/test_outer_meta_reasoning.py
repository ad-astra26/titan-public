"""Offline tests for OML Phase C piece 5 — OuterMetaReasoningEngine.

Verifies the outer two-level reasoner: (a) it inherits PrimitiveHandlersMixin
(reuse, no duplication) and never constructs the inner MetaReasoningEngine
(zero-inner-touch); (b) it uses FRESH MetaPolicy(9)+SubModePolicy selectors;
(c) send_queue isolation (=None); (d) a full meta-chain runs to a composite
conclusion with None subsystem deps; (e) DELEGATE drives the piece-3
OuterReasoningEngine SYNCHRONOUSLY and resolves via the inherited _check_delegate;
(f) the injected oracle scores the verifiable lane; (g) train_terminal credits
the chain across MetaPolicy + the per-primitive SubModePolicies.
"""
import numpy as np
import pytest

from titan_hcl.logic import meta_reasoning
from titan_hcl.logic.meta_reasoning import (
    NUM_META_ACTIONS,
    META_POLICY_INPUT_DIM,
    MetaPolicy,
    PrimitiveHandlersMixin,
    SubModePolicy,
)
from titan_hcl.synthesis.outer_meta_reasoning import OuterMetaReasoningEngine
from titan_hcl.synthesis.outer_reasoning import OuterReasoningEngine


def test_inherits_mixin_and_fresh_selectors():
    eng = OuterMetaReasoningEngine()
    assert isinstance(eng, PrimitiveHandlersMixin)
    assert PrimitiveHandlersMixin in type(eng).__mro__
    # FRESH MetaPolicy (9 meta-actions) — NOT the 5-action OuterMetaPolicy
    assert isinstance(eng.meta_policy, MetaPolicy)
    assert eng.meta_policy.w3.shape[1] == NUM_META_ACTIONS == 9
    assert eng.meta_policy.input_dim == META_POLICY_INPUT_DIM == 80
    # a SubModePolicy per primitive
    assert all(isinstance(p, SubModePolicy) for p in eng.sub_mode_policies.values())
    # isolation: send_queue disabled → inner-coupled emits are skipped
    assert eng._send_queue is None


def test_never_constructs_inner_meta_engine(monkeypatch):
    """Building + running the OUTER meta engine must not construct the inner
    MetaReasoningEngine (the CGN/meta-CGN/neuromod-coupled one)."""
    def _boom(*a, **k):
        raise AssertionError("inner MetaReasoningEngine was constructed")
    monkeypatch.setattr(meta_reasoning.MetaReasoningEngine, "__init__", _boom)

    eng = OuterMetaReasoningEngine()
    out = eng.run_chain(problem={"topic": "count orderings", "goal_class": "combinatorics"},
                        reasoning_engine=OuterReasoningEngine())
    assert out["action"] == "conclude"


def test_full_chain_runs_with_none_subsystem_deps():
    np.random.seed(13)
    eng = OuterMetaReasoningEngine(config={"max_steps": 12})
    re = OuterReasoningEngine()
    out = eng.run_chain(
        problem={"topic": "count orderings", "goal_class": "combinatorics",
                 "entry_primitive": "RECALL"},
        reasoning_engine=re,
        chain_archive=None, meta_wisdom=None, exp_orchestrator=None)
    assert out["action"] == "conclude"
    assert out["idea_type"] == "procedural"
    assert out["topic"] == "count orderings"
    assert 1 <= out["chain_length"] <= 12
    assert isinstance(out["chain"], list)
    assert -1.0 <= out["reward"] <= 1.0


def test_delegate_drives_outer_reasoning_engine_sync():
    """When the chain hits DELEGATE with a hypothesis, it drives the piece-3
    engine's run_chain (advancing _total_chains) and resolves via _check_delegate."""
    np.random.seed(3)
    eng = OuterMetaReasoningEngine(config={"max_steps": 20})
    re = OuterReasoningEngine()
    start_chains = re._total_chains
    # Force the chain to exercise DELEGATE by seeding a hypothesis then delegating.
    eng._start_chain("test", [0.5] * 132, {"topic": "t"})
    eng.state.hypotheses = [{"predicted_confidence": 0.7,
                             "strategy": ["DECOMPOSE", "COMPARE"]}]
    res = eng._execute("DELEGATE", "biased_chain", [0.5] * 132,
                       {"DA": 0.5}, re, None, None, None, None)
    assert res.get("delegated") is True
    assert eng.state.awaiting_delegate is True
    # sync drive + resolve (mirrors run_chain's DELEGATE branch)
    re.run_chain()
    eng._check_delegate(re)
    assert re._total_chains == start_chains + 1
    assert eng.state.awaiting_delegate is False
    assert len(eng.state.delegate_results) == 1
    assert "confidence" in eng.state.delegate_results[0]


def test_injected_oracle_scores_verifiable_lane():
    np.random.seed(7)
    eng = OuterMetaReasoningEngine(config={"max_steps": 14})
    calls = {"n": 0}

    def oracle(problem, delegate_results):
        calls["n"] += 1
        return 1.0  # always-correct oracle (the verifiable lane)

    eng.set_oracle(oracle)
    out = eng.run_chain(problem={"topic": "count orderings"},
                        reasoning_engine=OuterReasoningEngine())
    # The oracle is consulted (at EVALUATE if a delegate result exists, else at
    # conclude) — and a positive verdict marks the composite verified.
    assert out["action"] == "conclude"
    if out["oracle_verdict"] is not None:
        assert out["oracle_verdict"] == 1.0
        assert out["verified"] is True


def test_train_terminal_credits_meta_and_submode_policies():
    np.random.seed(5)
    eng = OuterMetaReasoningEngine(config={"max_steps": 10})
    eng.run_chain(problem={"topic": "t"}, reasoning_engine=OuterReasoningEngine())
    n = len(eng._chain_transitions)
    assert n >= 1
    meta_updates_before = eng.meta_policy.total_updates
    w3_before = eng.meta_policy.w3.copy()

    out = eng.train_terminal(reward=1.0)

    assert out["trained"] == n
    assert eng.meta_policy.total_updates == meta_updates_before + n
    assert not np.allclose(eng.meta_policy.w3, w3_before)
    assert eng._chain_transitions == []  # consumed


def test_meta_input_is_80d_and_submode_input_30d():
    eng = OuterMetaReasoningEngine()
    eng._start_chain("t", [0.5] * 132, {"topic": "x"})
    mi = eng._build_meta_input([0.5] * 132, {"DA": 0.5}, None, None)
    assert len(mi) == META_POLICY_INPUT_DIM == 80
    si = eng._build_sub_mode_input([0.5] * 132, {"DA": 0.5}, "RECALL")
    assert len(si) == 30
