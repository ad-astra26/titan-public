"""Anti-runaway regularization for OuterMetaPolicy (2026-06-11).

The unregularized REINFORCE + off-policy `tool` attribution (every verified
tool-use credits `tool`) drove the policy weights unbounded → scores ~1100 →
argmax collapsed to always-`tool`, feature-independent (verified live T3).
These tests pin the fix: weight_decay + per-matrix max-norm cap bound the
runaway, is_pathological() detects a persisted collapse (worker self-heal),
and feature-dependent routing is recoverable after a balanced regimen."""
import numpy as np

from titan_hcl.synthesis.outer_meta_policy import (
    OUTER_ACTIONS, OUTER_POLICY_INPUT_DIM, OuterMetaPolicy)

TOOL = OUTER_ACTIONS.index("tool")
DIRECT = OUTER_ACTIONS.index("direct")


def _ctx(requires_tool: float) -> np.ndarray:
    """An 11-D feature vector; index 6 = requires_tool, 0 = bias."""
    x = np.zeros(OUTER_POLICY_INPUT_DIM, dtype=np.float32)
    x[0] = 1.0                 # bias
    x[1] = 0.4                 # recall_top_cosine
    x[6] = float(requires_tool)  # requires_tool
    return x


def _hammer_tool(policy, steps: int) -> None:
    """Simulate the off-policy `tool` attribution: reinforce `tool` on every
    step across varied contexts with a constant positive advantage."""
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = _ctx(rng.integers(0, 2))
        x[1] = float(rng.random())
        policy.train_step(x, TOOL, advantage=1.0)


def _max_abs_score(policy) -> float:
    x = _ctx(1.0)
    return float(np.max(np.abs(policy.forward(x))))


def test_weight_decay_bounds_scores():
    """The fix: weight_decay + norm-cap keep scores bounded under the same regimen."""
    p = OuterMetaPolicy(weight_decay=0.001, max_weight_norm=6.0)
    _hammer_tool(p, 5000)
    assert np.all(np.isfinite(p.forward(_ctx(1.0))))
    assert _max_abs_score(p) < 100.0   # is_pathological threshold — never reached


def test_max_weight_norm_caps_matrices():
    """The hard backstop: no weight matrix exceeds max_weight_norm after training."""
    cap = 6.0
    p = OuterMetaPolicy(weight_decay=0.0, max_weight_norm=cap)
    _hammer_tool(p, 4000)
    for w in (p.w1, p.w2, p.w3):
        assert float(np.linalg.norm(w)) <= cap + 1e-3


def test_is_pathological_detects_runaway_and_fresh_is_clean():
    fresh = OuterMetaPolicy()
    assert fresh.is_pathological() is False
    # Reconstruct the live-T3 collapse directly (inflated weights → huge scores),
    # since the softmax gradient vanishes at convergence and can't be trained
    # into a runaway in a unit test. This is the state the worker self-heals.
    runaway = OuterMetaPolicy(weight_decay=0.0, max_weight_norm=0.0)
    runaway.w1 = runaway.w1 * 50.0
    runaway.w3 = runaway.w3 * 50.0
    assert _max_abs_score(runaway) > 100.0
    assert runaway.is_pathological() is True


def test_norm_cap_reins_in_an_inflated_policy():
    """The hard cap actively reduces a runaway: one train_step on an inflated
    policy rescales every weight matrix back under max_weight_norm."""
    p = OuterMetaPolicy(weight_decay=0.001, max_weight_norm=6.0)
    p.w1 = p.w1 * 50.0
    p.w3 = p.w3 * 50.0
    p.train_step(_ctx(1.0), TOOL, advantage=1.0)
    for w in (p.w1, p.w2, p.w3):
        assert float(np.linalg.norm(w)) <= 6.0 + 1e-3


def test_feature_dependent_routing_recoverable():
    """After a BALANCED regimen (tool when requires_tool=1, direct when 0), the
    regularized policy routes feature-dependently — the property the runaway
    destroyed (it routed everything to tool regardless of requires_tool)."""
    p = OuterMetaPolicy(weight_decay=0.001, max_weight_norm=6.0)
    rng = np.random.default_rng(1)
    for _ in range(6000):
        if rng.random() < 0.5:
            x = _ctx(1.0); x[1] = float(rng.random())
            p.train_step(x, TOOL, advantage=1.0)
        else:
            x = _ctx(0.0); x[1] = float(rng.random())
            p.train_step(x, DIRECT, advantage=1.0)
    assert int(np.argmax(p.forward(_ctx(1.0)))) == TOOL      # computable → tool
    assert int(np.argmax(p.forward(_ctx(0.0)))) == DIRECT    # conversational → direct
