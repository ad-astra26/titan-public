"""Fix 2 (§24.9) — the cold-start feature-discriminating seed.

A FRESH OuterMetaPolicy's argmax is ~uniform; the first dense live reward stream
(the turn-judge, which rewards a single action regardless of features) collapses
it to that action BEFORE it learns to discriminate on features — the routing-
collapse failure mode quantified live across T1/T2/T3 (2026-06-12). `_seed_cold_start`
warm-starts a cold policy from the SAME verifiable structural oracle the idle pass
teaches (`structural_target_action`), on a synthetic balanced lane set, via the
cross-entropy `train_step`. These tests pin that the SEEDED policy routes feature-
conditionally BEFORE any live turn — INCLUDING under live-like non-zero MSL noise
(the seed jitters the MSL block precisely so the net keys on the base features, not
a memorized vector) — that it is bounded (not pathological), and that it is
deterministic (identical warm-start every cold boot)."""
import numpy as np

from titan_hcl.modules.self_learning_worker import (
    _COLD_START_LANES, _seed_cold_start, _synth_feature_vec)
from titan_hcl.synthesis.outer_meta_policy import (
    MSL_CONTEXT_DIM, OUTER_ACTIONS, OUTER_POLICY_INPUT_DIM, OuterMetaPolicy,
    _BASE_FEATURE_NAMES, structural_target_action)

DIRECT = OUTER_ACTIONS.index("direct")
TOOL = OUTER_ACTIONS.index("tool")
SKILL = OUTER_ACTIONS.index("skill_delegate")
RESEARCH = OUTER_ACTIONS.index("research")
IDK = OUTER_ACTIONS.index("IDK")

_LANE_ACTION = {
    "tool": TOOL, "direct": DIRECT, "research": RESEARCH, "skill_delegate": SKILL}

# the minimal cfg the seed reads (defaults mirror _DEFAULTS)
_CFG = {
    "cold_start_seed_epochs": 300,
    "cold_start_seed_advantage": 3.0,
    "explore_know_threshold": 0.65,   # calibrated to grounded recall_known_floor (§24.9)
    "explore_skill_floor": 0.3,
}


def _eval_vec(rng, overrides: dict) -> np.ndarray:
    """A held-out LIVE-LIKE eval vector for one lane: the pinned discriminating
    base features (same as the seed) but FRESH non-zero MSL/retrieval noise from
    an independent rng — so a pass proves the seeded routing survives live MSL,
    not just the exact training distribution."""
    n_base = len(_BASE_FEATURE_NAMES)
    vec = np.empty(OUTER_POLICY_INPUT_DIM, dtype=np.float32)
    vec[:n_base] = rng.uniform(0.0, 0.2, size=n_base).astype(np.float32)
    vec[0] = 1.0
    vec[n_base:n_base + MSL_CONTEXT_DIM] = rng.uniform(
        -1.0, 1.0, size=MSL_CONTEXT_DIM).astype(np.float32)
    vec[n_base + MSL_CONTEXT_DIM:] = rng.uniform(
        0.0, 0.3, size=OUTER_POLICY_INPUT_DIM - n_base - MSL_CONTEXT_DIM).astype(np.float32)
    for idx, val in overrides.items():
        vec[idx] = float(val)
    return vec


# ── the seed teaches the 4 feature-discriminable lanes ──────────────────────
def test_seed_routes_feature_conditionally_under_live_msl():
    """The closer: a cold-seeded policy routes computable→tool, known→direct,
    unknowable→research, skill→skill_delegate — under independent live-like MSL
    noise (16 held-out vecs per lane). This is what the live judge-collapse
    destroyed; the seed installs it from boot."""
    np.random.seed(0)
    policy = OuterMetaPolicy()
    n = _seed_cold_start(policy, _CFG)
    assert n == _CFG["cold_start_seed_epochs"] * len(_COLD_START_LANES)

    rng = np.random.default_rng(1234)   # independent of the seed's rng(0)
    for label, overrides, _afford in _COLD_START_LANES:
        want = _LANE_ACTION[label]
        for _ in range(16):
            vec = _eval_vec(rng, overrides)
            assert int(policy.exploit_action(vec)) == want, (
                f"lane {label}: routed {OUTER_ACTIONS[int(policy.exploit_action(vec))]}, "
                f"want {OUTER_ACTIONS[want]}")


def test_fresh_policy_does_NOT_route_conditionally():
    """Control: WITHOUT the seed, a fresh policy does not feature-discriminate
    across the 4 lanes (it argmaxes one/few actions regardless of features) —
    i.e. the seed is what installs the routing, not the lane vectors themselves."""
    np.random.seed(0)
    policy = OuterMetaPolicy()
    rng = np.random.default_rng(1234)
    correct = 0
    total = 0
    for label, overrides, _afford in _COLD_START_LANES:
        want = _LANE_ACTION[label]
        for _ in range(16):
            if int(policy.exploit_action(_eval_vec(rng, overrides))) == want:
                correct += 1
            total += 1
    # a fresh HE-random policy is far from the 4/4 the seed achieves
    assert correct < total, "fresh policy already perfectly routes — seed is a no-op?"


# ── the seed is bounded + deterministic ─────────────────────────────────────
def test_seeded_policy_not_pathological():
    """The seed must not blow the policy past the runaway threshold (the §24.2
    regularization bounds the cross-entropy steps)."""
    np.random.seed(0)
    policy = OuterMetaPolicy()
    _seed_cold_start(policy, _CFG)
    assert not policy.is_pathological()


def test_seed_is_deterministic():
    """Fixed rng(0) ⇒ identical warm-start every cold boot (same starting policy
    weights ⇒ same seeded weights)."""
    np.random.seed(7)
    p1 = OuterMetaPolicy()
    flat0 = p1.to_flat().copy()
    p2 = OuterMetaPolicy.from_flat(flat0)   # identical start
    _seed_cold_start(p1, _CFG)
    _seed_cold_start(p2, _CFG)
    assert np.allclose(p1.to_flat(), p2.to_flat())


# ── the synthetic vecs land in the intended structural lane ─────────────────
def test_synth_vecs_map_to_intended_targets():
    """Every lane's synthetic vec must make `structural_target_action` return the
    intended action — else the seed would teach the wrong target."""
    rng = np.random.default_rng(0)
    for label, overrides, afford in _COLD_START_LANES:
        for _ in range(50):
            vec = _synth_feature_vec(rng, overrides)
            t = structural_target_action(
                vec, affordable=afford,
                know_threshold=_CFG["explore_know_threshold"],
                skill_floor=_CFG["explore_skill_floor"])
            assert t == _LANE_ACTION[label], (
                f"lane {label}: synth vec → {OUTER_ACTIONS[t]}, want {OUTER_ACTIONS[label]}")


def test_seed_disabled_when_epochs_zero():
    """`cold_start_seed_epochs=0` ⇒ no-op (the disable path)."""
    np.random.seed(0)
    policy = OuterMetaPolicy()
    before = policy.to_flat().copy()
    n = _seed_cold_start(policy, {**_CFG, "cold_start_seed_epochs": 0})
    assert n == 0
    assert np.array_equal(before, policy.to_flat())
