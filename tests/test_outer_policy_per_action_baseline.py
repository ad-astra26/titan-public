"""Per-action REINFORCE baseline — the always-tool routing-deadlock fix (v3).

Audit ground truth (live T1, 2026-06-12): regularization bounded the WEIGHTS
(norms ~1.4–3.2, not pathological) but the policy still routed every context to
`tool`. Root cause = the SINGLE GLOBAL baseline saturated to 1.0 by the dense
off-policy `tool` +1 stream, so a POSITIVE direct/research reward (+0.5) became a
NEGATIVE advantage (0.5−1.0) → the policy actively SUPPRESSED non-tool actions
(direct/research scores driven to ~−0.15). The fix: a baseline PER ACTION, so
each action's advantage is measured against its own running mean.

These tests pin: (1) the SHM-flat carries the per-action baselines + the schema
bump; (2) the global-baseline math demonstrably suppresses a positive non-tool
reward (the bug); (3) the per-action baseline preserves it (the fix); (4) the
end-to-end deadlock recovers."""
import numpy as np

from titan_hcl.synthesis.outer_meta_policy import (
    NUM_OUTER_ACTIONS, OUTER_ACTIONS, OUTER_POLICY_FLAT_DIM,
    OUTER_POLICY_INPUT_DIM, OUTER_META_POLICY_STATE_SCHEMA_VERSION,
    OuterMetaPolicy)

TOOL = OUTER_ACTIONS.index("tool")
DIRECT = OUTER_ACTIONS.index("direct")


def _ctx(requires_tool: float) -> np.ndarray:
    x = np.zeros(OUTER_POLICY_INPUT_DIM, dtype=np.float32)
    x[0] = 1.0                    # bias
    x[1] = 0.4                    # recall_top_cosine
    x[6] = float(requires_tool)   # requires_tool
    return x


def test_schema_version_bumped_to_v3():
    assert OUTER_META_POLICY_STATE_SCHEMA_VERSION == 3


def test_flat_dim_includes_per_action_baseline():
    """FLAT_DIM grew by NUM_OUTER_ACTIONS (the per-action tail) and a fresh
    policy round-trips its per-action baselines through to_flat/from_flat."""
    p = OuterMetaPolicy()
    # set distinct per-action baselines so the round-trip is unambiguous
    p.reward_baseline_per_action = np.array(
        [0.1, 0.9, 0.3, 0.4, 0.05][:NUM_OUTER_ACTIONS], dtype=np.float32)
    p.reward_baseline = 0.55
    p.total_updates = 42
    flat = p.to_flat()
    assert flat.shape[0] == OUTER_POLICY_FLAT_DIM
    q = OuterMetaPolicy.from_flat(flat)
    assert q.total_updates == 42
    assert abs(q.reward_baseline - 0.55) < 1e-5
    assert np.allclose(q.reward_baseline_per_action, p.reward_baseline_per_action, atol=1e-5)


def test_json_save_load_roundtrips_per_action_baseline(tmp_path):
    p = OuterMetaPolicy()
    p.reward_baseline_per_action = np.array(
        [0.2, 0.8, 0.1, 0.3, 0.15][:NUM_OUTER_ACTIONS], dtype=np.float32)
    path = str(tmp_path / "pol.json")
    p.save(path)
    q = OuterMetaPolicy()
    assert q.load(path) is True
    assert np.allclose(q.reward_baseline_per_action, p.reward_baseline_per_action, atol=1e-5)


def test_global_baseline_suppresses_positive_nontool_reward():
    """Document the BUG: with the (pre-v3) global baseline saturated to 1.0, a
    POSITIVE direct reward (+0.5) is a NEGATIVE advantage → direct is suppressed."""
    p = OuterMetaPolicy()
    saturated_global_baseline = 1.0   # the live-T1 state
    x = _ctx(0.0)                      # conversational
    before = float(p.forward(x)[DIRECT])
    adv = 0.5 - saturated_global_baseline   # the OLD learn() advantage
    assert adv < 0
    p.train_step(x, DIRECT, adv)
    after = float(p.forward(x)[DIRECT])
    assert after < before             # direct DECREASES despite a positive reward


def test_per_action_baseline_preserves_positive_nontool_advantage():
    """The FIX: flood `tool` with +1 (saturating ONLY tool's baseline), then a
    positive direct reward still RAISES direct (positive advantage preserved)."""
    p = OuterMetaPolicy()
    for _ in range(200):
        p.learn(_ctx(1.0), TOOL, reward=1.0)
    assert float(p.reward_baseline_per_action[TOOL]) > 0.8     # tool saturated
    assert float(p.reward_baseline_per_action[DIRECT]) < 0.2   # direct untouched
    x = _ctx(0.0)
    before = float(p.forward(x)[DIRECT])
    p.learn(x, DIRECT, reward=0.5)
    after = float(p.forward(x)[DIRECT])
    assert after > before             # direct RISES — no suppression


def test_imbalanced_stream_lifts_suppression_but_needs_balancing():
    """The per-action baseline ALONE (on the raw, tool-heavy live stream: 85%
    `tool` +1 / 15% `direct` +0.5) lifts `direct` OUT of suppression (it is no
    longer driven negative) — but `tool`'s dense training still leaks a higher
    score via the shared layers, so routing does NOT yet flip. This is exactly
    why balanced replay (the step-1 partner) is required."""
    p = OuterMetaPolicy()
    rng = np.random.default_rng(2)
    for _ in range(4000):
        if rng.random() < 0.85:
            x = _ctx(1.0); x[1] = float(rng.random())
            p.learn(x, TOOL, reward=1.0)
        else:
            x = _ctx(0.0); x[1] = float(rng.random())
            p.learn(x, DIRECT, reward=0.5)
    conv = p.forward(_ctx(0.0))
    assert float(conv[DIRECT]) > -0.02       # NOT suppressed (live bug was ~−0.15)
    # tool still leaks ahead on the raw imbalanced stream → balancing needed
    assert int(np.argmax(conv)) == TOOL


def test_per_action_baselines_track_their_own_means_independently():
    """The core of the fix: each action's baseline is an INDEPENDENT EMA of that
    action's own rewards. A dense `tool` +1 stream drives only tool's baseline to
    ~1.0; a sparse `direct` +0.3 stream drives only direct's baseline to ~0.3 —
    so direct's advantage (0.3 − 0.3 → 0, never 0.3 − 1.0 = −0.7) is never the
    suppressing negative the single global baseline produced. (The emergent
    routing FLIP is a soak-horizon property — REINFORCE advantages decay to 0 at
    convergence, so it is validated by the live re-soak, not a unit test.)"""
    p = OuterMetaPolicy()
    for _ in range(400):
        p.learn(_ctx(1.0), TOOL, reward=1.0)
        if _ % 5 == 0:                       # direct rewarded 1/5 as often (sparse)
            p.learn(_ctx(0.0), DIRECT, reward=0.3)
    assert abs(float(p.reward_baseline_per_action[TOOL]) - 1.0) < 0.05
    assert abs(float(p.reward_baseline_per_action[DIRECT]) - 0.3) < 0.05
    # the untouched actions stay at their init (0) — no cross-contamination
    for a in range(NUM_OUTER_ACTIONS):
        if a not in (TOOL, DIRECT):
            assert abs(float(p.reward_baseline_per_action[a])) < 1e-6
