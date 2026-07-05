"""B.2 — turn-judge-learned per-model quality for the adaptive router.

RFP_load_adaptive_inference_routing §7.B.2. The router's static `_qprior` is
displaced by a LEARNED per-model quality EMA folded from the synthesis turn-judge
reward (delivered via MODEL_QUALITY_FEEDBACK). These tests assert the fold, the
normalization ([-1,+1] judge → [0,1] quality), the cold-start handover at
`min_quality_samples`, the composite-reward effect, and v1→v2 state migration.
"""
import json
import os
import tempfile

from titan_hcl.inference.adaptive_router import AdaptiveRouter


def _router(**cfg):
    cfg.setdefault("router_enabled", True)
    return AdaptiveRouter(cfg)


def test_feedback_quality_normalizes_judge_reward_to_unit_scale():
    """good=+1→1.0, ok=0→0.5, poor=-1→0.0 (single sample = the normalized value)."""
    r = _router(quality_alpha=1.0)  # alpha=1 → EMA == latest observation
    r.feedback_quality("m_good", 1.0)
    r.feedback_quality("m_ok", 0.0)
    r.feedback_quality("m_poor", -1.0)
    assert r._quality_ema["m_good"] == 1.0
    assert r._quality_ema["m_ok"] == 0.5
    assert r._quality_ema["m_poor"] == 0.0


def test_quality_ema_folds_and_counts():
    r = _router(quality_alpha=0.5)
    r.feedback_quality("m", 1.0)     # q_obs 1.0 → ema 1.0 (first)
    r.feedback_quality("m", -1.0)    # q_obs 0.0 → ema 0.5
    assert abs(r._quality_ema["m"] - 0.5) < 1e-9
    assert r._quality_n["m"] == 2


def test_q_for_uses_prior_until_min_samples_then_learned():
    """`_q_for` returns the static prior until the model has min_quality_samples
    judged turns, then switches to the learned EMA."""
    r = _router(min_quality_samples=3, quality_alpha=1.0)
    managed = r.managed
    prior = r._qprior[managed]
    # below threshold → prior
    r.feedback_quality(managed, -1.0)   # n=1, ema 0.0
    r.feedback_quality(managed, -1.0)   # n=2
    assert r._q_for(managed) == prior
    # at/above threshold → learned (which we've driven to 0.0, well below prior)
    r.feedback_quality(managed, -1.0)   # n=3
    assert r._q_for(managed) == 0.0
    assert r._q_for(managed) != prior


def test_learned_poor_quality_lowers_composite_reward():
    """A model judged consistently poor earns a lower composite than the same
    model at its prior — the quality term actually moves the reward."""
    r = _router(min_quality_samples=1, quality_alpha=1.0,
                w_latency_chat=0.0, w_quality_chat=1.0, w_cost_chat=0.0)
    managed = r.managed
    baseline = r._reward(managed, latency_s=1.0, is_chat=True)  # prior quality
    for _ in range(2):
        r.feedback_quality(managed, -1.0)                       # learn poor
    learned = r._reward(managed, latency_s=1.0, is_chat=True)
    assert learned < baseline


def test_disabled_router_ignores_quality_feedback():
    r = _router(router_enabled=False)
    r.feedback_quality("m", 1.0)
    assert "m" not in r._quality_ema


def test_state_roundtrips_quality():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.json")
        r = _router(router_state_path=p, quality_alpha=1.0)
        r.feedback_quality("ministral-3:14b", 1.0)
        r._table = {"low|chat": {"gemma4:31b": {"r": 0.9, "n": 5}}}
        r.save()
        r2 = _router(router_state_path=p)
        assert r2._quality_ema.get("ministral-3:14b") == 1.0
        assert r2._quality_n.get("ministral-3:14b") == 1
        assert r2._table["low|chat"]["gemma4:31b"]["n"] == 5


def test_v1_state_upgrades_without_wiping_bandit_table():
    """A pre-B.2 (schema v1) state file — bandit table only, no quality keys — must
    load its learned buckets intact (never wiped on upgrade); quality starts empty."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.json")
        with open(p, "w") as f:
            json.dump({"schema": 1,
                       "buckets": {"high|chat": {"gemma4:31b": {"r": 0.7, "n": 42}}}}, f)
        r = _router(router_state_path=p)
        assert r._table["high|chat"]["gemma4:31b"]["n"] == 42   # preserved
        assert r._quality_ema == {}                              # fresh, re-learns
        assert r._quality_n == {}
