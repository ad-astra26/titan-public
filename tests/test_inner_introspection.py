"""Tests for the Inner Turn primitives (RFP_introspective_inner_turn Phase A).

Asserts the LOAD-BEARING invariants:
  • INV-IT-1 — r_inner is pure measured-telemetry error (no LLM/judge/quality).
  • Q1/INV-IT-4 — the inner domain is a SEPARATE net/predictor/level (own dims).
  • G2 — the inner level CLIMBS from self-prediction accuracy (engagement-indep).
  • INV-IT-9 — the local refractory: WIN→θ−α / LOSE→θ+β, bounded.
  • G6 — firing is great-pulse-gated, never a timer.
  • None-safety of the SHM-reader assembly (cold-start → zeros).
"""
import inspect

import numpy as np

from titan_hcl.synthesis import inner_introspection as ii
from titan_hcl.synthesis.mastery_level import MasteryLevel


def _rng():
    return np.random.default_rng(7)


def test_dims_and_stances():
    assert ii.INNER_NUM_STANCES == 5
    assert ii.INNER_STANCES == ("body", "mind", "spirit", "affect", "trajectory")
    assert ii.INNER_STATE_DIM == 71
    # Q7: dedicated φ dim ≠ 30 (auto-excluded from outer len-30 filter).
    assert ii.INNER_PHI_DIM != 30


def test_assemble_shapes_and_none_safety():
    body = {"values": [0.1] * 5}
    mind = {"values": [0.2] * 15}
    spirit = {"values": list(np.linspace(-1, 1, 45))}
    neuro = {"modulators": {n: {"level": 0.5} for n in ii.NEUROMOD_ORDER}}
    s = ii.assemble_inner_state(body, mind, spirit, neuro)
    assert s.shape == (71,)
    # cold-start: every reader None → all-zero, never raises (INV-IT-3 step 1).
    sN = ii.assemble_inner_state(None, None, None, None)
    assert sN.shape == (71,) and np.allclose(sN, 0.0)


def test_znorm_per_channel_scales_blocks_comparably():
    # spirit (45-D) raw has a much larger scale than body (5-D); after per-channel
    # z-norm each block is ~unit-std, so spirit cannot dominate the error norm (Q5).
    s = np.concatenate([np.full(5, 0.1), np.full(15, 0.2),
                        np.linspace(-50, 50, 45), np.full(6, 0.3)]).astype("float32")
    n = ii.znorm_channels(s)
    spirit = n[20:65]
    assert abs(float(spirit.std()) - 1.0) < 0.2          # standardized
    # a constant block → zeros (no information), never NaN.
    assert np.all(np.isfinite(n))
    assert np.allclose(n[:5], 0.0)                        # body block is constant


def test_reward_kernel_is_pure_telemetry():
    """INV-IT-1 / G3 — the reward is a deterministic function of the four numeric
    inputs ONLY. Static guard: the kernel source contains no judge/score/LLM/
    provider call."""
    # Scan the EXECUTABLE body only (strip the docstring, which legitimately
    # describes what the kernel must NOT do).
    fn = ii.inner_reward_kernel
    src_no_doc = inspect.getsource(fn).replace(fn.__doc__ or "", "")
    for forbidden in ("judge", "provider", "complete(", "llm", "verify_safety"):
        assert forbidden not in src_no_doc.lower(), f"reward kernel references {forbidden!r}"
    # perfect prediction → reward 1.0; worst → clipped at −1.0.
    s0 = _rng().normal(0, 1, 71).astype("float32")
    s1 = s0 * 0.9
    perfect = ii.inner_reward_kernel(s0, s1 - s0, s0, s1)
    assert abs(perfect["reward"] - 1.0) < 1e-5
    assert perfect["e_descr"] < 1e-5 and perfect["e_delta"] < 1e-5
    bad = ii.inner_reward_kernel(s0 + 100, s1 * 0 + 100, s0, s1)
    assert bad["reward"] == -1.0


def test_self_predictor_learns_and_reduces_error():
    """The self-model improves with practice → error falls (the climb signal)."""
    rng = _rng()
    pred = ii.InnerSelfPredictor()
    early, late = [], []
    for it in range(400):
        s = rng.normal(0, 1, 71).astype("float32")
        n = ii.znorm_channels(s)
        phi = ii.build_inner_phi(n)
        nxt = n * 0.9
        d, dl = pred.predict(phi, 2)
        r = ii.inner_reward_kernel(d, dl, n, nxt)
        (early if it < 30 else late).append(r["e_descr"] + r["e_delta"])
        pred.learn(phi, 2, n, nxt - n)
    assert np.mean(late) < 0.6 * np.mean(early)          # clear improvement


def test_inner_level_climbs_from_accuracy_engagement_independent():
    """G2/INV-IT-4 — with ZERO chat traffic, the InnerMasteryLevel level rises as
    self-prediction accuracy improves. Uses a SEPARATE MasteryLevel instance."""
    rng = _rng()
    pred = ii.InnerSelfPredictor()
    iq = ii.InnerIQL()
    lvl = MasteryLevel()
    first = last = None
    for ep in range(500):
        s = rng.normal(0, 1, 71).astype("float32")
        n = ii.znorm_channels(s)
        phi = ii.build_inner_phi(n)
        stance = iq.select_stance(phi)
        nxt = ii.znorm_channels(s * 0.9 + rng.normal(0, 0.05, 71).astype("float32"))
        d, dl = pred.predict(phi, stance)
        r = ii.inner_reward_kernel(d, dl, n, nxt)
        pred.learn(phi, stance, n, nxt - n)
        iq.train_iql([{"state": phi, "action": stance, "reward": r["reward"],
                       "next_state": None, "terminal": True}] * 4, steps=4)
        ro = lvl.update(iq.value_symlog(phi), iq.advantage_positive_rate())
        if ep == 0:
            first = ro["level"]
        last = ro["level"]
    assert last > first                                  # the level moved up


def test_inner_iql_trains_and_persists():
    rng = _rng()
    iq = ii.InnerIQL()
    trans = [{"state": ii.build_inner_phi(ii.znorm_channels(rng.normal(0, 1, 71))),
              "action": int(rng.integers(0, 5)), "reward": float(rng.uniform(0, 1)),
              "next_state": None, "terminal": True} for _ in range(40)]
    out = iq.train_iql(trans, steps=20)
    assert out["iql_updates"] == 20
    assert 0.0 <= out["adv_pos_rate"] <= 1.0
    # round-trips exactly.
    iq2 = ii.InnerIQL()
    assert iq2.load_dict(iq.to_dict())
    assert np.isclose(iq2.value_symlog(trans[0]["state"]), iq.value_symlog(trans[0]["state"]))


def test_drive_refractory_direction_and_gating():
    """INV-IT-9 — WIN lowers θ, LOSE raises θ, bounded. G6 — firing needs a
    great pulse (never a timer)."""
    drv = ii.IntrospectiveDrive()
    t0 = drv.theta
    drv.record_outcome(win=True, reward=0.8)
    t1 = drv.theta
    drv.record_outcome(win=False, reward=-0.3)
    t2 = drv.theta
    assert t1 < t0 < t2                                   # WIN−α, LOSE+β
    assert drv.floor <= drv.theta <= drv.ceil
    # firing is great-pulse gated.
    assert drv.should_fire(drive=2.0, great_pulse_fired=True, metabolic_ok=True)
    assert not drv.should_fire(drive=2.0, great_pulse_fired=False, metabolic_ok=True)
    assert not drv.should_fire(drive=2.0, great_pulse_fired=True, metabolic_ok=False)
    assert not drv.should_fire(drive=0.0, great_pulse_fired=True, metabolic_ok=True)


def test_curiosity_blend_from_neuromod():
    hi = {"modulators": {"DA": {"level": 0.9}, "NE": {"level": 0.8},
                         "GABA": {"level": 0.1}, "5HT": {"level": 0.5},
                         "ACh": {"level": 0.5}, "Endorphin": {"level": 0.5}}}
    lo = {"modulators": {"DA": {"level": 0.1}, "NE": {"level": 0.1},
                         "GABA": {"level": 0.9}, "5HT": {"level": 0.5},
                         "ACh": {"level": 0.5}, "Endorphin": {"level": 0.5}}}
    assert ii.curiosity_from_neuromod(hi) > ii.curiosity_from_neuromod(lo)
    assert ii.curiosity_from_neuromod(None) == 0.0       # cold-start safe


def test_inner_voice_prompts_grounded_and_pure():
    """Phase B — the voice prompts are PURE + GROUNDED in the measured signals
    (neuromod names + body magnitude), and carry no fabricated content."""
    s_raw = np.concatenate([np.array([0.9, 0.1, 0.2, 0.3, 0.1]),  # body
                            np.full(15, 0.2), np.full(45, 0.1),
                            np.array([0.8, 0.5, 0.7, 0.5, 0.5, 0.2])]).astype("float32")
    neuro = {"modulators": {n: {"level": float(v)} for n, v in zip(
        ii.NEUROMOD_ORDER, [0.8, 0.5, 0.7, 0.5, 0.5, 0.2])}}
    p = ii.build_inner_voice_prompts(s_raw, neuro, stance=1, dialogue_turns=2)
    assert "system_prompt" in p and "user_prompt" in p
    assert "drive/seeking 0.80" in p["user_prompt"]      # grounded neuromod level
    assert "mind" in p["user_prompt"]                    # the stance (idx 1)
    assert "first person" in p["system_prompt"].lower()


def test_inv_it2_reward_kernel_has_no_narration_input():
    """INV-IT-2 / G3 (hard guard) — the reward kernel's signature accepts ONLY
    numeric telemetry; there is structurally no way for the LLM voice text to
    influence r_inner."""
    import inspect
    params = set(inspect.signature(ii.inner_reward_kernel).parameters)
    assert params == {"descr_pred", "delta_pred", "s0_norm", "s1_norm",
                      "w_d", "w_delta"}
    assert not any("voice" in p or "narration" in p or "text" in p for p in params)


def test_inner_slot_distinct_from_outer():
    from titan_hcl.synthesis.mastery_level import MASTERY_LEVEL_STATE_SLOT
    assert ii.MASTERY_LEVEL_INNER_STATE_SLOT != MASTERY_LEVEL_STATE_SLOT
    assert ii.MASTERY_LEVEL_INNER_STATE_SPEC.name == "mastery_level_inner_state"
