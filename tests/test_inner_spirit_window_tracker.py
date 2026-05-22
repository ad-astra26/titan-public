"""Unit tests for InnerSpiritWindowTracker (D-SPEC-101, rFP Dims Redesign
Closure Phase 1).

The tracker turns saturating cumulative counters into *breathing* 0..1 signals
via a dual time-decay EMA (fast ~90s ÷ slow ~30min baseline), so the
re-grounded inner_spirit dims carry true variance from self-observation.
"""
from titan_hcl.logic.inner_spirit_sidecar import (
    InnerSpiritWindowTracker,
    _AUTHENTICITY_CLUSTER,
)

_KEYS = (
    "fire_rate_INTUITION", "fire_rate_REFLECTION", "fire_rate_CREATIVITY",
    # D-SPEC-101 Phase-1 completion additions:
    "fire_rate_EMPATHY", "fire_rate_CURIOSITY", "fire_rate_INSPIRATION",
    "self_churn", "growth", "authenticity_change", "topo_change",
    "hormone_velocity", "coherence_depth", "clock_pulse_rate",
)


def _base():
    return [0.5] * 45


def test_first_call_seeds_and_returns_zero():
    t = InnerSpiritWindowTracker()
    out = t.update(0.0, {"INTUITION": 100}, _base(), [0.1] * 10, {"FOCUS": 0.5})
    assert set(out) == set(_KEYS)
    # All breath/rate signals seed at 0.0; coherence_depth is a smoothed LEVEL
    # whose neutral seed is 0.5 (not a breath).
    assert all(v == 0.0 for k, v in out.items() if k != "coherence_depth")
    assert out["coherence_depth"] == 0.5


def test_all_signals_bounded_0_1():
    t = InnerSpiritWindowTracker()
    fires = {"INTUITION": 0, "REFLECTION": 0, "CREATIVITY": 0}
    s = _base()
    for i in range(200):
        fires = {k: v + (i % 5) for k, v in fires.items()}
        s = [min(1.0, x + 0.001 * (i % 3)) for x in s]
        out = t.update(float(i), fires, s, [0.1 + 0.001 * i] * 10,
                       {"FOCUS": 0.5 + 0.001 * (i % 7)})
        for v in out.values():
            assert 0.0 <= v <= 1.0


def test_quiet_reads_low_burst_reads_higher():
    t = InnerSpiritWindowTracker()
    fires = {"INTUITION": 1000}
    s = _base()
    # warm a steady low baseline (tiny constant trickle)
    for i in range(40):
        fires = {"INTUITION": fires["INTUITION"] + 1}
        t.update(float(i), fires, s, [0.1] * 10, {"FOCUS": 0.5})
    quiet = t.update(40.0, {"INTUITION": fires["INTUITION"]}, s, [0.1] * 10,
                     {"FOCUS": 0.5})["fire_rate_INTUITION"]
    # now a burst (large delta in one tick)
    burst = t.update(41.0, {"INTUITION": fires["INTUITION"] + 500}, s,
                     [0.1] * 10, {"FOCUS": 0.5})["fire_rate_INTUITION"]
    assert burst > quiet


def test_fire_counter_reset_is_safe():
    """A cumulative counter that DECREASES (process restart / wrap) must not
    produce a negative rate."""
    t = InnerSpiritWindowTracker()
    t.update(0.0, {"INTUITION": 5000}, _base(), [0.1] * 10, {})
    t.update(1.0, {"INTUITION": 5050}, _base(), [0.1] * 10, {})
    out = t.update(2.0, {"INTUITION": 10}, _base(), [0.1] * 10, {})  # reset
    assert out["fire_rate_INTUITION"] >= 0.0


def test_authenticity_isolated_to_identity_cluster():
    """authenticity_change reflects ONLY movement of the identity cluster,
    not arbitrary dims (e.g. a fire-driven CHIT dim moving)."""
    t = InnerSpiritWindowTracker()
    s = _base()
    t.update(0.0, {}, s, [0.1] * 10, {})
    # move a NON-cluster dim only (idx 20 pattern_recognition)
    s2 = _base()
    s2[20] = 0.95
    for i in range(1, 30):
        out = t.update(float(i), {}, s2, [0.1] * 10, {})
    assert out["authenticity_change"] == 0.0
    # now move a cluster dim (idx 0 self_recognition)
    s3 = _base()
    s3[_AUTHENTICITY_CLUSTER[0]] = 0.95
    moved = False
    for i in range(30, 60):
        out = t.update(float(i), {}, s3, [0.1] * 10, {})
        if out["authenticity_change"] > 0.0:
            moved = True
    assert moved


def test_temporal_continuity_complement_semantics():
    """self_churn rises with 45D movement → the consuming dim computes
    1 − self_churn, so heavy movement ⇒ low continuity. Here we assert the
    tracker's self_churn rises when the 45D moves vs holds steady."""
    t = InnerSpiritWindowTracker()
    s = _base()
    for i in range(40):
        t.update(float(i), {}, s, [0.1] * 10, {})  # steady
    steady = t.update(40.0, {}, s, [0.1] * 10, {})["self_churn"]
    # now churn the whole 45D
    churn_out = 0.0
    for i in range(41, 60):
        moving = [0.5 + 0.3 * ((i + j) % 2) for j in range(45)]
        churn_out = t.update(float(i), {}, moving, [0.1] * 10, {})["self_churn"]
    assert churn_out > steady
