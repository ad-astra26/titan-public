"""Unit tests for the D-SPEC-101 dims-redesign breath trackers.

Covers the shared expression/willing/change/variance trackers plus the
extended InnerSpiritWindowTracker (Phase-1 completion + Phase-2 outer).
Run: python -m pytest tests/test_dims_redesign_trackers.py -v -p no:anchorpy
"""
from titan_hcl.logic.expression_window_tracker import (
    ExpressionWindowTracker, ChangeBreathTracker, EmaVarianceTracker, MODALITIES)
from titan_hcl.logic.inner_spirit_sidecar import InnerSpiritWindowTracker


# ── ExpressionWindowTracker ─────────────────────────────────────────────

def test_expression_window_first_call_returns_zero_baseline():
    t = ExpressionWindowTracker()
    out = t.update(0.0, {"image": 5, "sound": 2, "speak": 1, "word": 10})
    assert all(out[f"{m}_rate"] == 0.0 for m in MODALITIES)
    assert out["variety"] == 0.0 and out["volume"] == 0.0


def test_expression_window_breath_rises_on_activity():
    t = ExpressionWindowTracker()
    t.update(0.0, {"image": 0, "sound": 0, "speak": 0, "word": 0})
    # second call seeds the EMA baseline (breath still 0 on first observe)
    t.update(1.0, {"image": 5, "sound": 0, "speak": 0, "word": 0})
    # third call: sustained image output → breath > 0
    out = t.update(2.0, {"image": 10, "sound": 0, "speak": 0, "word": 0})
    assert out["image_rate"] > 0.0
    assert out["sound_rate"] == 0.0
    assert 0.0 <= out["variety"] <= 1.0


def test_expression_window_reset_safe_negative_delta():
    t = ExpressionWindowTracker()
    t.update(0.0, {"image": 100, "sound": 0, "speak": 0, "word": 0})
    # counter reset (process restart) → negative delta clamps to 0, no crash
    out = t.update(1.0, {"image": 3, "sound": 0, "speak": 0, "word": 0})
    assert out["image_rate"] >= 0.0


def test_expression_window_configurable_modalities():
    t = ExpressionWindowTracker(modalities=("a", "b", "c"))
    out = t.update(0.0, {"a": 1, "b": 2, "c": 3})
    assert "a_rate" in out and "b_rate" in out and "c_rate" in out
    assert "image_rate" not in out


def test_expression_window_sovereignty_ratio():
    t = ExpressionWindowTracker()
    t.update(0.0, {"image": 0, "self_authored": 0, "total": 0})
    # 6 self-authored of 10 total over the window → ratio 0.6
    out = t.update(1.0, {"image": 0, "self_authored": 6, "total": 10})
    assert abs(out["sovereignty"] - 0.6) < 1e-9


# ── ChangeBreathTracker ─────────────────────────────────────────────────

def test_change_breath_steady_level_low_moving_level_high():
    t = ChangeBreathTracker()
    t.update(0.0, {"entropy": 0.5})
    t.update(1.0, {"entropy": 0.5})  # seeds baseline
    steady = t.update(2.0, {"entropy": 0.5})["entropy_change"]
    moving = t.update(3.0, {"entropy": 0.9})["entropy_change"]
    assert moving > steady


# ── EmaVarianceTracker (π HRV) ──────────────────────────────────────────

def test_ema_variance_steady_signal_low_cv2():
    t = EmaVarianceTracker(half_life_s=10.0)
    r = 0.0
    for i in range(50):
        r = t.update(float(i), 3.0)  # perfectly steady
    assert r < 0.05  # near-zero variability


def test_ema_variance_volatile_signal_higher_cv2():
    steady = EmaVarianceTracker(half_life_s=10.0)
    volatile = EmaVarianceTracker(half_life_s=10.0)
    rs = rv = 0.0
    for i in range(80):
        rs = steady.update(float(i), 3.0)
        rv = volatile.update(float(i), 3.0 + (2.0 if i % 2 else -2.0))
    assert rv > rs


# ── InnerSpiritWindowTracker (extended) ─────────────────────────────────

def _spin(tracker, fires, n=6, clock_pulses_start=0.0):
    """Drive the tracker n ticks with steady cumulative fires growth."""
    out = {}
    for i in range(n):
        cum = {h: fires.get(h, 0.0) * (i + 1) for h in fires}
        out = tracker.update(
            now=float(i), fires=cum, spirit_45d=[0.5] * 45,
            topo10=[0.5] * 10, levels={"CURIOSITY": 0.5},
            clock_pulses=clock_pulses_start + (i + 1) * 5.0)
    return out


def test_inner_tracker_emits_new_fire_rate_keys():
    t = InnerSpiritWindowTracker()
    out = _spin(t, {"EMPATHY": 2.0, "CURIOSITY": 3.0, "INSPIRATION": 1.0})
    for k in ("fire_rate_EMPATHY", "fire_rate_CURIOSITY",
              "fire_rate_INSPIRATION", "coherence_depth", "clock_pulse_rate"):
        assert k in out, f"missing {k}"
    # sustained firing → positive breath
    assert out["fire_rate_EMPATHY"] > 0.0
    assert out["clock_pulse_rate"] > 0.0


def test_inner_tracker_coherence_depth_high_when_dims_uniform():
    t = InnerSpiritWindowTracker()
    # all 45 dims equal → variance 0 → coherence_depth → 1.0 (smoothed)
    for i in range(8):
        out = t.update(now=float(i), fires={}, spirit_45d=[0.5] * 45,
                       topo10=None, levels={})
    assert out["coherence_depth"] > 0.9


def test_inner_tracker_coherence_depth_low_when_dims_scattered():
    t = InnerSpiritWindowTracker()
    scattered = [0.0 if i % 2 else 1.0 for i in range(45)]
    for i in range(8):
        out = t.update(now=float(i), fires={}, spirit_45d=scattered,
                       topo10=None, levels={})
    # var ≈ 0.25 → coherence_depth ≈ 0
    assert out["coherence_depth"] < 0.2
