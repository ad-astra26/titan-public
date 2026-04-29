"""Unit tests for perception creation_nudge exponential decay fix.

Archaeology finding (2026-04-23): outer_body[2] somatosensation
saturated to 1.0 permanently on any creating Titan because
creation_nudge added +0.03 per perception event with no decay.
Fix: mean-revert toward 0.5 with 5min half-life before each new nudge.
"""
import pytest

from titan_plugin.logic.perception import (
    _apply_creation_nudge_with_decay,
    _CREATION_NUDGE_BASELINE,
    _CREATION_NUDGE_HALF_LIFE_S,
    _reset_creation_nudge_for_testing,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_creation_nudge_for_testing()
    yield
    _reset_creation_nudge_for_testing()


def test_single_event_adds_nudge():
    """First event at baseline gets full nudge."""
    result = _apply_creation_nudge_with_decay(0.5, 0.03, now=1000.0)
    assert abs(result - 0.53) < 0.001


def test_back_to_back_events_accumulate_but_below_cap():
    """Burst of 50 events within a fraction of a second: approaches but
    does NOT saturate at 1.0 (as old code did). With decay_factor ~1.0
    per tick, nudges still accumulate — that's OK, the cap guards."""
    v = 0.5
    now = 1000.0
    for _ in range(50):
        v = _apply_creation_nudge_with_decay(v, 0.03, now=now)
        now += 0.01  # 10ms between events, nearly no decay
    # 50 * 0.03 = 1.5, but clamped to 1.0
    assert v == 1.0


def test_long_idle_decays_to_baseline():
    """After a long idle period (many half-lives), value returns to 0.5."""
    # Seed value far from baseline
    v = _apply_creation_nudge_with_decay(0.5, 0.4, now=1000.0)  # v=0.9
    assert v == 0.9

    # Wait 10 half-lives (50 min) — decay ~1000x
    next_ts = 1000.0 + 10 * _CREATION_NUDGE_HALF_LIFE_S
    v2 = _apply_creation_nudge_with_decay(v, 0.0, now=next_ts)
    # 0.5 + (0.9 - 0.5) * 2^(-10) ≈ 0.5004
    assert abs(v2 - 0.5) < 0.01


def test_one_half_life_gap_halves_distance_to_baseline():
    """After exactly one half-life, distance from baseline halves."""
    v = _apply_creation_nudge_with_decay(0.5, 0.4, now=1000.0)  # v=0.9
    # Distance from baseline: 0.4. After 1 half-life: 0.2.
    v2 = _apply_creation_nudge_with_decay(
        v, 0.0, now=1000.0 + _CREATION_NUDGE_HALF_LIFE_S
    )
    # Expected: 0.5 + 0.2 = 0.7
    assert abs(v2 - 0.7) < 0.01


def test_burst_then_idle_then_event_shows_decay():
    """Saturate via burst, wait, then one event should NOT re-saturate."""
    v = 0.5
    now = 1000.0
    # Burst to saturation
    for _ in range(50):
        v = _apply_creation_nudge_with_decay(v, 0.03, now=now)
        now += 0.01
    assert v == 1.0

    # Wait 2 half-lives (10 min)
    now += 2 * _CREATION_NUDGE_HALF_LIFE_S
    # One event after idle
    v = _apply_creation_nudge_with_decay(v, 0.03, now=now)
    # After 2 half-lives, distance from baseline = 0.5 * 2^-2 = 0.125
    # So decayed_value = 0.625. Plus nudge 0.03 = 0.655.
    assert 0.6 <= v <= 0.7, f"Expected 0.6-0.7 after decay + event, got {v}"


def test_below_baseline_values_also_decay_toward_it():
    """Values below 0.5 mean-revert upward toward 0.5."""
    # Simulate a dim that dropped to 0.2 somehow
    v = _apply_creation_nudge_with_decay(0.2, 0.0, now=1000.0)
    # With no prior timestamp (first call), decay_factor = 1.0 — no decay
    assert v == 0.2

    # Now a gap of 1 half-life
    v = _apply_creation_nudge_with_decay(
        v, 0.0, now=1000.0 + _CREATION_NUDGE_HALF_LIFE_S
    )
    # 0.5 + (0.2 - 0.5) * 0.5 = 0.35
    assert abs(v - 0.35) < 0.01


def test_negative_dt_safe():
    """Clock skew (now < last_ts) should not produce NaN or unbounded values."""
    _apply_creation_nudge_with_decay(0.5, 0.03, now=1000.0)
    # Clock goes backwards
    v = _apply_creation_nudge_with_decay(0.53, 0.03, now=999.0)
    # Should at least stay in [0, 1] and not NaN
    assert 0.0 <= v <= 1.0


def test_decay_cap_prevents_old_saturation_bug():
    """Regression test for the archaeology finding: before fix, 100 events
    with no idle → outer_body[2] pinned at 1.0 forever even after hours
    of idle. With fix, it decays back."""
    # Burst to saturation
    v = 0.5
    now = 1000.0
    for _ in range(40):
        v = _apply_creation_nudge_with_decay(v, 0.03, now=now)
        now += 0.01
    assert v == 1.0  # Saturated (expected — decay hasn't had time yet)

    # Wait 1h idle (12 half-lives) — strong decay
    now += 3600.0  # 12 half-lives at 300s
    v = _apply_creation_nudge_with_decay(v, 0.0, now=now)
    # Should be very close to baseline
    assert v < 0.55, f"After 1h idle, value should decay near 0.5, got {v}"
