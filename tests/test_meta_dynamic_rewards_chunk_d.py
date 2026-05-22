"""Chunk D (RFP_meta-reasoning_CGN_FIX.md §4.4) — verify the gentler
α-ramp warm-up tier + time-escape hatch added to DynamicRewardAccumulator.

5-tier schedule:
  0–500     → α=0.10  (NEW warm-up tier)
  500–2000  → α=0.25
  2000–5000 → α=0.50
  5000–10K  → α=0.75
  10K–20K   → α=1.00  (steady)
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from titan_hcl.logic.meta_dynamic_rewards import DynamicRewardAccumulator


# ──────────────────────────────────────────────────────────────────────
# 5-tier count-based schedule
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("count,expected_alpha,expected_phase", [
    (0,      0.10, "warm_up"),
    (250,    0.10, "warm_up"),
    (499,    0.10, "warm_up"),
    (500,    0.25, "phase_0"),
    (1500,   0.25, "phase_0"),
    (1999,   0.25, "phase_0"),
    (2000,   0.50, "phase_1"),
    (4999,   0.50, "phase_1"),
    (5000,   0.75, "phase_2"),
    (9999,   0.75, "phase_2"),
    (10000,  1.00, "steady"),
    (15000,  1.00, "steady"),
    (50000,  1.00, "steady"),
])
def test_alpha_ramp_5_tier_schedule(count, expected_alpha, expected_phase):
    """Verify 5-tier schedule per RFP §4.4 (gentler than original 4-tier)."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    acc._total_outcomes = count
    assert acc.current_alpha() == pytest.approx(expected_alpha)
    assert acc.current_phase() == expected_phase


def test_alpha_ramp_disabled_returns_zero():
    """When alpha_ramp_enabled=False, α is hard-wired to 0.0 regardless
    of outcome count (Session 1 safety interlock)."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=False)
    acc._total_outcomes = 50000
    assert acc.current_alpha() == 0.0
    assert acc.current_phase() == "disabled"


def test_alpha_warmup_tier_is_new_in_chunk_d():
    """Pre-Chunk-D the schedule jumped from 0→0.25 at count=500. Chunk D
    inserts a 0.10 warm-up tier for the first 500 outcomes to prevent
    policy thrashing when the policy is 100% FORMULATE-dominant."""
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    # At count=0, pre-Chunk-D would have returned 0.0 (warm_up=no-α);
    # Chunk D returns 0.10.
    acc._total_outcomes = 0
    assert acc.current_alpha() == 0.10
    acc._total_outcomes = 1
    assert acc.current_alpha() == 0.10
    acc._total_outcomes = 499
    assert acc.current_alpha() == 0.10
    # Tier boundary is exclusive on the low side.
    acc._total_outcomes = 500
    assert acc.current_alpha() == 0.25


# ──────────────────────────────────────────────────────────────────────
# Time-escape hatch
# ──────────────────────────────────────────────────────────────────────


def test_time_escape_no_boost_before_threshold():
    """Time-escape boost only kicks in after `time_escape_seconds_per_step`
    elapses since the last tier promotion."""
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_seconds_per_step=10.0,  # tight for fast test
    )
    acc._total_outcomes = 100  # warm-up tier, α=0.10
    # Immediately after init, no boost.
    assert acc.current_alpha() == pytest.approx(0.10)


def test_time_escape_adds_increment_after_threshold():
    """After 1 escape window elapses, α += increment."""
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_seconds_per_step=10.0,
        time_escape_increment=0.10,
        time_escape_cap=1.0,
    )
    acc._total_outcomes = 100  # warm-up, base α=0.10
    # Backdate the anchor 11s so 1 escape window has elapsed.
    acc._last_tier_promotion_ts = time.time() - 11.0
    alpha = acc.current_alpha()
    # α = 0.10 (warm-up base) + 0.10 (1 escape step) = 0.20
    assert alpha == pytest.approx(0.20)


def test_time_escape_caps_at_one():
    """Accumulated boost is capped at time_escape_cap (default 1.0)."""
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_seconds_per_step=1.0,
        time_escape_increment=0.10,
        time_escape_cap=1.0,
    )
    acc._total_outcomes = 100  # base α=0.10
    # Backdate 1000s — 1000 escape steps × 0.10 = 100.0, capped at 1.0
    acc._last_tier_promotion_ts = time.time() - 1000.0
    assert acc.current_alpha() == pytest.approx(1.0)


def test_time_escape_disabled_returns_count_based_only():
    """When time_escape_enabled=False, α is the count-based schedule only."""
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_enabled=False,
        time_escape_seconds_per_step=1.0,
    )
    acc._total_outcomes = 100
    # Backdate aggressively to force time-escape if enabled.
    acc._last_tier_promotion_ts = time.time() - 100000.0
    # Should still return 0.10 (warm-up tier, no boost added).
    assert acc.current_alpha() == pytest.approx(0.10)


def test_time_escape_resets_on_tier_promotion():
    """When count-based α promotes to a higher tier, the escape timer
    resets and the accumulated boost clears."""
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_seconds_per_step=10.0,
        time_escape_increment=0.10,
    )
    acc._total_outcomes = 100  # warm-up (tier 1)
    acc._last_tier_promotion_ts = time.time() - 11.0
    # Touch current_alpha to accumulate a boost.
    _ = acc.current_alpha()
    assert acc._time_escape_alpha_boost == pytest.approx(0.10)
    # Promote tier — bump outcomes past warm-up threshold.
    acc._total_outcomes = 500  # phase_0 (tier 2)
    alpha = acc.current_alpha()
    # Promotion detected → boost reset; α back to count-based 0.25.
    assert acc._time_escape_alpha_boost == 0.0
    assert alpha == pytest.approx(0.25)


# ──────────────────────────────────────────────────────────────────────
# Tier promotion logging
# ──────────────────────────────────────────────────────────────────────


def test_time_escape_first_tier_promotion_seen(caplog):
    """First tier promotion (e.g., warm_up→phase_0) is logged at INFO."""
    import logging
    acc = DynamicRewardAccumulator(alpha_ramp_enabled=True)
    acc._total_outcomes = 100  # warm-up (tier 1)
    # Trigger tier-index check
    _ = acc.current_alpha()
    assert acc._last_observed_tier_index == 1

    with caplog.at_level(logging.INFO,
                         logger="titan_hcl.logic.meta_dynamic_rewards"):
        acc._total_outcomes = 600  # phase_0 (tier 2)
        _ = acc.current_alpha()

    promotions = [r for r in caplog.records
                  if "tier promoted" in r.getMessage()]
    assert promotions, "tier promotion log not emitted"


# ──────────────────────────────────────────────────────────────────────
# Acceptance criteria from RFP §4.4 expected outcomes
# ──────────────────────────────────────────────────────────────────────


def test_worst_case_reaches_alpha_one_within_nine_weeks():
    """RFP §4.4: 'Worst-case the system reaches α=1.0 within 9 weeks of
    activation regardless of outcome rate.' Verified: at default settings
    (7d per step × 0.10 increment), 10 steps × 0.10 = 1.0 → 70 days ≈ 10
    weeks. Some headroom built in.
    """
    acc = DynamicRewardAccumulator(
        alpha_ramp_enabled=True,
        time_escape_seconds_per_step=604800.0,  # 7 days
        time_escape_increment=0.10,
    )
    # Simulate: 100 outcomes, never crosses warm-up threshold (sparse rate).
    acc._total_outcomes = 100  # base α=0.10
    # Backdate 9 weeks = 9×7×86400 = 5443200s — 9 escape steps × 0.10 = 0.90
    # boost. Plus 0.10 base = 1.0. So 9 weeks gets us to α=1.0.
    acc._last_tier_promotion_ts = time.time() - 9 * 604800.0
    assert acc.current_alpha() == pytest.approx(1.0)
