"""Phase 1 base-level activation tests — D-SPEC-123 (SPEC v1.56.0 §25 /
arch §5.2). Pure-math validation of B_i and the Petrov tail.
"""
from __future__ import annotations

import math

import pytest

from titan_hcl.synthesis.activation import (
    DEFAULT_DECAY_D,
    DEFAULT_WINDOW_N,
    ActivationState,
    base_level,
    record_access,
    recompute_all,
)


# ─────────────────────────────────────────────────────────────────────────
# record_access — use-gated reinforcement (INV-Syn-5)
# ─────────────────────────────────────────────────────────────────────────

def test_record_access_increments_count_and_log():
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=100.0)
    assert s.access_count == 1
    assert s.access_log == [100.0]
    assert s.first_access == 100.0
    assert s.last_access == 100.0


def test_record_access_window_caps_log_to_n():
    s = ActivationState(item_id="kuzu:1")
    for i in range(30):
        record_access(s, ts=float(i), window_n=20)
    assert s.access_count == 30           # all-time count keeps growing
    assert len(s.access_log) == 20         # window kept at cap
    assert s.access_log[0] == 10.0         # oldest retained is i=10
    assert s.access_log[-1] == 29.0
    assert s.first_access == 0.0          # earliest EVER access preserved


def test_record_access_first_access_pinned_to_earliest():
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=50.0)
    record_access(s, ts=51.0)
    record_access(s, ts=52.0)
    assert s.first_access == 50.0


# ─────────────────────────────────────────────────────────────────────────
# base_level — cold-start
# ─────────────────────────────────────────────────────────────────────────

def test_base_level_cold_start_returns_neg_inf():
    s = ActivationState(item_id="kuzu:1")
    bi = base_level(s, now=100.0)
    assert bi == float("-inf")


def test_base_level_zero_count_returns_neg_inf_even_with_log_history():
    # Edge case: shouldn't happen in practice (log + count are co-
    # maintained by record_access), but if access_count == 0 the result
    # must be -inf to flag "no activation yet".
    s = ActivationState(item_id="kuzu:1", access_log=[100.0])
    bi = base_level(s, now=101.0)
    assert bi == float("-inf")


# ─────────────────────────────────────────────────────────────────────────
# base_level — canonical ACT-R math (no tail)
# ─────────────────────────────────────────────────────────────────────────

def test_base_level_single_access_known_value():
    # B_i = ln((now - t_j)^(-d)) = -d * ln(now - t_j)
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=0.0)
    bi = base_level(s, now=100.0, d=0.5)
    expected = -0.5 * math.log(100.0)
    assert bi == pytest.approx(expected, rel=1e-9)


def test_base_level_two_accesses_known_value():
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=0.0)
    record_access(s, ts=50.0)
    # B_i = ln(100^-0.5 + 50^-0.5)
    bi = base_level(s, now=100.0, d=0.5)
    expected = math.log(100.0 ** -0.5 + 50.0 ** -0.5)
    assert bi == pytest.approx(expected, rel=1e-9)


def test_base_level_decays_over_time():
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=0.0)
    bi_at_10 = base_level(s, now=10.0)
    bi_at_100 = base_level(s, now=100.0)
    bi_at_1000 = base_level(s, now=1000.0)
    # Strictly decreasing as time passes — older accesses contribute less.
    assert bi_at_10 > bi_at_100 > bi_at_1000


def test_base_level_repeated_access_strengthens():
    s_once = ActivationState(item_id="a")
    s_three = ActivationState(item_id="b")
    record_access(s_once, ts=0.0)
    record_access(s_three, ts=0.0)
    record_access(s_three, ts=1.0)
    record_access(s_three, ts=2.0)
    bi_once = base_level(s_once, now=10.0)
    bi_three = base_level(s_three, now=10.0)
    assert bi_three > bi_once


def test_base_level_same_tick_repeat_uses_epsilon_floor():
    # Two accesses at exactly `now` should not blow up to +inf — the
    # epsilon-age floor caps each contribution at age = 1ms.
    s = ActivationState(item_id="kuzu:1")
    record_access(s, ts=100.0)
    record_access(s, ts=100.0)
    bi = base_level(s, now=100.0, d=0.5)
    assert math.isfinite(bi)
    # Should be approximately ln(2 * (1e-3)^-0.5)
    expected = math.log(2 * (1e-3) ** -0.5)
    assert bi == pytest.approx(expected, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# Petrov 2006 O(1) tail
# ─────────────────────────────────────────────────────────────────────────

def test_petrov_tail_only_kicks_in_when_count_exceeds_log():
    # access_count == len(access_log) → no tail correction needed.
    s = ActivationState(item_id="kuzu:1")
    for i in range(5):
        record_access(s, ts=float(i))
    bi_no_tail = base_level(s, now=100.0, d=0.5, window_n=20)
    expected = math.log(sum((100.0 - float(i)) ** -0.5 for i in range(5)))
    assert bi_no_tail == pytest.approx(expected, rel=1e-9)


def test_petrov_tail_adds_to_activation_when_count_exceeds_window():
    # 30 accesses, window_n=20 → tail of 10 missing entries.
    s = ActivationState(item_id="kuzu:1")
    for i in range(30):
        record_access(s, ts=float(i), window_n=20)
    bi_with_tail = base_level(s, now=100.0, d=0.5, window_n=20)
    # Hand-compute: sum over the 20 retained (i=10..29) + Petrov tail for
    # the missing 10 (i=0..9, approximated as tail_count * age_first^-d /
    # (1-d) where age_first = now - first_access = 100 - 0 = 100).
    retained_sum = sum((100.0 - float(i)) ** -0.5 for i in range(10, 30))
    tail_sum = 10 * (100.0 ** -0.5) / (1.0 - 0.5)
    expected = math.log(retained_sum + tail_sum)
    assert bi_with_tail == pytest.approx(expected, rel=1e-9)


def test_petrov_tail_makes_high_count_items_outrank_low_count():
    # Two items with identical recent-access patterns but different all-
    # time counts: the heavier history should rank higher.
    s_light = ActivationState(item_id="light")
    s_heavy = ActivationState(item_id="heavy")
    for i in range(20):
        record_access(s_light, ts=float(i), window_n=20)
    for i in range(100):
        record_access(s_heavy, ts=float(i), window_n=20)
    # Both have the same access_log (last 20 of 0..N-1); s_heavy has 80
    # extra tail entries.
    # Re-align the heavy item's recent log to match the light item's.
    s_heavy.access_log = list(range(20))
    s_heavy.first_access = 0.0
    s_light.access_log = list(range(20))
    s_light.first_access = 0.0
    bi_light = base_level(s_light, now=100.0, d=0.5, window_n=20)
    bi_heavy = base_level(s_heavy, now=100.0, d=0.5, window_n=20)
    assert bi_heavy > bi_light


# ─────────────────────────────────────────────────────────────────────────
# recompute_all — 60s synthesis_worker job
# ─────────────────────────────────────────────────────────────────────────

def test_recompute_all_updates_base_level_and_timestamp():
    s1 = ActivationState(item_id="a")
    s2 = ActivationState(item_id="b")
    record_access(s1, ts=0.0)
    record_access(s2, ts=10.0)
    n = recompute_all([s1, s2], now=100.0)
    assert n == 2
    assert s1.last_recompute == 100.0
    assert s2.last_recompute == 100.0
    assert math.isfinite(s1.base_level)
    assert math.isfinite(s2.base_level)


def test_recompute_all_skips_unchanged_when_no_new_access():
    s = ActivationState(item_id="a")
    record_access(s, ts=0.0)
    # First recompute writes.
    n1 = recompute_all([s], now=100.0)
    assert n1 == 1
    snap_recompute = s.last_recompute
    snap_base = s.base_level
    # Second recompute at the SAME `now` produces identical base_level →
    # nothing changes, n returned should be 0.
    n2 = recompute_all([s], now=100.0)
    assert n2 == 0
    assert s.last_recompute == snap_recompute
    assert s.base_level == snap_base


def test_recompute_all_handles_cold_start_items():
    # Cold-start items (access_count=0) get base_level=-inf — must not
    # crash, must not falsely report a touch.
    s_cold = ActivationState(item_id="cold")
    s_warm = ActivationState(item_id="warm")
    record_access(s_warm, ts=0.0)
    n = recompute_all([s_cold, s_warm], now=100.0)
    # s_cold went from 0.0 to -inf (a change), so it counts as touched.
    # s_warm has a real B_i, also a change. n == 2.
    assert n == 2
    assert s_cold.base_level == float("-inf")
    assert math.isfinite(s_warm.base_level)
