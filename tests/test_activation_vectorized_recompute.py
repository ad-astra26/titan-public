"""Verify→Confirm: the vectorized B_i (`base_level_batch`) is numerically
identical to the per-item `base_level`, across edge cases — the proof required
before swapping the GIL-starving 50s scalar recompute loop for the numpy path
(2026-06-10 synthesis heartbeat_timeout crash-loop fix).

If these pass, the production recompute can use base_level_batch with confidence
that no ranking/decision changes — only the GIL behaviour (released) + wall-time.
"""
import math

import numpy as np
import pytest

from titan_hcl.synthesis.activation import (
    ActivationState, base_level, base_level_batch, record_access,
    DEFAULT_DECAY_D, DEFAULT_WINDOW_N, EPSILON_AGE_S,
)


def _assert_matches(states, now, d=DEFAULT_DECAY_D, window_n=DEFAULT_WINDOW_N):
    """Per-item base_level vs base_level_batch — exact -inf, tight rtol finite."""
    scalar = [base_level(s, now, d=d, window_n=window_n) for s in states]
    vec = base_level_batch(states, now, d=d, window_n=window_n)
    assert vec.shape == (len(states),)
    for i, (sv, vv) in enumerate(zip(scalar, vec)):
        if sv == float("-inf"):
            assert vv == float("-inf"), f"state {i}: scalar=-inf but vec={vv}"
        else:
            assert math.isfinite(vv), f"state {i}: scalar={sv} finite but vec={vv}"
            # libm pow may differ by a ULP between numpy-vectorized and scalar **;
            # rtol=1e-9 is ~1e6× looser than that yet catches any real formula bug.
            assert vv == pytest.approx(sv, rel=1e-9, abs=1e-12), \
                f"state {i}: scalar={sv!r} vec={vv!r}"


def _mk(access_log, access_count=None, first_access=None):
    """Build a state directly (bypass record_access) for precise edge control."""
    log = list(access_log)
    return ActivationState(
        item_id="x",
        access_log=log,
        access_count=access_count if access_count is not None else len(log),
        first_access=first_access if first_access is not None else (log[0] if log else 0.0),
        last_access=log[-1] if log else 0.0,
    )


NOW = 1_000_000.0


def test_empty_and_cold_states_return_neg_inf():
    states = [
        _mk([], access_count=0),                       # never accessed
        _mk([], access_count=3, first_access=999_000), # count>0 but empty log → -inf
    ]
    _assert_matches(states, NOW)
    vec = base_level_batch(states, NOW)
    assert vec[0] == float("-inf") and vec[1] == float("-inf")


def test_single_access():
    _assert_matches([_mk([NOW - 10.0])], NOW)


def test_no_tail_count_equals_retained():
    # access_count == len(access_log) → no Petrov tail term.
    log = [NOW - t for t in (100.0, 50.0, 10.0, 1.0)]
    _assert_matches([_mk(log, access_count=4)], NOW)


def test_with_petrov_tail():
    # access_count >> retained → tail term engaged.
    log = [NOW - t for t in (40.0, 30.0, 20.0, 10.0, 5.0)]
    _assert_matches([_mk(log, access_count=137, first_access=NOW - 9_000.0)], NOW)


def test_full_window_with_tail():
    log = sorted(NOW - float(t) for t in range(1, DEFAULT_WINDOW_N + 1))
    _assert_matches([_mk(log, access_count=500, first_access=NOW - 50_000.0)], NOW)


def test_same_tick_access_hits_epsilon_floor():
    # now - t == 0 → clamped to EPSILON_AGE_S in BOTH paths.
    _assert_matches([_mk([NOW, NOW, NOW - 1.0], access_count=3)], NOW)


def test_first_access_zero_is_legitimate_timestamp():
    # first_access==0.0 is a real ts (not a sentinel) — tail uses now-0.
    _assert_matches([_mk([NOW - 5.0], access_count=50, first_access=0.0)], NOW)


def test_empty_input_returns_empty_array():
    out = base_level_batch([], NOW)
    assert isinstance(out, np.ndarray) and out.shape == (0,)


def test_non_default_d_and_window():
    log = [NOW - t for t in (8.0, 4.0, 2.0, 1.0)]
    _assert_matches([_mk(log, access_count=20, first_access=NOW - 800.0)],
                    NOW, d=0.3, window_n=10)


def test_large_random_population_matches():
    rng = np.random.default_rng(42)
    states = []
    for _ in range(2000):
        r = int(rng.integers(0, DEFAULT_WINDOW_N + 3))   # 0..22 incl over-window
        if r == 0:
            states.append(_mk([], access_count=int(rng.integers(0, 2))))
            continue
        ages = np.sort(rng.uniform(0.0, 100_000.0, size=r))[::-1]   # oldest..newest gap
        log = sorted(NOW - float(a) for a in ages)
        cnt = r + int(rng.integers(0, 400))               # maybe a tail
        first = NOW - float(rng.uniform(0.0, 1_000_000.0))
        states.append(_mk(log, access_count=cnt, first_access=first))
    _assert_matches(states, NOW)


def test_matches_states_built_via_record_access():
    # End-to-end with the real record_access path (caps log at window_n).
    states = []
    for k in range(1, 30):
        s = ActivationState(item_id=f"i{k}")
        for j in range(k):
            record_access(s, NOW - (k - j) * 3.0)
        states.append(s)
    _assert_matches(states, NOW)
