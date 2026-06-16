"""RFP_supervision_lifecycle §7.B/§7.D — mem/CPU-aware boot-wave pacing.

A calm box boots EXACTLY as fast as today (base cap + stagger, untouched — so dev
boots never drag); a contended box (low MemAvailable / high load — the other Titan
running on a shared box) shrinks in-flight + stretches the stagger so the ~40-
module wave can't over-subscribe → no shm_pid_dead death-storm (T3 2026-06-16,
32× churn). HARD-BOUNDED: cap ≥ floor, stagger ≤ ceiling — never a runaway boot.

Run isolated: python -m pytest tests/test_adaptive_boot_pacing.py -v -p no:anchorpy
"""
from unittest.mock import patch

from titan_hcl.orchestrator.core import (
    Orchestrator, BOOT_PACING_CAP_FLOOR, BOOT_PACING_STAGGER_MAX_S,
    BOOT_PACING_MEM_FLOOR_MB)
from titan_hcl.orchestrator.boot_throttle import BoxPressure


def _pace(mem_mb, load1, *, base_cap=8, base_stagger=1.5, ncpu=4):
    box = BoxPressure(mem_available_mb=mem_mb, swap_used_mb=0.0,
                      swap_total_mb=0.0, load1=load1, ncpu=ncpu)
    with patch("titan_hcl.orchestrator.boot_throttle.read_box_pressure",
               return_value=box), \
         patch("titan_hcl.orchestrator.core.os.cpu_count", return_value=ncpu):
        return Orchestrator._adaptive_boot_pacing(object(), base_cap, base_stagger)


# ── 1. CALM box → untouched (the "no dev-boot slowdown" guarantee) ──────────────

def test_calm_box_uses_base_values_exactly():
    cap, stagger = _pace(mem_mb=3000.0, load1=2.0)   # mem ample, load low
    assert cap == 8 and stagger == 1.5               # identical to today


def test_calm_box_just_under_thresholds():
    # MemAvailable just above the floor, load just under cores*1.5 → still calm.
    cap, stagger = _pace(mem_mb=BOOT_PACING_MEM_FLOOR_MB + 1, load1=5.9, ncpu=4)
    assert (cap, stagger) == (8, 1.5)


# ── 2. CONTENDED box → throttled, BOUNDED ──────────────────────────────────────

def test_mem_tight_only_halves_cap():
    cap, stagger = _pace(mem_mb=500.0, load1=2.0)     # mem < 768 floor
    assert cap == 4                                   # base//2
    assert 1.5 < stagger <= BOOT_PACING_STAGGER_MAX_S


def test_load_high_only_halves_cap():
    cap, stagger = _pace(mem_mb=3000.0, load1=10.0, ncpu=4)  # load1 > 4*1.5=6
    assert cap == 4
    assert stagger > 1.5


def test_severe_both_drops_to_floor():
    cap, stagger = _pace(mem_mb=400.0, load1=12.0, ncpu=4)   # tight AND high
    assert cap == BOOT_PACING_CAP_FLOOR               # == 2
    assert stagger == min(1.5 * 2.5, BOOT_PACING_STAGGER_MAX_S)


# ── 3. HARD BOUNDS — never a runaway boot ──────────────────────────────────────

def test_stagger_never_exceeds_ceiling():
    # A large base stagger under severe pressure is capped at the ceiling.
    cap, stagger = _pace(mem_mb=300.0, load1=20.0, base_stagger=3.0, ncpu=4)
    assert stagger == BOOT_PACING_STAGGER_MAX_S        # 3.0*2.5=7.5 → capped 4.0


def test_cap_never_below_floor():
    # A small base cap under pressure still keeps forward progress.
    cap, _ = _pace(mem_mb=500.0, load1=2.0, base_cap=3)
    assert cap >= BOOT_PACING_CAP_FLOOR                # max(2, 3//2=1) = 2


# ── 4. FAIL-OPEN — pacing never breaks a boot ──────────────────────────────────

def test_read_error_falls_back_to_base():
    with patch("titan_hcl.orchestrator.boot_throttle.read_box_pressure",
               side_effect=RuntimeError("proc read failed")):
        cap, stagger = Orchestrator._adaptive_boot_pacing(object(), 8, 1.5)
    assert (cap, stagger) == (8, 1.5)                  # behaves exactly as today
