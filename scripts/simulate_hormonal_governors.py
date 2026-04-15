#!/usr/bin/env python3
"""
Hormonal System Simulation — Self-Emergent Governors

Tests the 4 proposed self-emergent governors across 6 scenarios:
  1. Awake steady-state (normal activity)
  2. Dreaming state (high GABA, suppressed accumulation)
  3. High activity burst (all stimuli maxed)
  4. Low Chi emergency (metabolic stress)
  5. Chi recovery (returning from low to healthy)
  6. Edge cases (GABA=1.0, Chi=0, all hormones zeroed)

Compares CURRENT system (fixed rates) vs PROPOSED system (self-emergent).

Usage: python scripts/simulate_hormonal_governors.py
"""

import math

# ── Exact parameters from production code ────────────────────────

CROSS_TALK = {
    "REFLEX":      {"excitors": {"VIGILANCE": 0.3, "IMPULSE": 0.2}, "inhibitors": {"REFLECTION": 0.2, "CREATIVITY": 0.3}},
    "FOCUS":       {"excitors": {"CURIOSITY": 0.1, "IMPULSE": 0.1}, "inhibitors": {"REFLECTION": 0.1}},
    "INTUITION":   {"excitors": {"FOCUS": 0.2, "REFLECTION": 0.1}, "inhibitors": {"REFLEX": 0.2}},
    "IMPULSE":     {"excitors": {"REFLEX": 0.2, "CURIOSITY": 0.1}, "inhibitors": {"REFLECTION": 0.2}},
    "VIGILANCE":   {"excitors": {"REFLEX": 0.3, "FOCUS": 0.2}, "inhibitors": {"REFLECTION": 0.3, "CREATIVITY": 0.2}},
    "CREATIVITY":  {"excitors": {"INSPIRATION": 0.4, "CURIOSITY": 0.2, "REFLECTION": 0.1}, "inhibitors": {"REFLEX": 0.3, "VIGILANCE": 0.2}},
    "CURIOSITY":   {"excitors": {"INSPIRATION": 0.3, "CREATIVITY": 0.2}, "inhibitors": {"REFLEX": 0.5, "VIGILANCE": 0.1}},
    "EMPATHY":     {"excitors": {"CURIOSITY": 0.2, "REFLECTION": 0.2}, "inhibitors": {"REFLEX": 0.2, "VIGILANCE": 0.1}},
    "REFLECTION":  {"excitors": {"EMPATHY": 0.2, "INSPIRATION": 0.1}, "inhibitors": {"REFLEX": 0.4, "IMPULSE": 0.3}},
    "INSPIRATION": {"excitors": {"CURIOSITY": 0.3, "CREATIVITY": 0.4, "REFLECTION": 0.2}, "inhibitors": {"REFLEX": 0.2, "VIGILANCE": 0.1}},
}

PARAMS = {
    "REFLEX":      {"base": 0.008, "sens": 1.5, "decay": 0.003, "thresh": 0.5, "refrac_str": 0.9, "refrac_dec": 0.015},
    "FOCUS":       {"base": 0.006, "sens": 1.3, "decay": 0.002, "thresh": 0.4, "refrac_str": 0.7, "refrac_dec": 0.012},
    "INTUITION":   {"base": 0.004, "sens": 1.0, "decay": 0.002, "thresh": 0.5, "refrac_str": 0.8, "refrac_dec": 0.010},
    "IMPULSE":     {"base": 0.005, "sens": 1.2, "decay": 0.002, "thresh": 0.5, "refrac_str": 0.8, "refrac_dec": 0.010},
    "VIGILANCE":   {"base": 0.006, "sens": 1.4, "decay": 0.003, "thresh": 0.5, "refrac_str": 0.8, "refrac_dec": 0.012},
    "CREATIVITY":  {"base": 0.003, "sens": 1.0, "decay": 0.001, "thresh": 0.6, "refrac_str": 0.9, "refrac_dec": 0.008},
    "CURIOSITY":   {"base": 0.004, "sens": 1.2, "decay": 0.001, "thresh": 0.5, "refrac_str": 0.7, "refrac_dec": 0.010},
    "EMPATHY":     {"base": 0.003, "sens": 1.0, "decay": 0.001, "thresh": 0.5, "refrac_str": 0.7, "refrac_dec": 0.010},
    "REFLECTION":  {"base": 0.002, "sens": 0.8, "decay": 0.0008, "thresh": 0.6, "refrac_str": 0.9, "refrac_dec": 0.006},
    "INSPIRATION": {"base": 0.002, "sens": 0.8, "decay": 0.0008, "thresh": 0.7, "refrac_str": 0.9, "refrac_dec": 0.005},
}

CIRCADIAN_AWAKE = {
    "REFLEX": 1.0, "FOCUS": 1.0, "INTUITION": 1.0, "IMPULSE": 1.0,
    "VIGILANCE": 1.0, "CREATIVITY": 0.7, "CURIOSITY": 1.2,
    "EMPATHY": 1.0, "REFLECTION": 0.6, "INSPIRATION": 0.8,
}
CIRCADIAN_DREAM = {
    "REFLEX": 0.3, "FOCUS": 0.3, "INTUITION": 0.8, "IMPULSE": 0.2,
    "VIGILANCE": 0.2, "CREATIVITY": 1.3, "CURIOSITY": 0.4,
    "EMPATHY": 0.5, "REFLECTION": 1.5, "INSPIRATION": 1.5,
}

# Expression composite consumption (from expression_composites.py)
COMPOSITES = {
    "SPEAK": {"CREATIVITY": 0.3, "REFLECTION": 0.4, "EMPATHY": 0.3},
    "ART":   {"CREATIVITY": 0.5, "INSPIRATION": 0.3, "IMPULSE": 0.2},
    "MUSIC": {"CREATIVITY": 0.4, "INTUITION": 0.4, "REFLECTION": 0.2},
    "SOCIAL":{"EMPATHY": 0.5, "CURIOSITY": 0.3, "IMPULSE": 0.2},
    "KIN":   {"EMPATHY": 0.3, "CURIOSITY": 0.25, "REFLECTION": 0.2, "INSPIRATION": 0.15, "IMPULSE": 0.1},
}
COMPOSITE_CONSUMPTION = 0.65  # average consumption_rate across composites

NAMES = list(PARAMS.keys())
DT = 3.45  # Schumann body clock
MATURITY = 0.8  # current T1 maturity


class Hormone:
    def __init__(self, name, p):
        self.name = name
        self.level = 0.0
        self.threshold = p["thresh"]
        self.refractory = 0.0
        self.base_rate = p["base"]
        self.sensitivity = p["sens"]
        self.decay_rate = p["decay"]
        self.refrac_str = p["refrac_str"]
        self.refrac_dec = p["refrac_dec"]
        self.fire_count = 0
        self.excitors = CROSS_TALK[name]["excitors"]
        self.inhibitors = CROSS_TALK[name]["inhibitors"]


def run_simulation(mode="current", scenario="awake_steady", duration_min=60, verbose=False):
    """
    Run one simulation.
    mode: "current" (fixed rates) or "proposed" (self-emergent governors)
    scenario: defines the environment conditions over time
    """
    hormones = {n: Hormone(n, PARAMS[n]) for n in NAMES}
    ticks = int(duration_min * 60 / DT)

    # Stats tracking
    stats = {n: {"fires": 0, "max": 0.0, "min_nonzero": 999.0,
                 "sum": 0.0, "saturated_ticks": 0, "zero_ticks": 0}
             for n in NAMES}
    fire_events = []

    for tick in range(ticks):
        t_min = tick * DT / 60.0  # current time in minutes

        # ── Scenario-dependent environment ──
        env = _get_environment(scenario, t_min, duration_min)
        stimulus = env["stimulus"]       # dict: name → [0,1]
        gaba = env["gaba"]               # 0.0-1.0
        chi = env["chi"]                 # 0.0-1.0
        acc_gain = env["acc_gain"]        # neuromod accumulation_rate_gain
        is_dreaming = env["dreaming"]
        circadian = CIRCADIAN_DREAM if is_dreaming else CIRCADIAN_AWAKE

        # Collect current levels for cross-talk
        levels = {n: h.level for n, h in hormones.items()}

        for name in NAMES:
            h = hormones[name]
            p = PARAMS[name]
            stim = stimulus.get(name, 0.3)

            # ── Cross-talk ──
            cross_talk = 1.0
            for exc_name, weight in h.excitors.items():
                if mode == "proposed":
                    # Governor 5 (bonus): normalize by threshold
                    cross_talk += weight * (levels.get(exc_name, 0) / max(0.01, hormones[exc_name].threshold))
                else:
                    cross_talk += weight * levels.get(exc_name, 0)
            for inh_name, weight in h.inhibitors.items():
                if mode == "proposed":
                    cross_talk -= weight * (levels.get(inh_name, 0) / max(0.01, hormones[inh_name].threshold))
                else:
                    cross_talk -= weight * levels.get(inh_name, 0)
            cross_talk = max(0.1, min(3.0, cross_talk))

            # ── Accumulation ──
            secretion = (h.base_rate + stim * h.sensitivity)
            secretion *= cross_talk * circadian.get(name, 1.0) * DT

            if mode == "proposed":
                # Governor 3: neuromod-governed accumulation
                secretion *= acc_gain

            # Refractory suppresses
            effective = secretion * (1.0 - h.refractory * 0.8)
            h.level += max(0.0, effective)

            # ── Decay ──
            if mode == "proposed":
                # Governor 1: GABA-governed decay
                decay_mult = 1.0 + gaba * 3.0
                decay = h.decay_rate * DT * (1.0 + h.level * 0.5) * decay_mult
            else:
                # Current: fixed decay
                decay = h.decay_rate * DT * (1.0 + h.level * 0.5)
            h.level *= max(0.0, 1.0 - decay)
            h.level = max(0.0, h.level)

            # ── Cap ──
            if mode == "proposed":
                # Governor 2: Chi-governed capacity
                chi_factor = max(0.3, chi)  # floor at 30%
                effective_cap = (2.0 + MATURITY * 1.0) * chi_factor  # mature cap = 3.0 × chi
            else:
                # Current: fixed maturity cap
                infant_cap = 2.0
                mature_cap = 5.0
                effective_cap = infant_cap + MATURITY * (mature_cap - infant_cap)
            max_level = h.threshold * effective_cap
            if h.level > max_level:
                h.level = max_level

            # ── Refractory decay ──
            eff_refrac_dec = h.refrac_dec * (1.0 + MATURITY * 0.5)
            h.refractory *= (1.0 - eff_refrac_dec * DT)
            h.refractory = max(0.0, h.refractory)

            # ── Fire check ──
            if h.level >= h.threshold and h.refractory < 0.15:
                intensity = h.level / max(0.01, h.threshold)
                h.fire_count += 1
                stats[name]["fires"] += 1
                h.level *= 0.15  # dramatic drop
                h.refractory = h.refrac_str
                fire_events.append((tick, name, intensity))

        # ── Expression composite consumption (every tick, all 5 fire) ──
        # This simulates the current behavior where all composites fire every epoch
        if tick % 3 == 0:  # composites fire roughly every 3 ticks (~10s)
            for comp_name, weights in COMPOSITES.items():
                for horm_name, weight in weights.items():
                    depletion = weight * COMPOSITE_CONSUMPTION
                    h = hormones[horm_name]
                    if mode == "proposed":
                        # Governor 4: proportional consumption
                        cap = h.threshold * max(0.3, chi) * (2.0 + MATURITY * 1.0)
                        h.level = max(0.0, h.level * (1.0 - min(1.0, depletion / max(0.01, cap))))
                    else:
                        # Current: absolute consumption
                        h.level = max(0.0, h.level - depletion)

        # ── Track stats ──
        for name in NAMES:
            lv = hormones[name].level
            s = stats[name]
            s["sum"] += lv
            if lv > s["max"]:
                s["max"] = lv
            if lv > 0.001 and lv < s["min_nonzero"]:
                s["min_nonzero"] = lv
            if lv >= hormones[name].threshold * 4.0:
                s["saturated_ticks"] += 1
            if lv < 0.001:
                s["zero_ticks"] += 1

    # ── Compile results ──
    results = {}
    for name in NAMES:
        s = stats[name]
        results[name] = {
            "fires": s["fires"],
            "avg": round(s["sum"] / ticks, 4),
            "max": round(s["max"], 4),
            "min": round(s["min_nonzero"], 4) if s["min_nonzero"] < 999 else 0.0,
            "saturated_pct": round(100 * s["saturated_ticks"] / ticks, 1),
            "zero_pct": round(100 * s["zero_ticks"] / ticks, 1),
            "final": round(hormones[name].level, 4),
        }
    return results


def _get_environment(scenario, t_min, duration):
    """Return environment state for a given scenario and time."""

    # Defaults
    env = {
        "stimulus": {n: 0.3 for n in NAMES},  # moderate baseline
        "gaba": 0.35,      # healthy
        "chi": 0.60,       # healthy
        "acc_gain": 1.008,  # from current neuromod state
        "dreaming": False,
    }

    if scenario == "awake_steady":
        # Moderate stimuli, healthy state throughout
        env["stimulus"]["CURIOSITY"] = 0.5
        env["stimulus"]["CREATIVITY"] = 0.4
        env["stimulus"]["EMPATHY"] = 0.4

    elif scenario == "dreaming":
        # Dreaming: high GABA, suppressed stimuli
        env["gaba"] = 0.7
        env["dreaming"] = True
        env["stimulus"] = {n: 0.1 for n in NAMES}
        env["stimulus"]["REFLECTION"] = 0.5
        env["stimulus"]["INTUITION"] = 0.4

    elif scenario == "high_activity":
        # Burst of high stimuli (all programs active)
        env["stimulus"] = {n: 0.8 for n in NAMES}
        env["gaba"] = 0.2  # low inhibition during activity
        env["acc_gain"] = 1.15  # high DA → high accumulation

    elif scenario == "low_chi":
        # Metabolic stress: Chi drops to 0.15
        env["chi"] = 0.15
        env["gaba"] = 0.5
        env["stimulus"] = {n: 0.2 for n in NAMES}

    elif scenario == "chi_recovery":
        # Chi goes from 0.1 → 0.7 over the duration
        progress = t_min / duration
        env["chi"] = 0.1 + 0.6 * progress
        env["gaba"] = 0.6 - 0.3 * progress  # GABA decreases as Chi recovers

    elif scenario == "edge_gaba_max":
        # Edge case: GABA at maximum (1.0)
        env["gaba"] = 1.0
        env["stimulus"] = {n: 0.5 for n in NAMES}

    elif scenario == "edge_chi_zero":
        # Edge case: Chi at absolute zero
        env["chi"] = 0.0
        env["gaba"] = 0.8
        env["stimulus"] = {n: 0.1 for n in NAMES}

    elif scenario == "awake_to_dream":
        # Mixed: awake for first half, dreaming for second half
        if t_min < duration / 2:
            env["stimulus"]["CURIOSITY"] = 0.5
            env["stimulus"]["CREATIVITY"] = 0.4
        else:
            env["gaba"] = 0.7
            env["dreaming"] = True
            env["stimulus"] = {n: 0.1 for n in NAMES}
            env["stimulus"]["REFLECTION"] = 0.5

    return env


def print_comparison(scenario, duration_min=60):
    """Run both modes and print side-by-side comparison."""
    current = run_simulation("current", scenario, duration_min)
    proposed = run_simulation("proposed", scenario, duration_min)

    print(f"\n{'='*90}")
    print(f"  SCENARIO: {scenario} ({duration_min} min)")
    print(f"{'='*90}")
    print(f"  {'Program':12s} │ {'CURRENT':>38s} │ {'PROPOSED (Self-Emergent)':>38s}")
    print(f"  {'':12s} │ {'fires  avg    max   sat%  zero%':>38s} │ {'fires  avg    max   sat%  zero%':>38s}")
    print(f"  {'─'*12}─┼─{'─'*38}─┼─{'─'*38}")

    for name in NAMES:
        c = current[name]
        p = proposed[name]
        print(f"  {name:12s} │ {c['fires']:5d} {c['avg']:6.3f} {c['max']:6.3f} {c['saturated_pct']:5.1f} {c['zero_pct']:5.1f} │ "
              f"{p['fires']:5d} {p['avg']:6.3f} {p['max']:6.3f} {p['saturated_pct']:5.1f} {p['zero_pct']:5.1f}")

    # Summary
    c_fires = sum(v["fires"] for v in current.values())
    p_fires = sum(v["fires"] for v in proposed.values())
    c_sat = sum(v["saturated_pct"] for v in current.values()) / len(NAMES)
    p_sat = sum(v["saturated_pct"] for v in proposed.values()) / len(NAMES)
    c_zero = sum(v["zero_pct"] for v in current.values()) / len(NAMES)
    p_zero = sum(v["zero_pct"] for v in proposed.values()) / len(NAMES)
    c_impulse = current["IMPULSE"]["fires"]
    p_impulse = proposed["IMPULSE"]["fires"]

    print(f"  {'─'*12}─┼─{'─'*38}─┼─{'─'*38}")
    print(f"  {'TOTAL':12s} │ {c_fires:5d} fires, {c_sat:5.1f}% sat, {c_zero:5.1f}% zero │ "
          f"{p_fires:5d} fires, {p_sat:5.1f}% sat, {p_zero:5.1f}% zero")
    print(f"  {'IMPULSE':12s} │ {c_impulse:5d} fires ({('DEAD' if c_impulse < 3 else 'LOW' if c_impulse < 20 else 'OK'):>4s}){'':>22s} │ "
          f"{p_impulse:5d} fires ({('DEAD' if p_impulse < 3 else 'LOW' if p_impulse < 20 else 'OK'):>4s})")


def print_edge_case_safety():
    """Test boundary conditions for proposed system."""
    print(f"\n{'='*90}")
    print(f"  EDGE CASE SAFETY TESTS")
    print(f"{'='*90}")

    tests = [
        ("edge_gaba_max", "GABA=1.0 (maximum inhibition)"),
        ("edge_chi_zero", "Chi=0.0 (total metabolic failure)"),
    ]

    for scenario, desc in tests:
        results = run_simulation("proposed", scenario, 30)
        max_any = max(v["max"] for v in results.values())
        min_fires = min(v["fires"] for v in results.values())
        total_fires = sum(v["fires"] for v in results.values())
        avg_sat = sum(v["saturated_pct"] for v in results.values()) / len(NAMES)
        impulse_fires = results["IMPULSE"]["fires"]

        passed = max_any < 10.0 and avg_sat < 50.0  # basic sanity
        status = "PASS" if passed else "FAIL"

        print(f"\n  [{status}] {desc}")
        print(f"    Max hormone level: {max_any:.3f}")
        print(f"    Total fires: {total_fires}")
        print(f"    IMPULSE fires: {impulse_fires}")
        print(f"    Avg saturation: {avg_sat:.1f}%")
        for name in NAMES:
            r = results[name]
            flag = " *** SATURATED" if r["saturated_pct"] > 50 else ""
            flag2 = " *** DEAD" if r["fires"] == 0 and r["max"] < 0.01 else ""
            print(f"    {name:12s}  fires={r['fires']:4d}  avg={r['avg']:.3f}  max={r['max']:.3f}  final={r['final']:.3f}{flag}{flag2}")


if __name__ == "__main__":
    scenarios = [
        ("awake_steady", 60),
        ("dreaming", 30),
        ("high_activity", 30),
        ("low_chi", 30),
        ("chi_recovery", 60),
        ("awake_to_dream", 60),
    ]

    for scenario, duration in scenarios:
        print_comparison(scenario, duration)

    print_edge_case_safety()

    print(f"\n{'='*90}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*90}")
