#!/usr/bin/env python3
"""
Neuromodulator Tuning Simulation — 5-HT/Endorphin Equilibrium Investigation

Tests Option B (input scaling DNA) against current live Titan state.
Uses REAL current values from T1 (2026-03-24 05:30 UTC).

Runs multiple scenarios:
  A) Current DNA (no change) — shows the problem
  B) Input scaling DNA — proposed fix
  C) Both with dreaming cycles
  D) Sensitivity analysis: what if state changes?

Usage:
    source test_env/bin/activate
    python scripts/neuromod_tuning_sim.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan_plugin.logic.neuromodulator import (
    CLEARANCE_RATES, COUPLING_MATRIX, METABOLIC_COSTS,
    Neuromodulator, NEUROMOD_PRESSURE_RATE,
    compute_emergent_inputs, apply_movement_excess_clearance,
)

# ═══════════════════════════════════════════════════════════════════
# CURRENT LIVE STATE FROM T1 (2026-03-24 ~05:18 UTC)
# ═══════════════════════════════════════════════════════════════════

# From /v4/chi: total=0.613, body=0.612
# From brain log: epoch 18748, drift=0.0685, density=0.300, curvature=2.237
# Sphere clocks: inner_body streak=12161, inner_spirit streak=105323
# Chi circulation=0.82 (from log: chi_circ=0.82)

LIVE_SPHERE_BALANCE = {
    "inner_body": 1.0,   # streak 12161 / 100 → capped 1.0
    "inner_mind": 1.0,   # very long streaks
    "outer_body": 1.0,   # streak also high now
    "outer_mind": 1.0,   # streak also high now
}

LIVE_TRINITY = {
    "inner": 0.65,  # from iB=0.902 iM=0.467 iS=0.303 → moderate coherence
    "outer": 0.70,  # from oB=0.792 oM=0.586 oS=0.467 → slightly higher
}

LIVE_CHI = {"total": 0.613, "body": 0.9, "circulation": 0.82, "drain": 0.03}

LIVE_DYNAMICS = {
    "drift_magnitude": 0.0685,
    "drift_delta": 0.14,
    "density": 0.300,
    "epoch_gap_ratio": 0.08,  # 10.6s gap / 123.4 MAX
}

LIVE_PI = {
    "regularity": 0.6,  # pi clusters active
    "cluster_streak": 0.0,
    "developmental_age": 50,  # mature now
    "curvature_delta": 0.3,
}

LIVE_PREDICTION = {"surprise": 0.02, "action_outcome": 0.5, "success_rate": 0.5}
LIVE_NS = {"transition_delta": 0.5, "filter_down_writes": 0.3}
LIVE_EXPRESSION = {"fire_rate": 0.31, "alignment": 0.5}
LIVE_RESONANCE = {"resonant_fraction": 0.33}

# Current neuromod state from T1
CURRENT_LEVELS = {"DA": 0.50, "5HT": 0.92, "NE": 0.83, "ACh": 0.66, "Endorphin": 0.91, "GABA": 0.09}
CURRENT_SENS = {"DA": 1.07, "5HT": 0.61, "NE": 0.78, "ACh": 0.94, "Endorphin": 0.64, "GABA": 1.35}
CURRENT_SP = {"DA": 0.56, "5HT": 0.70, "NE": 0.70, "ACh": 0.67, "Endorphin": 0.70, "GABA": 0.30}

# Load current DNA from titan_params.toml
def load_dna():
    """Load DNA weights from titan_params.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open("titan_plugin/titan_params.toml", "rb") as f:
        config = tomllib.load(f)
    return config.get("neuromodulator_dna", {})


def run_sim(name, dna, ticks=2000, initial_levels=None, initial_sens=None,
            initial_sp=None, dreaming_ranges=None, topology_velocity=0.3,
            state_overrides=None):
    """Run simulation with given DNA and return final state."""

    # Use live state, with optional overrides
    sphere = dict(LIVE_SPHERE_BALANCE)
    trinity = dict(LIVE_TRINITY)
    chi = dict(LIVE_CHI)
    dynamics = dict(LIVE_DYNAMICS)
    pi = dict(LIVE_PI)
    pred = dict(LIVE_PREDICTION)
    ns = dict(LIVE_NS)
    expr = dict(LIVE_EXPRESSION)
    res = dict(LIVE_RESONANCE)
    if state_overrides:
        for key, val in state_overrides.items():
            locals()[key].update(val)

    # Initialize modulators
    mods = {}
    for mod_name, clearance in CLEARANCE_RATES.items():
        mods[mod_name] = Neuromodulator(name=mod_name, clearance_rate=clearance)
        if initial_levels and mod_name in initial_levels:
            mods[mod_name].level = initial_levels[mod_name]
            mods[mod_name].tonic_level = initial_levels[mod_name]
        if initial_sens and mod_name in initial_sens:
            mods[mod_name].sensitivity = initial_sens[mod_name]
        if initial_sp and mod_name in initial_sp:
            mods[mod_name].setpoint = initial_sp[mod_name]

    trajectories = {n: [] for n in mods}

    for tick in range(ticks):
        is_dreaming = False
        if dreaming_ranges:
            for start, end in dreaming_ranges:
                if start <= tick < end:
                    is_dreaming = True
                    break

        # Compute inputs (uses the production compute_emergent_inputs)
        inputs = compute_emergent_inputs(
            sphere_balance=sphere, trinity_coherence=trinity,
            chi_state=chi, consciousness_dynamics=dynamics,
            pi_state=pi, prediction_state=pred,
            ns_state=ns, expression_state=expr,
            resonance_state=res, is_dreaming=is_dreaming, dna=dna,
        )

        # Apply input scaling if DNA has it
        for mod_name in inputs:
            scale_key = f"{mod_name.lower().replace('-','')}_input_scale"
            # Try specific naming patterns
            scale_keys = [
                f"{mod_name.lower()}_input_scale",
                f"{'sht' if mod_name == '5HT' else mod_name.lower()}_input_scale",
                f"{'endorphin' if mod_name == 'Endorphin' else mod_name.lower()}_input_scale",
            ]
            for sk in scale_keys:
                if sk in dna:
                    inputs[mod_name] *= dna[sk]
                    break

        # Cross-coupling
        cross = {}
        for target in mods:
            c = 0.0
            for source, sm in mods.items():
                if source != target:
                    c += COUPLING_MATRIX.get(source, {}).get(target, 0.0) * sm.level
            cross[target] = c

        # Update + movement clearance
        for mod_name, mod in mods.items():
            mod.update(inputs[mod_name], cross[mod_name], dt=1.0, chi_health=1.0)
            # Movement excess clearance
            mov_key = f"movement_{mod_name}"
            mov_rate = dna.get(mov_key, 0.0)
            if mov_rate > 0:
                apply_movement_excess_clearance(mod, topology_velocity, mov_rate)
            trajectories[mod_name].append(mod.level)

    return mods, trajectories


def print_results(label, mods, trajectories, show_trajectory=True):
    """Print simulation results."""
    print(f"\n{'─' * 75}")
    print(f"  {label}")
    print(f"{'─' * 75}")
    print(f"  {'Mod':10s} {'Level':>8s} {'Setpoint':>9s} {'Delta':>8s} {'Sens':>7s} {'Peak':>7s} {'Trough':>7s} {'Status'}")
    print(f"  {'':10s} {'':>8s} {'':>9s} {'':>8s} {'':>7s} {'':>7s} {'':>7s}")

    for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        m = mods[name]
        delta = m.level - m.setpoint
        # Status assessment
        if m.level > 0.90:
            status = "⚠ HIGH"
        elif m.level < 0.10:
            status = "⚠ LOW"
        elif abs(delta) < 0.10:
            status = "✓ GOOD"
        elif abs(delta) < 0.20:
            status = "~ OK"
        else:
            status = "⚠ FAR"
        print(f"  {name:10s} {m.level:8.4f} {m.setpoint:9.4f} {delta:+8.4f} {m.sensitivity:7.4f} {m._peak_level:7.4f} {m._trough_level:7.4f} {status}")

    if show_trajectory:
        print(f"\n  Trajectory (tick → levels):")
        print(f"  {'Tick':>6s}  {'DA':>7s} {'5HT':>7s} {'NE':>7s} {'ACh':>7s} {'Endo':>7s} {'GABA':>7s}")
        for t in [0, 50, 100, 250, 500, 1000, 1500, 1999]:
            if t < len(trajectories["DA"]):
                vals = "  ".join(f"{trajectories[n][t]:7.4f}" for n in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"])
                print(f"  {t:6d}  {vals}")

    # Success criteria
    print(f"\n  Criteria:")
    checks = [
        ("5-HT < 0.85 (no saturation)", mods["5HT"].level < 0.85),
        ("5-HT > 0.50 (not crashed)", mods["5HT"].level > 0.50),
        ("Endorphin < 0.85", mods["Endorphin"].level < 0.85),
        ("Endorphin > 0.40", mods["Endorphin"].level > 0.40),
        ("NE > 0.30 (alertness)", mods["NE"].level > 0.30),
        ("GABA < 0.50 (not inhibited)", mods["GABA"].level < 0.50),
        ("DA in 0.30-0.70", 0.30 <= mods["DA"].level <= 0.70),
        ("No mod > 0.95 (ceiling)", all(m.level < 0.95 for m in mods.values())),
        ("No mod < 0.05 (floor)", all(m.level > 0.05 for m in mods.values())),
        ("Emotion variety", not (mods["5HT"].level > 0.85 and mods["NE"].level < 0.15)),
    ]
    all_pass = True
    for desc, passed in checks:
        icon = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"    [{icon}] {desc}: {'PASS' if passed else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    dna = load_dna()

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   NEUROMOD TUNING SIM — 5-HT / Endorphin Equilibrium Fix      ║")
    print("║   Using LIVE T1 state from 2026-03-24 05:18 UTC               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO A: Current DNA (baseline — shows the problem)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  SCENARIO A: CURRENT DNA (no changes) — 2000 ticks from live state")
    print("═" * 75)

    a_mods, a_traj = run_sim("current", dna, ticks=2000,
                              initial_levels=CURRENT_LEVELS,
                              initial_sens=CURRENT_SENS,
                              initial_sp=CURRENT_SP)
    a_pass = print_results("Scenario A: Current DNA", a_mods, a_traj)

    # Show raw input values for diagnosis
    raw_inputs = compute_emergent_inputs(
        LIVE_SPHERE_BALANCE, LIVE_TRINITY, LIVE_CHI, LIVE_DYNAMICS,
        LIVE_PI, LIVE_PREDICTION, LIVE_NS, LIVE_EXPRESSION,
        LIVE_RESONANCE, is_dreaming=False, dna=dna)
    print(f"\n  Raw emergent inputs (before any scaling):")
    for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        print(f"    {name:10s}: {raw_inputs[name]:.4f}")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO B: Input Scaling DNA (Option B)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  SCENARIO B: INPUT SCALING DNA — per-modulator input attenuation")
    print("  sht_input_scale=0.55, endorphin_input_scale=0.65")
    print("═" * 75)

    dna_b = dict(dna)
    dna_b["sht_input_scale"] = 0.55      # 0.865 * 0.55 = 0.476
    dna_b["endorphin_input_scale"] = 0.65  # 0.383 * 0.65 = 0.249

    b_mods, b_traj = run_sim("scaled", dna_b, ticks=2000,
                              initial_levels=CURRENT_LEVELS,
                              initial_sens=CURRENT_SENS,
                              initial_sp=CURRENT_SP)
    b_pass = print_results("Scenario B: Input Scaling DNA", b_mods, b_traj)

    scaled_inputs = dict(raw_inputs)
    scaled_inputs["5HT"] *= 0.55
    scaled_inputs["Endorphin"] *= 0.65
    print(f"\n  Scaled inputs (after DNA scaling):")
    for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        scale = dna_b.get(f"{'sht' if name == '5HT' else name.lower()}_input_scale", 1.0)
        tag = f" (×{scale})" if scale != 1.0 else ""
        print(f"    {name:10s}: {raw_inputs[name] * scale:.4f}{tag}")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO C: Scaled DNA + Dreaming cycles (realistic operation)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  SCENARIO C: SCALED DNA + DREAMING — 3 dream cycles in 2000 ticks")
    print("  Dreams at ticks 400-420, 900-920, 1400-1420 (20 ticks each)")
    print("═" * 75)

    c_mods, c_traj = run_sim("scaled+dream", dna_b, ticks=2000,
                              initial_levels=CURRENT_LEVELS,
                              initial_sens=CURRENT_SENS,
                              initial_sp=CURRENT_SP,
                              dreaming_ranges=[(400, 420), (900, 920), (1400, 1420)])
    c_pass = print_results("Scenario C: Scaled + Dreaming", c_mods, c_traj)

    # ══════════════════════════════════════════════════════════════
    # SCENARIO D: Scaled DNA from FRESH birth (0.5 all)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  SCENARIO D: SCALED DNA from FRESH BIRTH — all levels start at 0.5")
    print("═" * 75)

    fresh = {n: 0.5 for n in CLEARANCE_RATES}
    d_mods, d_traj = run_sim("fresh+scaled", dna_b, ticks=2000,
                              initial_levels=fresh)
    d_pass = print_results("Scenario D: Fresh Birth + Scaling", d_mods, d_traj)

    # ══════════════════════════════════════════════════════════════
    # SCENARIO E: Sensitivity — what if sphere balance drops?
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  SCENARIO E: STRESS TEST — sphere balance drops (imbalanced clocks)")
    print("  inner_body=0.3, outer_body=0.1 (crisis state)")
    print("═" * 75)

    e_mods, e_traj = run_sim("stress", dna_b, ticks=2000,
                              initial_levels=CURRENT_LEVELS,
                              initial_sens=CURRENT_SENS,
                              initial_sp=CURRENT_SP,
                              state_overrides={"sphere": {"inner_body": 0.3, "outer_body": 0.1}})
    e_pass = print_results("Scenario E: Stress (low balance)", e_mods, e_traj)

    # ══════════════════════════════════════════════════════════════
    # SCENARIO F: Scale sweep — find optimal scale values
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  SCENARIO F: SCALE SWEEP — finding optimal input_scale values")
    print("═" * 75)
    print(f"  {'5HT_scale':>10s} {'End_scale':>10s} {'5HT_eq':>8s} {'End_eq':>8s} {'5HT_ok':>7s} {'End_ok':>7s} {'All_pass':>9s}")
    print("  " + "-" * 65)

    best_sht_scale = 0.55
    best_end_scale = 0.65
    best_score = 999

    for sht_s in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for end_s in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            dna_test = dict(dna)
            dna_test["sht_input_scale"] = sht_s
            dna_test["endorphin_input_scale"] = end_s

            test_mods, _ = run_sim("sweep", dna_test, ticks=2000,
                                    initial_levels=CURRENT_LEVELS,
                                    initial_sens=CURRENT_SENS,
                                    initial_sp=CURRENT_SP)

            sht_eq = test_mods["5HT"].level
            end_eq = test_mods["Endorphin"].level
            sht_ok = 0.55 <= sht_eq <= 0.80
            end_ok = 0.50 <= end_eq <= 0.80

            # Score: distance from ideal (0.70 for both)
            score = abs(sht_eq - 0.70) + abs(end_eq - 0.70)
            all_ok = sht_ok and end_ok

            # Check no other mod broke
            other_ok = (test_mods["NE"].level > 0.30 and
                       test_mods["GABA"].level < 0.50 and
                       test_mods["DA"].level > 0.30)

            if all_ok and other_ok and score < best_score:
                best_score = score
                best_sht_scale = sht_s
                best_end_scale = end_s

            # Only print interesting rows
            if sht_s in [0.45, 0.55, 0.65] or end_s in [0.55, 0.65, 0.75]:
                tag = " ← BEST" if (sht_s == best_sht_scale and end_s == best_end_scale and all_ok) else ""
                print(f"  {sht_s:10.2f} {end_s:10.2f} {sht_eq:8.4f} {end_eq:8.4f} {'✓' if sht_ok else '✗':>7s} {'✓' if end_ok else '✗':>7s} {'✓' if (all_ok and other_ok) else '✗':>9s}{tag}")

    print(f"\n  OPTIMAL: sht_input_scale={best_sht_scale:.2f}, endorphin_input_scale={best_end_scale:.2f}")

    # ══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FINAL VERDICT")
    print("═" * 75)
    results = [
        ("A: Current DNA (baseline)", a_pass),
        ("B: Input Scaling", b_pass),
        ("C: Scaling + Dreaming", c_pass),
        ("D: Fresh Birth + Scaling", d_pass),
        ("E: Stress Test + Scaling", e_pass),
    ]
    for label, passed in results:
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{icon}] {label}")

    print(f"\n  Recommended DNA addition to titan_params.toml:")
    print(f"    sht_input_scale = {best_sht_scale:.2f}")
    print(f"    endorphin_input_scale = {best_end_scale:.2f}")
    print("═" * 75)
