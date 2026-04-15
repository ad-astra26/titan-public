#!/usr/bin/env python3
"""
Neuromodulator Emergent Redesign — Pre-Implementation Simulation

Simulates the proposed architecture-derived neuromodulator inputs using
REAL live data from both Titans. Runs 1000 virtual ticks to find equilibrium
and validates all success criteria before any production code is changed.

Usage:
    source test_env/bin/activate
    python scripts/neuromod_simulation.py
"""

import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan_plugin.logic.neuromodulator import (
    CLEARANCE_RATES, COUPLING_MATRIX, METABOLIC_COSTS,
    Neuromodulator, NEUROMOD_PRESSURE_RATE,
)

# ═══════════════════════════════════════════════════════════════════
# DNA WEIGHTS (proposed birth parameters for titan_params.toml)
# ═══════════════════════════════════════════════════════════════════

DNA = {
    # DA: Reward Prediction Error (minimal change — already good)
    "da_prediction_surprise": 0.30,
    "da_action_outcome": 0.40,
    "da_success_trend": 0.20,
    "da_pi_curvature_reward": 0.10,

    # 5-HT: Balance-Derived Stability
    "sht_sphere_balance": 0.35,
    "sht_pi_regularity": 0.20,
    "sht_chi_circulation": 0.15,
    "sht_drift_stability": 0.15,
    "sht_developmental_maturity": 0.15,

    # NE: Chi-Tonic + Trinity Coherence
    "ne_chi_tonic": 0.15,
    "ne_trinity_coherence": 0.20,
    "ne_prediction_surprise": 0.25,
    "ne_state_change_rate": 0.15,
    "ne_action_uncertainty": 0.15,
    "ne_system_excitation": 0.10,

    # ACh: Attention / Learning Demand
    "ach_state_change_rate": 0.30,
    "ach_ns_learning_rate": 0.25,
    "ach_pi_irregularity": 0.20,
    "ach_filter_down_activity": 0.25,

    # Endorphin: Flow / Intrinsic Reward
    "endorphin_action_alignment": 0.25,
    "endorphin_pi_flow": 0.25,
    "endorphin_resonance_harmony": 0.20,
    "endorphin_chi_body_vitality": 0.15,
    "endorphin_chi_circulation": 0.15,

    # GABA: Inhibition / Rest Need (NO circular neuromod deps)
    "gaba_dreaming": 0.40,
    "gaba_metabolic_drain": 0.20,
    "gaba_expression_fire_rate": 0.15,
    "gaba_chi_stagnation": 0.15,
    "gaba_epoch_saturation": 0.10,

    # Balance DNA
    "balance_body_weight": 1,
    "balance_mind_weight": 3,
    "balance_streak_normalization": 100,
}


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def compute_emergent_inputs(
    sphere_balance: dict,
    trinity_coherence: dict,
    chi_state: dict,
    consciousness_dynamics: dict,
    pi_state: dict,
    prediction_state: dict,
    ns_state: dict,
    expression_state: dict,
    resonance_state: dict,
    is_dreaming: bool,
    dna: dict,
) -> dict:
    """Compute neuromodulator inputs from architectural state + DNA weights.

    Every signal is derived from a different architectural subsystem.
    No circular dependencies between neuromod inputs.
    Weight coefficients are DNA (birth parameters).
    """

    # ── DA: Reward Prediction Error ──
    da_input = clamp(
        dna["da_prediction_surprise"] * prediction_state.get("surprise", 0.0) +
        dna["da_action_outcome"] * prediction_state.get("action_outcome", 0.5) +
        dna["da_success_trend"] * prediction_state.get("success_rate", 0.5) +
        dna["da_pi_curvature_reward"] * pi_state.get("curvature_delta", 0.0)
    )

    # ── 5-HT: Balance-Derived Stability ──
    # Sphere clock balance: body×1, mind×3 weighted
    body_bal = (sphere_balance.get("inner_body", 0.0) + sphere_balance.get("outer_body", 0.0)) / 2
    mind_bal = (sphere_balance.get("inner_mind", 0.0) + sphere_balance.get("outer_mind", 0.0)) / 2
    bw = dna["balance_body_weight"]
    mw = dna["balance_mind_weight"]
    weighted_balance = (body_bal * bw + mind_bal * mw) / (bw + mw)

    sht_input = clamp(
        dna["sht_sphere_balance"] * weighted_balance +
        dna["sht_pi_regularity"] * pi_state.get("regularity", 0.0) +
        dna["sht_chi_circulation"] * chi_state.get("circulation", 0.0) +
        dna["sht_drift_stability"] * (1.0 - consciousness_dynamics.get("drift_magnitude", 0.5)) +
        dna["sht_developmental_maturity"] * min(1.0, pi_state.get("developmental_age", 0) / 50.0)
    )

    # ── NE: Chi-Tonic + Trinity Coherence ──
    avg_coherence = (trinity_coherence.get("inner", 0.5) + trinity_coherence.get("outer", 0.5)) / 2
    ne_input = clamp(
        dna["ne_chi_tonic"] * chi_state.get("total", 0.5) +
        dna["ne_trinity_coherence"] * avg_coherence +
        dna["ne_prediction_surprise"] * prediction_state.get("surprise", 0.0) +
        dna["ne_state_change_rate"] * consciousness_dynamics.get("drift_delta", 0.0) +
        dna["ne_action_uncertainty"] * (1.0 - prediction_state.get("success_rate", 0.5)) +
        dna["ne_system_excitation"] * consciousness_dynamics.get("density", 0.0)
    )

    # ── ACh: Attention / Learning Demand ──
    ach_input = clamp(
        dna["ach_state_change_rate"] * consciousness_dynamics.get("drift_delta", 0.0) +
        dna["ach_ns_learning_rate"] * ns_state.get("transition_delta", 0.0) +
        dna["ach_pi_irregularity"] * (1.0 - pi_state.get("regularity", 0.5)) +
        dna["ach_filter_down_activity"] * ns_state.get("filter_down_writes", 0.0)
    )

    # ── Endorphin: Flow / Intrinsic Reward ──
    endorphin_input = clamp(
        dna["endorphin_action_alignment"] * expression_state.get("alignment", 0.5) +
        dna["endorphin_pi_flow"] * pi_state.get("cluster_streak", 0.0) +
        dna["endorphin_resonance_harmony"] * resonance_state.get("resonant_fraction", 0.0) +
        dna["endorphin_chi_body_vitality"] * chi_state.get("body", 0.5) +
        dna["endorphin_chi_circulation"] * chi_state.get("circulation", 0.0)
    )

    # ── GABA: Inhibition / Rest Need (NO circular deps) ──
    gaba_input = clamp(
        dna["gaba_dreaming"] * (1.0 if is_dreaming else 0.0) +
        dna["gaba_metabolic_drain"] * chi_state.get("drain", 0.0) +
        dna["gaba_expression_fire_rate"] * expression_state.get("fire_rate", 0.0) +
        dna["gaba_chi_stagnation"] * (1.0 - chi_state.get("circulation", 0.0)) +
        dna["gaba_epoch_saturation"] * consciousness_dynamics.get("epoch_gap_ratio", 0.0)
    )

    result = {
        "DA": da_input,
        "5HT": sht_input,
        "NE": ne_input,
        "ACh": ach_input,
        "Endorphin": endorphin_input,
        "GABA": gaba_input,
    }

    # Dreaming modulation (same as current — suppresses production, boosts GABA)
    if is_dreaming:
        _gaba_level = result["GABA"]
        _dream_suppression = max(0.1, 1.0 - _gaba_level * 0.8)
        for key in result:
            if key != "GABA":
                result[key] *= _dream_suppression
        result["GABA"] = min(1.0, result["GABA"] * 1.3)

    return result


def get_gain(level, setpoint, sensitivity):
    """Compute gain multiplier for downstream modulation."""
    deviation = (level - setpoint) / max(0.01, setpoint)
    gain = 1.0 * (1.0 + deviation * sensitivity)
    return max(0.3, min(3.0, gain))


# ═══════════════════════════════════════════════════════════════════
# REAL STATE FROM LIVE TITANS (collected via API + filesystem)
# ═══════════════════════════════════════════════════════════════════

# T1 sphere clock balance (from sphere_clock_state.json)
T1_SPHERE_BALANCE = {
    "inner_body": min(1.0, 512 / DNA["balance_streak_normalization"]),     # 1.0
    "inner_mind": min(1.0, 1487 / DNA["balance_streak_normalization"]),    # 1.0
    "outer_body": min(1.0, 1 / DNA["balance_streak_normalization"]),       # 0.01
    "outer_mind": min(1.0, 0 / DNA["balance_streak_normalization"]),       # 0.0
}

# T1 trinity coherence (from latest epochs: iB=0.941 iM=0.468 iS=0.291 | oB=0.792 oM=0.444 oS=0.465)
def compute_coherence(vals):
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return max(0.0, 1.0 - std / 0.4)

T1_TRINITY = {
    "inner": compute_coherence([0.941, 0.468, 0.291]),
    "outer": compute_coherence([0.792, 0.444, 0.465]),
}

# T1 Chi state (from /v4/chi)
T1_CHI = {"total": 0.614, "body": 0.612, "circulation": 0.003, "drain": 0.03}

# T1 consciousness dynamics (from latest epochs)
# drift ~0.0005 avg, density ~0.015, epoch gap ~9s, MIN_GAP ~8.8s
T1_DYNAMICS = {"drift_magnitude": 0.0005, "drift_delta": 0.3, "density": 0.015, "epoch_gap_ratio": 0.5}

# T1 pi state (from pi_heartbeat_state.json)
# heartbeat_ratio = total_pi / total_observed = 681/9540 = 0.071
T1_PI = {"regularity": 0.071, "cluster_streak": 0.0, "developmental_age": 6, "curvature_delta": 0.3}

# T1 prediction state
T1_PREDICTION = {"surprise": 0.02, "action_outcome": 0.5, "success_rate": 0.5}

# T1 NS state (transitions ~86k over ~7500 ticks → ~11.5/tick, filter_down ~1/epoch)
T1_NS = {"transition_delta": 0.5, "filter_down_writes": 0.3}

# T1 expression state (fire counts: ART=761, MUSIC=723, SOCIAL=819 over 7518 ticks)
# fire_rate = total_fires / ticks ≈ 2303/7518 ≈ 0.31
# alignment = fire_count / eval_count
T1_EXPRESSION = {"fire_rate": 0.31, "alignment": 0.5}

# T1 resonance (BIG PULSE count from live log: spirit=13, mind=2, body=0)
T1_RESONANCE = {"resonant_fraction": 0.33}  # 1 out of 3 pairs consistently resonant

# T2 equivalents (similar but slightly different)
T2_SPHERE_BALANCE = {
    "inner_body": 1.0, "inner_mind": 1.0,
    "outer_body": 0.0, "outer_mind": 0.0,  # T2 outer often unbalanced
}
T2_TRINITY = {
    "inner": compute_coherence([0.916, 0.440, 0.337]),
    "outer": compute_coherence([0.766, 0.500, 0.418]),
}
T2_CHI = {"total": 0.588, "body": 0.587, "circulation": 0.010, "drain": 0.02}


def run_simulation(name, sphere_bal, trinity, chi, dynamics, pi_state,
                   pred, ns, expr, resonance, initial_levels=None,
                   initial_sensitivity=None, initial_setpoints=None,
                   ticks=1000, dreaming_at=None):
    """Run neuromod simulation for N ticks and return trajectories."""

    # Initialize modulators
    mods = {}
    for mod_name, clearance in CLEARANCE_RATES.items():
        mods[mod_name] = Neuromodulator(
            name=mod_name,
            clearance_rate=clearance,
        )
        if initial_levels and mod_name in initial_levels:
            mods[mod_name].level = initial_levels[mod_name]
            mods[mod_name].tonic_level = initial_levels[mod_name]
        if initial_sensitivity and mod_name in initial_sensitivity:
            mods[mod_name].sensitivity = initial_sensitivity[mod_name]
        if initial_setpoints and mod_name in initial_setpoints:
            mods[mod_name].setpoint = initial_setpoints[mod_name]

    # Track trajectories
    trajectories = {n: [] for n in mods}
    input_trajectories = {n: [] for n in mods}

    for tick in range(ticks):
        is_dreaming = dreaming_at is not None and dreaming_at[0] <= tick < dreaming_at[1]

        # Compute emergent inputs
        inputs = compute_emergent_inputs(
            sphere_balance=sphere_bal,
            trinity_coherence=trinity,
            chi_state=chi,
            consciousness_dynamics=dynamics,
            pi_state=pi_state,
            prediction_state=pred,
            ns_state=ns,
            expression_state=expr,
            resonance_state=resonance,
            is_dreaming=is_dreaming,
            dna=DNA,
        )

        # Compute cross-coupling
        cross_couplings = {}
        for target in mods:
            coupling_sum = 0.0
            for source, source_mod in mods.items():
                if source == target:
                    continue
                weight = COUPLING_MATRIX.get(source, {}).get(target, 0.0)
                coupling_sum += weight * source_mod.level
            cross_couplings[target] = coupling_sum

        # Update each modulator
        for mod_name, mod in mods.items():
            mod.update(inputs[mod_name], cross_couplings[mod_name], dt=1.0, chi_health=1.0)
            trajectories[mod_name].append(mod.level)
            input_trajectories[mod_name].append(inputs[mod_name])

    return mods, trajectories, input_trajectories


def print_comparison(label, old_levels, new_mods, new_inputs):
    """Print before/after comparison."""
    print(f"\n{'=' * 75}")
    print(f"  {label}")
    print(f"{'=' * 75}")
    print(f"{'Mod':12s} {'Old Level':>10s} {'New Level':>10s} {'Change':>10s} {'Input':>8s} {'Setpoint':>10s} {'Sens':>8s} {'Status':>10s}")
    print("-" * 75)
    for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        old = old_levels[name]
        new = new_mods[name].level
        inp = new_inputs[name][-1] if new_inputs[name] else 0
        sp = new_mods[name].setpoint
        sens = new_mods[name].sensitivity
        delta = new - old
        status = "FIXED" if abs(delta) > 0.1 else ("shifted" if abs(delta) > 0.03 else "stable")
        if name == "NE" and new > 0.15:
            status = "FIXED ✓"
        elif name == "5HT" and new < 0.85:
            status = "FIXED ✓"
        elif name == "GABA" and new < 0.80:
            status = "FIXED ✓"
        print(f"  {name:10s} {old:10.4f} {new:10.4f} {delta:+10.4f} {inp:8.4f} {sp:10.4f} {sens:8.4f} {status:>10s}")


def print_gains(label, mods):
    """Print downstream gain impacts."""
    print(f"\n  Downstream Gains ({label}):")
    da, sht, ne, ach, endo, gaba = (mods[n] for n in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"])

    gains = {
        "sensory_gain (NE)": ne.get_gain(),
        "exploration_temp (NE)": ne.get_gain(),
        "filter_down (NE)": ne.get_gain(),
        "accum_rate (1/5HT)": 1.0 / max(0.3, sht.get_gain()),
        "refractory (5HT)": sht.get_gain(),
        "threshold_raise (GABA)": gaba.get_gain(),
        "system_energy (1/GABA)": 1.0 / max(0.3, gaba.get_gain()),
        "learning_rate (DA)": da.get_gain(),
        "motivation (Endo)": endo.get_gain(),
    }
    for name, val in gains.items():
        ok = "OK" if 0.5 < val < 2.0 else "WATCH" if 0.3 < val < 3.0 else "HIGH"
        print(f"    {name:25s}: {val:.3f}  [{ok}]")


def print_trajectory_samples(trajectories, ticks_to_show=(0, 50, 100, 250, 500, 999)):
    """Print trajectory at key tick points."""
    print(f"\n  Trajectory snapshots:")
    print(f"  {'Tick':>6s}  {'DA':>7s} {'5HT':>7s} {'NE':>7s} {'ACh':>7s} {'Endo':>7s} {'GABA':>7s}")
    for t in ticks_to_show:
        if t < len(trajectories["DA"]):
            print(f"  {t:6d}  " + "  ".join(
                f"{trajectories[n][t]:7.4f}" for n in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]))


def check_success_criteria(mods, trajectories):
    """Validate all success criteria from the plan."""
    print("\n  SUCCESS CRITERIA:")
    checks = [
        ("NE in 0.20-0.50", 0.20 <= mods["NE"].level <= 0.50),
        ("5-HT in 0.50-0.75", 0.50 <= mods["5HT"].level <= 0.75),
        ("GABA below 0.80", mods["GABA"].level < 0.80),
        ("ACh above 0.40", mods["ACh"].level > 0.40),
        ("No mod at floor (<0.05)", all(m.level > 0.05 for m in mods.values())),
        ("No mod at ceiling (>0.95)", all(m.level < 0.95 for m in mods.values())),
        ("Emotion variety possible", not (mods["5HT"].level > 0.85 and mods["NE"].level < 0.15)),
    ]
    all_pass = True
    for desc, passed in checks:
        icon = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"    [{icon}] {desc}")

    # Learning testsuite check
    ne_ok = mods["NE"].level >= 0.15
    gaba_ok = mods["GABA"].level < 0.85
    ts_icon = "✓ PASS" if (ne_ok and gaba_ok) else "✗ FAIL"
    if not (ne_ok and gaba_ok):
        all_pass = False
    print(f"    [{ts_icon}] Learning testsuite unblocked (NE≥0.15 AND GABA<0.85)")

    return all_pass


# ═══════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Current (broken) levels from both Titans
    T1_CURRENT = {"DA": 0.649, "5HT": 0.906, "NE": 0.094, "ACh": 0.387, "Endorphin": 0.885, "GABA": 0.831}
    T2_CURRENT = {"DA": 0.675, "5HT": 0.933, "NE": 0.090, "ACh": 0.365, "Endorphin": 0.921, "GABA": 0.820}

    # Current sensitivity and setpoints (from state files)
    T1_SENS = {"DA": 0.95, "5HT": 0.10, "NE": 1.67, "ACh": 1.21, "Endorphin": 0.11, "GABA": 0.39}
    T1_SP = {"DA": 0.69, "5HT": 0.70, "NE": 0.35, "ACh": 0.45, "Endorphin": 0.70, "GABA": 0.59}

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   NEUROMODULATOR EMERGENT REDESIGN — PRE-IMPLEMENTATION SIM     ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # ── Step 1: Show new inputs vs old ──
    print("\n" + "=" * 75)
    print("  STEP 1: NEW EMERGENT INPUTS (from live Titan state)")
    print("=" * 75)

    new_inputs = compute_emergent_inputs(
        T1_SPHERE_BALANCE, T1_TRINITY, T1_CHI, T1_DYNAMICS, T1_PI,
        T1_PREDICTION, T1_NS, T1_EXPRESSION, T1_RESONANCE,
        is_dreaming=False, dna=DNA)

    # Old inputs for comparison
    from titan_plugin.logic.neuromodulator import compute_inputs_from_titan
    old_inputs = compute_inputs_from_titan(
        prediction_surprise=0.02, action_outcome=0.5,
        middle_path_stability=0.65, pi_regularity=0.071,
        developmental_age=6, episodic_growth_rate=0.0,
        action_success_rate=0.5, outcome_variance=0.0,
        new_info_rate=0.0, action_state_alignment=0.5,
        creative_quality=0.0, system_excitation=0.015,
        is_dreaming=False, chi_total=0.614, chi_body_vitality=0.612, chi_circulation=0.003)

    print(f"  {'Mod':12s} {'OLD Input':>10s} {'NEW Input':>10s} {'Change':>10s} {'Source':>30s}")
    print("-" * 75)
    sources = {
        "DA": "prediction + experience",
        "5HT": "sphere balance (body×1, mind×3)",
        "NE": "Chi tonic + trinity coherence",
        "ACh": "state change + NS learning",
        "Endorphin": "π-flow + resonance",
        "GABA": "dreaming + metabolic drain",
    }
    for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        delta = new_inputs[name] - old_inputs[name]
        print(f"  {name:10s} {old_inputs[name]:10.4f} {new_inputs[name]:10.4f} {delta:+10.4f} {sources[name]:>30s}")

    # ── Step 2: Simulation from CURRENT state (with adapted sensitivity/setpoints) ──
    print("\n\n" + "=" * 75)
    print("  STEP 2A: T1 SIMULATION — From CURRENT state (1000 ticks, ~57 min)")
    print("=" * 75)

    t1_mods, t1_traj, t1_inp = run_simulation(
        "T1", T1_SPHERE_BALANCE, T1_TRINITY, T1_CHI, T1_DYNAMICS, T1_PI,
        T1_PREDICTION, T1_NS, T1_EXPRESSION, T1_RESONANCE,
        initial_levels=T1_CURRENT,
        initial_sensitivity=T1_SENS,
        initial_setpoints=T1_SP,
    )
    print_comparison("T1: Current → After 1000 ticks with NEW inputs", T1_CURRENT, t1_mods, t1_inp)
    print_trajectory_samples(t1_traj)
    print_gains("T1", t1_mods)
    t1_pass = check_success_criteria(t1_mods, t1_traj)

    # ── Step 2B: Simulation from FRESH state (birth — all 0.5) ──
    print("\n\n" + "=" * 75)
    print("  STEP 2B: FRESH BIRTH SIMULATION — From 0.5 (1000 ticks)")
    print("=" * 75)

    fresh_levels = {n: 0.5 for n in CLEARANCE_RATES}
    fresh_mods, fresh_traj, fresh_inp = run_simulation(
        "Fresh", T1_SPHERE_BALANCE, T1_TRINITY, T1_CHI, T1_DYNAMICS, T1_PI,
        T1_PREDICTION, T1_NS, T1_EXPRESSION, T1_RESONANCE,
        initial_levels=fresh_levels,
    )
    print_comparison("Fresh Birth → After 1000 ticks with NEW inputs", fresh_levels, fresh_mods, fresh_inp)
    print_trajectory_samples(fresh_traj)
    print_gains("Fresh", fresh_mods)
    fresh_pass = check_success_criteria(fresh_mods, fresh_traj)

    # ── Step 3: Dreaming cycle test ──
    print("\n\n" + "=" * 75)
    print("  STEP 3: DREAMING CYCLE TEST — Dream at tick 400-420 (20 ticks)")
    print("=" * 75)

    dream_mods, dream_traj, dream_inp = run_simulation(
        "Dream", T1_SPHERE_BALANCE, T1_TRINITY, T1_CHI, T1_DYNAMICS, T1_PI,
        T1_PREDICTION, T1_NS, T1_EXPRESSION, T1_RESONANCE,
        initial_levels=T1_CURRENT,
        initial_sensitivity=T1_SENS,
        initial_setpoints=T1_SP,
        dreaming_at=(400, 420),
    )
    print_comparison("T1 with dream cycle → After 1000 ticks", T1_CURRENT, dream_mods, dream_inp)
    print_trajectory_samples(dream_traj, ticks_to_show=(0, 100, 399, 400, 410, 420, 421, 500, 999))
    dream_pass = check_success_criteria(dream_mods, dream_traj)

    # ── Step 4: T2 simulation ──
    print("\n\n" + "=" * 75)
    print("  STEP 4: T2 SIMULATION — From T2 current state (1000 ticks)")
    print("=" * 75)

    t2_mods, t2_traj, t2_inp = run_simulation(
        "T2", T2_SPHERE_BALANCE, T2_TRINITY, T2_CHI, T1_DYNAMICS, T1_PI,
        T1_PREDICTION, T1_NS, T1_EXPRESSION, T1_RESONANCE,
        initial_levels=T2_CURRENT,
    )
    print_comparison("T2: Current → After 1000 ticks with NEW inputs", T2_CURRENT, t2_mods, t2_inp)
    t2_pass = check_success_criteria(t2_mods, t2_traj)

    # ── Step 5: GABA cascade verification ──
    print("\n\n" + "=" * 75)
    print("  STEP 5: GABA CASCADE VERIFICATION")
    print("=" * 75)
    print(f"  GABA has NO circular dependency on sht_input or ne_input")
    print(f"  GABA input sources: dreaming={0.0}, drain={T1_CHI['drain']:.3f}, "
          f"expr_rate={T1_EXPRESSION['fire_rate']:.3f}, "
          f"chi_stag={1.0-T1_CHI['circulation']:.3f}, "
          f"epoch_sat={T1_DYNAMICS['epoch_gap_ratio']:.3f}")
    gaba_input_raw = (
        DNA["gaba_dreaming"] * 0.0 +
        DNA["gaba_metabolic_drain"] * T1_CHI["drain"] +
        DNA["gaba_expression_fire_rate"] * T1_EXPRESSION["fire_rate"] +
        DNA["gaba_chi_stagnation"] * (1.0 - T1_CHI["circulation"]) +
        DNA["gaba_epoch_saturation"] * T1_DYNAMICS["epoch_gap_ratio"]
    )
    print(f"  Raw GABA input (not dreaming): {gaba_input_raw:.4f}")
    print(f"  Old GABA input was:            0.4760 (with circular sht/ne deps)")
    direction = "DROPS (GOOD)" if gaba_input_raw < 0.4760 else "RISES (CHECK)"
    print(f"  Direction: GABA {direction}")

    # ── Final verdict ──
    print("\n" + "=" * 75)
    all_pass = t1_pass and fresh_pass and dream_pass and t2_pass
    if all_pass:
        print("  ✓ ALL SIMULATIONS PASS — SAFE TO IMPLEMENT")
    else:
        print("  ✗ SOME CRITERIA FAILED — REVIEW BEFORE IMPLEMENTING")
        if not t1_pass: print("    - T1 simulation failed")
        if not fresh_pass: print("    - Fresh birth simulation failed")
        if not dream_pass: print("    - Dreaming cycle simulation failed")
        if not t2_pass: print("    - T2 simulation failed")
    print("=" * 75)
