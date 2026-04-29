"""
titan_plugin/logic/nervous_system.py — Nervous System Dispatcher (T4).

Dispatches 11 TitanVM micro-programs that operate on observable state.
TitanVM is the Trinity-level peer of CGN — the hand-coded supervisor
that provides deterministic baseline priors which the V5 neural
NervousSystem learns from during warmup and blends with outcome rewards.

Programs (5 original + 6 added 2026-04-16 for rFP β Stage 1):

INNER (autonomic — reactive to state):
  1. REFLEX      — fires when velocity spikes (sudden perturbation)
  2. FOCUS       — fires when polarity sustained beyond threshold
  3. INTUITION   — fires when direction drops (erratic movement)
  4. IMPULSE     — fires when magnitude + coherence both high (ready to act)
  5. METABOLISM  — fires on metabolic stress (high inner magnitude OR low coherence)
  6. VIGILANCE   — fires continuously at low level, spikes on velocity/perturbation

OUTER (personality — cognitive/social):
  7. INSPIRATION — fires when coherence high + velocity positive + direction novel
  8. CREATIVITY  — fires on coherent outer_body + meaningful direction
  9. CURIOSITY   — fires on ambivalent polarity + sustained outer_mind engagement
 10. EMPATHY     — fires on outer-part harmony (social resonance signature)
 11. REFLECTION  — fires on stillness + inner coherence (meditation/rest signature)

Each program reads from context (flattened observables dict), outputs a
score via SCORE opcode. Score > 0 means the signal fires with that urgency.
The score also becomes the vm_baseline target the V5 NN learns during
supervised warmup (neural_nervous_system.py _compute_targets).

The NervousSystem is wired into InnerTrinityCoordinator (T3) and runs
on every tick cycle.

rFP β reference: titan-docs/rFP_ns_program_signal_restoration.md § 4a-bis
"""
import logging
from typing import Optional

from titan_plugin.logic.titan_vm import TitanVM, Op, VMResult

logger = logging.getLogger(__name__)


# ── Observable context flattening ────────────────────────────────────

def flatten_observables(observables: dict[str, dict]) -> dict[str, float]:
    """
    Flatten 6-part × 5-observable dict into dotted paths for TitanVM LOAD.

    Input:  {"inner_body": {"coherence": 0.9, "velocity": 0.1, ...}, ...}
    Output: {"inner_body.coherence": 0.9, "inner_body.velocity": 0.1, ...}

    Also computes aggregate averages:
      - "inner.coherence_avg", "outer.coherence_avg", "all.coherence_avg"
      - Same for magnitude, velocity, direction, polarity
    """
    flat: dict[str, float] = {}

    inner_sums: dict[str, float] = {}
    outer_sums: dict[str, float] = {}
    all_sums: dict[str, float] = {}
    inner_count = 0
    outer_count = 0

    for part_name, obs in observables.items():
        for metric_name, value in obs.items():
            flat[f"{part_name}.{metric_name}"] = float(value)

            all_sums[metric_name] = all_sums.get(metric_name, 0.0) + float(value)

            if part_name.startswith("inner_"):
                inner_sums[metric_name] = inner_sums.get(metric_name, 0.0) + float(value)
                if metric_name == "coherence":
                    inner_count += 1
            elif part_name.startswith("outer_"):
                outer_sums[metric_name] = outer_sums.get(metric_name, 0.0) + float(value)
                if metric_name == "coherence":
                    outer_count += 1

    total_count = inner_count + outer_count or 1

    for metric in ("coherence", "magnitude", "velocity", "direction", "polarity"):
        if inner_count > 0:
            flat[f"inner.{metric}_avg"] = inner_sums.get(metric, 0.0) / inner_count
        if outer_count > 0:
            flat[f"outer.{metric}_avg"] = outer_sums.get(metric, 0.0) / outer_count
        flat[f"all.{metric}_avg"] = all_sums.get(metric, 0.0) / total_count

    return flat


# ── Nervous System Programs (TitanVM bytecode) ──────────────────────
#
# Each _make_<name>_program() factory accepts an optional `config` dict
# corresponding to titan_params.toml [titan_vm.programs.<name>]. When
# `config["v2_enabled"] == True`, the factory returns the rFP_titan_vm_v2
# Phase 2 variant (smooth thresholds, rich observables, LOAD_EMA/LOAD_DT).
# Otherwise it returns the legacy v1 bytecode. Default = v1 (gate flips
# happen per-Titan via config; calibration-neutral on deploy).


def _v2_enabled(config, name: str) -> bool:
    """True iff [titan_vm.programs.<name>].v2_enabled is truthy."""
    if not config:
        return False
    programs = (config.get("programs") or {}) if isinstance(config, dict) else {}
    entry = (programs.get(name.lower()) or programs.get(name.upper()) or {})
    return bool(entry.get("v2_enabled", False))


def _v2_cfg(config, name: str, key: str, default):
    """Read [titan_vm.programs.<name>].<key> with default fallback."""
    if not config:
        return default
    programs = (config.get("programs") or {}) if isinstance(config, dict) else {}
    entry = (programs.get(name.lower()) or programs.get(name.upper()) or {})
    val = entry.get(key, default)
    # TOML parsers return ints for whole numbers; coerce to float for math.
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    return val


def _v2_weighted_sum(
    terms: list,
    clamp_lo: float = 0.0,
    clamp_hi: float = 1.0,
) -> list:
    """Build bytecode: score = clamp(Σ weight_i × smooth_gate_i, lo, hi).

    Each term is a tuple of bytecode fragments (already producing a single
    stack value). This helper chains `ADD` across all terms and then
    clamps + scores. Leaves output on the stack before HALT.
    """
    out: list = []
    # First term pushed as-is
    if not terms:
        return [(Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,)]
    out.extend(terms[0])
    for frag in terms[1:]:
        out.extend(frag)
        out.append((Op.ADD,))
    out.extend([
        (Op.PUSH, clamp_lo),
        (Op.PUSH, clamp_hi),
        (Op.CLAMP,),
        (Op.SCORE,),
        (Op.HALT,),
    ])
    return out


def _gate_load_soft_gt(path: str, threshold: float, steepness: float,
                      weight: float) -> list:
    """Push: weight · sigmoid(k · (LOAD(path) − threshold))."""
    return [
        (Op.LOAD, path),
        (Op.SOFT_GT, threshold, steepness),
        (Op.PUSH, weight),
        (Op.MUL,),
    ]


def _gate_load_soft_lt(path: str, threshold: float, steepness: float,
                      weight: float) -> list:
    """Push: weight · sigmoid(k · (threshold − LOAD(path)))."""
    return [
        (Op.LOAD, path),
        (Op.SOFT_LT, threshold, steepness),
        (Op.PUSH, weight),
        (Op.MUL,),
    ]


def _gate_ema_soft_gt(path: str, alpha: float, threshold: float,
                     steepness: float, weight: float) -> list:
    """Push: weight · sigmoid(k · (EMA(path, α) − threshold))."""
    return [
        (Op.LOAD_EMA, path, alpha),
        (Op.SOFT_GT, threshold, steepness),
        (Op.PUSH, weight),
        (Op.MUL,),
    ]


def _gate_dt_soft_gt(path: str, threshold: float, steepness: float,
                   weight: float) -> list:
    """Push: weight · sigmoid(k · (DT(path) − threshold)).

    Useful for detecting rising signals (positive derivative above
    threshold). LOAD_DT pushes (current − previous); a rising signal has
    positive DT, so SOFT_GT(0) lights up only on the rise.
    """
    return [
        (Op.LOAD_DT, path),
        (Op.SOFT_GT, threshold, steepness),
        (Op.PUSH, weight),
        (Op.MUL,),
    ]




def _make_reflex_program(config: dict | None = None) -> list:
    """REFLEX: fires when average velocity spikes above threshold.

    v2 observables: velocity + consciousness.drift spike + NE + Δdrift.
    NE is the biological reflex-arousal correlate; drift-DT catches
    fast-onset salience events.
    """
    if _v2_enabled(config, "REFLEX"):
        k = _v2_cfg(config, "REFLEX", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_load_soft_gt("all.velocity_avg",
                               _v2_cfg(config, "REFLEX", "velocity_threshold", 0.3),
                               k,
                               _v2_cfg(config, "REFLEX", "w_velocity", 0.4)),
            _gate_load_soft_gt("neuromod.NE",
                               _v2_cfg(config, "REFLEX", "ne_threshold", 0.7),
                               k,
                               _v2_cfg(config, "REFLEX", "w_ne", 0.3)),
            _gate_dt_soft_gt("consciousness.drift",
                             _v2_cfg(config, "REFLEX", "drift_dt_threshold", 0.05),
                             k,
                             _v2_cfg(config, "REFLEX", "w_drift_dt", 0.3)),
        ])
    return [
        (Op.LOAD, "all.velocity_avg"),
        (Op.PUSH, 0.3),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = velocity_avg clamped to [0, 1]
        (Op.LOAD, "all.velocity_avg"),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_focus_program(config: dict | None = None) -> list:
    """FOCUS: fires when polarity is sustained beyond threshold (strong bias).

    v2 observables: polarity magnitude + ACh EMA + reasoning_reward_ema.
    ACh is the biological attention/focus correlate; reasoning EMA
    biases focus toward cognitive tasks with recent reward.
    """
    if _v2_enabled(config, "FOCUS"):
        k = _v2_cfg(config, "FOCUS", "steepness", 10.0)
        # ABS via SUB-trick: polarity is in [-1, 1]; use magnitude as
        # LOAD of mind_tensor.magnitude instead (positive signal).
        return _v2_weighted_sum([
            _gate_load_soft_gt("all.coherence_avg",
                               _v2_cfg(config, "FOCUS", "coherence_threshold", 0.5),
                               k,
                               _v2_cfg(config, "FOCUS", "w_coherence", 0.3)),
            _gate_ema_soft_gt("neuromod.ACh",
                              _v2_cfg(config, "FOCUS", "ach_alpha", 0.1),
                              _v2_cfg(config, "FOCUS", "ach_threshold", 0.5),
                              k,
                              _v2_cfg(config, "FOCUS", "w_ach", 0.4)),
            _gate_load_soft_gt("cgn.reasoning_reward_ema",
                               _v2_cfg(config, "FOCUS", "reasoning_threshold", 0.1),
                               k,
                               _v2_cfg(config, "FOCUS", "w_reasoning", 0.3)),
        ])
    return [
        (Op.LOAD, "all.polarity_avg"),
        (Op.ABS,),
        (Op.PUSH, 0.15),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = |polarity| clamped to [0, 1]
        (Op.LOAD, "all.polarity_avg"),
        (Op.ABS,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_intuition_program(config: dict | None = None) -> list:
    """INTUITION: fires when direction drops (erratic, unpredictable movement).

    v2 observables: inner_mind coherence EMA + DA + GABA + active_haovs.
    DA + GABA signal "open-but-contained" exploratory state; active_haovs
    > 0 indicates live hypotheses to leap from.
    """
    if _v2_enabled(config, "INTUITION"):
        k = _v2_cfg(config, "INTUITION", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_ema_soft_gt("inner_mind.coherence",
                              _v2_cfg(config, "INTUITION", "coh_alpha", 0.1),
                              _v2_cfg(config, "INTUITION", "coh_threshold", 0.5),
                              k,
                              _v2_cfg(config, "INTUITION", "w_coherence", 0.3)),
            _gate_load_soft_gt("neuromod.DA",
                               _v2_cfg(config, "INTUITION", "da_threshold", 0.55),
                               k,
                               _v2_cfg(config, "INTUITION", "w_da", 0.25)),
            _gate_load_soft_gt("neuromod.GABA",
                               _v2_cfg(config, "INTUITION", "gaba_threshold", 0.35),
                               k,
                               _v2_cfg(config, "INTUITION", "w_gaba", 0.2)),
            _gate_load_soft_gt("cgn.active_haovs",
                               _v2_cfg(config, "INTUITION", "haov_threshold", 0.5),
                               k,
                               _v2_cfg(config, "INTUITION", "w_haov", 0.25)),
        ])
    return [
        (Op.LOAD, "all.direction_avg"),
        (Op.PUSH, 0.5),
        (Op.CMP_LT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = 1 - direction clamped to [0, 1]
        (Op.PUSH, 1.0),
        (Op.LOAD, "all.direction_avg"),
        (Op.SUB,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_impulse_program(config: dict | None = None) -> list:
    """IMPULSE: fires when magnitude + coherence both high (ready to act).

    v2 observables: magnitude + coherence + DA + ACh + outer-body direction-DT.
    DA drives action-tendency; ACh + direction-DT catch motor-onset signature.
    """
    if _v2_enabled(config, "IMPULSE"):
        k = _v2_cfg(config, "IMPULSE", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_load_soft_gt("all.coherence_avg",
                               _v2_cfg(config, "IMPULSE", "coh_threshold", 0.6),
                               k,
                               _v2_cfg(config, "IMPULSE", "w_coherence", 0.2)),
            _gate_load_soft_gt("all.magnitude_avg",
                               _v2_cfg(config, "IMPULSE", "mag_threshold", 0.4),
                               k,
                               _v2_cfg(config, "IMPULSE", "w_magnitude", 0.2)),
            _gate_load_soft_gt("neuromod.DA",
                               _v2_cfg(config, "IMPULSE", "da_threshold", 0.55),
                               k,
                               _v2_cfg(config, "IMPULSE", "w_da", 0.25)),
            _gate_load_soft_gt("neuromod.ACh",
                               _v2_cfg(config, "IMPULSE", "ach_threshold", 0.5),
                               k,
                               _v2_cfg(config, "IMPULSE", "w_ach", 0.15)),
            _gate_dt_soft_gt("outer_body.direction",
                             _v2_cfg(config, "IMPULSE", "direction_dt_threshold", 0.02),
                             k,
                             _v2_cfg(config, "IMPULSE", "w_direction_dt", 0.2)),
        ])
    return [
        # Check coherence > 0.7
        (Op.LOAD, "all.coherence_avg"),
        (Op.PUSH, 0.7),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "check_mag"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_mag",),
        # Check magnitude > 0.5
        (Op.LOAD, "all.magnitude_avg"),
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = coherence * magnitude
        (Op.LOAD, "all.coherence_avg"),
        (Op.LOAD, "all.magnitude_avg"),
        (Op.MUL,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_inspiration_program(config: dict | None = None) -> list:
    """INSPIRATION: fires when coherence high + velocity moderate + novel direction.

    v2 observables: outer_mind coherence EMA + DA + DA-DT + grounded_density
    + reasoning reward EMA. DA rising is the eureka signature; grounded
    density confirms inspiration has real concept traction behind it.
    """
    if _v2_enabled(config, "INSPIRATION"):
        k = _v2_cfg(config, "INSPIRATION", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_ema_soft_gt("outer_mind.coherence",
                              _v2_cfg(config, "INSPIRATION", "coh_alpha", 0.05),
                              _v2_cfg(config, "INSPIRATION", "coh_threshold", 0.55),
                              k,
                              _v2_cfg(config, "INSPIRATION", "w_coherence", 0.3)),
            _gate_load_soft_gt("neuromod.DA",
                               _v2_cfg(config, "INSPIRATION", "da_threshold", 0.6),
                               k,
                               _v2_cfg(config, "INSPIRATION", "w_da", 0.2)),
            _gate_dt_soft_gt("neuromod.DA",
                             _v2_cfg(config, "INSPIRATION", "da_dt_threshold", 0.01),
                             k,
                             _v2_cfg(config, "INSPIRATION", "w_da_dt", 0.2)),
            _gate_load_soft_gt("cgn.grounded_density",
                               _v2_cfg(config, "INSPIRATION", "density_threshold", 0.5),
                               k,
                               _v2_cfg(config, "INSPIRATION", "w_density", 0.15)),
            _gate_load_soft_gt("cgn.reasoning_reward_ema",
                               _v2_cfg(config, "INSPIRATION", "reasoning_threshold", 0.1),
                               k,
                               _v2_cfg(config, "INSPIRATION", "w_reasoning", 0.15)),
        ])
    return [
        # Coherence > 0.75
        (Op.LOAD, "all.coherence_avg"),
        (Op.PUSH, 0.75),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "check_vel"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_vel",),
        # Velocity > 0.05 (some movement happening)
        (Op.LOAD, "all.velocity_avg"),
        (Op.PUSH, 0.05),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "check_dir"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_dir",),
        # Direction between 0.3-0.8 (novel but not erratic)
        (Op.LOAD, "all.direction_avg"),
        (Op.PUSH, 0.3),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "upper_bound"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("upper_bound",),
        (Op.LOAD, "all.direction_avg"),
        (Op.PUSH, 0.8),
        (Op.CMP_LT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = coherence * (1 - |direction - 0.5|) — peaks at novelty sweet spot
        (Op.LOAD, "all.coherence_avg"),
        (Op.LOAD, "all.direction_avg"),
        (Op.PUSH, 0.5),
        (Op.SUB,),
        (Op.ABS,),
        (Op.PUSH, 1.0),
        (Op.SWAP,),
        (Op.SUB,),
        (Op.MUL,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_metabolism_program(config: dict | None = None) -> list:
    """METABOLISM: fires when metabolic reserves are stressed.

    v2 observables: inner_body magnitude + coherence (low) + sol balance EMA
    + GABA (low = stress) + NE (inverse — high-NE hypervigilance also
    taxes metabolic reserves). Slow SOL EMA picks up wallet-drain regimes.

    Fire conditions (OR'd):
      - inner_body.magnitude > 0.4 (body signaling load)
      - all.coherence_avg < 0.3 (disorganized state needs energy consolidation)
    Urgency = max signal strength, clamped [0, 1].
    """
    if _v2_enabled(config, "METABOLISM"):
        k = _v2_cfg(config, "METABOLISM", "steepness", 8.0)
        return _v2_weighted_sum([
            _gate_load_soft_gt("inner_body.magnitude",
                               _v2_cfg(config, "METABOLISM", "mag_threshold", 0.4),
                               k,
                               _v2_cfg(config, "METABOLISM", "w_magnitude", 0.3)),
            _gate_load_soft_lt("all.coherence_avg",
                               _v2_cfg(config, "METABOLISM", "coh_threshold", 0.3),
                               k,
                               _v2_cfg(config, "METABOLISM", "w_low_coherence", 0.2)),
            _gate_ema_soft_gt("metabolic.sol_balance",
                              _v2_cfg(config, "METABOLISM", "sol_alpha", 0.01),
                              _v2_cfg(config, "METABOLISM", "sol_threshold", 0.05),
                              k,
                              _v2_cfg(config, "METABOLISM", "w_sol_ema", 0.15)),
            _gate_load_soft_lt("neuromod.GABA",
                               _v2_cfg(config, "METABOLISM", "gaba_threshold", 0.3),
                               k,
                               _v2_cfg(config, "METABOLISM", "w_low_gaba", 0.15)),
            _gate_load_soft_gt("neuromod.NE",
                               _v2_cfg(config, "METABOLISM", "ne_threshold", 0.75),
                               k,
                               _v2_cfg(config, "METABOLISM", "w_high_ne", 0.2)),
        ])
    return [
        # Primary: inner_body magnitude high?
        (Op.LOAD, "inner_body.magnitude"),
        (Op.PUSH, 0.4),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire_mag"),
        # Secondary: global coherence low?
        (Op.LOAD, "all.coherence_avg"),
        (Op.PUSH, 0.3),
        (Op.CMP_LT,),
        (Op.BRANCH_IF, "fire_coh"),
        # Neither triggered
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire_mag",),
        # Urgency = inner_body.magnitude clamped
        (Op.LOAD, "inner_body.magnitude"),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
        ("fire_coh",),
        # Urgency = 1 - coherence_avg clamped
        (Op.PUSH, 1.0),
        (Op.LOAD, "all.coherence_avg"),
        (Op.SUB,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_creativity_program(config: dict | None = None) -> list:
    """CREATIVITY: fires on coherent outer_body + meaningful direction.

    v2 observables: outer_body coherence + direction-DT + Endorphin + ACh
    + language reward EMA. Endorphin is the flow-state correlate; direction-
    DT catches the "forming" phase of a creative move.

    Fire conditions (AND):
      - outer_body.coherence > 0.5 (body has ordered state)
      - outer_body.direction > 0.4 (direction carries meaning)
    Urgency = coherence × direction (peaks when both high).
    """
    if _v2_enabled(config, "CREATIVITY"):
        k = _v2_cfg(config, "CREATIVITY", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_load_soft_gt("outer_body.coherence",
                               _v2_cfg(config, "CREATIVITY", "coh_threshold", 0.5),
                               k,
                               _v2_cfg(config, "CREATIVITY", "w_coherence", 0.25)),
            _gate_dt_soft_gt("outer_body.direction",
                             _v2_cfg(config, "CREATIVITY", "direction_dt_threshold", 0.01),
                             k,
                             _v2_cfg(config, "CREATIVITY", "w_direction_dt", 0.25)),
            _gate_load_soft_gt("neuromod.Endorphin",
                               _v2_cfg(config, "CREATIVITY", "endo_threshold", 0.5),
                               k,
                               _v2_cfg(config, "CREATIVITY", "w_endorphin", 0.25)),
            _gate_load_soft_gt("neuromod.ACh",
                               _v2_cfg(config, "CREATIVITY", "ach_threshold", 0.5),
                               k,
                               _v2_cfg(config, "CREATIVITY", "w_ach", 0.1)),
            _gate_load_soft_gt("cgn.language_reward_ema",
                               _v2_cfg(config, "CREATIVITY", "language_threshold", 0.1),
                               k,
                               _v2_cfg(config, "CREATIVITY", "w_language", 0.15)),
        ])
    return [
        # Check: outer_body.coherence > 0.5
        (Op.LOAD, "outer_body.coherence"),
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "check_dir"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_dir",),
        # Check: outer_body.direction > 0.4
        (Op.LOAD, "outer_body.direction"),
        (Op.PUSH, 0.4),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = coherence × direction
        (Op.LOAD, "outer_body.coherence"),
        (Op.LOAD, "outer_body.direction"),
        (Op.MUL,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_curiosity_program(config: dict | None = None) -> list:
    """CURIOSITY: fires on exploratory state — ambivalent polarity + engagement.

    v2 observables: outer_mind magnitude + rising DA + low GABA + grounded
    density + Δgrounded density. Rising DA in open-GABA state = biological
    novelty-seeking; rising grounded-density means curiosity is finding
    things (reinforces the loop).

    Fire conditions (AND):
      - |outer_mind.polarity| < 0.25 (no strong pre-commitment — open to learning)
      - outer_mind.magnitude > 0.35 (mind actively engaged, not idle)
    Urgency = (1 - |polarity|) × magnitude (peaks at full ambivalence + engagement).
    """
    if _v2_enabled(config, "CURIOSITY"):
        k = _v2_cfg(config, "CURIOSITY", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_load_soft_gt("outer_mind.magnitude",
                               _v2_cfg(config, "CURIOSITY", "mag_threshold", 0.35),
                               k,
                               _v2_cfg(config, "CURIOSITY", "w_magnitude", 0.2)),
            _gate_dt_soft_gt("neuromod.DA",
                             _v2_cfg(config, "CURIOSITY", "da_dt_threshold", 0.005),
                             k,
                             _v2_cfg(config, "CURIOSITY", "w_da_dt", 0.25)),
            _gate_load_soft_lt("neuromod.GABA",
                               _v2_cfg(config, "CURIOSITY", "gaba_threshold", 0.35),
                               k,
                               _v2_cfg(config, "CURIOSITY", "w_low_gaba", 0.2)),
            _gate_load_soft_gt("cgn.grounded_density",
                               _v2_cfg(config, "CURIOSITY", "density_threshold", 0.3),
                               k,
                               _v2_cfg(config, "CURIOSITY", "w_density", 0.2)),
            _gate_dt_soft_gt("cgn.grounded_density",
                             _v2_cfg(config, "CURIOSITY", "density_dt_threshold", 0.01),
                             k,
                             _v2_cfg(config, "CURIOSITY", "w_density_dt", 0.15)),
        ])
    return [
        # Check: |outer_mind.polarity| < 0.25
        (Op.LOAD, "outer_mind.polarity"),
        (Op.ABS,),
        (Op.PUSH, 0.25),
        (Op.CMP_LT,),
        (Op.BRANCH_IF, "check_mag"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_mag",),
        # Check: outer_mind.magnitude > 0.35
        (Op.LOAD, "outer_mind.magnitude"),
        (Op.PUSH, 0.35),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = (1 - |polarity|) × magnitude
        (Op.PUSH, 1.0),
        (Op.LOAD, "outer_mind.polarity"),
        (Op.ABS,),
        (Op.SUB,),
        (Op.LOAD, "outer_mind.magnitude"),
        (Op.MUL,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_empathy_program(config: dict | None = None) -> list:
    """EMPATHY: fires on outer-part harmony — signature of social resonance.

    v2 observables: outer.coherence EMA + outer_mind.coherence + 5HT +
    Endorphin + social reward EMA. 5HT is the social-bonding correlate;
    social reward EMA biases empathy toward sustained interaction patterns.

    Fire conditions:
      - outer.coherence_avg > 0.5 (outer parts integrated, receptive to other)
    Urgency = outer.coherence_avg clamped.

    Distinct from INSPIRATION (whole-system novelty) — EMPATHY is specifically
    the OUTER harmony that signals openness toward encounter.
    """
    if _v2_enabled(config, "EMPATHY"):
        k = _v2_cfg(config, "EMPATHY", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_ema_soft_gt("outer.coherence_avg",
                              _v2_cfg(config, "EMPATHY", "outer_coh_alpha", 0.1),
                              _v2_cfg(config, "EMPATHY", "outer_coh_threshold", 0.5),
                              k,
                              _v2_cfg(config, "EMPATHY", "w_outer_coh", 0.25)),
            _gate_load_soft_gt("outer_mind.coherence",
                               _v2_cfg(config, "EMPATHY", "mind_coh_threshold", 0.5),
                               k,
                               _v2_cfg(config, "EMPATHY", "w_mind_coh", 0.15)),
            _gate_load_soft_gt("neuromod.5HT",
                               _v2_cfg(config, "EMPATHY", "fiveht_threshold", 0.5),
                               k,
                               _v2_cfg(config, "EMPATHY", "w_5ht", 0.25)),
            _gate_load_soft_gt("neuromod.Endorphin",
                               _v2_cfg(config, "EMPATHY", "endo_threshold", 0.5),
                               k,
                               _v2_cfg(config, "EMPATHY", "w_endorphin", 0.15)),
            _gate_load_soft_gt("cgn.social_reward_ema",
                               _v2_cfg(config, "EMPATHY", "social_threshold", 0.1),
                               k,
                               _v2_cfg(config, "EMPATHY", "w_social", 0.2)),
        ])
    return [
        # Check: outer.coherence_avg > 0.5
        (Op.LOAD, "outer.coherence_avg"),
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = outer.coherence_avg clamped
        (Op.LOAD, "outer.coherence_avg"),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_reflection_program(config: dict | None = None) -> list:
    """REFLECTION: fires on stillness + inner coherence — meditation signature.

    v2 observables: low velocity + high coherence + low NE + ACh +
    reasoning reward EMA. Low NE is the biological settled-attention state
    that reflection requires; reasoning EMA biases reflection toward
    cognitive-replay cadence.

    Fire conditions (AND):
      - all.velocity_avg < 0.15 (stillness — movement has settled)
      - all.coherence_avg > 0.5 (clear state, not just exhausted)
    Urgency = coherence × (1 - velocity) — peaks at stillness + clarity.
    """
    if _v2_enabled(config, "REFLECTION"):
        k = _v2_cfg(config, "REFLECTION", "steepness", 10.0)
        return _v2_weighted_sum([
            _gate_load_soft_lt("all.velocity_avg",
                               _v2_cfg(config, "REFLECTION", "velocity_threshold", 0.15),
                               k,
                               _v2_cfg(config, "REFLECTION", "w_low_velocity", 0.2)),
            _gate_load_soft_gt("all.coherence_avg",
                               _v2_cfg(config, "REFLECTION", "coh_threshold", 0.5),
                               k,
                               _v2_cfg(config, "REFLECTION", "w_coherence", 0.2)),
            _gate_load_soft_lt("neuromod.NE",
                               _v2_cfg(config, "REFLECTION", "ne_threshold", 0.55),
                               k,
                               _v2_cfg(config, "REFLECTION", "w_low_ne", 0.2)),
            _gate_load_soft_gt("neuromod.ACh",
                               _v2_cfg(config, "REFLECTION", "ach_threshold", 0.5),
                               k,
                               _v2_cfg(config, "REFLECTION", "w_ach", 0.15)),
            _gate_load_soft_gt("cgn.reasoning_reward_ema",
                               _v2_cfg(config, "REFLECTION", "reasoning_threshold", 0.1),
                               k,
                               _v2_cfg(config, "REFLECTION", "w_reasoning", 0.25)),
        ])
    return [
        # Check: velocity_avg < 0.15
        (Op.LOAD, "all.velocity_avg"),
        (Op.PUSH, 0.15),
        (Op.CMP_LT,),
        (Op.BRANCH_IF, "check_coh"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("check_coh",),
        # Check: coherence_avg > 0.5
        (Op.LOAD, "all.coherence_avg"),
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "fire"),
        (Op.PUSH, 0.0), (Op.SCORE,), (Op.HALT,),
        ("fire",),
        # Urgency = coherence × (1 - velocity)
        (Op.LOAD, "all.coherence_avg"),
        (Op.PUSH, 1.0),
        (Op.LOAD, "all.velocity_avg"),
        (Op.SUB,),
        (Op.MUL,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def _make_vigilance_program(config: dict | None = None) -> list:
    """VIGILANCE: fires continuously at a low base level, spikes on perturbation.

    Unlike the other 10 programs, VIGILANCE ALWAYS produces a positive score:
      - Base = 0.15 (always-on tonic activation — biological NE baseline)
      - Additive: + velocity_avg (spike on perturbation)
    Urgency = clamp(0.15 + velocity_avg, 0, 1).

    This maps directly to the biological noradrenergic vigilance system,
    which has tonic baseline firing + phasic spikes on salience events.
    rFP β § 4h clarifies that Stage 2 will use NE dynamics as VIGILANCE's
    reward signal — here we just provide the hand-coded VM baseline the
    NN learns from during warmup.

    v2 observables: tonic base + NE level + NE-DT spike + inner_body
    magnitude (somatic arousal) + inner_body magnitude-DT.
    """
    if _v2_enabled(config, "VIGILANCE"):
        k = _v2_cfg(config, "VIGILANCE", "steepness", 10.0)
        base = _v2_cfg(config, "VIGILANCE", "tonic_base", 0.15)
        # Build weighted sum starting with a PUSH base (tonic floor).
        return _v2_weighted_sum([
            [(Op.PUSH, base)],  # tonic baseline (always present)
            _gate_load_soft_gt("neuromod.NE",
                               _v2_cfg(config, "VIGILANCE", "ne_threshold", 0.55),
                               k,
                               _v2_cfg(config, "VIGILANCE", "w_ne", 0.25)),
            _gate_dt_soft_gt("neuromod.NE",
                             _v2_cfg(config, "VIGILANCE", "ne_dt_threshold", 0.02),
                             k,
                             _v2_cfg(config, "VIGILANCE", "w_ne_dt", 0.2)),
            _gate_load_soft_gt("inner_body.magnitude",
                               _v2_cfg(config, "VIGILANCE", "body_threshold", 0.3),
                               k,
                               _v2_cfg(config, "VIGILANCE", "w_body", 0.15)),
            _gate_dt_soft_gt("inner_body.magnitude",
                             _v2_cfg(config, "VIGILANCE", "body_dt_threshold", 0.02),
                             k,
                             _v2_cfg(config, "VIGILANCE", "w_body_dt", 0.15)),
        ])
    return [
        # Urgency = 0.15 + velocity_avg, clamped [0, 1]
        (Op.PUSH, 0.15),
        (Op.LOAD, "all.velocity_avg"),
        (Op.ADD,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def load_nervous_system_programs(program_names: list[str] | None = None,
                                  config: dict | None = None) -> dict[str, list]:
    """Load TitanVM bytecode programs by auto-discovery from this module.

    For each program NAME requested, looks up `_make_{name.lower()}_program`
    in the module namespace and calls it. This matches the auto-registration
    pattern used by V5 NeuralNervousSystem (which reads program names from
    `titan_params.toml [neural_nervous_system.programs.*]`).

    Args:
        program_names: Optional list of program names to load. If None,
                       discovers ALL `_make_*_program` functions in this
                       module (currently 11). Pass a config-derived list
                       to filter to only the enabled programs from TOML.

    Returns:
        Dict of {program_name_upper: bytecode_list}. Programs without a
        matching `_make_X_program` function are skipped with a WARNING
        (not silently dropped).

    Design (rFP β Stage 1 follow-up, 2026-04-16): originally a hardcoded
    dict of 5 programs. The asymmetry with NeuralNS auto-registration was
    one cause of the 6/11 supervision gap — adding a TOML entry didn't
    auto-create a VM supervisor. Refactored to name-based auto-discovery
    so adding a new program = (a) add config entry + (b) add
    `_make_NAME_program()` function in this module. No third place to update.

    Original V5 rFP design intent (memory/session_20260316_r5_r6_titanvm.md):
    "Spirit could modify its own scoring programs (self-improvement)."
    Auto-discovery is the foundation for that future capability.
    """
    import sys as _sys
    module = _sys.modules[__name__]

    # Discover all _make_*_program functions in this module
    discovered = {}
    for attr_name in dir(module):
        if attr_name.startswith("_make_") and attr_name.endswith("_program"):
            # _make_metabolism_program → METABOLISM
            prog_key = attr_name[len("_make_"):-len("_program")].upper()
            fn = getattr(module, attr_name)
            if callable(fn):
                discovered[prog_key] = fn

    # If no list passed, return all discovered programs
    if program_names is None:
        return {name: fn(config) for name, fn in discovered.items()}

    # Otherwise filter to requested names; warn on missing
    out = {}
    for name in program_names:
        key = name.upper()
        if key in discovered:
            out[key] = discovered[key](config)
        else:
            logger.warning(
                "[NervousSystem] Program '%s' requested but no "
                "_make_%s_program() found in nervous_system.py — "
                "supervision gap; NN will train against vm_baseline=0",
                key, key.lower())
    return out


# ── Nervous System Dispatcher ───────────────────────────────────────

class NervousSystem:
    """Dispatches TitanVM micro-programs based on observable state."""

    def __init__(self, vm: Optional[TitanVM] = None, config: Optional[dict] = None):
        """
        Args:
            vm: TitanVM instance. If None, creates a lightweight one
                without state_register (programs use context only).
            config: Optional dict passed to TitanVM (usually the parsed
                [titan_vm] section from titan_params.toml). Ignored when a
                pre-built ``vm`` is supplied. Plumbed 2026-04-16 so the toml
                section actually reaches the VM — previously stranded.
        """
        self.vm = vm or TitanVM(config=config)
        # rFP_titan_vm_v2 Phase 2: pass VM config through so each factory
        # can select v1 vs v2 bytecode per-program via
        # [titan_vm.programs.<name>].v2_enabled.
        self.programs = load_nervous_system_programs(config=config)
        self._last_signals: list[dict] = []
        self._total_evaluations: int = 0

    def evaluate(self, observables: dict[str, dict],
                 neuromod_state: Optional[dict] = None,
                 cgn_state: Optional[dict] = None) -> list[dict]:
        """
        Run all nervous system programs against current observables.

        Args:
            observables: Dict from ObservableEngine ({part: {5 metrics}}).
            neuromod_state: Optional dict of neuromodulator levels (keys like
                DA/NE/5HT/ACh/Endorphin/GABA with float values). Merged into
                context with "neuromod." prefix so TitanVM bytecode can
                `LOAD neuromod.DA` etc (rFP_titan_vm_v2 Phase 2 §3.7).
            cgn_state: Optional dict of CGN grounding state (keys like
                grounded_density, active_haovs, reasoning_V, language_V …).
                Merged into context with "cgn." prefix (§3.8).

        Returns:
            List of fired signals [{system, urgency, result}].
            Only includes programs that scored > 0.
        """
        context = flatten_observables(observables)

        # rFP_titan_vm_v2 Phase 2: inject neuromod + cgn state as
        # pre-flattened "neuromod.<name>" / "cgn.<key>" keys. TitanVM's
        # _load_value uses `if path in context` as its first resolution
        # step, so this is the minimal-surface way to expose new
        # observable namespaces without widening _load_value itself.
        if neuromod_state:
            for name, level in neuromod_state.items():
                if isinstance(level, (int, float)):
                    context[f"neuromod.{name}"] = float(level)
        if cgn_state:
            for key, val in cgn_state.items():
                if isinstance(val, (int, float)):
                    context[f"cgn.{key}"] = float(val)

        signals = []

        for name, program in self.programs.items():
            try:
                # v2 (rFP_titan_vm_v2 Phase 1a): pass program_key so LOAD_EMA +
                # LOAD_DT attribute state per-program (preparation for Phase 2
                # rewrites that use temporal opcodes). Telemetry also accumulates.
                result = self.vm.execute(program, context=context, program_key=name)
                if result.score > 0:
                    signals.append({
                        "system": name,
                        "urgency": round(result.score, 4),
                        "duration_ms": round(result.duration_ms, 3),
                    })
            except Exception as e:
                logger.debug("[NervousSystem] %s error: %s", name, e)

        self._last_signals = signals
        self._total_evaluations += 1
        return signals

    def get_stats(self) -> dict:
        return {
            "program_count": len(self.programs),
            "total_evaluations": self._total_evaluations,
            "last_signals": self._last_signals,
        }
