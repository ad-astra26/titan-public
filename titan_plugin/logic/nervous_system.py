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

def _make_reflex_program() -> list:
    """REFLEX: fires when average velocity spikes above threshold."""
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


def _make_focus_program() -> list:
    """FOCUS: fires when polarity is sustained beyond threshold (strong bias)."""
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


def _make_intuition_program() -> list:
    """INTUITION: fires when direction drops (erratic, unpredictable movement)."""
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


def _make_impulse_program() -> list:
    """IMPULSE: fires when magnitude + coherence both high (ready to act)."""
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


def _make_inspiration_program() -> list:
    """INSPIRATION: fires when coherence high + velocity moderate + novel direction."""
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


def _make_metabolism_program() -> list:
    """METABOLISM: fires when metabolic reserves are stressed.

    Fire conditions (OR'd):
      - inner_body.magnitude > 0.4 (body signaling load)
      - all.coherence_avg < 0.3 (disorganized state needs energy consolidation)
    Urgency = max signal strength, clamped [0, 1].
    """
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


def _make_creativity_program() -> list:
    """CREATIVITY: fires on coherent outer_body + meaningful direction.

    Fire conditions (AND):
      - outer_body.coherence > 0.5 (body has ordered state)
      - outer_body.direction > 0.4 (direction carries meaning)
    Urgency = coherence × direction (peaks when both high).
    """
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


def _make_curiosity_program() -> list:
    """CURIOSITY: fires on exploratory state — ambivalent polarity + engagement.

    Fire conditions (AND):
      - |outer_mind.polarity| < 0.25 (no strong pre-commitment — open to learning)
      - outer_mind.magnitude > 0.35 (mind actively engaged, not idle)
    Urgency = (1 - |polarity|) × magnitude (peaks at full ambivalence + engagement).
    """
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


def _make_empathy_program() -> list:
    """EMPATHY: fires on outer-part harmony — signature of social resonance.

    Fire conditions:
      - outer.coherence_avg > 0.5 (outer parts integrated, receptive to other)
    Urgency = outer.coherence_avg clamped.

    Distinct from INSPIRATION (whole-system novelty) — EMPATHY is specifically
    the OUTER harmony that signals openness toward encounter.
    """
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


def _make_reflection_program() -> list:
    """REFLECTION: fires on stillness + inner coherence — meditation signature.

    Fire conditions (AND):
      - all.velocity_avg < 0.15 (stillness — movement has settled)
      - all.coherence_avg > 0.5 (clear state, not just exhausted)
    Urgency = coherence × (1 - velocity) — peaks at stillness + clarity.
    """
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


def _make_vigilance_program() -> list:
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
    """
    return [
        # Urgency = 0.15 + velocity_avg, clamped [0, 1]
        (Op.PUSH, 0.15),
        (Op.LOAD, "all.velocity_avg"),
        (Op.ADD,),
        (Op.PUSH, 0.0), (Op.PUSH, 1.0), (Op.CLAMP,),
        (Op.SCORE,), (Op.HALT,),
    ]


def load_nervous_system_programs(program_names: list[str] | None = None) -> dict[str, list]:
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
        return {name: fn() for name, fn in discovered.items()}

    # Otherwise filter to requested names; warn on missing
    out = {}
    for name in program_names:
        key = name.upper()
        if key in discovered:
            out[key] = discovered[key]()
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

    def __init__(self, vm: Optional[TitanVM] = None):
        """
        Args:
            vm: TitanVM instance. If None, creates a lightweight one
                without state_register (programs use context only).
        """
        self.vm = vm or TitanVM()
        self.programs = load_nervous_system_programs()
        self._last_signals: list[dict] = []
        self._total_evaluations: int = 0

    def evaluate(self, observables: dict[str, dict]) -> list[dict]:
        """
        Run all nervous system programs against current observables.

        Args:
            observables: Dict from ObservableEngine ({part: {5 metrics}}).

        Returns:
            List of fired signals [{system, urgency, result}].
            Only includes programs that scored > 0.
        """
        context = flatten_observables(observables)
        signals = []

        for name, program in self.programs.items():
            try:
                result = self.vm.execute(program, context=context)
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
