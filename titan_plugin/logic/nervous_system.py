"""
titan_plugin/logic/nervous_system.py — Nervous System Dispatcher (T4).

Dispatches 5 TitanVM micro-programs that operate on observable state:
  1. REFLEX   — fires when velocity spikes (sudden perturbation)
  2. FOCUS    — fires when polarity sustained beyond threshold
  3. INTUITION — fires when direction drops (erratic movement)
  4. IMPULSE  — fires when magnitude + coherence both high (ready to act)
  5. INSPIRATION — fires when coherence high + velocity positive + direction novel

Each program reads from context (flattened observables dict), outputs a
score via SCORE opcode. Score > 0 means the signal fires with that urgency.

The NervousSystem is wired into InnerTrinityCoordinator (T3) and runs
on every tick cycle.
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


def load_nervous_system_programs() -> dict[str, list]:
    """Load all 5 nervous system programs."""
    return {
        "REFLEX": _make_reflex_program(),
        "FOCUS": _make_focus_program(),
        "INTUITION": _make_intuition_program(),
        "IMPULSE": _make_impulse_program(),
        "INSPIRATION": _make_inspiration_program(),
    }


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
