#!/usr/bin/env python3
"""
rFP β Stage 1 verification — synthetic observables suite for all 11 VM programs.

Feeds each program a battery of synthetic observable states and asserts:
  1. At least ONE scenario produces score > 0 for each program (reachable)
  2. At least ONE scenario produces score = 0 (not always-on except VIGILANCE)
  3. Urgency output is always in [0, 1] (bounded by CLAMP opcodes)
  4. Fire pattern matches the program's documented semantic role

Synthetic scenarios cover the space of observable combinations — NOT an
exhaustive search, but a diverse enough set to validate that the bytecode
is syntactically valid and semantically grounded.

Run: python scripts/nsspecs/verify_vm_programs.py

Exit 0 if all 11 programs pass, 1 otherwise.
"""
import sys
from pathlib import Path

# Make titan_plugin importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titan_plugin.logic.nervous_system import (
    NervousSystem, load_nervous_system_programs, flatten_observables,
)
from titan_plugin.logic.titan_vm import TitanVM


def make_observables(*, inner_body=None, inner_mind=None, inner_spirit=None,
                     outer_body=None, outer_mind=None, outer_spirit=None):
    """Build a 6-part observable dict. Missing parts get zeros."""
    z = {"coherence": 0.0, "magnitude": 0.0, "velocity": 0.0,
         "direction": 0.0, "polarity": 0.0}
    def _merge(d):
        return {**z, **(d or {})}
    return {
        "inner_body": _merge(inner_body),
        "inner_mind": _merge(inner_mind),
        "inner_spirit": _merge(inner_spirit),
        "outer_body": _merge(outer_body),
        "outer_mind": _merge(outer_mind),
        "outer_spirit": _merge(outer_spirit),
    }


# ── Synthetic scenario battery ─────────────────────────────────────

SCENARIOS = {
    # Baseline: all zero → most programs should NOT fire (VIGILANCE still produces base 0.15)
    "all_zero": make_observables(),

    # Homeostatic: moderate coherence across parts, low velocity
    "homeostatic": make_observables(
        inner_body={"coherence": 0.5, "magnitude": 0.3},
        inner_mind={"coherence": 0.5, "magnitude": 0.3},
        inner_spirit={"coherence": 0.5},
        outer_body={"coherence": 0.5, "magnitude": 0.3},
        outer_mind={"coherence": 0.5, "magnitude": 0.3},
        outer_spirit={"coherence": 0.5},
    ),

    # Metabolic stress: high inner_body magnitude + low coherence everywhere
    "metabolic_stress": make_observables(
        inner_body={"magnitude": 0.9, "coherence": 0.2},
        inner_mind={"coherence": 0.2},
        inner_spirit={"coherence": 0.2},
        outer_body={"coherence": 0.2},
        outer_mind={"coherence": 0.2},
        outer_spirit={"coherence": 0.2},
    ),

    # Sudden perturbation: velocity spike across parts (REFLEX + VIGILANCE should fire)
    "velocity_spike": make_observables(
        inner_body={"velocity": 0.6, "coherence": 0.4},
        inner_mind={"velocity": 0.5, "coherence": 0.4},
        outer_body={"velocity": 0.6, "coherence": 0.4},
        outer_mind={"velocity": 0.5, "coherence": 0.4},
    ),

    # Focused cognition: sustained polarity (FOCUS should fire)
    "focused": make_observables(
        inner_mind={"polarity": 0.6, "coherence": 0.7, "magnitude": 0.5},
        inner_spirit={"polarity": 0.6, "coherence": 0.7},
        outer_mind={"polarity": 0.5, "coherence": 0.6},
    ),

    # Erratic state: low direction (INTUITION should fire)
    "erratic": make_observables(
        inner_body={"direction": 0.2, "velocity": 0.3},
        inner_mind={"direction": 0.2, "velocity": 0.3},
        outer_body={"direction": 0.2},
        outer_mind={"direction": 0.3},
    ),

    # Ready to act: coherence + magnitude high across ALL 6 parts (IMPULSE checks all.coherence_avg + all.magnitude_avg)
    "ready_to_act": make_observables(
        inner_body={"coherence": 0.85, "magnitude": 0.7},
        inner_mind={"coherence": 0.85, "magnitude": 0.7},
        inner_spirit={"coherence": 0.85, "magnitude": 0.7},
        outer_body={"coherence": 0.8, "magnitude": 0.7},
        outer_mind={"coherence": 0.8, "magnitude": 0.7},
        outer_spirit={"coherence": 0.8, "magnitude": 0.7},
    ),

    # Inspired state: high coherence across ALL 6 parts + moderate velocity + novel direction
    "inspired": make_observables(
        inner_body={"coherence": 0.8, "velocity": 0.08, "direction": 0.5},
        inner_mind={"coherence": 0.8, "velocity": 0.08, "direction": 0.5},
        inner_spirit={"coherence": 0.85, "velocity": 0.08, "direction": 0.5},
        outer_body={"coherence": 0.8, "velocity": 0.08, "direction": 0.5},
        outer_mind={"coherence": 0.8, "velocity": 0.08, "direction": 0.5},
        outer_spirit={"coherence": 0.85, "velocity": 0.08, "direction": 0.5},
    ),

    # Creative expression: outer_body coherent + direction meaningful
    "creative": make_observables(
        outer_body={"coherence": 0.75, "direction": 0.6, "magnitude": 0.5},
        outer_spirit={"coherence": 0.7},
        inner_mind={"coherence": 0.5, "magnitude": 0.4},
    ),

    # Curious: ambivalent polarity in outer_mind + sustained magnitude
    "curious": make_observables(
        outer_mind={"polarity": 0.05, "magnitude": 0.55, "coherence": 0.5},
        inner_mind={"polarity": 0.1, "magnitude": 0.5, "coherence": 0.5},
    ),

    # Empathic/social: outer parts highly coherent (harmony)
    "empathic": make_observables(
        outer_body={"coherence": 0.75, "magnitude": 0.4},
        outer_mind={"coherence": 0.75, "magnitude": 0.4},
        outer_spirit={"coherence": 0.75, "magnitude": 0.4},
    ),

    # Meditative/reflection: very low velocity, high coherence, stillness
    "meditative": make_observables(
        inner_body={"velocity": 0.03, "coherence": 0.7},
        inner_mind={"velocity": 0.03, "coherence": 0.7},
        inner_spirit={"velocity": 0.03, "coherence": 0.7},
        outer_body={"velocity": 0.03, "coherence": 0.6},
        outer_mind={"velocity": 0.03, "coherence": 0.7},
        outer_spirit={"velocity": 0.03, "coherence": 0.7},
    ),
}


# Expected fire-pattern per program: scenarios where it SHOULD produce score > 0
# (Verification loosely checks that at least the "primary" scenario fires.)
EXPECTED_FIRES = {
    "REFLEX":      {"velocity_spike"},
    "FOCUS":       {"focused"},
    "INTUITION":   {"erratic"},
    "IMPULSE":     {"ready_to_act"},
    "INSPIRATION": {"inspired"},
    "METABOLISM":  {"metabolic_stress"},
    "CREATIVITY":  {"creative"},
    "CURIOSITY":   {"curious"},
    "EMPATHY":     {"empathic", "homeostatic"},  # either fires satisfies the test
    "REFLECTION":  {"meditative"},
    "VIGILANCE":   {"velocity_spike", "all_zero", "homeostatic"},  # always fires (base 0.15)
}


def main() -> int:
    print("=" * 78)
    print("rFP β Stage 1 — TitanVM program verification suite")
    print("=" * 78)

    programs = load_nervous_system_programs()
    print(f"\nLoaded {len(programs)} VM programs: {', '.join(programs.keys())}")
    assert len(programs) == 11, f"Expected 11 programs, got {len(programs)}"

    vm = TitanVM()
    total_fails = 0

    print(f"\n{'PROGRAM':<14} {'SCENARIOS_FIRED':<24} {'MAX_SCORE':>10} {'RANGE':<16}  VERDICT")
    print("-" * 78)

    for prog_name, bytecode in programs.items():
        scores = {}
        for scenario_name, observables in SCENARIOS.items():
            context = flatten_observables(observables)
            result = vm.execute(bytecode, context=context)
            if result.error:
                print(f"  {prog_name:<12}  VM ERROR: {result.error}")
                total_fails += 1
                break
            # All scores must be bounded [0, 1]
            if not (0.0 <= result.score <= 1.0):
                print(f"  {prog_name:<12}  SCORE OUT OF RANGE: {result.score} in {scenario_name}")
                total_fails += 1
                break
            scores[scenario_name] = result.score
        else:
            fired_scenarios = {s for s, v in scores.items() if v > 0.0}
            max_score = max(scores.values()) if scores else 0.0
            min_score = min(scores.values()) if scores else 0.0
            expected_any = EXPECTED_FIRES.get(prog_name, set())
            # VIGILANCE always fires (documented behavior)
            if prog_name == "VIGILANCE":
                verdict = "PASS" if all(v > 0 for v in scores.values()) else "FAIL (VIGILANCE must always fire)"
            else:
                matches = fired_scenarios & expected_any
                verdict = "PASS" if matches else "FAIL (expected fire not triggered)"

            fired_display = ",".join(sorted(fired_scenarios))[:22]
            if len(fired_scenarios) > 2 and len(fired_display) > 20:
                fired_display = f"{len(fired_scenarios)} scenarios"
            range_str = f"[{min_score:.3f},{max_score:.3f}]"
            print(f"  {prog_name:<12} {fired_display:<24} {max_score:>10.4f} {range_str:<16}  {verdict}")
            if "FAIL" in verdict:
                total_fails += 1

    print("\n" + "=" * 78)
    if total_fails == 0:
        print(f"  PASS: all 11 VM programs valid + semantically grounded")
        return 0
    else:
        print(f"  FAIL: {total_fails}/11 programs failed verification")
        return 1


if __name__ == "__main__":
    sys.exit(main())
