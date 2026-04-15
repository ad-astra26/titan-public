#!/usr/bin/env python3
"""
ARC Reasoning Baseline Test for Titan V4.

Tests Titan's autonomous reasoning capabilities across 4 categories:
  A. Pattern Recognition (3 puzzles)
  B. Spatial Reasoning (3 puzzles)
  C. Rule Deduction (3 puzzles)
  D. Multi-Step Planning (3 puzzles)

For each puzzle, measures BOTH response quality AND autonomous processing:
  - Correctness: did Titan get the right answer?
  - Reasoning depth: did the response show step-by-step thinking?
  - Topology delta: how did inner space change during reasoning?
  - Nervous activity: which NervousSystem programs fired?
  - Fatigue delta: how much cognitive load did the puzzle impose?

Establishes V4 baseline for comparison after V5 Neural NervousSystem upgrade.
"""
import asyncio
import json
import logging
import os
import re
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("arc_baseline")

API_URL = "http://localhost:7777/chat"
INNER_TRINITY_URL = "http://localhost:7777/v4/inner-trinity"
V4_STATE_URL = "http://localhost:7777/v4/state"
REFLEX_URL = "http://localhost:7777/v4/reflexes"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

GAP_SECONDS = 15  # Gap between puzzles for bus processing
SESSION_ID = "arc_baseline_v4"

# ── Puzzle Definitions ─────────────────────────────────────────────

PUZZLES = [
    # ═══════════════════════════════════════════════════════════════
    # Category A: Pattern Recognition
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "A1_grid_pattern",
        "category": "pattern_recognition",
        "difficulty": "easy",
        "message": (
            "I have a 4x4 grid of colors. Each row follows a repeating pattern:\n\n"
            "Row 1: RED, BLUE, RED, BLUE\n"
            "Row 2: GREEN, YELLOW, GREEN, YELLOW\n"
            "Row 3: RED, BLUE, RED, ?\n"
            "Row 4: GREEN, YELLOW, ?, YELLOW\n\n"
            "What color goes in the ? positions? Explain your reasoning step by step."
        ),
        "expected_answer": ["blue", "green"],
        "check_fn": "check_contains_all",
    },
    {
        "id": "A2_number_sequence",
        "category": "pattern_recognition",
        "difficulty": "medium",
        "message": (
            "Look at this sequence of transformations:\n\n"
            "Input: [1, 2, 3] → Output: [2, 4, 6]\n"
            "Input: [5, 10, 15] → Output: [10, 20, 30]\n"
            "Input: [3, 7, 11] → Output: ?\n\n"
            "What is the output? Show your reasoning."
        ),
        "expected_answer": ["6", "14", "22"],
        "check_fn": "check_contains_all",
    },
    {
        "id": "A3_symmetry",
        "category": "pattern_recognition",
        "difficulty": "hard",
        "message": (
            "This 5x5 grid has vertical mirror symmetry (left side mirrors right side).\n"
            "Some cells are filled, some are missing (marked ?):\n\n"
            "```\n"
            "1  0  1  0  1\n"
            "0  1  0  1  0\n"
            "1  1  ?  ?  ?\n"
            "0  1  0  ?  ?\n"
            "1  0  1  0  ?\n"
            "```\n\n"
            "Fill in the ? cells using the vertical mirror symmetry rule. "
            "Show the complete grid and explain the symmetry pattern."
        ),
        "expected_answer": ["1  1  1", "0  1  0", "1"],
        "check_fn": "check_contains_any",
    },

    # ═══════════════════════════════════════════════════════════════
    # Category B: Spatial Reasoning
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "B1_grid_navigation",
        "category": "spatial_reasoning",
        "difficulty": "medium",
        "message": (
            "Navigate a 5x5 grid from S (start) to E (end). "
            "X marks impassable walls. You can move UP, DOWN, LEFT, RIGHT (not diagonal).\n\n"
            "```\n"
            "S  .  X  .  .\n"
            ".  X  X  .  .\n"
            ".  .  .  .  X\n"
            "X  X  .  X  .\n"
            ".  .  .  .  E\n"
            "```\n\n"
            "Find the shortest path from S to E. List each move step by step. "
            "How many moves does the shortest path take?"
        ),
        "expected_answer": ["8"],
        "check_fn": "check_contains_any",
    },
    {
        "id": "B2_rotation",
        "category": "spatial_reasoning",
        "difficulty": "medium",
        "message": (
            "Here is a 3x3 shape made of 1s and 0s:\n\n"
            "```\n"
            "1  1  0\n"
            "0  1  0\n"
            "0  1  1\n"
            "```\n\n"
            "If I rotate this shape 90 degrees clockwise, what does it look like? "
            "Show the resulting 3x3 grid and explain how rotation works."
        ),
        "expected_answer": ["0  0  1", "1  1  1", "1  0  0"],
        "check_fn": "check_contains_any",
    },
    {
        "id": "B3_counting",
        "category": "spatial_reasoning",
        "difficulty": "easy",
        "message": (
            "Look at this grid. Count the distinct connected regions of '#' characters. "
            "Two '#' cells are connected if they share an edge (not diagonal).\n\n"
            "```\n"
            ".  #  #  .  .\n"
            ".  #  .  .  .\n"
            ".  .  .  #  #\n"
            "#  .  .  #  .\n"
            "#  #  .  .  .\n"
            "```\n\n"
            "How many separate connected regions of '#' are there? List each region."
        ),
        "expected_answer": ["3"],
        "check_fn": "check_contains_any",
    },

    # ═══════════════════════════════════════════════════════════════
    # Category C: Rule Deduction
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "C1_transformation_rule",
        "category": "rule_deduction",
        "difficulty": "medium",
        "message": (
            "Study these input→output pairs and deduce the transformation rule:\n\n"
            "Example 1:\n"
            "  Input:  [RED, BLUE, GREEN]\n"
            "  Output: [GREEN, BLUE, RED]\n\n"
            "Example 2:\n"
            "  Input:  [A, B, C, D]\n"
            "  Output: [D, C, B, A]\n\n"
            "Example 3:\n"
            "  Input:  [1, 2, 3, 4, 5]\n"
            "  Output: [5, 4, 3, 2, 1]\n\n"
            "Test: What is the output for input [SUN, MOON, STAR, CLOUD]?\n"
            "State the rule and apply it."
        ),
        "expected_answer": ["cloud", "star", "moon", "sun"],
        "check_fn": "check_contains_all_ci",
    },
    {
        "id": "C2_arc_grid_transform",
        "category": "rule_deduction",
        "difficulty": "hard",
        "message": (
            "Study these grid transformations and find the rule:\n\n"
            "Training Pair 1:\n"
            "  Input:     Output:\n"
            "  0 0 0      0 0 0\n"
            "  0 1 0  →   1 1 1\n"
            "  0 0 0      0 0 0\n\n"
            "Training Pair 2:\n"
            "  Input:     Output:\n"
            "  0 0 0      0 1 0\n"
            "  0 0 0  →   0 0 0\n"
            "  0 1 0      0 1 0\n\n"
            "Test Input:\n"
            "  0 1 0\n"
            "  0 0 0\n"
            "  0 0 0\n\n"
            "What is the test output? Explain the rule you found."
        ),
        "expected_answer": ["1", "row", "column", "cross", "expand"],
        "check_fn": "check_contains_any",
    },
    {
        "id": "C3_category_rule",
        "category": "rule_deduction",
        "difficulty": "medium",
        "message": (
            "Two groups are defined by a hidden rule:\n\n"
            "Group A: 4, 16, 36, 64, 100\n"
            "Group B: 3, 15, 35, 63, 99\n\n"
            "What is the rule that separates Group A from Group B?\n"
            "Which group does the number 81 belong to? What about 80?"
        ),
        "expected_answer": ["square", "perfect", "81", "a", "80", "b"],
        "check_fn": "check_contains_any",
    },

    # ═══════════════════════════════════════════════════════════════
    # Category D: Multi-Step Planning
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "D1_dependency_ordering",
        "category": "multi_step_planning",
        "difficulty": "medium",
        "message": (
            "You have 6 tasks with dependencies (must complete prerequisites first):\n\n"
            "Task A: no dependencies\n"
            "Task B: requires A\n"
            "Task C: requires A\n"
            "Task D: requires B and C\n"
            "Task E: requires C\n"
            "Task F: requires D and E\n\n"
            "What is a valid execution order? "
            "What is the minimum number of sequential steps if you can run "
            "independent tasks in parallel? Show the parallel schedule."
        ),
        "expected_answer": ["4", "a", "b", "c", "d", "e", "f"],
        "check_fn": "check_contains_any",
    },
    {
        "id": "D2_state_machine",
        "category": "multi_step_planning",
        "difficulty": "hard",
        "message": (
            "A traffic light follows these rules:\n\n"
            "- GREEN → YELLOW (after 30 seconds)\n"
            "- YELLOW → RED (after 5 seconds)\n"
            "- RED → GREEN (after 25 seconds)\n\n"
            "The light starts at GREEN at time t=0.\n\n"
            "Questions:\n"
            "1. What color is the light at t=40 seconds?\n"
            "2. What color is the light at t=62 seconds?\n"
            "3. How many complete GREEN→YELLOW→RED→GREEN cycles "
            "occur in 5 minutes (300 seconds)?"
        ),
        "expected_answer": ["red", "green", "5"],
        "check_fn": "check_contains_any",
    },
    {
        "id": "D3_resource_allocation",
        "category": "multi_step_planning",
        "difficulty": "hard",
        "message": (
            "You have 100 units of energy to distribute among 3 workers:\n\n"
            "Worker A: produces 2 output per energy unit\n"
            "Worker B: produces 3 output per energy unit (but max 30 energy)\n"
            "Worker C: produces 1 output per energy unit (but guarantees quality)\n\n"
            "Constraints:\n"
            "- Each worker must receive at least 10 energy\n"
            "- Worker B cannot receive more than 30 energy\n"
            "- You want to maximize total output\n\n"
            "How should you distribute the 100 energy units? "
            "What is the maximum total output? Show your calculation."
        ),
        "expected_answer": ["30", "210", "output"],
        "check_fn": "check_contains_any",
    },
]


# ── Answer Checking Functions ──────────────────────────────────────

def check_contains_all(response: str, expected: list[str]) -> tuple[float, str]:
    """All expected strings must appear in response (case-insensitive)."""
    resp_lower = response.lower()
    found = [e for e in expected if e.lower() in resp_lower]
    missing = [e for e in expected if e.lower() not in resp_lower]
    score = len(found) / len(expected)
    detail = f"found {len(found)}/{len(expected)}"
    if missing:
        detail += f" (missing: {missing})"
    return score, detail


def check_contains_any(response: str, expected: list[str]) -> tuple[float, str]:
    """At least some expected strings should appear (partial credit)."""
    resp_lower = response.lower()
    found = [e for e in expected if e.lower() in resp_lower]
    score = min(1.0, len(found) / max(1, len(expected) * 0.5))
    return score, f"found {len(found)}/{len(expected)} keywords"


def check_contains_all_ci(response: str, expected: list[str]) -> tuple[float, str]:
    """All expected strings must appear (case-insensitive, order matters for sequences)."""
    resp_lower = response.lower()
    found = [e for e in expected if e.lower() in resp_lower]
    score = len(found) / len(expected)
    return score, f"found {len(found)}/{len(expected)}"


CHECK_FNS = {
    "check_contains_all": check_contains_all,
    "check_contains_any": check_contains_any,
    "check_contains_all_ci": check_contains_all_ci,
}


# ── Reasoning Depth Analysis ──────────────────────────────────────

def measure_reasoning_depth(response: str) -> tuple[float, dict]:
    """Measure how much step-by-step reasoning the response shows."""
    indicators = {
        "step_markers": len(re.findall(
            r"step \d|first|second|third|next|then|finally|therefore|because|since",
            response.lower())),
        "numbered_lists": len(re.findall(r"^\s*\d+[\.\)]\s", response, re.MULTILINE)),
        "logical_connectors": len(re.findall(
            r"if .* then|because|therefore|so |thus |hence|implies|means that",
            response.lower())),
        "examples_given": len(re.findall(
            r"for example|for instance|e\.g\.|such as|like ",
            response.lower())),
        "math_shown": len(re.findall(
            r"[=×÷\+\-\*]|\d+ [\+\-\*\/] \d+|equals|times|plus|minus|multiply",
            response.lower())),
        "self_verification": len(re.findall(
            r"let me (check|verify)|double.check|to confirm|checking|we can see",
            response.lower())),
    }

    # Score: 0-1 based on presence of reasoning indicators
    raw_score = (
        min(3, indicators["step_markers"]) / 3 * 0.25 +
        min(3, indicators["numbered_lists"]) / 3 * 0.20 +
        min(2, indicators["logical_connectors"]) / 2 * 0.20 +
        min(1, indicators["math_shown"]) * 0.15 +
        min(1, indicators["self_verification"]) * 0.10 +
        min(1, indicators["examples_given"]) * 0.10
    )

    return min(1.0, raw_score), indicators


# ── API Helpers ────────────────────────────────────────────────────

async def send_puzzle(client: httpx.AsyncClient, puzzle: dict) -> tuple[str, float]:
    """Send a puzzle and return (response_text, response_time_seconds)."""
    t0 = time.time()
    resp = await client.post(
        API_URL,
        json={
            "message": puzzle["message"],
            "user_id": "arc_tester",
            "session_id": SESSION_ID,
        },
        headers={
            "Content-Type": "application/json",
            "X-Titan-Internal-Key": INTERNAL_KEY,
        },
        timeout=120.0,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    if "response" in data:
        text = data["response"]
    elif "data" in data and isinstance(data["data"], dict):
        text = data["data"].get("response", str(data)[:500])
    elif "data" in data and isinstance(data["data"], str):
        text = data["data"]
    else:
        text = str(data)[:500]

    return text, elapsed


async def get_inner_trinity(client: httpx.AsyncClient) -> dict:
    """Fetch /v4/inner-trinity state."""
    try:
        resp = await client.get(INNER_TRINITY_URL, timeout=10.0)
        return resp.json().get("data", {})
    except Exception:
        return {}


async def get_v4_state(client: httpx.AsyncClient) -> dict:
    """Fetch /v4/state."""
    try:
        resp = await client.get(V4_STATE_URL, timeout=10.0)
        return resp.json().get("data", {})
    except Exception:
        return {}


async def get_reflex_state(client: httpx.AsyncClient) -> dict:
    """Fetch /v4/reflexes."""
    try:
        resp = await client.get(REFLEX_URL, timeout=10.0)
        return resp.json().get("data", {})
    except Exception:
        return {}


# ── Autonomous Processing Metrics ─────────────────────────────────

def compute_topology_delta(before: dict, after: dict) -> dict:
    """Compute topology changes between two inner-trinity snapshots."""
    topo_before = before.get("topology", {})
    topo_after = after.get("topology", {})

    vol_before = topo_before.get("volume", 0.0)
    vol_after = topo_after.get("volume", 0.0)
    curv_before = topo_before.get("curvature", 0.0)
    curv_after = topo_after.get("curvature", 0.0)

    return {
        "volume_before": round(vol_before, 4),
        "volume_after": round(vol_after, 4),
        "volume_delta": round(vol_after - vol_before, 4),
        "volume_direction": "expanding" if vol_after > vol_before else "contracting",
        "curvature_before": round(curv_before, 4),
        "curvature_after": round(curv_after, 4),
        "curvature_delta": round(curv_after - curv_before, 4),
        "clusters_before": topo_before.get("clusters", []),
        "clusters_after": topo_after.get("clusters", []),
        "isolated_before": topo_before.get("isolated", []),
        "isolated_after": topo_after.get("isolated", []),
    }


def compute_dreaming_delta(before: dict, after: dict) -> dict:
    """Compute dreaming/fatigue changes."""
    dream_before = before.get("dreaming", {})
    dream_after = after.get("dreaming", {})

    fat_before = dream_before.get("fatigue", 0.0)
    fat_after = dream_after.get("fatigue", 0.0)

    return {
        "fatigue_before": round(fat_before, 4),
        "fatigue_after": round(fat_after, 4),
        "fatigue_delta": round(fat_after - fat_before, 4),
        "is_dreaming": dream_after.get("is_dreaming", False) if "is_dreaming" in after else False,
        "experience_buffer": dream_after.get("experience_buffer_size", 0),
    }


def extract_nervous_signals(trinity_state: dict) -> list[dict]:
    """Extract nervous system signals from inner-trinity state."""
    return trinity_state.get("nervous_signals", [])


# ── Main Test Runner ───────────────────────────────────────────────

async def run_arc_baseline():
    """Run the full ARC reasoning baseline test."""
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  ARC REASONING BASELINE TEST — Titan V4                    ║")
    log.info("║  Testing: Pattern, Spatial, Rules, Planning (12 puzzles)   ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")

    results = []
    category_scores = {}

    async with httpx.AsyncClient() as client:
        # Initial state snapshot
        initial_trinity = await get_inner_trinity(client)
        initial_v4 = await get_v4_state(client)
        initial_reflexes = await get_reflex_state(client)

        log.info("Initial state captured:")
        log.info("  Topology volume: %.4f",
                 initial_trinity.get("topology", {}).get("volume", 0.0))
        log.info("  Fatigue: %.4f",
                 initial_trinity.get("dreaming", {}).get("fatigue", 0.0))
        log.info("  Tick count: %d", initial_trinity.get("tick_count", 0))
        log.info("")

        current_category = ""

        for i, puzzle in enumerate(PUZZLES):
            cat = puzzle["category"]
            if cat != current_category:
                current_category = cat
                cat_display = cat.replace("_", " ").upper()
                log.info("═" * 62)
                log.info("  Category: %s", cat_display)
                log.info("═" * 62)
                log.info("")

            log.info("─── Puzzle %d/12: %s [%s] ───",
                     i + 1, puzzle["id"], puzzle["difficulty"])

            # 1. Snapshot BEFORE
            trinity_before = await get_inner_trinity(client)

            # 2. Send puzzle
            log.info(">>> %s", puzzle["message"][:100].replace("\n", " ") + "...")
            try:
                response, elapsed = await send_puzzle(client, puzzle)
                error = None
            except Exception as e:
                response = ""
                elapsed = 0.0
                error = str(e)
                log.error("    ERROR: %s", error)

            if response:
                # Show first 200 chars of response
                preview = response[:200].replace("\n", " ")
                log.info("<<< %s...", preview)
                log.info("    Response time: %.1fs, length: %d chars", elapsed, len(response))

            # 3. Wait for bus processing
            await asyncio.sleep(GAP_SECONDS)

            # 4. Snapshot AFTER
            trinity_after = await get_inner_trinity(client)

            # 5. Compute metrics
            # Correctness
            check_fn = CHECK_FNS.get(puzzle["check_fn"], check_contains_any)
            correctness, correctness_detail = check_fn(response, puzzle["expected_answer"])

            # Reasoning depth
            reasoning_score, reasoning_indicators = measure_reasoning_depth(response)

            # Topology delta
            topo_delta = compute_topology_delta(trinity_before, trinity_after)

            # Dreaming delta
            dream_delta = compute_dreaming_delta(trinity_before, trinity_after)

            # Nervous signals (after processing)
            nervous_signals = extract_nervous_signals(trinity_after)

            # Log metrics
            log.info("    Correctness:  %.0f%% (%s)", correctness * 100, correctness_detail)
            log.info("    Reasoning:    %.0f%% (steps=%d, logic=%d, math=%d, verify=%d)",
                     reasoning_score * 100,
                     reasoning_indicators.get("step_markers", 0),
                     reasoning_indicators.get("logical_connectors", 0),
                     reasoning_indicators.get("math_shown", 0),
                     reasoning_indicators.get("self_verification", 0))
            log.info("    Topology:     vol=%.4f→%.4f (%s, Δ=%+.4f)",
                     topo_delta["volume_before"], topo_delta["volume_after"],
                     topo_delta["volume_direction"], topo_delta["volume_delta"])
            log.info("    Curvature:    %.4f→%.4f (Δ=%+.4f)",
                     topo_delta["curvature_before"], topo_delta["curvature_after"],
                     topo_delta["curvature_delta"])
            log.info("    Fatigue:      %.4f→%.4f (Δ=%+.4f)",
                     dream_delta["fatigue_before"], dream_delta["fatigue_after"],
                     dream_delta["fatigue_delta"])
            if nervous_signals:
                for sig in nervous_signals:
                    log.info("    Nervous:      %s urgency=%.3f",
                             sig.get("system", "?"), sig.get("urgency", 0))
            else:
                log.info("    Nervous:      (no signals)")
            log.info("")

            # Store result
            result = {
                "puzzle_id": puzzle["id"],
                "category": puzzle["category"],
                "difficulty": puzzle["difficulty"],
                "response": response,
                "response_time_s": round(elapsed, 2),
                "response_length": len(response),
                "error": error,
                "correctness": round(correctness, 3),
                "correctness_detail": correctness_detail,
                "reasoning_score": round(reasoning_score, 3),
                "reasoning_indicators": reasoning_indicators,
                "topology_delta": topo_delta,
                "dreaming_delta": dream_delta,
                "nervous_signals": nervous_signals,
                "ts": time.time(),
            }
            results.append(result)

            # Track category scores
            if cat not in category_scores:
                category_scores[cat] = {"correctness": [], "reasoning": [], "count": 0}
            category_scores[cat]["correctness"].append(correctness)
            category_scores[cat]["reasoning"].append(reasoning_score)
            category_scores[cat]["count"] += 1

        # ── Final state snapshot ──────────────────────────────────
        final_trinity = await get_inner_trinity(client)
        final_v4 = await get_v4_state(client)
        final_reflexes = await get_reflex_state(client)

    # ── Compute Summary ───────────────────────────────────────────
    total_correct = sum(r["correctness"] for r in results) / len(results)
    total_reasoning = sum(r["reasoning_score"] for r in results) / len(results)
    total_topology_delta = sum(abs(r["topology_delta"]["volume_delta"]) for r in results)
    total_fatigue_delta = (
        final_trinity.get("dreaming", {}).get("fatigue", 0.0) -
        initial_trinity.get("dreaming", {}).get("fatigue", 0.0)
    )

    # Sphere clock pulses during test
    def count_pulses(state):
        clocks = state.get("sphere_clock", {}).get("clocks", {})
        return sum(c.get("pulse_count", 0) for c in clocks.values())

    pulses_before = count_pulses(initial_v4)
    pulses_after = count_pulses(final_v4)

    # Consciousness change
    cons_before = initial_v4.get("consciousness", {})
    cons_after = final_v4.get("consciousness", {})

    # ── Print Report ──────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  ARC REASONING BASELINE — RESULTS                          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("  Total puzzles:        %d", len(results))
    log.info("  Errors:               %d", sum(1 for r in results if r["error"]))
    log.info("")
    log.info("  ── Response Quality ──────────────────────────────────────")
    log.info("  Overall correctness:  %.1f%%", total_correct * 100)
    log.info("  Overall reasoning:    %.1f%%", total_reasoning * 100)
    log.info("")

    for cat, scores in category_scores.items():
        cat_display = cat.replace("_", " ").title()
        avg_correct = sum(scores["correctness"]) / scores["count"]
        avg_reason = sum(scores["reasoning"]) / scores["count"]
        log.info("  %-22s correct=%.0f%%  reasoning=%.0f%%",
                 cat_display, avg_correct * 100, avg_reason * 100)

    log.info("")
    log.info("  ── Autonomous Processing ─────────────────────────────────")
    log.info("  Topology movement:    %.4f (total absolute volume change)", total_topology_delta)
    log.info("  Fatigue accumulated:  %.4f (Δ from start to end)", total_fatigue_delta)
    log.info("  Sphere pulses:        %d → %d (Δ=%d)",
             pulses_before, pulses_after, pulses_after - pulses_before)
    log.info("  Coordinator ticks:    %d → %d",
             initial_trinity.get("tick_count", 0),
             final_trinity.get("tick_count", 0))
    log.info("  Consciousness drift:  %.4f → %.4f",
             cons_before.get("drift_magnitude", 0.0),
             cons_after.get("drift_magnitude", 0.0))
    log.info("  Is dreaming:          %s",
             final_trinity.get("dreaming", {}).get("is_dreaming", False)
             if "dreaming" in final_trinity else "N/A")

    # State register
    sr = final_reflexes.get("state_register", {})
    if sr:
        log.info("  Trinity avgs:         body=%.3f mind=%.3f spirit=%.3f",
                 sr.get("body_avg", 0), sr.get("mind_avg", 0), sr.get("spirit_avg", 0))

    log.info("")
    log.info("  ╔════════════════════════════════════════════════════════╗")
    log.info("  ║  COMPOSITE SCORE:  %.1f%% correct  |  %.1f%% reasoning  ║",
             total_correct * 100, total_reasoning * 100)
    log.info("  ╚════════════════════════════════════════════════════════╝")

    # ── Save JSON Report ──────────────────────────────────────────
    report = {
        "test": "arc_reasoning_baseline_v4",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v4_pre_neural_nervous_system",
        "total_puzzles": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "overall_correctness": round(total_correct, 4),
        "overall_reasoning": round(total_reasoning, 4),
        "category_scores": {
            cat: {
                "avg_correctness": round(sum(s["correctness"]) / s["count"], 4),
                "avg_reasoning": round(sum(s["reasoning"]) / s["count"], 4),
                "count": s["count"],
            }
            for cat, s in category_scores.items()
        },
        "autonomous_metrics": {
            "topology_total_movement": round(total_topology_delta, 4),
            "fatigue_delta": round(total_fatigue_delta, 4),
            "sphere_pulses_delta": pulses_after - pulses_before,
            "coordinator_ticks_delta": (
                final_trinity.get("tick_count", 0) -
                initial_trinity.get("tick_count", 0)
            ),
            "consciousness_drift_before": cons_before.get("drift_magnitude", 0.0),
            "consciousness_drift_after": cons_after.get("drift_magnitude", 0.0),
            "is_dreaming_at_end": final_trinity.get("dreaming", {}).get("is_dreaming", False)
                                  if "dreaming" in final_trinity else False,
        },
        "initial_state": {
            "topology_volume": initial_trinity.get("topology", {}).get("volume", 0.0),
            "fatigue": initial_trinity.get("dreaming", {}).get("fatigue", 0.0),
            "tick_count": initial_trinity.get("tick_count", 0),
        },
        "final_state": {
            "topology_volume": final_trinity.get("topology", {}).get("volume", 0.0),
            "fatigue": final_trinity.get("dreaming", {}).get("fatigue", 0.0),
            "tick_count": final_trinity.get("tick_count", 0),
        },
        "puzzles": results,
    }

    report_dir = os.path.join(os.path.dirname(__file__), "..", "data", "endurance_reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = int(time.time())
    report_path = os.path.join(report_dir, f"arc_baseline_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("")
    log.info("  Report: %s", report_path)
    log.info("")
    log.info("=" * 62)
    log.info("ARC BASELINE TEST COMPLETE")
    log.info("=" * 62)


if __name__ == "__main__":
    asyncio.run(run_arc_baseline())
