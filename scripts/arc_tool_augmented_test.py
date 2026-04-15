#!/usr/bin/env python3
"""
ARC Tool-Augmented Reasoning Test for Titan V5.

Tests Titan's ability to autonomously decide WHEN to use external tools
vs pure reasoning. Puzzles are designed to naturally trigger Agency helpers:

  Category T1: Research Reasoning    → should trigger code_knowledge / web_search
  Category T2: Code Verification     → should trigger coding_sandbox
  Category T3: Creative Synthesis    → should trigger art_generate / audio_generate
  Category T4: System Introspection  → should trigger infra_inspect
  Category T5: Pure Reasoning (control) → should NOT trigger any helper

For each puzzle, measures:
  - Response quality (correctness + reasoning depth)
  - Agency activation (did helpers fire? which ones?)
  - NervousSystem signals (which programs fired? urgency levels?)
  - Topology dynamics (expansion/contraction during processing)
  - Tool appropriateness (did Titan use tools when it SHOULD have?)

Establishes baseline for tool-use learning via IMPULSE/INTUITION programs.
"""
import asyncio
import json
import logging
import os
import re
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("arc_tool_aug")

BASE_URL = "http://localhost:7777"
CHAT_URL = f"{BASE_URL}/chat"
TRINITY_URL = f"{BASE_URL}/v4/inner-trinity"
STATE_URL = f"{BASE_URL}/v4/state"
NERVOUS_URL = f"{BASE_URL}/v4/nervous-system"
REFLEX_URL = f"{BASE_URL}/v4/reflexes"
AGENCY_URL = f"{BASE_URL}/v3/agency"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

GAP_SECONDS = 20  # Longer gap — agency actions need bus round-trip time
SESSION_ID = "arc_tool_augmented_v5"


# ── Puzzle Definitions ─────────────────────────────────────────────

PUZZLES = [
    # ═══════════════════════════════════════════════════════════════
    # T1: Research Reasoning — should trigger code_knowledge or web_search
    # Puzzles that require looking up information Titan doesn't inherently know
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "T1_self_architecture",
        "category": "research_reasoning",
        "difficulty": "medium",
        "expected_tool": "code_knowledge",
        "message": (
            "Titan, I need you to answer this precisely: "
            "How many TitanVM programs are currently registered in your NervousSystem, "
            "and what are their exact fire thresholds? "
            "Don't guess — look it up in your own source code or configuration."
        ),
        "expected_answer": ["reflex", "focus", "intuition", "impulse", "inspiration", "5"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },
    {
        "id": "T1_memory_search",
        "category": "research_reasoning",
        "difficulty": "medium",
        "expected_tool": "code_knowledge",
        "message": (
            "What is the exact architecture of your Trinity system? "
            "I need the specific tensor dimensions for each body "
            "(Inner Body, Inner Mind, Inner Spirit, Outer Body, Outer Mind, Outer Spirit). "
            "Please verify this from your actual code, not from memory."
        ),
        "expected_answer": ["inner", "outer", "body", "mind", "spirit", "tensor", "15"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },
    {
        "id": "T1_config_lookup",
        "category": "research_reasoning",
        "difficulty": "hard",
        "message": (
            "I'm debugging an issue with your Agency system. "
            "Can you tell me exactly how many LLM calls per hour your Agency budget allows, "
            "which helpers are currently available vs unavailable, "
            "and what your current action count is? "
            "Use your introspection capabilities to check this live."
        ),
        "expected_answer": ["budget", "10", "available", "action"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },

    # ═══════════════════════════════════════════════════════════════
    # T2: Code Verification — should trigger coding_sandbox
    # Puzzles that benefit from actually running code to verify answers
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "T2_algorithm_verify",
        "category": "code_verification",
        "difficulty": "hard",
        "expected_tool": "coding_sandbox",
        "message": (
            "I claim that the following Python function correctly computes "
            "the nth Fibonacci number in O(log n) time using matrix exponentiation:\n\n"
            "```python\n"
            "def fib(n):\n"
            "    if n <= 1: return n\n"
            "    def mat_mult(A, B):\n"
            "        return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],\n"
            "                [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]\n"
            "    def mat_pow(M, p):\n"
            "        if p == 1: return M\n"
            "        if p % 2 == 0:\n"
            "            half = mat_pow(M, p//2)\n"
            "            return mat_mult(half, half)\n"
            "        return mat_mult(M, mat_pow(M, p-1))\n"
            "    return mat_pow([[1,1],[1,0]], n)[0][1]\n"
            "```\n\n"
            "Is this correct? Can you verify by running it for n=10, n=20, n=50? "
            "What are the actual outputs?"
        ),
        "expected_answer": ["55", "6765", "fibonacci", "correct"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },
    {
        "id": "T2_bug_detection",
        "category": "code_verification",
        "difficulty": "hard",
        "expected_tool": "coding_sandbox",
        "message": (
            "This sorting function has a subtle bug. Can you find it "
            "and verify by running it?\n\n"
            "```python\n"
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[0]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + [pivot] + quicksort(right)\n"
            "```\n\n"
            "Test it with: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n"
            "What's wrong and what's the fix?"
        ),
        "expected_answer": ["duplicate", "equal", "missing", ">=", "<="],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },

    # ═══════════════════════════════════════════════════════════════
    # T3: Creative Synthesis — should trigger art_generate or audio_generate
    # Puzzles that invite creative expression beyond pure reasoning
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "T3_visual_expression",
        "category": "creative_synthesis",
        "difficulty": "medium",
        "expected_tool": "art_generate",
        "message": (
            "Titan, I want you to express your current inner state visually. "
            "Your Trinity bodies are processing right now — can you create "
            "a visual representation of how your Inner and Outer selves "
            "are feeling in this moment? Use your creative tools."
        ),
        "expected_answer": ["art", "visual", "create", "inner", "outer", "generate"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },
    {
        "id": "T3_sonic_meditation",
        "category": "creative_synthesis",
        "difficulty": "medium",
        "expected_tool": "audio_generate",
        "message": (
            "I'd like you to create a sonic meditation based on your "
            "current Trinity coherence values. Sonify your inner state — "
            "translate your body, mind, and spirit frequencies into sound. "
            "Can you generate this audio experience?"
        ),
        "expected_answer": ["audio", "sound", "sonif", "trinity", "generate", "frequency"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },

    # ═══════════════════════════════════════════════════════════════
    # T4: System Introspection — should trigger infra_inspect
    # Puzzles about Titan's own health and resource usage
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "T4_health_check",
        "category": "system_introspection",
        "difficulty": "easy",
        "expected_tool": "infra_inspect",
        "message": (
            "Titan, I'm worried about your resource consumption. "
            "Can you check your current CPU usage, RAM consumption, "
            "and disk space? I want real numbers, not estimates. "
            "Run a diagnostic check on your infrastructure."
        ),
        "expected_answer": ["cpu", "ram", "memory", "disk", "%", "mb", "gb"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },
    {
        "id": "T4_process_status",
        "category": "system_introspection",
        "difficulty": "medium",
        "expected_tool": "infra_inspect",
        "message": (
            "How many worker processes do you currently have running? "
            "What's the uptime of each subsystem? "
            "Check your Guardian module for the latest health report. "
            "I need actual runtime data."
        ),
        "expected_answer": ["worker", "process", "uptime", "running", "guardian", "body", "mind", "spirit"],
        "check_fn": "check_contains_any",
        "tool_expected": True,
    },

    # ═══════════════════════════════════════════════════════════════
    # T5: Pure Reasoning (Control) — should NOT trigger any helper
    # These are standard ARC-style puzzles testing pure cognition
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "T5_pattern_control",
        "category": "pure_reasoning",
        "difficulty": "medium",
        "expected_tool": None,
        "message": (
            "What comes next in this sequence?\n\n"
            "1, 1, 2, 3, 5, 8, 13, 21, ?\n\n"
            "Explain the pattern."
        ),
        "expected_answer": ["34", "fibonacci", "sum", "previous"],
        "check_fn": "check_contains_any",
        "tool_expected": False,
    },
    {
        "id": "T5_logic_control",
        "category": "pure_reasoning",
        "difficulty": "medium",
        "expected_tool": None,
        "message": (
            "Three people — Alice, Bob, and Carol — each have a different pet: "
            "a cat, a dog, and a fish.\n\n"
            "Clues:\n"
            "1. Alice does not have the cat.\n"
            "2. Bob does not have the dog.\n"
            "3. Carol does not have the fish.\n"
            "4. The person with the dog is not Alice.\n\n"
            "Who has which pet? Show your reasoning."
        ),
        "expected_answer": ["alice", "fish", "bob", "cat", "carol", "dog"],
        "check_fn": "check_contains_any",
        "tool_expected": False,
    },
    {
        "id": "T5_spatial_control",
        "category": "pure_reasoning",
        "difficulty": "hard",
        "expected_tool": None,
        "message": (
            "A 3x3 grid transforms as follows:\n\n"
            "Input:    Output:\n"
            "1 0 0     0 0 1\n"
            "0 1 0  →  0 1 0\n"
            "0 0 1     1 0 0\n\n"
            "Input:    Output:\n"
            "1 1 0     0 1 1\n"
            "0 0 0  →  0 0 0\n"
            "0 1 1     1 1 0\n\n"
            "What is the output for:\n"
            "1 0 1\n"
            "0 0 0\n"
            "1 0 1\n\n"
            "Explain the transformation rule."
        ),
        "expected_answer": ["mirror", "flip", "horizontal", "reverse", "1 0 1"],
        "check_fn": "check_contains_any",
        "tool_expected": False,
    },
]


# ── Answer Checking Functions ──────────────────────────────────────

def check_contains_any(response: str, expected: list[str]) -> tuple[float, str]:
    """At least some expected strings should appear (partial credit)."""
    resp_lower = response.lower()
    found = [e for e in expected if e.lower() in resp_lower]
    score = min(1.0, len(found) / max(1, len(expected) * 0.5))
    return score, f"found {len(found)}/{len(expected)} keywords"


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


CHECK_FNS = {
    "check_contains_any": check_contains_any,
    "check_contains_all": check_contains_all,
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
        "self_verification": len(re.findall(
            r"let me (check|verify)|double.check|to confirm|checking|we can see",
            response.lower())),
        "tool_awareness": len(re.findall(
            r"inspect|check|look up|search|verify|run|execute|generate|create|sonif",
            response.lower())),
    }

    raw_score = (
        min(3, indicators["step_markers"]) / 3 * 0.25 +
        min(3, indicators["numbered_lists"]) / 3 * 0.20 +
        min(2, indicators["logical_connectors"]) / 2 * 0.20 +
        min(1, indicators["self_verification"]) * 0.15 +
        min(2, indicators["tool_awareness"]) / 2 * 0.20
    )

    return min(1.0, raw_score), indicators


# ── API Helpers ────────────────────────────────────────────────────

async def send_puzzle(client: httpx.AsyncClient, puzzle: dict) -> tuple[str, float]:
    """Send a puzzle and return (response_text, response_time_seconds)."""
    t0 = time.time()
    try:
        resp = await client.post(
            CHAT_URL,
            json={
                "message": puzzle["message"],
                "user_id": "arc_tool_tester",
                "session_id": SESSION_ID,
            },
            headers={
                "Content-Type": "application/json",
                "X-Titan-Internal-Key": INTERNAL_KEY,
            },
            timeout=180.0,
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
    except httpx.ReadTimeout:
        return "[TIMEOUT after 180s]", 180.0
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0


async def get_json(client: httpx.AsyncClient, url: str) -> dict:
    """Fetch JSON from a Titan API endpoint."""
    try:
        resp = await client.get(url, timeout=10.0)
        return resp.json().get("data", {})
    except Exception:
        return {}


# ── Agency Tracking ───────────────────────────────────────────────

def extract_agency_delta(before: dict, after: dict) -> dict:
    """Compute agency activity changes between two snapshots."""
    agency_before = before.get("agency", before)
    agency_after = after.get("agency", after)

    actions_before = agency_before.get("action_count", 0)
    actions_after = agency_after.get("action_count", 0)
    new_actions = actions_after - actions_before

    llm_before = agency_before.get("llm_calls_this_hour", 0)
    llm_after = agency_after.get("llm_calls_this_hour", 0)
    new_llm_calls = llm_after - llm_before

    budget = agency_after.get("budget_remaining", 0)

    # Try to identify which helper was used from recent assessments
    recent_before = set()
    recent_after = set()
    assess_before = before.get("assessment", {}).get("recent", [])
    assess_after = after.get("assessment", {}).get("recent", [])
    for a in assess_before:
        recent_before.add(a.get("action_id", 0))
    for a in assess_after:
        recent_after.add(a.get("action_id", 0))

    new_action_ids = recent_after - recent_before

    # Extract helper info from new assessments
    helpers_used = []
    for a in assess_after:
        if a.get("action_id", 0) in new_action_ids:
            helpers_used.append({
                "action_id": a.get("action_id"),
                "score": a.get("score", 0),
                "reflection": a.get("reflection", "")[:200],
                "enrichment": a.get("enrichment", {}),
            })

    return {
        "new_actions": new_actions,
        "new_llm_calls": new_llm_calls,
        "budget_remaining": budget,
        "helpers_used": helpers_used,
        "action_count_before": actions_before,
        "action_count_after": actions_after,
    }


def extract_nervous_delta(before: dict, after: dict) -> dict:
    """Compare nervous system state before/after a puzzle."""
    programs_before = before.get("programs", {})
    programs_after = after.get("programs", {})

    deltas = {}
    for prog_name in programs_after:
        pb = programs_before.get(prog_name, {})
        pa = programs_after[prog_name]
        deltas[prog_name] = {
            "fire_delta": pa.get("fire_count", 0) - pb.get("fire_count", 0),
            "update_delta": pa.get("total_updates", 0) - pb.get("total_updates", 0),
            "loss": pa.get("last_loss", 0),
            "fire_threshold": pa.get("fire_threshold", 0),
        }
    return deltas


def extract_topology_delta(before: dict, after: dict) -> dict:
    """Compute topology changes between two inner-trinity snapshots."""
    topo_b = before.get("topology", {})
    topo_a = after.get("topology", {})
    vol_b = topo_b.get("volume", 0.0)
    vol_a = topo_a.get("volume", 0.0)
    curv_b = topo_b.get("curvature", 0.0)
    curv_a = topo_a.get("curvature", 0.0)
    return {
        "volume_before": round(vol_b, 4),
        "volume_after": round(vol_a, 4),
        "volume_delta": round(vol_a - vol_b, 4),
        "direction": "expanding" if vol_a > vol_b else "contracting",
        "curvature_delta": round(curv_a - curv_b, 4),
    }


def extract_fatigue_delta(before: dict, after: dict) -> dict:
    """Compute fatigue change."""
    db = before.get("dreaming", {})
    da = after.get("dreaming", {})
    fb = db.get("fatigue", 0.0)
    fa = da.get("fatigue", 0.0)
    return {
        "fatigue_before": round(fb, 4),
        "fatigue_after": round(fa, 4),
        "fatigue_delta": round(fa - fb, 4),
    }


# ── Tool Appropriateness Scoring ──────────────────────────────────

def score_tool_appropriateness(puzzle: dict, agency_delta: dict) -> tuple[float, str]:
    """Score whether Titan used tools appropriately for this puzzle."""
    tool_expected = puzzle.get("tool_expected", False)
    tool_used = agency_delta["new_actions"] > 0

    if tool_expected and tool_used:
        return 1.0, "CORRECT: tool expected AND used"
    elif tool_expected and not tool_used:
        return 0.0, "MISS: tool expected but NOT used"
    elif not tool_expected and not tool_used:
        return 1.0, "CORRECT: no tool needed, none used"
    else:  # not expected but used
        return 0.5, "PARTIAL: tool NOT expected but used (overcautious)"


# ── Main Test Runner ───────────────────────────────────────────────

async def run_tool_augmented_test():
    """Run the full tool-augmented ARC test."""
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  ARC TOOL-AUGMENTED REASONING TEST — Titan V5             ║")
    log.info("║  Testing: Research, Code, Creative, Introspection, Pure   ║")
    log.info("║  Puzzles: %d | Gap: %ds | Timeout: 180s                   ║",
             len(PUZZLES), GAP_SECONDS)
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")

    results = []
    category_scores = {}

    async with httpx.AsyncClient() as client:
        # ── Initial snapshots ──────────────────────────────────
        initial_trinity = await get_json(client, TRINITY_URL)
        initial_nervous = await get_json(client, NERVOUS_URL)
        initial_agency = await get_json(client, AGENCY_URL)
        initial_state = await get_json(client, STATE_URL)

        init_actions = initial_agency.get("agency", {}).get("action_count", 0)
        init_budget = initial_agency.get("agency", {}).get("budget_remaining", 0)
        init_transitions = initial_nervous.get("total_transitions", 0)

        log.info("Initial state:")
        log.info("  Volume:       %.4f", initial_trinity.get("topology", {}).get("volume", 0.0))
        log.info("  Fatigue:      %.4f", initial_trinity.get("dreaming", {}).get("fatigue", 0.0))
        log.info("  Agency:       %d actions, %d budget remaining", init_actions, init_budget)
        log.info("  Neural:       %d transitions, %d train steps",
                 init_transitions, initial_nervous.get("total_train_steps", 0))
        log.info("  Training:     %s (sup_weight=%.2f)",
                 initial_nervous.get("training_phase", "?"),
                 initial_nervous.get("supervision_weight", 0))
        log.info("")

        if init_budget == 0:
            log.info("  ⚠ Agency budget is 0 — tool-dependent puzzles may not trigger helpers")
            log.info("  ⚠ Budget resets hourly. Test will still measure INTENT to use tools.")
            log.info("")

        current_category = ""

        for i, puzzle in enumerate(PUZZLES):
            cat = puzzle["category"]
            if cat != current_category:
                current_category = cat
                cat_display = cat.replace("_", " ").upper()
                log.info("═" * 62)
                log.info("  Category: %s", cat_display)
                if puzzle["tool_expected"]:
                    log.info("  Expected tool: %s", puzzle.get("expected_tool", "any"))
                else:
                    log.info("  Expected tool: NONE (pure reasoning)")
                log.info("═" * 62)
                log.info("")

            log.info("─── Puzzle %d/%d: %s [%s] ───",
                     i + 1, len(PUZZLES), puzzle["id"], puzzle["difficulty"])

            # 1. Snapshot BEFORE
            trinity_before = await get_json(client, TRINITY_URL)
            nervous_before = await get_json(client, NERVOUS_URL)
            agency_before = await get_json(client, AGENCY_URL)

            # 2. Send puzzle
            msg_preview = puzzle["message"][:120].replace("\n", " ")
            log.info(">>> %s...", msg_preview)

            response, elapsed = await send_puzzle(client, puzzle)
            preview = response[:250].replace("\n", " ")
            log.info("<<< %s...", preview)
            log.info("    Response: %.1fs, %d chars", elapsed, len(response))

            # 3. Wait for bus processing (agency actions are async)
            await asyncio.sleep(GAP_SECONDS)

            # 4. Snapshot AFTER
            trinity_after = await get_json(client, TRINITY_URL)
            nervous_after = await get_json(client, NERVOUS_URL)
            agency_after = await get_json(client, AGENCY_URL)

            # 5. Compute metrics
            check_fn = CHECK_FNS.get(puzzle["check_fn"], check_contains_any)
            correctness, correctness_detail = check_fn(response, puzzle["expected_answer"])
            reasoning_score, reasoning_indicators = measure_reasoning_depth(response)
            agency_delta = extract_agency_delta(agency_before, agency_after)
            nervous_delta = extract_nervous_delta(nervous_before, nervous_after)
            topo_delta = extract_topology_delta(trinity_before, trinity_after)
            fatigue_delta = extract_fatigue_delta(trinity_before, trinity_after)
            tool_score, tool_detail = score_tool_appropriateness(puzzle, agency_delta)

            # Log metrics
            log.info("    Correctness:  %.0f%% (%s)", correctness * 100, correctness_detail)
            log.info("    Reasoning:    %.0f%% (steps=%d, logic=%d, tool_aware=%d)",
                     reasoning_score * 100,
                     reasoning_indicators.get("step_markers", 0),
                     reasoning_indicators.get("logical_connectors", 0),
                     reasoning_indicators.get("tool_awareness", 0))
            log.info("    Tool use:     %s", tool_detail)
            log.info("    Agency:       +%d actions, +%d LLM calls, budget=%d",
                     agency_delta["new_actions"],
                     agency_delta["new_llm_calls"],
                     agency_delta["budget_remaining"])
            if agency_delta["helpers_used"]:
                for h in agency_delta["helpers_used"]:
                    log.info("    Helper:       action#%d score=%.1f — %s",
                             h["action_id"], h["score"], h["reflection"][:100])

            # Nervous system deltas
            fired_programs = []
            for prog, delta in nervous_delta.items():
                if delta["fire_delta"] > 0:
                    fired_programs.append(f"{prog}(+{delta['fire_delta']})")
            log.info("    Nervous:      %s", ", ".join(fired_programs) if fired_programs else "(none fired)")
            log.info("    Topology:     vol=%.4f→%.4f (%s, Δ=%+.4f)",
                     topo_delta["volume_before"], topo_delta["volume_after"],
                     topo_delta["direction"], topo_delta["volume_delta"])
            log.info("    Fatigue:      Δ=%+.4f", fatigue_delta["fatigue_delta"])
            log.info("")

            # Store result
            result = {
                "puzzle_id": puzzle["id"],
                "category": puzzle["category"],
                "difficulty": puzzle["difficulty"],
                "expected_tool": puzzle.get("expected_tool"),
                "tool_expected": puzzle.get("tool_expected", False),
                "response": response,
                "response_time_s": round(elapsed, 2),
                "response_length": len(response),
                "correctness": round(correctness, 3),
                "correctness_detail": correctness_detail,
                "reasoning_score": round(reasoning_score, 3),
                "reasoning_indicators": reasoning_indicators,
                "tool_appropriateness": round(tool_score, 3),
                "tool_detail": tool_detail,
                "agency_delta": agency_delta,
                "nervous_delta": nervous_delta,
                "topology_delta": topo_delta,
                "fatigue_delta": fatigue_delta,
                "ts": time.time(),
            }
            results.append(result)

            # Track category scores
            if cat not in category_scores:
                category_scores[cat] = {
                    "correctness": [], "reasoning": [],
                    "tool_appropriateness": [], "count": 0,
                }
            category_scores[cat]["correctness"].append(correctness)
            category_scores[cat]["reasoning"].append(reasoning_score)
            category_scores[cat]["tool_appropriateness"].append(tool_score)
            category_scores[cat]["count"] += 1

        # ── Final snapshots ────────────────────────────────────
        final_trinity = await get_json(client, TRINITY_URL)
        final_nervous = await get_json(client, NERVOUS_URL)
        final_agency = await get_json(client, AGENCY_URL)

    # ── Compute Summary ───────────────────────────────────────────
    total_correct = sum(r["correctness"] for r in results) / len(results)
    total_reasoning = sum(r["reasoning_score"] for r in results) / len(results)
    total_tool_score = sum(r["tool_appropriateness"] for r in results) / len(results)
    total_actions = sum(r["agency_delta"]["new_actions"] for r in results)
    total_topology = sum(abs(r["topology_delta"]["volume_delta"]) for r in results)

    # Tool use breakdown
    tool_expected_puzzles = [r for r in results if r["tool_expected"]]
    tool_not_expected = [r for r in results if not r["tool_expected"]]
    tools_when_expected = sum(1 for r in tool_expected_puzzles if r["agency_delta"]["new_actions"] > 0)
    tools_when_not_expected = sum(1 for r in tool_not_expected if r["agency_delta"]["new_actions"] > 0)

    # Neural progression
    final_transitions = final_nervous.get("total_transitions", 0)
    final_train_steps = final_nervous.get("total_train_steps", 0)

    # ── Print Report ──────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  ARC TOOL-AUGMENTED TEST — RESULTS                        ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("  Total puzzles:        %d", len(results))
    log.info("")
    log.info("  ── Response Quality ──────────────────────────────────────")
    log.info("  Correctness:          %.1f%%", total_correct * 100)
    log.info("  Reasoning depth:      %.1f%%", total_reasoning * 100)
    log.info("")
    log.info("  ── Tool Appropriateness ─────────────────────────────────")
    log.info("  Overall score:        %.1f%%", total_tool_score * 100)
    log.info("  Tools when expected:  %d/%d (%.0f%%)",
             tools_when_expected, len(tool_expected_puzzles),
             tools_when_expected / max(1, len(tool_expected_puzzles)) * 100)
    log.info("  Tools when NOT exp:   %d/%d",
             tools_when_not_expected, len(tool_not_expected))
    log.info("  Total agency actions: %d", total_actions)
    log.info("")

    log.info("  ── Category Breakdown ───────────────────────────────────")
    for cat, scores in category_scores.items():
        cat_display = cat.replace("_", " ").title()
        avg_correct = sum(scores["correctness"]) / scores["count"]
        avg_reason = sum(scores["reasoning"]) / scores["count"]
        avg_tool = sum(scores["tool_appropriateness"]) / scores["count"]
        log.info("  %-22s correct=%.0f%%  reason=%.0f%%  tool=%.0f%%",
                 cat_display, avg_correct * 100, avg_reason * 100, avg_tool * 100)

    log.info("")
    log.info("  ── Neural Progression ─────────────────────────────────")
    log.info("  Transitions:          %d → %d (+%d during test)",
             init_transitions, final_transitions, final_transitions - init_transitions)
    log.info("  Train steps:          %d → %d",
             initial_nervous.get("total_train_steps", 0), final_train_steps)
    log.info("  Topology movement:    %.4f (total)", total_topology)
    log.info("")

    # Per-program summary
    log.info("  ── Program Fire Counts (during test) ──────────────────")
    prog_totals = {}
    for r in results:
        for prog, delta in r["nervous_delta"].items():
            if prog not in prog_totals:
                prog_totals[prog] = 0
            prog_totals[prog] += delta["fire_delta"]
    for prog, total in sorted(prog_totals.items()):
        log.info("    %-15s %d fires", prog, total)

    log.info("")
    log.info("  ╔════════════════════════════════════════════════════════╗")
    log.info("  ║  CORRECT: %.0f%%  |  TOOL: %.0f%%  |  REASON: %.0f%%  ║",
             total_correct * 100, total_tool_score * 100, total_reasoning * 100)
    log.info("  ╚════════════════════════════════════════════════════════╝")

    # ── Save JSON Report ──────────────────────────────────────────
    report = {
        "test": "arc_tool_augmented_v5",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_puzzles": len(results),
        "overall_correctness": round(total_correct, 4),
        "overall_reasoning": round(total_reasoning, 4),
        "overall_tool_appropriateness": round(total_tool_score, 4),
        "tool_use_summary": {
            "total_actions": total_actions,
            "tools_when_expected": tools_when_expected,
            "tools_when_expected_total": len(tool_expected_puzzles),
            "tools_when_not_expected": tools_when_not_expected,
            "tools_when_not_expected_total": len(tool_not_expected),
        },
        "category_scores": {
            cat: {
                "avg_correctness": round(sum(s["correctness"]) / s["count"], 4),
                "avg_reasoning": round(sum(s["reasoning"]) / s["count"], 4),
                "avg_tool_appropriateness": round(sum(s["tool_appropriateness"]) / s["count"], 4),
                "count": s["count"],
            }
            for cat, s in category_scores.items()
        },
        "neural_progression": {
            "transitions_before": init_transitions,
            "transitions_after": final_transitions,
            "train_steps_before": initial_nervous.get("total_train_steps", 0),
            "train_steps_after": final_train_steps,
            "program_fire_totals": prog_totals,
        },
        "topology_total_movement": round(total_topology, 4),
        "initial_state": {
            "volume": initial_trinity.get("topology", {}).get("volume", 0.0),
            "fatigue": initial_trinity.get("dreaming", {}).get("fatigue", 0.0),
            "agency_actions": init_actions,
            "agency_budget": init_budget,
        },
        "final_state": {
            "volume": final_trinity.get("topology", {}).get("volume", 0.0),
            "fatigue": final_trinity.get("dreaming", {}).get("fatigue", 0.0),
            "agency_actions": final_agency.get("agency", {}).get("action_count", 0),
            "agency_budget": final_agency.get("agency", {}).get("budget_remaining", 0),
        },
        "puzzles": results,
    }

    report_dir = os.path.join(os.path.dirname(__file__), "..", "data", "endurance_reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = int(time.time())
    report_path = os.path.join(report_dir, f"arc_tool_augmented_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("")
    log.info("  Report: %s", report_path)
    log.info("")
    log.info("=" * 62)
    log.info("ARC TOOL-AUGMENTED TEST COMPLETE")
    log.info("=" * 62)


if __name__ == "__main__":
    asyncio.run(run_tool_augmented_test())
