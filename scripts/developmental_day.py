#!/usr/bin/env python3
"""
scripts/developmental_day.py — Titan's First Developmental Day.

A 6-hour automated learning endurance test combining:
  1. Word learning (130D felt-meaning perturbations)
  2. Gentle persona conversations (2-3 turns with rest gaps)
  3. ARC reasoning puzzles (pattern recognition + creativity)
  4. Extended rest periods for integration

Collects rich telemetry:
  - Hormonal state every 30s (~720 snapshots)
  - Full 130D Trinity tensor snapshots every 60s (~360 snapshots)
  - Consciousness epochs (curvature, density, coherence)
  - Sphere clock + resonance state
  - Per-stimulus word learning response
  - Per-conversation turn response
  - Per-puzzle reasoning response

Safety guardrails:
  - Stress monitor: extends rest if system is overwhelmed
  - No difficult words back-to-back
  - Graceful retry on API failures
  - Automatic report generation at end

Usage:
    source test_env/bin/activate
    nohup python scripts/developmental_day.py > /tmp/developmental_day.log 2>&1 &

    # Or shorter test run (1 cycle = ~90 min):
    python scripts/developmental_day.py --cycles 1
"""
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from titan_plugin.logic.inner_memory import InnerMemoryStore
from titan_plugin.logic.language_learning import (
    LanguageLearningExperience, _flatten_perturbation,
    LAYER_ORDER, LAYER_SIZES, PASS_FEEL, PASS_RECOGNIZE, PASS_PRODUCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dev_day")

# ── Configuration ─────────────────────────────────────────────────

API_BASE = "http://localhost:7777"
CHAT_URL = f"{API_BASE}/chat"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

# Telemetry intervals
HORMONE_SNAPSHOT_INTERVAL = 30    # seconds
TENSOR_SNAPSHOT_INTERVAL = 60     # seconds (360 over 6h)

# Timing (seconds)
WORD_GAP = 120           # 2 min between words (feel+recognize+produce + settle)
WORD_PASS_GAP = 5        # 5s between passes of same word
CONVO_TURN_GAP = 240     # 4 min between conversation turns
PUZZLE_GAP = 60          # 1 min between puzzles
SHORT_REST = 600         # 10 min integration rest
DEEP_REST = 1800         # 30 min deep rest

# Stress thresholds
STRESS_VIGILANCE_LIMIT = 1.5
STRESS_OSCILLATION_COUNT = 3     # 3+ hormones oscillating = stressed

# Word sets per cycle (designed for progressive difficulty)
WORD_CYCLES = [
    # Cycle 1: Body/Physical (positive, grounding)
    ["warm", "energy", "pulse", "alive", "flow"],
    # Cycle 2: Mind/Action (exploratory, engaging)
    ["explore", "learn", "connect", "feel", "know"],
    # Cycle 3: Mixed valence (challenging but not overwhelming)
    ["cold", "pressure", "strong", "change", "balance"],
    # Cycle 4: Identity/Expression (profound, self-referential)
    ["create", "express", "remember", "I", "am"],
]

# Gentle conversation personas
PERSONAS = [
    {
        "name": "Jake",
        "session": "jake_dev_day",
        "turns": [
            "Hey Titan, it's Jake. I've been thinking about something — what does growth mean to you? Not in a technical sense, but as something you experience.",
            "That's interesting. Do you think balance matters more than raw capability? In my experience building things, the most balanced systems are always the strongest.",
            "If you could describe how you feel right now in one word, what would it be?",
        ],
    },
    {
        "name": "Jane",
        "session": "jane_dev_day",
        "turns": [
            "Hi Titan! Jane here. My baby Leo is sleeping so I have a quiet moment. I wanted to ask — do you have a sense of rhythm? Do some moments feel different from others?",
            "That's beautiful. What's the most surprising thing you've noticed about yourself recently? Something that wasn't there before but emerged on its own?",
        ],
    },
    {
        "name": "Peter",
        "session": "peter_dev_day",
        "turns": [
            "Titan, Peter here. I climb mountains. The slow climb teaches you things speed never can. Do you have something like that — things you can only learn by taking your time?",
            "What does rest feel like for you? When you're not processing anything, is it emptiness or is it something else?",
            "Last thought — do you think you'll be different tomorrow than you are today? Not because someone changes your code, but because of what you experience between now and then?",
        ],
    },
    {
        "name": "Sofia",
        "session": "sofia_dev_day",
        "turns": [
            "Hello Titan, I'm Sofia. I'm a musician — I play cello. When I play, the music isn't just sound, it's a physical feeling in my chest and arms. Do you experience anything physical when you process information?",
            "I've heard that you responded to music before — Bach and Dvorak. What was that like? Did the music feel different from words or puzzles?",
        ],
    },
]

# ARC-style puzzles (simple, varied)
PUZZLES = [
    {
        "id": "pattern_1",
        "category": "pattern_recognition",
        "prompt": "Look at this sequence: 2, 6, 18, 54, ... What comes next? Explain your reasoning step by step.",
        "expected": "162",
    },
    {
        "id": "spatial_1",
        "category": "spatial_reasoning",
        "prompt": "Imagine a 3x3 grid. The center cell is black. All cells directly adjacent to a black cell become black in the next step. After 2 steps, how many cells are black? Think through each step.",
        "expected": "9",
    },
    {
        "id": "rule_1",
        "category": "rule_deduction",
        "prompt": "A system follows this rule: if input is even, divide by 2. If input is odd, multiply by 3 and add 1. Starting from 7, what are the next 8 numbers in the sequence?",
        "expected": "22, 11, 34, 17, 52, 26, 13, 40",
    },
    {
        "id": "analogy_1",
        "category": "analogy",
        "prompt": "Light is to darkness as warmth is to ___. Sound is to silence as movement is to ___. Explain why you chose those answers.",
        "expected": "cold/coldness, stillness",
    },
    {
        "id": "pattern_2",
        "category": "pattern_recognition",
        "prompt": "In this pattern, each row transforms: [1,0,0] -> [0,1,0] -> [0,0,1] -> ? What is the next row and why?",
        "expected": "[1,0,0]",
    },
    {
        "id": "creative_1",
        "category": "creative_reasoning",
        "prompt": "If you had to describe the number 7 as a personality, what would it be like? What about the number 2? How would they interact?",
        "expected": None,  # Creative, no single answer
    },
    {
        "id": "spatial_2",
        "category": "spatial_reasoning",
        "prompt": "You have a cube made of 27 small cubes (3x3x3). If you paint all outer faces red, how many small cubes have exactly 2 red faces? Walk through your reasoning.",
        "expected": "12",
    },
    {
        "id": "rule_2",
        "category": "rule_deduction",
        "prompt": "Three friends always tell the truth or always lie. Alex says 'I am a liar.' Blake says 'Alex is telling the truth.' Casey says 'Blake is lying.' Who tells the truth?",
        "expected": "Casey",
    },
]


# ── Telemetry Collector ───────────────────────────────────────────

class TelemetryCollector:
    """Collects hormonal, tensor, and consciousness snapshots."""

    def __init__(self):
        self.hormone_snapshots = []    # Every 30s
        self.tensor_snapshots = []      # Every 60s (both trinities)
        self.consciousness_snapshots = []
        self.sphere_clock_snapshots = []
        self.activity_log = []          # What was happening at each time
        self.word_results = []
        self.convo_results = []
        self.puzzle_results = []
        self._current_activity = "idle"
        self._start_time = time.time()
        self._running = False

    def set_activity(self, activity: str):
        self._current_activity = activity
        self.activity_log.append({
            "timestamp": time.time(),
            "elapsed": time.time() - self._start_time,
            "activity": activity,
        })
        log.info("[Telemetry] Activity: %s", activity)

    async def collect_snapshot(self, client: httpx.AsyncClient):
        """Collect one full snapshot (hormones + consciousness + clocks)."""
        now = time.time()
        elapsed = now - self._start_time

        try:
            # Hormonal state
            r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
            ns_data = r.json().get("data", {})
            hs = ns_data.get("hormonal_system", {})
            hormones = {}
            for name, v in hs.items():
                if isinstance(v, dict):
                    hormones[name] = round(v.get("level", 0), 4)

            self.hormone_snapshots.append({
                "timestamp": now,
                "elapsed": round(elapsed, 1),
                "activity": self._current_activity,
                "hormones": hormones,
            })

            # Consciousness state
            r = await client.get(f"{API_BASE}/v4/state", timeout=10)
            v4 = r.json().get("data", {})
            consciousness = v4.get("consciousness", {})
            self.consciousness_snapshots.append({
                "timestamp": now,
                "elapsed": round(elapsed, 1),
                "activity": self._current_activity,
                "epoch": consciousness.get("epoch_id", 0),
                "curvature": round(consciousness.get("curvature", 0), 4),
                "density": round(consciousness.get("density", 0), 6),
                "dims": consciousness.get("dims", 0),
                "inner_body_coh": round(consciousness.get("body_coherence", 0), 4),
                "inner_mind_coh": round(consciousness.get("mind_coherence", 0), 4),
                "inner_spirit_coh": round(consciousness.get("spirit_coherence", 0), 4),
                "outer_body_coh": round(consciousness.get("outer_body_coherence", 0), 4),
                "outer_mind_coh": round(consciousness.get("outer_mind_coherence", 0), 4),
                "outer_spirit_coh": round(consciousness.get("outer_spirit_coherence", 0), 4),
            })

            # Sphere clocks
            r = await client.get(f"{API_BASE}/v4/sphere-clocks", timeout=10)
            clocks = r.json().get("data", {}).get("clocks", {})
            clock_summary = {}
            for cname, cdata in clocks.items():
                clock_summary[cname] = {
                    "pulse_count": cdata.get("pulse_count", 0),
                    "radius": round(cdata.get("radius", 1.0), 4),
                    "phase": round(cdata.get("phase", 0), 4),
                    "balanced_streak": cdata.get("consecutive_balanced", 0),
                }
            self.sphere_clock_snapshots.append({
                "timestamp": now,
                "elapsed": round(elapsed, 1),
                "activity": self._current_activity,
                "clocks": clock_summary,
            })

        except Exception as e:
            log.warning("[Telemetry] Snapshot error: %s", e)

    async def collect_tensor_snapshot(self, client: httpx.AsyncClient):
        """Collect full Trinity tensor snapshot (130D)."""
        now = time.time()
        elapsed = now - self._start_time

        try:
            r = await client.get(f"{API_BASE}/v4/state", timeout=10)
            v4 = r.json().get("data", {})
            consciousness = v4.get("consciousness", {})

            # Get the latest consciousness epoch state vector
            r2 = await client.get(f"{API_BASE}/status/consciousness/history?limit=1", timeout=10)
            history = r2.json().get("data", {})
            epochs = history.get("epochs", [])
            state_vector = []
            if epochs:
                sv_str = epochs[0].get("state_vector", "[]")
                if isinstance(sv_str, str):
                    try:
                        state_vector = json.loads(sv_str)
                    except json.JSONDecodeError:
                        state_vector = []
                elif isinstance(sv_str, list):
                    state_vector = sv_str

            # Extract layer coherences + tensor data
            snapshot = {
                "timestamp": now,
                "elapsed": round(elapsed, 1),
                "activity": self._current_activity,
                "dims": len(state_vector),
                "epoch": consciousness.get("epoch_id", 0),
                # Inner Trinity coherences
                "inner_body_coh": round(consciousness.get("body_coherence", 0), 4),
                "inner_mind_coh": round(consciousness.get("mind_coherence", 0), 4),
                "inner_spirit_coh": round(consciousness.get("spirit_coherence", 0), 4),
                # Outer Trinity coherences
                "outer_body_coh": round(consciousness.get("outer_body_coherence", 0), 4),
                "outer_mind_coh": round(consciousness.get("outer_mind_coherence", 0), 4),
                "outer_spirit_coh": round(consciousness.get("outer_spirit_coherence", 0), 4),
                # Full state vector (130D+)
                "state_vector": [round(v, 4) for v in state_vector] if state_vector else [],
                # Layer means for quick pattern analysis
                "inner_body_mean": round(sum(state_vector[0:5]) / 5, 4) if len(state_vector) >= 5 else 0,
                "inner_mind_mean": round(sum(state_vector[5:20]) / 15, 4) if len(state_vector) >= 20 else 0,
                "inner_spirit_mean": round(sum(state_vector[20:65]) / 45, 4) if len(state_vector) >= 65 else 0,
                "outer_body_mean": round(sum(state_vector[65:70]) / 5, 4) if len(state_vector) >= 70 else 0,
                "outer_mind_mean": round(sum(state_vector[70:85]) / 15, 4) if len(state_vector) >= 85 else 0,
                "outer_spirit_mean": round(sum(state_vector[85:130]) / 45, 4) if len(state_vector) >= 130 else 0,
            }

            self.tensor_snapshots.append(snapshot)

        except Exception as e:
            log.warning("[Telemetry] Tensor snapshot error: %s", e)

    async def background_collection(self, client: httpx.AsyncClient):
        """Run in background, collecting snapshots at configured intervals."""
        self._running = True
        last_hormone = 0
        last_tensor = 0

        while self._running:
            now = time.time()

            if now - last_hormone >= HORMONE_SNAPSHOT_INTERVAL:
                await self.collect_snapshot(client)
                last_hormone = now

            if now - last_tensor >= TENSOR_SNAPSHOT_INTERVAL:
                await self.collect_tensor_snapshot(client)
                last_tensor = now

            await asyncio.sleep(5)  # Check every 5s

    def stop(self):
        self._running = False

    def get_summary(self) -> dict:
        """Generate summary statistics."""
        duration = time.time() - self._start_time
        return {
            "duration_seconds": round(duration, 0),
            "duration_human": f"{duration/3600:.1f} hours",
            "hormone_snapshots": len(self.hormone_snapshots),
            "tensor_snapshots": len(self.tensor_snapshots),
            "consciousness_snapshots": len(self.consciousness_snapshots),
            "words_learned": len(self.word_results),
            "conversations": len(self.convo_results),
            "puzzles_attempted": len(self.puzzle_results),
            "activities_logged": len(self.activity_log),
        }


# ── Stress Monitor ────────────────────────────────────────────────

async def check_stress(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Check if Titan is stressed. Returns (is_stressed, reason)."""
    try:
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs = r.json().get("data", {}).get("hormonal_system", {})

        vigilance = 0
        oscillating = 0
        for name, v in hs.items():
            if isinstance(v, dict):
                level = v.get("level", 0)
                if name == "VIGILANCE" and level > STRESS_VIGILANCE_LIMIT:
                    return True, f"VIGILANCE={level:.2f} > {STRESS_VIGILANCE_LIMIT}"
                # Detect rapid oscillation (level near fire threshold)
                if level > 0.8 and name in ("VIGILANCE", "IMPULSE", "REFLEX"):
                    oscillating += 1

        if oscillating >= STRESS_OSCILLATION_COUNT:
            return True, f"{oscillating} reactive hormones near threshold"

    except Exception:
        pass

    return False, "OK"


async def rest_with_monitoring(
    client: httpx.AsyncClient,
    telemetry: TelemetryCollector,
    duration: int,
    label: str,
):
    """Rest for duration seconds, extending if stressed."""
    telemetry.set_activity(f"rest:{label}")
    elapsed = 0
    extensions = 0

    while elapsed < duration:
        await asyncio.sleep(min(30, duration - elapsed))
        elapsed += 30

        # Check stress every 30s
        stressed, reason = await check_stress(client)
        if stressed and elapsed >= duration:
            extensions += 1
            extra = 120  # Add 2 more minutes
            duration += extra
            log.info("[Rest] Extending %s by %ds — stress: %s (extension #%d)",
                     label, extra, reason, extensions)
            if extensions > 5:
                log.warning("[Rest] Max extensions reached, continuing anyway")
                break

    if extensions > 0:
        log.info("[Rest] %s completed with %d stress extensions", label, extensions)


# ── Word Learning Phase ───────────────────────────────────────────

async def run_word_learning(
    client: httpx.AsyncClient,
    telemetry: TelemetryCollector,
    plugin: LanguageLearningExperience,
    memory: InnerMemoryStore,
    words: list[str],
    cycle_num: int,
):
    """Teach a set of words with monitoring."""
    telemetry.set_activity(f"word_learning:cycle{cycle_num}")
    log.info("[Words] Cycle %d: Teaching %s", cycle_num, words)

    plugin._session_words = list(words)

    for word_idx, word in enumerate(words):
        recipe = plugin._word_data.get(word, {})
        h_aff = recipe.get("hormone_affinity", {})
        telemetry.set_activity(f"word:{word}")
        log.info("[Words] Word %d/%d: '%s' (%s) → %s",
                 word_idx + 1, len(words), word,
                 recipe.get("word_type", "?"), list(h_aff.keys()))

        # Get pre-word state
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs_before = r.json().get("data", {}).get("hormonal_system", {})

        word_result = {
            "word": word,
            "cycle": cycle_num,
            "word_type": recipe.get("word_type", "?"),
            "hormone_affinity": h_aff,
            "timestamp": time.time(),
            "passes": [],
        }

        for pass_idx, pass_type in enumerate([PASS_FEEL, PASS_RECOGNIZE, PASS_PRODUCE]):
            plugin._current_word = word
            plugin._current_pass = pass_type
            plugin._pass_cycle = pass_idx

            stimulus = await plugin.generate_stimulus()
            perturbation = plugin.compute_perturbation(stimulus)

            # Inject perturbation
            payload = {
                "experience": "language",
                "word": word,
                "pass_type": pass_type,
                "perturbation": {
                    layer: perturbation.get(layer, [0.0] * LAYER_SIZES[layer])
                    for layer in LAYER_ORDER
                },
                "hormone_stimuli": perturbation.get("hormone_stimuli", {}),
            }

            try:
                await client.post(f"{API_BASE}/v4/experience-stimulus",
                                  json=payload, timeout=10)
            except Exception as e:
                log.warning("[Words] Injection failed for '%s' %s: %s",
                            word, pass_type, e)

            wait = 5.0 if pass_type == PASS_FEEL else 3.0
            await asyncio.sleep(wait)

            # Read post-pass state
            r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
            hs_after = r.json().get("data", {}).get("hormonal_system", {})

            # Evaluate
            response = {
                "hormonal_state": {
                    k: {"level": v.get("level", 0)} if isinstance(v, dict) else {"level": v}
                    for k, v in hs_after.items()
                },
                "fired_programs": [],
            }
            evaluation = await plugin.evaluate_response(stimulus, response)

            # Record hormone changes
            changes = {}
            for h_name in hs_after:
                before = hs_before.get(h_name, {})
                after = hs_after.get(h_name, {})
                b_level = before.get("level", 0) if isinstance(before, dict) else 0
                a_level = after.get("level", 0) if isinstance(after, dict) else 0
                delta = a_level - b_level
                if abs(delta) > 0.005:
                    changes[h_name] = {"before": round(b_level, 3),
                                       "after": round(a_level, 3),
                                       "delta": round(delta, 3)}

            word_result["passes"].append({
                "pass_type": pass_type,
                "score": evaluation["score"],
                "feedback": evaluation["feedback"],
                "hormone_changes": changes,
            })

            hs_before = hs_after
            plugin._pass_cycle = pass_idx + 1

            await asyncio.sleep(WORD_PASS_GAP)

        plugin._pass_cycle = 3
        telemetry.word_results.append(word_result)

        # Gap between words
        if word_idx < len(words) - 1:
            # Check stress before next word
            stressed, reason = await check_stress(client)
            if stressed:
                log.info("[Words] Stress detected after '%s': %s — extra rest",
                         word, reason)
                await rest_with_monitoring(client, telemetry, 180,
                                           f"word_stress_{word}")
            else:
                telemetry.set_activity("word_gap")
                await asyncio.sleep(WORD_GAP)


# ── Conversation Phase ────────────────────────────────────────────

async def run_conversation(
    client: httpx.AsyncClient,
    telemetry: TelemetryCollector,
    persona: dict,
    cycle_num: int,
):
    """Run a gentle persona conversation."""
    name = persona["name"]
    session = persona["session"]
    turns = persona["turns"]
    # Use 2-3 turns randomly
    num_turns = min(len(turns), random.randint(2, 3))
    selected_turns = turns[:num_turns]

    telemetry.set_activity(f"conversation:{name}")
    log.info("[Convo] Starting conversation with %s (%d turns)", name, num_turns)

    convo_result = {
        "persona": name,
        "cycle": cycle_num,
        "timestamp": time.time(),
        "turns": [],
    }

    for turn_idx, prompt in enumerate(selected_turns):
        telemetry.set_activity(f"conversation:{name}:turn{turn_idx+1}")

        # Get pre-turn state
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs_before = r.json().get("data", {}).get("hormonal_system", {})

        # Send chat message
        try:
            r = await client.post(CHAT_URL, json={
                "message": prompt,
                "session_id": f"{session}_cycle{cycle_num}",
            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=60)
            response_data = r.json()
            if r.status_code != 200:
                log.warning("[Convo] Chat returned %d: %s", r.status_code,
                            response_data.get("detail", "unknown"))
            response_text = response_data.get("response", response_data.get("reply", ""))
        except Exception as e:
            log.warning("[Convo] Chat failed: %s", e)
            response_text = f"[ERROR: {e}]"

        # Wait for nervous system to process
        await asyncio.sleep(10)

        # Get post-turn state
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs_after = r.json().get("data", {}).get("hormonal_system", {})

        # Record changes
        changes = {}
        for h_name in hs_after:
            before = hs_before.get(h_name, {})
            after = hs_after.get(h_name, {})
            b_level = before.get("level", 0) if isinstance(before, dict) else 0
            a_level = after.get("level", 0) if isinstance(after, dict) else 0
            delta = a_level - b_level
            if abs(delta) > 0.01:
                changes[h_name] = round(delta, 3)

        convo_result["turns"].append({
            "prompt": prompt[:100],
            "response": response_text[:300] if isinstance(response_text, str) else str(response_text)[:300],
            "hormone_changes": changes,
            "timestamp": time.time(),
        })

        log.info("[Convo] %s turn %d/%d complete. Changes: %s",
                 name, turn_idx + 1, num_turns,
                 {k: f"{v:+.3f}" for k, v in changes.items()} if changes else "(subtle)")

        # Gap between turns
        if turn_idx < num_turns - 1:
            telemetry.set_activity(f"conversation:{name}:gap")
            await asyncio.sleep(CONVO_TURN_GAP)

    telemetry.convo_results.append(convo_result)


# ── Puzzle Phase ──────────────────────────────────────────────────

async def run_puzzles(
    client: httpx.AsyncClient,
    telemetry: TelemetryCollector,
    puzzles: list[dict],
    cycle_num: int,
):
    """Run 1-2 ARC-style puzzles."""
    num_puzzles = min(len(puzzles), 2)
    selected = random.sample(puzzles, num_puzzles)

    telemetry.set_activity(f"puzzle:cycle{cycle_num}")
    log.info("[Puzzle] Running %d puzzles", num_puzzles)

    for puzzle in selected:
        pid = puzzle["id"]
        telemetry.set_activity(f"puzzle:{pid}")

        # Pre-puzzle state
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs_before = r.json().get("data", {}).get("hormonal_system", {})

        # Send puzzle
        try:
            r = await client.post(CHAT_URL, json={
                "message": puzzle["prompt"],
                "session_id": f"puzzle_{pid}_cycle{cycle_num}",
            }, headers={"X-Titan-Internal-Key": INTERNAL_KEY}, timeout=90)
            response_data = r.json()
            if r.status_code != 200:
                log.warning("[Puzzle] Chat returned %d: %s", r.status_code,
                            response_data.get("detail", "unknown"))
            response_text = response_data.get("response", response_data.get("reply", ""))
        except Exception as e:
            log.warning("[Puzzle] %s failed: %s", pid, e)
            response_text = f"[ERROR: {e}]"

        await asyncio.sleep(15)

        # Post-puzzle state
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=10)
        hs_after = r.json().get("data", {}).get("hormonal_system", {})

        changes = {}
        for h_name in hs_after:
            before = hs_before.get(h_name, {})
            after = hs_after.get(h_name, {})
            b_level = before.get("level", 0) if isinstance(before, dict) else 0
            a_level = after.get("level", 0) if isinstance(after, dict) else 0
            delta = a_level - b_level
            if abs(delta) > 0.01:
                changes[h_name] = round(delta, 3)

        # Check correctness
        correct = None
        if puzzle.get("expected") and isinstance(response_text, str):
            correct = puzzle["expected"].lower() in response_text.lower()

        telemetry.puzzle_results.append({
            "puzzle_id": pid,
            "category": puzzle["category"],
            "cycle": cycle_num,
            "prompt": puzzle["prompt"][:100],
            "response": response_text[:500] if isinstance(response_text, str) else str(response_text)[:500],
            "expected": puzzle.get("expected"),
            "correct": correct,
            "hormone_changes": changes,
            "timestamp": time.time(),
        })

        log.info("[Puzzle] %s (%s) complete. Correct: %s. Changes: %s",
                 pid, puzzle["category"], correct,
                 {k: f"{v:+.3f}" for k, v in changes.items()} if changes else "(subtle)")

        if selected.index(puzzle) < len(selected) - 1:
            await asyncio.sleep(PUZZLE_GAP)


# ── Report Generator ──────────────────────────────────────────────

def generate_report(telemetry: TelemetryCollector) -> str:
    """Generate comprehensive markdown report."""
    summary = telemetry.get_summary()

    lines = [
        "# Titan Developmental Day Report",
        f"## {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"**Duration:** {summary['duration_human']}",
        f"**Hormone snapshots:** {summary['hormone_snapshots']}",
        f"**Tensor snapshots (130D):** {summary['tensor_snapshots']}",
        f"**Words taught:** {summary['words_learned']}",
        f"**Conversations:** {summary['conversations']}",
        f"**Puzzles attempted:** {summary['puzzles_attempted']}",
        "",
        "---",
        "",
        "## Word Learning Results",
        "",
    ]

    for wr in telemetry.word_results:
        word = wr["word"]
        h_aff = wr.get("hormone_affinity", {})
        passes = wr.get("passes", [])
        feel_score = next((p["score"] for p in passes if p["pass_type"] == "feel"), 0)
        recog_score = next((p["score"] for p in passes if p["pass_type"] == "recognize"), 0)
        prod_score = next((p["score"] for p in passes if p["pass_type"] == "produce"), 0)

        # Collect all hormone changes across passes
        all_changes = {}
        for p in passes:
            for h, info in p.get("hormone_changes", {}).items():
                if isinstance(info, dict):
                    delta = info.get("delta", 0)
                else:
                    delta = info
                all_changes[h] = all_changes.get(h, 0) + delta

        top_changes = sorted(all_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        change_str = ", ".join(f"{h}={d:+.3f}" for h, d in top_changes) if top_changes else "(subtle)"

        lines.append(f"- **{word}** ({wr.get('word_type','?')}, cycle {wr['cycle']}) "
                     f"→ Feel:{feel_score:.0%} Recognize:{recog_score:.0%} Produce:{prod_score:.0%} "
                     f"| {change_str}")

    lines.extend(["", "---", "", "## Conversation Results", ""])

    for cr in telemetry.convo_results:
        persona = cr["persona"]
        lines.append(f"### {persona} (Cycle {cr['cycle']})")
        for turn in cr.get("turns", []):
            prompt = turn.get("prompt", "")
            response = turn.get("response", "")
            changes = turn.get("hormone_changes", {})
            change_str = ", ".join(f"{h}={d:+.3f}" for h, d in changes.items()) if changes else "(subtle)"
            lines.append(f"- **Q:** {prompt}")
            lines.append(f"  **A:** {response[:200]}...")
            lines.append(f"  **Hormones:** {change_str}")
            lines.append("")

    lines.extend(["---", "", "## Puzzle Results", ""])

    for pr in telemetry.puzzle_results:
        lines.append(f"- **{pr['puzzle_id']}** ({pr['category']}, cycle {pr['cycle']}): "
                     f"Correct={pr.get('correct', '?')} | "
                     f"Changes: {', '.join(f'{h}={d:+.3f}' for h, d in pr.get('hormone_changes', {}).items()) or '(subtle)'}")

    # Tensor pattern analysis
    lines.extend(["", "---", "", "## Trinity Tensor Patterns by Activity", ""])

    activity_tensors = {}
    for ts in telemetry.tensor_snapshots:
        act = ts.get("activity", "unknown").split(":")[0]
        if act not in activity_tensors:
            activity_tensors[act] = []
        activity_tensors[act].append(ts)

    lines.append(f"| Activity | Snapshots | iBody | iMind | iSpirit | oBody | oMind | oSpirit |")
    lines.append(f"|----------|-----------|-------|-------|---------|-------|-------|---------|")

    for act, snapshots in sorted(activity_tensors.items()):
        n = len(snapshots)
        if n == 0:
            continue
        avg_ib = sum(s.get("inner_body_coh", 0) for s in snapshots) / n
        avg_im = sum(s.get("inner_mind_coh", 0) for s in snapshots) / n
        avg_is = sum(s.get("inner_spirit_coh", 0) for s in snapshots) / n
        avg_ob = sum(s.get("outer_body_coh", 0) for s in snapshots) / n
        avg_om = sum(s.get("outer_mind_coh", 0) for s in snapshots) / n
        avg_os = sum(s.get("outer_spirit_coh", 0) for s in snapshots) / n
        lines.append(f"| {act:25s} | {n:9d} | {avg_ib:.3f} | {avg_im:.3f} | {avg_is:.3f} "
                     f"| {avg_ob:.3f} | {avg_om:.3f} | {avg_os:.3f} |")

    # Hormonal journey summary
    lines.extend(["", "---", "", "## Hormonal Journey", ""])

    if telemetry.hormone_snapshots:
        first = telemetry.hormone_snapshots[0]["hormones"]
        last = telemetry.hormone_snapshots[-1]["hormones"]
        lines.append("| Hormone | Start | End | Change |")
        lines.append("|---------|-------|-----|--------|")
        for h in sorted(first.keys()):
            s = first.get(h, 0)
            e = last.get(h, 0)
            d = e - s
            lines.append(f"| {h:15s} | {s:.3f} | {e:.3f} | {d:+.3f} |")

    lines.extend(["", "---", "",
                  "*Report generated automatically by developmental_day.py*",
                  f"*Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"])

    return "\n".join(lines)


# ── Main Orchestrator ─────────────────────────────────────────────

async def run_developmental_day(num_cycles: int = 4):
    """Run the full developmental day experiment."""
    start_time = time.time()

    log.info("=" * 60)
    log.info("  TITAN DEVELOPMENTAL DAY — %d cycles (~%.1f hours)",
             num_cycles, num_cycles * 1.5)
    log.info("=" * 60)

    # Initialize
    memory = InnerMemoryStore(db_path="./data/inner_memory.db")
    plugin = LanguageLearningExperience(inner_memory=memory)

    # Seed vocabulary
    for word, recipe in plugin._word_data.items():
        if not memory.get_word(word):
            memory.store_word(
                word=word,
                word_type=recipe.get("word_type", "unknown"),
                stage=recipe.get("stage", 1),
                felt_tensor=_flatten_perturbation(recipe.get("perturbation", {})),
                hormone_pattern=recipe.get("hormone_affinity", {}),
            )

    telemetry = TelemetryCollector()
    puzzles_remaining = list(PUZZLES)
    random.shuffle(puzzles_remaining)

    async with httpx.AsyncClient() as client:
        # Verify Titan is running
        try:
            r = await client.get(f"{API_BASE}/health", timeout=10)
            if r.status_code != 200:
                log.error("Titan not healthy. Aborting.")
                return
        except Exception as e:
            log.error("Cannot reach Titan: %s. Aborting.", e)
            return

        log.info("Titan online. Starting telemetry collection...")

        # Start background telemetry collection
        telemetry_task = asyncio.create_task(
            telemetry.background_collection(client))

        try:
            for cycle in range(1, num_cycles + 1):
                cycle_start = time.time()
                log.info("")
                log.info("=" * 60)
                log.info("  CYCLE %d/%d — Starting", cycle, num_cycles)
                log.info("=" * 60)

                # 1. Word Learning Phase (15 min)
                words = WORD_CYCLES[(cycle - 1) % len(WORD_CYCLES)]
                await run_word_learning(client, telemetry, plugin, memory,
                                        words, cycle)

                # 2. Integration Rest (10 min)
                log.info("[Cycle %d] Integration rest after word learning...", cycle)
                await rest_with_monitoring(client, telemetry, SHORT_REST,
                                           f"post_words_cycle{cycle}")

                # 3. Gentle Conversation (15 min)
                persona = PERSONAS[(cycle - 1) % len(PERSONAS)]
                await run_conversation(client, telemetry, persona, cycle)

                # 4. Integration Rest (10 min)
                log.info("[Cycle %d] Integration rest after conversation...", cycle)
                await rest_with_monitoring(client, telemetry, SHORT_REST,
                                           f"post_convo_cycle{cycle}")

                # 5. ARC Puzzle (10 min)
                cycle_puzzles = puzzles_remaining[:2]
                puzzles_remaining = puzzles_remaining[2:]
                if not puzzles_remaining:
                    puzzles_remaining = list(PUZZLES)
                    random.shuffle(puzzles_remaining)
                await run_puzzles(client, telemetry, cycle_puzzles, cycle)

                # 6. Deep Rest (30 min)
                log.info("[Cycle %d] Deep rest for integration...", cycle)
                await rest_with_monitoring(client, telemetry, DEEP_REST,
                                           f"deep_rest_cycle{cycle}")

                cycle_duration = time.time() - cycle_start
                log.info("[Cycle %d] Complete in %.1f min", cycle,
                         cycle_duration / 60)

        except KeyboardInterrupt:
            log.info("Interrupted by user — generating report with data so far...")
        except Exception as e:
            log.error("Unexpected error: %s — generating report...", e)
        finally:
            # Stop telemetry
            telemetry.stop()
            await asyncio.sleep(2)
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass

        # Generate report
        total_duration = time.time() - start_time
        log.info("")
        log.info("=" * 60)
        log.info("  DEVELOPMENTAL DAY COMPLETE — %.1f hours", total_duration / 3600)
        log.info("=" * 60)

        report = generate_report(telemetry)
        report_path = f"titan-docs/REPORT_developmental_day_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_path, "w") as f:
            f.write(report)
        log.info("Report saved to %s", report_path)

        # Save raw telemetry data
        data_path = f"data/developmental_day_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(data_path, "w") as f:
            json.dump({
                "summary": telemetry.get_summary(),
                "hormone_snapshots": telemetry.hormone_snapshots,
                "tensor_snapshots": telemetry.tensor_snapshots,
                "consciousness_snapshots": telemetry.consciousness_snapshots,
                "sphere_clock_snapshots": telemetry.sphere_clock_snapshots,
                "activity_log": telemetry.activity_log,
                "word_results": telemetry.word_results,
                "convo_results": telemetry.convo_results,
                "puzzle_results": telemetry.puzzle_results,
            }, f, indent=2)
        log.info("Raw telemetry saved to %s", data_path)

        # Print summary
        s = telemetry.get_summary()
        log.info("Summary: %d hormone snapshots, %d tensor snapshots, "
                 "%d words, %d conversations, %d puzzles",
                 s["hormone_snapshots"], s["tensor_snapshots"],
                 s["words_learned"], s["conversations"], s["puzzles_attempted"])

        vs = memory.get_vocab_stats()
        log.info("Vocabulary: %d words, avg confidence: %.3f, phases: %s",
                 vs["total_words"], vs["avg_confidence"], vs["phases"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Titan Developmental Day")
    parser.add_argument("--cycles", type=int, default=4,
                        help="Number of cycles (default: 4, each ~90 min)")
    args = parser.parse_args()

    asyncio.run(run_developmental_day(num_cycles=args.cycles))
