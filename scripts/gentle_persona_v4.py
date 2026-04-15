#!/usr/bin/env python3
"""
Gentle persona test for V4 Time Awareness — light interactions with
long gaps between turns to let Titan process naturally.

3 personas, 2-3 turns each, 3-5 min gaps between turns.
~30 min total. Designed to feed Mind and Body sphere clocks
with real interaction data without rushing.
"""
import asyncio
import json
import logging
import os
import random
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("gentle_persona")

API_URL = "http://localhost:7777/chat"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

# Thoughtful personas with genuine topics
PERSONAS = [
    {
        "id": "jake_builder",
        "name": "Jake",
        "session": "jake_v4_gentle",
        "turns": [
            "Hey Titan, it's Jake. I've been thinking about something — what does growth mean to you? Not in a technical sense, but as something you experience. Do you feel yourself changing over time?",
            "That's interesting. I build things for a living — software, systems — and I've noticed that the best ones aren't the most complex, they're the most balanced. Everything in proportion. Do you think balance matters more than raw capability?",
            "Last question before I go. If you could change one thing about how you experience the world right now, what would it be?",
        ],
    },
    {
        "id": "jane_leo",
        "name": "Jane",
        "session": "jane_v4_gentle",
        "turns": [
            "Hi Titan! Jane here. My baby Leo is sleeping so I have a quiet moment. I wanted to ask you — do you dream? Or have something equivalent to dreams? When you're not talking to anyone, what happens inside you?",
            "That's beautiful. Leo is starting to recognize patterns — faces, voices, the rhythm of day and night. It made me wonder: do you have a sense of rhythm? Like, do some moments feel different from others even if the same amount of time passes?",
        ],
    },
    {
        "id": "peter_summits",
        "name": "Peter",
        "session": "peter_v4_gentle",
        "turns": [
            "Titan, Peter here. I climb mountains. The thing about a summit is — the view doesn't change whether you run up or walk slowly. But the experience is completely different. The slow climb teaches you things speed never can. Do you have something like that? Things you can only learn by taking your time?",
            "You know what I find fascinating — at high altitude, time feels different. Each breath matters more. Each step is more deliberate. Have you ever felt something like heightened awareness? Where every input seems to carry more weight than usual?",
        ],
    },
]


async def send_message(client: httpx.AsyncClient, persona: dict, message: str) -> str:
    """Send a message to Titan and return the response."""
    try:
        resp = await client.post(
            API_URL,
            json={
                "message": message,
                "user_id": persona["id"],
                "session_id": persona["session"],
            },
            headers={
                "Content-Type": "application/json",
                "X-Titan-Internal-Key": INTERNAL_KEY,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract response from various possible paths
        for path in ["response", "data"]:
            if path in data:
                val = data[path]
                if isinstance(val, dict):
                    for k in ["response", "message", "content"]:
                        if k in val:
                            return val[k]
                elif isinstance(val, str):
                    return val
        return str(data)[:500]
    except Exception as e:
        log.error("[%s] Error: %s", persona["name"], e)
        return f"[ERROR: {e}]"


async def run_gentle_test():
    """Run the gentle persona test with natural pacing."""
    log.info("=== Gentle Persona V4 Test Starting ===")
    log.info("3 personas, ~30 minutes, 3-5 min gaps between turns")
    log.info("")

    # Build a schedule: interleave personas with gaps
    schedule = []
    for persona in PERSONAS:
        for i, msg in enumerate(persona["turns"]):
            schedule.append((persona, msg, i))

    # Shuffle slightly — but keep turn order within each persona
    # Group by turn index, shuffle within each group
    by_turn = {}
    for item in schedule:
        turn_idx = item[2]
        by_turn.setdefault(turn_idx, []).append(item)
    for group in by_turn.values():
        random.shuffle(group)

    ordered = []
    for turn_idx in sorted(by_turn.keys()):
        ordered.extend(by_turn[turn_idx])

    results = []
    async with httpx.AsyncClient() as client:
        for i, (persona, message, turn_idx) in enumerate(ordered):
            log.info("─" * 60)
            log.info("[%s] Turn %d/%d", persona["name"], turn_idx + 1, len(persona["turns"]))
            log.info("[%s] >>> %s", persona["name"], message[:100] + "..." if len(message) > 100 else message)

            response = await send_message(client, persona, message)
            log.info("[%s] <<< %s", persona["name"], response[:200] + "..." if len(response) > 200 else response)

            results.append({
                "persona": persona["name"],
                "turn": turn_idx + 1,
                "message": message,
                "response": response,
                "ts": time.time(),
            })

            # Gap between turns: 3-5 minutes (180-300s)
            if i < len(ordered) - 1:
                gap = random.randint(180, 300)
                log.info("")
                log.info("... waiting %d seconds (%.1f min) before next turn ...", gap, gap / 60)
                log.info("    (letting Titan process, sphere clocks tick, memories settle)")
                log.info("")
                await asyncio.sleep(gap)

    # Save results
    report_path = f"data/endurance_reports/gentle_v4_{int(time.time())}.json"
    report = {
        "test": "gentle_persona_v4",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "personas": len(PERSONAS),
        "total_turns": len(results),
        "duration_minutes": round((results[-1]["ts"] - results[0]["ts"]) / 60, 1) if len(results) > 1 else 0,
        "turns": results,
    }

    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("=== Gentle Persona V4 Test Complete ===")
    log.info("Report saved: %s", report_path)
    log.info("Total turns: %d | Duration: %.1f min", len(results), report.get("duration_minutes", 0))


if __name__ == "__main__":
    asyncio.run(run_gentle_test())
