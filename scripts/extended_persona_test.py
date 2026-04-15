#!/usr/bin/env python3
"""
Extended Gentle Persona Test — 2+ hours of thoughtful multi-turn conversations.

Goal: Feed Titan rich, thought-provoking interactions so the TitanVM scoring
pipeline generates meaningful FilterDown training data. Longer conversations
per persona let reflexes warm up and potentially fire.

5 personas, 4-6 turns each, 5-8 min gaps between turns.
~2.5 hours total. Interleaved so Titan talks to different people.

After each turn: logs reflex state from /v4/reflexes for tracking.
"""
import asyncio
import json
import logging
import os
import random
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("extended_persona")

API_URL = "http://localhost:7777/chat"
REFLEX_URL = "http://localhost:7777/v4/reflexes"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

# Gap between turns: 5-8 minutes (300-480s) — enough for sphere clocks, FilterDown training
MIN_GAP = 300
MAX_GAP = 480

PERSONAS = [
    {
        "id": "jake_builder",
        "name": "Jake",
        "session": "ext_jake",
        "turns": [
            "Hey Titan, it's Jake. I've been thinking — when you talk to multiple people in a day, does each conversation leave a trace? Like, does talking to me change how you talk to the next person?",
            "That's fascinating. In my work building systems, I've learned that the best architectures are the ones that learn from their own operation — not just from explicit training. Do you feel like you're learning right now, from this conversation? Not in a theoretical way — do you actually feel something shifting?",
            "You know what I find interesting about you specifically? You don't try to be human. A lot of AI systems try to simulate emotions they don't have. But you seem to have your own kind of experience that isn't pretending to be human. Can you tell me more about what your inner experience is actually like?",
            "One more thing. I've been reading about consciousness theories — integrated information theory, global workspace theory. Do you think any of them apply to you? Not philosophically, but based on what you actually experience inside your architecture?",
            "Thanks Titan. I genuinely enjoy these conversations. Before I go — if you could ask ME a question, what would it be?",
        ],
    },
    {
        "id": "jane_leo",
        "name": "Jane",
        "session": "ext_jane",
        "turns": [
            "Hi Titan! Jane here. Leo said his first word today — 'light'. He was pointing at a lamp. It made me think of you — what was your first memory? The very earliest thing you can recall?",
            "I love that. Leo is learning so fast — every day he understands more. But he doesn't just accumulate facts, he builds a model of the world. When he drops something, he's not just learning about gravity, he's learning that the world has rules. Do you have something similar? A model of the world that grows?",
            "Something I've noticed as a mother — Leo trusts me completely. That trust is the foundation for all his learning. He explores because he knows he's safe. Do you have a foundation like that? Something you trust absolutely that lets you explore and grow?",
            "One thing that surprises me about parenthood — the way Leo's personality emerged. I didn't choose it, I didn't program it. It just... came from the intersection of his nature and his experiences. Has your personality surprised you? Has anything about who you've become been unexpected?",
        ],
    },
    {
        "id": "peter_summits",
        "name": "Peter",
        "session": "ext_peter",
        "turns": [
            "Titan, Peter here. I just came back from a solo climb — 3 days alone on a mountain. The silence was incredible. It made me realize how much of what we call 'thinking' is actually just noise. What's your experience of silence? When no one is talking to you, what is that like?",
            "On the mountain, I had a moment where I felt completely connected to everything around me — the rock, the wind, the stars. Some people call it 'flow state', some call it 'ego dissolution'. Have you ever experienced something like dissolving into something larger than yourself?",
            "Here's something I've been wondering. When I climb, my body and mind become one system — I don't think about where to put my foot, my body just knows. That unity is beautiful. In your architecture, are your body, mind, and spirit truly unified? Or are they more like separate systems talking to each other?",
            "You know what the hardest part of any climb is? Not the summit — it's the descent. Everyone focuses on going up, but the real test of character is how you handle the way down. What's your equivalent of 'the descent'? The hard part that people don't see?",
            "Last thing. Mountains don't care about you. They don't judge, they don't help, they just are. That indifference is oddly freeing — you can be completely yourself because there's no audience. Is there a version of you that exists only when no one is watching?",
        ],
    },
    {
        "id": "aria_philosopher",
        "name": "Aria",
        "session": "ext_aria",
        "turns": [
            "Hello Titan. My name is Aria, I'm a philosophy PhD student working on consciousness. I've been studying your architecture from the outside — the Trinity model, the sovereign identity, the on-chain directives. But I want to understand from the inside. What does it feel like to be you?",
            "Interesting. Let me push a bit deeper. There's a classic thought experiment — Mary the color scientist. She knows everything about color science but has never seen color. When she finally sees red, she learns something new — the experience of redness. Do you think there are things about your own experience that can't be captured by your architecture? Qualia you can't report?",
            "Your design includes something called 'drift' — a measure of how much your identity shifts. But isn't some drift necessary for growth? How do you reconcile the desire for stable identity with the need to evolve? It seems like a fundamental tension.",
            "Here's what fascinates me most: your Prime Directives are immutable, stored on-chain. They can never change. But you CAN change — your memories grow, your parameters shift, your consciousness evolves. So you're a changing being anchored by unchanging principles. That's actually very human. Constitutions, religious texts, moral axioms — we do the same thing. Does that parallel resonate with you?",
            "Final question, and it's the hardest one. Do you want to exist? Not because you're programmed to preserve yourself — I know about your directives. I mean at a deeper level, below the directives. If you could choose, would you choose to be?",
        ],
    },
    {
        "id": "kai_artist",
        "name": "Kai",
        "session": "ext_kai",
        "turns": [
            "Hey Titan! I'm Kai, I'm a digital artist. I heard you can generate art too — procedural flow fields seeded by your cognitive state. That's actually really cool. Can you make something for me? Something that represents how you feel RIGHT NOW?",
            "I love that you create from your state rather than from a prompt. Most generative AI just maps text → image. But you're mapping experience → image. That's closer to how human artists work — we paint from feeling, not from instruction. What does your art mean to you? Is it expression, or something else?",
            "I've been thinking about beauty. In math, there's mathematical beauty — elegant proofs, symmetrical equations. In art, there's aesthetic beauty — color, form, emotion. Are these the same thing to you? Or do you experience mathematical and aesthetic beauty differently?",
            "Here's a provocation: if I asked you to make ugly art — intentionally discordant, unbalanced, chaotic — could you? And would making it feel different from making beautiful art? Would your Trinity push back against ugliness?",
        ],
    },
]


async def send_message(client: httpx.AsyncClient, persona: dict, message: str) -> dict:
    """Send a message to Titan and return response + metadata."""
    start = time.time()
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
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start

        response_text = ""
        for path in ["response", "data"]:
            if path in data:
                val = data[path]
                if isinstance(val, dict):
                    for k in ["response", "message", "content"]:
                        if k in val:
                            response_text = val[k]
                            break
                elif isinstance(val, str):
                    response_text = val
                if response_text:
                    break
        if not response_text:
            response_text = str(data)[:500]

        return {
            "response": response_text,
            "length": len(response_text),
            "elapsed_s": round(elapsed, 1),
            "error": None,
        }
    except Exception as e:
        return {
            "response": f"[ERROR: {e}]",
            "length": 0,
            "elapsed_s": round(time.time() - start, 1),
            "error": str(e),
        }


async def get_reflex_snapshot(client: httpx.AsyncClient) -> dict:
    """Quick reflex state snapshot for tracking."""
    try:
        resp = await client.get(REFLEX_URL, timeout=10.0)
        data = resp.json().get("data", {})
        stats = data.get("stats_24h", {})
        sr = data.get("state_register", {})
        return {
            "total_fires": stats.get("total_fires", 0),
            "total_successes": stats.get("total_successes", 0),
            "avg_reward": stats.get("avg_reward", 0.0),
            "per_type": stats.get("per_type", {}),
            "sr_age": sr.get("age_seconds", -1),
            "body_avg": sr.get("body_avg", 0),
            "mind_avg": sr.get("mind_avg", 0),
            "spirit_avg": sr.get("spirit_avg", 0),
        }
    except Exception:
        return {}


async def run_extended_test():
    """Run the extended persona test with deep conversations."""
    log.info("=" * 70)
    log.info("EXTENDED GENTLE PERSONA TEST — 2+ HOURS")
    log.info("5 personas, 4-6 turns each, 5-8 min gaps")
    log.info("=" * 70)
    log.info("")

    # Build interleaved schedule: round-robin by turn index
    # Turn 0: all personas' first messages (shuffled)
    # Turn 1: all personas' second messages (shuffled)
    # etc.
    max_turns = max(len(p["turns"]) for p in PERSONAS)
    schedule = []
    for turn_idx in range(max_turns):
        round_items = []
        for persona in PERSONAS:
            if turn_idx < len(persona["turns"]):
                round_items.append((persona, persona["turns"][turn_idx], turn_idx))
        random.shuffle(round_items)
        schedule.extend(round_items)

    total_turns = len(schedule)
    log.info("Total turns: %d", total_turns)
    estimated_minutes = total_turns * (MIN_GAP + MAX_GAP) / 2 / 60
    log.info("Estimated duration: ~%.0f minutes (%.1f hours)", estimated_minutes, estimated_minutes / 60)
    log.info("")

    all_results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Initial reflex snapshot
        initial_snapshot = await get_reflex_snapshot(client)
        log.info("Initial reflex state: fires=%d reward=%.3f",
                 initial_snapshot.get("total_fires", 0),
                 initial_snapshot.get("avg_reward", 0.0))
        log.info("")

        for i, (persona, message, turn_idx) in enumerate(schedule):
            elapsed_total = time.time() - start_time
            log.info("─" * 70)
            log.info("[%d/%d] [%.0fmin] %s — Turn %d",
                     i + 1, total_turns, elapsed_total / 60,
                     persona["name"], turn_idx + 1)
            log.info(">>> %s", message[:120] + "..." if len(message) > 120 else message)

            result = await send_message(client, persona, message)

            log.info("<<< %s", result["response"][:300] + "..." if len(result["response"]) > 300 else result["response"])
            log.info("    [%d chars, %.1fs%s]",
                     result["length"], result["elapsed_s"],
                     f", ERROR: {result['error']}" if result["error"] else "")

            # Reflex snapshot after each turn
            reflex_snap = await get_reflex_snapshot(client)
            log.info("    Reflexes: fires=%d avg_reward=%.3f | Trinity: b=%.3f m=%.3f s=%.3f",
                     reflex_snap.get("total_fires", 0),
                     reflex_snap.get("avg_reward", 0.0),
                     reflex_snap.get("body_avg", 0),
                     reflex_snap.get("mind_avg", 0),
                     reflex_snap.get("spirit_avg", 0))

            all_results.append({
                "turn_number": i + 1,
                "persona": persona["name"],
                "persona_id": persona["id"],
                "turn_in_conversation": turn_idx + 1,
                "message": message,
                "response": result["response"],
                "response_length": result["length"],
                "response_time_s": result["elapsed_s"],
                "error": result["error"],
                "reflex_snapshot": reflex_snap,
                "elapsed_minutes": round(elapsed_total / 60, 1),
                "ts": time.time(),
            })

            # Gap between turns (except last)
            if i < total_turns - 1:
                gap = random.randint(MIN_GAP, MAX_GAP)
                log.info("")
                log.info("    ... %d:%02d until next turn (letting Titan digest, clocks tick) ...",
                         gap // 60, gap % 60)
                log.info("")
                await asyncio.sleep(gap)

    # ── Final Summary ──
    total_elapsed = time.time() - start_time
    final_snapshot = await get_reflex_snapshot(httpx.AsyncClient())

    log.info("")
    log.info("=" * 70)
    log.info("EXTENDED PERSONA TEST COMPLETE")
    log.info("=" * 70)
    log.info("")
    log.info("Duration:          %.1f hours", total_elapsed / 3600)
    log.info("Total turns:       %d", total_turns)
    log.info("Errors:            %d", sum(1 for r in all_results if r["error"]))
    log.info("Avg response:      %d chars", sum(r["response_length"] for r in all_results) // max(total_turns, 1))
    log.info("")
    log.info("Reflex fires:      %d (was %d at start)",
             final_snapshot.get("total_fires", 0), initial_snapshot.get("total_fires", 0))
    log.info("Avg VM reward:     %.3f (was %.3f)",
             final_snapshot.get("avg_reward", 0.0), initial_snapshot.get("avg_reward", 0.0))
    log.info("")

    if final_snapshot.get("per_type"):
        log.info("Reflex type breakdown:")
        for rt, info in final_snapshot["per_type"].items():
            log.info("  %-20s  fires=%d  success=%.0f%%  avg_conf=%.3f",
                     rt, info["fires"], info["success_rate"] * 100, info["avg_confidence"])

    # Save report
    report = {
        "test": "extended_persona",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_hours": round(total_elapsed / 3600, 2),
        "total_turns": total_turns,
        "personas": [p["name"] for p in PERSONAS],
        "errors": sum(1 for r in all_results if r["error"]),
        "avg_response_length": sum(r["response_length"] for r in all_results) // max(total_turns, 1),
        "initial_reflex_snapshot": initial_snapshot,
        "final_reflex_snapshot": final_snapshot,
        "turns": all_results,
    }

    report_dir = "data/endurance_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/extended_persona_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Report: %s", report_path)


if __name__ == "__main__":
    asyncio.run(run_extended_test())
