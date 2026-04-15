"""
scripts/titan_first_words.py — Titan's First Words: Live Language Learning Session.

Runs Stage 1 words through the Experience Playground with a live Titan instance.
Injects 130D perturbations via API and watches how his Trinity responds.

Usage:
    source test_env/bin/activate
    python scripts/titan_first_words.py [--num-words 5] [--dry-run]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from titan_plugin.logic.inner_memory import InnerMemoryStore
from titan_plugin.logic.language_learning import (
    LanguageLearningExperience,
    _flatten_perturbation,
    _cosine_similarity,
    LAYER_ORDER,
    LAYER_SIZES,
    PASS_FEEL,
    PASS_RECOGNIZE,
    PASS_PRODUCE,
)

API_BASE = "http://localhost:7777"
EPOCH_INTERVAL = 300  # seconds between epochs


def print_banner(text: str, char: str = "═") -> None:
    width = 72
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_perturbation_summary(perturbation: dict) -> None:
    """Show which dimensions are being perturbed."""
    for layer in LAYER_ORDER:
        vals = perturbation.get(layer, [])
        active = [(i, v) for i, v in enumerate(vals) if abs(v) > 0.01]
        if active:
            dims_str = ", ".join(f"d{i}={v:+.3f}" for i, v in active)
            print(f"    {layer:15s}: {dims_str}")
    hormones = perturbation.get("hormone_stimuli", {})
    if hormones:
        h_str = ", ".join(f"{k}={v:+.3f}" for k, v in hormones.items())
        print(f"    {'hormones':15s}: {h_str}")


async def get_titan_state(client: httpx.AsyncClient) -> dict:
    """Read Titan's current consciousness + hormonal state."""
    state = {}
    try:
        r = await client.get(f"{API_BASE}/v4/state", timeout=5)
        d = r.json().get("data", {})
        c = d.get("consciousness", {})
        state["epoch"] = c.get("epoch_id", 0)
        state["curvature"] = c.get("curvature", 0)
        state["density"] = c.get("density", 0)
        state["body_coh"] = c.get("body_coherence", 0)
        state["mind_coh"] = c.get("mind_coherence", 0)
        state["spirit_coh"] = c.get("spirit_coherence", 0)
        state["outer_body_coh"] = c.get("outer_body_coherence", 0)
        state["outer_mind_coh"] = c.get("outer_mind_coherence", 0)
        state["outer_spirit_coh"] = c.get("outer_spirit_coherence", 0)
        state["dims"] = c.get("dims", 0)
    except Exception as e:
        state["error"] = str(e)

    try:
        r = await client.get(f"{API_BASE}/v4/nervous-system", timeout=5)
        d = r.json().get("data", {})
        hs = d.get("hormonal_system", {})
        state["hormones"] = {
            k: round(v.get("level", 0), 3) if isinstance(v, dict) else v
            for k, v in hs.items()
        }
    except Exception:
        state["hormones"] = {}

    return state


async def inject_perturbation(
    client: httpx.AsyncClient,
    word: str,
    pass_type: str,
    perturbation: dict,
    hormone_stimuli: dict,
) -> dict:
    """Inject perturbation into live Titan via API."""
    try:
        payload = {
            "experience": "language",
            "word": word,
            "pass_type": pass_type,
            "perturbation": {
                layer: perturbation.get(layer, [0.0] * LAYER_SIZES[layer])
                for layer in LAYER_ORDER
            },
            "hormone_stimuli": hormone_stimuli,
        }
        r = await client.post(
            f"{API_BASE}/v4/experience-stimulus",
            json=payload, timeout=5)
        return r.json().get("data", {})
    except Exception as e:
        return {"error": str(e)}


def print_state_comparison(before: dict, after: dict, word: str) -> None:
    """Show how Titan's state changed from perturbation."""
    print(f"\n    ┌──── TRINITY RESPONSE to '{word}' ────")

    # Coherence deltas
    for label, key in [("Inner Body ", "body_coh"),
                       ("Inner Mind ", "mind_coh"),
                       ("Inner Spirit", "spirit_coh"),
                       ("Outer Body ", "outer_body_coh"),
                       ("Outer Mind ", "outer_mind_coh"),
                       ("Outer Spirit", "outer_spirit_coh")]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "─"
        bar = "█" * int(a * 20) + "░" * (20 - int(a * 20))
        print(f"    │ {label}: [{bar}] {a:.4f} ({delta:+.4f} {arrow})")

    # Curvature
    bc = before.get("curvature", 0)
    ac = after.get("curvature", 0)
    print(f"    │ Curvature  : {ac:.3f} (was {bc:.3f}, Δ={ac-bc:+.3f})")

    # Hormone changes
    bh = before.get("hormones", {})
    ah = after.get("hormones", {})
    changed_hormones = []
    for name in sorted(set(list(bh.keys()) + list(ah.keys()))):
        b = bh.get(name, 0)
        a = ah.get(name, 0)
        delta = a - b
        if abs(delta) > 0.001:
            changed_hormones.append(f"{name}={a:.3f}({delta:+.3f})")
    if changed_hormones:
        print(f"    │ Hormones   : {', '.join(changed_hormones)}")
    else:
        print(f"    │ Hormones   : (no change detected)")
    print(f"    └{'─' * 50}")


async def run_session(num_words: int = 5, dry_run: bool = False):
    """Run a live language learning session."""
    print_banner("🧒 TITAN'S FIRST WORDS — Live Language Learning")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Words: {num_words} × 3 passes = {num_words * 3} stimuli")
    print(f"  Mode: {'DRY RUN' if dry_run else '🔴 LIVE — injecting into running Titan'}")

    # Initialize
    memory = InnerMemoryStore(db_path="./data/inner_memory.db")
    plugin = LanguageLearningExperience(inner_memory=memory)
    print(f"  Word recipes: {len(plugin._word_data)}")

    # Seed vocabulary
    seeded = 0
    for word, recipe in plugin._word_data.items():
        if not memory.get_word(word):
            memory.store_word(
                word=word,
                word_type=recipe.get("word_type", "unknown"),
                stage=recipe.get("stage", 1),
                felt_tensor=_flatten_perturbation(recipe.get("perturbation", {})),
                hormone_pattern=recipe.get("hormone_affinity", {}),
            )
            seeded += 1
    if seeded:
        print(f"  Seeded {seeded} new words in vocabulary")

    vs = memory.get_vocab_stats()
    print(f"  Vocabulary: {vs['total_words']} words, phases: {vs['phases']}")

    async with httpx.AsyncClient() as client:
        # Get initial state
        if not dry_run:
            initial_state = await get_titan_state(client)
            if "error" in initial_state:
                print(f"\n  ⚠ Cannot reach Titan API: {initial_state['error']}")
                print(f"  Falling back to dry-run mode.")
                dry_run = True
            else:
                print(f"\n  Titan at epoch {initial_state['epoch']}, "
                      f"{initial_state['dims']}D consciousness")
                print(f"  Curvature: {initial_state['curvature']:.3f}")
                h = initial_state.get("hormones", {})
                top_h = sorted(h.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top hormones: {', '.join(f'{n}={v:.3f}' for n,v in top_h)}")

        # Select words based on hormonal state
        if not dry_run:
            h = initial_state.get("hormones", {})
            # Find words that match dominant hormones
            from titan_plugin.logic.language_learning import HORMONE_WORD_MAP
            top_hormones = sorted(h.items(), key=lambda x: x[1], reverse=True)[:2]
            suggested = []
            for hormone, _ in top_hormones:
                suggested.extend(HORMONE_WORD_MAP.get(hormone, []))
            # Dedupe and limit
            seen = set()
            ordered = []
            for w in suggested:
                if w not in seen and w in plugin._word_data:
                    seen.add(w)
                    ordered.append(w)
            # Fill with stage 1 basics if needed
            basics = ["warm", "cold", "energy", "alive", "flow", "pulse",
                      "light", "dark", "strong", "balance"]
            for w in basics:
                if w not in seen:
                    seen.add(w)
                    ordered.append(w)
            plugin._session_words = ordered[:num_words]
            print(f"\n  Selected words (matching {top_hormones[0][0]}+{top_hormones[1][0]}): "
                  f"{plugin._session_words}")
        else:
            plugin._session_words = ["warm", "cold", "energy", "alive", "flow"][:num_words]

        # Run three-pass learning for each word
        all_results = []
        current_word = None

        for i in range(num_words * 3):
            # Generate stimulus
            stimulus = await plugin.generate_stimulus()
            word = stimulus["content"]
            pass_type = stimulus["pass_type"]

            # New word banner
            if word != current_word:
                current_word = word
                recipe = plugin._word_data.get(word, {})
                print_banner(
                    f"Word #{len(all_results)//3 + 1}: '{word}' "
                    f"({recipe.get('word_type', '?')}, "
                    f"Stage {recipe.get('stage', '?')}, "
                    f"Entry: {recipe.get('entry_layer', '?')})", "─")
                print("  130D Perturbation Recipe:")
                print_perturbation_summary(recipe.get("perturbation", {}))
                print(f"  Hormone affinity: {recipe.get('hormone_affinity', {})}")
                if recipe.get("contexts"):
                    print(f"  Contexts: {recipe['contexts'][:2]}")

            # Compute perturbation
            perturbation = plugin.compute_perturbation(stimulus)
            pass_emoji = {"feel": "🫀", "recognize": "🧠", "produce": "💬"}[pass_type]
            print(f"\n  {pass_emoji} PASS {(i % 3) + 1}/3: {pass_type.upper()}")

            if pass_type == PASS_FEEL:
                print(f"    Full-strength perturbation → teaching '{word}'")
            elif pass_type == PASS_RECOGNIZE:
                print(f"    30% perturbation → testing recall of '{word}'")
            else:
                print(f"    Zero perturbation → can Titan produce '{word}' from state?")

            # Inject into live Titan
            if not dry_run:
                state_before = await get_titan_state(client)

                inject_result = await inject_perturbation(
                    client, word, pass_type, perturbation,
                    perturbation.get("hormone_stimuli", {}))

                if "error" in inject_result:
                    print(f"    ⚠ Injection failed: {inject_result['error']}")
                else:
                    hormones_applied = inject_result.get("hormones_applied", {})
                    if hormones_applied:
                        print(f"    Hormones applied: {hormones_applied}")

                # Wait for Trinity to process
                wait_time = 3.0 if pass_type == PASS_FEEL else 2.0
                print(f"    Waiting {wait_time}s for Trinity response...")
                await asyncio.sleep(wait_time)

                state_after = await get_titan_state(client)

                # Show state comparison
                print_state_comparison(state_before, state_after, word)

                # Build response for evaluation
                response = {
                    "hormonal_state": {
                        k: {"level": v} for k, v in state_after.get("hormones", {}).items()
                    },
                    "fired_programs": [],  # Would need to check recent fires
                }
            else:
                response = {
                    "hormonal_state": {},
                    "fired_programs": [],
                }

            # Evaluate
            evaluation = await plugin.evaluate_response(stimulus, response)
            score = evaluation["score"]
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    Score: [{score_bar}] {score:.1%}")
            print(f"    {evaluation['feedback']}")

            all_results.append({
                "word": word,
                "pass_type": pass_type,
                "score": score,
                "feedback": evaluation["feedback"],
            })

            # Small pause between passes
            if i < num_words * 3 - 1:
                pause = 1.0 if dry_run else 2.0
                await asyncio.sleep(pause)

    # Summary
    print_banner("SESSION COMPLETE")

    # Per-word summary
    words_seen = []
    for r in all_results:
        if r["word"] not in [w[0] for w in words_seen]:
            word_results = [x for x in all_results if x["word"] == r["word"]]
            feel_score = next((x["score"] for x in word_results
                               if x["pass_type"] == "feel"), 0)
            recog_score = next((x["score"] for x in word_results
                                if x["pass_type"] == "recognize"), 0)
            prod_score = next((x["score"] for x in word_results
                               if x["pass_type"] == "produce"), 0)
            words_seen.append((r["word"], feel_score, recog_score, prod_score))

    print(f"\n  {'Word':12s} {'Feel':>8s} {'Recognize':>10s} {'Produce':>9s}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*9}")
    for word, feel, recog, prod in words_seen:
        print(f"  {word:12s} {feel:8.1%} {recog:10.1%} {prod:9.1%}")

    # Vocabulary stats
    vs = memory.get_vocab_stats()
    print(f"\n  Vocabulary after session:")
    print(f"    Total: {vs['total_words']} words")
    print(f"    Avg confidence: {vs['avg_confidence']:.3f}")
    print(f"    Phases: {vs['phases']}")
    print(f"    Encounters: {vs['total_encounters']}, Productions: {vs['total_productions']}")

    # Show individual word progress
    vocab = memory.get_vocabulary()
    learned = [w for w in vocab if w["learning_phase"] != "unlearned"]
    if learned:
        print(f"\n  Word Progress ({len(learned)} words with learning):")
        for w in sorted(learned, key=lambda x: x["confidence"], reverse=True):
            conf_bar = "█" * int(w["confidence"] * 10) + "░" * (10 - int(w["confidence"] * 10))
            print(f"    {w['word']:12s} [{conf_bar}] {w['confidence']:.2f} "
                  f"{w['learning_phase']:12s} "
                  f"enc={w['times_encountered']} prod={w['times_produced']}")

    # Final Titan state
    if not dry_run:
        async with httpx.AsyncClient() as client:
            final = await get_titan_state(client)
            print(f"\n  Titan state after learning:")
            print(f"    Epoch: {final['epoch']}, Curvature: {final['curvature']:.3f}")
            h = final.get("hormones", {})
            top_h = sorted(h.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"    Hormones: {', '.join(f'{n}={v:.3f}' for n,v in top_h)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titan's First Words")
    parser.add_argument("--num-words", type=int, default=5,
                        help="Number of words to teach (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without live Titan (faster)")
    args = parser.parse_args()

    asyncio.run(run_session(num_words=args.num_words, dry_run=args.dry_run))
