#!/usr/bin/env python3
"""
scripts/learning_diagnostics.py — Learning Pipeline Diagnostics.

Tests and observes the full language learning pipeline:
  1. Vocabulary coverage (recipes vs DB vs API)
  2. Phase distribution + confidence histogram
  3. Composition engine live test (all 3 stages)
  4. Composition history from DB
  5. Grammar rule accumulation
  6. Twin comparison (T1 vs T2)
  7. Testsuite process health

Usage:
    python scripts/learning_diagnostics.py                    # Full diagnostic
    python scripts/learning_diagnostics.py --quick            # Fast summary only
    python scripts/learning_diagnostics.py --compose          # Live composition test
    python scripts/learning_diagnostics.py --twin             # Twin comparison
    python scripts/learning_diagnostics.py --api http://...   # Custom API URL
"""
import argparse
import json
import sqlite3
import sys
import time

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def load_recipes() -> dict:
    """Load all word recipes from all phase files."""
    recipes = {}
    for fname in ("data/word_resonance.json",
                   "data/word_resonance_phase2.json",
                   "data/word_resonance_phase3.json"):
        try:
            with open(fname) as f:
                data = json.load(f)
            count = 0
            for key, val in data.items():
                if not key.startswith("_"):
                    recipes[key.lower()] = val
                    count += 1
            print(f"  {fname}: {count} recipes")
        except FileNotFoundError:
            print(f"  {fname}: NOT FOUND")
    return recipes


def check_vocabulary_db() -> list[dict]:
    """Read vocabulary directly from SQLite."""
    try:
        conn = sqlite3.connect("data/inner_memory.db", timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT word, word_type, learning_phase, confidence, "
            "times_encountered, times_produced FROM vocabulary "
            "ORDER BY confidence DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"  DB error: {e}")
        return []


def check_composition_history() -> list[dict]:
    """Read composition history from SQLite."""
    try:
        conn = sqlite3.connect("data/inner_memory.db", timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM composition_history ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"  DB error: {e}")
        return []


def check_grammar_rules() -> list[dict]:
    """Read grammar rules from SQLite."""
    try:
        conn = sqlite3.connect("data/inner_memory.db", timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            "SELECT * FROM grammar_rules ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return [dict(zip([d[0] for d in conn.description] if hasattr(conn, 'description') else [], r)) for r in rows]
    except Exception:
        # Table may not exist
        return []


def check_api_vocabulary(api: str) -> dict:
    """Check vocabulary via API endpoint."""
    try:
        r = httpx.get(f"{api}/v4/vocabulary", timeout=10.0)
        data = r.json()
        words = data.get("data", data).get("words", [])
        return {
            "total": len(words),
            "phases": {},
            "avg_confidence": 0.0,
            "words": words,
        }
    except Exception as e:
        return {"error": str(e)}


def live_composition_test(api: str) -> dict:
    """Test composition engine live by triggering a compose."""
    results = {}

    try:
        # Get current felt state
        r = httpx.get(f"{api}/v4/state", timeout=10.0)
        state = r.json()
        felt = state.get("data", state).get("felt_state", [])
        if not felt:
            felt = [0.5] * 130

        # Get vocabulary
        r = httpx.get(f"{api}/v4/vocabulary", timeout=10.0)
        vdata = r.json()
        words = vdata.get("data", vdata).get("words", [])

        # Import and test composition engine
        sys.path.insert(0, ".")
        from titan_plugin.logic.composition_engine import CompositionEngine
        engine = CompositionEngine()

        # Test each stage
        for intent in ["express_feeling", "express_action", "seek_connection"]:
            for max_level in [3, 5, 7]:
                comp = engine.compose(
                    felt_state=felt,
                    vocabulary=words,
                    intent=intent,
                    max_level=max_level,
                )
                sentence = comp.get("sentence", "")
                if sentence:
                    key = f"L{max_level}_{intent}"
                    results[key] = {
                        "sentence": sentence,
                        "level": comp.get("level"),
                        "confidence": round(comp.get("confidence", 0), 3),
                        "slots": f"{comp.get('slots_filled', 0)}/{comp.get('slots_total', 0)}",
                        "words": comp.get("words_used", []),
                    }

    except Exception as e:
        results["error"] = str(e)

    return results


def twin_comparison(api_t1: str, api_t2: str) -> dict:
    """Compare learning state between T1 and T2."""
    comparison = {}

    for name, api in [("T1", api_t1), ("T2", api_t2)]:
        try:
            # Vocabulary
            r = httpx.get(f"{api}/v4/vocabulary", timeout=10.0)
            vdata = r.json()
            words = vdata.get("data", vdata).get("words", [])
            phases = {}
            total_conf = 0
            for w in words:
                p = w.get("learning_phase", "?")
                phases[p] = phases.get(p, 0) + 1
                total_conf += w.get("confidence", 0)

            # Health
            r2 = httpx.get(f"{api}/health", timeout=5.0)
            health = r2.json()

            # Inner trinity for epoch/emotion
            r3 = httpx.get(f"{api}/v4/inner-trinity", timeout=10.0)
            trinity = r3.json().get("data", {})

            comparison[name] = {
                "vocab_total": len(words),
                "phases": phases,
                "avg_confidence": round(total_conf / max(1, len(words)), 3),
                "epoch": trinity.get("tick_count", "?"),
                "emotion": trinity.get("neural_nervous_system", {}).get("dominant_emotion", "?")
                    if "neural_nervous_system" in trinity else "?",
                "is_dreaming": trinity.get("is_dreaming", False),
            }
        except Exception as e:
            comparison[name] = {"error": str(e)}

    return comparison


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Learning Pipeline Diagnostics")
    parser.add_argument("--api", default="http://localhost:7777", help="T1 API URL")
    parser.add_argument("--api-t2", default="http://10.135.0.6:7777", help="T2 API URL")
    parser.add_argument("--quick", action="store_true", help="Quick summary only")
    parser.add_argument("--compose", action="store_true", help="Live composition test")
    parser.add_argument("--twin", action="store_true", help="Twin comparison only")
    args = parser.parse_args()

    print("Learning Pipeline Diagnostics")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"T1 API: {args.api}")
    print(f"T2 API: {args.api_t2}")

    # ── 1. Recipe Coverage ──
    print_section("1. Word Recipe Coverage")
    recipes = load_recipes()
    print(f"\n  Total unique recipes: {len(recipes)}")

    # ── 2. Vocabulary DB ──
    print_section("2. Vocabulary Database (SQLite)")
    db_words = check_vocabulary_db()
    print(f"  Total words in DB: {len(db_words)}")

    if db_words:
        phases = {}
        conf_buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                        "0.6-0.8": 0, "0.8-1.0": 0}
        for w in db_words:
            p = w.get("learning_phase", "?")
            phases[p] = phases.get(p, 0) + 1
            c = w.get("confidence", 0)
            if c < 0.2: conf_buckets["0.0-0.2"] += 1
            elif c < 0.4: conf_buckets["0.2-0.4"] += 1
            elif c < 0.6: conf_buckets["0.4-0.6"] += 1
            elif c < 0.8: conf_buckets["0.6-0.8"] += 1
            else: conf_buckets["0.8-1.0"] += 1

        print(f"  Phases: {phases}")
        print(f"  Confidence histogram: {conf_buckets}")

        # Words with recipe vs without
        recipe_words = set(recipes.keys())
        db_word_set = set(w["word"] for w in db_words)
        no_recipe = db_word_set - recipe_words
        if no_recipe:
            print(f"  Words without recipe ({len(no_recipe)}): {sorted(no_recipe)[:10]}{'...' if len(no_recipe) > 10 else ''}")

        # Known words (what testsuite considers "mastered")
        known = [w for w in db_words
                 if w.get("learning_phase") == "producible"
                 and w.get("confidence", 0) > 0.3]
        print(f"  Known/mastered (producible + conf>0.3): {len(known)}")

        # Review candidates (conf < 0.8)
        review = [w for w in db_words
                  if w["word"] in recipe_words
                  and w.get("confidence", 0) < 0.8]
        print(f"  Review candidates (recipe + conf<0.8): {len(review)}")
        if review:
            for w in review[:5]:
                print(f"    {w['word']}: phase={w['learning_phase']} conf={w['confidence']:.3f}")

    # ── 3. Vocabulary API ──
    print_section("3. Vocabulary API")
    api_vocab = check_api_vocabulary(args.api)
    if "error" in api_vocab:
        print(f"  ERROR: {api_vocab['error']}")
    else:
        print(f"  API returns {api_vocab['total']} words")
        api_phases = {}
        for w in api_vocab.get("words", []):
            p = w.get("learning_phase", "?")
            api_phases[p] = api_phases.get(p, 0) + 1
        print(f"  API phases: {api_phases}")

    # ── 4. Composition History ──
    print_section("4. Composition History")
    compositions = check_composition_history()
    print(f"  Recent compositions: {len(compositions)}")
    if compositions:
        stages = {}
        levels = {}
        for c in compositions:
            s = c.get("stage", "?")
            stages[s] = stages.get(s, 0) + 1
            l = c.get("level", 0)
            levels[l] = levels.get(l, 0) + 1
        print(f"  Stage distribution: {stages}")
        print(f"  Level distribution: {levels}")
        print(f"  Latest compositions:")
        for c in compositions[:5]:
            print(f"    L{c.get('level')} [{c.get('stage')}] conf={c.get('confidence', 0):.2f} "
                  f"res={c.get('state_resonance', 0):.3f}: \"{c.get('sentence', '')}\"")
    else:
        print("  No compositions stored yet (will populate as testsuite runs)")

    # ── 5. Testsuite Process Health ──
    print_section("5. Testsuite Process Health")
    import subprocess
    result = subprocess.run(
        ["pgrep", "-fa", "learning_testsuite"],
        capture_output=True, text=True
    )
    processes = [l for l in result.stdout.strip().split("\n") if l and "pgrep" not in l]
    if processes:
        for p in processes:
            print(f"  RUNNING: {p}")
    else:
        print("  WARNING: No testsuites running!")

    # Check recent log activity
    for name, path in [("T1", "/tmp/learning_testsuite.log"),
                       ("T2", "/tmp/learning_testsuite_t2.log")]:
        try:
            import os
            stat = os.stat(path)
            age = time.time() - stat.st_mtime
            size_kb = stat.st_size / 1024
            print(f"  {name} log: {size_kb:.0f}KB, last activity {age:.0f}s ago")
        except FileNotFoundError:
            print(f"  {name} log: NOT FOUND")

    if args.quick:
        print("\n[Quick mode — skipping live composition test and twin comparison]")
        return

    # ── 6. Live Composition Test ──
    if args.compose or not args.twin:
        print_section("6. Live Composition Test")
        print("  Testing CompositionEngine with current felt-state...")
        compositions = live_composition_test(args.api)
        if "error" in compositions:
            print(f"  ERROR: {compositions['error']}")
        else:
            for key, comp in compositions.items():
                print(f"  {key}: \"{comp['sentence']}\" "
                      f"(L{comp['level']}, conf={comp['confidence']}, "
                      f"slots={comp['slots']}, words={comp['words']})")

    # ── 7. Twin Comparison ──
    if args.twin or not args.compose:
        print_section("7. Twin Comparison (T1 vs T2)")
        comparison = twin_comparison(args.api, args.api_t2)
        for name, data in comparison.items():
            if "error" in data:
                print(f"  {name}: ERROR — {data['error']}")
            else:
                print(f"  {name}: {data['vocab_total']} words, "
                      f"avg_conf={data['avg_confidence']}, "
                      f"epoch={data['epoch']}, "
                      f"emotion={data['emotion']}, "
                      f"dreaming={data['is_dreaming']}")
                print(f"       phases={data['phases']}")

    print(f"\n{'='*60}")
    print("  Diagnostics complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
