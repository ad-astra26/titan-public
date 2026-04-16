#!/usr/bin/env python3
"""
ARC-AGI-3 Competition Runner for Titan1.

Standalone script — does NOT connect to the live consciousness loop.
Loads NS weights as read-only copies for personality signals.
Trains action-scorer NN from ARC rewards during play.

Usage:
    # Train on ls20 (10 games)
    python scripts/arc_competition.py --game ls20 --episodes 10

    # Evaluate all games (no training)
    python scripts/arc_competition.py --all --evaluate

    # Full competition run with saved results
    python scripts/arc_competition.py --all --episodes 20 --save-results

    # Quick test (1 episode, verbose)
    python scripts/arc_competition.py --game ls20 --episodes 1 -v
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan_plugin.logic.arc import ArcSDKBridge, GridPerception, ActionMapper, ArcSession, StateActionMemory
from titan_plugin.logic.neural_reflex_net import NeuralReflexNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "neural_nervous_system")
ARC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "arc_agi_3")

# Programs whose personality signals guide ARC strategy
PERSONALITY_PROGRAMS = ["CURIOSITY", "INTUITION", "CREATIVITY", "FOCUS", "IMPULSE"]

ALL_GAMES = ["ls20", "ft09", "vc33"]


def load_arc_config() -> dict:
    """Load ARC-AGI-3 config: titan_params.toml (base) + merged Titan config
    (config.toml + ~/.titan/secrets.toml, where [arc_agi_3].api_key lives)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    params_path = os.path.join(PROJECT_ROOT, "titan_plugin", "titan_params.toml")

    arc: dict = {}
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            arc.update(tomllib.load(f).get("arc_agi_3", {}))

    # Merged config (config.toml + ~/.titan/secrets.toml) overrides titan_params.toml —
    # the api_key secret lives in ~/.titan/secrets.toml.
    try:
        from titan_plugin.config_loader import load_titan_config
        override = load_titan_config().get("arc_agi_3", {})
        arc.update(override)
    except Exception:
        pass

    return arc


def get_game_profile(arc_config: dict, game_id: str) -> dict:
    """Get per-game strategy profile. Falls back to defaults."""
    profiles = arc_config.get("profiles", {})
    profile = profiles.get(game_id, {})
    # Defaults (balanced)
    return {
        "type": profile.get("type", "unknown"),
        "epsilon_start": profile.get("epsilon_start", 0.15),
        "epsilon_decay": profile.get("epsilon_decay", 0.99),
        "stuck_threshold": profile.get("stuck_threshold", 100),
        "curiosity_bonus": profile.get("curiosity_bonus", 0.2),
        "max_resets": profile.get("max_resets", 3),
    }

TITAN_API_URL = "http://localhost:7777"
DREAMING_CHECK_INTERVAL = 10  # seconds between dreaming checks


def check_titan_dreaming() -> dict:
    """Check if Titan is currently dreaming via the live API.

    Returns dict with dreaming state, or empty dict if API unreachable.
    Respects Titan's circadian rhythm — external play should pause during dreams.
    """
    try:
        import urllib.request
        req = urllib.request.Request(f"{TITAN_API_URL}/v4/inner-trinity", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        inner = data.get("data", data)
        dreaming = inner.get("dreaming", {})
        return {
            "is_dreaming": dreaming.get("is_dreaming", inner.get("is_dreaming", False)),
            "fatigue": dreaming.get("fatigue", 0.0),
            "cycle_count": dreaming.get("cycle_count", 0),
            "api_reachable": True,
        }
    except Exception:
        return {"is_dreaming": False, "api_reachable": False}


def wait_for_wakefulness(force: bool = False) -> None:
    """Wait until Titan is awake before starting ARC play.

    If Titan is dreaming, pauses and checks every DREAMING_CHECK_INTERVAL seconds.
    This respects the GABA-governed consolidation phase and ensures we use
    freshest NS weights (post-dream = resensitized receptors, cleaner signals).
    """
    if force:
        return
    state = check_titan_dreaming()
    if not state.get("api_reachable"):
        logging.info("[ARC] Titan API not reachable — proceeding without dreaming check")
        return
    if not state.get("is_dreaming"):
        logging.info("[ARC] Titan is awake (fatigue=%.3f, cycles=%d) — ready to play",
                     state.get("fatigue", 0), state.get("cycle_count", 0))
        return

    logging.info("[ARC] Titan is DREAMING (cycle=%d) — waiting for consolidation to complete...",
                 state.get("cycle_count", 0))
    while state.get("is_dreaming"):
        time.sleep(DREAMING_CHECK_INTERVAL)
        state = check_titan_dreaming()
        if not state.get("api_reachable"):
            logging.info("[ARC] Lost API connection — proceeding")
            return

    logging.info("[ARC] Titan has woken up! Proceeding with ARC play")


def load_ns_programs_readonly() -> dict[str, NeuralReflexNet]:
    """Load NS program weights as read-only copies for personality signals."""
    programs = {}
    for name in PERSONALITY_PROGRAMS:
        weights_path = os.path.join(NS_WEIGHTS_DIR, f"{name.lower()}_weights.json")
        if os.path.exists(weights_path):
            # Read config from weights file to get correct architecture
            with open(weights_path) as f:
                data = json.load(f)
            net = NeuralReflexNet(
                name=name,
                input_dim=data.get("input_dim", 55),
                hidden_1=data.get("hidden_1", 48),
                hidden_2=data.get("hidden_2", 24),
                learning_rate=0.0,  # read-only, no training
                fire_threshold=data.get("fire_threshold", 0.5),
            )
            if net.load(weights_path):
                programs[name] = net
                logging.info("[ARC] Loaded %s (input=%dD, %d updates, threshold=%.2f)",
                             name, net.input_dim, net.total_updates, net.fire_threshold)
            else:
                logging.warning("[ARC] Failed to load %s from %s", name, weights_path)
        else:
            logging.debug("[ARC] No weights for %s at %s", name, weights_path)

    return programs


def load_action_scorer(mapper: ActionMapper, game_id: str) -> None:
    """Load previously trained action-scorer weights for a game."""
    scorer_path = os.path.join(ARC_DATA_DIR, f"{game_id}_scorer.json")
    if os.path.exists(scorer_path):
        # Check architecture compatibility before loading
        try:
            with open(scorer_path) as f:
                saved = json.load(f)
            if saved.get("input_dim") != mapper._action_scorer.input_dim:
                logging.warning("[ARC] Scorer architecture mismatch for %s "
                                "(saved=%dD, current=%dD) — training fresh",
                                game_id, saved.get("input_dim"), mapper._action_scorer.input_dim)
                return
        except Exception:
            pass
        if mapper._action_scorer.load(scorer_path):
            logging.info("[ARC] Loaded scorer for %s (%d updates)",
                         game_id, mapper._action_scorer.total_updates)


def save_action_scorer(mapper: ActionMapper, game_id: str) -> None:
    """Save action-scorer weights for a game."""
    os.makedirs(ARC_DATA_DIR, exist_ok=True)
    scorer_path = os.path.join(ARC_DATA_DIR, f"{game_id}_scorer.json")
    mapper._action_scorer.save(scorer_path)
    logging.info("[ARC] Saved scorer for %s to %s", game_id, scorer_path)


def save_results(results: dict) -> None:
    """Save results to JSON for dashboard API."""
    os.makedirs(ARC_DATA_DIR, exist_ok=True)
    results_path = os.path.join(ARC_DATA_DIR, "latest_results.json")
    tmp = results_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, results_path)
    logging.info("[ARC] Results saved to %s", results_path)


def print_report(report, game_id: str):
    """Pretty-print a session report."""
    print(f"\n{'='*60}")
    print(f"  ARC-AGI-3 Report: {game_id}")
    print(f"{'='*60}")
    print(f"  Episodes:      {report.num_episodes}")
    print(f"  Avg Steps:     {report.avg_steps:.1f}")
    print(f"  Avg Levels:    {report.avg_levels:.2f}")
    print(f"  Best Levels:   {report.best_levels}")
    print(f"  Avg Reward:    {report.avg_reward:.4f}")
    print(f"  Best Reward:   {report.best_reward:.4f}")
    print(f"  Duration:      {report.duration_s:.1f}s")

    if report.episodes:
        print(f"\n  Per-Episode:")
        for i, ep in enumerate(report.episodes):
            fires = ", ".join(f"{k}:{v}" for k, v in sorted(ep.nervous_fires.items()))
            resets = f", resets={ep.reset_count}" if ep.reset_count > 0 else ""
            print(f"    #{i+1}: steps={ep.steps}, levels={ep.levels_completed}/{ep.win_levels}, "
                  f"reward={ep.total_reward:.3f}, state={ep.final_state}{resets}")
            if fires:
                print(f"         NS fires: {fires}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Titan ARC-AGI-3 Competition Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", choices=ALL_GAMES, default="ls20",
                        help="Game to play (default: ls20)")
    parser.add_argument("--all", action="store_true",
                        help="Play all available games")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per game (default: 10)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluation mode (no training, no exploration)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per game (default: 500)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to data/arc_agi_3/")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--force", action="store_true",
                        help="Skip dreaming check (don't wait for Titan to wake)")
    parser.add_argument("--online", action="store_true",
                        help="Use ONLINE mode for public leaderboard scorecard")
    parser.add_argument("--tags", type=str, default="",
                        help="Comma-separated tags for scorecard (e.g. 'titan1,v4,spatial')")
    parser.add_argument("--cycle", action="store_true",
                        help="Cycle through games: primary×N, secondary×2, primary×N, ...")
    parser.add_argument("--cycle-break", type=int, default=2,
                        help="Episodes on secondary game during cycling (default: 2)")
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable reasoning engine integration (deliberate cognition during play)")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "="*60)
    print("  TITAN ARC-AGI-3 COMPETITION RUNNER")
    print("="*60)

    # 0. Respect Titan's dreaming cycle
    print("\n[0/4] Checking Titan's circadian state...")
    wait_for_wakefulness(force=args.force)

    # 1. Load NS personality programs (read-only)
    print("\n[1/4] Loading NS personality programs...")
    ns_programs = load_ns_programs_readonly()
    if ns_programs:
        print(f"  Loaded {len(ns_programs)} programs: {list(ns_programs.keys())}")
    else:
        print("  No NS programs found — running with action-scorer only")

    # 2. Create SDK bridge (with API key + optional ONLINE mode)
    print("\n[2/4] Initializing ARC-AGI-3 SDK...")
    arc_config = load_arc_config()
    api_key = arc_config.get("api_key", "")
    sdk = ArcSDKBridge(api_key=api_key, online=args.online)
    if not sdk.initialize():
        print("  ERROR: SDK initialization failed!")
        sys.exit(1)

    envs = sdk.get_environments()
    print(f"  {len(envs)} environments available:")
    for e in envs:
        print(f"    {e['game_id']}: baselines={e['baseline_actions']}")

    # Create scorecard for leaderboard tracking
    scorecard_id = None
    if api_key:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        tags = tags or ["titan1", "v4", "autonomous"]
        scorecard_id = sdk.create_scorecard(
            tags=tags,
            source_url="https://github.com/iamtitan-tech/titan",
        )
        if scorecard_id:
            print(f"  Scorecard: {scorecard_id} (tags={tags})")
            if args.online:
                print(f"  PUBLIC LEADERBOARD: https://three.arcprize.org/scorecards/{scorecard_id}")
        else:
            print("  WARNING: Scorecard creation failed — results won't appear on leaderboard")

    # 3. Run games
    games = ALL_GAMES if args.all else [args.game]
    all_results = {}
    t0 = time.time()

    # Build game schedule: cycling rotates primary×chunk, secondary×break, ...
    game_schedule = []
    if args.cycle and not args.all:
        primary = args.game
        others = [g for g in ALL_GAMES if g != primary]
        remaining = args.episodes
        chunk = max(3, args.episodes // 4)  # primary chunk size
        other_idx = 0
        while remaining > 0:
            n = min(chunk, remaining)
            game_schedule.extend([(primary, n)])
            remaining -= n
            if remaining > 0 and others:
                brk = min(args.cycle_break, remaining)
                game_schedule.append((others[other_idx % len(others)], brk))
                remaining -= brk
                other_idx += 1
        logging.info("[ARC] Cycling schedule: %s", game_schedule)
        print(f"  Cycling: {game_schedule}")
    else:
        game_schedule = [(g, args.episodes) for g in games]

    for game_id, num_episodes in game_schedule:
        print(f"\n[3/4] {'Evaluating' if args.evaluate else 'Training'} on {game_id} "
              f"({num_episodes} episodes, max {args.max_steps} steps)...")

        # Create fresh components per game (separate action-scorer per game)
        perception = GridPerception(max_steps=args.max_steps)
        mapper = ActionMapper(grid_feature_dim=30, pattern_dim=7)

        # Load previous action-scorer weights if they exist
        load_action_scorer(mapper, game_id)

        # Apply per-game profile (Improvement #6)
        profile = get_game_profile(arc_config, game_id)

        # Use shorter max_steps for navigation games (L1 baseline=21 steps)
        game_max_steps = min(args.max_steps, 200) if profile.get("type") == "navigation" else args.max_steps
        session = ArcSession(sdk, perception, mapper, ns_programs, game_max_steps)
        session.curiosity_bonus = profile["curiosity_bonus"]
        session.stuck_threshold = profile["stuck_threshold"]
        session.max_resets = profile["max_resets"]
        session.epsilon_start = profile["epsilon_start"]
        session.epsilon_decay = profile["epsilon_decay"]
        logging.info("[ARC] Profile for %s: type=%s, eps=%.2f→decay %.3f, "
                     "stuck=%d, curiosity=%.2f, resets=%d",
                     game_id, profile["type"], profile["epsilon_start"],
                     profile["epsilon_decay"], profile["stuck_threshold"],
                     profile["curiosity_bonus"], profile["max_resets"])

        # Wire reasoning engine (Option B: deliberate cognition during play)
        if args.reasoning:
            try:
                from titan_plugin.logic.reasoning import ReasoningEngine
                # Fresh reasoning engine per game — learns ARC-specific chains
                re_config = {
                    "max_chain_length": 10,
                    "min_chain_length": 3,
                    "confidence_threshold": 0.6,
                    "policy_input_dim": 99,
                    "policy_h1": 64,
                    "policy_h2": 32,
                    "learning_rate": 0.001,
                    "save_dir": os.path.join(ARC_DATA_DIR, f"reasoning_{game_id}"),
                }
                session.reasoning_engine = ReasoningEngine(config=re_config)
                session._reasoning_steps_threshold = 10
                logging.info("[ARC] Reasoning engine ENABLED for %s (min_chain=3, threshold=10 steps)",
                             game_id)
            except Exception as re_err:
                logging.warning("[ARC] Reasoning engine failed to init: %s", re_err)

        # Wire NS accumulation model (Improvement #7)
        ns_accum_config = arc_config.get("ns_accumulation", {})
        if ns_accum_config.get("enabled", False):
            session.ns_accumulation_enabled = True
            session.ns_accum_decay = ns_accum_config.get("decay_rate", 0.95)
            session.ns_accum_threshold = ns_accum_config.get("fire_threshold", 1.5)

        # Wire state-action memory (Improvement #2: cross-episode learning)
        memory = StateActionMemory()
        memory_path = os.path.join(ARC_DATA_DIR, f"{game_id}_memory.json")
        if memory.load(memory_path):
            logging.info("[ARC] Loaded state memory for %s (%d states, %d records)",
                         game_id, memory.get_stats()["total_states"],
                         memory.get_stats()["total_records"])
        session.state_memory = memory

        # Wire dreaming check so long sessions pause between episodes when Titan sleeps
        if not args.force:
            session.dreaming_check = lambda: wait_for_wakefulness(force=False)

        # Create environment
        if not sdk.make_env(game_id):
            print(f"  ERROR: Failed to create environment for {game_id}")
            continue

        # Run
        if args.evaluate:
            report = session.evaluate(game_id, num_episodes)
        else:
            report = session.train_session(game_id, num_episodes)

        print_report(report, game_id)

        # Save action-scorer + state memory (only in training mode)
        if not args.evaluate:
            save_action_scorer(mapper, game_id)
            memory.save(memory_path)
            mem_stats = memory.get_stats()
            logging.info("[ARC] Memory: %d states, %d new this session, %d winning sequences",
                         mem_stats["total_states"], mem_stats["new_this_episode"],
                         mem_stats["winning_sequences"])

        # Collect results
        all_results[game_id] = {
            "num_episodes": report.num_episodes,
            "avg_steps": report.avg_steps,
            "avg_levels": report.avg_levels,
            "best_levels": report.best_levels,
            "avg_reward": report.avg_reward,
            "best_reward": report.best_reward,
            "duration_s": report.duration_s,
        }

    total_duration = time.time() - t0

    # 4. Final scorecard
    print("\n[4/4] Final Scorecard")
    # Close scorecard first (finalizes for leaderboard)
    if scorecard_id:
        closed = sdk.close_scorecard()
        if closed:
            print(f"  Scorecard CLOSED: {scorecard_id}")
            if args.online:
                print(f"  Results will appear on leaderboard within ~15 minutes")
                print(f"  View: https://three.arcprize.org/scorecards/{scorecard_id}")
    scorecard = sdk.get_scorecard()
    if scorecard:
        print(json.dumps(scorecard, indent=2, default=str))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(games)} game(s), {total_duration:.1f}s total")
    print(f"{'='*60}")
    for gid, res in all_results.items():
        print(f"  {gid}: {res['avg_levels']:.2f} avg levels, "
              f"{res['best_levels']} best, {res['avg_steps']:.0f} avg steps")

    # Save results if requested
    if args.save_results:
        results_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "evaluate" if args.evaluate else "train",
            "episodes_per_game": args.episodes,
            "max_steps": args.max_steps,
            "ns_programs": list(ns_programs.keys()),
            "games": all_results,
            "scorecard": scorecard,
            "total_duration_s": round(total_duration, 2),
        }
        save_results(results_data)

    sdk.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
