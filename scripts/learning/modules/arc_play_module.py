"""
ARC-AGI-3 Play Module for Experience Playground.

Lets Titan play ARC-AGI-3 puzzles during his learning sessions.
Uses the standalone ARC competition infrastructure but wrapped
as a testsuite module so the SmartScheduler can pick it up.

The module runs a short ARC play session (1-3 games), records results,
and reports back to the testsuite for curriculum tracking. Titan learns
spatial reasoning, pattern recognition, and strategic thinking — all
general-purpose cognitive skills that benefit his overall development.

This is NOT game-specific optimization — it's cognitive exercise,
like a parent giving a child puzzles to develop problem-solving skills.
"""
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger("testsuite.arc_play")

# ARC module lives in titan_plugin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

ARC_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data", "arc_agi_3")
NS_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data", "neural_nervous_system")

# Games to cycle through
GAMES = ["ls20", "ft09", "vc33"]
PERSONALITY_PROGRAMS = ["CURIOSITY", "INTUITION", "CREATIVITY", "FOCUS", "IMPULSE"]


class ArcPlayModule:
    """
    ARC-AGI-3 puzzle play as a testsuite learning module.

    Runs 1-3 short ARC game episodes during a learning window.
    Uses Titan's NS personality programs (read-only) for strategy.
    Results stored in data/arc_agi_3/ for analysis.
    """

    def __init__(self, instance_name: str = "titan1"):
        self._instance = instance_name
        self._session_count = 0
        self._game_index = 0  # Cycle through games
        self._sdk_initialized = False
        logger.info("[ArcPlay] Init for %s", instance_name)

    async def run(self, client, api: str, state: dict, curriculum) -> dict:
        """Run one ARC reasoning session (1-3 game episodes)."""
        import asyncio
        t0 = time.time()
        self._session_count += 1

        # Don't play during dreaming (redundant safety — scheduler checks too)
        if state.get("is_dreaming"):
            return self._result(t0, success=False, error="dreaming")

        # Don't play if Chi is too low (cognitive energy needed)
        if state.get("chi_total", 0.5) < 0.35:
            return self._result(t0, success=False, error="low_chi")

        logger.info("[ArcPlay] Session %d starting (game_index=%d)",
                    self._session_count, self._game_index)

        # Run ARC play in a thread (SDK is synchronous, testsuite is async)
        try:
            result = await asyncio.to_thread(self._play_sync, state)
            return result
        except Exception as e:
            logger.error("[ArcPlay] Session failed: %s", e)
            return self._result(t0, success=False, error=str(e))

    def _play_sync(self, state: dict) -> dict:
        """Synchronous ARC play — runs in thread."""
        import json
        t0 = time.time()

        try:
            from titan_plugin.logic.arc import (
                ArcSDKBridge, GridPerception, ActionMapper,
                ArcSession, StateActionMemory,
            )
            from titan_plugin.logic.neural_reflex_net import NeuralReflexNet
        except ImportError as e:
            return self._result(t0, success=False, error=f"import: {e}")

        # Select game (cycle through ls20, ft09, vc33)
        game_id = GAMES[self._game_index % len(GAMES)]
        self._game_index += 1

        # Determine episode count based on energy/alertness
        chi = state.get("chi_total", 0.5)
        ne = state.get("ne", 0.5)
        episodes = 3 if (chi > 0.7 and ne > 0.5) else 2 if chi > 0.5 else 1

        # Load NS programs (read-only)
        ns_programs = {}
        for name in PERSONALITY_PROGRAMS:
            path = os.path.join(NS_WEIGHTS_DIR, f"{name.lower()}_weights.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                net = NeuralReflexNet(
                    name=name, input_dim=data.get("input_dim", 55),
                    hidden_1=data.get("hidden_1", 48),
                    hidden_2=data.get("hidden_2", 24),
                    learning_rate=0.0, fire_threshold=data.get("fire_threshold", 0.5),
                )
                if net.load(path):
                    ns_programs[name] = net

        # Create SDK + components
        sdk = ArcSDKBridge()
        if not sdk.initialize():
            return self._result(t0, success=False, error="sdk_init_failed")

        perception = GridPerception(max_steps=300)
        mapper = ActionMapper(grid_feature_dim=30)

        # Load previous scorer
        scorer_path = os.path.join(ARC_DATA_DIR, f"{game_id}_scorer.json")
        if os.path.exists(scorer_path):
            mapper._action_scorer.load(scorer_path)

        session = ArcSession(sdk, perception, mapper, ns_programs, max_steps=300)

        # Inject real Titan neuromod state (from learning framework API snapshot)
        session.real_neuromods = {
            "DA": state.get("da", 0.5),
            "5-HT": state.get("5ht", 0.5),
            "NE": state.get("ne", 0.5),
            "ACh": state.get("ach", 0.5),
            "Endorphin": state.get("endorphin", 0.5),
            "GABA": state.get("gaba", 0.3),
        }
        session.real_body_state = {
            "fatigue": 1.0 - state.get("chi_total", 0.5),
            "chi_total": state.get("chi_total", 0.5),
            "is_dreaming": state.get("is_dreaming", False),
            "emotion": state.get("emotion", "neutral"),
        }

        # Apply per-game profile + NS accumulation from titan_params.toml
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            params_path = os.path.join(os.path.dirname(ARC_DATA_DIR), "titan_plugin", "titan_params.toml")
            if not os.path.exists(params_path):
                params_path = os.path.join(os.path.dirname(os.path.dirname(ARC_DATA_DIR)),
                                           "titan_plugin", "titan_params.toml")
            if os.path.exists(params_path):
                with open(params_path, "rb") as f:
                    arc_cfg = tomllib.load(f).get("arc_agi_3", {})
                profile = arc_cfg.get("profiles", {}).get(game_id, {})
                if profile:
                    session.curiosity_bonus = profile.get("curiosity_bonus", 0.2)
                    session.stuck_threshold = profile.get("stuck_threshold", 100)
                    session.max_resets = profile.get("max_resets", 3)
                    session.epsilon_start = profile.get("epsilon_start", 0.15)
                    session.epsilon_decay = profile.get("epsilon_decay", 0.99)
                    # epsilon_min floor (2026-04-15 anti-collapse fix)
                    session.epsilon_min = profile.get("epsilon_min", session.epsilon_min)
                ns_accum = arc_cfg.get("ns_accumulation", {})
                if ns_accum.get("enabled", False):
                    session.ns_accumulation_enabled = True
                    session.ns_accum_decay = ns_accum.get("decay_rate", 0.95)
                    session.ns_accum_threshold = ns_accum.get("fire_threshold", 1.5)
                # rFP_arc_training_fix (2026-04-13; rebalanced 2026-04-15)
                _reward_shaping = arc_cfg.get("reward_shaping", {})
                session.goal_distance_reward_k = float(
                    _reward_shaping.get("goal_distance_reward_k", 0.0))
                session.character_target_reward_k = float(
                    _reward_shaping.get("character_target_reward_k", 0.0))
                session.episode_diagnostics_enabled = bool(
                    _reward_shaping.get("episode_diagnostics_enabled", False))
                session.novelty_reward_cap_per_step = float(_reward_shaping.get(
                    "novelty_reward_cap_per_step", session.novelty_reward_cap_per_step))
                logger.info(
                    "[ArcPlay] Profile loaded for %s: stuck=%d, curiosity=%.2f, "
                    "eps_min=%.2f, goal_k=%.2f, char_target_k=%.2f, novelty_cap=%.3f, diag=%s",
                    game_id, session.stuck_threshold, session.curiosity_bonus,
                    session.epsilon_min, session.goal_distance_reward_k,
                    session.character_target_reward_k, session.novelty_reward_cap_per_step,
                    session.episode_diagnostics_enabled,
                )
        except Exception as _cfg_err:
            logger.debug("[ArcPlay] Config load: %s (using defaults)", _cfg_err)

        # Load state memory
        memory = StateActionMemory()
        memory_path = os.path.join(ARC_DATA_DIR, f"{game_id}_memory.json")
        memory.load(memory_path)
        session.state_memory = memory

        # ── Forward Model (Phase A1): action-effect prediction ──────────
        try:
            from titan_plugin.logic.arc.forward_model import ForwardModel
            fm = ForwardModel(feature_dim=30, num_actions=7, learning_rate=0.001)
            fm_path = os.path.join(ARC_DATA_DIR, f"{game_id}_forward_model.json")
            fm.load(fm_path)
            session._forward_model = fm
            logger.info("[ArcPlay] Forward model loaded: %d updates, %d buffer",
                        fm.total_updates, len(fm._buffer))
        except Exception as _fm_err:
            logger.debug("[ArcPlay] Forward model not available: %s", _fm_err)

        # ── CGN Reasoning Consumer ──────────────────────────────────────
        # Register ARC as a cognitive consumer in the Concept Grounding
        # Network. CGN Worker (Guardian module) handles training centrally.
        # ARC uses CGNConsumerClient for local forward pass + sends transitions via API.
        cgn = None
        try:
            from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
            _project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            cgn = CGNConsumerClient(
                "reasoning",
                state_dir=os.path.join(_project_root, "data", "cgn"))
            session._cgn = cgn
            logger.info("[ArcPlay] CGN reasoning consumer client ready (via /dev/shm)")
        except Exception as _cgn_err:
            logger.debug("[ArcPlay] CGN not available (standalone mode): %s", _cgn_err)

        # Create environment and play
        if not sdk.make_env(game_id):
            sdk.close()
            return self._result(t0, success=False, error=f"env_{game_id}_failed")

        report = session.train_session(game_id, episodes)

        # ── CGN: Ground ARC discoveries (via CGN Worker) ────────────────
        _cgn_transitions = 0
        if cgn and report.episodes:
            try:
                import httpx as _cgn_httpx
                for ep in report.episodes:
                    # Build concept features
                    _emb = [0.0] * 130
                    if hasattr(session, '_last_pattern_block') and session._last_pattern_block:
                        pb = session._last_pattern_block
                        for i, p in enumerate(["symmetry", "translation", "alignment",
                                               "containment", "adjacency", "repetition", "shape"]):
                            _emb[i] = pb.pattern_deltas.get(p, 0.0)

                    concept_f = {
                        "concept_id": f"arc_{game_id}",
                        "embedding": _emb,
                        "confidence": min(1.0, ep.total_reward / 30.0),
                        "encounter_count": ep.steps,
                        "production_count": ep.levels_completed,
                        "age_epochs": state.get("epoch", 0),
                    }
                    ctx_f = {
                        "epoch": state.get("epoch", 0),
                        "neuromods": session.real_neuromods or {},
                        "concept_confidences": state.get("concept_confidences", {}),
                        "encounter_type": "reasoning",
                    }

                    # Local forward pass (from /dev/shm weights)
                    result = cgn.ground(concept_f, ctx_f)

                    # Forward transition + reward to CGN Worker via API
                    reward = min(0.15, ep.total_reward / 200.0)
                    if ep.levels_completed > 0:
                        reward += 0.5
                    try:
                        _cgn_httpx.post(
                            f"{self._api_base}/v4/social-perception",
                            json={"titan_id": self._titan_id, "events": [{
                                "cgn_transition": {
                                    "consumer": "reasoning",
                                    "concept_id": f"arc_{game_id}",
                                    "reward": reward,
                                    "outcome_context": {
                                        "game_id": game_id,
                                        "levels": ep.levels_completed,
                                        "steps": ep.steps,
                                        "total_reward": ep.total_reward,
                                    },
                                }
                            }]}, timeout=5)
                    except Exception:
                        pass
                    # Also send transition data if client has bus
                    if result.transition:
                        result.transition["reward"] = reward
                        cgn.send_transition(result.transition)
                    _cgn_transitions += 1

                # No local _save_state() — CGN Worker handles persistence
                logger.info("[ArcPlay] CGN: %d reasoning transitions sent to Worker",
                            _cgn_transitions)
            except Exception as _cgn_err:
                logger.warning("[ArcPlay] CGN grounding failed: %s", _cgn_err)

        # Save scorer + memory + forward model
        os.makedirs(ARC_DATA_DIR, exist_ok=True)
        mapper._action_scorer.save(scorer_path)
        memory.save(memory_path)
        if session._forward_model:
            fm_path = os.path.join(ARC_DATA_DIR, f"{game_id}_forward_model.json")
            session._forward_model.save(fm_path)
            _fm_stats = session._forward_model.get_stats()
            logger.info("[ArcPlay] Forward model saved: buffer=%d, updates=%d, accuracy=%.3f",
                        _fm_stats["buffer_size"], _fm_stats["total_updates"],
                        _fm_stats["avg_prediction_accuracy"])

        # Save session report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instance": self._instance,
            "game_id": game_id,
            "episodes": report.num_episodes,
            "avg_steps": report.avg_steps,
            "avg_levels": report.avg_levels,
            "best_levels": report.best_levels,
            "avg_reward": report.avg_reward,
            "best_reward": report.best_reward,
            "duration_s": report.duration_s,
            "memory_states": memory.get_stats()["total_states"],
            "scorer_updates": mapper._action_scorer.total_updates,
        }
        report_path = os.path.join(ARC_DATA_DIR, f"{game_id}_session_log.jsonl")
        with open(report_path, "a") as f:
            f.write(json.dumps(report_data) + "\n")

        sdk.close()

        duration = time.time() - t0
        # Compute accuracy as levels_solved / total_possible
        accuracy = report.avg_levels / max(1, report.episodes[0].win_levels if report.episodes else 7)

        logger.info(
            "[ArcPlay] Session %d: %s — %d episodes, %.1f avg levels, "
            "%.2f avg reward, %d memory states, %.1fs",
            self._session_count, game_id, report.num_episodes,
            report.avg_levels, report.avg_reward,
            memory.get_stats()["total_states"], duration,
        )

        return {
            "type": "arc_play",
            "success": True,
            "game_id": game_id,
            "episodes": report.num_episodes,
            "avg_levels": report.avg_levels,
            "best_levels": report.best_levels,
            "avg_reward": round(report.avg_reward, 3),
            "accuracy": round(accuracy, 3),
            "memory_states": memory.get_stats()["total_states"],
            "cgn_transitions": _cgn_transitions,
            "forward_model": session._forward_model.get_stats() if session._forward_model else {},
            "duration": round(duration, 1),
        }

    def _result(self, t0: float, success: bool = False, error: str = "") -> dict:
        return {
            "type": "arc_play",
            "success": success,
            "game_id": "",
            "episodes": 0,
            "avg_levels": 0.0,
            "best_levels": 0,
            "avg_reward": 0.0,
            "accuracy": 0.0,
            "memory_states": 0,
            "duration": round(time.time() - t0, 1),
            "error": error,
        }
