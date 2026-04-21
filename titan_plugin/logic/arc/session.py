"""
titan_plugin/logic/arc/session.py — ARC-AGI-3 Session Manager.

Manages the full ARC-AGI-3 interaction loop via the official SDK:
  1. Reset game → receive initial frame
  2. GridPerception encodes frame → Trinity tensor update
  3. NS personality programs provide strategic signals (read-only)
  4. ActionMapper selects action from signals + learned scorer
  5. SDK step → receive next frame
  6. Derive reward from level completion
  7. Train action-scorer NN from rewards
  8. Repeat until WIN, GAME_OVER, or max steps
  9. HAOV loop: Hypothesize→Act→Observe→Verify (Phase A2)

All operations are synchronous (SDK is blocking).
NS programs are loaded READ-ONLY — never modified during ARC play.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── HAOV: Hypothesis-Act-Observe-Verify Loop (Phase A2) ─────────────────


@dataclass
class Hypothesis:
    """A rule hypothesis about how the puzzle works."""
    rule: str                      # e.g. "action_3_increases_order"
    action: int                    # action that the hypothesis is about
    predicted_effect: str          # "order_up", "entropy_down", "movement", "novel_state"
    predicted_magnitude: float     # expected effect size (0-1)
    confidence: float = 0.1       # grows with verification, decays with falsification
    tests: int = 0                # number of times tested
    confirmations: int = 0        # number of times confirmed
    falsifications: int = 0       # number of times falsified
    source_pattern: str = ""      # pattern that inspired this hypothesis


class HAOVTracker:
    """Tracks hypotheses about puzzle rules across steps within an episode.

    The HAOV loop maps to meta-reasoning primitives:
      HYPOTHESIZE → generate rule from pattern observation
      ACT → execute action to test hypothesis
      OBSERVE → compare predicted vs actual grid outcome
      VERIFY → update confidence: confirm or falsify

    Verified hypotheses become high-confidence action selection signals.
    """

    def __init__(self, max_hypotheses: int = 20):
        self._hypotheses: list[Hypothesis] = []
        self._max = max_hypotheses
        self._active_test: Optional[dict] = None  # currently testing hypothesis
        self._stats = {
            "formed": 0, "tested": 0, "confirmed": 0, "falsified": 0,
            "used_for_action": 0,
        }
        # Persistent across episodes (cleared on full reset only)
        self._verified_rules: list[Hypothesis] = []  # confidence > 0.6

    def reset_episode(self):
        """Reset per-episode state, keep verified rules."""
        self._active_test = None
        # Decay low-confidence hypotheses
        self._hypotheses = [h for h in self._hypotheses if h.confidence > 0.05]

    def hypothesize(self, action: int, pattern_info: dict,
                    features: dict) -> Optional[Hypothesis]:
        """HYPOTHESIZE: form a rule hypothesis from observed patterns.

        Called when PGL detects a significant pattern or the forward model
        makes a confident prediction.

        Args:
            action: the action being hypothesized about
            pattern_info: dict with keys like 'best_match', 'surprise',
                          'pattern_deltas', 'salient_observation'
            features: current grid features dict

        Returns:
            New hypothesis or None if nothing worth hypothesizing.
        """
        best_match = pattern_info.get("best_match", "")
        surprise = pattern_info.get("surprise", 0.0)
        deltas = pattern_info.get("pattern_deltas", {})

        # Find the dominant pattern change
        if deltas:
            dominant_pattern = max(deltas, key=lambda k: abs(deltas[k]))
            dominant_delta = deltas[dominant_pattern]
        else:
            return None

        # Only hypothesize when there's a clear signal
        if abs(dominant_delta) < 0.05:
            return None

        # Determine predicted effect based on pattern type
        if dominant_pattern in ("symmetry", "alignment"):
            effect = "order_up"
            magnitude = abs(dominant_delta)
        elif dominant_pattern == "translation":
            effect = "movement"
            magnitude = abs(dominant_delta)
        elif dominant_pattern == "repetition":
            effect = "entropy_down"
            magnitude = abs(dominant_delta) * 0.8
        else:
            effect = "novel_state"
            magnitude = surprise

        rule_name = f"action_{action}_{effect}_{dominant_pattern}"

        # Check if we already have this hypothesis
        for h in self._hypotheses:
            if h.rule == rule_name:
                return h  # Already tracking this one

        h = Hypothesis(
            rule=rule_name,
            action=action,
            predicted_effect=effect,
            predicted_magnitude=magnitude,
            source_pattern=dominant_pattern,
        )
        self._hypotheses.append(h)
        self._stats["formed"] += 1

        # Evict lowest-confidence if over limit
        if len(self._hypotheses) > self._max:
            self._hypotheses.sort(key=lambda x: x.confidence)
            self._hypotheses.pop(0)

        logger.debug("[HAOV] HYPOTHESIZE: %s (pattern=%s, magnitude=%.2f)",
                     rule_name, dominant_pattern, magnitude)
        return h

    def select_test_action(self, available_actions: list[int],
                           features: dict) -> Optional[int]:
        """ACT: select an action to test a hypothesis.

        Picks the hypothesis with highest uncertainty (moderate confidence,
        not yet well-tested) and returns its action for testing.

        Returns:
            Action to execute for testing, or None if no hypothesis to test.
        """
        if self._active_test:
            return None  # Already testing one

        # Find best hypothesis to test: moderate confidence (0.15-0.7), fewest tests
        testable = [h for h in self._hypotheses
                    if h.action in available_actions
                    and 0.05 <= h.confidence <= 0.7
                    and h.tests < 10]

        if not testable:
            return None

        # Prefer high-magnitude, low-test-count hypotheses
        testable.sort(key=lambda h: h.predicted_magnitude / max(1, h.tests + 1),
                      reverse=True)
        target = testable[0]

        # Record what we're testing
        order = features.get("semantic", [0.5] * 5)[3] if "semantic" in features else 0.5
        entropy = features.get("inner_body", [0.5] * 5)[1] if "inner_body" in features else 0.5
        self._active_test = {
            "hypothesis": target,
            "pre_order": order,
            "pre_entropy": entropy,
            "pre_features": {k: list(v) if hasattr(v, '__iter__') else v
                             for k, v in features.items()
                             if k in ("inner_body", "inner_mind", "spatial", "semantic")},
        }
        self._stats["tested"] += 1
        target.tests += 1

        return target.action

    def verify(self, action_taken: int, features_after: dict,
               reward: float, is_novel: bool) -> Optional[dict]:
        """OBSERVE + VERIFY: compare prediction to reality.

        Called after an action is executed. If the action matches the
        active test, verify the hypothesis.

        Returns:
            Dict with verification result, or None if no active test.
        """
        if not self._active_test:
            return None

        test = self._active_test
        h = test["hypothesis"]

        # Only verify if we actually took the hypothesized action
        if action_taken != h.action:
            self._active_test = None
            return None

        # OBSERVE: measure actual effect
        post_order = features_after.get("semantic", [0.5] * 5)[3] if "semantic" in features_after else 0.5
        post_entropy = features_after.get("inner_body", [0.5] * 5)[1] if "inner_body" in features_after else 0.5

        # VERIFY: did the predicted effect happen?
        confirmed = False
        prediction_error = 0.0

        if h.predicted_effect == "order_up":
            actual_delta = post_order - test["pre_order"]
            confirmed = actual_delta > 0.01
            prediction_error = abs(h.predicted_magnitude - actual_delta)

        elif h.predicted_effect == "entropy_down":
            actual_delta = test["pre_entropy"] - post_entropy
            confirmed = actual_delta > 0.01
            prediction_error = abs(h.predicted_magnitude - actual_delta)

        elif h.predicted_effect == "movement":
            # Check if spatial features changed significantly
            pre_spatial = test["pre_features"].get("spatial", [0.5] * 5)
            post_spatial = features_after.get("spatial", [0.5] * 5)
            spatial_delta = sum(abs(a - b) for a, b in zip(post_spatial, pre_spatial)) / 5.0
            confirmed = spatial_delta > 0.03
            prediction_error = abs(h.predicted_magnitude - spatial_delta)

        elif h.predicted_effect == "novel_state":
            confirmed = is_novel or reward > 0.1
            prediction_error = 0.0 if confirmed else h.predicted_magnitude

        # Update hypothesis confidence
        if confirmed:
            h.confirmations += 1
            h.confidence = min(0.95, h.confidence + 0.15 * (1.0 - h.confidence))
            self._stats["confirmed"] += 1
            # Promote to verified rules
            if h.confidence > 0.6 and h not in self._verified_rules:
                self._verified_rules.append(h)
                logger.info("[HAOV] VERIFIED rule: %s (conf=%.2f, %d/%d confirmed)",
                            h.rule, h.confidence, h.confirmations, h.tests)
        else:
            h.falsifications += 1
            h.confidence = max(0.01, h.confidence * 0.6)
            self._stats["falsified"] += 1

        self._active_test = None

        return {
            "hypothesis": h.rule,
            "confirmed": confirmed,
            "confidence": h.confidence,
            "prediction_error": prediction_error,
            "tests": h.tests,
        }

    def suggest_action(self, available_actions: list[int]) -> Optional[int]:
        """Use verified hypotheses to suggest an action.

        Returns action from the highest-confidence verified hypothesis
        that matches available actions, or None.
        """
        candidates = [h for h in self._verified_rules
                      if h.action in available_actions
                      and h.confidence > 0.5]
        if not candidates:
            return None

        best = max(candidates, key=lambda h: h.confidence * h.predicted_magnitude)
        self._stats["used_for_action"] += 1
        return best.action

    def get_stats(self) -> dict:
        """Return HAOV statistics for logging."""
        return {
            **self._stats,
            "active_hypotheses": len(self._hypotheses),
            "verified_rules": len(self._verified_rules),
            "top_rules": [
                {"rule": h.rule, "conf": round(h.confidence, 2),
                 "tests": h.tests, "confirmed": h.confirmations}
                for h in sorted(self._verified_rules,
                                key=lambda x: x.confidence, reverse=True)[:5]
            ],
        }

    def get_verified_concepts(self) -> list[dict]:
        """Return verified hypotheses as concept candidates for CGN grounding."""
        return [
            {
                "concept_id": f"haov_{h.rule}",
                "confidence": h.confidence,
                "pattern": h.source_pattern,
                "effect": h.predicted_effect,
                "tests": h.tests,
                "confirmations": h.confirmations,
            }
            for h in self._verified_rules if h.confidence > 0.5
        ]


@dataclass
class EpisodeResult:
    """Result of a single game play (all levels)."""
    game_id: str
    steps: int
    levels_completed: int
    win_levels: int
    total_reward: float
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    levels_at_step: list[int] = field(default_factory=list)
    nervous_fires: dict = field(default_factory=dict)
    reset_count: int = 0
    duration_s: float = 0.0
    success: bool = False
    final_state: str = "NOT_FINISHED"


@dataclass
class SessionReport:
    """Report from a training/evaluation session."""
    game_id: str
    num_episodes: int
    avg_reward: float
    avg_steps: float
    avg_levels: float
    best_levels: int
    best_reward: float
    episodes: list[EpisodeResult] = field(default_factory=list)
    duration_s: float = 0.0


class ArcSession:
    """
    Manages ARC-AGI-3 game sessions — the bridge between Titan's
    nervous system and the interactive reasoning benchmark.

    Core loop: perceive → think → act → learn → repeat.
    NS programs provide personality signals (read-only copies).
    Action-scorer NN learns ARC-specific quality from rewards.
    """

    def __init__(
        self,
        sdk_bridge,
        grid_perception,
        action_mapper,
        ns_programs: Optional[dict] = None,
        max_steps: int = 500,
    ):
        """
        Args:
            sdk_bridge: ArcSDKBridge instance
            grid_perception: GridPerception instance
            action_mapper: ActionMapper instance
            ns_programs: Dict of {name: NeuralReflexNet} read-only copies
            max_steps: Safety limit per game (across all levels)
        """
        self._sdk = sdk_bridge
        self._perception = grid_perception
        self._mapper = action_mapper
        self._ns_programs = ns_programs or {}
        self._max_steps = max_steps
        self._episode_history: list[EpisodeResult] = []
        # Optional callback: called between episodes to respect dreaming cycles.
        # Should block until Titan is awake. Set by competition runner.
        self.dreaming_check: Optional[callable] = None
        # State-action memory graph (Improvement #2) — persists across episodes
        self.state_memory = None  # Set by caller (ArcSession doesn't own lifecycle)
        # Intrinsic curiosity reward config (Improvement #1)
        self.curiosity_bonus = 0.2          # reward for novel states
        self.curiosity_bonus_large = 0.4    # reward for states with large frame delta
        self.large_delta_threshold = 0.3    # frame delta threshold for large bonus
        self.stuck_threshold = 100          # steps before strategic reset triggers
        self.stuck_penalty = -0.01          # penalty per step when stuck (pre-reset)
        self.max_resets = 3                 # max strategic resets per game
        # Per-game epsilon with decay (Improvement #6)
        self.epsilon_start = 0.15           # starting exploration rate
        self.epsilon_decay = 0.99           # decay per episode
        self.epsilon_min = 0.08             # floor: exploration never fully dies (2026-04-15 fix)
        self._current_epsilon = 0.15        # tracks decaying epsilon
        self._episode_count = 0
        # Anti-collapse watchdog (2026-04-15): force exploration when action
        # distribution collapses to one action. 50-step sliding window; if any
        # single action >80% of window, force epsilon = 0.5 for next 30 steps.
        self.anti_collapse_enabled = True
        self.anti_collapse_window = 50
        self.anti_collapse_threshold = 0.80
        self.anti_collapse_force_steps = 30
        self.anti_collapse_force_epsilon = 0.5
        # Novelty-reward cap (2026-04-15): previous novelty shaped-reward summed
        # to ~0.41 per step, drowning G2 character-target reward (~0.015 per
        # step). Cap keeps task signal visible to scorer.
        self.novelty_reward_cap_per_step = 0.15
        # NS accumulation model (Improvement #7)
        self.ns_accumulation_enabled = False
        self.ns_accum_decay = 0.95
        self.ns_accum_threshold = 1.5
        self._ns_accumulators: dict[str, float] = {}
        # Reasoning engine integration (Option B)
        self.reasoning_engine = None  # Set by caller (ReasoningEngine instance)
        self._reasoning_steps_threshold = 10  # Start reasoning after N stuck steps
        self._reasoning_action_bias: Optional[str] = None  # "explore" or "exploit"

        # rFP_arc_training_fix (2026-04-13) — goal-signal reward shaping
        # G1 empirical capture (all games) + G2 ls20 character-target heuristic
        # Feature-flagged via goal_distance_reward_k coefficient (0 = disabled,
        # falls back to existing shaped reward).
        self._goal_detector = None
        self.goal_distance_reward_k = 0.0           # coefficient for G1 similarity reward
        self.character_target_reward_k = 0.0        # coefficient for G2 ls20 manhattan reward
        self.character_target_normalize_by_grid = True   # iter-2 default; set False for iter-3
        self.arc_iter_3_enabled = False             # feature flag — True activates iter-3 reward rebalance
        self.episode_diagnostics_enabled = False    # T1 diagnostic JSONL dump
        try:
            from titan_plugin.logic.arc.goal_detector import GoalDetector
            self._goal_detector = GoalDetector()
        except Exception as _gd_err:
            logger.warning("[ArcSession] GoalDetector init failed: %s", _gd_err)
        # Auto-load reward_shaping config so any caller (arc_competition.py cron,
        # arc_play_module.py testsuite, tests) picks up the flags uniformly.
        # Callers may still override via direct attribute assignment.
        try:
            import os as _os
            try:
                import tomllib as _tomllib
            except ImportError:
                import tomli as _tomllib
            # __file__ = .../titan_plugin/logic/arc/session.py
            # → 4 dirnames reach project root, then titan_plugin/titan_params.toml
            _here = _os.path.abspath(__file__)
            _project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_here))))
            _rs_path = _os.path.join(_project_root, "titan_plugin", "titan_params.toml")
            if _os.path.exists(_rs_path):
                with open(_rs_path, "rb") as _rsf:
                    _rs_cfg = _tomllib.load(_rsf).get("arc_agi_3", {}).get("reward_shaping", {})
                self.goal_distance_reward_k = float(_rs_cfg.get("goal_distance_reward_k", 0.0))
                self.character_target_reward_k = float(_rs_cfg.get("character_target_reward_k", 0.0))
                self.episode_diagnostics_enabled = bool(_rs_cfg.get("episode_diagnostics_enabled", False))
                self.novelty_reward_cap_per_step = float(_rs_cfg.get(
                    "novelty_reward_cap_per_step", self.novelty_reward_cap_per_step))
                # iter-3 (2026-04-20): master gate + per-cell-reward toggle
                self.arc_iter_3_enabled = bool(_rs_cfg.get("arc_iter_3_enabled", False))
                # Default True (iter-2 behavior); TOML override wins. If iter-3 is the gate,
                # a False value flips the math to unnormalized per-cell progress reward.
                self.character_target_normalize_by_grid = bool(_rs_cfg.get(
                    "character_target_normalize_by_grid", True))
                if self.goal_distance_reward_k > 0 or self.character_target_reward_k > 0:
                    logger.info(
                        "[ArcSession] Reward shaping loaded: goal_k=%.2f char_target_k=%.2f "
                        "novelty_cap=%.3f normalize=%s iter_3=%s diag=%s",
                        self.goal_distance_reward_k, self.character_target_reward_k,
                        self.novelty_reward_cap_per_step,
                        self.character_target_normalize_by_grid,
                        self.arc_iter_3_enabled,
                        self.episode_diagnostics_enabled,
                    )
        except Exception as _rs_err:
            logger.debug("[ArcSession] reward_shaping config load skipped: %s", _rs_err)
        self._reasoning_chains_fired = 0
        self._reasoning_commits = 0
        # Real neuromod state from Titan (set by caller from learning framework state dict)
        self.real_neuromods: Optional[dict] = None  # {DA: float, 5HT: float, ...}
        self.real_body_state: Optional[dict] = None  # {fatigue, chi_total, is_dreaming, ...}
        # CGN reasoning consumer (set by caller — shared V(s) with language + social)
        self._cgn = None  # ConceptGroundingNetwork instance
        # Forward Model (Phase A1): predicts next features from current + action
        self._forward_model = None  # ForwardModel instance, set by caller
        self._lookahead_enabled = False  # Enable after sufficient training data
        self._lookahead_min_buffer = 50  # Min transitions before enabling.
                                         # 2026-04-08 (later) audit: lowered
                                         # 100 → 50 because a 50-episode run
                                         # produces ~200 transitions total but
                                         # the forward model has to activate
                                         # mid-run to be useful. With buffer=100,
                                         # lookahead never activated until the
                                         # end of session, defeating its purpose.
        # Forward model usage telemetry — counts how often lookahead actually
        # selected an action (vs scorer/exploration). Helps verify the forward
        # model is actually contributing to decisions, not just sitting idle.
        self._fm_lookahead_used_count = 0
        self._fm_scorer_used_count = 0
        self._prev_features_flat = None  # For recording transitions
        # Sequence Replay (Phase A3): replay winning action chains
        self._active_sequence: Optional[list[int]] = None
        self._sequence_step: int = 0
        # HAOV: Hypothesis-Act-Observe-Verify loop (Phase A2)
        self._haov = HAOVTracker(max_hypotheses=20)
        self._haov_enabled = False  # Enable after forward model has data
        self._haov_min_fm_buffer = 50  # Min forward model transitions before HAOV
        self._haov_test_probability = 0.25  # 25% chance of testing a hypothesis vs normal play
        self._haov_cgn_synced = False  # Track if we registered ARC verifier on CGN
        # Perceptual Grounding Layer (mini-reasoning + pattern primitives)
        self._mini_reasoning = None
        self._pattern_primitives = None
        self._surprise_ns_boost = 0.3  # CURIOSITY accumulator boost on surprise
        self._last_pattern_block = None
        try:
            from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
            from titan_plugin.logic.pattern_primitives import PatternPrimitives
            self._mini_reasoning = MiniReasoningEngine(trend_window=10)
            self._pattern_primitives = PatternPrimitives()
            logger.info("[ArcSession] Perceptual Grounding Layer ENABLED")
        except ImportError as _pgl_err:
            logger.debug("[ArcSession] Perceptual Grounding not available: %s", _pgl_err)

    def _apply_neuromod_scaling(self) -> dict:
        """Apply live neuromod state to strategy parameters (Phase A4).

        Maps Titan's emotional/arousal state to puzzle-solving strategy:
          NE (arousal)     → exploration rate (high NE = try new things)
          DA (reward)      → exploitation bias (high DA = trust known patterns)
          5-HT (patience)  → stuck threshold (high 5HT = persist longer)
          ACh (attention)  → surprise sensitivity (high ACh = notice more)
          GABA (inhibition)→ impulsivity gate (low GABA = act on impulse)
          Endorphin        → episode persistence (high = keep going)

        Returns dict of scaling factors applied (for logging).
        """
        if not self.real_neuromods:
            return {}

        nm = self.real_neuromods
        _get = lambda k, alt=None, default=0.5: float(
            nm.get(k, nm.get(alt, default)) if alt else nm.get(k, default))

        ne = _get("NE")
        da = _get("DA")
        sht = _get("5-HT", "5HT")
        ach = _get("ACh")
        gaba = _get("GABA")
        endorphin = _get("Endorphin")

        scales = {}

        # NE → exploration: high arousal = more exploration
        # Base epsilon scaled by 1.0 + (NE - 0.5) * 2.0
        # NE=0.3 → epsilon*0.6, NE=0.5 → epsilon*1.0, NE=0.8 → epsilon*1.6
        ne_factor = max(0.3, min(2.0, 1.0 + (ne - 0.5) * 2.0))
        self._current_epsilon *= ne_factor
        scales["ne_epsilon_factor"] = round(ne_factor, 2)

        # DA → exploitation: high dopamine = reduce exploration (trust known)
        # Counteracts NE somewhat — balance between explore and exploit
        if da > 0.6:
            da_exploit = 1.0 - (da - 0.6) * 0.5  # DA=0.8 → epsilon * 0.9
            self._current_epsilon *= max(0.5, da_exploit)
            scales["da_exploit_factor"] = round(da_exploit, 2)

        # 5-HT → patience: high serotonin = longer before reset
        # stuck_threshold scaled by 1.0 + (5HT - 0.5) * 3.0
        # 5HT=0.3 → threshold*0.4, 5HT=0.5 → threshold*1.0, 5HT=0.9 → threshold*2.2
        sht_factor = max(0.3, min(2.5, 1.0 + (sht - 0.5) * 3.0))
        self.stuck_threshold = max(20, int(self.stuck_threshold * sht_factor))
        scales["sht_patience_factor"] = round(sht_factor, 2)

        # ACh → attention/surprise sensitivity: high ACh = stronger curiosity
        # curiosity_bonus scaled by 0.5 + ACh (range: 0.5-1.5)
        ach_factor = max(0.5, min(1.5, 0.5 + ach))
        self.curiosity_bonus *= ach_factor
        self.curiosity_bonus_large *= ach_factor
        scales["ach_curiosity_factor"] = round(ach_factor, 2)

        # GABA → impulsivity: low GABA = skip careful scoring, act on NS signals
        # This is captured naturally by NS accumulation threshold
        if self.ns_accumulation_enabled and gaba < 0.3:
            # Lower fire threshold = NS programs fire faster = more impulsive
            self.ns_accum_threshold *= max(0.5, gaba / 0.3)
            scales["gaba_impulsivity"] = round(gaba, 2)

        # Endorphin → persistence: high endorphin = allow more resets (keep trying)
        if endorphin > 0.6:
            extra_resets = int((endorphin - 0.6) * 5)  # Up to +2 resets
            self.max_resets += extra_resets
            scales["endorphin_extra_resets"] = extra_resets

        return scales

    def play_game(self, game_id: str, training: bool = True) -> Optional[EpisodeResult]:
        """
        Play one complete game (all levels) of an ARC-AGI-3 environment.

        Args:
            game_id: Game identifier (e.g., "ls20")
            training: Whether to train action-scorer from rewards

        Returns:
            EpisodeResult with full trajectory data.
        """
        t0 = time.time()
        self._perception.reset()
        self._episode_count += 1
        # Decay epsilon per episode (Improvement #6), clamped to epsilon_min floor
        # so exploration never dies (2026-04-15 fix for single-action collapse).
        _decayed_eps = self.epsilon_start * (self.epsilon_decay ** (self._episode_count - 1))
        self._current_epsilon = max(self.epsilon_min, _decayed_eps)

        # Per-term reward accumulators + anti-collapse tracker (2026-04-15).
        # Reward breakdown surfaces in the episode diagnostic JSONL so we can
        # tell which term is actually driving learning.
        _reward_sums = {
            "sdk_extrinsic": 0.0,
            "curiosity_intrinsic": 0.0,
            "novelty_shaped": 0.0,   # grid order/entropy/diff/navigation bonuses
            "goal_distance_g1": 0.0,
            "character_target_g2": 0.0,
        }
        _char_target_detected_steps = 0
        _recent_actions: list[int] = []
        _force_explore_remaining = 0

        # Phase A4: Apply neuromod scaling to strategy parameters.
        # Save base values before scaling (they're modified in-place by scaling).
        _base_stuck = self.stuck_threshold
        _base_curiosity = self.curiosity_bonus
        _base_curiosity_lg = self.curiosity_bonus_large
        _base_resets = self.max_resets
        _base_accum_thresh = self.ns_accum_threshold
        _neuro_scales = self._apply_neuromod_scaling()
        if _neuro_scales:
            logger.info("[ArcSession] Neuromod scaling: eps=%.3f stuck=%d curiosity=%.2f %s",
                        self._current_epsilon, self.stuck_threshold,
                        self.curiosity_bonus, _neuro_scales)

        # Reset NS accumulators for new episode
        if self.ns_accumulation_enabled:
            self._ns_accumulators = {name: 0.0 for name in self._ns_programs}
        # Reset perceptual grounding for new episode
        if self._mini_reasoning:
            self._mini_reasoning.reset()
        # Reset HAOV per-episode state (keep verified rules)
        if self._haov:
            self._haov.reset_episode()

        # Reset game
        frame = self._sdk.reset(game_id)
        if not frame:
            logger.warning("[ArcSession] Failed to reset game %s", game_id)
            return None

        logger.info("[ArcSession] Starting %s (%d levels, max %d steps, eps=%.3f)",
                    game_id, frame.win_levels, self._max_steps, self._current_epsilon)

        actions = []
        rewards = []
        levels_at_step = []
        nervous_fires = {}
        step = 0
        steps_since_level_change = 0
        reset_count = 0
        mem = self.state_memory  # may be None

        if mem:
            mem.reset_episode_counter()

        # SOAR+CGN: register ARC verifier once (if CGN available)
        if self._cgn and not self._haov_cgn_synced:
            try:
                if hasattr(self._cgn, 'register_verifier'):
                    def _arc_verify(obs_before, obs_after, action_ctx):
                        """ARC-specific hypothesis verifier for CGN GHAOV."""
                        pre_order = obs_before.get("semantic", [0.5]*5)[3] if isinstance(obs_before.get("semantic"), list) else 0.5
                        post_order = obs_after.get("semantic", [0.5]*5)[3] if isinstance(obs_after.get("semantic"), list) else 0.5
                        pre_entropy = obs_before.get("inner_body", [0.5]*5)[1] if isinstance(obs_before.get("inner_body"), list) else 0.5
                        post_entropy = obs_after.get("inner_body", [0.5]*5)[1] if isinstance(obs_after.get("inner_body"), list) else 0.5
                        order_improved = post_order > pre_order + 0.01
                        entropy_decreased = post_entropy < pre_entropy - 0.01
                        confirmed = order_improved or entropy_decreased
                        error = abs(post_order - pre_order) + abs(post_entropy - pre_entropy)
                        return confirmed, error
                    self._cgn.register_verifier("reasoning", _arc_verify)
                    self._haov_cgn_synced = True
            except Exception:
                pass

        while step < self._max_steps and not frame.done:
            # Anti-collapse watchdog (2026-04-15): if the last N actions are
            # >80% concentrated on one action, force epsilon high for the next
            # M steps to break the deterministic-scorer + decayed-epsilon lock
            # that produced single-action episodes (e.g. all 500 ft09 steps = 6).
            _epsilon_override: Optional[float] = None
            if self.anti_collapse_enabled and training:
                if _force_explore_remaining > 0:
                    _epsilon_override = self.anti_collapse_force_epsilon
                    _force_explore_remaining -= 1
                elif len(_recent_actions) >= self.anti_collapse_window:
                    _window = _recent_actions[-self.anti_collapse_window:]
                    _max_freq = max(_window.count(a) for a in set(_window))
                    if _max_freq / self.anti_collapse_window >= self.anti_collapse_threshold:
                        _force_explore_remaining = self.anti_collapse_force_steps
                        _epsilon_override = self.anti_collapse_force_epsilon
                        logger.warning(
                            "[ArcSession] Anti-collapse triggered at step %d — "
                            "action %d dominated %d/%d of window; forcing epsilon=%.2f for %d steps",
                            step, max(set(_window), key=_window.count),
                            _max_freq, self.anti_collapse_window,
                            self.anti_collapse_force_epsilon, self.anti_collapse_force_steps,
                        )

            # SOAR: impasse detection before crude reset (smarter stuck handling)
            if (self._cgn and hasattr(self._cgn, 'detect_impasse')
                    and steps_since_level_change >= self.stuck_threshold * 0.7
                    and steps_since_level_change < self.stuck_threshold):
                _impasse = self._cgn.detect_impasse("reasoning")
                if _impasse:
                    _imp_type = _impasse.get("type", "?")
                    logger.info("[ArcSession] SOAR impasse: %s — HAOV will guide next action",
                                _imp_type)
                    # rFP Step D (2026-04-20): on CGN reasoning impasse, emit
                    # CGN_KNOWLEDGE_REQ so knowledge_worker scrapes relevant
                    # concepts. Best-effort; 3s timeout; cooldown 300s per session
                    # so we don't hammer the endpoint every step during an
                    # extended impasse.
                    _now_ts = time.time()
                    if (_now_ts - getattr(self, "_last_impasse_kreq_ts", 0)) > 300:
                        try:
                            import requests as _rq
                            _topic = f"ARC {game_id} puzzle strategy: {_imp_type}"
                            _rq.post(
                                "http://127.0.0.1:7777/v4/knowledge-request",
                                json={"topic": _topic, "urgency": 0.6},
                                timeout=3.0,
                            )
                            self._last_impasse_kreq_ts = _now_ts
                            logger.info(
                                "[ArcSession] Impasse→knowledge request emitted: %s",
                                _topic)
                        except Exception as _kr_err:
                            logger.debug(
                                "[ArcSession] knowledge-request emit failed: %s",
                                _kr_err)
                    # Don't reset — let HAOV hypothesis testing guide next steps

            # IMPROVEMENT #3: Strategic Reset — when stuck, restart with knowledge
            if (steps_since_level_change >= self.stuck_threshold
                    and reset_count < self.max_resets):
                new_states = mem.new_transitions_this_episode if mem else 0
                logger.info(
                    "[ArcSession] STRATEGIC RESET at step %d — stuck %d steps, "
                    "%d new states discovered, reset #%d",
                    step, steps_since_level_change, new_states, reset_count + 1,
                )
                # Reset the game — SDK restarts from level beginning
                reset_frame = self._sdk.reset(game_id)
                if reset_frame is not None:
                    frame = reset_frame
                    self._perception.reset()
                    steps_since_level_change = 0
                    reset_count += 1
                    if mem:
                        mem.reset_episode_counter()
                    continue  # Start fresh loop iteration with reset frame

            # Hash current state for memory graph
            state_hash = mem.hash_state(frame.grid) if mem else ""

            # 1. Perceive: grid → Trinity tensors (A5: pass ACh for quadrant focus)
            _ach = self.real_neuromods.get("ACh", 0.5) if self.real_neuromods else 0.5
            features = self._perception.perceive(
                grid=frame.grid,
                reward=frame.reward,
                available_actions=len(frame.available_actions),
                step=step,
                ach_level=_ach,
            )

            # 1b. Perceptual Grounding: mini-reasoning + pattern primitives
            _pgl_block = None
            if self._mini_reasoning and self._pattern_primitives:
                _pgl_grid = np.asarray(frame.grid, dtype=np.float64)
                _pgl_profile = self._pattern_primitives.compute_profile(_pgl_grid)
                _pgl_action_id = actions[-1] if actions else -1
                _pgl_block = self._mini_reasoning.process_step(
                    _pgl_grid, action_id=_pgl_action_id, pattern_profile=_pgl_profile)
                self._last_pattern_block = _pgl_block
                # Surprise → NS boost: boost CURIOSITY accumulator on surprise
                if (_pgl_block.surprise > 0.4 and self.ns_accumulation_enabled
                        and self._ns_accumulators):
                    _boost = _pgl_block.surprise * self._surprise_ns_boost
                    if "CURIOSITY" in self._ns_accumulators:
                        self._ns_accumulators["CURIOSITY"] += _boost
                    if "CREATIVITY" in self._ns_accumulators and _pgl_block.surprise > 0.6:
                        self._ns_accumulators["CREATIVITY"] += _boost * 0.5
                    if step % 50 == 0 or _pgl_block.surprise > 0.7:
                        logger.info("[ArcSession] SURPRISE=%.2f → CURIOSITY boost +%.2f | %s",
                                    _pgl_block.surprise, _boost, _pgl_block.salient_observation)
                # Inject pattern profile into features for enhanced scorer input
                if _pgl_profile:
                    features["pattern_profile"] = self._pattern_primitives.profile_to_vector(_pgl_profile)

                # CGN: ground surprising pattern discoveries as reasoning concepts
                if (self._cgn and _pgl_block.surprise > 0.3
                        and step % 20 == 0):  # Throttle: every 20 steps max
                    try:
                        from titan_plugin.logic.cgn import ConceptFeatures, SensoryContext
                        _pname = _pgl_block.best_match or "unknown"
                        _pconcept = ConceptFeatures(
                            concept_id=f"pattern_{_pname}",
                            embedding=np.zeros(130, dtype=np.float32),
                            confidence=1.0 - _pgl_block.surprise,
                            encounter_count=1,
                        )
                        for _pi, _pp in enumerate(["symmetry", "translation", "alignment",
                                                    "containment", "adjacency", "repetition", "shape"]):
                            _pconcept.embedding[_pi] = _pgl_block.pattern_deltas.get(_pp, 0.0)
                        _pctx = SensoryContext(
                            encounter_type="arc_discovery",
                            neuromods=self.real_neuromods or {},
                        )
                        self._cgn.ground("reasoning", _pconcept, _pctx)
                    except Exception:
                        pass  # Non-critical — don't break ARC play

            # 1c. HAOV: form hypotheses from pattern observations (Phase A2)
            if (self._haov_enabled and self._haov and _pgl_block
                    and _pgl_block.surprise > 0.2 and actions):
                _last_action = actions[-1]
                _haov_info = {
                    "best_match": _pgl_block.best_match,
                    "surprise": _pgl_block.surprise,
                    "pattern_deltas": _pgl_block.pattern_deltas,
                    "salient_observation": getattr(_pgl_block, "salient_observation", ""),
                }
                self._haov.hypothesize(_last_action, _haov_info, features)

            # 2. Think: get personality signals from read-only NS programs
            signals = self._get_ns_signals(features)

            # Track nervous fires
            for sig in signals:
                prog = sig["system"]
                nervous_fires[prog] = nervous_fires.get(prog, 0) + 1

            # 2b. REASON: deliberate cognition when stuck or complex state detected
            reasoning_bias = None
            if self.reasoning_engine and steps_since_level_change >= self._reasoning_steps_threshold:
                reasoning_bias = self._reason_about_grid(
                    features, signals, steps_since_level_change, frame.available_actions)

            # 2c. FORWARD MODEL: 1-step lookahead planning (Phase A1)
            # Build flat feature vector for forward model (30D from Trinity layers)
            _fm_features = None
            _fm_action = None
            if self._forward_model:
                _fm_features = np.concatenate([
                    features.get("inner_body", [0.5]*5),
                    features.get("inner_mind", [0.5]*5),
                    features.get("inner_spirit", [0.5]*5),
                    features.get("spatial", [0.5]*5),
                    features.get("semantic", [0.5]*5),
                    features.get("resonance", [0.5]*5),
                ]).astype(np.float32)

                # Record previous transition (state → action → next_state)
                if self._prev_features_flat is not None and actions:
                    self._forward_model.record_transition(
                        self._prev_features_flat, actions[-1], _fm_features)
                    # Train every 8 steps (lightweight)
                    if step % 8 == 0 and self._forward_model._buffer:
                        self._forward_model.train_step(batch_size=min(32, len(self._forward_model._buffer)))
                self._prev_features_flat = _fm_features

                # Enable lookahead once we have enough training data
                if (not self._lookahead_enabled
                        and len(self._forward_model._buffer) >= self._lookahead_min_buffer):
                    self._lookahead_enabled = True
                    logger.info("[ArcSession] Forward model LOOKAHEAD enabled (buffer=%d)",
                                len(self._forward_model._buffer))

                # Enable HAOV once forward model has some data
                if (not self._haov_enabled and self._haov
                        and len(self._forward_model._buffer) >= self._haov_min_fm_buffer):
                    self._haov_enabled = True
                    logger.info("[ArcSession] HAOV loop enabled (fm_buffer=%d)",
                                len(self._forward_model._buffer))

            # 3. Act: consult state memory first, then mapper
            action = None

            # Phase A3: Sequence replay — replay entire winning action chains
            if mem and mem._winning_sequences:
                _seq = mem.suggest_sequence(state_hash, frame.available_actions)
                if _seq and not self._active_sequence:
                    self._active_sequence = _seq
                    self._sequence_step = 0
                    logger.info("[ArcSession] Starting sequence replay: %d actions", len(_seq))

            # Execute active sequence if one is running
            if self._active_sequence and self._sequence_step < len(self._active_sequence):
                seq_action = self._active_sequence[self._sequence_step]
                if seq_action in frame.available_actions:
                    action = seq_action
                    self._sequence_step += 1
                    if self._sequence_step >= len(self._active_sequence):
                        logger.info("[ArcSession] Sequence replay complete (%d actions)",
                                    len(self._active_sequence))
                        self._active_sequence = None
                        self._sequence_step = 0
                else:
                    # Sequence diverged — abort
                    logger.info("[ArcSession] Sequence aborted at step %d (action %d unavailable)",
                                self._sequence_step, seq_action)
                    self._active_sequence = None
                    self._sequence_step = 0

            # Phase A2: HAOV hypothesis testing or verified-rule action
            if action is None and self._haov_enabled and self._haov and training:
                # 25% of the time: test a hypothesis (scientific exploration)
                if np.random.random() < self._haov_test_probability:
                    _haov_action = self._haov.select_test_action(
                        frame.available_actions, features)
                    if _haov_action is not None:
                        action = _haov_action
                # Otherwise: use verified rules for exploitation
                if action is None:
                    _haov_action = self._haov.suggest_action(frame.available_actions)
                    if _haov_action is not None:
                        action = _haov_action

            if action is None and mem and not training:
                # In eval mode, prefer memory-suggested actions (exploit knowledge)
                action = mem.suggest_action(state_hash, frame.available_actions)

            # 3b. FORWARD MODEL LOOKAHEAD: predict outcomes for all actions
            if (action is None and self._forward_model and self._lookahead_enabled
                    and _fm_features is not None and len(frame.available_actions) > 1):
                # Predict next features for each available action
                predictions = self._forward_model.predict_all_actions(
                    _fm_features, frame.available_actions)
                # Score each predicted state using action-scorer
                best_score = -float("inf")
                best_fm_action = None
                for a, pred_features in predictions.items():
                    # Use existing scorer to evaluate predicted state
                    pred_dict = {
                        "inner_body": pred_features[:5].tolist(),
                        "inner_mind": pred_features[5:10].tolist(),
                        "inner_spirit": pred_features[10:15].tolist(),
                        "spatial": pred_features[15:20].tolist(),
                        "semantic": pred_features[20:25].tolist(),
                        "resonance": pred_features[25:30].tolist(),
                    }
                    score = self._mapper.score_state(pred_dict, a)
                    if score > best_score:
                        best_score = score
                        best_fm_action = a
                # Use lookahead action with probability (1 - epsilon)
                _lookahead_eps = _epsilon_override if _epsilon_override is not None else self._current_epsilon
                if best_fm_action is not None and np.random.random() > _lookahead_eps:
                    action = best_fm_action
                    _fm_action = best_fm_action
                    self._fm_lookahead_used_count += 1
                else:
                    self._fm_scorer_used_count += 1

            if action is None:
                # If reasoning committed with "explore" bias, boost epsilon
                _base_eps = _epsilon_override if _epsilon_override is not None else self._current_epsilon
                effective_epsilon = _base_eps if training else 0.0
                if reasoning_bias == "explore":
                    effective_epsilon = min(1.0, effective_epsilon + 0.3)
                elif reasoning_bias == "exploit" and mem:
                    # Reasoning says exploit — try memory suggestion even in training
                    action = mem.suggest_action(state_hash, frame.available_actions)

            if action is None:
                action = self._mapper.select_action(
                    available_actions=frame.available_actions,
                    nervous_signals=signals,
                    grid_features=features,
                    epsilon=effective_epsilon if training else 0.0,
                )

            if action is None:
                logger.warning("[ArcSession] No action available — ending")
                break

            actions.append(action)
            levels_at_step.append(frame.levels_completed)
            _recent_actions.append(action)  # anti-collapse watchdog window

            # 4. Execute action via SDK
            next_frame = self._sdk.step(game_id, action)
            if next_frame is None:
                logger.warning("[ArcSession] SDK step failed — ending")
                break

            # 5. Compute reward: extrinsic (level completion) + intrinsic (curiosity)
            reward_extrinsic = next_frame.reward
            reward_intrinsic = 0.0

            if mem:
                next_hash = mem.hash_state(next_frame.grid)
                is_novel = mem.record(state_hash, action, next_hash, reward_extrinsic)

                # IMPROVEMENT #1: Intrinsic curiosity reward
                if is_novel:
                    # Novel state bonus — dense reward for exploration
                    delta = features.get("inner_mind", [0]*5)[3]  # delta_from_prev
                    if delta > self.large_delta_threshold:
                        reward_intrinsic = self.curiosity_bonus_large
                    else:
                        reward_intrinsic = self.curiosity_bonus
                else:
                    # Revisiting known state — small novelty from visit count
                    novelty = mem.get_novelty_ratio(next_hash)
                    if novelty > 0.5:
                        reward_intrinsic = self.curiosity_bonus * 0.3 * novelty

            # SHAPED REWARD: gradient toward goal using 30D semantic features
            # Peek at next frame's features to compute progress signal
            reward_shaped = 0.0
            _novelty_shaped = 0.0  # accumulate novelty portion separately so we can cap it
            _next_grid = np.asarray(next_frame.grid, dtype=np.float64)
            # Quick feature extraction on next grid (reuse helpers, no state mutation)
            _prev_order = features.get("semantic", [0]*5)[3]  # structural_order
            _prev_entropy = features.get("inner_body", [0]*5)[1]  # color_entropy
            # Compute next grid's order: regularity × symmetry
            _next_regularity = self._perception._pattern_regularity(_next_grid)
            _next_symmetry = (self._perception._mirror_symmetry(_next_grid, "horizontal") +
                              self._perception._mirror_symmetry(_next_grid, "vertical")) / 2.0
            _next_order = _next_regularity * 0.6 + _next_symmetry * 0.4
            _next_entropy = self._perception._shannon_entropy(_next_grid)
            # Reward grid becoming more ordered
            if _next_order > _prev_order + 0.02:
                _novelty_shaped += 0.1
            # Reward entropy decrease (simplification)
            if _next_entropy < _prev_entropy - 0.02:
                _novelty_shaped += 0.05
            # Reward significant grid change (action had real effect)
            if self._perception._prev_grid is not None:
                _diff_frac = float(np.sum(_next_grid != self._perception._prev_grid)) / max(1, _next_grid.size)
                if _diff_frac > 0.15:
                    _novelty_shaped += 0.08

            # NAVIGATION REWARD: for movement-based games (ls20)
            # Track where changes happen — consistent directional movement = progress
            _prev_spatial = features.get("spatial", [0.5] * 5)
            if self._perception._prev_grid is not None:
                _diff_mask = (_next_grid != self._perception._prev_grid)
                if _diff_mask.any():
                    _rows, _cols = _next_grid.shape
                    _ys, _xs = np.where(_diff_mask)
                    _cx = float(_xs.mean()) / max(1, _cols - 1)
                    _cy = float(_ys.mean()) / max(1, _rows - 1)
                    # Reward movement to new spatial region (away from center of prev changes)
                    _prev_cx = _prev_spatial[0]  # change_centroid_x
                    _prev_cy = _prev_spatial[1]  # change_centroid_y
                    _movement = (((_cx - _prev_cx) ** 2 + (_cy - _prev_cy) ** 2) ** 0.5)
                    if _movement > 0.05:
                        _novelty_shaped += 0.12  # moving to new area!
                    # Reward concentrated changes (focused action, not noise)
                    if len(_xs) > 1:
                        _disp = (float(np.std(_xs)) / max(1, _cols) +
                                 float(np.std(_ys)) / max(1, _rows)) / 2.0
                        if _disp < 0.1:
                            _novelty_shaped += 0.06  # precise, focused change

            # Cap the novelty portion so task-directed rewards (G1/G2) can dominate.
            # Pre-2026-04-15, uncapped novelty summed to ~0.41/step and drowned G2
            # (~0.015/step). Cap + rebalanced k lets scorer see the gradient.
            if self.novelty_reward_cap_per_step > 0:
                _novelty_shaped = min(_novelty_shaped, self.novelty_reward_cap_per_step)
            reward_shaped += _novelty_shaped

            # ── GOAL-SIGNAL REWARD (rFP_arc_training_fix 2026-04-13) ──
            # G1 empirical capture: if we've seen a WIN for this game, reward
            # is proportional to similarity improvement toward the goal grid.
            # G2 ls20 heuristic: if no goal yet, use character-target manhattan.
            # Feature-flagged via coefficients (0 = disabled).
            _g1_term = 0.0
            _g2_term = 0.0
            if self._goal_detector is not None and (
                self.goal_distance_reward_k > 0 or self.character_target_reward_k > 0
            ):
                _prev_grid_np = self._perception._prev_grid
                if _prev_grid_np is not None:
                    # G1: goal-distance delta (only if we have a captured goal)
                    if self.goal_distance_reward_k > 0:
                        _goal = self._goal_detector.get_goal(game_id)
                        if _goal is not None:
                            _delta = self._goal_detector.goal_distance_delta(
                                prev_grid=_prev_grid_np,
                                new_grid=_next_grid,
                                goal_grid=_goal,
                            )
                            _g1_term = self.goal_distance_reward_k * _delta
                            reward_shaped += _g1_term

                    # G2: ls20 character-target (only for ls20, only if no goal yet
                    # — after first win, G1's similarity is a stronger signal)
                    if (self.character_target_reward_k > 0 and game_id == "ls20"
                            and not self._goal_detector.has_goal(game_id)):
                        _target = self._goal_detector.detect_target(
                            _next_grid.astype(np.int8), background=0)
                        if _target is not None:
                            # Character color = the color that moved between prev and curr
                            _char_pos = self._goal_detector.detect_character(
                                _prev_grid_np, _next_grid)
                            if _char_pos is not None:
                                _char_color = int(_next_grid[_char_pos])
                                if _char_color != 0:
                                    _ct_delta = self._goal_detector.character_target_reward(
                                        prev_grid=_prev_grid_np,
                                        curr_grid=_next_grid,
                                        character_color=_char_color,
                                        target=_target,
                                        normalize_by_grid=self.character_target_normalize_by_grid,
                                    )
                                    _g2_term = self.character_target_reward_k * _ct_delta
                                    reward_shaped += _g2_term
                                    _char_target_detected_steps += 1

            # Per-term reward accumulators (diagnostic)
            _reward_sums["sdk_extrinsic"] += float(reward_extrinsic)
            _reward_sums["curiosity_intrinsic"] += float(reward_intrinsic)
            _reward_sums["novelty_shaped"] += float(_novelty_shaped)
            _reward_sums["goal_distance_g1"] += float(_g1_term)
            _reward_sums["character_target_g2"] += float(_g2_term)

            reward_total = reward_extrinsic + reward_intrinsic + reward_shaped

            # Phase A2: HAOV verification — compare prediction to reality
            _haov_verified = None
            if self._haov_enabled and self._haov:
                # Use features from next frame for verification (lightweight — reuse existing vars)
                _verify_features = features  # fallback: use pre-action features
                if next_frame:
                    # Quick extraction of next grid features for verification
                    _next_grid_np = np.asarray(next_frame.grid, dtype=np.float64)
                    _verify_features = {
                        "inner_body": features.get("inner_body", [0.5]*5),
                        "inner_mind": features.get("inner_mind", [0.5]*5),
                        "inner_spirit": features.get("inner_spirit", [0.5]*5),
                        "spatial": [_next_order, _next_entropy, 0.5, 0.5, 0.5],
                        "semantic": features.get("semantic", [0.5]*5),
                        "resonance": features.get("resonance", [0.5]*5),
                    }
                    # Update semantic order from computed values
                    _v_sem = list(_verify_features["semantic"])
                    _v_sem[3] = _next_order  # structural_order
                    _verify_features["semantic"] = _v_sem
                    _v_body = list(_verify_features["inner_body"])
                    _v_body[1] = _next_entropy  # color_entropy
                    _verify_features["inner_body"] = _v_body
                _is_novel = False
                if mem:
                    try:
                        _is_novel = is_novel
                    except NameError:
                        _is_novel = False
                _haov_verified = self._haov.verify(
                    action, _verify_features, reward_total, _is_novel)
                if _haov_verified:
                    # Bonus reward for confirmed hypotheses (scientific discovery!)
                    if _haov_verified["confirmed"]:
                        reward_total += 0.05 * _haov_verified["confidence"]

                # Also: hypothesize from forward model prediction errors
                if (self._forward_model and _fm_features is not None
                        and _fm_action == action and step % 5 == 0):
                    # If forward model was very wrong, that's surprising — hypothesize
                    _fm_next = np.concatenate([
                        _verify_features.get("inner_body", [0.5]*5),
                        _verify_features.get("inner_mind", [0.5]*5),
                        _verify_features.get("inner_spirit", [0.5]*5),
                        _verify_features.get("spatial", [0.5]*5),
                        _verify_features.get("semantic", [0.5]*5),
                        _verify_features.get("resonance", [0.5]*5),
                    ]).astype(np.float32)
                    _fm_acc = self._forward_model.prediction_accuracy(
                        _fm_features, action, _fm_next)
                    if _fm_acc < 0.3:  # Very surprising outcome
                        _fm_hyp_info = {
                            "best_match": "forward_model_surprise",
                            "surprise": 1.0 - _fm_acc,
                            "pattern_deltas": {"translation": abs(reward_total) * 0.5,
                                               "shape": (1.0 - _fm_acc) * 0.3},
                            "salient_observation": f"fm_error={1.0-_fm_acc:.2f}",
                        }
                        self._haov.hypothesize(action, _fm_hyp_info, _verify_features)

            # Stuck penalty (overrides small curiosity bonus when truly stuck)
            steps_since_level_change += 1
            if steps_since_level_change > self.stuck_threshold:
                reward_total = min(reward_total, self.stuck_penalty)

            rewards.append(reward_total)

            if training:
                self._mapper.record_outcome(action, reward_total, features)

            # Log level completions
            if next_frame.levels_completed > frame.levels_completed:
                logger.info(
                    "[ArcSession] Level %d/%d completed at step %d!",
                    next_frame.levels_completed, next_frame.win_levels, step,
                )
                steps_since_level_change = 0
                # Record level completion in memory
                if mem:
                    mem.record_level_completion(state_hash, actions[-20:])

            frame = next_frame
            step += 1

        duration = time.time() - t0

        result = EpisodeResult(
            game_id=game_id,
            steps=step,
            levels_completed=frame.levels_completed if frame else 0,
            win_levels=frame.win_levels if frame else 0,
            total_reward=round(sum(rewards), 4),
            actions=actions,
            rewards=rewards,
            levels_at_step=levels_at_step,
            nervous_fires=nervous_fires,
            reset_count=reset_count,
            duration_s=round(duration, 2),
            success=frame.state == "WIN" if frame else False,
            final_state=frame.state if frame else "ERROR",
        )

        self._episode_history.append(result)

        # ── GOAL CAPTURE + T1 DIAGNOSTIC DUMP (rFP_arc_training_fix 2026-04-13) ──
        # Capture final grid on WIN (G1 empirical); dump per-episode diagnostics
        # so we can iterate on reward design from real data (I8 rotation: one file per day).
        if self._goal_detector is not None and frame is not None and frame.grid is not None:
            try:
                # rFP Step C (2026-04-20): when ArcSession records a local WIN, pass
                # our titan_id so GoalDetector broadcasts the fresh goal to kin.
                # On non-WIN outcomes on_episode_end is a no-op anyway.
                _kin_src = os.environ.get("TITAN_KIN_SOURCE") or None
                self._goal_detector.on_episode_end(
                    game_id=game_id,
                    final_state=frame.state,
                    final_grid=np.asarray(frame.grid, dtype=np.int8),
                    source_titan_id=_kin_src,
                )
            except Exception as _gd_err:
                logger.warning("[ArcSession] GoalDetector.on_episode_end failed: %s", _gd_err)

        if self.episode_diagnostics_enabled:
            try:
                import json as _json
                import os as _os
                from datetime import datetime as _dt
                # I8: daily rotation — one file per day, keep 30 days (caller manages cleanup)
                _date = _dt.utcnow().strftime("%Y%m%d")
                _diag_dir = "data/arc_agi_3"
                _os.makedirs(_diag_dir, exist_ok=True)
                _diag_path = _os.path.join(_diag_dir, f"episode_diagnostics_{_date}.jsonl")
                _final_grid = frame.grid.astype(int).tolist() if frame and frame.grid is not None else None
                _goal_known = (
                    self._goal_detector is not None
                    and self._goal_detector.has_goal(game_id)
                )
                _sim_to_goal = None
                if _goal_known and _final_grid is not None:
                    _sim_to_goal = float(self._goal_detector.similarity(
                        np.array(_final_grid, dtype=np.int8),
                        self._goal_detector.get_goal(game_id),
                    ))
                _action_counts: dict[int, int] = {}
                for _a in result.actions:
                    _action_counts[_a] = _action_counts.get(_a, 0) + 1
                # Action entropy as a normalized collapse metric ∈ [0, 1]
                # (1.0 = uniform over available actions; 0.0 = single action)
                _total_acts = sum(_action_counts.values()) or 1
                _probs = [c / _total_acts for c in _action_counts.values()]
                import math as _math
                _action_entropy = -sum(p * _math.log(p) for p in _probs if p > 0)
                _max_entropy = _math.log(max(1, len(_action_counts)))
                _action_entropy_norm = (_action_entropy / _max_entropy) if _max_entropy > 0 else 0.0
                _diag = {
                    "ts_utc": _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "game_id": game_id,
                    "final_state": result.final_state,
                    "levels_completed": result.levels_completed,
                    "steps": result.steps,
                    "total_reward": result.total_reward,
                    "goal_known": bool(_goal_known),
                    "similarity_to_goal": _sim_to_goal,
                    "action_counts": _action_counts,
                    "action_entropy_norm": round(_action_entropy_norm, 3),
                    "reset_count": result.reset_count,
                    "duration_s": result.duration_s,
                    "epsilon_at_start": round(self._current_epsilon, 4),
                    "epsilon_min": self.epsilon_min,
                    "reward_breakdown": {k: round(v, 4) for k, v in _reward_sums.items()},
                    "character_target_signal": {
                        "detected_steps": _char_target_detected_steps,
                        "detection_rate": round(_char_target_detected_steps / max(1, result.steps), 3),
                    },
                    "reward_config": {
                        "goal_distance_reward_k": self.goal_distance_reward_k,
                        "character_target_reward_k": self.character_target_reward_k,
                        "novelty_reward_cap_per_step": self.novelty_reward_cap_per_step,
                    },
                }
                with open(_diag_path, "a") as _dfh:
                    _dfh.write(_json.dumps(_diag) + "\n")
            except Exception as _diag_err:
                logger.warning("[ArcSession] Episode diagnostic dump failed: %s", _diag_err)

        # Log perceptual grounding stats if enabled
        _pgl_info = ""
        if self._mini_reasoning:
            _pgl_stats = self._mini_reasoning.get_stats()
            _pgl_info = (f", surprise_rate={_pgl_stats['surprise_rate']:.2f}"
                         f", high_surprises={_pgl_stats['high_surprise_count']}"
                         f", actions_known={_pgl_stats['actions_known']}")

        # Log HAOV stats
        _haov_info = ""
        if self._haov_enabled and self._haov:
            _hs = self._haov.get_stats()
            _haov_info = (f", haov={_hs['formed']}H/{_hs['tested']}T/"
                          f"{_hs['confirmed']}C/{_hs['falsified']}F"
                          f" verified={_hs['verified_rules']}")
            if _hs["top_rules"]:
                _top = _hs["top_rules"][0]
                _haov_info += f" best={_top['rule']}({_top['conf']:.0%})"

            # Ground verified hypotheses in CGN as high-confidence reasoning concepts
            if self._cgn:
                for concept in self._haov.get_verified_concepts():
                    try:
                        from titan_plugin.logic.cgn import ConceptFeatures, SensoryContext
                        _hconcept = ConceptFeatures(
                            concept_id=concept["concept_id"],
                            embedding=np.zeros(130, dtype=np.float32),
                            confidence=concept["confidence"],
                            encounter_count=concept["tests"],
                            production_count=concept["confirmations"],
                        )
                        # Encode pattern type in embedding
                        _pattern_idx = {"symmetry": 0, "translation": 1, "alignment": 2,
                                        "containment": 3, "adjacency": 4, "repetition": 5,
                                        "shape": 6}.get(concept["pattern"], 7)
                        _hconcept.embedding[_pattern_idx] = concept["confidence"]
                        _hconcept.embedding[8] = {"order_up": 0.8, "entropy_down": 0.6,
                                                   "movement": 0.4, "novel_state": 0.2
                                                   }.get(concept["effect"], 0.1)
                        _hctx = SensoryContext(
                            encounter_type="arc_discovery",
                            neuromods=self.real_neuromods or {},
                        )
                        self._cgn.ground("reasoning", _hconcept, _hctx)
                        # Higher reward for verified hypotheses
                        self._cgn.record_outcome("reasoning", concept["concept_id"],
                                                 min(0.3, concept["confidence"] * 0.4), {
                                                     "source": "haov_verified",
                                                     "pattern": concept["pattern"],
                                                     "tests": concept["tests"],
                                                 })
                    except Exception:
                        pass

            # Sync local HAOV verified rules to CGN's generalized tracker
            if self._cgn and hasattr(self._cgn, 'get_haov'):
                _cgn_haov = self._cgn.get_haov("reasoning")
                if _cgn_haov:
                    for concept in self._haov.get_verified_concepts():
                        _cgn_haov.hypothesize(
                            action_context={"concept_id": concept["concept_id"]},
                            observation={
                                "effect": concept.get("effect", "verified"),
                                "magnitude": concept["confidence"],
                                "rule_name": concept["concept_id"],
                                "source": "arc_haov_sync",
                            },
                        )
                        # Set confidence directly on synced hypothesis
                        for h in _cgn_haov._hypotheses:
                            if h.rule == concept["concept_id"]:
                                h.confidence = concept["confidence"]
                                h.confirmations = concept.get("confirmations", 0)
                                h.tests = concept.get("tests", 0)
                                if h.confidence > 0.6 and h not in _cgn_haov._verified_rules:
                                    _cgn_haov._verified_rules.append(h)
                                break

        # Forward model telemetry — added 2026-04-08 to verify lookahead is
        # actually contributing to action selection (not just sitting idle).
        _fm_info = ""
        if self._forward_model:
            _fm_total = self._fm_lookahead_used_count + self._fm_scorer_used_count
            _fm_buffer = len(self._forward_model._buffer) if self._forward_model._buffer else 0
            _fm_pct = (100.0 * self._fm_lookahead_used_count / _fm_total) if _fm_total > 0 else 0
            _fm_info = (f" fm_lookahead={self._fm_lookahead_used_count}/{_fm_total}"
                        f" ({_fm_pct:.0f}%) buf={_fm_buffer} enabled={self._lookahead_enabled}")

        logger.info(
            "[ArcSession] Game complete: %s — %d steps, %d/%d levels, "
            "reward=%.2f, state=%s, %.1fs%s%s%s",
            game_id, step, result.levels_completed, result.win_levels,
            result.total_reward, result.final_state, duration, _pgl_info,
            _haov_info, _fm_info,
        )

        # Restore base strategy values (neuromod scaling is per-episode)
        self.stuck_threshold = _base_stuck
        self.curiosity_bonus = _base_curiosity
        self.curiosity_bonus_large = _base_curiosity_lg
        self.max_resets = _base_resets
        self.ns_accum_threshold = _base_accum_thresh

        return result

    def train_session(self, game_id: str, num_episodes: int = 10) -> SessionReport:
        """Run multiple training games, respecting dreaming cycles between episodes."""
        t0 = time.time()
        episodes = []

        for i in range(num_episodes):
            # Respect Titan's dreaming cycle between episodes
            if self.dreaming_check and i > 0:
                self.dreaming_check()

            logger.info("[ArcSession] Training game %d/%d (%s)",
                        i + 1, num_episodes, game_id)

            result = self.play_game(game_id, training=True)
            if result:
                episodes.append(result)

        return self._build_report(game_id, episodes, time.time() - t0)

    def evaluate(self, game_id: str, num_episodes: int = 5) -> SessionReport:
        """Run evaluation games (no training, no exploration)."""
        t0 = time.time()
        episodes = []

        for i in range(num_episodes):
            logger.info("[ArcSession] Evaluation %d/%d (%s)",
                        i + 1, num_episodes, game_id)

            result = self.play_game(game_id, training=False)
            if result:
                episodes.append(result)

        return self._build_report(game_id, episodes, time.time() - t0)

    def _get_ns_signals(self, grid_features: dict) -> list[dict]:
        """
        Get personality signals from read-only NS program copies.

        Maps 15D ARC grid features into 55D NS observable format,
        then forwards through each program. Programs that exceed
        their fire threshold return as active signals.

        This gives Titan's personality influence over ARC strategy:
        - CURIOSITY: fires when grid has high entropy → explore
        - INTUITION: fires when patterns detected → repeat known patterns
        - CREATIVITY: fires when stuck → try novel actions
        - FOCUS: fires when making progress → exploit current strategy
        """
        if not self._ns_programs:
            return []

        # ── "Feel the Puzzle" — wire ARC perception into full 55D NS input ──
        # Inner Trinity: direct grid features (what the puzzle IS)
        body = grid_features.get("inner_body", [0.5] * 5)
        mind = grid_features.get("inner_mind", [0.5] * 5)
        spirit = grid_features.get("inner_spirit", [0.5] * 5)
        # Outer Body: spatial awareness (WHERE things change = body perception)
        outer_body = grid_features.get("spatial", [0.5] * 5)
        # Outer Mind: semantic features (WHAT the puzzle MEANS = mind perception)
        outer_mind = grid_features.get("semantic", [0.5] * 5)
        # Outer Spirit: resonance features (HOW it echoes inside = spirit perception)
        outer_spirit = grid_features.get("resonance", [0.5] * 5)
        # Tier 2: blend of all layers for deeper NS input (25D)
        # First 5D: body-mind cross-features, next 5D: stuck/progress signals,
        # remaining 15D: repeat key signals at different scales for gradient depth
        tier2_cross = [
            body[0] * mind[0],   # density × objects
            body[2] * mind[4],   # symmetry × regularity
            spirit[2],           # stuck_indicator (raw — critical for REFLECTION)
            spirit[0] * spirit[4],  # exploration × progress
            outer_mind[4] if len(outer_mind) > 4 else 0.5,  # narrative_weight
        ]
        tier2_progress = [
            spirit[1],           # reward_trend
            1.0 - spirit[2],    # inverse stuck = progress indicator
            mind[3],             # delta_from_prev (novelty)
            outer_spirit[1] if len(outer_spirit) > 1 else 0.5,  # exploration_reward
            outer_spirit[3] if len(outer_spirit) > 3 else 0.5,  # progress_confidence
        ]
        # Remaining 15D: repeat key features at dampened scale for gradient stability
        tier2_echo = ([v * 0.7 for v in body] +
                      [v * 0.7 for v in mind] +
                      [v * 0.7 for v in spirit[:5]])
        tier2 = tier2_cross + tier2_progress + tier2_echo

        input_55d = np.array(
            body + mind + spirit + outer_body + outer_mind + outer_spirit + tier2,
            dtype=np.float64,
        )
        # Truncate to 55D (standard feature set) in case of mismatch
        input_55d = input_55d[:55]

        # Compute urgency for all programs
        raw_urgencies = {}
        for name, net in self._ns_programs.items():
            try:
                inp = input_55d[:net.input_dim]
                if len(inp) < net.input_dim:
                    inp = np.pad(inp, (0, net.input_dim - len(inp)),
                                 constant_values=0.5)
                raw_urgencies[name] = float(net.forward(inp))
            except Exception as e:
                logger.debug("[ArcSession] NS program %s failed: %s", name, e)

        if not raw_urgencies:
            return []

        # ── Improvement #7: NS Accumulation Model ──
        # Instead of single-pass fire, accumulate urgency over steps like
        # hormones build up over time. Programs fire when accumulated signal
        # exceeds threshold, then reset (refractory). This lets ALL programs
        # contribute over time, not just the top 2 per step.
        if self.ns_accumulation_enabled and self._ns_accumulators:
            signals = []
            for name, raw in raw_urgencies.items():
                # Decay + accumulate
                self._ns_accumulators[name] = (
                    self._ns_accumulators.get(name, 0.0) * self.ns_accum_decay + raw
                )
                # Fire when accumulated signal exceeds threshold
                if self._ns_accumulators[name] > self.ns_accum_threshold:
                    signals.append({
                        "system": name,
                        "urgency": min(1.0, self._ns_accumulators[name] / (self.ns_accum_threshold * 2)),
                        "intensity": min(1.0, self._ns_accumulators[name] / (self.ns_accum_threshold * 2)),
                        "learned": True,
                        "hormonal": True,  # accumulated like hormones
                    })
                    # Refractory reset
                    self._ns_accumulators[name] = 0.0
            return signals

        # Fallback: relative ranking (original method)
        # NS programs were trained on consciousness data, so ARC inputs produce
        # low absolute values (0.001-0.01). Use RELATIVE ranking: normalize to
        # [0,1] range and fire the top programs above median.
        min_u = min(raw_urgencies.values())
        max_u = max(raw_urgencies.values())
        spread = max_u - min_u
        if spread < 1e-10:
            # All equal — no personality differentiation
            return []

        signals = []
        for name, raw in raw_urgencies.items():
            normalized = (raw - min_u) / spread  # 0.0 to 1.0
            # Fire programs in the top half (normalized > 0.5)
            if normalized > 0.4:
                signals.append({
                    "system": name,
                    "urgency": normalized,
                    "intensity": normalized,
                    "learned": True,
                    "hormonal": False,
                })

        return signals

    def _reason_about_grid(
        self, features: dict, ns_signals: list, stuck_steps: int,
        available_actions: list[int],
    ) -> Optional[str]:
        """Run reasoning chain on current grid state.

        Uses the same 7 logic primitives as Titan's main reasoning engine,
        but applied to ARC grid perception features.

        Returns:
            "explore" — reasoning concluded: try something different
            "exploit" — reasoning concluded: repeat known strategy
            None — reasoning was inconclusive or didn't fire
        """
        re = self.reasoning_engine
        if not re:
            return None

        # Build observation from grid features (30D: body+mind+spirit+spatial+semantic+resonance)
        body = features.get("inner_body", [0.5] * 5)
        mind = features.get("inner_mind", [0.5] * 5)
        spirit = features.get("inner_spirit", [0.5] * 5)
        spatial = features.get("spatial", [0.5] * 5)
        semantic = features.get("semantic", [0.5] * 5)
        resonance = features.get("resonance", [0.5] * 5)
        obs_30d = np.array(body + mind + spirit + spatial + semantic + resonance, dtype=np.float64)

        # Pad to match reasoning engine's expected input (79D enriched)
        obs_padded = np.zeros(79, dtype=np.float64)
        obs_padded[:min(30, len(obs_30d))] = obs_30d[:30]
        # Fill tier2 region [30:55] with cross-features for richer input
        obs_padded[30] = spirit[2]       # stuck_indicator
        obs_padded[31] = spirit[1]       # reward_trend
        obs_padded[32] = mind[3]         # delta_from_prev (novelty)
        obs_padded[33] = spirit[0]       # exploration_rate
        obs_padded[34] = spirit[4]       # episode_progress
        # Fill neuromod region [55:67] from stuck state
        stuck_norm = min(1.0, stuck_steps / max(1, self.stuck_threshold))
        obs_padded[55] = stuck_norm                # "urgency" signal
        obs_padded[56] = 1.0 - stuck_norm         # "patience" remaining
        obs_padded[57] = len(available_actions) / 7.0  # action diversity

        # Build gut signals from NS urgencies
        gut_signals = {s["system"]: s["urgency"] for s in ns_signals}
        # Add stuck signal as synthetic gut
        gut_signals["STUCK"] = stuck_norm

        # Body state for reasoning — prefer real Titan state, fallback to synthetic
        if self.real_body_state:
            body_state = self.real_body_state
        else:
            body_state = {
                "fatigue": stuck_norm * 0.5,
                "chi_total": 1.0 - stuck_norm * 0.3,
                "is_dreaming": False,
            }

        # Neuromod state — prefer real Titan neuromods, fallback to synthetic
        if self.real_neuromods:
            raw_neuromods = {k: max(0.1, v) for k, v in self.real_neuromods.items()}
        else:
            # Synthetic fallback (grid features as proxy)
            raw_neuromods = {
                "DA": max(0.1, spirit[1]),
                "5-HT": max(0.1, 1.0 - spirit[2]),
                "NE": max(0.1, mind[3]),
                "ACh": max(0.1, body[0]),
                "Endorphin": max(0.1, spirit[4]),
                "GABA": max(0.1, stuck_norm * 0.3),
            }

        # Run reasoning tick
        result = re.tick(
            observation=obs_padded,
            gut_signals=gut_signals,
            body_state=body_state,
            raw_neuromods=raw_neuromods,
            working_memory_items=[],
            dt=1.0,
        )

        action = result.get("action", "IDLE")
        self._reasoning_chains_fired += 1 if action in ("COMMIT", "ABANDON", "CONTINUE") else 0

        if action == "COMMIT":
            self._reasoning_commits += 1
            conf = result.get("confidence", 0.5)
            chain_len = result.get("chain_length", 0)
            # Analyze reasoning chain to determine bias
            chain = result.get("chain", [])
            plan = result.get("reasoning_plan", {})

            # High confidence + decomposition → exploit (found a pattern)
            if conf > 0.7 and "DECOMPOSE" in chain:
                logger.info("[ARC-REASON] COMMIT(exploit) conf=%.2f chain=%d plan=%s",
                            conf, chain_len, plan.get("intent", "?"))
                return "exploit"
            # Lower confidence or association → explore (need more info)
            else:
                logger.info("[ARC-REASON] COMMIT(explore) conf=%.2f chain=%d plan=%s",
                            conf, chain_len, plan.get("intent", "?"))
                return "explore"

        elif action == "ABANDON":
            # Reasoning couldn't build confidence — explore
            if stuck_steps > self.stuck_threshold * 0.7:
                logger.debug("[ARC-REASON] ABANDON at stuck=%d → forced explore", stuck_steps)
                return "explore"

        # IDLE or CONTINUE — reasoning still thinking, no bias yet
        return None

    def _build_report(self, game_id: str, episodes: list[EpisodeResult],
                      duration: float) -> SessionReport:
        """Build aggregate session report."""
        if episodes:
            avg_reward = sum(e.total_reward for e in episodes) / len(episodes)
            avg_steps = sum(e.steps for e in episodes) / len(episodes)
            avg_levels = sum(e.levels_completed for e in episodes) / len(episodes)
            best_levels = max(e.levels_completed for e in episodes)
            best_reward = max(e.total_reward for e in episodes)
        else:
            avg_reward = avg_steps = avg_levels = 0.0
            best_levels = 0
            best_reward = 0.0

        return SessionReport(
            game_id=game_id,
            num_episodes=len(episodes),
            avg_reward=round(avg_reward, 4),
            avg_steps=round(avg_steps, 1),
            avg_levels=round(avg_levels, 2),
            best_levels=best_levels,
            best_reward=round(best_reward, 4),
            episodes=episodes,
            duration_s=round(duration, 2),
        )

    def get_stats(self) -> dict:
        """Return session statistics."""
        return {
            "total_episodes": len(self._episode_history),
            "total_steps": sum(e.steps for e in self._episode_history),
            "total_reward": sum(e.total_reward for e in self._episode_history),
            "games_played": list(set(e.game_id for e in self._episode_history)),
            "ns_programs_loaded": list(self._ns_programs.keys()),
        }
