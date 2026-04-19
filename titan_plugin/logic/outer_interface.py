"""
titan_plugin/logic/outer_interface.py — Outer Interface: ACTION→REACTION→OBSERVATION.

Bidirectional interface between Titan's outer Trinity and the physical/digital world.
Mirrors the inner Interface (InputExtractor + OutputColoring + InterfaceAdvisor)
but faces OUTWARD toward the world.

Implements the closed loop:
    INNER URGE → ACTION (on world) → REACTION (world responds) → OBSERVATION (mapped to Trinity)

Three speeds:
    1. FAST PATH (ActionDecoder): result → numerical dims, milliseconds
    2. NARRATOR PATH (ActionNarrator): result → words → vocabulary reinforcement, milliseconds
    3. PASSTHROUGH: external interaction → self-exploration pauses

Two modes:
    - SELF_EXPLORE: autonomous exploration driven by expression composites
    - EXTERNAL_PASSTHROUGH: external interaction has priority, exploration paused

Higher cognitive actions (SPEAK, social, research, code) enrich BOTH trinities.
Simple actions (art, audio, memo) enrich outer Trinity only.

Part of the Self-Exploration Outer Interface.
"""
import logging
import time
from typing import Optional

from .action_decoder import ActionDecoder
from .action_narrator import ActionNarrator
from .self_exploration_advisor import SelfExplorationAdvisor, SCHUMANN_MIND

logger = logging.getLogger(__name__)

# GABA-governed cooldown after external interaction (no human time)
# Cooldown = SCHUMANN_BODY × cooldown_multiplier × GABA
# The multiplier 9 = 3³ = Body→Mind→Spirit cycle (Schumann-derived)
# At GABA=1.0: 3.45 × 9 × 1.0 = 31s. At GABA=0.1: 3.1s.
SCHUMANN_BODY = SCHUMANN_MIND * 3  # 3.45s
DEFAULT_COOLDOWN_MULTIPLIER = 9
DEFAULT_INNER_STRENGTH = 0.5


class OuterInterface:
    """Bidirectional interface between Titan's outer Trinity and the world.

    Orchestrates: ActionDecoder (fast path) + ActionNarrator (words) +
    SelfExplorationAdvisor (timing) + mode switching (explore vs passthrough).
    """

    def __init__(self, word_recipe_dir: str = "data",
                 inner_memory=None, dna_params: dict = None,
                 params_config: dict = None):
        # Load params from [self_exploration], [action_decoder], [action_narrator]
        se_cfg = params_config.get("self_exploration", {}) if params_config else {}
        ad_cfg = params_config.get("action_decoder", {}) if params_config else {}
        an_cfg = params_config.get("action_narrator", {}) if params_config else {}

        self.decoder = ActionDecoder(config=params_config)
        self.narrator = ActionNarrator(word_recipe_dir=word_recipe_dir, config=params_config)
        self.advisor = SelfExplorationAdvisor(dna_params=dna_params, params_config=se_cfg)

        self._cooldown_multiplier = float(se_cfg.get(
            "external_cooldown_multiplier", DEFAULT_COOLDOWN_MULTIPLIER))
        self._inner_strength = float(se_cfg.get(
            "inner_enrichment_strength", DEFAULT_INNER_STRENGTH))
        self._word_perturbation_strength = float(an_cfg.get(
            "word_perturbation_strength", 0.3))

        self._inner_memory = inner_memory  # for vocabulary lookup
        self._mode = "SELF_EXPLORE"
        self._external_active = False
        self._last_external_time = 0.0

        self._stats = {
            "total_actions_processed": 0,
            "total_narrations": 0,
            "total_words_reinforced": 0,
            "total_words_unknown": 0,
            "total_explore_ticks": 0,
            "total_actions_queued": 0,
            "mode_switches": 0,
        }

    # ── Mode Control ──────────────────────────────────────────────

    def on_external_interaction(self) -> None:
        """External interaction arriving — pause self-exploration."""
        if self._mode != "EXTERNAL_PASSTHROUGH":
            self._stats["mode_switches"] += 1
        self._external_active = True
        self._mode = "EXTERNAL_PASSTHROUGH"
        self._last_external_time = time.time()
        logger.debug("[OuterInterface] External interaction → PASSTHROUGH mode")

    def check_resume(self, gaba_level: float = 0.5) -> None:
        """Check if external cooldown has expired (GABA-governed, no human time).

        Cooldown = SCHUMANN_BODY × 9 × GABA
        At GABA=0.85: ~26.4s cooldown (calm Titan resumes slowly)
        At GABA=0.10: ~3.1s cooldown (alert Titan resumes quickly)
        """
        if not self._external_active:
            return
        if self._mode != "EXTERNAL_PASSTHROUGH":
            return

        cooldown = SCHUMANN_BODY * self._cooldown_multiplier * max(0.1, gaba_level)
        elapsed = time.time() - self._last_external_time

        if elapsed >= cooldown:
            self._external_active = False
            self._mode = "SELF_EXPLORE"
            self._stats["mode_switches"] += 1
            logger.info("[OuterInterface] Cooldown expired (%.1fs, GABA=%.2f) → SELF_EXPLORE",
                        elapsed, gaba_level)

    @property
    def mode(self) -> str:
        return self._mode

    # ── OBSERVATION Pipeline ──────────────────────────────────────

    def process_action_result(self, action_type: str, result: dict) -> dict:
        """Full ACTION→REACTION→OBSERVATION pipeline.

        Called after Agency executes a helper. Converts the result into
        outer Trinity dimension deltas + vocabulary reinforcement.

        Returns observation dict with:
            outer_body_deltas, outer_mind_deltas (fast path)
            inner_body_deltas, inner_mind_deltas (higher cognitive only)
            narration, known_words, unknown_words (narrator path)
            features (extracted from result)
            action_type
        """
        # 1. FAST PATH: decode result → sensory dims
        observation = self.decoder.decode(action_type, result)

        # 2. NARRATOR PATH: describe result in words
        narration = self.narrator.narrate(
            action_type, result, observation.get("features", {}))
        observation["narration"] = narration

        # 3. VOCABULARY: extract known/unknown words
        vocab_analysis = self.narrator.extract_vocabulary_words(
            narration, self._inner_memory)
        observation["known_words"] = vocab_analysis["known_words"]
        observation["unknown_words"] = vocab_analysis["unknown_words"]

        # 4. HIGHER COGNITIVE: enrich both trinities
        if ActionDecoder.is_higher_cognitive(action_type):
            observation["inner_deltas"] = self._compute_inner_enrichment(
                action_type, observation)

        # 5. WORD PERTURBATIONS: collect for injection
        word_perturbations = []
        for kw in vocab_analysis["known_words"]:
            perturb = self.narrator.get_word_perturbation(kw["word"])
            if perturb:
                word_perturbations.append({
                    "word": kw["word"],
                    "perturbation": perturb,
                    "confidence": kw["confidence"],
                })
        observation["word_perturbations"] = word_perturbations

        # Stats
        self._stats["total_actions_processed"] += 1
        self._stats["total_narrations"] += 1
        self._stats["total_words_reinforced"] += len(vocab_analysis["known_words"])
        self._stats["total_words_unknown"] += len(vocab_analysis["unknown_words"])

        logger.info(
            "[OuterInterface] Processed %s: %d body_deltas, %d mind_deltas, "
            "narration='%s', %d known words, %d unknown",
            action_type,
            len(observation.get("outer_body_deltas", {})),
            len(observation.get("outer_mind_deltas", {})),
            narration[:60],
            len(vocab_analysis["known_words"]),
            len(vocab_analysis["unknown_words"]),
        )

        return observation

    def _compute_inner_enrichment(self, action_type: str,
                                   observation: dict) -> dict:
        """Compute inner Trinity deltas for higher cognitive actions.

        Higher cognitive = SPEAK, social, research, code.
        These combine thinking + feeling + willing across levels,
        so enriching BOTH trinities accelerates integration.

        Inner deltas are applied at 50% strength (self-observation is
        quieter than external stimulus — like bone conduction vs speaker).
        """
        features = observation.get("features", {})
        inner_mind_deltas = {}
        inner_body_deltas = {}

        INNER_STRENGTH = self._inner_strength

        if action_type == "self_express":
            # SPEAK: hearing own words → inner feeling + thinking
            inner_mind_deltas[5] = 0.03 * INNER_STRENGTH   # inner_hearing
            inner_mind_deltas[6] = 0.02 * INNER_STRENGTH   # inner_touch (self-contact)

        elif action_type == "social_post":
            # Social: empathy activation + social cognition
            inner_mind_deltas[1] = 0.03 * INNER_STRENGTH   # social_cognition
            inner_mind_deltas[5] = 0.02 * INNER_STRENGTH   # inner_hearing (social echo)

        elif action_type == "web_search":
            # Research: knowledge → inner thinking
            inner_mind_deltas[0] = 0.04 * INNER_STRENGTH   # memory_depth
            inner_mind_deltas[4] = 0.02 * INNER_STRENGTH   # conceptual_thinking

        elif action_type in ("code_knowledge", "coding_sandbox"):
            # Self-reflection: meta-cognition
            inner_mind_deltas[0] = 0.03 * INNER_STRENGTH   # memory_depth
            inner_mind_deltas[2] = 0.03 * INNER_STRENGTH   # perceptual_thinking

        return {
            "inner_body_deltas": inner_body_deltas,
            "inner_mind_deltas": inner_mind_deltas,
        }

    # ── Self-Exploration Tick ─────────────────────────────────────

    def tick_self_exploration(self, expression_fires: list,
                              neuromodulators: dict,
                              chi: dict,
                              hormonal_system: dict = None) -> list:
        """Called every Tier 2 tick (3.45s). Returns actions to execute.

        Checks if any EXPRESSION composites fired, asks advisor if
        refractory allows, returns list of actions for Agency dispatch.

        Args:
            expression_fires: list of fired composites from ExpressionManager
            neuromodulators: dict with modulator name → {level, ...}
            chi: dict with 'circulation', 'total', etc.
            hormonal_system: dict with hormone name → {level, ...} (for CURIOSITY)
        """
        self._stats["total_explore_ticks"] += 1

        # Check mode
        if self._mode == "EXTERNAL_PASSTHROUGH":
            return []

        # Extract control signals from Titan's state
        gaba = 0.5
        if isinstance(neuromodulators, dict):
            gaba_mod = neuromodulators.get("GABA")
            if isinstance(gaba_mod, dict):
                gaba = gaba_mod.get("level", 0.5)
            elif isinstance(gaba_mod, (int, float)):
                gaba = gaba_mod

        # CURIOSITY from hormonal system (NS program hormone level)
        curiosity = 0.5
        if hormonal_system:
            curiosity_h = hormonal_system.get("CURIOSITY")
            if isinstance(curiosity_h, dict):
                curiosity = curiosity_h.get("level", 0.5)
            elif isinstance(curiosity_h, (int, float)):
                curiosity = curiosity_h

        chi_circ = 0.01  # Default very low
        if isinstance(chi, dict):
            chi_circ = chi.get("circulation", 0.01)

        # Check each expression fire against advisor refractory
        actions_to_execute = []
        for fire in expression_fires:
            action_helper = fire.get("action_helper", "")
            if not action_helper:
                continue

            if self.advisor.should_explore(action_helper, gaba, curiosity, chi_circ):
                actions_to_execute.append(fire)
                self.advisor.record_action(action_helper)
                self._stats["total_actions_queued"] += 1

        return actions_to_execute

    # ── Stats & Observability ─────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "mode": self._mode,
            "external_active": self._external_active,
            **self._stats,
            "decoder": self.decoder.get_stats(),
            "narrator": self.narrator.get_stats(),
            "advisor": self.advisor.get_stats(),
        }

    # ── Hot-Reload State ───────────────────────────────────────────

    def get_state(self) -> dict:
        """Return ALL mutable state for hot-reload persistence."""
        return {
            "_mode": self._mode,
            "_external_active": self._external_active,
            "_last_external_time": self._last_external_time,
            "_stats": dict(self._stats),
            # Sub-component state
            "advisor": {
                "_last_action_time": dict(self.advisor._last_action_time),
                "_action_outcomes": {k: list(v) for k, v in self.advisor._action_outcomes.items()},
                "_total_explorations": self.advisor._total_explorations,
                "_total_blocked": self.advisor._total_blocked,
            },
            "decoder": {
                "_total_decodes": self.decoder._total_decodes,
            },
            "narrator": {
                "_stats": dict(self.narrator._stats),
            },
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from hot-reload snapshot."""
        self._mode = state.get("_mode", self._mode)
        self._external_active = state.get("_external_active", self._external_active)
        self._last_external_time = state.get("_last_external_time", self._last_external_time)
        self._stats = state.get("_stats", self._stats)

        # Sub-component: advisor
        adv = state.get("advisor", {})
        if adv:
            self.advisor._last_action_time = adv.get("_last_action_time", self.advisor._last_action_time)
            self.advisor._action_outcomes = adv.get("_action_outcomes", self.advisor._action_outcomes)
            self.advisor._total_explorations = adv.get("_total_explorations", self.advisor._total_explorations)
            self.advisor._total_blocked = adv.get("_total_blocked", self.advisor._total_blocked)

        # Sub-component: decoder
        dec = state.get("decoder", {})
        if dec:
            self.decoder._total_decodes = dec.get("_total_decodes", self.decoder._total_decodes)

        # Sub-component: narrator
        nar = state.get("narrator", {})
        if nar:
            self.narrator._stats = nar.get("_stats", self.narrator._stats)

        logger.info("[OuterInterface] State restored: mode=%s, %d actions processed, "
                    "advisor=%d explorations, decoder=%d decodes",
                    self._mode, self._stats.get("total_actions_processed", 0),
                    self.advisor._total_explorations, self.decoder._total_decodes)
