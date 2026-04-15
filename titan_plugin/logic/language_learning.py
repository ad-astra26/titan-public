"""
titan_plugin/logic/language_learning.py — Language Learning Experience Plugin.

Titan's first steps toward autonomous language expression.
Each word is taught through THREE PASSES (infant analogy):

  1. FEEL:      Strong perturbation + word → embodied association (receptive)
  2. RECOGNIZE: Word alone → Titan's Trinity shifts to recalled pattern (recognition)
  3. PRODUCE:   Current state → Titan selects the word himself (expression)

Words are selected based on Titan's current hormonal state:
  - High CURIOSITY → teach exploration words
  - High CREATIVITY → teach creative words
  - Don't fight the hormonal state — ride it

Depends on:
  - Experience Playground (P1+P2)
  - 130D Trinity (132D consciousness)
  - Inner Memory vocabulary table
  - data/word_resonance.json (40 word recipes)
"""
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

from titan_plugin.logic.experience_playground import ExperiencePlugin

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

WORD_RESONANCE_PATH = Path(__file__).parent.parent.parent / "data" / "word_resonance.json"
TOTAL_130D = 130  # 5 + 15 + 45 + 5 + 15 + 45
LAYER_SIZES = {
    "inner_body": 5, "inner_mind": 15, "inner_spirit": 45,
    "outer_body": 5, "outer_mind": 15, "outer_spirit": 45,
}
LAYER_ORDER = ["inner_body", "inner_mind", "inner_spirit",
               "outer_body", "outer_mind", "outer_spirit"]

# Hormone → word affinity mapping for state-driven selection
HORMONE_WORD_MAP = {
    "CURIOSITY": ["explore", "search", "learn", "grow", "change", "see"],
    "CREATIVITY": ["create", "express", "flow", "light", "energy"],
    "EMPATHY": ["connect", "warm", "give", "feel"],
    "REFLECTION": ["rest", "still", "slow", "dark", "balance", "remember", "observe"],
    "VIGILANCE": ["cold", "pain", "weak", "protect", "pressure"],
    "FOCUS": ["focus", "observe", "think", "pulse", "stable", "see", "hear"],
    "IMPULSE": ["fast", "strong", "energy", "pressure", "want"],
    "INSPIRATION": ["alive", "light", "create", "grow"],
    "INTUITION": ["know", "feel", "see"],
    "REFLEX": ["strong", "fast"],
}

# Learning phases and pass types
PASS_FEEL = "feel"
PASS_RECOGNIZE = "recognize"
PASS_PRODUCE = "produce"


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors, auto-padding shorter one."""
    max_len = max(len(a), len(b))
    va = a + [0.0] * (max_len - len(a))
    vb = b + [0.0] * (max_len - len(b))
    dot = sum(x * y for x, y in zip(va, vb))
    mag_a = math.sqrt(sum(x * x for x in va)) or 1e-10
    mag_b = math.sqrt(sum(x * x for x in vb)) or 1e-10
    return dot / (mag_a * mag_b)


def _flatten_perturbation(perturbation: dict) -> list:
    """Flatten 6-layer perturbation dict into 130D vector."""
    flat = []
    for layer in LAYER_ORDER:
        vals = perturbation.get(layer, [])
        expected = LAYER_SIZES[layer]
        flat.extend(vals[:expected])
        if len(vals) < expected:
            flat.extend([0.0] * (expected - len(vals)))
    return flat


class LanguageLearningExperience(ExperiencePlugin):
    """Titan learns words through felt-meaning association.

    Three-pass learning per word:
      Feel → Recognize → Produce

    Words are selected based on Titan's current hormonal state
    to teach what the body is ready to learn.
    """

    name = "language"
    description = "Learn words through 130D felt-meaning association"
    difficulty_levels = 3  # Stage 1 body, Stage 2 mind, Stage 3 (future)

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        inner_memory=None,
        hormonal_system=None,
        word_resonance_path: Optional[str] = None,
    ):
        super().__init__(llm_fn, inner_memory, hormonal_system)
        self._word_data: dict = {}
        self._current_word: Optional[str] = None
        self._current_pass: str = PASS_FEEL
        self._pass_cycle: int = 0  # Tracks which pass we're on per word
        self._words_taught: list = []
        self._session_words: list = []  # Words queued for this session
        self._load_word_resonance(word_resonance_path)

    def _load_word_resonance(self, path: Optional[str] = None) -> None:
        """Load word recipes from JSON dictionary."""
        fpath = Path(path) if path else WORD_RESONANCE_PATH
        if not fpath.exists():
            logger.warning("[LanguageLearning] Word resonance file not found: %s", fpath)
            return
        with open(fpath, "r") as f:
            data = json.load(f)
        # Skip _meta key
        self._word_data = {
            k: v for k, v in data.items() if not k.startswith("_")
        }
        logger.info("[LanguageLearning] Loaded %d word recipes", len(self._word_data))

    def _select_words_for_session(self, num_words: int = 5) -> list[str]:
        """Select words to teach based on hormonal state and learning progress.

        Priority:
        1. Words matching current dominant hormone (ride the state)
        2. Words not yet learned (unlearned > felt > recognized)
        3. Lowest confidence first within same phase
        """
        # Get current dominant hormones
        dominant_hormones = []
        if self._hormonal:
            try:
                hormone_levels = []
                for name in HORMONE_WORD_MAP:
                    h = self._hormonal.get(name)
                    if h:
                        hormone_levels.append((name, h.level))
                hormone_levels.sort(key=lambda x: x[1], reverse=True)
                dominant_hormones = [h[0] for h in hormone_levels[:3]]
            except Exception:
                pass

        # Get learning state from memory
        word_states = {}
        if self._memory:
            try:
                vocab = self._memory.get_vocabulary()
                word_states = {w["word"]: w for w in vocab}
            except Exception:
                pass

        # Score each word
        candidates = []
        stage = self._current_level
        for word, recipe in self._word_data.items():
            if recipe.get("stage", 1) > stage:
                continue  # Don't teach above current level

            state = word_states.get(word, {})
            phase = state.get("learning_phase", "unlearned")
            confidence = state.get("confidence", 0.0)

            # Phase priority: unlearned > felt > recognized > producible
            phase_score = {"unlearned": 4, "felt": 3, "recognized": 2,
                           "producible": 1}.get(phase, 0)

            # Hormone alignment score
            hormone_score = 0
            for h in dominant_hormones:
                if word in HORMONE_WORD_MAP.get(h, []):
                    hormone_score += 2

            # Antonym bonus: if we just taught a word, its antonym is next
            antonym_bonus = 0
            if self._words_taught and recipe.get("antonym") in self._words_taught[-3:]:
                antonym_bonus = 3

            total_score = phase_score * 10 + hormone_score + antonym_bonus - confidence * 5
            candidates.append((word, total_score, phase))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in candidates[:num_words]]

        if selected:
            logger.info("[LanguageLearning] Selected words for session: %s "
                        "(dominant hormones: %s)",
                        selected, dominant_hormones[:2])
        return selected

    async def generate_stimulus(self) -> dict:
        """Generate next stimulus: word + pass type.

        Cycles through: FEEL → RECOGNIZE → PRODUCE for each word,
        then moves to next word.
        """
        # Need new word?
        if self._current_word is None or self._pass_cycle >= 3:
            if self._session_words:
                self._current_word = self._session_words.pop(0)
                self._pass_cycle = 0
            else:
                # Reload session words
                self._session_words = self._select_words_for_session(5)
                if not self._session_words:
                    # Fallback: random word from current stage
                    stage_words = [w for w, r in self._word_data.items()
                                   if r.get("stage", 1) <= self._current_level]
                    self._session_words = random.sample(
                        stage_words, min(5, len(stage_words)))
                self._current_word = self._session_words.pop(0)
                self._pass_cycle = 0

        # Determine current pass
        passes = [PASS_FEEL, PASS_RECOGNIZE, PASS_PRODUCE]
        self._current_pass = passes[self._pass_cycle]

        word = self._current_word
        recipe = self._word_data.get(word, {})

        stimulus = {
            "content": word,
            "type": "word",
            "level": self._current_level,
            "pass_type": self._current_pass,
            "word_type": recipe.get("word_type", "unknown"),
            "entry_layer": recipe.get("entry_layer", "outer_body"),
            "contexts": recipe.get("contexts", []),
            "expected": {
                "word": word,
                "felt_tensor": _flatten_perturbation(recipe.get("perturbation", {})),
                "hormone_affinity": recipe.get("hormone_affinity", {}),
            },
            "metadata": {
                "pass_number": self._pass_cycle + 1,
                "stage": recipe.get("stage", 1),
                "antonym": recipe.get("antonym", ""),
            },
        }

        self._pass_cycle += 1
        return stimulus

    def compute_perturbation(self, stimulus: dict) -> dict:
        """Compute 130D perturbation based on pass type.

        FEEL:      Full perturbation (strong, from recipe)
        RECOGNIZE: Reduced perturbation (30% strength, test recall)
        PRODUCE:   No perturbation (Titan must generate from current state)
        """
        word = stimulus["content"]
        recipe = self._word_data.get(word, {})
        pass_type = stimulus.get("pass_type", PASS_FEEL)

        if pass_type == PASS_PRODUCE:
            # No perturbation — Titan must find the word from his state
            return {
                "inner_body": [0.0] * 5,
                "inner_mind": [0.0] * 15,
                "inner_spirit": [0.0] * 45,
                "outer_body": [0.0] * 5,
                "outer_mind": [0.0] * 15,
                "outer_spirit": [0.0] * 45,
                "hormone_stimuli": {},
            }

        perturbation = recipe.get("perturbation", {})
        hormone_affinity = recipe.get("hormone_affinity", {})

        # Scale factor based on pass type
        # Reduced from 1.0/0.3 to test autonomous word selection with less scaffolding
        # All 128 words are producible — Titan should maintain accuracy with weaker perturbation
        scale = 0.5 if pass_type == PASS_FEEL else 0.15  # FEEL=50%, RECOGNIZE=15%

        result = {}
        for layer in LAYER_ORDER:
            vals = perturbation.get(layer, [])
            expected = LAYER_SIZES[layer]
            scaled = [v * scale for v in vals[:expected]]
            if len(scaled) < expected:
                scaled.extend([0.0] * (expected - len(scaled)))
            result[layer] = scaled

        # Hormone stimuli also scaled
        result["hormone_stimuli"] = {
            k: v * scale for k, v in hormone_affinity.items()
        }

        return result

    async def evaluate_response(self, stimulus: dict, response: dict) -> dict:
        """Evaluate Titan's response based on pass type.

        FEEL:      Did Trinity shift in the expected direction? (lenient)
        RECOGNIZE: Does Trinity state resemble the word's pattern? (moderate)
        PRODUCE:   Can Titan select the correct word from state? (strict)
        """
        pass_type = stimulus.get("pass_type", PASS_FEEL)
        word = stimulus["content"]
        expected = stimulus.get("expected", {})
        expected_tensor = expected.get("felt_tensor", [])
        expected_hormones = expected.get("hormone_affinity", {})

        # Get Titan's current hormonal state from response
        hormonal_state = response.get("hormonal_state", {})
        fired_programs = response.get("fired_programs", [])

        score = 0.5  # Default neutral
        feedback = ""
        correction = None
        reinforcement = None

        if pass_type == PASS_FEEL:
            # FEEL: Check if any relevant hormone was stimulated
            # Lenient — the goal is just to CREATE the association
            hormone_match = 0.0
            for hormone, expected_delta in expected_hormones.items():
                state = hormonal_state.get(hormone, {})
                if isinstance(state, dict) and state.get("level", 0) > 0.1:
                    hormone_match += 0.3
            # Any program fire during feel = positive engagement
            if fired_programs:
                hormone_match += 0.2
            score = min(1.0, 0.5 + hormone_match)
            feedback = f"FEEL: '{word}' perturbation applied. Engagement: {score:.1%}"
            reinforcement = {"association": "forming", "pass": "feel"}

        elif pass_type == PASS_RECOGNIZE:
            # RECOGNIZE: Weaker perturbation — does Titan still respond?
            # Check if hormonal response aligns with expected pattern
            alignment = 0.0
            for hormone, expected_delta in expected_hormones.items():
                state = hormonal_state.get(hormone, {})
                if isinstance(state, dict) and state.get("level", 0) > 0.15:
                    alignment += 0.4
            if fired_programs:
                alignment += 0.1
            score = min(1.0, 0.3 + alignment)
            feedback = f"RECOGNIZE: '{word}' recall test. Alignment: {score:.1%}"
            if score < 0.5:
                correction = {"word": word, "needs": "more_exposure"}

        elif pass_type == PASS_PRODUCE:
            # PRODUCE: No perturbation — can Titan select this word?
            # Use cosine similarity between Titan's current state and word's tensor
            if self._memory and expected_tensor:
                try:
                    similar = self._memory.find_similar_words(
                        expected_tensor, top_k=5, min_confidence=0.0)
                    if similar:
                        # Check if the target word is in top matches
                        top_words = [w for w, s in similar]
                        if word in top_words:
                            rank = top_words.index(word)
                            score = max(0.3, 1.0 - rank * 0.2)
                            feedback = (f"PRODUCE: '{word}' found at rank {rank+1}! "
                                        f"Score: {score:.1%}")
                            reinforcement = {"word": word, "rank": rank + 1}
                        else:
                            score = 0.2
                            feedback = (f"PRODUCE: '{word}' not in top 5. "
                                        f"Top match: '{top_words[0] if top_words else 'none'}'")
                            correction = {"expected": word,
                                          "produced": top_words[0] if top_words else None}
                    else:
                        score = 0.1
                        feedback = f"PRODUCE: No vocabulary for matching yet"
                except Exception as e:
                    score = 0.3
                    feedback = f"PRODUCE: Evaluation error: {e}"
            else:
                # Without memory, score based on hormonal alignment
                score = 0.4
                feedback = f"PRODUCE: '{word}' — no memory for word matching"

        # Update vocabulary in inner memory
        if self._memory:
            try:
                phase_map = {PASS_FEEL: "felt", PASS_RECOGNIZE: "recognized",
                             PASS_PRODUCE: "producible"}
                new_phase = phase_map.get(pass_type, "felt")

                # Only advance phase if score is good enough
                if score < 0.5:
                    # Don't advance, keep current phase
                    current = self._memory.get_word(word)
                    if current:
                        new_phase = current.get("learning_phase", "unlearned")

                # Store/update felt tensor if FEEL pass
                felt_tensor = expected_tensor if pass_type == PASS_FEEL else None

                self._memory.update_word_learning(
                    word=word,
                    phase=new_phase,
                    felt_tensor=felt_tensor,
                    confidence_delta=0.1 * score if score > 0.5 else -0.02,
                    encountered=True,
                    produced=(pass_type == PASS_PRODUCE and score > 0.5),
                )

                if pass_type == PASS_FEEL and word not in self._words_taught:
                    self._words_taught.append(word)

            except Exception as e:
                logger.debug("[LanguageLearning] Memory update failed: %s", e)

        return {
            "score": score,
            "feedback": feedback,
            "correction": correction,
            "reinforcement": reinforcement,
            "pass_type": pass_type,
            "word": word,
        }

    def get_stats(self) -> dict:
        """Extended stats including language-specific metrics."""
        base = super().get_stats()
        base["words_taught"] = len(self._words_taught)
        base["current_word"] = self._current_word
        base["current_pass"] = self._current_pass
        base["word_count"] = len(self._word_data)

        if self._memory:
            try:
                base["vocab_stats"] = self._memory.get_vocab_stats()
            except Exception:
                pass

        return base
