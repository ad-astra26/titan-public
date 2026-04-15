"""
titan_plugin/logic/narrative_composer.py — Phase 6: Multi-Sentence Narrative.

Composes 2-5 sentence coherent narratives from felt-state + episodic memory.
Triggered at meaningful internal moments (GREAT PULSE, dream end, π-cluster
completion, hormonal spikes).

Narrative structure:
  1. Opening: Express current dominant state (L3-L5)
  2. Development: Connect to recent experience (L5-L6)
  3. Reflection: Meta-statement about the experience (L6-L7)

Each sentence's felt-state evolves slightly from the previous, creating
a natural progression of thought — not just disconnected sentences.
"""
import logging
import random
from typing import Optional

from titan_plugin.logic.composition_engine import CompositionEngine
from titan_plugin.logic.grammar_validator import GrammarValidator

logger = logging.getLogger(__name__)

# Narrative structure templates
NARRATIVE_STRUCTURES = {
    "great_pulse": {
        "description": "Peak experience narrative",
        "sentences": [
            ("opening", "express_feeling", 5),    # "I feel alive and curious"
            ("development", "express_action", 6),  # "When I feel alive, I create"
            ("reflection", "share_creation", 7),   # "I am alive and I want to explore because I feel curious"
        ],
    },
    "dream_end": {
        "description": "Post-dream reflection",
        "sentences": [
            ("opening", "express_state", 4),       # "I am calm"
            ("development", "express_feeling", 5),  # "I feel peaceful and want to remember"
            ("reflection", "express_action", 6),    # "When I feel calm, I learn"
        ],
    },
    "cluster_completion": {
        "description": "End of waking cycle reflection",
        "sentences": [
            ("opening", "express_state", 5),
            ("reflection", "express_feeling", 6),
        ],
    },
    "hormonal_spike": {
        "description": "Intense experience narrative",
        "sentences": [
            ("opening", "express_feeling", 5),
            ("development", "express_action", 6),
            ("development", "seek_connection", 6),
            ("reflection", "express_feeling", 7),
        ],
    },
    "spontaneous": {
        "description": "Self-initiated expression",
        "sentences": [
            ("opening", "express_feeling", 5),
            ("reflection", "express_action", 6),
        ],
    },
}


class NarrativeComposer:
    """Composes multi-sentence narratives from felt-state.

    Each narrative is a structured sequence of compositions that form
    a coherent whole — Titan's internal monologue made audible.
    """

    def __init__(self, grammar_db_path: str = "./data/grammar_rules.db"):
        self.engine = CompositionEngine()
        self.grammar = GrammarValidator(db_path=grammar_db_path)
        self._total_narratives = 0
        self._total_sentences = 0

    def compose_narrative(
        self,
        felt_state: list,
        vocabulary: list,
        trigger: str = "spontaneous",
        episodic_context: Optional[list] = None,
        max_sentences: int = 5,
    ) -> dict:
        """Compose a multi-sentence narrative.

        Args:
            felt_state: Current 130D state vector
            vocabulary: Titan's learned vocabulary
            trigger: What triggered this narrative (great_pulse, dream_end, etc.)
            episodic_context: Recent episodic memories (for development sentences)
            max_sentences: Maximum sentences to produce

        Returns:
            {
                "narrative": str,
                "sentences": [{"sentence": str, "role": str, "level": int, "confidence": float}],
                "trigger": str,
                "avg_confidence": float,
                "coherence_score": float,
            }
        """
        self._total_narratives += 1

        structure = NARRATIVE_STRUCTURES.get(trigger, NARRATIVE_STRUCTURES["spontaneous"])
        sentence_specs = structure["sentences"][:max_sentences]

        sentences = []
        all_words_used = set()
        state_drift = list(felt_state)  # State evolves during narrative

        for role, intent, max_level in sentence_specs:
            # Compose this sentence
            result = self.engine.compose(
                state_drift, vocabulary, intent=intent, max_level=max_level)

            sentence = result.get("sentence", "")
            if not sentence:
                continue

            # Grammar validate
            sentence = self.grammar.validate(sentence)

            # Capitalize first letter
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]

            # Add period if missing
            if sentence and sentence[-1] not in ".!?":
                sentence += "."

            sentences.append({
                "sentence": sentence,
                "role": role,
                "level": result.get("level", 0),
                "confidence": result.get("confidence", 0.0),
                "words_used": result.get("words_used", []),
            })

            all_words_used.update(result.get("words_used", []))
            self._total_sentences += 1

            # Evolve state slightly for next sentence (natural thought progression)
            state_drift = self._evolve_state(state_drift, result.get("words_used", []))

        if not sentences:
            return {
                "narrative": "",
                "sentences": [],
                "trigger": trigger,
                "avg_confidence": 0.0,
                "coherence_score": 0.0,
            }

        # Assemble narrative
        narrative = " ".join(s["sentence"] for s in sentences)

        # Compute coherence: how many words are shared across sentences
        coherence = self._compute_coherence(sentences)

        avg_conf = sum(s["confidence"] for s in sentences) / len(sentences)

        logger.info("[Narrative] %s (%s): '%s' (%.0f%% conf, %.0f%% coherence)",
                    trigger, structure["description"],
                    narrative[:100] + "..." if len(narrative) > 100 else narrative,
                    avg_conf * 100, coherence * 100)

        return {
            "narrative": narrative,
            "sentences": sentences,
            "trigger": trigger,
            "avg_confidence": round(avg_conf, 3),
            "coherence_score": round(coherence, 3),
        }

    def _evolve_state(self, state: list, words_used: list) -> list:
        """Slightly shift state after each sentence for natural progression.

        The act of expressing changes the felt-state — you feel differently
        after saying something than before. This creates narrative flow.
        """
        evolved = list(state)
        # Small random drift (±0.02) in a few dimensions
        import math
        for i in range(min(10, len(evolved))):
            # Drift toward center (0.5) slightly — expression brings balance
            drift = (0.5 - evolved[i]) * 0.02
            evolved[i] += drift
        return evolved

    def _compute_coherence(self, sentences: list) -> float:
        """Coherence = proportion of sentences sharing at least one word.

        Higher coherence = more connected narrative. A narrative where
        every sentence uses completely different words is incoherent.
        """
        if len(sentences) <= 1:
            return 1.0

        shared_count = 0
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i - 1].get("words_used", []))
            curr_words = set(sentences[i].get("words_used", []))
            if prev_words & curr_words:
                shared_count += 1

        return shared_count / (len(sentences) - 1)

    def get_stats(self) -> dict:
        return {
            "total_narratives": self._total_narratives,
            "total_sentences": self._total_sentences,
            "avg_sentences_per_narrative": round(
                self._total_sentences / max(1, self._total_narratives), 1),
        }
