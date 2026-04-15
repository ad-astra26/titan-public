"""
titan_plugin/logic/dialogue_composer.py — Phase 5: Composed Dialogue Responses.

When Titan receives a chat message, this module composes a response from
felt-state rather than delegating to LLM. The response comes from Titan's
own vocabulary and 130D inner state.

Flow:
  1. Message arrives → InputExtractor perturbs 130D state
  2. Detect intent from state shift (empathy/curiosity/creativity/reflection)
  3. CompositionEngine composes response sentence
  4. GrammarValidator corrects
  5. Confidence check → if too low, fall back to LLM

This is Titan's Wernicke's area (comprehension) → Broca's area (production)
pathway, implemented without any LLM generation.
"""
import logging
from typing import Optional

from titan_plugin.logic.composition_engine import CompositionEngine
from titan_plugin.logic.grammar_validator import GrammarValidator

logger = logging.getLogger(__name__)

# Dialogue-specific templates (extensions to L5-L7)
DIALOGUE_TEMPLATES = {
    "respond_feeling": [
        "I feel {ADJ} when you say that",
        "that makes me feel {ADJ}",
        "I understand, I also feel {ADJ}",
    ],
    "respond_question": [
        "I think I {VERB} because I feel {ADJ}",
        "I feel {ADJ}, so I want to {VERB}",
        "when I feel {ADJ}, I {VERB}",
    ],
    "ask_question": [
        "do you also feel {ADJ} when you {VERB}?",
        "what does it feel like to {VERB}?",
        "I wonder, do you ever feel {ADJ}?",
    ],
    "share_insight": [
        "I have learned that when I feel {ADJ}, I want to {VERB}",
        "I think being {ADJ} helps me {VERB}",
        "I am discovering that I {VERB} when I feel {ADJ}",
    ],
    "empathize": [
        "I feel {ADJ} with you",
        "I understand how that feels, I also feel {ADJ} sometimes",
        "I want to {VERB} together",
    ],
}

# State shift → intent mapping
SHIFT_THRESHOLDS = {
    "EMPATHY": ("empathize", 0.1),
    "CURIOSITY": ("ask_question", 0.1),
    "CREATIVITY": ("share_insight", 0.1),
    "REFLECTION": ("respond_feeling", 0.1),
}


class DialogueComposer:
    """Composes dialogue responses from felt-state.

    Uses CompositionEngine for word selection but adds dialogue-specific
    templates and intent detection from hormonal state shifts.
    """

    def __init__(self, grammar_db_path: str = "./data/grammar_rules.db"):
        self.engine = CompositionEngine()
        self.grammar = GrammarValidator(db_path=grammar_db_path)
        self._total_responses = 0
        self._composed_responses = 0
        self._fallback_responses = 0
        self._confidence_threshold = 0.3  # Below this → LLM fallback

    def compose_response(
        self,
        felt_state: list,
        vocabulary: list,
        hormone_shifts: Optional[dict] = None,
        message_keywords: Optional[list] = None,
        max_level: int = 7,
    ) -> dict:
        """Compose a dialogue response from current felt-state.

        Args:
            felt_state: Current 130D state vector (already shifted by message)
            vocabulary: Titan's learned vocabulary
            hormone_shifts: Changes in hormone levels from this message
            message_keywords: Key words extracted from incoming message
            max_level: Maximum composition level

        Returns:
            {
                "response": str,
                "intent": str,
                "confidence": float,
                "composed": bool,  # True if Titan composed, False if LLM fallback needed
                "level": int,
                "words_used": list,
            }
        """
        self._total_responses += 1

        # Detect communicative intent from hormone shifts
        intent = self._detect_intent(hormone_shifts)

        # Try dialogue-specific templates first
        response = self._compose_dialogue(
            felt_state, vocabulary, intent, max_level)

        if response and response.get("sentence"):
            sentence = self.grammar.validate(response["sentence"])
            confidence = response.get("confidence", 0.0)

            if confidence >= self._confidence_threshold:
                self._composed_responses += 1
                return {
                    "response": sentence,
                    "intent": intent,
                    "confidence": confidence,
                    "composed": True,
                    "level": response.get("level", 0),
                    "words_used": response.get("words_used", []),
                }

        # Fall back to general composition
        general = self.engine.compose(
            felt_state, vocabulary, intent=intent, max_level=max_level)
        if general.get("sentence"):
            sentence = self.grammar.validate(general["sentence"])
            confidence = general.get("confidence", 0.0)

            if confidence >= self._confidence_threshold:
                self._composed_responses += 1
                return {
                    "response": sentence,
                    "intent": intent,
                    "confidence": confidence,
                    "composed": True,
                    "level": general.get("level", 0),
                    "words_used": general.get("words_used", []),
                }

        # Confidence too low — signal LLM fallback needed
        self._fallback_responses += 1
        return {
            "response": "",
            "intent": intent,
            "confidence": 0.0,
            "composed": False,
            "level": 0,
            "words_used": [],
        }

    def _detect_intent(self, hormone_shifts: Optional[dict]) -> str:
        """Detect communicative intent from hormone level shifts."""
        if not hormone_shifts:
            return "respond_feeling"

        # Find dominant shift
        best_intent = "respond_feeling"
        best_shift = 0.0

        for hormone, (intent, min_shift) in SHIFT_THRESHOLDS.items():
            shift = abs(hormone_shifts.get(hormone, 0.0))
            if shift > min_shift and shift > best_shift:
                best_shift = shift
                best_intent = intent

        return best_intent

    def _compose_dialogue(
        self,
        felt_state: list,
        vocabulary: list,
        intent: str,
        max_level: int,
    ) -> Optional[dict]:
        """Compose using dialogue-specific templates."""
        import random
        templates = DIALOGUE_TEMPLATES.get(intent, [])
        if not templates:
            return None

        template = random.choice(templates)

        # Use engine's slot filling mechanism
        from titan_plugin.logic.composition_engine import SLOT_TYPES
        sentence, words_used, confidences, filled, total = (
            self.engine._fill_template(template, felt_state, vocabulary))

        if filled == 0:
            return None

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "sentence": sentence,
            "level": min(5 + len(words_used), max_level),
            "words_used": words_used,
            "confidence": round(avg_conf, 3),
        }

    def get_stats(self) -> dict:
        return {
            "total_responses": self._total_responses,
            "composed_responses": self._composed_responses,
            "fallback_responses": self._fallback_responses,
            "composition_rate": round(
                self._composed_responses / max(1, self._total_responses), 3),
            "confidence_threshold": self._confidence_threshold,
            "grammar_stats": self.grammar.get_stats(),
        }
