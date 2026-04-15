"""
Language Mini-Reasoner — analyzes composition quality and vocabulary dynamics.

Primitives:
  PARSE_INTENT:       Analyze recent composition queue → dominant intent, emotional tone
  MATCH_PATTERN:      Compare compositions vs grammar patterns → novelty score
  EVALUATE_EXPRESSION: Rate quality of recent compositions → improvement direction

Runs at Mind rate (23.49 Hz computation gate). Feeds language analysis
to main reasoning via MiniReasonerRegistry.query("language").
"""
import logging
import numpy as np

from .mini_experience import MiniReasoner

logger = logging.getLogger(__name__)


class LanguageMiniReasoner(MiniReasoner):
    domain = "language"
    primitives = ["PARSE_INTENT", "MATCH_PATTERN", "EVALUATE_EXPRESSION"]
    rate_tier = "mind"
    observation_dim = 20  # Vocabulary + composition + teacher metrics

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._confidence_history = []
        self._level_history = []

    def perceive(self, context: dict) -> np.ndarray:
        """Build 20D from vocabulary stats + composition metrics + teacher state."""
        parts = []

        # Vocabulary stats (5D)
        vstats = context.get("vocabulary_stats", {})
        parts.append(min(1.0, vstats.get("total_words", 0) / 200.0))      # normalized vocab size
        parts.append(vstats.get("avg_confidence", 0.0))                     # avg word confidence
        parts.append(min(1.0, vstats.get("adjective_count", 0) / 30.0))   # adjective diversity
        parts.append(min(1.0, vstats.get("verb_count", 0) / 30.0))        # verb diversity
        parts.append(min(1.0, vstats.get("noun_count", 0) / 20.0))        # noun diversity

        # Composition queue stats (7D)
        queue = context.get("composition_queue", [])
        n_queued = len(queue)
        parts.append(min(1.0, n_queued / 10.0))                            # queue fullness
        if queue:
            avg_conf = sum(q.get("confidence", 0) for q in queue) / n_queued
            avg_level = sum(q.get("level", 0) for q in queue) / n_queued
            max_level = max(q.get("level", 0) for q in queue)
            intents = set(q.get("intent", "default") for q in queue)
        else:
            avg_conf, avg_level, max_level = 0.0, 0.0, 0
            intents = set()
        parts.append(avg_conf)                                              # avg composition confidence
        parts.append(min(1.0, avg_level / 9.0))                           # normalized avg level
        parts.append(min(1.0, max_level / 9.0))                           # normalized max level
        parts.append(min(1.0, len(intents) / 5.0))                        # intent diversity
        # Recent word diversity (last 5 compositions)
        recent_words = set()
        for q in queue[-5:]:
            recent_words.update(q.get("words_used", []))
        parts.append(min(1.0, len(recent_words) / 20.0))                  # recent word diversity
        # Level trend
        if len(self._level_history) >= 3:
            trend = float(np.mean(self._level_history[-3:])) - float(np.mean(self._level_history[-6:-3])) if len(self._level_history) >= 6 else 0.0
        else:
            trend = 0.0
        parts.append(max(-1, min(1, trend)))                               # level trend

        # Teacher state (4D)
        teacher = context.get("teacher_state", {})
        parts.append(1.0 if teacher.get("active", False) else 0.0)        # teacher active
        parts.append(min(1.0, teacher.get("sessions_total", 0) / 100.0))  # normalized session count
        parts.append(teacher.get("last_mode_confidence", 0.0))            # last mode effectiveness
        parts.append(min(1.0, teacher.get("no_response_count", 0) / 3.0)) # teacher reliability

        # Grammar pattern stats (4D)
        gp = context.get("grammar_patterns", {})
        parts.append(min(1.0, gp.get("total_patterns", 0) / 20.0))       # pattern count
        parts.append(gp.get("avg_success_rate", 0.0))                     # pattern success rate
        parts.append(min(1.0, gp.get("unique_templates", 0) / 10.0))     # template diversity
        parts.append(gp.get("l8_active", 0.0))                            # L8 firing

        arr = np.array(parts[:self.observation_dim], dtype=np.float64)
        if len(arr) < self.observation_dim:
            arr = np.concatenate([arr, np.zeros(self.observation_dim - len(arr))])

        # Track history for trend analysis
        self._confidence_history.append(avg_conf)
        self._level_history.append(avg_level)
        if len(self._confidence_history) > 30:
            self._confidence_history = self._confidence_history[-30:]
            self._level_history = self._level_history[-30:]

        return arr

    def execute_primitive(self, primitive_idx: int, observation: np.ndarray) -> dict:
        if primitive_idx == 0:
            return self._parse_intent(observation)
        elif primitive_idx == 1:
            return self._match_pattern(observation)
        else:
            return self._evaluate_expression(observation)

    def _parse_intent(self, obs: np.ndarray) -> dict:
        """Analyze dominant intent from composition metrics."""
        queue_fullness = obs[5]
        avg_conf = obs[6]
        intent_diversity = obs[9]
        vocab_size = obs[0]

        # Determine dominant intent from feature balance
        if avg_conf < 0.3:
            dominant = "learning"
            tone = "uncertain"
        elif intent_diversity > 0.6:
            dominant = "exploring"
            tone = "curious"
        elif vocab_size > 0.5 and avg_conf > 0.6:
            dominant = "expressing"
            tone = "confident"
        else:
            dominant = "practicing"
            tone = "focused"

        return {
            "relevance": max(queue_fullness, 0.2),
            "confidence": max(0.4, avg_conf),
            "dominant_intent": dominant,
            "emotional_tone": tone,
            "vocabulary_readiness": round(float(vocab_size), 3),
        }

    def _match_pattern(self, obs: np.ndarray) -> dict:
        """Compare composition novelty vs grammar patterns."""
        pattern_count = obs[16]
        pattern_success = obs[17]
        template_diversity = obs[18]
        word_diversity = obs[10]
        level_trend = obs[11]

        novelty = max(0, 1.0 - pattern_count) * 0.5 + word_diversity * 0.3 + max(0, level_trend) * 0.2
        return {
            "relevance": min(1.0, novelty),
            "confidence": max(0.3, pattern_success),
            "novelty_score": round(novelty, 3),
            "pattern_maturity": round(float(pattern_count), 3),
            "improvement_possible": level_trend > 0,
        }

    def _evaluate_expression(self, obs: np.ndarray) -> dict:
        """Rate quality trajectory of recent compositions."""
        avg_conf = obs[6]
        avg_level = obs[7]
        max_level = obs[8]
        word_diversity = obs[10]

        quality = avg_conf * 0.3 + avg_level * 0.3 + word_diversity * 0.2 + max_level * 0.2
        # Improvement direction
        if avg_conf < 0.4:
            direction = "vocabulary"
        elif avg_level < 0.5:
            direction = "complexity"
        elif word_diversity < 0.4:
            direction = "diversity"
        else:
            direction = "refinement"

        return {
            "relevance": quality,
            "confidence": max(0.4, avg_conf),
            "expression_quality": round(quality, 3),
            "improvement_direction": direction,
            "current_level": round(float(max_level * 9), 1),
        }

    def format_summary(self) -> dict:
        s = dict(self._latest_summary)
        s.setdefault("relevance", 0.0)
        s.setdefault("confidence", 0.3)
        s.setdefault("primitive", self.primitives[0])
        return s
