"""
Experience Plugins — Domain-specific extractors for the Experience Orchestrator.

Each plugin knows how to:
  1. Extract perception features from its domain context
  2. Score outcomes for its domain
  3. Summarize experiences into distilled patterns

Perception Key Design (aligned with 132D Trinity architecture):
  State vector: [0:5] Inner Body, [5:20] Inner Mind (Think5+Feel5+Will5),
  [20:65] Inner Spirit (SAT15+CHIT15+ANANDA15), [65:70] Outer Body,
  [70:85] Outer Mind (Think5+Feel5+Will5), [85:130] Outer Spirit.

  Each plugin extracts dimensions meaningful for its domain from
  the full inner_state (132D) or from pre-sliced context fields.
  The orchestrator is dimension-agnostic (stores as JSON).
"""

import json
from collections import Counter
from titan_plugin.logic.experience_orchestrator import ExperiencePlugin


def _safe_slice(lst: list, start: int, end: int, default: float = 0.5) -> list[float]:
    """Safely slice a list, padding with default if too short."""
    result = []
    for i in range(start, end):
        result.append(float(lst[i]) if i < len(lst) else default)
    return result


def _get_hormones_5d(context: dict) -> list[float]:
    """Extract top 5 hormones as a normalized 5D vector."""
    hormones = context.get("intent_hormones", context.get("hormonal_snapshot", {}))
    if isinstance(hormones, str):
        import json
        try:
            hormones = json.loads(hormones)
        except (json.JSONDecodeError, TypeError):
            hormones = {}
    if not isinstance(hormones, dict):
        return [0.5] * 5
    # Top 5 programs in consistent order for stable perception keys
    return [
        float(hormones.get("CURIOSITY", 0.5)),
        float(hormones.get("CREATIVITY", 0.5)),
        float(hormones.get("EMPATHY", 0.5)),
        float(hormones.get("INSPIRATION", 0.5)),
        float(hormones.get("REFLECTION", 0.5)),
    ]


class ArcPuzzlePlugin(ExperiencePlugin):
    """ARC-AGI-3 puzzle solving — 20D perception aligned with GridPerception (4×5D).

    Perception key (20D):
      [0:5]   Inner Body 5D       — physical grid properties
      [5:10]  Inner Mind first 5D  — pattern features (Thinking sub-layer)
      [10:15] Inner Spirit first 5D — episode state (SAT sub-layer)
      [15:20] Spatial 5D           — Outer Body (change location features)
    """

    @property
    def domain(self) -> str:
        return "arc_puzzle"

    def extract_perception_key(self, context: dict) -> list[float]:
        # Prefer full inner_state (132D) if available
        state = context.get("inner_state", [])
        if len(state) >= 70:
            return (
                _safe_slice(state, 0, 5)       # Inner Body [0:5]
                + _safe_slice(state, 5, 10)    # Inner Mind Thinking [5:10]
                + _safe_slice(state, 20, 25)   # Inner Spirit SAT first 5 [20:25]
                + _safe_slice(state, 65, 70)   # Outer Body [65:70]
            )
        # Fallback to pre-sliced context fields
        return (
            _safe_slice(context.get("inner_body", []), 0, 5)
            + _safe_slice(context.get("inner_mind", []), 0, 5)
            + _safe_slice(context.get("inner_spirit", []), 0, 5)
            + _safe_slice(context.get("spatial_features", []), 0, 5)
        )

    def compute_outcome_score(self, result: dict) -> float:
        levels = result.get("levels_completed", 0)
        total = max(1, result.get("total_levels", 7))
        reward = result.get("reward", 0.0)
        level_score = min(1.0, levels / total)
        reward_score = min(1.0, max(0.0, reward / 30.0))
        return max(level_score, reward_score)

    def summarize_for_distillation(self, experiences: list[dict]) -> dict:
        actions = Counter(e.get("action_taken", "unknown") for e in experiences)
        successes = Counter(
            e.get("action_taken", "unknown")
            for e in experiences
            if e.get("outcome_score", 0) > 0.5
        )
        best = successes.most_common(1)
        best_action = best[0][0] if best else "explore"
        avg_score = (
            sum(e.get("outcome_score", 0) for e in experiences) / max(1, len(experiences))
        )
        return {
            "pattern": f"arc:best={best_action},avg={avg_score:.2f},n={len(experiences)}",
            "action_distribution": dict(actions),
            "best_action": best_action,
        }


class LanguageLearningPlugin(ExperiencePlugin):
    """Language learning — 30D perception from Inner Mind + Outer Feeling + Hormones.

    Perception key (30D):
      [0:5]   Inner Body 5D            — physical felt-state (drives word resonance)
      [5:20]  Inner Mind 15D           — Thinking/Feeling/Willing (core of language)
      [20:25] Outer Mind Feeling 5D    — external perception state [state 75:80]
      [25:30] Hormonal context 5D      — emotional drive for composition
    """

    @property
    def domain(self) -> str:
        return "language"

    def extract_perception_key(self, context: dict) -> list[float]:
        # Prefer full inner_state (132D) if available
        state = context.get("inner_state", [])
        if len(state) >= 80:
            return (
                _safe_slice(state, 0, 5)       # Inner Body [0:5]
                + _safe_slice(state, 5, 20)    # Inner Mind full 15D [5:20]
                + _safe_slice(state, 75, 80)   # Outer Mind Feeling [75:80]
                + _get_hormones_5d(context)     # Hormonal context 5D
            )
        # Fallback: construct from available fields
        body = _safe_slice(context.get("inner_body", []), 0, 5)
        mind = context.get("inner_mind", [])
        if len(mind) < 15:
            # Try felt_tensor as fallback (older call sites)
            felt = context.get("felt_tensor", [])
            if isinstance(felt, str):
                import json
                try:
                    felt = json.loads(felt)
                except (json.JSONDecodeError, TypeError):
                    felt = []
            mind = _safe_slice(list(felt), 5, 20) if len(felt) >= 20 else _safe_slice(mind, 0, 15, 0.5)
        else:
            mind = _safe_slice(mind, 0, 15)
        outer_feel = _safe_slice(context.get("spatial_features", []), 10, 15)
        return body + mind + outer_feel + _get_hormones_5d(context)

    def compute_outcome_score(self, result: dict) -> float:
        return float(result.get("confidence", result.get("score", 0.0)))

    def summarize_for_distillation(self, experiences: list[dict]) -> dict:
        scores = [e.get("outcome_score", 0) for e in experiences]
        avg = sum(scores) / max(1, len(scores))
        actions = Counter(e.get("action_taken", "self_express") for e in experiences)

        # Separate compositions from comprehensions
        compositions = [e for e in experiences
                        if not str(e.get("action_taken", "")).startswith("comprehend:")]
        comprehensions = [e for e in experiences
                          if str(e.get("action_taken", "")).startswith("comprehend:")]
        comp_avg = (sum(e.get("outcome_score", 0) for e in compositions)
                    / max(1, len(compositions)))
        compr_avg = (sum(e.get("outcome_score", 0) for e in comprehensions)
                     / max(1, len(comprehensions)))

        # Extract word frequency from composition contexts
        word_counts = Counter()
        for e in compositions:
            ctx = e.get("context", "{}")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except Exception:
                    ctx = {}
            for w in ctx.get("words_used", []):
                word_counts[w] += 1

        # Composition pattern analysis (safe import)
        patterns_summary = ""
        try:
            from titan_plugin.logic.sentence_pattern import SentencePatternExtractor
            extractor = SentencePatternExtractor()
            patterns = extractor.extract_patterns()
            if patterns.get("total_compositions", 0) > 0:
                patterns_summary = (
                    f",fill_rate={patterns['slot_fill_rate']:.2f}"
                    f",top_level={max(patterns.get('level_distribution', {0: 0}), key=patterns.get('level_distribution', {0: 0}).get, default=0)}"
                )
        except Exception:
            pass

        return {
            "pattern": (
                f"language:avg_conf={avg:.2f},n={len(experiences)}"
                f",compose={len(compositions)},comprehend={len(comprehensions)}"
                f",comp_conf={comp_avg:.2f},compr_conf={compr_avg:.2f}"
                f"{patterns_summary}"
            ),
            "composition_count": len(compositions),
            "comprehension_count": len(comprehensions),
            "composition_confidence": comp_avg,
            "comprehension_confidence": compr_avg,
            "top_words": word_counts.most_common(5),
        }


class CreativeExpressionPlugin(ExperiencePlugin):
    """Creative expression — 30D perception from Trinity state at moment of creation.

    Captures the MOMENT of creation through the full Trinity. SpatialPerception
    features of what was CREATED flow back naturally through the SENSE_VISUAL →
    SensoryHub → Outer Trinity path in subsequent epochs.

    Perception key (30D):
      [0:5]   Inner Mind Feeling 5D   — how Titan FEELS while creating [state 10:15]
      [5:10]  Outer Body 5D           — system state during creation   [state 65:70]
      [10:15] Outer Mind Feeling 5D   — external perception state      [state 75:80]
      [15:20] Hormonal context 5D     — CURIOSITY, CREATIVITY, EMPATHY, INSPIRATION, REFLECTION
      [20:25] Inner Body 5D           — physical state                 [state 0:5]
      [25:30] Inner Spirit first 5D   — consciousness state            [state 20:25]
    """

    @property
    def domain(self) -> str:
        return "creative"

    def extract_perception_key(self, context: dict) -> list[float]:
        # Prefer full inner_state (132D) if available
        state = context.get("inner_state", [])
        if len(state) >= 80:
            return (
                _safe_slice(state, 10, 15)     # Inner Mind Feeling [10:15]
                + _safe_slice(state, 65, 70)   # Outer Body [65:70]
                + _safe_slice(state, 75, 80)   # Outer Mind Feeling [75:80]
                + _get_hormones_5d(context)     # Hormonal context 5D
                + _safe_slice(state, 0, 5)     # Inner Body [0:5]
                + _safe_slice(state, 20, 25)   # Inner Spirit SAT [20:25]
            )
        # Fallback to pre-sliced fields
        hormones = _get_hormones_5d(context)
        body = _safe_slice(context.get("inner_body", []), 0, 5)
        mind_feel = _safe_slice(context.get("inner_mind", []), 5, 10)
        spirit = _safe_slice(context.get("inner_spirit", []), 0, 5)
        outer_body = _safe_slice(context.get("spatial_features", []), 0, 5)
        outer_feel = _safe_slice(context.get("spatial_features", []), 10, 15)
        return mind_feel + outer_body + outer_feel + hormones + body + spirit

    def compute_outcome_score(self, result: dict) -> float:
        return float(
            result.get("assessment_score", result.get("score", 0.5))
        )

    def summarize_for_distillation(self, experiences: list[dict]) -> dict:
        by_type: dict[str, list[float]] = {}
        for e in experiences:
            t = e.get("action_taken", "art_generate")
            by_type.setdefault(t, []).append(e.get("outcome_score", 0))

        best_type = "art_generate"
        best_avg = 0.0
        for t, scores in by_type.items():
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_type = t

        return {
            "pattern": f"creative:best={best_type},avg={best_avg:.2f},n={len(experiences)}",
            "best_medium": best_type,
        }


class CommunicationPlugin(ExperiencePlugin):
    """Social communication — 20D perception from emotional + social state.

    Perception key (20D):
      [0:5]   Inner Body 5D           — physical state during exchange  [state 0:5]
      [5:10]  Inner Mind Feeling 5D   — emotional state                 [state 10:15]
      [10:15] Outer Mind Feeling 5D   — external perception (social)    [state 75:80]
      [15:20] Hormonal context 5D     — CURIOSITY, CREATIVITY, EMPATHY, INSPIRATION, REFLECTION
    """

    @property
    def domain(self) -> str:
        return "communication"

    def extract_perception_key(self, context: dict) -> list[float]:
        # Prefer full inner_state (132D) if available
        state = context.get("inner_state", [])
        if len(state) >= 80:
            return (
                _safe_slice(state, 0, 5)       # Inner Body [0:5]
                + _safe_slice(state, 10, 15)   # Inner Mind Feeling [10:15]
                + _safe_slice(state, 75, 80)   # Outer Mind Feeling [75:80]
                + _get_hormones_5d(context)     # Hormonal context 5D
            )
        # Fallback to pre-sliced fields
        body = _safe_slice(context.get("inner_body", []), 0, 5)
        mind_feel = _safe_slice(context.get("inner_mind", []), 5, 10)
        outer_feel = _safe_slice(context.get("spatial_features", []), 10, 15)
        return body + mind_feel + outer_feel + _get_hormones_5d(context)

    def compute_outcome_score(self, result: dict) -> float:
        return float(
            result.get("resonance", result.get("score", 0.5))
        )

    def summarize_for_distillation(self, experiences: list[dict]) -> dict:
        scores = [e.get("outcome_score", 0) for e in experiences]
        success_rate = sum(1 for s in scores if s > 0.6) / max(1, len(scores))
        return {
            "pattern": f"communication:success={success_rate:.2f},n={len(experiences)}",
            "success_rate": success_rate,
        }
