"""
titan_plugin/logic/intuition.py — INTUITION: RL-based suggestions for the Trinity.

Uses the FILTER_DOWN value network to suggest attention allocation.
Given the current Trinity state, evaluates which senses to focus on
and suggests behavioral postures.

Suggestions are never overrides — sovereignty is always preserved.
Asymmetric penalty: bad INTUITION updates the model's confidence,
but ignoring good INTUITION costs trust (tracked, not enforced).

Postures:
  0 = rest      — reduce activity, conserve energy (body stressed)
  1 = research  — seek new information (vision fading)
  2 = socialize — engage more (hearing/taste low)
  3 = create    — express, generate art/audio (mood needs lift)
  4 = meditate  — reduce load, seek equilibrium (overall drift high)
"""
import logging
import time
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Posture definitions
POSTURES = {
    0: {"name": "rest", "desc": "Conserve energy, reduce activity"},
    1: {"name": "research", "desc": "Seek new information, refresh knowledge"},
    2: {"name": "socialize", "desc": "Engage with others, strengthen connections"},
    3: {"name": "create", "desc": "Express through art or sound"},
    4: {"name": "meditate", "desc": "Seek equilibrium, observe inner state"},
}

CENTER = 0.5

# Trust tracking
TRUST_DECAY = 0.99       # Trust decays slowly
TRUST_BOOST = 0.05       # Good outcome boosts trust
TRUST_PENALTY = 0.03     # Bad outcome after ignoring suggestion penalizes


class IntuitionEngine:
    """
    Suggests behavioral postures based on Trinity state analysis.

    Uses a simple heuristic mapping from tensor deficits to postures,
    combined with value network confidence when FILTER_DOWN is trained.
    """

    def __init__(self):
        self._trust = 0.5  # Start neutral
        self._last_suggestion: Optional[dict] = None
        self._last_suggestion_ts = 0.0
        self._suggestion_count = 0
        self._outcome_history: list[dict] = []

    def suggest(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
        filter_down=None,
    ) -> Optional[dict]:
        """
        Analyze Trinity state and suggest a posture.

        Returns dict with posture info, or None if no strong suggestion.
        """
        # Compute per-dimension deficits (how far from center)
        body_deficit = [abs(v - CENTER) for v in body]
        mind_deficit = [abs(v - CENTER) for v in mind]

        # Find the dimension with the largest deficit
        all_deficits = []
        for i, d in enumerate(body_deficit):
            all_deficits.append(("body", i, d))
        for i, d in enumerate(mind_deficit):
            all_deficits.append(("mind", i, d))

        all_deficits.sort(key=lambda x: x[2], reverse=True)

        # Only suggest if the worst deficit is significant
        if not all_deficits or all_deficits[0][2] < 0.15:
            return None

        worst_layer, worst_dim, worst_deficit = all_deficits[0]

        # Map deficit to posture
        posture_id = self._deficit_to_posture(worst_layer, worst_dim, body, mind)

        # Confidence: based on deficit magnitude and trust
        confidence = min(1.0, worst_deficit * 2) * self._trust

        # Don't repeat the same suggestion too quickly
        if (self._last_suggestion and
            self._last_suggestion.get("posture_id") == posture_id and
            time.time() - self._last_suggestion_ts < 120):
            return None

        suggestion = {
            "posture_id": posture_id,
            "posture_name": POSTURES[posture_id]["name"],
            "posture_desc": POSTURES[posture_id]["desc"],
            "confidence": round(confidence, 3),
            "reason": f"{worst_layer}[{worst_dim}] deficit={worst_deficit:.3f}",
            "trust": round(self._trust, 3),
            "ts": time.time(),
        }

        self._last_suggestion = suggestion
        self._last_suggestion_ts = time.time()
        self._suggestion_count += 1

        logger.info("[Intuition] Suggest: %s (confidence=%.2f, reason=%s)",
                     suggestion["posture_name"], confidence, suggestion["reason"])
        return suggestion

    def record_outcome(self, followed: bool, loss_before: float, loss_after: float) -> None:
        """
        Record whether the suggestion was followed and the outcome.

        If followed and loss improved → boost trust
        If ignored and loss worsened → penalize trust
        """
        improved = loss_after < loss_before

        if followed and improved:
            self._trust = min(1.0, self._trust + TRUST_BOOST)
        elif not followed and not improved:
            self._trust = max(0.0, self._trust - TRUST_PENALTY)

        # Decay trust slightly over time
        self._trust *= TRUST_DECAY

        self._outcome_history.append({
            "ts": time.time(),
            "followed": followed,
            "improved": improved,
            "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4),
            "trust_after": round(self._trust, 3),
        })

        # Keep history bounded
        if len(self._outcome_history) > 200:
            self._outcome_history = self._outcome_history[-100:]

    def get_stats(self) -> dict:
        return {
            "trust": round(self._trust, 3),
            "suggestion_count": self._suggestion_count,
            "last_suggestion": self._last_suggestion,
            "outcome_count": len(self._outcome_history),
        }

    @staticmethod
    def _deficit_to_posture(layer: str, dim: int, body: Sequence[float], mind: Sequence[float]) -> int:
        """Map a deficit to the most appropriate posture."""
        if layer == "body":
            # Body[0]=interoception(SOL), [1]=proprioception(net), [2]=somatosensation(resources),
            # [3]=entropy(errors), [4]=thermal(load)
            if dim in (0, 2):  # Energy or resources depleted
                return 0  # rest
            if dim == 3:  # High entropy (errors)
                return 4  # meditate (reduce chaos)
            if dim in (1, 4):  # Network or load issues
                return 0  # rest
        else:  # mind
            # Mind[0]=vision, [1]=hearing, [2]=taste(social), [3]=smell(env), [4]=touch(mood)
            if dim == 0:  # Vision fading (stale knowledge)
                return 1  # research
            if dim in (1, 2):  # Hearing/taste (social quality)
                return 2  # socialize
            if dim == 4:  # Touch (mood) low
                return 3  # create
            if dim == 3:  # Smell (environmental)
                return 4  # meditate

        # Default: meditate (general equilibrium seeking)
        return 4
