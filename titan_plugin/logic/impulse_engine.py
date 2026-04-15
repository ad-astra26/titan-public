"""
titan_plugin/logic/impulse_engine.py — Impulse Engine (Step 7.1).

Real-time observer of the Inner Trinity state. Detects significant deficits
in Body/Mind/Spirit tensors and fires IMPULSE events when the deficit
exceeds an adaptive threshold.

The threshold self-regulates via EMA outcome tracking:
  - Success (Trinity improved after action): threshold decreases (fire more)
  - Failure (Trinity worsened or no effect): threshold increases (fire less)
  - Floor=0.1 (never suppress all impulses), Ceil=0.7 (never fire on every tick)

Runs every Spirit publish cycle (60s). Impulses are the raw "urge" that
gets enriched by Interface into an INTENT for the Agency Module.

This is how Titan decides he wants to do something — not because he was
told to, but because his inner state demands it.
"""
import logging
import time
from collections import deque
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Center of the Middle Path
CENTER = 0.5


class ImpulseEngine:
    """
    Detects Trinity deficits and fires IMPULSE events when above threshold.

    Adaptive threshold uses asymmetric EMA:
      - α=0.01: threshold decreases slowly on success (encourage more action)
      - β=0.015: threshold increases faster on failure (suppress bad impulses)
    """

    def __init__(
        self,
        threshold: float = 0.3,
        alpha: float = 0.01,
        beta: float = 0.015,
        floor: float = 0.1,
        ceil: float = 0.7,
        cooldown_seconds: float = 300.0,
    ):
        self._threshold = threshold
        self._alpha = alpha
        self._beta = beta
        self._floor = floor
        self._ceil = ceil
        self._cooldown = cooldown_seconds

        # State
        self._impulse_counter = 0
        self._last_impulse_ts = 0.0
        self._last_impulse_posture: Optional[str] = None
        self._outcome_history: deque = deque(maxlen=200)

        # Pending impulse (waiting for outcome)
        self._pending: Optional[dict] = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def impulse_count(self) -> int:
        return self._impulse_counter

    def observe(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
        intuition_suggestion: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Observe current Trinity state and maybe fire an IMPULSE.

        Returns IMPULSE payload dict or None if below threshold / in cooldown.
        """
        now = time.time()

        # Cooldown check
        if now - self._last_impulse_ts < self._cooldown:
            return None

        # Compute per-dimension deficits
        body_deficits = [(i, abs(v - CENTER)) for i, v in enumerate(body)]
        mind_deficits = [(i, abs(v - CENTER)) for i, v in enumerate(mind)]

        # Also check spirit (dims 0-2: WHO/WHY/WHAT; 3-4 are scalar summaries)
        spirit_deficits = [(i, abs(v - CENTER)) for i, v in enumerate(spirit[:3])]

        # Combine all deficits with layer labels
        all_deficits = (
            [("body", i, d) for i, d in body_deficits]
            + [("mind", i, d) for i, d in mind_deficits]
            + [("spirit", i, d) for i, d in spirit_deficits]
        )
        all_deficits.sort(key=lambda x: x[2], reverse=True)

        if not all_deficits:
            return None

        # Find the strongest deficit
        source_layer, source_dim, max_deficit = all_deficits[0]

        # Check threshold
        if max_deficit < self._threshold:
            return None

        # Urgency: how far above threshold (0..1 normalized)
        urgency = min(1.0, (max_deficit - self._threshold) / (1.0 - self._threshold))

        # Determine posture from deficit (reuse Intuition logic)
        posture = self._deficit_to_posture(source_layer, source_dim, body, mind)

        # Don't repeat the same posture too quickly (even after cooldown)
        if posture == self._last_impulse_posture and now - self._last_impulse_ts < self._cooldown * 2:
            # Check if there's a secondary deficit with a different posture
            for layer, dim, deficit in all_deficits[1:]:
                if deficit < self._threshold:
                    break
                alt_posture = self._deficit_to_posture(layer, dim, body, mind)
                if alt_posture != posture:
                    posture = alt_posture
                    source_layer, source_dim, max_deficit = layer, dim, deficit
                    break
            else:
                return None  # No alternative found, wait

        # Collect source dims above threshold
        source_dims = [
            (layer, dim, deficit) for layer, dim, deficit in all_deficits
            if deficit >= self._threshold
        ]

        # Build IMPULSE payload
        self._impulse_counter += 1
        impulse_id = self._impulse_counter

        # Include intuition confidence if available
        intuition_confidence = 0.0
        if intuition_suggestion and intuition_suggestion.get("posture_name") == posture:
            intuition_confidence = intuition_suggestion.get("confidence", 0.0)

        payload = {
            "impulse_id": impulse_id,
            "posture": posture,
            "source_layer": source_layer,
            "source_dims": [(l, d) for l, d, _ in source_dims[:5]],
            "deficit_values": [round(def_v, 4) for _, _, def_v in source_dims[:5]],
            "intuition_confidence": round(intuition_confidence, 4),
            "urgency": round(urgency, 4),
            "threshold": round(self._threshold, 4),
            "trinity_snapshot": {
                "body": [round(v, 4) for v in body],
                "mind": [round(v, 4) for v in mind],
                "spirit": [round(v, 4) for v in spirit],
            },
            "ts": now,
        }

        # Update state
        self._last_impulse_ts = now
        self._last_impulse_posture = posture
        self._pending = payload

        logger.info(
            "[ImpulseEngine] IMPULSE #%d: posture=%s urgency=%.2f "
            "source=%s[%d] deficit=%.3f threshold=%.3f",
            impulse_id, posture, urgency, source_layer, source_dim,
            max_deficit, self._threshold,
        )

        return payload

    def record_outcome(
        self,
        impulse_id: int,
        trinity_before: dict,
        trinity_after: dict,
    ) -> dict:
        """
        Record the outcome of an impulse action and adjust threshold.

        Args:
            impulse_id: The impulse that triggered the action
            trinity_before: {"body": [...], "mind": [...], "spirit": [...]}
            trinity_after: {"body": [...], "mind": [...], "spirit": [...]}

        Returns:
            Outcome dict with delta and threshold adjustment.
        """
        from titan_plugin.logic.middle_path import middle_path_loss

        loss_before = middle_path_loss(
            trinity_before["body"], trinity_before["mind"], trinity_before["spirit"],
        )
        loss_after = middle_path_loss(
            trinity_after["body"], trinity_after["mind"], trinity_after["spirit"],
        )

        delta = loss_after - loss_before  # Negative = improved
        success = delta < 0

        # Adjust threshold
        old_threshold = self._threshold
        if success:
            self._threshold = max(self._floor, self._threshold - self._alpha)
        else:
            self._threshold = min(self._ceil, self._threshold + self._beta)

        outcome = {
            "impulse_id": impulse_id,
            "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4),
            "delta": round(delta, 4),
            "success": success,
            "threshold_before": round(old_threshold, 4),
            "threshold_after": round(self._threshold, 4),
            "ts": time.time(),
        }

        self._outcome_history.append(outcome)
        self._pending = None

        logger.info(
            "[ImpulseEngine] Outcome #%d: %s delta=%.4f threshold=%.3f→%.3f",
            impulse_id, "SUCCESS" if success else "FAILURE",
            delta, old_threshold, self._threshold,
        )

        return outcome

    def get_stats(self) -> dict:
        """Return engine statistics."""
        successes = sum(1 for o in self._outcome_history if o["success"])
        total = len(self._outcome_history)
        return {
            "threshold": round(self._threshold, 4),
            "impulse_count": self._impulse_counter,
            "cooldown_seconds": self._cooldown,
            "outcome_count": total,
            "success_rate": round(successes / total, 3) if total > 0 else 0.0,
            "pending_impulse": self._pending is not None,
            "last_impulse_ts": self._last_impulse_ts,
        }

    @staticmethod
    def _deficit_to_posture(
        layer: str, dim: int,
        body: Sequence[float], mind: Sequence[float],
    ) -> str:
        """Map a deficit location to a behavioral posture."""
        if layer == "body":
            if dim in (0, 2):  # Interoception/somatosensation
                return "rest"
            if dim == 3:  # Entropy
                return "meditate"
            return "rest"  # Proprioception/thermal
        elif layer == "mind":
            if dim == 0:  # Vision (knowledge staleness)
                return "research"
            if dim in (1, 2):  # Hearing/taste (social)
                return "socialize"
            if dim == 4:  # Touch (mood)
                return "create"
            if dim == 3:  # Smell (environment)
                return "meditate"
        elif layer == "spirit":
            if dim == 0:  # WHO (identity)
                return "meditate"
            if dim == 1:  # WHY (drift)
                return "meditate"
            if dim == 2:  # WHAT (trajectory)
                return "research"

        return "meditate"
