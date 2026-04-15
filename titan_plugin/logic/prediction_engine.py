"""
titan_plugin/logic/prediction_engine.py — Predictive Processing Engine.

The brain CONSTANTLY predicts. Every perception is compared against
prediction. The difference (prediction error) drives learning and attention.

A system that EXPECTS and is SURPRISED is fundamentally different
from one that merely reacts.

Titan predicts his next consciousness state from current state + trajectory.
When the actual state arrives, surprise = how wrong the prediction was.
High surprise → CURIOSITY (novelty). Low surprise → confidence (familiarity).
"""
import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def _cosine_distance(a: list, b: list) -> float:
    """Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 1.0
    similarity = dot / (mag_a * mag_b)
    return max(0.0, 1.0 - similarity)


class PredictionEngine:
    """Predicts next consciousness state and measures surprise.

    After each epoch:
      1. predict_next(state, trajectory) → stores prediction
      2. Next epoch: compute_error(actual_state) → surprise score
      3. Surprise feeds CURIOSITY (novelty) and VIGILANCE (threat detection)
      4. Familiarity feeds confidence and stability signals
    """

    def __init__(self, error_window: int = 20):
        """
        Args:
            error_window: Number of recent errors to keep for rolling average.
        """
        self._last_prediction: Optional[list] = None
        self._errors: deque = deque(maxlen=error_window)
        self._total_predictions: int = 0
        self._total_surprises: int = 0  # errors > 0.3

    def predict_next(self, current_state: list, trajectory: list,
                     dt: float = 1.0) -> list:
        """Predict next state: predicted = current + trajectory × dt.

        Args:
            current_state: Current 130D (or any-D) state vector.
            trajectory: Trajectory vector (same dimensions).
            dt: Time scaling factor (default 1.0 = one epoch step).

        Returns:
            Predicted next state vector.
        """
        if not current_state:
            self._last_prediction = None
            return []

        # Ensure trajectory matches dimensions
        traj = trajectory if trajectory and len(trajectory) == len(current_state) else [0.0] * len(current_state)

        predicted = [s + t * dt for s, t in zip(current_state, traj)]
        self._last_prediction = predicted
        self._total_predictions += 1
        return predicted

    def compute_error(self, actual_state: list) -> float:
        """Compute prediction error (surprise) against last prediction.

        Args:
            actual_state: The actual state that occurred.

        Returns:
            Surprise score: 0.0 = perfectly predicted, 1.0 = completely unexpected.
        """
        if self._last_prediction is None or not actual_state:
            return 0.0

        error = _cosine_distance(self._last_prediction, actual_state)
        self._errors.append(error)

        if error > 0.3:
            self._total_surprises += 1

        return error

    def get_novelty_signal(self) -> float:
        """Rolling average of recent prediction errors.

        High novelty = unfamiliar territory → feeds CURIOSITY.
        """
        if not self._errors:
            return 0.0
        return sum(self._errors) / len(self._errors)

    def get_familiarity(self) -> float:
        """Inverse of novelty — how predictable is the environment.

        High familiarity = safe to act → feeds confidence.
        """
        return max(0.0, 1.0 - self.get_novelty_signal())

    def get_stats(self) -> dict:
        """Prediction engine statistics for API."""
        return {
            "total_predictions": self._total_predictions,
            "total_surprises": self._total_surprises,
            "novelty": round(self.get_novelty_signal(), 4),
            "familiarity": round(self.get_familiarity(), 4),
            "recent_errors": [round(e, 4) for e in list(self._errors)[-5:]],
        }
