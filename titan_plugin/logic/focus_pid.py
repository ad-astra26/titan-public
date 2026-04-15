"""
titan_plugin/logic/focus_pid.py — FOCUS: PID drift controller for the Trinity.

Monitors each layer's distance from the Middle Path (Divine Center).
Gentle nudges escalate if tensor values head off-cliff.

PID per dimension:
  P — proportional to current drift (how far from center)
  I — integral of accumulated drift (chronic imbalance)
  D — derivative of drift change (acute deterioration)

Output: per-dimension nudge values that Body/Mind can use to adjust sensitivity.
Nudges are suggestions — sovereignty is always preserved.
"""
import logging
import time
from typing import Sequence

logger = logging.getLogger(__name__)

CENTER = 0.5

# PID gains (tuned for 5DT tensors in 0-1 range)
KP = 1.0   # Proportional: direct response to drift
KI = 0.05  # Integral: slow correction for chronic drift
KD = 0.3   # Derivative: fast response to sudden changes

# Integral wind-up limits
INTEGRAL_MAX = 5.0
INTEGRAL_DECAY = 0.98  # Slight decay to prevent infinite accumulation

# Nudge magnitude limits
NUDGE_MAX = 0.5  # Maximum nudge magnitude per dimension
NUDGE_THRESHOLD = 0.05  # Only publish nudges above this magnitude


class FocusPID:
    """
    PID controller for one Trinity layer (Body or Mind).

    Tracks per-dimension error and produces nudge vectors.
    """

    def __init__(self, name: str, dims: int = 5):
        self.name = name
        self.dims = dims

        # PID state per dimension
        self._integral = [0.0] * dims
        self._prev_error = [0.0] * dims
        self._prev_ts = 0.0

    def update(self, tensor: Sequence[float]) -> list[float]:
        """
        Compute PID nudge vector for this layer.

        Args:
            tensor: current 5DT tensor values

        Returns:
            nudges: [5 floats] — positive = push up, negative = push down.
                    Magnitude indicates urgency. Zero = no action needed.
        """
        now = time.time()
        dt = now - self._prev_ts if self._prev_ts > 0 else 1.0
        dt = max(0.1, min(dt, 120.0))  # Clamp to reasonable range
        self._prev_ts = now

        nudges = []
        for i in range(self.dims):
            # Error: signed distance from center (positive = above center)
            error = tensor[i] - CENTER

            # P term
            p = KP * error

            # I term (with wind-up limit and decay)
            self._integral[i] = self._integral[i] * INTEGRAL_DECAY + error * dt
            self._integral[i] = max(-INTEGRAL_MAX, min(INTEGRAL_MAX, self._integral[i]))
            integral = KI * self._integral[i]

            # D term
            d_error = (error - self._prev_error[i]) / dt
            derivative = KD * d_error
            self._prev_error[i] = error

            # Combined PID output (negative: push toward center)
            raw_nudge = -(p + integral + derivative)

            # Clamp
            nudge = max(-NUDGE_MAX, min(NUDGE_MAX, raw_nudge))
            nudges.append(round(nudge, 4))

        return nudges

    def should_publish(self, nudges: list[float]) -> bool:
        """Only publish if at least one nudge exceeds threshold."""
        return any(abs(n) > NUDGE_THRESHOLD for n in nudges)

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "integral": [round(v, 4) for v in self._integral],
            "prev_error": [round(v, 4) for v in self._prev_error],
        }
