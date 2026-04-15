"""
titan_plugin/logic/ground_up.py — GROUND_UP Enrichment Mechanism.

Symmetric counterpart to FILTER_DOWN:
  - FILTER_DOWN: Spirit observes Mind/Body → nudges toward Divine Center from ABOVE
  - GROUND_UP: Lower topology observes Mind/Body → nudges toward stability from BELOW

Both forces point toward the SAME center (0.5 equilibrium). They don't cancel —
they create productive tension that stabilizes Mind between spiritual uplift
and material grounding. The balance IS coherence. This IS the Middle Path.

GROUND_UP modifies:
  - Body [0:5]: all 5 dimensions (full matter grounding)
  - Mind [10:15]: willing dimensions only (volition, intention, decision, commitment, agency)
  - Mind [0:9] and Spirit [0:45]: UNTOUCHED (sensory/cognitive Mind and Spirit are not grounded)
"""
import math


# Default grounding strength — conservative, same order as FILTER_DOWN
DEFAULT_STRENGTH = 0.1

# Damping factor to prevent oscillation overshoot
DEFAULT_DAMPING = 0.95

# Maximum nudge per dimension per epoch (safety clamp)
MAX_NUDGE = 0.05


class GroundUpEnricher:
    """Applies grounding enrichment from lower topology to Body and Mind willing."""

    def __init__(
        self,
        strength: float = DEFAULT_STRENGTH,
        damping: float = DEFAULT_DAMPING,
    ):
        """
        Args:
            strength: grounding force magnitude (0.0 = off, 1.0 = maximum)
            damping: smoothing factor to prevent oscillation (0.0-1.0)
        """
        self.strength = strength
        self.damping = damping
        self._prev_nudge_body: list[float] = [0.0] * 5
        self._prev_nudge_mind: list[float] = [0.0] * 5
        self._total_applications = 0
        self._total_body_delta = 0.0
        self._total_mind_delta = 0.0

    def compute_nudge(
        self,
        grounding_signal_10d: list[float],
        body_5d: list[float],
        mind_willing_5d: list[float],
    ) -> dict:
        """
        Compute grounding nudge vectors for Body and Mind willing.

        Args:
            grounding_signal_10d: gradient from LowerTopology (10D)
            body_5d: current body state [0:5]
            mind_willing_5d: current mind willing state [10:15]

        Returns:
            {"body_nudge": [5], "mind_nudge": [5], "total_magnitude": float}
        """
        signal = list(grounding_signal_10d)
        while len(signal) < 10:
            signal.append(0.0)

        # Split signal: body(0:5) + mind_willing(5:10)
        raw_body = signal[0:5]
        raw_mind = signal[5:10]

        # Apply damping (smooth with previous nudge to prevent jitter)
        body_nudge = [
            self.damping * prev + (1.0 - self.damping) * raw
            for prev, raw in zip(self._prev_nudge_body, raw_body)
        ]
        mind_nudge = [
            self.damping * prev + (1.0 - self.damping) * raw
            for prev, raw in zip(self._prev_nudge_mind, raw_mind)
        ]

        # Clamp to MAX_NUDGE (safety)
        body_nudge = [max(-MAX_NUDGE, min(MAX_NUDGE, n)) for n in body_nudge]
        mind_nudge = [max(-MAX_NUDGE, min(MAX_NUDGE, n)) for n in mind_nudge]

        self._prev_nudge_body = list(body_nudge)
        self._prev_nudge_mind = list(mind_nudge)

        total_mag = math.sqrt(
            sum(n * n for n in body_nudge) + sum(n * n for n in mind_nudge)
        )

        return {
            "body_nudge": body_nudge,
            "mind_nudge": mind_nudge,
            "total_magnitude": round(total_mag, 6),
        }

    def apply(
        self,
        body_5d: list[float],
        mind_15d: list[float],
        grounding_signal_10d: list[float],
        dt: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        """
        Apply grounding enrichment to Body and Mind.

        Args:
            body_5d: current body tensor [0:5]
            mind_15d: current FULL mind tensor [0:15] (only [10:15] is modified)
            grounding_signal_10d: gradient from LowerTopology
            dt: time delta (seconds since last epoch)

        Returns:
            (new_body_5d, new_mind_15d) — enriched tensors
        """
        nudge = self.compute_nudge(grounding_signal_10d, body_5d, mind_15d[10:15])

        # Apply to body (all 5D)
        new_body = list(body_5d)
        for i in range(min(5, len(new_body))):
            delta = nudge["body_nudge"][i] * self.strength * min(dt, 30.0)
            new_body[i] += delta
            # Clamp to [0, 1] — valid tensor range
            new_body[i] = max(0.0, min(1.0, new_body[i]))

        # Apply to mind (ONLY dims 10-14 = willing)
        new_mind = list(mind_15d)
        while len(new_mind) < 15:
            new_mind.append(0.0)
        for i in range(5):
            if 10 + i < len(new_mind):
                delta = nudge["mind_nudge"][i] * self.strength * min(dt, 30.0)
                new_mind[10 + i] += delta
                new_mind[10 + i] = max(0.0, min(1.0, new_mind[10 + i]))

        # Track stats
        self._total_applications += 1
        self._total_body_delta += sum(abs(n) for n in nudge["body_nudge"])
        self._total_mind_delta += sum(abs(n) for n in nudge["mind_nudge"])

        return new_body, new_mind

    def get_state(self) -> dict:
        """Serialize mutable state for hot-reload.

        Preserves damped nudge vectors and cumulative counters.
        The nudge computation itself runs each tick, but previous
        nudge values are needed for damping continuity.
        """
        return {
            "prev_nudge_body": list(self._prev_nudge_body),
            "prev_nudge_mind": list(self._prev_nudge_mind),
            "total_applications": self._total_applications,
            "total_body_delta": self._total_body_delta,
            "total_mind_delta": self._total_mind_delta,
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from hot-reload snapshot."""
        self._prev_nudge_body = list(state.get("prev_nudge_body", [0.0] * 5))
        self._prev_nudge_mind = list(state.get("prev_nudge_mind", [0.0] * 5))
        self._total_applications = state.get("total_applications", 0)
        self._total_body_delta = state.get("total_body_delta", 0.0)
        self._total_mind_delta = state.get("total_mind_delta", 0.0)

    def get_stats(self) -> dict:
        """Full stats for API."""
        return {
            "strength": self.strength,
            "damping": self.damping,
            "total_applications": self._total_applications,
            "cumulative_body_delta": round(self._total_body_delta, 4),
            "cumulative_mind_delta": round(self._total_mind_delta, 4),
            "last_body_nudge": [round(n, 6) for n in self._prev_nudge_body],
            "last_mind_nudge": [round(n, 6) for n in self._prev_nudge_mind],
        }
