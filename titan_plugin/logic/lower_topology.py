"""
titan_plugin/logic/lower_topology.py — Lower Space Topology (10DT).

Computes grounding topology for each trinity — the material pole that
counterbalances the spiritual pull from FILTER_DOWN.

Two variants:
  - "inner": ethereal matter — the mathematical form of grounding
  - "outer": dense matter — physical/digital reality grounding

Each lower topology is 10DT:
  - Dims 0-4: body grounding (full 5DT body mapping)
  - Dims 5-9: mind willing grounding (volitional 5DT of mind[10:15])

The grounding signal pushes UPWARD into Body and Mind, giving Mind
the missing anchor it needs to find coherence between spiritual
uplift (FILTER_DOWN) and material grounding (GROUND_UP).
"""
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Rolling window for velocity computation
HISTORY_SIZE = 20

# Reference state: balanced material equilibrium
GROUND_REFERENCE = [0.5] * 10


class LowerTopology:
    """10DT space topology for grounding a trinity in matter."""

    def __init__(
        self,
        variant: str = "inner",
        grounding_strength: float = 0.1,
        reference_state: Optional[list] = None,
    ):
        """
        Args:
            variant: "inner" (ethereal) or "outer" (dense physical)
            grounding_strength: magnitude of grounding gradient (0.0-1.0)
            reference_state: 10D equilibrium point (default: 0.5 vector)
        """
        self.variant = variant
        self.grounding_strength = grounding_strength
        self._reference = list(reference_state or GROUND_REFERENCE)
        self._history: list[list[float]] = []
        self._magnitude_history: list[float] = []
        self._last_state: Optional[list[float]] = None
        self._observables: dict = {
            "coherence": 0.0,
            "magnitude": 0.0,
            "velocity": 0.0,
            "direction": 0.0,
            "polarity": 0.0,
        }
        self._grounding_signal: list[float] = [0.0] * 10
        self._topology_10d: list[float] = [0.0] * 10
        self._compute_count = 0

    def compute(
        self,
        body_5d: list[float],
        mind_willing_5d: list[float],
        whole_10d: Optional[list[float]] = None,
    ) -> dict:
        """
        Compute lower topology from body state + mind willing state.

        Args:
            body_5d: body tensor [0:5] from this trinity
            mind_willing_5d: mind willing tensor [10:15] from this trinity
            whole_10d: WHOLE space topology (for polarity alignment)

        Returns:
            dict with topology_10d, observables, grounding_signal
        """
        # Combine into 10DT state
        state = list(body_5d[:5]) + list(mind_willing_5d[:5])
        # Pad if short
        while len(state) < 10:
            state.append(0.0)

        # Outer variant: willing dims grounded by physical-world validation
        # Each willing dimension modulated by its real-world counterpart
        if self.variant == "outer":
            try:
                import json as _json, os as _os, time as _time
                _anchor_path = _os.path.join(
                    _os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
                _anc = {}
                if _os.path.exists(_anchor_path):
                    with open(_anchor_path) as _af:
                        _anc = _json.load(_af)

                _since = _time.time() - _anc.get("last_anchor_time", _time.time())
                _anchor_fresh = max(0.3, 1.0 / (1.0 + _since / 300.0))
                _anchor_success = 1.0 if _anc.get("success") else 0.5
                _balance = _anc.get("sol_balance", 13.0)
                _balance_healthy = min(1.0, _balance / 10.0)  # 10+ SOL = healthy

                # state[5] = action_drive × anchor_success (will to act grounded by outcomes)
                state[5] = state[5] * (0.5 + 0.5 * _anchor_success)
                # state[7] = creative_will × anchor_freshness (creation grounded by permanence)
                state[7] = state[7] * (0.5 + 0.5 * _anchor_fresh)
                # state[9] = growth_will × balance_trend (growth grounded by resources)
                state[9] = state[9] * (0.5 + 0.5 * _balance_healthy)
            except Exception:
                pass  # Graceful degradation — use unmodulated willing

        self._topology_10d = state
        self._compute_count += 1

        # Track history
        self._history.append(list(state))
        if len(self._history) > HISTORY_SIZE:
            self._history.pop(0)

        # ── Observables ──
        # Coherence: cosine similarity with reference (balanced) state
        coherence = _cosine_sim(state, self._reference)

        # Magnitude: L2 norm
        magnitude = _l2_norm(state)
        self._magnitude_history.append(magnitude)
        if len(self._magnitude_history) > HISTORY_SIZE:
            self._magnitude_history.pop(0)

        # Velocity: rate of change of magnitude
        velocity = 0.0
        if len(self._magnitude_history) >= 2:
            velocity = self._magnitude_history[-1] - self._magnitude_history[-2]

        # Direction: contracting (positive) or expanding (negative)
        direction = 1.0 if velocity < 0 else -1.0 if velocity > 0 else 0.0

        # Polarity: alignment with WHOLE topology
        polarity = 0.0
        if whole_10d and len(whole_10d) >= 10:
            polarity = _cosine_sim(state, whole_10d[:10])

        self._observables = {
            "coherence": round(coherence, 6),
            "magnitude": round(magnitude, 6),
            "velocity": round(velocity, 6),
            "direction": round(direction, 2),
            "polarity": round(polarity, 6),
        }

        # ── Grounding signal ──
        # Gradient pointing from current state toward ground center
        ground_center = self._compute_ground_center(whole_10d)
        self._grounding_signal = [
            (gc - s) * self.grounding_strength
            for gc, s in zip(ground_center, state)
        ]

        self._last_state = list(state)

        return {
            "topology_10d": self._topology_10d,
            "observables": dict(self._observables),
            "grounding_signal": list(self._grounding_signal),
        }

    def _compute_ground_center(
        self, whole_10d: Optional[list[float]] = None
    ) -> list[float]:
        """
        Compute the material equilibrium point — where grounding pulls toward.

        The ground center is influenced by the WHOLE topology:
        - If WHOLE has high curvature → ground center shifts toward stability
        - If WHOLE is balanced → ground center is the reference state (0.5)

        For inner variant: slightly ethereal (closer to 0.5 center)
        For outer variant: slightly denser (biased toward actual body state)
        """
        center = list(self._reference)

        if whole_10d and len(whole_10d) >= 6:
            # WHOLE curvature shifts ground center
            curvature = whole_10d[1] if len(whole_10d) > 1 else 0.0
            # High curvature → pull harder toward stability
            curvature_factor = min(1.0, abs(curvature) / math.pi) * 0.1

            if self.variant == "outer":
                # Dense: ground center biased toward actual body readings
                if self._last_state:
                    for i in range(5):  # Body dims only
                        center[i] = (
                            self._reference[i] * 0.7
                            + self._last_state[i] * 0.3
                            + curvature_factor * 0.1
                        )
            else:
                # Ethereal: ground center stays closer to pure reference
                for i in range(10):
                    center[i] = self._reference[i] + curvature_factor * 0.05

        return center

    def get_observables(self) -> dict:
        """Return current observable metrics."""
        return dict(self._observables)

    def get_grounding_signal(self) -> list[float]:
        """Return current grounding gradient (10D)."""
        return list(self._grounding_signal)

    def get_state(self) -> dict:
        """Serialize mutable state for hot-reload.

        Preserves rolling histories, cached observables, grounding signal,
        and counters. The topology recomputes each tick, but histories
        and damped state must survive reload.
        """
        return {
            "variant": self.variant,
            "history": [list(h) for h in self._history],
            "magnitude_history": list(self._magnitude_history),
            "last_state": list(self._last_state) if self._last_state else None,
            "observables": dict(self._observables),
            "grounding_signal": list(self._grounding_signal),
            "topology_10d": list(self._topology_10d),
            "compute_count": self._compute_count,
        }

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from hot-reload snapshot."""
        self._history = [list(h) for h in state.get("history", [])]
        self._magnitude_history = list(state.get("magnitude_history", []))
        last = state.get("last_state")
        self._last_state = list(last) if last else None
        self._observables = dict(state.get("observables", self._observables))
        self._grounding_signal = list(state.get("grounding_signal", self._grounding_signal))
        self._topology_10d = list(state.get("topology_10d", self._topology_10d))
        self._compute_count = state.get("compute_count", 0)
        logger.info(
            "[LowerTopology:%s] Hot-reload restored: %d computes, history=%d",
            self.variant, self._compute_count, len(self._history),
        )

    def get_stats(self) -> dict:
        """Full stats for API."""
        return {
            "variant": self.variant,
            "grounding_strength": self.grounding_strength,
            "compute_count": self._compute_count,
            "topology_10d": [round(v, 4) for v in self._topology_10d],
            "observables": dict(self._observables),
            "grounding_signal_magnitude": round(
                _l2_norm(self._grounding_signal), 4
            ),
            "history_length": len(self._history),
        }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def _l2_norm(v: list[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))
