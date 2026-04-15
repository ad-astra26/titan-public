"""
titan_plugin/logic/observables.py — Tensor Observable Engine (T1).

Computes 5 observables for each of the 6 Trinity body parts:
  1. coherence  — internal alignment (from middle_path.layer_coherence)
  2. magnitude  — normalized L2 norm (how "activated" the tensor is)
  3. velocity   — L2 distance from previous tensor (rate of change)
  4. direction  — cosine similarity with previous tensor (same way or reversing?)
  5. polarity   — signed offset from center (mean - 0.5)

These observables are the foundation for:
  - Sphere clock velocity (coherence replaces layer_loss)
  - Space topology (T5: distance matrix, volume, curvature)
  - Dreaming cycle (T6: fatigue/readiness scoring)
  - Emergent GREAT PULSE (T7: topology convergence detection)
"""
import math
from typing import Sequence

from titan_plugin.logic.middle_path import layer_coherence


# ── Math helpers ─────────────────────────────────────────────────────

def l2_norm(tensor: Sequence[float]) -> float:
    """L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(v * v for v in tensor))


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """L2 distance between two vectors of equal length."""
    return math.sqrt(sum((va - vb) ** 2 for va, vb in zip(a, b)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two vectors. Returns [-1.0, 1.0].
    Returns 1.0 if both are zero vectors (no change = same direction).
    """
    dot = sum(va * vb for va, vb in zip(a, b))
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0  # no movement = same direction by convention
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def mean(tensor: Sequence[float]) -> float:
    """Arithmetic mean."""
    if not tensor:
        return 0.0
    return sum(tensor) / len(tensor)


# ── Observer per body part ───────────────────────────────────────────

class BodyPartObserver:
    """Computes 5 observables for one Trinity body part."""

    __slots__ = ("name", "_prev_tensor")

    def __init__(self, name: str, dim: int = 5):
        self.name = name
        self._prev_tensor: list[float] = [0.5] * dim

    def observe(self, tensor: Sequence[float]) -> dict:
        """
        Compute observables from current tensor.

        Args:
            tensor: Current tensor for this body part (5D body, 15D mind, 45D spirit).

        Returns:
            Dict with keys: coherence, magnitude, velocity, direction, polarity.
        """
        t = list(tensor)
        dim = len(t) or 1

        coherence = layer_coherence(t)
        magnitude = l2_norm(t) / math.sqrt(dim)     # normalized to [0, 1]
        velocity = l2_distance(t, self._prev_tensor)
        direction = cosine_similarity(t, self._prev_tensor)
        polarity = mean(t) - 0.5                     # signed [-0.5, 0.5]

        self._prev_tensor = t

        return {
            "coherence": round(coherence, 6),
            "magnitude": round(magnitude, 6),
            "velocity": round(velocity, 6),
            "direction": round(direction, 6),
            "polarity": round(polarity, 6),
        }


# ── Engine for all 6 parts ──────────────────────────────────────────

# Canonical ordering matches sphere_clock.ALL_COMPONENTS
ALL_PARTS = (
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
)

# True Trinity dimensions: Body=5D, Mind=15D (Feeling+Thinking+Willing), Spirit=45D
PART_DIMS = {
    "inner_body": 5, "inner_mind": 15, "inner_spirit": 45,
    "outer_body": 5, "outer_mind": 15, "outer_spirit": 45,
}


class ObservableEngine:
    """Computes observables for all 6 Trinity body parts at true dimensions."""

    def __init__(self):
        self.observers: dict[str, BodyPartObserver] = {
            name: BodyPartObserver(name, dim=PART_DIMS[name])
            for name in ALL_PARTS
        }

    def observe_all(self, tensors: dict[str, Sequence[float]]) -> dict[str, dict]:
        """
        Compute observables for all 6 parts.

        Args:
            tensors: Dict mapping part name → tensor values.
                     Missing parts default to [0.5]*dim per part.

        Returns:
            Dict mapping part name → {coherence, magnitude, velocity, direction, polarity}.
        """
        return {
            name: observer.observe(tensors.get(name, [0.5] * PART_DIMS[name]))
            for name, observer in self.observers.items()
        }

    def observe_inner(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
    ) -> dict[str, dict]:
        """Convenience: observe just the 3 inner parts."""
        return {
            "inner_body": self.observers["inner_body"].observe(body),
            "inner_mind": self.observers["inner_mind"].observe(mind),
            "inner_spirit": self.observers["inner_spirit"].observe(spirit),
        }

    def observe_outer(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
    ) -> dict[str, dict]:
        """Convenience: observe just the 3 outer parts."""
        return {
            "outer_body": self.observers["outer_body"].observe(body),
            "outer_mind": self.observers["outer_mind"].observe(mind),
            "outer_spirit": self.observers["outer_spirit"].observe(spirit),
        }

    def get_observations_30d(self, observables: dict) -> list[float]:
        """Flatten an observables dict into the canonical 30D vector.

        Pure function — does NOT mutate observer state. Safe to call from
        any consumer that received the dict via bus or shared state.

        Canonical order (matches ALL_PARTS): 6 body parts × 5 observables.
          part 0 (inner_body):   [coherence, magnitude, velocity, direction, polarity]
          part 1 (inner_mind):   [coh, mag, vel, dir, pol]
          part 2 (inner_spirit): [coh, mag, vel, dir, pol]
          part 3 (outer_body):   [coh, mag, vel, dir, pol]
          part 4 (outer_mind):   [coh, mag, vel, dir, pol]
          part 5 (outer_spirit): [coh, mag, vel, dir, pol]

        Missing part → 5 × 0.0 (observables are centered/signed; 0.0 is
        the neutral default, NOT 0.5 which is the felt-state midpoint).
        Missing observable key → 0.0.

        Returns exactly 30 floats.
        """
        OBS_KEYS = ("coherence", "magnitude", "velocity", "direction", "polarity")
        flat: list[float] = []
        for part in ALL_PARTS:
            part_obs = observables.get(part, {}) if isinstance(observables, dict) else {}
            for key in OBS_KEYS:
                flat.append(float(part_obs.get(key, 0.0)))
        return flat

    def get_coherences(self, observables: dict[str, dict] = None) -> dict[str, float]:
        """
        Extract just the coherence values from the last observation.

        If observables dict is passed, extract from it. Otherwise use
        the last _prev_tensor in each observer to recompute.
        """
        if observables:
            return {name: obs["coherence"] for name, obs in observables.items()}
        # Recompute from stored prev tensors
        return {
            name: round(layer_coherence(observer._prev_tensor), 6)
            for name, observer in self.observers.items()
        }
