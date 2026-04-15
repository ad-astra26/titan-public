"""
Spatial Mini-Reasoner — fast reflexive analysis of body/spatial topology.

Primitives:
  COMPARE_SPATIAL: Compare inner vs outer body regions → similarity, divergence
  DETECT_PATTERN:  Track rolling topology changes → pattern direction
  TRACK_MOVEMENT:  Monitor body coherence velocity → movement speed/direction

Runs at Body rate (7.83 Hz computation gate). Feeds structured spatial
conclusions to main reasoning via MiniReasonerRegistry.query("spatial").
"""
import logging
import numpy as np

from .mini_experience import MiniReasoner

logger = logging.getLogger(__name__)


class SpatialMiniReasoner(MiniReasoner):
    domain = "spatial"
    primitives = ["COMPARE_SPATIAL", "DETECT_PATTERN", "TRACK_MOVEMENT"]
    rate_tier = "body"
    observation_dim = 30  # ObservationSpace Tier 1: 6 parts × 5 metrics

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_topology = None
        self._topology_ema = None

    def perceive(self, context: dict) -> np.ndarray:
        """Extract 30D core observables from observation space."""
        obs = context.get("observables")
        if obs is None:
            return np.zeros(self.observation_dim)
        # observables is either a dict of parts or a flat array
        if isinstance(obs, dict):
            # Flatten: 6 parts × 5 metrics
            flat = []
            for part in ["inner_body", "inner_mind", "inner_spirit",
                         "outer_body", "outer_mind", "outer_spirit"]:
                part_data = obs.get(part, {})
                if isinstance(part_data, dict):
                    flat.extend([
                        part_data.get("coherence", 0.5),
                        part_data.get("magnitude", 0.5),
                        part_data.get("velocity", 0.0),
                        part_data.get("direction", 0.0),
                        part_data.get("polarity", 0.0),
                    ])
                elif isinstance(part_data, (list, np.ndarray)):
                    flat.extend(list(part_data)[:5])
                else:
                    flat.extend([0.5] * 5)
            return np.array(flat[:self.observation_dim], dtype=np.float64)
        elif isinstance(obs, (list, np.ndarray)):
            arr = np.array(obs, dtype=np.float64)
            if len(arr) >= self.observation_dim:
                return arr[:self.observation_dim]
            return np.concatenate([arr, np.zeros(self.observation_dim - len(arr))])
        return np.zeros(self.observation_dim)

    def execute_primitive(self, primitive_idx: int, observation: np.ndarray) -> dict:
        if primitive_idx == 0:
            return self._compare_spatial(observation)
        elif primitive_idx == 1:
            return self._detect_pattern(observation)
        else:
            return self._track_movement(observation)

    def _compare_spatial(self, obs: np.ndarray) -> dict:
        """Compare inner body (0:15) vs outer body (15:30)."""
        inner = obs[:15]
        outer = obs[15:30] if len(obs) >= 30 else np.zeros(15)
        diff = np.abs(inner - outer)
        similarity = 1.0 - float(np.mean(diff))
        max_divergence_dim = int(np.argmax(diff))
        return {
            "relevance": float(np.max(diff)),
            "confidence": max(0.3, similarity),
            "similarity": round(similarity, 3),
            "divergence": round(float(np.mean(diff)), 3),
            "max_divergence_dim": max_divergence_dim,
        }

    def _detect_pattern(self, obs: np.ndarray) -> dict:
        """Track topology changes via EMA deviation."""
        if self._topology_ema is None:
            self._topology_ema = obs.copy()
            self._prev_topology = obs.copy()
            return {"relevance": 0.0, "confidence": 0.3, "pattern_detected": False}

        # Update EMA
        alpha = 0.1
        self._topology_ema = self._topology_ema * (1 - alpha) + obs * alpha
        deviation = obs - self._topology_ema
        deviation_mag = float(np.linalg.norm(deviation))
        pattern_detected = deviation_mag > 0.3

        # Direction of change
        delta = obs - self._prev_topology
        direction = int(np.argmax(np.abs(delta)))
        self._prev_topology = obs.copy()

        return {
            "relevance": min(1.0, deviation_mag),
            "confidence": 0.5 + min(0.4, deviation_mag),
            "pattern_detected": pattern_detected,
            "deviation_magnitude": round(deviation_mag, 3),
            "change_direction": direction,
        }

    def _track_movement(self, obs: np.ndarray) -> dict:
        """Monitor velocity dimensions (indices 2, 7, 12, 17, 22, 27)."""
        velocity_indices = [2, 7, 12, 17, 22, 27]
        velocities = [obs[i] if i < len(obs) else 0.0 for i in velocity_indices]
        avg_speed = float(np.mean(np.abs(velocities)))
        max_idx = int(np.argmax(np.abs(velocities)))
        return {
            "relevance": min(1.0, avg_speed * 2),
            "confidence": 0.4 + min(0.5, avg_speed),
            "avg_speed": round(avg_speed, 3),
            "fastest_part": max_idx,
            "fastest_velocity": round(float(velocities[max_idx]), 3),
        }

    def format_summary(self) -> dict:
        s = dict(self._latest_summary)
        s.setdefault("relevance", 0.0)
        s.setdefault("confidence", 0.3)
        s.setdefault("primitive", self.primitives[0])
        return s
