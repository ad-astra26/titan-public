"""
titan_plugin/logic/topology.py — Space Topology Engine (T5).

Computes Titan's inner space topology from body part observables:
  - Distance matrix: pairwise L2 between 6 observable vectors
  - Volume: sum of all pairwise distances (how expanded inner space is)
  - Curvature: rate of change of volume (contracting=positive, expanding=negative)
  - Clusters: groups of body parts with similar observable signatures

These metrics feed:
  - InnerState.topology (T2)
  - SpiritState.topology (T2)
  - Dreaming cycle fatigue/readiness (T6)
  - GREAT PULSE convergence detection (T7)
"""
import logging
import math
from typing import Sequence
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# Observable keys for vector construction
OBSERVABLE_KEYS = ("coherence", "magnitude", "velocity", "direction", "polarity")

# Rolling window size for curvature computation
VOLUME_HISTORY_SIZE = 20

# Cluster distance threshold (parts closer than this are clustered)
DEFAULT_CLUSTER_THRESHOLD = 0.3


class TopologyEngine:
    """Computes Titan's inner space topology from body part observables."""

    def __init__(self, cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD):
        self._volume_history: list[float] = []
        self._cluster_threshold = cluster_threshold

    def compute(self, observables: dict[str, dict]) -> dict:
        """
        Compute full topology from current observables.

        Args:
            observables: {part_name: {coherence, magnitude, velocity, direction, polarity}}

        Returns:
            {
                "distance_matrix": {(a, b): dist, ...},
                "volume": float,
                "curvature": float,
                "clusters": [[part_names], ...],
                "isolated": [part_names],
                "mean_distance": float,
            }
        """
        if not observables or len(observables) < 2:
            return {"volume": 0.0, "curvature": 0.0, "clusters": [],
                    "isolated": [], "mean_distance": 0.0, "distance_matrix": {}}

        # Build observable vectors
        vectors = {}
        for name, obs in observables.items():
            vectors[name] = [obs.get(k, 0.0) for k in OBSERVABLE_KEYS]

        # Pairwise distances
        names = sorted(vectors.keys())
        distances = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = _l2_distance(vectors[a], vectors[b])
                distances[(a, b)] = round(d, 6)

        # Volume = sum of all pairwise distances
        volume = sum(distances.values())

        # Track volume history for curvature
        self._volume_history.append(volume)
        if len(self._volume_history) > VOLUME_HISTORY_SIZE:
            self._volume_history.pop(0)

        curvature = self._compute_curvature()

        # Clusters
        clusters, isolated = self._find_clusters(names, distances)

        mean_dist = volume / len(distances) if distances else 0.0

        return {
            "distance_matrix": distances,
            "volume": round(volume, 6),
            "curvature": round(curvature, 6),
            "clusters": clusters,
            "isolated": isolated,
            "mean_distance": round(mean_dist, 6),
        }

    def _compute_curvature(self) -> float:
        """
        Curvature = rate of change of volume.

        Positive curvature = contracting (inner space getting smaller)
        Negative curvature = expanding (inner space getting larger)
        Zero = stable
        """
        if len(self._volume_history) < 2:
            return 0.0

        # Simple: difference of last two volumes, normalized
        prev = self._volume_history[-2]
        curr = self._volume_history[-1]
        if prev < 1e-10:
            return 0.0
        # Positive when contracting (prev > curr), negative when expanding
        return (prev - curr) / prev

    def _find_clusters(
        self, names: list[str], distances: dict[tuple, float]
    ) -> tuple[list[list[str]], list[str]]:
        """
        Simple single-linkage clustering based on distance threshold.

        Returns:
            (clusters, isolated) — clusters are lists of 2+ names,
            isolated are names not in any cluster.
        """
        # Build adjacency
        adj: dict[str, set] = {n: set() for n in names}
        for (a, b), d in distances.items():
            if d <= self._cluster_threshold:
                adj[a].add(b)
                adj[b].add(a)

        # BFS to find connected components
        visited = set()
        clusters = []
        clustered = set()
        for name in names:
            if name in visited:
                continue
            if not adj[name]:
                visited.add(name)
                continue
            # BFS
            component = []
            queue = [name]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= 2:
                clusters.append(sorted(component))
                clustered.update(component)

        # Isolated = nodes not in any cluster (no close neighbors)
        isolated = sorted(n for n in names if n not in clustered)
        return clusters, isolated

    def is_convergence_peak(self) -> bool:
        """
        T7: True when volume just reached a local minimum (maximum contraction).

        A convergence peak means inner space contracted to its tightest point
        and is now expanding — the natural moment for a GREAT PULSE.
        """
        if len(self._volume_history) < 3:
            return False
        # Volume[-2] is a local minimum: lower than both neighbors
        return (self._volume_history[-2] < self._volume_history[-3] and
                self._volume_history[-2] <= self._volume_history[-1])

    def get_volume_trend(self, window: int = 5) -> float:
        """
        Average volume change over recent history.

        Positive = contracting, negative = expanding.
        """
        if len(self._volume_history) < 2:
            return 0.0
        recent = self._volume_history[-window:]
        if len(recent) < 2:
            return 0.0
        changes = [recent[i] - recent[i + 1] for i in range(len(recent) - 1)]
        return sum(changes) / len(changes)

    def get_state(self) -> dict:
        """Serialize mutable state for hot-reload.

        Preserves volume history (needed for curvature computation),
        sectional history, and previous 130D snapshot (needed for
        dimensional flow). The topology itself recomputes each tick,
        but these rolling windows must survive reload.
        """
        state: dict = {
            "volume_history": list(self._volume_history),
            "cluster_threshold": self._cluster_threshold,
        }
        if hasattr(self, '_sectional_history'):
            state["sectional_history"] = {
                k: list(v) for k, v in self._sectional_history.items()
            }
        if hasattr(self, '_prev_130d') and self._prev_130d is not None:
            state["prev_130d"] = list(self._prev_130d)
        if hasattr(self, '_last_cross_mirror'):
            state["last_cross_mirror"] = self._last_cross_mirror
        return state

    def restore_state(self, state: dict) -> None:
        """Restore mutable state from hot-reload snapshot."""
        self._volume_history = list(state.get("volume_history", []))
        self._cluster_threshold = state.get("cluster_threshold", self._cluster_threshold)
        if "sectional_history" in state:
            self._sectional_history = {
                k: list(v) for k, v in state["sectional_history"].items()
            }
        if "prev_130d" in state:
            self._prev_130d = list(state["prev_130d"])
        if "last_cross_mirror" in state:
            self._last_cross_mirror = state["last_cross_mirror"]
        logger.info(
            "[TopologyEngine] Hot-reload restored: volume_history=%d entries",
            len(self._volume_history),
        )

    def get_stats(self) -> dict:
        return {
            "volume_history_size": len(self._volume_history),
            "current_volume": self._volume_history[-1] if self._volume_history else 0.0,
            "current_curvature": self._compute_curvature(),
            "cluster_threshold": self._cluster_threshold,
        }


    # ── WHOLE 10DT Space Topology (Grounding Refinement) ─────────────

    def compute_whole_10d(
        self,
        basic_topology: dict,
        inner_lower: dict,
        outer_lower: dict,
        inner_mind_willing: list = None,
        outer_mind_willing: list = None,
        spirit_magnitudes: list = None,
    ) -> list:
        """
        Compute extended WHOLE space topology (10DT).

        Combines existing 6 metrics with 4 new grounding-aware dimensions:
          [0] volume, [1] curvature, [2] density, [3] mean_distance,
          [4] cross_layer_mirror, [5] cluster_count,
          [6] grounding_tension, [7] matter_spirit_ratio,
          [8] willing_coherence, [9] field_polarity

        Args:
            basic_topology: result of self.compute() (existing 6 metrics)
            inner_lower: inner LowerTopology observables dict
            outer_lower: outer LowerTopology observables dict
            inner_mind_willing: inner mind[10:15] (5D)
            outer_mind_willing: outer mind[10:15] (5D)
            spirit_magnitudes: [inner_spirit_mag, outer_spirit_mag]
        """
        inner_obs = inner_lower.get("observables", {})
        outer_obs = outer_lower.get("observables", {})

        # Existing 6 dimensions
        volume = basic_topology.get("volume", 0.0)
        curvature = basic_topology.get("curvature", 0.0)
        mean_dist = basic_topology.get("mean_distance", 0.0)
        density = 1.0 / max(0.01, mean_dist) if mean_dist > 0 else 0.0
        clusters = basic_topology.get("clusters", [])
        cluster_count = len(clusters)

        # Cross-layer mirror from extended topology (if available)
        cross_mirror = getattr(self, '_last_cross_mirror', 0.0)

        # ── 4 new dimensions ──

        # [6] Grounding tension: difference between inner/outer magnitudes + anchor freshness
        # Fresh on-chain anchor = stronger physical grounding = lower tension
        inner_mag = inner_obs.get("magnitude", 0.0)
        outer_mag = outer_obs.get("magnitude", 0.0)
        base_tension = abs(inner_mag - outer_mag)
        # Read anchor freshness from shared state (spirit_worker writes anchor_state.json)
        anchor_factor = 1.0  # no anchor data = full tension
        try:
            import json as _json, os as _os
            _anchor_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
            if _os.path.exists(_anchor_path):
                import time as _time
                with open(_anchor_path) as _af:
                    _anc = _json.load(_af)
                _since = _time.time() - _anc.get("last_anchor_time", _time.time())
                # Fresh anchor reduces tension: 0s=0.5x, 300s=0.75x, 600s=~1.0x
                anchor_factor = 0.5 + 0.5 * min(1.0, _since / 600.0)
        except Exception as _swallow_exc:
            swallow_warn('[logic.topology] TopologyEngine.compute_whole_10d: import json as _json, os as _os', _swallow_exc,
                         key='logic.topology.TopologyEngine.compute_whole_10d.line300', throttle=100)
        grounding_tension = base_tension * anchor_factor

        # [7] Matter-spirit ratio: balance between material and spiritual poles
        mean_lower_mag = (inner_mag + outer_mag) / 2.0
        mean_spirit_mag = 0.5  # Default
        if spirit_magnitudes and len(spirit_magnitudes) >= 2:
            mean_spirit_mag = sum(spirit_magnitudes) / len(spirit_magnitudes)
        matter_spirit_ratio = mean_lower_mag / max(0.01, mean_spirit_mag)

        # [8] Willing coherence: cosine similarity between inner and outer mind willing
        willing_coherence = 0.0
        if inner_mind_willing and outer_mind_willing:
            dot = sum(a * b for a, b in zip(inner_mind_willing, outer_mind_willing))
            mag_i = math.sqrt(sum(v * v for v in inner_mind_willing))
            mag_o = math.sqrt(sum(v * v for v in outer_mind_willing))
            if mag_i > 1e-10 and mag_o > 1e-10:
                willing_coherence = dot / (mag_i * mag_o)

        # [9] Field polarity: rate of oscillation (contraction↔expansion)
        field_polarity = self._compute_curvature()  # Reuse volume curvature

        # Track cross_mirror for next call
        self._last_cross_mirror = cross_mirror

        return [
            round(volume, 6),
            round(curvature, 6),
            round(density, 6),
            round(mean_dist, 6),
            round(cross_mirror, 6),
            float(cluster_count),
            round(grounding_tension, 6),
            round(matter_spirit_ratio, 6),
            round(willing_coherence, 6),
            round(field_polarity, 6),
        ]

    # ── 130D Extended Topology (DQ7) ────────────────────────────────

    def compute_extended(self, inner_65d: list, outer_65d: list) -> dict:
        """
        Compute extended topology for 130D space.

        Adds measures that require full Inner+Outer dimensional awareness:
        - Sectional curvature per dimension group (Body/Mind/Spirit)
        - Cross-layer mirror (inner↔outer correlation)
        - Dimensional flow (which dims change fastest)
        - Density landscape (centroid + variance per group)

        Returns extended topology dict (superset of basic compute()).
        """
        result = {}

        # ── Sectional curvature: per dimension group ──
        # Track volume change rate separately for each 5D/15D/45D subspace
        groups = {
            "inner_body": inner_65d[0:5],
            "inner_mind": inner_65d[5:20],
            "inner_spirit": inner_65d[20:65],
            "outer_body": outer_65d[0:5],
            "outer_mind": outer_65d[5:20],
            "outer_spirit": outer_65d[20:65],
        }
        sectional = {}
        for name, dims in groups.items():
            mag = math.sqrt(sum(v * v for v in dims)) if dims else 0.0
            # Track in _sectional_history
            if not hasattr(self, '_sectional_history'):
                self._sectional_history = {}
            hist = self._sectional_history.setdefault(name, [])
            hist.append(mag)
            if len(hist) > VOLUME_HISTORY_SIZE:
                hist.pop(0)
            # Curvature = rate of change
            if len(hist) >= 2 and hist[-2] > 1e-10:
                sectional[name] = round((hist[-2] - hist[-1]) / hist[-2], 6)
            else:
                sectional[name] = 0.0
        result["sectional_curvature"] = sectional

        # ── Cross-layer mirror: inner↔outer correlation ──
        # Cosine similarity between inner 65D and outer 65D
        dot = sum(a * b for a, b in zip(inner_65d, outer_65d))
        mag_i = math.sqrt(sum(v * v for v in inner_65d))
        mag_o = math.sqrt(sum(v * v for v in outer_65d))
        if mag_i > 1e-10 and mag_o > 1e-10:
            mirror = dot / (mag_i * mag_o)
        else:
            mirror = 0.0
        result["cross_layer_mirror"] = round(mirror, 6)

        # Per-group mirror (Body↔Body, Mind↔Mind, Spirit↔Spirit)
        group_pairs = [
            ("body", inner_65d[0:5], outer_65d[0:5]),
            ("mind", inner_65d[5:20], outer_65d[5:20]),
            ("spirit", inner_65d[20:65], outer_65d[20:65]),
        ]
        group_mirrors = {}
        for name, inner, outer in group_pairs:
            d = sum(a * b for a, b in zip(inner, outer))
            mi = math.sqrt(sum(v * v for v in inner))
            mo = math.sqrt(sum(v * v for v in outer))
            group_mirrors[name] = round(d / (mi * mo), 6) if mi > 1e-10 and mo > 1e-10 else 0.0
        result["group_mirrors"] = group_mirrors

        # ── Dimensional flow: velocity per dimension ──
        if not hasattr(self, '_prev_130d'):
            self._prev_130d = None
        full_130d = inner_65d + outer_65d
        if self._prev_130d and len(self._prev_130d) == len(full_130d):
            flow = [abs(a - b) for a, b in zip(full_130d, self._prev_130d)]
            # Top 10 most active dimensions
            indexed = sorted(enumerate(flow), key=lambda x: -x[1])
            result["dimensional_flow"] = {
                "total_flow": round(sum(flow), 4),
                "mean_flow": round(sum(flow) / max(1, len(flow)), 6),
                "top_active": [(i, round(v, 4)) for i, v in indexed[:10]],
            }
        else:
            result["dimensional_flow"] = {
                "total_flow": 0.0, "mean_flow": 0.0, "top_active": [],
            }
        self._prev_130d = list(full_130d)

        # ── Density landscape: centroid + variance per group ──
        density_landscape = {}
        for name, dims in groups.items():
            if dims:
                mean = sum(dims) / len(dims)
                variance = sum((v - mean) ** 2 for v in dims) / len(dims)
                density_landscape[name] = {
                    "centroid": round(mean, 4),
                    "variance": round(variance, 6),
                    "spread": round(math.sqrt(variance), 4),
                }
        result["density_landscape"] = density_landscape

        return result


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """L2 distance between two vectors."""
    return math.sqrt(sum((va - vb) ** 2 for va, vb in zip(a, b)))
