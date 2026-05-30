"""Vertical Intuition Resonance — Somatic Marker Architecture (M11-M13).

Connects the HIGHEST level (SPIRIT_SELF meta-reasoning, 132D observer) with
the LOWEST level (hormonal intuition, autonomic body signals) to create a
vertical resonance channel. When upper wisdom and lower instinct AGREE,
a soft intuition bias is injected into outer reasoning.

Based on Antonio Damasio's Somatic Marker Hypothesis (1994):
- Body Loop: FilterDown value network + hormonal system (bottom-up)
- As-If Loop: PredictionEngine + prediction error (top-down)
- vmPFC: InnerTrinityCoordinator integration hub
- Convergence: agreement between meta and gut → felt-state bias

M11: Convergence Detector — compute agreement between SPIRIT_SELF and hormones
M12: Intuition Injection — soft bias into outer reasoning (±0.3 max)
M13: IQL Reward Shaping — learn to trust convergent intuition

See: titan-docs/rFP_vertical_intuition_resonance.md
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Main reasoning primitives (must match reasoning.py PRIMITIVES)
MAIN_PRIMITIVES = [
    "COMPARE", "IF_THEN", "SEQUENCE", "ASSOCIATE",
    "DECOMPOSE", "LOOP", "NEGATE", "CONCLUDE",
]

# Hormonal program → reasoning primitive affinity mapping
# Based on Damasio: each autonomic drive has natural reasoning mode
HORMONE_PRIMITIVE_MAP = {
    "VIGILANCE":  {"COMPARE": 0.4, "DECOMPOSE": 0.2},   # Threat → careful examination
    "CURIOSITY":  {"ASSOCIATE": 0.3, "SEQUENCE": 0.2},   # Exploration → find connections
    "FOCUS":      {"SEQUENCE": 0.3, "DECOMPOSE": 0.2},   # Concentration → systematic
    "INTUITION":  {"IF_THEN": 0.3, "ASSOCIATE": 0.2},    # Gut feeling → conditional
    "REFLECTION": {"DECOMPOSE": 0.2, "COMPARE": 0.2},    # Introspection → analyze
    "IMPULSE":    {"CONCLUDE": 0.3, "SEQUENCE": 0.2},    # Action drive → commit fast
    "CREATIVITY": {"ASSOCIATE": 0.3, "LOOP": 0.2},       # Creative → explore variations
    "EMPATHY":    {"COMPARE": 0.2, "IF_THEN": 0.2},      # Social → perspective-taking
    "INSPIRATION":{"ASSOCIATE": 0.3, "IF_THEN": 0.2},    # Eureka → novel connections
    "REFLEX":     {"CONCLUDE": 0.3, "NEGATE": 0.2},      # Protective → stop/reject
}

# Meta-reasoning domain → primitive affinity mapping
DOMAIN_PRIMITIVE_MAP = {
    "body_mind":        {"COMPARE": 0.3, "DECOMPOSE": 0.2},
    "outer_perception": {"COMPARE": 0.3, "ASSOCIATE": 0.2},
    "inner_spirit":     {"ASSOCIATE": 0.3, "IF_THEN": 0.2},
    "outer_spirit":     {"ASSOCIATE": 0.3, "IF_THEN": 0.2},
    "general":          {"COMPARE": 0.2, "SEQUENCE": 0.2},
}


class IntuitionConvergenceDetector:
    """M11: Detects when meta-reasoning (top) and hormonal intuition (bottom) agree.

    Produces 8D bias vectors for main reasoning primitives when convergence > threshold.
    Uses cosine similarity between meta-level and gut-level assessments.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._convergence_threshold = cfg.get("convergence_threshold", 0.5)
        self._max_bias = cfg.get("max_bias", 0.3)
        self._learned_weight = cfg.get("initial_learned_weight", 0.3)
        self._warmup_rate = cfg.get("warmup_rate", 0.001)
        self._max_learned_weight = cfg.get("max_learned_weight", 1.0)

        # State
        self._last_meta_vec = np.zeros(8, dtype=np.float32)
        self._last_gut_vec = np.zeros(8, dtype=np.float32)
        self._last_convergence = 0.0
        self._last_bias: np.ndarray | None = None
        self._total_convergence_events = 0
        self._total_checks = 0

        # Tracking for per-Titan divergence
        self._convergence_history: list[dict] = []
        self._primitive_convergence_counts = np.zeros(8, dtype=np.float32)

    def compute_meta_intuition(self, domain: str = "general",
                                hypotheses: list[dict] | None = None,
                                chain_primitives: list[str] | None = None) -> np.ndarray:
        """Convert SPIRIT_SELF's assessment into 8D reasoning bias vector.

        Uses domain assessment + best hypothesis strategy to suggest which
        reasoning primitives would be most appropriate.
        """
        vec = np.zeros(8, dtype=np.float32)

        # Domain → primitive affinity
        domain_map = DOMAIN_PRIMITIVE_MAP.get(domain, DOMAIN_PRIMITIVE_MAP["general"])
        for prim_name, weight in domain_map.items():
            if prim_name in MAIN_PRIMITIVES:
                vec[MAIN_PRIMITIVES.index(prim_name)] += weight

        # Hypothesis refinement: boost primitives in the best hypothesis strategy
        if hypotheses:
            best = max(hypotheses,
                       key=lambda h: h.get("predicted_confidence", 0),
                       default=None)
            if best:
                for prim in best.get("strategy", []):
                    if prim in MAIN_PRIMITIVES:
                        vec[MAIN_PRIMITIVES.index(prim)] += 0.2

        # Chain history: boost primitives that have been productive
        if chain_primitives:
            for prim in chain_primitives[-3:]:  # Last 3 actions
                if prim in MAIN_PRIMITIVES:
                    vec[MAIN_PRIMITIVES.index(prim)] += 0.1

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        self._last_meta_vec = vec
        return vec

    def compute_gut_intuition(self, hormonal_system=None,
                               neuromod_levels: dict | None = None) -> np.ndarray:
        """Convert hormonal/autonomic state into 8D reasoning bias vector.

        Maps hormonal pressure profile to reasoning primitive affinities.
        Uses actual hormone levels from HormonalPressureEngine.
        """
        vec = np.zeros(8, dtype=np.float32)

        if hormonal_system and hasattr(hormonal_system, '_hormones'):
            for hormone_name, mapping in HORMONE_PRIMITIVE_MAP.items():
                hormone = hormonal_system._hormones.get(hormone_name)
                if hormone is None:
                    continue
                level = getattr(hormone, 'level', 0.0)
                for prim_name, weight in mapping.items():
                    if prim_name in MAIN_PRIMITIVES:
                        vec[MAIN_PRIMITIVES.index(prim_name)] += level * weight

        # Also integrate neuromodulator levels as secondary signal
        if neuromod_levels:
            # High NE → COMPARE (alertness → careful examination)
            ne = neuromod_levels.get("NE", 0.5)
            vec[MAIN_PRIMITIVES.index("COMPARE")] += ne * 0.15
            # High DA → CONCLUDE (reward → commit to action)
            da = neuromod_levels.get("DA", 0.5)
            vec[MAIN_PRIMITIVES.index("CONCLUDE")] += da * 0.15
            # High ACh → SEQUENCE (focus → systematic)
            ach = neuromod_levels.get("ACh", 0.5)
            vec[MAIN_PRIMITIVES.index("SEQUENCE")] += ach * 0.15
            # High 5HT → IF_THEN (patience → conditional reasoning)
            sht = neuromod_levels.get("5HT", 0.5)
            vec[MAIN_PRIMITIVES.index("IF_THEN")] += sht * 0.10

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        self._last_gut_vec = vec
        return vec

    def compute_convergence(self, meta_vec: np.ndarray,
                             gut_vec: np.ndarray) -> float:
        """Compute convergence score: cosine similarity of meta and gut intuition.

        Returns 0.0 (disagree) to 1.0 (perfect agreement).
        """
        dot = float(np.dot(meta_vec, gut_vec))
        self._last_convergence = max(0.0, dot)
        return self._last_convergence

    def compute_intuition_bias(self, meta_vec: np.ndarray, gut_vec: np.ndarray,
                                convergence: float,
                                intuition_trust: float = 1.0) -> np.ndarray | None:
        """Produce the final intuition bias vector for injection into reasoning.

        Only produces signal when convergence > threshold.
        Weighted by learned_weight (ramps with experience) and intuition trust.
        Softer than DELEGATE bias (max ±0.3 vs DELEGATE's ±3.0).
        """
        self._total_checks += 1

        if convergence < self._convergence_threshold:
            self._last_bias = None
            return None

        self._total_convergence_events += 1

        # Average of meta and gut, weighted by convergence
        combined = (meta_vec + gut_vec) / 2.0
        bias = combined * convergence * self._learned_weight * intuition_trust
        bias = np.clip(bias, -self._max_bias, self._max_bias)

        # Track which primitives get convergence support (per-Titan divergence!)
        self._primitive_convergence_counts += np.abs(bias)

        self._last_bias = bias.copy()

        # Slowly ramp learned_weight toward max (trust builds with experience)
        self._learned_weight = min(
            self._max_learned_weight,
            self._learned_weight + self._warmup_rate)

        return bias

    def check(self, domain: str = "general",
              hypotheses: list[dict] | None = None,
              chain_primitives: list[str] | None = None,
              hormonal_system=None,
              neuromod_levels: dict | None = None,
              intuition_trust: float = 1.0,
              current_epoch: int = 0,
              pi_value: float = 0.0) -> dict:
        """Full convergence check: compute meta + gut + convergence + bias.

        Returns dict with all computed values and optional bias vector.
        """
        meta_vec = self.compute_meta_intuition(domain, hypotheses, chain_primitives)
        gut_vec = self.compute_gut_intuition(hormonal_system, neuromod_levels)
        convergence = self.compute_convergence(meta_vec, gut_vec)
        bias = self.compute_intuition_bias(meta_vec, gut_vec, convergence,
                                            intuition_trust)

        result = {
            "convergence": round(convergence, 4),
            "has_bias": bias is not None,
            "meta_top3": self._top_primitives(meta_vec, 3),
            "gut_top3": self._top_primitives(gut_vec, 3),
            "learned_weight": round(self._learned_weight, 4),
        }

        # Log strong convergence events with pi (like "I AM" events)
        if convergence >= 0.7:
            self._convergence_history.append({
                "epoch": current_epoch,
                "convergence": round(convergence, 4),
                "pi_value": round(pi_value, 6),
                "meta_top": self._top_primitives(meta_vec, 1),
                "gut_top": self._top_primitives(gut_vec, 1),
                "timestamp": time.time(),
            })
            # Keep last 100
            if len(self._convergence_history) > 100:
                self._convergence_history = self._convergence_history[-100:]

            logger.info(
                "[INTUITION-CONVERGENCE] %.3f — meta=%s gut=%s weight=%.3f "
                "pi=%.4f epoch=%d",
                convergence,
                result["meta_top3"], result["gut_top3"],
                self._learned_weight, pi_value, current_epoch)

        if bias is not None:
            result["bias"] = {MAIN_PRIMITIVES[i]: round(float(bias[i]), 3)
                              for i in range(8) if abs(bias[i]) > 0.001}

        return result

    def _top_primitives(self, vec: np.ndarray, n: int) -> list[str]:
        """Return top N primitive names from a bias vector."""
        indices = np.argsort(-vec)[:n]
        return [MAIN_PRIMITIVES[i] for i in indices if vec[i] > 0.01]

    def get_convergence_profile(self) -> dict:
        """Per-Titan convergence personality: which primitives get intuition support.

        T1 (Hypothesizer) will naturally converge on ASSOCIATE/IF_THEN.
        T2 (Delegator) will converge on SEQUENCE/COMPARE.
        T3 (Evaluator) will converge on COMPARE/DECOMPOSE.
        These profiles emerge from the different meta-reasoning personalities.
        """
        total = float(self._primitive_convergence_counts.sum())
        if total < 1e-6:
            return {"profile": "insufficient_data", "total_events": 0}
        normalized = self._primitive_convergence_counts / total
        profile = {MAIN_PRIMITIVES[i]: round(float(normalized[i]), 3)
                   for i in range(8) if normalized[i] > 0.01}
        # Sort by strength
        profile = dict(sorted(profile.items(), key=lambda x: -x[1]))
        return {
            "profile": profile,
            "total_convergence_events": self._total_convergence_events,
            "total_checks": self._total_checks,
            "convergence_rate": round(self._total_convergence_events /
                                       max(1, self._total_checks), 3),
            "learned_weight": round(self._learned_weight, 4),
            "strong_events": len(self._convergence_history),
        }

    def to_dict(self) -> dict:
        return {
            "learned_weight": self._learned_weight,
            "total_convergence_events": self._total_convergence_events,
            "total_checks": self._total_checks,
            "primitive_convergence_counts": self._primitive_convergence_counts.tolist(),
            "convergence_history": self._convergence_history[-20:],
        }

    def from_dict(self, d: dict) -> None:
        self._learned_weight = d.get("learned_weight", 0.3)
        self._total_convergence_events = d.get("total_convergence_events", 0)
        self._total_checks = d.get("total_checks", 0)
        counts = d.get("primitive_convergence_counts")
        if counts and len(counts) == 8:
            self._primitive_convergence_counts = np.array(counts, dtype=np.float32)
        self._convergence_history = d.get("convergence_history", [])
        if self._total_convergence_events > 0:
            logger.info("[INTUITION] Restored: %d convergences, weight=%.3f, "
                        "profile=%s",
                        self._total_convergence_events, self._learned_weight,
                        self.get_convergence_profile().get("profile", {}))


def compute_intuition_reward_shaping(action_taken: int,
                                      intuition_bias: np.ndarray | None,
                                      outcome_score: float,
                                      config: dict | None = None) -> float:
    """M13: IQL reward shaping based on intuition alignment.

    Asymmetric learning:
    - Trust intuition + succeed = bonus (learn to listen)
    - Ignore intuition + fail = penalty (learn consequence)
    - Ignore intuition + succeed = neutral (valid override)
    - Trust intuition + fail = neutral (don't reinforce bad intuition)
    """
    if intuition_bias is None:
        return 0.0

    cfg = config or {}
    trust_bonus = cfg.get("trust_bonus", 0.05)
    ignore_penalty = cfg.get("ignore_penalty", 0.03)
    success_threshold = cfg.get("success_threshold", 0.5)
    failure_threshold = cfg.get("failure_threshold", 0.3)

    # What did intuition suggest?
    intuition_prim = int(np.argmax(intuition_bias))
    # Did reasoning follow intuition?
    aligned = (action_taken == intuition_prim)

    if aligned and outcome_score > success_threshold:
        return trust_bonus   # Trusted AND succeeded
    elif not aligned and outcome_score < failure_threshold:
        return -ignore_penalty  # Ignored AND failed
    return 0.0  # All other cases: neutral
