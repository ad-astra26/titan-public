"""
Self-Exploration Mini-Reasoner — Titan's introspective cognitive module.

Primitives:
  INTROSPECT:    Compare current felt-state vs recent EMA → state description
  COMPARE_SELF:  Evaluate neuromod balance vs known productive states → alignment
  HYPOTHESIZE:   Generate hypothesis about what improves current state → action proposal

Runs at Spirit rate (70.47 Hz computation gate). Slow and reflective.
Feeds self-understanding to main reasoning via MiniReasonerRegistry.query("self_exploration").

Critical: This module serves TITAN'S OWN self-understanding, not just outer interactions.
"""
import logging
import numpy as np

from .mini_experience import MiniReasoner

logger = logging.getLogger(__name__)


class SelfExplorationMiniReasoner(MiniReasoner):
    domain = "self_exploration"
    primitives = ["INTROSPECT", "COMPARE_SELF", "HYPOTHESIZE"]
    rate_tier = "spirit"
    observation_dim = 20  # Inner state changes + autonomy signals

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_ema = None
        self._productive_profile = None  # Learned "good state" profile
        self._hypothesis_outcomes = []   # Track hypothesis success

    def perceive(self, context: dict) -> np.ndarray:
        """Build 20D from inner state + autonomy signals."""
        parts = []

        # Chi state (5D)
        chi = context.get("chi_state", {})
        parts.append(chi.get("total", 0.5) if isinstance(chi.get("total"), (int, float)) else 0.5)
        parts.append(chi.get("circulation", 0.5) if isinstance(chi.get("circulation"), (int, float)) else 0.5)
        s_eff = chi.get("spirit", {})
        parts.append(s_eff.get("effective", 0.5) if isinstance(s_eff, dict) else 0.5)
        m_eff = chi.get("mind", {})
        parts.append(m_eff.get("effective", 0.5) if isinstance(m_eff, dict) else 0.5)
        b_eff = chi.get("body", {})
        parts.append(b_eff.get("effective", 0.5) if isinstance(b_eff, dict) else 0.5)

        # Neuromod deviation (3D)
        parts.append(min(1.0, context.get("neuromod_deviation", 0.0)))
        nm = context.get("neuromod_levels", {})
        da = nm.get("DA", 0.5)
        gaba = nm.get("GABA", 0.3)
        parts.append(da)
        parts.append(gaba)

        # Activity metrics (5D)
        parts.append(min(1.0, context.get("expression_fire_rate", 0.0)))
        parts.append(min(1.0, context.get("reasoning_commit_rate", 0.0)))
        parts.append(context.get("fatigue", 0.0))
        parts.append(min(1.0, context.get("drift_magnitude", 0.0)))
        parts.append(context.get("vocabulary_confidence", 0.0))

        # Consciousness quality (4D)
        parts.append(min(1.0, context.get("epoch_curvature", 0.0) / 3.14))
        parts.append(context.get("epoch_density", 0.0))
        parts.append(context.get("spirit_coherence", 0.5))
        parts.append(context.get("mind_coherence", 0.5))

        # Autonomy (3D)
        parts.append(min(1.0, context.get("self_explore_count", 0) / 10.0))
        parts.append(min(1.0, context.get("kin_resonance", 0.0)))
        parts.append(1.0 if context.get("is_dreaming", False) else 0.0)

        arr = np.array(parts[:self.observation_dim], dtype=np.float64)
        if len(arr) < self.observation_dim:
            arr = np.concatenate([arr, np.zeros(self.observation_dim - len(arr))])
        return arr

    def execute_primitive(self, primitive_idx: int, observation: np.ndarray) -> dict:
        if primitive_idx == 0:
            return self._introspect(observation)
        elif primitive_idx == 1:
            return self._compare_self(observation)
        else:
            return self._hypothesize(observation)

    def _introspect(self, obs: np.ndarray) -> dict:
        """Compare current state vs recent EMA → change description."""
        if self._state_ema is None:
            self._state_ema = obs.copy()
            return {"relevance": 0.2, "confidence": 0.3, "change": "initializing"}

        alpha = 0.05  # Slow EMA — spirit rate is reflective
        self._state_ema = self._state_ema * (1 - alpha) + obs * alpha
        delta = obs - self._state_ema
        delta_mag = float(np.linalg.norm(delta))
        max_change_dim = int(np.argmax(np.abs(delta)))

        # Interpret the change
        dim_labels = [
            "chi_total", "chi_circ", "chi_spirit", "chi_mind", "chi_body",
            "neuromod_dev", "DA", "GABA",
            "expression_rate", "reasoning_commit", "fatigue", "drift", "vocab_conf",
            "curvature", "density", "spirit_coh", "mind_coh",
            "self_explore", "kin_resonance", "dreaming",
        ]
        changing = dim_labels[max_change_dim] if max_change_dim < len(dim_labels) else f"dim_{max_change_dim}"
        direction = "rising" if delta[max_change_dim] > 0 else "falling"

        return {
            "relevance": min(1.0, delta_mag * 3),
            "confidence": 0.4 + min(0.5, self._total_ticks / 100.0),
            "change": f"{changing}_{direction}",
            "change_magnitude": round(delta_mag, 3),
            "state_stability": round(max(0, 1.0 - delta_mag * 2), 3),
        }

    def _compare_self(self, obs: np.ndarray) -> dict:
        """Evaluate current state vs known productive profile."""
        # Build productive profile from accumulated experience
        if self._productive_profile is None:
            # Default "productive" state: moderate chi, low fatigue, balanced neuromods
            self._productive_profile = np.array([
                0.6, 0.5, 0.5, 0.5, 0.5,   # chi: moderate
                0.1, 0.5, 0.3,               # neuromods: balanced
                0.3, 0.4, 0.2, 0.3, 0.5,    # activity: moderate
                0.5, 0.1, 0.6, 0.6,          # consciousness: coherent
                0.3, 0.5, 0.0,               # autonomy: engaged
            ][:self.observation_dim])
            if len(self._productive_profile) < self.observation_dim:
                self._productive_profile = np.concatenate([
                    self._productive_profile,
                    np.full(self.observation_dim - len(self._productive_profile), 0.5)
                ])

        diff = np.abs(obs - self._productive_profile)
        alignment = 1.0 - float(np.mean(diff))
        worst_dim = int(np.argmax(diff))

        dim_labels = [
            "chi_total", "chi_circ", "chi_spirit", "chi_mind", "chi_body",
            "neuromod_dev", "DA", "GABA",
            "expression_rate", "reasoning_commit", "fatigue", "drift", "vocab_conf",
            "curvature", "density", "spirit_coh", "mind_coh",
            "self_explore", "kin_resonance", "dreaming",
        ]
        worst_label = dim_labels[worst_dim] if worst_dim < len(dim_labels) else f"dim_{worst_dim}"

        return {
            "relevance": max(0.2, 1.0 - alignment),
            "confidence": 0.4 + min(0.4, self._total_ticks / 200.0),
            "alignment_score": round(alignment, 3),
            "weakest_dimension": worst_label,
            "weakest_gap": round(float(diff[worst_dim]), 3),
        }

    def _hypothesize(self, obs: np.ndarray) -> dict:
        """Generate hypothesis about what action would improve state."""
        fatigue = obs[10] if len(obs) > 10 else 0.0
        da = obs[6] if len(obs) > 6 else 0.5
        gaba = obs[7] if len(obs) > 7 else 0.3
        expression_rate = obs[8] if len(obs) > 8 else 0.0
        reasoning_commit = obs[9] if len(obs) > 9 else 0.0
        chi_circ = obs[1] if len(obs) > 1 else 0.5

        # Simple rule-based hypothesis generation (learned policy selects WHEN to fire this)
        if fatigue > 0.6:
            hypothesis = "rest_needed"
            action = "reduce_activity"
            confidence = 0.6 + fatigue * 0.3
        elif da < 0.3:
            hypothesis = "low_motivation"
            action = "seek_novelty"
            confidence = 0.5
        elif reasoning_commit < 0.2 and expression_rate > 0.5:
            hypothesis = "expression_without_thought"
            action = "deepen_reasoning"
            confidence = 0.5
        elif chi_circ < 0.2:
            hypothesis = "energy_stagnant"
            action = "increase_activity"
            confidence = 0.5
        elif gaba > 0.6:
            hypothesis = "over_inhibited"
            action = "explore_freely"
            confidence = 0.5
        else:
            hypothesis = "balanced"
            action = "continue_current"
            confidence = 0.7

        return {
            "relevance": max(0.2, 1.0 - confidence) if hypothesis != "balanced" else 0.1,
            "confidence": round(confidence, 3),
            "hypothesis": hypothesis,
            "suggested_action": action,
        }

    def format_summary(self) -> dict:
        s = dict(self._latest_summary)
        s.setdefault("relevance", 0.0)
        s.setdefault("confidence", 0.3)
        s.setdefault("primitive", self.primitives[0])
        return s
