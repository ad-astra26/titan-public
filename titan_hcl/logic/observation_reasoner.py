"""
Observation Mini-Reasoner — monitors neurochemical state and consciousness drift.

Primitives:
  COMPARE_STATE:  Compare neuromod levels vs setpoints → deviation map
  DETECT_ANOMALY: Flag dimensions deviating >2σ from EMA → anomaly alert
  TRACK_DRIFT:    Monitor 132D consciousness drift trend → acceleration

Runs at Body rate. Feeds structured observations to main reasoning
via MiniReasonerRegistry.query("observation").
"""
import logging
import numpy as np

from .mini_experience import MiniReasoner

logger = logging.getLogger(__name__)


class ObservationMiniReasoner(MiniReasoner):
    domain = "observation"
    primitives = ["COMPARE_STATE", "DETECT_ANOMALY", "TRACK_DRIFT"]
    rate_tier = "body"
    observation_dim = 25  # T5 (12D neurochemical) + T6 (12D dynamics) + 1D drift

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_ema = None
        self._state_var = None  # Running variance for anomaly detection
        self._drift_history = []

    def perceive(self, context: dict) -> np.ndarray:
        """Build 25D from neurochemical + dynamics + drift."""
        parts = []
        # Neurochemical (12D): 6 neuromod levels + 6 chi components
        nm = context.get("neuromod_levels", {})
        for name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
            parts.append(nm.get(name, 0.5))
        chi = context.get("chi_state", {})
        parts.append(chi.get("total", 0.5))
        parts.append(chi.get("circulation", 0.5))
        parts.append(chi.get("spirit", {}).get("effective", 0.5) if isinstance(chi.get("spirit"), dict) else 0.5)
        parts.append(chi.get("mind", {}).get("effective", 0.5) if isinstance(chi.get("mind"), dict) else 0.5)
        parts.append(chi.get("body", {}).get("effective", 0.5) if isinstance(chi.get("body"), dict) else 0.5)
        parts.append(chi.get("state", "BALANCED") == "FLOWING" and 1.0 or 0.0)

        # Dynamics (12D)
        parts.append(context.get("metabolic_drain", 0.0))
        parts.append(context.get("sleep_drive", 0.0))
        parts.append(context.get("wake_drive", 0.5))
        parts.append(context.get("fatigue", 0.0))
        parts.append(context.get("expression_fire_rate", 0.0))
        parts.append(1.0 if context.get("teacher_active", False) else 0.0)
        parts.append(context.get("vocabulary_confidence", 0.0))
        parts.append(min(1.0, context.get("time_since_dream", 0) / 600.0))
        parts.append(context.get("reasoning_active", 0.0))
        parts.append(context.get("reasoning_chain_length", 0.0))
        parts.append(context.get("reasoning_confidence", 0.0))
        parts.append(context.get("reasoning_gut_agreement", 0.0))

        # Drift magnitude (1D)
        parts.append(min(1.0, context.get("drift_magnitude", 0.0)))

        arr = np.array(parts[:self.observation_dim], dtype=np.float64)
        if len(arr) < self.observation_dim:
            arr = np.concatenate([arr, np.zeros(self.observation_dim - len(arr))])
        return arr

    def execute_primitive(self, primitive_idx: int, observation: np.ndarray) -> dict:
        if primitive_idx == 0:
            return self._compare_state(observation)
        elif primitive_idx == 1:
            return self._detect_anomaly(observation)
        else:
            return self._track_drift(observation)

    def _compare_state(self, obs: np.ndarray) -> dict:
        """Compare neuromod levels (0:6) vs typical setpoints."""
        setpoints = np.array([0.5, 0.7, 0.7, 0.66, 0.7, 0.3])  # DA, 5HT, NE, ACh, Endo, GABA
        levels = obs[:6]
        deviations = levels - setpoints
        abs_dev = np.abs(deviations)
        max_dev_idx = int(np.argmax(abs_dev))
        names = ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]
        return {
            "relevance": float(np.max(abs_dev)),
            "confidence": max(0.4, 1.0 - float(np.mean(abs_dev))),
            "max_deviation": names[max_dev_idx],
            "deviation_value": round(float(deviations[max_dev_idx]), 3),
            "overall_homeostasis": round(1.0 - float(np.mean(abs_dev)), 3),
        }

    def _detect_anomaly(self, obs: np.ndarray) -> dict:
        """Flag dimensions deviating >2σ from running EMA."""
        if self._state_ema is None:
            self._state_ema = obs.copy()
            self._state_var = np.ones_like(obs) * 0.01
            return {"relevance": 0.0, "confidence": 0.3, "anomaly_detected": False}

        # Update EMA and variance
        alpha = 0.05
        self._state_ema = self._state_ema * (1 - alpha) + obs * alpha
        diff_sq = (obs - self._state_ema) ** 2
        self._state_var = self._state_var * (1 - alpha) + diff_sq * alpha

        # Detect anomalies (>2σ)
        std = np.sqrt(self._state_var + 1e-10)
        z_scores = np.abs(obs - self._state_ema) / std
        anomaly_mask = z_scores > 2.0
        n_anomalies = int(np.sum(anomaly_mask))
        max_z_dim = int(np.argmax(z_scores))

        return {
            "relevance": min(1.0, float(np.max(z_scores)) / 3.0),
            "confidence": 0.5 + min(0.4, n_anomalies * 0.1),
            "anomaly_detected": n_anomalies > 0,
            "anomaly_count": n_anomalies,
            "max_z_score_dim": max_z_dim,
            "max_z_score": round(float(z_scores[max_z_dim]), 2),
        }

    def _track_drift(self, obs: np.ndarray) -> dict:
        """Monitor consciousness drift magnitude trend."""
        drift_val = float(obs[-1]) if len(obs) >= self.observation_dim else 0.0
        self._drift_history.append(drift_val)
        if len(self._drift_history) > 20:
            self._drift_history = self._drift_history[-20:]

        if len(self._drift_history) < 3:
            return {"relevance": drift_val, "confidence": 0.3, "drift_trend": "stable"}

        recent = self._drift_history[-5:]
        earlier = self._drift_history[-10:-5] if len(self._drift_history) >= 10 else self._drift_history[:5]
        recent_avg = float(np.mean(recent))
        earlier_avg = float(np.mean(earlier))
        acceleration = recent_avg - earlier_avg

        if acceleration > 0.05:
            trend = "accelerating"
        elif acceleration < -0.05:
            trend = "decelerating"
        else:
            trend = "stable"

        return {
            "relevance": min(1.0, abs(acceleration) * 5),
            "confidence": 0.5 + min(0.4, len(self._drift_history) / 20.0),
            "drift_trend": trend,
            "drift_current": round(drift_val, 3),
            "drift_acceleration": round(acceleration, 4),
        }

    def format_summary(self) -> dict:
        s = dict(self._latest_summary)
        s.setdefault("relevance", 0.0)
        s.setdefault("confidence", 0.3)
        s.setdefault("primitive", self.primitives[0])
        return s
