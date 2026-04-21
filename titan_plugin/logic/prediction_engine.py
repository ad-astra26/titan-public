"""
titan_plugin/logic/prediction_engine.py — Predictive Processing Engine.

The brain CONSTANTLY predicts. Every perception is compared against
prediction. The difference (prediction error) drives learning and attention.

A system that EXPECTS and is SURPRISED is fundamentally different
from one that merely reacts.

Titan predicts his next consciousness state from current state + trajectory.
When the actual state arrives, surprise = how wrong the prediction was.
High surprise → CURIOSITY (novelty). Low surprise → confidence (familiarity).

v2 (2026-04-21, rFP_prediction_engine_v2):
  - Persistence: _errors + EMAs saved to data/prediction/novelty_state.json
    → no more 20-epoch novelty blackout after restart.
  - Adaptive surprise threshold: rolling top-quartile of recent errors
    (not hardcoded 0.3). Tracks live distribution; well-learned trajectories
    stop firing surprise while genuinely novel ones always do.
  - Z-score calibration: novelty signal = sigmoid of z-scored error against
    EMA mean/std. Stable [0, 1] range regardless of trajectory variance.
  - Cold-boot neutral default (0.5, configurable) instead of 0.0.
"""
import json
import logging
import math
import os
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


def _load_cfg() -> dict:
    """Load [prediction_engine] TOML section. Defaults if missing."""
    try:
        from titan_plugin.params import get_params
        cfg = get_params("prediction_engine") or {}
    except Exception:
        cfg = {}
    return {
        "ema_alpha": float(cfg.get("ema_alpha", 0.005)),
        "neutral_default": float(cfg.get("neutral_default", 0.5)),
        "surprise_threshold_floor": float(cfg.get("surprise_threshold_floor", 0.1)),
        "surprise_threshold_bootstrap": float(cfg.get("surprise_threshold_bootstrap", 0.3)),
        "save_every_evals": int(cfg.get("save_every_evals", 100)),
        "error_window": int(cfg.get("error_window", 50)),
    }


class PredictionEngine:
    """Predicts next consciousness state and measures surprise.

    After each epoch:
      1. predict_next(state, trajectory) → stores prediction
      2. Next epoch: compute_error(actual_state) → surprise score (also updates EMAs + novelty_ema)
      3. Surprise feeds CURIOSITY (novelty) and VIGILANCE (threat detection)
      4. Familiarity feeds confidence and stability signals

    Persistence: state saved to data/prediction/novelty_state.json every
    `save_every_evals` calls to compute_error. Loaded on __init__. No more
    20-epoch novelty blackout after restart.
    """

    def __init__(
        self,
        error_window: Optional[int] = None,
        data_dir: str = "data/prediction",
        load_state: bool = True,
    ):
        """
        Args:
            error_window: Number of recent errors to keep for rolling average.
                          If None, uses TOML `[prediction_engine].error_window` (default 50).
            data_dir: Directory for persisted novelty_state.json.
            load_state: If True, attempt to load state from disk on boot.
        """
        cfg = _load_cfg()
        self._ema_alpha = cfg["ema_alpha"]
        self._neutral_default = cfg["neutral_default"]
        self._threshold_floor = cfg["surprise_threshold_floor"]
        self._threshold_bootstrap = cfg["surprise_threshold_bootstrap"]
        self._save_every = cfg["save_every_evals"]
        window = error_window if error_window is not None else cfg["error_window"]

        self._last_prediction: Optional[list] = None
        self._errors: deque = deque(maxlen=window)
        self._total_predictions: int = 0
        self._total_surprises: int = 0

        # EMA of raw cosine-distance errors (for z-score calibration)
        self._error_mean_ema: float = 0.0
        self._error_std_ema: float = 0.01
        # EMA of novelty signal output (consumer-facing, in [0, 1])
        # spirit_worker.py reads this via getattr(prediction_engine, '_novelty_ema', None)
        # to populate META-CGN's self_prediction_accuracy subsystem signal.
        self._novelty_ema: float = self._neutral_default

        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "novelty_state.json")

        if load_state:
            self._try_load_state()

    # ── Core prediction API (unchanged from v1) ──

    def predict_next(self, current_state: list, trajectory: list,
                     dt: float = 1.0) -> list:
        """Predict next state: predicted = current + trajectory × dt."""
        if not current_state:
            self._last_prediction = None
            return []

        traj = trajectory if trajectory and len(trajectory) == len(current_state) else [0.0] * len(current_state)
        predicted = [s + t * dt for s, t in zip(current_state, traj)]
        self._last_prediction = predicted
        self._total_predictions += 1
        return predicted

    def compute_error(self, actual_state: list) -> float:
        """Compute prediction error, update EMAs, and count surprises adaptively.

        Returns:
            Surprise score: raw cosine distance (0 = perfect, 1 = orthogonal).
        """
        if self._last_prediction is None or not actual_state:
            return 0.0

        error = _cosine_distance(self._last_prediction, actual_state)
        self._errors.append(error)

        # Update EMAs for z-score calibration
        alpha = self._ema_alpha
        prev_mean = self._error_mean_ema
        self._error_mean_ema = (1.0 - alpha) * prev_mean + alpha * error
        residual = abs(error - self._error_mean_ema)
        self._error_std_ema = max(0.001, (1.0 - alpha) * self._error_std_ema + alpha * residual)

        # Adaptive surprise count
        if error > self.get_surprise_threshold():
            self._total_surprises += 1

        # Update consumer-facing novelty EMA (post-calibration signal)
        nov = self.get_novelty_signal()
        self._novelty_ema = (1.0 - alpha) * self._novelty_ema + alpha * nov

        # Periodic save
        if self._total_predictions > 0 and self._total_predictions % self._save_every == 0:
            self._save_state()

        return error

    # ── Signal readouts ──

    def get_novelty_signal(self) -> float:
        """Z-score-calibrated novelty signal in [0, 1].

        Returns:
            Sigmoid(z-score of rolling-mean error). 0.5 on cold-boot when no
            errors accumulated yet (configurable via neutral_default).
        """
        if not self._errors:
            return self._neutral_default
        raw = sum(self._errors) / len(self._errors)
        z = (raw - self._error_mean_ema) / max(self._error_std_ema, 0.01)
        return 1.0 / (1.0 + math.exp(-z))

    def get_familiarity(self) -> float:
        """Inverse of novelty — how predictable is the environment."""
        return max(0.0, 1.0 - self.get_novelty_signal())

    def get_surprise_threshold(self) -> float:
        """Adaptive surprise threshold: top-quartile of recent errors.

        Returns bootstrap default until ≥10 errors accumulate. Floor at
        threshold_floor prevents collapse to 0 on static trajectories.
        """
        if len(self._errors) < 10:
            return self._threshold_bootstrap
        sorted_errs = sorted(self._errors)
        p75_idx = int(len(sorted_errs) * 0.75)
        p75_idx = min(p75_idx, len(sorted_errs) - 1)
        return max(self._threshold_floor, sorted_errs[p75_idx])

    def get_stats(self) -> dict:
        """Prediction engine statistics for API."""
        return {
            "total_predictions": self._total_predictions,
            "total_surprises": self._total_surprises,
            "novelty": round(self.get_novelty_signal(), 4),
            "novelty_ema": round(self._novelty_ema, 4),
            "familiarity": round(self.get_familiarity(), 4),
            "surprise_threshold": round(self.get_surprise_threshold(), 4),
            "error_mean_ema": round(self._error_mean_ema, 4),
            "error_std_ema": round(self._error_std_ema, 4),
            "recent_errors": [round(e, 4) for e in list(self._errors)[-5:]],
        }

    # ── Persistence ──

    def get_state(self) -> dict:
        """Return serializable state dict (schema v1)."""
        return {
            "schema": 1,
            "errors": list(self._errors),
            "error_mean_ema": self._error_mean_ema,
            "error_std_ema": self._error_std_ema,
            "novelty_ema": self._novelty_ema,
            "ema_alpha": self._ema_alpha,
            "total_predictions": self._total_predictions,
            "total_surprises": self._total_surprises,
            "last_saved_epoch": self._total_predictions,
        }

    def _save_state(self) -> None:
        """Atomic tmp+rename save with corruption-guard pre-flight encode."""
        try:
            state = self.get_state()
            payload = json.dumps(state)  # pre-flight encode — catches serialization errors
            os.makedirs(self._data_dir, exist_ok=True)
            tmp_path = self._state_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(payload)
            os.replace(tmp_path, self._state_path)
        except Exception as e:
            logger.warning("[PredictionEngine] Save failed (non-fatal): %s", e)

    def _try_load_state(self) -> None:
        """Load state from disk if present; fall through to defaults on any error."""
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r") as f:
                state = json.load(f)
            if not isinstance(state, dict) or state.get("schema") != 1:
                logger.warning("[PredictionEngine] State schema mismatch; initializing defaults")
                return
            loaded_errors = state.get("errors", [])
            if isinstance(loaded_errors, list):
                filtered = [float(e) for e in loaded_errors if isinstance(e, (int, float))]
                self._errors = deque(filtered, maxlen=self._errors.maxlen)
            self._error_mean_ema = float(state.get("error_mean_ema", 0.0))
            self._error_std_ema = max(0.001, float(state.get("error_std_ema", 0.01)))
            self._novelty_ema = float(state.get("novelty_ema", self._neutral_default))
            self._total_predictions = int(state.get("total_predictions", 0))
            self._total_surprises = int(state.get("total_surprises", 0))
            logger.info(
                "[PredictionEngine] Loaded state: errors=%d predictions=%d surprises=%d "
                "mean_ema=%.4f std_ema=%.4f novelty_ema=%.4f",
                len(self._errors),
                self._total_predictions,
                self._total_surprises,
                self._error_mean_ema,
                self._error_std_ema,
                self._novelty_ema,
            )
        except Exception as e:
            logger.warning("[PredictionEngine] Load failed, initializing defaults: %s", e)
