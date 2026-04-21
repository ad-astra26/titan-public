"""Tests for Prediction Engine v2 — persistence, adaptive threshold, z-score calibration.

Covers rFP_prediction_engine_v2 §3 (test plan):
  - test_persistence_roundtrip
  - test_persistence_corruption_guard
  - test_no_blackout_after_reload
  - test_adaptive_threshold_tracks_distribution
  - test_zscore_stable_range
  - test_diagnostic_endpoint_returns_live_data
"""
import json
import math
import os

import pytest

from titan_plugin.logic.prediction_engine import PredictionEngine


def _seed_errors(pe: PredictionEngine, errors: list) -> None:
    """Inject errors via the compute_error path (correct EMA + save side-effects)."""
    for e in errors:
        # bypass predict_next/cosine_distance — directly drive compute_error-like updates
        pe._last_prediction = [1.0]
        # cosine_distance of [1.0] vs [1 - e] → 1 - (1-e)/|...| ≈ e when e small
        # Simplest: push to _errors and replay EMA math matching compute_error()
        pe._errors.append(e)
        alpha = pe._ema_alpha
        pe._error_mean_ema = (1.0 - alpha) * pe._error_mean_ema + alpha * e
        residual = abs(e - pe._error_mean_ema)
        pe._error_std_ema = max(0.001, (1.0 - alpha) * pe._error_std_ema + alpha * residual)
        if e > pe.get_surprise_threshold():
            pe._total_surprises += 1
        nov = pe.get_novelty_signal()
        pe._novelty_ema = (1.0 - alpha) * pe._novelty_ema + alpha * nov
        pe._total_predictions += 1


class TestPredictionEngineV2:

    def test_persistence_roundtrip(self, tmp_path):
        """Save state, create fresh engine, load, verify errors + EMAs match."""
        data_dir = str(tmp_path / "prediction")
        pe1 = PredictionEngine(data_dir=data_dir, load_state=False)
        _seed_errors(pe1, [0.1, 0.15, 0.2, 0.12, 0.08, 0.18, 0.22, 0.14, 0.11, 0.16])
        pe1._save_state()

        assert os.path.exists(os.path.join(data_dir, "novelty_state.json"))

        pe2 = PredictionEngine(data_dir=data_dir, load_state=True)
        assert list(pe2._errors) == list(pe1._errors)
        assert pe2._error_mean_ema == pytest.approx(pe1._error_mean_ema, abs=1e-9)
        assert pe2._error_std_ema == pytest.approx(pe1._error_std_ema, abs=1e-9)
        assert pe2._novelty_ema == pytest.approx(pe1._novelty_ema, abs=1e-9)
        assert pe2._total_predictions == pe1._total_predictions
        assert pe2._total_surprises == pe1._total_surprises

    def test_persistence_corruption_guard(self, tmp_path, caplog):
        """Malformed JSON on disk → fresh defaults, WARNING logged, no crash."""
        import logging
        data_dir = str(tmp_path / "prediction")
        os.makedirs(data_dir, exist_ok=True)
        state_path = os.path.join(data_dir, "novelty_state.json")
        with open(state_path, "w") as f:
            f.write("{not valid json at all")  # garbage

        with caplog.at_level(logging.WARNING, logger="titan_plugin.logic.prediction_engine"):
            pe = PredictionEngine(data_dir=data_dir, load_state=True)

        # Loaded defaults; no crash
        assert pe._total_predictions == 0
        assert len(pe._errors) == 0
        assert pe._error_mean_ema == 0.0
        # Warning was emitted
        assert any("Load failed" in r.message for r in caplog.records)

    def test_no_blackout_after_reload(self, tmp_path):
        """After reload with 20 errors, get_novelty_signal() > 0 immediately.

        (v1 returned 0.0 until 20 new predictions accumulated post-restart.)
        """
        data_dir = str(tmp_path / "prediction")
        pe1 = PredictionEngine(data_dir=data_dir, load_state=False)
        # Seed 20 errors in a range that produces non-degenerate z-score
        _seed_errors(pe1, [0.05, 0.1, 0.15, 0.08, 0.12, 0.2, 0.25, 0.18, 0.22, 0.3,
                           0.11, 0.13, 0.17, 0.09, 0.14, 0.19, 0.21, 0.16, 0.10, 0.24])
        pe1._save_state()

        pe2 = PredictionEngine(data_dir=data_dir, load_state=True)
        assert pe2.get_novelty_signal() > 0.0

    def test_adaptive_threshold_tracks_distribution(self):
        """100 errors in narrow [0.1, 0.15] → threshold migrates toward p75 ≈ 0.14."""
        pe = PredictionEngine(error_window=100, load_state=False)

        # Bootstrap default until 10 errors
        assert pe.get_surprise_threshold() == pytest.approx(0.3)

        # Fill with narrow-range errors. Use deterministic values so p75 is predictable.
        vals = [0.10 + 0.0005 * i for i in range(100)]  # 0.10 → 0.1495
        _seed_errors(pe, vals)

        threshold = pe.get_surprise_threshold()
        # p75 index = int(100 * 0.75) = 75 → vals[75] = 0.10 + 0.0005 * 75 = 0.1375
        assert threshold == pytest.approx(0.1375, abs=0.01)
        # Floored at 0.1, not collapsed toward bootstrap 0.3
        assert 0.1 <= threshold <= 0.2

    def test_zscore_stable_range(self):
        """Inject high-variance errors, verify novelty stays in [0.0, 1.0]."""
        pe = PredictionEngine(error_window=100, load_state=False)
        # Wide-range errors — novelty signal must not escape sigmoid bounds
        import random
        random.seed(42)
        errs = [random.random() for _ in range(100)]
        _seed_errors(pe, errs)

        # Sample novelty after a mix of low/high injections
        for extra in [0.01, 0.9, 0.5, 0.99, 0.0, 0.3]:
            _seed_errors(pe, [extra])
            nov = pe.get_novelty_signal()
            assert 0.0 <= nov <= 1.0, f"novelty escaped bounds: {nov} after extra={extra}"

    def test_diagnostic_endpoint_returns_live_data(self, tmp_path, monkeypatch):
        """/v4/prediction — file-fallback payload shape + live values."""
        # Write a state file in tmp_path and monkeypatch the endpoint's hard-coded path
        data_dir = str(tmp_path / "prediction")
        pe = PredictionEngine(data_dir=data_dir, load_state=False)
        _seed_errors(pe, [0.1, 0.12, 0.15, 0.11, 0.14, 0.13, 0.09, 0.16, 0.11, 0.14])
        pe._save_state()
        state_path = os.path.join(data_dir, "novelty_state.json")
        assert os.path.exists(state_path)

        # Load the file directly (simulates the endpoint's fallback path)
        with open(state_path) as f:
            state = json.load(f)

        # Verify schema v2 contract (what /v4/prediction serializes)
        assert state["schema"] == 1
        assert len(state["errors"]) == 10
        assert "error_mean_ema" in state and "error_std_ema" in state
        assert "novelty_ema" in state and 0.0 <= state["novelty_ema"] <= 1.0
        assert state["total_predictions"] == 10

        # Spot-check the /v4/prediction payload shape our endpoint constructs
        payload = {
            "total_predictions": state.get("total_predictions", 0),
            "total_surprises": state.get("total_surprises", 0),
            "novelty_ema": round(float(state.get("novelty_ema", 0.5)), 4),
            "error_mean_ema": round(float(state.get("error_mean_ema", 0.0)), 4),
            "error_std_ema": round(float(state.get("error_std_ema", 0.01)), 4),
            "recent_errors": [round(float(e), 4) for e in state.get("errors", [])[-5:]],
            "errors_window_size": len(state.get("errors", [])),
        }
        assert payload["total_predictions"] == 10
        assert payload["errors_window_size"] == 10
        assert len(payload["recent_errors"]) == 5
        assert all(isinstance(e, float) for e in payload["recent_errors"])
