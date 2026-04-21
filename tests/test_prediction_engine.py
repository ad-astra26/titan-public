"""Tests for Prediction Engine — predictive processing and surprise detection."""
import pytest
from titan_plugin.logic.prediction_engine import PredictionEngine, _cosine_distance


class TestPredictionEngine:

    def test_predict_linear(self):
        pe = PredictionEngine()
        predicted = pe.predict_next([1.0, 0.0, 0.0], [0.1, 0.0, 0.0])
        assert predicted == pytest.approx([1.1, 0.0, 0.0])

    def test_predict_with_dt(self):
        pe = PredictionEngine()
        predicted = pe.predict_next([1.0, 0.0], [0.1, 0.2], dt=2.0)
        assert predicted == pytest.approx([1.2, 0.4])

    def test_error_identical_states(self):
        pe = PredictionEngine()
        pe.predict_next([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        error = pe.compute_error([1.0, 0.0, 0.0])
        assert error == pytest.approx(0.0, abs=0.01)

    def test_error_different_states(self):
        pe = PredictionEngine()
        pe.predict_next([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        error = pe.compute_error([0.0, 1.0, 0.0])
        assert error > 0.5  # Very different

    def test_error_no_prediction(self):
        pe = PredictionEngine()
        assert pe.compute_error([1.0, 0.0]) == 0.0

    def test_novelty_rolling_average(self):
        # v2 semantics: novelty is z-score + sigmoid of rolling error mean against
        # EMA mean/std. Always in [0, 1]; exact midpoint depends on EMA state.
        pe = PredictionEngine(error_window=3)
        pe.predict_next([1.0, 0.0], [0.0, 0.0])
        pe.compute_error([1.0, 0.0])  # error ≈ 0
        pe.predict_next([1.0, 0.0], [0.0, 0.0])
        pe.compute_error([0.0, 1.0])  # error > 0.5
        novelty = pe.get_novelty_signal()
        assert 0.0 <= novelty <= 1.0

    def test_familiarity_inverse(self):
        # v2 semantics: with no errors, get_novelty_signal() returns the configured
        # neutral_default (0.5) → familiarity = 0.5. familiarity = 1 - novelty always.
        pe = PredictionEngine()
        assert pe.get_familiarity() == pytest.approx(1.0 - pe.get_novelty_signal())

    def test_stats_complete(self):
        pe = PredictionEngine()
        pe.predict_next([1.0], [0.1])
        pe.compute_error([1.1])
        stats = pe.get_stats()
        assert "total_predictions" in stats
        assert "novelty" in stats
        assert "familiarity" in stats
        assert stats["total_predictions"] == 1

    def test_surprise_counter(self):
        pe = PredictionEngine()
        pe.predict_next([1.0, 0.0], [0.0, 0.0])
        pe.compute_error([0.0, 1.0])  # High surprise
        assert pe._total_surprises == 1

    def test_empty_state(self):
        pe = PredictionEngine()
        result = pe.predict_next([], [])
        assert result == []
        assert pe._last_prediction is None
