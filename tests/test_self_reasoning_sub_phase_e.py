"""tests/test_self_reasoning_sub_phase_e.py — L2 housekeeping closure.

Sub-phase E from `finished/rFP_tuning_012_compound_rewards_v2.md` —
INTROSPECT signal grounding (self_prediction_accuracy +
self_profile_divergence). Closes the long-standing
`meta_reasoning_rewards.py:317` stub note "They'll read 0.0 in this
session" by giving the dream-end consolidation a real value to push.
"""
from __future__ import annotations

import pytest

from titan_hcl.logic.self_reasoning import (
    SelfProfile,
    SelfReasoningEngine,
    _profile_divergence,
)


class TestProfileDivergence:
    def _baseline_profile(self, **overrides) -> SelfProfile:
        defaults = dict(
            epoch=100,
            vocab_total=500,
            vocab_productive=200,
            neuromod_levels={"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                             "ACh": 0.5, "GABA": 0.5, "Endorphin": 0.5},
            i_confidence=0.5,
            i_depth=0.3,
            chi_coherence=0.6,
            concept_confidences={"I": 0.5, "YOU": 0.3, "WE": 0.1,
                                  "YES": 0.4, "NO": 0.2, "THEY": 0.1},
            commit_rate=0.4,
            eureka_count=5,
            wisdom_count=2,
            prediction_accuracy=0.6,
        )
        defaults.update(overrides)
        return SelfProfile(**defaults)

    def test_identical_profiles_zero_divergence(self):
        a = self._baseline_profile()
        b = self._baseline_profile()
        d = _profile_divergence(a, b)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_neuromod_swing_increases_divergence(self):
        prior = self._baseline_profile(
            neuromod_levels={"DA": 0.2, "5HT": 0.8, "NE": 0.3,
                             "ACh": 0.7, "GABA": 0.4, "Endorphin": 0.6})
        current = self._baseline_profile(
            neuromod_levels={"DA": 0.8, "5HT": 0.2, "NE": 0.7,
                             "ACh": 0.3, "GABA": 0.6, "Endorphin": 0.4})
        d = _profile_divergence(prior, current)
        assert d > 0.05, "full neuromod flip should register clear divergence"

    def test_identity_collapse_high_divergence(self):
        # I-confidence collapses from 0.9 to 0.1 + concept confidences
        # invert. This is a major identity shift.
        prior = self._baseline_profile(
            i_confidence=0.9, i_depth=0.8, chi_coherence=0.9,
            concept_confidences={"I": 0.9, "YOU": 0.8, "WE": 0.7,
                                  "YES": 0.7, "NO": 0.1, "THEY": 0.5})
        current = self._baseline_profile(
            i_confidence=0.1, i_depth=0.05, chi_coherence=0.1,
            concept_confidences={"I": 0.05, "YOU": 0.05, "WE": 0.0,
                                  "YES": 0.05, "NO": 0.9, "THEY": 0.1})
        d = _profile_divergence(prior, current)
        assert d > 0.2

    def test_divergence_bounded_unit_interval(self):
        prior = self._baseline_profile()
        # Pathological "fully different" profile — every axis at the
        # opposite extreme.
        current = self._baseline_profile(
            vocab_total=100000, vocab_productive=50000,
            neuromod_levels={k: 1.0 for k in prior.neuromod_levels},
            i_confidence=1.0, i_depth=1.0, chi_coherence=1.0,
            concept_confidences={k: 1.0 for k in prior.concept_confidences},
            commit_rate=1.0, eureka_count=10000, wisdom_count=10000,
            prediction_accuracy=1.0,
        )
        d = _profile_divergence(prior, current)
        assert 0.0 <= d <= 1.0

    def test_partial_field_overlap_uses_what_is_available(self):
        prior = self._baseline_profile(neuromod_levels={"DA": 0.5})
        current = self._baseline_profile(neuromod_levels={"DA": 0.5})
        # Missing modulators default to 0.0 on both sides → contribute 0 delta.
        d = _profile_divergence(prior, current)
        assert d == pytest.approx(0.0, abs=0.001)


class TestComputeIntrospectSignals:
    @pytest.fixture
    def engine(self, tmp_path):
        # Minimal engine — we only need the EMA + _last_profile slots.
        eng = SelfReasoningEngine.__new__(SelfReasoningEngine)
        eng._prediction_accuracy_ema = 0.0
        eng._last_profile = None
        return eng

    def test_cold_start_returns_max_divergence(self, engine):
        # No prior + no current profile → divergence convention 1.0
        # (first introspection is by definition a "new view of self").
        engine._prediction_accuracy_ema = 0.7
        sigs = engine.compute_introspect_signals()
        assert sigs["self_prediction_accuracy"] == pytest.approx(0.7)
        assert sigs["self_profile_divergence"] == 1.0

    def test_prediction_accuracy_clamped(self, engine):
        # Defensive clamp: EMA could drift outside [0,1] from bad input.
        engine._prediction_accuracy_ema = 1.5
        sigs = engine.compute_introspect_signals()
        assert sigs["self_prediction_accuracy"] == 1.0

        engine._prediction_accuracy_ema = -0.3
        sigs = engine.compute_introspect_signals()
        assert sigs["self_prediction_accuracy"] == 0.0

    def test_divergence_from_prior_and_current(self, engine):
        engine._prediction_accuracy_ema = 0.5
        engine._last_profile = SelfProfile(
            i_confidence=0.5, i_depth=0.5, chi_coherence=0.5,
            concept_confidences={"I": 0.5})
        current = SelfProfile(
            i_confidence=0.5, i_depth=0.5, chi_coherence=0.5,
            concept_confidences={"I": 0.5})
        sigs = engine.compute_introspect_signals(current_profile=current)
        # Identical profiles → divergence 0.0
        assert sigs["self_profile_divergence"] == pytest.approx(0.0, abs=0.01)

    def test_no_current_with_prior_returns_zero_divergence(self, engine):
        # When dream-end consolidation didn't build a new profile but a
        # prior exists, signal "no new insight detected" = 0.0 (not 1.0).
        engine._last_profile = SelfProfile(i_confidence=0.5)
        sigs = engine.compute_introspect_signals(current_profile=None)
        assert sigs["self_profile_divergence"] == 0.0

    def test_signals_in_unit_interval(self, engine):
        # Any combination should return values in [0,1].
        engine._prediction_accuracy_ema = 0.42
        engine._last_profile = SelfProfile(
            i_confidence=0.1, neuromod_levels={"DA": 0.1})
        curr = SelfProfile(
            i_confidence=0.9, neuromod_levels={"DA": 0.9})
        sigs = engine.compute_introspect_signals(current_profile=curr)
        assert 0.0 <= sigs["self_prediction_accuracy"] <= 1.0
        assert 0.0 <= sigs["self_profile_divergence"] <= 1.0


class TestConsolidateTrainingProducesIntrospectSignals:
    """Smoke-test: consolidate_training() result dict now carries
    introspect_signals — the wire used by self_reflection_worker."""

    def test_consolidate_training_includes_introspect_signals(self, tmp_path):
        import sqlite3

        db_path = str(tmp_path / "self_reasoning.db")
        # Initialize minimal DB so consolidate_training's
        # self_insights query doesn't crash.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE self_insights ("
            "id INTEGER PRIMARY KEY, "
            "timestamp REAL NOT NULL)")
        conn.execute(
            "CREATE TABLE self_predictions ("
            "id INTEGER PRIMARY KEY, "
            "verified INTEGER DEFAULT 0)")
        conn.commit()
        conn.close()

        # Build a minimal engine bypassing the heavy __init__.
        eng = SelfReasoningEngine.__new__(SelfReasoningEngine)
        eng._db_path = db_path
        eng._active_predictions = []
        eng._prediction_accuracy_ema = 0.65
        eng._last_profile = None

        result = eng.consolidate_training()
        assert isinstance(result, dict)
        assert "introspect_signals" in result
        sigs = result["introspect_signals"]
        assert "self_prediction_accuracy" in sigs
        assert "self_profile_divergence" in sigs
        assert sigs["self_prediction_accuracy"] == pytest.approx(0.65)
