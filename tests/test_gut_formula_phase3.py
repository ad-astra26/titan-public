"""rFP β Phase 3 — Gut formula redesign tests.

Tests the new primitive-affinity gut formula:
- Replaces degenerate (1 - abs(conf - mean_urgency)) with semantic alignment
- Symmetric formula: contribution = urgency × (affinity - 0.5) × 2
- Time-smoothed via EMA (alpha=0.3)
- Hormone-augmented urgencies (separate test for NeuralNS method)
- Backward compat (legacy mode when no primitive passed)
"""
import pytest


class TestPrimitiveAffinityTable:
    def test_all_11_programs_present(self):
        from titan_plugin.logic.reasoning import PRIMITIVE_AFFINITY
        expected = {"REFLEX", "FOCUS", "INTUITION", "IMPULSE", "INSPIRATION",
                    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
                    "METABOLISM", "VIGILANCE"}
        assert set(PRIMITIVE_AFFINITY.keys()) == expected

    def test_affinity_values_in_valid_range(self):
        """All affinity values must be in [0, 1] (formula assumption)."""
        from titan_plugin.logic.reasoning import PRIMITIVE_AFFINITY
        for prog, affs in PRIMITIVE_AFFINITY.items():
            for prim, val in affs.items():
                assert 0.0 <= val <= 1.0, f"{prog}.{prim}={val} out of range"

    def test_only_real_primitives_referenced(self):
        """Affinity entries must reference primitives that actually exist."""
        from titan_plugin.logic.reasoning import PRIMITIVE_AFFINITY, PRIMITIVES
        for prog, affs in PRIMITIVE_AFFINITY.items():
            for prim in affs:
                assert prim in PRIMITIVES, (
                    f"{prog} references unknown primitive '{prim}'")

    def test_each_program_has_at_least_one_primary(self):
        """Every program must have at least one primary affinity (1.0)."""
        from titan_plugin.logic.reasoning import PRIMITIVE_AFFINITY
        for prog, affs in PRIMITIVE_AFFINITY.items():
            assert any(v >= 0.95 for v in affs.values()), (
                f"{prog} has no primary affinity (max={max(affs.values())})")


class TestGutFormula:
    """Direct tests of _update_gut_agreement logic (no full ReasoningEngine)."""

    @pytest.fixture
    def mock_engine(self):
        """Mock with the minimum attrs _update_gut_agreement reads."""
        class M:
            confidence = 0.5
            spirit_nudge = 0.0
            gut_agreement = 0.5  # initial neutral
            _gut_ema_alpha = 1.0  # disable smoothing for direct-formula tests
        return M()

    def test_aligned_program_pulls_gut_high(self, mock_engine):
        """FOCUS firing strongly while DECOMPOSE executes → high gut."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={"FOCUS": 0.8}, current_primitive="DECOMPOSE")
        # FOCUS affinity for DECOMPOSE = 1.0 → contribution = 0.8 × 1.0 = 0.8
        # net = 0.8 / 0.8 = 1.0 → gut = 0.5 + 0.5 × 1.0 = 1.0
        assert mock_engine.gut_agreement >= 0.95

    def test_anti_affinity_pulls_gut_low(self, mock_engine):
        """CURIOSITY firing strongly while CONCLUDE executes → low gut."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={"CURIOSITY": 0.8}, current_primitive="CONCLUDE")
        # CURIOSITY affinity for CONCLUDE = 0.3 → contribution = 0.8 × -0.4 = -0.32
        # net = -0.32 / 0.8 = -0.4 → gut = 0.5 + 0.5 × -0.4 = 0.3
        assert 0.25 <= mock_engine.gut_agreement <= 0.35

    def test_neutral_program_no_contribution(self, mock_engine):
        """REFLEX only knows CONCLUDE. With DECOMPOSE → neutral 0.5."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={"REFLEX": 0.6}, current_primitive="DECOMPOSE")
        # REFLEX has no DECOMPOSE entry → uses NEUTRAL_AFFINITY=0.5
        # contribution = 0.6 × 0 = 0 → gut = 0.5 (neutral)
        assert 0.45 <= mock_engine.gut_agreement <= 0.55

    def test_mixed_signals_balance(self, mock_engine):
        """FOCUS (loves DECOMPOSE) + METABOLISM (neutral on DECOMPOSE) → high."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine,
            gut_signals={"FOCUS": 0.6, "METABOLISM": 0.4},
            current_primitive="DECOMPOSE")
        # FOCUS: 0.6 × (1.0 - 0.5) × 2 = 0.6
        # METABOLISM: 0.4 × (0.5 - 0.5) × 2 = 0 (no DECOMPOSE entry → neutral)
        # net = 0.6 / 1.0 = 0.6 → gut = 0.8
        assert 0.75 <= mock_engine.gut_agreement <= 0.85

    def test_metabolism_anti_loop(self, mock_engine):
        """METABOLISM has 0.3 affinity for LOOP → opposes loop."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={"METABOLISM": 0.7}, current_primitive="LOOP")
        # METABOLISM.LOOP = 0.3 → contribution = 0.7 × -0.4 = -0.28
        # net = -0.28 / 0.7 = -0.4 → gut = 0.3
        assert 0.25 <= mock_engine.gut_agreement <= 0.35

    def test_empty_signals_neutral(self, mock_engine):
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={}, current_primitive="COMPARE")
        assert mock_engine.gut_agreement == 0.5

    def test_below_noise_floor_neutral(self, mock_engine):
        """Urgencies below 0.05 don't contribute (filtered noise)."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        ReasoningEngine._update_gut_agreement(
            mock_engine,
            gut_signals={"FOCUS": 0.01, "CURIOSITY": 0.02},
            current_primitive="CONCLUDE")
        assert mock_engine.gut_agreement == 0.5

    def test_legacy_mode_no_primitive(self, mock_engine):
        """Backward compat: no primitive passed → falls back to mean-urgency."""
        from titan_plugin.logic.reasoning import ReasoningEngine
        mock_engine.confidence = 0.7
        ReasoningEngine._update_gut_agreement(
            mock_engine, gut_signals={"FOCUS": 0.6})
        # Legacy: 1 - abs(0.7 - 0.6) = 0.9
        assert 0.85 <= mock_engine.gut_agreement <= 0.95


class TestGutEMA:
    def test_ema_smoothing_applies(self):
        """Multiple calls should EMA-smooth gut, not snap to instantaneous value."""
        from titan_plugin.logic.reasoning import ReasoningEngine

        class M:
            confidence = 0.5
            spirit_nudge = 0.0
            gut_agreement = 0.5
            _gut_ema_alpha = 0.3  # default smoothing

        engine = M()
        # Push gut up via aligned signal
        for _ in range(5):
            ReasoningEngine._update_gut_agreement(
                engine, gut_signals={"FOCUS": 0.8}, current_primitive="DECOMPOSE")
        # Should be approaching 1.0 but not snap to it
        assert 0.7 <= engine.gut_agreement <= 0.99
        # Now flip to anti-affinity — should not snap down immediately
        prev = engine.gut_agreement
        ReasoningEngine._update_gut_agreement(
            engine, gut_signals={"CURIOSITY": 0.8}, current_primitive="CONCLUDE")
        # New raw value would be 0.3, but EMA: 0.7×prev + 0.3×0.3
        expected = 0.7 * prev + 0.3 * 0.3
        assert abs(engine.gut_agreement - expected) < 0.01

    def test_ema_alpha_disabled_uses_instantaneous(self):
        """Setting alpha=1.0 should make gut snap to instantaneous value."""
        from titan_plugin.logic.reasoning import ReasoningEngine

        class M:
            confidence = 0.5
            spirit_nudge = 0.0
            gut_agreement = 0.5
            _gut_ema_alpha = 1.0

        engine = M()
        ReasoningEngine._update_gut_agreement(
            engine, gut_signals={"FOCUS": 0.8}, current_primitive="DECOMPOSE")
        assert engine.gut_agreement >= 0.95


class TestHormoneAugmentation:
    def test_blend_zero_returns_pure_nn(self):
        """hormone_blend=0 should return raw _all_urgencies unchanged."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem

        # Minimal mock — only need _all_urgencies + flags
        class M:
            _all_urgencies = {"FOCUS": 0.3, "METABOLISM": 0.5}
            _hormonal_enabled = True
            _hormonal = None
        result = NeuralNervousSystem.get_augmented_urgencies(M(), hormone_blend=0)
        assert result == {"FOCUS": 0.3, "METABOLISM": 0.5}

    def test_blend_with_hormone_above_threshold(self):
        """Hormone level/threshold > 1 should be clamped at 1.0 in blend."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem

        class MockHormone:
            level = 1.5
            threshold = 0.5
            # level/threshold = 3, clamped to 1.0

        class MockHormonal:
            _hormones = {"FOCUS": MockHormone()}

        class M:
            _all_urgencies = {"FOCUS": 0.0}  # NN at zero (collapsed)
            _hormonal_enabled = True
            _hormonal = MockHormonal()
        result = NeuralNervousSystem.get_augmented_urgencies(M(), hormone_blend=0.3)
        # 0.7 × 0.0 + 0.3 × 1.0 = 0.3
        assert abs(result["FOCUS"] - 0.3) < 0.01

    def test_blend_normal_case(self):
        """Standard blend: 70% NN + 30% hormone-ratio."""
        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem

        class MockHormone:
            level = 0.4
            threshold = 0.5
            # level/threshold = 0.8

        class MockHormonal:
            _hormones = {"FOCUS": MockHormone()}

        class M:
            _all_urgencies = {"FOCUS": 0.6}
            _hormonal_enabled = True
            _hormonal = MockHormonal()
        result = NeuralNervousSystem.get_augmented_urgencies(M(), hormone_blend=0.3)
        # 0.7 × 0.6 + 0.3 × 0.8 = 0.42 + 0.24 = 0.66
        assert abs(result["FOCUS"] - 0.66) < 0.01
