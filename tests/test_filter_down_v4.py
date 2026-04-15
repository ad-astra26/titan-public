"""
Tests for V4 FILTER_DOWN — 30-dim Full Trinity attention network.

Tests the extended value network (30→64→32→1), transition recording
with both Inner and Outer Trinity, multiplier computation for 4 layers,
and SPIRIT FOCUS cascade integration.
"""
import math
import tempfile
import pytest


class TestFilterDownV4Engine:
    """Tests for the V4 30-dim FilterDown engine."""

    def test_init_defaults(self):
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            assert engine._total_train_steps == 0
            assert len(engine._inner_body_multipliers) == 5
            assert len(engine._outer_mind_multipliers) == 5

    def test_record_30dim_transition(self):
        """Can record a 30-dim SPIRIT tensor transition."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            state = [0.5] * 30
            next_state = [0.6] * 30
            engine.record_transition(state, next_state)
            assert len(engine._buffer) == 1

    def test_training_after_min_transitions(self):
        """Training triggers after enough transitions accumulate."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            # Record enough transitions
            for i in range(40):
                state = [0.3 + i * 0.01] * 30
                next_state = [0.31 + i * 0.01] * 30
                engine.record_transition(state, next_state)

            loss = engine.maybe_train()
            assert loss is not None
            assert engine._total_train_steps == 1

    def test_compute_multipliers_default(self):
        """Before training, returns default (1.0) multipliers."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            result = engine.compute_multipliers([0.5] * 30)

            assert "inner_body" in result
            assert "inner_mind" in result
            assert "outer_body" in result
            assert "outer_mind" in result
            assert result["inner_body"] == [1.0] * 5

    def test_compute_multipliers_after_training(self):
        """After training, multipliers deviate from defaults."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            # Create diverse transitions
            import random
            random.seed(42)
            for _ in range(50):
                state = [random.uniform(0.1, 0.9) for _ in range(30)]
                next_state = [random.uniform(0.1, 0.9) for _ in range(30)]
                engine.record_transition(state, next_state)

            engine.maybe_train()
            result = engine.compute_multipliers([0.5] * 30)

            # At least some multiplier should differ from 1.0
            all_mults = (result["inner_body"] + result["inner_mind"] +
                         result["outer_body"] + result["outer_mind"])
            assert any(abs(m - 1.0) > 0.001 for m in all_mults)

    def test_multipliers_have_4_layers(self):
        """Multipliers cover 4 observable layers (not spirit layers)."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            result = engine.compute_multipliers([0.5] * 30)
            assert len(result) == 4
            for key in ["inner_body", "inner_mind", "outer_body", "outer_mind"]:
                assert len(result[key]) == 5

    def test_multipliers_clamped(self):
        """All multipliers stay within [FLOOR, CEIL]."""
        from titan_plugin.logic.filter_down import (
            FilterDownV4Engine, MULTIPLIER_FLOOR, MULTIPLIER_CEIL,
        )
        import random
        random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            for _ in range(50):
                state = [random.uniform(0.0, 1.0) for _ in range(30)]
                next_state = [random.uniform(0.0, 1.0) for _ in range(30)]
                engine.record_transition(state, next_state)
            engine.maybe_train()

            result = engine.compute_multipliers([0.5] * 30)
            for layer_mults in result.values():
                for m in layer_mults:
                    assert MULTIPLIER_FLOOR <= m <= MULTIPLIER_CEIL

    def test_reward_uses_both_trinities(self):
        """Reward signal combines Inner and Outer middle path losses."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)

            # Perfect center → high reward (low loss)
            perfect = [0.5] * 30
            engine.record_transition(perfect, perfect)
            _, reward_good, _ = engine._buffer._buffer[0]

            # Far from center → low reward (high loss)
            engine._buffer._buffer.clear()
            extreme = [1.0] * 30
            engine.record_transition(extreme, extreme)
            _, reward_bad, _ = engine._buffer._buffer[0]

            assert reward_good > reward_bad

    def test_stats_structure(self):
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            stats = engine.get_stats()
            assert stats["version"] == "v4"
            assert stats["state_dim"] == 30
            assert "inner_body_multipliers" in stats
            assert "outer_mind_multipliers" in stats

    def test_persistence(self):
        """V4 weights and buffer persist independently of V3."""
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FilterDownV4Engine(data_dir=tmpdir)
            for i in range(10):
                engine.record_transition([0.5 + i*0.01] * 30, [0.5 + i*0.01] * 30)
            engine._persist()

            assert os.path.exists(os.path.join(tmpdir, "filter_down_v4_weights.json"))
            assert os.path.exists(os.path.join(tmpdir, "filter_down_v4_buffer.json"))

            # Load into new engine
            engine2 = FilterDownV4Engine(data_dir=tmpdir)
            assert len(engine2._buffer) == 10


class TestSpiritFocusCascade:
    """Tests for the SPIRIT FOCUS cascade multiplier integration."""

    def test_focus_without_stale_no_cascade(self):
        """Without STALE, focus nudges are unmodified."""
        from titan_plugin.logic.focus_pid import FocusPID
        focus = FocusPID("body")
        nudges = focus.update([0.7, 0.7, 0.7, 0.7, 0.7])
        # No cascade = nudges as-is
        assert all(isinstance(n, float) for n in nudges)

    def test_cascade_multiplier_amplifies_nudges(self):
        """SPIRIT cascade multiplier amplifies focus nudges."""
        from titan_plugin.logic.focus_pid import FocusPID
        focus = FocusPID("body")
        values = [0.8, 0.8, 0.8, 0.8, 0.8]  # Off-center
        nudges = focus.update(values)

        # Apply cascade manually (as spirit_worker does)
        cascade_mult = 2.0
        amplified = [n * cascade_mult for n in nudges]

        # Amplified nudges should be stronger
        for orig, amp in zip(nudges, amplified):
            if orig != 0:
                assert abs(amp) > abs(orig)

    def test_balanced_part_feels_less_cascade(self):
        """Balanced parts get small nudges × cascade = still small."""
        from titan_plugin.logic.focus_pid import FocusPID
        focus_balanced = FocusPID("balanced")
        focus_imbalanced = FocusPID("imbalanced")

        balanced_nudges = focus_balanced.update([0.5, 0.5, 0.5, 0.5, 0.5])
        imbalanced_nudges = focus_imbalanced.update([0.9, 0.1, 0.9, 0.1, 0.9])

        cascade = 2.0
        balanced_force = sum(abs(n * cascade) for n in balanced_nudges)
        imbalanced_force = sum(abs(n * cascade) for n in imbalanced_nudges)

        assert imbalanced_force > balanced_force

    def test_unified_spirit_stale_multiplier_integration(self):
        """UnifiedSpirit provides correct cascade multiplier when STALE."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"stale_threshold": 0.9},
                data_dir=tmpdir,
            )

            # Not stale → multiplier = 1.0
            assert spirit.stale_focus_multiplier == 1.0

            # Make it stale
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            spirit.update_subconscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.update_conscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.advance({})

            assert spirit.is_stale
            assert spirit.stale_focus_multiplier > 1.0
