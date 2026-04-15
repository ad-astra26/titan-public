"""
Tests for V4 Time Awareness — End-to-End Integration.

Verifies the complete V4 pipeline:
  Inner Trinity → SphereClocks → Resonance → GREAT PULSE → UnifiedSpirit
  Outer Trinity → SphereClocks → Resonance → GREAT PULSE → Enrichment
  STALE detection → FOCUS cascade → FilterDown V4 (30-dim)

These tests exercise the full data flow without requiring actual
multiprocessing (all components instantiated in-process).
"""
import math
import tempfile
import time
import pytest


class TestV4EndToEndPipeline:
    """Full pipeline: sphere clocks → resonance → GREAT PULSE → advance."""

    def _create_full_stack(self, tmpdir, clock_speed=10.0):
        """Create all V4 components wired together."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        from titan_plugin.logic.resonance import ResonanceDetector
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        from titan_plugin.logic.filter_down import FilterDownV4Engine

        config = {
            "base_contraction_speed": clock_speed,
            "balance_threshold": 0.30,
            "resonance_cycles": 1,  # Low for testing
            "pulse_window": 120.0,
        }

        sphere_clock = SphereClockEngine(config=config, data_dir=tmpdir)
        resonance = ResonanceDetector(config=config, data_dir=tmpdir)
        spirit = UnifiedSpirit(data_dir=tmpdir)
        filter_v4 = FilterDownV4Engine(data_dir=tmpdir)

        return sphere_clock, resonance, spirit, filter_v4

    def test_balanced_tensors_produce_pulses(self):
        """Balanced Inner+Outer Trinity tensors generate sphere pulses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            balanced = [0.5] * 5
            inner_pulses = sc.tick_inner(balanced, balanced, balanced)
            outer_pulses = sc.tick_outer(balanced, balanced, balanced)

            # With speed=10 and perfect center, should pulse immediately
            assert len(inner_pulses) == 3
            assert len(outer_pulses) == 3

    def test_pulses_trigger_resonance(self):
        """Sphere pulses from both sides trigger resonance detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            balanced = [0.5] * 5
            inner_pulses = sc.tick_inner(balanced, balanced, balanced)
            outer_pulses = sc.tick_outer(balanced, balanced, balanced)

            big_pulses = []
            for p in inner_pulses + outer_pulses:
                result = res.record_pulse(p)
                if result:
                    big_pulses.append(result)

            assert len(big_pulses) > 0

    def test_full_pipeline_great_pulse_to_epoch(self):
        """Complete pipeline: balanced tensors → resonance → GREAT PULSE → epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            # Update SPIRIT with balanced tensors
            balanced = [0.5] * 5
            spirit.update_subconscious(balanced, balanced, balanced)
            spirit.update_conscious(balanced, balanced, balanced)

            # Generate pulses
            inner_pulses = sc.tick_inner(balanced, balanced, balanced)
            outer_pulses = sc.tick_outer(balanced, balanced, balanced)

            # Feed pulses to resonance detector
            great_ready = False
            for p in inner_pulses + outer_pulses:
                result = res.record_pulse(p)
                if result and result.get("great_pulse_ready"):
                    great_ready = True

            # If GREAT PULSE condition met, advance spirit
            if great_ready:
                epoch = spirit.advance(res.get_stats())
                assert epoch is not None
                assert epoch.epoch_id == 1
                assert len(epoch.spirit_tensor) == 30

                # Compute enrichment
                enrichment = spirit.compute_enrichment()
                assert len(enrichment) == 6

    def test_imbalanced_tensors_slow_pipeline(self):
        """Imbalanced tensors produce slow sphere clocks → no quick resonance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir, clock_speed=0.01)

            # Very imbalanced
            imbalanced = [1.0, 0.0, 1.0, 0.0, 1.0]
            inner_pulses = sc.tick_inner(imbalanced, imbalanced, imbalanced)
            outer_pulses = sc.tick_outer(imbalanced, imbalanced, imbalanced)

            # With slow speed + imbalanced, no pulses should fire
            assert len(inner_pulses) == 0
            assert len(outer_pulses) == 0

    def test_velocity_tracks_growth_over_epochs(self):
        """Velocity increases when tensor magnitude grows across epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            # Epoch 1: small tensor
            spirit.update_subconscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.update_conscious([0.2]*5, [0.2]*5, [0.2]*5)
            e1 = spirit.advance({})

            # Epoch 2: growing tensor
            spirit.update_subconscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
            e2 = spirit.advance({})

            # Epoch 3: larger tensor
            spirit.update_subconscious([0.8]*5, [0.8]*5, [0.8]*5)
            spirit.update_conscious([0.8]*5, [0.8]*5, [0.8]*5)
            e3 = spirit.advance({})

            assert e3.velocity > 1.0  # Growing

    def test_stale_triggers_focus_cascade(self):
        """STALE detection provides correct focus cascade multiplier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            # Start with large tensor
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            # Shrink (STALE condition)
            spirit.update_subconscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.update_conscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.advance({})

            assert spirit.is_stale
            mult = spirit.stale_focus_multiplier
            assert mult > 1.0

            # Apply cascade to focus nudges
            from titan_plugin.logic.focus_pid import FocusPID
            focus = FocusPID("body")
            nudges = focus.update([0.8]*5)
            cascaded = [n * mult for n in nudges]
            assert sum(abs(c) for c in cascaded) > sum(abs(n) for n in nudges)

    def test_filter_down_v4_records_30dim_transitions(self):
        """V4 FilterDown records transitions from full 30DT tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            spirit.update_subconscious([0.4]*5, [0.5]*5, [0.6]*5)
            spirit.update_conscious([0.3]*5, [0.5]*5, [0.7]*5)
            prev = spirit.tensor

            spirit.update_subconscious([0.5]*5, [0.6]*5, [0.7]*5)
            spirit.update_conscious([0.4]*5, [0.6]*5, [0.8]*5)
            curr = spirit.tensor

            fv4.record_transition(prev, curr)
            assert len(fv4._buffer) == 1

    def test_enrichment_proportional_to_balance(self):
        """GREAT PULSE enrichment rewards balanced components more."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sc, res, spirit, fv4 = self._create_full_stack(tmpdir)

            # Inner body perfectly balanced, inner mind off-center
            spirit.update_subconscious([0.5]*5, [0.9]*5, [0.5]*5)
            spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.advance({})

            enrichment = spirit.compute_enrichment()
            assert enrichment["inner_body"]["reward"] > enrichment["inner_mind"]["reward"]

    def test_sphere_clock_persistence_across_restart(self):
        """Full V4 state survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First session: generate some state
            sc1, res1, sp1, fv1 = self._create_full_stack(tmpdir)
            sc1.tick_inner([0.5]*5, [0.5]*5, [0.5]*5)
            for p in sc1.tick_inner([0.5]*5, [0.5]*5, [0.5]*5):
                res1.record_pulse(p)
            sp1.update_subconscious([0.5]*5, [0.5]*5, [0.5]*5)
            sp1.advance({})

            sc1.save_state()
            res1.save_state()
            sp1.save_state()

            # Second session: verify state restored
            sc2, res2, sp2, fv2 = self._create_full_stack(tmpdir)
            assert sp2.epoch_count == 1
            total_pulses = sum(c.pulse_count for c in sc2.clocks.values())
            assert total_pulses > 0


class TestV4ConfigIntegration:
    """Tests for V4 configuration sections."""

    def test_sphere_clock_config_loads(self):
        """[sphere_clock] config section loads correctly."""
        import os
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "titan_plugin", "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            sc_cfg = config.get("sphere_clock", {})
            assert "base_contraction_speed" in sc_cfg
            assert "min_radius" in sc_cfg
            assert "balance_threshold" in sc_cfg
            assert sc_cfg["balance_threshold"] == 0.20
        except FileNotFoundError:
            pytest.skip("config.toml not found")

    def test_epochs_config_loads(self):
        """[epochs] config section has production timing."""
        import os
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "titan_plugin", "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            epochs = config.get("epochs", {})
            assert epochs["small_epoch_interval"] == 21600   # 6 hours
            assert epochs["greater_epoch_interval"] == 86400  # 24 hours
            assert "outer_trinity_interval" in epochs
        except FileNotFoundError:
            pytest.skip("config.toml not found")

    def test_bus_message_types_defined(self):
        """All V4 bus message types are defined."""
        from titan_plugin.bus import (
            OUTER_TRINITY_STATE, SPHERE_PULSE, BIG_PULSE,
            GREAT_PULSE, FILTER_DOWN_V4,
        )
        assert OUTER_TRINITY_STATE == "OUTER_TRINITY_STATE"
        assert SPHERE_PULSE == "SPHERE_PULSE"
        assert BIG_PULSE == "BIG_PULSE"
        assert GREAT_PULSE == "GREAT_PULSE"
        assert FILTER_DOWN_V4 == "FILTER_DOWN_V4"


class TestV4TensorLayout:
    """Tests verifying the 30DT SPIRIT tensor layout matches the design."""

    def test_30dt_layout(self):
        """30DT tensor layout: [inner_body|inner_mind|inner_spirit|outer_body|outer_mind|outer_spirit]."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)

            spirit.update_subconscious(
                inner_body=[0.1, 0.2, 0.3, 0.4, 0.5],
                inner_mind=[0.6, 0.7, 0.8, 0.9, 1.0],
                inner_spirit=[0.11, 0.22, 0.33, 0.44, 0.55],
            )
            spirit.update_conscious(
                outer_body=[0.15, 0.25, 0.35, 0.45, 0.55],
                outer_mind=[0.65, 0.75, 0.85, 0.95, 0.05],
                outer_spirit=[0.12, 0.23, 0.34, 0.45, 0.56],
            )

            t = spirit.tensor
            assert len(t) == 30

            # Verify layout
            assert t[0:5] == [0.1, 0.2, 0.3, 0.4, 0.5]       # Inner Body
            assert t[5:10] == [0.6, 0.7, 0.8, 0.9, 1.0]       # Inner Mind
            assert t[10:15] == [0.11, 0.22, 0.33, 0.44, 0.55]  # Inner Spirit
            assert t[15:20] == [0.15, 0.25, 0.35, 0.45, 0.55]  # Outer Body
            assert t[20:25] == [0.65, 0.75, 0.85, 0.95, 0.05]  # Outer Mind
            assert t[25:30] == [0.12, 0.23, 0.34, 0.45, 0.56]  # Outer Spirit

    def test_inner_outer_symmetry(self):
        """Inner and Outer Trinity slices are symmetric (15 dims each)."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit, INNER_DIMS, OUTER_DIMS
        assert INNER_DIMS == 15
        assert OUTER_DIMS == 15
        assert INNER_DIMS + OUTER_DIMS == 30
