"""
Tests for V4 Time Awareness — SphereClockEngine.

Tests sphere contraction/expansion mechanics, pulse generation,
balance-dependent velocity, IQL scoring hooks, persistence, and
the 6-clock engine orchestration.
"""
import json
import math
import os
import tempfile
import pytest


# ── SphereClock unit tests ────────────────────────────────────────────

class TestSphereClock:
    """Tests for individual SphereClock instances."""

    def test_init_defaults(self):
        """Clock starts fully expanded with scalar at edge."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test")
        assert clock.radius == 1.0
        assert clock.scalar_position == 1.0
        assert clock.phase == 0.0
        assert clock.pulse_count == 0
        assert clock.contraction_velocity == 0.0

    def test_contraction_moves_toward_center(self):
        """Ticking with balanced tensor moves scalar closer to center."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.1)
        initial_pos = clock.scalar_position

        # delta=0.1 (fairly balanced) → velocity = 0.1 * (1.0 - 0.1) = 0.09
        clock.tick(delta_from_center=0.1)
        assert clock.scalar_position < initial_pos
        assert clock.scalar_position == pytest.approx(1.0 - 0.09, abs=0.001)

    def test_balanced_tensor_faster_contraction(self):
        """Lower delta (more balanced) produces faster contraction."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock_balanced = SphereClock("balanced", base_speed=0.1)
        clock_imbalanced = SphereClock("imbalanced", base_speed=0.1)

        clock_balanced.tick(delta_from_center=0.1)   # velocity = 0.09
        clock_imbalanced.tick(delta_from_center=0.8) # velocity = 0.02

        assert clock_balanced.contraction_velocity > clock_imbalanced.contraction_velocity
        assert clock_balanced.scalar_position < clock_imbalanced.scalar_position

    def test_imbalanced_tensor_slow_contraction(self):
        """High delta (imbalanced) produces near-zero contraction."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.05)

        clock.tick(delta_from_center=0.95)
        # velocity = 0.05 * (1.0 - 0.95) = 0.0025
        assert clock.contraction_velocity == pytest.approx(0.0025, abs=0.001)
        # Almost no movement
        assert clock.scalar_position > 0.99

    def test_pulse_fires_when_scalar_reaches_center(self):
        """Pulse is generated when scalar_position reaches 0."""
        from titan_plugin.logic.sphere_clock import SphereClock
        # Set up a clock that will pulse on next tick
        clock = SphereClock("test", base_speed=1.0)
        clock.scalar_position = 0.05  # Very close to center

        pulse = clock.tick(delta_from_center=0.0)  # velocity = 1.0 * 1.0 = 1.0
        assert pulse is not None
        assert pulse["component"] == "test"
        assert pulse["pulse_count"] == 1
        assert clock.pulse_count == 1

    def test_sphere_expands_after_pulse(self):
        """After pulse, scalar resets to edge of (new) sphere radius."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=1.0, pulse_shrink_rate=0.02)
        clock.scalar_position = 0.01

        pulse = clock.tick(delta_from_center=0.1)  # Balanced → radius shrinks
        assert pulse is not None
        assert clock.scalar_position == clock.radius
        assert clock.scalar_position > 0  # Not stuck at center

    def test_phase_wraps_at_2pi(self):
        """Phase stays within [0, 2π)."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.5)

        # Tick many times to accumulate phase
        for _ in range(100):
            clock.tick(delta_from_center=0.1)
            assert 0.0 <= clock.phase < 2.0 * math.pi

    def test_pulse_count_increments(self):
        """Each pulse increments the counter."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=10.0)  # Very fast

        pulse_events = []
        for _ in range(50):
            pulse = clock.tick(delta_from_center=0.0)
            if pulse:
                pulse_events.append(pulse)

        assert len(pulse_events) > 1
        assert pulse_events[-1]["pulse_count"] == len(pulse_events)

    def test_sphere_shrinks_for_balanced_parts(self):
        """Consistently balanced components get tighter (smaller) spheres."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=10.0, pulse_shrink_rate=0.05)

        initial_radius = clock.radius
        # Force several balanced pulses
        for _ in range(100):
            clock.tick(delta_from_center=0.1)

        # After multiple balanced pulses, radius should be smaller
        assert clock.radius < initial_radius

    def test_sphere_grows_for_imbalanced_pulse(self):
        """Imbalanced pulse makes sphere slightly larger (slower next cycle)."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=10.0, pulse_shrink_rate=0.05)

        # First force a shrink
        clock.radius = 0.5
        clock.scalar_position = 0.01

        # Imbalanced pulse (delta > threshold)
        pulse = clock.tick(delta_from_center=0.9)
        assert pulse is not None
        assert pulse["radius_after"] > 0.5  # Grew

    def test_iql_scoring_balanced_at_center(self):
        """IQL: +1 when balanced and at center."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", balance_threshold=0.20)
        clock.scalar_position = 0.005  # Near center

        score = clock.get_iql_score(delta_from_center=0.1)
        assert score == 1

    def test_iql_scoring_imbalanced(self):
        """IQL: -1 when outside balance threshold."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", balance_threshold=0.20)

        score = clock.get_iql_score(delta_from_center=0.5)
        assert score == -1

    def test_iql_scoring_neutral(self):
        """IQL: 0 when balanced but not at center yet."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", balance_threshold=0.20)
        clock.scalar_position = 0.5  # Not at center

        score = clock.get_iql_score(delta_from_center=0.1)
        assert score == 0

    def test_balance_threshold_check(self):
        """20% delta threshold correctly classifies balanced/imbalanced."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=10.0, balance_threshold=0.20)

        # Balanced (delta=0.15 < threshold=0.20)
        clock.scalar_position = 0.01
        pulse = clock.tick(delta_from_center=0.15)
        assert pulse is not None
        assert pulse["balanced"] is True

        # Imbalanced (delta=0.25 > threshold=0.20)
        clock.scalar_position = 0.01
        pulse = clock.tick(delta_from_center=0.25)
        assert pulse is not None
        assert pulse["balanced"] is False

    def test_persistence_save_load(self):
        """Clock state survives save/load cycle."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.1)

        # Advance state
        for _ in range(5):
            clock.tick(delta_from_center=0.2)
        clock.pulse_count = 42

        # Serialize
        data = clock.to_dict()
        assert data["pulse_count"] == 42

        # Restore into fresh clock
        clock2 = SphereClock("test")
        clock2.from_dict(data)
        assert clock2.pulse_count == 42
        assert clock2.radius == pytest.approx(clock.radius, abs=0.001)
        assert clock2.scalar_position == pytest.approx(clock.scalar_position, abs=0.001)


# ── SphereClockEngine integration tests ────────────────────────────────

class TestSphereClockEngine:
    """Tests for the 6-clock engine orchestration."""

    def test_engine_creates_6_clocks(self):
        """Engine initializes with all 6 Trinity component clocks."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine, ALL_COMPONENTS
        engine = SphereClockEngine()

        assert len(engine.clocks) == 6
        for name in ALL_COMPONENTS:
            assert name in engine.clocks

    def test_get_all_phases_returns_6_entries(self):
        """get_all_phases returns phase for all 6 clocks."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine()

        phases = engine.get_all_phases()
        assert len(phases) == 6
        for phase in phases.values():
            assert isinstance(phase, float)

    def test_get_paired_phases(self):
        """get_paired_phases returns 3 pairs (body, mind, spirit)."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine()

        pairs = engine.get_paired_phases()
        assert len(pairs) == 3
        assert "body" in pairs
        assert "mind" in pairs
        assert "spirit" in pairs
        for inner_phase, outer_phase in pairs.values():
            assert isinstance(inner_phase, float)
            assert isinstance(outer_phase, float)

    def test_tick_inner_returns_pulse_events(self):
        """tick_inner processes 3 inner clocks and returns any pulses."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine(config={"base_contraction_speed": 10.0})

        body = [0.5] * 5    # Perfect center
        mind = [0.5] * 5
        spirit = [0.5] * 5

        pulses = engine.tick_inner(body, mind, spirit)
        # With speed=10 and perfect center (delta≈0), should pulse immediately
        assert isinstance(pulses, list)
        assert len(pulses) == 3  # All 3 should pulse on first tick (speed=10)
        for p in pulses:
            assert p["component"].startswith("inner_")

    def test_tick_outer_returns_pulse_events(self):
        """tick_outer processes 3 outer clocks and returns any pulses."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine(config={"base_contraction_speed": 10.0})

        outer_body = [0.5] * 5
        outer_mind = [0.5] * 5
        outer_spirit = [0.5] * 5

        pulses = engine.tick_outer(outer_body, outer_mind, outer_spirit)
        assert isinstance(pulses, list)
        assert len(pulses) == 3
        for p in pulses:
            assert p["component"].startswith("outer_")

    def test_tick_inner_no_pulse_when_imbalanced(self):
        """Very imbalanced tensors produce no pulse on single tick."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine(config={"base_contraction_speed": 0.01})

        # Extreme imbalance: all at 1.0 (max distance from center 0.5)
        body = [1.0] * 5
        mind = [1.0] * 5
        spirit = [1.0] * 5

        pulses = engine.tick_inner(body, mind, spirit)
        assert len(pulses) == 0  # Too slow to pulse in one tick

    def test_engine_persistence(self):
        """Engine state persists across save/load."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine1 = SphereClockEngine(
                config={"base_contraction_speed": 10.0},
                data_dir=tmpdir,
            )
            # Generate some pulses
            engine1.tick_inner([0.5] * 5, [0.5] * 5, [0.5] * 5)
            total_before = sum(c.pulse_count for c in engine1.clocks.values())
            assert total_before > 0

            engine1.save_state()

            # Load into fresh engine
            engine2 = SphereClockEngine(
                config={"base_contraction_speed": 10.0},
                data_dir=tmpdir,
            )
            total_after = sum(c.pulse_count for c in engine2.clocks.values())
            assert total_after == total_before

    def test_engine_stats(self):
        """get_stats returns structured data."""
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        engine = SphereClockEngine()
        stats = engine.get_stats()

        assert "clocks" in stats
        assert len(stats["clocks"]) == 6
        assert "total_pulses" in stats
        assert "config" in stats
        assert stats["config"]["base_speed"] > 0
