"""
tests/test_t1_coherence_observables.py — T1: Coherence + Observables tests.

12 tests covering:
  - layer_coherence correctness + backward compat with layer_loss
  - Each of the 5 observables
  - ObservableEngine processes all 6 parts
  - Sphere clock uses coherence correctly
  - Balanced tensor → fast clock, incoherent tensor → slow clock
  - Clock still pulses at min_velocity when incoherent
"""
import math
import pytest


# ── T1.1: layer_coherence ──────────────────────────────────────────────

class TestLayerCoherence:
    def test_uniform_tensor_returns_one(self):
        """Uniform tensor (all same value) → coherence = 1.0."""
        from titan_plugin.logic.middle_path import layer_coherence
        # All 0.3 — perfectly coherent (zero variance)
        assert layer_coherence([0.3, 0.3, 0.3, 0.3, 0.3]) == 1.0
        # All 0.9 — also perfectly coherent
        assert layer_coherence([0.9, 0.9, 0.9, 0.9, 0.9]) == 1.0
        # All 0.5 (center)
        assert layer_coherence([0.5, 0.5, 0.5, 0.5, 0.5]) == 1.0

    def test_maximally_varied_returns_near_zero(self):
        """Half at 0, half at 1 → coherence ≈ 0.0."""
        from titan_plugin.logic.middle_path import layer_coherence
        # 3 at 0.0, 2 at 1.0 — not exactly 0.0 because uneven split
        # but [0, 0, 1, 1] (even split in 4-dim) is exactly 0.0
        assert layer_coherence([0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)
        # 5-dim: close to zero but not exactly (3/2 split)
        coh = layer_coherence([0.0, 0.0, 0.0, 1.0, 1.0])
        assert coh < 0.1  # very low coherence

    def test_backward_compat_with_layer_loss(self):
        """layer_loss still works and returns L2 distance from center."""
        from titan_plugin.logic.middle_path import layer_loss, layer_coherence
        tensor = [0.3, 0.7, 0.5, 0.2, 0.8]
        loss = layer_loss(tensor)
        # L2 from center should be positive
        assert loss > 0.0
        # layer_loss should be unchanged — still L2 from center
        expected = math.sqrt(sum((v - 0.5) ** 2 for v in tensor))
        assert loss == pytest.approx(expected, abs=1e-10)
        # coherence should be independent (not 1 - loss)
        coh = layer_coherence(tensor)
        assert 0.0 <= coh <= 1.0

    def test_single_dim_returns_one(self):
        """Single dimension tensor → coherence = 1.0 (no variance possible)."""
        from titan_plugin.logic.middle_path import layer_coherence
        assert layer_coherence([0.7]) == 1.0


# ── T1.2: Individual observables ──────────────────────────────────────

class TestObservables:
    def test_coherence_observable(self):
        """BodyPartObserver.observe() returns correct coherence."""
        from titan_plugin.logic.observables import BodyPartObserver
        obs = BodyPartObserver("test")
        result = obs.observe([0.5, 0.5, 0.5, 0.5, 0.5])
        assert result["coherence"] == pytest.approx(1.0, abs=1e-6)

    def test_magnitude_observable(self):
        """Magnitude: normalized L2 norm. Zero vector → 0, all-1 → 1."""
        from titan_plugin.logic.observables import BodyPartObserver
        obs = BodyPartObserver("test")
        # All zeros → magnitude = 0
        result = obs.observe([0.0, 0.0, 0.0, 0.0, 0.0])
        assert result["magnitude"] == pytest.approx(0.0, abs=1e-6)
        # All ones → magnitude = sqrt(5)/sqrt(5) = 1.0
        result = obs.observe([1.0, 1.0, 1.0, 1.0, 1.0])
        assert result["magnitude"] == pytest.approx(1.0, abs=1e-6)

    def test_velocity_observable(self):
        """Velocity: L2 distance from previous tensor."""
        from titan_plugin.logic.observables import BodyPartObserver
        obs = BodyPartObserver("test")
        # First call: distance from default [0.5]*5
        result = obs.observe([0.5, 0.5, 0.5, 0.5, 0.5])
        assert result["velocity"] == pytest.approx(0.0, abs=1e-6)
        # Second call: move to all-1 → distance = sqrt(5 * 0.25)
        result = obs.observe([1.0, 1.0, 1.0, 1.0, 1.0])
        expected = math.sqrt(5 * 0.25)
        assert result["velocity"] == pytest.approx(expected, abs=1e-4)

    def test_direction_observable(self):
        """Direction: cosine similarity with previous tensor."""
        from titan_plugin.logic.observables import BodyPartObserver
        obs = BodyPartObserver("test")
        # Move to [1,1,1,1,1]
        obs.observe([1.0, 1.0, 1.0, 1.0, 1.0])
        # Move further in same direction — stay at [1,1,1,1,1] (no change)
        result = obs.observe([1.0, 1.0, 1.0, 1.0, 1.0])
        # No change → direction = 1.0 (same direction by convention)
        assert result["direction"] == pytest.approx(1.0, abs=1e-4)

    def test_polarity_observable(self):
        """Polarity: mean - 0.5. Positive = above center, negative = below."""
        from titan_plugin.logic.observables import BodyPartObserver
        obs = BodyPartObserver("test")
        # All at 0.8 → polarity = 0.3
        result = obs.observe([0.8, 0.8, 0.8, 0.8, 0.8])
        assert result["polarity"] == pytest.approx(0.3, abs=1e-4)
        # All at 0.2 → polarity = -0.3
        result = obs.observe([0.2, 0.2, 0.2, 0.2, 0.2])
        assert result["polarity"] == pytest.approx(-0.3, abs=1e-4)


# ── T1.2: ObservableEngine ────────────────────────────────────────────

class TestObservableEngine:
    def test_processes_all_six_parts(self):
        """ObservableEngine.observe_all() returns observables for all 6 parts."""
        from titan_plugin.logic.observables import ObservableEngine, ALL_PARTS
        engine = ObservableEngine()
        tensors = {name: [0.5] * 5 for name in ALL_PARTS}
        result = engine.observe_all(tensors)
        assert set(result.keys()) == set(ALL_PARTS)
        for name, obs in result.items():
            assert set(obs.keys()) == {"coherence", "magnitude", "velocity", "direction", "polarity"}
            assert obs["coherence"] == pytest.approx(1.0, abs=1e-6)


# ── T1.1 + T1.3: Sphere clock coherence wiring ──────────────────────

class TestSphereClockCoherence:
    def test_coherent_tensor_fast_clock(self):
        """High coherence → high velocity → fast contraction."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.05, min_velocity_factor=0.15)
        # High coherence = 0.95
        clock.tick(0.95)
        assert clock.contraction_velocity == pytest.approx(0.05 * 0.95, abs=1e-6)

    def test_incoherent_tensor_slow_clock(self):
        """Low coherence → low velocity (floored at min_velocity_factor)."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.05, min_velocity_factor=0.15)
        # Low coherence = 0.05 (below min_velocity_factor)
        clock.tick(0.05)
        assert clock.contraction_velocity == pytest.approx(0.05 * 0.15, abs=1e-6)

    def test_clock_always_pulses_when_incoherent(self):
        """Even at zero coherence, clock eventually pulses due to velocity floor."""
        from titan_plugin.logic.sphere_clock import SphereClock
        clock = SphereClock("test", base_speed=0.05, min_velocity_factor=0.15)
        # Tick with zero coherence many times — should eventually pulse
        pulse = None
        for _ in range(200):  # 200 ticks should be enough at 15% speed
            result = clock.tick(0.0)
            if result is not None:
                pulse = result
                break
        assert pulse is not None, "Clock should pulse even at zero coherence"
        assert pulse["component"] == "test"
        assert pulse["pulse_count"] == 1


# ── rFP #1 Phase 2: get_observations_30d pure helper ────────────────

class TestGetObservations30D:
    """Pure flattening helper — used by state_register for the 30D topology key."""

    def _sample_dict(self):
        """Canonical observables dict with distinct per-part values for ordering check."""
        keys = ("coherence", "magnitude", "velocity", "direction", "polarity")
        parts = ("inner_body", "inner_mind", "inner_spirit",
                 "outer_body", "outer_mind", "outer_spirit")
        return {
            parts[pi]: {keys[ki]: 0.1 * (pi + 1) + 0.01 * ki for ki in range(5)}
            for pi in range(6)
        }

    def test_returns_exactly_30_floats(self):
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        out = eng.get_observations_30d(self._sample_dict())
        assert len(out) == 30
        assert all(isinstance(x, float) for x in out)

    def test_canonical_order_inner_first_outer_last(self):
        """Part 0 = inner_body (0.10-0.14); part 5 = outer_spirit (0.60-0.64)."""
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        out = eng.get_observations_30d(self._sample_dict())
        # First 5 floats = inner_body's 5 observables
        assert out[0] == pytest.approx(0.10, abs=1e-9)
        assert out[4] == pytest.approx(0.14, abs=1e-9)
        # Last 5 floats = outer_spirit's 5 observables
        assert out[25] == pytest.approx(0.60, abs=1e-9)
        assert out[29] == pytest.approx(0.64, abs=1e-9)

    def test_missing_part_pads_with_zeros(self):
        """Dict missing outer_spirit → last 5 floats are 0.0 (not 0.5)."""
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        partial = self._sample_dict()
        del partial["outer_spirit"]
        out = eng.get_observations_30d(partial)
        assert len(out) == 30
        assert out[25:30] == [0.0] * 5

    def test_empty_dict_returns_all_zeros(self):
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        assert eng.get_observations_30d({}) == [0.0] * 30

    def test_non_dict_input_returns_all_zeros(self):
        """Defensive: None / non-dict input → 30 zeros, no crash."""
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        assert eng.get_observations_30d(None) == [0.0] * 30

    def test_observer_state_not_mutated(self):
        """Pure function — engine's observer _prev_tensor values untouched."""
        from titan_plugin.logic.observables import ObservableEngine
        eng = ObservableEngine()
        before = [list(obs._prev_tensor) for obs in eng.observers.values()]
        eng.get_observations_30d(self._sample_dict())
        after = [list(obs._prev_tensor) for obs in eng.observers.values()]
        assert before == after
