"""
tests/test_t3_inner_coordinator.py — T3: Inner Trinity Coordinator tests.

6 tests covering:
  - Coordinator tick computes observables
  - Coordinator buffers outer snapshots
  - SpiritState updated after tick
  - InnerState updated after tick
  - Coordinator works without TitanVM (graceful degradation)
  - Backward compatible with existing spirit_worker behavior
"""
import pytest


def _make_coordinator(with_vm=False):
    """Helper: create a fully wired coordinator."""
    from titan_plugin.logic.inner_state import InnerState
    from titan_plugin.logic.spirit_state import SpiritState
    from titan_plugin.logic.observables import ObservableEngine
    from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator

    inner = InnerState()
    spirit = SpiritState()
    obs_engine = ObservableEngine()

    vm = None
    if with_vm:
        # Mock VM with a tick counter
        class MockVM:
            def __init__(self):
                self.tick_count = 0
            def tick(self):
                self.tick_count += 1
        vm = MockVM()

    coord = InnerTrinityCoordinator(inner, spirit, obs_engine, vm=vm)
    return coord, inner, spirit, obs_engine, vm


class TestCoordinatorTick:
    def test_tick_computes_observables(self):
        """Coordinator.tick() returns observables for all provided parts."""
        coord, inner, spirit, _, _ = _make_coordinator()

        inner_tensors = {
            "inner_body": [0.6, 0.6, 0.6, 0.6, 0.6],
            "inner_mind": [0.4, 0.4, 0.4, 0.4, 0.4],
            "inner_spirit": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
        outer_tensors = {
            "outer_body": [0.7, 0.7, 0.7, 0.7, 0.7],
            "outer_mind": [0.3, 0.3, 0.3, 0.3, 0.3],
            "outer_spirit": [0.5, 0.5, 0.5, 0.5, 0.5],
        }

        result = coord.tick(inner_tensors, outer_tensors)

        # Should have all 6 parts
        assert len(result) == 6
        for name in ["inner_body", "inner_mind", "inner_spirit",
                      "outer_body", "outer_mind", "outer_spirit"]:
            assert name in result
            obs = result[name]
            assert "coherence" in obs
            assert "magnitude" in obs
            assert "velocity" in obs
            assert "direction" in obs
            assert "polarity" in obs

        # Uniform tensors → coherence = 1.0
        assert result["inner_body"]["coherence"] == pytest.approx(1.0, abs=1e-4)

    def test_inner_state_updated_after_tick(self):
        """InnerState.observables is populated after coordinator.tick()."""
        coord, inner, _, _, _ = _make_coordinator()

        inner_tensors = {
            "inner_body": [0.6] * 5,
            "inner_mind": [0.4] * 5,
            "inner_spirit": [0.5] * 5,
        }
        coord.tick(inner_tensors)

        assert len(inner.observables) == 6  # all 6 parts (outer defaults)
        assert inner.observables["inner_body"]["coherence"] == pytest.approx(1.0, abs=1e-4)
        assert inner.snapshot()["update_count"] >= 1

    def test_spirit_state_updated_after_tick(self):
        """SpiritState is assembled after coordinator.tick()."""
        coord, _, spirit, _, _ = _make_coordinator()

        inner_tensors = {
            "inner_body": [0.8] * 5,
            "inner_mind": [0.3] * 5,
            "inner_spirit": [0.5] * 5,
        }
        coord.tick(inner_tensors)

        snap = spirit.snapshot()
        assert snap["assembly_count"] >= 1
        assert len(snap["observables"]) == 6


class TestCoordinatorOuterSnapshot:
    def test_buffers_outer_snapshots(self):
        """Coordinator.on_outer_snapshot() buffers in InnerState."""
        coord, inner, _, _, _ = _make_coordinator()

        snapshot = {"body_tensor": [0.6] * 5, "mind_tensor": [0.4] * 5,
                    "spirit_tensor": [0.5] * 5, "ts": 1234.0}
        coord.on_outer_snapshot(snapshot)

        assert inner.snapshot()["experience_buffer_size"] == 1

        coord.on_outer_snapshot({"ts": 1235.0})
        assert inner.snapshot()["experience_buffer_size"] == 2


class TestCoordinatorGracefulDegradation:
    def test_works_without_vm(self):
        """Coordinator functions normally when vm=None."""
        coord, inner, spirit, _, vm = _make_coordinator(with_vm=False)
        assert vm is None

        result = coord.tick({"inner_body": [0.5] * 5, "inner_mind": [0.5] * 5,
                             "inner_spirit": [0.5] * 5})
        assert len(result) == 6
        stats = coord.get_stats()
        assert stats["has_vm"] is False
        assert stats["tick_count"] == 1


class TestCoordinatorConvenienceMethods:
    def test_tick_inner_only_returns_coherences(self):
        """tick_inner_only returns observables + coherences dict."""
        coord, inner, _, _, _ = _make_coordinator()

        obs, coherences = coord.tick_inner_only(
            [0.6] * 5, [0.4] * 5, [0.5] * 5,
        )

        assert "inner_body" in obs
        assert "inner_mind" in obs
        assert "inner_spirit" in obs
        assert coherences["inner_body"] == pytest.approx(1.0, abs=1e-4)
        assert coherences["inner_mind"] == pytest.approx(1.0, abs=1e-4)

        # InnerState should have the observables
        assert "inner_body" in inner.observables
