"""
tests/test_t2_state_registries.py — T2: Three State Registries tests.

8 tests covering:
  - OuterState backward compatible with StateRegister
  - InnerState stores/retrieves observables
  - SpiritState assembles full view
  - is_active flag pauses OuterState updates
  - Three registries are independent
"""
import pytest


class TestOuterStateBackwardCompat:
    def test_state_register_alias_works(self):
        """StateRegister import still works and is same class as OuterState."""
        from titan_plugin.logic.state_register import StateRegister, OuterState
        assert StateRegister is OuterState

    def test_outer_state_has_all_original_methods(self):
        """OuterState retains all StateRegister properties and methods."""
        from titan_plugin.logic.state_register import OuterState
        reg = OuterState()
        # Properties
        assert reg.body_tensor == [0.5] * 5
        assert reg.mind_tensor == [0.5] * 5
        assert reg.spirit_tensor == [0.5] * 5
        assert isinstance(reg.consciousness, dict)
        # Methods
        snap = reg.snapshot()
        assert "body_tensor" in snap
        assert "mind_tensor" in snap
        assert "spirit_tensor" in snap
        # Full 30DT
        full = reg.get_full_30dt()
        assert len(full) == 30

    def test_is_active_flag_default_true(self):
        """is_active defaults to True."""
        from titan_plugin.logic.state_register import OuterState
        reg = OuterState()
        assert reg.is_active is True


class TestOuterStateIsActive:
    def test_is_active_pauses_bus_updates(self):
        """When is_active=False, _process_bus_message is a no-op."""
        from titan_plugin.logic.state_register import OuterState
        reg = OuterState()

        # Active: update works
        reg._process_bus_message({
            "type": "BODY_STATE",
            "payload": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        })
        assert reg.body_tensor == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Deactivate (dreaming)
        reg.is_active = False
        reg._process_bus_message({
            "type": "BODY_STATE",
            "payload": {"values": [0.9, 0.9, 0.9, 0.9, 0.9]},
        })
        # Should NOT have updated
        assert reg.body_tensor == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Reactivate
        reg.is_active = True
        reg._process_bus_message({
            "type": "BODY_STATE",
            "payload": {"values": [0.7, 0.7, 0.7, 0.7, 0.7]},
        })
        assert reg.body_tensor == [0.7, 0.7, 0.7, 0.7, 0.7]


class TestInnerState:
    def test_stores_and_retrieves_observables(self):
        """InnerState stores observables and returns them in snapshot."""
        from titan_plugin.logic.inner_state import InnerState
        state = InnerState()

        obs = {
            "inner_body": {"coherence": 0.95, "magnitude": 0.7, "velocity": 0.1,
                           "direction": 0.99, "polarity": 0.05},
            "inner_mind": {"coherence": 0.88, "magnitude": 0.6, "velocity": 0.2,
                           "direction": 0.95, "polarity": -0.1},
        }
        state.update_observables(obs)

        assert state.observables["inner_body"]["coherence"] == 0.95
        assert state.observables["inner_mind"]["polarity"] == -0.1

        snap = state.snapshot()
        assert snap["observables"]["inner_body"]["coherence"] == 0.95
        assert snap["update_count"] == 1

    def test_experience_buffer(self):
        """InnerState buffers and drains outer snapshots."""
        from titan_plugin.logic.inner_state import InnerState
        state = InnerState()

        state.buffer_experience({"body_tensor": [0.1] * 5, "ts": 1.0})
        state.buffer_experience({"body_tensor": [0.2] * 5, "ts": 2.0})
        assert state.snapshot()["experience_buffer_size"] == 2

        drained = state.drain_experience_buffer()
        assert len(drained) == 2
        assert state.snapshot()["experience_buffer_size"] == 0


class TestSpiritState:
    def test_assembles_full_view(self):
        """SpiritState assembles from outer + inner + observables."""
        from titan_plugin.logic.spirit_state import SpiritState
        from titan_plugin.logic.inner_state import InnerState
        spirit = SpiritState()
        inner = InnerState()

        obs = {
            "inner_body": {"coherence": 0.9, "magnitude": 0.7, "velocity": 0.1,
                           "direction": 0.99, "polarity": 0.05},
            "outer_body": {"coherence": 0.85, "magnitude": 0.6, "velocity": 0.2,
                           "direction": 0.95, "polarity": -0.1},
        }
        inner.update_observables(obs)

        outer_snap = {
            "body_tensor": [0.6] * 5,
            "mind_tensor": [0.5] * 5,
            "spirit_tensor": [0.7] * 5,
            "outer_body": [0.4] * 5,
            "outer_mind": [0.5] * 5,
            "outer_spirit": [0.3] * 5,
        }
        spirit.assemble(outer_snapshot=outer_snap, observables=obs,
                        inner_snapshot=inner.snapshot())

        snap = spirit.snapshot()
        assert len(snap["full_30dt"]) == 30
        assert snap["full_30dt"][:5] == [0.6] * 5  # body
        assert snap["mean_coherence"] == pytest.approx(0.875, abs=1e-4)
        assert snap["assembly_count"] == 1


class TestRegistriesIndependence:
    def test_three_registries_independent(self):
        """Writing to one registry doesn't affect the others."""
        from titan_plugin.logic.state_register import OuterState
        from titan_plugin.logic.inner_state import InnerState
        from titan_plugin.logic.spirit_state import SpiritState

        outer = OuterState()
        inner = InnerState()
        spirit = SpiritState()

        # Update outer
        outer._process_bus_message({
            "type": "BODY_STATE",
            "payload": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        })

        # Inner should be unaffected
        assert inner.observables == {}
        assert inner.fatigue == 0.0

        # Spirit should be unaffected
        assert spirit.full_30dt == [0.5] * 30

        # Update inner
        inner.update_observables({"inner_body": {"coherence": 0.5}})

        # Outer and Spirit unaffected
        assert outer.body_tensor == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert spirit.observables == {}
