"""
tests/test_reasoning_observables_feed.py

Regression test for BUG-REASONING-TIER1-OBSERVABLES-STARVED (2026-06-11).

The Phase C NNS-loop migration read the reasoning observation's tier1
observables via ``coordinator.observable_engine.get_observables()``, but:

  * the InnerTrinityCoordinator stores the engine as ``.observables``
    (NOT ``.observable_engine`` — inner_coordinator.py:60), and
  * ObservableEngine exposes no ``get_observables``/``snapshot`` method.

So the ``observables`` dict passed to ObservationSpace.update was ALWAYS ``{}``
→ tier1 (observation[:30]) was all-zeros every tick → the COMPARE primitive
(inner trinity observation[:15] vs outer trinity observation[15:30]) was never
``significant`` → the +0.1 confidence boost never fired → ReasoningEngine never
crossed the 0.6 commit threshold (commit_rate=0 fleet-wide, 231k conclusions /
0 commits, verified live T1 2026-06-11).

The fix (cognitive_worker.py NNS loop) reads the live observables the
coordinator already maintains on ``coordinator.inner.observables``. These tests
lock the data chain (observables → tier1 → COMPARE.significant) and the exact
attribute topology the bug got wrong.
"""
import numpy as np


def _realistic_observables() -> dict:
    """Compute real 30D observables from divergent inner/outer trinity tensors
    via the real ObservableEngine — the same shape coordinator.tick() produces
    and stores on inner.observables."""
    from titan_hcl.logic.observables import ObservableEngine
    eng = ObservableEngine()
    rng = np.random.RandomState(7)
    # Inner trinity low/coherent, outer trinity high/divergent → inner != outer.
    tensors = {
        "inner_body": list(rng.uniform(0.10, 0.30, 5)),
        "inner_mind": list(rng.uniform(0.10, 0.30, 15)),
        "inner_spirit": list(rng.uniform(0.10, 0.30, 45)),
        "outer_body": list(rng.uniform(0.60, 0.95, 5)),
        "outer_mind": list(rng.uniform(0.60, 0.95, 15)),
        "outer_spirit": list(rng.uniform(0.60, 0.95, 45)),
    }
    # observe twice so velocity/direction (prev-tensor based) are populated,
    # mirroring a steadily-ticking engine.
    eng.observe_all(tensors)
    return eng.observe_all(tensors)


class TestObservablesFeedTier1:
    def test_real_observables_populate_tier1_nonzero(self):
        """A populated observables dict → tier1 (obs[:30]) carries real values."""
        from titan_hcl.logic.observation_space import ObservationSpace
        space = ObservationSpace()
        space.update(observables=_realistic_observables())
        enriched = space.build_input("enriched")
        assert enriched.shape[0] >= 30
        assert np.count_nonzero(enriched[:30]) > 0, \
            "tier1 must be populated when observables are supplied"

    def test_empty_observables_zero_tier1_is_the_bug_signature(self):
        """observables={} (what the buggy _nn_obs produced) → tier1 all-zeros.
        This is the exact starvation signature the fix removes upstream."""
        from titan_hcl.logic.observation_space import ObservationSpace
        space = ObservationSpace()
        space.update(observables={})
        enriched = space.build_input("enriched")
        assert np.count_nonzero(enriched[:30]) == 0


class TestCompareFiresOnRealObservables:
    """COMPARE is the only primitive that emits the +0.1 strong confidence
    signal from the observation alone, and it reads ONLY tier1."""

    def test_compare_significant_with_real_observables(self):
        from titan_hcl.logic.observation_space import ObservationSpace
        from titan_hcl.logic import reasoning as R
        space = ObservationSpace()
        space.update(observables=_realistic_observables())
        enriched = space.build_input("enriched")
        # ACh ~ live T1; threshold = 0.3 + (1-ACh)*0.4
        res = R._primitive_compare(enriched, [], {"ACh": 0.54})
        assert res["significant"] is True, \
            "COMPARE must be able to fire when tier1 carries divergent " \
            "inner/outer observables — this is the +0.1 commit path"

    def test_compare_never_significant_on_zero_tier1(self):
        """With the starved tier1 (the bug), COMPARE can never be significant."""
        from titan_hcl.logic.observation_space import ObservationSpace
        from titan_hcl.logic import reasoning as R
        space = ObservationSpace()
        space.update(observables={})
        enriched = space.build_input("enriched")
        res = R._primitive_compare(enriched, [], {"ACh": 0.54})
        assert res["significant"] is False


class TestFixSourceAttributeTopology:
    """Lock the exact attribute the fix relies on vs the one the bug read."""

    def test_inner_state_exposes_observables_dict(self):
        from titan_hcl.logic.inner_state import InnerState
        st = InnerState()
        assert st.observables == {}  # pre-tick
        obs = _realistic_observables()
        st.update_observables(obs)
        assert st.observables == obs  # exactly what the fix reads

    def test_coordinator_has_inner_observables_not_observable_engine(self):
        """The bug read coordinator.observable_engine (never existed); the fix
        reads coordinator.inner.observables (the live dict)."""
        from titan_hcl.logic.inner_state import InnerState
        from titan_hcl.logic.spirit_state import SpiritState
        from titan_hcl.logic.observables import ObservableEngine
        from titan_hcl.logic.inner_coordinator import InnerTrinityCoordinator
        coord = InnerTrinityCoordinator(
            InnerState(), SpiritState(), ObservableEngine())
        # The buggy lookup target — never an attribute on the coordinator:
        assert getattr(coord, "observable_engine", None) is None
        # The fix's source — present, and a dict:
        assert getattr(coord, "inner", None) is not None
        assert isinstance(coord.inner.observables, dict)
        # And the engine truly lives under `.observables`, not `.observable_engine`:
        assert coord.observables is not None
