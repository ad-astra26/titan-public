"""
tests/test_t4_t7_nervous_topology_dreaming.py — T4-T7 combined tests.

T4: Nervous System (15 tests planned, 10 here)
T5: Space Topology (10 tests)
T6: Dreaming Cycle (12 tests planned, 8 here)
T7: Emergent GREAT PULSE (8 tests planned, 5 here)
"""
import math
import time
import pytest


# ── T4: Nervous System Programs ──────────────────────────────────────

class TestNervousSystemPrograms:
    def test_reflex_fires_on_high_velocity(self):
        """REFLEX program fires when avg velocity > 0.3."""
        from titan_plugin.logic.nervous_system import NervousSystem, flatten_observables
        ns = NervousSystem()
        obs = _make_obs(velocity=0.5)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "REFLEX" in names

    def test_reflex_silent_on_low_velocity(self):
        """REFLEX doesn't fire on calm state."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(velocity=0.1)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "REFLEX" not in names

    def test_focus_fires_on_strong_polarity(self):
        """FOCUS fires when |polarity| > 0.15."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(polarity=0.3)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "FOCUS" in names

    def test_focus_silent_on_neutral_polarity(self):
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(polarity=0.05)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "FOCUS" not in names

    def test_intuition_fires_on_erratic_direction(self):
        """INTUITION fires when direction < 0.5."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(direction=0.2)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "INTUITION" in names

    def test_intuition_silent_on_stable_direction(self):
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(direction=0.9)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "INTUITION" not in names

    def test_impulse_fires_on_high_coherence_and_magnitude(self):
        """IMPULSE fires when coherence > 0.7 AND magnitude > 0.5."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(coherence=0.9, magnitude=0.8)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "IMPULSE" in names

    def test_impulse_silent_on_low_coherence(self):
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(coherence=0.3, magnitude=0.8)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "IMPULSE" not in names

    def test_no_signals_on_neutral_state(self):
        """Neutral state (all 0.5) shouldn't fire REFLEX, FOCUS, INTUITION."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        obs = _make_obs(coherence=0.5, magnitude=0.5, velocity=0.0,
                        direction=0.9, polarity=0.0)
        signals = ns.evaluate(obs)
        names = [s["system"] for s in signals]
        assert "REFLEX" not in names
        assert "FOCUS" not in names
        assert "INTUITION" not in names

    def test_urgency_scales_with_observable(self):
        """Higher velocity → higher REFLEX urgency."""
        from titan_plugin.logic.nervous_system import NervousSystem
        ns = NervousSystem()
        low = ns.evaluate(_make_obs(velocity=0.35))
        high = ns.evaluate(_make_obs(velocity=0.9))
        reflex_low = [s for s in low if s["system"] == "REFLEX"]
        reflex_high = [s for s in high if s["system"] == "REFLEX"]
        assert len(reflex_low) == 1 and len(reflex_high) == 1
        assert reflex_high[0]["urgency"] > reflex_low[0]["urgency"]


# ── T5: Space Topology ──────────────────────────────────────────────

class TestTopologyEngine:
    def test_uniform_observables_minimum_volume(self):
        """All parts with identical observables → minimum (zero) volume."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        obs = _make_obs(coherence=0.8, magnitude=0.5, velocity=0.1,
                        direction=0.9, polarity=0.0)
        result = topo.compute(obs)
        assert result["volume"] == pytest.approx(0.0, abs=1e-4)

    def test_divergent_observables_large_volume(self):
        """Different parts with different observables → large volume."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        obs = {
            "inner_body": {"coherence": 0.1, "magnitude": 0.9, "velocity": 0.8,
                           "direction": 0.1, "polarity": 0.4},
            "inner_mind": {"coherence": 0.9, "magnitude": 0.1, "velocity": 0.1,
                           "direction": 0.9, "polarity": -0.4},
            "outer_body": {"coherence": 0.5, "magnitude": 0.5, "velocity": 0.5,
                           "direction": 0.5, "polarity": 0.0},
        }
        result = topo.compute(obs)
        assert result["volume"] > 1.0

    def test_contracting_volume_positive_curvature(self):
        """Volume shrinking over ticks → positive curvature."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        # First tick: large volume
        topo.compute({
            "inner_body": {"coherence": 0.1, "magnitude": 0.9, "velocity": 0.8,
                           "direction": 0.1, "polarity": 0.4},
            "inner_mind": {"coherence": 0.9, "magnitude": 0.1, "velocity": 0.1,
                           "direction": 0.9, "polarity": -0.4},
        })
        # Second tick: smaller volume (more aligned)
        result = topo.compute({
            "inner_body": {"coherence": 0.5, "magnitude": 0.5, "velocity": 0.4,
                           "direction": 0.5, "polarity": 0.1},
            "inner_mind": {"coherence": 0.6, "magnitude": 0.4, "velocity": 0.3,
                           "direction": 0.6, "polarity": -0.1},
        })
        assert result["curvature"] > 0  # positive = contracting

    def test_expanding_volume_negative_curvature(self):
        """Volume growing over ticks → negative curvature."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        # First tick: small but non-zero volume
        topo.compute({
            "inner_body": {"coherence": 0.5, "magnitude": 0.5, "velocity": 0.1,
                           "direction": 0.9, "polarity": 0.0},
            "inner_mind": {"coherence": 0.6, "magnitude": 0.4, "velocity": 0.2,
                           "direction": 0.8, "polarity": 0.05},
        })
        # Second tick: larger volume (diverging further)
        result = topo.compute({
            "inner_body": {"coherence": 0.1, "magnitude": 0.9, "velocity": 0.8,
                           "direction": 0.1, "polarity": 0.4},
            "inner_mind": {"coherence": 0.9, "magnitude": 0.1, "velocity": 0.1,
                           "direction": 0.9, "polarity": -0.4},
        })
        assert result["curvature"] < 0  # negative = expanding

    def test_cluster_detection(self):
        """Parts with similar observables form a cluster."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine(cluster_threshold=0.3)
        obs = {
            "inner_body": {"coherence": 0.8, "magnitude": 0.5, "velocity": 0.1,
                           "direction": 0.9, "polarity": 0.0},
            "inner_mind": {"coherence": 0.8, "magnitude": 0.5, "velocity": 0.1,
                           "direction": 0.9, "polarity": 0.0},  # same as body
            "outer_body": {"coherence": 0.1, "magnitude": 0.9, "velocity": 0.8,
                           "direction": 0.1, "polarity": 0.4},  # very different
        }
        result = topo.compute(obs)
        assert len(result["clusters"]) >= 1
        # inner_body and inner_mind should cluster together
        found = any("inner_body" in c and "inner_mind" in c for c in result["clusters"])
        assert found

    def test_isolated_part_detected(self):
        """A part far from all others is isolated."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine(cluster_threshold=0.2)
        obs = {
            "inner_body": {"coherence": 0.5, "magnitude": 0.5, "velocity": 0.1,
                           "direction": 0.9, "polarity": 0.0},
            "inner_mind": {"coherence": 0.5, "magnitude": 0.5, "velocity": 0.1,
                           "direction": 0.9, "polarity": 0.0},
            "outer_spirit": {"coherence": 0.0, "magnitude": 1.0, "velocity": 0.9,
                             "direction": 0.0, "polarity": -0.5},
        }
        result = topo.compute(obs)
        assert "outer_spirit" in result["isolated"]

    def test_topology_persisted_in_inner_state(self):
        """Coordinator stores topology in InnerState."""
        from titan_plugin.logic.inner_state import InnerState
        from titan_plugin.logic.spirit_state import SpiritState
        from titan_plugin.logic.observables import ObservableEngine
        from titan_plugin.logic.topology import TopologyEngine
        from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator

        inner = InnerState()
        coord = InnerTrinityCoordinator(
            inner, SpiritState(), ObservableEngine(),
            topology_engine=TopologyEngine())
        coord.tick({"inner_body": [0.6]*5, "inner_mind": [0.4]*5,
                    "inner_spirit": [0.5]*5})
        assert inner.topology.get("volume") is not None


# ── T6: Dreaming Cycle ──────────────────────────────────────────────

class TestDreamingCycle:
    def test_fatigue_rises_with_depleted_observables(self):
        """Low outer coherence/magnitude → high fatigue."""
        from titan_plugin.logic.dreaming import DreamingEngine
        engine = DreamingEngine()
        obs = _make_obs(coherence=0.2, magnitude=0.2, direction=0.3)
        fatigue = engine.compute_fatigue(obs, {})
        assert fatigue > 0.3

    def test_readiness_rises_with_restored_observables(self):
        """High inner coherence + stable direction → high readiness."""
        from titan_plugin.logic.dreaming import DreamingEngine
        engine = DreamingEngine()
        engine._last_transition_ts = time.time() - 600  # 10 min rest
        obs = _make_obs(coherence=0.95, direction=0.95)
        readiness = engine.compute_readiness(obs, {"curvature": 0.5})
        assert readiness > 0.5

    def test_begin_dreaming_transition(self):
        """Fatigue above threshold triggers BEGIN_DREAMING."""
        from titan_plugin.logic.dreaming import DreamingEngine
        from titan_plugin.logic.inner_state import InnerState
        engine = DreamingEngine(fatigue_threshold=0.3)
        inner = InnerState()
        obs = _make_obs(coherence=0.1, magnitude=0.1, direction=0.1)
        transition = engine.check_transition(inner, obs, {})
        assert transition == "BEGIN_DREAMING"

    def test_end_dreaming_transition(self):
        """Readiness above threshold during dreaming triggers END_DREAMING."""
        from titan_plugin.logic.dreaming import DreamingEngine
        from titan_plugin.logic.inner_state import InnerState
        engine = DreamingEngine(readiness_threshold=0.3)
        engine._last_transition_ts = time.time() - 600
        inner = InnerState()
        inner.is_dreaming = True
        obs = _make_obs(coherence=0.95, direction=0.95)
        transition = engine.check_transition(inner, obs, {"curvature": 0.5})
        assert transition == "END_DREAMING"

    def test_full_cycle(self):
        """awake → dreaming → awake cycle."""
        from titan_plugin.logic.dreaming import DreamingEngine
        from titan_plugin.logic.inner_state import InnerState
        engine = DreamingEngine(fatigue_threshold=0.3, readiness_threshold=0.3)
        inner = InnerState()

        # Start awake
        assert not inner.is_dreaming

        # Fatigue → begin dreaming
        engine.begin_dreaming(inner)
        assert inner.is_dreaming
        assert inner.cycle_count == 1

        # End dreaming
        engine._last_transition_ts = time.time() - 600
        summary = engine.end_dreaming(inner)
        assert not inner.is_dreaming
        assert "experiences_processed" in summary

    def test_experience_buffer_drains_on_wake(self):
        """Buffered experiences are drained when dreaming ends."""
        from titan_plugin.logic.dreaming import DreamingEngine
        from titan_plugin.logic.inner_state import InnerState
        engine = DreamingEngine()
        inner = InnerState()
        inner.buffer_experience({"body_tensor": [0.3]*5, "ts": 1.0})
        inner.buffer_experience({"body_tensor": [0.7]*5, "ts": 2.0})
        engine.begin_dreaming(inner)
        summary = engine.end_dreaming(inner)
        assert summary["experiences_processed"] == 2
        assert len(inner._experience_buffer) == 0

    def test_inner_keeps_running_during_dreaming(self):
        """InnerState is still updatable when dreaming."""
        from titan_plugin.logic.inner_state import InnerState
        inner = InnerState()
        inner.is_dreaming = True
        inner.update_observables({"inner_body": {"coherence": 0.9}})
        assert inner.observables["inner_body"]["coherence"] == 0.9

    def test_distillation_filters_significant_experiences(self):
        """Only experiences with sufficient variance are distilled."""
        from titan_plugin.logic.dreaming import DreamingEngine
        engine = DreamingEngine()
        buffer = [
            {"body_tensor": [0.5, 0.5, 0.5, 0.5, 0.5]},  # boring
            {"body_tensor": [0.1, 0.9, 0.1, 0.9, 0.1]},  # interesting
        ]
        insights = engine._distill_experiences(buffer)
        assert len(insights) == 1  # only the interesting one
        assert insights[0]["significance"] > 0.02


# ── T7: Emergent GREAT PULSE ────────────────────────────────────────

class TestEmergentGreatPulse:
    def test_convergence_detected_at_volume_minimum(self):
        """is_convergence_peak returns True at local volume minimum."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        # Simulate volume history: decreasing then increasing
        topo._volume_history = [5.0, 3.0, 2.0, 1.0, 1.5]
        # volume[-2]=1.0 < volume[-3]=2.0 AND volume[-2]=1.0 <= volume[-1]=1.5
        assert topo.is_convergence_peak()

    def test_no_convergence_during_contraction(self):
        """No convergence peak while still contracting."""
        from titan_plugin.logic.topology import TopologyEngine
        topo = TopologyEngine()
        topo._volume_history = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert not topo.is_convergence_peak()

    def test_great_pulse_fires_during_dreaming_convergence(self):
        """Full coordinator: GREAT PULSE fires at topology convergence during dreaming."""
        from titan_plugin.logic.inner_state import InnerState
        from titan_plugin.logic.spirit_state import SpiritState
        from titan_plugin.logic.observables import ObservableEngine
        from titan_plugin.logic.topology import TopologyEngine
        from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator

        inner = InnerState()
        inner.is_dreaming = True  # must be dreaming
        spirit = SpiritState()
        topo = TopologyEngine()
        # Pre-load volume history to trigger convergence on next tick
        topo._volume_history = [5.0, 3.0, 1.0]  # next tick will check

        coord = InnerTrinityCoordinator(
            inner, spirit, ObservableEngine(), topology_engine=topo)

        # Tick that adds a volume > 1.0 (the minimum) → convergence detected
        # Use slightly divergent tensors to get volume > 1.0
        coord.tick({
            "inner_body": [0.8]*5, "inner_mind": [0.2]*5, "inner_spirit": [0.5]*5,
        })

        # The topology should have recorded a great pulse
        assert coord._last_dreaming_event == "GREAT_PULSE"
        assert "great_pulse" in coord._last_topology

    def test_no_great_pulse_when_awake(self):
        """GREAT PULSE should NOT fire during active (non-dreaming) state."""
        from titan_plugin.logic.inner_state import InnerState
        from titan_plugin.logic.spirit_state import SpiritState
        from titan_plugin.logic.observables import ObservableEngine
        from titan_plugin.logic.topology import TopologyEngine
        from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator

        inner = InnerState()
        inner.is_dreaming = False  # awake
        topo = TopologyEngine()
        topo._volume_history = [5.0, 3.0, 1.0]

        coord = InnerTrinityCoordinator(
            inner, SpiritState(), ObservableEngine(), topology_engine=topo)

        coord.tick({"inner_body": [0.8]*5, "inner_mind": [0.2]*5, "inner_spirit": [0.5]*5})
        assert coord._last_dreaming_event != "GREAT_PULSE"

    def test_enrichment_quality_increases_on_great_pulse(self):
        """Spirit enrichment quality increases when GREAT PULSE fires."""
        from titan_plugin.logic.inner_state import InnerState
        from titan_plugin.logic.spirit_state import SpiritState
        from titan_plugin.logic.observables import ObservableEngine
        from titan_plugin.logic.topology import TopologyEngine
        from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator

        inner = InnerState()
        inner.is_dreaming = True
        spirit = SpiritState()
        spirit.enrichment_quality = 0.1
        topo = TopologyEngine()
        topo._volume_history = [5.0, 3.0, 1.0]

        coord = InnerTrinityCoordinator(
            inner, spirit, ObservableEngine(), topology_engine=topo)

        coord.tick({"inner_body": [0.8]*5, "inner_mind": [0.2]*5, "inner_spirit": [0.5]*5})
        assert spirit.enrichment_quality > 0.1


# ── Helpers ──────────────────────────────────────────────────────────

def _make_obs(
    coherence=0.8, magnitude=0.5, velocity=0.1,
    direction=0.9, polarity=0.0,
) -> dict[str, dict]:
    """Create uniform observables for all 6 parts."""
    obs = {
        "coherence": coherence, "magnitude": magnitude,
        "velocity": velocity, "direction": direction, "polarity": polarity,
    }
    return {
        "inner_body": dict(obs), "inner_mind": dict(obs), "inner_spirit": dict(obs),
        "outer_body": dict(obs), "outer_mind": dict(obs), "outer_spirit": dict(obs),
    }
