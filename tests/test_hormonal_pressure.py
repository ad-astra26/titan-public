"""Tests for the Hormonal Pressure System (Phase H)."""
import time
import pytest
from titan_plugin.logic.hormonal_pressure import (
    HormonalPressure, HormonalSystem, extract_stimuli,
    DEFAULT_CROSS_TALK, CIRCADIAN, DEFAULT_HORMONE_PARAMS,
)


# ── HormonalPressure unit tests ─────────────────────────────────────

class TestHormonalPressure:
    def _make(self, **kw):
        defaults = dict(
            name="TEST", base_secretion_rate=0.01,
            stimulus_sensitivity=1.0, decay_rate=0.001,
            fire_threshold=0.5, refractory_strength=0.8,
            refractory_decay=0.02,
        )
        defaults.update(kw)
        return HormonalPressure(**defaults)

    def test_pressure_accumulates_over_time(self):
        h = self._make()
        assert h.level == 0.0
        h.accumulate(stimulus=0.5, dt=10.0)
        assert h.level > 0.0

    def test_pressure_decays_naturally(self):
        h = self._make(decay_rate=0.1)
        h.level = 1.0
        h.accumulate(stimulus=0.0, dt=1.0)
        assert h.level < 1.0

    def test_fire_at_threshold(self):
        h = self._make(fire_threshold=0.1)
        # Accumulate enough to fire
        for _ in range(50):
            h.accumulate(stimulus=1.0, dt=1.0)
        assert h.should_fire()
        intensity = h.fire()
        assert intensity > 1.0  # Over threshold
        assert h.fire_count == 1

    def test_refractory_prevents_immediate_refire(self):
        h = self._make(fire_threshold=0.1, refractory_strength=0.9)
        # Build and fire
        for _ in range(50):
            h.accumulate(stimulus=1.0, dt=1.0)
        h.fire()
        # Immediately try again
        h.level = 1.0  # Force high level
        assert not h.should_fire()  # Refractory blocks

    def test_refractory_decays(self):
        h = self._make(refractory_decay=0.5)
        h.refractory = 1.0
        h.accumulate(stimulus=0.0, dt=10.0)
        assert h.refractory < 0.1

    def test_fire_returns_intensity(self):
        h = self._make(fire_threshold=0.5)
        h.level = 1.0  # 2x threshold
        h.refractory = 0.0
        intensity = h.fire()
        assert abs(intensity - 2.0) < 0.01

    def test_adapt_threshold_positive_reward(self):
        h = self._make(fire_threshold=0.5)
        h.adapt_threshold(reward=0.5, lr=0.1)
        assert h.threshold < 0.5  # Lowered

    def test_adapt_threshold_negative_reward(self):
        h = self._make(fire_threshold=0.5)
        h.adapt_threshold(reward=-0.5, lr=0.1)
        assert h.threshold > 0.5  # Raised

    def test_threshold_clamped(self):
        h = self._make(fire_threshold=0.5)
        # Try to push below floor
        for _ in range(100):
            h.adapt_threshold(reward=1.0, lr=0.1)
        assert h.threshold >= 0.1
        # Try to push above ceiling
        for _ in range(100):
            h.adapt_threshold(reward=-1.0, lr=0.1)
        assert h.threshold <= 2.0

    def test_cross_talk_excitation(self):
        h = self._make()
        h.excitors = {"OTHER": 0.5}
        # Accumulate with excitatory cross-talk
        h.accumulate(stimulus=0.5, dt=1.0, other_levels={"OTHER": 1.0})
        level_with_excitation = h.level

        h2 = self._make()
        h2.accumulate(stimulus=0.5, dt=1.0, other_levels={})
        level_without = h2.level

        assert level_with_excitation > level_without

    def test_cross_talk_inhibition(self):
        h = self._make()
        h.inhibitors = {"OTHER": 0.5}
        h.accumulate(stimulus=0.5, dt=1.0, other_levels={"OTHER": 1.0})
        level_with_inhibition = h.level

        h2 = self._make()
        h2.accumulate(stimulus=0.5, dt=1.0, other_levels={})
        level_without = h2.level

        assert level_with_inhibition < level_without

    def test_circadian_modulation(self):
        h1 = self._make()
        h1.accumulate(stimulus=0.5, dt=1.0, circadian_multiplier=1.5)
        level_high = h1.level

        h2 = self._make()
        h2.accumulate(stimulus=0.5, dt=1.0, circadian_multiplier=0.3)
        level_low = h2.level

        assert level_high > level_low

    def test_save_restore_state(self):
        h = self._make()
        h.level = 0.42
        h.threshold = 0.55
        h.refractory = 0.3
        h.fire_count = 7
        state = h.get_state()

        h2 = self._make()
        h2.restore_state(state)
        assert abs(h2.level - 0.42) < 1e-6
        assert abs(h2.threshold - 0.55) < 1e-6
        assert h2.fire_count == 7

    def test_dt_clamped(self):
        """Long dt (e.g., after restart) shouldn't cause explosion."""
        h = self._make()
        h.accumulate(stimulus=1.0, dt=99999.0)
        level_clamped = h.level

        h2 = self._make()
        h2.accumulate(stimulus=1.0, dt=30.0)
        level_30s = h2.level

        # dt=99999 should produce same result as dt=30 (clamped)
        assert abs(level_clamped - level_30s) < 1e-6


# ── HormonalSystem tests ────────────────────────────────────────────

class TestHormonalSystem:
    def _make_system(self, names=None):
        names = names or ["REFLEX", "FOCUS", "CURIOSITY", "CREATIVITY",
                          "EMPATHY", "REFLECTION", "INSPIRATION",
                          "INTUITION", "IMPULSE", "VIGILANCE"]
        return HormonalSystem(program_names=names)

    def test_initializes_all_hormones(self):
        sys = self._make_system()
        assert len(sys.get_levels()) == 10
        for name, level in sys.get_levels().items():
            assert level == 0.0

    def test_accumulate_all(self):
        sys = self._make_system()
        stimuli = {n: 0.5 for n in sys.get_levels()}
        sys.accumulate_all(stimuli, dt=10.0)
        for name, level in sys.get_levels().items():
            assert level > 0.0

    def test_fire_candidates(self):
        sys = self._make_system(["FOCUS"])
        h = sys.get_hormone("FOCUS")
        h.level = 1.0
        h.refractory = 0.0
        candidates = sys.get_fire_candidates()
        assert "FOCUS" in candidates

    def test_fire_resets_pressure(self):
        sys = self._make_system(["FOCUS"])
        h = sys.get_hormone("FOCUS")
        h.level = 1.0
        h.refractory = 0.0
        intensity = sys.fire("FOCUS")
        assert intensity > 0
        assert h.level < 0.2
        assert h.refractory > 0.5

    def test_dreaming_modulates(self):
        sys1 = self._make_system(["REFLECTION"])
        sys1.accumulate_all({"REFLECTION": 0.5}, dt=2.0, is_dreaming=True)
        level_dreaming = sys1.get_levels()["REFLECTION"]

        sys2 = self._make_system(["REFLECTION"])
        sys2.accumulate_all({"REFLECTION": 0.5}, dt=2.0, is_dreaming=False)
        level_awake = sys2.get_levels()["REFLECTION"]

        # REFLECTION should be higher during dreaming (circadian mult 1.5 vs 0.6)
        assert level_dreaming > level_awake

    def test_cross_talk_visible(self):
        sys = self._make_system(["INSPIRATION", "CREATIVITY"])
        # Set INSPIRATION level high
        sys.get_hormone("INSPIRATION").level = 0.5
        # CREATIVITY has INSPIRATION as excitor (+0.4)
        sys.accumulate_all({"CREATIVITY": 0.3, "INSPIRATION": 0.0}, dt=2.0)
        creativity_level = sys.get_levels()["CREATIVITY"]

        sys2 = self._make_system(["INSPIRATION", "CREATIVITY"])
        sys2.accumulate_all({"CREATIVITY": 0.3, "INSPIRATION": 0.0}, dt=2.0)
        creativity_without = sys2.get_levels()["CREATIVITY"]

        assert creativity_level > creativity_without

    def test_save_load(self, tmp_path):
        sys = self._make_system(["FOCUS", "CURIOSITY"])
        sys.get_hormone("FOCUS").level = 0.42
        sys.get_hormone("CURIOSITY").fire_count = 5
        path = str(tmp_path / "hormones.json")
        sys.save(path)

        sys2 = self._make_system(["FOCUS", "CURIOSITY"])
        sys2.load(path)
        assert abs(sys2.get_levels()["FOCUS"] - 0.42) < 1e-6
        assert sys2.get_hormone("CURIOSITY").fire_count == 5


# ── Stimulus Extraction tests ────────────────────────────────────────

class TestStimulusExtraction:
    def _make_obs(self, coherence=0.5, magnitude=0.5, velocity=0.0):
        """Create observables dict with uniform values."""
        parts = ["inner_body", "inner_mind", "inner_spirit",
                 "outer_body", "outer_mind", "outer_spirit"]
        return {p: [coherence, magnitude, velocity, 0.0, 0.5] for p in parts}

    def test_returns_all_programs(self):
        obs = self._make_obs()
        stimuli = extract_stimuli(obs)
        expected = {"REFLEX", "FOCUS", "INTUITION", "IMPULSE", "VIGILANCE",
                    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
                    "INSPIRATION"}
        assert set(stimuli.keys()) == expected

    def test_all_bounded_0_1(self):
        obs = self._make_obs(coherence=1.0, magnitude=1.0, velocity=1.0)
        stimuli = extract_stimuli(obs, topology={"volume": 1.0, "curvature": 5.0})
        for name, val in stimuli.items():
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_curiosity_boredom_grows(self):
        obs = self._make_obs()
        s1 = extract_stimuli(obs, events={"time_since_explore": 100})
        s2 = extract_stimuli(obs, events={"time_since_explore": 3600})
        assert s2["CURIOSITY"] > s1["CURIOSITY"]

    def test_reflex_responds_to_velocity(self):
        obs_calm = self._make_obs(velocity=0.0)
        obs_fast = self._make_obs(velocity=0.8)
        s_calm = extract_stimuli(obs_calm)
        s_fast = extract_stimuli(obs_fast)
        assert s_fast["REFLEX"] > s_calm["REFLEX"]

    def test_empathy_responds_to_social(self):
        obs = self._make_obs()
        s1 = extract_stimuli(obs, events={"time_since_social": 100})
        s2 = extract_stimuli(obs, events={"time_since_social": 3600})
        assert s2["EMPATHY"] > s1["EMPATHY"]
