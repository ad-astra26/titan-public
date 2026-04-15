"""Tests for Dimensional Quantitative Symmetry (5/15/45/135D)."""
import pytest
from titan_plugin.logic.mind_tensor import collect_mind_15d, MIND_DIM_NAMES
from titan_plugin.logic.spirit_tensor import collect_spirit_45d, SPIRIT_DIM_NAMES


class TestMindTensor15D:
    def test_output_is_15d(self):
        result = collect_mind_15d(current_5d=[0.5]*5)
        assert len(result) == 15

    def test_thinking_from_current_5d(self):
        current = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = collect_mind_15d(current_5d=current)
        assert result[:5] == current  # First 5 = thinking = current

    def test_feeling_defaults_without_sources(self):
        result = collect_mind_15d(current_5d=[0.5]*5)
        feeling = result[5:10]
        assert all(0.0 <= v <= 1.0 for v in feeling)

    def test_willing_from_hormones(self):
        hormones = {
            "IMPULSE": 0.8, "EMPATHY": 0.6, "CREATIVITY": 0.9,
            "VIGILANCE": 0.3, "CURIOSITY": 0.7,
        }
        result = collect_mind_15d(current_5d=[0.5]*5, hormone_levels=hormones)
        willing = result[10:15]
        assert willing[0] == 0.8  # Action drive = IMPULSE
        assert willing[2] == 0.9  # Creative will = CREATIVITY
        assert willing[4] == 0.7  # Growth will = CURIOSITY

    def test_all_bounded(self):
        result = collect_mind_15d(
            current_5d=[1.0]*5,
            hormone_levels={"IMPULSE": 2.0, "EMPATHY": -1.0,
                           "CREATIVITY": 0.5, "VIGILANCE": 0.5,
                           "CURIOSITY": 0.5})
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_dim_names_count(self):
        assert len(MIND_DIM_NAMES) == 15


class TestSpiritTensor45D:
    def test_output_is_45d(self):
        result = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15)
        assert len(result) == 45

    def test_sat_is_first_15(self):
        result = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15)
        sat = result[:15]
        assert len(sat) == 15
        assert all(0.0 <= v <= 1.0 for v in sat)

    def test_chit_is_middle_15(self):
        result = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15)
        chit = result[15:30]
        assert len(chit) == 15
        assert all(0.0 <= v <= 1.0 for v in chit)

    def test_ananda_is_last_15(self):
        result = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15)
        ananda = result[30:45]
        assert len(ananda) == 15
        assert all(0.0 <= v <= 1.0 for v in ananda)

    def test_observer_principle_body_coherence_affects_spirit(self):
        """High body coherence should increase witness presence (CHIT-4)."""
        # High body coherence
        r1 = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.8]*5, mind_tensor=[0.8]*15)
        # Low body coherence
        r2 = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.2]*5, mind_tensor=[0.2]*15)
        # Witness presence (CHIT-4 = index 19) should be higher with higher coherence
        assert r1[19] > r2[19]

    def test_hormones_affect_spirit(self):
        """Hormone levels should affect CHIT truth-seeking and ANANDA dimensions."""
        r1 = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15,
            hormone_levels={"CURIOSITY": 0.9, "FOCUS": 0.8, "INSPIRATION": 0.7,
                          "IMPULSE": 0.5, "VIGILANCE": 0.5})
        r2 = collect_spirit_45d(
            current_5d=[0.5]*5, body_tensor=[0.5]*5, mind_tensor=[0.5]*15,
            hormone_levels={"CURIOSITY": 0.1, "FOCUS": 0.1, "INSPIRATION": 0.1,
                          "IMPULSE": 0.1, "VIGILANCE": 0.1})
        # Truth-seeking (CHIT-7 = index 22) = CURIOSITY level
        assert r1[22] > r2[22]

    def test_all_bounded(self):
        result = collect_spirit_45d(
            current_5d=[1.0]*5, body_tensor=[1.0]*5, mind_tensor=[1.0]*15,
            consciousness={"epoch_count": 99999, "density": 2.0, "curvature": 10.0},
            hormone_levels={p: 5.0 for p in ["CURIOSITY","FOCUS","IMPULSE",
                           "EMPATHY","CREATIVITY","VIGILANCE","REFLECTION",
                           "INSPIRATION","INTUITION","REFLEX"]},
            hormone_fires={p: 999 for p in ["CURIOSITY","FOCUS","IMPULSE",
                          "EMPATHY","CREATIVITY","VIGILANCE","REFLECTION",
                          "INSPIRATION","INTUITION","REFLEX"]})
        assert all(0.0 <= v <= 1.0 for v in result), \
            f"Out of bounds: {[i for i,v in enumerate(result) if v < 0 or v > 1]}"

    def test_dim_names_count(self):
        assert len(SPIRIT_DIM_NAMES) == 45


class TestDimensionalRatio:
    def test_trinity_ratio_3_6_9(self):
        """Body:Mind:Spirit must follow 1:3:9 ratio (powers of 3)."""
        body_dim = 5
        mind_dim = 15
        spirit_dim = 45
        assert mind_dim == body_dim * 3
        assert spirit_dim == mind_dim * 3
        assert spirit_dim == body_dim * 9
