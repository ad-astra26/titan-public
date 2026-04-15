"""Tests for the Dual-Layer Inner Memory System (Phase M)."""
import time
import pytest
from titan_plugin.logic.inner_memory import InnerMemoryStore


@pytest.fixture
def mem(tmp_path):
    return InnerMemoryStore(str(tmp_path / "test_inner_memory.db"))


class TestInnerMemorySchema:
    def test_init_creates_schema(self, mem):
        stats = mem.get_stats()
        assert "hormone_snapshots" in stats
        assert "program_fires" in stats
        assert "action_chains" in stats
        assert "creative_works" in stats
        assert "event_markers" in stats
        assert all(v == 0 for v in stats.values())


class TestHormoneSnapshots:
    def test_record_and_retrieve(self, mem):
        mem.record_hormone_snapshot(
            epoch_id=100,
            levels={"CURIOSITY": 0.5, "EMPATHY": 0.3},
            thresholds={"CURIOSITY": 0.5, "EMPATHY": 0.5},
            refractory={"CURIOSITY": 0.0, "EMPATHY": 0.1},
            fired=["CURIOSITY"],
            stimuli={"CURIOSITY": 0.8, "EMPATHY": 0.2})
        assert mem.get_stats()["hormone_snapshots"] == 1

    def test_hormone_history(self, mem):
        mem.record_hormone_snapshot(
            epoch_id=1, levels={"FOCUS": 0.3}, thresholds={},
            refractory={}, fired=[], stimuli={})
        mem.record_hormone_snapshot(
            epoch_id=2, levels={"FOCUS": 0.6}, thresholds={},
            refractory={}, fired=["FOCUS"], stimuli={})
        history = mem.get_hormone_history("FOCUS", limit=10)
        assert len(history) == 2
        assert history[0]["level"] == 0.6  # Most recent first
        assert history[0]["fired"] is True


class TestProgramFires:
    def test_record_fire(self, mem):
        mem.record_program_fire(
            program="CURIOSITY", layer="outer",
            intensity=1.5, pressure=0.75, threshold=0.5,
            stimulus=0.8, body=[0.5]*5, mind=[0.5]*5, spirit=[0.5]*5)
        assert mem.get_stats()["program_fires"] == 1

    def test_get_recent_fires(self, mem):
        for prog in ("CURIOSITY", "REFLEX", "CURIOSITY"):
            mem.record_program_fire(
                program=prog, layer="inner",
                intensity=1.0, pressure=0.6, threshold=0.5)
        all_fires = mem.get_recent_fires()
        assert len(all_fires) == 3
        curiosity_fires = mem.get_recent_fires(program="CURIOSITY")
        assert len(curiosity_fires) == 2


class TestActionChains:
    def test_record_action(self, mem):
        mem.record_action_chain(
            impulse_id=1, triggering_program="CREATIVITY",
            posture="create", helper="art_generate",
            success=True, score=0.85,
            reasoning="Good art",
            trinity_before={"body": [0.5]*5})
        assert mem.get_stats()["action_chains"] == 1

    def test_get_action_patterns(self, mem):
        mem.record_action_chain(
            impulse_id=1, triggering_program="CURIOSITY",
            posture="explore", helper="web_search",
            success=True, score=0.9)
        mem.record_action_chain(
            impulse_id=2, triggering_program="CREATIVITY",
            posture="create", helper="art_generate",
            success=True, score=0.7)
        patterns = mem.get_action_patterns(helper="web_search")
        assert len(patterns) == 1
        assert patterns[0]["score"] == 0.9

    def test_get_best_actions(self, mem):
        mem.record_action_chain(
            impulse_id=1, triggering_program="X",
            posture="a", helper="h", success=True, score=0.9)
        mem.record_action_chain(
            impulse_id=2, triggering_program="X",
            posture="b", helper="h", success=False, score=0.3)
        best = mem.get_best_actions(min_score=0.7)
        assert len(best) == 1
        assert best[0]["score"] == 0.9


class TestCreativeWorks:
    def test_record_work(self, mem):
        mem.record_creative_work(
            work_type="art", file_path="/tmp/test.jpg",
            triggering_program="CREATIVITY", posture="create",
            assessment_score=0.85, hormone_level=0.6)
        assert mem.get_stats()["creative_works"] == 1


class TestEventMarkers:
    def test_record_event(self, mem):
        mem.record_event("explore", program="CURIOSITY",
                         details={"query": "sacred geometry"})
        assert mem.get_stats()["event_markers"] == 1

    def test_time_since_last_default(self, mem):
        elapsed = mem.time_since_last("explore")
        assert elapsed >= 3600.0  # Default when no events

    def test_time_since_last_with_event(self, mem):
        mem.record_event("explore", program="CURIOSITY")
        elapsed = mem.time_since_last("explore")
        assert elapsed < 2.0  # Just recorded

    def test_time_since_last_multiple_types(self, mem):
        mem.record_event("explore")
        mem.record_event("social")
        assert mem.time_since_last("explore") < 2.0
        assert mem.time_since_last("social") < 2.0
        assert mem.time_since_last("create") >= 3600.0  # Never happened
