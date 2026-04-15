"""Tests for Experience Playground Framework (P1+P2)."""
import asyncio
import pytest


# ── Test Plugin ──────────────────────────────────────────────────────

class MockExperiencePlugin:
    """Simple test plugin that counts stimuli."""

    name = "test_experience"
    description = "Test plugin for framework validation"
    difficulty_levels = 3

    def __init__(self, **kwargs):
        self._current_level = 1
        self._total_stimuli = 0
        self._correct_responses = 0
        self._session_history = []
        self._stimulus_counter = 0

    async def generate_stimulus(self):
        self._stimulus_counter += 1
        return {
            "content": f"stimulus_{self._stimulus_counter}",
            "type": "test",
            "level": self._current_level,
            "expected": {"answer": "correct"},
            "metadata": {},
        }

    def compute_perturbation(self, stimulus):
        return {
            "inner_body": [0.1] * 5,
            "inner_mind": [0.2] * 15,
            "inner_spirit": [0.3] * 45,
            "outer_body": [0.1] * 5,
            "outer_mind": [0.2] * 15,
            "outer_spirit": [0.3] * 45,
            "hormone_stimuli": {"CURIOSITY": 0.3},
        }

    async def evaluate_response(self, stimulus, response):
        return {
            "score": 0.9,
            "feedback": "Good response",
            "correction": None,
            "reinforcement": {"association": "strong"},
        }

    def should_advance_level(self):
        if self._total_stimuli < 10:
            return False
        recent = self._session_history[-20:]
        if len(recent) < 10:
            return False
        correct = sum(1 for r in recent if r.get("score", 0) > 0.7)
        return correct / len(recent) > 0.8

    def get_stats(self):
        return {
            "name": self.name,
            "description": self.description,
            "level": self._current_level,
            "max_level": self.difficulty_levels,
            "total_stimuli": self._total_stimuli,
            "correct": self._correct_responses,
            "accuracy": round(self._correct_responses /
                              max(1, self._total_stimuli), 3),
            "history_size": len(self._session_history),
        }


# ── Framework Tests ──────────────────────────────────────────────────

class TestExperiencePlugin:
    """Test the base plugin interface."""

    def test_plugin_has_required_attributes(self):
        from titan_plugin.logic.experience_playground import ExperiencePlugin
        plugin = ExperiencePlugin()
        assert hasattr(plugin, 'name')
        assert hasattr(plugin, 'description')
        assert hasattr(plugin, 'difficulty_levels')
        assert hasattr(plugin, '_current_level')
        assert hasattr(plugin, '_total_stimuli')

    def test_plugin_stats(self):
        from titan_plugin.logic.experience_playground import ExperiencePlugin
        plugin = ExperiencePlugin()
        stats = plugin.get_stats()
        assert "name" in stats
        assert "level" in stats
        assert "accuracy" in stats
        assert stats["level"] == 1
        assert stats["accuracy"] == 0.0

    def test_should_advance_needs_minimum_stimuli(self):
        from titan_plugin.logic.experience_playground import ExperiencePlugin
        plugin = ExperiencePlugin()
        assert not plugin.should_advance_level()  # No stimuli yet


class TestExperiencePlayground:
    """Test the playground runner."""

    def test_register_plugin(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        plugin = MockExperiencePlugin()
        pg.register(plugin)
        assert "test_experience" in pg._plugins
        assert len(pg.list_experiences()) == 1

    def test_list_experiences(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        pg.register(MockExperiencePlugin())
        experiences = pg.list_experiences()
        assert len(experiences) == 1
        assert experiences[0]["name"] == "test_experience"

    def test_is_active_default_false(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        assert not pg.is_active

    @pytest.mark.asyncio
    async def test_run_session_unknown_experience(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        result = await pg.run_session("nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_session_basic(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        pg.register(MockExperiencePlugin())
        result = await pg.run_session("test_experience",
                                      num_stimuli=3, pause_between=0.1)
        assert result["experience"] == "test_experience"
        assert result["stimuli_count"] == 3
        assert result["accuracy"] > 0.5
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_session_records_scores(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        pg.register(MockExperiencePlugin())
        result = await pg.run_session("test_experience",
                                      num_stimuli=5, pause_between=0.1)
        for r in result["results"]:
            assert "score" in r
            assert "stimulus" in r
            assert "feedback" in r
            assert r["score"] == 0.9

    @pytest.mark.asyncio
    async def test_session_updates_plugin_stats(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        plugin = MockExperiencePlugin()
        pg.register(plugin)
        await pg.run_session("test_experience",
                             num_stimuli=5, pause_between=0.1)
        assert plugin._total_stimuli == 5
        assert plugin._correct_responses == 5  # All scored 0.9 > 0.7

    @pytest.mark.asyncio
    async def test_level_advancement(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        plugin = MockExperiencePlugin()
        pg.register(plugin)
        # Run 15 stimuli (>10 required, all scoring 0.9)
        result = await pg.run_session("test_experience",
                                      num_stimuli=15, pause_between=0.05)
        # Should have advanced (80%+ accuracy after 10+ stimuli)
        assert result["level_end"] > result["level_start"]

    @pytest.mark.asyncio
    async def test_not_active_after_session(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        pg.register(MockExperiencePlugin())
        await pg.run_session("test_experience",
                             num_stimuli=2, pause_between=0.1)
        assert not pg.is_active

    def test_get_stats(self):
        from titan_plugin.logic.experience_playground import ExperiencePlayground
        pg = ExperiencePlayground()
        pg.register(MockExperiencePlugin())
        stats = pg.get_stats()
        assert "registered_experiences" in stats
        assert "test_experience" in stats["registered_experiences"]
        assert stats["total_sessions"] == 0

    def test_perturbation_is_130d(self):
        """Verify perturbation covers full 130D Trinity."""
        plugin = MockExperiencePlugin()
        perturbation = plugin.compute_perturbation({"content": "test"})
        total_dims = (
            len(perturbation["inner_body"]) +
            len(perturbation["inner_mind"]) +
            len(perturbation["inner_spirit"]) +
            len(perturbation["outer_body"]) +
            len(perturbation["outer_mind"]) +
            len(perturbation["outer_spirit"])
        )
        assert total_dims == 130
        assert "hormone_stimuli" in perturbation
