"""Tests for the Experience Orchestrator — general-purpose experiential learning."""

import os
import tempfile
import pytest
from titan_plugin.logic.experience_orchestrator import (
    ExperienceOrchestrator,
    ExperienceBias,
    ExperiencePlugin,
    _cosine_sim,
)
from titan_plugin.logic.experience_plugins import (
    ArcPuzzlePlugin,
    LanguageLearningPlugin,
    CreativeExpressionPlugin,
    CommunicationPlugin,
)


@pytest.fixture
def tmp_db():
    """Create a temporary DB path for testing."""
    with tempfile.TemporaryDirectory() as td:
        yield os.path.join(td, "test_eo.db")


@pytest.fixture
def orchestrator(tmp_db):
    """Create an orchestrator with all plugins registered."""
    eo = ExperienceOrchestrator(db_path=tmp_db)
    eo.register_plugin(ArcPuzzlePlugin())
    eo.register_plugin(LanguageLearningPlugin())
    eo.register_plugin(CreativeExpressionPlugin())
    eo.register_plugin(CommunicationPlugin())
    return eo


# ── Plugin Registration ───────────────────────────────────────────

class TestPluginRegistration:
    def test_register_plugin(self, orchestrator):
        assert "arc_puzzle" in orchestrator._plugins
        assert "language" in orchestrator._plugins
        assert "creative" in orchestrator._plugins
        assert "communication" in orchestrator._plugins

    def test_register_duplicate(self, orchestrator):
        orchestrator.register_plugin(ArcPuzzlePlugin())
        assert len(orchestrator._plugins) == 4  # no duplicate

    def test_stats_show_plugins(self, orchestrator):
        stats = orchestrator.get_stats()
        assert set(stats["plugins"]) == {"arc_puzzle", "language", "creative", "communication"}


# ── Phase 1: Record ───────────────────────────────────────────────

class TestRecord:
    def test_record_basic(self, orchestrator):
        rid = orchestrator.record_outcome(
            domain="arc_puzzle",
            perception_features=[0.5] * 15,
            inner_state_132d=[0.5] * 132,
            hormonal_snapshot={"CURIOSITY": 0.7},
            action_taken="ACTION1",
            outcome_score=0.8,
            epoch_id=100,
        )
        assert rid > 0
        stats = orchestrator.get_stats()
        assert stats["total_records"] == 1
        assert stats["undistilled"] == 1

    def test_record_multiple_domains(self, orchestrator):
        for domain in ["arc_puzzle", "language", "creative"]:
            orchestrator.record_outcome(
                domain=domain,
                perception_features=[0.5] * 10,
                inner_state_132d=[0.5] * 132,
                hormonal_snapshot={},
                action_taken="test",
                outcome_score=0.6,
            )
        assert orchestrator.get_stats()["total_records"] == 3

    def test_record_with_context(self, orchestrator):
        rid = orchestrator.record_outcome(
            domain="language",
            perception_features=[0.3] * 10,
            inner_state_132d=[0.4] * 132,
            hormonal_snapshot={"CREATIVITY": 0.9},
            action_taken="self_express",
            outcome_score=0.9,
            context={"word": "warm", "sentence": "I am warm"},
        )
        assert rid > 0


# ── Phase 2: Distill ──────────────────────────────────────────────

class TestDistill:
    def _seed_records(self, eo, domain, n=10, score_range=(0.3, 0.9)):
        import random
        random.seed(42)
        for i in range(n):
            score = score_range[0] + (score_range[1] - score_range[0]) * (i / max(1, n - 1))
            eo.record_outcome(
                domain=domain,
                perception_features=[0.5 + i * 0.01] * 15,
                inner_state_132d=[0.5 + i * 0.005] * 132,
                hormonal_snapshot={"CURIOSITY": 0.5 + i * 0.02},
                action_taken=f"ACTION{(i % 3) + 1}",
                outcome_score=score,
                epoch_id=i,
            )

    def test_distill_empty(self, orchestrator):
        insights = orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        assert insights == []

    def test_distill_produces_wisdom(self, orchestrator):
        self._seed_records(orchestrator, "arc_puzzle", n=10)
        insights = orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        assert len(insights) == 1
        assert insights[0]["domain"] == "arc_puzzle"
        assert insights[0]["confidence"] > 0
        assert insights[0]["experience_count"] == 10
        assert orchestrator.get_stats()["total_wisdom"] == 1

    def test_distill_marks_records(self, orchestrator):
        self._seed_records(orchestrator, "language", n=5)
        orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        assert orchestrator.get_stats()["undistilled"] == 0

    def test_distill_updates_action_stats(self, orchestrator):
        self._seed_records(orchestrator, "arc_puzzle", n=9)
        orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        assert orchestrator.get_stats()["total_action_stats"] >= 3  # ACTION1, ACTION2, ACTION3

    def test_distill_multiple_domains(self, orchestrator):
        self._seed_records(orchestrator, "arc_puzzle", n=5)
        self._seed_records(orchestrator, "language", n=5)
        insights = orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        assert len(insights) == 2
        domains = {i["domain"] for i in insights}
        assert domains == {"arc_puzzle", "language"}

    def test_distill_idempotent(self, orchestrator):
        self._seed_records(orchestrator, "arc_puzzle", n=5)
        orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=100)
        insights2 = orchestrator.distill_cycle(dream_cycle=2, current_epoch_id=200)
        assert insights2 == []  # already distilled


# ── Phase 3: Bias ─────────────────────────────────────────────────

class TestBias:
    def _seed_and_distill(self, eo, domain="arc_puzzle", n=20):
        for i in range(n):
            score = 0.8 if i % 2 == 0 else 0.2
            eo.record_outcome(
                domain=domain,
                perception_features=[0.5] * 15,
                inner_state_132d=[0.5] * 132,
                hormonal_snapshot={},
                action_taken="ACTION1" if score > 0.5 else "ACTION2",
                outcome_score=score,
            )
        eo.distill_cycle(dream_cycle=1, current_epoch_id=100)

    def test_bias_no_data(self, orchestrator):
        bias = orchestrator.get_experience_bias(
            domain="arc_puzzle",
            current_perception=[0.5] * 15,
            current_inner_state=[0.5] * 132,
            candidate_actions=["ACTION1"],
        )
        assert bias.confidence == 0.0
        assert bias.relevant_experiences == 0

    def test_bias_with_history(self, orchestrator):
        self._seed_and_distill(orchestrator)
        bias = orchestrator.get_experience_bias(
            domain="arc_puzzle",
            current_perception=[0.5] * 15,
            current_inner_state=[0.5] * 132,
            candidate_actions=["ACTION1", "ACTION2"],
        )
        assert bias.confidence > 0
        assert bias.relevant_experiences > 0
        assert "ACTION1" in bias.action_scores
        assert bias.action_scores["ACTION1"] > bias.action_scores["ACTION2"]

    def test_bias_threshold_modulation(self, orchestrator):
        bias = ExperienceBias(
            confidence=0.8, success_rate=0.9, relevant_experiences=20
        )
        adjusted = bias.apply_to_threshold(0.5)
        assert adjusted < 0.5  # high success → lower threshold

        bias2 = ExperienceBias(
            confidence=0.8, success_rate=0.1, relevant_experiences=20
        )
        adjusted2 = bias2.apply_to_threshold(0.5)
        assert adjusted2 > 0.5  # low success → higher threshold

    def test_bias_insufficient_data_no_change(self, orchestrator):
        bias = ExperienceBias(
            confidence=0.1, success_rate=0.9, relevant_experiences=1
        )
        assert bias.apply_to_threshold(0.5) == 0.5  # not enough data

    def test_full_record_distill_bias_cycle(self, orchestrator):
        # Record → Distill → Bias (full loop)
        for i in range(15):
            orchestrator.record_outcome(
                domain="language",
                perception_features=[0.6] * 10,
                inner_state_132d=[0.6] * 132,
                hormonal_snapshot={"CREATIVITY": 0.8},
                action_taken="self_express",
                outcome_score=0.85,
            )

        insights = orchestrator.distill_cycle(dream_cycle=1, current_epoch_id=50)
        assert len(insights) == 1

        bias = orchestrator.get_experience_bias(
            domain="language",
            current_perception=[0.6] * 10,
            current_inner_state=[0.6] * 132,
            candidate_actions=["self_express"],
        )
        assert bias.confidence > 0.3
        assert bias.success_rate > 0.5
        threshold = bias.apply_to_threshold(0.5)
        assert threshold < 0.5  # success should lower threshold


# ── Plugin Tests ──────────────────────────────────────────────────

class TestPlugins:
    def test_arc_perception_key(self):
        """ArcPuzzle: 20D = Body(5) + Mind Thinking(5) + Spirit SAT(5) + Outer Body(5)."""
        p = ArcPuzzlePlugin()
        # Full inner_state path (preferred)
        key = p.extract_perception_key({
            "inner_state": [0.1] * 5 + [0.6] * 15 + [0.3] * 45 + [0.8] * 5 + [0.5] * 62,
        })
        assert len(key) == 20
        assert key[0] == pytest.approx(0.1)   # Inner Body[0]
        assert key[5] == pytest.approx(0.6)   # Inner Mind Thinking[0]
        assert key[10] == pytest.approx(0.3)  # Inner Spirit SAT[0]
        assert key[15] == pytest.approx(0.8)  # Outer Body[0]
        # Fallback path (pre-sliced fields)
        key2 = p.extract_perception_key({
            "inner_body": [0.1, 0.2, 0.3, 0.4, 0.5],
            "inner_mind": [0.6, 0.7, 0.8, 0.9, 1.0],
            "inner_spirit": [0.1] * 5,
            "spatial_features": [0.9] * 5,
        })
        assert len(key2) == 20

    def test_arc_outcome_score(self):
        p = ArcPuzzlePlugin()
        assert p.compute_outcome_score({"levels_completed": 3, "total_levels": 7}) == pytest.approx(3 / 7)
        assert p.compute_outcome_score({"reward": 25.0}) == pytest.approx(25 / 30)

    def test_language_perception_key(self):
        """Language: 30D = Body(5) + Mind(15) + Outer Mind Feeling(5) + Hormones(5)."""
        p = LanguageLearningPlugin()
        # Full inner_state path
        key = p.extract_perception_key({
            "inner_state": [0.1] * 5 + [0.5] * 15 + [0.3] * 45 + [0.7] * 5 + [0.4] * 5 + [0.8] * 5 + [0.5] * 52,
            "intent_hormones": {"CURIOSITY": 0.9, "CREATIVITY": 0.8},
        })
        assert len(key) == 30
        assert key[0] == pytest.approx(0.1)    # Inner Body[0]
        assert key[5] == pytest.approx(0.5)    # Inner Mind[0]
        assert key[20] == pytest.approx(0.8)   # Outer Mind Feeling[0] (state[75])
        assert key[25] == pytest.approx(0.9)   # CURIOSITY (first hormone)

    def test_creative_perception_key(self):
        """Creative: 30D = MindFeel(5) + OuterBody(5) + OuterMindFeel(5) + Hormones(5) + Body(5) + Spirit(5)."""
        p = CreativeExpressionPlugin()
        key = p.extract_perception_key({
            "inner_state": [0.2] * 5 + [0.4] * 5 + [0.6] * 5 + [0.5] * 5 + [0.3] * 45 + [0.8] * 5 + [0.7] * 5 + [0.9] * 5 + [0.5] * 52,
            "intent_hormones": {"CURIOSITY": 0.7},
        })
        assert len(key) == 30
        assert key[0] == pytest.approx(0.6)    # Inner Mind Feeling[0] (state[10])
        assert key[5] == pytest.approx(0.8)    # Outer Body[0] (state[65])

    def test_communication_perception_key(self):
        """Communication: 20D = Body(5) + Mind Feeling(5) + Outer Mind Feeling(5) + Hormones(5)."""
        p = CommunicationPlugin()
        key = p.extract_perception_key({
            "inner_state": [0.1] * 5 + [0.4] * 5 + [0.9] * 5 + [0.5] * 5 + [0.3] * 45 + [0.7] * 5 + [0.6] * 5 + [0.8] * 5 + [0.5] * 52,
            "intent_hormones": {"EMPATHY": 0.95, "CURIOSITY": 0.7},
        })
        assert len(key) == 20
        assert key[0] == pytest.approx(0.1)    # Inner Body[0]
        assert key[5] == pytest.approx(0.9)    # Inner Mind Feeling[0] (state[10])
        assert key[10] == pytest.approx(0.8)   # Outer Mind Feeling[0] (state[75])
        assert key[15] == pytest.approx(0.7)   # CURIOSITY (first in _get_hormones_5d)

    def test_arc_distillation_summary(self):
        p = ArcPuzzlePlugin()
        experiences = [
            {"action_taken": "ACTION1", "outcome_score": 0.8},
            {"action_taken": "ACTION1", "outcome_score": 0.7},
            {"action_taken": "ACTION2", "outcome_score": 0.3},
        ]
        summary = p.summarize_for_distillation(experiences)
        assert "pattern" in summary
        assert "ACTION1" in summary["pattern"]

    def test_unknown_domain_records_fine(self, orchestrator):
        rid = orchestrator.record_outcome(
            domain="unknown_domain",
            perception_features=[0.5] * 5,
            inner_state_132d=[0.5] * 132,
            hormonal_snapshot={},
            action_taken="test",
            outcome_score=0.5,
        )
        assert rid > 0


# ── Utility Tests ─────────────────────────────────────────────────

class TestUtils:
    def test_cosine_sim_identical(self):
        assert _cosine_sim([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_cosine_sim_orthogonal(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_cosine_sim_empty(self):
        assert _cosine_sim([], [1, 2]) == 0.0

    def test_cosine_sim_different_lengths(self):
        sim = _cosine_sim([1, 0, 0], [1, 0])
        assert sim == pytest.approx(1.0)
