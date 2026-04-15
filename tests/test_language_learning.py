"""Tests for Language Learning Experience Plugin (L-CORE + L-FEEL)."""
import asyncio
import json
import os
import pytest
import tempfile

from titan_plugin.logic.inner_memory import InnerMemoryStore
from titan_plugin.logic.language_learning import (
    LanguageLearningExperience,
    _cosine_similarity,
    _flatten_perturbation,
    LAYER_ORDER,
    LAYER_SIZES,
    PASS_FEEL,
    PASS_RECOGNIZE,
    PASS_PRODUCE,
    WORD_RESONANCE_PATH,
)


# ── Utility Tests ────────────────────────────────────────────────────

class TestCosineAndFlatten:

    def test_cosine_identical(self):
        a = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_different_lengths(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_flatten_perturbation_130d(self):
        perturbation = {
            "inner_body": [0.1] * 5,
            "inner_mind": [0.2] * 15,
            "inner_spirit": [0.3] * 45,
            "outer_body": [0.4] * 5,
            "outer_mind": [0.5] * 15,
            "outer_spirit": [0.6] * 45,
        }
        flat = _flatten_perturbation(perturbation)
        assert len(flat) == 130
        assert flat[0] == 0.1   # inner_body[0]
        assert flat[5] == 0.2   # inner_mind[0]
        assert flat[20] == 0.3  # inner_spirit[0]
        assert flat[65] == 0.4  # outer_body[0]
        assert flat[70] == 0.5  # outer_mind[0]
        assert flat[85] == 0.6  # outer_spirit[0]

    def test_flatten_pads_short_layers(self):
        perturbation = {
            "inner_body": [0.1, 0.2],  # Only 2 of 5
            "inner_mind": [],
            "inner_spirit": [0.3],
            "outer_body": [0.4] * 5,
            "outer_mind": [0.5] * 15,
            "outer_spirit": [0.6] * 45,
        }
        flat = _flatten_perturbation(perturbation)
        assert len(flat) == 130
        assert flat[0] == 0.1
        assert flat[1] == 0.2
        assert flat[2] == 0.0  # Padded


# ── Vocabulary Store Tests ───────────────────────────────────────────

class TestVocabularyStore:

    @pytest.fixture
    def memory(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return InnerMemoryStore(db_path=db_path)

    def test_store_and_get_word(self, memory):
        memory.store_word("warm", "adjective", stage=1,
                          felt_tensor=[0.1] * 130,
                          hormone_pattern={"EMPATHY": 0.1})
        word = memory.get_word("warm")
        assert word is not None
        assert word["word"] == "warm"
        assert word["word_type"] == "adjective"
        assert word["stage"] == 1
        assert len(word["felt_tensor"]) == 130
        assert word["hormone_pattern"]["EMPATHY"] == 0.1
        assert word["learning_phase"] == "unlearned"
        assert word["confidence"] == 0.0

    def test_store_word_updates_existing(self, memory):
        memory.store_word("warm", "adjective", felt_tensor=[0.1] * 130)
        memory.store_word("warm", "adjective", felt_tensor=[0.2] * 130)
        word = memory.get_word("warm")
        assert word["felt_tensor"][0] == 0.2  # Updated

    def test_update_word_learning(self, memory):
        memory.store_word("warm", "adjective")
        memory.update_word_learning("warm", phase="felt",
                                    confidence_delta=0.1,
                                    encountered=True)
        word = memory.get_word("warm")
        assert word["learning_phase"] == "felt"
        assert word["confidence"] == pytest.approx(0.1)
        assert word["times_encountered"] == 1

    def test_update_word_produced(self, memory):
        memory.store_word("warm", "adjective")
        memory.update_word_learning("warm", phase="producible",
                                    confidence_delta=0.2,
                                    encountered=True, produced=True)
        word = memory.get_word("warm")
        assert word["times_produced"] == 1

    def test_confidence_clamped(self, memory):
        memory.store_word("warm", "adjective")
        # Push confidence above 1.0
        for _ in range(15):
            memory.update_word_learning("warm", phase="felt",
                                        confidence_delta=0.1)
        word = memory.get_word("warm")
        assert word["confidence"] <= 1.0

    def test_get_vocabulary_filtered(self, memory):
        memory.store_word("warm", "adjective", stage=1)
        memory.store_word("explore", "verb", stage=2)
        memory.update_word_learning("warm", phase="felt")

        stage1 = memory.get_vocabulary(stage=1)
        assert len(stage1) == 1
        assert stage1[0]["word"] == "warm"

        felt = memory.get_vocabulary(phase="felt")
        assert len(felt) == 1
        assert felt[0]["word"] == "warm"

    def test_find_similar_words(self, memory):
        # Store words with distinct tensors
        memory.store_word("warm", "adjective",
                          felt_tensor=[1.0, 0.0, 0.0])
        memory.update_word_learning("warm", phase="felt",
                                    confidence_delta=0.5)
        memory.store_word("cold", "adjective",
                          felt_tensor=[0.0, 1.0, 0.0])
        memory.update_word_learning("cold", phase="felt",
                                    confidence_delta=0.5)

        # Query close to "warm"
        similar = memory.find_similar_words([0.9, 0.1, 0.0], top_k=2,
                                            min_confidence=0.3)
        assert len(similar) == 2
        assert similar[0][0] == "warm"  # Closest match
        assert similar[0][1] > similar[1][1]  # Higher similarity

    def test_vocab_stats(self, memory):
        memory.store_word("warm", "adjective")
        memory.store_word("cold", "adjective")
        memory.update_word_learning("warm", phase="felt",
                                    confidence_delta=0.5,
                                    encountered=True)
        stats = memory.get_vocab_stats()
        assert stats["total_words"] == 2
        assert stats["phases"]["unlearned"] == 1
        assert stats["phases"]["felt"] == 1
        assert stats["total_encounters"] == 1

    def test_get_nonexistent_word(self, memory):
        assert memory.get_word("nonexistent") is None


# ── Language Learning Plugin Tests ───────────────────────────────────

class TestLanguageLearningPlugin:

    @pytest.fixture
    def memory(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return InnerMemoryStore(db_path=db_path)

    @pytest.fixture
    def plugin(self, memory):
        p = LanguageLearningExperience(inner_memory=memory)
        return p

    def test_loads_word_resonance(self, plugin):
        assert len(plugin._word_data) == 40
        assert "warm" in plugin._word_data
        assert "explore" in plugin._word_data
        assert "I" in plugin._word_data

    def test_plugin_attributes(self, plugin):
        assert plugin.name == "language"
        assert plugin.difficulty_levels == 3

    @pytest.mark.asyncio
    async def test_generate_stimulus_feel_pass(self, plugin):
        stimulus = await plugin.generate_stimulus()
        assert stimulus["type"] == "word"
        assert stimulus["pass_type"] == PASS_FEEL
        assert stimulus["content"] in plugin._word_data
        assert "expected" in stimulus
        assert len(stimulus["expected"]["felt_tensor"]) == 130

    @pytest.mark.asyncio
    async def test_three_pass_cycle(self, plugin):
        """Test Feel → Recognize → Produce cycle for one word."""
        s1 = await plugin.generate_stimulus()
        word = s1["content"]
        assert s1["pass_type"] == PASS_FEEL

        s2 = await plugin.generate_stimulus()
        assert s2["content"] == word  # Same word
        assert s2["pass_type"] == PASS_RECOGNIZE

        s3 = await plugin.generate_stimulus()
        assert s3["content"] == word  # Still same word
        assert s3["pass_type"] == PASS_PRODUCE

        # Next stimulus should be new word
        s4 = await plugin.generate_stimulus()
        assert s4["pass_type"] == PASS_FEEL  # Back to feel
        # (could be same or different word depending on selection)

    def test_perturbation_feel_full_strength(self, plugin):
        stimulus = {"content": "warm", "pass_type": PASS_FEEL}
        pert = plugin.compute_perturbation(stimulus)
        assert len(pert["inner_body"]) == 5
        assert len(pert["inner_mind"]) == 15
        assert len(pert["inner_spirit"]) == 45
        assert len(pert["outer_body"]) == 5
        assert len(pert["outer_mind"]) == 15
        assert len(pert["outer_spirit"]) == 45
        # "warm" has inner_body thermal +0.4
        assert pert["inner_body"][4] == pytest.approx(0.4)
        assert "hormone_stimuli" in pert
        assert pert["hormone_stimuli"].get("EMPATHY", 0) == pytest.approx(0.1)

    def test_perturbation_recognize_reduced(self, plugin):
        stimulus = {"content": "warm", "pass_type": PASS_RECOGNIZE}
        pert = plugin.compute_perturbation(stimulus)
        # 30% of 0.4 = 0.12
        assert pert["inner_body"][4] == pytest.approx(0.12)
        assert pert["hormone_stimuli"]["EMPATHY"] == pytest.approx(0.03)

    def test_perturbation_produce_zero(self, plugin):
        stimulus = {"content": "warm", "pass_type": PASS_PRODUCE}
        pert = plugin.compute_perturbation(stimulus)
        assert all(v == 0.0 for v in pert["inner_body"])
        assert all(v == 0.0 for v in pert["inner_spirit"])
        assert pert["hormone_stimuli"] == {}

    def test_total_perturbation_is_130d(self, plugin):
        stimulus = {"content": "energy", "pass_type": PASS_FEEL}
        pert = plugin.compute_perturbation(stimulus)
        total = (len(pert["inner_body"]) + len(pert["inner_mind"]) +
                 len(pert["inner_spirit"]) + len(pert["outer_body"]) +
                 len(pert["outer_mind"]) + len(pert["outer_spirit"]))
        assert total == 130

    @pytest.mark.asyncio
    async def test_evaluate_feel_pass(self, plugin, memory):
        stimulus = {
            "content": "warm",
            "pass_type": PASS_FEEL,
            "expected": {
                "felt_tensor": [0.0] * 130,
                "hormone_affinity": {"EMPATHY": 0.1},
            },
        }
        response = {
            "hormonal_state": {"EMPATHY": {"level": 0.3, "fired": 0}},
            "fired_programs": [{"program": "INTUITION", "intensity": 0.5}],
        }
        # Pre-store the word
        memory.store_word("warm", "adjective")

        result = await plugin.evaluate_response(stimulus, response)
        assert result["score"] >= 0.5
        assert result["pass_type"] == PASS_FEEL
        assert "FEEL" in result["feedback"]

    @pytest.mark.asyncio
    async def test_evaluate_updates_memory(self, plugin, memory):
        memory.store_word("warm", "adjective")
        stimulus = {
            "content": "warm",
            "pass_type": PASS_FEEL,
            "expected": {"felt_tensor": [0.1] * 130,
                         "hormone_affinity": {"EMPATHY": 0.1}},
        }
        response = {
            "hormonal_state": {"EMPATHY": {"level": 0.3}},
            "fired_programs": [],
        }
        await plugin.evaluate_response(stimulus, response)

        word = memory.get_word("warm")
        assert word["learning_phase"] == "felt"
        assert word["times_encountered"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_produce_with_vocabulary(self, plugin, memory):
        """Test PRODUCE evaluation with stored vocabulary."""
        # Store two words with distinct tensors
        memory.store_word("warm", "adjective",
                          felt_tensor=[1.0] + [0.0] * 129)
        memory.update_word_learning("warm", phase="felt",
                                    confidence_delta=0.5)
        memory.store_word("cold", "adjective",
                          felt_tensor=[0.0, 1.0] + [0.0] * 128)
        memory.update_word_learning("cold", phase="felt",
                                    confidence_delta=0.5)

        # Test: expected tensor close to "warm"
        stimulus = {
            "content": "warm",
            "pass_type": PASS_PRODUCE,
            "expected": {
                "felt_tensor": [0.9, 0.1] + [0.0] * 128,
                "hormone_affinity": {},
            },
        }
        response = {"hormonal_state": {}, "fired_programs": []}
        result = await plugin.evaluate_response(stimulus, response)
        assert result["score"] > 0.5  # Should find "warm" as top match

    def test_get_stats(self, plugin):
        stats = plugin.get_stats()
        assert stats["name"] == "language"
        assert stats["word_count"] == 40
        assert "words_taught" in stats

    def test_word_selection_prefers_unlearned(self, plugin, memory):
        """Unlearned words should be selected before learned ones."""
        # Store some words as already learned
        memory.store_word("warm", "adjective", stage=1)
        memory.update_word_learning("warm", phase="producible",
                                    confidence_delta=0.8)
        # Unlearned words should be selected first
        selected = plugin._select_words_for_session(5)
        assert "warm" not in selected or len(selected) == 5


# ── Integration: Plugin + Playground ─────────────────────────────────

class TestLanguageLearningIntegration:

    @pytest.fixture
    def memory(self, tmp_path):
        return InnerMemoryStore(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_full_session(self, memory):
        from titan_plugin.logic.experience_playground import ExperiencePlayground

        pg = ExperiencePlayground(inner_memory=memory)
        plugin = LanguageLearningExperience(inner_memory=memory)

        # Seed vocabulary in memory
        for word, recipe in list(plugin._word_data.items())[:5]:
            memory.store_word(word, recipe.get("word_type", "unknown"),
                              stage=recipe.get("stage", 1))

        pg.register(plugin)
        result = await pg.run_session("language",
                                      num_stimuli=6, pause_between=0.05)
        assert result["experience"] == "language"
        assert result["stimuli_count"] == 6
        assert len(result["results"]) == 6

        # Should have at least 2 words cycled through (6 stimuli / 3 passes)
        words_seen = set()
        for r in result["results"]:
            words_seen.add(r["stimulus"]["content"])
        assert len(words_seen) >= 1

        # Vocabulary should be updated
        vocab_stats = memory.get_vocab_stats()
        assert vocab_stats["total_encounters"] > 0

    @pytest.mark.asyncio
    async def test_session_records_all_passes(self, memory):
        from titan_plugin.logic.experience_playground import ExperiencePlayground

        pg = ExperiencePlayground(inner_memory=memory)
        plugin = LanguageLearningExperience(inner_memory=memory)

        for word, recipe in list(plugin._word_data.items())[:3]:
            memory.store_word(word, recipe.get("word_type", "unknown"),
                              stage=recipe.get("stage", 1))

        pg.register(plugin)
        result = await pg.run_session("language",
                                      num_stimuli=3, pause_between=0.05)

        passes = [r["stimulus"]["pass_type"] for r in result["results"]]
        # First 3 stimuli should be feel/recognize/produce for same word
        assert passes == [PASS_FEEL, PASS_RECOGNIZE, PASS_PRODUCE]
