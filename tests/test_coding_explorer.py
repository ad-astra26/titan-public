"""
Tests for CodingExplorer — Self-Directed Cognitive Development.

Tests exercise selection, reward computation, state vector building,
concept tracking, and sandbox integration.
"""

import os
import sqlite3
import tempfile
import time

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def explorer(temp_db):
    """Create a CodingExplorer with test config."""
    from titan_plugin.logic.coding_explorer import CodingExplorer
    config = {
        "cooldown_epochs": 5,
        "max_exercises_per_dream": 3,
        "research_before_code": False,
        "arc_cross_pollinate": False,
    }
    return CodingExplorer(send_queue=None, config=config, db_path=temp_db)


class TestCodingExplorerInit:
    def test_init_creates_db_table(self, temp_db):
        from titan_plugin.logic.coding_explorer import CodingExplorer
        ce = CodingExplorer(db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "coding_exercises" in tables

    def test_init_default_stats(self, explorer):
        stats = explorer.get_stats()
        assert stats["total_exercises"] == 0
        assert stats["total_successes"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["concepts_attempted"] == 0

    def test_can_explore_initially(self, explorer):
        assert explorer.can_explore is True

    def test_cooldown_ticks(self, explorer):
        explorer._cooldown = 3
        assert explorer.can_explore is False
        explorer.tick_cooldown()
        explorer.tick_cooldown()
        explorer.tick_cooldown()
        assert explorer.can_explore is True


class TestExerciseTemplates:
    def test_all_templates_have_required_fields(self):
        from titan_plugin.logic.coding_explorer import EXERCISE_TEMPLATES
        for name, template in EXERCISE_TEMPLATES.items():
            assert "concept" in template, f"{name} missing concept"
            assert "difficulty" in template, f"{name} missing difficulty"
            assert "description" in template, f"{name} missing description"
            assert "code_template" in template, f"{name} missing code_template"
            assert "test_assertions" in template, f"{name} missing test_assertions"
            assert "associations" in template, f"{name} missing associations"

    def test_template_count(self):
        from titan_plugin.logic.coding_explorer import EXERCISE_TEMPLATES
        assert len(EXERCISE_TEMPLATES) >= 8

    def test_all_action_exercise_mappings_valid(self):
        from titan_plugin.logic.coding_explorer import (
            ACTION_EXERCISE_MAP, EXERCISE_TEMPLATES)
        for action, exercises in ACTION_EXERCISE_MAP.items():
            for ex_name in exercises:
                assert ex_name in EXERCISE_TEMPLATES, (
                    f"Action '{action}' references unknown template '{ex_name}'")


class TestActionSelection:
    def test_fallback_seek_novelty(self, explorer):
        trigger = {"action": "seek_novelty", "gap_metric": "DA"}
        action, idx = explorer._select_action(trigger, 1000, {}, {})
        assert action == "implement"
        assert idx == 2

    def test_fallback_consolidate(self, explorer):
        trigger = {"action": "consolidate", "gap_metric": "i_confidence"}
        action, idx = explorer._select_action(trigger, 1000, {}, {})
        assert action == "refactor"
        assert idx == 4

    def test_fallback_introspect(self, explorer):
        trigger = {"action": "introspect", "gap_metric": "dominant_primitive"}
        action, idx = explorer._select_action(trigger, 1000, {}, {})
        assert action == "decompose"
        assert idx == 0

    def test_fallback_rest(self, explorer):
        trigger = {"action": "rest"}
        action, idx = explorer._select_action(trigger, 1000, {}, {})
        assert action == "abstract"
        assert idx == 1

    def test_fallback_default(self, explorer):
        trigger = {"action": "unknown_action"}
        action, idx = explorer._select_action(trigger, 1000, {}, {})
        assert action == "implement"
        assert idx == 2


class TestStateVector:
    def test_state_vector_shape(self, explorer):
        trigger = {"urgency": 0.6, "gap_metric": "DA"}
        neuromods = {"DA": 0.7, "5HT": 0.5, "NE": 0.3, "ACh": 0.6,
                     "GABA": 0.2, "Endorphin": 0.4}
        context = {"i_confidence": 0.9, "chi_coherence": 0.5,
                   "total_chains": 500, "commit_rate": 0.6,
                   "vocab_total": 200}
        vec = explorer._build_state_vector(trigger, 100000, neuromods, context)
        assert vec.shape == (30,)
        assert vec.dtype.name.startswith("float")

    def test_state_vector_neuromods(self, explorer):
        trigger = {"urgency": 0.5}
        neuromods = {"DA": 0.8, "5HT": 0.3}
        vec = explorer._build_state_vector(trigger, 1000, neuromods, {})
        assert abs(vec[0] - 0.8) < 0.01  # DA
        assert abs(vec[1] - 0.3) < 0.01  # 5HT

    def test_state_vector_urgency(self, explorer):
        trigger = {"urgency": 0.9, "gap_metric": "NE"}
        vec = explorer._build_state_vector(trigger, 1000, {}, {})
        assert abs(vec[6] - 0.9) < 0.01  # urgency
        assert abs(vec[7] - 0.3) < 0.01  # NE gap encoding


class TestRewardComputation:
    def test_all_tests_pass(self, explorer):
        template = {"test_assertions": 3}
        sandbox_result = {"success": True, "result": "All tests passed"}
        reward, passed, total = explorer._compute_reward(sandbox_result, template)
        assert reward == 0.10
        assert passed == 3
        assert total == 3

    def test_syntax_error(self, explorer):
        template = {"test_assertions": 2}
        sandbox_result = {"success": False, "error": "Syntax error: invalid syntax"}
        reward, passed, total = explorer._compute_reward(sandbox_result, template)
        assert reward == -0.02
        assert passed == 0

    def test_runtime_error(self, explorer):
        template = {"test_assertions": 2}
        sandbox_result = {"success": False, "error": "NameError: x undefined"}
        reward, passed, total = explorer._compute_reward(sandbox_result, template)
        assert reward == -0.01

    def test_timeout(self, explorer):
        template = {"test_assertions": 2}
        sandbox_result = {"success": False, "error": "Timeout: exceeded 30s"}
        reward, passed, total = explorer._compute_reward(sandbox_result, template)
        assert reward == -0.03

    def test_compiles_no_test(self, explorer):
        template = {"test_assertions": 0}
        sandbox_result = {"success": True, "result": "42"}
        reward, passed, total = explorer._compute_reward(sandbox_result, template)
        assert reward == 0.02


class TestExerciseSelection:
    def test_picks_template_for_action(self, explorer):
        trigger = {"action": "implement"}
        template = explorer._pick_exercise("implement", trigger)
        assert template is not None
        assert "concept" in template

    def test_novelty_preference(self, explorer):
        # First pick: should prefer least attempted
        t1 = explorer._pick_exercise("implement", {})
        concept1 = t1["concept"]
        # Record some attempts
        explorer._concept_attempts[concept1] = 100
        # Second pick: should avoid the well-attempted concept
        t2 = explorer._pick_exercise("implement", {})
        assert t2["concept"] != concept1 or len(
            explorer._concept_attempts) < 2

    def test_returns_none_for_empty(self, explorer):
        # Use a monkeypatched empty map without modifying globals
        import titan_plugin.logic.coding_explorer as mod
        orig_map = mod.ACTION_EXERCISE_MAP
        orig_templates = dict(mod.EXERCISE_TEMPLATES)
        try:
            mod.ACTION_EXERCISE_MAP = {"nonexistent": ["no_such_template"]}
            result = explorer._pick_exercise("nonexistent", {})
            assert result is None
        finally:
            mod.ACTION_EXERCISE_MAP = orig_map
            mod.EXERCISE_TEMPLATES.update(orig_templates)


class TestExploreExecution:
    def test_explore_with_sandbox(self, explorer):
        """Full integration test: trigger → action → sandbox → reward."""
        trigger = {"action": "seek_novelty", "urgency": 0.6,
                   "gap_metric": "DA", "reason": "test"}
        neuromods = {"DA": 0.7, "5HT": 0.5, "NE": 0.3}
        result = explorer.explore(trigger, 1000, neuromods)
        assert result is not None
        assert result.epoch == 1000
        assert result.action in ["decompose", "abstract", "implement",
                                  "test", "refactor", "compose"]
        assert result.concept != ""
        assert result.reward != 0.0  # Should get some reward/penalty
        # Stats should update
        stats = explorer.get_stats()
        assert stats["total_exercises"] == 1
        assert stats["concepts_attempted"] >= 1

    def test_cooldown_prevents_double_explore(self, explorer):
        trigger = {"action": "seek_novelty", "urgency": 0.5,
                   "gap_metric": "DA", "reason": "test"}
        # First explore works
        r1 = explorer.explore(trigger, 1000, {})
        assert r1 is not None
        # Second is blocked by cooldown
        r2 = explorer.explore(trigger, 1001, {})
        assert r2 is None

    def test_dream_counter(self, explorer):
        explorer._max_per_dream = 1
        trigger = {"action": "seek_novelty", "urgency": 0.5,
                   "gap_metric": "DA", "reason": "test"}
        r1 = explorer.explore(trigger, 1000, {})
        assert r1 is not None
        # Reset cooldown manually
        explorer._cooldown = 0
        # Should be blocked by per-dream limit
        r2 = explorer.explore(trigger, 1001, {})
        assert r2 is None
        # Dream start resets
        explorer.on_dream_start()
        explorer._cooldown = 0
        r3 = explorer.explore(trigger, 1002, {})
        assert r3 is not None

    def test_db_persistence(self, explorer):
        trigger = {"action": "seek_novelty", "urgency": 0.5,
                   "gap_metric": "DA", "reason": "test"}
        result = explorer.explore(trigger, 1000, {})
        assert result is not None
        # Check DB
        conn = sqlite3.connect(explorer._db_path)
        rows = conn.execute(
            "SELECT concept, action, reward FROM coding_exercises").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == result.concept
        assert rows[0][1] == result.action


class TestNmFloat:
    def test_simple_float(self, explorer):
        assert explorer._nm_float({"DA": 0.7}, "DA") == 0.7

    def test_dict_value(self, explorer):
        assert explorer._nm_float({"DA": {"level": 0.8}}, "DA") == 0.8

    def test_alt_keys_serotonin(self, explorer):
        assert explorer._nm_float({"Serotonin": 0.6}, "5HT") == 0.6
        assert explorer._nm_float({"5-HT": 0.7}, "5HT") == 0.7

    def test_default(self, explorer):
        assert explorer._nm_float({}, "DA") == 0.5
        assert explorer._nm_float({}, "DA", default=0.3) == 0.3


class TestCodingInterpreter:
    def test_interpreter_registered(self):
        from titan_plugin.logic.reasoning_interpreter import ReasoningInterpreter
        ri = ReasoningInterpreter()
        coding = ri.registry.get("coding")
        assert coding is not None
        assert coding.domain == "coding"
        assert len(coding.action_names) == 6
        assert "decompose" in coding.action_names
        assert "compose" in coding.action_names

    def test_interpreter_build_features(self):
        from titan_plugin.logic.reasoning_interpreter import CodingInterpreter
        ci = CodingInterpreter()
        plan = {"intent": "reflect", "structure": "decomposition"}
        context = {"success_rate": 0.5, "sandbox_available": 1.0}
        features = ci.build_features(plan, context)
        assert features.shape == (24,)

    def test_interpreter_interpret(self):
        from titan_plugin.logic.reasoning_interpreter import CodingInterpreter
        ci = CodingInterpreter()
        reasoning_output = {
            "action": "COMMIT",
            "reasoning_plan": {"intent": "reflect", "structure": "decomposition"},
            "confidence": 0.7,
        }
        context = {"success_rate": 0.5}
        result = ci.interpret(reasoning_output, context)
        assert result["domain"] == "coding"
        assert result["action_name"] in ci.action_names


class TestSandboxTemplates:
    """Verify that all exercise templates actually compile and pass.

    Note: exec() needs globals dict for imports and recursive calls to work.
    The real sandbox runs code as a subprocess file, so it works naturally.
    """

    def _run_template(self, name):
        from titan_plugin.logic.coding_explorer import EXERCISE_TEMPLATES
        code = EXERCISE_TEMPLATES[name]["code_template"]
        # Use a shared globals dict so imports and recursive calls work
        g = {}
        exec(compile(code, f"<{name}>", "exec"), g)

    def test_symmetry_detection(self):
        self._run_template("symmetry_detection")

    def test_sequence_prediction(self):
        self._run_template("sequence_prediction")

    def test_grid_rotation(self):
        self._run_template("grid_rotation")

    def test_decomposition(self):
        self._run_template("decomposition")

    def test_abstraction(self):
        self._run_template("abstraction")

    def test_composition(self):
        self._run_template("composition")

    def test_statistical_analysis(self):
        self._run_template("statistical_analysis")

    def test_recursive_structure(self):
        self._run_template("recursive_structure")
