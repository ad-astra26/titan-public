"""Tests for the Perceptual Grounding Layer (mini-reasoning + pattern primitives)."""
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
# Mini-Reasoning Tests
# ═══════════════════════════════════════════════════════════════

class TestDecompose:
    """Test DECOMPOSE primitive — grid → connected components."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        self.engine = MiniReasoningEngine()

    def test_empty_grid(self):
        grid = np.zeros((10, 10))
        objects = self.engine.decompose(grid)
        assert len(objects) == 0

    def test_single_object(self):
        grid = np.zeros((10, 10))
        grid[2:5, 3:6] = 1  # 3x3 square
        objects = self.engine.decompose(grid)
        assert len(objects) == 1
        assert objects[0].color == 1
        assert objects[0].size == 9

    def test_multiple_objects(self):
        grid = np.zeros((10, 10))
        grid[1:3, 1:3] = 1  # top-left
        grid[7:9, 7:9] = 2  # bottom-right
        objects = self.engine.decompose(grid)
        assert len(objects) == 2
        colors = {o.color for o in objects}
        assert colors == {1, 2}

    def test_separate_same_color(self):
        grid = np.zeros((10, 10))
        grid[0, 0] = 1
        grid[9, 9] = 1  # same color, not connected
        objects = self.engine.decompose(grid)
        assert len(objects) == 2

    def test_bbox_and_centroid(self):
        grid = np.zeros((10, 10))
        grid[2:5, 3:6] = 1
        objects = self.engine.decompose(grid)
        obj = objects[0]
        assert obj.bbox == (2, 3, 4, 5)
        assert obj.centroid == (3.0, 4.0)


class TestFilter:
    """Test FILTER primitive — what changed between steps."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        self.engine = MiniReasoningEngine()

    def test_no_change(self):
        grid = np.zeros((10, 10))
        grid[2:4, 2:4] = 1
        objs = self.engine.decompose(grid)
        effect = self.engine.filter_changes(grid, grid, objs, objs, action_id=1)
        assert effect.changed_cells == 0
        assert effect.essence == "no_change"
        assert effect.magnitude == 0.0

    def test_object_moved(self):
        grid1 = np.zeros((10, 10))
        grid1[2:4, 2:4] = 1
        grid2 = np.zeros((10, 10))
        grid2[2:4, 3:5] = 1  # moved right
        objs1 = self.engine.decompose(grid1)
        objs2 = self.engine.decompose(grid2)
        effect = self.engine.filter_changes(grid1, grid2, objs1, objs2, action_id=2)
        assert effect.changed_cells > 0
        assert "right" in effect.essence
        assert len(effect.moved_objects) > 0

    def test_object_appeared(self):
        grid1 = np.zeros((10, 10))
        grid2 = np.zeros((10, 10))
        grid2[5, 5] = 3  # new object
        objs1 = self.engine.decompose(grid1)
        objs2 = self.engine.decompose(grid2)
        effect = self.engine.filter_changes(grid1, grid2, objs1, objs2, action_id=1)
        assert len(effect.appeared) == 1


class TestSurprise:
    """Test surprise signal computation."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine, ActionEffect
        self.engine = MiniReasoningEngine()
        self.ActionEffect = ActionEffect

    def test_first_action_moderate_surprise(self):
        effect = self.ActionEffect(
            action_id=1, changed_cells=5, moved_objects=[], appeared=[],
            disappeared=[], essence="cells_changed", magnitude=0.05)
        surprise = self.engine.compute_surprise(1, effect, {})
        assert 0.3 <= surprise <= 0.7  # moderate for first encounter

    def test_consistent_action_low_surprise(self):
        # Record several consistent effects
        for _ in range(5):
            effect = self.ActionEffect(
                action_id=1, changed_cells=5, moved_objects=[], appeared=[],
                disappeared=[], essence="same_thing", magnitude=0.05)
            self.engine.record_effect(1, effect)
        # Same effect again → low surprise
        surprise = self.engine.compute_surprise(1, effect, {})
        assert surprise < 0.3

    def test_novel_effect_high_surprise(self):
        # Record consistent effects
        for _ in range(5):
            effect = self.ActionEffect(
                action_id=1, changed_cells=5, moved_objects=[], appeared=[],
                disappeared=[], essence="same_thing", magnitude=0.05)
            self.engine.record_effect(1, effect)
        # New effect → higher surprise
        novel = self.ActionEffect(
            action_id=1, changed_cells=50, moved_objects=[], appeared=[],
            disappeared=[], essence="totally_different", magnitude=0.5)
        surprise = self.engine.compute_surprise(1, novel, {})
        assert surprise > 0.3  # novel effect adds surprise


class TestCausalMemory:
    """Test action-effect causal memory."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine, ActionEffect
        self.engine = MiniReasoningEngine()
        self.ActionEffect = ActionEffect

    def test_unknown_action(self):
        profile = self.engine.get_action_profile(99)
        assert profile["known"] is False

    def test_record_and_retrieve(self):
        effect = self.ActionEffect(
            action_id=2, changed_cells=10, moved_objects=[], appeared=[],
            disappeared=[], essence="moved_right", magnitude=0.1)
        self.engine.record_effect(2, effect)
        profile = self.engine.get_action_profile(2)
        assert profile["known"] is True
        assert profile["dominant_effect"] == "moved_right"
        assert profile["consistency"] == 1.0

    def test_consistency_tracking(self):
        for i in range(8):
            essence = "moved_right" if i < 6 else "moved_left"
            effect = self.ActionEffect(
                action_id=3, changed_cells=10, moved_objects=[], appeared=[],
                disappeared=[], essence=essence, magnitude=0.1)
            self.engine.record_effect(3, effect)
        profile = self.engine.get_action_profile(3)
        assert profile["dominant_effect"] == "moved_right"
        assert profile["consistency"] == 0.75  # 6/8


class TestTrendWindow:
    """Test pattern trend tracking."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        self.engine = MiniReasoningEngine(trend_window=9)

    def test_rising_trend(self):
        for i in range(9):
            self.engine.update_trend({"symmetry": 0.1 * i})
        trends = self.engine.get_trends()
        assert trends.get("symmetry") == "rising"

    def test_falling_trend(self):
        for i in range(9):
            self.engine.update_trend({"alignment": 0.9 - 0.1 * i})
        trends = self.engine.get_trends()
        assert trends.get("alignment") == "falling"

    def test_stable_trend(self):
        for i in range(9):
            self.engine.update_trend({"shape": 0.5})
        trends = self.engine.get_trends()
        assert trends.get("shape") == "stable"


class TestProcessStep:
    """Test full mini-reasoning pipeline."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        from titan_plugin.logic.pattern_primitives import PatternPrimitives
        self.engine = MiniReasoningEngine()
        self.pp = PatternPrimitives()

    def test_first_step(self):
        grid = np.zeros((10, 10))
        grid[3:6, 3:6] = 1
        profile = self.pp.compute_profile(grid)
        block = self.engine.process_step(grid, action_id=-1, pattern_profile=profile)
        assert block.step == 1
        assert block.action_effect is None  # first step, no previous
        assert "initial_state" in block.salient_observation

    def test_second_step_with_change(self):
        grid1 = np.zeros((10, 10))
        grid1[3:6, 3:6] = 1
        profile1 = self.pp.compute_profile(grid1)
        self.engine.process_step(grid1, action_id=-1, pattern_profile=profile1)

        grid2 = np.zeros((10, 10))
        grid2[3:6, 4:7] = 1  # shifted right
        profile2 = self.pp.compute_profile(grid2)
        block = self.engine.process_step(grid2, action_id=2, pattern_profile=profile2)
        assert block.step == 2
        assert block.action_effect is not None
        assert block.action_effect.changed_cells > 0


# ═══════════════════════════════════════════════════════════════
# Pattern Primitives Tests
# ═══════════════════════════════════════════════════════════════

class TestPatternPrimitives:
    """Test pattern detection on synthetic grids."""

    def setup_method(self):
        from titan_plugin.logic.pattern_primitives import PatternPrimitives
        self.pp = PatternPrimitives()

    def test_symmetric_grid(self):
        grid = np.zeros((10, 10))
        grid[4, 3:7] = 1  # horizontal line centered
        score = self.pp.symmetry(grid)
        assert score > 0.8  # highly symmetric

    def test_asymmetric_grid(self):
        grid = np.zeros((10, 10))
        grid[0, 0] = 1  # single cell in corner
        score = self.pp.symmetry(grid)
        assert score > 0.9  # mostly empty = mostly symmetric (bg matches)

    def test_horizontal_line_shape(self):
        grid = np.zeros((10, 10))
        grid[5, 2:8] = 1  # horizontal line
        score = self.pp.shape_score(grid)
        assert score > 0.3  # line detected

    def test_alignment_concentrated(self):
        grid = np.zeros((10, 10))
        grid[5, :] = 1  # entire row
        score = self.pp.alignment(grid)
        assert score > 0.15  # concentrated in one row

    def test_adjacency_clustered(self):
        grid = np.zeros((10, 10))
        grid[4:7, 4:7] = 1  # 3x3 cluster
        score = self.pp.adjacency(grid)
        assert score > 0.5  # highly adjacent

    def test_adjacency_scattered(self):
        grid = np.zeros((10, 10))
        grid[0, 0] = 1
        grid[9, 9] = 1
        grid[0, 9] = 1
        grid[9, 0] = 1  # corners only
        score = self.pp.adjacency(grid)
        assert score < 0.3  # scattered

    def test_profile_returns_all_patterns(self):
        grid = np.ones((10, 10))
        profile = self.pp.compute_profile(grid)
        assert len(profile) == 7
        for name in self.pp.PATTERN_NAMES:
            assert name in profile
            assert 0.0 <= profile[name] <= 1.0

    def test_profile_to_vector(self):
        grid = np.ones((10, 10))
        profile = self.pp.compute_profile(grid)
        vec = self.pp.profile_to_vector(profile)
        assert len(vec) == 7
        assert all(isinstance(v, float) for v in vec)

    def test_repetition_repeated_blocks(self):
        grid = np.zeros((8, 8))
        grid[0:2, 0:2] = 1
        grid[0:2, 4:6] = 1
        grid[4:6, 0:2] = 1
        grid[4:6, 4:6] = 1  # 4 identical 2x2 blocks
        score = self.pp.repetition(grid)
        assert score > 0.3  # repeated pattern

    def test_containment_nested(self):
        grid = np.zeros((10, 10))
        grid[2:8, 2:8] = 1  # large square
        grid[4:6, 4:6] = 2  # small square inside
        score = self.pp.containment(grid)
        assert score > 0.0  # containment detected


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """Test mini-reasoning + pattern primitives working together."""

    def setup_method(self):
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        from titan_plugin.logic.pattern_primitives import PatternPrimitives
        self.engine = MiniReasoningEngine()
        self.pp = PatternPrimitives()

    def test_full_episode_simulation(self):
        """Simulate 10 steps of ARC play with pattern tracking."""
        grid = np.zeros((10, 10))
        grid[3:6, 3:6] = 1

        for step in range(10):
            profile = self.pp.compute_profile(grid)
            action_id = (step % 4) + 1
            block = self.engine.process_step(grid, action_id=action_id,
                                              pattern_profile=profile)
            # Shift grid slightly each step
            grid = np.roll(grid, 1, axis=1)

        stats = self.engine.get_stats()
        assert stats["steps"] == 10
        assert stats["actions_known"] > 0
        assert stats["trend_window_size"] == 10

    def test_surprise_decreases_with_repetition(self):
        """Same action with same effect → surprise should decrease."""
        grid1 = np.zeros((10, 10))
        grid1[3:6, 3:6] = 1
        grid2 = np.zeros((10, 10))
        grid2[3:6, 4:7] = 1  # shifted right

        surprises = []
        for _ in range(6):
            p1 = self.pp.compute_profile(grid1)
            self.engine.process_step(grid1, action_id=-1, pattern_profile=p1)
            p2 = self.pp.compute_profile(grid2)
            block = self.engine.process_step(grid2, action_id=2, pattern_profile=p2)
            surprises.append(block.surprise)
            self.engine._prev_grid = None  # reset for next round

        # Later surprises should be lower than first
        assert surprises[-1] <= surprises[0]

    def test_pattern_features_in_scorer_input(self):
        """Verify pattern_profile is passed through features dict."""
        grid = np.zeros((10, 10))
        grid[3:6, 3:6] = 1
        profile = self.pp.compute_profile(grid)
        features = {"inner_body": [0.5]*5, "inner_mind": [0.5]*5,
                     "inner_spirit": [0.5]*5, "spatial": [0.5]*5,
                     "pattern_profile": self.pp.profile_to_vector(profile)}
        assert len(features["pattern_profile"]) == 7
