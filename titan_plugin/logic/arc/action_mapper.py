"""
titan_plugin/logic/arc/action_mapper.py — ARC-AGI-3 Action Mapper.

Maps Titan's nervous system signals to ARC game actions (int IDs 1-7).
Autonomy-first: uses neural signals directly, no LLM involved.

Strategy hierarchy:
  1. INTUITION fires → pattern-based action (repeat successful pattern)
  2. CURIOSITY fires → exploration action (try unvisited state)
  3. FOCUS fires high → exploit current best action
  4. CREATIVITY fires → novel/random action (creative exploration)
  5. Default → score all actions with action-scoring NN, pick highest
"""
import logging
import random
from typing import Optional

import numpy as np

from ..neural_reflex_net import NeuralReflexNet

logger = logging.getLogger(__name__)

# Maximum action ID in ARC-AGI-3 (ACTION1-ACTION7)
MAX_ACTION_ID = 7


class ActionMapper:
    """
    Map nervous system signals to ARC-AGI-3 game actions.

    Autonomy-first: decisions made by NervousSystem signals,
    not by LLM. LLM can optionally narrate afterward.
    """

    def __init__(self, grid_feature_dim: int = 30, pattern_dim: int = 7):
        """
        Args:
            grid_feature_dim: Dimension of grid features (30D: Trinity 15D + spatial 5D + focused 10D)
            pattern_dim: Dimension of pattern profile (7D from PatternPrimitives, 0 to disable)
        """
        # Action scoring NN — learns to rank actions by expected reward
        # Input: grid features (30D) + pattern profile (7D) + action one-hot (7D) = 44D
        # A5: focused_body(5) + focused_mind(5) added 10D from quadrant perception
        scorer_input_dim = grid_feature_dim + pattern_dim + MAX_ACTION_ID
        self._action_scorer = NeuralReflexNet(
            name="ARC_ACTION_SCORER",
            input_dim=scorer_input_dim,
            hidden_1=48,    # was 16 — 3x more capacity
            hidden_2=24,    # was 8  — 3x more capacity
            learning_rate=0.0005,  # slightly lower for stability with larger net
            fire_threshold=0.0,  # not used for scoring
        )
        self._grid_feature_dim = grid_feature_dim
        self._action_history: list[dict] = []
        self._successful_patterns: list[dict] = []  # Actions that got rewards

    def select_action(
        self,
        available_actions: list[int],
        nervous_signals: list[dict],
        grid_features: dict,
        epsilon: float = 0.1,
    ) -> Optional[int]:
        """
        Select best action using nervous system signals.

        Args:
            available_actions: List of action IDs (e.g. [1, 2, 3, 4])
            nervous_signals: Output from NS evaluate() or _get_ns_signals()
            grid_features: Dict from GridPerception.perceive()
            epsilon: Random exploration probability

        Returns:
            Selected action ID (int), or None if no actions available.
        """
        if not available_actions:
            return None

        # Build signal lookup
        signals = {s["system"]: s["urgency"] for s in nervous_signals}

        # 1. INTUITION → repeat successful pattern
        if signals.get("INTUITION", 0) > 0.5 and self._successful_patterns:
            pattern = self._find_matching_pattern(available_actions)
            if pattern is not None:
                logger.debug("[ActionMapper] INTUITION -> pattern-based action %d", pattern)
                return pattern

        # 2. CURIOSITY → explore (pick least-tried action)
        if signals.get("CURIOSITY", 0) > 0.3:
            action = self._exploration_action(available_actions)
            if action is not None:
                logger.debug("[ActionMapper] CURIOSITY -> exploration action %d", action)
                return action

        # 3. CREATIVITY → novel random action (creative exploration)
        if signals.get("CREATIVITY", 0) > 0.4:
            action = random.choice(available_actions)
            logger.debug("[ActionMapper] CREATIVITY -> random action %d", action)
            return action

        # 4. Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.choice(available_actions)

        # 5. Default: score all actions, pick highest
        return self._score_and_select(available_actions, grid_features)

    def record_outcome(self, action: int, reward: float,
                       grid_features: dict) -> None:
        """
        Record action outcome for learning.

        Args:
            action: The action ID that was taken
            reward: Reward received
            grid_features: Grid features at time of action
        """
        # Clear stale history if architecture changed (dimension mismatch)
        if self._action_history:
            test_inp = self._build_scorer_input(grid_features, action)
            if len(test_inp) != self._action_scorer.input_dim:
                logger.warning("[ActionMapper] Clearing %d stale history entries "
                               "(dim %d != %d)", len(self._action_history),
                               len(test_inp), self._action_scorer.input_dim)
                self._action_history.clear()
                self._successful_patterns.clear()

        self._action_history.append({
            "action": action,
            "reward": reward,
            "features": grid_features,
        })

        # Remember successful patterns
        if reward > 0.3:
            self._successful_patterns.append({
                "action": action,
                "reward": reward,
            })
            # Keep only recent successes
            if len(self._successful_patterns) > 50:
                self._successful_patterns = self._successful_patterns[-50:]

        # Train action scorer from recent history
        if len(self._action_history) >= 8:
            self._train_scorer()

    def _encode_action(self, action_id: int) -> list[float]:
        """Encode action ID as one-hot vector (7D for ACTION1-ACTION7)."""
        one_hot = [0.0] * MAX_ACTION_ID
        idx = action_id - 1  # ACTION1=1 → index 0
        if 0 <= idx < MAX_ACTION_ID:
            one_hot[idx] = 1.0
        return one_hot

    def _build_scorer_input(self, grid_features: dict, action_id: int) -> np.ndarray:
        """Build scorer input: grid features + action one-hot.

        Layout: body(5) + mind(5) + spirit(5) + spatial(5) + focused(10) + pattern(7) + action(7)
        A5: focused_body(5) + focused_mind(5) adds 10D from quadrant perception.
        Adapts to whatever GridPerception provides (backward compat if no focused features).
        """
        body = grid_features.get("inner_body", [0.5] * 5)
        mind = grid_features.get("inner_mind", [0.5] * 5)
        spirit = grid_features.get("inner_spirit", [0.5] * 5)
        spatial = grid_features.get("spatial", [])  # 5D if available
        focused_body = grid_features.get("focused_body", [])  # 5D if A5 enabled
        focused_mind = grid_features.get("focused_mind", [])  # 5D if A5 enabled
        pattern = grid_features.get("pattern_profile", [])  # 7D if PGL enabled
        action_enc = self._encode_action(action_id)
        return np.array(body + mind + spirit + spatial + focused_body + focused_mind +
                         pattern + action_enc, dtype=np.float64)

    def _score_and_select(self, actions: list[int],
                          grid_features: dict) -> int:
        """Score actions using the action-scoring NN and pick highest."""
        best_action = actions[0]
        best_score = -1.0

        for action in actions:
            features = self._build_scorer_input(grid_features, action)
            score = self._action_scorer.forward(features)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def score_state(self, grid_features: dict, action_id: int) -> float:
        """Score a (predicted) state-action pair. Used by forward model lookahead.

        Args:
            grid_features: Dict with inner_body, inner_mind, inner_spirit, etc.
            action_id: action that was (hypothetically) taken

        Returns:
            Scalar score (higher = better predicted outcome)
        """
        features = self._build_scorer_input(grid_features, action_id)
        return float(self._action_scorer.forward(features))

    def _exploration_action(self, actions: list[int]) -> Optional[int]:
        """Pick the least-tried action for exploration."""
        if not self._action_history:
            return random.choice(actions)

        # Count how many times each action was used recently
        action_counts: dict[int, int] = {}
        for entry in self._action_history[-50:]:
            a = entry["action"]
            action_counts[a] = action_counts.get(a, 0) + 1

        # Find action with lowest count
        least_tried = None
        min_count = float("inf")
        for action in actions:
            count = action_counts.get(action, 0)
            if count < min_count:
                min_count = count
                least_tried = action

        return least_tried

    def _find_matching_pattern(self, actions: list[int]) -> Optional[int]:
        """Find an available action that matches a successful pattern."""
        for pattern in reversed(self._successful_patterns):
            if pattern["action"] in actions:
                return pattern["action"]
        return None

    def _train_scorer(self) -> None:
        """Train action scoring NN from recent history."""
        recent = self._action_history[-32:]
        if len(recent) < 8:
            return

        inputs = []
        targets = []
        for entry in recent:
            inp = self._build_scorer_input(entry["features"], entry["action"])
            # Skip entries with wrong dimension (from previous architecture)
            if len(inp) != self._action_scorer.input_dim:
                continue
            inputs.append(inp.tolist())
            targets.append([max(0.0, min(1.0, entry["reward"]))])

        if len(inputs) < 4:
            return  # Not enough valid samples after filtering

        inputs_arr = np.array(inputs, dtype=np.float64)
        targets_arr = np.array(targets, dtype=np.float64)

        loss = self._action_scorer.train_step(inputs_arr, targets_arr)
        if loss is not None:
            logger.debug("[ActionMapper] Scorer trained: loss=%.4f", loss)

    def get_stats(self) -> dict:
        """Return mapper statistics."""
        return {
            "action_history_size": len(self._action_history),
            "successful_patterns": len(self._successful_patterns),
            "scorer_stats": self._action_scorer.get_stats(),
        }
