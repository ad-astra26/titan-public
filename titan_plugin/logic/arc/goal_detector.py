"""
titan_plugin/logic/arc/goal_detector.py — ARC goal-signal inference.

Per rFP_arc_training_fix.md (2026-04-13): ARC-AGI-3 SDK does not expose a goal
grid — discovering it IS the challenge. This module provides three goal-signal
strategies:

  G1 EMPIRICAL CAPTURE (all games)
    On state=WIN, snapshot final grid as goal template per game. After first win,
    subsequent episodes use similarity-to-goal as a dense reward gradient.

  G2 ls20 CHARACTER-TARGET HEURISTIC (ls20-first MVP)
    ls20 is navigation (21-step L1 baseline):
      character = cell whose position changes step-to-step with identity preserved
      target    = static distinctive (rare-color) cell
    Reward = change in manhattan(character, target)

  G3/G4 (ft09, vc33) — deferred post-MVP; G1 empirical capture unblocks them
    once a first win on each game occurs.

Design: pure numpy, no torch/heavy deps. Persists to data/arc_agi_3/goal_grids.json
for cross-episode + cross-session continuity. Fails soft on all persistence errors.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# PERSISTENCE_BY_DESIGN: GoalDetector._goals / _meta are per-session state
# rebuilt from ARC game replay on boot — detecting goals from scratch each
# run is safer than trusting stale goal inferences across sessions.
class GoalDetector:
    """Per-game goal-signal state + detection helpers.

    Thread-safety note: accessed from single-threaded ArcSession.play_game loop.
    Not designed for concurrent per-game updates.
    """

    def __init__(self, persist_dir: str = "data/arc_agi_3"):
        self._persist_dir = persist_dir
        self._path = os.path.join(persist_dir, "goal_grids.json")
        self._goals: dict[str, np.ndarray] = {}       # game_id → goal grid
        self._meta: dict[str, dict] = {}              # game_id → captured_at_utc, shape, etc
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Best-effort load; corrupt or missing file → empty state, no crash."""
        try:
            if os.path.exists(self._path):
                with open(self._path) as fh:
                    data = json.load(fh)
                for gid, entry in data.items():
                    try:
                        shape = tuple(entry["shape"])
                        flat = entry["grid"]
                        self._goals[gid] = np.array(flat, dtype=np.int8).reshape(shape)
                        self._meta[gid] = {
                            "captured_at_utc": entry.get("captured_at_utc"),
                            "shape": shape,
                        }
                    except Exception as e:
                        logger.warning(
                            "[GoalDetector] Could not restore goal for %s: %s", gid, e)
        except Exception as e:
            logger.warning("[GoalDetector] Corrupted persistence at %s: %s — starting fresh", self._path, e)

    def _save(self) -> None:
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            payload = {}
            for gid, grid in self._goals.items():
                payload[gid] = {
                    "grid": grid.astype(int).tolist(),
                    "shape": list(grid.shape),
                    "captured_at_utc": self._meta.get(gid, {}).get(
                        "captured_at_utc",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ),
                }
            tmp = self._path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            logger.warning("[GoalDetector] Save failed: %s", e)

    # ── G1 empirical capture ──────────────────────────────────────────

    def get_goal(self, game_id: str) -> Optional[np.ndarray]:
        return self._goals.get(game_id)

    def has_goal(self, game_id: str) -> bool:
        return game_id in self._goals

    def on_episode_end(
        self, game_id: str, final_state: str, final_grid: np.ndarray,
    ) -> None:
        """Called by ArcSession at episode end. Captures goal on WIN only."""
        if final_state != "WIN":
            return
        if final_grid is None or final_grid.size == 0:
            return
        grid = np.asarray(final_grid, dtype=np.int8)
        # Always update on new win — the latest win state is the "freshest" goal
        self._goals[game_id] = grid.copy()
        self._meta[game_id] = {
            "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shape": tuple(grid.shape),
        }
        self._save()
        logger.info(
            "[GoalDetector] Captured goal for %s (shape=%s, %d non-zero cells)",
            game_id, grid.shape, int((grid != 0).sum()),
        )

    # ── Similarity + goal-distance reward ─────────────────────────────

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of cells matching between two grids of the same shape.
        Returns 0.0 on shape mismatch (defensive)."""
        a = np.asarray(a, dtype=np.int8)
        b = np.asarray(b, dtype=np.int8)
        if a.shape != b.shape or a.size == 0:
            return 0.0
        return float((a == b).sum()) / float(a.size)

    def goal_distance_delta(
        self, prev_grid: np.ndarray, new_grid: np.ndarray, goal_grid: np.ndarray,
    ) -> float:
        """Return similarity_new - similarity_prev.
        Positive = moved closer to goal; negative = moved away; 0 = no change."""
        sim_prev = self.similarity(prev_grid, goal_grid)
        sim_new = self.similarity(new_grid, goal_grid)
        return sim_new - sim_prev

    # ── G2 ls20 character-target heuristics ──────────────────────────

    @staticmethod
    def detect_character(
        prev_grid: np.ndarray, curr_grid: np.ndarray,
    ) -> Optional[tuple[int, int]]:
        """Detect character position by finding cell that changed between frames.

        Simple heuristic: returns the position of the FIRST non-background cell
        in curr_grid that differs from prev_grid. For ls20-style 1-cell character,
        this cleanly identifies the new position. Returns None if no change.
        """
        prev = np.asarray(prev_grid, dtype=np.int8)
        curr = np.asarray(curr_grid, dtype=np.int8)
        if prev.shape != curr.shape:
            return None
        diff = (prev != curr)
        if not diff.any():
            return None
        # Prefer non-background cell on the NEW grid (character's new position)
        ys, xs = np.where(diff & (curr != 0))
        if len(ys) == 0:
            # Character might have moved OUT of a cell (curr now background)
            ys, xs = np.where(diff)
        return (int(ys[0]), int(xs[0]))

    @staticmethod
    def detect_target(
        grid: np.ndarray, background: int = 0,
    ) -> Optional[tuple[int, int]]:
        """Detect target as the rarest non-background cell.

        For ls20: target is typically a single distinctive-colored cell that
        remains static across frames. Rarity heuristic picks it out from the
        common-colored terrain.
        """
        grid = np.asarray(grid, dtype=np.int8)
        if grid.size == 0:
            return None
        # Count each non-background color
        non_bg_mask = (grid != background)
        if not non_bg_mask.any():
            return None
        non_bg_values = grid[non_bg_mask]
        values, counts = np.unique(non_bg_values, return_counts=True)
        # Pick color with minimum count (rarest)
        rarest_color = int(values[int(np.argmin(counts))])
        ys, xs = np.where(grid == rarest_color)
        if len(ys) == 0:
            return None
        # Return centroid of that color's cells
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))
        return (cy, cx)

    @staticmethod
    def character_target_reward(
        prev_grid: np.ndarray, curr_grid: np.ndarray,
        character_color: int, target: tuple[int, int],
    ) -> float:
        """Manhattan-distance reward for ls20-style navigation.

        Returns: prev_distance - new_distance (positive = character moved toward target).
        If character color not found in either grid, returns 0 (signal absent).
        """
        prev = np.asarray(prev_grid, dtype=np.int8)
        curr = np.asarray(curr_grid, dtype=np.int8)
        if prev.shape != curr.shape:
            return 0.0

        def _centroid(g: np.ndarray, color: int) -> Optional[tuple[int, int]]:
            ys, xs = np.where(g == color)
            if len(ys) == 0:
                return None
            return (int(round(float(ys.mean()))), int(round(float(xs.mean()))))

        prev_pos = _centroid(prev, character_color)
        new_pos = _centroid(curr, character_color)
        if prev_pos is None or new_pos is None:
            return 0.0
        ty, tx = target
        prev_dist = abs(prev_pos[0] - ty) + abs(prev_pos[1] - tx)
        new_dist = abs(new_pos[0] - ty) + abs(new_pos[1] - tx)
        # Normalize by grid dimension so reward magnitude is grid-size-independent
        norm = float(prev.shape[0] + prev.shape[1])
        return (prev_dist - new_dist) / max(norm, 1.0)
