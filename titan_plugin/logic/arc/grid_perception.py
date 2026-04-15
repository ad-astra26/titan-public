"""
titan_plugin/logic/arc/grid_perception.py — ARC-AGI-3 Grid Perception Engine.

Encodes ARC-AGI-3 game frames (numpy int8 arrays, typically 64×64, values 0-12)
into Titan-compatible Trinity tensor features using pure NumPy.

No CNN needed — grids are small enough for direct mathematical feature extraction.
Same philosophy as Titan's Vision Sense: pure math on raw data, no pretrained models.

Grid → Trinity Mapping:
  Inner Body (5D): Physical grid features (density, entropy, symmetry, edges, frequency)
  Inner Mind (5D): Pattern features (repetition, objects, variance, delta, progress)
  Inner Spirit (5D): Episode state (exploration, reward trend, stuck, confidence, progress)
"""
import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GridPerception:
    """
    Encode ARC-AGI-3 grid frames into Titan Trinity tensor format.

    All features are normalized to [0, 1] for direct tensor injection.
    Maintains episode state for delta/progress tracking.
    """

    def __init__(self, max_steps: int = 200):
        self._max_steps = max_steps
        self._prev_grid: Optional[np.ndarray] = None
        self._frame_history: list[np.ndarray] = []
        self._reward_history: list[float] = []
        self._unique_states: set[bytes] = set()
        self._steps_since_progress: int = 0
        self._step_count: int = 0

    def perceive(self, grid,
                 reward: float = 0.0,
                 available_actions: int = 0,
                 step: int = -1,
                 **kwargs) -> dict:
        """
        Convert a game frame to Titan Trinity tensor updates.

        Args:
            grid: numpy ndarray (64x64 int8, values 0-12) or 2D list of ints
            reward: Reward from last action (0.0 if first frame)
            available_actions: Number of available actions
            step: Current step number (-1 = auto-increment)

        Returns: {
            "inner_body": [5 floats],
            "inner_mind": [5 floats],
            "inner_spirit": [5 floats],
        }
        """
        g = np.asarray(grid, dtype=np.float64)
        if step >= 0:
            self._step_count = step
        else:
            self._step_count += 1

        # Track state for episode features
        state_hash = g.tobytes()
        is_new_state = state_hash not in self._unique_states
        self._unique_states.add(state_hash)
        self._reward_history.append(reward)
        self._frame_history.append(g)

        if reward > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        # Extract features
        body = self._extract_body(g)
        mind = self._extract_mind(g)
        spirit = self._extract_spirit(reward, is_new_state, available_actions)
        spatial = self._extract_spatial(g)  # Improvement #4: WHERE things change
        semantic = self._extract_semantic(g, body, mind)  # 30D expansion: meaning
        resonance = self._extract_resonance(body, mind, spirit)  # 30D: inner echo

        # Update prev grid
        self._prev_grid = g.copy()

        result = {
            "inner_body": [round(v, 4) for v in body],
            "inner_mind": [round(v, 4) for v in mind],
            "inner_spirit": [round(v, 4) for v in spirit],
            "spatial": [round(v, 4) for v in spatial],
            "semantic": [round(v, 4) for v in semantic],
            "resonance": [round(v, 4) for v in resonance],
        }

        # A5: Multi-scale perception — ACh-gated quadrant focus
        ach_level = kwargs.get("ach_level", 0.5)
        focused_body, focused_mind = self._extract_focused_features(g, ach_level)
        result["focused_body"] = [round(v, 4) for v in focused_body]
        result["focused_mind"] = [round(v, 4) for v in focused_mind]

        return result

    def reset(self):
        """Reset episode state for new game."""
        self._prev_grid = None
        self._frame_history = []
        self._reward_history = []
        self._unique_states = set()
        self._steps_since_progress = 0
        self._step_count = 0

    # ── Inner Body (5D): Physical Grid Features ────────────────────

    def _extract_body(self, grid: np.ndarray) -> list[float]:
        """
        Physical grid properties — what the grid LOOKS like.

        [0] density:       fraction of non-zero cells
        [1] color_entropy: Shannon entropy of color distribution
        [2] symmetry:      horizontal + vertical mirror correlation
        [3] edge_density:  fraction of cell boundaries with different values
        [4] spatial_freq:  mean FFT magnitude (busy vs calm)
        """
        rows, cols = grid.shape
        total = rows * cols

        # [0] Density
        density = float(np.count_nonzero(grid)) / max(1, total)

        # [1] Color entropy
        color_entropy = self._shannon_entropy(grid)

        # [2] Symmetry (average of horizontal + vertical mirror)
        h_sym = self._mirror_symmetry(grid, axis="horizontal")
        v_sym = self._mirror_symmetry(grid, axis="vertical")
        symmetry = (h_sym + v_sym) / 2.0

        # [3] Edge density
        edge_density = self._edge_density(grid)

        # [4] Spatial frequency (FFT)
        spatial_freq = self._spatial_frequency(grid)

        return [density, color_entropy, symmetry, edge_density, spatial_freq]

    # ── Inner Mind (5D): Pattern Features ──────────────────────────

    def _extract_mind(self, grid: np.ndarray) -> list[float]:
        """
        Pattern recognition features — what the grid MEANS.

        [0] object_count:   number of connected components (normalized)
        [1] size_variance:  variance in object sizes
        [2] color_count:    number of distinct colors used (normalized)
        [3] delta_from_prev: change from previous frame
        [4] pattern_score:  regularity / repeating pattern detection
        """
        # Downsample large grids for BFS performance
        small = self._downsample(grid.astype(np.int16), target=16) if grid.shape[0] > 16 else grid.astype(np.int16)

        # [0] Object count (connected components)
        objects = self._count_objects(small)
        object_count = min(1.0, objects / 20.0)  # normalize

        # [1] Size variance
        sizes = self._object_sizes(small)
        if len(sizes) > 1:
            size_variance = min(1.0, float(np.std(sizes)) / (float(np.mean(sizes)) + 1e-10))
        else:
            size_variance = 0.0

        # [2] Color count
        unique_colors = len(np.unique(grid))
        color_count = min(1.0, unique_colors / 10.0)  # max 10 colors

        # [3] Delta from previous frame
        if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
            diff = np.sum(grid != self._prev_grid)
            delta = float(diff) / max(1, grid.size)
        else:
            delta = 1.0  # First frame = maximum novelty

        # [4] Pattern score (row/column repetition)
        pattern_score = self._pattern_regularity(grid)

        return [object_count, size_variance, color_count, delta, pattern_score]

    # ── Inner Spirit (5D): Episode State ───────────────────────────

    def _extract_spirit(self, reward: float, is_new_state: bool,
                        available_actions: int) -> list[float]:
        """
        Episode-level state — how the JOURNEY is going.

        [0] exploration_rate:  unique states / total steps
        [1] reward_trend:      recent reward trajectory (slope)
        [2] stuck_indicator:   steps since last progress / max
        [3] action_diversity:  available actions / expected
        [4] episode_progress:  current step / max steps
        """
        # [0] Exploration rate
        total_steps = max(1, self._step_count)
        exploration_rate = min(1.0, len(self._unique_states) / total_steps)

        # [1] Reward trend (slope of recent rewards)
        if len(self._reward_history) >= 3:
            recent = self._reward_history[-10:]
            x = np.arange(len(recent), dtype=np.float64)
            if np.std(x) > 0:
                slope = np.corrcoef(x, recent)[0, 1]
                reward_trend = (float(slope) + 1.0) / 2.0 if np.isfinite(slope) else 0.5
            else:
                reward_trend = 0.5
        else:
            reward_trend = 0.5

        # [2] Stuck indicator
        stuck = min(1.0, self._steps_since_progress / max(1.0, self._max_steps * 0.2))

        # [3] Action diversity
        action_diversity = min(1.0, available_actions / 10.0) if available_actions > 0 else 0.5

        # [4] Episode progress
        progress = min(1.0, self._step_count / max(1.0, self._max_steps))

        return [exploration_rate, reward_trend, stuck, action_diversity, progress]

    # ── Spatial Awareness (5D): WHERE things change ────────────────
    # General-purpose spatial perception — useful for ARC puzzles,
    # art self-assessment, visual sense, environment understanding.

    def _extract_spatial(self, grid: np.ndarray) -> list[float]:
        """
        Spatial cause-effect features — WHERE and HOW the grid changed.

        These are general perceptual primitives, not game-specific:
        [0] change_centroid_x: horizontal center of changes (0=left, 1=right)
        [1] change_centroid_y: vertical center of changes (0=top, 1=bottom)
        [2] change_dispersion: clustered (0) vs scattered (1) changes
        [3] change_direction: dominant movement direction (0=left/up, 1=right/down)
        [4] new_colors_ratio: fraction of novel colors introduced
        """
        if self._prev_grid is None or self._prev_grid.shape != grid.shape:
            return [0.5, 0.5, 0.0, 0.5, 0.0]  # neutral on first frame

        diff_mask = (grid != self._prev_grid)
        if not diff_mask.any():
            return [0.5, 0.5, 0.0, 0.5, 0.0]  # nothing changed

        rows, cols = grid.shape
        ys, xs = np.where(diff_mask)

        # [0] Horizontal center of change
        cx = float(xs.mean()) / max(1, cols - 1)

        # [1] Vertical center of change
        cy = float(ys.mean()) / max(1, rows - 1)

        # [2] Dispersion: std of change positions (clustered vs scattered)
        if len(xs) > 1:
            disp_x = float(np.std(xs)) / max(1, cols)
            disp_y = float(np.std(ys)) / max(1, rows)
            dispersion = min(1.0, (disp_x + disp_y) / 2.0)
        else:
            dispersion = 0.0

        # [3] Change direction: compare change centroid to grid center
        # >0.5 = changes shifted right/down, <0.5 = shifted left/up
        # This tells the agent which direction its action affected
        grid_cx = cols / 2.0
        grid_cy = rows / 2.0
        dx = (float(xs.mean()) - grid_cx) / max(1, cols)
        dy = (float(ys.mean()) - grid_cy) / max(1, rows)
        direction = min(1.0, max(0.0, 0.5 + (dx + dy) / 2.0))

        # [4] New colors: fraction of colors in changed cells that are novel
        prev_colors = set(self._prev_grid[diff_mask].flatten().tolist())
        cur_colors = set(grid[diff_mask].flatten().tolist())
        new_colors = cur_colors - prev_colors
        new_ratio = len(new_colors) / max(1, len(cur_colors)) if cur_colors else 0.0

        return [cx, cy, dispersion, direction, new_ratio]

    # ── Helper Functions ───────────────────────────────────────────

    @staticmethod
    def _downsample(grid: np.ndarray, target: int = 16) -> np.ndarray:
        """Downsample grid to target size via block-mode for BFS performance.

        For 64x64 → 16x16: each 4x4 block takes the most common non-zero value
        (or 0 if all zeros). Preserves object structure better than averaging.
        """
        rows, cols = grid.shape
        if rows <= target and cols <= target:
            return grid
        br = max(1, rows // target)
        bc = max(1, cols // target)
        out_r = min(target, rows // br)
        out_c = min(target, cols // bc)
        result = np.zeros((out_r, out_c), dtype=grid.dtype)
        for r in range(out_r):
            for c in range(out_c):
                block = grid[r*br:(r+1)*br, c*bc:(c+1)*bc].flatten()
                nonzero = block[block != 0]
                if len(nonzero) > 0:
                    vals, counts = np.unique(nonzero, return_counts=True)
                    result[r, c] = vals[np.argmax(counts)]
        return result

    @staticmethod
    def _shannon_entropy(grid: np.ndarray) -> float:
        """Shannon entropy of color distribution, normalized to [0, 1]."""
        flat = grid.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        if len(unique) <= 1:
            return 0.0
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(13)  # max 13 values (0-12)
        return min(1.0, entropy / max_entropy)

    @staticmethod
    def _mirror_symmetry(grid: np.ndarray, axis: str = "horizontal") -> float:
        """Correlation between grid and its mirror."""
        if axis == "horizontal":
            mirrored = grid[:, ::-1]
        else:
            mirrored = grid[::-1, :]
        matching = np.sum(grid == mirrored)
        return float(matching) / max(1, grid.size)

    @staticmethod
    def _edge_density(grid: np.ndarray) -> float:
        """Fraction of adjacent cell pairs with different values."""
        rows, cols = grid.shape
        total_edges = 0
        diff_edges = 0

        # Horizontal edges
        if cols > 1:
            h_diff = grid[:, :-1] != grid[:, 1:]
            total_edges += h_diff.size
            diff_edges += int(h_diff.sum())

        # Vertical edges
        if rows > 1:
            v_diff = grid[:-1, :] != grid[1:, :]
            total_edges += v_diff.size
            diff_edges += int(v_diff.sum())

        return float(diff_edges) / max(1, total_edges)

    @staticmethod
    def _spatial_frequency(grid: np.ndarray) -> float:
        """Mean FFT magnitude (high = busy pattern, low = calm)."""
        if grid.size == 0:
            return 0.0
        fft = np.fft.fft2(grid)
        magnitudes = np.abs(fft)
        # Exclude DC component
        magnitudes[0, 0] = 0
        mean_mag = float(magnitudes.mean())
        # Normalize by grid size
        return min(1.0, mean_mag / (grid.size ** 0.5 + 1e-10))

    @staticmethod
    def _count_objects(grid: np.ndarray) -> int:
        """Count connected components (4-connectivity) of non-zero cells."""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        count = 0

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    # BFS flood fill
                    count += 1
                    color = grid[r, c]
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (0 <= cr < rows and 0 <= cc < cols and
                                not visited[cr, cc] and grid[cr, cc] == color):
                            visited[cr, cc] = True
                            stack.extend([(cr+1, cc), (cr-1, cc),
                                          (cr, cc+1), (cr, cc-1)])
        return count

    @staticmethod
    def _object_sizes(grid: np.ndarray) -> list[int]:
        """Get sizes of all connected components."""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        sizes = []

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    color = grid[r, c]
                    size = 0
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (0 <= cr < rows and 0 <= cc < cols and
                                not visited[cr, cc] and grid[cr, cc] == color):
                            visited[cr, cc] = True
                            size += 1
                            stack.extend([(cr+1, cc), (cr-1, cc),
                                          (cr, cc+1), (cr, cc-1)])
                    sizes.append(size)
        return sizes

    @staticmethod
    def _pattern_regularity(grid: np.ndarray) -> float:
        """Detect repeating patterns in rows/columns."""
        rows, cols = grid.shape
        row_repeats = 0
        col_repeats = 0

        # Check row repetition
        for r in range(1, rows):
            if np.array_equal(grid[r], grid[r - 1]):
                row_repeats += 1

        # Check column repetition
        for c in range(1, cols):
            if np.array_equal(grid[:, c], grid[:, c - 1]):
                col_repeats += 1

        total = max(1, (rows - 1) + (cols - 1))
        return float(row_repeats + col_repeats) / total

    # ── Semantic Features (5D): What does this image MEAN? ─────────

    def _extract_semantic(self, grid: np.ndarray,
                          body: list[float], mind: list[float]) -> list[float]:
        """
        Higher-order meaning features — derived from body+mind primitives.

        [0] complexity_score:   cognitive load (objects × color × edges)
        [1] beauty_score:       harmony × symmetry (felt beauty)
        [2] emotional_warmth:   color warmth proxy (density × entropy balance)
        [3] structural_order:   regularity × symmetry (ordered vs chaotic)
        [4] narrative_weight:   delta magnitude (how much this frame matters)
        """
        # [0] Complexity: product of object count, color diversity, edge density
        complexity = min(1.0, mind[0] * mind[2] * body[3] * 8.0)

        # [1] Beauty: harmony between symmetry and spatial frequency
        beauty = body[2] * (1.0 - abs(body[4] - 0.5) * 2)  # peak when freq is moderate

        # [2] Warmth: dense + colorful = warm; sparse + monotone = cold
        warmth = (body[0] * 0.6 + body[1] * 0.4)

        # [3] Order: high regularity + high symmetry = ordered
        structural_order = (mind[4] * 0.6 + body[2] * 0.4)

        # [4] Narrative weight: how much changed (delta is importance signal)
        narrative_weight = mind[3]  # delta_from_prev already normalized

        return [complexity, beauty, warmth, structural_order, narrative_weight]

    # ── Resonance Features (5D): How does this echo inside? ──────

    def _extract_resonance(self, body: list[float], mind: list[float],
                           spirit: list[float]) -> list[float]:
        """
        Cross-layer resonance — how body/mind/spirit features harmonize.

        [0] body_mind_coherence:  correlation between physical and pattern features
        [1] exploration_reward:   spirit exploration × mind novelty
        [2] stuck_frustration:    spirit stuck × low mind delta (nothing changing)
        [3] progress_confidence:  spirit progress × mind pattern clarity
        [4] overall_harmony:      mean deviation from 0.5 across all layers
        """
        # [0] Body-mind coherence: do physical and pattern features align?
        bm_diff = sum(abs(b - m) for b, m in zip(body, mind)) / 5.0
        coherence = 1.0 - min(1.0, bm_diff)

        # [1] Exploration reward: exploring AND finding novel things
        exploration_reward = spirit[0] * mind[3]  # exploration_rate × delta

        # [2] Stuck frustration: stuck AND nothing changing
        stuck_frustration = spirit[2] * (1.0 - mind[3])  # stuck × low delta

        # [3] Progress confidence: making progress AND patterns are clear
        progress_confidence = spirit[4] * mind[4]  # episode_progress × regularity

        # [4] Overall harmony: how close to center (divine center = 0.5)
        all_vals = body + mind + spirit
        harmony = 1.0 - min(1.0, sum(abs(v - 0.5) for v in all_vals) / len(all_vals))

        return [coherence, exploration_reward, stuck_frustration,
                progress_confidence, harmony]

    # ── A5: Multi-Scale Perception ──────────────────���───────────────

    def _extract_focused_features(self, grid: np.ndarray,
                                  ach_level: float = 0.5) -> tuple:
        """ACh-gated quadrant focus (Phase A5).

        High ACh (>0.6) = focus on the most-changed quadrant.
        Low ACh (<0.4) = global average of all quadrants.
        In between = weighted blend.

        Returns:
            (focused_body: list[5], focused_mind: list[5])
        """
        rows, cols = grid.shape
        if rows < 4 or cols < 4:
            # Grid too small for quadrant split — return global features
            return self._extract_body(grid), self._extract_mind(grid)

        mid_r, mid_c = rows // 2, cols // 2
        quads = [
            grid[:mid_r, :mid_c],    # top-left
            grid[:mid_r, mid_c:],    # top-right
            grid[mid_r:, :mid_c],    # bottom-left
            grid[mid_r:, mid_c:],    # bottom-right
        ]

        quad_bodies = [self._extract_body(q) for q in quads]
        quad_minds = [self._extract_mind(q) for q in quads]

        # Find most-changed quadrant (requires prev_grid)
        focus_idx = self._find_focus_quadrant(grid, mid_r, mid_c)

        # Global average
        global_body = [sum(qb[i] for qb in quad_bodies) / 4.0 for i in range(5)]
        global_mind = [sum(qm[i] for qm in quad_minds) / 4.0 for i in range(5)]

        # ACh-gated blend: high ACh → focused, low ACh → global
        alpha = max(0.0, min(1.0, (ach_level - 0.3) / 0.4))  # ramp from 0.3-0.7
        focused_body = [alpha * quad_bodies[focus_idx][i] + (1 - alpha) * global_body[i]
                        for i in range(5)]
        focused_mind = [alpha * quad_minds[focus_idx][i] + (1 - alpha) * global_mind[i]
                        for i in range(5)]

        return focused_body, focused_mind

    def _find_focus_quadrant(self, grid: np.ndarray,
                             mid_r: int, mid_c: int) -> int:
        """Find the quadrant with the most change from previous frame.

        Returns 0-3 (quadrant index). Falls back to 0 if no previous frame.
        """
        if self._prev_grid is None or self._prev_grid.shape != grid.shape:
            return 0

        diff = (grid != self._prev_grid)
        # Count changes per quadrant
        counts = [
            int(diff[:mid_r, :mid_c].sum()),
            int(diff[:mid_r, mid_c:].sum()),
            int(diff[mid_r:, :mid_c].sum()),
            int(diff[mid_r:, mid_c:].sum()),
        ]
        return counts.index(max(counts))

    def get_stats(self) -> dict:
        """Return current perception state for debugging."""
        return {
            "step_count": self._step_count,
            "unique_states": len(self._unique_states),
            "frame_history_size": len(self._frame_history),
            "reward_history_size": len(self._reward_history),
            "steps_since_progress": self._steps_since_progress,
        }
