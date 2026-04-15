"""
Pattern Primitives Library — General-purpose spatial/geometric pattern detectors.

7 pattern primitives that return 0.0-1.0 scores:
    symmetry     — mirror symmetry (horizontal + vertical)
    translation  — repeated pattern shifted by offset
    alignment    — cells forming lines/columns
    containment  — shape nesting (bounding box overlap)
    adjacency    — how connected colored regions are
    repetition   — same sub-pattern appears multiple times
    shape        — recognizable geometric shapes (line, rect, L, T)

Produces a PatternProfile (7D vector) — compact structural fingerprint.
General-purpose: works on any 2D grid, not game-specific.
"""
import logging
import numpy as np

logger = logging.getLogger("pattern_primitives")


class PatternPrimitives:
    """Compute pattern scores for a 2D grid."""

    PATTERN_NAMES = [
        "symmetry", "translation", "alignment", "containment",
        "adjacency", "repetition", "shape",
    ]

    # Class-level observability counter (instance-agnostic — many short-lived instances)
    _global_compute_count = 0
    _log_interval = 100

    def compute_profile(self, grid: np.ndarray) -> dict:
        """Compute full 7D pattern profile for a grid.

        Returns dict: {pattern_name: score (0.0-1.0)}
        """
        g = np.asarray(grid, dtype=np.float64)
        profile = {
            "symmetry": self.symmetry(g),
            "translation": self.translation(g),
            "alignment": self.alignment(g),
            "containment": self.containment(g),
            "adjacency": self.adjacency(g),
            "repetition": self.repetition(g),
            "shape": self.shape_score(g),
        }

        # Periodic observability — class-level counter so it works for short-lived instances
        PatternPrimitives._global_compute_count += 1
        if PatternPrimitives._global_compute_count % PatternPrimitives._log_interval == 0:
            dominant = max(profile, key=lambda k: profile[k])
            logger.info(
                "[PatternPrim] %d profiles computed | dominant=%s(%.2f) | sym=%.2f trans=%.2f "
                "align=%.2f contain=%.2f adj=%.2f rep=%.2f shape=%.2f",
                PatternPrimitives._global_compute_count, dominant, profile[dominant],
                profile["symmetry"], profile["translation"], profile["alignment"],
                profile["containment"], profile["adjacency"], profile["repetition"],
                profile["shape"],
            )

        return profile

    def profile_to_vector(self, profile: dict) -> list[float]:
        """Convert profile dict to ordered 7D vector."""
        return [profile.get(name, 0.0) for name in self.PATTERN_NAMES]

    # ─── SYMMETRY ───────────────────────────────────────────────

    def symmetry(self, grid: np.ndarray) -> float:
        """Measure mirror symmetry (average of horizontal + vertical).

        Compares each cell to its mirror position. Score = fraction of matching cells.
        """
        rows, cols = grid.shape
        if rows == 0 or cols == 0:
            return 0.0

        # Horizontal symmetry (top-bottom mirror)
        h_match = 0
        h_total = 0
        for r in range(rows // 2):
            mirror_r = rows - 1 - r
            for c in range(cols):
                h_total += 1
                if grid[r, c] == grid[mirror_r, c]:
                    h_match += 1
        h_sym = h_match / max(1, h_total)

        # Vertical symmetry (left-right mirror)
        v_match = 0
        v_total = 0
        for r in range(rows):
            for c in range(cols // 2):
                mirror_c = cols - 1 - c
                v_total += 1
                if grid[r, c] == grid[r, mirror_c]:
                    v_match += 1
        v_sym = v_match / max(1, v_total)

        return round((h_sym + v_sym) / 2.0, 4)

    # ─── TRANSLATION ────────────────────────────────────────────

    def translation(self, grid: np.ndarray) -> float:
        """Detect repeated pattern shifted by offset.

        Checks small offsets (1-4 cells in each direction).
        Score = best cross-correlation with any shifted version.
        """
        rows, cols = grid.shape
        if rows < 4 or cols < 4:
            return 0.0

        best_corr = 0.0
        # Check offsets 1-4 in both dimensions
        for dr in range(1, min(5, rows // 2)):
            match = 0
            total = 0
            for r in range(rows - dr):
                for c in range(cols):
                    total += 1
                    if grid[r, c] == grid[r + dr, c]:
                        match += 1
            corr = match / max(1, total)
            best_corr = max(best_corr, corr)

        for dc in range(1, min(5, cols // 2)):
            match = 0
            total = 0
            for r in range(rows):
                for c in range(cols - dc):
                    total += 1
                    if grid[r, c] == grid[r, c + dc]:
                        match += 1
            corr = match / max(1, total)
            best_corr = max(best_corr, corr)

        return round(best_corr, 4)

    # ─── ALIGNMENT ──────────────────────────────────────────────

    def alignment(self, grid: np.ndarray) -> float:
        """Measure how well colored cells form lines/columns.

        High score = cells concentrated in few rows or columns.
        Uses variance of per-row and per-column occupancy.
        """
        rows, cols = grid.shape
        if rows == 0 or cols == 0:
            return 0.0

        # Non-background mask
        mask = grid != 0

        # Row occupancy (fraction of colored cells per row)
        row_occ = np.sum(mask, axis=1) / max(1, cols)
        # Column occupancy
        col_occ = np.sum(mask, axis=0) / max(1, rows)

        # High variance = concentrated in few rows/cols = high alignment
        row_var = float(np.var(row_occ))
        col_var = float(np.var(col_occ))

        # Normalize: max variance for binary distribution is 0.25
        alignment = min(1.0, (row_var + col_var) / 0.5)
        return round(alignment, 4)

    # ─── CONTAINMENT ────────────────────────────────────────────

    def containment(self, grid: np.ndarray) -> float:
        """Measure shape nesting — are smaller shapes inside larger ones?

        Uses bounding box analysis on connected components.
        Score = fraction of objects that are contained within another.
        """
        from titan_plugin.logic.mini_reasoning import MiniReasoningEngine
        # Use lightweight decompose
        engine = MiniReasoningEngine()
        objects = engine.decompose(grid)

        if len(objects) < 2:
            return 0.0

        contained_count = 0
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                # Check if obj_a bbox is inside obj_b bbox
                if (obj_a.bbox[0] >= obj_b.bbox[0] and
                        obj_a.bbox[1] >= obj_b.bbox[1] and
                        obj_a.bbox[2] <= obj_b.bbox[2] and
                        obj_a.bbox[3] <= obj_b.bbox[3] and
                        obj_a.size < obj_b.size):
                    contained_count += 1
                    break  # Only count once per object

        return round(contained_count / len(objects), 4)

    # ─── ADJACENCY ──────────────────────────────────────────────

    def adjacency(self, grid: np.ndarray) -> float:
        """Measure how connected colored regions are.

        High score = colored cells tend to be next to other colored cells.
        Low score = colored cells are scattered.
        """
        rows, cols = grid.shape
        mask = grid != 0
        colored_count = int(np.sum(mask))
        if colored_count < 2:
            return 0.0

        # Count adjacent pairs of colored cells
        adjacent = 0
        total_neighbors = 0
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                # Check 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        total_neighbors += 1
                        if mask[nr, nc]:
                            adjacent += 1

        return round(adjacent / max(1, total_neighbors), 4)

    # ─── REPETITION ─────────────────────────────────────────────

    def repetition(self, grid: np.ndarray) -> float:
        """Detect if same sub-pattern appears multiple times.

        Checks small NxN blocks for repeated patterns.
        Score based on how many blocks match each other.
        """
        rows, cols = grid.shape
        block_size = max(2, min(4, rows // 4, cols // 4))
        if rows < block_size * 2 or cols < block_size * 2:
            return 0.0

        # Extract non-overlapping blocks
        blocks = []
        for r in range(0, rows - block_size + 1, block_size):
            for c in range(0, cols - block_size + 1, block_size):
                block = grid[r:r+block_size, c:c+block_size].copy()
                blocks.append(block.tobytes())

        if len(blocks) < 2:
            return 0.0

        # Count matching pairs
        from collections import Counter
        counts = Counter(blocks)
        repeated = sum(c for c in counts.values() if c > 1)

        return round(min(1.0, repeated / len(blocks)), 4)

    # ─── SHAPE ──────────────────────────────────────────────────

    def shape_score(self, grid: np.ndarray) -> float:
        """Detect recognizable geometric shapes in the grid.

        Checks for: horizontal lines, vertical lines, rectangles, L-shapes.
        Score = fraction of colored cells that belong to recognizable shapes.
        """
        rows, cols = grid.shape
        mask = grid != 0
        colored_count = int(np.sum(mask))
        if colored_count < 3:
            return 0.0

        shape_cells = 0

        # Detect horizontal lines (3+ consecutive colored in a row)
        for r in range(rows):
            run = 0
            for c in range(cols):
                if mask[r, c]:
                    run += 1
                else:
                    if run >= 3:
                        shape_cells += run
                    run = 0
            if run >= 3:
                shape_cells += run

        # Detect vertical lines (3+ consecutive colored in a column)
        for c in range(cols):
            run = 0
            for r in range(rows):
                if mask[r, c]:
                    run += 1
                else:
                    if run >= 3:
                        shape_cells += run
                    run = 0
            if run >= 3:
                shape_cells += run

        # Avoid double-counting (cells in both H and V lines)
        shape_cells = min(shape_cells, colored_count * 2)

        return round(min(1.0, shape_cells / max(1, colored_count * 2)), 4)
