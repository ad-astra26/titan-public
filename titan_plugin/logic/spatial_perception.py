"""
titan_plugin/logic/spatial_perception.py — General-Purpose Spatial Perception.

Extracts 30D visual features from RGB images, mapped to Titan's outer Trinity.
Pure function: no bus access, no state mutation beyond internal frame tracking.

Feature groups (6 x 5D = 30D):
  Physical (5D) → outer_body[0:5]       — what the image LOOKS like
  Pattern  (5D) → oMind Feeling[5:10]   — what PATTERNS exist
  Spatial  (5D) → modulates Physical     — WHERE things change
  Semantic (5D) → logged, not wired yet  — what the image MEANS
  Journey  (5D) → logged, not wired yet  — visual experience over time
  Resonance(5D) → logged, not wired yet  — cross-modal harmony

Wired to Trinity: 10D (Physical + Pattern). Spatial modulates Physical.
Remaining 15D available via creative journal and future enrichment paths.
"""
import hashlib
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class SpatialPerception:
    """General-purpose visual perception: RGB image -> 30D features."""

    def __init__(self):
        self._prev_gray: np.ndarray | None = None  # for delta/spatial tracking
        self._seen_hashes: set = set()
        self._seen_count: int = 0
        self._novelty_history: list[float] = []

    def perceive(self, rgb_array: np.ndarray) -> dict:
        """
        Extract 30D features from an RGB image array (H, W, 3).

        Args:
            rgb_array: numpy array of shape (H, W, 3), dtype float64, values 0-255.

        Returns:
            Dict with keys: physical, pattern, spatial, semantic, journey,
            resonance (each list[5 floats] in [0,1]), and flat_30d (list[30]).
        """
        # Convert to grayscale for structural features
        gray = np.mean(rgb_array, axis=2)

        # Quantize for BFS object detection: 16 levels on 16x16 grid
        gray_q = (gray * 15 / 255.0).astype(np.int16)
        small = self._downsample(gray_q, target=16)

        # Image hash for journey tracking
        img_hash = hashlib.md5(gray.astype(np.float32).tobytes()).hexdigest()[:12]
        is_new = img_hash not in self._seen_hashes
        self._seen_hashes.add(img_hash)
        self._seen_count += 1

        # Extract all 6 feature groups
        physical = self._extract_physical(rgb_array, gray)
        pattern = self._extract_pattern(gray, gray_q, small)
        spatial = self._extract_spatial(gray)
        semantic = self._extract_semantic(physical, pattern)
        journey = self._extract_journey(is_new)
        resonance = self._extract_resonance(physical, semantic)

        # Update prev frame for next call
        self._prev_gray = gray.copy()

        # Round all values
        def _r(vals):
            return [round(max(0.0, min(1.0, v)), 4) for v in vals]

        phys_r = _r(physical)
        patt_r = _r(pattern)
        spat_r = _r(spatial)
        sem_r = _r(semantic)
        jour_r = _r(journey)
        res_r = _r(resonance)

        return {
            "physical": phys_r,
            "pattern": patt_r,
            "spatial": spat_r,
            "semantic": sem_r,
            "journey": jour_r,
            "resonance": res_r,
            "flat_30d": phys_r + patt_r + spat_r + sem_r + jour_r + res_r,
        }

    # ── Physical (5D): What the image LOOKS like ─────────────────────

    def _extract_physical(self, rgb: np.ndarray, gray: np.ndarray) -> list[float]:
        """
        [0] color_entropy:  information density of RGB histogram
        [1] edge_density:   fraction of strong gradient pixels
        [2] symmetry:       left-right mirror correlation
        [3] spatial_freq:   FFT high-frequency energy ratio
        [4] harmony:        combined beauty measure
        """
        color_entropy = self._rgb_color_entropy(rgb)
        edge_density = self._edge_density_sobel(gray)
        symmetry = self._mirror_symmetry_corr(gray)
        spatial_freq = self._spatial_frequency_fft(gray)
        harmony = (color_entropy * 0.3 + symmetry * 0.4 +
                   (1.0 - abs(spatial_freq - 0.5) * 2) * 0.3)
        return [color_entropy, edge_density, symmetry, spatial_freq, harmony]

    # ── Pattern (5D): What PATTERNS exist ────────────────────────────

    def _extract_pattern(self, gray: np.ndarray, gray_q: np.ndarray,
                         small: np.ndarray) -> list[float]:
        """
        [0] object_count:      connected components (normalized)
        [1] size_variance:     variance in object sizes
        [2] color_count:       chromatic diversity (distinct gray levels)
        [3] delta_from_prev:   change from previous image
        [4] pattern_regularity: repeating row/column patterns
        """
        # [0] Object count via BFS on quantized 16x16
        objects = self._count_objects(small)
        object_count = min(1.0, objects / 20.0)

        # [1] Size variance
        sizes = self._object_sizes(small)
        if len(sizes) > 1:
            size_variance = min(1.0, float(np.std(sizes)) / (float(np.mean(sizes)) + 1e-10))
        else:
            size_variance = 0.0

        # [2] Color count (unique quantized levels)
        unique_colors = len(np.unique(gray_q))
        color_count = min(1.0, unique_colors / 12.0)

        # [3] Delta from previous
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = np.mean(np.abs(gray - self._prev_gray)) / 255.0
            delta = min(1.0, diff * 4.0)  # Scale: 25% mean change = 1.0
        else:
            delta = 1.0  # First image = maximum novelty

        # [4] Pattern regularity
        pattern_regularity = self._pattern_regularity(small)

        return [object_count, size_variance, color_count, delta, pattern_regularity]

    # ── Spatial (5D): WHERE things change ────────────────────────────

    def _extract_spatial(self, gray: np.ndarray) -> list[float]:
        """
        [0] change_centroid_x: horizontal center of changes
        [1] change_centroid_y: vertical center of changes
        [2] change_dispersion: clustered vs scattered changes
        [3] change_direction:  dominant movement direction
        [4] new_regions_ratio: fraction of significantly changed regions
        """
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return [0.5, 0.5, 0.0, 0.5, 0.0]

        diff = np.abs(gray - self._prev_gray)
        threshold = 15.0  # Minimum gray-level change to count
        diff_mask = diff > threshold

        if not diff_mask.any():
            return [0.5, 0.5, 0.0, 0.5, 0.0]

        rows, cols = gray.shape
        ys, xs = np.where(diff_mask)

        # [0] Horizontal center of change
        cx = float(xs.mean()) / max(1, cols - 1)

        # [1] Vertical center of change
        cy = float(ys.mean()) / max(1, rows - 1)

        # [2] Dispersion
        if len(xs) > 1:
            disp_x = float(np.std(xs)) / max(1, cols)
            disp_y = float(np.std(ys)) / max(1, rows)
            dispersion = min(1.0, (disp_x + disp_y) / 2.0)
        else:
            dispersion = 0.0

        # [3] Direction: offset from center
        grid_cx = cols / 2.0
        grid_cy = rows / 2.0
        dx = (float(xs.mean()) - grid_cx) / max(1, cols)
        dy = (float(ys.mean()) - grid_cy) / max(1, rows)
        direction = min(1.0, max(0.0, 0.5 + (dx + dy) / 2.0))

        # [4] New regions ratio: fraction of pixels with significant change
        new_ratio = float(diff_mask.sum()) / max(1, diff_mask.size)

        return [cx, cy, dispersion, direction, min(1.0, new_ratio * 5.0)]

    # ── Semantic (5D): What the image MEANS ──────────────────────────

    def _extract_semantic(self, physical: list[float],
                          pattern: list[float]) -> list[float]:
        """
        Derived from Physical + Pattern (no new computation).
        [0] complexity:      cognitive demand (edges x colors x objects)
        [1] beauty:          harmony from physical
        [2] warmth:          warm vs cool (derived from color entropy + harmony)
        [3] structural_order: symmetry x regularity
        [4] narrative_weight: delta (how much this differs from last)
        """
        complexity = min(1.0, physical[1] * pattern[2] * pattern[0] * 8.0)
        beauty = physical[4]  # harmony
        # Warmth approximation: high entropy + high harmony = warm
        warmth = min(1.0, (physical[0] * 0.6 + physical[4] * 0.4))
        structural_order = min(1.0, physical[2] * 0.6 + pattern[4] * 0.4)
        narrative_weight = pattern[3]  # delta from previous
        return [complexity, beauty, warmth, structural_order, narrative_weight]

    # ── Journey (5D): Visual experience over time ────────────────────

    def _extract_journey(self, is_new: bool) -> list[float]:
        """
        [0] exploration_rate:  unique images / total images seen
        [1] novelty_trend:     recent novelty direction
        [2] stuck_indicator:   seeing same things repeatedly
        [3] source_diversity:  (placeholder, always 0.5)
        [4] image_count_norm:  how many images seen (normalized)
        """
        total = max(1, self._seen_count)
        exploration_rate = min(1.0, len(self._seen_hashes) / total)

        # Track novelty
        self._novelty_history.append(1.0 if is_new else 0.0)
        if len(self._novelty_history) > 50:
            self._novelty_history = self._novelty_history[-50:]

        # Novelty trend: compare recent vs older
        if len(self._novelty_history) >= 6:
            recent = np.mean(self._novelty_history[-3:])
            older = np.mean(self._novelty_history[-6:-3])
            novelty_trend = min(1.0, max(0.0, 0.5 + (recent - older)))
        else:
            novelty_trend = 0.5

        # Stuck: low exploration rate = seeing repeats
        stuck = 1.0 - exploration_rate

        image_count_norm = min(1.0, self._seen_count / 100.0)

        return [exploration_rate, novelty_trend, stuck, 0.5, image_count_norm]

    # ── Resonance (5D): Cross-modal harmony ──────────────────────────

    @staticmethod
    def _extract_resonance(physical: list[float],
                           semantic: list[float]) -> list[float]:
        """
        Cross-products of Physical and Semantic features.
        Not wired to Trinity yet — logged for future use.
        """
        harmony = physical[4]
        felt_impact = semantic[4]  # narrative_weight (delta)
        creative_resonance = min(1.0, harmony * semantic[0])  # beauty × complexity
        temporal_context = semantic[3]  # structural_order
        integration_depth = min(1.0, (harmony + semantic[1]) / 2.0)
        return [harmony, felt_impact, creative_resonance, temporal_context, integration_depth]

    # ── Helper methods ───────────────────────────────────────────────

    @staticmethod
    def _rgb_color_entropy(rgb: np.ndarray) -> float:
        """Shannon entropy of RGB color histogram, normalized to 0-1."""
        bins = 32
        entropy_sum = 0.0
        for ch in range(min(3, rgb.shape[2] if rgb.ndim == 3 else 1)):
            channel = rgb[:, :, ch] if rgb.ndim == 3 else rgb
            hist, _ = np.histogram(channel, bins=bins, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            h = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
            entropy_sum += h
        max_entropy = 3 * math.log2(bins)
        return min(1.0, entropy_sum / max_entropy)

    @staticmethod
    def _edge_density_sobel(gray: np.ndarray) -> float:
        """Fraction of pixels with strong gradient (Sobel-like), 0-1."""
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        min_h = min(gx.shape[0], gy.shape[0])
        min_w = min(gx.shape[1], gy.shape[1])
        magnitude = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)
        edge_count = np.sum(magnitude > 25.5)
        denom = magnitude.size * 0.5
        return min(1.0, edge_count / denom) if denom > 0 else 0.0

    @staticmethod
    def _mirror_symmetry_corr(gray: np.ndarray) -> float:
        """Left-right symmetry via Pearson correlation, 0-1."""
        w = gray.shape[1]
        if w < 4:
            return 0.5
        left = gray[:, :w // 2]
        right = gray[:, w // 2:w // 2 + left.shape[1]][:, ::-1]
        if left.shape != right.shape:
            min_w = min(left.shape[1], right.shape[1])
            left, right = left[:, :min_w], right[:, :min_w]
        corr = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        if np.isnan(corr):
            return 0.5
        return max(0.0, min(1.0, (corr + 1.0) / 2.0))

    @staticmethod
    def _spatial_frequency_fft(gray: np.ndarray) -> float:
        """Ratio of high-frequency energy in 2D FFT, 0-1."""
        if gray.size == 0:
            return 0.0
        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        low_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= r ** 2
        total_energy = np.sum(magnitude ** 2) + 1e-10
        low_energy = np.sum(magnitude[low_mask] ** 2)
        return min(1.0, max(0.0, 1.0 - low_energy / total_energy))

    @staticmethod
    def _downsample(grid: np.ndarray, target: int = 16) -> np.ndarray:
        """Downsample grid to target size via block-mode for BFS performance."""
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
                block = grid[r * br:(r + 1) * br, c * bc:(c + 1) * bc].flatten()
                nonzero = block[block != 0]
                if len(nonzero) > 0:
                    vals, counts = np.unique(nonzero, return_counts=True)
                    result[r, c] = vals[np.argmax(counts)]
        return result

    @staticmethod
    def _count_objects(grid: np.ndarray) -> int:
        """Count connected components (4-connectivity) of non-zero cells."""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    count += 1
                    color = grid[r, c]
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (0 <= cr < rows and 0 <= cc < cols
                                and not visited[cr, cc] and grid[cr, cc] == color):
                            visited[cr, cc] = True
                            stack.extend([(cr + 1, cc), (cr - 1, cc),
                                          (cr, cc + 1), (cr, cc - 1)])
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
                        if (0 <= cr < rows and 0 <= cc < cols
                                and not visited[cr, cc] and grid[cr, cc] == color):
                            visited[cr, cc] = True
                            size += 1
                            stack.extend([(cr + 1, cc), (cr - 1, cc),
                                          (cr, cc + 1), (cr, cc - 1)])
                    sizes.append(size)
        return sizes

    @staticmethod
    def _pattern_regularity(grid: np.ndarray) -> float:
        """Detect repeating patterns in rows/columns."""
        rows, cols = grid.shape
        row_repeats = 0
        col_repeats = 0
        for r in range(1, rows):
            if np.array_equal(grid[r], grid[r - 1]):
                row_repeats += 1
        for c in range(1, cols):
            if np.array_equal(grid[:, c], grid[:, c - 1]):
                col_repeats += 1
        total = max(1, (rows - 1) + (cols - 1))
        return float(row_repeats + col_repeats) / total
