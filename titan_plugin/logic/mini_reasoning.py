"""
Mini-Reasoning Layer — Fast, automatic perceptual grounding for meta-reasoning.

System 1 cognition: runs every step, produces structured PatternBlocks
that meta-reasoning (System 2) can reason about meaningfully.

5 primitives:
    DECOMPOSE — break grid into objects (connected components)
    FILTER    — focus on what changed between steps
    DISTILL   — reduce delta to single salient observation
    COMPARE   — structural difference between two states
    MATCH     — check against pattern primitive library

Also tracks:
    - Action-Effect Causal Memory (which actions produce which effects)
    - Pattern Trend Window (sliding window of pattern profiles)
    - Surprise Signal (unexpected pattern changes → NS boost)

General-purpose: nothing here is game-specific.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger("mini_reasoning")


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class GridObject:
    """A connected component in the grid."""
    obj_id: int
    color: int
    cells: list  # list of (row, col) tuples
    size: int
    bbox: tuple  # (min_row, min_col, max_row, max_col)
    centroid: tuple  # (row, col) floats


@dataclass
class ActionEffect:
    """What a single action DID to the grid."""
    action_id: int
    changed_cells: int
    moved_objects: list  # [{obj_id, color, displacement: (dr, dc)}]
    appeared: list  # [GridObject] — new objects
    disappeared: list  # [GridObject] — removed objects
    essence: str  # single-sentence description
    magnitude: float  # 0.0 = no change, 1.0 = total change


@dataclass
class PatternBlock:
    """Pre-compiled observation for meta-reasoning consumption."""
    step: int
    action_effect: Optional[ActionEffect]
    pattern_deltas: dict  # {pattern_name: delta} from COMPARE
    best_match: str  # dominant pattern in current grid
    best_match_score: float
    trend: dict  # {pattern_name: "rising"|"falling"|"stable"}
    surprise: float  # 0.0 = expected, 1.0 = totally unexpected
    salient_observation: str  # DISTILL output — most important thing


# ═══════════════════════════════════════════════════════════════
# Mini-Reasoning Engine
# ═══════════════════════════════════════════════════════════════

class MiniReasoningEngine:
    """
    Fast perceptual grounding layer.

    Runs every ARC step. Produces PatternBlocks for meta-reasoning.
    Maintains action-effect causal memory and pattern trend window.
    """

    def __init__(self, trend_window: int = 10):
        self._trend_window = trend_window
        self._prev_objects: list[GridObject] = []
        self._prev_grid: Optional[np.ndarray] = None

        # Action-Effect Causal Memory: {action_id: [ActionEffect, ...]}
        self._causal_memory: dict[int, list[ActionEffect]] = {}

        # Pattern Trend Window: list of {pattern_name: score} dicts
        self._pattern_history: list[dict] = []

        # Surprise tracking
        self._surprise_history: list[float] = []

        # Step counter
        self._step = 0

    def reset(self):
        """Reset for new episode."""
        prev_step = self._step
        self._prev_objects = []
        self._prev_grid = None
        self._causal_memory = {}
        self._pattern_history = []
        self._surprise_history = []
        self._step = 0
        if prev_step > 0:
            logger.info("[MiniReason] reset after %d steps", prev_step)

    # ─── DECOMPOSE ──────────────────────────────────────────────

    def decompose(self, grid: np.ndarray) -> list[GridObject]:
        """Break grid into connected components (objects) by color.

        Uses 4-connectivity flood fill. Background (color 0) is ignored.
        Returns list of GridObject with bbox, centroid, size.
        """
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        obj_id = 0

        for r in range(rows):
            for c in range(cols):
                if visited[r, c] or grid[r, c] == 0:
                    continue

                # Flood fill from (r, c)
                color = int(grid[r, c])
                cells = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr < 0 or cr >= rows or cc < 0 or cc >= cols
                            or visited[cr, cc] or grid[cr, cc] != color):
                        continue
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])

                if not cells:
                    continue

                rows_list = [p[0] for p in cells]
                cols_list = [p[1] for p in cells]
                bbox = (min(rows_list), min(cols_list),
                        max(rows_list), max(cols_list))
                centroid = (sum(rows_list) / len(cells),
                            sum(cols_list) / len(cells))

                objects.append(GridObject(
                    obj_id=obj_id, color=color, cells=cells,
                    size=len(cells), bbox=bbox, centroid=centroid,
                ))
                obj_id += 1

        return objects

    # ─── FILTER ─────────────────────────────────────────────────

    def filter_changes(self, prev_grid: np.ndarray, curr_grid: np.ndarray,
                       prev_objects: list[GridObject],
                       curr_objects: list[GridObject],
                       action_id: int = -1) -> ActionEffect:
        """Focus on what changed between two grid states.

        Computes cell-level diff and tracks object movement/appearance/disappearance.
        """
        diff_mask = prev_grid != curr_grid
        changed_cells = int(np.sum(diff_mask))
        total_cells = max(1, prev_grid.size)
        magnitude = changed_cells / total_cells

        # Track object movement by matching color + overlap
        moved = []
        appeared = []
        disappeared = list(prev_objects)  # start assuming all disappeared

        for curr_obj in curr_objects:
            best_match = None
            best_overlap = 0
            curr_set = set(curr_obj.cells)

            for prev_obj in prev_objects:
                if prev_obj.color != curr_obj.color:
                    continue
                overlap = len(curr_set & set(prev_obj.cells))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = prev_obj

            if best_match and best_overlap > 0:
                # Matched — compute displacement
                dr = curr_obj.centroid[0] - best_match.centroid[0]
                dc = curr_obj.centroid[1] - best_match.centroid[1]
                if abs(dr) > 0.5 or abs(dc) > 0.5:
                    moved.append({
                        "obj_id": best_match.obj_id,
                        "color": curr_obj.color,
                        "displacement": (round(dr, 1), round(dc, 1)),
                    })
                # Remove from disappeared list
                if best_match in disappeared:
                    disappeared.remove(best_match)
            else:
                appeared.append(curr_obj)

        # Build essence description
        essence = self._build_essence(changed_cells, moved, appeared, disappeared)

        return ActionEffect(
            action_id=action_id,
            changed_cells=changed_cells,
            moved_objects=moved,
            appeared=appeared,
            disappeared=disappeared,
            essence=essence,
            magnitude=magnitude,
        )

    def _build_essence(self, changed_cells: int, moved: list,
                       appeared: list, disappeared: list) -> str:
        """Reduce action effect to single salient description."""
        if changed_cells == 0:
            return "no_change"

        parts = []
        if moved:
            # Most salient: largest object that moved
            biggest = max(moved, key=lambda m: abs(m["displacement"][0]) + abs(m["displacement"][1]))
            dr, dc = biggest["displacement"]
            direction = ""
            if abs(dr) > abs(dc):
                direction = "down" if dr > 0 else "up"
            elif abs(dc) > 0:
                direction = "right" if dc > 0 else "left"
            if direction:
                parts.append(f"color{biggest['color']}_moved_{direction}")

        if appeared:
            parts.append(f"{len(appeared)}_appeared")
        if disappeared:
            parts.append(f"{len(disappeared)}_disappeared")
        if not parts:
            parts.append(f"{changed_cells}_cells_changed")

        return "|".join(parts)

    # ─── DISTILL ────────────────────────────────────────────────

    def distill(self, effect: ActionEffect, pattern_deltas: dict) -> str:
        """Reduce all observations to single most salient fact."""
        if effect.magnitude == 0:
            return "action_had_no_effect"

        # Find strongest pattern change
        if pattern_deltas:
            strongest = max(pattern_deltas.items(), key=lambda x: abs(x[1]))
            if abs(strongest[1]) > 0.05:
                direction = "increased" if strongest[1] > 0 else "decreased"
                return f"{strongest[0]}_{direction}|{effect.essence}"

        return effect.essence

    # ─── COMPARE ────────────────────────────────────────────────

    def compare(self, prev_profile: dict, curr_profile: dict) -> dict:
        """Compute pattern-level structural difference between two states."""
        deltas = {}
        for key in curr_profile:
            prev_val = prev_profile.get(key, 0.0)
            curr_val = curr_profile.get(key, 0.0)
            deltas[key] = round(curr_val - prev_val, 4)
        return deltas

    # ─── MATCH ──────────────────────────────────────────────────

    def match_best(self, profile: dict) -> tuple[str, float]:
        """Find dominant pattern in current grid from pattern profile."""
        if not profile:
            return "none", 0.0
        best_name = max(profile, key=lambda k: profile[k])
        return best_name, profile[best_name]

    # ─── SURPRISE SIGNAL ────────────────────────────────────────

    def compute_surprise(self, action_id: int, effect: ActionEffect,
                         pattern_deltas: dict) -> float:
        """Compute surprise: how unexpected was this action's effect?

        Uses causal memory to predict expected effect, then measures
        divergence from actual effect. High surprise = hypothesis opportunity.
        """
        if action_id < 0:
            return 0.0

        # Get expected magnitude from causal memory
        past_effects = self._causal_memory.get(action_id, [])
        if not past_effects:
            # First time seeing this action — moderate surprise
            return 0.5

        # Expected magnitude = running average
        expected_magnitude = sum(e.magnitude for e in past_effects) / len(past_effects)

        # Surprise = absolute deviation from expectation, normalized
        magnitude_surprise = abs(effect.magnitude - expected_magnitude)

        # Pattern surprise: did pattern deltas differ from typical?
        pattern_surprise = 0.0
        if pattern_deltas and len(past_effects) >= 3:
            # Compare current pattern deltas to historical average
            for key, delta in pattern_deltas.items():
                # Historical delta for this pattern under this action
                hist_deltas = []
                for pe in past_effects[-10:]:  # last 10 uses
                    # We don't store pattern deltas in causal memory yet,
                    # so compare effect direction instead
                    pass
                # Simple heuristic: large absolute delta = more surprising
                pattern_surprise += abs(delta)
            pattern_surprise = min(1.0, pattern_surprise / max(1, len(pattern_deltas)))

        # Combined surprise (magnitude deviation + pattern change size)
        surprise = min(1.0, magnitude_surprise * 0.6 + pattern_surprise * 0.4)

        # Effect direction surprise: did the action do something NEW?
        if effect.essence != "no_change":
            past_essences = {e.essence for e in past_effects[-5:]}
            if effect.essence not in past_essences:
                surprise = min(1.0, surprise + 0.2)  # novel effect boost

        return round(surprise, 4)

    # ─── ACTION-EFFECT CAUSAL MEMORY ────────────────────────────

    def record_effect(self, action_id: int, effect: ActionEffect):
        """Store action→effect mapping in causal memory."""
        if action_id not in self._causal_memory:
            self._causal_memory[action_id] = []
        self._causal_memory[action_id].append(effect)
        # Keep last 20 effects per action
        if len(self._causal_memory[action_id]) > 20:
            self._causal_memory[action_id] = self._causal_memory[action_id][-20:]

    def get_action_profile(self, action_id: int) -> dict:
        """Get summary of what an action typically does.

        Returns dict with dominant_effect, avg_magnitude, consistency.
        Designed for T2's RECALL personality.
        """
        effects = self._causal_memory.get(action_id, [])
        if not effects:
            return {"known": False}

        avg_mag = sum(e.magnitude for e in effects) / len(effects)
        essences = [e.essence for e in effects]
        # Most common essence = dominant effect
        from collections import Counter
        dominant = Counter(essences).most_common(1)[0]
        consistency = dominant[1] / len(essences)

        return {
            "known": True,
            "uses": len(effects),
            "dominant_effect": dominant[0],
            "consistency": round(consistency, 3),
            "avg_magnitude": round(avg_mag, 4),
        }

    # ─── PATTERN TREND WINDOW ───────────────────────────────────

    def update_trend(self, profile: dict):
        """Add pattern profile to trend window."""
        self._pattern_history.append(profile)
        if len(self._pattern_history) > self._trend_window:
            self._pattern_history = self._pattern_history[-self._trend_window:]

    def get_trends(self) -> dict:
        """Compute trend direction for each pattern over the window.

        Returns {pattern_name: "rising"|"falling"|"stable"}.
        Designed for T3's FORMULATE personality.
        """
        if len(self._pattern_history) < 3:
            return {}

        trends = {}
        # Compare first third vs last third of window
        n = len(self._pattern_history)
        first_third = self._pattern_history[:n // 3]
        last_third = self._pattern_history[-(n // 3):]

        all_keys = set()
        for p in self._pattern_history:
            all_keys.update(p.keys())

        for key in all_keys:
            early = sum(p.get(key, 0.0) for p in first_third) / max(1, len(first_third))
            late = sum(p.get(key, 0.0) for p in last_third) / max(1, len(last_third))
            delta = late - early
            if delta > 0.03:
                trends[key] = "rising"
            elif delta < -0.03:
                trends[key] = "falling"
            else:
                trends[key] = "stable"

        return trends

    # ─── MAIN PROCESS STEP ──────────────────────────────────────

    def process_step(self, grid: np.ndarray, action_id: int = -1,
                     pattern_profile: Optional[dict] = None) -> PatternBlock:
        """Run full mini-reasoning pipeline for one step.

        Called every ARC step. Returns PatternBlock for meta-reasoning.

        Args:
            grid: Current grid state (numpy array)
            action_id: Action that was just taken (-1 for first step)
            pattern_profile: Pattern scores from pattern_primitives module

        Returns:
            PatternBlock with all pre-compiled observations
        """
        self._step += 1
        curr_grid = np.asarray(grid, dtype=np.float64)
        profile = pattern_profile or {}

        # DECOMPOSE: break grid into objects
        curr_objects = self.decompose(curr_grid)

        # FILTER + COMPARE: what changed?
        effect = None
        pattern_deltas = {}
        surprise = 0.0

        if self._prev_grid is not None:
            # FILTER: compute action effect
            effect = self.filter_changes(
                self._prev_grid, curr_grid,
                self._prev_objects, curr_objects,
                action_id=action_id,
            )

            # COMPARE: pattern-level difference
            if self._pattern_history:
                prev_profile = self._pattern_history[-1]
                pattern_deltas = self.compare(prev_profile, profile)

            # SURPRISE: how unexpected was this?
            surprise = self.compute_surprise(action_id, effect, pattern_deltas)
            self._surprise_history.append(surprise)

            # Record in causal memory
            if action_id >= 0:
                self.record_effect(action_id, effect)

        # MATCH: dominant pattern
        best_match, best_score = self.match_best(profile)

        # Update trend window
        self.update_trend(profile)
        trends = self.get_trends()

        # DISTILL: most salient observation
        salient = ""
        if effect:
            salient = self.distill(effect, pattern_deltas)
        elif curr_objects:
            salient = f"initial_state|{len(curr_objects)}_objects"

        # Store state for next step
        self._prev_grid = curr_grid.copy()
        self._prev_objects = curr_objects

        # Periodic observability summary (every 100 steps)
        if self._step % 100 == 0:
            stats = self.get_stats()
            logger.info(
                "[MiniReason] step=%d actions_known=%d effects=%d avg_surprise=%.3f "
                "surprise_rate=%.1f%% trend_size=%d best_match=%s(%.2f)",
                self._step, stats["actions_known"], stats["total_effects_recorded"],
                stats["avg_surprise"], stats["surprise_rate"] * 100,
                stats["trend_window_size"], best_match, best_score,
            )

        return PatternBlock(
            step=self._step,
            action_effect=effect,
            pattern_deltas=pattern_deltas,
            best_match=best_match,
            best_match_score=best_score,
            trend=trends,
            surprise=surprise,
            salient_observation=salient,
        )

    # ─── STATS ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return summary statistics for logging/reporting."""
        total_effects = sum(len(v) for v in self._causal_memory.values())
        avg_surprise = (sum(self._surprise_history) / len(self._surprise_history)
                        if self._surprise_history else 0.0)
        high_surprise_count = sum(1 for s in self._surprise_history if s > 0.5)

        return {
            "steps": self._step,
            "actions_known": len(self._causal_memory),
            "total_effects_recorded": total_effects,
            "trend_window_size": len(self._pattern_history),
            "avg_surprise": round(avg_surprise, 4),
            "high_surprise_count": high_surprise_count,
            "surprise_rate": round(high_surprise_count / max(1, self._step), 4),
        }
