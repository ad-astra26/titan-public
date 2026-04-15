"""
titan_plugin/logic/coding_explorer.py — Self-Directed Coding Explorer.

Orchestrates Titan's self-directed cognitive development through code:
  1. INTROSPECT detects a cognitive gap
  2. Self-exploration triggers coding exercise
  3. CGN "coding" consumer selects action (decompose/abstract/implement/test/refactor/compose)
  4. Sandbox executes code with strict isolation
  5. Execution result → objective reward signal → CGN
  6. Cross-pollinate insights with ARC "reasoning" consumer

The LLM narrator generates actual code for now. Titan decides WHAT to code
(via CGN action selection). The narrator decides HOW to write it.

See: titan-docs/rFP_coding_consumer_self_directed_development.md
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("titan.coding_explorer")


# ── Exercise Templates ────────────────────────────────────────────────
# Built-in exercises that Titan can attempt, organized by difficulty.
# Each template maps a cognitive concept to a coding task.

EXERCISE_TEMPLATES = {
    # Pattern detection exercises (cross-pollinate with ARC)
    "symmetry_detection": {
        "concept": "symmetry",
        "difficulty": 0.3,
        "description": "Detect if a 2D list has horizontal or vertical symmetry",
        "code_template": (
            "def check_symmetry(grid):\n"
            "    \"\"\"Return 'horizontal', 'vertical', 'both', or 'none'.\"\"\"\n"
            "    rows = len(grid)\n"
            "    cols = len(grid[0]) if rows > 0 else 0\n"
            "    h_sym = all(grid[i] == grid[rows-1-i] for i in range(rows//2))\n"
            "    v_sym = all(grid[r][c] == grid[r][cols-1-c]\n"
            "               for r in range(rows) for c in range(cols//2))\n"
            "    if h_sym and v_sym: return 'both'\n"
            "    if h_sym: return 'horizontal'\n"
            "    if v_sym: return 'vertical'\n"
            "    return 'none'\n\n"
            "# Tests\n"
            "assert check_symmetry([[1,2,1],[3,4,3],[1,2,1]]) == 'both'\n"
            "assert check_symmetry([[1,2,3],[1,2,3]]) == 'horizontal'\n"
            "assert check_symmetry([[1,2,1],[3,4,3]]) == 'vertical'\n"
            "print('All symmetry tests passed')\n"
        ),
        "test_assertions": 3,
        "associations": ["pattern_matching", "grid_transform"],
    },
    "sequence_prediction": {
        "concept": "sequence_reasoning",
        "difficulty": 0.4,
        "description": "Predict the next element in a numerical sequence",
        "code_template": (
            "def predict_next(seq):\n"
            "    \"\"\"Predict the next number in a sequence using differences.\"\"\"\n"
            "    if len(seq) < 2:\n"
            "        return seq[-1] if seq else 0\n"
            "    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]\n"
            "    if len(set(diffs)) == 1:\n"
            "        return seq[-1] + diffs[0]\n"
            "    # Second-order differences\n"
            "    diffs2 = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]\n"
            "    if len(set(diffs2)) == 1:\n"
            "        next_diff = diffs[-1] + diffs2[0]\n"
            "        return seq[-1] + next_diff\n"
            "    return seq[-1] + diffs[-1]\n\n"
            "# Tests\n"
            "assert predict_next([2, 4, 6, 8]) == 10\n"
            "assert predict_next([1, 4, 9, 16]) == 25\n"
            "assert predict_next([1, 3, 7, 13]) == 21  # +2,+4,+6,+8\n"
            "print('All sequence tests passed')\n"
        ),
        "test_assertions": 3,
        "associations": ["pattern_matching", "abstraction"],
    },
    "grid_rotation": {
        "concept": "spatial_transform",
        "difficulty": 0.3,
        "description": "Rotate a 2D grid 90 degrees clockwise",
        "code_template": (
            "def rotate_90(grid):\n"
            "    \"\"\"Rotate grid 90 degrees clockwise.\"\"\"\n"
            "    rows = len(grid)\n"
            "    cols = len(grid[0]) if rows else 0\n"
            "    return [[grid[rows-1-r][c] for r in range(rows)]\n"
            "            for c in range(cols)]\n\n"
            "# Tests\n"
            "assert rotate_90([[1,2],[3,4]]) == [[3,1],[4,2]]\n"
            "assert rotate_90([[1,2,3]]) == [[1],[2],[3]]\n"
            "print('All rotation tests passed')\n"
        ),
        "test_assertions": 2,
        "associations": ["rotation", "grid_transform", "spatial_reasoning"],
    },
    "decomposition": {
        "concept": "decomposition",
        "difficulty": 0.5,
        "description": "Break a complex problem into simpler sub-problems",
        "code_template": (
            "def decompose_grid(grid):\n"
            "    \"\"\"Break grid into quadrants and analyze each.\"\"\"\n"
            "    rows = len(grid)\n"
            "    cols = len(grid[0]) if rows else 0\n"
            "    mid_r, mid_c = rows // 2, cols // 2\n"
            "    quadrants = {\n"
            "        'top_left': [row[:mid_c] for row in grid[:mid_r]],\n"
            "        'top_right': [row[mid_c:] for row in grid[:mid_r]],\n"
            "        'bot_left': [row[:mid_c] for row in grid[mid_r:]],\n"
            "        'bot_right': [row[mid_c:] for row in grid[mid_r:]],\n"
            "    }\n"
            "    stats = {}\n"
            "    for name, q in quadrants.items():\n"
            "        flat = [v for row in q for v in row]\n"
            "        stats[name] = {\n"
            "            'sum': sum(flat),\n"
            "            'unique': len(set(flat)),\n"
            "            'size': len(flat),\n"
            "        }\n"
            "    return stats\n\n"
            "# Tests\n"
            "result = decompose_grid([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])\n"
            "assert result['top_left']['sum'] == 14  # 1+2+5+6\n"
            "assert result['bot_right']['unique'] == 4\n"
            "print('All decomposition tests passed')\n"
        ),
        "test_assertions": 2,
        "associations": ["pattern_matching", "abstraction", "analysis"],
    },
    "abstraction": {
        "concept": "abstraction",
        "difficulty": 0.6,
        "description": "Extract a reusable transformation rule from examples",
        "code_template": (
            "def learn_rule(examples):\n"
            "    \"\"\"Given (input, output) pairs, learn the transform rule.\"\"\"\n"
            "    # Detect if it's a constant offset\n"
            "    offsets = [o - i for i, o in examples]\n"
            "    if len(set(offsets)) == 1:\n"
            "        offset = offsets[0]\n"
            "        return lambda x: x + offset\n"
            "    # Detect if it's a multiplier\n"
            "    ratios = [o / i if i != 0 else None for i, o in examples]\n"
            "    ratios_clean = [r for r in ratios if r is not None]\n"
            "    if ratios_clean and len(set(ratios_clean)) == 1:\n"
            "        factor = ratios_clean[0]\n"
            "        return lambda x: x * factor\n"
            "    # Fallback: linear regression (y = ax + b)\n"
            "    n = len(examples)\n"
            "    sx = sum(i for i, o in examples)\n"
            "    sy = sum(o for i, o in examples)\n"
            "    sxy = sum(i * o for i, o in examples)\n"
            "    sxx = sum(i * i for i, o in examples)\n"
            "    a = (n * sxy - sx * sy) / max(1, n * sxx - sx * sx)\n"
            "    b = (sy - a * sx) / max(1, n)\n"
            "    return lambda x: a * x + b\n\n"
            "# Tests\n"
            "rule1 = learn_rule([(1, 3), (2, 4), (3, 5)])\n"
            "assert rule1(10) == 12  # +2 offset\n"
            "rule2 = learn_rule([(2, 6), (3, 9), (4, 12)])\n"
            "assert rule2(5) == 15  # *3 multiplier\n"
            "print('All abstraction tests passed')\n"
        ),
        "test_assertions": 2,
        "associations": ["pattern_matching", "generalization", "induction"],
    },
    "composition": {
        "concept": "composition",
        "difficulty": 0.6,
        "description": "Compose multiple transforms into a pipeline",
        "code_template": (
            "def compose(*transforms):\n"
            "    \"\"\"Compose multiple transform functions into one.\"\"\"\n"
            "    def composed(x):\n"
            "        result = x\n"
            "        for fn in transforms:\n"
            "            result = fn(result)\n"
            "        return result\n"
            "    return composed\n\n"
            "def apply_to_grid(grid, fn):\n"
            "    \"\"\"Apply a scalar function to every element.\"\"\"\n"
            "    return [[fn(cell) for cell in row] for row in grid]\n\n"
            "# Tests\n"
            "double = lambda x: x * 2\n"
            "add_one = lambda x: x + 1\n"
            "pipeline = compose(double, add_one)\n"
            "assert pipeline(3) == 7  # (3*2)+1\n"
            "grid = [[1, 2], [3, 4]]\n"
            "result = apply_to_grid(grid, pipeline)\n"
            "assert result == [[3, 5], [7, 9]]\n"
            "print('All composition tests passed')\n"
        ),
        "test_assertions": 2,
        "associations": ["function_composition", "pipeline", "abstraction"],
    },
    "statistical_analysis": {
        "concept": "data_analysis",
        "difficulty": 0.4,
        "description": "Compute basic statistics and detect outliers",
        "code_template": (
            "import statistics\n\n"
            "def analyze(data):\n"
            "    \"\"\"Return stats and outliers (>2 std from mean).\"\"\"\n"
            "    if len(data) < 2:\n"
            "        return {'mean': data[0] if data else 0, 'outliers': []}\n"
            "    mean = statistics.mean(data)\n"
            "    stdev = statistics.stdev(data)\n"
            "    outliers = [x for x in data if abs(x - mean) > 2 * stdev]\n"
            "    return {\n"
            "        'mean': round(mean, 2),\n"
            "        'median': statistics.median(data),\n"
            "        'stdev': round(stdev, 2),\n"
            "        'outliers': outliers,\n"
            "    }\n\n"
            "# Tests\n"
            "result = analyze([1, 2, 3, 4, 5, 100])\n"
            "assert 100 in result['outliers']\n"
            "assert result['median'] == 3.5\n"
            "print('All analysis tests passed')\n"
        ),
        "test_assertions": 2,
        "associations": ["statistics", "outlier_detection", "data_understanding"],
    },
    "recursive_structure": {
        "concept": "recursion",
        "difficulty": 0.7,
        "description": "Build and traverse a recursive tree structure",
        "code_template": (
            "def build_tree(values):\n"
            "    \"\"\"Build a balanced binary tree from sorted values.\"\"\"\n"
            "    if not values:\n"
            "        return None\n"
            "    mid = len(values) // 2\n"
            "    return {\n"
            "        'val': values[mid],\n"
            "        'left': build_tree(values[:mid]),\n"
            "        'right': build_tree(values[mid+1:]),\n"
            "    }\n\n"
            "def tree_depth(node):\n"
            "    if node is None:\n"
            "        return 0\n"
            "    return 1 + max(tree_depth(node['left']), tree_depth(node['right']))\n\n"
            "def inorder(node):\n"
            "    if node is None:\n"
            "        return []\n"
            "    return inorder(node['left']) + [node['val']] + inorder(node['right'])\n\n"
            "# Tests\n"
            "tree = build_tree([1, 2, 3, 4, 5, 6, 7])\n"
            "assert tree['val'] == 4\n"
            "assert tree_depth(tree) == 3\n"
            "assert inorder(tree) == [1, 2, 3, 4, 5, 6, 7]\n"
            "print('All recursion tests passed')\n"
        ),
        "test_assertions": 3,
        "associations": ["recursion", "tree_structure", "divide_conquer"],
    },
}


# ── Reward Table ──────────────────────────────────────────────────────
# Maps sandbox execution outcomes to reward signals.

REWARD_TABLE = {
    "all_tests_pass":   0.10,   # Code compiles, all assertions pass
    "partial_tests":    0.05,   # Code compiles, some assertions pass
    "compiles_no_test": 0.02,   # Code compiles, no assertions
    "syntax_error":    -0.02,   # AST validation fails
    "runtime_error":   -0.01,   # Execution raises exception
    "timeout":         -0.03,   # Code exceeds time limit
    "planning_only":    0.03,   # Decompose/abstract without execution
}


# ── Action → Exercise Mapping ────────────────────────────────────────
# Maps CGN action names to suitable exercise concepts.

ACTION_EXERCISE_MAP = {
    "decompose": ["decomposition", "grid_rotation"],
    "abstract":  ["abstraction", "sequence_prediction"],
    "implement": ["symmetry_detection", "statistical_analysis",
                  "recursive_structure"],
    "test":      ["symmetry_detection", "sequence_prediction",
                  "decomposition"],
    "refactor":  ["composition", "abstraction"],
    "compose":   ["composition", "recursive_structure"],
}


# ── Data Classes ──────────────────────────────────────────────────────

@dataclass
class CodingExerciseResult:
    """Result of a single coding exercise."""
    concept: str = ""
    action: str = ""
    action_index: int = 0
    difficulty: float = 0.0
    sandbox_success: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    reward: float = 0.0
    execution_time_ms: float = 0.0
    error: str = ""
    output: str = ""
    epoch: int = 0
    timestamp: float = 0.0


# ── Coding Explorer ──────────────────────────────────────────────────

class CodingExplorer:
    """Orchestrates self-directed coding exercises.

    Flow:
      1. Receives trigger from self-exploration (coherence gap or INTROSPECT)
      2. CGN "coding" consumer selects action (ground())
      3. Picks exercise template matching the action
      4. Sandbox executes the code
      5. Computes reward from execution result
      6. Records outcome to CGN + DB
      7. Optionally cross-pollinates with ARC reasoning consumer

    The sandbox provides an objective, unfakeable reward signal —
    code either works or doesn't. No LLM self-rating needed.
    """

    def __init__(self, send_queue=None, config: dict = None,
                 db_path: str = "data/inner_memory.db"):
        self._send_queue = send_queue
        self._config = config or {}
        self._db_path = db_path

        # Config
        self._cooldown_epochs = self._config.get("cooldown_epochs", 50)
        self._max_per_dream = self._config.get("max_exercises_per_dream", 3)
        self._research_before = self._config.get("research_before_code", True)
        self._arc_cross = self._config.get("arc_cross_pollinate", True)

        # CGN consumer client (loaded lazily — needs /dev/shm from CGN worker)
        self._cgn_client = None
        self._cgn_client_init_attempted = False

        # Sandbox helper
        from titan_plugin.logic.agency.helpers.coding_sandbox import (
            CodingSandboxHelper)
        self._sandbox = CodingSandboxHelper()

        # State
        self._cooldown = 0
        self._exercises_this_dream = 0
        self._total_exercises = 0
        self._total_successes = 0
        self._concept_attempts: Dict[str, int] = {}
        self._concept_successes: Dict[str, int] = {}

        # Initialize DB table
        self._init_db()

        logger.info("[CodingExplorer] Initialized (cooldown=%d, max_dream=%d, "
                    "sandbox=%s)",
                    self._cooldown_epochs, self._max_per_dream,
                    self._sandbox.status())

    def _init_db(self):
        """Create coding_exercises table if not exists."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coding_exercises (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    epoch INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    concept TEXT NOT NULL,
                    action TEXT NOT NULL,
                    action_index INTEGER DEFAULT 0,
                    difficulty REAL DEFAULT 0.0,
                    sandbox_success INTEGER DEFAULT 0,
                    tests_passed INTEGER DEFAULT 0,
                    tests_total INTEGER DEFAULT 0,
                    reward REAL DEFAULT 0.0,
                    execution_time_ms REAL DEFAULT 0.0,
                    error TEXT DEFAULT '',
                    output TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ce_epoch
                    ON coding_exercises(epoch)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ce_concept
                    ON coding_exercises(concept)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("[CodingExplorer] DB init failed: %s", e)

    def _ensure_cgn_client(self):
        """Lazily initialize CGN consumer client."""
        if self._cgn_client is not None:
            return True
        if self._cgn_client_init_attempted:
            return False
        self._cgn_client_init_attempted = True
        try:
            from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient
            self._cgn_client = CGNConsumerClient(
                "coding", self._send_queue, "spirit")
            logger.info("[CodingExplorer] CGN 'coding' client initialized")
            return True
        except Exception as e:
            logger.warning("[CodingExplorer] CGN client init failed: %s", e)
            return False

    # ── Public API ───────────────────────────────────────────────────

    @property
    def can_explore(self) -> bool:
        """Whether coding exploration is currently allowed."""
        return self._cooldown <= 0

    def tick_cooldown(self):
        """Decrement cooldown. Called each epoch."""
        if self._cooldown > 0:
            self._cooldown -= 1

    def on_dream_start(self):
        """Reset per-dream counter."""
        self._exercises_this_dream = 0

    def explore(self, trigger: dict, epoch: int,
                neuromods: dict, context: dict = None) -> Optional[CodingExerciseResult]:
        """Execute a coding exploration exercise.

        Args:
            trigger: from get_exploration_triggers() or SELF_EXPLORE_TRIGGER msg
            epoch: current consciousness epoch
            neuromods: current neuromodulator levels
            context: optional extra context (msl, reasoning, etc.)

        Returns:
            CodingExerciseResult or None if exercise skipped.
        """
        context = context or {}

        if not self.can_explore:
            logger.debug("[CodingExplorer] Cooldown active (%d epochs left)",
                         self._cooldown)
            return None

        if self._exercises_this_dream >= self._max_per_dream:
            logger.debug("[CodingExplorer] Max exercises per dream reached (%d)",
                         self._max_per_dream)
            return None

        # ── 1. Select action via CGN ────────────────────────────────
        action_name, action_index = self._select_action(
            trigger, epoch, neuromods, context)

        # ── 2. Pick exercise template ───────────────────────────────
        template = self._pick_exercise(action_name, trigger)
        if not template:
            logger.debug("[CodingExplorer] No template for action=%s", action_name)
            return None

        concept = template.get("concept", action_name)
        difficulty = template.get("difficulty", 0.5)

        # ── 3. Execute in sandbox ───────────────────────────────────
        code = template.get("code_template", "")
        if not code:
            # Planning-only action (decompose/abstract)
            result = CodingExerciseResult(
                concept=concept,
                action=action_name,
                action_index=action_index,
                difficulty=difficulty,
                sandbox_success=True,
                reward=REWARD_TABLE["planning_only"],
                epoch=epoch,
                timestamp=time.time(),
                output="Planning step (no execution)",
            )
            self._record_result(result)
            return result

        t0 = time.time()
        # Sandbox execute is async but we call sync wrapper
        sandbox_result = self._sandbox._run_code(code)
        exec_ms = (time.time() - t0) * 1000

        # ── 4. Compute reward ───────────────────────────────────────
        reward, tests_passed, tests_total = self._compute_reward(
            sandbox_result, template)

        result = CodingExerciseResult(
            concept=concept,
            action=action_name,
            action_index=action_index,
            difficulty=difficulty,
            sandbox_success=sandbox_result.get("success", False),
            tests_passed=tests_passed,
            tests_total=tests_total,
            reward=reward,
            execution_time_ms=round(exec_ms, 1),
            error=sandbox_result.get("error", "") or "",
            output=(sandbox_result.get("result", "") or "")[:500],
            epoch=epoch,
            timestamp=time.time(),
        )

        # ── 5. Record to CGN + DB ──────────────────────────────────
        self._record_result(result)

        # ── 6. Cross-pollinate with ARC ────────────────────────────
        if self._arc_cross and reward > 0 and self._send_queue:
            self._cross_pollinate_arc(result)

        # ── 7. Update state ────────────────────────────────────────
        self._cooldown = self._cooldown_epochs
        self._exercises_this_dream += 1
        self._total_exercises += 1
        if result.sandbox_success and result.tests_passed > 0:
            self._total_successes += 1

        # Track per-concept stats
        self._concept_attempts[concept] = (
            self._concept_attempts.get(concept, 0) + 1)
        if result.sandbox_success and result.tests_passed > 0:
            self._concept_successes[concept] = (
                self._concept_successes.get(concept, 0) + 1)

        logger.info("[CodingExplorer] Exercise: %s/%s — sandbox=%s tests=%d/%d "
                    "reward=%.3f time=%.0fms (total=%d, success_rate=%.0f%%)",
                    action_name, concept,
                    "PASS" if result.sandbox_success else "FAIL",
                    tests_passed, tests_total,
                    reward, exec_ms,
                    self._total_exercises,
                    (self._total_successes / max(1, self._total_exercises)) * 100)

        return result

    # ── Action Selection ─────────────────────────────────────────────

    def _select_action(self, trigger: dict, epoch: int,
                       neuromods: dict, context: dict
                       ) -> tuple:
        """Use CGN consumer client to select a coding action.

        Falls back to heuristic selection if CGN not available.
        Returns (action_name, action_index).
        """
        action_names = ["decompose", "abstract", "implement",
                        "test", "refactor", "compose"]

        if self._ensure_cgn_client():
            try:
                # Build 30D state vector from current cognitive state
                state_vec = self._build_state_vector(
                    trigger, epoch, neuromods, context)

                # Local inference via CGN consumer client
                result = self._cgn_client.infer_action(state_vec)
                if result is not None:
                    action_idx = result.action_index
                    action_idx = min(action_idx, len(action_names) - 1)
                    return action_names[action_idx], action_idx
            except Exception as e:
                logger.debug("[CodingExplorer] CGN action select failed: %s", e)

        # Fallback: map trigger to action heuristically
        trigger_action = trigger.get("action", "")
        gap_metric = trigger.get("gap_metric", "")

        if trigger_action == "seek_novelty" or gap_metric == "DA":
            return "implement", 2
        elif trigger_action == "consolidate" or gap_metric == "i_confidence":
            return "refactor", 4
        elif trigger_action == "introspect" or gap_metric == "dominant_primitive":
            return "decompose", 0
        elif trigger_action == "rest":
            return "abstract", 1
        elif trigger_action == "adjust_attention":
            return "test", 3
        else:
            return "implement", 2  # Default: try building something

    def _build_state_vector(self, trigger: dict, epoch: int,
                            neuromods: dict, context: dict) -> np.ndarray:
        """Build 30D state vector for CGN inference.

        Encodes: neuromod state (6D) + trigger info (4D) + cognitive context
        (10D) + exercise history (5D) + padding (5D).
        """
        vec = np.zeros(30, dtype=np.float32)

        # Neuromods (6D): DA, 5HT, NE, ACh, GABA, Endorphin
        nm = neuromods or {}
        vec[0] = self._nm_float(nm, "DA")
        vec[1] = self._nm_float(nm, "5HT")
        vec[2] = self._nm_float(nm, "NE")
        vec[3] = self._nm_float(nm, "ACh")
        vec[4] = self._nm_float(nm, "GABA")
        vec[5] = self._nm_float(nm, "Endorphin")

        # Trigger info (4D)
        urgency = trigger.get("urgency", 0.5)
        vec[6] = urgency
        # Encode gap type
        gap_metric = trigger.get("gap_metric", "")
        gap_map = {"DA": 0.1, "5HT": 0.2, "NE": 0.3, "ACh": 0.4,
                   "GABA": 0.5, "i_confidence": 0.6, "vocab_total": 0.7,
                   "total_chains": 0.8, "dominant_primitive": 0.9}
        vec[7] = gap_map.get(gap_metric, 0.5)
        vec[8] = min(1.0, self._total_exercises / 100.0)
        vec[9] = min(1.0, self._exercises_this_dream / max(1, self._max_per_dream))

        # Cognitive context (10D)
        ctx = context or {}
        vec[10] = ctx.get("i_confidence", 0.5)
        vec[11] = ctx.get("chi_coherence", 0.5)
        vec[12] = min(1.0, ctx.get("total_chains", 0) / 1000.0)
        vec[13] = ctx.get("commit_rate", 0.5)
        vec[14] = min(1.0, ctx.get("vocab_total", 0) / 300.0)
        vec[15] = ctx.get("prediction_accuracy", 0.5)
        vec[16] = min(1.0, epoch / 500000.0)  # Maturity
        vec[17] = ctx.get("fatigue", 0.0)
        vec[18] = ctx.get("chi_total", 0.5)
        vec[19] = 1.0 if ctx.get("is_dreaming", False) else 0.0

        # Exercise history (5D)
        success_rate = (self._total_successes /
                        max(1, self._total_exercises))
        vec[20] = success_rate
        vec[21] = min(1.0, len(self._concept_attempts) / 8.0)
        vec[22] = min(1.0, self._cooldown / max(1, self._cooldown_epochs))
        vec[23] = 1.0 if self._sandbox.status() == "available" else 0.0
        vec[24] = min(1.0, self._exercises_this_dream / 3.0)

        return vec

    # ── Exercise Selection ───────────────────────────────────────────

    def _pick_exercise(self, action_name: str,
                       trigger: dict) -> Optional[dict]:
        """Pick an exercise template matching the CGN action.

        Prefers exercises with lower attempt count (novelty-seeking).
        """
        candidates = ACTION_EXERCISE_MAP.get(action_name, [])
        if not candidates:
            # Fallback: any template
            candidates = list(EXERCISE_TEMPLATES.keys())

        if not candidates:
            return None

        # Score candidates: prefer least-attempted concepts
        scored = []
        for name in candidates:
            template = EXERCISE_TEMPLATES.get(name)
            if not template:
                continue
            attempts = self._concept_attempts.get(
                template["concept"], 0)
            # Lower attempts = higher score (novelty bonus)
            novelty_score = 1.0 / (1 + attempts)
            scored.append((novelty_score, name, template))

        if not scored:
            return None

        # Sort by novelty (highest first), pick top
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][2]

    # ── Reward Computation ───────────────────────────────────────────

    def _compute_reward(self, sandbox_result: dict,
                        template: dict) -> tuple:
        """Compute reward from sandbox execution result.

        Returns (reward, tests_passed, tests_total).
        """
        tests_total = template.get("test_assertions", 0)

        if not sandbox_result.get("success", False):
            error = sandbox_result.get("error", "")
            if "Timeout" in error:
                return REWARD_TABLE["timeout"], 0, tests_total
            elif "Syntax" in error or "Validation" in error:
                return REWARD_TABLE["syntax_error"], 0, tests_total
            else:
                return REWARD_TABLE["runtime_error"], 0, tests_total

        # Success — check how many tests passed
        output = sandbox_result.get("result", "")

        if "passed" in output.lower():
            # All tests passed (our templates print "All X tests passed")
            return REWARD_TABLE["all_tests_pass"], tests_total, tests_total

        if tests_total > 0:
            # Code ran but no "passed" output — probably partial
            return REWARD_TABLE["partial_tests"], 1, tests_total

        return REWARD_TABLE["compiles_no_test"], 0, 0

    # ── Recording & Persistence ──────────────────────────────────────

    def _record_result(self, result: CodingExerciseResult):
        """Persist exercise result to DB and send CGN transition."""
        # DB
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "INSERT INTO coding_exercises "
                "(epoch, timestamp, concept, action, action_index, "
                "difficulty, sandbox_success, tests_passed, tests_total, "
                "reward, execution_time_ms, error, output) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (result.epoch, result.timestamp, result.concept,
                 result.action, result.action_index, result.difficulty,
                 1 if result.sandbox_success else 0,
                 result.tests_passed, result.tests_total,
                 result.reward, result.execution_time_ms,
                 result.error[:500], result.output[:500]))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("[CodingExplorer] DB insert failed: %s", e)

        # CGN transition (outcome)
        if self._send_queue:
            try:
                self._send_queue.put_nowait({
                    "type": "CGN_TRANSITION",
                    "src": "spirit",
                    "dst": "cgn",
                    "ts": time.time(),
                    "payload": {
                        "type": "outcome",
                        "consumer": "coding",
                        "concept_id": result.concept,
                        "reward": result.reward,
                        "outcome_context": {
                            "action": result.action,
                            "sandbox_success": result.sandbox_success,
                            "tests_passed": result.tests_passed,
                            "tests_total": result.tests_total,
                            "difficulty": result.difficulty,
                            "execution_time_ms": result.execution_time_ms,
                        },
                    },
                })
            except Exception as e:
                logger.debug("[CodingExplorer] CGN transition send failed: %s", e)

        # ── META-CGN producers #5 + #6: coding.problem_solved / coding.test_failed ──
        # v3 Phase D rollout (rFP_meta_cgn_v3 § 12 rows 5+6). Paired emission at the
        # single canonical exercise-completion point. Fires ONLY when sandbox actually
        # ran tests (tests_total > 0) — planning-only actions (code_template empty)
        # and sandbox infrastructure failures produce neither signal.
        #
        # LATENT during META-CGN Phase D rollout: CodingExplorer.explore() is not
        # triggered under current Titan states (self_reasoning gap conditions rarely
        # fire on saturated-stable Titans). See rFP_coding_explorer_activation.md
        # for activation plan deferred post-META-CGN. Producer code is correct and
        # ready; empirical validation arrives when coding runs.
        #
        # Partial success (tests_passed > 0 AND tests_passed < tests_total) fires BOTH
        # producers — semantically correct: the chain synthesized something that works
        # AND failed some tests, both deserve downstream reinforcement signals.
        if self._send_queue and result.sandbox_success and result.tests_total > 0:
            try:
                from titan_plugin.bus import emit_meta_cgn_signal
                # Producer #5 — problem_solved: any test passed
                if result.tests_passed > 0:
                    _p5_ratio = float(result.tests_passed) / float(result.tests_total)
                    _p5_sent = emit_meta_cgn_signal(
                        self._send_queue,
                        src="coding",
                        consumer="coding",
                        event_type="problem_solved",
                        intensity=min(1.0, _p5_ratio),
                        domain=str(result.concept)[:40],
                        reason=f"coding exercise {result.action}/{result.concept} "
                               f"tests_passed={result.tests_passed}/{result.tests_total} "
                               f"reward={result.reward:.3f}",
                    )
                    if _p5_sent:
                        logger.info(
                            "[META-CGN] coding.problem_solved EMIT — concept=%s action=%s "
                            "tests=%d/%d reward=%.3f intensity=%.2f",
                            result.concept, result.action, result.tests_passed,
                            result.tests_total, result.reward, _p5_ratio)
                    else:
                        logger.warning(
                            "[META-CGN] Producer #5 coding.problem_solved DROPPED by bus "
                            "— concept=%s tests=%d/%d (rate-gate or queue-full)",
                            result.concept, result.tests_passed, result.tests_total)
                # Producer #6 — test_failed: any test failed
                if result.tests_passed < result.tests_total:
                    _p6_failed = result.tests_total - result.tests_passed
                    _p6_ratio = float(_p6_failed) / float(result.tests_total)
                    _p6_sent = emit_meta_cgn_signal(
                        self._send_queue,
                        src="coding",
                        consumer="coding",
                        event_type="test_failed",
                        intensity=min(1.0, _p6_ratio),
                        domain=str(result.concept)[:40],
                        reason=f"coding exercise {result.action}/{result.concept} "
                               f"tests_failed={_p6_failed}/{result.tests_total}",
                    )
                    if _p6_sent:
                        logger.info(
                            "[META-CGN] coding.test_failed EMIT — concept=%s action=%s "
                            "tests_failed=%d/%d intensity=%.2f",
                            result.concept, result.action, _p6_failed,
                            result.tests_total, _p6_ratio)
                    else:
                        logger.warning(
                            "[META-CGN] Producer #6 coding.test_failed DROPPED by bus "
                            "— concept=%s tests_failed=%d/%d (rate-gate or queue-full)",
                            result.concept, _p6_failed, result.tests_total)
            except Exception as _cp_emit_err:
                logger.warning(
                    "[META-CGN] Producer #5/#6 coding emit FAILED — concept=%s "
                    "err=%s (signal missed)", result.concept, _cp_emit_err)

        # Language bridge: send pattern discoveries for vocabulary building
        if self._send_queue and result.reward > 0 and result.tests_passed > 0:
            template = EXERCISE_TEMPLATES.get(result.concept, {})
            associations = template.get("associations", [])
            _lang_concepts = {"symmetry", "spatial_transform", "decomposition",
                              "pattern_matching", "composition", "abstraction",
                              "sequence_reasoning", "grid_transform"}
            _matched = [a for a in associations if a in _lang_concepts]
            if _matched:
                try:
                    self._send_queue.put_nowait({
                        "type": "CGN_KNOWLEDGE_REQ",
                        "src": "spirit",
                        "dst": "cgn",
                        "ts": time.time(),
                        "payload": {
                            "topic": _matched[0],
                            "requestor": "coding_explorer",
                            "urgency": min(0.8, result.reward * 1.5),
                            "context": {
                                "source": "coding_pattern_discovery",
                                "concept": result.concept,
                                "action": result.action,
                                "tests_passed": result.tests_passed,
                            },
                        },
                    })
                    logger.debug("[CodingExplorer] Language bridge: %s → %s",
                                 result.concept, _matched[0])
                except Exception:
                    pass

    # ── ARC Cross-Pollination ────────────────────────────────────────

    def _cross_pollinate_arc(self, result: CodingExerciseResult):
        """Send surprise event to ARC reasoning consumer on coding success.

        Shared concepts (symmetry, rotation, decomposition) bridge the
        ARC visual domain and the coding logical domain.
        """
        # Only cross-pollinate concepts that exist in both domains
        arc_relevant = {"symmetry", "spatial_transform", "decomposition",
                        "pattern_matching", "composition", "abstraction"}

        template = EXERCISE_TEMPLATES.get(result.concept, {})
        associations = set(template.get("associations", []))
        shared = associations & arc_relevant

        if not shared:
            return

        try:
            self._send_queue.put_nowait({
                "type": "CGN_SURPRISE",
                "src": "spirit",
                "dst": "cgn",
                "ts": time.time(),
                "payload": {
                    "consumer": "reasoning",
                    "concept_id": result.concept,
                    "magnitude": abs(result.reward) * 2.0,
                    "context": {
                        "source": "coding_cross_pollination",
                        "coding_action": result.action,
                        "coding_reward": result.reward,
                        "shared_concepts": list(shared),
                    },
                },
            })
            logger.debug("[CodingExplorer] ARC cross-pollination: %s → %s "
                         "(magnitude=%.3f)",
                         result.concept, list(shared),
                         abs(result.reward) * 2.0)
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _nm_float(nm: dict, key: str, default: float = 0.5) -> float:
        """Extract float from neuromods dict."""
        v = nm.get(key)
        if v is None:
            # Try alternate keys
            alt_map = {"5HT": ["5-HT", "Serotonin"], "Endorphin": ["endorphin"]}
            for alt in alt_map.get(key, []):
                v = nm.get(alt)
                if v is not None:
                    break
        if v is None:
            return default
        if isinstance(v, dict):
            return float(v.get("level", default))
        return float(v)

    def get_stats(self) -> dict:
        """Return coding explorer statistics for monitoring."""
        success_rate = (self._total_successes /
                        max(1, self._total_exercises))
        return {
            "total_exercises": self._total_exercises,
            "total_successes": self._total_successes,
            "success_rate": round(success_rate, 4),
            "cooldown": self._cooldown,
            "exercises_this_dream": self._exercises_this_dream,
            "concepts_attempted": len(self._concept_attempts),
            "concept_stats": {
                c: {
                    "attempts": self._concept_attempts.get(c, 0),
                    "successes": self._concept_successes.get(c, 0),
                }
                for c in self._concept_attempts
            },
            "sandbox_available": self._sandbox.status() == "available",
            "cgn_client_ready": self._cgn_client is not None,
        }
