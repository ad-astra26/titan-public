"""
titan_plugin/logic/experience_memory.py — Experience Memory (ex_mem).

Stores memories of completed outer-world actions with full Inner+Outer
Trinity state snapshots and hormonal outcome scoring.

The hormonal delta IS the natural score — Titan doesn't need an external
reward signal. INSPIRATION went up? Good. VIGILANCE went up? Threatening.

Inner↔Outer Symmetry:
  INNER: hormonal urge → INTUITION queries ex_mem → confidence signal
  OUTER: action executes → records Trinity snapshots → hormonal delta
  NEXT:  same urge → INTUITION recalls → "conditions match past success"
"""
import json
import logging
import math
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


class ExperienceMemory:
    """Outer-world action experience store with Trinity state snapshots.

    Records completed actions with:
    - What was done (task_type)
    - Why (intent_hormones — inner state that drove the action)
    - Context (inner + outer Trinity snapshots before/after)
    - How it felt (hormonal_delta — natural scoring)
    - Whether it worked (outcome_score, success)
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size = -8000")    # 8MB cap (was unbounded on 78MB DB)
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("[ex_mem] Initialized at %s", db_path)

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experience_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                intent_hormones TEXT,
                inner_state_before TEXT,
                outer_state_before TEXT,
                outer_state_after TEXT,
                hormonal_delta TEXT,
                outcome_score REAL NOT NULL DEFAULT 0.0,
                success INTEGER NOT NULL DEFAULT 0,
                similar_count INTEGER NOT NULL DEFAULT 1,
                epoch_id INTEGER,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_exmem_task_type
            ON experience_memory(task_type)
        """)
        self._conn.commit()

    def record_experience(
        self,
        task_type: str,
        intent_hormones: dict = None,
        inner_before: list = None,
        outer_before: list = None,
        outer_after: list = None,
        hormonal_delta: dict = None,
        outcome_score: float = 0.0,
        success: bool = False,
        epoch_id: int = 0,
    ) -> int:
        """Record a completed action experience.

        Auto-increments similar_count for same task_type.
        """
        # Count existing experiences of this type
        existing = self.get_experience_count(task_type)

        cur = self._conn.execute("""
            INSERT INTO experience_memory
            (task_type, intent_hormones, inner_state_before, outer_state_before,
             outer_state_after, hormonal_delta, outcome_score, success,
             similar_count, epoch_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_type,
            json.dumps(intent_hormones) if intent_hormones else None,
            json.dumps(inner_before) if inner_before else None,
            json.dumps(outer_before) if outer_before else None,
            json.dumps(outer_after) if outer_after else None,
            json.dumps(hormonal_delta) if hormonal_delta else None,
            outcome_score,
            1 if success else 0,
            existing + 1,
            epoch_id,
            time.time(),
        ))
        self._conn.commit()
        row_id = cur.lastrowid

        logger.info(
            "[ex_mem] Recorded %s #%d (score=%.2f, success=%s, total=%d)",
            task_type, row_id, outcome_score, success, existing + 1)

        return row_id

    def recall_similar(self, task_type: str, current_inner: list = None,
                       top_k: int = 5) -> list:
        """Find past experiences of similar tasks in similar inner states.

        If current_inner provided, sorts by inner_state cosine similarity.
        Otherwise returns most recent experiences of this task_type.
        """
        rows = self._conn.execute(
            "SELECT * FROM experience_memory WHERE task_type = ? "
            "ORDER BY created_at DESC LIMIT 50",
            (task_type,)
        ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            d = dict(row)
            # Deserialize JSON fields
            for field in ("intent_hormones", "inner_state_before",
                          "outer_state_before", "outer_state_after", "hormonal_delta"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Compute similarity if inner state available
            if current_inner and d.get("inner_state_before") and isinstance(d["inner_state_before"], list):
                min_len = min(len(current_inner), len(d["inner_state_before"]))
                d["similarity"] = round(
                    _cosine_sim(current_inner[:min_len], d["inner_state_before"][:min_len]), 4)
            else:
                d["similarity"] = 0.0

            results.append(d)

        # Sort by similarity if inner state was provided
        if current_inner:
            results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def get_success_rate(self, task_type: str, last_n: int = 10) -> float:
        """Success rate for a task type. Feeds CONFIDENCE signal."""
        rows = self._conn.execute(
            "SELECT success FROM experience_memory WHERE task_type = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (task_type, last_n)
        ).fetchall()

        if not rows:
            return 0.0
        return sum(r["success"] for r in rows) / len(rows)

    def get_best_conditions(self, task_type: str) -> Optional[dict]:
        """Inner state that correlated with best outcomes.

        Returns the inner_state_before of the highest-scoring experience.
        Answers: 'When does this action work best?'
        """
        row = self._conn.execute(
            "SELECT inner_state_before, outcome_score FROM experience_memory "
            "WHERE task_type = ? AND inner_state_before IS NOT NULL "
            "ORDER BY outcome_score DESC LIMIT 1",
            (task_type,)
        ).fetchone()

        if not row or not row["inner_state_before"]:
            return None

        try:
            return {
                "inner_state": json.loads(row["inner_state_before"]),
                "score": row["outcome_score"],
            }
        except (json.JSONDecodeError, TypeError):
            return None

    def get_experience_count(self, task_type: str) -> int:
        """How many times has Titan done this type of action?"""
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_memory WHERE task_type = ?",
            (task_type,)
        ).fetchone()
        return row["c"] if row else 0

    def count(self) -> int:
        """Total stored experiences."""
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM experience_memory"
        ).fetchone()
        return row["c"] if row else 0

    def get_stats(self) -> dict:
        """Experience memory statistics for API."""
        total = self.count()
        # Get unique task types
        rows = self._conn.execute(
            "SELECT task_type, COUNT(*) as cnt, AVG(outcome_score) as avg_score, "
            "AVG(success) as success_rate FROM experience_memory "
            "GROUP BY task_type ORDER BY cnt DESC"
        ).fetchall()

        by_type = {}
        for r in rows:
            by_type[r["task_type"]] = {
                "count": r["cnt"],
                "avg_score": round(r["avg_score"] or 0, 3),
                "success_rate": round(r["success_rate"] or 0, 3),
            }

        return {
            "total": total,
            "by_type": by_type,
        }

    def close(self):
        if self._conn:
            self._conn.close()
