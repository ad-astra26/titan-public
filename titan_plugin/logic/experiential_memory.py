"""
titan_plugin/logic/experiential_memory.py — Experiential Memory (e_mem).

Stores dream-distilled insights for waking recall.
Dreams are significance-weighted distillations of impactful experiences.
What varies most gets remembered. What was never recalled gets forgotten.
Bookmarked dreams (core memories) persist forever.

Retention is self-governed by Titan's developmental age (π-cluster count),
not human clock time. Young Titan forgets faster; mature Titan remembers longer.

CRITICAL: This is the bridge between DREAMING and WAKING cognition.
Without it, Titan sleeps but doesn't learn from sleep.
"""
import json
import logging
import math
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors.

    rFP #3 Phase 1 defensive guard: isinstance check before len() prevents
    TypeError when scalar/None values leak in (e.g. legacy DB rows written
    with a scalar 'felt_tensor' before the rFP #3 store-side fix).
    """
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


class ExperientialMemory:
    """Dream-distilled insight store with bookmarking and self-governed retention.

    Three-tier retention:
      Bookmarked — forever (core memories)
      Recalled   — 2× retention window (proved useful during waking)
      Ordinary   — 1× retention window (never recalled, low significance)

    Retention window grows logarithmically with developmental age.
    """

    # Auto-bookmark thresholds
    AUTO_BOOKMARK_SIGNIFICANCE = 0.8
    IDENTITY_HORMONES = ("INSPIRATION", "CREATIVITY", "REFLECTION")
    IDENTITY_FIRE_THRESHOLD = 0.5
    IDENTITY_FIRE_MIN_COUNT = 2

    # Retention base (in π-clusters ≈ 7 human days at ~21 min/cluster)
    BASE_RETENTION_CLUSTERS = 480
    MAX_RETENTION_MULTIPLIER = 5.0

    def __init__(self, db_path: str, developmental_age_fn=None):
        """
        Args:
            db_path: SQLite database file path.
            developmental_age_fn: Callable returning current π-cluster count.
                If None, retention window uses BASE_RETENTION_CLUSTERS.
        """
        self._db_path = db_path
        self._dev_age_fn = developmental_age_fn
        # 2026-04-09 fix: connection is shared across threads (check_same_thread=False),
        # but Python's sqlite3 module does NOT serialize concurrent access on the same
        # connection — that's the caller's job. Without this lock, the spirit_worker
        # query-drain thread (calling get_stats) races dream-consolidation writes
        # (store_insight / prune_stale), producing sqlite3.InterfaceError: SQLITE_MISUSE
        # which kills get_coordinator and makes /health appear unreachable. RLock so
        # nested locked methods (get_stats → count → ...) don't deadlock.
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size = -4000")    # 4MB cap (small DB)
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # rFP #3 Phase 4: config-ified recall similarity floor
        # (overridden via set_min_recall_similarity at construction site)
        self._min_recall_similarity: float = 0.1
        logger.info("[e_mem] Initialized at %s", db_path)

    def set_min_recall_similarity(self, value: float) -> None:
        """Override default recall similarity floor.

        Called by the construction site with the titan_params.toml
        [dreaming].min_recall_similarity value. Kept as a setter to avoid
        coupling e_mem init to the config loader.
        """
        self._min_recall_similarity = float(value)

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS experiential_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    significance REAL NOT NULL,
                    felt_tensor TEXT NOT NULL,
                    epoch_id INTEGER NOT NULL,
                    dream_cycle INTEGER NOT NULL,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled REAL,
                    bookmarked INTEGER NOT NULL DEFAULT 0,
                    bookmark_reason TEXT,
                    bookmark_felt_state TEXT,
                    bookmarked_at REAL,
                    created_at REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emem_dream_cycle
                ON experiential_memory(dream_cycle)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emem_bookmarked
                ON experiential_memory(bookmarked)
            """)
            self._conn.commit()

    def store_insight(self, insight: dict, dream_cycle: int) -> int:
        """Store a distilled dream insight with auto-bookmark check.

        Args:
            insight: Dict with 'significance', 'felt_tensor' (or 'tensor_mean'),
                     optionally 'epoch_id', 'hormones', 'ts'.
            dream_cycle: Which dream cycle produced this insight.

        Returns:
            Inserted row ID.
        """
        significance = insight.get("significance", 0.0)
        # rFP #3 Phase 1: drop the tensor_mean fallback. tensor_mean is a SCALAR
        # summary, not a vector — prior fallback caused the [D6] Epoch recall error
        # spam. Default to empty list if felt_tensor absent; validate type and dim.
        felt_tensor = insight.get("felt_tensor", [])
        if not isinstance(felt_tensor, list):
            logger.warning(
                "[e_mem] store_insight: non-list felt_tensor rejected "
                "(type=%s, value=%.100s) — would have corrupted the column",
                type(felt_tensor).__name__, repr(felt_tensor))
            felt_tensor = []
        if felt_tensor and len(felt_tensor) not in (65, 130):
            logger.warning(
                "[e_mem] store_insight: unexpected felt_tensor dim=%d "
                "(expect 65 or 130 per architectural contract)",
                len(felt_tensor))
        epoch_id = insight.get("epoch_id", 0)

        # Auto-bookmark check
        bookmarked = 0
        bookmark_reason = None

        if significance > self.AUTO_BOOKMARK_SIGNIFICANCE:
            bookmarked = 1
            bookmark_reason = "auto:high_significance"

        hormones = insight.get("hormones", {})
        identity_count = sum(
            1 for h in self.IDENTITY_HORMONES
            if hormones.get(h, 0) > self.IDENTITY_FIRE_THRESHOLD
        )
        if identity_count >= self.IDENTITY_FIRE_MIN_COUNT:
            bookmarked = 1
            bookmark_reason = "auto:identity_resonance"

        with self._lock:
            cur = self._conn.execute("""
                INSERT INTO experiential_memory
                (significance, felt_tensor, epoch_id, dream_cycle,
                 bookmarked, bookmark_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                significance,
                json.dumps(felt_tensor) if isinstance(felt_tensor, list) else str(felt_tensor),
                epoch_id, dream_cycle,
                bookmarked, bookmark_reason, time.time(),
            ))
            self._conn.commit()
            row_id = cur.lastrowid

        if bookmarked:
            logger.info(
                "[e_mem] Bookmarked dream #%d (sig=%.3f, reason=%s)",
                row_id, significance, bookmark_reason)

        return row_id

    def recall_by_state(self, current_state: list[float], top_k: int = 3) -> list[dict]:
        """Find insights most similar to current 130D state.

        Increments recall_count for returned insights.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiential_memory ORDER BY id DESC LIMIT 200"
            ).fetchall()

            if not rows:
                return []

            scored = []
            for row in rows:
                try:
                    tensor = json.loads(row["felt_tensor"])
                except (json.JSONDecodeError, TypeError):
                    continue
                # rFP #3 Phase 1: guard against legacy scalar-corrupted rows
                # (298/300 rows pre-rFP #3 stored scalar floats as felt_tensor due
                # to the tensor_mean fallback bug). Skip silently — row remains in
                # DB with valid significance/ts/dream_cycle but invisible to recall.
                if not isinstance(tensor, list):
                    continue
                sim = _cosine_sim(current_state, tensor)
                scored.append((sim, dict(row)))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            now = time.time()

            for sim, row_dict in scored[:top_k]:
                if sim < self._min_recall_similarity:
                    break
                row_dict["similarity"] = round(sim, 4)
                row_dict["felt_tensor"] = json.loads(row_dict["felt_tensor"])
                results.append(row_dict)
                # Increment recall count
                self._conn.execute(
                    "UPDATE experiential_memory SET recall_count = recall_count + 1, "
                    "last_recalled = ? WHERE id = ?",
                    (now, row_dict["id"]))

            if results:
                self._conn.commit()

            return results

    def recall_by_recency(self, limit: int = 5) -> list[dict]:
        """Most recent distilled insights."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiential_memory ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            try:
                d["felt_tensor"] = json.loads(d["felt_tensor"])
            except (json.JSONDecodeError, TypeError):
                d["felt_tensor"] = []
            results.append(d)
        return results

    def bookmark_insight(self, insight_id: int, reason_tensor: list[float] = None):
        """Titan intentionally marks a dream as permanently important.

        reason_tensor: felt-state at the moment of bookmarking — captures
        WHY this dream mattered (felt-reason, not text).

        rFP #3 Phase 3 dimensional note: `bookmark_felt_state` column now
        holds MIXED dims:
          - Legacy rows (pre rFP #3): 65D reason_tensor
          - New rows (post rFP #3):  130D reason_tensor
        Downstream consumers reading this column MUST check length before
        use — don't assume a single size.
        """
        with self._lock:
            self._conn.execute("""
                UPDATE experiential_memory
                SET bookmarked = 1, bookmark_reason = 'intentional',
                    bookmark_felt_state = ?, bookmarked_at = ?
                WHERE id = ?
            """, (
                json.dumps(reason_tensor) if reason_tensor else None,
                time.time(),
                insight_id,
            ))
            self._conn.commit()
        logger.info("[e_mem] Intentionally bookmarked dream #%d", insight_id)

    def get_recall_ratio(self) -> float:
        """Fraction of stored insights recalled at least once.

        Feeds Outer CHIT-25 (dream_recall → material action integration).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN recall_count > 0 THEN 1 ELSE 0 END) as recalled "
                "FROM experiential_memory"
            ).fetchone()
        total = row["total"] or 0
        if total == 0:
            return 0.0
        return (row["recalled"] or 0) / total

    def get_dream_quality(self, last_n_cycles: int = 5) -> float:
        """Average significance of insights from recent dream cycles.

        Feeds Inner CHIT-25 (dream_awareness).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT AVG(significance) as avg_sig "
                "FROM experiential_memory "
                "WHERE dream_cycle >= (SELECT MAX(dream_cycle) FROM experiential_memory) - ?",
                (last_n_cycles,)
            ).fetchone()
        return row["avg_sig"] if row and row["avg_sig"] is not None else 0.0

    def compute_retention_window(self) -> int:
        """Retention window in π-clusters. Grows with maturity.

        dev_age 50:   ~480 clusters (~7 human days)
        dev_age 200:  ~720 clusters (~10 days)
        dev_age 1000: ~1200 clusters (~17 days)
        dev_age 5000: ~1680 clusters (~24 days)
        """
        dev_age = self._dev_age_fn() if self._dev_age_fn else 100
        maturity = max(1, dev_age)
        growth = 1.0 + math.log(max(1, maturity / 50.0))
        return int(self.BASE_RETENTION_CLUSTERS * min(growth, self.MAX_RETENTION_MULTIPLIER))

    def prune_stale(self):
        """Three-tier retention pruning.

        Bookmarked = forever, Recalled = 2× window, Ordinary = 1× window.
        """
        window = self.compute_retention_window()
        with self._lock:
            # Use max dream_cycle as proxy for current developmental position
            row = self._conn.execute(
                "SELECT MAX(dream_cycle) as max_cycle FROM experiential_memory"
            ).fetchone()
            current_cycle = row["max_cycle"] if row and row["max_cycle"] is not None else 0

            cutoff_ordinary = current_cycle - window
            cutoff_recalled = current_cycle - (window * 2)

            # Prune ordinary (never recalled, not bookmarked)
            c1 = self._conn.execute(
                "DELETE FROM experiential_memory "
                "WHERE dream_cycle < ? AND bookmarked = 0 AND recall_count = 0",
                (cutoff_ordinary,)
            )
            # Prune old recalled (but not bookmarked)
            c2 = self._conn.execute(
                "DELETE FROM experiential_memory "
                "WHERE dream_cycle < ? AND bookmarked = 0 AND recall_count > 0",
                (cutoff_recalled,)
            )
            self._conn.commit()

            pruned = (c1.rowcount or 0) + (c2.rowcount or 0)
        if pruned > 0:
            logger.info(
                "[e_mem] Pruned %d stale dreams (window=%d, ordinary_cutoff=%d, recalled_cutoff=%d)",
                pruned, window, cutoff_ordinary, cutoff_recalled)
        return pruned

    def count(self) -> int:
        """Total stored insights."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM experiential_memory"
            ).fetchone()
        return row["c"] if row else 0

    def count_bookmarked(self) -> int:
        """Total bookmarked (core) memories."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM experiential_memory WHERE bookmarked = 1"
            ).fetchone()
        return row["c"] if row else 0

    def get_stats(self) -> dict:
        """Complete e_mem statistics for API.

        Defensive: any single stat that fails returns a sane default rather
        than killing the whole stats payload (which would propagate up to
        get_coordinator and make /health appear unreachable).
        """
        def _safe(fn, default):
            try:
                return fn()
            except Exception as exc:
                logger.warning("[e_mem] get_stats: %s failed: %s", fn.__name__, exc)
                return default

        return {
            "total": _safe(self.count, 0),
            "bookmarked": _safe(self.count_bookmarked, 0),
            "recall_ratio": round(_safe(self.get_recall_ratio, 0.0), 3),
            "dream_quality": round(_safe(self.get_dream_quality, 0.0), 3),
            "retention_window": _safe(self.compute_retention_window, self.BASE_RETENTION_CLUSTERS),
        }

    def close(self):
        """Close database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
