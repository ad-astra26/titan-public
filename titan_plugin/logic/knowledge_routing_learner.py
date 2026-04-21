"""
knowledge_routing_learner — smart per-backend routing based on observed data.

Closes rFP_knowledge_pipeline_v2.md §3.4 (Layer 4, "longer term"). Tracks,
per (query_type, backend), a rolling 7-day window of:

  * n_attempts — dispatch attempts that reached this backend
  * n_success — attempts that came back successful
  * avg_latency_ms — exponential moving average of latency
  * avg_quality — EMA of quality_score from knowledge_worker's quality gate
  * n_usage — downstream concept usage (from CGN_KNOWLEDGE_USAGE events)

Reputation = 0.7 * success_rate + 0.3 * avg_quality + 0.1 * tanh(usage/10)
(capped at 1.0). Backends with reputation < demote_threshold AND ≥ min_samples
are dropped from the chain entirely for that query type. Remaining backends
are reordered by reputation descending.

If stats are thin for a query type (< min_samples for every backend), the
learner returns None — dispatcher falls back to the static route() chain.
This preserves cold-start behaviour until we have enough signal.

Persistence: SQLite at data/knowledge_routing_stats.db, WAL journal, atomic
schema init. Stats flushed on every record_outcome/record_usage. Window
pruning happens lazily at read time (learned_chain) to avoid a background
thread.

See: rFP_knowledge_pipeline_v2.md §3.4 + KP-8.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

from titan_plugin.logic.knowledge_router import QueryType

logger = logging.getLogger(__name__)

# Event kinds fired via on_event callback (KP-8.1 observability)
EVENT_KIND_CHAIN_REORDERED = "chain_reordered"

# Callback signature: (kind: str, ctx: dict) -> None. Fires OUTSIDE the
# tracker lock so callbacks may publish bus events / hit network.
EventCallback = Callable[[str, dict], None]


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_SAMPLES = 10            # per (qt, backend) before reputation applies
DEFAULT_DEMOTE_THRESHOLD = 0.10     # drop from chain below this reputation
DEFAULT_SUCCESS_WEIGHT = 0.70
DEFAULT_QUALITY_WEIGHT = 0.30
DEFAULT_USAGE_BONUS = 0.10          # max boost from usage (tanh-scaled)
DEFAULT_EMA_ALPHA = 0.20            # EMA responsiveness for latency + quality


# ── Data shape ───────────────────────────────────────────────────────

@dataclass
class BackendStats:
    query_type: str
    backend: str
    n_attempts: int = 0
    n_success: int = 0
    n_usage: int = 0
    avg_latency_ms: float = 0.0
    avg_quality: float = 0.0
    last_update_ts: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.n_attempts <= 0:
            return 0.0
        return self.n_success / self.n_attempts

    def reputation(self,
                    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
                    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
                    usage_bonus: float = DEFAULT_USAGE_BONUS) -> float:
        """Combine success-rate + avg-quality + usage-tanh into [0, 1+]."""
        base = (success_weight * self.success_rate
                + quality_weight * self.avg_quality)
        boost = usage_bonus * math.tanh(self.n_usage / 10.0)
        return min(1.0, base + boost)


# ── Schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_stats (
    query_type     TEXT NOT NULL,
    backend        TEXT NOT NULL,
    n_attempts     INTEGER NOT NULL DEFAULT 0,
    n_success      INTEGER NOT NULL DEFAULT 0,
    n_usage        INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL    NOT NULL DEFAULT 0.0,
    avg_quality    REAL    NOT NULL DEFAULT 0.0,
    last_update_ts REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (query_type, backend)
);
CREATE INDEX IF NOT EXISTS idx_routing_last ON routing_stats(last_update_ts);
"""


# ── Learner ──────────────────────────────────────────────────────────

class RoutingLearner:
    """Persistent per-backend reputation tracker + chain reorderer.

    Thread-safe via a single lock; SQLite WAL handles reader/writer
    concurrency between knowledge_worker + WebSearchHelper processes.
    """

    def __init__(self,
                 db_path: str = "data/knowledge_routing_stats.db",
                 window_days: int = DEFAULT_WINDOW_DAYS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 demote_threshold: float = DEFAULT_DEMOTE_THRESHOLD,
                 success_weight: float = DEFAULT_SUCCESS_WEIGHT,
                 quality_weight: float = DEFAULT_QUALITY_WEIGHT,
                 usage_bonus: float = DEFAULT_USAGE_BONUS,
                 ema_alpha: float = DEFAULT_EMA_ALPHA,
                 enabled: bool = True,
                 on_event: Optional[EventCallback] = None):
        self.db_path = db_path
        self.window_seconds = int(window_days * 86400)
        self.min_samples = int(min_samples)
        self.demote_threshold = float(demote_threshold)
        self.success_weight = float(success_weight)
        self.quality_weight = float(quality_weight)
        self.usage_bonus = float(usage_bonus)
        self.ema_alpha = float(ema_alpha)
        self.enabled = bool(enabled)
        self._on_event = on_event
        self._lock = Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
            logger.info("[RoutingLearner] Schema OK (%s, window=%dd, "
                        "min_samples=%d)",
                        self.db_path, self.window_seconds // 86400,
                        self.min_samples)
        except Exception as e:
            logger.error("[RoutingLearner] Schema init failed: %s", e)
            raise

    # ── Recording ───────────────────────────────────────────────────

    def record_outcome(self,
                        query_type: QueryType,
                        backend: str,
                        success: bool,
                        latency_ms: float = 0.0,
                        quality: float = 0.0) -> None:
        """Update stats after a dispatch attempt reaches a backend.

        Called by dispatcher._run_chain() regardless of cache vs live fetch
        (cached hits still count — they reflect that this backend WAS
        reputable enough to ship a valid result last time). Quality is
        forwarded by knowledge_worker after its quality_gate; 0 for raw
        mode callers who don't run the gate.
        """
        if not self.enabled:
            return
        now = time.time()
        qt_val = query_type.value if isinstance(query_type, QueryType) else str(query_type)
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                row = conn.execute(
                    "SELECT n_attempts, n_success, n_usage, avg_latency_ms, "
                    "avg_quality FROM routing_stats "
                    "WHERE query_type = ? AND backend = ?",
                    (qt_val, backend)).fetchone()

                if row is None:
                    n_attempts = 1
                    n_success = 1 if success else 0
                    n_usage = 0
                    avg_lat = float(latency_ms) if latency_ms > 0 else 0.0
                    avg_q = float(quality)
                else:
                    n_attempts = int(row[0]) + 1
                    n_success = int(row[1]) + (1 if success else 0)
                    n_usage = int(row[2])
                    prev_lat = float(row[3])
                    prev_q = float(row[4])
                    avg_lat = (self._ema(prev_lat, latency_ms)
                                if latency_ms > 0 else prev_lat)
                    # Only update quality when caller supplied one (>0)
                    avg_q = (self._ema(prev_q, quality)
                             if quality > 0 else prev_q)

                conn.execute(
                    "INSERT OR REPLACE INTO routing_stats "
                    "(query_type, backend, n_attempts, n_success, n_usage, "
                    "avg_latency_ms, avg_quality, last_update_ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (qt_val, backend, n_attempts, n_success, n_usage,
                     avg_lat, avg_q, now))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(
                    "[RoutingLearner] record_outcome(%s,%s) error: %s",
                    qt_val, backend, e)

    def record_quality(self,
                        query_type: QueryType,
                        backend: str,
                        quality: float) -> None:
        """Update avg_quality EMA without touching attempt counters.

        Called by knowledge_worker after its quality gate so reputation
        reflects actual downstream-usable signal, not just HTTP success.
        Dispatcher's record_outcome() already moved n_attempts/n_success
        during the fetch; this is an orthogonal refinement of the row.
        """
        if not self.enabled or quality <= 0:
            return
        qt_val = query_type.value if isinstance(query_type, QueryType) else str(query_type)
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                row = conn.execute(
                    "SELECT avg_quality FROM routing_stats "
                    "WHERE query_type = ? AND backend = ?",
                    (qt_val, backend)).fetchone()
                if row is None:
                    # No prior observation — create with just the quality
                    conn.execute(
                        "INSERT INTO routing_stats "
                        "(query_type, backend, n_attempts, n_success, "
                        "n_usage, avg_latency_ms, avg_quality, "
                        "last_update_ts) VALUES (?, ?, 0, 0, 0, 0, ?, ?)",
                        (qt_val, backend, float(quality), time.time()))
                else:
                    avg_q = self._ema(float(row[0]), quality)
                    conn.execute(
                        "UPDATE routing_stats SET avg_quality = ?, "
                        "last_update_ts = ? "
                        "WHERE query_type = ? AND backend = ?",
                        (avg_q, time.time(), qt_val, backend))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(
                    "[RoutingLearner] record_quality(%s,%s) error: %s",
                    qt_val, backend, e)

    def record_usage(self, query_type: QueryType, backend: str) -> None:
        """Downstream consumer used a cached concept from this backend.

        Boosts reputation without changing success-rate semantics. Called
        by knowledge_worker on CGN_KNOWLEDGE_USAGE for backends it recorded
        during grounding.
        """
        if not self.enabled:
            return
        qt_val = query_type.value if isinstance(query_type, QueryType) else str(query_type)
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                cur = conn.execute(
                    "UPDATE routing_stats "
                    "SET n_usage = n_usage + 1, last_update_ts = ? "
                    "WHERE query_type = ? AND backend = ?",
                    (time.time(), qt_val, backend))
                if cur.rowcount == 0:
                    # First observation — create row with single usage
                    conn.execute(
                        "INSERT INTO routing_stats "
                        "(query_type, backend, n_attempts, n_success, "
                        "n_usage, last_update_ts) VALUES (?, ?, 0, 0, 1, ?)",
                        (qt_val, backend, time.time()))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(
                    "[RoutingLearner] record_usage(%s,%s) error: %s",
                    qt_val, backend, e)

    # ── EMA helper ──────────────────────────────────────────────────

    def _ema(self, prev: float, sample: float) -> float:
        if prev <= 0:
            return float(sample)
        return (1 - self.ema_alpha) * prev + self.ema_alpha * float(sample)

    # ── Window prune (lazy) ────────────────────────────────────────

    def _prune_stale(self, conn: sqlite3.Connection) -> int:
        """Drop rows whose last_update_ts is older than the window.

        Caller holds the lock. Returns rows deleted for telemetry.
        """
        cutoff = time.time() - self.window_seconds
        cur = conn.execute(
            "DELETE FROM routing_stats WHERE last_update_ts < ?",
            (cutoff,))
        return int(cur.rowcount or 0)

    # ── Chain reordering ────────────────────────────────────────────

    def learned_chain(self,
                       query_type: QueryType,
                       static_chain: List[str]) -> Optional[List[str]]:
        """Return a reordered+filtered backend chain, or None if cold.

        Cold start: if NO backend in static_chain has ≥ min_samples
        within the window, return None so dispatcher uses static_chain.

        Warm: for every backend in static_chain that DOES have ≥
        min_samples, compute reputation. Drop those below demote_threshold.
        Keep cold backends (those below min_samples) in their original
        position as fallback candidates. Sort the warm-above-threshold
        set by reputation descending, then append cold candidates in
        original order.
        """
        if not self.enabled or not static_chain:
            return None
        qt_val = query_type.value if isinstance(query_type, QueryType) else str(query_type)
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                self._prune_stale(conn)
                rows = conn.execute(
                    "SELECT backend, n_attempts, n_success, n_usage, "
                    "avg_latency_ms, avg_quality FROM routing_stats "
                    "WHERE query_type = ?", (qt_val,)).fetchall()
                conn.close()
            except Exception as e:
                logger.debug("[RoutingLearner] learned_chain error: %s", e)
                return None

        stats_by_backend: Dict[str, BackendStats] = {}
        for r in rows:
            s = BackendStats(
                query_type=qt_val, backend=r[0],
                n_attempts=int(r[1]), n_success=int(r[2]),
                n_usage=int(r[3]),
                avg_latency_ms=float(r[4]), avg_quality=float(r[5]),
            )
            stats_by_backend[r[0]] = s

        warm: List[Tuple[float, int, str]] = []   # (rep, idx, backend)
        cold: List[Tuple[int, str]] = []          # (idx, backend)
        any_warm = False
        for idx, bk in enumerate(static_chain):
            st = stats_by_backend.get(bk)
            if st is not None and st.n_attempts >= self.min_samples:
                any_warm = True
                rep = st.reputation(self.success_weight,
                                     self.quality_weight,
                                     self.usage_bonus)
                if rep >= self.demote_threshold:
                    warm.append((rep, idx, bk))
                else:
                    logger.info(
                        "[RoutingLearner] Demoting %s for qt=%s "
                        "(rep=%.3f < %.3f, %d attempts)",
                        bk, qt_val, rep, self.demote_threshold,
                        st.n_attempts)
            else:
                cold.append((idx, bk))

        if not any_warm:
            return None  # cold start — dispatcher keeps static chain

        warm_sorted = [bk for rep, idx, bk in
                       sorted(warm, key=lambda t: (-t[0], t[1]))]
        cold_sorted = [bk for idx, bk in sorted(cold, key=lambda t: t[0])]

        reordered = warm_sorted + cold_sorted
        # No change → return None so dispatcher's logging can say "static"
        if reordered == list(static_chain):
            return None
        logger.debug(
            "[RoutingLearner] Reordered chain for qt=%s: %s → %s",
            qt_val, static_chain, reordered)

        # KP-8.1 observability — fire chain_reordered event with before/after
        # chains + demoted backends list. Outside the lock above; no need
        # for extra isolation since SQLite txn already committed.
        if self._on_event is not None:
            demoted = [bk for bk in static_chain if bk not in reordered]
            try:
                self._on_event(EVENT_KIND_CHAIN_REORDERED, {
                    "query_type": qt_val,
                    "static_chain": list(static_chain),
                    "reordered_chain": list(reordered),
                    "demoted": demoted,
                    "n_warm": sum(
                        1 for s in stats_by_backend.values()
                        if s.n_attempts >= self.min_samples),
                })
            except Exception as e:
                logger.warning(
                    "[RoutingLearner] on_event callback failed: %s", e)

        return reordered

    # ── Stats for API + arch_map ────────────────────────────────────

    def snapshot(self) -> dict:
        """Current learner state for /v4/search-pipeline/learning."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                self._prune_stale(conn)
                rows = conn.execute(
                    "SELECT query_type, backend, n_attempts, n_success, "
                    "n_usage, avg_latency_ms, avg_quality, last_update_ts "
                    "FROM routing_stats ORDER BY query_type, backend"
                ).fetchall()
                conn.close()
            except Exception as e:
                logger.warning("[RoutingLearner] snapshot error: %s", e)
                rows = []
        out = {"ts": time.time(), "config": {
            "enabled": self.enabled,
            "window_days": self.window_seconds // 86400,
            "min_samples": self.min_samples,
            "demote_threshold": self.demote_threshold,
        }, "by_query_type": {}}
        for r in rows:
            qt = r[0]
            by_qt = out["by_query_type"].setdefault(qt, [])
            s = BackendStats(
                query_type=qt, backend=r[1],
                n_attempts=int(r[2]), n_success=int(r[3]),
                n_usage=int(r[4]),
                avg_latency_ms=float(r[5]), avg_quality=float(r[6]),
                last_update_ts=float(r[7]),
            )
            by_qt.append({
                "backend": s.backend,
                "n_attempts": s.n_attempts,
                "n_success": s.n_success,
                "n_usage": s.n_usage,
                "success_rate": round(s.success_rate, 4),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "avg_quality": round(s.avg_quality, 3),
                "reputation": round(s.reputation(
                    self.success_weight, self.quality_weight,
                    self.usage_bonus), 4),
                "last_update_ts": s.last_update_ts,
                "warm": s.n_attempts >= self.min_samples,
            })
        return out


__all__ = [
    "BackendStats",
    "RoutingLearner",
    "DEFAULT_WINDOW_DAYS",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_DEMOTE_THRESHOLD",
    "EVENT_KIND_CHAIN_REORDERED",
    "EventCallback",
]
