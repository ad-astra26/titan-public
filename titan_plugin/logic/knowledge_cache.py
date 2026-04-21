"""
knowledge_cache — persistent SQLite-backed cache for knowledge_router.

Replaces Stage A's in-memory 500-entry FIFO cache with a durable
per-query-type-TTL store that survives knowledge_worker restarts.
Cache key = sha256(normalized_query + query_type + backend) per
knowledge_router.query_hash().

Per-type TTLs (rFP_knowledge_pipeline_v2.md §3.2):
    dictionary / dictionary_phrase : 30 days   (definitions stable)
    wikipedia_like                  : 7 days    (encyclopedic drift)
    conceptual                      : 24 hours  (strategy queries rotate)
    technical                       : 12 hours  (docs update)
    news                            : 1 hour    (fresh by nature)

Failure caching (Q-KP3 decisions, confirmed 2026-04-20):
    empty / http_4xx                : 1 hour    (content isn't appearing)
    parse_error                     : 1 hour    (transient)
    rate_limit                      : 5 min     (backend recovery)
    http_5xx                        : 10 min    (backend recovery)
    network / timeout               : 0         (do NOT cache, retry)

Eviction (Q-KP2 decision, confirmed 2026-04-20):
    Primary:  TTL-based, opportunistic during get() + explicit evict_expired()
    Secondary: LRU-within-TTL via ts_last_hit when row count > size_cap
               (default 10 000 entries)

See rFP_knowledge_pipeline_v2.md §3.2.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from titan_plugin.logic.knowledge_router import QueryType

logger = logging.getLogger(__name__)


# ── TTL configuration ────────────────────────────────────────────────

TTL_SUCCESS: Dict[QueryType, int] = {
    QueryType.DICTIONARY: 30 * 86400,
    QueryType.DICTIONARY_PHRASE: 30 * 86400,
    QueryType.WIKIPEDIA_LIKE: 7 * 86400,
    QueryType.CONCEPTUAL: 24 * 3600,
    QueryType.TECHNICAL: 12 * 3600,
    QueryType.NEWS: 1 * 3600,
    # Stable verdict — caching it avoids re-classifying the same internal
    # name many times per day on log-noisy Titans.
    QueryType.INTERNAL_REJECTED: 24 * 3600,
}

TTL_FAILURE: Dict[str, int] = {
    "empty":       3600,   # content isn't going to appear suddenly
    "http_4xx":    3600,   # client-side issue, won't fix itself soon
    "parse_error": 3600,   # backend schema issue
    "rate_limit":   300,   # 5 min
    "http_5xx":     600,   # 10 min
    "network":        0,   # DO NOT CACHE — retry next call
    "timeout":        0,   # DO NOT CACHE
}


def resolve_ttl(qt: QueryType, success: bool, error_type: str = "") -> int:
    """Pick the correct TTL for a cache write. 0 = do not cache."""
    if success:
        return TTL_SUCCESS.get(qt, 3600)
    return TTL_FAILURE.get(error_type, 0)


# ── Data shape ───────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query_hash: str
    query_text: str
    query_type: str
    backend: str
    result_json: str
    success: bool
    error_type: str = ""
    quality_score: float = 0.0
    bytes_consumed: int = 0
    ts_cached: float = 0.0
    ts_last_hit: float = 0.0
    hit_count: int = 0
    ttl_seconds: int = 0

    @property
    def is_expired(self) -> bool:
        """True if this entry has exceeded its TTL (ttl_seconds==0 never expires)."""
        if self.ttl_seconds <= 0:
            # ttl_seconds == 0 stored as "do not cache" — shouldn't be in DB
            # but if it is, treat as immediately expired.
            return True
        return (time.time() - self.ts_cached) >= self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.ts_cached


# ── Schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash     TEXT PRIMARY KEY,
    query_text     TEXT NOT NULL,
    query_type     TEXT NOT NULL,
    backend        TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    success        INTEGER NOT NULL DEFAULT 0,
    error_type     TEXT    NOT NULL DEFAULT '',
    quality_score  REAL    NOT NULL DEFAULT 0.0,
    bytes_consumed INTEGER NOT NULL DEFAULT 0,
    ts_cached      REAL    NOT NULL,
    ts_last_hit    REAL    NOT NULL DEFAULT 0,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    ttl_seconds    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_ts_cached   ON search_cache(ts_cached);
CREATE INDEX IF NOT EXISTS idx_cache_ts_last_hit ON search_cache(ts_last_hit);
CREATE INDEX IF NOT EXISTS idx_cache_type        ON search_cache(query_type);
CREATE INDEX IF NOT EXISTS idx_cache_backend     ON search_cache(backend);
"""


# ── KnowledgeCache ───────────────────────────────────────────────────

class KnowledgeCache:
    """Persistent SQLite-backed cache.

    Thread-safety model: short-lived connection per operation, WAL journal
    for concurrent reader/writer safety. knowledge_worker is a single
    subprocess so contention is minimal; WAL still protects against
    long-running read queries blocking writers.
    """

    def __init__(self, db_path: str, size_cap: int = 10_000):
        self.db_path = db_path
        self.size_cap = int(size_cap)
        # In-memory rolling counters (UTC-day-scoped)
        self._hits_today = 0
        self._misses_today = 0
        self._bytes_saved_today = 0
        self._counter_day_epoch = self._current_day_epoch()
        self._init_schema()

    # ── Schema + counter bookkeeping ────────────────────────────────

    def _init_schema(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
            logger.info("[KnowledgeCache] Schema OK (%s)", self.db_path)
        except Exception as e:
            logger.error("[KnowledgeCache] Schema init failed: %s", e)
            raise

    @staticmethod
    def _current_day_epoch() -> int:
        """UTC day epoch (integer days since 1970-01-01)."""
        return int(time.time() // 86400)

    def _maybe_reset_counters(self) -> None:
        today = self._current_day_epoch()
        if today != self._counter_day_epoch:
            self._counter_day_epoch = today
            self._hits_today = 0
            self._misses_today = 0
            self._bytes_saved_today = 0

    # ── Get ─────────────────────────────────────────────────────────

    def get(self, query_hash: str) -> Optional[CacheEntry]:
        """Look up a cache entry by hash.

        On hit:  updates ts_last_hit + increments hit_count, returns entry
                  UNLESS expired, in which case we delete and return None.
        On miss: returns None.

        Counters (hits/misses_today) update accordingly.
        """
        self._maybe_reset_counters()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM search_cache WHERE query_hash = ?",
                (query_hash,)
            ).fetchone()

            if row is None:
                conn.close()
                self._misses_today += 1
                return None

            entry = CacheEntry(
                query_hash=row["query_hash"],
                query_text=row["query_text"],
                query_type=row["query_type"],
                backend=row["backend"],
                result_json=row["result_json"],
                success=bool(row["success"]),
                error_type=row["error_type"] or "",
                quality_score=float(row["quality_score"]),
                bytes_consumed=int(row["bytes_consumed"]),
                ts_cached=float(row["ts_cached"]),
                ts_last_hit=float(row["ts_last_hit"] or 0),
                hit_count=int(row["hit_count"]),
                ttl_seconds=int(row["ttl_seconds"]),
            )

            if entry.is_expired:
                # Opportunistic eviction on read path
                conn.execute("DELETE FROM search_cache WHERE query_hash = ?",
                             (query_hash,))
                conn.commit()
                conn.close()
                self._misses_today += 1
                return None

            # Hit — update stats + persist hit telemetry
            now = time.time()
            conn.execute(
                "UPDATE search_cache SET ts_last_hit = ?, "
                "hit_count = hit_count + 1 WHERE query_hash = ?",
                (now, query_hash))
            conn.commit()
            conn.close()

            entry.ts_last_hit = now
            entry.hit_count += 1
            self._hits_today += 1
            self._bytes_saved_today += entry.bytes_consumed
            return entry
        except Exception as e:
            logger.warning("[KnowledgeCache] get(%s) error: %s",
                           query_hash[:16], e)
            return None

    # ── Put ─────────────────────────────────────────────────────────

    def put(self,
            query_hash: str,
            query_text: str,
            query_type: QueryType,
            backend: str,
            result_payload: Dict[str, Any],
            success: bool,
            error_type: str = "",
            quality_score: float = 0.0,
            bytes_consumed: int = 0) -> bool:
        """Insert or replace a cache entry.

        Returns False without persisting if resolve_ttl says "don't cache"
        (ttl == 0 — timeout/network errors that should retry on next call).

        Returns True on successful upsert.
        """
        ttl = resolve_ttl(query_type, success, error_type)
        if ttl <= 0:
            return False

        now = time.time()
        try:
            payload_json = json.dumps(result_payload, default=str)
        except Exception as e:
            logger.warning("[KnowledgeCache] put() JSON encode failed "
                           "for %s: %s", query_hash[:16], e)
            return False

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO search_cache "
                "(query_hash, query_text, query_type, backend, result_json, "
                "success, error_type, quality_score, bytes_consumed, "
                "ts_cached, ts_last_hit, hit_count, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)",
                (query_hash, query_text, query_type.value, backend,
                 payload_json,
                 1 if success else 0,
                 error_type, float(quality_score),
                 int(bytes_consumed), now, ttl))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.warning("[KnowledgeCache] put(%s) error: %s",
                           query_hash[:16], e)
            return False

    # ── Eviction ────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Delete all rows whose TTL has passed. Returns count evicted."""
        try:
            now = time.time()
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cur = conn.execute(
                "DELETE FROM search_cache "
                "WHERE (ts_cached + ttl_seconds) < ?",
                (now,))
            evicted = cur.rowcount
            conn.commit()
            conn.close()
            if evicted:
                logger.info("[KnowledgeCache] evict_expired: %d rows", evicted)
            return evicted
        except Exception as e:
            logger.warning("[KnowledgeCache] evict_expired error: %s", e)
            return 0

    def evict_lru(self, max_entries: Optional[int] = None) -> int:
        """If row count exceeds max_entries, drop oldest by ts_last_hit.

        LRU semantic: ts_last_hit is updated on every get() hit, so an
        entry that has been heavily used stays fresh even if old. A freshly
        inserted entry has ts_last_hit=0 (never hit), which sorts to the
        BOTTOM of the eviction list (likely to go first) — that's intended:
        speculative inserts shouldn't squat if they're never used.

        Actually we want the opposite priority: prefer evicting LEAST-
        recently-used. For never-hit entries (ts_last_hit=0), fall back to
        ts_cached so a freshly-inserted-never-hit entry has ordering based
        on its insert time, not 0.
        """
        cap = int(max_entries if max_entries is not None else self.size_cap)
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            row = conn.execute(
                "SELECT COUNT(*) FROM search_cache").fetchone()
            count = int(row[0] or 0)
            if count <= cap:
                conn.close()
                return 0
            overflow = count - cap
            # Evict by max(ts_last_hit, ts_cached) ASC — catches both
            # never-hit-but-old AND old-and-stale-hit entries. Batch in
            # groups to avoid a pathological single-tx delete.
            batch = min(overflow, 1000)
            cur = conn.execute(
                "DELETE FROM search_cache WHERE query_hash IN ("
                "  SELECT query_hash FROM search_cache "
                "  ORDER BY MAX(ts_last_hit, ts_cached) ASC "
                "  LIMIT ?"
                ")",
                (batch,))
            evicted = cur.rowcount
            conn.commit()
            conn.close()
            if evicted:
                logger.info("[KnowledgeCache] evict_lru: %d rows "
                            "(count=%d → %d, cap=%d)",
                            evicted, count, count - evicted, cap)
            return evicted
        except Exception as e:
            logger.warning("[KnowledgeCache] evict_lru error: %s", e)
            return 0

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Health snapshot for /v4/search-pipeline/health (KP-5)."""
        self._maybe_reset_counters()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            total = int(conn.execute(
                "SELECT COUNT(*) FROM search_cache").fetchone()[0] or 0)
            by_type_rows = conn.execute(
                "SELECT query_type, COUNT(*) FROM search_cache "
                "GROUP BY query_type"
            ).fetchall()
            by_backend_rows = conn.execute(
                "SELECT backend, COUNT(*) FROM search_cache "
                "GROUP BY backend"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("[KnowledgeCache] stats error: %s", e)
            total = 0
            by_type_rows = []
            by_backend_rows = []

        hits = self._hits_today
        misses = self._misses_today
        hit_rate = hits / max(1, hits + misses)
        return {
            "entries": total,
            "hits_24h": hits,
            "misses_24h": misses,
            "hit_rate": round(hit_rate, 4),
            "bytes_saved_24h_estimate": int(self._bytes_saved_today),
            "by_query_type": {t: int(c) for t, c in by_type_rows},
            "by_backend": {b: int(c) for b, c in by_backend_rows},
            "size_cap": self.size_cap,
            "db_path": self.db_path,
        }


__all__ = [
    "CacheEntry",
    "KnowledgeCache",
    "TTL_FAILURE",
    "TTL_SUCCESS",
    "resolve_ttl",
]
