"""IMW metrics — simple thread/async-safe counters + gauges for observability.

Exposed at GET /v4/imw-health. Also used by arch_map services/health checks.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


class IMWMetrics:
    """Lightweight metrics registry for the service process.

    All counters are ints; all gauges are floats. Thread-safe via a single lock
    (low-contention: metrics updates are far less frequent than writes).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()

        # Counters (lifetime)
        self.total_writes = 0
        self.total_errors = 0
        self.total_batches = 0
        self.total_per_write_fallbacks = 0
        self.journal_replay_count_lifetime = 0
        self.connections_accepted = 0

        # Gauges (current)
        self.connections_active = 0
        self.queue_depth = 0
        self.in_flight_requests = 0
        self.service_wal_size_mb = 0.0

        # Rolling window (last 60s) for rate calculations
        self._commit_events: Deque[tuple] = deque(maxlen=4096)   # (ts, n_writes, latency_ms)

        # Last error (string)
        self.last_error: str | None = None
        self.last_error_ts: float | None = None

        # Canonical tables (Phase 3 state)
        self.tables_canonical: list[str] = []

    # ── counter ops ──────────────────────────────────────────────────

    def incr_writes(self, n: int = 1) -> None:
        with self._lock:
            self.total_writes += n

    def incr_errors(self, err: str) -> None:
        with self._lock:
            self.total_errors += 1
            self.last_error = err
            self.last_error_ts = time.time()

    def incr_batches(self) -> None:
        with self._lock:
            self.total_batches += 1

    def incr_per_write_fallback(self) -> None:
        with self._lock:
            self.total_per_write_fallbacks += 1

    def incr_journal_replay(self, n: int = 1) -> None:
        with self._lock:
            self.journal_replay_count_lifetime += n

    def incr_connections(self) -> None:
        with self._lock:
            self.connections_accepted += 1
            self.connections_active += 1

    def decr_connections(self) -> None:
        with self._lock:
            self.connections_active = max(0, self.connections_active - 1)

    # ── gauge ops ────────────────────────────────────────────────────

    def set_queue_depth(self, n: int) -> None:
        with self._lock:
            self.queue_depth = n

    def set_in_flight(self, n: int) -> None:
        with self._lock:
            self.in_flight_requests = n

    def set_wal_size_mb(self, mb: float) -> None:
        with self._lock:
            self.service_wal_size_mb = mb

    def set_tables_canonical(self, tables: list[str]) -> None:
        with self._lock:
            self.tables_canonical = list(tables)

    def record_commit(self, n_writes: int, latency_ms: float) -> None:
        with self._lock:
            self._commit_events.append((time.time(), n_writes, latency_ms))

    # ── derived ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            recent = [e for e in self._commit_events if e[0] >= cutoff]
            writes_1m = sum(e[1] for e in recent)
            if recent:
                avg_lat = sum(e[2] for e in recent) / len(recent)
                avg_batch = writes_1m / len(recent)
            else:
                avg_lat = 0.0
                avg_batch = 0.0
            return {
                "status": "ok" if self.last_error is None else "degraded",
                "uptime_sec": now - self._started_at,
                "connections_active": self.connections_active,
                "connections_accepted_lifetime": self.connections_accepted,
                "queue_depth": self.queue_depth,
                "in_flight_requests": self.in_flight_requests,
                "commits_per_sec_1m": writes_1m / 60.0,
                "avg_batch_fill": avg_batch,
                "avg_commit_latency_ms": avg_lat,
                "total_writes": self.total_writes,
                "total_errors": self.total_errors,
                "total_batches": self.total_batches,
                "total_per_write_fallbacks": self.total_per_write_fallbacks,
                "journal_replay_count_lifetime": self.journal_replay_count_lifetime,
                "service_wal_size_mb": self.service_wal_size_mb,
                "last_error": self.last_error,
                "last_error_ts": self.last_error_ts,
                "tables_canonical": list(self.tables_canonical),
            }
