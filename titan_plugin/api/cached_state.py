"""
titan_plugin/api/cached_state.py — bus-event-backed state cache.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

The api_subprocess holds a `CachedState` dict that the BusSubscriber updates
on every relevant *_UPDATED bus event from the kernel. Endpoint code reads
from the cache via the StateAccessor abstraction — zero IPC, sub-microsecond
latency, no sync RPC fragility.

Cache invalidation policy: pure event-driven (no TTL). The kernel is the
single source of truth; it publishes update events on every state change.
Bootstrap: BusSubscriber issues STATE_SNAPSHOT_REQUEST at api_subprocess
start; kernel responds with current state; cache populated.

Stale-tolerance: every entry carries `wall_ns` (insertion timestamp).
`get(key, default=None)` returns the value plain; `get_with_age(key)`
returns (value, age_seconds). Endpoints that care about freshness can
choose; most don't (state is updated frequently enough that staleness is
imperceptible).

Thread safety: all reads/writes guarded by a single RLock. Read path is
fast (dict lookup + lock acquisition ~100ns). Writer is the BusSubscriber's
listener thread; readers are the FastAPI endpoint coroutines (single
asyncio loop) plus diagnostic threads.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class CachedState:
    """Thread-safe state cache populated by bus events.

    Keys are dotted strings matching state paths (e.g. "network.balance",
    "guardian.status", "agency.action_count"). Values are arbitrary
    msgpack-compatible Python objects (dicts, lists, scalars).

    The kernel side publishes per-key update events; this class is a
    passive consumer.
    """

    __slots__ = ("_data", "_wall_ns", "_lock", "_bootstrap_done")

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._wall_ns: dict[str, int] = {}
        self._lock = threading.RLock()
        self._bootstrap_done = threading.Event()

    # -- writes (BusSubscriber side) -----------------------------------

    def set(self, key: str, value: Any, wall_ns: int | None = None) -> None:
        """Update a single key. Idempotent."""
        ts = wall_ns if wall_ns is not None else time.time_ns()
        with self._lock:
            self._data[key] = value
            self._wall_ns[key] = ts

    def bulk_update(self, mapping: dict[str, Any], wall_ns: int | None = None) -> None:
        """Atomic-ish bulk update — used by bootstrap snapshot."""
        ts = wall_ns if wall_ns is not None else time.time_ns()
        with self._lock:
            for k, v in mapping.items():
                self._data[k] = v
                self._wall_ns[k] = ts

    def mark_bootstrap_done(self) -> None:
        """Signal that the initial STATE_SNAPSHOT_RESPONSE has been processed.
        Endpoint code may wait on this before serving requests if it requires
        a populated cache.
        """
        self._bootstrap_done.set()

    # -- reads (endpoint side) ----------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value or default. Fast path."""
        with self._lock:
            return self._data.get(key, default)

    def get_with_age(self, key: str, default: Any = None) -> tuple[Any, float | None]:
        """Return (value, age_seconds) — age is None if key missing."""
        with self._lock:
            if key not in self._data:
                return (default, None)
            age_s = (time.time_ns() - self._wall_ns[key]) / 1e9
            return (self._data[key], age_s)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> dict[str, Any]:
        """Defensive shallow copy — used by /v4/cache-staleness diagnostic."""
        with self._lock:
            return dict(self._data)

    def staleness_report(self) -> dict[str, float]:
        """Per-key age in seconds — diagnostic for cache health gates."""
        now_ns = time.time_ns()
        with self._lock:
            return {
                k: (now_ns - ts) / 1e9
                for k, ts in self._wall_ns.items()
            }

    def wait_bootstrap(self, timeout: float = 10.0) -> bool:
        """Block until the bootstrap snapshot has populated the cache. Returns
        True on success, False on timeout. Used by api_subprocess startup
        to gate /health on a populated cache."""
        return self._bootstrap_done.wait(timeout=timeout)

    @property
    def bootstrap_done(self) -> bool:
        return self._bootstrap_done.is_set()

    # -- introspection -------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        with self._lock:
            return f"<CachedState keys={len(self._data)} bootstrap={self._bootstrap_done.is_set()}>"
