"""ThreadLocalRoSqlite — a thread-safe read-only sqlite reader (one connection PER THREAD).

Why this exists: a single `sqlite3.Connection` is NOT safe for concurrent `execute()`
across threads — even with `check_same_thread=False` (which only silences the
thread-OWNERSHIP check, it does not serialize access). Under concurrent chat turns the
agno PreHook runs recall (`RuleEvaluator._exec_fork_read`) on ONE shared
`timechain/index.db` connection from multiple threads → `sqlite3.InterfaceError:
bad parameter or other API misuse`, dropping that turn's FORK_READ recall (degraded
answer). Latent since 2026-05-28 (recall_reader); the chat-concurrency arc exposed it.

The fix — NOT IMW (IMW serializes WRITES to registered DBs; this is a `mode=ro` READ):
sqlite allows UNLIMITED concurrent readers via SEPARATE connections with zero write-lock
contention, so a per-thread connection is both correct AND contention-free (a single
shared lock would serialize all recall — a bottleneck that defeats the concurrency arc).

Drop-in for the read-only-connection subset callers use: `.execute()` (returns the real
per-thread cursor → `.fetchone()`/`.fetchall()`/iteration all work), truthiness (always
True → the "index present" branch), and any other Connection attr via `__getattr__`.
`row_factory` is set per new connection from the constructor (do NOT post-hoc setattr on
the wrapper — that would not reach the per-thread connections).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ThreadLocalRoSqlite:
    """A read-only sqlite reader that opens ONE connection per accessing thread."""

    def __init__(self, uri: str, *, timeout: float = 1.0,
                 row_factory: Optional[Callable[[sqlite3.Cursor, Any], Any]] = None):
        self._uri = uri
        self._timeout = float(timeout)
        self._row_factory = row_factory
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(
                self._uri, uri=True, check_same_thread=False, timeout=self._timeout)
            if self._row_factory is not None:
                c.row_factory = self._row_factory
            self._local.conn = c
        return c

    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        return self._conn().execute(sql, params)

    def close(self) -> None:
        """Close THIS thread's connection (idempotent). Other threads' connections
        close on their own thread's teardown / process exit — read-only, harmless."""
        c = getattr(self._local, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
            self._local.conn = None

    def __getattr__(self, name: str) -> Any:
        # Proxy any other Connection method (cursor/executemany/…) to the per-thread
        # connection. Only reached for names NOT found normally (so _uri/_local/execute
        # never hit here → no recursion).
        return getattr(self._conn(), name)
