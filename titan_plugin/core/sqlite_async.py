"""Async-safe sqlite3 helpers for FastAPI endpoints.

Built 2026-04-14 after `arch_map async-blocks` scanner identified 30+
sqlite3.connect call sites reachable from FastAPI endpoints. Each was
a potential event-loop block (sqlite3 is synchronous, holds the GIL,
plus disk I/O can be seconds under load).

Use these helpers from any `async def` function instead of bare
`sqlite3.connect`. The helpers wrap the blocking I/O in
`asyncio.to_thread()` so the event loop stays responsive.

For sync functions (workers, scripts, init paths), continue to use
`sqlite3.connect` directly — wrapping there adds overhead with no
benefit.

Concurrency notes:
  - SQLite has process-wide write lock; concurrent writers serialize
    via the SQLITE_BUSY/timeout mechanism.
  - These helpers honor the `timeout=` kwarg passed to `connect()`
    (default 10s).
  - WAL mode is recommended for read-heavy DBs to allow concurrent
    readers (callers should `PRAGMA journal_mode=WAL` once at init).
"""
from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Iterable


async def query(db_path: str, sql: str, params: Iterable[Any] = (),
                timeout: float = 10.0, fetch: str = "all",
                row_factory: Any = None) -> list | tuple | None:
    """Execute a read query off the asyncio event loop.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL statement (typically SELECT).
        params: Bind parameters.
        timeout: SQLite busy-wait timeout in seconds.
        fetch: "all" (list of rows), "one" (single row or None), or
               "none" (return None — useful for DDL).
        row_factory: optional sqlite3 row_factory (e.g., sqlite3.Row for
               dict-like access by column name). Default: tuples.

    Returns:
        Per `fetch` mode.
    """
    def _do():
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            if row_factory is not None:
                conn.row_factory = row_factory
            cur = conn.execute(sql, tuple(params))
            if fetch == "all":
                return cur.fetchall()
            if fetch == "one":
                return cur.fetchone()
            return None
    return await asyncio.to_thread(_do)


async def execute(db_path: str, sql: str, params: Iterable[Any] = (),
                  timeout: float = 10.0) -> int:
    """Execute a write statement off the asyncio event loop.

    Returns the affected rowcount (rows changed).
    Commits the implicit transaction via `with` block.
    """
    def _do():
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            cur = conn.execute(sql, tuple(params))
            return cur.rowcount
    return await asyncio.to_thread(_do)


async def executemany(db_path: str, sql: str, seq_of_params: Iterable[Iterable[Any]],
                      timeout: float = 10.0) -> int:
    """Bulk-execute a write statement off the asyncio event loop.

    Single transaction across all rows for atomicity + speed.
    """
    def _do():
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            cur = conn.executemany(sql, [tuple(p) for p in seq_of_params])
            return cur.rowcount
    return await asyncio.to_thread(_do)


async def with_connection(db_path: str, fn, timeout: float = 10.0,
                          row_factory: Any = None):
    """Run a callable that takes a sqlite3.Connection, off the event loop.

    Use when the operation needs multiple statements within one transaction
    or returns shaped results that don't fit query()/execute().

    Example:
        async def get_user_with_donations(uid):
            def _fn(conn):
                user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
                donations = conn.execute(
                    "SELECT * FROM donations WHERE user_id=?", (uid,)).fetchall()
                return {"user": user, "donations": donations}
            return await with_connection(db_path, _fn, row_factory=sqlite3.Row)
    """
    def _do():
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            if row_factory is not None:
                conn.row_factory = row_factory
            return fn(conn)
    return await asyncio.to_thread(_do)
