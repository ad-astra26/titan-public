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
  - These helpers honor the `timeout=` kwarg passed to `connect()`.
  - WAL mode is recommended for read-heavy DBs to allow concurrent
    readers (callers should `PRAGMA journal_mode=WAL` once at init).

2026-04-21 — BUG-SQLITE-WRITER-CONTENTION Option A:
  Default timeout for `query()` lowered from 10.0s → 2.0s. Rationale: readers
  under concurrent load (≥5 concurrent chat requests on the public Observatory)
  were hitting sqlite3's 10-second BUSY wait when a writer held the lock,
  driving p99 latency to ~15s (httpx client timeout). Fast-fail to 2.0s lets
  readers degrade to partial results instead of blocking the event loop.
  `execute()` + `with_connection()` keep default 10.0s since writes need
  more time and callers that KNOW they're reading can pass `timeout=2.0`.
  Module-level `_contention_counter` tracks OperationalError events for the
  new `/v4/db-contention` diagnostic endpoint.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from threading import Lock
from typing import Any, Iterable


# ─────────────────────────────────────────────────────────────────
# Contention telemetry (2026-04-21 BUG-SQLITE Option A diagnostic)
# ─────────────────────────────────────────────────────────────────
#
# Records sqlite3.OperationalError ("database is locked") events across all
# helper call sites. Readable via a dashboard endpoint for live p99 triage.
# Lock keeps increment-and-read atomic across asyncio.to_thread() callers.

_contention_lock = Lock()
_contention_counter: dict[str, Any] = {
    "timeouts_total": 0,
    "timeouts_by_op": {"query": 0, "execute": 0,
                       "executemany": 0, "with_connection": 0},
    "last_timeout_ts": 0.0,
    "last_timeout_db": "",
    "last_timeout_op": "",
    "last_timeout_msg": "",
}


def _record_contention(op: str, db_path: str, err: Exception) -> None:
    """Record a sqlite busy/timeout event for diagnostic reporting."""
    msg = str(err)
    if "locked" not in msg.lower() and "busy" not in msg.lower():
        # Only count lock/busy errors — real SQL errors are a different class.
        return
    with _contention_lock:
        _contention_counter["timeouts_total"] += 1
        by_op = _contention_counter["timeouts_by_op"]
        by_op[op] = by_op.get(op, 0) + 1
        _contention_counter["last_timeout_ts"] = time.time()
        _contention_counter["last_timeout_db"] = db_path
        _contention_counter["last_timeout_op"] = op
        _contention_counter["last_timeout_msg"] = msg[:200]


def get_contention_stats() -> dict:
    """Snapshot for `/v4/db-contention` endpoint. Thread-safe shallow copy."""
    with _contention_lock:
        return {
            "timeouts_total": _contention_counter["timeouts_total"],
            "timeouts_by_op": dict(
                _contention_counter["timeouts_by_op"]),
            "last_timeout_ts": _contention_counter["last_timeout_ts"],
            "last_timeout_db": _contention_counter["last_timeout_db"],
            "last_timeout_op": _contention_counter["last_timeout_op"],
            "last_timeout_msg": _contention_counter["last_timeout_msg"],
            "last_timeout_age_s": (
                round(time.time() - _contention_counter["last_timeout_ts"], 1)
                if _contention_counter["last_timeout_ts"] else None),
        }


async def query(db_path: str, sql: str, params: Iterable[Any] = (),
                timeout: float = 2.0, fetch: str = "all",
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
        try:
            with sqlite3.connect(db_path, timeout=timeout) as conn:
                if row_factory is not None:
                    conn.row_factory = row_factory
                cur = conn.execute(sql, tuple(params))
                if fetch == "all":
                    return cur.fetchall()
                if fetch == "one":
                    return cur.fetchone()
                return None
        except sqlite3.OperationalError as _err:
            _record_contention("query", db_path, _err)
            raise
    return await asyncio.to_thread(_do)


async def execute(db_path: str, sql: str, params: Iterable[Any] = (),
                  timeout: float = 10.0) -> int:
    """Execute a write statement off the asyncio event loop.

    Returns the affected rowcount (rows changed).
    Commits the implicit transaction via `with` block.
    """
    def _do():
        try:
            with sqlite3.connect(db_path, timeout=timeout) as conn:
                cur = conn.execute(sql, tuple(params))
                return cur.rowcount
        except sqlite3.OperationalError as _err:
            _record_contention("execute", db_path, _err)
            raise
    return await asyncio.to_thread(_do)


async def executemany(db_path: str, sql: str, seq_of_params: Iterable[Iterable[Any]],
                      timeout: float = 10.0) -> int:
    """Bulk-execute a write statement off the asyncio event loop.

    Single transaction across all rows for atomicity + speed.
    """
    def _do():
        try:
            with sqlite3.connect(db_path, timeout=timeout) as conn:
                cur = conn.executemany(sql, [tuple(p) for p in seq_of_params])
                return cur.rowcount
        except sqlite3.OperationalError as _err:
            _record_contention("executemany", db_path, _err)
            raise
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
        try:
            with sqlite3.connect(db_path, timeout=timeout) as conn:
                if row_factory is not None:
                    conn.row_factory = row_factory
                return fn(conn)
        except sqlite3.OperationalError as _err:
            _record_contention("with_connection", db_path, _err)
            raise
    return await asyncio.to_thread(_do)
