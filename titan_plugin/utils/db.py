"""Centralized SQLite connection helper for Titan.

All code that opens a SQLite database should use ``safe_connect`` instead
of raw ``sqlite3.connect`` to ensure:

1. **WAL mode** — allows concurrent readers + serialised writers.
   WAL is a database-level flag, but re-issuing the PRAGMA on every
   connection is cheap and guarantees it after a crash/recovery.
2. **Consistent busy_timeout** — short timeouts (1-3s) fail under
   multi-process contention on CPU-starved VPS.  Default 10s.
3. **Synchronous=NORMAL** — safe with WAL; avoids fsync on every
   commit, reducing write latency by ~5-10x without durability risk.

Usage::

    from titan_plugin.utils.db import safe_connect

    conn = safe_connect("data/inner_memory.db")
    # ... use conn ...
    conn.close()

Or as a context manager::

    with safe_connect("data/inner_memory.db") as conn:
        conn.execute("INSERT INTO ...")
"""

from __future__ import annotations

import sqlite3

# Default busy timeout in seconds.  Under multi-process contention on a
# 4-vCPU VPS running 2 Titans, 2-3s is routinely insufficient.  10s
# matches the Layer 2 convention established in reasoning.py.
DEFAULT_TIMEOUT = 10.0


def safe_connect(
    db_path: str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    check_same_thread: bool = True,
    wal: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection with WAL + busy_timeout guarantees.

    Parameters
    ----------
    db_path : str
        Path to the ``.db`` file (relative or absolute).
    timeout : float
        Busy-wait timeout in **seconds** (maps to sqlite3_busy_timeout).
    check_same_thread : bool
        Pass ``False`` for persistent connections shared across threads
        within one process.  Never share across *processes*.
    wal : bool
        Set ``PRAGMA journal_mode=WAL``.  Default True for all shared
        databases.  Set False only for single-process throwaway DBs.
    """
    conn = sqlite3.connect(db_path, timeout=timeout,
                           check_same_thread=check_same_thread)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn
