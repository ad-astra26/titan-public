"""IMW writer_service periodic SQLite-WAL truncate (WAL-hygiene).

Proves the fix for the 2026-07-07 finding — consciousness.db-wal pinned at a
2.3GB high-water mark because autocheckpoint keeps the WAL functionally small
(passive, in-place) but never shrinks the *file*; only close() truncated it, so a
long-lived writer's `.db-wal` grew to GBs. `IMWDaemon._maybe_truncate_sqlite_wal`
runs a cadence-gated `PRAGMA wal_checkpoint(TRUNCATE)` on the daemon's own
connection to bound the file. Verifies: (1) TRUNCATE shrinks a grown WAL file,
(2) the cadence gate suppresses a too-soon second truncate, (3) the interval<=0
kill-switch disables it entirely.
"""
import os
import sqlite3

import pytest

from titan_hcl.persistence.config import IMWConfig
from titan_hcl.persistence.writer_service import IMWDaemon


def _grow_wal(conn: sqlite3.Connection, n: int = 20_000) -> None:
    conn.execute("BEGIN")
    for i in range(n):
        conn.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 64))
    conn.execute("COMMIT")


def _daemon_with(conn: sqlite3.Connection, interval_s: float) -> IMWDaemon:
    """Minimal daemon carrying only what _maybe_truncate_sqlite_wal touches."""
    d = IMWDaemon.__new__(IMWDaemon)   # bypass async __init__
    d._conn = conn
    d._shadow_conn = None
    d._cfg = IMWConfig(wal_truncate_interval_s=interval_s)
    d._last_wal_truncate = 0.0         # 0 ⇒ first call is due
    return d


@pytest.fixture()
def wal_conn(tmp_path):
    db = str(tmp_path / "hygiene.db")
    conn = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")   # let the WAL grow so we can shrink it
    conn.execute("CREATE TABLE t (x INTEGER, y TEXT)")
    yield conn, db + "-wal"
    conn.close()


def test_truncate_shrinks_grown_wal_file(wal_conn):
    conn, wal_path = wal_conn
    _grow_wal(conn)
    grew = os.path.getsize(wal_path)
    assert grew > 500_000, f"WAL should have grown, got {grew} bytes"

    _daemon_with(conn, 300.0)._maybe_truncate_sqlite_wal()

    after = os.path.getsize(wal_path)
    assert after < grew, f"TRUNCATE should shrink the WAL file ({grew} -> {after})"


def test_cadence_gate_suppresses_too_soon_second_truncate(wal_conn):
    conn, wal_path = wal_conn
    d = _daemon_with(conn, 300.0)
    _grow_wal(conn)
    d._maybe_truncate_sqlite_wal()          # first truncate (due: last=0.0)
    _grow_wal(conn)                          # regrow the WAL
    regrew = os.path.getsize(wal_path)
    d._maybe_truncate_sqlite_wal()          # within 300s ⇒ must NOT truncate
    assert os.path.getsize(wal_path) == regrew, "cadence gate should skip the 2nd truncate"


def test_kill_switch_disables_truncate(wal_conn):
    conn, wal_path = wal_conn
    _grow_wal(conn)
    grew = os.path.getsize(wal_path)
    _daemon_with(conn, 0.0)._maybe_truncate_sqlite_wal()   # interval<=0 ⇒ disabled
    assert os.path.getsize(wal_path) == grew, "kill-switch (0) must not truncate"


def test_best_effort_never_raises_on_bad_conn(wal_conn):
    conn, _ = wal_conn
    conn.close()                             # force PRAGMA to error
    # Must swallow the sqlite error, not propagate (write path must never break).
    _daemon_with(conn, 300.0)._maybe_truncate_sqlite_wal()


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:anchorpy"]))
