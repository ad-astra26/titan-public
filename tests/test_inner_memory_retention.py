"""Inner-memory retention (A, 2026-06-18): bound the high-frequency `program_fires`
telemetry log (478k rows / 1.1GB on T1). Readers only need a recent window, so
keeping the most-recent N rows is safe. Verifies the retention SQL keeps exactly
the recent window, is a no-op when the table is small, and routes through the IMW
writer with NO VACUUM.
"""
import sqlite3

from titan_hcl.logic.inner_memory import InnerMemoryStore

_PRUNE_SQL = ("DELETE FROM program_fires "
              "WHERE id <= (SELECT MAX(id) FROM program_fires) - ?")


def _make_program_fires(db_path, n):
    c = sqlite3.connect(str(db_path))
    c.execute("CREATE TABLE program_fires (id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "timestamp REAL, program TEXT)")
    for i in range(n):
        c.execute("INSERT INTO program_fires (timestamp, program) VALUES (?, ?)",
                  (float(i), "VIGILANCE"))
    c.commit()
    return c


def test_retention_sql_keeps_exactly_the_recent_window(tmp_path):
    c = _make_program_fires(tmp_path / "im.db", 1000)
    c.execute(_PRUNE_SQL, (100,))
    c.commit()
    assert c.execute("SELECT COUNT(*) FROM program_fires").fetchone()[0] == 100
    lo, hi = c.execute("SELECT MIN(id), MAX(id) FROM program_fires").fetchone()
    assert (lo, hi) == (901, 1000)  # the most recent 100, newest untouched
    c.close()


def test_retention_noop_when_table_small(tmp_path):
    c = _make_program_fires(tmp_path / "im.db", 10)
    c.execute(_PRUNE_SQL, (50_000,))  # cutoff negative → deletes nothing
    c.commit()
    assert c.execute("SELECT COUNT(*) FROM program_fires").fetchone()[0] == 10
    c.close()


def test_prune_routes_through_writer_no_vacuum():
    captured = {}

    class _FakeClient:
        def write(self, sql, params, table=None):
            captured.update(sql=sql, params=params, table=table)

    store = InnerMemoryStore.__new__(InnerMemoryStore)  # bypass __init__ (no real client)
    store._client = _FakeClient()
    store.prune_program_fires(keep_rows=12_345)

    assert "DELETE FROM program_fires" in captured["sql"]
    assert "VACUUM" not in captured["sql"].upper()  # never VACUUM (T3 2026-04-21 lesson)
    assert captured["params"] == (12_345,)
    assert captured["table"] == "program_fires"
