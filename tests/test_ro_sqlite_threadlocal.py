"""ThreadLocalRoSqlite — thread-safe read-only sqlite reader.

Regression for the timechain/index.db concurrent-read bug: a single shared
sqlite connection (check_same_thread=False) raised sqlite3.InterfaceError under
concurrent FORK_READ recall from parallel chat turns. The thread-local reader
gives each thread its own mode=ro connection (unlimited concurrent readers,
contention-free).
"""
import os
import sqlite3
import threading

from titan_hcl.synthesis.ro_sqlite import ThreadLocalRoSqlite


def _make_db(path, n=200):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.executemany("INSERT INTO t (id, v) VALUES (?,?)",
                  [(i, f"row{i}") for i in range(n)])
    c.commit()
    c.close()


def test_execute_reads_rows(tmp_path):
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    row = r.execute("SELECT v FROM t WHERE id=?", (5,)).fetchone()
    assert row[0] == "row5"


def test_truthy_like_a_connection(tmp_path):
    # RuleEvaluator does `if self._index_db is not None` / `if not self._index_db`
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    assert r is not None and bool(r) is True


def test_row_factory_applied_per_connection(tmp_path):
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro", row_factory=sqlite3.Row)
    row = r.execute("SELECT id, v FROM t WHERE id=?", (7,)).fetchone()
    assert row["v"] == "row7"          # string-key access (Row) works


def test_getattr_proxies_cursor(tmp_path):
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    cur = r.cursor()                    # proxied to the per-thread connection
    cur.execute("SELECT count(*) FROM t")
    assert cur.fetchone()[0] == 200


def test_distinct_connection_per_thread(tmp_path):
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    seen = {}

    def grab(tid):
        seen[tid] = id(r._conn())      # the underlying per-thread connection object

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(seen.values())) == 4   # 4 threads → 4 distinct connections


def test_heavy_concurrent_reads_never_error(tmp_path):
    # THE regression: many threads reading the SAME ThreadLocalRoSqlite concurrently
    # must all succeed (a shared single connection raised InterfaceError here).
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    errors = []
    results = []

    def hammer():
        try:
            for i in range(50):
                row = r.execute("SELECT v FROM t WHERE id=?", (i % 200,)).fetchone()
                results.append(row[0])
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=hammer) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent reads raised: {errors[:3]}"
    assert len(results) == 12 * 50       # every read returned


def test_close_is_idempotent(tmp_path):
    db = str(tmp_path / "x.db")
    _make_db(db)
    r = ThreadLocalRoSqlite(f"file:{db}?mode=ro")
    r.execute("SELECT 1").fetchone()
    r.close()
    r.close()                            # idempotent
    # re-opens lazily after close
    assert r.execute("SELECT count(*) FROM t").fetchone()[0] == 200
