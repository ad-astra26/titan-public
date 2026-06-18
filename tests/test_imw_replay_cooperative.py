"""IMW boot-resilience (2026-06-18): the service-WAL replay must be COOPERATIVE +
batched so a large uncommitted wal can't GIL-starve the heartbeat/bus threads →
guardian kill → infinite boot loop (observed live on T1: 61MB wal, per-record
COMMIT, no yield). Verifies a large wal (> 2× the replay batch) replays every good
record, a poison-pill record is skipped (not fatal), and the offset advances.
"""
import asyncio
import sqlite3

from titan_hcl.persistence.config import IMWConfig
from titan_hcl.persistence.writer_service import IMWDaemon
from titan_hcl.persistence.service_wal import ServiceWAL


def _daemon_with_wal(tmp_path, populate):
    db = tmp_path / "m.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
    conn.commit()
    conn.close()
    walp = str(tmp_path / "imw.wal")
    w = ServiceWAL(walp, max_mb=64)
    populate(w)
    w.close()

    cfg = IMWConfig.from_dict({
        "enabled": True, "mode": "canonical", "tables_canonical": ["t"],
        "socket_path": str(tmp_path / "imw.sock"), "wal_path": walp,
        "journal_dir": str(tmp_path / "j"), "db_path": str(db),
    })
    cfg.ensure_runtime_dirs()
    daemon = IMWDaemon(cfg)
    # Minimal pre-replay setup (mirrors the head of daemon.start()).
    daemon._wal = ServiceWAL(walp, max_mb=64)
    daemon._conn = daemon._open_db(str(db))
    return daemon, db


def test_large_wal_replays_every_record_batched(tmp_path):
    N = 2500  # > 2× the 1000-record replay batch → exercises multiple commits+yields

    def _pop(w):
        for i in range(N):
            w.append_request(f"r{i}", "INSERT INTO t (id, v) VALUES (?, ?)", [i, i * 2])

    daemon, db = _daemon_with_wal(tmp_path, _pop)
    asyncio.run(daemon._replay_service_wal())

    c = sqlite3.connect(str(db))
    assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == N
    assert c.execute("SELECT v FROM t WHERE id=0").fetchone()[0] == 0
    assert c.execute("SELECT v FROM t WHERE id=2499").fetchone()[0] == 4998
    c.close()
    # the checkpoint offset advanced to the wal end (nothing left to replay)
    assert daemon._wal._last_ckpt_offset > 0


def test_poison_pill_record_is_skipped_not_fatal(tmp_path):
    def _pop(w):
        w.append_request("good1", "INSERT INTO t (id, v) VALUES (?, ?)", [1, 10])
        # 2 placeholders, 1 param → sqlite ProgrammingError (the live "11 vs 3" shape)
        w.append_request("poison", "INSERT INTO t (id, v) VALUES (?, ?)", [2])
        w.append_request("good2", "INSERT INTO t (id, v) VALUES (?, ?)", [3, 30])

    daemon, db = _daemon_with_wal(tmp_path, _pop)
    asyncio.run(daemon._replay_service_wal())  # must NOT raise

    c = sqlite3.connect(str(db))
    ids = [r[0] for r in c.execute("SELECT id FROM t ORDER BY id").fetchall()]
    assert ids == [1, 3]  # both good records applied; poison skipped
    c.close()


def test_replay_yields_to_event_loop(tmp_path):
    # Cooperative proof: while a >batch replay runs, a concurrent coroutine must get
    # scheduled (it couldn't if the replay never awaited).
    def _pop(w):
        for i in range(2200):
            w.append_request(f"r{i}", "INSERT INTO t (id, v) VALUES (?, ?)", [i, i])

    daemon, db = _daemon_with_wal(tmp_path, _pop)

    async def _drive():
        ticks = {"n": 0}

        async def _ticker():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0)
        tk = asyncio.ensure_future(_ticker())
        await daemon._replay_service_wal()
        tk.cancel()
        return ticks["n"]

    ticks = asyncio.run(_drive())
    # ≥2 batch boundaries (2200/1000) → the ticker ran during replay
    assert ticks >= 2
