"""IMW commit-loop resilience (B, 2026-06-18): an unexpected exception in
_commit_batch must NEVER terminate the commit loop. Before the guard, a single
unguarded exception silently killed the loop → commits stopped → the service-wal
accumulated unbounded (the 61MB → boot-loop root condition on T1).
"""
import asyncio

from titan_hcl.persistence.config import IMWConfig
from titan_hcl.persistence.writer_service import IMWDaemon
from titan_hcl.persistence.service_wal import ServiceWAL


def _cfg(tmp_path):
    cfg = IMWConfig.from_dict({
        "enabled": True, "mode": "canonical", "tables_canonical": ["t"],
        "socket_path": str(tmp_path / "imw.sock"),
        "wal_path": str(tmp_path / "imw.wal"),
        "journal_dir": str(tmp_path / "j"),
        "db_path": str(tmp_path / "m.db"),
        "batch_window_ms": 5,
        "service_wal_max_mb": 1,  # small cap so the backing-up predicate is testable
    })
    cfg.ensure_runtime_dirs()
    return cfg


def test_commit_loop_survives_unexpected_exception(tmp_path):
    cfg = _cfg(tmp_path)
    daemon = IMWDaemon(cfg)
    daemon._wal = ServiceWAL(str(cfg.wal_path), max_mb=64)  # for _warn_if_wal_backing_up
    calls = {"n": 0}

    async def _boom_then_ok(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic commit fault")  # the silent-killer scenario
        # subsequent batches: no-op (we're proving loop survival, not the commit)

    daemon._commit_batch = _boom_then_ok  # type: ignore[assignment]

    async def _drive():
        task = asyncio.ensure_future(daemon._commit_loop())
        await daemon._queue.put(object())      # batch 1 → raises
        await asyncio.sleep(0.3)
        assert not task.done(), "commit loop died on an exception (B regression)"
        await daemon._queue.put(object())      # batch 2 → must still be processed
        await asyncio.sleep(0.8)               # past the error backoff
        daemon._stop_event.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(_drive())
    assert calls["n"] >= 2, "loop did not keep processing after the fault"


def test_wal_backing_up_warning_is_throttled(tmp_path):
    # B observability: the warning fires when the wal exceeds 60% of the cap and is
    # throttled (no log spam). We just verify the predicate + throttle don't raise.
    cfg = _cfg(tmp_path)
    daemon = IMWDaemon(cfg)
    daemon._wal = ServiceWAL(str(cfg.wal_path), max_mb=1)  # tiny cap → easy to exceed
    big = "x" * 1024
    for i in range(1000):
        daemon._wal.append_request(f"r{i}", "INSERT INTO t VALUES (?)", [big])
    assert daemon._wal.size_mb() >= 0.6  # ~1MB now (cap=1MB) → predicate true
    daemon._warn_if_wal_backing_up()
    t1 = daemon._last_wal_warn_ts
    daemon._warn_if_wal_backing_up()
    assert daemon._last_wal_warn_ts == t1  # throttled (no re-fire within 30s)
    assert t1 > 0.0                         # it did fire once
