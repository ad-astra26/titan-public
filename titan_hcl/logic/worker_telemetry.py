"""Worker bottleneck telemetry — persistent, low-overhead op/memory instrumentation.

RFP_worker_telemetry_bottleneck_instrumentation §7.A (Phase A).

WHY: synthesis + agno are the heavy workers; under sustained load their main
loops stall long enough to age out the heartbeat → `shm_pid_dead` → the guardian
DISABLES them (observed fleet-wide 2026-06-15/16 within ~1h of deploy). The boxes
ran fine for weeks — the workers are under-optimized, not the VPS. We cannot fix
what we cannot see. This lib lets a worker SELF-TIME its operations (tagged by
`feature` + a per-chat-turn `trigger_id`) and sample RssAnon, persisting to a
per-worker SQLite-WAL DB that SURVIVES the restart the stall causes — so the
precise bottleneck becomes a fact, not a guess.

DESIGN (the invariants this file realizes):
  • INV-TEL-1 — the telemetry must not BECOME the load it measures. The hot path
    (`timed`/`record_*`) only appends to a bounded in-memory ring (a `deque` whose
    append is atomic in CPython — no lock, no disk). ALL SQLite writes happen ONLY
    on the single low-priority flusher thread, batched.
  • INV-TEL-2 — RssAnon (real, non-reclaimable) is the leak signal, NOT VmRSS
    (which over-counts reclaimable mmap'd FAISS/Kuzu/DuckDB pages). We sample both
    but alert on RssAnon.
  • INV-TEL-3 — bounded + self-pruning: the ring is `maxlen` (drops oldest on
    overflow, never blocks the worker); the DB prunes rows older than retention.
  • INV-TEL-4 — per-worker isolation: each worker writes ONLY its own
    `telemetry_<worker>.db` (single-writer; honors the storage-topology G21
    discipline). Disposable diagnostic index → not backed up.
  • INV-TEL-5 — never raises into the worker: every call is best-effort +
    exc_info-logged; a telemetry fault degrades to no-data, never a worker crash.

Default ON (Maker flag rule — features ship enabled; `[telemetry] enabled=false`
is the kill-switch).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Defaults — overridable via `[telemetry]` (titan_params/config) or the ctor.
_DEFAULTS: Dict[str, Any] = {
    "enabled": True,            # Maker rule: default ON; =false is the kill-switch
    "db_dir": "data",           # cwd-relative, per-worker file telemetry_<worker>.db
    "flush_s": 5.0,             # flusher drain cadence
    "mem_sample_s": 30.0,       # RssAnon/VmRSS sample cadence
    "warn_ms": 5000.0,          # an op longer than this is flagged `stall=1`
    "ring_cap": 10000,          # bounded in-mem ring (drops oldest on overflow)
    "retention_days": 7.0,      # prune op_events/memory_samples older than this
    "prune_every_flushes": 720, # ~1h at flush_s=5 — prune is not free, do it rarely
}


def _resolve_config(override: Optional[dict]) -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    # titan_params/config `[telemetry]` section (best-effort — absent → defaults)
    try:
        from titan_hcl.params import get_params
        section = get_params("telemetry") or {}
        for k in _DEFAULTS:
            if k in section:
                cfg[k] = section[k]
    except Exception:
        pass
    if override:
        cfg.update({k: v for k, v in override.items() if k in _DEFAULTS})
    return cfg


def _read_proc_status_kb(field: str) -> float:
    """Return the kB value of a /proc/self/status field (e.g. 'RssAnon'), 0.0 on
    any failure. Format: 'RssAnon:\\t    1492 kB'. Never raises."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return float(line.split()[1])  # the number before 'kB'
    except Exception:
        pass
    return 0.0


def _read_rss_anon_mb() -> float:
    """RssAnon in MB — the leak signal (real anon memory, not reclaimable mmap).
    NOTE: nothing else reads RssAnon today; the heartbeat (`bus_socket
    ._enrich_heartbeat`) reads /proc/self/stat = VmRSS. This is the fresh read."""
    return _read_proc_status_kb("RssAnon") / 1024.0


def _read_vmrss_mb() -> float:
    return _read_proc_status_kb("VmRSS") / 1024.0


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS op_events (
        ts REAL, op TEXT, feature TEXT, trigger_id TEXT,
        duration_ms REAL, rss_anon_mb REAL, ctx_json TEXT, stall INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS ix_op_events_feature ON op_events(feature)",
    "CREATE INDEX IF NOT EXISTS ix_op_events_trigger ON op_events(trigger_id)",
    "CREATE INDEX IF NOT EXISTS ix_op_events_ts ON op_events(ts)",
    """CREATE TABLE IF NOT EXISTS memory_samples (
        ts REAL, rss_anon_mb REAL, vmrss_mb REAL, sizes_json TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_mem_ts ON memory_samples(ts)",
    """CREATE TABLE IF NOT EXISTS restart_events (
        ts REAL, reason TEXT, prev_uptime_s REAL
    )""",
)


class Telemetry:
    """Per-worker telemetry recorder. ONE instance per worker process.

    Hot path: `with tel.timed("op", feature="research"): …` → appends to the ring
    (atomic, no disk). A daemon flusher thread persists to SQLite-WAL + samples
    memory. Every public method is best-effort and NEVER raises (INV-TEL-5)."""

    def __init__(self, worker_name: str, config: Optional[dict] = None) -> None:
        self.worker = worker_name
        self._cfg = _resolve_config(config)
        self._enabled = bool(self._cfg["enabled"])
        self._ring: "deque[tuple]" = deque(maxlen=int(self._cfg["ring_cap"]))
        self._db_path = os.path.join(
            str(self._cfg["db_dir"]), f"telemetry_{worker_name}.db")
        self._stop = threading.Event()
        self._flusher: Optional[threading.Thread] = None
        self._started_monotonic = time.monotonic()
        if not self._enabled:
            logger.info("[telemetry:%s] DISABLED (kill-switch)", worker_name)
            return
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._init_db()
            self._flusher = threading.Thread(
                target=self._flush_loop, name=f"telemetry-{worker_name}",
                daemon=True)
            self._flusher.start()
            logger.info("[telemetry:%s] ON → %s (flush=%.0fs mem=%.0fs warn=%dms)",
                        worker_name, self._db_path, self._cfg["flush_s"],
                        self._cfg["mem_sample_s"], int(self._cfg["warn_ms"]))
        except Exception:
            logger.warning("[telemetry:%s] init failed → disabled", worker_name,
                           exc_info=True)
            self._enabled = False

    # ── DB ──────────────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            for stmt in _SCHEMA:
                conn.execute(stmt)
            conn.commit()
        finally:
            conn.close()

    # ── hot path (INV-TEL-1: ring-append only, never blocks) ────────────────
    @contextmanager
    def timed(self, op: str, feature: Optional[str] = None,
              trigger_id: Optional[str] = None, **ctx: Any):
        """Time an operation: `with tel.timed("consolidation", feature="...",
        clusters=N): …`. No-op (cheap yield) when disabled."""
        if not self._enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            try:
                dur_ms = (time.monotonic() - t0) * 1000.0
                self._record(op, feature, trigger_id, dur_ms, ctx)
            except Exception:
                logger.warning("[telemetry:%s] timed record failed",
                               self.worker, exc_info=True)

    def record_stage(self, op: str, duration_ms: float,
                     rss_mb: Optional[float] = None,
                     feature: Optional[str] = None,
                     trigger_id: Optional[str] = None, **ctx: Any) -> None:
        """Record an ALREADY-timed op (for callers that own their timer, e.g.
        agno `_ph_stage` / synthesis `_timed`). rss_mb optional — read live if None."""
        if not self._enabled:
            return
        try:
            self._record(op, feature, trigger_id, float(duration_ms), ctx,
                         rss_mb=rss_mb)
        except Exception:
            logger.warning("[telemetry:%s] record_stage failed", self.worker,
                           exc_info=True)

    def record_boot(self, reason: str = "", prev_uptime_s: float = 0.0) -> None:
        """Record a worker (re)boot / disable reason → restart_events."""
        if not self._enabled:
            return
        self._ring.append(("restart", float(time.time()), str(reason),
                           float(prev_uptime_s)))

    def _record(self, op: str, feature: Optional[str], trigger_id: Optional[str],
                dur_ms: float, ctx: dict, rss_mb: Optional[float] = None) -> None:
        rss = float(rss_mb) if rss_mb is not None else _read_rss_anon_mb()
        stall = 1 if dur_ms > float(self._cfg["warn_ms"]) else 0
        ctx_json = json.dumps(ctx, default=str) if ctx else None
        # deque.append is atomic in CPython → no lock on the hot path (INV-TEL-1).
        self._ring.append(("op", float(time.time()), op, feature, trigger_id,
                           dur_ms, rss, ctx_json, stall))

    # ── flusher thread (the ONLY writer — INV-TEL-1/-4) ─────────────────────
    def _drain_ring(self) -> list:
        out = []
        while True:
            try:
                out.append(self._ring.popleft())   # atomic
            except IndexError:
                break
        return out

    def _flush_loop(self) -> None:
        flush_s = float(self._cfg["flush_s"])
        mem_s = float(self._cfg["mem_sample_s"])
        prune_every = int(self._cfg["prune_every_flushes"])
        last_mem = 0.0
        flushes = 0
        while not self._stop.wait(flush_s):
            try:
                batch = self._drain_ring()
                now = time.monotonic()
                do_mem = (now - last_mem) >= mem_s
                if not batch and not do_mem:
                    continue
                conn = self._connect()
                try:
                    ops = [r[1:] for r in batch if r[0] == "op"]
                    restarts = [r[1:] for r in batch if r[0] == "restart"]
                    if ops:
                        conn.executemany(
                            "INSERT INTO op_events(ts,op,feature,trigger_id,"
                            "duration_ms,rss_anon_mb,ctx_json,stall) "
                            "VALUES(?,?,?,?,?,?,?,?)", ops)
                    if restarts:
                        conn.executemany(
                            "INSERT INTO restart_events(ts,reason,prev_uptime_s) "
                            "VALUES(?,?,?)", restarts)
                    if do_mem:
                        last_mem = now
                        conn.execute(
                            "INSERT INTO memory_samples(ts,rss_anon_mb,vmrss_mb,"
                            "sizes_json) VALUES(?,?,?,?)",
                            (time.time(), _read_rss_anon_mb(), _read_vmrss_mb(),
                             None))
                    conn.commit()
                    flushes += 1
                    if prune_every and flushes % prune_every == 0:
                        self._prune(conn)
                        conn.commit()
                finally:
                    conn.close()
            except Exception:
                logger.warning("[telemetry:%s] flush failed (continuing)",
                               self.worker, exc_info=True)

    def _prune(self, conn: sqlite3.Connection) -> None:
        cutoff = time.time() - float(self._cfg["retention_days"]) * 86400.0
        conn.execute("DELETE FROM op_events WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM memory_samples WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM restart_events WHERE ts < ?", (cutoff,))

    def flush_now(self) -> None:
        """Force a synchronous drain+write (for tests / shutdown). Best-effort."""
        if not self._enabled:
            return
        try:
            batch = self._drain_ring()
            if not batch:
                return
            conn = self._connect()
            try:
                ops = [r[1:] for r in batch if r[0] == "op"]
                restarts = [r[1:] for r in batch if r[0] == "restart"]
                if ops:
                    conn.executemany(
                        "INSERT INTO op_events(ts,op,feature,trigger_id,"
                        "duration_ms,rss_anon_mb,ctx_json,stall) "
                        "VALUES(?,?,?,?,?,?,?,?)", ops)
                if restarts:
                    conn.executemany(
                        "INSERT INTO restart_events(ts,reason,prev_uptime_s) "
                        "VALUES(?,?,?)", restarts)
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning("[telemetry:%s] flush_now failed", self.worker,
                           exc_info=True)

    def close(self) -> None:
        self._stop.set()
        self.flush_now()


# ── per-process factory (one Telemetry per worker) ──────────────────────────
_INSTANCES: Dict[str, Telemetry] = {}
_INSTANCES_LOCK = threading.Lock()


def get_telemetry(worker_name: str, config: Optional[dict] = None) -> Telemetry:
    """Return the process's Telemetry for `worker_name` (created once)."""
    with _INSTANCES_LOCK:
        tel = _INSTANCES.get(worker_name)
        if tel is None:
            tel = Telemetry(worker_name, config)
            _INSTANCES[worker_name] = tel
        return tel
