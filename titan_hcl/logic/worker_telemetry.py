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
    "hb_gap_warn_ms": 20000.0,  # inter-heartbeat gap over this → HEARTBEAT_GAP row
                                # (normal beat ≈10s; >20s = a missed beat / loop block)
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
    """CREATE TABLE IF NOT EXISTS freeze_dumps (
        ts REAL, stack_text TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_freeze_ts ON freeze_dumps(ts)",
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
        # C4 — component-size provider (best-effort callable → dict), sampled
        # alongside RssAnon into memory_samples.sizes_json. Never on the hot path.
        self._size_provider = None
        # C5 — freeze-dump sink: the worker points its faulthandler all-thread
        # dump at this file; the flusher ingests new dump text into freeze_dumps.
        self._freeze_dump_fp = None
        self._freeze_dump_path: Optional[str] = None
        self._freeze_dump_pos = 0
        self._boot_freeze_text: Optional[str] = None
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

    def record_heartbeat_gap(self, gap_ms: float) -> None:
        """C2 — record a HEARTBEAT_GAP op_event when the inter-heartbeat gap
        exceeds `hb_gap_warn_ms` (a long gap = the worker loop was blocked → the
        post-hoc stall signal the heartbeat itself can't emit during a freeze).
        Normal ~10s beats are NOT recorded (only abnormal gaps). Best-effort."""
        if not self._enabled:
            return
        try:
            if gap_ms > float(self._cfg["hb_gap_warn_ms"]):
                self._record("HEARTBEAT_GAP", "heartbeat", None,
                             float(gap_ms), {})
        except Exception:
            logger.warning("[telemetry:%s] record_heartbeat_gap failed",
                           self.worker, exc_info=True)

    def record_boot(self, reason: str = "boot") -> None:
        """C3 — record THIS worker (re)boot → restart_events, deriving the prior
        run's uptime + the downtime from the DB itself (the prior boot's
        restart_events.ts and the last memory_samples.ts = last-alive). Combined
        with the op_events stall rows + memory_samples RssAnon trend already in
        the DB, this lets `analyze` correlate a stall→DISABLE→restart across the
        very reboot the stall caused. Called ONCE at worker boot. Never raises."""
        if not self._enabled:
            return
        try:
            prev_boot_ts, last_alive_ts = self._read_prev_run_marks()
            prev_uptime_s = 0.0
            detail = str(reason)
            if last_alive_ts > 0.0:
                if prev_boot_ts > 0.0:
                    prev_uptime_s = max(0.0, last_alive_ts - prev_boot_ts)
                downtime_s = max(0.0, time.time() - last_alive_ts)
                detail = (f"{reason} (prev_uptime={prev_uptime_s:.0f}s "
                          f"downtime={downtime_s:.0f}s)")
            self._ring.append(("restart", float(time.time()), detail,
                               float(prev_uptime_s)))
        except Exception:
            logger.warning("[telemetry:%s] record_boot failed", self.worker,
                           exc_info=True)

    def _read_prev_run_marks(self) -> tuple:
        """(prev_boot_ts, last_alive_ts) from the persisted DB — best-effort,
        (0.0, 0.0) on any failure / empty DB."""
        prev_boot_ts = 0.0
        last_alive_ts = 0.0
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT MAX(ts) FROM restart_events").fetchone()
                if row and row[0] is not None:
                    prev_boot_ts = float(row[0])
                row = conn.execute(
                    "SELECT MAX(ts) FROM memory_samples").fetchone()
                if row and row[0] is not None:
                    last_alive_ts = float(row[0])
            finally:
                conn.close()
        except Exception:
            pass
        return prev_boot_ts, last_alive_ts

    def uptime_s(self) -> float:
        """Seconds since this Telemetry (≈ this worker process) started."""
        return time.monotonic() - self._started_monotonic

    # ── C4: component-size sampling ─────────────────────────────────────────
    def set_size_provider(self, fn) -> None:
        """Register a best-effort callable returning a dict of component sizes
        (e.g. {"faiss_rows": N, "wiki_queue": M, "spine_nodes": K}). The flusher
        calls it every mem_sample_s and stores the dict as memory_samples
        .sizes_json. Called OFF the hot path; a faulty provider degrades to no
        sizes, never a worker fault (INV-TEL-5)."""
        self._size_provider = fn

    # ── C5: freeze-dump capture ─────────────────────────────────────────────
    def freeze_dump_file(self):
        """Open (once) `data/freeze_<worker>.txt` and return the file object the
        worker passes to `faulthandler.dump_traceback_later(file=…)`. The flusher
        ingests new dump text from it into `freeze_dumps`. Returns None on failure
        (caller then keeps the default stderr dump). Best-effort, never raises."""
        if not self._enabled:
            return None
        try:
            if self._freeze_dump_fp is None:
                path = os.path.join(
                    str(self._cfg["db_dir"]), f"freeze_{self.worker}.txt")
                # Capture any dump left by the PRIOR run (incl. the freeze that
                # killed it — the signal we most want) into a one-shot buffer,
                # then truncate so subsequent boots never re-ingest it.
                try:
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        with open(path, "r", errors="replace") as f:
                            self._boot_freeze_text = f.read().strip() or None
                except Exception:
                    self._boot_freeze_text = None
                self._freeze_dump_fp = open(path, "w", buffering=1)  # truncates
                self._freeze_dump_path = path
                self._freeze_dump_pos = 0
            return self._freeze_dump_fp
        except Exception:
            logger.warning("[telemetry:%s] freeze_dump_file open failed",
                           self.worker, exc_info=True)
            return None

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
                             self._sample_sizes_json()))
                    # C5 — persist a prior-run freeze (once) + any new live dump.
                    if self._boot_freeze_text:
                        conn.execute(
                            "INSERT INTO freeze_dumps(ts, stack_text) VALUES(?,?)",
                            (time.time(), "[prior-run]\n" + self._boot_freeze_text))
                        self._boot_freeze_text = None
                    self._ingest_freeze_dumps(conn)
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

    def _sample_sizes_json(self) -> Optional[str]:
        """C4 — capture the registered component sizes as JSON (off the hot
        path, in the flusher). None if no provider / empty / fault."""
        fn = self._size_provider
        if fn is None:
            return None
        try:
            sizes = fn()
            if sizes:
                return json.dumps(sizes, default=str)
        except Exception:
            logger.warning("[telemetry:%s] size provider failed", self.worker,
                           exc_info=True)
        return None

    def _ingest_freeze_dumps(self, conn: sqlite3.Connection) -> None:
        """C5 — read any new faulthandler dump text appended since the last
        flush and persist it as a freeze_dumps row. Best-effort."""
        path = self._freeze_dump_path
        if not path:
            return
        try:
            size = os.path.getsize(path)
            if size < self._freeze_dump_pos:   # truncated / rotated → restart
                self._freeze_dump_pos = 0
            if size <= self._freeze_dump_pos:
                return
            with open(path, "r", errors="replace") as f:
                f.seek(self._freeze_dump_pos)
                new_text = f.read()
                self._freeze_dump_pos = f.tell()
            new_text = new_text.strip()
            if new_text:
                conn.execute(
                    "INSERT INTO freeze_dumps(ts, stack_text) VALUES(?,?)",
                    (time.time(), new_text))
        except Exception:
            logger.warning("[telemetry:%s] freeze-dump ingest failed",
                           self.worker, exc_info=True)

    def _prune(self, conn: sqlite3.Connection) -> None:
        cutoff = time.time() - float(self._cfg["retention_days"]) * 86400.0
        conn.execute("DELETE FROM op_events WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM memory_samples WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM restart_events WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM freeze_dumps WHERE ts < ?", (cutoff,))

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
        try:
            if self._freeze_dump_fp is not None:
                self._freeze_dump_fp.close()
                self._freeze_dump_fp = None
        except Exception:
            pass


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


def get_active_telemetry() -> Optional[Telemetry]:
    """Return THIS process's single Telemetry instance, or None.

    Shared infra (e.g. `bus_socket._enrich_heartbeat`, the ONE heartbeat
    chokepoint in every process) uses this to attribute a heartbeat-gap to the
    heavy worker WITHOUT creating a writer: a worker process (synthesis/agno)
    has created exactly one instance; every other process has zero → None →
    no-op. NEVER creates an instance (that would open a spurious 2nd writer and
    break INV-TEL-4); only returns an already-created one."""
    with _INSTANCES_LOCK:
        if len(_INSTANCES) == 1:
            return next(iter(_INSTANCES.values()))
        return None
