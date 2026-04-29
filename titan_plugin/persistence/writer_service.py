"""IMW daemon — single-writer asyncio service.

Runs as a Guardian-supervised module. Owns the ONLY write connection to
data/inner_memory.db. Accepts framed msgpack requests over a unix domain
socket, journals via service_wal, group-commits batches to the DB, and
responds to callers with ACK/NAK.

See titan-docs/PLAN_inner_memory_writer_service.md §8 for full design.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .config import IMWConfig
from .journal import CallerJournal, scan_orphan_journals
from .metrics import IMWMetrics
from .service_wal import ServiceWAL
from .wire_format import (
    MAX_FRAME_BYTES,
    WireFormatError,
    WriteRequest,
    WriteResponse,
    _LEN_HEADER,
    encode_frame,
)
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("titan.imw.service")


class _PendingRequest:
    __slots__ = ("req", "wal_offset", "ack_queue", "rowcount", "last_row_id", "err")

    def __init__(self, req: WriteRequest, wal_offset: int, ack_queue: "asyncio.Queue[WriteResponse]") -> None:
        self.req = req
        self.wal_offset = wal_offset
        self.ack_queue = ack_queue
        self.rowcount: Optional[int] = None
        self.last_row_id: Optional[int] = None
        self.err: Optional[dict] = None


class IMWDaemon:
    """Asyncio service: accept connections, batch-commit to SQLite, ACK to callers."""

    def __init__(self, config: IMWConfig) -> None:
        self._cfg = config
        self._cfg.ensure_runtime_dirs()
        self._metrics = IMWMetrics()
        self._server: Optional[asyncio.base_events.Server] = None
        self._wal: Optional[ServiceWAL] = None
        self._conn: Optional[sqlite3.Connection] = None       # primary
        self._shadow_conn: Optional[sqlite3.Connection] = None
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._seen_req_ids: set[str] = set()
        self._commit_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_committed_wal_offset = 0
        # dedup cache size — keep last 50k req_ids
        self._dedup_cap = 50_000

    def _get_conn(self, target_db: str) -> sqlite3.Connection:
        if target_db == "shadow" and self._shadow_conn is not None:
            return self._shadow_conn
        return self._conn  # primary (default)

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        self._wal = ServiceWAL(self._cfg.wal_path, max_mb=self._cfg.service_wal_max_mb)
        self._conn = self._open_db(self._cfg.db_path)
        # Optional shadow connection — needed by any mode that fires shadow
        # writes (Phase 1 shadow mode + Phase 3 hybrid mode for non-canonical
        # tables). Without this, target_db="shadow" requests silently fall
        # back to the primary connection in _get_conn(), corrupting primary
        # with duplicate rows.
        if self._cfg.mode in ("shadow", "hybrid") and self._cfg.shadow_db_path:
            self._shadow_conn = self._open_db(self._cfg.shadow_db_path)
            logger.info("[imw] %s mode: secondary DB at %s",
                        self._cfg.mode, self._cfg.shadow_db_path)
            # Replicate primary schema → shadow so shadow writes land against
            # the right tables. Without this, every shadow INSERT fails with
            # "no such table: X" until someone runs the DDL manually.
            # Learned the hard way 2026-04-20 on the first shadow-mode boot.
            self._sync_shadow_schema()
        await self._replay_service_wal()
        await self._replay_orphan_caller_journals()
        # Bind unix socket — remove stale if present
        sock_path = Path(self._cfg.socket_path)
        if sock_path.exists():
            sock_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self._cfg.socket_path,
        )
        try:
            os.chmod(self._cfg.socket_path, 0o660)
        except OSError as _swallow_exc:
            swallow_warn('[persistence.writer_service] IMWDaemon.start: os.chmod(self._cfg.socket_path, 432)', _swallow_exc,
                         key='persistence.writer_service.IMWDaemon.start.line101', throttle=100)
        self._commit_task = asyncio.create_task(self._commit_loop(), name="imw.commit_loop")
        logger.info(
            "[imw] booted: socket=%s wal=%s db=%s batch=%dms/%d",
            self._cfg.socket_path, self._cfg.wal_path, self._cfg.db_path,
            self._cfg.batch_window_ms, self._cfg.max_batch_size,
        )

    def _sync_shadow_schema(self) -> None:
        """Copy all CREATE TABLE/INDEX statements from primary to shadow.

        Idempotent — uses IF NOT EXISTS. Only touches schema, not data.
        """
        if self._conn is None or self._shadow_conn is None:
            return
        schema_rows = self._conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL ORDER BY type DESC"
        ).fetchall()
        created = 0
        skipped = 0
        for _type, name, sql in schema_rows:
            if not sql:
                continue
            # Force IF NOT EXISTS for idempotence (safe re-run on daemon restart)
            sql_safe = sql
            if "IF NOT EXISTS" not in sql_safe.upper():
                for kw in ("CREATE TABLE ", "CREATE UNIQUE INDEX ", "CREATE INDEX "):
                    if sql_safe.startswith(kw):
                        sql_safe = kw + "IF NOT EXISTS " + sql_safe[len(kw):]
                        break
            try:
                self._shadow_conn.execute(sql_safe)
                created += 1
            except sqlite3.Error as e:
                swallow_warn(f'[imw] shadow schema skip {_type} {name}', e,
                             key="persistence.writer_service.shadow_schema_skip", throttle=100)
                skipped += 1
        logger.info("[imw] shadow schema sync: %d objects created, %d skipped",
                      created, skipped)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._commit_task is not None:
            try:
                await asyncio.wait_for(self._commit_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._commit_task.cancel()
        for c in (self._conn, self._shadow_conn):
            if c is not None:
                try:
                    c.close()
                except Exception as _swallow_exc:
                    swallow_warn('[persistence.writer_service] IMWDaemon.stop: c.close()', _swallow_exc,
                                 key='persistence.writer_service.IMWDaemon.stop.line158', throttle=100)
        if self._wal is not None:
            self._wal.close()
        try:
            sp = Path(self._cfg.socket_path)
            if sp.exists():
                sp.unlink()
        except OSError as _swallow_exc:
            swallow_warn('[persistence.writer_service] IMWDaemon.stop: sp = Path(self._cfg.socket_path)', _swallow_exc,
                         key='persistence.writer_service.IMWDaemon.stop.line166', throttle=100)

    async def serve_forever(self) -> None:
        await self._stop_event.wait()

    def metrics_snapshot(self) -> dict:
        if self._wal is not None:
            self._metrics.set_wal_size_mb(self._wal.size_mb())
        self._metrics.set_queue_depth(self._queue.qsize())
        self._metrics.set_tables_canonical(self._cfg.tables_canonical)
        return self._metrics.snapshot()

    # ── db connection ────────────────────────────────────────────────

    def _open_db(self, db_path: str) -> sqlite3.Connection:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            db_path,
            timeout=self._cfg.busy_timeout_sec,
            isolation_level=None,   # manage transactions manually
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-16000")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        return conn

    # ── replay on boot ───────────────────────────────────────────────

    async def _replay_service_wal(self) -> None:
        if self._wal is None or self._conn is None:
            return
        count = 0
        for offset, rec in self._wal.iter_uncommitted():
            req_id = rec.get("req_id")
            sql = rec.get("sql")
            params = rec.get("params")
            target_db = rec.get("target_db", "primary")
            if not req_id or not sql:
                continue
            if req_id in self._seen_req_ids:
                continue
            conn = self._get_conn(target_db)
            if conn is None:
                # e.g., shadow record but no shadow_conn configured this boot
                continue
            try:
                conn.execute("BEGIN")
                conn.execute(sql, params or ())
                conn.execute("COMMIT")
                self._seen_req_ids.add(req_id)
                count += 1
            except sqlite3.Error as e:
                try:
                    conn.execute("ROLLBACK")
                except Exception as _swallow_exc:
                    swallow_warn("[persistence.writer_service] IMWDaemon._replay_service_wal: conn.execute('ROLLBACK')", _swallow_exc,
                                 key='persistence.writer_service.IMWDaemon._replay_service_wal.line223', throttle=100)
                logger.warning("[imw] WAL replay of %s failed: %s", req_id, e)
        if count:
            logger.info("[imw] service-WAL replay applied %d uncommitted writes", count)
            self._wal.checkpoint(self._wal._last_ckpt_offset)

    async def _replay_orphan_caller_journals(self) -> None:
        if self._conn is None:
            return
        # Match the writer client's per-instance journal naming: each daemon
        # only scans journals owned by its own writer client (derived from
        # the same cfg.socket_path stem). See journal.scan_orphan_journals
        # for the rationale.
        instance = Path(self._cfg.socket_path).stem or "imw"
        for path, pid in scan_orphan_journals(
            self._cfg.journal_dir, instance_prefix=instance):
            try:
                jnl = CallerJournal(str(path), pid=pid)
                replayed = 0
                for _offset, _flen, rec in jnl.iter_unacked():
                    req_id = rec.get("req_id")
                    sql = rec.get("sql")
                    params = rec.get("params")
                    if not req_id or not sql:
                        continue
                    if req_id in self._seen_req_ids:
                        continue
                    try:
                        self._conn.execute("BEGIN")
                        self._conn.execute(sql, params or ())
                        self._conn.execute("COMMIT")
                        self._seen_req_ids.add(req_id)
                        replayed += 1
                    except sqlite3.Error as e:
                        try:
                            self._conn.execute("ROLLBACK")
                        except Exception as _swallow_exc:
                            swallow_warn("[persistence.writer_service] IMWDaemon._replay_orphan_caller_journals: self._conn.execute('ROLLBACK')", _swallow_exc,
                                         key='persistence.writer_service.IMWDaemon._replay_orphan_caller_journals.line254', throttle=100)
                        logger.warning("[imw] orphan %s replay %s failed: %s", path.name, req_id, e)
                jnl.close()
                if replayed:
                    self._metrics.incr_journal_replay(replayed)
                    logger.info("[imw] replayed %d orphan writes from pid=%d", replayed, pid)
                # delete orphan file after successful replay
                try:
                    path.unlink()
                except OSError as e:
                    logger.warning("[imw] could not delete %s: %s", path, e)
            except Exception as e:
                logger.warning("[imw] orphan replay of %s failed: %s", path, e)

    # ── connection handler ──────────────────────────────────────────

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "<unix>"
        self._metrics.incr_connections()
        ack_queue: asyncio.Queue[WriteResponse] = asyncio.Queue()
        sender_task = asyncio.create_task(
            self._send_responses(writer, ack_queue), name="imw.sender"
        )
        try:
            while True:
                try:
                    head = await reader.readexactly(4)
                except asyncio.IncompleteReadError:
                    break
                (n,) = _LEN_HEADER.unpack(head)
                if n <= 0 or n > MAX_FRAME_BYTES:
                    logger.warning("[imw] bad frame length %d from %r", n, peer)
                    break
                try:
                    body = await reader.readexactly(n)
                except asyncio.IncompleteReadError:
                    break
                try:
                    req = WriteRequest.from_msgpack(body)
                except (WireFormatError, KeyError, TypeError, ValueError) as e:
                    logger.warning("[imw] unparseable request from %r: %s", peer, e)
                    continue

                await self._handle_request(req, ack_queue)
        finally:
            # allow sender to drain queued responses
            await ack_queue.put(None)  # type: ignore
            try:
                await asyncio.wait_for(sender_task, timeout=2.0)
            except asyncio.TimeoutError:
                sender_task.cancel()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as _swallow_exc:
                swallow_warn('[persistence.writer_service] IMWDaemon._handle_connection: writer.close()', _swallow_exc,
                             key='persistence.writer_service.IMWDaemon._handle_connection.line313', throttle=100)
            self._metrics.decr_connections()

    async def _send_responses(
        self,
        writer: asyncio.StreamWriter,
        ack_queue: "asyncio.Queue[Optional[WriteResponse]]",
    ) -> None:
        try:
            while True:
                resp = await ack_queue.get()
                if resp is None:
                    return
                try:
                    writer.write(encode_frame(resp.to_msgpack()))
                    await writer.drain()
                except (ConnectionError, OSError) as e:
                    swallow_warn('[imw] send_responses write failed', e,
                                 key="persistence.writer_service.send_responses_write_failed", throttle=100)
                    return
        except asyncio.CancelledError:
            raise

    async def _handle_request(
        self,
        req: WriteRequest,
        ack_queue: "asyncio.Queue[Optional[WriteResponse]]",
    ) -> None:
        if req.op == "ping":
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=True, committed_at=time.time()
            ))
            return
        if req.op == "flush":
            # force-flush: drain queue. Simplest: enqueue a sentinel we wait on.
            # For now, the ACK is immediate — caller awaits their own writes' acks.
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=True, committed_at=time.time()
            ))
            return
        if req.op not in ("write", "writemany"):
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=False,
                error={"type": "BadOp", "msg": f"unknown op {req.op!r}"}
            ))
            return

        if req.req_id in self._seen_req_ids:
            # idempotent: already processed (e.g., journal replay)
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=True, committed_at=time.time()
            ))
            return

        if req.sql is None:
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=False,
                error={"type": "BadRequest", "msg": "sql is required"}
            ))
            return

        # Journal to service WAL before committing
        try:
            wal_offset = self._wal.append_request(req.req_id, req.sql, req.params, req.target_db)
        except Exception as e:
            logger.error("[imw] service WAL append failed: %s", e)
            self._metrics.incr_errors(f"wal_append: {e}")
            await ack_queue.put(WriteResponse(
                req_id=req.req_id, ok=False,
                error={"type": "WALError", "msg": str(e)}
            ))
            return

        self._queue.put_nowait(_PendingRequest(req, wal_offset, ack_queue))
        self._metrics.set_queue_depth(self._queue.qsize())

    # ── commit loop ──────────────────────────────────────────────────

    async def _commit_loop(self) -> None:
        batch_window = self._cfg.batch_window_ms / 1000.0
        max_batch = self._cfg.max_batch_size
        while not self._stop_event.is_set():
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=batch_window)
            except asyncio.TimeoutError:
                continue
            batch: list[_PendingRequest] = [first]
            # Fast-path: if alone and fast-path enabled, commit immediately
            # Otherwise, drain up to max_batch from queue
            if self._cfg.fast_path_enabled and self._queue.empty():
                pass  # commit single write
            else:
                while len(batch) < max_batch:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            self._metrics.set_queue_depth(self._queue.qsize())
            self._metrics.set_in_flight(len(batch))
            t0 = time.time()
            await self._commit_batch(batch)
            dt_ms = (time.time() - t0) * 1000.0
            self._metrics.record_commit(len(batch), dt_ms)
            self._metrics.incr_batches()
            self._metrics.set_in_flight(0)

    async def _commit_batch(self, batch: list[_PendingRequest]) -> None:
        """Try transactional commit; on failure, fall back to per-write."""
        # Pre-skip req_ids we've already applied (dedup)
        fresh = [p for p in batch if p.req.req_id not in self._seen_req_ids]
        already = [p for p in batch if p.req.req_id in self._seen_req_ids]
        for p in already:
            await p.ack_queue.put(WriteResponse(
                req_id=p.req.req_id, ok=True, committed_at=time.time()
            ))
        if not fresh:
            return

        # Group by target_db so we can BEGIN/COMMIT per connection
        by_target: dict = {}
        for p in fresh:
            by_target.setdefault(p.req.target_db, []).append(p)

        try:
            commit_ts = None
            for target, group in by_target.items():
                conn = self._get_conn(target)
                conn.execute("BEGIN")
                for p in group:
                    try:
                        if p.req.op == "writemany":
                            cur = conn.executemany(p.req.sql, p.req.params or [])
                        else:
                            cur = conn.execute(p.req.sql, p.req.params or ())
                        p.rowcount = cur.rowcount
                        p.last_row_id = cur.lastrowid
                    except sqlite3.Error as e:
                        # Any error in this group's transaction
                        raise _BatchError(p, e)
                conn.execute("COMMIT")
            commit_ts = time.time()
            # Record req_ids + ACK
            for p in fresh:
                self._seen_req_ids.add(p.req.req_id)
                await p.ack_queue.put(WriteResponse(
                    req_id=p.req.req_id, ok=True,
                    rowcount=p.rowcount, last_row_id=p.last_row_id,
                    committed_at=commit_ts,
                ))
            self._metrics.incr_writes(len(fresh))
            self._last_committed_wal_offset = max(p.wal_offset for p in fresh)
            self._wal.checkpoint(self._last_committed_wal_offset)
            self._trim_dedup_cache()
        except _BatchError:
            # Roll back anything in-flight on both connections and retry per-write
            for c in (self._conn, self._shadow_conn):
                if c is not None:
                    try:
                        c.execute("ROLLBACK")
                    except Exception as _swallow_exc:
                        swallow_warn("[persistence.writer_service] IMWDaemon._commit_batch: c.execute('ROLLBACK')", _swallow_exc,
                                     key='persistence.writer_service.IMWDaemon._commit_batch.line473', throttle=100)
            self._metrics.incr_per_write_fallback()
            logger.warning("[imw] batch failed; per-write fallback for %d requests", len(fresh))
            await self._commit_per_write(fresh)
        except sqlite3.Error as e:
            # Unexpected (e.g., BEGIN itself failed)
            for c in (self._conn, self._shadow_conn):
                if c is not None:
                    try:
                        c.execute("ROLLBACK")
                    except Exception as _swallow_exc:
                        swallow_warn("[persistence.writer_service] IMWDaemon._commit_batch: c.execute('ROLLBACK')", _swallow_exc,
                                     key='persistence.writer_service.IMWDaemon._commit_batch.line484', throttle=100)
            logger.error("[imw] batch outer error: %s", e)
            self._metrics.incr_errors(f"batch_outer: {e}")
            for p in fresh:
                await p.ack_queue.put(WriteResponse(
                    req_id=p.req.req_id, ok=False,
                    error={"type": type(e).__name__, "msg": str(e)},
                ))

    async def _commit_per_write(self, requests: list[_PendingRequest]) -> None:
        for p in requests:
            conn = self._get_conn(p.req.target_db)
            try:
                conn.execute("BEGIN")
                if p.req.op == "writemany":
                    cur = conn.executemany(p.req.sql, p.req.params or [])
                else:
                    cur = conn.execute(p.req.sql, p.req.params or ())
                conn.execute("COMMIT")
                self._seen_req_ids.add(p.req.req_id)
                await p.ack_queue.put(WriteResponse(
                    req_id=p.req.req_id, ok=True,
                    rowcount=cur.rowcount, last_row_id=cur.lastrowid,
                    committed_at=time.time(),
                ))
                self._metrics.incr_writes(1)
                self._last_committed_wal_offset = max(self._last_committed_wal_offset, p.wal_offset)
            except sqlite3.Error as e:
                try:
                    conn.execute("ROLLBACK")
                except Exception as _swallow_exc:
                    swallow_warn("[persistence.writer_service] IMWDaemon._commit_per_write: conn.execute('ROLLBACK')", _swallow_exc,
                                 key='persistence.writer_service.IMWDaemon._commit_per_write.line515', throttle=100)
                self._metrics.incr_errors(f"per_write: {e}")
                await p.ack_queue.put(WriteResponse(
                    req_id=p.req.req_id, ok=False,
                    error={"type": type(e).__name__, "msg": str(e)},
                ))
        # Checkpoint on whatever we did commit
        if self._last_committed_wal_offset:
            self._wal.checkpoint(self._last_committed_wal_offset)
        self._trim_dedup_cache()

    def _trim_dedup_cache(self) -> None:
        if len(self._seen_req_ids) > self._dedup_cap * 2:
            # Keep most recent half — set doesn't preserve insertion order in py<3.7;
            # for our usage, dropping arbitrary half is acceptable (replay re-adds).
            overflow = len(self._seen_req_ids) - self._dedup_cap
            to_drop = set(list(self._seen_req_ids)[:overflow])
            self._seen_req_ids -= to_drop


class _BatchError(Exception):
    """Internal: signals a batch failure so caller can roll back + retry per-write."""

    def __init__(self, pending: _PendingRequest, err: sqlite3.Error) -> None:
        self.pending = pending
        self.err = err


# Generic alias — preferred name for new code.
# Per rFP_universal_sqlite_writer Phase 4 (2026-04-27).
SqliteWriterDaemon = IMWDaemon


async def _run_daemon(cfg: IMWConfig, stop_event: asyncio.Event) -> None:
    daemon = IMWDaemon(cfg)
    await daemon.start()
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
    finally:
        await daemon.stop()


def run_service(cfg: IMWConfig) -> None:
    """Blocking entry point — runs asyncio event loop until interrupt."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _signal_stop(signum, frame):  # noqa: ARG001
        logger.info("[imw] signal %s — stopping", signum)
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_stop)
        except (ValueError, OSError) as _swallow_exc:
            swallow_warn('[persistence.writer_service] run_service: signal.signal(sig, _signal_stop)', _swallow_exc,
                         key='persistence.writer_service.run_service.line567', throttle=100)

    try:
        loop.run_until_complete(_run_daemon(cfg, stop_event))
    finally:
        loop.close()
