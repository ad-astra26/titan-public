"""Per-process client library for the IMW service.

Callers use get_client().write(sql, params, table="...") instead of
raw sqlite3.connect().execute(...). The client handles:

- Routing based on config mode + per-table canonical list
- Journaling (data/run/imw_<pid>.jrn) before sending
- Synchronous sqlite-style API backed by an asyncio loop in a daemon thread
- Automatic reconnect + replay of journal tail on reconnect
- Fallback to direct safe_connect() write when IMW disabled

See titan-docs/PLAN_inner_memory_writer_service.md §9.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import IMWConfig
from .journal import CallerJournal
from .metrics import IMWMetrics
from .transport import TransportError, UnixSocketTransport
from .wire_format import (
    WireFormatError,
    WriteRequest,
    WriteResponse,
    decode_length,
    encode_frame,
)
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("titan.imw.client")

_TABLE_RE = re.compile(
    r"^\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|REPLACE\s+INTO)\s+['\"`]?(\w+)['\"`]?",
    re.IGNORECASE,
)


class WriterError(RuntimeError):
    """Base class for IMW client errors."""


class WriterDisabledError(WriterError):
    """Raised when IMW is disabled but code expects it to be active."""


@dataclass
class WriteResult:
    ok: bool
    rowcount: Optional[int] = None
    last_row_id: Optional[int] = None
    error: Optional[str] = None
    via: str = "direct"  # "direct" | "imw"
    target_db: str = "primary"


def detect_table(sql: str) -> Optional[str]:
    """Best-effort extraction of the primary table name from a SQL statement."""
    if not sql:
        return None
    m = _TABLE_RE.match(sql)
    return m.group(1) if m else None


class InnerMemoryWriterClient:
    """Per-process singleton client.

    All methods are thread-safe. Sync methods block on a dedicated asyncio
    loop running in a daemon thread.
    """

    def __init__(self, cfg: IMWConfig, caller_name: str = "") -> None:
        self._cfg = cfg
        self._caller = f"{caller_name or 'main'}:{os.getpid()}"
        self._journal: Optional[CallerJournal] = None
        self._transport: Optional[UnixSocketTransport] = None
        self._pending: dict[str, "asyncio.Future[WriteResponse]"] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._closed = False
        self._connect_lock: Optional[asyncio.Lock] = None

        if cfg.enabled and cfg.mode != "disabled":
            self._start_loop_thread()

    # ── loop thread management ───────────────────────────────────────

    def _start_loop_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop_main, name="imw.client_loop",
                                          daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise WriterError("IMW client loop thread failed to start")

    def _loop_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Create asyncio primitives inside this loop
        self._connect_lock = asyncio.Lock()
        try:
            self._loop.run_until_complete(self._init_and_run())
        except Exception as e:
            logger.error("[imw.client] loop crashed: %s", e, exc_info=True)
        finally:
            try:
                self._loop.close()
            except Exception as _swallow_exc:
                swallow_warn('[persistence.writer_client] InnerMemoryWriterClient._loop_main: self._loop.close()', _swallow_exc,
                             key='persistence.writer_client.InnerMemoryWriterClient._loop_main.line123', throttle=100)

    async def _init_and_run(self) -> None:
        # Open journal + replay any unacked tail before signaling ready.
        #
        # Journal file is named after the writer instance (derived from
        # cfg.socket_path stem) so multiple writer instances in the same
        # process don't share a journal file. Pre-fix bug: both
        # inner_memory_client (`imw.sock`) and observatory_writer_client
        # (`observatory_writer.sock`) used the literal `imw_<pid>.jrn`,
        # which made each daemon's `scan_orphan_journals(glob="imw_*.jrn")`
        # pick up the OTHER instance's journal on orphan replay → the
        # inner_memory daemon would try to replay observatory writes
        # against `inner_memory.db` and fail with "no such table:
        # vital_snapshots/trinity_snapshots/growth_snapshots". Fixed by
        # prefixing the journal filename with the socket basename so each
        # instance's journals are visible only to its own daemon.
        instance = Path(self._cfg.socket_path).stem or "imw"
        journal_path = Path(self._cfg.journal_dir) / f"{instance}_{os.getpid()}.jrn"
        self._journal = CallerJournal(str(journal_path), pid=os.getpid())
        await self._connect_with_retry()
        # Replay unacked records from this PID's journal (if any survived from a previous crash)
        await self._replay_own_journal()
        self._started.set()
        # Keep loop alive
        while not self._closed:
            await asyncio.sleep(0.5)

    async def _connect_with_retry(self) -> None:
        if self._connect_lock is None:
            return
        async with self._connect_lock:
            if self._transport is not None and self._transport.is_connected():
                return
            backoff = self._cfg.reconnect_backoff_min_ms / 1000.0
            max_backoff = self._cfg.reconnect_backoff_max_ms / 1000.0
            attempts = 0
            while not self._closed:
                try:
                    self._transport = UnixSocketTransport(
                        self._cfg.socket_path,
                        connect_timeout=self._cfg.connect_timeout_sec,
                    )
                    await self._transport.connect()
                    # Launch reader task
                    self._reader_task = asyncio.create_task(
                        self._read_loop(), name="imw.client.reader"
                    )
                    logger.info("[imw.client] connected to %s", self._cfg.socket_path)
                    return
                except (TransportError, Exception) as e:
                    attempts += 1
                    if attempts == 1 or attempts % 10 == 0:
                        logger.warning("[imw.client] connect failed (attempt %d): %s",
                                         attempts, e)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    body = await self._transport.recv_frame()
                except TransportError as e:
                    logger.warning("[imw.client] transport broken: %s; reconnecting", e)
                    await self._handle_disconnect()
                    return
                try:
                    resp = WriteResponse.from_msgpack(body)
                except (WireFormatError, KeyError) as e:
                    logger.warning("[imw.client] malformed response: %s", e)
                    continue
                fut = self._pending.pop(resp.req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(resp)
        except asyncio.CancelledError:
            raise

    async def _handle_disconnect(self) -> None:
        # Fail all pending futures so sync callers unblock
        for req_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(TransportError("connection lost"))
        self._pending.clear()
        try:
            await self._transport.close()
        except Exception as _swallow_exc:
            swallow_warn('[persistence.writer_client] InnerMemoryWriterClient._handle_disconnect: await self._transport.close()', _swallow_exc,
                         key='persistence.writer_client.InnerMemoryWriterClient._handle_disconnect.line196', throttle=100)
        self._transport = None
        # Reconnect in background
        asyncio.create_task(self._reconnect_and_replay())

    async def _reconnect_and_replay(self) -> None:
        await self._connect_with_retry()
        if not self._closed:
            await self._replay_own_journal()

    async def _replay_own_journal(self) -> None:
        if self._journal is None or self._transport is None:
            return
        replayed = 0
        for offset, flen, rec in self._journal.iter_unacked():
            req_id = rec.get("req_id")
            sql = rec.get("sql")
            params = rec.get("params")
            if not req_id or not sql:
                continue
            req = WriteRequest(
                req_id=req_id, caller=self._caller, op="write",
                sql=sql, params=params, sync=True,
                target_db="primary",  # journal records default to primary
            )
            fut: asyncio.Future[WriteResponse] = self._loop.create_future()
            self._pending[req.req_id] = fut
            try:
                await self._transport.send_frame(req.to_msgpack())
            except TransportError:
                # give up replay; reconnect will retry
                return
            try:
                resp = await asyncio.wait_for(fut, timeout=30.0)
                if resp.ok:
                    self._journal.ack(offset, flen)
                    replayed += 1
            except asyncio.TimeoutError:
                logger.warning("[imw.client] replay timeout for %s", req_id)
                return
        if replayed:
            logger.info("[imw.client] replayed %d unacked journal entries", replayed)

    # ── main write API ───────────────────────────────────────────────

    def write(self, sql: str, params: Any = (), *, table: Optional[str] = None,
              sync: bool = True, timeout: float = 60.0) -> WriteResult:
        """Primary write entry point. Thread-safe, synchronous.

        Routing:
        - If IMW disabled  → direct safe_connect write
        - mode=shadow      → direct write (canonical) + fire-and-forget IMW to shadow DB
        - mode=dual        → direct write + IMW to primary (caller sees direct result)
        - mode=canonical   → IMW if table in tables_canonical, else direct
        - mode=hybrid      → IMW if table in tables_canonical, else direct + shadow IMW
                             (the per-table cutover analog of canonical, but keeps
                             the Phase-1 shadow safety net for non-listed tables —
                             added 2026-04-26 for OBS-imw-phase3-self-insights canary)
        """
        if self._closed:
            raise WriterError("client closed")
        table_name = table or detect_table(sql)

        if not self._cfg.enabled or self._cfg.mode == "disabled":
            return self._route_direct(sql, params)

        if self._cfg.mode == "shadow":
            res = self._route_direct(sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] shadow fire-and-forget failed: %s", e)
            return res

        if self._cfg.mode == "dual":
            res = self._route_direct(sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="primary")
            except Exception as e:
                logger.warning("[imw.client] dual fire-and-forget failed: %s", e)
            return res

        if self._cfg.mode == "hybrid":
            if table_name and self._cfg.is_table_canonical(table_name):
                return self._route_imw(sql, params, target_db="primary",
                                         sync=sync, timeout=timeout)
            # non-canonical table: direct + shadow safety net
            res = self._route_direct(sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] hybrid shadow fire-and-forget failed: %s", e)
            return res

        # canonical mode: per-table decision
        if table_name and self._cfg.is_table_canonical(table_name):
            return self._route_imw(sql, params, target_db="primary",
                                     sync=sync, timeout=timeout)
        # table not yet canonical → direct
        return self._route_direct(sql, params)

    def write_many(self, sql: str, rows: list, *, table: Optional[str] = None,
                    sync: bool = True, timeout: float = 60.0) -> WriteResult:
        """Batched insert. Routed identically to write()."""
        if self._closed:
            raise WriterError("client closed")
        table_name = table or detect_table(sql)
        if not self._cfg.enabled or self._cfg.mode == "disabled":
            return self._route_direct_many(sql, rows)

        if self._cfg.mode in ("shadow", "dual"):
            res = self._route_direct_many(sql, rows)
            target = "shadow" if self._cfg.mode == "shadow" else "primary"
            try:
                self._fire_and_forget_imw_many(sql, rows, target_db=target)
            except Exception as e:
                logger.warning("[imw.client] %s many fire-and-forget failed: %s",
                                  self._cfg.mode, e)
            return res

        if self._cfg.mode == "hybrid":
            if table_name and self._cfg.is_table_canonical(table_name):
                return self._route_imw_many(sql, rows, target_db="primary",
                                              sync=sync, timeout=timeout)
            res = self._route_direct_many(sql, rows)
            try:
                self._fire_and_forget_imw_many(sql, rows, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] hybrid many shadow fire-and-forget failed: %s", e)
            return res

        if table_name and self._cfg.is_table_canonical(table_name):
            return self._route_imw_many(sql, rows, target_db="primary",
                                          sync=sync, timeout=timeout)
        return self._route_direct_many(sql, rows)

    async def awrite(self, sql: str, params: Any = (), *, table: Optional[str] = None,
                       sync: bool = True, timeout: float = 60.0) -> WriteResult:
        """Async variant of write() for callers already in an asyncio loop.

        Routes identically to sync write() but runs the direct-path sqlite
        operations in a threadpool executor (asyncio.to_thread) so the caller's
        event loop is not blocked. The IMW path is already async.
        """
        if self._closed:
            raise WriterError("client closed")
        table_name = table or detect_table(sql)

        if not self._cfg.enabled or self._cfg.mode == "disabled":
            return await asyncio.to_thread(self._route_direct, sql, params)

        if self._cfg.mode == "shadow":
            res = await asyncio.to_thread(self._route_direct, sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] shadow fire-and-forget failed: %s", e)
            return res

        if self._cfg.mode == "dual":
            res = await asyncio.to_thread(self._route_direct, sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="primary")
            except Exception as e:
                logger.warning("[imw.client] dual fire-and-forget failed: %s", e)
            return res

        if self._cfg.mode == "hybrid":
            if table_name and self._cfg.is_table_canonical(table_name):
                return await asyncio.to_thread(
                    self._route_imw, sql, params,
                    target_db="primary", sync=sync, timeout=timeout,
                )
            res = await asyncio.to_thread(self._route_direct, sql, params)
            try:
                self._fire_and_forget_imw(sql, params, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] hybrid shadow fire-and-forget failed: %s", e)
            return res

        # canonical mode: per-table decision
        if table_name and self._cfg.is_table_canonical(table_name):
            return await asyncio.to_thread(
                self._route_imw, sql, params,
                target_db="primary", sync=sync, timeout=timeout,
            )
        return await asyncio.to_thread(self._route_direct, sql, params)

    async def awrite_many(self, sql: str, rows: list, *, table: Optional[str] = None,
                            sync: bool = True, timeout: float = 60.0) -> WriteResult:
        """Async variant of write_many()."""
        if self._closed:
            raise WriterError("client closed")
        table_name = table or detect_table(sql)
        if not self._cfg.enabled or self._cfg.mode == "disabled":
            return await asyncio.to_thread(self._route_direct_many, sql, rows)

        if self._cfg.mode in ("shadow", "dual"):
            res = await asyncio.to_thread(self._route_direct_many, sql, rows)
            target = "shadow" if self._cfg.mode == "shadow" else "primary"
            try:
                self._fire_and_forget_imw_many(sql, rows, target_db=target)
            except Exception as e:
                logger.warning("[imw.client] %s many fire-and-forget failed: %s",
                                  self._cfg.mode, e)
            return res

        if self._cfg.mode == "hybrid":
            if table_name and self._cfg.is_table_canonical(table_name):
                return await asyncio.to_thread(
                    self._route_imw_many, sql, rows,
                    target_db="primary", sync=sync, timeout=timeout,
                )
            res = await asyncio.to_thread(self._route_direct_many, sql, rows)
            try:
                self._fire_and_forget_imw_many(sql, rows, target_db="shadow")
            except Exception as e:
                logger.warning("[imw.client] hybrid many shadow fire-and-forget failed: %s", e)
            return res

        if table_name and self._cfg.is_table_canonical(table_name):
            return await asyncio.to_thread(
                self._route_imw_many, sql, rows,
                target_db="primary", sync=sync, timeout=timeout,
            )
        return await asyncio.to_thread(self._route_direct_many, sql, rows)

    def ping(self, timeout: float = 5.0) -> bool:
        if not self._cfg.enabled:
            return False
        req = WriteRequest.new_ping(caller=self._caller)
        try:
            resp = self._submit_and_wait(req, timeout=timeout)
            return resp.ok
        except Exception:
            return False

    def flush(self, timeout: float = 30.0) -> None:
        """Wait until all in-flight writes are acknowledged."""
        if not self._cfg.enabled:
            return
        deadline = time.time() + timeout
        while self._pending and time.time() < deadline:
            time.sleep(0.05)

    # ── direct path (legacy / fallback) ──────────────────────────────

    def _route_direct(self, sql: str, params: Any) -> WriteResult:
        try:
            conn = sqlite3.connect(self._cfg.db_path, timeout=self._cfg.busy_timeout_sec)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                cur = conn.execute(sql, params or ())
                conn.commit()
                return WriteResult(ok=True, rowcount=cur.rowcount,
                                     last_row_id=cur.lastrowid, via="direct")
            finally:
                conn.close()
        except sqlite3.Error as e:
            return WriteResult(ok=False, error=str(e), via="direct")

    def _route_direct_many(self, sql: str, rows: list) -> WriteResult:
        try:
            conn = sqlite3.connect(self._cfg.db_path, timeout=self._cfg.busy_timeout_sec)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                cur = conn.executemany(sql, rows)
                conn.commit()
                return WriteResult(ok=True, rowcount=cur.rowcount,
                                     last_row_id=cur.lastrowid, via="direct")
            finally:
                conn.close()
        except sqlite3.Error as e:
            return WriteResult(ok=False, error=str(e), via="direct")

    # ── IMW path ─────────────────────────────────────────────────────

    def _route_imw(self, sql: str, params: Any, *, target_db: str,
                    sync: bool, timeout: float) -> WriteResult:
        req = WriteRequest.new_write(caller=self._caller, sql=sql, params=params,
                                        sync=sync, target_db=target_db)
        return self._submit_writing(req, timeout=timeout, target_db=target_db)

    def _route_imw_many(self, sql: str, rows: list, *, target_db: str,
                          sync: bool, timeout: float) -> WriteResult:
        req = WriteRequest.new_writemany(caller=self._caller, sql=sql, rows=rows,
                                            sync=sync, target_db=target_db)
        return self._submit_writing(req, timeout=timeout, target_db=target_db)

    def _submit_writing(self, req: WriteRequest, *, timeout: float,
                          target_db: str) -> WriteResult:
        # Only journal PRIMARY writes — shadow replays aren't worth journaling
        if target_db == "primary" and self._journal is not None:
            try:
                # fsync-on-disk happens inside append()
                offset = self._journal.append(req.req_id, req.sql, req.params)
            except Exception as e:
                return WriteResult(ok=False, error=f"journal append: {e}",
                                     via="imw", target_db=target_db)
        else:
            offset = None
        try:
            resp = self._submit_and_wait(req, timeout=timeout)
        except Exception as e:
            # transport/timeout — caller blocks on disconnect handler's replay
            return WriteResult(ok=False, error=str(e), via="imw", target_db=target_db)
        if resp.ok and offset is not None:
            # ack journal entry (flen = 4 header + msgpack body)
            import msgpack
            frame_len = 4 + len(msgpack.packb({
                "req_id": req.req_id, "sql": req.sql, "params": req.params,
                "journaled_at": req.ts,
            }, use_bin_type=True))
            self._journal.ack(offset, frame_len)
        err = None if resp.ok else (resp.error or {}).get("msg", "unknown")
        return WriteResult(ok=resp.ok, rowcount=resp.rowcount,
                             last_row_id=resp.last_row_id, error=err,
                             via="imw", target_db=target_db)

    def _fire_and_forget_imw(self, sql: str, params: Any, *, target_db: str) -> None:
        """Fire a dual/shadow write without blocking the caller on its ACK."""
        req = WriteRequest.new_write(caller=self._caller, sql=sql, params=params,
                                        sync=False, target_db=target_db)
        self._submit_no_wait(req)

    def _fire_and_forget_imw_many(self, sql: str, rows: list, *, target_db: str) -> None:
        req = WriteRequest.new_writemany(caller=self._caller, sql=sql, rows=rows,
                                            sync=False, target_db=target_db)
        self._submit_no_wait(req)

    def _submit_no_wait(self, req: WriteRequest) -> None:
        if self._loop is None or self._transport is None:
            return
        async def _send():
            try:
                fut: asyncio.Future[WriteResponse] = self._loop.create_future()
                self._pending[req.req_id] = fut
                await self._transport.send_frame(req.to_msgpack())
                # drop the future — we don't await it. The reader will still drain.
            except Exception as e:
                swallow_warn('[imw.client] fire-and-forget send failed', e,
                             key="persistence.writer_client.fire_and_forget_send_failed", throttle=100)
                self._pending.pop(req.req_id, None)
        asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def _submit_and_wait(self, req: WriteRequest, timeout: float) -> WriteResponse:
        if self._loop is None:
            raise WriterError("loop not running")
        # Ensure connected (may trigger reconnect)
        async def _ensure_and_send():
            if self._transport is None or not self._transport.is_connected():
                await self._connect_with_retry()
            fut: asyncio.Future[WriteResponse] = self._loop.create_future()
            self._pending[req.req_id] = fut
            await self._transport.send_frame(req.to_msgpack())
            return await asyncio.wait_for(fut, timeout=timeout)

        cf = asyncio.run_coroutine_threadsafe(_ensure_and_send(), self._loop)
        try:
            return cf.result(timeout=timeout + 1.0)
        except Exception:
            self._pending.pop(req.req_id, None)
            raise

    # ── shutdown ─────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            async def _stop():
                if self._transport is not None:
                    await self._transport.close()
                if self._journal is not None:
                    self._journal.close()
            fut = asyncio.run_coroutine_threadsafe(_stop(), self._loop)
            try:
                fut.result(timeout=3.0)
            except Exception as _swallow_exc:
                swallow_warn('[persistence.writer_client] InnerMemoryWriterClient.close: fut.result(timeout=3.0)', _swallow_exc,
                             key='persistence.writer_client.InnerMemoryWriterClient.close.line525', throttle=100)
            self._loop.call_soon_threadsafe(self._loop.stop)


# ── module-level singleton ───────────────────────────────────────────

_client: Optional[InnerMemoryWriterClient] = None
_client_lock = threading.Lock()


def get_client(caller_name: str = "") -> InnerMemoryWriterClient:
    global _client
    with _client_lock:
        if _client is None:
            cfg = IMWConfig.from_titan_config()
            _client = InnerMemoryWriterClient(cfg, caller_name=caller_name)
            atexit.register(_atexit_close)
        return _client


def reset_client() -> None:
    """Testing hook: tear down the singleton so a new one can be built."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception as _swallow_exc:
                swallow_warn('[persistence.writer_client] reset_client: _client.close()', _swallow_exc,
                             key='persistence.writer_client.reset_client.line553', throttle=100)
            _client = None


def _atexit_close() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception as _swallow_exc:
            swallow_warn('[persistence.writer_client] _atexit_close: _client.close()', _swallow_exc,
                         key='persistence.writer_client._atexit_close.line563', throttle=100)
