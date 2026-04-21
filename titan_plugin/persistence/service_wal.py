"""IMW service-side WAL — append every accepted request BEFORE it is committed to the DB.

Purpose: if the service crashes between receiving a request and committing it to
inner_memory.db, the next boot replays the WAL tail so no acknowledged-by-write
request is ever lost.

Semantics:
- Each request appended with fsync BEFORE the service commits to the DB.
- After a successful DB commit, the requests in that batch are marked "committed"
  by writing a checkpoint record (a marker frame containing last_committed_offset).
- On service boot: read WAL, find last checkpoint, replay everything after.

File format:
    [4-byte length][msgpack payload]
    where payload is one of:
        {"k": "req", "req_id": str, "sql": str, "params": list, "ts": float}
        {"k": "ckpt", "offset": int, "ts": float}

The WAL is truncated to the last checkpoint on clean shutdown. Rotation at max_mb.
"""
from __future__ import annotations

import logging
import os
import struct
import time
from pathlib import Path
from threading import RLock
from typing import Iterator

import msgpack

from .wire_format import MAX_FRAME_BYTES, _LEN_HEADER

logger = logging.getLogger("titan.imw.wal")


class ServiceWALError(RuntimeError):
    """Raised on service WAL integrity / write failures."""


class ServiceWAL:
    def __init__(self, path: str, max_mb: int = 64) -> None:
        self._path = Path(path)
        self._max_bytes = max_mb * 1024 * 1024
        self._lock = RLock()
        self._last_ckpt_offset = 0
        self._file = None
        self._fd = -1
        self._closed = False
        self._open_or_create()

    def _open_or_create(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists() or self._path.stat().st_size == 0
        self._file = open(self._path, "ab+")
        self._fd = self._file.fileno()
        if new_file:
            # Start with no checkpoint (0 = beginning)
            self._last_ckpt_offset = 0
        else:
            self._scan_last_checkpoint()
        self._file.seek(0, os.SEEK_END)

    def _scan_last_checkpoint(self) -> None:
        """Read from start, find offset of the last checkpoint record."""
        with open(self._path, "rb") as f:
            cur = 0
            last_ckpt_end = 0
            size = self._path.stat().st_size
            while cur < size:
                f.seek(cur)
                head = f.read(4)
                if len(head) < 4:
                    break
                (n,) = _LEN_HEADER.unpack(head)
                if n <= 0 or n > MAX_FRAME_BYTES:
                    logger.warning("[imw.wal] bad frame length %d at offset %d", n, cur)
                    break
                body = f.read(n)
                if len(body) < n:
                    break
                try:
                    rec = msgpack.unpackb(body, raw=False)
                except Exception:
                    break
                if isinstance(rec, dict) and rec.get("k") == "ckpt":
                    last_ckpt_end = cur + 4 + n
                cur += 4 + n
            self._last_ckpt_offset = last_ckpt_end

    # ── write API ────────────────────────────────────────────────────

    def append_request(self, req_id: str, sql: str, params, target_db: str = "primary") -> int:
        """Append a request record and fsync. Returns byte offset of the record."""
        with self._lock:
            if self._closed:
                raise ServiceWALError("WAL is closed")
            record = msgpack.packb({
                "k": "req",
                "req_id": req_id,
                "sql": sql,
                "params": params,
                "target_db": target_db,
                "ts": time.time(),
            }, use_bin_type=True)
            frame = _LEN_HEADER.pack(len(record)) + record
            offset = self._file.tell()
            self._file.write(frame)
            self._file.flush()
            os.fsync(self._fd)
            return offset

    def checkpoint(self, committed_up_to_offset: int) -> None:
        """Write a checkpoint marker AFTER a successful DB commit."""
        with self._lock:
            if self._closed:
                return
            record = msgpack.packb({
                "k": "ckpt",
                "offset": committed_up_to_offset,
                "ts": time.time(),
            }, use_bin_type=True)
            frame = _LEN_HEADER.pack(len(record)) + record
            offset = self._file.tell()
            self._file.write(frame)
            self._file.flush()
            os.fsync(self._fd)
            self._last_ckpt_offset = offset + len(frame)
            # Rotation check
            if self._last_ckpt_offset > self._max_bytes:
                self._truncate_past_checkpoint()

    def _truncate_past_checkpoint(self) -> None:
        """Safe-rotate: after a checkpoint we can compact by starting fresh
        (anything before the checkpoint is committed to DB, so it's safe to drop)."""
        with self._lock:
            try:
                self._file.close()
                # Create a temp file with just a fresh start (no records yet)
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, self._path)
                self._file = open(self._path, "ab+")
                self._fd = self._file.fileno()
                self._last_ckpt_offset = 0
                logger.info("[imw.wal] truncated past checkpoint — fresh WAL")
            except OSError as e:
                logger.error("[imw.wal] truncate failed: %s", e)

    # ── replay on boot ───────────────────────────────────────────────

    def iter_uncommitted(self) -> Iterator[tuple]:
        """Yield (offset, rec_dict) for all request records after the last checkpoint."""
        with self._lock:
            self._file.flush()
            path = self._path
            ckpt = self._last_ckpt_offset
            size = path.stat().st_size
        with open(path, "rb") as f:
            cur = ckpt
            while cur < size:
                f.seek(cur)
                head = f.read(4)
                if len(head) < 4:
                    break
                (n,) = _LEN_HEADER.unpack(head)
                if n <= 0 or n > MAX_FRAME_BYTES:
                    break
                body = f.read(n)
                if len(body) < n:
                    break
                try:
                    rec = msgpack.unpackb(body, raw=False)
                except Exception:
                    break
                if isinstance(rec, dict) and rec.get("k") == "req":
                    yield (cur, rec)
                cur += 4 + n

    def size_mb(self) -> float:
        try:
            return self._path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._file is not None:
                try:
                    self._file.flush()
                    os.fsync(self._fd)
                except OSError:
                    pass
                self._file.close()
            self._closed = True

    def __enter__(self) -> "ServiceWAL":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
