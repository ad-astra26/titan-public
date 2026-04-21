"""Per-caller crash-safe journal for the IMW client.

Design (see titan-docs/PLAN_inner_memory_writer_service.md §7):
- File: data/run/imw_<pid>.jrn
- Header (first HEADER_SIZE bytes, fixed): msgpack + zero-pad
    {magic, pid, last_acked_offset, created_at}
- Records (appended after header): length-prefixed msgpack
    {req_id, sql, params, journaled_at}
- Append-only; fsync after every append + after header rewrite
- On clean shutdown: flush ACKs → rewrite header → truncate to last_acked_offset
- On crash: file persists; next boot replays records after header.last_acked_offset
- On rotate (size > max_mb): rename to .jrn.old, new journal starts with
  header.last_acked_offset=HEADER_SIZE. Old is deleted only after service
  confirms replay drained.
"""
from __future__ import annotations

import logging
import os
import struct
import time
from pathlib import Path
from threading import RLock
from typing import Iterator, Optional

import msgpack

from .wire_format import MAX_FRAME_BYTES, _LEN_HEADER

logger = logging.getLogger("titan.imw.journal")

HEADER_SIZE = 256  # fixed; plenty for our header fields
HEADER_MAGIC = b"IMW_JOURNAL_V1"


class JournalError(RuntimeError):
    """Raised on journal integrity / serialization failures."""


class CallerJournal:
    """Append-only, fsync-safe per-caller journal file.

    Thread-safe via RLock. File handles stay open for the life of the journal.
    """

    def __init__(self, path: str, pid: Optional[int] = None, max_mb: int = 100) -> None:
        self._path = Path(path)
        self._pid = pid if pid is not None else os.getpid()
        self._max_bytes = max_mb * 1024 * 1024
        self._lock = RLock()
        self._last_acked_offset = HEADER_SIZE
        self._ack_counter = 0
        self._last_header_flush = time.time()
        self._file = None  # type: Optional[object]
        self._fd = -1
        self._closed = False
        self._open_or_create()

    # ── file lifecycle ───────────────────────────────────────────────

    def _open_or_create(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists() and self._path.stat().st_size > 0:
            self._load_header()
        else:
            self._write_fresh_header()
        # Open in append-binary mode so writes go to end regardless of seek
        self._file = open(self._path, "rb+")
        self._fd = self._file.fileno()
        # position at end
        self._file.seek(0, os.SEEK_END)

    def _write_fresh_header(self) -> None:
        self._last_acked_offset = HEADER_SIZE
        header = self._pack_header()
        with open(self._path, "wb") as f:
            f.write(header)
            f.flush()
            os.fsync(f.fileno())

    def _pack_header(self) -> bytes:
        payload = msgpack.packb({
            "magic": HEADER_MAGIC,
            "pid": self._pid,
            "last_acked_offset": self._last_acked_offset,
            "created_at": time.time(),
        }, use_bin_type=True)
        if len(payload) > HEADER_SIZE - 4:
            raise JournalError(f"header too big: {len(payload)} > {HEADER_SIZE - 4}")
        # 4-byte length + msgpack + zero-pad to HEADER_SIZE
        return _LEN_HEADER.pack(len(payload)) + payload + b"\x00" * (HEADER_SIZE - 4 - len(payload))

    def _load_header(self) -> None:
        with open(self._path, "rb") as f:
            raw = f.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise JournalError(f"journal header truncated: {len(raw)}/{HEADER_SIZE}")
        n = struct.unpack(">I", raw[:4])[0]
        if n <= 0 or n > HEADER_SIZE - 4:
            raise JournalError(f"journal header length invalid: {n}")
        obj = msgpack.unpackb(raw[4:4 + n], raw=False)
        if obj.get("magic") != HEADER_MAGIC:
            raise JournalError(f"journal magic mismatch: {obj.get('magic')!r}")
        self._last_acked_offset = int(obj.get("last_acked_offset", HEADER_SIZE))
        if self._last_acked_offset < HEADER_SIZE:
            self._last_acked_offset = HEADER_SIZE

    # ── append + ack ─────────────────────────────────────────────────

    def append(self, req_id: str, sql: Optional[str], params, journaled_at: Optional[float] = None) -> int:
        """Append a record, fsync, return byte offset of this record."""
        with self._lock:
            if self._closed:
                raise JournalError("journal is closed")
            record = msgpack.packb({
                "req_id": req_id,
                "sql": sql,
                "params": params,
                "journaled_at": journaled_at if journaled_at is not None else time.time(),
            }, use_bin_type=True)
            if len(record) > MAX_FRAME_BYTES:
                raise JournalError(f"journal record too large: {len(record)}")
            frame = _LEN_HEADER.pack(len(record)) + record
            offset = self._file.tell()
            self._file.write(frame)
            self._file.flush()
            os.fsync(self._fd)
            # size-based rotation check
            if offset + len(frame) > self._max_bytes:
                self._rotate()
            return offset

    def ack(self, offset: int, frame_len_incl_header: int) -> None:
        """Mark record at `offset` as acknowledged. Update header periodically."""
        with self._lock:
            end = offset + frame_len_incl_header
            if end > self._last_acked_offset:
                self._last_acked_offset = end
            self._ack_counter += 1
            now = time.time()
            if self._ack_counter >= 100 or (now - self._last_header_flush) > 1.0:
                self._flush_header()
                self._ack_counter = 0
                self._last_header_flush = now

    def _flush_header(self) -> None:
        if self._file is None:
            return
        pos = self._file.tell()
        try:
            self._file.seek(0)
            self._file.write(self._pack_header())
            self._file.flush()
            os.fsync(self._fd)
        finally:
            self._file.seek(pos)

    # ── replay ───────────────────────────────────────────────────────

    def iter_unacked(self) -> Iterator[tuple]:
        """Yield (offset, frame_len_incl_header, record_dict) after last_acked_offset."""
        with self._lock:
            self._file.flush()
            cur = self._last_acked_offset
            end = self._path.stat().st_size
            while cur < end:
                self._file.seek(cur)
                head = self._file.read(4)
                if len(head) < 4:
                    break
                (n,) = _LEN_HEADER.unpack(head)
                if n <= 0 or n > MAX_FRAME_BYTES:
                    logger.warning("[imw.journal] bad frame length %d at offset %d; stopping replay", n, cur)
                    break
                body = self._file.read(n)
                if len(body) < n:
                    logger.warning("[imw.journal] truncated frame at offset %d; stopping replay", cur)
                    break
                try:
                    rec = msgpack.unpackb(body, raw=False)
                except Exception as e:
                    logger.warning("[imw.journal] unparseable frame at %d: %s; stopping", cur, e)
                    break
                yield (cur, 4 + n, rec)
                cur += 4 + n

    # ── rotation + close ─────────────────────────────────────────────

    def _rotate(self) -> None:
        old_path = self._path.with_suffix(self._path.suffix + ".old")
        # Close current, rename to .old, reopen fresh
        self._flush_header()
        self._file.close()
        try:
            if old_path.exists():
                old_path.unlink()
            os.rename(self._path, old_path)
        except OSError as e:
            logger.error("[imw.journal] rotate rename failed: %s", e)
        self._write_fresh_header()
        self._file = open(self._path, "rb+")
        self._fd = self._file.fileno()
        self._file.seek(0, os.SEEK_END)
        logger.info("[imw.journal] rotated %s → %s", self._path.name, old_path.name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._file.flush()
                size = self._path.stat().st_size
                # If all records are acked, reset to a clean fresh header.
                # Otherwise keep the unacked tail on disk for next boot's replay.
                if self._last_acked_offset >= size:
                    self._last_acked_offset = HEADER_SIZE
                    self._file.seek(0)
                    self._file.write(self._pack_header())
                    self._file.flush()
                    self._file.truncate(HEADER_SIZE)
                    os.fsync(self._fd)
                else:
                    self._flush_header()
            except OSError as e:
                logger.warning("[imw.journal] close truncate failed: %s", e)
            if self._file is not None:
                self._file.close()
            self._closed = True

    def __enter__(self) -> "CallerJournal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def scan_orphan_journals(journal_dir: str, exclude_pid: Optional[int] = None) -> list:
    """Return a list of (path, pid) for journal files whose owner is dead/absent.

    The IMW service daemon calls this on startup to find orphan journals to replay.
    """
    out = []
    exclude_pid = exclude_pid if exclude_pid is not None else os.getpid()
    p = Path(journal_dir)
    if not p.exists():
        return out
    for f in p.glob("imw_*.jrn"):
        # Extract pid from filename: imw_<pid>.jrn
        try:
            pid_str = f.stem.split("_", 1)[1]
            pid = int(pid_str)
        except (IndexError, ValueError):
            logger.warning("[imw.journal] unparseable journal filename: %s", f.name)
            continue
        if pid == exclude_pid:
            continue
        try:
            os.kill(pid, 0)
            # alive — owner still owns it, skip
            continue
        except ProcessLookupError:
            out.append((f, pid))
        except PermissionError:
            # exists but not ours — skip conservatively
            continue
    return out
