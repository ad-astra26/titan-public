"""ThoughtSidecar — lock-free ``tx_hash → promoted-thought content`` store.

Synthesis spine, Phase B (RFP_synthesis_spine_reads_real_data §7 / B1). At
promotion the real thought (``user_prompt`` / ``agent_response``) is written here
keyed by the deterministic Timechain per-TX hash; the recall deref reads it.

Why a dedicated sqlite-WAL file (NOT ``titan_memory.duckdb``): DuckDB takes an
EXCLUSIVE cross-process lock — a read-only open of the live
``titan_memory.duckdb`` fails with *"Conflicting lock is held"* (verified live),
so a cross-process deref cannot read it, and the seal drops the per-TX content
(``db_ref``/``node_id``) from the chain envelope. SQLite in WAL mode allows many
concurrent readers alongside the single writer, cross-process — exactly what the
hot-path recall deref needs (no RPC, no lock; G18-G20 aligned).

Ownership: written ONLY by ``memory_worker`` (the promotion / DB owner) via
:class:`ThoughtSidecar`. Read R/O by any consumer process (agno / cognitive
recall) via :class:`ThoughtSidecarReader`. Both soft-fail — a missing/locked
sidecar yields ``None`` so the deref simply drops the candidate.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

SIDECAR_NAME = "thought_sidecar.db"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS thought_content ("
    " tx_hash TEXT PRIMARY KEY,"
    " node_id INTEGER,"
    " user_prompt TEXT,"
    " agent_response TEXT,"
    " memory_type TEXT,"
    " fork TEXT,"
    " ts DOUBLE)"
)


def _sidecar_path(data_dir: str) -> str:
    return os.path.join(data_dir, SIDECAR_NAME)


class ThoughtSidecar:
    """Writer side — owned by memory_worker. sqlite-WAL, internally serialized."""

    def __init__(self, data_dir: str):
        self._path = _sidecar_path(data_dir)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._path, check_same_thread=False, timeout=5.0)
        # WAL → concurrent cross-process readers alongside this single writer.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def put(self, *, tx_hash: str, node_id, user_prompt: str,
            agent_response: str, memory_type: str, fork: str,
            ts: Optional[float] = None) -> None:
        """Upsert one promoted thought's content keyed by its per-TX hash.
        Idempotent (INSERT OR REPLACE) — re-promotion / re-run is safe. Never
        raises (a sidecar write must not break promotion)."""
        if not tx_hash:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO thought_content "
                    "(tx_hash, node_id, user_prompt, agent_response, "
                    "memory_type, fork, ts) VALUES (?,?,?,?,?,?,?)",
                    (tx_hash, node_id, user_prompt, agent_response,
                     memory_type, fork,
                     ts if ts is not None else time.time()),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("[ThoughtSidecar] put failed for %s: %s",
                               str(tx_hash)[:12], e)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class ThoughtSidecarReader:
    """Read-only, lock-free reader for any consumer process (recall deref).

    Opens a normal connection with ``PRAGMA query_only=1`` rather than
    ``mode=ro`` — a WAL database needs to attach its shared-memory index, which a
    strict read-only file open cannot create; ``query_only`` gives a safe
    no-write connection that still reads WAL correctly. Lazy + soft-fail."""

    def __init__(self, data_dir: str):
        self._path = _sidecar_path(data_dir)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _ensure(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        if not os.path.exists(self._path):
            return None
        try:
            conn = sqlite3.connect(
                self._path, check_same_thread=False, timeout=2.0)
            conn.execute("PRAGMA query_only=1")
            conn.row_factory = sqlite3.Row
            self._conn = conn
        except Exception as e:
            logger.debug("[ThoughtSidecarReader] open failed: %s", e)
            return None
        return self._conn

    def get(self, tx_hash: str) -> Optional[dict]:
        """Return ``{tx_hash, node_id, user_prompt, agent_response, memory_type,
        fork, ts}`` for ``tx_hash`` or ``None``. Soft-fail."""
        if not tx_hash:
            return None
        with self._lock:
            conn = self._ensure()
            if conn is None:
                return None
            try:
                row = conn.execute(
                    "SELECT tx_hash, node_id, user_prompt, agent_response, "
                    "memory_type, fork, ts FROM thought_content "
                    "WHERE tx_hash = ? LIMIT 1", (tx_hash,)).fetchone()
                return dict(row) if row is not None else None
            except Exception as e:
                logger.debug("[ThoughtSidecarReader] get %s failed: %s",
                             str(tx_hash)[:12], e)
                return None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


__all__ = ["ThoughtSidecar", "ThoughtSidecarReader", "SIDECAR_NAME"]
