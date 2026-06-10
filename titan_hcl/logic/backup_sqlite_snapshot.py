"""SQLite source-snapshot consistency for backup.

ARCHITECTURE_backup_restore §24.5.a + INV-BR-11 / INV-BRS-4/11
(RFP_backup_redesign_spine Phase A).

Before the §24.5 diff is encoded, a live source DB must be captured as a
*transactionally-consistent* image — a hardlink shares the inode (torn read
under concurrent write/checkpoint) and even a plain copy of a WAL-mode SQLite
DB is not consistent. This module is the SINGLE source of truth for:

  • SQLite-header detection (route only real SQLite files here — a `.db` may be
    DuckDB/Kuzu, which are NOT SQLite and must keep the copy/hardlink path);
  • the IMW-owned-DB registry — derived from the enabled `[persistence_*]`
    config sections whose `db_path` realpath-matches (INV-BRS-11: "never guess"),
    the SAME test `events_teacher.py` uses to decide whether to route writes
    through the writer;
  • the consistent online backup itself (`sqlite3.Connection.backup`, stepped
    pages+sleep) — IMW-owned DBs THROUGH the single-writer IMW `snapshot` op,
    self-written DBs via our OWN read-only connection.

Both the live sync build path (`backup_upload_pipeline._build_*_payload`) and
the async `DiffEngine` call these helpers, so the classification can never
diverge between the two paths.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("titan.backup.sqlite_snapshot")

_SQLITE_MAGIC = b"SQLite format 3\x00"  # 16-byte file header

# conn.backup step size — mirrors writer_service._SNAPSHOT_BACKUP_* so the IMW
# op and the self-written read-conn path step at the same cadence. Stepped
# pages+sleep (NOT VACUUM INTO) so a hot DB yields the source lock between
# batches; the online-backup API guarantees a consistent point-in-time image
# even under concurrent writes (auto-restart on mid-copy source change).
_BACKUP_PAGES = 1024       # ~4 MB per step at the 4 KiB default page size
_BACKUP_SLEEP_S = 0.001    # 1 ms yield between page batches
_BACKUP_TIMEOUT_S = 30.0


def is_sqlite_file(path: str) -> bool:
    """True iff `path` begins with the SQLite file magic. Header-detection, not
    extension-trust: a `.db`/`.kuzu` could be DuckDB or Kuzu (different engines,
    no `conn.backup`) — those MUST NOT take the SQLite online-backup path."""
    try:
        with open(path, "rb") as f:
            return f.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def _iter_persistence_sections(full: Optional[dict] = None):
    """Yield (section_name, IMWConfig) for every ENABLED `[persistence*]` section.

    `full` = an already-parsed config dict (tests inject one); None → read the
    live config.toml. Lazy imports keep this off the boot import graph."""
    from titan_hcl.persistence.config import IMWConfig, _load_config_toml_cached
    if full is None:
        cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
        if not cfg_path.exists():
            return
        full = _load_config_toml_cached(cfg_path)
    for key, section in full.items():
        if key != "persistence" and not key.startswith("persistence_"):
            continue
        if not isinstance(section, dict):
            continue
        cfg = IMWConfig.from_dict(section)
        if not cfg.enabled or cfg.mode == "disabled" or not cfg.db_path:
            continue
        yield key, cfg


def imw_owned_realpaths(full: Optional[dict] = None) -> frozenset:
    """The realpath'd db_paths that are IMW-owned = every ENABLED `[persistence_*]`
    section's `db_path` (INV-BRS-11 / INV-BR-11). Derived from config, never a
    hardcoded list. Today: inner_memory.db, social_graph.db, events_teacher.db,
    observatory.db, consciousness.db (only the first three are in the backup set).
    """
    return frozenset(os.path.realpath(cfg.db_path) for _, cfg in _iter_persistence_sections(full))


def _imw_client_for_db(realpath_db: str):
    """Return a writer client for the IMW-owned DB at `realpath_db` (its cfg
    carries the right per-DB `socket_path` → the correct daemon), or None if it
    can't be resolved (caller falls back to a read-conn backup)."""
    from titan_hcl.persistence.writer_client import get_client
    for key, cfg in _iter_persistence_sections():
        if os.path.realpath(cfg.db_path) == realpath_db:
            return get_client(f"backup_snapshot_{key}", cfg=cfg)
    return None


def _prepare_dest(dest_path: str) -> None:
    """A partial prior image (and its sidecars) must never survive into a fresh
    snapshot."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(dest) + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError as e:
                logger.warning("[sqlite_snapshot] could not unlink %s: %s", p, e)


def _read_conn_backup(src_path: str, dest_path: str) -> int:
    """Consistent online backup of a SELF-written SQLite DB via our OWN
    read-only connection (WAL-safe, concurrency-safe across processes). Stepped
    pages+sleep. Returns the snapshot size in bytes. Synchronous — run off the
    heartbeat thread by the caller (INV-BRS-3)."""
    import sqlite3
    _prepare_dest(dest_path)
    src = sqlite3.connect(src_path, timeout=_BACKUP_TIMEOUT_S)
    try:
        src.execute("PRAGMA query_only=1")  # read-only source connection
        dst = sqlite3.connect(dest_path)
        try:
            src.backup(dst, pages=_BACKUP_PAGES, sleep=_BACKUP_SLEEP_S)
        finally:
            dst.close()
    finally:
        src.close()
    return os.path.getsize(dest_path) if os.path.exists(dest_path) else 0


def snapshot_sqlite_sync(src_path: str, dest_path: str) -> None:
    """Capture a transactionally-consistent image of the live SQLite DB at
    `src_path` → `dest_path` (SYNC — the live build path). IMW-owned → the IMW
    `snapshot` op (single-writer, INV-BR-11); self-written → our own read-only
    connection. On an IMW-owned DB whose client can't be reached, falls back to
    a read-conn backup (still consistent — the WAL lets a separate reader run)."""
    rp = os.path.realpath(src_path)
    if rp in imw_owned_realpaths():
        client = _imw_client_for_db(rp)
        if client is not None and client.snapshot(dest_path):
            return
        logger.warning("[sqlite_snapshot] %s is IMW-owned but the writer was "
                       "unreachable — read-conn backup fallback", src_path)
    _read_conn_backup(src_path, dest_path)


async def snapshot_sqlite_async(src_path: str, dest_path: str) -> None:
    """Async variant (the `DiffEngine` path) — never blocks the caller's loop:
    the IMW op awaits the async client; the self-written read-conn backup runs
    in a thread (`asyncio.to_thread`, GIL-releasing — INV-BRS-3)."""
    rp = os.path.realpath(src_path)
    if rp in imw_owned_realpaths():
        client = _imw_client_for_db(rp)
        if client is not None and await client.asnapshot(dest_path):
            return
        logger.warning("[sqlite_snapshot] %s is IMW-owned but the writer was "
                       "unreachable — read-conn backup fallback", src_path)
    await asyncio.to_thread(_read_conn_backup, src_path, dest_path)


def snapshot_dest_for(src_path: str, scratch_dir: Optional[str] = None) -> str:
    """A fresh dest path for a SQLite online-backup image. KEEPS the source
    extension so the encoder dispatch (`diff_encoders.select_encoder`) routes the
    snapshot to the same encoder as the live source (a `.db` → xdelta3). The
    image lives in the `.bksnap_scratch` dir (OUT of the source tree, so a leaked
    snapshot can never be re-snapshotted by the next event's directory walk)."""
    from .diff_encoders import full_ship  # lazy — avoid import cost at module load
    ext = os.path.splitext(src_path)[1] or ".db"
    base = os.path.basename(src_path)
    d = scratch_dir or full_ship._scratch_dir_for(src_path)
    os.makedirs(d, exist_ok=True)
    fd, p = tempfile.mkstemp(prefix=f"{base}.snap.", suffix=ext, dir=d)
    os.close(fd)
    os.unlink(p)  # backup writes a fresh file; mkstemp's empty stub must go
    return p


def prepare_sqlite_snapshot(src_path: str,
                            scratch_dir: Optional[str] = None) -> Optional[str]:
    """If `src_path` is a live SQLite DB (header-detected), capture a consistent
    online-backup image and return its path (the CALLER owns cleanup); else None
    (the source is not SQLite → use it directly). SYNC — the live build path.

    This is the §24.5.a / INV-BR-11 entry point for `backup_upload_pipeline`:
    every SQLite source is routed through a consistent image BEFORE the §24.6
    skip-hash, the §24.5 encode, and the §24.7 Merkle, so all three see the same
    transactionally-consistent bytes (no TOCTOU between hash and packed bytes)."""
    if not is_sqlite_file(src_path):
        return None
    dest = snapshot_dest_for(src_path, scratch_dir)
    snapshot_sqlite_sync(src_path, dest)
    return dest
