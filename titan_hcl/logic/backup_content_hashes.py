"""SPEC §24.6 — Skip-if-unchanged content_hash cache.

Per-file SHA256 cache at `data/backup_content_hashes.json`. Computed before
tarball build. If hash unchanged since last upload of this file, manifest
event lists it under `skipped_files` with a pointer to the previous tx_id /
event_id that still contains it.

Restore walks back to the most recent event that physically uploaded the
file — never silently drops a file from coverage (per SPEC §24.6 + rFP §1.9
locked-in principle).

Cache schema:
    {
      "titan_id": "T1",
      "schema_version": 1,
      "files": {
        "data/inner_memory.db": {
          "sha256": "<hex>",
          "last_upload_event_id": "<uuid4>",
          "last_upload_tx_id": "<arweave_tx | local_event_id>",
          "last_upload_ts": <float>,
          "size_bytes": <int>
        },
        ...
      }
    }

Cheap pre-check: on-disk file mtime > cache.last_upload_ts → SHA256 recompute.
Otherwise: trust cache. Mtime is a heuristic; if mtime hasn't changed but
content somehow did (clock skew / out-of-band rewrite), the SHA256 mismatch
that would surface at restore time triggers BACKUP_MERKLE_MISMATCH — never
a silent miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


CONTENT_HASH_SCHEMA_VERSION = 1


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: str, data: dict) -> None:
    """Atomic write per SPEC §11.H.2 — single-file, no .bak rotation
    (this cache is rebuildable from current files + manifest; not load-
    bearing for crash recovery). Tmp + rename only.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ContentHashCache:
    """Per-Titan content_hash cache for skip-if-unchanged dedup."""

    def __init__(self, titan_id: str, base_dir: str = "data"):
        self.titan_id = titan_id
        self.base_dir = base_dir
        # Single file per Titan (not per plane — local + Arweave share the
        # same hash universe since the content is identical).
        self.path = os.path.join(base_dir, f"backup_content_hashes_{titan_id}.json")
        self._data: dict = {
            "titan_id": titan_id,
            "schema_version": CONTENT_HASH_SCHEMA_VERSION,
            "files": {},
        }

    @classmethod
    def load(cls, titan_id: str, base_dir: str = "data") -> "ContentHashCache":
        obj = cls(titan_id=titan_id, base_dir=base_dir)
        if not os.path.exists(obj.path):
            return obj
        try:
            with open(obj.path) as f:
                data = json.load(f)
            if data.get("titan_id") != titan_id:
                logger.warning(
                    "[ContentHashCache] titan_id mismatch in %s: file has %r, "
                    "loader expected %r — starting fresh cache (rebuildable)",
                    obj.path, data.get("titan_id"), titan_id)
                return obj
            if data.get("schema_version") != CONTENT_HASH_SCHEMA_VERSION:
                logger.warning(
                    "[ContentHashCache] schema version %r in %s != current %r — "
                    "starting fresh cache",
                    data.get("schema_version"), obj.path,
                    CONTENT_HASH_SCHEMA_VERSION)
                return obj
            obj._data = data
            obj._data.setdefault("files", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "[ContentHashCache] Load failed for %s: %s — starting fresh "
                "cache (cache is rebuildable from disk + manifest)",
                obj.path, e)
        return obj

    def save(self) -> None:
        _atomic_write_json(self.path, self._data)

    # ── core ops ─────────────────────────────────────────────────────────

    def compute_current_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 of `file_path` (or None if file missing).

        Public method — does NOT touch cache. Used both by skip-check and
        by post-upload record. Returns hex string."""
        try:
            return _sha256_file(file_path)
        except (OSError, FileNotFoundError):
            return None

    def get_cached_entry(self, file_path: str) -> Optional[dict]:
        """Return the cache entry for `file_path` (or None if absent).

        Entry shape: {sha256, last_upload_event_id, last_upload_tx_id,
                      last_upload_ts, size_bytes}.
        """
        return self._data["files"].get(file_path)

    def is_unchanged_since_last_upload(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Cheap mtime pre-check → SHA256 confirm. Returns (unchanged, current_sha256).

        Cheap-path: file mtime ≤ cached.last_upload_ts → True without
        recomputing SHA256 (filesystem hasn't touched the file since last
        upload). Recompute on miss to detect out-of-band rewrites that
        preserved mtime (rare).

        Returns:
            (True, None)            — unchanged, used the cheap mtime path
                                       (caller can trust cached.sha256)
            (True, "<hex>")          — unchanged, verified via SHA256 recompute
            (False, "<hex>")         — CHANGED, caller must ship; pass this
                                       hash to record_upload after success
            (False, None)            — file missing on disk
        """
        if not os.path.exists(file_path):
            return (False, None)

        cached = self.get_cached_entry(file_path)
        if cached is None:
            current = self.compute_current_hash(file_path)
            return (False, current)

        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return (False, None)

        last_upload_ts = cached.get("last_upload_ts", 0.0)
        if mtime <= last_upload_ts:
            # Cheap path: file hasn't been touched since last upload.
            return (True, None)

        # mtime changed — must recompute to know if CONTENT changed
        current = self.compute_current_hash(file_path)
        if current == cached.get("sha256"):
            return (True, current)
        return (False, current)

    def record_upload(self, file_path: str, sha256: str, event_id: str,
                      tx_id: Optional[str], size_bytes: Optional[int] = None,
                      now: Optional[float] = None) -> None:
        """Record a successful upload — call AFTER the upload + manifest
        event lands so we don't poison the cache on transient failures.

        Args:
            file_path: in-scope archive path (key in cache.files dict)
            sha256: content hash (must match what was uploaded)
            event_id: manifest event_id this upload was anchored in
            tx_id: Arweave tx_id (or local event_id for local plane) —
                   pointer that restore uses to find this file's content
            size_bytes: optional file size at upload time (diagnostic)
            now: optional ts override (for tests)
        """
        if size_bytes is None:
            try:
                size_bytes = os.path.getsize(file_path)
            except OSError:
                size_bytes = 0
        self._data["files"][file_path] = {
            "sha256": sha256,
            "last_upload_event_id": event_id,
            "last_upload_tx_id": tx_id,
            "last_upload_ts": now if now is not None else time.time(),
            "size_bytes": size_bytes,
        }

    def make_skipped_file_pointer(self, file_path: str) -> Optional[dict]:
        """Build the manifest event's `skipped_files[]` entry for an
        unchanged file. Returns None if no cached entry exists (file has
        never been uploaded — must ship).

        Pointer shape per SPEC §24.3:
            {"path": <archive_path>,
             "prev_tx_id": <last_upload_tx_id>,
             "prev_event_id": <last_upload_event_id>}
        Restore walks back to that event to retrieve the file content.
        """
        cached = self.get_cached_entry(file_path)
        if cached is None:
            return None
        return {
            "path": file_path,
            "prev_tx_id": cached.get("last_upload_tx_id"),
            "prev_event_id": cached.get("last_upload_event_id"),
        }
