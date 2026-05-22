"""Content-addressed store (CAS) — Synthesis Engine Phase 0 / 0B.

A Timechain payload carries a `content_hash` reference instead of inlining large
content; the bytes live here, stored exactly once. SPEC §11.H.1 (critical-data row
`data/content_blobs/`) + §24.4.D (Arweave backup tier). D-SPEC-102 / v1.40.0.

Invariants:
- **Content-addressed**: blob filename = `sha256(content)` (hex). Identical content
  collapses to one file (free dedup) — `put` is idempotent and write-once.
- **Self-verifying**: `get` recomputes the hash and rejects a blob whose bytes no
  longer match its name (corruption / tamper).
- **Immutable + append-only**: blobs are never mutated or deleted here. Reference-
  counted GC is a later phase (read-only audit in Phase 0); canonical blobs are never
  removed (INV-3 / G16).
- **Atomic writes**: tempfile in the shard dir → fsync → `os.replace` (POSIX atomic).
  No rotate-`.bak` pattern — immutable content needs none.

This module is pure storage: it has no Timechain, bus, or SHM coupling. It is safe to
import and exercise in isolation (Phase-0 Step 1).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

# Two-level fan-out (ab/cd/<hash>) keeps any single directory well under the point
# where ext4 htree lookups degrade, even at tens of millions of blobs.
_SHARD_PREFIX_LEN = 2
_SHARD_LEVELS = 2
_HASH_HEX_LEN = 64  # sha256
_DIR_NAME = "content_blobs"


class ContentStoreError(Exception):
    """Base class for CAS errors."""


class BlobNotFound(ContentStoreError, KeyError):
    """Requested hash is not present in the store."""


class CorruptBlob(ContentStoreError):
    """A blob's bytes no longer hash to its filename (corruption / tamper)."""


def _resolve_data_dir() -> Path:
    """Project data dir, honoring TITAN_DATA_DIR (shadow-swap aware)."""
    env = os.environ.get("TITAN_DATA_DIR")
    if env:
        return Path(env)
    # Project root = three levels up from this file (titan_hcl/synthesis/x.py).
    return Path(__file__).resolve().parent.parent.parent / "data"


def _is_hash(h: str) -> bool:
    if len(h) != _HASH_HEX_LEN:
        return False
    try:
        int(h, 16)
    except ValueError:
        return False
    return True


class ContentStore:
    """Sharded, content-addressed blob store rooted at `<data>/content_blobs/`."""

    def __init__(self, root: Optional[Path] = None) -> None:
        base = Path(root) if root is not None else _resolve_data_dir()
        # If caller passes a dir that isn't already the store, nest content_blobs/.
        self._root = base if base.name == _DIR_NAME else base / _DIR_NAME
        # NB: no mkdir here — construction is side-effect-free so read-only
        # consumers (e.g. cas_audit) never create dirs on live data. The shard
        # dirs are created lazily in `put()`.

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, h: str) -> Path:
        parts = [h[i * _SHARD_PREFIX_LEN:(i + 1) * _SHARD_PREFIX_LEN] for i in range(_SHARD_LEVELS)]
        return self._root.joinpath(*parts, h)

    def put(self, data: bytes) -> str:
        """Store `data`; return its sha256 hex. Idempotent + write-once (dedup)."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"ContentStore.put expects bytes, got {type(data).__name__}")
        data = bytes(data)
        h = hashlib.sha256(data).hexdigest()
        dest = self._path_for(h)
        if dest.exists():
            return h  # identical content already stored — free dedup
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write tmp in the same shard dir, fsync, then rename into place.
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, dest)
            tmp_name = None
        finally:
            if tmp_name is not None and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return h

    def exists(self, h: str) -> bool:
        return _is_hash(h) and self._path_for(h).exists()

    def get(self, h: str) -> bytes:
        """Return the blob for `h`, verifying integrity on read."""
        if not _is_hash(h):
            raise BlobNotFound(f"not a valid sha256 hex: {h!r}")
        path = self._path_for(h)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise BlobNotFound(h) from None
        if hashlib.sha256(data).hexdigest() != h:
            raise CorruptBlob(
                f"blob {h} failed integrity check (bytes do not match filename hash)"
            )
        return data

    def stat(self) -> dict:
        """Read-only store summary (blob count + total bytes) for GC audit / metrics."""
        count = 0
        total = 0
        if not self._root.exists():
            return {"root": str(self._root), "blob_count": 0, "total_bytes": 0}
        for p in self._root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                count += 1
                total += p.stat().st_size
        return {"root": str(self._root), "blob_count": count, "total_bytes": total}


_default_store: Optional[ContentStore] = None


def get_content_store() -> ContentStore:
    """Process-wide default store rooted at the resolved data dir."""
    global _default_store
    if _default_store is None:
        _default_store = ContentStore()
    return _default_store
