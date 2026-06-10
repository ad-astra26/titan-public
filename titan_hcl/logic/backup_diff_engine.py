"""DiffEngine — the ONE shared, symmetric, bounded-RSS build/restore library.

RFP_backup_redesign_spine Phase A (INV-BRS-2/3/4; ARCHITECTURE_backup_restore
§24.5.a). Build and restore are symmetric, so the snapshot router + the
diff/pack/apply steps live in ONE async library used by both the BackupWorker
(build) and the RestoreWorker (restore). It WRAPS the existing, proven encoders
+ pack + restore-apply (INV-BRS-8 / INV-BR-8) — it reimplements no crypto, no
diff math, no Merkle. The ONLY new behaviour is the snapshot router: every live
SQLite DB is captured as a transactionally-consistent image (`conn.backup`)
BEFORE encode, never read live (INV-BR-11).

    src  → snapshot(src)        → consistent, truncation/torn-read-free artifact
         → encode(snap, base)   → diff_dict (full_ship / xdelta3 / timechain_tail)
         → pack(specs, out)     → event tarball (streamed)  [build]
    tarball → apply(...)        → reconstructed bytes        [restore, symmetric]

Every step is bounded-RSS (O(1 MiB)-chunk) + CPU-cooperative (heavy work runs
off the caller's loop via `asyncio.to_thread`; the IMW snapshot awaits the async
writer client) — INV-BRS-3.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Iterable, Optional

from . import diff_encoders
from .diff_encoders import full_ship
from .backup_event_tarball import FileDiffSpec, pack_event_tarball
from .backup_restore import apply_event_components
from .backup_sqlite_snapshot import (
    is_sqlite_file,
    snapshot_dest_for,
    snapshot_sqlite_async,
)

logger = logging.getLogger("titan.backup.diff_engine")


class DiffEngine:
    """Stateless async helpers (no `self` god-state)."""

    async def snapshot(self, src_path: str, *,
                       scratch_dir: Optional[str] = None) -> tuple[str, bool]:
        """Capture `src_path` as an immutable, internally-consistent artifact
        BEFORE encode (INV-BR-11). Routes by source class:
          • live SQLite DB (header-detected) → consistent online backup
            (`conn.backup`) — IMW-owned through the IMW op, self-written via a
            read-only conn;
          • `.json`/`.jsonl` + small non-DB → copy; large binary → hardlink
            (the existing `full_ship._race_safe_snapshot`).
        Returns (snapshot_path, owned) — owned=True ⇒ caller/pack unlinks it.
        """
        if is_sqlite_file(src_path):
            dest = snapshot_dest_for(src_path, scratch_dir)
            await snapshot_sqlite_async(src_path, dest)
            return dest, True
        return await asyncio.to_thread(full_ship._race_safe_snapshot, src_path)

    async def encode(self, current_path: str, baseline_path: Optional[str] = None,
                     *, format_hint: Optional[str] = None) -> dict:
        """Wrap `diff_encoders.encode_diff` (full_ship / xdelta3 / timechain_tail).
        `current_path` is the consistent SNAPSHOT from `snapshot()` — never a live
        source. Runs off the loop (xdelta3 subprocess / file IO)."""
        return await asyncio.to_thread(
            diff_encoders.encode_diff, current_path, baseline_path, format_hint)

    async def pack(self, *, event_id: str, event_type: str, component: str,
                   file_specs: Iterable[FileDiffSpec], output_path: str,
                   ts_unix: Optional[float] = None,
                   extra_metadata: Optional[dict] = None) -> dict:
        """Wrap `pack_event_tarball` — ONE streamed pass over the staged patch
        artifacts (zstd stream_writer + tar, 1 MiB chunks, atomic .tmp→replace);
        `info.size` is bound to the immutable snapshot, never a live file."""
        return await asyncio.to_thread(
            lambda: pack_event_tarball(
                event_id=event_id, event_type=event_type, component=component,
                file_specs=file_specs, output_path=output_path,
                ts_unix=ts_unix, extra_metadata=extra_metadata))

    async def apply(self, components: dict, target_dir: str,
                    arc_to_target: Callable[[str, str], str], *,
                    verify_patch_hash: bool = True, best_effort: bool = False,
                    arweave_fetch_skipped_sync: Optional[Callable[[str], bytes]] = None
                    ) -> dict:
        """Wrap `apply_event_components` — the streamed, symmetric restore-apply
        (member extract → decode → write). Same library as build (INV-BRS-2)."""
        return await asyncio.to_thread(
            apply_event_components, components, target_dir, arc_to_target,
            arweave_fetch_skipped_sync, verify_patch_hash, best_effort)
