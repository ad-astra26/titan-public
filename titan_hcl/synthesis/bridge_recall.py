"""BridgeRecall — watermark-gated cross-process read of activation_state.

Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0 §25 INV-Syn-4 / G18).

Cross-process consumers (e.g. an agno_worker that needs activation
lookups but does NOT already hold a DuckDB connection to
titan_memory.duckdb) read activation_state DIRECTLY through this reader,
gating every query on the `synth_status.bin` SHM watermark that
synthesis_worker publishes after each 60s recompute. INV-Syn-4: zero new
sync `bus.request` patterns — reads are watermark-guarded file reads.

In-process callers that ALREADY hold a TieredMemoryGraph (so already
have a read-write connection to titan_memory.duckdb open) MUST NOT
construct a BridgeRecall with a separate db_path against the same file —
DuckDB rejects two connections to the same file with different configs
in the same process. Those callers should reuse their existing
connection for the activation_state SELECT and use BridgeRecall ONLY for
the watermark check (pass `db_path=None` skip mode handled via the
`existing_conn` constructor param, OR just call read_watermark() /
is_fresh() and bypass activation_lookup()).

Soft-fail policy: when the watermark is stale (last_consistent_event_ts
older than `freshness_window_s` — default 300s) OR the SHM slot is
missing entirely (worker not booted), the lookup returns an empty dict.
The caller (composite_score) then treats every item as cold-start →
ranking degrades to cosine + importance, never crashes.

The DuckDB read is also defensive: if the file is locked by the writer
(synthesis_worker has a write connection open), DuckDB read-only mode
co-exists in most cases — but we MUST NOT raise; on any DuckDB error,
return empty dict and let the caller degrade.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
from typing import Optional

import numpy as np

from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryReader,
    resolve_shm_root,
)

logger = logging.getLogger(__name__)

# Default freshness tolerance: 300s = 5× the 60s recompute interval. If
# the worker hasn't recomputed in 5 cycles, treat the state as stale and
# degrade. Tunable per consumer.
DEFAULT_FRESHNESS_WINDOW_S = 300.0

# Mirror of titan_hcl/modules/synthesis_worker.py constants — kept inline
# to avoid importing the worker module (which pulls in DuckDB at import-
# time even in read-only consumer processes that never write).
_SYNTH_STATUS_SLOT_NAME = "synth_status"
_SYNTH_STATUS_PAYLOAD_BYTES = 24       # struct '<ddII'
_SYNTH_STATUS_SCHEMA_VERSION = 1


def _build_synth_status_spec() -> RegistrySpec:
    """Build the RegistrySpec mirroring SynthStatusWriter (must stay in sync)."""
    return RegistrySpec(
        name=_SYNTH_STATUS_SLOT_NAME,
        dtype=np.dtype(np.uint8),
        shape=(_SYNTH_STATUS_PAYLOAD_BYTES,),
        feature_flag="",
        schema_version=_SYNTH_STATUS_SCHEMA_VERSION,
        variable_size=True,
    )


class BridgeRecall:
    """Read-only consumer of synth_status.bin watermark + activation_state.

    Construct ONCE per consumer process (it caches the SHM mmap + a
    DuckDB read-only connection). Thread-safe — DuckDB connections are
    NOT thread-safe by default, so we serialize through a lock.

    `activation_lookup(item_ids)` is the hot-path API for composite_score.
    Returns `{item_id: base_level}` for items present in activation_state;
    absent items remain absent in the dict (composite_score substitutes
    cold-start). Returns `{}` on any failure (stale watermark, missing
    SHM slot, DuckDB error) → composite_score gracefully degrades.
    """

    def __init__(
        self,
        titan_id: Optional[str] = None,
        db_path: Optional[str] = None,
        freshness_window_s: float = DEFAULT_FRESHNESS_WINDOW_S,
    ) -> None:
        from titan_hcl.core.state_registry import resolve_titan_id
        self._titan_id = resolve_titan_id(titan_id)
        # Mirror TieredMemoryGraph's data-dir resolution (core/memory.py)
        # so BUG-B1-SHARED-LOCKS shadow kernels (TITAN_DATA_DIR set →
        # data_shadow_<port>/) hit the right DuckDB. Without this,
        # BridgeRecall would always open the canonical `data/` path,
        # which is locked by the live kernel for shadow tests.
        if db_path is None:
            data_dir = os.environ.get("TITAN_DATA_DIR", "data")
            db_path = os.path.join(data_dir, "titan_memory.duckdb")
        self._db_path = db_path
        self._freshness_window_s = freshness_window_s

        # Watermark reader — built lazily (SHM slot may not exist yet at
        # consumer import time; synthesis_worker boots later).
        self._spec = _build_synth_status_spec()
        self._reader: Optional[StateRegistryReader] = None
        self._reader_lock = threading.Lock()

        # DuckDB read-only — also lazy; opening it before the writer has
        # created the file would create an empty DB and silently mask the
        # real one.
        self._duck_conn = None
        self._duck_lock = threading.Lock()
        self._duck_error_logged = False

    # ── watermark ───────────────────────────────────────────────────────

    def read_watermark(self) -> Optional[tuple[float, float, int, int]]:
        """Read the current synth_status.bin payload.

        Returns `(last_consistent_event_ts, last_recompute_ts,
        items_tracked, recompute_count)` or None if the slot is missing /
        unreadable / uninitialized.
        """
        with self._reader_lock:
            if self._reader is None:
                shm_root = resolve_shm_root(self._titan_id)
                self._reader = StateRegistryReader(self._spec, shm_root)
            payload = self._reader.read_variable()
        if payload is None or len(payload) != _SYNTH_STATUS_PAYLOAD_BYTES:
            return None
        return struct.unpack("<ddII", payload)

    def is_fresh(self, now: float) -> bool:
        """True if the watermark is fresh enough for the consumer to trust
        activation_state reads. False → caller should degrade."""
        wm = self.read_watermark()
        if wm is None:
            return False
        last_consistent_event_ts, last_recompute_ts, _, _ = wm
        # Either signal counts as "the worker has done work recently."
        most_recent = max(last_consistent_event_ts, last_recompute_ts)
        if most_recent == 0.0:
            return False
        return (now - most_recent) <= self._freshness_window_s

    # ── activation lookup ──────────────────────────────────────────────

    def activation_lookup(self, item_ids: list[str]) -> dict[str, float]:
        """Read `base_level` for the requested item_ids via the JSON
        activation snapshot synthesis_worker exports each recompute pass.

        Returns a dict ONLY for items present in the snapshot — absent
        items are omitted (composite_score substitutes cold-start).

        DuckDB 1.5+ enforces an exclusive file lock even for read_only
        opens, so reading synthesis.duckdb directly from a cross-process
        consumer would conflict with synthesis_worker's R/W connection.
        Snapshot file (sibling of synthesis.duckdb, named
        `activation_snapshot.json`) sidesteps this entirely.

        Soft-fail: returns `{}` on stale watermark, missing snapshot, or
        any JSON parse error. Never raises to the caller.
        """
        if not item_ids:
            return {}

        import time as _time
        if not self.is_fresh(_time.time()):
            return {}

        # Snapshot path = sibling of self._db_path
        snapshot_path = os.path.join(
            os.path.dirname(self._db_path) or ".",
            "activation_snapshot.json")
        if not os.path.exists(snapshot_path):
            return {}

        with self._duck_lock:
            # Lazy-cache the snapshot — re-read only on mtime change.
            try:
                mtime = os.path.getmtime(snapshot_path)
                if not hasattr(self, "_snapshot_cache") \
                        or self._snapshot_mtime != mtime:
                    import json
                    with open(snapshot_path) as f:
                        self._snapshot_cache = json.load(f)
                    self._snapshot_mtime = mtime
            except Exception as exc:
                if not self._duck_error_logged:
                    logger.warning(
                        "[BridgeRecall] snapshot read failed (%s): %s "
                        "— degrading to cosine-only",
                        snapshot_path, exc)
                    self._duck_error_logged = True
                return {}
            snapshot = self._snapshot_cache or {}

        out: dict[str, float] = {}
        for item_id in item_ids:
            v = snapshot.get(item_id)
            if v is None:
                continue
            try:
                out[item_id] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    # ── bundle read (Phase 2 D-P2-4) ──────────────────────────────────

    def read_bundle(
        self, entity_class: str, entity_id: str, fork: str,
    ) -> list[dict]:
        """Read the materialized standing-contract bundle for
        `(entity_class, entity_id, fork)` via the cross-process snapshot
        file synthesis_worker exports each recompute pass.

        Returns the bundle list (newest-first list of tx records) when:
          - the synth_status.bin watermark is fresh, AND
          - bundle_snapshot.json exists + parses, AND
          - the composite key is present in the snapshot.

        Returns `[]` on any soft-fail path (stale watermark, missing
        snapshot, missing key, parse error) — consumers should treat the
        empty result as "no materialized bundle yet, fall back to a
        SEARCH" (PLAN_synthesis_engine_Phase2.md §2D engine recall).

        Soft-fail rationale (INV-Syn-4): cross-process readers MUST
        degrade gracefully when the writer is missing/stale, never raise
        to the caller.
        """
        # Synthesis-bundle snapshot lives next to synthesis.duckdb, NOT
        # next to titan_memory.duckdb. The two DBs share `data/` (or
        # data_shadow_<port>/ under TITAN_DATA_DIR) but have distinct
        # filenames. We resolve the synthesis snapshot path independently
        # of self._db_path (which points at titan_memory.duckdb for the
        # P1 activation_lookup path).
        snapshot_path = self._resolve_bundle_snapshot_path()
        if not os.path.exists(snapshot_path):
            return []

        import time as _time
        if not self.is_fresh(_time.time()):
            return []

        composite_key = f"{entity_class}|{entity_id}|{fork}"

        with self._duck_lock:
            # Lazy-cache the bundle snapshot — re-read only on mtime change.
            try:
                mtime = os.path.getmtime(snapshot_path)
                if not hasattr(self, "_bundle_snapshot_cache") \
                        or self._bundle_snapshot_mtime != mtime:
                    import json
                    with open(snapshot_path) as f:
                        self._bundle_snapshot_cache = json.load(f)
                    self._bundle_snapshot_mtime = mtime
            except Exception as exc:
                if not getattr(self, "_bundle_error_logged", False):
                    logger.warning(
                        "[BridgeRecall] bundle snapshot read failed (%s): "
                        "%s — degrading to empty",
                        snapshot_path, exc)
                    self._bundle_error_logged = True
                return []
            snapshot = self._bundle_snapshot_cache or {}

        bundles = snapshot.get("bundles", {}) if isinstance(snapshot, dict) else {}
        result = bundles.get(composite_key, [])
        if not isinstance(result, list):
            return []
        return [dict(r) for r in result if isinstance(r, dict)]

    def _resolve_bundle_snapshot_path(self) -> str:
        """Resolve `data/bundle_snapshot.json` (or shadow equivalent) —
        sibling of synthesis.duckdb, NOT of self._db_path
        (titan_memory.duckdb)."""
        # Mirror synthesis_worker's path resolution:
        #   db_path = os.path.join(data_dir, "synthesis.duckdb")
        #   snapshot_path = os.path.join(data_dir, BUNDLE_SNAPSHOT_NAME)
        data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        from titan_hcl.synthesis.standing_store import BUNDLE_SNAPSHOT_NAME
        return os.path.join(data_dir, BUNDLE_SNAPSHOT_NAME)

    def close(self) -> None:
        with self._reader_lock:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = None
        with self._duck_lock:
            if self._duck_conn is not None:
                try:
                    self._duck_conn.close()
                except Exception:
                    pass
                self._duck_conn = None


# ── process-singleton accessor ─────────────────────────────────────────

_bridge_recall_singleton: Optional[BridgeRecall] = None
_bridge_recall_lock = threading.Lock()


def get_bridge_recall(
    titan_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> BridgeRecall:
    """Process-local BridgeRecall singleton. Construct once per process
    (cheap — no I/O at construction; SHM + DuckDB attach lazily on first
    read).

    `db_path` lets callers override the default `data/titan_memory.duckdb`
    (which is what BUG-B1-SHARED-LOCKS' TITAN_DATA_DIR-aware callers need
    to point at — e.g. shadow kernels using `data_shadow_<port>/`). When
    None, BridgeRecall resolves the env var TITAN_DATA_DIR itself.
    """
    global _bridge_recall_singleton
    with _bridge_recall_lock:
        if _bridge_recall_singleton is None:
            _bridge_recall_singleton = BridgeRecall(
                titan_id=titan_id, db_path=db_path)
    return _bridge_recall_singleton


__all__ = [
    "BridgeRecall", "get_bridge_recall",
    "DEFAULT_FRESHNESS_WINDOW_S",
]
