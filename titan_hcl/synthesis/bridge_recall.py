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
        """Read `base_level` for the requested item_ids. Returns a dict
        ONLY for items present in activation_state — absent items are
        omitted (composite_score substitutes the cold-start default).

        Soft-fail: returns `{}` on stale watermark, missing SHM slot,
        DuckDB error, or any other failure. Never raises to the caller.
        """
        if not item_ids:
            return {}

        import time as _time
        if not self.is_fresh(_time.time()):
            return {}

        # Open DuckDB lazily (read-only) — guards against the writer
        # creating the file later in the same process lifetime.
        with self._duck_lock:
            if self._duck_conn is None:
                if not os.path.exists(self._db_path):
                    return {}
                try:
                    import duckdb
                    self._duck_conn = duckdb.connect(
                        self._db_path, read_only=True)
                except Exception as exc:
                    if not self._duck_error_logged:
                        logger.warning(
                            "[BridgeRecall] DuckDB open failed (%s): %s "
                            "— degrading to cosine-only",
                            self._db_path, exc)
                        self._duck_error_logged = True
                    return {}

            try:
                placeholders = ", ".join(["?"] * len(item_ids))
                rows = self._duck_conn.execute(
                    f"SELECT item_id, base_level FROM activation_state "
                    f"WHERE item_id IN ({placeholders})",
                    list(item_ids),
                ).fetchall()
            except Exception as exc:
                if not self._duck_error_logged:
                    logger.warning(
                        "[BridgeRecall] activation_state query failed: %s "
                        "— degrading to cosine-only", exc)
                    self._duck_error_logged = True
                return {}

        out: dict[str, float] = {}
        for item_id, base_level in rows:
            if base_level is None:
                continue
            try:
                bl = float(base_level)
            except (TypeError, ValueError):
                continue
            # Cold-start sentinel stored in DuckDB as -inf; pass through
            # so composite_score's substitution logic gets the right
            # signal.
            out[item_id] = bl
        return out

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
