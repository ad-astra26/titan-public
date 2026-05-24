"""StandingBundleStore — materialized state for standing contracts.

Synthesis Engine Phase 2 (PLAN_synthesis_engine_Phase2.md §2B, D-P2-4
+ D-P2-5). Owner: synthesis_worker process (sole writer G21 /
INV-Syn-3). Storage: `data/synthesis.duckdb` / `association_bundles`
table (lives in the synthesis-owned DB from P1, NOT memory_worker's
titan_memory.duckdb).

A *standing contract* converts the bundled-entity RECALL from a SEARCH
(arch §12.1) into a READ (single SELECT) by maintaining an
incrementally-updated ring-buffer of recent TX records per
`(entity_class, entity_id, fork)` triple. Each `maintain_bundle` action
fired by the post-seal contract hook (timechain_v2 Mempool/Builder)
calls `.maintain(...)` here.

Cross-process consumers do NOT open synthesis.duckdb (DuckDB 1.5+
exclusive file lock; arch §0 v0.5.0 lesson 1). Instead, this writer
exports `data/bundle_snapshot.json` after each maintain pass and
consumers read via `BridgeRecall.read_bundle(...)` watermark-gated by
synth_status.bin (INV-Syn-4 / G18).

Schema (DuckDB):
    CREATE TABLE association_bundles (
        entity_class TEXT NOT NULL,
        entity_id    TEXT NOT NULL,
        fork         TEXT NOT NULL,
        bundle       BLOB,         -- msgpack list[dict] tx records, LRU-ordered
        last_updated DOUBLE,
        version      INTEGER DEFAULT 1,
        PRIMARY KEY (entity_class, entity_id, fork)
    );

Bundle layout (msgpack): list of TX-record dicts, newest first:
    [{"tx_hash": "<hex>", "epoch_id": int, "ts": float,
      "significance": float, "source": str}, ...]
Ring-buffer eviction: when len(bundle) > ring_size, oldest entries dropped.

Caps (titan_params.toml [synthesis.standing]):
  - user_bundle_max_txs   (default 50): ring-buffer size per entity.
  - user_bundle_max_users (default 2000): hard cap on distinct keys.
The user_bundle_max_users cap is enforced as an LRU eviction over the
`last_updated` column when a new key would push the table over.

Phase 2 ships ONE standing contract (`actr_user_conversation_bundle`,
per-user conversation bundle, D-P2-5). The store is entity-class-
agnostic by design — additional standing contracts in future phases
register new classes without schema change.
"""
from __future__ import annotations

import logging
import json
import os
import threading
import time
from typing import Any, Optional

import duckdb
import msgpack

logger = logging.getLogger(__name__)


# Snapshot file name (sibling of synthesis.duckdb). Mirrors the
# activation_snapshot.json pattern from Phase 1 (arch §3.4 BridgeRecall).
BUNDLE_SNAPSHOT_NAME = "bundle_snapshot.json"

# Schema version stamped on every bundle write — bump if the msgpack
# record shape changes incompatibly.
BUNDLE_SCHEMA_VERSION = 1

# Defaults match titan_params.toml [synthesis.standing] (PLAN §2B).
DEFAULT_RING_SIZE = 50
DEFAULT_MAX_ENTITIES = 2000


class StandingBundleStore:
    """Sole-writer (synthesis_worker process, G21 / INV-Syn-3) persistence
    for association_bundles.

    `.maintain(entity_class, entity_id, fork, tx_record)` is the hot-path
    write — called from synthesis_worker's recv loop on every
    MAINTAIN_BUNDLE bus event. The hot-path goal is <1ms per call so the
    worker can absorb high-rate chat traffic without backpressure.

    `.read(entity_class, entity_id, fork)` is the in-process read API —
    other readers (different processes) use `BridgeRecall.read_bundle()`.

    `.export_snapshot(path)` writes the full {key -> bundle} dict atomically
    (write-tmp + os.replace) so cross-process consumers see a coherent
    snapshot even mid-write.
    """

    def __init__(
        self,
        db_path: str,
        ring_size: int = DEFAULT_RING_SIZE,
        max_entities: int = DEFAULT_MAX_ENTITIES,
    ) -> None:
        self._db_path = db_path
        self._ring_size = max(1, int(ring_size))
        self._max_entities = max(1, int(max_entities))
        # Lock guarding the in-mem cache + DuckDB connection. Hot path is
        # single-threaded (synthesis_worker recv loop) but the recompute
        # daemon thread may also call export_snapshot — so the lock is
        # required.
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS association_bundles (
                entity_class TEXT NOT NULL,
                entity_id    TEXT NOT NULL,
                fork         TEXT NOT NULL,
                bundle       BLOB,
                last_updated DOUBLE,
                version      INTEGER DEFAULT 1,
                PRIMARY KEY (entity_class, entity_id, fork)
            )
            """
        )

        # In-memory cache `(class, id, fork) -> list[record dict]`. Loaded
        # at construction so reads + maintains hit RAM, with DuckDB as the
        # durable backing store. Modest size — bounded by max_entities.
        self._cache: dict[tuple[str, str, str], list[dict]] = {}
        # last_updated mirror — used for LRU eviction without scanning DB.
        self._last_updated: dict[tuple[str, str, str], float] = {}
        self._load_existing()

        # Aggregate stats — surface on Observatory + tests.
        self._total_maintains = 0
        self._total_evictions = 0
        self._total_lru_evictions = 0

    # ── bootstrap ─────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        try:
            rows = self._conn.execute(
                "SELECT entity_class, entity_id, fork, bundle, last_updated "
                "FROM association_bundles"
            ).fetchall()
        except Exception as exc:
            logger.warning(
                "[StandingBundleStore] failed to load existing rows: %s",
                exc, exc_info=True)
            return
        loaded = 0
        for ec, eid, fork, blob, last_updated in rows:
            try:
                bundle = list(msgpack.unpackb(blob, raw=False)) if blob else []
            except Exception:
                logger.warning(
                    "[StandingBundleStore] corrupt bundle (%s,%s,%s) — "
                    "starting fresh", ec, eid, fork)
                bundle = []
            key = (ec, eid, fork)
            self._cache[key] = bundle
            self._last_updated[key] = float(last_updated or 0.0)
            loaded += 1
        if loaded:
            logger.info(
                "[StandingBundleStore] loaded %d bundles from %s",
                loaded, self._db_path)

    # ── hot path ──────────────────────────────────────────────────────

    def maintain(
        self,
        entity_class: str,
        entity_id: str,
        fork: str,
        tx_record: dict,
    ) -> None:
        """Insert `tx_record` at the head of the (class, id, fork) bundle;
        ring-buffer evict if over `ring_size`; LRU-evict entire bundles if
        we'd cross `max_entities`. Idempotent on duplicate tx_hash (in-place
        no-op — most-recent stays at head).
        """
        key = (str(entity_class), str(entity_id), str(fork))
        now = time.time()
        with self._lock:
            bundle = self._cache.get(key)
            if bundle is None:
                # New entity — enforce max_entities cap.
                if len(self._cache) >= self._max_entities:
                    self._evict_lru_locked()
                bundle = []
                self._cache[key] = bundle

            # Idempotency on tx_hash — if the same TX hash already exists,
            # leave the bundle untouched (preserves original recency order;
            # post-seal hook may double-fire on builder retry).
            tx_hash = tx_record.get("tx_hash", "")
            if tx_hash:
                for existing in bundle:
                    if existing.get("tx_hash") == tx_hash:
                        self._last_updated[key] = now
                        self._persist_locked(key, bundle, now)
                        self._total_maintains += 1
                        return

            # Insert newest-first.
            bundle.insert(0, dict(tx_record))
            # Ring-buffer eviction.
            if len(bundle) > self._ring_size:
                bundle[:] = bundle[: self._ring_size]
                self._total_evictions += 1
            self._last_updated[key] = now
            self._persist_locked(key, bundle, now)
            self._total_maintains += 1

    def _persist_locked(
        self, key: tuple[str, str, str], bundle: list[dict], now: float,
    ) -> None:
        """Caller must hold self._lock. Upserts the bundle to DuckDB."""
        ec, eid, fork = key
        blob = msgpack.packb(bundle, use_bin_type=True)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO association_bundles "
                "(entity_class, entity_id, fork, bundle, last_updated, version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ec, eid, fork, blob, now, BUNDLE_SCHEMA_VERSION),
            )
        except Exception as exc:
            logger.warning(
                "[StandingBundleStore] persist failed (%s,%s,%s): %s",
                ec, eid, fork, exc, exc_info=True)

    def _evict_lru_locked(self) -> None:
        """Caller holds self._lock. Removes the single least-recently-
        updated bundle to make room for a new entity. Conservative: only
        one per overflow event (the next maintain triggers next eviction
        if needed)."""
        if not self._last_updated:
            return
        oldest_key = min(self._last_updated.items(), key=lambda kv: kv[1])[0]
        self._cache.pop(oldest_key, None)
        self._last_updated.pop(oldest_key, None)
        ec, eid, fork = oldest_key
        try:
            self._conn.execute(
                "DELETE FROM association_bundles "
                "WHERE entity_class = ? AND entity_id = ? AND fork = ?",
                (ec, eid, fork),
            )
        except Exception as exc:
            logger.warning(
                "[StandingBundleStore] LRU evict DB delete failed: %s", exc)
        self._total_lru_evictions += 1
        logger.info(
            "[StandingBundleStore] LRU-evicted bundle (%s,%s,%s) — "
            "total entities now %d (cap %d)",
            ec, eid, fork, len(self._cache), self._max_entities)

    # ── read ──────────────────────────────────────────────────────────

    def read(
        self, entity_class: str, entity_id: str, fork: str,
    ) -> list[dict]:
        """Return the bundle list (newest-first). Empty list if absent.
        Safe to call across the recompute thread (lock held)."""
        key = (str(entity_class), str(entity_id), str(fork))
        with self._lock:
            bundle = self._cache.get(key, [])
            # Return a shallow copy so caller mutations cannot corrupt the
            # in-memory state.
            return [dict(r) for r in bundle]

    def entities_tracked(self) -> int:
        with self._lock:
            return len(self._cache)

    # ── snapshot export ───────────────────────────────────────────────

    def export_snapshot(self, snapshot_path: str) -> int:
        """Atomic JSON export of the full {composite_key -> bundle} mapping
        for cross-process readers.

        Schema:
            {
              "version": 1,
              "exported_at": <wall-clock seconds>,
              "bundles": {
                 "user|hash1|conversation": [<tx_record>, ...],
                 "user|hash2|conversation": [...],
                 ...
              }
            }
        Composite key separator `|` chosen because entity_id values are
        hashed (no `|` byte) and fork names are alphanumeric.

        Atomic write via tmp + os.replace. Returns the count of bundles
        in the snapshot.
        """
        with self._lock:
            payload = {
                "version": BUNDLE_SCHEMA_VERSION,
                "exported_at": time.time(),
                "bundles": {
                    f"{ec}|{eid}|{fork}": [dict(r) for r in bundle]
                    for (ec, eid, fork), bundle in self._cache.items()
                },
            }
            count = len(payload["bundles"])

        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = snapshot_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp_path, snapshot_path)
        return count

    # ── stats / lifecycle ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Aggregate stats for Observatory / tests."""
        with self._lock:
            return {
                "entities_tracked": len(self._cache),
                "max_entities": self._max_entities,
                "ring_size": self._ring_size,
                "total_maintains": self._total_maintains,
                "total_ring_evictions": self._total_evictions,
                "total_lru_evictions": self._total_lru_evictions,
            }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                logger.exception("[StandingBundleStore] close failed")


__all__ = [
    "StandingBundleStore",
    "BUNDLE_SNAPSHOT_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_RING_SIZE",
    "DEFAULT_MAX_ENTITIES",
]
