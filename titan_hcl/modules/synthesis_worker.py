"""synthesis_worker — Python L2 module, sole writer (G21 / INV-Syn-3) for
the synthesis engine's activation_state table + synth_status.bin SHM
watermark.

Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0 §25 / §9.B `synthesis_worker`
block). Phase 1 of `rFP_outer_memory_enhancement.md §18`.

What this worker owns:
  1. activation_state DuckDB writes — record_access + 60s recompute of B_i
     per arch §5.2 (use-gated reinforcement INV-Syn-5: only items the LLM
     cited / acted upon).
  2. synth_status.bin SHM watermark writer (G21 sole writer) — exposes the
     `last_consistent_event_ts` watermark that cross-process consumers use
     to BridgeRecall-gate their DuckDB+Kuzu reads (INV-Syn-4 / G18).
  3. Plug registries (substrate / truth-oracle / meaning-oracle / proof /
     tool) — instantiated empty in Phase 1; concrete plugs register per
     phase per arch §3.3.
  4. Phase 2: StandingBundleStore writes (G21 / association_bundles), the
     post-seal hook's MAINTAIN_BUNDLE consumer.
  5. Phase 2D: EngineRecall — process-local contract-driven recall
     coordinator. Constructs its own RuleEvaluator (with index.db R/O
     handle + lazy FAISS reader) and exposes a process-local accessor
     so future in-process consumers (agno tool layer, bridge) can use
     contract-driven recall without a sync bus.request (INV-Syn-2 / G19).

Bus subscriptions:
  • MEMORY_RETRIEVAL_USED — use-gated record_access (INV-Syn-5)
  • MAINTAIN_BUNDLE       — Phase 2 standing-contract maintenance event
                            (D-P2-4): post-seal contract hook emits one
                            per sealed TX matching an active contract
                            with action="maintain_bundle"; routed to the
                            StandingBundleStore.maintain() hot path.
  • KERNEL_EPOCH_TICK     — 60s recompute cadence trigger (loose — actual
                            recompute runs on a wall-clock timer in a
                            daemon thread, the tick just nudges it forward
                            on the kernel's heartbeat for observability)
  • MODULE_SHUTDOWN       — clean exit

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • MODULE_HEARTBEAT         — every 30s
  • MODULE_READY             — once at boot (after DuckDB + SHM writer init)
  • SYNTHESIS_RECOMPUTE_DONE — every 60s, observability only

Periodic work pattern: per
`feedback_recv_queue_except_empty_periodic_trap.md` periodic work runs in
SEPARATE daemon threads (heartbeat + recompute), NEVER inside `except
Empty:` after `recv_queue.get(timeout=...)` — the recv loop only handles
inbound messages.

The recompute loop is a no-op until `MEMORY_RETRIEVAL_USED` events arrive
(Phase 1.5+ producers — none today). Worker boots clean either way.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
import time
from queue import Empty
from typing import Any, Optional

import duckdb
import msgpack

from titan_hcl import bus
from titan_hcl.bus import (
    DREAM_STATE_CHANGED,
    KERNEL_EPOCH_TICK,
    MAINTAIN_BUNDLE,
    MEMORY_RETRIEVAL_USED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    SYNTHESIS_RECOMPUTE_DONE,
    make_msg,
)
from titan_hcl.synthesis.activation import (
    DEFAULT_DECAY_D,
    DEFAULT_WINDOW_N,
    ActivationState,
    base_level,
    record_access,
)
from titan_hcl.synthesis.standing_store import (
    BUNDLE_SNAPSHOT_NAME,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_RING_SIZE,
    StandingBundleStore,
)
from titan_hcl.synthesis.recall import EngineRecall

# Process-local EngineRecall singleton (PLAN §2D). Constructed during
# synthesis_worker_main boot; exposed via get_engine_recall() so future
# in-process consumers (e.g. an agno tool wrapper that lives inside this
# worker) can drive contract-driven recall without a sync bus.request
# (INV-Syn-2 / G19). Cross-process consumers must NOT use this — they go
# through BridgeRecall.
_engine_recall_singleton: Optional[EngineRecall] = None
_engine_recall_lock = threading.Lock()


def get_engine_recall() -> Optional[EngineRecall]:
    """Return the process-local EngineRecall instance, or None when this
    process is not the synthesis_worker (the worker is the only writer
    of `_engine_recall_singleton`). Other processes can publish recall
    requests on the bus (Phase 3+ surface) or use BridgeRecall directly
    for the watermark-gated reader path."""
    with _engine_recall_lock:
        return _engine_recall_singleton


def _set_engine_recall(er: Optional[EngineRecall]) -> None:
    """Internal — synthesis_worker_main writes the singleton at boot."""
    global _engine_recall_singleton
    with _engine_recall_lock:
        _engine_recall_singleton = er

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 30.0
RECOMPUTE_INTERVAL_S = 60.0          # arch §5.2 60s recompute cadence
SYNTH_STATUS_SLOT_NAME = "synth_status"     # /dev/shm/titan_<id>/synth_status.bin
SYNTH_STATUS_PAYLOAD_BYTES = 24             # struct '<ddII'

# Cross-process activation snapshot. DuckDB 1.5+ enforces an exclusive
# file lock even for read_only opens, so the writer process (this worker)
# cannot share data/synthesis.duckdb with reader processes (memory_worker
# in particular). Workaround: export a small JSON snapshot of the
# activation_state {item_id -> base_level} mapping after each recompute
# pass. Readers consume the snapshot file (no DB lock), gated by the
# same synth_status.bin watermark for freshness. Snapshot atomic via
# write-tmp + os.replace.
ACTIVATION_SNAPSHOT_NAME = "activation_snapshot.json"

# Phase 4 FU-1 — concept-spine cross-process read surface. synthesis_worker
# exports the full Kuzu spine state to JSON each 60s recompute pass; the api
# process reads this snapshot (NOT the Kuzu file directly) per the same
# pattern as ACTIVATION_SNAPSHOT_NAME + BUNDLE_SNAPSHOT_NAME. Kuzu 0.11's
# read_only=True still acquires the exclusive write lock — cross-process
# reads MUST go through this snapshot.
SPINE_SNAPSHOT_NAME = "spine_snapshot.json"

# Phase 1 plug-registry container — instantiated empty; concrete plugs land
# per phase per arch §3.3.
_PLUG_KINDS = ("substrate", "truth_oracle", "meaning_oracle", "proof", "tool")


# ─────────────────────────────────────────────────────────────────────────
# Plug registry
# ─────────────────────────────────────────────────────────────────────────

class PlugRegistry:
    """Per-kind plug registry. Phase 1 holds nothing — concrete plugs
    register per phase (arch §3.3): Phase 2 = Timechain SubstratePlug;
    Phase 4 = Kuzu/FAISS substrates; Phase 6 = TruthOraclePlug concrete
    (coding_sandbox / solana_rpc / web_api / x_oracle), MeaningOraclePlug
    (CGN), ProofStrategyPlug (Merkle / ZK), ToolPlug (coding_sandbox /
    events_teacher / knowledge / x_research).
    """

    def __init__(self) -> None:
        self._by_kind: dict[str, dict[str, Any]] = {k: {} for k in _PLUG_KINDS}

    def register(self, kind: str, plug_id: str, plug: Any) -> None:
        if kind not in self._by_kind:
            raise ValueError(f"unknown plug kind: {kind!r}; "
                             f"expected one of {_PLUG_KINDS}")
        self._by_kind[kind][plug_id] = plug

    def get(self, kind: str, plug_id: str) -> Optional[Any]:
        return self._by_kind.get(kind, {}).get(plug_id)

    def list(self, kind: str) -> list[str]:
        return list(self._by_kind.get(kind, {}).keys())

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._by_kind.items()}


# ─────────────────────────────────────────────────────────────────────────
# synth_status.bin SHM watermark writer
# ─────────────────────────────────────────────────────────────────────────

class SynthStatusWriter:
    """Sole writer (G21 / INV-Syn-3) for synth_status.bin — the SHM
    watermark cross-process BridgeRecall readers gate every DuckDB+Kuzu
    query on (INV-Syn-4 / G18).

    Payload layout (24 bytes, little-endian, struct '<ddII'):
        f64 last_consistent_event_ts   — most recent monotonic event ts
                                          the synthesis_worker has fully
                                          committed to activation_state
        f64 last_recompute_ts          — last 60s recompute pass timestamp
        u32 items_tracked              — current size of activation_state
        u32 recompute_count            — monotonic recompute pass counter
    """

    def __init__(self, titan_id: str) -> None:
        from titan_hcl.core.state_registry import (
            RegistrySpec,
            StateRegistryWriter,
            ensure_shm_root,
        )
        import numpy as np

        self._spec = RegistrySpec(
            name=SYNTH_STATUS_SLOT_NAME,
            dtype=np.dtype(np.uint8),
            shape=(SYNTH_STATUS_PAYLOAD_BYTES,),
            feature_flag="",                # always-on
            schema_version=1,
            variable_size=True,
        )
        shm_root = ensure_shm_root(titan_id)
        self._writer = StateRegistryWriter(self._spec, shm_root)
        # Boot-state values; published lazily on first publish() call.
        self._last_consistent_event_ts = 0.0
        self._last_recompute_ts = 0.0
        self._items_tracked = 0
        self._recompute_count = 0

    def publish(
        self,
        last_consistent_event_ts: Optional[float] = None,
        last_recompute_ts: Optional[float] = None,
        items_tracked: Optional[int] = None,
        recompute_count_increment: int = 0,
    ) -> None:
        """Update + publish in one call. Any None arg keeps the previous
        value. `recompute_count_increment` is an offset (typically 1)."""
        if last_consistent_event_ts is not None:
            self._last_consistent_event_ts = last_consistent_event_ts
        if last_recompute_ts is not None:
            self._last_recompute_ts = last_recompute_ts
        if items_tracked is not None:
            self._items_tracked = items_tracked
        if recompute_count_increment:
            self._recompute_count += recompute_count_increment

        payload = struct.pack(
            "<ddII",
            self._last_consistent_event_ts,
            self._last_recompute_ts,
            self._items_tracked,
            self._recompute_count,
        )
        assert len(payload) == SYNTH_STATUS_PAYLOAD_BYTES
        self._writer.write_variable(payload)

    def close(self) -> None:
        self._writer.close()


# ─────────────────────────────────────────────────────────────────────────
# activation_state DuckDB persistence
# ─────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Sole-writer DuckDB persistence for activation_state (G21 / INV-Syn-3).

    In-memory cache + lazy persistence: record_access mutates the in-mem
    ActivationState then writes back on each recompute pass (batches access
    log updates to keep the hot path cheap).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # DuckDB locks at the process level: TieredMemoryGraph (in
        # memory_worker process) holds titan_memory.duckdb R/W. If
        # synthesis_worker shared that file, the lock would conflict
        # (different processes → DuckDB v0.8+ rejects). So synthesis_worker
        # owns its OWN file `data/synthesis.duckdb` per G21 / INV-Syn-3 —
        # sole writer of activation_state + (Phase 7) actr_buffers +
        # (Phase 8) procedural_skills. Memory_worker + cross-process
        # readers open this same file R/O via BridgeRecall +
        # _in_process_activation_lookup.
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path)
        # Owner-side schema creation. CREATE TABLE IF NOT EXISTS is
        # idempotent across restarts. Mirrors the schema previously held
        # in titan_hcl/core/direct_memory.py (D-SPEC-123 v1; relocated
        # 2026-05-23 to resolve the cross-worker lock conflict).
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS activation_state (
                item_id TEXT PRIMARY KEY,
                last_access DOUBLE,
                access_log BLOB,
                access_count INTEGER DEFAULT 0,
                first_access DOUBLE,
                base_level DOUBLE DEFAULT 0.0,
                last_recompute DOUBLE DEFAULT 0.0
            )
        """)
        # Cache loaded lazily on first access — read existing rows so the
        # worker resumes ACT-R activation across restarts.
        self._cache: dict[str, ActivationState] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        rows = self._conn.execute(
            "SELECT item_id, last_access, access_log, access_count, "
            "first_access, base_level, last_recompute FROM activation_state"
        ).fetchall()
        for item_id, last_access, log_blob, count, first, base, recompute in rows:
            log: list[float] = []
            if log_blob:
                try:
                    log = list(msgpack.unpackb(log_blob, raw=False))
                except Exception as exc:
                    logger.warning(
                        "[synthesis_worker] corrupt access_log for %s: %s "
                        "(starting fresh)", item_id, exc)
                    log = []
            self._cache[item_id] = ActivationState(
                item_id=item_id,
                last_access=last_access or 0.0,
                access_log=log,
                access_count=count or 0,
                first_access=first or 0.0,
                base_level=base or 0.0,
                last_recompute=recompute or 0.0,
            )
        if self._cache:
            logger.info(
                "[synthesis_worker] loaded %d activation_state rows from %s",
                len(self._cache), self._db_path)

    def record_access(self, item_id: str, ts: float) -> None:
        """Use-gated reinforcement entry point (INV-Syn-5). Called from the
        recv loop on MEMORY_RETRIEVAL_USED."""
        state = self._cache.get(item_id)
        if state is None:
            state = ActivationState(item_id=item_id)
            self._cache[item_id] = state
        record_access(state, ts)

    def recompute_and_persist(
        self,
        now: float,
        d: float = DEFAULT_DECAY_D,
        window_n: int = DEFAULT_WINDOW_N,
    ) -> int:
        """Recompute every cached state's B_i and persist to DuckDB.
        Returns the count of rows actually upserted (states that changed).
        """
        n_touched = 0
        for state in self._cache.values():
            new_bi = base_level(state, now, d=d, window_n=window_n)
            changed = (new_bi != state.base_level
                       or state.last_recompute == 0.0)
            if changed:
                state.base_level = new_bi
                state.last_recompute = now
                self._persist(state)
                n_touched += 1
        return n_touched

    def export_snapshot(self, snapshot_path: str) -> int:
        """Atomic JSON export of {item_id -> base_level} for cross-process
        readers (DuckDB 1.5+ exclusive-lock workaround). Cold-start sentinels
        (-inf, NaN) are filtered out — readers treat absent items as
        cold-start by default. Returns the count of items written.
        """
        import json
        import math
        payload: dict[str, float] = {}
        for state in self._cache.values():
            bl = state.base_level
            if bl is None or not math.isfinite(bl):
                continue
            payload[state.item_id] = bl
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = snapshot_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp_path, snapshot_path)
        return len(payload)

    def _persist(self, state: ActivationState) -> None:
        log_blob = msgpack.packb(state.access_log, use_bin_type=True)
        # base_level may be -inf for cold-start states; DuckDB DOUBLE
        # supports infinities natively, so this round-trips cleanly.
        self._conn.execute(
            "INSERT OR REPLACE INTO activation_state "
            "(item_id, last_access, access_log, access_count, first_access, "
            " base_level, last_recompute) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (state.item_id, state.last_access, log_blob, state.access_count,
             state.first_access, state.base_level, state.last_recompute),
        )

    def items_tracked(self) -> int:
        return len(self._cache)

    def bulk_base_level(self, item_ids: list[str]) -> dict[str, float]:
        """Phase 2 D-P2-1 — bulk activation lookup for EngineRecall's
        composite-score pass. Returns `{item_id: base_level}` ONLY for
        items present in the in-memory cache; absent items are omitted
        (composite_score substitutes cold-start). Pure read; never raises."""
        out: dict[str, float] = {}
        cache = self._cache
        for iid in item_ids:
            state = cache.get(iid)
            if state is None:
                continue
            out[iid] = state.base_level
        return out

    def close(self) -> None:
        self._conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Worker loops
# ─────────────────────────────────────────────────────────────────────────

def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 compliance)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[synthesis_worker] _send %s → %s failed: %s", msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str,
                    stop_event: threading.Event) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s."""
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _recompute_loop(store: "ActivationStore",
                    bundle_store: "StandingBundleStore",
                    status_writer: "SynthStatusWriter",
                    snapshot_path: str,
                    bundle_snapshot_path: str,
                    spine_exporter_holder: dict,
                    send_queue, name: str,
                    interval_s: float,
                    stop_event: threading.Event,
                    cache_lock: threading.Lock) -> None:
    """Daemon thread — 60s recompute of B_i across activation_state + SHM
    watermark publish + standing-bundle snapshot export (arch §5.2 +
    Phase 2 D-P2-4). Separate from the recv loop so a slow DuckDB COMMIT
    never delays inbound MEMORY_RETRIEVAL_USED / MAINTAIN_BUNDLE handling.

    Per feedback_recv_queue_except_empty_periodic_trap.md: periodic work
    MUST run in its own thread, never inside `except Empty:`.
    """
    # Initial settle so DuckDB and SHM are warm before the first pass.
    stop_event.wait(min(interval_s, 10.0))
    pass_count = 0
    error_count = 0
    while not stop_event.is_set():
        t0 = time.monotonic()
        n_touched = 0
        bundles_exported = 0
        try:
            now = time.time()
            with cache_lock:
                n_touched = store.recompute_and_persist(now)
                items = store.items_tracked()
                # Export atomic snapshot for cross-process readers
                # (DuckDB 1.5+ lock workaround — see ACTIVATION_SNAPSHOT_NAME).
                store.export_snapshot(snapshot_path)
            # Bundle snapshot has its own internal lock — no need to hold
            # cache_lock (and shouldn't, since maintain() is a separate
            # write path with shorter critical section).
            bundles_exported = bundle_store.export_snapshot(bundle_snapshot_path)
            # Phase 4 FU-1 — spine snapshot for cross-process api reads.
            # Holder pattern: synthesis_worker_main sets the exporter to
            # ConceptStore.export_snapshot AFTER kuzu_graph_obj is wired;
            # default no-op until that completes.
            try:
                spine_exporter_holder["fn"]()
            except Exception as _spine_exp_err:
                logger.debug(
                    "[synthesis_worker] spine_exporter call failed: %s",
                    _spine_exp_err,
                )
            status_writer.publish(
                last_consistent_event_ts=now,
                last_recompute_ts=now,
                items_tracked=items,
                recompute_count_increment=1,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            _send(send_queue, SYNTHESIS_RECOMPUTE_DONE, name, "all", {
                "items_recomputed": n_touched,
                "items_tracked": items,
                "bundles_exported": bundles_exported,
                "duration_ms": duration_ms,
            })
            pass_count += 1
            if pass_count % 60 == 1:    # ~hourly summary
                logger.info(
                    "[synthesis_worker] recompute pass #%d — items=%d "
                    "touched=%d bundles=%d duration=%dms errors=%d",
                    pass_count, items, n_touched, bundles_exported,
                    duration_ms, error_count)
        except Exception as exc:
            error_count += 1
            logger.warning(
                "[synthesis_worker] recompute pass error (#%d): %s",
                error_count, exc, exc_info=(error_count % 50 == 1))
        # Sleep for the remaining slice of the interval (compensates for
        # the work just done).
        slept = time.monotonic() - t0
        remaining = max(interval_s - slept, 0.1)
        stop_event.wait(remaining)


def synthesis_worker_main(recv_queue, send_queue, name: str,
                          config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id; construct ActivationStore (DuckDB
         activation_state) + SynthStatusWriter (synth_status.bin SHM) +
         PlugRegistry (empty in Phase 1).
      2. Start heartbeat thread (30s) + recompute thread (60s).
      3. Emit MODULE_READY.
      4. Main loop: drain recv_queue, route MEMORY_RETRIEVAL_USED to
         store.record_access (INV-Syn-5), handle MODULE_SHUTDOWN.
    """
    from titan_hcl.core.state_registry import resolve_titan_id

    titan_id = resolve_titan_id(
        (config or {}).get("titan_id") if config else None)

    # DuckDB path — synthesis_worker owns its OWN file (not titan_memory.duckdb)
    # per G21 / INV-Syn-3 to avoid the cross-worker R/W lock conflict
    # (DuckDB v0.8+ rejects two R/W connections to one file across
    # processes). `memory_db_path` config key is honored if explicitly
    # set (for tests / migrations), else defaults to data/synthesis.duckdb.
    db_path = (config or {}).get("memory_db_path") or os.path.join(
        "data", "synthesis.duckdb")

    interval_s = float((config or {}).get("recompute_interval_s",
                                          RECOMPUTE_INTERVAL_S))

    logger.info(
        "[synthesis_worker] booting — titan_id=%s name=%s db=%s "
        "recompute_interval=%.1fs (D-SPEC-123 / SPEC §25)",
        titan_id, name, db_path, interval_s)

    # Construct sole-writer surfaces (G21 / INV-Syn-3).
    store = ActivationStore(db_path)
    status_writer = SynthStatusWriter(titan_id)
    registry = PlugRegistry()

    # Phase 2 D-P2-4 standing-bundle store — sole writer of
    # data/synthesis.duckdb / association_bundles. Shares the synthesis.duckdb
    # file with ActivationStore (same process, separate connections fine —
    # the DuckDB cross-PROCESS lock is the constraint).
    standing_cfg = (config or {}).get("standing") or {}
    ring_size = int(standing_cfg.get(
        "user_bundle_max_txs", DEFAULT_RING_SIZE))
    max_entities = int(standing_cfg.get(
        "user_bundle_max_users", DEFAULT_MAX_ENTITIES))
    bundle_store = StandingBundleStore(
        db_path, ring_size=ring_size, max_entities=max_entities)

    # Cross-process activation snapshot file — sits next to synthesis.duckdb
    # in the same data_dir. Readers consume via plain JSON read.
    snapshot_path = os.path.join(
        os.path.dirname(db_path) or ".", ACTIVATION_SNAPSHOT_NAME)
    bundle_snapshot_path = os.path.join(
        os.path.dirname(db_path) or ".", BUNDLE_SNAPSHOT_NAME)
    # Phase 4 FU-1 — Kuzu Concept-spine cross-process read surface.
    spine_snapshot_path = os.path.join(
        os.path.dirname(db_path) or ".", SPINE_SNAPSHOT_NAME)

    # Lock around the in-mem activation cache for the recv-loop ↔
    # recompute-loop interleave. record_access is cheap; the lock holds
    # only for the mutation itself, never around bus I/O. The
    # StandingBundleStore has its own internal lock so its hot path
    # doesn't contend with activation recompute.
    cache_lock = threading.Lock()

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(send_queue, name, stop_event),
        daemon=True, name=f"synthesis-hb-{name}")
    hb_thread.start()

    # Phase 4 FU-1 — spine exporter holder (late-bound). Default is a
    # no-op so the recompute loop fires safely before the ConceptStore is
    # ready. The consolidation wiring below sets the real exporter.
    spine_exporter_holder: dict = {"fn": lambda: None}

    rc_thread = threading.Thread(
        target=_recompute_loop,
        args=(store, bundle_store, status_writer, snapshot_path,
              bundle_snapshot_path, spine_exporter_holder,
              send_queue, name,
              interval_s, stop_event, cache_lock),
        daemon=True, name=f"synthesis-recompute-{name}")
    rc_thread.start()

    # Phase 4 §P4.A/§P4.H — synthesis spine Kuzu graph (G21 sole writer
    # per INV-Syn-7). Uses its OWN Kuzu file `data/synthesis_spine.kuzu`
    # — distinct from `data/knowledge_graph.kuzu` which memory_worker
    # owns in RW for Person/Topic/Trinity entities. Sharing one Kuzu
    # file across two RW processes triggers Kuzu's exclusive-write-lock
    # rejection (same class of cross-process conflict the Phase 1 lesson
    # 1 solved for DuckDB by giving synthesis_worker its own .duckdb).
    # Cross-process readers open this file with read_only=True (api
    # process for /v6/synthesis/concepts/* — Kuzu 0.11 supports
    # concurrent read-only opens against an active writer).
    kuzu_graph_obj: Optional[Any] = None
    try:
        from titan_hcl.core.direct_memory import TitanKnowledgeGraph
        kuzu_graph_obj = TitanKnowledgeGraph(
            os.path.join(os.path.dirname(db_path) or ".", "synthesis_spine.kuzu"),
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] Kuzu spine graph open failed: %s — spine "
            "recall + consolidation will degrade to no-op",
            exc,
        )

    # Phase 2 D-P2-1 EngineRecall — contract-driven recall coordinator.
    # Lazy-open a read-only sqlite handle on data/timechain/index.db so
    # the RuleEvaluator's FORK_READ + CROSS_REF ops have a substrate.
    # The FAISS reader is NOT wired in Phase 2D scope — production
    # consumers inject a concrete adapter (P2D scope = structure + accessor
    # + contract-driven recall via direct call with mock-friendly handles).
    # If index.db is missing (fresh boot, no chain yet), engine_recall
    # falls back to no-op (returns None on recall) — matches PLAN policy.
    engine_recall: Optional[EngineRecall] = None
    try:
        import sqlite3 as _sqlite3
        index_db_path = os.path.join(
            os.path.dirname(db_path) or ".", "timechain", "index.db")
        index_db_conn = None
        if os.path.exists(index_db_path):
            # Read-only URI open — never writes.
            index_db_conn = _sqlite3.connect(
                f"file:{index_db_path}?mode=ro", uri=True,
                check_same_thread=False, timeout=1.0)
        from titan_hcl.logic.timechain_v2 import RuleEvaluator
        evaluator = RuleEvaluator(
            orchestrator=None,
            faiss_reader=None,    # P2D: caller-injected later (deferred)
            index_db=index_db_conn,
        )
        engine_recall = EngineRecall(
            rule_evaluator=evaluator,
            activation_lookup=store.bulk_base_level,
            embedder=None,        # P2D: lazy-injected by future consumer
            # Phase 4 §P4.H — kuzu_reader for concept-granularity recall.
            kuzu_reader=kuzu_graph_obj,
        )
        _set_engine_recall(engine_recall)
        logger.info(
            "[synthesis_worker] EngineRecall constructed — index_db=%s "
            "faiss=deferred embedder=deferred kuzu_reader=%s",
            "attached" if index_db_conn else "missing",
            "attached" if kuzu_graph_obj is not None else "missing")
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] EngineRecall construction failed: %s — "
            "recall surface will return None (caller falls back)",
            exc, exc_info=True)

    _send(send_queue, MODULE_READY, name, "guardian", {
        "titan_id": titan_id,
        "module": "synthesis_worker",
        "version": "1.2.0",
        "schema_version": 1,
        "spec_ref": "SPEC §25 / D-SPEC-123 + PLAN_synthesis_engine_Phase2.md §2B+2D",
        "plug_counts": registry.counts(),
        "items_tracked_at_boot": store.items_tracked(),
        "bundles_tracked_at_boot": bundle_store.entities_tracked(),
        "engine_recall_ready": engine_recall is not None,
    })
    logger.info(
        "[synthesis_worker] MODULE_READY emitted — items_at_boot=%d "
        "bundles_at_boot=%d plug_counts=%s",
        store.items_tracked(), bundle_store.entities_tracked(),
        registry.counts())

    events_recorded = 0
    bundles_maintained = 0

    # ── Phase 4 §P4.G — dream-boundary consolidation pass wiring ──────
    # Subscribes to DREAM_STATE_CHANGED; on `dreaming=True` transition,
    # runs a ConsolidationPass in a worker thread (does NOT block the bus
    # loop). Rate-limited: at most 1 pass per dream window. LLM provider
    # is best-effort — if inference module is unavailable or unconfigured,
    # the proposer returns all-reject (pass still mines + anchors summary
    # TXs for audit; spine writes simply don't happen until provider lands).
    consolidation_pass: Optional[Any] = None
    # Phase 4 FU-1 — the ConceptStore instance is hoisted into the outer
    # scope so the recompute loop can call concept_store.export_snapshot()
    # each 60s tick regardless of whether ConsolidationPass construction
    # succeeded. Stays None if kuzu_graph_obj is unavailable.
    concept_store: Optional[Any] = None
    last_dream_pass_started_ts: float = 0.0
    consolidation_thread_lock = threading.Lock()
    try:
        from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
        from titan_hcl.synthesis.concept_store import ConceptStore
        from titan_hcl.synthesis.consolidation import (
            ConsolidationPass, LLMProposal,
        )
        from titan_hcl.synthesis.consolidation_defaults import (
            default_mine_recent_txs, make_default_llm_propose,
        )
        # Reuse the kuzu_graph_obj constructed above for EngineRecall.
        # Soft-fail if it's missing: the worker keeps running with
        # consolidation disabled (logged WARN; pass-summary TXs cannot
        # be anchored, but every other synthesis surface still works).
        if kuzu_graph_obj is None:
            raise RuntimeError(
                "kuzu_graph_obj unavailable (synthesis_spine.kuzu open "
                "failed earlier) — consolidation pass disabled this session"
            )
        kuzu_graph = kuzu_graph_obj

        # OuterMemoryWriter wired to the worker's send_queue (every
        # concept-version TX + consolidation_pass TX flows through here).
        from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
        writer = OuterMemoryWriter(
            send_queue=send_queue, src="synthesis_worker",
        )
        cgn_bridge = CGNRegistrationBridge(
            registry_path=os.path.join("data", "synthesis_spine_concepts.json"),
        )
        concept_store = ConceptStore(kuzu_graph, writer)

        # Phase 4 FU-1 — wire the spine_exporter so the recompute loop's
        # next 60s tick starts writing data/spine_snapshot.json. The api
        # process reads this JSON (NOT Kuzu directly — Kuzu 0.11 holds
        # the exclusive lock even for read_only=True opens).
        spine_exporter_holder["fn"] = (
            lambda: concept_store.export_snapshot(spine_snapshot_path)
        )
        # Initial export so the snapshot is non-empty + present before
        # the first 60s tick (api process gets data immediately on first
        # poll after worker boot).
        try:
            initial_n = concept_store.export_snapshot(spine_snapshot_path)
            logger.info(
                "[synthesis_worker] initial spine snapshot exported "
                "(%d concept rows) → %s",
                initial_n, spine_snapshot_path,
            )
        except Exception as _initial_exp_err:
            logger.warning(
                "[synthesis_worker] initial spine snapshot failed: %s",
                _initial_exp_err,
            )

        # LLM provider — best-effort. Falls back to all-reject proposer
        # when the inference module isn't importable / configured.
        propose_fn = None
        try:
            from titan_hcl import inference as _inference_mod
            # Resolve provider via the existing get_provider surface; the
            # specific model + cfg are part of the broader inference setup
            # this worker doesn't own. If provider construction fails the
            # proposer falls through to the no-op.
            provider = getattr(_inference_mod, "get_default_provider", None)
            if callable(provider):
                p = provider()  # may raise if not configured
                propose_fn = make_default_llm_propose(p)
        except Exception as e:
            logger.info(
                "[synthesis_worker] LLM proposer unavailable at boot (%s) — "
                "consolidation will run with all-reject proposer until "
                "provider is wired; pass-summary TXs still anchored",
                e,
            )
        if propose_fn is None:
            def propose_fn(_cluster) -> "LLMProposal":  # type: ignore[no-redef]
                return LLMProposal(
                    action="reject", reason="llm_proposer_unconfigured",
                )

        consolidation_pass = ConsolidationPass(
            concept_store=concept_store,
            cgn_bridge=cgn_bridge,
            outer_memory_writer=writer,
            mine_recent_txs_fn=default_mine_recent_txs,
            llm_propose_fn=propose_fn,
        )
        logger.info(
            "[synthesis_worker] ConsolidationPass ready — DREAM_STATE_CHANGED "
            "subscription active; rate-limit = 1 pass / dream window",
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] ConsolidationPass construction failed: %s — "
            "dream-boundary consolidation will be a no-op this session",
            exc, exc_info=True,
        )

    def _maybe_run_consolidation_async(dream_start_ts: float) -> None:
        """Fire a ConsolidationPass in a worker thread; never blocks the
        bus loop. Rate-limited by dream window — second DREAM_STATE_CHANGED
        within the same window is a no-op."""
        nonlocal last_dream_pass_started_ts
        if consolidation_pass is None:
            return
        with consolidation_thread_lock:
            # Rate-limit: at most 1 pass per dream-start timestamp. A new
            # dream window must arrive (last_dream_pass_started_ts differs)
            # before another pass fires.
            if dream_start_ts <= last_dream_pass_started_ts:
                logger.debug(
                    "[synthesis_worker] consolidation already ran for "
                    "dream window %.3f — skipping",
                    dream_start_ts,
                )
                return
            last_dream_pass_started_ts = dream_start_ts

        def _run():
            try:
                result = consolidation_pass.run()
                logger.info(
                    "[synthesis_worker] consolidation_pass %s done — "
                    "created=%d bumped=%d rejected=%d llm_calls=%d "
                    "txs_mined=%d duration_ms=%.1f",
                    result.pass_id,
                    len(result.concepts_created),
                    len(result.concepts_bumped),
                    result.rejected_clusters,
                    result.llm_calls,
                    result.txs_mined,
                    result.duration_ms,
                )
            except Exception as e:
                logger.warning(
                    "[synthesis_worker] consolidation_pass crashed: %s",
                    e, exc_info=True,
                )

        threading.Thread(
            target=_run, name="synthesis-consolidation",
            daemon=True,
        ).start()

    try:
        while True:
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                # IMPORTANT: do NOT do periodic work here. Periodic work
                # runs in the recompute_loop daemon thread per
                # feedback_recv_queue_except_empty_periodic_trap.md.
                continue

            if msg is None:
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            if msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[synthesis_worker] MODULE_SHUTDOWN received — exiting "
                    "(events_recorded=%d items_tracked=%d "
                    "bundles_maintained=%d bundles_tracked=%d)",
                    events_recorded, store.items_tracked(),
                    bundles_maintained, bundle_store.entities_tracked())
                break

            if msg_type == MEMORY_RETRIEVAL_USED:
                item_id = payload.get("item_id")
                ts = payload.get("ts")
                if not isinstance(item_id, str) or not isinstance(ts, (int, float)):
                    continue
                with cache_lock:
                    store.record_access(item_id, float(ts))
                events_recorded += 1
                continue

            if msg_type == MAINTAIN_BUNDLE:
                # Phase 2 D-P2-4 standing-contract maintenance event.
                # Payload shape (see titan_hcl/bus.py MAINTAIN_BUNDLE doc):
                #   {entity_class, entity_id, fork, tx_hash, epoch_id, ts,
                #    significance?, source?}
                entity_class = payload.get("entity_class")
                entity_id = payload.get("entity_id")
                fork = payload.get("fork")
                tx_hash = payload.get("tx_hash")
                if not all(isinstance(x, str) and x for x in
                           (entity_class, entity_id, fork, tx_hash)):
                    logger.debug(
                        "[synthesis_worker] MAINTAIN_BUNDLE missing required "
                        "field(s); payload=%s", payload)
                    continue
                tx_record = {
                    "tx_hash": tx_hash,
                    "epoch_id": int(payload.get("epoch_id", 0)),
                    "ts": float(payload.get("ts", time.time())),
                    "significance": float(payload.get("significance", 0.0)),
                    "source": str(payload.get("source", "")),
                }
                try:
                    bundle_store.maintain(
                        entity_class, entity_id, fork, tx_record)
                    bundles_maintained += 1
                except Exception as exc:
                    logger.warning(
                        "[synthesis_worker] bundle_store.maintain failed: %s",
                        exc, exc_info=True)
                continue

            if msg_type == DREAM_STATE_CHANGED:
                # Phase 4 §P4.G — dream-boundary consolidation listener.
                # On dreaming=True transition fire one ConsolidationPass
                # in a worker thread (rate-limited by dream window so a
                # noisy publisher can't blow the LLM budget). On
                # dreaming=False: log only — the pass running in the
                # background must finish on its own.
                dreaming = bool(payload.get("dreaming", False))
                if dreaming:
                    dream_start_ts = float(
                        payload.get("ts", time.time())
                    )
                    logger.info(
                        "[synthesis_worker] DREAM_STATE_CHANGED dreaming=True "
                        "ts=%.3f — scheduling consolidation pass",
                        dream_start_ts,
                    )
                    _maybe_run_consolidation_async(dream_start_ts)
                else:
                    logger.debug(
                        "[synthesis_worker] DREAM_STATE_CHANGED dreaming=False "
                        "— pass continues in background if running",
                    )
                continue

            # KERNEL_EPOCH_TICK + everything else: no-op in Phase 1. The
            # recompute_loop drives on wall-clock, not on this tick.

    finally:
        stop_event.set()
        _set_engine_recall(None)
        try:
            store.close()
        except Exception:
            logger.exception("[synthesis_worker] store close failed")
        try:
            bundle_store.close()
        except Exception:
            logger.exception("[synthesis_worker] bundle_store close failed")
        try:
            status_writer.close()
        except Exception:
            logger.exception("[synthesis_worker] status_writer close failed")
        logger.info("[synthesis_worker] shutdown complete")
