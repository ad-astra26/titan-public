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
  • (MODULE_READY retired — Phase 11 §11.I.2 SHM slot state=booted is the contract)
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
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    SYNTHESIS_FORK_COMMAND,
    SYNTHESIS_FORK_COMMAND_RESULT,
    SYNTHESIS_BUFFER_COMMAND,
    SYNTHESIS_RECOMPUTE_DONE,
    USER_FEEDBACK_SIGNAL,
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
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

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


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel.
# Flipped True after ActivationStore + StandingBundleStore + EngineRecall init
# complete. Gates SHM-slot heartbeat (see _heartbeat_loop) so titan_hcl's
# 1Hz SHM poll + MODULE_PROBE_REQUEST dispatcher see real liveness.
_WORKER_READY: bool = False


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

# Phase 5 §P5.I — hypothesis-fork cross-process read surface. Mirrors
# SPINE_SNAPSHOT_NAME (Phase 4 FU-1) for the same reason: synthesis_worker
# is the sole writer (INV-Syn-8) to hypothesis_forks DuckDB + HypothesisFork
# Kuzu nodes; the api process reads this JSON snapshot, never the DBs
# directly.
FORKS_SNAPSHOT_NAME = "forks_snapshot.json"

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
                    stop_event: threading.Event,
                    state_writer: Optional[Any] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s.

    Phase 11 §11.I.5 (Chunk 11N): also publishes
    ModuleStateWriter.heartbeat() on the SHM slot when `state_writer` is
    provided so guardian_hcl's SHM-staleness detector + observatory
    `/v6/readiness` see fresh data. SHM heartbeat suppressed while
    `_WORKER_READY` is False so the slot stays in "starting"/"booted"
    until the recv-loop probe handler transitions it to "running".
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        if state_writer is not None and _WORKER_READY:
            try:
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _recompute_loop(store: "ActivationStore",
                    bundle_store: "StandingBundleStore",
                    status_writer: "SynthStatusWriter",
                    snapshot_path: str,
                    bundle_snapshot_path: str,
                    spine_exporter_holder: dict,
                    fork_exporter_holder: dict,
                    fork_activation_updater_holder: dict,
                    oracle_exporter_holder: dict,
                    metrics_exporter_holder: dict,
                    tx_index_holder: dict,
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
            # Phase 5 §P5.C — push current B_i for every open fork into
            # hypothesis_forks.activation so find_below_floor() reflects
            # live activation. The updater is late-bound at fork-store
            # wire time (default no-op so the loop fires before P5 wiring).
            try:
                fork_activation_updater_holder["fn"]()
            except Exception as _fork_act_err:
                logger.debug(
                    "[synthesis_worker] fork_activation_updater call failed: "
                    "%s", _fork_act_err,
                )
            # Phase 5 §P5.I — forks_snapshot.json for cross-process api reads.
            try:
                fork_exporter_holder["fn"]()
            except Exception as _fork_exp_err:
                logger.debug(
                    "[synthesis_worker] fork_exporter call failed: %s",
                    _fork_exp_err,
                )
            # Phase 6 §P6.K — oracles_snapshot.json for cross-process api reads.
            try:
                oracle_exporter_holder["fn"]()
            except Exception as _ora_exp_err:
                logger.debug(
                    "[synthesis_worker] oracle_exporter call failed: %s",
                    _ora_exp_err,
                )
            # Phase 10 §P10.B — synthesis_metrics_snapshot.json (observation-only).
            try:
                metrics_exporter_holder["fn"]()
            except Exception as _met_exp_err:
                logger.debug(
                    "[synthesis_worker] metrics_exporter call failed: %s",
                    _met_exp_err,
                )
            # Operator-closure Phase A2 — incrementally index new chain TXs into
            # the tx_hash-native FAISS shards (bounded per tick; no-op until the
            # builder is wired). Keeps SEARCH current with live chain growth.
            try:
                tx_index_holder["fn"]()
            except Exception as _tix_exp_err:
                logger.debug(
                    "[synthesis_worker] tx_index tick failed: %s",
                    _tix_exp_err,
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


@with_error_envelope(module_name="synthesis", subsystem="entry", severity=_phase11_sev.FATAL)
def synthesis_worker_main(recv_queue, send_queue, name: str,
                          config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id; construct ActivationStore (DuckDB
         activation_state) + SynthStatusWriter (synth_status.bin SHM) +
         PlugRegistry (empty in Phase 1).
      2. Start heartbeat thread (30s) + recompute thread (60s).
      3. Phase 11 §11.I.2 — transition SHM slot starting→booted (no MODULE_READY emit).
      4. Main loop: drain recv_queue, route MEMORY_RETRIEVAL_USED to
         store.record_access (INV-Syn-5), handle MODULE_SHUTDOWN.
    """
    # Phase 11 §11.I.5 (Chunk 11N) — readiness flag reset per entry.
    global _WORKER_READY
    _WORKER_READY = False

    from titan_hcl.core.state_registry import resolve_titan_id

    titan_id = resolve_titan_id(
        (config or {}).get("titan_id") if config else None)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21 per worker) ──
    # Constructed BEFORE the slow DuckDB + Kuzu + EngineRecall init so the
    # slot publishes state="starting" immediately. The heartbeat thread
    # (started below) calls state_writer.heartbeat() every 30s so
    # guardian_hcl's SHM-staleness detector doesn't kill the worker mid-boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name="synthesis",
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[synthesis_worker] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy path): %s", _sw_err)

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
    # Phase 5 §P5.I — hypothesis-forks cross-process read surface.
    forks_snapshot_path = os.path.join(
        os.path.dirname(db_path) or ".", FORKS_SNAPSHOT_NAME)

    # Lock around the in-mem activation cache for the recv-loop ↔
    # recompute-loop interleave. record_access is cheap; the lock holds
    # only for the mutation itself, never around bus I/O. The
    # StandingBundleStore has its own internal lock so its hot path
    # doesn't contend with activation recompute.
    cache_lock = threading.Lock()

    stop_event = threading.Event()
    # Phase 11 §11.I.5 — pass state_writer so heartbeat thread mirrors
    # MODULE_HEARTBEAT to the SHM slot once _WORKER_READY flips True.
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer),
        daemon=True, name=f"synthesis-hb-{name}")
    hb_thread.start()

    # Phase 4 FU-1 — spine exporter holder (late-bound). Default is a
    # no-op so the recompute loop fires safely before the ConceptStore is
    # ready. The consolidation wiring below sets the real exporter.
    spine_exporter_holder: dict = {"fn": lambda: None}
    # Phase 5 §P5.I — forks exporter + activation updater holders
    # (same late-bound pattern; HypothesisForkStore wires real callables
    # once it has been constructed; both default to no-op).
    fork_exporter_holder: dict = {"fn": lambda: None}
    fork_activation_updater_holder: dict = {"fn": lambda: None}
    # Phase 6 §P6.K — oracle snapshot exporter holder (same late-bound
    # pattern; Phase 6 wiring sets the real exporter callable once
    # OracleRouter/SpendStore/GateConfig are ready).
    oracle_exporter_holder: dict = {"fn": lambda: None}
    # Phase 10 §P10.B — metrics snapshot exporter holder (late-bound; set to
    # MetricsAggregator.export once the meter + aggregator are constructed).
    metrics_exporter_holder: dict = {"fn": lambda: None}
    # Operator-closure Phase A2 — tx_hash-index incremental builder holder
    # (late-bound; set once the SynthesisVectorStore + TxIndexBuilder are wired
    # below). Default no-op so the recompute loop fires safely before wiring.
    tx_index_holder: dict = {"fn": lambda: None}

    # Operator-closure Phase A — ONE shared embedder for the whole worker (the
    # tx_hash FAISS store, EngineRecall's query embed, the consolidation cosine
    # path, AND skill_store) so only a single fastembed model is resident (RSS
    # discipline per feedback_eager_init_needs_rss_root_cause_first — lazy-loads
    # on first embed, never at boot).
    _data_dir_sw = os.path.dirname(db_path) or "."

    def _shared_embedder(text: str):
        try:
            from fastembed import TextEmbedding
            import numpy as np
            if not hasattr(_shared_embedder, "_model"):
                _shared_embedder._model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5")
            vecs = list(_shared_embedder._model.embed([text]))
            v = np.array(vecs[0], dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            return v
        except Exception as e:
            logger.debug("[synthesis_worker] shared_embedder failed: %s", e)
            return None

    # Operator-closure Phase A2 — the tx_hash-native FAISS store (sole writer,
    # G21). Binds outer-memory vectors to the chain by the canonical block_hash
    # (arch §3.6 / INV-15). Serves as the in-process faiss_reader for this
    # worker's EngineRecall (so SEARCH finally returns hits) + the by-tx_hash
    # vector source for ConsolidationPass clustering (W4).
    synth_vector_store = None
    try:
        from titan_hcl.synthesis.synthesis_vector_index import SynthesisVectorStore
        synth_vector_store = SynthesisVectorStore(
            data_dir=_data_dir_sw, embedder=_shared_embedder)
        logger.info(
            "[synthesis_worker] tx_hash FAISS store ready (forks=%s) — "
            "binding outer memory to the chain spine (INV-15)",
            list(synth_vector_store.stats().keys()))
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] tx_hash FAISS store wiring failed: %s — "
            "SEARCH stays cold (recall falls back to FORK_READ/CROSS_REF)", exc)
        synth_vector_store = None

    rc_thread = threading.Thread(
        target=_recompute_loop,
        args=(store, bundle_store, status_writer, snapshot_path,
              bundle_snapshot_path, spine_exporter_holder,
              fork_exporter_holder, fork_activation_updater_holder,
              oracle_exporter_holder, metrics_exporter_holder,
              tx_index_holder,
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
            # Operator-closure Phase A: the tx_hash-native FAISS store IS the
            # faiss_reader the SEARCH op always needed (arch §12.1 / INV-15).
            # No longer deferred — SEARCH returns real hits in-process.
            faiss_reader=synth_vector_store,
            index_db=index_db_conn,
        )
        engine_recall = EngineRecall(
            rule_evaluator=evaluator,
            activation_lookup=store.bulk_base_level,
            embedder=_shared_embedder,   # operator-closure: was deferred (B1)
            # Phase 4 §P4.H — kuzu_reader for concept-granularity recall.
            kuzu_reader=kuzu_graph_obj,
        )
        _set_engine_recall(engine_recall)
        logger.info(
            "[synthesis_worker] EngineRecall constructed — index_db=%s "
            "faiss=%s embedder=wired kuzu_reader=%s",
            "attached" if index_db_conn else "missing",
            "tx_hash_store" if synth_vector_store is not None else "none",
            "attached" if kuzu_graph_obj is not None else "missing")

        # Operator-closure Phase A2 — wire the incremental tx-index builder onto
        # the recompute-loop holder. Each 60s tick indexes new conversation/
        # declarative/procedural blocks since the watermark (bounded), keeping
        # the tx_hash spine current with live chain growth. Read-only on the
        # chain; the store's atomic save makes the new vectors visible to
        # cross-process FaissReaders (agno/cognitive).
        if synth_vector_store is not None and index_db_conn is not None:
            try:
                from titan_hcl.synthesis.tx_index_builder import TxIndexBuilder
                _tx_index_builder = TxIndexBuilder(
                    store=synth_vector_store,
                    data_dir=_data_dir_sw,
                    index_db=index_db_conn,
                )

                def _tx_index_tick() -> None:
                    summary = _tx_index_builder.run(max_blocks=2000)
                    if summary.get("indexed"):
                        logger.info(
                            "[synthesis_worker] tx-index tick: +%d vectors "
                            "(scanned=%d) shards=%s",
                            summary["indexed"], summary["scanned"],
                            synth_vector_store.stats())

                tx_index_holder["fn"] = _tx_index_tick
                logger.info("[synthesis_worker] tx-index incremental builder wired")
            except Exception as _tix_err:
                logger.warning(
                    "[synthesis_worker] tx-index builder wiring failed: %s — "
                    "index stays at backfill state (no incremental growth)",
                    _tix_err)
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] EngineRecall construction failed: %s — "
            "recall surface will return None (caller falls back)",
            exc, exc_info=True)

    # Phase 11 §11.I.2 — slot transition: starting → booted (D-SPEC-141 / v1.65.0).
    # MODULE_READY bus emit DELETED per locked D2 (no shim, no dual-publish).
    # SHM slot is the contract; titan_hcl's 1Hz poll detects "booted" and
    # dispatches MODULE_PROBE_REQUEST → handler below transitions slot to "running".
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[synthesis_worker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl) — "
                "items_at_boot=%d bundles_at_boot=%d plug_counts=%s",
                store.items_tracked(), bundle_store.entities_tracked(),
                registry.counts())
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[synthesis_worker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    events_recorded = 0      # used_by_llm=True → record_access (INV-Syn-23)
    events_surfaced = 0      # used_by_llm=False → surfaced-not-cited telemetry
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

        # Phase 4 FU-2 — Ollama Cloud LLM proposer for ConsolidationPass.
        # The synthesis worker's ModuleSpec config carries the [inference]
        # block; we construct an OllamaCloudProvider via the existing
        # factory + wrap it with make_default_llm_propose. Best-effort —
        # missing API key or import failure degrades to the all-reject
        # proposer so the pass still runs (mines TXs + anchors summary
        # for audit), just produces no spine writes.
        propose_fn = None
        inference_cfg = (config or {}).get("inference", {}) or {}
        try:
            from titan_hcl.inference import get_provider as _get_provider
            api_key = inference_cfg.get("ollama_cloud_api_key", "") or ""
            if api_key:
                provider = _get_provider("ollama_cloud", inference_cfg)
                propose_fn = make_default_llm_propose(provider)
                model = inference_cfg.get(
                    "ollama_cloud_model", "") or "default"
                logger.info(
                    "[synthesis_worker] Ollama Cloud LLM proposer wired "
                    "(model=%s) — ConsolidationPass will make real "
                    "NEW/VERSION_BUMP/REJECT verdicts",
                    model,
                )
            else:
                logger.info(
                    "[synthesis_worker] [inference] ollama_cloud_api_key "
                    "missing — ConsolidationPass falls back to all-reject "
                    "proposer (pass summary TXs still anchored)",
                )
        except Exception as e:
            logger.warning(
                "[synthesis_worker] Ollama Cloud provider construction "
                "failed (%s) — ConsolidationPass falls back to all-reject "
                "proposer until provider is fixed",
                e,
            )
        if propose_fn is None:
            def propose_fn(_cluster) -> "LLMProposal":  # type: ignore[no-redef]
                return LLMProposal(
                    action="reject", reason="llm_proposer_unconfigured",
                )

        # Operator-closure Phase B4 (W4) — real embeddings into consolidation.
        # default_mine_recent_txs returns embedding=None (the "FAISS fetch is
        # Phase 7" TODO, never done) → tag-only clustering → no real concepts
        # ever formed. Fill each mined candidate's embedding from the tx_hash
        # FAISS store (Phase A) so ConsolidationPass clusters by COSINE (0.85)
        # AND tags, not tags alone — the precondition for real concept synthesis.
        def _mine_with_embeddings(since_ts, exclude_forks):
            cands = default_mine_recent_txs(since_ts, exclude_forks)
            if synth_vector_store is None:
                return cands
            for c in cands:
                if c.embedding is None:
                    vec = synth_vector_store.get_vector(c.fork, c.tx_hash)
                    if vec is not None:
                        c.embedding = tuple(float(x) for x in vec)
            return cands

        consolidation_pass = ConsolidationPass(
            concept_store=concept_store,
            cgn_bridge=cgn_bridge,
            outer_memory_writer=writer,
            mine_recent_txs_fn=_mine_with_embeddings,
            llm_propose_fn=propose_fn,
        )
        logger.info(
            "[synthesis_worker] ConsolidationPass ready — DREAM_STATE_CHANGED "
            "subscription active; rate-limit = 1 pass / dream window; "
            "embeddings=%s (cosine clustering)",
            "tx_hash_store" if synth_vector_store is not None else "tag-only",
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] ConsolidationPass construction failed: %s — "
            "dream-boundary consolidation will be a no-op this session",
            exc, exc_info=True,
        )

    # ── Phase 5 §P5.A–P5.H — HypothesisForkStore + ForkGC wiring ────────
    # Constructed AFTER ConceptStore / OuterMemoryWriter are wired so it
    # can reuse them for graduation paths (INV-10 / INV-Syn-11). Soft-fail
    # if any dependency is missing: forks become a no-op for the session,
    # synthesis_worker keeps running.
    hypothesis_fork_store: Optional[Any] = None
    fork_gc: Optional[Any] = None
    try:
        if (kuzu_graph_obj is None
                or concept_store is None
                or 'writer' not in locals()):
            raise RuntimeError(
                "missing dependency: kuzu_graph / concept_store / writer "
                "not wired — hypothesis-fork lifecycle disabled this session"
            )
        from titan_hcl.synthesis.hypothesis_fork_store import (
            HypothesisForkStore,
        )
        from titan_hcl.synthesis.fork_gc import ForkGC

        hypothesis_fork_store = HypothesisForkStore(
            duckdb_conn=store._conn,    # same DuckDB conn as ActivationStore
            kuzu_graph=kuzu_graph_obj,
            concept_store=concept_store,
            outer_memory_writer=writer,
            activation_store=store,      # ActivationStore from above
            # P8.X (D-SPEC-PHASE8 fold-in): write-through snapshot path so
            # every create/record/graduate/abandon synchronously refreshes
            # forks_snapshot.json. Closes the "new fork visible in snapshot
            # never appeared after 6s" P5 cascade flake. The 60s recompute-
            # loop snapshot stays as a heartbeat but is no longer load-
            # bearing for visibility.
            snapshot_path=forks_snapshot_path,
        )

        # Wire forks snapshot exporter into the recompute loop's late-bind
        # slot — the next 60s tick begins writing forks_snapshot.json.
        fork_exporter_holder["fn"] = (
            lambda: hypothesis_fork_store.export_snapshot(forks_snapshot_path)
        )
        # Initial export so the snapshot is non-empty/present immediately.
        try:
            initial_forks = hypothesis_fork_store.export_snapshot(
                forks_snapshot_path,
            )
            logger.info(
                "[synthesis_worker] initial forks snapshot exported "
                "(%d fork rows) → %s",
                initial_forks, forks_snapshot_path,
            )
        except Exception as _initial_fexp_err:
            logger.warning(
                "[synthesis_worker] initial forks snapshot failed: %s",
                _initial_fexp_err,
            )

        # Wire per-tick fork-activation pusher: for every open fork, look
        # up its current B_i in the ActivationStore cache and persist via
        # hypothesis_fork_store.update_activation. Cheap because the open-
        # fork count is bounded (active probationary set is small —
        # graduation/abandonment removes them).
        def _push_fork_activations() -> None:
            if hypothesis_fork_store is None:
                return
            with cache_lock:
                fork_ids = [
                    f.fork_id for f in hypothesis_fork_store.list_active()
                ]
                bi_map = store.bulk_base_level(
                    [f"fork:{fid}" for fid in fork_ids]
                )
            for fid in fork_ids:
                bi = bi_map.get(f"fork:{fid}")
                if bi is None:
                    continue
                hypothesis_fork_store.update_activation(fid, bi)
        fork_activation_updater_holder["fn"] = _push_fork_activations

        fork_gc = ForkGC(
            fork_store=hypothesis_fork_store,
            synthesis_duckdb_conn=store._conn,
            kuzu_graph=kuzu_graph_obj,
            activation_store=store,
            memory_db_conn=None,   # Phase 5 v1: synthesis.duckdb scope only
        )
        logger.info(
            "[synthesis_worker] HypothesisForkStore + ForkGC ready — "
            "DREAM_STATE_CHANGED subscription will trigger nightly sweep",
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] HypothesisForkStore wiring failed: %s — "
            "hypothesis-fork lifecycle disabled this session",
            exc, exc_info=True,
        )

    # Sweep mode: per Maker decision 2026-05-27 + PLAN §P5.H, first 24h
    # of T3 soak runs in dry-run. Production behavior is set by the
    # `phase5.fork_gc_live` config flag (default False; flip after soak).
    fork_gc_live_mode: bool = bool(
        (config or {}).get("synthesis", {}).get("fork_gc_live", False)
    )

    # ── Phase 9 P9.D — skill-outcome sink (late-bound) ──────────────────
    # The ToolPlugs (built in the Phase 6 block below) need a skill-outcome
    # callback, but skill_store + SkillFailureTracker are built later (Phase 8
    # block). Use a late-bound holder the sink closes over; fill it after the
    # Phase 8 block constructs the store + tracker. When unfilled (skill store
    # disabled), the sink is a no-op. Closes the P8 increment_* gap + drives
    # repair-fork-on-failure (§9.3 / INV-Syn-24 not — this is the failure loop).
    _skill_outcome_holder: dict = {"store": None, "tracker": None, "meter": None}

    def _skill_outcome_sink(skill_id: str, success: bool) -> None:
        _st = _skill_outcome_holder.get("store")
        _tr = _skill_outcome_holder.get("tracker")
        _mt = _skill_outcome_holder.get("meter")
        if _st is not None:
            try:
                if success:
                    _st.increment_success(skill_id)
                else:
                    _st.increment_failure(skill_id)
            except Exception as _se:
                logger.debug("[synthesis_worker] skill utility update failed: %s", _se)
        if _tr is not None:
            try:
                _tr.record_outcome(skill_id, success=success)
            except Exception as _te:
                logger.debug("[synthesis_worker] failure tracker failed: %s", _te)
        # Phase 10 — a successful delegated skill is a recall-satisfied moment.
        if _mt is not None and success:
            try:
                _mt.record_knowledge_moment()
                _mt.record_recall_satisfied(kind="skill_delegation")
            except Exception:
                pass

    # ── Phase 6 §P6.A–K — Oracle middleware + proof middleware + CGN ─────
    # MeaningOraclePlug + 4 ToolPlugs + coverage analyzer + snapshot
    # exporter. Soft-fail: if any dependency is missing the worker keeps
    # running with Phase 6 disabled this session (matches the P5 pattern).
    oracle_router: Optional[Any] = None
    proof_registry: Optional[Any] = None
    cgn_meaning_oracle: Optional[Any] = None
    oracle_snapshot_exporter: Optional[Any] = None
    try:
        if 'writer' not in locals() or writer is None:
            raise RuntimeError(
                "missing dependency: OuterMemoryWriter not wired — "
                "Phase 6 oracle/proof middleware disabled this session"
            )
        from titan_hcl.synthesis.oracle_gate import (
            OracleGate, build_gate_config, ensure_oracle_daily_spend_table,
            zk_privacy_domains,
        )
        from titan_hcl.synthesis.oracle_router import (
            OracleRouter, OracleSpendStore,
        )
        from titan_hcl.synthesis.oracle_coverage import CoverageAnalyzer
        from titan_hcl.synthesis.oracle_snapshot import OracleSnapshotExporter
        from titan_hcl.synthesis.proofs.merkle_proof import MerkleProofStrategy
        from titan_hcl.synthesis.proofs.zk_proof import ZKProofStrategy
        from titan_hcl.synthesis.proofs.registry import ProofStrategyRegistry
        from titan_hcl.synthesis.cgn_meaning_oracle import CGNMeaningOracle
        from titan_hcl.synthesis.oracles.coding_sandbox_oracle import (
            CodingSandboxOracle,
        )
        from titan_hcl.synthesis.oracles.solana_rpc_oracle import SolanaRpcOracle
        from titan_hcl.synthesis.oracles.web_api_oracle import WebApiOracle
        from titan_hcl.synthesis.oracles.x_oracle import XOracle
        from titan_hcl.synthesis.tools.coding_sandbox_tool import (
            CodingSandboxTool,
        )
        from titan_hcl.synthesis.tools.events_teacher_tool import (
            EventsTeacherTool,
        )
        from titan_hcl.synthesis.tools.knowledge_tool import KnowledgeTool
        from titan_hcl.synthesis.tools.x_research_tool import XResearchTool

        # Merged config (the worker received `config` from the kernel).
        gate_config = build_gate_config(config or {})
        gate = OracleGate(gate_config)

        # Spend store shares the synthesis_worker's existing DuckDB
        # connection (INV-Syn-3 sole writer).
        ensure_oracle_daily_spend_table(store._conn)
        spend_store = OracleSpendStore(store._conn)

        # Balance provider — best-effort; gate_config.balance_sol_baseline
        # ensures sensible behavior when balance lookup degrades. The
        # synthesis_worker is a subprocess so we cannot directly reach
        # TitanHCL.network — use a config-supplied callable or fall back
        # to a static 1.0 (= baseline; admit_score = importance).
        def _balance_lookup() -> float:
            try:
                # network_state.bin carries balance_sol (G18 SHM read);
                # placeholder static for now — synthesis_worker boot
                # extension can wire ShmReaderBank in a follow-up.
                return float(
                    (config or {}).get("synthesis", {}).get("balance_sol_fallback", 1.0)
                )
            except Exception:
                return 1.0

        oracle_router = OracleRouter(
            gate=gate,
            spend_store=spend_store,
            outer_memory_writer=writer,
            balance_provider=_balance_lookup,
        )

        # Proof strategy registry — Merkle default; ZK injected via the
        # ZK Vault commit/verify functions. Phase 6 v1 leaves the ZK
        # callables as no-ops (they raise if invoked); ZK fires only
        # when INV-Syn-14 triggers AND the worker has wired the ZK
        # Vault submitter via a follow-up integration commit.
        proof_registry = ProofStrategyRegistry(
            merkle=MerkleProofStrategy(),
            zk=ZKProofStrategy(),
            privacy_domains=zk_privacy_domains(config or {}),
        )

        # CGN meaning oracle — concept_reader bound to ConceptStore;
        # cgn_grounder reserved for the bus-RPC follow-up (returns None
        # for now → degraded grounding per the P6.H defensive contract).
        def _concept_reader(concept_id: str, version: int):
            if concept_store is None:
                return None
            try:
                # ConceptStore exposes spine reads via its _read_spine_*
                # methods (P4); we use the public read_concept_strands
                # helper if present, else fall back to the registry
                # ensure_grounded for shape compatibility.
                getter = getattr(concept_store, "read_spine_strands", None)
                if callable(getter):
                    return getter(concept_id, version)
            except Exception:
                logger.exception("[synthesis_worker] concept_reader failed")
            return None

        cgn_meaning_oracle = CGNMeaningOracle(
            concept_reader=_concept_reader,
        )

        # Concrete truth oracles
        sandbox_oracle = CodingSandboxOracle()
        solana_oracle = SolanaRpcOracle(
            rpc_url=(config or {}).get("network", {}).get("premium_rpc_url"),
            fallback_urls=list((config or {}).get("network", {}).get("public_rpc_urls", [])),
        )
        web_api_oracle = WebApiOracle()  # default search_fn + judge_fn
        # x_oracle / x_research need a real SocialXGateway instance — best-
        # effort wire from existing plugin path (if available via config).
        # Sandbox-as-tool and sandbox-as-oracle share the same helper.

        # Register truth oracles with the router.
        oracle_router.register(sandbox_oracle)
        oracle_router.register(solana_oracle)
        oracle_router.register(web_api_oracle)
        oracle_router.register(cgn_meaning_oracle)  # MeaningOraclePlug — for /v6/synthesis/oracles/router visibility

        # Coverage analyzer — readers default to no-op; integration
        # boot can wire them to the chain index DB in a follow-up.
        coverage_analyzer = CoverageAnalyzer()

        # Snapshot exporter — wired to the 60s tick via a holder pattern
        # (mirrors forks_snapshot pattern). Buffers for recent verdicts
        # + proofs are constructed here so the router/proof paths can
        # push entries when they fire.
        recent_verdict_buffer: list = []
        recent_proof_buffer: list = []
        oracle_snapshot_path = os.path.join(
            os.environ.get("TITAN_DATA_DIR", "data"),
            "oracles_snapshot.json",
        )
        oracle_snapshot_exporter = OracleSnapshotExporter(
            router=oracle_router,
            spend_store=spend_store,
            gate_config=gate_config,
            coverage_analyzer=coverage_analyzer,
            snapshot_path=oracle_snapshot_path,
            recent_verdict_buffer=recent_verdict_buffer,
            recent_proof_buffer=recent_proof_buffer,
        )

        # Initial export so the file exists from boot — Observatory
        # routes get a real (possibly empty) payload immediately.
        try:
            oracle_snapshot_exporter.export()
            logger.info(
                "[synthesis_worker] initial oracle snapshot exported → %s",
                oracle_snapshot_path,
            )
        except Exception as _osx_err:
            logger.warning(
                "[synthesis_worker] initial oracle snapshot failed: %s",
                _osx_err,
            )

        # Wire the exporter into the 60s recompute tick.
        oracle_exporter_holder["fn"] = lambda: oracle_snapshot_exporter.export()

        # ToolPlugs — same OuterMemoryWriter (so procedural TXs anchor
        # via the single canonical write path per INV-4); router is
        # passed in so companion-verdict triggers fire.
        sandbox_tool = CodingSandboxTool(
            writer=writer, router=oracle_router, oracle=sandbox_oracle,
            skill_outcome_sink=_skill_outcome_sink,
        )
        events_teacher_tool = EventsTeacherTool(
            writer=writer, router=oracle_router,
            skill_outcome_sink=_skill_outcome_sink,
        )
        knowledge_tool = KnowledgeTool(
            writer=writer, router=oracle_router,
            skill_outcome_sink=_skill_outcome_sink,
        )
        # x_research_tool needs a gateway — leave unwired here; the
        # main plugin (TitanHCL) constructs it with the live
        # SocialXGateway. agno_tools fall-back gracefully when missing.
        synthesis_tool_plugs = {
            "coding_sandbox": sandbox_tool,
            "events_teacher": events_teacher_tool,
            "knowledge": knowledge_tool,
        }
        logger.info(
            "[synthesis_worker] Phase 6 oracle/proof middleware ready — "
            "router=%d plugs, proof_registry=Merkle+ZK, exporter=on",
            len(oracle_router.registered_oracles()),
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] Phase 6 wiring failed: %s — "
            "oracle/proof middleware disabled this session",
            exc, exc_info=True,
        )
        synthesis_tool_plugs = {}

    # ── Phase 7 §P7.A/H — ActrBufferStore wiring (D-SPEC-PHASE7) ─────────
    # Sole writer of `actr_buffers` (INV-Syn-16). Constructed AFTER
    # ActivationStore so it can share the existing synthesis.duckdb conn;
    # soft-fail mirrors P5/P6 — if construction raises, buffers are a
    # no-op for the session and synthesis_worker keeps running.
    actr_buffer_store: Optional[Any] = None
    buffers_snapshot_path = os.path.join(
        os.environ.get("TITAN_DATA_DIR", "data"),
        "buffers_snapshot.json",
    )
    try:
        from titan_hcl.synthesis.buffer_store import ActrBufferStore
        actr_buffer_store = ActrBufferStore(
            duckdb_conn=store._conn,           # share synthesis.duckdb (INV-Syn-3)
            snapshot_path=buffers_snapshot_path,
        )
        # Initial export so the snapshot file exists from boot — agno's
        # BufferCache.hydrate + Observatory routes get a real (possibly
        # empty) payload immediately. The 60s recompute does NOT need
        # to re-export buffers (every set/clear already triggers an
        # atomic export inside ActrBufferStore.persist/clear).
        try:
            actr_buffer_store.snapshot_export()
            logger.info(
                "[synthesis_worker] Phase 7 working-memory buffers ready — "
                "store=ok, snapshot=%s",
                buffers_snapshot_path,
            )
        except Exception as _bsx_err:
            logger.warning(
                "[synthesis_worker] initial buffers snapshot failed: %s",
                _bsx_err,
            )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] Phase 7 ActrBufferStore wiring failed: %s — "
            "working-memory buffers disabled this session",
            exc, exc_info=True,
        )

    # ── Phase 8 §P8.A-G — Procedural pipeline (D-SPEC-PHASE8) ──────────
    # Constructs: ProceduralSkillStore (INV-Syn-19) → SkillVerifier
    # (INV-Syn-20) → LLMJudge (INV-Syn-21) → ProceduralMiner. Each is
    # independent; partial wiring failures leave the rest functional.
    procedural_skill_store: Optional[Any] = None
    skill_verifier: Optional[Any] = None
    llm_judge: Optional[Any] = None
    procedural_miner: Optional[Any] = None

    skill_cfg = (config or {}).get("synthesis", {}).get("skill", {}) or {}
    _data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    skills_faiss_path = os.path.join(_data_dir, "skills_vectors.faiss")
    skills_snapshot_path = os.path.join(_data_dir, "skills_snapshot.json")
    chain_dir = os.path.join(_data_dir, "timechain")

    def _skill_soft_retire_emit(skill_id: str, utility: float) -> None:
        try:
            _send(send_queue, "META_SKILL_SOFT_RETIRED", name, "all", {
                "skill_id": skill_id, "utility_score": float(utility),
            })
        except Exception as e:
            logger.debug("[synthesis_worker] soft_retire emit failed: %s", e)

    try:
        from titan_hcl.synthesis.skill_store import ProceduralSkillStore

        # Operator-closure Phase A: reuse the ONE shared embedder (no second
        # fastembed model — RSS discipline). Same BAAI/bge-small-en-v1.5 path.
        procedural_skill_store = ProceduralSkillStore(
            duckdb_conn=store._conn,
            faiss_path=skills_faiss_path,
            snapshot_path=skills_snapshot_path,
            embedder=_shared_embedder,
            soft_retire_floor=float(skill_cfg.get("soft_retire_floor", -0.5)),
            on_soft_retire=_skill_soft_retire_emit,
        )
        procedural_skill_store.snapshot_export()
        logger.info(
            "[synthesis_worker] Phase 8 ProceduralSkillStore ready — "
            "faiss=%s snapshot=%s", skills_faiss_path, skills_snapshot_path,
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] Phase 8 ProceduralSkillStore wiring failed: %s",
            exc, exc_info=True,
        )

    if procedural_skill_store is not None:
        try:
            from titan_hcl.synthesis.procedural_tx_reader import ChainContentHashReader
            from titan_hcl.synthesis.skill_verifier import SkillVerifier
            _chain_content_reader = ChainContentHashReader(chain_dir=chain_dir)

            def _skill_lifecycle_emit(ev: str, payload: dict) -> None:
                try:
                    _send(send_queue, ev, name, "all", payload)
                except Exception:
                    pass

            skill_verifier = SkillVerifier(
                skill_store=procedural_skill_store,
                chain_reader=_chain_content_reader,
                outer_memory_writer=writer,
                bus_emit=_skill_lifecycle_emit,
            )
            logger.info("[synthesis_worker] Phase 8 SkillVerifier ready (INV-Syn-20)")
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 8 SkillVerifier wiring failed: %s",
                exc, exc_info=True,
            )

    if procedural_skill_store is not None:
        try:
            from titan_hcl.synthesis.llm_judge import LLMJudge
            from titan_hcl.synthesis.procedural_tx_reader import default_procedural_tool_call_reader

            def _llm_judge_call(prompt: str, timeout_s: float) -> str:
                try:
                    from titan_hcl.inference import get_provider as _get_provider
                    provider = _get_provider("ollama_cloud", (config or {}).get("inference", {}) or {})
                    fn = getattr(provider, "generate", None) or getattr(provider, "complete", None)
                    if fn is None:
                        return ""
                    out = fn(prompt, max_tokens=300, temperature=0.2)
                    if isinstance(out, str):
                        return out
                    return getattr(out, "text", "") or str(out or "")
                except Exception as e:
                    logger.debug("[synthesis_worker] llm_judge provider failed: %s", e)
                    return ""

            def _judge_bus_emit(ev: str, payload: dict) -> None:
                try:
                    _send(send_queue, ev, name, "all", payload)
                except Exception:
                    pass

            llm_judge = LLMJudge(
                tool_call_reader=lambda since_ts, lim: default_procedural_tool_call_reader(
                    since_ts, lim, chain_dir=chain_dir,
                ),
                llm_provider=_llm_judge_call,
                outer_memory_writer=writer,
                model_id=(config or {}).get("inference", {}).get("ollama_cloud_model") or "ollama_cloud_deepseek",
                bus_emit=_judge_bus_emit,
                per_pass_cap=int(skill_cfg.get("llm_judge_per_pass_cap", 200)),
                timeout_s=float(skill_cfg.get("llm_judge_timeout_s", 30.0)),
            )
            logger.info("[synthesis_worker] Phase 8 LLMJudge ready (INV-Syn-21)")
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 8 LLMJudge wiring failed: %s",
                exc, exc_info=True,
            )

    if procedural_skill_store is not None:
        try:
            from titan_hcl.synthesis.procedural_miner import ProceduralMiner
            from titan_hcl.synthesis.procedural_tx_reader import default_procedural_tool_call_reader

            def _miner_llm_propose(cluster_meta: dict, kind: str):
                try:
                    from titan_hcl.inference import get_provider as _get_provider
                    provider = _get_provider("ollama_cloud", (config or {}).get("inference", {}) or {})
                    fn = getattr(provider, "generate", None) or getattr(provider, "complete", None)
                    if fn is None:
                        return None
                    seq_str = " → ".join(
                        f"{tool}({args_shape})" for tool, args_shape in cluster_meta.get("sequence", [])
                    )
                    prompt = (
                        f"Abstract this recurrent tool-call sequence ({kind}) into a parametrized skill. "
                        f"Output STRICT JSON with keys nl_description (string), executable_spec (object), "
                        f"preconditions (list), postconditions (list). Sequence: {seq_str}. "
                        f"Occurrence count: {cluster_meta.get('occurrence_count')}. Kind: {kind}. "
                        f"If kind==negative, nl_description should start with "
                        f"'Approach X fails for task-shape Y'."
                    )
                    raw = fn(prompt, max_tokens=600, temperature=0.3)
                    text = raw if isinstance(raw, str) else (getattr(raw, "text", "") or str(raw or ""))
                    if not text:
                        return None
                    start = text.find("{")
                    end = text.rfind("}")
                    if start < 0 or end <= start:
                        return None
                    return json.loads(text[start:end + 1])
                except Exception as e:
                    logger.debug("[synthesis_worker] miner_llm_propose failed: %s", e)
                    return None

            def _miner_bus_emit(ev: str, payload: dict) -> None:
                try:
                    _send(send_queue, ev, name, "all", payload)
                except Exception:
                    pass

            procedural_miner = ProceduralMiner(
                skill_store=procedural_skill_store,
                tool_call_reader=lambda since, lim: default_procedural_tool_call_reader(
                    since, lim, chain_dir=chain_dir,
                ),
                llm_proposer=_miner_llm_propose,
                outer_memory_writer=writer,
                bus_emit=_miner_bus_emit,
                window_hours=int(skill_cfg.get("miner_window_hours", 168)),
                min_seq_len=int(skill_cfg.get("miner_min_seq_len", 2)),
                max_seq_len=int(skill_cfg.get("miner_max_seq_len", 8)),
                min_occurrences=int(skill_cfg.get("miner_min_occurrences", 3)),
                max_skills_per_pass=int(skill_cfg.get("miner_max_skills_per_pass", 10)),
            )
            logger.info("[synthesis_worker] Phase 8 ProceduralMiner ready")
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 8 ProceduralMiner wiring failed: %s",
                exc, exc_info=True,
            )

    # ── Phase 9 P9.D/P9.E — repair-fork-on-failure + Tier-2 override ────
    # SkillFailureTracker (consumes the skill-outcome sink → spawns a repair
    # fork at N consecutive failures, §9.3) + UserFeedbackOverride (INV-Syn-24).
    # Soft-fail; filled into the late-bound _skill_outcome_holder so the
    # ToolPlugs' sink (built earlier) drives them.
    skill_failure_tracker: Optional[Any] = None
    user_feedback_override: Optional[Any] = None
    meta_cfg = dict((config or {}).get("synthesis", {}).get("meta", {}) or {})
    if procedural_skill_store is not None:
        try:
            from titan_hcl.synthesis.skill_failure_tracker import SkillFailureTracker
            from titan_hcl.synthesis.user_feedback import UserFeedbackOverride

            def _tracker_bus_emit(ev: str, payload: dict) -> None:
                try:
                    _send(send_queue, ev, name, "all", payload)
                except Exception:
                    pass

            # concept_resolver: map a compiled skill → its parent canonical
            # concept for a repair-fork root. P8 skills compile from tool-call
            # TXs (no direct skill→concept edge yet), so v1 returns None →
            # net-new exploration fork ("repair_skill:<id>") per §9.3 (∅ root).
            # Refinement: root at the spine concept once a skill→concept link
            # exists.
            def _skill_concept_resolver(skill_id: str):
                return None

            if hypothesis_fork_store is not None:
                skill_failure_tracker = SkillFailureTracker(
                    fork_store=hypothesis_fork_store,
                    concept_resolver=_skill_concept_resolver,
                    bus_emit=_tracker_bus_emit,
                    failure_threshold=int(
                        meta_cfg.get("repair_fork_failure_threshold", 3)),
                )
            user_feedback_override = UserFeedbackOverride(
                outer_memory_writer=writer,
                skill_store=procedural_skill_store,
                user_feedback_delta=float(meta_cfg.get("user_feedback_delta", 0.15)),
            )
            # Fill the late-bound holder so the ToolPlug skill-outcome sink
            # (built in the Phase 6 block) now drives the P8 utility loop +
            # the failure tracker.
            _skill_outcome_holder["store"] = procedural_skill_store
            _skill_outcome_holder["tracker"] = skill_failure_tracker
            logger.info(
                "[synthesis_worker] Phase 9 repair-fork + Tier-2 override ready "
                "(failure_threshold=%s, feedback_delta=%s, tracker=%s)",
                meta_cfg.get("repair_fork_failure_threshold", 3),
                meta_cfg.get("user_feedback_delta", 0.15),
                "on" if skill_failure_tracker is not None else "off(no fork store)",
            )
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 9 P9.D/E wiring failed: %s — "
                "repair-fork + Tier-2 disabled this session", exc, exc_info=True,
            )

    # ── Phase 10 §P10.A/B — Observatory metrics (D-SPEC-PHASE10) ────────
    # SovereigntyRatioMeter (headline) + MetricsAggregator (full bundle →
    # synthesis_metrics_snapshot.json at the recompute tail). INV-Syn-25:
    # observation only. Soft-fail. The meter is recorded from the two
    # unambiguous signals synthesis_worker sees: a cited recall
    # (used_by_llm=True) and a delegated-skill success (skill_outcome_sink);
    # surfaced-not-cited marks a knowledge_moment only. (v1 is item-granular;
    # per-turn denominators are a refinement — the B.6 gate is the trend.)
    sovereignty_meter: Optional[Any] = None
    metrics_aggregator: Optional[Any] = None
    retrieval_latency_ring: Optional[Any] = None
    metrics_cfg = dict((config or {}).get("synthesis", {}).get("metrics", {}) or {})
    if bool(metrics_cfg.get("metrics_snapshot_enabled", True)):
        try:
            from titan_hcl.synthesis.sovereignty_meter import SovereigntyRatioMeter
            from titan_hcl.synthesis.metrics_aggregator import (
                LatencyRing, MetricsAggregator,
            )
            sovereignty_meter = SovereigntyRatioMeter(
                windows=list(metrics_cfg.get("sovereignty_windows", ["24h", "7d", "all"])),
            )
            retrieval_latency_ring = LatencyRing(
                maxlen=int(metrics_cfg.get("retrieval_latency_ring_size", 1000)))

            def _chi_stats_provider() -> dict:
                # Best-effort chi-budget compliance readout (B.5).
                try:
                    ev = getattr(engine_recall, "_evaluator", None)
                    if ev is not None and hasattr(ev, "get_stats"):
                        return dict(ev.get_stats() or {})
                except Exception:
                    pass
                return {}

            metrics_snapshot_path = os.path.join(
                os.environ.get("TITAN_DATA_DIR", "data"),
                "synthesis_metrics_snapshot.json")
            metrics_aggregator = MetricsAggregator(
                sovereignty_meter=sovereignty_meter,
                snapshot_path=metrics_snapshot_path,
                data_dir=os.environ.get("TITAN_DATA_DIR", "data"),
                latency_ring=retrieval_latency_ring,
                chi_stats_provider=_chi_stats_provider,
                groundedness_top_n=int(metrics_cfg.get("groundedness_heatmap_top_n", 50)),
            )
            metrics_aggregator.export()  # initial snapshot from boot
            metrics_exporter_holder["fn"] = lambda: metrics_aggregator.export()
            _skill_outcome_holder["meter"] = sovereignty_meter
            logger.info(
                "[synthesis_worker] Phase 10 metrics ready — sovereignty meter "
                "+ aggregator, snapshot=%s", metrics_snapshot_path)
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 10 metrics wiring failed: %s — "
                "observability disabled this session", exc, exc_info=True)

    # Phase 8 dream dispatchers — judge BEFORE miner (INV-Syn-21).
    last_llm_judge_started_ts: float = 0.0
    llm_judge_lock = threading.Lock()

    def _maybe_run_llm_judge_async(dream_start_ts: float) -> None:
        nonlocal last_llm_judge_started_ts
        if llm_judge is None:
            return
        with llm_judge_lock:
            if dream_start_ts <= last_llm_judge_started_ts:
                return
            last_llm_judge_started_ts = dream_start_ts
        window_hours = int(skill_cfg.get("miner_window_hours", 168))
        since_ts = dream_start_ts - window_hours * 3600.0

        def _run_judge():
            try:
                summary = llm_judge.score_window(since_ts=since_ts)
                logger.info(
                    "[synthesis_worker] llm_judge done — tool_calls=%d "
                    "unscored=%d scored=%d llm_calls=%d failures=%d",
                    summary.get("tool_calls_in_window", 0),
                    summary.get("unscored_in_window", 0),
                    summary.get("scored_now", 0),
                    summary.get("llm_calls", 0),
                    summary.get("llm_failures", 0),
                )
            except Exception as e:
                logger.warning("[synthesis_worker] llm_judge crashed: %s", e, exc_info=True)

        threading.Thread(target=_run_judge, name="synthesis-llm-judge", daemon=True).start()

    last_miner_started_ts: float = 0.0
    miner_lock = threading.Lock()

    def _maybe_run_procedural_miner_async(dream_start_ts: float) -> None:
        nonlocal last_miner_started_ts
        if procedural_miner is None:
            return
        with miner_lock:
            if dream_start_ts <= last_miner_started_ts:
                return
            last_miner_started_ts = dream_start_ts

        def _run_miner():
            try:
                summary = procedural_miner.mine_pass(
                    dream_pass_id=f"dream_{int(dream_start_ts)}",
                )
                logger.info(
                    "[synthesis_worker] procedural_miner done — txs=%d "
                    "recurrent=%d positive=%d negative=%d llm_calls=%d failures=%d",
                    summary.get("txs_scanned", 0),
                    summary.get("clusters_recurrent", 0),
                    summary.get("positive_skills_compiled", 0),
                    summary.get("negative_skills_compiled", 0),
                    summary.get("llm_calls", 0),
                    summary.get("llm_failures", 0),
                )
            except Exception as e:
                logger.warning("[synthesis_worker] procedural_miner crashed: %s", e, exc_info=True)

        threading.Thread(target=_run_miner, name="synthesis-procedural-miner", daemon=True).start()

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

    # Phase 5 §P5.H — nightly ForkGC sweep, dream-boundary triggered.
    # Same rate-limit / threading pattern as consolidation: never blocks
    # the bus loop. Fires AFTER consolidation_pass (sequencing isn't
    # critical — both are independent — but the consolidation pass may
    # mark new graduated/abandoned forks that the sweep then handles).
    last_fork_sweep_started_ts: float = 0.0
    fork_sweep_lock = threading.Lock()

    def _maybe_run_fork_gc_async(dream_start_ts: float) -> None:
        nonlocal last_fork_sweep_started_ts
        if fork_gc is None:
            return
        with fork_sweep_lock:
            if dream_start_ts <= last_fork_sweep_started_ts:
                logger.debug(
                    "[synthesis_worker] fork_gc already swept for dream "
                    "window %.3f — skipping", dream_start_ts,
                )
                return
            last_fork_sweep_started_ts = dream_start_ts

        def _run_sweep():
            try:
                report = fork_gc.sweep(dry_run=not fork_gc_live_mode)
                logger.info(
                    "[synthesis_worker] fork_gc sweep done — visited=%d "
                    "pruned=%d skipped=%d dropped=%d dry_run=%s",
                    report.forks_visited, report.forks_pruned,
                    report.forks_skipped, report.total_nodes_dropped,
                    report.dry_run,
                )
            except Exception as e:
                logger.warning(
                    "[synthesis_worker] fork_gc sweep crashed: %s",
                    e, exc_info=True,
                )

        threading.Thread(
            target=_run_sweep, name="synthesis-fork-gc",
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

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
            if msg_type == MODULE_PROBE_REQUEST and _state_writer is not None:
                try:
                    from titan_hcl.core.probe_dispatcher import (
                        handle_module_probe_request,
                    )
                    handle_module_probe_request(
                        msg,
                        probe_fn=None,
                        send_queue=send_queue,
                        module_name=name,
                        state_writer=_state_writer,
                    )
                except Exception as _probe_err:  # noqa: BLE001
                    logger.warning(
                        "[synthesis_worker] MODULE_PROBE_REQUEST handler "
                        "failed: %s", _probe_err)
                continue

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
                # INV-Syn-23 strict cited gate (Phase 9): reinforce ONLY items
                # the LLM actually cited (used_by_llm=True, set by the agno
                # CitedUseDetector). Surfaced-not-cited (False / legacy-missing)
                # is telemetry-only — no record_access, no rich-get-richer
                # runaway (rFP §240). The actr_working_memory_decay contract is
                # the SPEC-correct authority; this gate makes the producer honest.
                if payload.get("used_by_llm") is True:
                    with cache_lock:
                        store.record_access(item_id, float(ts))
                    events_recorded += 1
                    # Phase 10 — a cited recall is a recall-satisfied moment.
                    if sovereignty_meter is not None:
                        try:
                            sovereignty_meter.record_knowledge_moment(float(ts))
                            sovereignty_meter.record_recall_satisfied(
                                kind="cited_recall", ts=float(ts))
                        except Exception:
                            pass
                else:
                    events_surfaced += 1
                    # Surfaced-not-cited: a knowledge moment that recall did
                    # not satisfy (the LLM re-derived instead).
                    if sovereignty_meter is not None:
                        try:
                            sovereignty_meter.record_knowledge_moment(float(ts))
                        except Exception:
                            pass
                continue

            if msg_type == USER_FEEDBACK_SIGNAL:
                # Phase 9 INV-Syn-24 — Tier-2 explicit user feedback override.
                if user_feedback_override is None:
                    continue
                _tc_tx = payload.get("tool_call_tx")
                _verdict = payload.get("verdict")
                if not isinstance(_tc_tx, str) or not _tc_tx:
                    continue
                try:
                    user_feedback_override.apply(
                        tool_call_tx=_tc_tx,
                        verdict=str(_verdict or ""),
                        skill_id=payload.get("skill_id"),
                        source=str(payload.get("source", "explicit")),
                    )
                except Exception as _uf_err:
                    logger.warning(
                        "[synthesis_worker] user feedback override failed: %s",
                        _uf_err)
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

            if msg_type == SYNTHESIS_FORK_COMMAND:
                # Phase 5 §P5.A-G — hypothesis-fork lifecycle command
                # dispatch. api process publishes; synthesis_worker (sole
                # writer per INV-Syn-8) executes + emits a result event
                # carrying request_id so the api can correlate.
                request_id = payload.get("request_id", "")
                op = payload.get("op", "")
                if hypothesis_fork_store is None:
                    _send(send_queue, SYNTHESIS_FORK_COMMAND_RESULT,
                          name, "all", {
                              "request_id": request_id, "op": op,
                              "ok": False,
                              "error": "hypothesis_fork_store_not_wired",
                          })
                    continue
                try:
                    if op == "create":
                        new_fid = hypothesis_fork_store.create_fork(
                            intent=str(payload.get("intent", "")),
                            root_anchor=payload.get("root_anchor"),
                            parent_concept_id=payload.get("parent_concept_id"),
                        )
                        result = {"ok": True, "fork_id": new_fid}
                    elif op == "record_exploration_tx":
                        hypothesis_fork_store.record_exploration_tx(
                            str(payload.get("fork_id", "")),
                            str(payload.get("tx_hash", "")),
                        )
                        result = {"ok": True}
                    elif op == "graduate_manual":
                        verdict = {
                            "oracle_id": "manual:maker",
                            "verdict": "true",
                            "evidence_ref": str(
                                payload.get("evidence_ref", "manual_trigger")
                            ),
                            "cost": 0.0, "latency_ms": 0,
                            "ts": time.time(),
                        }
                        tx = hypothesis_fork_store.graduate_oracle(
                            fork_id=str(payload.get("fork_id", "")),
                            oracle_verdict=verdict,
                            concept_name=payload.get("concept_name"),
                        )
                        result = {"ok": True, "anchor_tx": tx}
                    elif op == "abandon":
                        tx = hypothesis_fork_store.abandon(
                            fork_id=str(payload.get("fork_id", "")),
                            reason=str(payload.get(
                                "reason", "manual_abandon")),
                        )
                        result = {"ok": True, "tombstone_tx": tx}
                    elif op == "sweep":
                        if fork_gc is None:
                            result = {
                                "ok": False,
                                "error": "fork_gc_not_wired",
                            }
                        else:
                            dry_run = bool(payload.get(
                                "dry_run", not fork_gc_live_mode,
                            ))
                            report = fork_gc.sweep(dry_run=dry_run)
                            result = {
                                "ok": True,
                                "dry_run": report.dry_run,
                                "forks_visited": report.forks_visited,
                                "forks_pruned": report.forks_pruned,
                                "forks_skipped": report.forks_skipped,
                                "total_nodes_dropped":
                                    report.total_nodes_dropped,
                            }
                    else:
                        result = {
                            "ok": False, "error": f"unknown_op:{op}",
                        }
                    # Re-export snapshot eagerly so the next api GET sees
                    # the new state immediately (don't wait 60s).
                    try:
                        fork_exporter_holder["fn"]()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] SYNTHESIS_FORK_COMMAND op=%s "
                        "failed: %s", op, e, exc_info=True,
                    )
                    result = {"ok": False, "error": str(e)}

                _send(send_queue, SYNTHESIS_FORK_COMMAND_RESULT,
                      name, "all", {
                          "request_id": request_id, "op": op, **result,
                      })
                continue

            if msg_type == SYNTHESIS_BUFFER_COMMAND:
                # Phase 7 §P7.H — ACT-R working-memory buffer write surface.
                # agno_worker (caller-side BufferCache) publishes set/clear
                # commands; synthesis_worker is sole writer per INV-Syn-16.
                # Soft-fail: bad ops + bad payloads are logged at WARN and
                # dropped (no caller is waiting on a response — write-
                # through is fire-and-forget per INV-Syn-17).
                if actr_buffer_store is None:
                    logger.debug(
                        "[synthesis_worker] SYNTHESIS_BUFFER_COMMAND dropped "
                        "— ActrBufferStore not wired this session"
                    )
                    continue
                op = (payload.get("op") or "").lower()
                chat_id = payload.get("chat_id") or ""
                buf_name = payload.get("buffer_name") or ""
                try:
                    if op == "set":
                        actr_buffer_store.persist(
                            chat_id=chat_id,
                            buffer_name=buf_name,
                            content=payload.get("content") or "",
                            concept_ids=payload.get("concept_ids") or [],
                            ts=payload.get("ts"),
                        )
                    elif op == "clear":
                        actr_buffer_store.clear(
                            chat_id=chat_id, buffer_name=buf_name,
                        )
                    else:
                        logger.warning(
                            "[synthesis_worker] SYNTHESIS_BUFFER_COMMAND "
                            "unknown op=%r (chat_id=%s buffer=%s) — dropping",
                            op, chat_id, buf_name,
                        )
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] SYNTHESIS_BUFFER_COMMAND op=%s "
                        "chat_id=%s buffer=%s failed: %s",
                        op, chat_id, buf_name, e,
                    )
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
                    # Phase 8 (INV-Syn-21) — LLM judge runs FIRST so the
                    # procedural miner sees a fully-scored window. Both run
                    # in independent threads; rate-limits inside each dispatcher
                    # ensure ≤ 1 of each per dream window.
                    _maybe_run_llm_judge_async(dream_start_ts)
                    _maybe_run_consolidation_async(dream_start_ts)
                    # Phase 5 §P5.H — fire the nightly ForkGC sweep on the
                    # SAME dream-boundary tick. Independent thread; the
                    # rate-limit inside _maybe_run_fork_gc_async ensures
                    # at most 1 sweep per dream window.
                    _maybe_run_fork_gc_async(dream_start_ts)
                    # Phase 8 (P8.G) — procedural miner runs AFTER the judge
                    # so its FORK_READ sees the just-anchored scored_by patches.
                    _maybe_run_procedural_miner_async(dream_start_ts)
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
