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

Bus subscriptions:
  • MEMORY_RETRIEVAL_USED — use-gated record_access (INV-Syn-5)
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
    KERNEL_EPOCH_TICK,
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

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 30.0
RECOMPUTE_INTERVAL_S = 60.0          # arch §5.2 60s recompute cadence
SYNTH_STATUS_SLOT_NAME = "synth_status"     # /dev/shm/titan_<id>/synth_status.bin
SYNTH_STATUS_PAYLOAD_BYTES = 24             # struct '<ddII'

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
        self._conn = duckdb.connect(db_path)
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
                    status_writer: "SynthStatusWriter",
                    send_queue, name: str,
                    interval_s: float,
                    stop_event: threading.Event,
                    cache_lock: threading.Lock) -> None:
    """Daemon thread — 60s recompute of B_i across activation_state + SHM
    watermark publish (arch §5.2). Separate from the recv loop so a slow
    DuckDB COMMIT never delays inbound MEMORY_RETRIEVAL_USED handling.

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
        try:
            now = time.time()
            with cache_lock:
                n_touched = store.recompute_and_persist(now)
                items = store.items_tracked()
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
                "duration_ms": duration_ms,
            })
            pass_count += 1
            if pass_count % 60 == 1:    # ~hourly summary
                logger.info(
                    "[synthesis_worker] recompute pass #%d — items=%d "
                    "touched=%d duration=%dms errors=%d",
                    pass_count, items, n_touched, duration_ms, error_count)
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

    # DuckDB path — same resolution as TitanDuckDB.
    db_path = (config or {}).get("memory_db_path") or os.path.join(
        "data", "titan_memory.duckdb")

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

    # Lock around the in-mem cache for the recv-loop ↔ recompute-loop
    # interleave. record_access is cheap; the lock holds only for the
    # mutation itself, never around bus I/O.
    cache_lock = threading.Lock()

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(send_queue, name, stop_event),
        daemon=True, name=f"synthesis-hb-{name}")
    hb_thread.start()

    rc_thread = threading.Thread(
        target=_recompute_loop,
        args=(store, status_writer, send_queue, name, interval_s,
              stop_event, cache_lock),
        daemon=True, name=f"synthesis-recompute-{name}")
    rc_thread.start()

    _send(send_queue, MODULE_READY, name, "guardian", {
        "titan_id": titan_id,
        "module": "synthesis_worker",
        "version": "1.0.0",
        "schema_version": 1,
        "spec_ref": "SPEC §25 / D-SPEC-123",
        "plug_counts": registry.counts(),
        "items_tracked_at_boot": store.items_tracked(),
    })
    logger.info(
        "[synthesis_worker] MODULE_READY emitted — items_at_boot=%d "
        "plug_counts=%s", store.items_tracked(), registry.counts())

    events_recorded = 0

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
                    "(events_recorded=%d items_tracked=%d)",
                    events_recorded, store.items_tracked())
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

            # KERNEL_EPOCH_TICK + everything else: no-op in Phase 1. The
            # recompute_loop drives on wall-clock, not on this tick.

    finally:
        stop_event.set()
        try:
            store.close()
        except Exception:
            logger.exception("[synthesis_worker] store close failed")
        try:
            status_writer.close()
        except Exception:
            logger.exception("[synthesis_worker] status_writer close failed")
        logger.info("[synthesis_worker] shutdown complete")
