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

import faulthandler
import json
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
    CGN_CONCEPT_GROUNDED,
    DREAM_STATE_CHANGED,
    ENGRAM_FELT_CANDIDATE,
    KERNEL_EPOCH_TICK,
    KNOWLEDGE_MOMENT,
    MAINTAIN_BUNDLE,
    MEMORY_RETRIEVAL_USED,
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    RETRIEVAL_SAMPLE,
    SYNTHESIS_FORK_COMMAND,
    TOOL_CALL_VERDICT_RECORD,
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
from titan_hcl.synthesis.writer import (
    SynthesisWriter,
    BOOT_SYNC_TIMEOUT_S,
    guard_conn,
    resolve_writer,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev
from titan_hcl.errors import ModuleError, ModuleErrorCode
from titan_hcl.bus import publish_module_error

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
# Freeze-watchdog deadline (2026-06-05 heartbeat-timeout hunt). The heartbeat
# thread re-arms a faulthandler all-thread stack dump every beat; if the process
# freezes (a thread holds the GIL or blocks a lock) and the heartbeat loop cannot
# re-arm within this many seconds, faulthandler dumps EVERY thread's stack to
# stderr → the journal, naming the exact frozen op. Sized < the 60s guardian
# heartbeat_timeout so the dump lands BEFORE the kill, and > HEARTBEAT_INTERVAL_S
# so a healthy beat always cancels+re-arms it (no dump in steady state).
HB_FREEZE_DUMP_S = 50.0
# Recompute-pass duration above which we log a WARNING regardless of pass count
# (a single slow pass on the recompute thread is a prime heartbeat-freeze suspect).
RECOMPUTE_SLOW_WARN_MS = 10_000
# Bounded boot-alive window (root-cause fix 2026-06-04): the heartbeat thread
# emits the SHM heartbeat DURING boot (not just after _WORKER_READY) up to this
# many seconds, so a slow boot under a respawn cascade reports ALIVE instead of
# being false-detected as CRASHED (shm_pid_dead). Generous enough to ride out a
# cascade; bounded so a GENUINELY stuck boot still stops heartbeating and gets
# restarted (no hidden hang). State stays starting/booted; readiness stays
# probe-gated (SPEC §11.I.2 / §11.I.7).
BOOT_HEARTBEAT_GRACE_S = 300.0
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

# WAL-hygiene (bugfix 2026-06-05) — checkpoint the synthesis_spine Kuzu graph
# whenever its un-checkpointed `.wal` crosses this size, so its replay-on-open
# cost stays small. Kept low: a Guardian-supervised module is killed (never
# cleanly closed), so the only thing that ever clears the WAL is an explicit
# checkpoint. (T2 2026-06-05: a 632 KB WAL already replayed to +420 MB.)
_KUZU_WAL_CHECKPOINT_MB = 4.0

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

    def __init__(self, db_path: str, writer: Any = None) -> None:
        self._db_path = db_path
        # ── Single-writer-thread discipline (INV-Syn single-writer / G21 at
        # the THREAD level; AUDIT_synthesis_engine_crashloop_concurrency_
        # 20260602.md). This connection is the ONE shared synthesis.duckdb
        # handle (ActivationStore + StandingBundleStore + ActrBufferStore +
        # HypothesisForkStore + OracleSpendStore + ProceduralSkillStore all
        # use it). DuckDB Connections are NOT thread-safe → it is created AND
        # only ever invoked on the single SynthesisWriter thread. guard_conn()
        # raises WriterThreadViolation on any off-thread .execute(), so a
        # forgotten submit() fails loudly in tests instead of segfaulting in
        # production. Tests pass no writer → a synchronous InlineWriter runs
        # ops inline. (Process-level sole-writer per G21 / INV-Syn-3 still
        # holds: synthesis_worker owns its own data/synthesis.duckdb file;
        # cross-process readers use atomic JSON snapshots, never this handle.)
        self._writer = resolve_writer(writer)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # BOOT_SYNC_TIMEOUT_S (not the 30s steady-state default): this is the
        # boot-CRITICAL connect+schema+load. Under a respawn cascade the writer
        # thread can be CPU-starved, so it gets room to complete instead of a
        # TimeoutError crash on init (root-cause fix 2026-06-04 — T1 mainnet).
        self._conn = guard_conn(
            self._writer,
            self._writer.submit_sync(
                lambda: duckdb.connect(db_path), timeout=BOOT_SYNC_TIMEOUT_S))
        # Cache resumes ACT-R activation across restarts. Schema + resume-load
        # run on the writer thread (boot is single-threaded, so they complete
        # before any other thread submits an op).
        self._cache: dict[str, ActivationState] = {}
        self._writer.submit_sync(self._init_schema, timeout=BOOT_SYNC_TIMEOUT_S)
        self._writer.submit_sync(self._load_existing, timeout=BOOT_SYNC_TIMEOUT_S)

    def _init_schema(self) -> None:
        # CREATE TABLE IF NOT EXISTS is idempotent across restarts. Mirrors the
        # schema previously in core/direct_memory.py (D-SPEC-123 v1; relocated
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
        rows: list[tuple] = []
        for state in self._cache.values():
            new_bi = base_level(state, now, d=d, window_n=window_n)
            changed = (new_bi != state.base_level
                       or state.last_recompute == 0.0)
            if changed:
                state.base_level = new_bi
                state.last_recompute = now
                rows.append(self._row_tuple(state))
        # The B_i math ran on the CALLER thread (under cache_lock); rows are
        # immutable tuples, so the DuckDB upsert is submitted as ONE batch op
        # to the writer thread — heavy compute never runs on the writer, and
        # the persist never races another handle user.
        if rows:
            self._writer.submit(lambda: self._persist_rows(rows))
        return len(rows)

    def export_snapshot(self, snapshot_path: str) -> int:
        """Atomic JSON export of {item_id -> base_level} for cross-process
        readers (DuckDB 1.5+ exclusive-lock workaround). Cold-start sentinels
        (-inf, NaN) are filtered out — readers treat absent items as
        cold-start by default. Returns the count of items written.
        """
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

    def _row_tuple(self, state: ActivationState) -> tuple:
        """Materialize an immutable upsert row from a live state — called under
        the caller's cache_lock so the writer op never reads mutating state."""
        log_blob = msgpack.packb(state.access_log, use_bin_type=True)
        # base_level may be -inf for cold-start states; DuckDB DOUBLE supports
        # infinities natively, so this round-trips cleanly.
        return (state.item_id, state.last_access, log_blob, state.access_count,
                state.first_access, state.base_level, state.last_recompute)

    def _persist_rows(self, rows: list[tuple]) -> None:
        """Batch upsert on the writer thread. ON CONFLICT DO UPDATE (NOT
        INSERT OR REPLACE — that is DELETE+INSERT and churns the PK ART index,
        the documented 2026-06-01 crash class). INV-Syn-3 sole writer."""
        self._conn.executemany(
            "INSERT INTO activation_state "
            "(item_id, last_access, access_log, access_count, first_access, "
            " base_level, last_recompute) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (item_id) DO UPDATE SET "
            "last_access=excluded.last_access, access_log=excluded.access_log, "
            "access_count=excluded.access_count, "
            "first_access=excluded.first_access, "
            "base_level=excluded.base_level, "
            "last_recompute=excluded.last_recompute",
            rows,
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
        # §11.H.3 + AUDIT §C (rFP §P2): force a final DuckDB CHECKPOINT before
        # close so the WAL is durably flushed on graceful shutdown (kill-respawn
        # boots fresh from synthesis.duckdb). _persist's INSERTs already
        # autocommit per statement (DuckDB default — committed txns are
        # WAL-recovered even on SIGKILL), so this only guards the
        # WAL-not-yet-checkpointed window, not uncommitted data.
        def _op() -> None:
            try:
                self._conn.execute("CHECKPOINT")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[synthesis_worker] CHECKPOINT on close failed: %s", e)
            self._conn.close()
        self._writer.submit_sync(_op)


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
                    state_writer: Optional[Any] = None,
                    boot_deadline: Optional[float] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s.

    Phase 11 §11.I.5 (Chunk 11N): also publishes
    ModuleStateWriter.heartbeat() on the SHM slot when `state_writer` is
    provided so guardian_hcl's SHM-staleness detector + observatory
    `/v6/readiness` see fresh data. SHM heartbeat suppressed while
    `_WORKER_READY` is False so the slot stays in "starting"/"booted"
    until the recv-loop probe handler transitions it to "running".
    """
    _hb_last = time.monotonic()
    while not stop_event.is_set():
        # Freeze-watchdog (2026-06-05): re-arm a faulthandler all-thread stack
        # dump at the TOP of every beat. A healthy next beat (≤30s) cancels +
        # re-arms it; if THIS process freezes (a thread holds the GIL, or
        # heartbeat() blocks on _write_lock) the loop cannot re-arm within
        # HB_FREEZE_DUMP_S → faulthandler dumps EVERY thread's stack to stderr
        # → the journal, naming the exact frozen op BEFORE the 60s guardian kill.
        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(HB_FREEZE_DUMP_S, repeat=False)
        except Exception:  # never let the watchdog break the heartbeat
            pass
        _hb_gap = time.monotonic() - _hb_last
        _hb_last = time.monotonic()
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        if state_writer is not None:
            # Emit the SHM heartbeat once READY (steady state) OR during boot
            # while still within the boot-grace window (root-cause fix
            # 2026-06-04). A slow boot under a respawn cascade then reports
            # ALIVE (state stays starting/booted — heartbeat() preserves state;
            # readiness stays probe-gated) instead of being false-detected as
            # CRASHED via a stale slot and killed mid-init. Past the boot
            # deadline a still-not-ready worker stops heartbeating → a genuinely
            # stuck boot is caught by EMPTY/CRASHED supervision (no hidden hang).
            _booting_ok = (boot_deadline is not None
                           and time.monotonic() < boot_deadline)
            if _WORKER_READY or _booting_ok:
                _hb_w0 = time.monotonic()
                try:
                    state_writer.heartbeat()
                except Exception as _hb_err:
                    # NEVER silent (directive_error_visibility / SPEC §11.I.4):
                    # a failed SHM heartbeat write is exactly how the slot goes
                    # stale → guardian heartbeat_timeout. Surface it in the journal
                    # AND on the MODULE_ERROR cascade (WARN = informational, no
                    # restart impact) so the next freeze names its own cause.
                    logger.warning(
                        "[synthesis_worker] HB WRITE FAILED — state_writer."
                        "heartbeat() raised %r — SHM slot will go stale → "
                        "guardian heartbeat_timeout", _hb_err, exc_info=True)
                    try:
                        publish_module_error(send_queue, ModuleError(
                            module_name=name, subsystem="heartbeat",
                            error_code=ModuleErrorCode.SHM_WRITE_FAILED,
                            severity=_phase11_sev.WARN,
                            message="synthesis SHM heartbeat write failed",
                            detail=repr(_hb_err)))
                    except Exception as _casc_err:
                        logger.debug(
                            "[synthesis_worker] HB error cascade publish failed: "
                            "%s", _casc_err)
                else:
                    # A large GAP since the last beat = this thread was starved
                    # (GIL held by a bulk op on another thread). A large WRITE
                    # duration = heartbeat() blocked on ModuleStateWriter._write_lock
                    # (a slow concurrent publish). Either → SHM slot stale →
                    # guardian heartbeat_timeout. Log BOTH so a near-miss is visible
                    # even when faulthandler's harder threshold didn't trip.
                    _hb_write_ms = (time.monotonic() - _hb_w0) * 1000.0
                    if _WORKER_READY and (_hb_gap > HEARTBEAT_INTERVAL_S * 1.5
                                          or _hb_write_ms > 2000.0):
                        logger.warning(
                            "[synthesis_worker] HB SLOW — gap=%.1fs write=%.0fms "
                            "(expected gap ~%.1fs, write <100ms) — heartbeat path "
                            "delayed; SHM slot at risk of staleness",
                            _hb_gap, _hb_write_ms, HEARTBEAT_INTERVAL_S)
            else:
                logger.warning(
                    "[synthesis_worker] HB SUPPRESSED — not READY + past boot "
                    "grace (%.0fs into life) — SHM slot stale, guardian will "
                    "restart a genuinely stuck boot", _hb_gap)
        stop_event.wait(HEARTBEAT_INTERVAL_S)
    # Loop exiting (shutdown) — disarm the freeze-watchdog so a clean stop
    # never trips a spurious dump.
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


def _tx_index_loop(tx_index_holder: dict, stop_event: threading.Event,
                   interval_s: float, name: str) -> None:
    """Daemon thread — incrementally index new chain TXs into the tx_hash FAISS
    shards on its OWN cadence (G13, AUDIT §5.3). Decoupled from the recompute
    loop so a slow embed (the embedder lazy-load is seconds) never delays the
    SHM freshness watermark publish that cross-process BridgeRecall consumers
    gate on. No-op until the builder is wired (holder default). FAISS writes
    route through the single SynthesisWriter (Option C)."""
    stop_event.wait(min(interval_s, 10.0))
    while not stop_event.is_set():
        try:
            tx_index_holder["fn"]()
        except Exception as e:
            logger.debug("[synthesis_worker] tx_index tick failed: %s", e)
        stop_event.wait(interval_s)


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
                    kuzu_checkpoint_holder: dict,
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
            # EngramStore.export_snapshot AFTER kuzu_graph_obj is wired;
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
            # WAL-hygiene (bugfix 2026-06-05) — bound the Kuzu spine WAL so its
            # replay-on-open cost can never grow into the OOM-loop danger zone.
            # No-op until wired (kuzu_graph_obj open) + only checkpoints past the
            # size threshold / every ~30 passes (self-throttled inside the tick).
            try:
                kuzu_checkpoint_holder["fn"]()
            except Exception as _kck_err:
                logger.debug(
                    "[synthesis_worker] kuzu_checkpoint call failed: %s",
                    _kck_err,
                )
            # G13 (AUDIT §5.3): the tx-index FAISS build (embed + add, bounded)
            # runs on its OWN `synthesis-tx-index` daemon thread now — NOT
            # inline here — so a slow embed (the embedder lazy-load is seconds)
            # never delays this watermark publish, which cross-process
            # BridgeRecall consumers gate freshness on.
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
            # Heartbeat-freeze hunt (2026-06-05): a single slow recompute pass on
            # this thread is a prime suspect for starving the 30s heartbeat. Warn
            # on ANY pass over the threshold (independent of the hourly summary) so
            # a creeping recompute names itself well before it hits 60s.
            if duration_ms > RECOMPUTE_SLOW_WARN_MS:
                logger.warning(
                    "[synthesis_worker] recompute pass #%d SLOW — duration=%dms "
                    "(items=%d touched=%d) — a pass approaching 60s will starve "
                    "the heartbeat → guardian heartbeat_timeout",
                    pass_count, duration_ms, items, n_touched)
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

    # ── G10 / INV-Syn-13 — live SOL balance reader for the metabolic gate ──
    # ShmReaderBank reads network_state.bin (kernel monitor_tick = G21 single
    # writer) via SeqLock/PROT_READ — a pure SHM read, NOT a DuckDB/Kuzu/FAISS
    # handle op, so it deliberately does NOT route through db_writer (Option C
    # serializes only the non-thread-safe native handles; the SHM reader is
    # safe from any thread). NOT hand-rolled offsets (feedback_never_hand_roll_
    # shm_reads_use_shmreaderbank) and NOT bus/RPC (G18/G19). __init__ does lazy
    # mmap attach (no SHM touched at construction), so this is cheap + cold-slot
    # safe. None bank → the gate uses balance_sol_baseline.
    _shm_bank = None
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        _shm_bank = ShmReaderBank(titan_id)
    except Exception as _bank_err:  # noqa: BLE001
        logger.warning(
            "[synthesis_worker] ShmReaderBank init failed — INV-Syn-13 gate "
            "uses balance_sol_baseline (no live balance): %s", _bank_err)

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

    # ── Liveness BEFORE the heavy init (root-cause fix 2026-06-04) ──────────
    # Start the heartbeat thread NOW — before the boot-critical duckdb.connect +
    # schema/load below — and let it emit the SHM heartbeat DURING boot (bounded
    # by BOOT_HEARTBEAT_GRACE_S). Previously the hb thread started AFTER the
    # connect, and the SHM heartbeat was suppressed until _WORKER_READY (end of
    # boot), so a slow boot under a respawn cascade (box CPU/I/O starved) emitted
    # no SHM heartbeat → guardian false-detected the alive-but-still-booting
    # worker as CRASHED (shm_pid_dead) and killed it mid-init. State stays
    # starting/booted (heartbeat() preserves state); readiness is still
    # probe-gated (state→running only via the recv-loop probe, post-_WORKER_READY),
    # so nothing routes work early. SPEC-aligned: §11.I.2 (state machine
    # unchanged) + §11.I.7 (staleness-supervision is a RUNNING-state concern).
    stop_event = threading.Event()
    _boot_hb_deadline = time.monotonic() + BOOT_HEARTBEAT_GRACE_S
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer, _boot_hb_deadline),
        daemon=True, name=f"synthesis-hb-{name}")
    hb_thread.start()

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

    # ── The single persistence-writer thread (Option C; AUDIT 2026-06-02).
    # Born BEFORE ActivationStore (which submits its boot DDL/load through it)
    # and BEFORE rc_thread/recv/dream threads start, so every native-handle op
    # in the worker — DuckDB, Kuzu spine, FAISS — is serialized on this one
    # thread. Realizes the sole-writer invariant at the THREAD level (G21);
    # concurrent .execute() is impossible by construction → no SIGSEGV, and no
    # locks → no lock-ordering deadlock. NOT the OuterMemoryWriter (`writer`
    # below, bus-only) — this is the DB/Kuzu/FAISS invoker.
    db_writer = SynthesisWriter(name=name).start()

    # Construct sole-writer surfaces (G21 / INV-Syn-3).
    store = ActivationStore(db_path, writer=db_writer)
    status_writer = SynthStatusWriter(titan_id)
    registry = PlugRegistry()

    # §7.E.0 — per-Engram citation attribution: the membership reverse-index +
    # the live `fluent` axis feed + §7.E's `(axes_at_recall, cited?)` reward log.
    # Shares ActivationStore's ONE guarded conn (store._conn) + the sole writer
    # (G21 / INV-Syn-28). Soft: a schema-DDL failure disables attribution for the
    # session; synthesis is otherwise unaffected. In scope from here for the
    # ConsolidationPass injection AND the KNOWLEDGE_MOMENT handler below.
    from titan_hcl.synthesis.recall_attribution import RecallAttribution
    recall_attribution: Optional[RecallAttribution] = RecallAttribution(
        store._conn, db_writer)
    if not recall_attribution.ensure_schema():
        recall_attribution = None

    # Inner↔Outer Felt-Teaching Bridge (RFP_inner_outer_felt_teaching_bridge) —
    # shares ActivationStore's ONE guarded conn (store._conn) + the sole writer
    # (db_writer), like recall_attribution. Owns the §7.1 decompose cache
    # (engram_objects) + the §7.2 event-sourced CGN grounded-set
    # (cgn_grounded_objects). In scope from here for the ConsolidationPass injection
    # AND the CGN_CONCEPT_GROUNDED handler below. Soft → None (synthesis unaffected).
    from titan_hcl.synthesis.felt_bridge import FeltBridge
    felt_bridge: Optional[FeltBridge] = FeltBridge(store._conn, db_writer)
    if not felt_bridge.ensure_schema():
        felt_bridge = None

    # Phase 2 D-P2-4 standing-bundle store — sole writer of
    # data/synthesis.duckdb / association_bundles. CONN-2 FOLD (AUDIT §5.2):
    # shares ActivationStore's ONE guarded connection (store._conn) rather than
    # opening its own duckdb.connect to the same file — two Connection objects
    # mutating one file's WAL/buffer-manager from two threads was a confirmed
    # corruption path. All writes route through db_writer.
    standing_cfg = (config or {}).get("standing") or {}
    ring_size = int(standing_cfg.get(
        "user_bundle_max_txs", DEFAULT_RING_SIZE))
    max_entities = int(standing_cfg.get(
        "user_bundle_max_users", DEFAULT_MAX_ENTITIES))
    bundle_store = StandingBundleStore(
        db_path, ring_size=ring_size, max_entities=max_entities,
        conn=store._conn, writer=db_writer)

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

    # (stop_event + the heartbeat thread are now started EARLIER — before the
    # heavy duckdb init above — so liveness covers the boot-critical connect.
    # See the "Liveness BEFORE the heavy init" block near the top of boot.)

    # Phase 4 FU-1 — spine exporter holder (late-bound). Default is a
    # no-op so the recompute loop fires safely before the EngramStore is
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
    # WAL-hygiene (bugfix 2026-06-05) — periodic Kuzu spine checkpoint holder
    # (late-bound; set after the spine graph opens). Bounds the synthesis_spine
    # WAL so its replay-on-open can never grow into the OOM-loop danger zone.
    kuzu_checkpoint_holder: dict = {"fn": lambda: None}

    # Operator-closure Phase A — ONE shared embedder for the whole worker (the
    # tx_hash FAISS store, EngineRecall's query embed, the consolidation cosine
    # path, AND skill_store). Routed to the fleet-standard llama.cpp embedder
    # singleton (Phase 13 §3J.1) so only a single ~197 MB model is resident — this
    # is the RSS fix: fastembed/onnxruntime's CPU arena never returned memory to
    # the OS and drove the bulk backfill to ~5 GB (feedback_eager_init_needs_rss_
    # root_cause_first). llama.cpp is flat. Lazy-loads on first embed, never at boot.
    _data_dir_sw = os.path.dirname(db_path) or "."

    def _ensure_embed_model():
        from titan_hcl.utils.text_embedder import get_text_embedder
        return get_text_embedder()

    def _shared_embedder(text: str):
        try:
            import numpy as np
            # Singleton returns an L2-normalized 1-D vector already.
            return np.asarray(_ensure_embed_model().encode(text), dtype=np.float32)
        except Exception as e:
            logger.debug("[synthesis_worker] shared_embedder failed: %s", e)
            return None

    def _shared_batch_embedder(texts: list):
        # ONE embed call for the whole list — far faster + lighter than N single
        # calls (the per-tick tx-index path embeds in bulk). Singleton normalizes.
        try:
            import numpy as np
            vecs = np.asarray(_ensure_embed_model().encode(list(texts)),
                              dtype=np.float32)
            return [vecs[i] for i in range(vecs.shape[0])]
        except Exception as e:
            logger.debug("[synthesis_worker] batch_embedder failed: %s", e)
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
            data_dir=_data_dir_sw, embedder=_shared_embedder,
            batch_embedder=_shared_batch_embedder, writer=db_writer)
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
              tx_index_holder, kuzu_checkpoint_holder,
              send_queue, name,
              interval_s, stop_event, cache_lock),
        daemon=True, name=f"synthesis-recompute-{name}")
    rc_thread.start()

    # G13 — tx-index FAISS build runs on its OWN daemon thread (off the
    # recompute/watermark thread). No-op until the builder wires the holder
    # below; FAISS writes route through the SynthesisWriter (Option C).
    tx_index_thread = threading.Thread(
        target=_tx_index_loop,
        args=(tx_index_holder, stop_event, interval_s, name),
        daemon=True, name=f"synthesis-tx-index-{name}")
    tx_index_thread.start()

    # Phase 4 §P4.A/§P4.H — synthesis spine Kuzu graph (G21 sole writer
    # per INV-Syn-7). Uses its OWN Kuzu file `data/synthesis_spine.kuzu`
    # — distinct from `data/knowledge_graph.kuzu` which memory_worker
    # owns in RW for Person/Topic/Trinity entities. Sharing one Kuzu
    # file across two RW processes triggers Kuzu's exclusive-write-lock
    # rejection (same class of cross-process conflict the Phase 1 lesson
    # 1 solved for DuckDB by giving synthesis_worker its own .duckdb).
    # Cross-process readers open this file with read_only=True (api
    # process for /v6/synthesis/engrams/* — Kuzu 0.11 supports
    # concurrent read-only opens against an active writer).
    kuzu_graph_obj: Optional[Any] = None
    try:
        from titan_hcl.core.direct_memory import TitanKnowledgeGraph
        kuzu_graph_obj = TitanKnowledgeGraph(
            os.path.join(os.path.dirname(db_path) or ".", "synthesis_spine.kuzu"),
        )
        # WAL hygiene (bugfix 2026-06-05) — a Guardian-supervised module is
        # KILLED, never cleanly closed, so the Kuzu WAL only ever GROWS across
        # boots and is replayed in full on every open. Once that replay cost
        # crosses synthesis's RSS cap the module OOM-loops *before* it can ever
        # checkpoint → dirty WAL forever (T2 2026-06-05: 632 KB WAL → +420 MB
        # per boot → DISABLED). Checkpoint once at boot to clear the prior
        # session's + the schema-DDL WAL immediately, then bound it periodically
        # in the recompute loop (see kuzu_checkpoint_holder below).
        _boot_wal = kuzu_graph_obj.wal_size_mb()
        if kuzu_graph_obj.checkpoint():
            logger.info(
                "[synthesis_worker] Kuzu spine boot checkpoint — WAL %.1fMB → cleared",
                _boot_wal)
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] Kuzu spine graph open failed: %s — spine "
            "recall + consolidation will degrade to no-op",
            exc,
        )

    # Bound the spine WAL during a long-running session: each recompute tick,
    # checkpoint when the WAL crosses a low threshold OR every ~30 passes, so it
    # can never accumulate back to the OOM-loop danger zone. Default no-op until
    # wired here (kuzu_graph_obj may be None on open failure). G21 sole-writer
    # (INV-Syn-7) — safe to checkpoint from the recompute thread.
    if kuzu_graph_obj is not None:
        _kuzu_ckpt_state = {"pass": 0}

        def _kuzu_checkpoint_tick() -> None:
            try:
                _kuzu_ckpt_state["pass"] += 1
                wal_mb = kuzu_graph_obj.wal_size_mb()
                if wal_mb >= _KUZU_WAL_CHECKPOINT_MB or _kuzu_ckpt_state["pass"] % 30 == 0:
                    if kuzu_graph_obj.checkpoint() and wal_mb >= 1.0:
                        logger.info(
                            "[synthesis_worker] Kuzu spine checkpoint — WAL "
                            "%.1fMB → cleared (pass=%d)",
                            wal_mb, _kuzu_ckpt_state["pass"])
            except Exception as _ck_err:
                logger.debug(
                    "[synthesis_worker] Kuzu checkpoint tick failed: %s", _ck_err)

        kuzu_checkpoint_holder["fn"] = _kuzu_checkpoint_tick

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
            # sqlite3.Row so consumers (TxIndexBuilder, RuleEvaluator, EngineRecall)
            # can use string-key access. Row is tuple-compatible (positional access
            # still works), so this is safe for positional consumers. WITHOUT this,
            # TxIndexBuilder._resolve_fork_ids did `r["fork_name"]` on a plain tuple
            # → "tuple indices must be integers or slices, not str" → the incremental
            # tx-index builder failed every boot ("no incremental growth"). (2026-06-01)
            index_db_conn.row_factory = _sqlite3.Row
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
                    # 2026-06-01: bound per-tick scope so the COLD-START backfill
                    # on a big chain (T1 mainnet) builds the spine GRADUALLY over
                    # many 60s ticks instead of a single 2000-block scan+embed
                    # spike (which drove synthesis RSS to ~3.5GB → guardian
                    # restart loop). Each tick stays small + bounded; the
                    # watermark persists progress, so the full index still builds
                    # — just over minutes, not in one memory spike. Steady-state
                    # (caught up) ticks index only the few new blocks per minute.
                    summary = _tx_index_builder.run(max_blocks=300)
                    if summary.get("indexed"):
                        logger.info(
                            "[synthesis_worker] tx-index tick: +%d vectors "
                            "(scanned=%d) shards=%s",
                            summary["indexed"], summary["scanned"],
                            synth_vector_store.stats())
                    # Phase C (RFP spine): index PROMOTED thoughts from the
                    # content sidecar (real thought text keyed by per-TX hash) so
                    # SEARCH returns promoted-thought tx_hashes the deref resolves
                    # to real content — not the chain envelope.
                    sc_summary = _tx_index_builder.run_sidecar(max_items=500)
                    if sc_summary.get("indexed"):
                        logger.info(
                            "[synthesis_worker] tx-index sidecar tick: +%d real "
                            "thoughts (scanned=%d) by_fork=%s",
                            sc_summary["indexed"], sc_summary["scanned"],
                            sc_summary.get("by_fork"))

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
    # Phase 4 FU-1 — the EngramStore instance is hoisted into the outer
    # scope so the recompute loop can call engram_store.export_snapshot()
    # each 60s tick regardless of whether ConsolidationPass construction
    # succeeded. Stays None if kuzu_graph_obj is unavailable.
    engram_store: Optional[Any] = None
    last_dream_pass_started_ts: float = 0.0
    consolidation_thread_lock = threading.Lock()
    try:
        from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
        from titan_hcl.synthesis.engram_store import EngramStore
        from titan_hcl.synthesis.consolidation import (
            ConsolidationPass, LLMProposal,
        )
        from titan_hcl.synthesis.consolidation_defaults import (
            default_mine_recent_thoughts, make_default_llm_propose,
            make_default_decompose, derive_domain_hint,
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
        engram_store = EngramStore(kuzu_graph, writer, db_writer=db_writer)

        # Phase 4 FU-1 — wire the spine_exporter so the recompute loop's
        # next 60s tick starts writing data/spine_snapshot.json. The api
        # process reads this JSON (NOT Kuzu directly — Kuzu 0.11 holds
        # the exclusive lock even for read_only=True opens).
        spine_exporter_holder["fn"] = (
            lambda: engram_store.export_snapshot(spine_snapshot_path)
        )
        # Phase D (§7.D) — one-time boot recompute of the population grounding
        # scalar. This MIGRATES the existing population onto the percentile-blend
        # (backfilling axis_used from the stored provisional for pre-D Engrams) so
        # groundedness discriminates immediately on deploy — independent of whether
        # a dream creates new concepts. Idempotent + cheap (bounded Engram count);
        # ongoing re-ranking happens at each dream boundary (ConsolidationPass.run).
        try:
            # §7.E.0 — restore the live `fluent` axis from persisted citation
            # counters on boot (counts survive restarts in synthesis.duckdb), so
            # fluent discriminates day-1 just like axis_used does.
            _boot_fluent_lookup = None
            _boot_axes_sink = None
            if recall_attribution is not None:
                try:
                    _bfm = recall_attribution.fluent_map()
                    _boot_fluent_lookup = (
                        lambda cid, ver, _m=_bfm: _m.get((str(cid), int(ver))))
                    _boot_axes_sink = recall_attribution.update_axes_cache
                except Exception:
                    pass
            migrated = engram_store.recompute_population_groundedness(
                fluent_lookup=_boot_fluent_lookup, axes_sink=_boot_axes_sink)
            logger.info(
                "[synthesis_worker] boot population groundedness recompute: "
                "%d Engrams percentile-blended (§7.D)", migrated)
        except Exception as _pop_err:
            logger.warning(
                "[synthesis_worker] boot population recompute failed: %s", _pop_err)

        # §7.F — one-time content backfill of the advisory `domain_hint` for
        # pre-Phase-F Engrams (BUG-ENGRAM-DOMAIN-HINT-NOT-BACKFILLED). Phase F
        # set it only at consolidation; existing Engrams predate it. Cheap
        # deterministic name-classifier (fix-plan-sanctioned), idempotent (only
        # touches blanks), owns the Kuzu RW handle here in the SynthesisWriter.
        # Precedent = the §7.D axis_used boot backfill above.
        try:
            migrated_dh = engram_store.backfill_domain_hints(derive_domain_hint)
            if migrated_dh:
                logger.info(
                    "[synthesis_worker] boot domain_hint backfill: %d Engrams "
                    "labeled (§7.F)", migrated_dh)
        except Exception as _dh_err:
            logger.warning(
                "[synthesis_worker] boot domain_hint backfill failed: %s",
                _dh_err)

        # §7.E — offline train-step of the learned grounding combiner on boot
        # (idle: pre-tick). Self-gating — activates only if it beats the §7.D
        # blend on held-out citation AUC, else the store keeps using the blend.
        # With the current near-degenerate data the guard short-circuits before
        # any sklearn import (no negative class yet). Safe to ship (falls back).
        if recall_attribution is not None:
            try:
                _gc_metrics = engram_store.train_grounding_combiner(
                    recall_attribution.read_training_events())
                logger.info(
                    "[synthesis_worker] boot grounding-combiner train (§7.E): %s",
                    _gc_metrics)
            except Exception as _gc_err:
                logger.warning(
                    "[synthesis_worker] boot grounding-combiner train failed: %s",
                    _gc_err)

        # Initial export so the snapshot is non-empty + present before
        # the first 60s tick (api process gets data immediately on first
        # poll after worker boot).
        try:
            initial_n = engram_store.export_snapshot(spine_snapshot_path)
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
        decompose_fn = None  # Bridge §7.1 — Engram→Object decompose (same provider)
        inference_cfg = (config or {}).get("inference", {}) or {}
        try:
            from titan_hcl.inference import get_provider as _get_provider
            api_key = inference_cfg.get("ollama_cloud_api_key", "") or ""
            if api_key:
                provider = _get_provider("ollama_cloud", inference_cfg)
                propose_fn = make_default_llm_propose(provider)
                decompose_fn = make_default_decompose(provider)  # Bridge §7.1
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

        # RFP_synthesis_spine_reads_real_data §7.D — consolidation reads REAL
        # promoted thoughts. default_mine_recent_thoughts returns embedding=None
        # (sidecar holds content, not vectors); fill each candidate's embedding
        # from the `conversation` FAISS shard (the shard run_sidecar indexes into
        # and recall SEARCHes) so ConsolidationPass clusters by COSINE over real
        # thought content — the precondition for real concept synthesis.
        import dataclasses as _dc

        def _mine_with_embeddings(since_ts, exclude_forks):
            # RFP_synthesis_spine_reads_real_data §7.D (D1/D2): mine the REAL
            # promoted thoughts from the sidecar (content keyed by per-TX hash),
            # then fill each candidate's embedding from the `conversation` FAISS
            # shard — the SAME shard the recall helper SEARCHes and run_sidecar
            # indexes into. (The old chain-block mine keyed candidates by block_hash
            # on the declarative/episodic shards → double-miss → 0 concepts.)
            import numpy as _np_emb
            cands = default_mine_recent_thoughts(
                since_ts=since_ts, exclude_forks=exclude_forks)
            if synth_vector_store is None:
                return cands
            # TxCandidate is a FROZEN dataclass — cannot assign c.embedding in
            # place ('cannot assign to field embedding'); rebuild via replace().
            # Keep the embedding as a numpy float32 ARRAY (not a Python tuple): the
            # consolidation cosine then runs via numpy (GIL-RELEASING C compute), so
            # clustering hundreds of vectors on the dream thread never starves the
            # 30s heartbeat (RFP §7.D flap fix — pure-Python tuple cosine held the
            # GIL across ~10^5 dot products → shm_pid_dead).
            out = []
            for c in cands:
                if c.embedding is None:
                    vec = synth_vector_store.get_vector("conversation", c.tx_hash)
                    if vec is not None:
                        c = _dc.replace(
                            c, embedding=_np_emb.asarray(vec, dtype=_np_emb.float32))
                out.append(c)
            return out

        # Tunables (RFP §7.D / D2-D3): cosine clustering over the sidecar source;
        # wide window so backfilled history participates; min_cluster default 2.
        consolidation_cfg = (config or {}).get("synthesis", {}).get(
            "consolidation", {}) or {}
        _min_cluster = int(consolidation_cfg.get("min_cluster_size", 2))
        _window_hours = float(consolidation_cfg.get("window_hours", 720.0))

        import numpy as _np_cos
        def _np_cosine(a, b) -> float:
            # Vectorized cosine — numpy dot/norm RELEASE the GIL during the C-level
            # compute, so the dream-thread clustering NO LONGER starves the 30s
            # heartbeat thread. The default pure-Python `_default_cosine` held the
            # GIL across ~10^5 dot products over the embedded candidates → heartbeat
            # went stale → guardian shm_pid_dead flap (RFP §7.D: "no bulk ops on the
            # hot path" — the hot path here is the shared GIL).
            try:
                na = float(_np_cos.linalg.norm(a))
                nb = float(_np_cos.linalg.norm(b))
                if na <= 0.0 or nb <= 0.0:
                    return 0.0
                return float(_np_cos.dot(a, b) / (na * nb))
            except Exception:
                return 0.0

        consolidation_pass = ConsolidationPass(
            engram_store=engram_store,
            cgn_bridge=cgn_bridge,
            outer_memory_writer=writer,
            mine_recent_txs_fn=_mine_with_embeddings,
            llm_propose_fn=propose_fn,
            cosine_fn=_np_cosine,
            window_hours=_window_hours,
            min_cluster_size=_min_cluster,
            recall_attribution=recall_attribution,  # §7.E.0 membership + fluent feed
            decompose_fn=decompose_fn,              # Bridge §7.1 Engram→Object decompose
            felt_bridge=felt_bridge,                # Bridge §7.1/§7.2 tables
            emit_candidate_fn=(                     # Bridge §7.3 handoff → felt_teaching
                lambda _p: _send(send_queue, ENGRAM_FELT_CANDIDATE, name,
                                 "felt_teaching_worker", _p)),
        )
        logger.info(
            "[synthesis_worker] ConsolidationPass ready — DREAM_STATE_CHANGED "
            "subscription active; rate-limit = 1 pass / dream window; "
            "source=sidecar(promoted thoughts); embeddings=%s; "
            "min_cluster=%d window_h=%.0f",
            "conversation_shard" if synth_vector_store is not None else "tag-only",
            _min_cluster, _window_hours,
        )
    except Exception as exc:
        logger.warning(
            "[synthesis_worker] ConsolidationPass construction failed: %s — "
            "dream-boundary consolidation will be a no-op this session",
            exc, exc_info=True,
        )

    # ── Phase 5 §P5.A–P5.H — HypothesisForkStore + ForkGC wiring ────────
    # Constructed AFTER EngramStore / OuterMemoryWriter are wired so it
    # can reuse them for graduation paths (INV-10 / INV-Syn-11). Soft-fail
    # if any dependency is missing: forks become a no-op for the session,
    # synthesis_worker keeps running.
    hypothesis_fork_store: Optional[Any] = None
    fork_gc: Optional[Any] = None
    try:
        if (kuzu_graph_obj is None
                or engram_store is None
                or 'writer' not in locals()):
            raise RuntimeError(
                "missing dependency: kuzu_graph / engram_store / writer "
                "not wired — hypothesis-fork lifecycle disabled this session"
            )
        from titan_hcl.synthesis.hypothesis_fork_store import (
            HypothesisForkStore,
        )
        from titan_hcl.synthesis.fork_gc import ForkGC

        hypothesis_fork_store = HypothesisForkStore(
            writer=db_writer,           # single-writer-thread (Option C)
            duckdb_conn=store._conn,    # same DuckDB conn as ActivationStore
            kuzu_graph=kuzu_graph_obj,
            engram_store=engram_store,
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
            writer=db_writer,           # single-writer-thread (Option C)
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
    _skill_outcome_holder: dict = {"store": None, "tracker": None}

    def _skill_outcome_sink(skill_id: str, success: bool) -> None:
        _st = _skill_outcome_holder.get("store")
        _tr = _skill_outcome_holder.get("tracker")
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
        # P3 (RFP_synthesis_decision_authority): a successful delegated skill's
        # sovereignty is captured in that turn's per-reply S (high E — the skill
        # supplied the substrate from Titan's own compiled procedure), so the
        # meter no longer records a separate skill-delegation satisfied mark.
        # The skill OUTCOME recording above (utility + failure tracker) stays.

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

        # Spend store shares the synthesis_worker's existing DuckDB connection
        # (INV-Syn-3 sole writer). OracleSpendStore.__init__ ensures the table
        # exists ON the writer thread — no standalone DDL here (the guarded
        # conn rejects off-writer-thread .execute by design, Option C).
        spend_store = OracleSpendStore(store._conn, writer=db_writer)

        # ── Balance provider (INV-Syn-13) — LIVE SOL via G18 SHM read (G10) ──
        # Reads balance_sol from network_state.bin (kernel monitor_tick = G21
        # single writer) through ShmReaderBank — replaces the static 1.0 that
        # made admit_score == importance always (the gate never tightened on
        # low balance). Short monotonic TTL so a burst of metered claims in one
        # dream pass shares one SHM read. Pure SHM read → does NOT route through
        # db_writer. Soft-fail to gate_config.balance_sol_baseline (the neutral
        # "balance unknown, don't penalize" value, NOT a magic 1.0) on
        # cold-boot / torn-read / decode-fail.
        _balance_refresh_s = 2.0
        _balance_cache: dict = {"value": None, "mono": 0.0}

        def _balance_lookup() -> float:
            baseline = gate_config.balance_sol_baseline
            if _shm_bank is None:
                return baseline
            now = time.monotonic()
            if (_balance_cache["value"] is not None
                    and (now - _balance_cache["mono"]) < _balance_refresh_s):
                return _balance_cache["value"]
            bal = baseline
            try:
                st = _shm_bank.read_network_state()
                if isinstance(st, dict) and st.get("balance_sol") is not None:
                    bal = float(st["balance_sol"])
            except Exception:  # noqa: BLE001 — cold slot / torn read → baseline
                bal = baseline
            _balance_cache["value"] = bal
            _balance_cache["mono"] = now
            return bal

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

        # CGN meaning oracle — concept_reader bound to EngramStore;
        # cgn_grounder reserved for the bus-RPC follow-up (returns None
        # for now → degraded grounding per the P6.H defensive contract).
        def _concept_reader(concept_id: str, version: int):
            # G3 (AUDIT §5.3): EngramStore.read_spine_strands now exists (the
            # getattr probe used to silently return None → meaning_of empty
            # fleet-wide). Call it directly; it runs on the writer thread
            # (@on_writer) and returns the four Timechain-anchor strands, or
            # None for a missing concept. Soft-fail so a Kuzu hiccup on this
            # dream-orchestrator path never crash-loops the worker.
            if engram_store is None:
                return None
            try:
                return engram_store.read_spine_strands(concept_id, version)
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

        # Coverage analyzer — wired to the procedural-fork tool_call TXs on the
        # chain so /v6/synthesis/oracles/coverage reflects REAL invocations.
        # (2026-06-01) The analyzer was previously left on the no-op default
        # reader, so the Observatory coverage endpoint always read 0 even when
        # tool_call TXs were on chain — the deferred "follow-up" never landed.
        # Now exposed once the tool-backstop actually produces tool_call TXs.
        from titan_hcl.synthesis.procedural_tx_reader import (
            default_procedural_tool_call_reader as _cov_tc_reader,
        )
        _cov_data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        _cov_timechain_dir = os.path.join(_cov_data_dir, "timechain")
        coverage_analyzer = CoverageAnalyzer(
            tool_call_reader=lambda since_ts, lim: _cov_tc_reader(
                since_ts, lim,
                index_db_path=os.path.join(_cov_timechain_dir, "index.db"),
                chain_dir=_cov_timechain_dir,
            ),
        )

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
            writer=db_writer,                  # single-writer-thread (Option C)
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
            writer=db_writer,                  # single-writer-thread (Option C)
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
                    import asyncio as _aio
                    from titan_hcl.inference import get_provider as _get_provider
                    provider = _get_provider("ollama_cloud", (config or {}).get("inference", {}) or {})
                    # provider.complete is ASYNC — bridge via asyncio.run (same
                    # class of bug as the miner proposer: calling it synchronously
                    # returned an un-awaited coroutine → judge scored 0 TXs →
                    # nothing got scored_by → the miner's Tier-1-verified filter
                    # starved → 0 skills. 2026-06-02.
                    out = _aio.run(provider.complete(
                        prompt=prompt, temperature=0.2, max_tokens=300,
                        timeout=float(timeout_s),
                    ))
                    return out if isinstance(out, str) else (str(out or ""))
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
                    import asyncio as _aio
                    from titan_hcl.inference import get_provider as _get_provider
                    provider = _get_provider("ollama_cloud", (config or {}).get("inference", {}) or {})
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
                    # provider.complete is an ASYNC coroutine — bridge via
                    # asyncio.run (mirrors consolidation_defaults.make_default_
                    # llm_propose). The prior code grabbed `complete` and called
                    # it SYNCHRONOUSLY → returned an un-awaited coroutine →
                    # 'coroutine never awaited' → EVERY skill abstraction failed
                    # (recurrent clusters found but 0 skills compiled). 2026-06-02.
                    text = _aio.run(provider.complete(
                        prompt=prompt,
                        system="You are a procedural-skill abstraction engine. "
                               "Output ONLY strict JSON, no prose.",
                        temperature=0.3, max_tokens=600, timeout=45.0,
                    ))
                    if not text:
                        return None
                    start = text.find("{")
                    end = text.rfind("}")
                    if start < 0 or end <= start:
                        return None
                    return json.loads(text[start:end + 1])
                except Exception as e:
                    # Error-visibility: a silent proposer failure = 0 skills with
                    # no journal trace. Surface it (the recurrent clusters are
                    # real; only the abstraction is failing).
                    logger.warning(
                        "[synthesis_worker] miner_llm_propose failed: %s", e,
                        exc_info=True)
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
    # Cross-process recall telemetry accumulator (2026-06-01) — fed by
    # RETRIEVAL_SAMPLE events from agno/cognitive recall; read by the chi
    # provider + the message-loop handler (both below). Declared here so it's
    # in scope regardless of the metrics-enabled branch.
    _xproc_chi = {"total_chi_spent": 0.0, "total_evaluations": 0,
                  "total_chi_exhausted": 0}
    metrics_cfg = dict((config or {}).get("synthesis", {}).get("metrics", {}) or {})
    if bool(metrics_cfg.get("metrics_snapshot_enabled", True)):
        try:
            from titan_hcl.synthesis.sovereignty_meter import (
                SovereigntyRatioMeter, boot_seed_from_marks,
            )
            from titan_hcl.synthesis.metrics_aggregator import (
                LatencyRing, MetricsAggregator,
            )

            # G9 (INV-Syn-25 / Phase F): the meter's rolling windows are durable
            # because every mark is persisted to synthesis.duckdb::sovereignty_marks
            # (sole writer = this worker's SynthesisWriter, INV-Syn-28) and replayed
            # on boot. Replaces the v0.22.0 conv-TX boot-seed: the conversation
            # fork's v2 batch envelopes drop per-TX content at seal, so reading
            # `sovereignty` back found 0 — the metric's durable home is synthesis's
            # OWN db (exactly where BRAIN reads the §18 gauge, RFP_brain §12), not
            # the chain envelope. The meter class stays pure; we inject an on_record
            # callback that INSERTs each mark and boot-seed replays in-window rows.
            _seed_window_s = float(metrics_cfg.get(
                "sovereignty_boot_seed_window_s", 7 * 24 * 3600))
            _prune_margin_s = 86400.0  # keep 1 day past the longest window
            _prune_every = int(metrics_cfg.get("sovereignty_marks_prune_every", 2000))
            _prune_state = {"n": 0}
            _persist_mark = None
            try:
                # Schema DDL — serialized on the writer thread (INV-Syn-28).
                # P3: each mark is now (ts, s, e, v) — one per-reply sovereignty
                # score S = 0.7E+0.3V AND its E/V components (so the soul-Chronicle
                # re-source renders the faithful 3-axis narrative without
                # recomputing). One-time migration: a table missing any of
                # `s`/`e`/`v` (the old count-ratio store, or the pre-E/V (ts,s)
                # store) is dropped + recreated. The marks are a rebuildable
                # metric (INV-Syn-25), never Titan data, so discarding old rows
                # is safe — the rolling window refills from new replies.
                def _ensure_marks_schema():
                    cols = {r[0] for r in store._conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'sovereignty_marks'").fetchall()}
                    if cols and not {"s", "e", "v"} <= cols:
                        store._conn.execute("DROP TABLE sovereignty_marks")
                    store._conn.execute(
                        "CREATE TABLE IF NOT EXISTS sovereignty_marks "
                        "(ts DOUBLE, s DOUBLE, e DOUBLE, v DOUBLE)")
                db_writer.submit_sync(_ensure_marks_schema)

                def _persist_mark(ts, s, e=0.0, v=0.0):
                    # Fired by the meter OUTSIDE its lock; the actual write is
                    # serialized on the SynthesisWriter. Soft-fail.
                    try:
                        db_writer.submit(lambda: store._conn.execute(
                            "INSERT INTO sovereignty_marks (ts, s, e, v) "
                            "VALUES (?, ?, ?, ?)",
                            [float(ts), float(s), float(e), float(v)]))
                        _prune_state["n"] += 1
                        if _prune_state["n"] % _prune_every == 0:
                            _cut = time.time() - _seed_window_s - _prune_margin_s
                            db_writer.submit(lambda: store._conn.execute(
                                "DELETE FROM sovereignty_marks WHERE ts < ?",
                                [_cut]))
                    except Exception:
                        pass
            except Exception as _ddl_err:
                logger.warning(
                    "[synthesis_worker] G9 sovereignty_marks DDL failed — durable "
                    "reseed disabled this session (live meter still works): %s",
                    _ddl_err)
                _persist_mark = None

            sovereignty_meter = SovereigntyRatioMeter(
                windows=list(metrics_cfg.get("sovereignty_windows", ["24h", "7d", "all"])),
                on_record=_persist_mark,
            )

            # Boot-seed: replay in-window durable marks (writer-serialized read,
            # `_persist=False` so the replay never re-writes). Bounded + soft-fail.
            if _persist_mark is not None:
                def _marks_query_fn(since_ts, limit):
                    try:
                        return db_writer.submit_sync(lambda: store._conn.execute(
                            "SELECT ts, s, e, v FROM sovereignty_marks "
                            "WHERE ts > ? ORDER BY ts ASC LIMIT ?",
                            [float(since_ts), int(limit)]).fetchall())
                    except Exception:
                        return []
                try:
                    _seed = boot_seed_from_marks(
                        sovereignty_meter, _marks_query_fn,
                        since_ts=time.time() - _seed_window_s,
                        cap=int(metrics_cfg.get("sovereignty_boot_seed_cap", 50000)))
                    logger.info(
                        "[synthesis_worker] G9 sovereignty boot-seed (durable marks): "
                        "scanned=%d replies=%d capped=%s",
                        _seed.get("scanned", 0), _seed.get("replies", 0),
                        _seed.get("capped", False))
                    # One-shot boot prune so a long-idle table is trimmed even if
                    # live inserts are sparse.
                    _cut0 = time.time() - _seed_window_s - _prune_margin_s
                    db_writer.submit(lambda: store._conn.execute(
                        "DELETE FROM sovereignty_marks WHERE ts < ?", [_cut0]))
                except Exception as _seed_err:
                    logger.warning(
                        "[synthesis_worker] G9 sovereignty boot-seed skipped "
                        "(non-blocking): %s", _seed_err)

            retrieval_latency_ring = LatencyRing(
                maxlen=int(metrics_cfg.get("retrieval_latency_ring_size", 1000)))

            # Operator-closure telemetry (2026-06-01): the §3 operator RECALL runs
            # in agno_worker (chat) + cognitive_worker (per-epoch), NOT here, so the
            # local `engine_recall._evaluator` is idle (chi=0) and the latency ring
            # was never fed. Cross-process RETRIEVAL_SAMPLE events aggregate the real
            # work into `_xproc_chi` (declared above) + the ring so
            # /v6/synthesis/metrics is honest.
            def _chi_stats_provider() -> dict:
                # chi-budget readout = cross-process recall chi (the real work) +
                # this worker's own evaluator (usually idle). Both summed so the
                # metric reflects ALL evaluator activity fleet-locally (B.5).
                local = {}
                try:
                    ev = getattr(engine_recall, "_evaluator", None)
                    if ev is not None and hasattr(ev, "get_stats"):
                        local = dict(ev.get_stats() or {})
                except Exception:
                    local = {}
                out = dict(local)
                out["available"] = True
                out["total_chi_spent"] = round(
                    float(local.get("total_chi_spent", 0.0))
                    + _xproc_chi["total_chi_spent"], 6)
                out["total_evaluations"] = (
                    int(local.get("total_evaluations", 0))
                    + _xproc_chi["total_evaluations"])
                out["total_chi_exhausted"] = (
                    int(local.get("total_chi_exhausted", 0))
                    + _xproc_chi["total_chi_exhausted"])
                return out

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
            logger.info(
                "[synthesis_worker] Phase 10 metrics ready — sovereignty meter "
                "+ aggregator, snapshot=%s", metrics_snapshot_path)
        except Exception as exc:
            logger.warning(
                "[synthesis_worker] Phase 10 metrics wiring failed: %s — "
                "observability disabled this session", exc, exc_info=True)

    # ── Dream-boundary orchestrator (Option C) ────────────────────────────
    # Replaces the 4 fire-and-forget dispatchers. ONE ordered thread runs the
    # dream passes SEQUENTIALLY so (a) the LLM judge's scored_by writes are
    # durably visible before the miner reads the window (INV-Syn-21 happens-
    # before, enforced by db_writer.flush()), and (b) newly-compiled skills are
    # verified right after mining (INV-Syn-20). Compute/LLM runs on THIS thread;
    # every store handle-op it triggers routes to the single SynthesisWriter
    # thread (the consolidation LLM await holds NO handle), so nothing races.
    # Rate-limited to <=1 orchestration per dream window.
    last_dream_orchestration_ts: float = 0.0
    dream_orchestration_lock = threading.Lock()

    def _maybe_run_dream_passes_async(dream_start_ts: float) -> None:
        nonlocal last_dream_orchestration_ts
        with dream_orchestration_lock:
            if dream_start_ts <= last_dream_orchestration_ts:
                logger.debug(
                    "[synthesis_worker] dream passes already ran for window "
                    "%.3f — skipping", dream_start_ts)
                return
            last_dream_orchestration_ts = dream_start_ts
        window_hours = int(skill_cfg.get("miner_window_hours", 168))
        since_ts = dream_start_ts - window_hours * 3600.0

        def _run_dream_sequence() -> None:
            _dseq_t0 = time.monotonic()
            logger.info(
                "[synthesis_worker] DREAM SEQUENCE START (window_since=%.0f) — "
                "compute runs on the synthesis-dream thread", since_ts)
            # 1) Oracle companion-verdict flush FIRST so the just-anchored
            #    verdict refs are in the window the judge + miner read (W6).
            if oracle_router is not None:
                try:
                    flushed = oracle_router.flush_companion_batches()
                    if flushed:
                        logger.info(
                            "[synthesis_worker] oracle companion batches flushed "
                            "at dream boundary: %s", flushed)
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] companion flush failed: %s", e,
                        exc_info=True)
            # 2) LLM judge — scores the tool-call window (INV-Syn-21: BEFORE miner).
            if llm_judge is not None:
                try:
                    summary = llm_judge.score_window(since_ts=since_ts)
                    logger.info(
                        "[synthesis_worker] llm_judge done — tool_calls=%d "
                        "unscored=%d scored=%d llm_calls=%d failures=%d",
                        summary.get("tool_calls_in_window", 0),
                        summary.get("unscored_in_window", 0),
                        summary.get("scored_now", 0),
                        summary.get("llm_calls", 0),
                        summary.get("llm_failures", 0))
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] llm_judge crashed: %s", e,
                        exc_info=True)
                # Barrier: the judge's scored_by-patch writes must be durably
                # flushed before the miner reads the window (INV-Syn-21).
                try:
                    db_writer.flush()
                except Exception:
                    pass
            # 3) Procedural miner — now sees a fully-scored window.
            if procedural_miner is not None:
                try:
                    summary = procedural_miner.mine_pass(
                        dream_pass_id=f"dream_{int(dream_start_ts)}")
                    logger.info(
                        "[synthesis_worker] procedural_miner done — txs=%d "
                        "recurrent=%d positive=%d negative=%d llm_calls=%d "
                        "failures=%d",
                        summary.get("txs_scanned", 0),
                        summary.get("clusters_recurrent", 0),
                        summary.get("positive_skills_compiled", 0),
                        summary.get("negative_skills_compiled", 0),
                        summary.get("llm_calls", 0),
                        summary.get("llm_failures", 0))
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] procedural_miner crashed: %s", e,
                        exc_info=True)
            # 4) Verify newly-compiled skills (INV-Syn-20). Without this pass
            #    verified_at stays NULL forever -> read_for_match(verified_only)
            #    is always empty -> no skill ever delegatable (skills=0 cause b).
            if skill_verifier is not None and procedural_skill_store is not None:
                try:
                    unverified = procedural_skill_store.list_unverified()
                    verified_n = 0
                    for sid in unverified:
                        try:
                            if skill_verifier.verify_skill(sid):
                                verified_n += 1
                        except Exception as e:
                            logger.warning(
                                "[synthesis_worker] verify_skill(%s) failed: %s",
                                sid, e, exc_info=True)
                    if unverified:
                        logger.info(
                            "[synthesis_worker] skill verification — "
                            "candidates=%d verified=%d",
                            len(unverified), verified_n)
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] skill verification pass failed: %s",
                        e, exc_info=True)
            # 5) ConsolidationPass — concept synthesis (independent of skills).
            if consolidation_pass is not None:
                try:
                    result = consolidation_pass.run()
                    logger.info(
                        "[synthesis_worker] consolidation_pass %s done — "
                        "created=%d bumped=%d rejected=%d llm_calls=%d "
                        "txs_mined=%d duration_ms=%.1f",
                        result.pass_id, len(result.concepts_created),
                        len(result.concepts_bumped), result.rejected_clusters,
                        result.llm_calls, result.txs_mined, result.duration_ms)
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] consolidation_pass crashed: %s", e,
                        exc_info=True)
            # 6) ForkGC sweep — after consolidation (which may mark new
            #    graduated/abandoned forks the sweep then handles).
            if fork_gc is not None:
                try:
                    report = fork_gc.sweep(dry_run=not fork_gc_live_mode)
                    logger.info(
                        "[synthesis_worker] fork_gc sweep done — visited=%d "
                        "pruned=%d skipped=%d dropped=%d dry_run=%s",
                        report.forks_visited, report.forks_pruned,
                        report.forks_skipped, report.total_nodes_dropped,
                        report.dry_run)
                except Exception as e:
                    logger.warning(
                        "[synthesis_worker] fork_gc sweep crashed: %s", e,
                        exc_info=True)
            logger.info(
                "[synthesis_worker] DREAM SEQUENCE END — total=%.1fs (GIL-held "
                "compute in these steps starves the 30s heartbeat → shm_pid_dead)",
                time.monotonic() - _dseq_t0)

        threading.Thread(
            target=_run_dream_sequence, name="synthesis-dream",
            daemon=True).start()

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
                # Operator-closure C1: MEMORY_RETRIEVAL_USED is now the per-ITEM
                # REINFORCEMENT signal ONLY (record_access on the cited gate).
                # The knowledge_moment denominator moved to the per-TURN
                # KNOWLEDGE_MOMENT signal below (SPEC §25.9) — recording a moment
                # per surfaced item inflated the denominator N× and crushed the
                # sovereignty ratio. Surfaced-not-cited is telemetry only.
                if payload.get("used_by_llm") is True:
                    with cache_lock:
                        store.record_access(item_id, float(ts))
                    events_recorded += 1
                else:
                    events_surfaced += 1
                continue

            if msg_type == RETRIEVAL_SAMPLE:
                # Operator-closure telemetry (2026-06-01) — a recall ran in
                # agno/cognitive (their own evaluator/ring); fold its latency +
                # chi into THIS worker's §18 metrics so they reflect the real
                # loop. Metrics-only (INV-Syn-25). Soft-fail.
                try:
                    if retrieval_latency_ring is not None:
                        retrieval_latency_ring.record(
                            str(payload.get("fork", "conversation")),
                            str(payload.get("source", "recall")),
                            float(payload.get("latency_ms", 0.0)))
                    _xproc_chi["total_chi_spent"] += float(
                        payload.get("chi_spent", 0.0) or 0.0)
                    _xproc_chi["total_evaluations"] += int(
                        payload.get("evaluations", 0) or 0)
                except Exception:
                    pass
                continue

            if msg_type == KNOWLEDGE_MOMENT:
                # P3 (RFP_synthesis_decision_authority) — the per-TURN post-LLM
                # event (the RFP's CHAT_TURN_COMPLETE, realized on this existing
                # non-blocking topic) now carries `s_reply` = the ONE sovereignty
                # score 0.7E+0.3V, computed agno-side at the OVG boundary. The
                # meter records it as one per-reply mark (rolling-mean S). The
                # recall-attribution leg below is unchanged.
                _km_ts = payload.get("ts")
                if not isinstance(_km_ts, (int, float)):
                    _km_ts = time.time()
                if sovereignty_meter is not None:
                    _s_reply = payload.get("s_reply")
                    if isinstance(_s_reply, (int, float)):
                        # E/V components ride the same event (chronicle re-source
                        # 3-axis). Absent/old emitters → 0.0, only zeroing the
                        # rolling-E/V projection, never the headline S.
                        _e_reply = payload.get("e_reply")
                        _v_reply = payload.get("v_reply")
                        _e_reply = float(_e_reply) if isinstance(
                            _e_reply, (int, float)) else 0.0
                        _v_reply = float(_v_reply) if isinstance(
                            _v_reply, (int, float)) else 0.0
                        try:
                            sovereignty_meter.record_reply(
                                float(_s_reply), float(_km_ts),
                                e=_e_reply, v=_v_reply)
                        except Exception:
                            pass
                # §7.E.0 — per-Engram citation attribution (OFF the chat hot path):
                # resolve the surfaced + cited tx_hash sets → their latest Engram(s)
                # and record recall-citation (feeds the live `fluent` axis + §7.E's
                # `(axes_at_recall, cited?)` reward). Independent of the meter; soft.
                if recall_attribution is not None:
                    try:
                        recall_attribution.record_recall(
                            payload.get("surfaced_tx") or [],
                            payload.get("cited_tx") or [],
                            float(_km_ts))
                    except Exception:
                        pass
                continue

            if msg_type == CGN_CONCEPT_GROUNDED:
                # Inner↔Outer Felt-Teaching Bridge §7.2 — event-sourced CGN
                # grounded-set. cgn_worker emits this when a concept matures across
                # ≥2 consumers (cgn_worker.py:877; payload {concept_id, consumers,
                # first_consumer, felt_centroid}). We absorb the concept_id (the Object
                # label) into FeltBridge's durable grounded-set so the next dream no
                # longer flags it as a felt-gap. CGN-felt RFP Phase B — also absorb the
                # felt_centroid (G18 read-down) so the producer can do a true
                # felt-vector frame_dependent comparison. G18: fire-and-forget event
                # consumption — NEVER a sync RPC into cgn_worker. Soft.
                if felt_bridge is not None:
                    try:
                        felt_bridge.record_grounded(
                            payload.get("concept_id"),
                            felt_centroid=payload.get("felt_centroid"))
                    except Exception:
                        pass
                continue

            if msg_type == TOOL_CALL_VERDICT_RECORD:
                # Operator-closure C2 (W7) — a chat-time self-oracle tool
                # (coding_sandbox in agno) shipped its pre-computed verdict.
                # Buffer it into the OracleRouter companion buffer for the
                # dream-boundary OracleVerdictBatch flush (→ §A.6 coverage).
                # No plug re-run; the tool already verified by executing.
                if oracle_router is None:
                    continue
                _ptx = payload.get("parent_tool_call_tx")
                if not isinstance(_ptx, str) or not _ptx:
                    continue
                try:
                    oracle_router.record_companion_verdict(
                        parent_tool_call_tx=_ptx,
                        oracle_id=str(payload.get("oracle_id", "coding_sandbox")),
                        verdict=str(payload.get("verdict", "unknown")),
                        evidence_ref=str(payload.get("evidence_ref", "")),
                        latency_ms=int(payload.get("latency_ms", 0) or 0),
                        fork="procedural",
                    )
                except Exception as _tcv_err:
                    logger.debug(
                        "[synthesis_worker] tool-call verdict record failed: %s",
                        _tcv_err)
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
                # The DREAM_STATE_CHANGED payload key is `is_dreaming` (set by
                # dream_state_worker; life_force/observatory/timechain all read
                # `is_dreaming`). This consumer read `dreaming` → ALWAYS False →
                # the entire dream-boundary operator loop (ConsolidationPass ③,
                # procedural miner + LLM judge ④, oracle companion flush W6,
                # ForkGC) NEVER fired on any real dream. Root cause of "0 real
                # concepts / 0 compiled skills". (2026-06-01)
                dreaming = bool(payload.get("is_dreaming", False))
                if dreaming:
                    dream_start_ts = float(
                        payload.get("ts", time.time())
                    )
                    logger.info(
                        "[synthesis_worker] DREAM_STATE_CHANGED dreaming=True "
                        "ts=%.3f — scheduling consolidation pass",
                        dream_start_ts,
                    )
                    # Option C — ONE ordered orchestrator thread runs the full
                    # dream sequence (oracle-flush → judge → flush-barrier →
                    # miner → skill-verify → consolidation → fork-gc) so the
                    # INV-Syn-21 happens-before holds (miner sees the judge's
                    # scored_by) and nothing races on the shared DuckDB/Kuzu/
                    # FAISS handles. Rate-limited to ≤1 per dream window inside.
                    _maybe_run_dream_passes_async(dream_start_ts)
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
        # Stop the single writer thread LAST — store.close()/bundle_store.close()
        # submit their final CHECKPOINT + conn-close ops through it (submit_sync,
        # so they have already drained by the time we get here); this joins the
        # now-idle writer thread.
        try:
            db_writer.close()
        except Exception:
            logger.exception("[synthesis_worker] db_writer close failed")
        logger.info("[synthesis_worker] shutdown complete")
