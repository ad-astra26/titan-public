"""
meditation_worker — Python L2 module owning the full meditation lifecycle:
M3 emergent driver + MeditationWatchdog cadence + orchestrator + phase
state machine + post-completion side effects + SOCIAL_CATALYST(dream_summary)
publisher + meditation_state.bin SHM writer.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.D`.
Maker greenlit Q1-Q5 inline 2026-05-15.

What this worker owns:
  1. `_meditation_tracker` state dict (last_epoch, count, count_since_nft,
     last_ts, in_meditation, current_phase) — restored from
     data/meditation_state.json on boot, persisted on SAVE_NOW + shutdown.
  2. M3 emergent meditation driver — epoch_gap + drain + GABA convergence
     check on every KERNEL_EPOCH_TICK; emits MEDITATION_REQUEST to self when
     conditions converge. Reads drain via chi_state.bin SHM + GABA via
     hormonal_state.bin SHM + is_dreaming via dream_state.bin SHM.
  3. MeditationWatchdog instance — `logic/meditation_watchdog.py` class
     unchanged; runs `.check()` at `_med_watchdog_interval` cadence; emits
     MEDITATION_HEALTH_ALERT + MEDITATION_RECOVERY_TIER_1 + Tier-2 escalation.
  4. Meditation orchestrator (dedicated thread with own asyncio loop) —
     receives MEDITATION_REQUEST → awaits memory_worker readiness via
     Guardian probe → emits work-RPC QUERY(action="run_meditation") to
     memory worker (300s allowlisted G19 per SPEC §3.4.1) → awaits RESPONSE
     via InFlightRegistry → publishes MEDITATION_COMPLETE 3-target fan-out
     (spirit + timechain + backup) + SOCIAL_CATALYST(dream_summary) +
     EPOCH_TICK + writes data/backup_trigger.json atomic.
  5. Phase state machine: idle → entering → deep → exiting → idle; publishes
     MEDITATION_PHASE_CHANGED per transition.
  6. `meditation_state.bin` SHM slot writer (G21 single writer; sole writer
     under Phase C) — dual-trigger republish on every KERNEL_EPOCH_TICK +
     on every transition.

Bus subscriptions (REQUIRED):
  • MEDITATION_REQUEST        — emergent (self-emit), dashboard force-trigger,
                                 watchdog Tier-1, EXPRESSION_FIRED KIN_SENSE bridge
  • MEDITATION_FORCE_END      — Maker dashboard manual abort
  • EXPRESSION_FIRED          — kin_sense trigger fan-out
  • KERNEL_EPOCH_TICK         — driver-check cadence + watchdog cadence + SHM republish
  • MODULE_SHUTDOWN           — clean shutdown signal

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • MEDITATION_REQUEST (self-emit only — emergent / watchdog / kin_sense bridge)
  • MEDITATION_PHASE_CHANGED    (on every phase transition, dst="all")
  • MEDITATION_COMPLETE         (on success, dst=spirit + timechain + backup)
  • MEDITATION_INTERRUPTED      (on abnormal termination, dst="all")
  • MEDITATION_HEALTH_ALERT     (watchdog detections, dst="core")
  • MEDITATION_RECOVERY_TIER_1  (watchdog Tier-1 recovery actions, dst="core")
  • MEDITATION_RECOVERY_TIER_2  (watchdog Tier-2 escalation, dst="core")
  • SOCIAL_CATALYST(type=dream_summary, dst="social")
  • EPOCH_TICK                  (post-completion, dst="all" + "timechain")
  • MODULE_HEARTBEAT            (every 30s)
  • MODULE_READY                (on init complete after watchdog self_test pass)

Persisted state: data/meditation_state.json (tracker dict — counts +
last_epoch + last_ts persist across restart).

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • REQUIRED: memory          — work-RPC target; readiness probe before first cycle
  • SOFT:     studio, observatory, social — cosmetic/side-effect targets

See:
  - SPEC v1.8.3 §9.B `meditation_worker` block
  - SPEC v1.8.3 §7.1 `meditation_state.bin` SHM slot row
  - SPEC v1.8.3 §8.7 bus event rows (3 new + 5 producer updates)
  - SPEC v1.8.3 §21 D-SPEC-57
  - PLAN_microkernel_phase_c_meditation_worker_extraction.md
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from queue import Empty
from typing import Any, Optional

from titan_hcl._phase_c_constants import (
    MEDITATION_STATE_SCHEMA_VERSION,
)
from titan_hcl.bus import (
    EPOCH_TICK,
    EXPRESSION_FIRED,
    KERNEL_EPOCH_TICK,
    MEDITATION_COMPLETE,
    MEDITATION_FORCE_END,
    MEDITATION_HEALTH_ALERT,
    MEDITATION_INTERRUPTED,
    MEDITATION_PHASE_CHANGED,
    MEDITATION_RECOVERY_TIER_1,
    MEDITATION_RECOVERY_TIER_2,
    MEDITATION_REQUEST,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    QUERY,
    SAVE_NOW,
    SOCIAL_CATALYST,
    STUDIO_RENDER_REQUEST,
    make_msg,
)
from titan_hcl.logic.meditation_state_publisher import MeditationStatePublisher
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
SHM_REPUBLISH_CADENCE_S = 1.0
RUN_MEDITATION_TIMEOUT_S = 300.0  # G19 allowlisted per phase_c_rpc_exemptions.yaml
TRACKER_PERSISTENCE_PATH = os.path.join("data", "meditation_state.json")
BACKUP_TRIGGER_PATH = os.path.join("data", "backup_trigger.json")


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 compliance)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[MeditationWorker] _send %s → %s failed: %s",
            msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str, stop_event: threading.Event) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s."""
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        stop_event.wait(HEARTBEAT_INTERVAL_S)


# ── SHM readers for emergent driver inputs ──────────────────────────────

def _build_shm_readers(titan_id: str):
    """Build the three sub-µs SHM readers needed by the emergent driver:
    neuromod_state.bin (GABA, index 5 in NEUROMOD_NAMES per D-SPEC-54),
    chi_state.bin (drain proxy from body weight), dream_state.bin
    (is_dreaming gate).

    Returns dict with reader instances or None entries on attach failure.
    Readers are reattach-tolerant — first failure logs INFO, callers handle
    None gracefully and retry on next tick.
    """
    from titan_hcl.core.state_registry import (
        CHI_STATE,
        NEUROMOD_STATE,
        StateRegistryReader,
        ensure_shm_root,
    )

    shm_root = ensure_shm_root(titan_id)
    readers: dict[str, Any] = {
        "neuromod": None,
        "chi": None,
        "dream": None,
    }

    # Best-effort attach — each slot is independently optional.
    try:
        readers["neuromod"] = StateRegistryReader(NEUROMOD_STATE, shm_root)
    except Exception as e:
        logger.info(
            "[MeditationWorker] neuromod_state.bin reader unavailable "
            "(emergent driver will use defaults): %s", e)

    try:
        readers["chi"] = StateRegistryReader(CHI_STATE, shm_root)
    except Exception as e:
        logger.info(
            "[MeditationWorker] chi_state.bin reader unavailable "
            "(emergent driver will use defaults): %s", e)

    try:
        from titan_hcl.logic.meditation_state_specs import (
            MEDITATION_STATE_SPEC,
        )
        # placeholder import - dream reader uses dream_state_specs
        from titan_hcl.logic.dream_state_specs import DREAM_STATE_SPEC
        readers["dream"] = StateRegistryReader(DREAM_STATE_SPEC, shm_root)
        _ = MEDITATION_STATE_SPEC  # silence linter
    except Exception as e:
        logger.info(
            "[MeditationWorker] dream_state.bin reader unavailable "
            "(is_dreaming gate will default False): %s", e)

    return readers


# NEUROMOD_NAMES axis order matches state_registry.NEUROMOD_STATE spec
# (DA, 5HT, NE, ACh, Endorphin, GABA). GABA is index 5.
_GABA_INDEX = 5
_DA_INDEX = 0
_NE_INDEX = 2
_FIELD_LEVEL = 0  # axis 1 index for level field per D-SPEC-54
_FIELD_GAIN = 1   # axis 1 index for gain field (Modulator.get_gain(): 1.0 at setpoint)


def _read_all_gains(readers: dict[str, Any]) -> Optional[list[float]]:
    """Read all 6 modulator `gain` values (field 1) from neuromod_state.bin.

    `gain = 1.0 + (level − setpoint)/setpoint × sensitivity` (neuromodulator.py),
    so it is the LIVE homeostatic-deviation signal: gain == 1.0 exactly when a
    modulator sits at its adapted setpoint (balanced). The meditation onset uses
    `mean|gain − 1|` as the homeostatic-imbalance ("agitation") measure
    (rFP_meditation_emergent_onset_redesign §2.1). Returns None on read failure.
    """
    reader = readers.get("neuromod")
    if reader is None:
        return None
    try:
        arr = reader.read()
        if arr is None or arr.shape != (6, 4):
            return None
        return [float(arr[i, _FIELD_GAIN]) for i in range(6)]
    except Exception:
        return None


def _read_arousal(readers: dict[str, Any]) -> float:
    """Arousal proxy = mean(NE, DA) level. High arousal resists meditation
    onset (agitation_drive multiplier). Default 0.5 on read failure."""
    reader = readers.get("neuromod")
    if reader is None:
        return 0.5
    try:
        arr = reader.read()
        if arr is None or arr.shape != (6, 4):
            return 0.5
        return float((arr[_NE_INDEX, _FIELD_LEVEL] + arr[_DA_INDEX, _FIELD_LEVEL]) / 2.0)
    except Exception:
        return 0.5


def evaluate_meditation_onset(
    *,
    gains: Optional[list[float]],
    arousal: float,
    epoch_gap: int,
    time_since_last: float,
    balance_ema: Optional[float],
    last_epoch: int,
    meditation_count: int,
    balance_band: float,
    balance_half_life: float,
    k_agit: float,
    debt_onset: int,
    debt_ramp: int,
    max_interval_s: float,
    min_epochs: int,
) -> dict:
    """Pure meditation onset-v2 decision (rFP_meditation_emergent_onset_redesign).

    Sleep-symmetric drive competition on the homeostatic-BALANCE axis. Pure +
    side-effect-free (no SHM/bus) so it is directly unit-testable; `_emergent_check`
    wraps it with the live reads + emit.

    Returns: {fire, reason, balance, agitation, balance_sustain, meditate_drive,
              agitation_drive, debt}.
    """
    agitation = (sum(abs(g - 1.0) for g in gains) / len(gains)) if gains else 1.0
    balance = max(0.0, min(1.0, 1.0 - agitation / max(1e-6, k_agit)))

    _alpha = 1.0 - 0.5 ** (1.0 / max(1.0, balance_half_life))
    balance_sustain = balance if balance_ema is None else (
        balance_ema + _alpha * (balance - balance_ema))

    debt = max(0.0, (epoch_gap - debt_onset) / max(1.0, debt_ramp))
    meditate_drive = balance_sustain * (1.0 + debt)
    agitation_drive = agitation * (0.5 + arousal * 0.5)

    first_ever = meditation_count == 0 and epoch_gap > min_epochs
    natural_fire = (
        last_epoch > 0
        and epoch_gap > min_epochs
        and agitation < balance_band      # genuinely balanced this tick (low deviation)
        and meditate_drive > agitation_drive
    )
    hard_floor = time_since_last > max_interval_s

    if hard_floor:
        reason = "hard_floor_max_interval"
    elif first_ever:
        reason = "first_ever"
    elif natural_fire:
        reason = "homeostatic_balance_sustained"
    else:
        reason = None

    return {
        "fire": reason is not None,
        "reason": reason,
        "balance": balance,
        "agitation": agitation,
        "balance_sustain": balance_sustain,
        "meditate_drive": meditate_drive,
        "agitation_drive": agitation_drive,
        "debt": debt,
    }


def _read_gaba(readers: dict[str, Any]) -> float:
    """Read GABA level from neuromod_state.bin SHM ((6,4) float32 LE).
    Default 0.5 on failure (neuromod default setpoint per legacy code).
    """
    reader = readers.get("neuromod")
    if reader is None:
        return 0.5
    try:
        arr = reader.read()
        if arr is None:
            return 0.5
        if arr.shape != (6, 4):
            return 0.5
        return float(arr[_GABA_INDEX, _FIELD_LEVEL])
    except Exception:
        return 0.5


# CHI_STATE field order per state_registry.CHI_STATE comment:
# total, spirit, mind, body, coherence, urgency.
_CHI_BODY_INDEX = 3


def _read_drain(readers: dict[str, Any]) -> float:
    """Read metabolic drain proxy from chi_state.bin SHM (low body chi =
    high drain). Returns approximate drain in [0.0, 1.0]. Default 0.5
    on read failure.
    """
    reader = readers.get("chi")
    if reader is None:
        return 0.5
    try:
        arr = reader.read()
        if arr is None:
            return 0.5
        if arr.shape != (6,):
            return 0.5
        body = float(arr[_CHI_BODY_INDEX])
        return max(0.0, min(1.0, 1.0 - body))
    except Exception:
        return 0.5


def _read_is_dreaming(readers: dict[str, Any]) -> bool:
    """Read is_dreaming flag from dream_state.bin SHM. Default False on failure."""
    reader = readers.get("dream")
    if reader is None:
        return False
    try:
        raw = reader.read_variable()
        if not raw:
            return False
        decoded = msgpack.unpackb(raw, raw=False)
        if not isinstance(decoded, dict):
            return False
        return bool(decoded.get("is_dreaming", False))
    except Exception:
        pass
    return False


# ── Tracker persistence helpers ─────────────────────────────────────────

def _load_tracker_from_disk() -> dict[str, Any]:
    """Load tracker dict from data/meditation_state.json. Empty dict on
    missing/malformed file. Used on boot to restore counts."""
    if not os.path.exists(TRACKER_PERSISTENCE_PATH):
        return {}
    try:
        with open(TRACKER_PERSISTENCE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        logger.warning(
            "[MeditationWorker] Failed to load tracker from %s: %s "
            "(starting fresh)", TRACKER_PERSISTENCE_PATH, e)
        return {}


def _persist_tracker(tracker: dict[str, Any]) -> None:
    """Atomic-write tracker dict to data/meditation_state.json.

    Called on SAVE_NOW + MODULE_SHUTDOWN + after every successful completion.
    """
    try:
        os.makedirs(os.path.dirname(TRACKER_PERSISTENCE_PATH), exist_ok=True)
        tmp = TRACKER_PERSISTENCE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(tracker, f)
        os.replace(tmp, TRACKER_PERSISTENCE_PATH)
    except Exception as e:
        logger.warning(
            "[MeditationWorker] Failed to persist tracker to %s: %s",
            TRACKER_PERSISTENCE_PATH, e)


def _write_backup_trigger_file(payload: dict, meditation_count: int) -> None:
    """Atomic write of data/backup_trigger.json for legacy file-trigger
    fallback compat with __init__.py:1187 backup watcher loop.

    Mirrors spirit_worker.py:8073-8087 atomic-write pattern.
    """
    try:
        os.makedirs(os.path.dirname(BACKUP_TRIGGER_PATH), exist_ok=True)
        # validate JSON-serializable
        json.dumps(payload)
        tmp = BACKUP_TRIGGER_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "payload": payload,
                "meditation_count": meditation_count,
                "ts": time.time(),
            }, f)
        os.replace(tmp, BACKUP_TRIGGER_PATH)
        logger.info(
            "[MeditationWorker] Backup trigger written for meditation #%d",
            meditation_count)
    except Exception as e:
        logger.error(
            "[MeditationWorker] F5 — Backup trigger write failed: %s",
            e, exc_info=True)
        raise


# ── Daemon-thread file-I/O executor (Chunk 1G, RFP §1.4) ─────────────────
#
# Per RFP_phase_c_enhancements.md §1.4 + §1G: move the two file writes off
# the meditation orchestrator hot path so disk-I/O latency variance can't
# extend cycle completion. The bus emit of MEDITATION_COMPLETE / *_PHASE_
# CHANGED stays synchronous; only the legacy-fallback file writes are
# offloaded. SAVE_NOW + MODULE_SHUTDOWN callsites stay synchronous (caller
# expects guaranteed persistence-on-completion).

_io_executor: Optional[Any] = None
_io_executor_lock = threading.Lock()


def _get_io_executor():
    """Lazy-init single-worker daemon-thread executor for off-hot-path I/O."""
    global _io_executor
    if _io_executor is None:
        with _io_executor_lock:
            if _io_executor is None:
                from concurrent.futures import ThreadPoolExecutor
                _io_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="meditation-io-")
    return _io_executor


def _shutdown_io_executor(wait: bool = True) -> None:
    """Drain pending file-I/O writes — called from meditation_worker_main
    finally block before the worker exits."""
    global _io_executor
    with _io_executor_lock:
        if _io_executor is not None:
            _io_executor.shutdown(wait=wait)
            _io_executor = None


def _schedule_backup_trigger(
    send_queue,
    name: str,
    med_payload: dict,
    meditation_count: int,
    titan_id: str,
) -> None:
    """Fire-and-forget _write_backup_trigger_file on daemon thread. On
    failure emit MEDITATION_HEALTH_ALERT via send_queue (thread-safe).
    Caller must NOT mutate med_payload after submit (we snapshot)."""
    payload_snapshot = dict(med_payload)

    def _runner() -> None:
        try:
            _write_backup_trigger_file(payload_snapshot, meditation_count)
        except Exception as e:
            try:
                _send(send_queue, MEDITATION_HEALTH_ALERT, name, "core", {
                    "severity": "MEDIUM",
                    "failure_mode": "F5_TRIGGER_FILE_WRITE",
                    "detail": f"Trigger file write failed: {e}",
                    "diagnostic": {
                        "trigger_path": BACKUP_TRIGGER_PATH,
                        "meditation_count": meditation_count,
                        "error": str(e),
                    },
                    "ts": time.time(),
                    "titan_id": titan_id,
                })
            except Exception:
                logger.exception(
                    "[MeditationWorker] Backup trigger alert emit failed")
    _get_io_executor().submit(_runner)


def _schedule_persist_tracker(tracker: dict) -> None:
    """Fire-and-forget _persist_tracker on daemon thread. Snapshot to
    detach from caller mutation."""
    snapshot = dict(tracker)
    _get_io_executor().submit(_persist_tracker, snapshot)


# ── In-flight registry for awaiting memory_worker RESPONSEs ──────────────

class _InFlightRegistry:
    """Minimal rid-keyed Future registry for work-RPC reply correlation.

    Mirrors `titan_hcl.modules._memory_dispatch.InFlightRegistry`. The
    main recv loop is the sole `recv_queue` reader; the orchestrator
    thread blocks on Futures registered here, the main loop resolves them
    when it sees a matching-rid RESPONSE.
    """

    def __init__(self) -> None:
        from concurrent.futures import Future
        self._Future = Future
        self._lock = threading.Lock()
        self._futures: dict[str, Any] = {}

    def register(self, rid: str):
        fut = self._Future()
        with self._lock:
            self._futures[rid] = fut
        return fut

    def resolve(self, msg: dict) -> bool:
        rid = msg.get("rid") if isinstance(msg, dict) else None
        if not rid:
            return False
        with self._lock:
            fut = self._futures.pop(rid, None)
        if fut is None:
            return False
        try:
            fut.set_result(msg)
        except Exception:
            pass
        return True

    def cancel(self, rid: str) -> None:
        with self._lock:
            self._futures.pop(rid, None)


# ── Orchestrator thread ─────────────────────────────────────────────────

class _OrchestratorState:
    """Mutable container shared between main loop + orchestrator thread."""

    def __init__(self):
        self.request_queue: deque[dict] = deque()
        self.request_event = threading.Event()
        self.busy = False  # True while a cycle is in flight
        self.lock = threading.Lock()

    def enqueue(self, payload: dict) -> bool:
        """Returns True if accepted (orchestrator was idle), False if busy."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.request_queue.append(payload)
        self.request_event.set()
        return True

    def mark_idle(self):
        with self.lock:
            self.busy = False

    def drain_one(self) -> Optional[dict]:
        with self.lock:
            return self.request_queue.popleft() if self.request_queue else None


def _orchestrator_loop(
    state: _OrchestratorState,
    in_flight: _InFlightRegistry,
    publisher: MeditationStatePublisher,
    send_queue,
    name: str,
    stop_event: threading.Event,
) -> None:
    """Daemon thread — drives meditation cycles serially.

    For each MEDITATION_REQUEST:
      1. Phase idle → entering, publish MEDITATION_PHASE_CHANGED + SHM republish
      2. Send QUERY(action="run_meditation") to memory worker, register rid
         in in-flight registry
      3. Phase entering → deep, publish
      4. Block on Future for up to RUN_MEDITATION_TIMEOUT_S (heartbeat through send_queue)
      5. Phase deep → exiting, publish
      6. On success: emit MEDITATION_COMPLETE × 3 targets + SOCIAL_CATALYST(dream_summary)
         + EPOCH_TICK + backup_trigger.json write + tracker update
      7. On timeout/error: emit MEDITATION_INTERRUPTED(reason=...)
      8. Phase exiting → idle, publish + persist tracker
    """
    from concurrent.futures import TimeoutError as _FutureTimeoutError

    while not stop_event.is_set():
        # Wait for a request.
        state.request_event.wait(timeout=1.0)
        state.request_event.clear()
        if stop_event.is_set():
            break

        req_payload = state.drain_one()
        if req_payload is None:
            state.mark_idle()
            continue

        trigger_source = str(req_payload.get("source", "unknown"))
        epoch_id = int(req_payload.get("epoch_id", 0) or 0)

        prev_phase = publisher.snapshot()["tracker"]["current_phase"]
        logger.info(
            "[MeditationWorker] cycle starting — trigger=%s epoch_id=%d phase=%s→entering",
            trigger_source, epoch_id, prev_phase)

        # Phase: idle → entering
        publisher.set_in_meditation(True)
        publisher.set_phase("entering")
        publisher.publish()
        _send(send_queue, MEDITATION_PHASE_CHANGED, name, "all", {
            "phase": "entering",
            "previous_phase": prev_phase,
            "epoch_id": epoch_id,
            "ts": time.time(),
        })

        # Send work-RPC.
        rid = uuid.uuid4().hex
        fut = in_flight.register(rid)
        query_payload = {"action": "run_meditation"}
        _send(send_queue, QUERY, name, "memory", query_payload, rid=rid)

        # Phase: entering → deep
        publisher.set_phase("deep")
        publisher.publish()
        _send(send_queue, MEDITATION_PHASE_CHANGED, name, "all", {
            "phase": "deep",
            "previous_phase": "entering",
            "epoch_id": epoch_id,
            "ts": time.time(),
        })

        # Block on Future with bounded slices so we can heartbeat through send_queue.
        deadline = time.time() + RUN_MEDITATION_TIMEOUT_S
        result: Optional[dict] = None
        last_hb = 0.0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                in_flight.cancel(rid)
                logger.warning(
                    "[MeditationWorker] run_meditation timeout after %.0fs "
                    "(rid=%s)", RUN_MEDITATION_TIMEOUT_S, rid[:8])
                break
            try:
                reply = fut.result(timeout=min(3.0, remaining))
                if isinstance(reply, dict):
                    result = reply.get("payload", {}) if isinstance(
                        reply.get("payload"), dict) else {}
                break
            except _FutureTimeoutError:
                if time.time() - last_hb > 3.0:
                    _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
                    last_hb = time.time()
                continue
            except Exception as e:
                logger.error(
                    "[MeditationWorker] Future result error (rid=%s): %s",
                    rid[:8], e)
                break

        # Phase: deep → exiting
        publisher.set_phase("exiting")
        publisher.publish()
        _send(send_queue, MEDITATION_PHASE_CHANGED, name, "all", {
            "phase": "exiting",
            "previous_phase": "deep",
            "epoch_id": epoch_id,
            "ts": time.time(),
        })

        # Branch on success / failure.
        if result is None:
            # Timeout or error path
            logger.warning(
                "[MeditationWorker] cycle did not complete (trigger=%s) — "
                "emitting MEDITATION_INTERRUPTED(reason=timeout)", trigger_source)
            publisher.set_in_meditation(False)
            publisher.set_phase("idle")
            publisher.publish()
            _send(send_queue, MEDITATION_INTERRUPTED, name, "all", {
                "reason": "timeout",
                "triggered_by": trigger_source,
                "meditation_count": publisher.get_count(),
                "ts": time.time(),
            })
            _send(send_queue, MEDITATION_PHASE_CHANGED, name, "all", {
                "phase": "idle",
                "previous_phase": "exiting",
                "epoch_id": epoch_id,
                "ts": time.time(),
            })
            state.mark_idle()
            continue

        # Success path — apply completion to tracker
        promoted = int(result.get("promoted", 0) or 0)
        pruned = int(result.get("pruned", 0) or 0)
        success = bool(result.get("success", False))
        completion = {
            "epoch": epoch_id,
            "promoted": promoted,
            "pruned": pruned,
            "trigger": trigger_source,
            "success": success,
            "ts": time.time(),
        }
        publisher.record_completion(epoch_id=epoch_id, completion=completion)
        publisher.publish()

        # MEDITATION_COMPLETE — single broadcast (dst="all"). Per the Rust
        # broker fanout (§8.2), dst="all" delivers to every subscriber whose
        # `subscribed_topics` includes the type. Verified current consumers:
        #   - timechain_worker: broadcast_topic ✓ (block-seal cascade)
        #   - backup_worker:    broadcast_topic ✓ (personality/soul backup)
        #   - cognitive_worker: broadcast_topic ✓ (coordinator.meditation_observe)
        #   - life_force_worker / memory_worker / social_worker: broadcast_topics ✓
        #   - titan_HCL parent: `subscribe("core", types=[MEDITATION_COMPLETE])`
        #     in `_meditation_chronicle_loop` — the broadcast topic-filter
        #     contract delivers dst="all" of that type to this virtual sub.
        #
        # This closes BUG-MEDITATION-COMPLETE-FANOUT-STARVES-BROADCAST-SUBSCRIBERS:
        # the legacy directed fan-out ("spirit","timechain","backup") starved the
        # 4 workers above (they declared the topic but only dst="all" matches the
        # broadcast contract). The "spirit" leg was already a dead target after
        # D-SPEC-116 spirit_worker retirement. The retained dst="all" is the
        # SPEC-correct broadcast pattern the broadcast_topics declarations imply.
        med_payload = dict(completion)
        _send(send_queue, MEDITATION_COMPLETE, name, "all", med_payload)

        # STUDIO_RENDER_REQUEST(type=meditation) — v1.9.4 §4.K wire-up.
        # Closes the post-§4.D regression where meditation art generation was
        # silently dropped when _meditation_loop was extracted (last meditation
        # art was 2026-05-15 15:26 UTC pre-§4.D shipment). Fire-and-forget:
        # studio_worker emits STUDIO_RENDER_COMPLETED back to broadcast; the
        # render history hook lives in observatory_writer (future). meditation
        # art generation is a cosmetic side-effect — not blocking the cycle.
        # Gated on promoted>0 OR success (matches legacy Fix #B2 from
        # rFP_meditation_worker_latency: art belongs alongside persisted
        # promotions even if the bus.request return arrived late).
        if promoted > 0 or success:
            persistent_count = int(result.get("persistent_count", 0) or 0)
            _send(send_queue, STUDIO_RENDER_REQUEST, name, "studio", {
                "request_id": uuid.uuid4().hex,
                "type": "meditation",
                "args": {
                    "state_root": f"MEDITATION_V3_E{publisher.get_count()}",
                    "age_nodes": persistent_count,
                    "avg_intensity": min(10, promoted + 3),
                },
                "ts": time.time(),
            })

        # SOCIAL_CATALYST(dream_summary) — D8-3 catalyst-producer site #5 close
        # (moved from spirit_worker.py:8110-8120).
        _send(send_queue, SOCIAL_CATALYST, name, "social", {
            "type": "dream_summary",
            "significance": 0.7,
            "content": "Meditation #%d: %d memories crystallized, %d pruned" % (
                publisher.get_count(), promoted, pruned),
            "data": med_payload,
        })

        # EPOCH_TICK — preserves legacy plugin.py:3869-3871 emit signaling
        # epoch boundary to TimeChain block-sealing path.
        epoch_payload = {"epoch": epoch_id, "promoted": promoted, "pruned": pruned}
        _send(send_queue, EPOCH_TICK, name, "all", epoch_payload)
        _send(send_queue, EPOCH_TICK, name, "timechain", epoch_payload)

        # backup_trigger.json (legacy file-trigger compat — __init__.py:1187
        # watcher loop fallback when bus path fails). Chunk 1G (RFP §1.4):
        # off-hot-path daemon thread; failure emits MEDITATION_HEALTH_ALERT
        # via send_queue from the daemon thread.
        _schedule_backup_trigger(
            send_queue, name, med_payload,
            publisher.get_count(),
            getattr(publisher, "_titan_id", "?"))

        # Persist tracker to disk — off-hot-path (Chunk 1G).
        _schedule_persist_tracker(publisher.get_tracker())

        # Phase: exiting → idle
        publisher.set_phase("idle")
        publisher.publish()
        _send(send_queue, MEDITATION_PHASE_CHANGED, name, "all", {
            "phase": "idle",
            "previous_phase": "exiting",
            "epoch_id": epoch_id,
            "ts": time.time(),
        })

        logger.info(
            "[MeditationWorker] cycle #%d COMPLETE — trigger=%s "
            "promoted=%d pruned=%d success=%s",
            publisher.get_count(), trigger_source, promoted, pruned, success)

        state.mark_idle()


# ── Main entry ──────────────────────────────────────────────────────────

@with_error_envelope(module_name="meditation", subsystem="entry", severity=_phase11_sev.FATAL)
def meditation_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id, load [meditation] config from titan_params.toml
      2. Restore tracker from data/meditation_state.json
      3. Build MeditationStatePublisher + MeditationWatchdog + self_test
      4. Build SHM readers for emergent driver inputs
      5. Start heartbeat thread + orchestrator thread
      6. First SHM publish (cold defaults restored from disk)
      7. Emit MODULE_READY (only after self_test pass)
      8. Main loop: drain recv_queue, dispatch:
           - QUERY (rid match) → in_flight.resolve (orchestrator awaits)
           - RESPONSE (rid match) → in_flight.resolve
           - MEDITATION_REQUEST → enqueue orchestrator (drop if busy)
           - MEDITATION_FORCE_END → reset in_meditation + emit INTERRUPTED
           - EXPRESSION_FIRED (composite=KIN_SENSE) → bridge to MEDITATION_REQUEST self-emit
           - KERNEL_EPOCH_TICK → emergent driver check + watchdog check + 1Hz SHM republish
           - SAVE_NOW / MODULE_SHUTDOWN → persist tracker + (shutdown only) exit
    """
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id(config.get("titan_id") if config else None)

    logger.info(
        "[MeditationWorker] booting — titan_id=%s name=%s "
        "(SPEC v1.8.3 §9.B / D-SPEC-57 / rFP §4.D)",
        titan_id, name)

    # ── Load [meditation] config from titan_params.toml ────────────
    med_cfg: dict[str, Any] = {}
    try:
        try:
            import tomllib as _tomllib  # py311+
        except ImportError:
            import tomli as _tomllib  # type: ignore
        params_path = os.path.join("titan_hcl", "titan_params.toml")
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                med_cfg = _tomllib.load(f).get("meditation", {}) or {}
    except Exception as e:
        logger.warning(
            "[MeditationWorker] Failed to load titan_params.toml "
            "[meditation] section: %s — using defaults", e)

    # Emergent driver thresholds (mirrors spirit_worker.py:2227-2231).
    _med_emergent = bool(med_cfg.get("emergent_enabled", True))
    _med_min_epochs = int(med_cfg.get("min_interval_epochs", 1500))
    # Onset v2 (rFP_meditation_emergent_onset_redesign / D-SPEC-107) — sleep-
    # symmetric emergent drive competition on the homeostatic-BALANCE axis.
    # Replaces the legacy unreachable `gaba > setpoint+offset` hard AND-gate.
    _med_balance_band = float(med_cfg.get("balance_band", 0.25))          # mean|gain−1| below this ⇒ "balanced"
    _med_balance_half_life = float(med_cfg.get("balance_half_life_epochs", 50.0))  # sustained-calm EMA half-life
    _med_debt_onset = int(med_cfg.get("debt_onset_epochs", 1500))         # epochs before meditation-debt starts ramping
    _med_debt_ramp = int(med_cfg.get("debt_ramp_epochs", 900))            # ramp width (mirror sleep debt)
    _med_max_interval_s = float(med_cfg.get("max_interval_seconds", 21600.0))  # HARD cadence floor (≈6h ⇒ ~4×/day)
    _med_inertia_s = float(med_cfg.get("meditation_inertia_seconds", 1800.0))  # refractory after a completed meditation
    _med_k_agit = float(med_cfg.get("agitation_norm", 1.0))               # normalizer: balance = 1 − agitation/k
    # Legacy emergent params (drain_threshold/gaba_offset) retained for the
    # observable log only; no longer gate onset (the hard gaba gate is removed).
    _med_drain_threshold = float(med_cfg.get("drain_threshold", 0.55))
    _med_gaba_offset = float(med_cfg.get("gaba_offset", 0.10))

    # Watchdog config (mirrors spirit_worker.py:2236-2247).
    _med_watchdog_interval = float(
        med_cfg.get("watchdog_check_interval_seconds", 60))
    _med_watchdog_detection_only = bool(
        med_cfg.get("watchdog_detection_only", True))
    _med_tier2_window_s = float(
        med_cfg.get("watchdog_tier2_window_seconds", 600.0))
    _med_tier2_threshold = int(
        med_cfg.get("watchdog_tier2_reset_threshold", 2))
    _med_tier2_cooldown_s = float(
        med_cfg.get("watchdog_tier2_cooldown_seconds", 1800.0))
    _med_tier2_enabled = bool(med_cfg.get("watchdog_tier2_enabled", True))
    _med_watchdog_last_check = 0.0
    _med_tier1_reset_history: dict[str, list[float]] = {}
    _med_tier2_recent: float = 0.0

    # ── Restore tracker from disk ────────────────────────────────────
    restored_tracker = _load_tracker_from_disk()

    # ── Build publisher + watchdog ───────────────────────────────────
    publisher = MeditationStatePublisher(titan_id)
    if restored_tracker:
        publisher.restore_tracker(restored_tracker)
        logger.info(
            "[MeditationWorker] tracker restored — count=%d count_since_nft=%d "
            "last_epoch=%d last_ts=%.0f",
            publisher.get_count(), publisher.get_count_since_nft(),
            publisher.snapshot()["tracker"]["last_epoch"],
            publisher.snapshot()["tracker"]["last_ts"])

    watchdog = None
    if med_cfg.get("watchdog_enabled", True):
        try:
            from titan_hcl.logic.meditation_watchdog import MeditationWatchdog
            watchdog = MeditationWatchdog(
                titan_id=titan_id,
                bootstrap_hours=float(med_cfg.get("watchdog_bootstrap_hours", 12.0)),
                min_alert_hours=float(med_cfg.get("watchdog_min_alert_hours", 3.0)),
                gap_window=int(med_cfg.get("watchdog_gap_window", 50)),
                stuck_threshold_seconds=float(
                    med_cfg.get("watchdog_stuck_threshold_seconds", 600.0)),
                backup_lag_threshold=int(
                    med_cfg.get("watchdog_backup_lag_threshold", 2)),
                zero_promoted_streak_threshold=int(
                    med_cfg.get("watchdog_zero_promoted_streak", 3)),
            )
            if not watchdog.self_test():
                logger.critical(
                    "[MeditationWorker] MeditationWatchdog self-test FAILED "
                    "— disabling watchdog (won't deploy unverified safety)")
                watchdog = None
            else:
                publisher.update_watchdog_snapshot(watchdog.health_snapshot())
                logger.info(
                    "[MeditationWorker] MeditationWatchdog initialized "
                    "(detection_only=%s, interval=%.0fs, bootstrap=%.0fh, "
                    "min_alert_floor=%.0fh)",
                    _med_watchdog_detection_only, _med_watchdog_interval,
                    watchdog.bootstrap_hours, watchdog.min_alert_hours)
        except Exception as e:
            logger.error(
                "[MeditationWorker] Watchdog init failed — continuing "
                "without watchdog: %s", e)
            watchdog = None

    # ── SHM readers for emergent driver ──────────────────────────────
    shm_readers = _build_shm_readers(titan_id)

    # ── Threading scaffolding ────────────────────────────────────────
    stop_event = threading.Event()
    in_flight = _InFlightRegistry()
    orch_state = _OrchestratorState()

    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(send_queue, name, stop_event),
        daemon=True, name=f"meditation-hb-{name}")
    hb_thread.start()

    orch_thread = threading.Thread(
        target=_orchestrator_loop,
        args=(orch_state, in_flight, publisher, send_queue, name, stop_event),
        daemon=True, name=f"meditation-orch-{name}")
    orch_thread.start()

    # ── First SHM publish (cold defaults + restored counts) ─────────
    publisher.publish()

    _send(send_queue, MODULE_READY, name, "guardian", {
        "titan_id": titan_id,
        "module": "meditation_worker",
        "version": "1.8.3",
        "schema_version": MEDITATION_STATE_SCHEMA_VERSION,
        "spec_ref": "D-SPEC-57",
        "restored_count": publisher.get_count(),
    })
    logger.info(
        "[MeditationWorker] MODULE_READY emitted — meditation_state.bin SHM "
        "initialized (count=%d, in_meditation=False, phase=idle)",
        publisher.get_count())

    # ── Counters ─────────────────────────────────────────────────────
    request_count = 0
    force_end_count = 0
    expression_fired_count = 0
    kernel_tick_count = 0
    last_shm_republish_ts = time.time()

    # Onset-v2 emergent state (rFP_meditation §2.2-2.5):
    #   balance_ema — sustained-calm EMA over epochs (None until first sample).
    # (π-CLUSTER_END acceleration is a future enhancement — the meditation
    #  worker has no cluster signal source today; epoch-EMA is the window.)
    _onset_state = {"balance_ema": None}

    # Helper: emergent meditation onset (called on every KERNEL_EPOCH_TICK).
    # Sleep-symmetric drive competition on the homeostatic-balance axis:
    #   agitation       = mean_i |gain_i − 1|            (live homeostatic deviation)
    #   balance         = 1 − agitation/k                (1 ⇒ all modulators at setpoint)
    #   balance_sustain = EMA(balance)                   (rewards a held calm window)
    #   debt            = max(0,(epoch_gap−onset)/ramp)  (mirror sleep debt — anti-starvation)
    #   meditate_drive  = balance_sustain × (1+debt)     [π-accel boosts]
    #   agitation_drive = agitation × (0.5+arousal×0.5)  (dysregulation/arousal resists)
    #   ONSET when meditate_drive > agitation_drive  (no hard threshold)
    #   HARD floor: time_since_last > max_interval_s ⇒ force (guaranteed ~4×/day).
    def _emergent_check(current_epoch_id: int) -> None:
        nonlocal request_count
        if not _med_emergent:
            return
        if publisher.is_in_meditation():
            return
        if _read_is_dreaming(shm_readers):
            return

        snap = publisher.snapshot()["tracker"]
        last_epoch = snap.get("last_epoch", 0)
        last_ts = float(snap.get("last_ts", 0.0) or 0.0)
        epoch_gap = current_epoch_id - last_epoch if current_epoch_id else 0
        now = time.time()
        time_since_last = (now - last_ts) if last_ts > 0 else float("inf")

        # Meditation-inertia refractory (mirror dreaming wake-inertia).
        if time_since_last < _med_inertia_s:
            return

        decision = evaluate_meditation_onset(
            gains=_read_all_gains(shm_readers),
            arousal=_read_arousal(shm_readers),
            epoch_gap=epoch_gap,
            time_since_last=time_since_last,
            balance_ema=_onset_state["balance_ema"],
            last_epoch=last_epoch,
            meditation_count=publisher.get_count(),
            balance_band=_med_balance_band,
            balance_half_life=_med_balance_half_life,
            k_agit=_med_k_agit,
            debt_onset=_med_debt_onset,
            debt_ramp=_med_debt_ramp,
            max_interval_s=_med_max_interval_s,
            min_epochs=_med_min_epochs,
        )
        # Always advance the sustained-calm EMA (even when not firing).
        _onset_state["balance_ema"] = decision["balance_sustain"]

        if not decision["fire"]:
            return

        request_count += 1
        logger.info(
            "[MEDITATION] Emergent onset (%s) — balance=%.3f sustain=%.3f "
            "agitation=%.3f m_drive=%.3f a_drive=%.3f debt=%.2f gap=%d "
            "since=%.0fs (self-emit MEDITATION_REQUEST)",
            decision["reason"], decision["balance"], decision["balance_sustain"],
            decision["agitation"], decision["meditate_drive"],
            decision["agitation_drive"], decision["debt"], epoch_gap,
            time_since_last)
        _send(send_queue, MEDITATION_REQUEST, name, "meditation", {
            "epoch_id": current_epoch_id,
            "epoch_gap": epoch_gap,
            "balance": round(decision["balance"], 4),
            "balance_sustain": round(decision["balance_sustain"], 4),
            "agitation": round(decision["agitation"], 4),
            "source": "emergent_driver_v2",
            "reason": decision["reason"],
            "ts": now,
        })

    # Helper: fire watchdog check (60s cadence inside KERNEL_EPOCH_TICK).
    def _watchdog_check(now: float) -> None:
        nonlocal _med_watchdog_last_check, _med_tier2_recent
        if watchdog is None:
            return
        if now - _med_watchdog_last_check < _med_watchdog_interval:
            return
        _med_watchdog_last_check = now

        # Read backup_state.json for F4 lag detection (best-effort).
        backup_count: Optional[int] = None
        try:
            bs_path = os.path.join("data", "backup_state.json")
            if os.path.exists(bs_path):
                with open(bs_path) as f:
                    backup_count = int(json.load(f).get("meditation_count", 0))
        except Exception:
            pass

        tracker_view = publisher.snapshot()["tracker"]
        try:
            alerts = watchdog.check(tracker_view, now, backup_state_count=backup_count)
        except Exception as e:
            logger.warning("[MeditationWatchdog] check raised: %s", e)
            return

        publisher.update_watchdog_snapshot(watchdog.health_snapshot())

        for alert in alerts:
            alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else alert
            logger.warning(
                "[MeditationWatchdog] %s %s — %s",
                alert_dict.get("severity"), alert_dict.get("failure_mode"),
                alert_dict.get("detail"))
            publisher.record_alert(alert_dict)
            _send(send_queue, MEDITATION_HEALTH_ALERT, name, "core", {
                **alert_dict,
                "detection_only": _med_watchdog_detection_only,
                "titan_id": titan_id,
            })

            # Tier-3 Maker alert (HIGH/CRITICAL via Telegram).
            if alert_dict.get("severity") in ("HIGH", "CRITICAL"):
                try:
                    from titan_hcl.utils.maker_alert import send_maker_alert
                    alert_key = (
                        f"meditation.{titan_id}.{alert_dict.get('failure_mode')}")
                    alert_text = (
                        f"🧘 *Titan {titan_id} meditation "
                        f"{alert_dict.get('severity')}*\n"
                        f"Mode: `{alert_dict.get('failure_mode')}`\n"
                        f"{alert_dict.get('detail')}\n"
                        f"Tier-1 recovery: "
                        f"{'ACTIVE' if not _med_watchdog_detection_only else 'detection-only'}"
                    )
                    send_maker_alert(alert_text, alert_key)
                except Exception as e:
                    logger.debug(
                        "[MeditationWatchdog] Tier-3 alert error: %s", e)

            # Tier-1 recovery actions (gated by detection_only flag).
            if _med_watchdog_detection_only:
                continue
            failure_mode = alert_dict.get("failure_mode")
            if failure_mode == "F1_F2_OVERDUE":
                # F1/F2 cadence force-trigger RETIRED (rFP_meditation_emergent_
                # onset_redesign / D-SPEC-107). Cadence is now owned by the onset-v2
                # hard floor (`max_interval_seconds` in _emergent_check), so the
                # watchdog no longer competes as a second force path. It still
                # DETECTS + alerts the overdue condition (Tier-3 maker alert above)
                # so a genuinely-stuck onset is still surfaced — it just doesn't
                # force here. The drain/GABA `classify_overdue` heuristic
                # (drain_flat AND gaba_flat) was the gate that never fired for the
                # live drain/GABA regime and is no longer used.
                logger.info(
                    "[MeditationWatchdog] F1/F2 overdue detected — detection-only "
                    "(cadence owned by onset-v2 max_interval floor; no watchdog "
                    "force-trigger). diagnostic=%s",
                    alert_dict.get("diagnostic", {}))
            elif failure_mode == "F3_F6_STUCK":
                stuck_min = alert_dict.get("diagnostic", {}).get(
                    "stuck_for_minutes", "?")
                logger.warning(
                    "[MeditationWatchdog] Tier-1: F3/F6 stuck %s min — "
                    "resetting in_meditation flag", stuck_min)
                publisher.set_in_meditation(False)
                publisher.set_phase("idle")
                publisher.publish()
                # Reset orchestrator busy flag (cycle is dead).
                orch_state.mark_idle()
                _send(send_queue, MEDITATION_RECOVERY_TIER_1, name, "core", {
                    "titan_id": titan_id,
                    "failure_mode": "F3_F6_STUCK",
                    "action": "reset_in_meditation_flag",
                    "stuck_for_minutes": stuck_min,
                })
                _send(send_queue, MEDITATION_INTERRUPTED, name, "all", {
                    "reason": "watchdog_tier1_reset",
                    "triggered_by": "F3_F6_STUCK",
                    "meditation_count": publisher.get_count(),
                    "ts": time.time(),
                })

                # Tier-2 escalation tracking.
                if _med_tier2_enabled:
                    hist = _med_tier1_reset_history.setdefault("F3_F6_STUCK", [])
                    hist.append(now)
                    hist[:] = [t for t in hist if now - t <= _med_tier2_window_s]
                    in_cooldown = (now - _med_tier2_recent) < _med_tier2_cooldown_s
                    if len(hist) >= _med_tier2_threshold and not in_cooldown:
                        _med_tier2_recent = now
                        logger.critical(
                            "[MeditationWatchdog] Tier-2 escalation: F3/F6 "
                            "fired %d times in %.0fmin — Tier-1 reset alone "
                            "insufficient. Memory_worker likely needs restart.",
                            len(hist), _med_tier2_window_s / 60)
                        _send(send_queue, MEDITATION_RECOVERY_TIER_2, name, "core", {
                            "titan_id": titan_id,
                            "failure_mode": "F3_F6_STUCK",
                            "action": "tier2_critical_log",
                            "escalation_count": len(hist),
                            "ts": now,
                        })

    # ── Main dispatch loop ───────────────────────────────────────────
    try:
        while True:
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                # 1Hz SHM republish floor (defensive against missing KERNEL_EPOCH_TICK).
                now = time.time()
                if (now - last_shm_republish_ts) >= SHM_REPUBLISH_CADENCE_S:
                    publisher.publish()
                    last_shm_republish_ts = now
                continue

            if msg is None:
                continue

            # Give in_flight registry first dibs (orchestrator awaits on rid).
            if isinstance(msg, dict) and in_flight.resolve(msg):
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            if msg_type == MEDITATION_REQUEST:
                request_count += 1
                logger.info(
                    "[MeditationWorker] MEDITATION_REQUEST received "
                    "(source=%s reason=%s count=%d)",
                    payload.get("source", "unknown"),
                    payload.get("reason", ""), request_count)
                # SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — Guardian's pre-start
                # dependency activation guarantees memory is RUNNING before
                # this worker enters its main loop. The legacy
                # `_wait_memory_ready` bus-event wait (pre-§11.G.2.5) was
                # redundant + broken under the cross-process Phase C model:
                # memory's MODULE_READY is targeted dst="guardian" (or
                # broadcast dst="all" — Phase 1.1) but boot_buffer only
                # caches targeted frames per SPEC §8.0.bis, and broadcasts
                # before a late-subscriber attach (memory boots ~52s on T3
                # devnet) are lost. Trust Guardian. If memory is genuinely
                # down, the orchestrator's `run_meditation_async` work-RPC
                # times out at RUN_MEDITATION_TIMEOUT_S=300s and the cycle
                # emits MEDITATION_INTERRUPTED naturally — §11.G.2 respawn
                # check + watchdog handle persistent failure.
                if not orch_state.enqueue(payload):
                    logger.info(
                        "[MeditationWorker] orchestrator busy — dropping "
                        "MEDITATION_REQUEST (in_meditation already True)")

            elif msg_type == MEDITATION_FORCE_END:
                force_end_count += 1
                logger.warning(
                    "[MeditationWorker] MEDITATION_FORCE_END received "
                    "(source=%s reason=%s) — resetting in_meditation",
                    payload.get("source", "?"), payload.get("reason", ""))
                if publisher.is_in_meditation():
                    publisher.set_in_meditation(False)
                    publisher.set_phase("idle")
                    publisher.publish()
                    orch_state.mark_idle()
                    _send(send_queue, MEDITATION_INTERRUPTED, name, "all", {
                        "reason": "force_end",
                        "triggered_by": payload.get("source", "dashboard"),
                        "meditation_count": publisher.get_count(),
                        "ts": time.time(),
                    })

            elif msg_type == EXPRESSION_FIRED:
                expression_fired_count += 1
                # kin_sense trigger bridge — translate to MEDITATION_REQUEST self-emit.
                if str(payload.get("composite", "")).upper() == "KIN_SENSE":
                    logger.info(
                        "[MeditationWorker] EXPRESSION_FIRED(KIN_SENSE) "
                        "bridge → MEDITATION_REQUEST self-emit")
                    _send(send_queue, MEDITATION_REQUEST, name, "meditation", {
                        "source": "kin_sense",
                        "reason": "kin_sense_expression",
                        "ts": time.time(),
                    })

            elif msg_type == KERNEL_EPOCH_TICK:
                kernel_tick_count += 1
                current_epoch = int(payload.get("epoch_id", 0) or 0)
                # SHM republish (dual-trigger Q1 greenlight).
                now = time.time()
                if (now - last_shm_republish_ts) >= SHM_REPUBLISH_CADENCE_S:
                    publisher.publish()
                    last_shm_republish_ts = now
                # Emergent driver + watchdog checks.
                _emergent_check(current_epoch)
                _watchdog_check(now)

            elif msg_type == SAVE_NOW:
                logger.info("[MeditationWorker] SAVE_NOW — persisting tracker")
                _persist_tracker(publisher.get_tracker())

            elif msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[MeditationWorker] MODULE_SHUTDOWN received — exiting "
                    "(stats: requests=%d force_ends=%d expression_fired=%d "
                    "kernel_ticks=%d count=%d)",
                    request_count, force_end_count, expression_fired_count,
                    kernel_tick_count, publisher.get_count())
                _persist_tracker(publisher.get_tracker())
                break

    except KeyboardInterrupt:
        logger.info("[MeditationWorker] KeyboardInterrupt — exiting")
    except Exception as e:
        logger.error(
            "[MeditationWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        raise
    finally:
        stop_event.set()
        orch_state.request_event.set()  # unblock orchestrator
        # Chunk 1G — drain off-hot-path I/O before exit so any in-flight
        # tracker / backup_trigger writes finish before the process dies.
        _shutdown_io_executor(wait=True)
        logger.info("[MeditationWorker] shutdown complete")
