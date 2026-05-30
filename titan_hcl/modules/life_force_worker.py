"""life_force_worker — Python L2 module hosting the LifeForceEngine instance.

Per rFP_titan_hcl_l2_separation_strategy.md §4.G (LOCKED 2026-05-05;
SHIPPED 2026-05-15) + SPEC v1.8.3 §9.B life_force_worker block +
D-SPEC-57.

ACTIVE: always-on autostart. No flag-gate (replaces cognitive_worker
chunk 8M.6 Track 1 drift per §0 vision target).

Owns:
  - LifeForceEngine (titan_hcl/logic/life_force.py — 545 LOC, 13
    methods: Chi (Λ) 3×3 Trinity vitality math, adaptive
    developmental-age weights, bidirectional Mind-bridge flow,
    circulation rate, behavioral-state classifier, contemplation /
    GRAND-CYCLE conviction, metabolic-drain accumulator).
  - In-memory chi_history ring (last 100 evaluate results — exposed
    via /v4/chi chi_trend + LifeForceProxy.get_chi_history).
  - data/life_force_state.json persistence (SAVE_NOW snapshot of
    LifeForceEngine.get_state() — 13 fields per life_force.py:492-509).
  - life_force_state.bin SHM slot (G21 single-writer; 1 Hz; payload
    per SPEC §7.1 v1.8.3).
  - LifeForceProxy dispatch handler (3 work-RPC actions per
    phase_c_rpc_exemptions.yaml::work_rpc_sites — all ≤5s per G19).

Bus subscriptions (SPEC §9.B life_force_worker block):
  REQUIRED — bus.QUERY (dst=life_force) for LifeForceProxy dispatch
             + KERNEL_EPOCH_TICK (drives evaluate)
             + DREAM_STATE_CHANGED (set_dreaming + recovery)
             + MEDITATION_COMPLETE (drain *= 0.85 per Maker Q4)
             + EXPRESSION_FIRED (accumulate_metabolic_pressure)
             + NEUROMOD_STATS_UPDATED (cached for next inputs read)
             + MODULE_SHUTDOWN + SAVE_NOW
  OPTIONAL — HORMONAL_STATE_UPDATED (cached; degraded path)

Bus publications:
  - LIFE_FORCE_UPDATED       (1Hz coalesced; bulk via SHM)
  - CHI_UPDATED              (per evaluate; producer flipped from
                              cognitive_worker per v1.8.3 §4.G —
                              feeds chi.state cache key for /v4/chi)
  - FATIGUE_LEVEL_CRITICAL   (single-shot on drain≥0.7 upward,
                              edge-debounced reset ≤0.6)
  - NEUROMOD_EXTERNAL_NUDGE  (per evaluate, source="life_force_chi_health" —
                              closes §4.Q D-SPEC-54 orphan nudge)
  - MODULE_HEARTBEAT / MODULE_SHUTDOWN (standard per §11; legacy MODULE_READY
                              retired per Phase 11 §11.I.2 — SHM slot
                              state=booted is the contract)

SHM reads:
  - life_force_inputs.bin (cognitive_worker is G21 writer; this worker
    reads the 16-input bridge payload per KERNEL_EPOCH_TICK)

SHM writes:
  - life_force_state.bin (G21 single-writer = this worker)

Migration map per SPEC §9.B v1.8.3:
  cognitive_worker.py:1545-1592 chunk 8M.6 init    → REMOVED (this worker)
  cognitive_worker.py:2364-2497 evaluate orchestration → REMOVED (this worker)
  cognitive_worker.py 5 chi reader sites           → use LifeForceShmReader
  spirit_worker.py 77 dead readers                 → DELETED (Phase A+B
                                                     accepted DISABLED)
  inner_coordinator.py self._life_force.set_dreaming → bus-event-only
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty
from typing import Any, Optional

import msgpack

from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    LIFE_FORCE_FATIGUE_RESET,
    LIFE_FORCE_FATIGUE_THRESHOLD,
    LIFE_FORCE_MEDITATION_RECOVERY_FACTOR,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

MODULE_NAME = "life_force"

# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to per-process SHM slot via ModuleStateWriter. Heartbeat
# publishes to the SHM slot only once True so slot stays at
# "starting"/"booted" during boot window.
_WORKER_READY: bool = False

# Cadence + lifecycle constants.
_HEARTBEAT_INTERVAL_S = 10.0            # SPEC §10.B MODULE_HEARTBEAT_INTERVAL_S
_POLL_INTERVAL_S = 0.2                  # recv loop poll cadence
_SHM_PUBLISH_INTERVAL_S = 1.0           # life_force_state.bin 1 Hz per SPEC §7.1
_STATS_NOTIFY_INTERVAL_S = 1.0          # LIFE_FORCE_UPDATED bus notification cadence

# Persisted state file (SAVE_NOW snapshot).
_PERSIST_PATH = Path("data/life_force_state.json")

# Topics subscribed by life_force_worker (per SPEC §9.B v1.8.3).
_LIFE_FORCE_WORKER_SUBSCRIBE_TOPICS: list[str] = [
    bus.QUERY,                       # LifeForceProxy dispatch (dst=life_force)
    bus.KERNEL_EPOCH_TICK,           # drives evaluate via inputs SHM read
    bus.DREAM_STATE_CHANGED,         # set_dreaming flag + drain recovery
    bus.MEDITATION_COMPLETE,         # drain *= 0.85 one-shot
    bus.EXPRESSION_FIRED,            # accumulate_metabolic_pressure
    bus.NEUROMOD_STATS_UPDATED,      # cached (informational; inputs bridge
                                     # already supplies neuromodulator
                                     # homeostasis via cognitive_worker side)
    bus.MODULE_SHUTDOWN,
    bus.SAVE_NOW,
]
# HORMONAL_STATE_UPDATED is added at runtime IF the constant exists in
# bus.py (per SPEC §9.B OPTIONAL — informational only; the actual
# hormonal_vitality input comes via life_force_inputs.bin bridge).
_OPTIONAL_HORMONAL_TOPIC = "HORMONAL_STATE_UPDATED"


# ── Lifecycle helpers (mirror metabolism_worker template) ─────────────


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict,
              rid=None) -> None:
    """Best-effort enqueue helper — never raises (heartbeat path)."""
    try:
        msg = {"type": msg_type, "src": src, "dst": dst, "payload": payload,
               "ts": time.time()}
        if rid is not None:
            msg["rid"] = rid
        send_queue.put(msg)
    except Exception:
        pass


def _send_heartbeat(send_queue, name: str, extra: Optional[dict] = None,
                    state_writer: Optional[Any] = None) -> None:
    """Emit MODULE_HEARTBEAT to guardian_HCL with current RSS.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided AND `_WORKER_READY`
    is True (the SHM slot stays at state="starting"/"booted" until then).
    """
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss_mb = 0.0
    payload = {"alive": True, "ts": time.time(), "rss_mb": round(rss_mb, 1)}
    if extra:
        payload.update(extra)
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", payload)
    if state_writer is not None and _WORKER_READY:
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash heartbeat
            pass


def _send_response(send_queue, src_name: str, dst: str, payload: dict,
                   rid) -> None:
    """Emit RESPONSE to bus.QUERY caller. rid is required."""
    if rid is None:
        return
    try:
        send_queue.put({
            "type": bus.RESPONSE,
            "src": src_name,
            "dst": dst,
            "rid": rid,
            "payload": payload,
            "ts": time.time(),
        })
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] send_response enqueue failed (rid=%s): %s",
            rid, e)


# ── LifeForceEngine init + persistence ────────────────────────────────


def _init_life_force_engine(config: dict) -> Any:
    """Construct LifeForceEngine + load drain_passive_decay from config.

    Mirrors the cognitive_worker chunk 8M.6 init at lines 1555-1590
    pre-extraction. Cold-boot safe.
    """
    try:
        from titan_hcl.logic.life_force import LifeForceEngine
    except Exception as e:
        logger.error(
            "[LifeForceWorker] LifeForceEngine import failed: %s", e,
            exc_info=True)
        return None

    engine = LifeForceEngine()

    # Load drain_passive_decay override from titan_params.toml [life_force].
    lf_cfg = (config.get("life_force", {}) or {})
    try:
        if "drain_passive_decay" in lf_cfg:
            engine._drain_passive_decay = float(lf_cfg["drain_passive_decay"])
            logger.info(
                "[LifeForceWorker] drain_passive_decay loaded from config: %.4f",
                engine._drain_passive_decay)
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] drain_passive_decay config load failed: %s", e)

    logger.info(
        "[LifeForceWorker] LifeForceEngine (Chi) booted — drain_passive_decay=%.4f "
        "metabolic_drain=%.4f", engine._drain_passive_decay,
        engine._metabolic_drain)
    return engine


def _try_load_persisted(engine) -> None:
    """Load data/life_force_state.json on boot (if present + decodable).

    Restores 13 fields per LifeForceEngine.restore_state. Cold-boot OK
    (no file = stays at __init__ defaults).
    """
    if engine is None:
        return
    try:
        if not _PERSIST_PATH.exists():
            logger.info(
                "[LifeForceWorker] persist file not found at %s — cold boot",
                _PERSIST_PATH)
            return
        with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict):
            engine.restore_state(state)
            logger.info(
                "[LifeForceWorker] persisted state restored from %s",
                _PERSIST_PATH)
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] persist load failed (%s): %s — proceeding "
            "with __init__ defaults",
            _PERSIST_PATH, e, exc_info=True)


def _save_persisted(engine) -> None:
    """Snapshot LifeForceEngine.get_state() to data/life_force_state.json.

    Atomic write via .tmp + rename. Called from SAVE_NOW handler + clean
    shutdown.
    """
    if engine is None:
        return
    try:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = engine.get_state()
        tmp_path = _PERSIST_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_path, _PERSIST_PATH)
        logger.info(
            "[LifeForceWorker] persisted state saved to %s (drain=%.3f "
            "conviction=%d evaluations=%d)",
            _PERSIST_PATH, state.get("_metabolic_drain", 0.0),
            state.get("_conviction_counter", 0),
            state.get("_total_evaluations", 0))
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] persist save failed: %s", e, exc_info=True)


# ── Main entry ────────────────────────────────────────────────────────


@with_error_envelope(module_name="life_force", subsystem="entry", severity=_phase11_sev.FATAL)
def life_force_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """Main loop for the life_force_worker subprocess.

    Hosts LifeForceEngine + serves bus.QUERY work-RPC dispatch +
    drives evaluate per KERNEL_EPOCH_TICK + publishes
    life_force_state.bin SHM slot at 1 Hz via dedicated thread.
    """
    # === BOILERPLATE: spawn-mode sys.path bootstrap ===
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # === BOILERPLATE: Phase B.2 §C7 socket-mode bus client setup ===
    # Build topics with optional HORMONAL_STATE_UPDATED (soft-dep).
    topics = list(_LIFE_FORCE_WORKER_SUBSCRIBE_TOPICS)
    if hasattr(bus, _OPTIONAL_HORMONAL_TOPIC):
        topics.append(getattr(bus, _OPTIONAL_HORMONAL_TOPIC))

    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    try:
        recv_queue, send_queue, _bus_client = setup_worker_bus(
            name, recv_queue, send_queue, topics=topics,
        )
    except Exception as _err:
        logger.error(
            "[LifeForceWorker] setup_worker_bus failed: %s — exiting",
            _err, exc_info=True)
        return

    # === BOILERPLATE: pdeathsig installation ===
    try:
        from titan_hcl.core.worker_lifecycle import install_parent_death_signal
        install_parent_death_signal()
    except Exception as _err:
        logger.debug(
            "[LifeForceWorker] pdeathsig install skipped: %s", _err)

    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel
    # on every entry.
    global _WORKER_READY
    _WORKER_READY = False

    from titan_hcl.core.state_registry import (
        StateRegistryReader, ensure_shm_root, resolve_titan_id,
    )
    titan_id = (
        (config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    boot_ts = time.time()

    logger.info(
        "[LifeForceWorker] Booting (titan_id=%s) — rFP §4.G + D-SPEC-57",
        titan_id)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[LifeForceWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    # === MODULE-SPECIFIC: LifeForceEngine init + persistence restore ===
    engine = _init_life_force_engine(config)
    if engine is None:
        logger.error(
            "[LifeForceWorker] engine init failed — exiting non-zero "
            "so guardian respawns")
        sys.exit(1)
    _try_load_persisted(engine)

    # === MODULE-SPECIFIC: SHM publisher init ===
    state_publisher = None
    try:
        from titan_hcl.logic.life_force_state_publisher import (
            LifeForceStatePublisher,
        )
        state_publisher = LifeForceStatePublisher(titan_id=titan_id)
    except Exception as _shm_err:
        logger.error(
            "[LifeForceWorker] LifeForceStatePublisher BOOT FAILED — "
            "worker continues without SHM visibility: %s",
            _shm_err, exc_info=True)

    # === MODULE-SPECIFIC: life_force_inputs.bin SHM reader ===
    inputs_reader = None
    try:
        from titan_hcl.logic.life_force_inputs_specs import (
            LIFE_FORCE_INPUTS_SPEC,
        )
        inputs_reader = StateRegistryReader(
            LIFE_FORCE_INPUTS_SPEC, ensure_shm_root(titan_id))
        logger.info(
            "[LifeForceWorker] life_force_inputs.bin reader attached "
            "(G18 SHM read; producer = cognitive_worker)")
    except Exception as _r_err:
        logger.error(
            "[LifeForceWorker] inputs reader init failed: %s — engine "
            "will fall back to evaluate() defaults until cognitive_worker "
            "publishes",
            _r_err, exc_info=True)

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # Legacy MODULE_READY bus emit retired per locked D2.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[LifeForceWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[LifeForceWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # Mutable closure cells (worker state shared across recv loop + thread).
    latest_chi: dict = {"result": None}    # most recent evaluate() output
    is_dreaming_box = {"value": False}     # cached from DREAM_STATE_CHANGED
    fatigue_armed = {"value": True}        # edge-debounce arm for FATIGUE_LEVEL_CRITICAL
    fatigue_emit_count = {"value": 0}      # heartbeat metric

    # === MODULE-SPECIFIC: 1 Hz SHM publisher thread ===
    _periodic_stop = threading.Event()

    def _periodic_publish_loop():
        last_shm = 0.0
        last_stats_notify = 0.0
        while not _periodic_stop.is_set():
            try:
                now = time.time()

                # SHM publish (1 Hz — even if no evaluate has fired, cold-
                # boot stub keeps consumers' chi.state cache populated).
                if state_publisher is not None and \
                        now - last_shm > _SHM_PUBLISH_INTERVAL_S:
                    try:
                        state_publisher.publish(
                            life_force_engine=engine,
                            chi_result=latest_chi["result"],
                            is_dreaming=is_dreaming_box["value"],
                        )
                    except Exception as _shm_err:
                        logger.warning(
                            "[LifeForceWorker] state publish raised: %s",
                            _shm_err, exc_info=True)
                    last_shm = now

                # 1Hz LIFE_FORCE_UPDATED bus notification (bulk via SHM).
                if now - last_stats_notify > _STATS_NOTIFY_INTERVAL_S:
                    _send_msg(
                        send_queue, bus.LIFE_FORCE_UPDATED, name, "all",
                        {"ts": now},
                    )
                    last_stats_notify = now
            except Exception as _per_err:
                logger.warning(
                    "[LifeForceWorker] periodic publish thread error: %s",
                    _per_err)
            _periodic_stop.wait(0.5)

    _periodic_thread = threading.Thread(
        target=_periodic_publish_loop,
        daemon=True,
        name="life_force-periodic-publish",
    )
    _periodic_thread.start()

    # === Main recv loop ===
    last_heartbeat = time.time()
    while True:
        now = time.time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL_S:
            stats_extra = {
                "state": getattr(engine, "_state", "UNKNOWN"),
                "drain": round(getattr(engine, "_metabolic_drain", 0.0), 4),
                "evaluations": getattr(engine, "_total_evaluations", 0),
                "fatigue_emits": fatigue_emit_count["value"],
                "is_dreaming": is_dreaming_box["value"],
            }
            _send_heartbeat(send_queue, name, extra=stats_extra,
                            state_writer=_state_writer)
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        # B.2.1 supervision-transfer dispatch
        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:
            pass

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
        if msg_type == bus.MODULE_PROBE_REQUEST:
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
                    "[LifeForceWorker] MODULE_PROBE_REQUEST handler "
                    "failed: %s", _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info(
                "[LifeForceWorker] Shutdown: %s",
                msg.get("payload", {}).get("reason"))
            break

        if msg_type == bus.SAVE_NOW:
            _save_persisted(engine)
            continue

        if msg_type == bus.KERNEL_EPOCH_TICK:
            _handle_epoch_tick(
                engine, inputs_reader, latest_chi, is_dreaming_box,
                fatigue_armed, fatigue_emit_count, send_queue, name,
            )
            continue

        if msg_type == bus.DREAM_STATE_CHANGED:
            payload = msg.get("payload", {}) or {}
            new_dreaming = bool(payload.get("is_dreaming", False))
            is_dreaming_box["value"] = new_dreaming
            try:
                engine.set_dreaming(new_dreaming)
            except Exception as e:
                logger.warning(
                    "[LifeForceWorker] set_dreaming failed: %s", e)
            continue

        if msg_type == bus.MEDITATION_COMPLETE:
            # Maker Q4 lock 2026-05-15 — proportional drain recovery
            # (matches dreaming *= 0.93 precedent at life_force.py:303).
            try:
                prev = float(getattr(engine, "_metabolic_drain", 0.0))
                new = prev * LIFE_FORCE_MEDITATION_RECOVERY_FACTOR
                engine._metabolic_drain = new
                logger.info(
                    "[LifeForceWorker] MEDITATION_COMPLETE → drain %.4f → %.4f "
                    "(factor=%.2f)",
                    prev, new, LIFE_FORCE_MEDITATION_RECOVERY_FACTOR)
            except Exception as e:
                logger.warning(
                    "[LifeForceWorker] MEDITATION_COMPLETE handling failed: %s",
                    e)
            continue

        if msg_type == bus.EXPRESSION_FIRED:
            # Accumulate metabolic_drain from expression activity.
            payload = msg.get("payload", {}) or {}
            try:
                # Tier-2 fires carry composite + neuromod_cost + somatic_cost
                # estimates. Fall back to small constant if absent (matches
                # life_force.py:accumulate_metabolic_pressure docstring's
                # 0-0.01 typical band).
                neuromod_pressure = float(
                    payload.get("neuromod_cost", 0.002) or 0.002
                )
                somatic_pressure = float(
                    payload.get("somatic_cost", 0.002) or 0.002
                )
                engine.accumulate_metabolic_pressure(
                    neuromod_pressure, somatic_pressure)
            except Exception as e:
                logger.warning(
                    "[LifeForceWorker] EXPRESSION_FIRED handling failed: %s",
                    e)
            continue

        # NEUROMOD_STATS_UPDATED + optional HORMONAL_STATE_UPDATED are
        # informational only — the actual neuromodulator_homeostasis +
        # hormonal_vitality inputs come via life_force_inputs.bin bridge
        # from cognitive_worker (cognitive_worker reads these bus events
        # too and packs the derived values into the inputs payload).
        # We accept them here for future use but do NOT recompute directly.
        if msg_type == bus.NEUROMOD_STATS_UPDATED:
            continue
        if hasattr(bus, _OPTIONAL_HORMONAL_TOPIC) and \
                msg_type == getattr(bus, _OPTIONAL_HORMONAL_TOPIC):
            continue

        if msg_type == bus.QUERY:
            _handle_query(
                msg, engine, latest_chi, send_queue, name,
            )
            continue

        logger.debug(
            "[LifeForceWorker] Unhandled msg_type=%s — ignoring", msg_type)

    # === Clean shutdown ===
    logger.info(
        "[LifeForceWorker] Exiting — stopping publisher thread "
        "(state=%s drain=%.3f evaluations=%d)",
        getattr(engine, "_state", "UNKNOWN"),
        getattr(engine, "_metabolic_drain", 0.0),
        getattr(engine, "_total_evaluations", 0))
    _periodic_stop.set()
    _save_persisted(engine)  # final SAVE_NOW on graceful exit
    logger.info("[LifeForceWorker] Exit complete")


# ── KERNEL_EPOCH_TICK handler — drive evaluate ────────────────────────


def _read_inputs_from_shm(inputs_reader) -> Optional[dict]:
    """Read life_force_inputs.bin and msgpack-decode. None on cold-boot."""
    if inputs_reader is None:
        return None
    try:
        raw = inputs_reader.read_variable()
    except Exception as e:
        logger.debug(
            "[LifeForceWorker] inputs SHM read raised: %s", e)
        return None
    if raw is None:
        return None
    try:
        decoded = msgpack.unpackb(raw, raw=False)
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] inputs msgpack decode raised: %s", e)
        return None
    return decoded if isinstance(decoded, dict) else None


def _handle_epoch_tick(
    engine,
    inputs_reader,
    latest_chi: dict,
    is_dreaming_box: dict,
    fatigue_armed: dict,
    fatigue_emit_count: dict,
    send_queue,
    name: str,
) -> None:
    """KERNEL_EPOCH_TICK driver — read inputs SHM, evaluate, publish events."""
    if engine is None:
        return

    inputs = _read_inputs_from_shm(inputs_reader)
    if inputs is None:
        # Cold boot — cognitive_worker hasn't published yet. Engine uses
        # evaluate() defaults; consumers see BOOTSTRAP state via the SHM
        # stub. Don't emit CHI_UPDATED until first real evaluate (consumers
        # have no signal to attach to anyway).
        return

    # Filter out meta-fields (schema_version, ts) — pass only the 16
    # named inputs to LifeForceEngine.evaluate.
    _evaluate_kwargs = {
        k: v for k, v in inputs.items()
        if k not in ("schema_version", "ts")
    }

    try:
        chi = engine.evaluate(**_evaluate_kwargs)
    except TypeError as e:
        # Unexpected kwarg from schema drift — log + drop.
        logger.error(
            "[LifeForceWorker] evaluate kwargs mismatch: %s — "
            "input keys=%s", e, sorted(_evaluate_kwargs.keys()))
        return
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] evaluate raised: %s", e, exc_info=True)
        return

    # Cache for SHM publisher + diagnostic surface.
    engine._latest_chi = chi
    latest_chi["result"] = chi

    # CHI_UPDATED — feeds chi.state cache key (api_subprocess
    # BusSubscriber maps payload → /v4/chi). Producer flipped from
    # cognitive_worker per v1.8.3 §4.G D-SPEC-57.
    _send_msg(send_queue, bus.CHI_UPDATED, name, "all", chi)

    # NEUROMOD_EXTERNAL_NUDGE — chi_health bridge (closes §4.Q D-SPEC-54
    # orphan from the now-deleted spirit_worker.py:3770 set_chi_health).
    try:
        drain = float(getattr(engine, "_metabolic_drain", 0.0) or 0.0)
        chi_health = max(0.1, 1.0 - drain * 0.6)
        _send_msg(
            send_queue, bus.NEUROMOD_EXTERNAL_NUDGE, name, "all",
            {
                "chi_health": chi_health,
                "source": "life_force_chi_health",
                "ts": time.time(),
            },
        )
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] chi_health NUDGE emit failed: %s", e)

    # FATIGUE_LEVEL_CRITICAL — single-shot on drain≥0.7 upward crossing,
    # edge-debounced reset ≤0.6 per Maker Q6 + SPEC v1.8.3 §8.7.
    try:
        drain = float(getattr(engine, "_metabolic_drain", 0.0) or 0.0)
        if fatigue_armed["value"] and drain >= LIFE_FORCE_FATIGUE_THRESHOLD:
            _send_msg(
                send_queue, bus.FATIGUE_LEVEL_CRITICAL, name, "all",
                {
                    "drain": drain,
                    "threshold": LIFE_FORCE_FATIGUE_THRESHOLD,
                    "state": str(getattr(engine, "_state", "UNKNOWN")),
                    "ts": time.time(),
                },
            )
            fatigue_armed["value"] = False
            fatigue_emit_count["value"] += 1
            logger.warning(
                "[LifeForceWorker] FATIGUE_LEVEL_CRITICAL emitted "
                "(drain=%.4f ≥ %.2f, state=%s)",
                drain, LIFE_FORCE_FATIGUE_THRESHOLD,
                getattr(engine, "_state", "UNKNOWN"))
        elif (not fatigue_armed["value"]) and drain <= LIFE_FORCE_FATIGUE_RESET:
            fatigue_armed["value"] = True
            logger.info(
                "[LifeForceWorker] FATIGUE re-armed (drain=%.4f ≤ %.2f)",
                drain, LIFE_FORCE_FATIGUE_RESET)
    except Exception as e:
        logger.warning(
            "[LifeForceWorker] FATIGUE edge-debounce raised: %s", e)


# ── Action dispatch ───────────────────────────────────────────────────


def _handle_query(
    msg: dict,
    engine,
    latest_chi: dict,
    send_queue,
    name: str,
) -> None:
    """Dispatch QUERY action to LifeForceEngine.

    Every action listed here corresponds to a row in
    phase_c_rpc_exemptions.yaml::work_rpc_sites under life_force_proxy:.
    Per G19: each call is bounded by caller timeout (≤5s on proxy side).
    """
    payload = msg.get("payload", {}) or {}
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "get_stats":
            stats = engine.get_stats() if engine is not None else {}
            _send_response(send_queue, name, src, {"result": stats}, rid)
            return

        if action == "get_chi_history":
            limit = int(payload.get("limit", 100))
            history = list(getattr(engine, "_chi_history", []) or [])
            data = history[-limit:] if limit > 0 else history
            _send_response(send_queue, name, src, {"result": data}, rid)
            return

        if action == "get_contemplation_status":
            # Pull the contemplation block from the most recent evaluate
            # if available; fall back to engine state otherwise.
            latest = latest_chi.get("result")
            if isinstance(latest, dict) and "contemplation" in latest:
                status = latest["contemplation"]
            else:
                status = {
                    "active": False,
                    "phase": int(getattr(engine, "_contemplation_phase", 0)),
                    "conviction": int(getattr(engine, "_conviction_counter", 0)),
                    "conviction_threshold": 300,
                    "mature_enough": False,
                }
            _send_response(send_queue, name, src, {"result": status}, rid)
            return

        logger.warning(
            "[LifeForceWorker] Unknown action: %s (payload=%s)",
            action, payload)
        _send_response(
            send_queue, name, src,
            {"error": f"unknown action: {action}"},
            rid,
        )

    except Exception as e:
        logger.error(
            "[LifeForceWorker] action=%s raised: %s",
            action, e, exc_info=True)
        _send_response(
            send_queue, name, src, {"error": str(e)}, rid,
        )
