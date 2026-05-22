"""
observatory_worker — Python L3 module owning Observatory output *production*
that has no domain owner: the V4 real-time event bridge + the periodic
ObservatoryDB history snapshots.

Phase C `RFP_phase_c_titan_hcl_cleanup.md` Phase A+B (Track 2). Carved out of
two residual in-process loops in `core/plugin.py` (`_v4_event_bridge_loop`
+ `_trinity_snapshot_loop`) so titan_HCL is a lean L2+L3 orchestrator and
each carve earns its own §9.B block (`feedback_phase_c_break_monolith_ethos`).

Producer, NOT writer-owner: ObservatoryDB write *serialization* is owned by
the `observatory_writer` L3 daemon (IMW pattern — `rFP_observatory_writer_service`,
[persistence_observatory] in config.toml). This worker's `record_*` calls
route through that daemon like every other writer's. It owns only the
ownerless production loops.

What this worker owns:
  1. V4 event bridge — subscribes the real-time broadcast event set
     (SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
     DREAM_INBOX_REPLAY, NEUROMOD_UPDATE, HORMONE_FIRED, EXPRESSION_FIRED),
     translates each to the frontend event shape, and publishes
     OBSERVATORY_EVENT (dst="api"). The api_subprocess re-emits to its
     subprocess-local EventBus for WebSocket fan-out (SPEC §A.4 / D-SPEC-82:
     the api subprocess is broadcast-free, so this translation must run
     OUTSIDE it — a worker, not the api process).
  2. Periodic ObservatoryDB history snapshots — trinity + growth + v4 +
     vital, on the [frontend] trinity_snapshot_interval cadence (default 60s).
     ALL reads SHM-direct via ShmReaderBank (G18 / INV-4); no proxy-RPC.

Explicitly NOT owned:
  • DREAM_INBOX_REPLAY → CHAT_REQUEST orchestration — moved to agno_worker
    (the chat-action owner); this worker only mirrors the replay to a WS event.
  • ObservatoryDB write serialization — observatory_writer (exists).
  • Domain record_* writes (meditation/agno) — stay with their owners.

Bus subscriptions (broadcast_topics on ModuleSpec):
  • SPHERE_PULSE, BIG_PULSE, GREAT_PULSE — resonance pulse events
  • DREAM_STATE_CHANGED                  — dream phase transition
  • DREAM_INBOX_REPLAY                    — dream-end inbox replay (WS mirror only)
  • NEUROMOD_UPDATE, HORMONE_FIRED, EXPRESSION_FIRED — neuromod/hormone/expression
  • MODULE_SHUTDOWN                       — clean shutdown

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • OBSERVATORY_EVENT (dst="api") — per translated event
  • MODULE_HEARTBEAT (every 30s)
  • MODULE_READY (on boot, after ShmReaderBank + ObservatoryDB construct)

Persisted state: none in SHM (writes go to ObservatoryDB via observatory_writer).

External I/O: bus client + ObservatoryDB writes (routed via observatory_writer
daemon socket / WAL). No Solana RPC, no HTTP.

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • SOFT: api_subprocess (OBSERVATORY_EVENT consumer — boots independently)
  • SOFT: observatory_writer (write target — obs_db falls back to direct WAL)
  • SOFT: producers of bridged events (cognitive/ns/expression/hormonal workers)

See:
  - RFP_phase_c_titan_hcl_cleanup.md Phase A + Phase B
  - ARCHITECTURE_api_family.md §6 (producer side) + §4 (api_subprocess)
  - SPEC §9.B observatory_worker block
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl.bus import (
    BIG_PULSE,
    DREAM_INBOX_REPLAY,
    DREAM_STATE_CHANGED,
    EXPRESSION_FIRED,
    GREAT_PULSE,
    HORMONE_FIRED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_SHUTDOWN,
    NEUROMOD_UPDATE,
    OBSERVATORY_EVENT,
    SPHERE_PULSE,
    make_msg,
)

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_SNAPSHOT_INTERVAL_S = 60

# The broadcast event types this worker bridges to OBSERVATORY_EVENT. Kept as
# a module constant so the ModuleSpec.broadcast_topics + the CI parity test
# reference one source (mirrors the V4_EVENT_TYPES set the parent v4_bridge
# used pre-extraction, minus DREAM_INBOX_REPLAY's orchestration which is now
# agno_worker's; the replay is still bridged here for the WS mirror event).
V4_EVENT_TYPES = (
    SPHERE_PULSE,
    BIG_PULSE,
    GREAT_PULSE,
    DREAM_STATE_CHANGED,
    DREAM_INBOX_REPLAY,
    NEUROMOD_UPDATE,
    HORMONE_FIRED,
    EXPRESSION_FIRED,
)


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 compliance)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[ObservatoryWorker] _send %s → %s failed: %s", msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str, stop_event: threading.Event) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s."""
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _emit_observatory_event(send_queue, name: str, event_type: str,
                            data: dict) -> None:
    """Publish a translated frontend event as OBSERVATORY_EVENT (dst="api").

    The api_subprocess bus listener translates this to its local
    event_bus.emit(event_type, data) for WebSocket subscribers (api_subprocess.py
    _bus_listener_loop). Under Phase C the api process is always separate
    (D-SPEC-82), so this is the sole path — no in-process event_bus fallback.
    """
    _send(send_queue, OBSERVATORY_EVENT, name, "api",
          {"event_type": event_type, "data": data})


def _translate_and_emit(send_queue, name: str, msg_type: str,
                        payload: dict) -> bool:
    """Translate one V4 broadcast event to its frontend shape + emit.

    Returns True if the type was a recognized V4 event (handled), else False.
    Payload shapes lifted verbatim from the parent _v4_event_bridge_loop so
    the WebSocket contract is byte-identical pre/post extraction.
    """
    if msg_type == SPHERE_PULSE:
        _emit_observatory_event(send_queue, name, "sphere_pulse", {
            "clock": payload.get("clock", ""),
            "pulse_count": payload.get("pulse_count", 0),
            "radius": payload.get("radius"),
            "phase": payload.get("phase"),
        })
    elif msg_type == BIG_PULSE:
        _emit_observatory_event(send_queue, name, "big_pulse", {
            "pair": payload.get("pair", ""),
            "big_pulse_count": payload.get("big_pulse_count", 0),
            "consecutive": payload.get("consecutive", 0),
        })
    elif msg_type == GREAT_PULSE:
        _emit_observatory_event(send_queue, name, "great_pulse", {
            "pair": payload.get("pair", ""),
            "great_pulse_count": payload.get("great_pulse_count", 0),
        })
    elif msg_type == DREAM_STATE_CHANGED:
        _emit_observatory_event(send_queue, name, "dream_state", {
            "is_dreaming": payload.get("is_dreaming", False),
            "state": payload.get("state", "awake"),
            "recovery_pct": payload.get("recovery_pct", 0.0),
            "remaining_epochs": payload.get("remaining_epochs", 0),
            "wake_transition": payload.get("wake_transition", False),
        })
    elif msg_type == DREAM_INBOX_REPLAY:
        # WS mirror ONLY — the CHAT_REQUEST re-emission moved to agno_worker
        # (the chat-action owner). This event keeps frontend parity.
        messages = payload.get("messages", []) or []
        _emit_observatory_event(send_queue, name, "dream_inbox_replay", {
            "replayed": len(messages),
            "total": len(messages),
        })
    elif msg_type == NEUROMOD_UPDATE:
        _emit_observatory_event(send_queue, name, "neuromod_update", payload)
    elif msg_type == HORMONE_FIRED:
        _emit_observatory_event(send_queue, name, "hormone_fired", payload)
    elif msg_type == EXPRESSION_FIRED:
        _emit_observatory_event(send_queue, name, "expression_fired", payload)
    else:
        return False
    return True


def _build_and_record_snapshot(shm_bank, obs_db) -> None:
    """Build the trinity/growth/v4/vital snapshot from SHM-direct reads and
    record it to ObservatoryDB. Lifted from the parent _trinity_snapshot_loop,
    with every proxy-RPC read converted to SHM-direct (G18 / INV-4).

    Raises nothing fatal — best-effort; the caller logs + continues so a
    transient SHM/DB hiccup never kills the snapshot cadence.
    """
    trinity = shm_bank.compose_trinity()

    body_tensor = trinity.get("body_values", [0.5] * 5)
    mind_tensor = trinity.get("mind_values", [0.5] * 5)
    spirit_tensor = trinity.get("spirit_tensor", [0.5] * 5)
    middle_path_loss = trinity.get("middle_path_loss", 0.0)
    body_center_dist = trinity.get("body_center_dist", 0.0)
    mind_center_dist = trinity.get("mind_center_dist", 0.0)

    obs_db.record_trinity_snapshot(
        body_tensor=body_tensor,
        mind_tensor=mind_tensor,
        spirit_tensor=spirit_tensor,
        middle_path_loss=middle_path_loss,
        body_center_dist=body_center_dist,
        mind_center_dist=mind_center_dist,
    )

    # Growth metrics — sourced from the consciousness state_vector (same
    # indices the parent loop used: sv[5]=learning, sv[6]=social, sv[4]=directive).
    consciousness = trinity.get("consciousness", {}) or {}
    sv = consciousness.get("state_vector", []) or []
    learning_velocity = sv[5] if len(sv) > 5 else 0.0
    social_density = sv[6] if len(sv) > 6 else 0.0
    metabolic_health = spirit_tensor[3] if len(spirit_tensor) > 3 else 0.5
    directive_alignment = sv[4] if len(sv) > 4 else 0.0

    obs_db.record_growth_snapshot(
        learning_velocity=learning_velocity,
        social_density=social_density,
        metabolic_health=metabolic_health,
        directive_alignment=directive_alignment,
    )

    # V4 Time-Awareness snapshot (only when sphere/unified data is present).
    if trinity.get("sphere_clock") or trinity.get("unified_spirit"):
        obs_db.record_v4_snapshot(
            sphere_clocks=trinity.get("sphere_clock"),
            resonance=trinity.get("resonance"),
            unified_spirit=trinity.get("unified_spirit"),
            consciousness=trinity.get("consciousness"),
            impulse_engine=trinity.get("impulse_engine"),
            filter_down=trinity.get("filter_down"),
            middle_path_loss=middle_path_loss,
        )

    # Vital snapshot — keeps /status/history populated. SHM-direct mood +
    # sol_balance + persistent_count (was mind_proxy / network / memory RPC).
    mind_pl = shm_bank.read_mind_state() or {}
    mood_label = mind_pl.get("mood_label", "Unknown")
    mood_valence = float(mind_pl.get("mood_valence", 0.5))

    # Neuromod-level dominance override (mirrors the coordinator's
    # current_emotion derivation): the dominant neuromodulator wins when its
    # level exceeds 0.5.
    neuromod_pl = shm_bank.read_neuromod() or {}
    nm_modulators = neuromod_pl.get("modulators") or {}
    if nm_modulators:
        top_name, top_val = max(
            ((n, float(p.get("level", 0.0))) for n, p in nm_modulators.items()),
            key=lambda kv: kv[1],
            default=(mood_label, 0.0),
        )
        if top_val > 0.5:
            mood_label = top_name
            mood_valence = top_val

    chi_pl = shm_bank.read_chi() or {}
    chi_total = float(chi_pl.get("total", 0.5))

    body_pl = shm_bank.read_body_state() or {}
    sol_balance = float(body_pl.get("sol_balance", 0.0))

    mem_pl = shm_bank.read_memory_state() or {}
    persistent_count = int(mem_pl.get("persistent_count", 0))

    life_pl = shm_bank.read_life_force_state() or {}
    energy_state = life_pl.get("state", "HIGH") or "HIGH"

    obs_db.record_vital_snapshot(
        sovereignty_pct=chi_total * 100,
        life_force_pct=metabolic_health * 100,
        sol_balance=sol_balance,
        energy_state=energy_state,
        mood_label=mood_label,
        mood_score=mood_valence,
        persistent_count=persistent_count,
        mempool_size=0,
        epoch_counter=consciousness.get("epoch_id", 0),
    )


def _snapshot_loop(shm_bank, obs_db, interval_s: int,
                   stop_event: threading.Event) -> None:
    """Daemon thread — record an ObservatoryDB history snapshot every
    interval_s seconds. Separate from the event-bridge recv loop so a
    blocking DB write never delays real-time WebSocket event translation."""
    # Initial settle so SHM slots have a chance to be written by producers.
    stop_event.wait(min(interval_s, 30))
    snapshot_count = 0
    error_count = 0
    while not stop_event.is_set():
        try:
            _build_and_record_snapshot(shm_bank, obs_db)
            snapshot_count += 1
            if snapshot_count % 60 == 0:
                logger.info(
                    "[ObservatoryWorker] snapshot heartbeat — recorded=%d "
                    "errors=%d", snapshot_count, error_count)
        except Exception as e:
            error_count += 1
            logger.warning(
                "[ObservatoryWorker] snapshot error (#%d): %s",
                error_count, e, exc_info=(error_count % 50 == 1))
        stop_event.wait(interval_s)


def observatory_worker_main(recv_queue, send_queue, name: str,
                            config: dict) -> None:
    """L3 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id; construct ShmReaderBank (process-agnostic SHM
         reader) + ObservatoryDB (per-process singleton, auto-wires the
         observatory_writer client per [persistence_observatory]).
      2. Start heartbeat thread (30s) + snapshot thread (interval cadence).
      3. Emit MODULE_READY.
      4. Main loop: drain recv_queue, translate V4 events → OBSERVATORY_EVENT,
         handle MODULE_SHUTDOWN.
    """
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id(config.get("titan_id") if config else None)

    interval_s = DEFAULT_SNAPSHOT_INTERVAL_S
    try:
        interval_s = int((config or {}).get("frontend", {}).get(
            "trinity_snapshot_interval", DEFAULT_SNAPSHOT_INTERVAL_S))
    except (TypeError, ValueError):
        interval_s = DEFAULT_SNAPSHOT_INTERVAL_S

    logger.info(
        "[ObservatoryWorker] booting — titan_id=%s name=%s snapshot_interval=%ds "
        "(RFP_phase_c_titan_hcl_cleanup Phase A+B / §9.B)",
        titan_id, name, interval_s)

    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    from titan_hcl.utils.observatory_db import get_observatory_db

    shm_bank = ShmReaderBank(titan_id=titan_id)
    obs_db = get_observatory_db()

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(send_queue, name, stop_event),
        daemon=True, name=f"observatory-hb-{name}")
    hb_thread.start()

    snap_thread = threading.Thread(
        target=_snapshot_loop,
        args=(shm_bank, obs_db, interval_s, stop_event),
        daemon=True, name=f"observatory-snap-{name}")
    snap_thread.start()

    _send(send_queue, MODULE_READY, name, "guardian", {
        "titan_id": titan_id,
        "module": "observatory_worker",
        "version": "1.0.0",
        "schema_version": 1,
        "spec_ref": "RFP_phase_c_titan_hcl_cleanup",
    })
    logger.info(
        "[ObservatoryWorker] MODULE_READY emitted — event bridge + snapshot "
        "loop (interval=%ds) active", interval_s)

    events_bridged = 0
    unknown_count = 0

    try:
        while True:
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                continue

            if msg is None:
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            if msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[ObservatoryWorker] MODULE_SHUTDOWN received — exiting "
                    "(events_bridged=%d)", events_bridged)
                break

            if _translate_and_emit(send_queue, name, msg_type, payload):
                events_bridged += 1
                if events_bridged % 1000 == 0:
                    logger.info(
                        "[ObservatoryWorker] event-bridge heartbeat — "
                        "bridged=%d", events_bridged)
            else:
                unknown_count += 1

    except KeyboardInterrupt:
        logger.info("[ObservatoryWorker] KeyboardInterrupt — exiting")
    except Exception as e:
        logger.error(
            "[ObservatoryWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        raise
    finally:
        stop_event.set()
        logger.info("[ObservatoryWorker] shutdown complete")
