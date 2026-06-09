"""
dream_state_worker — Python L2 module owning the canonical dream-state
publication + chat-during-dream inbox + DREAM_WAKE_REQUEST routing hub.

Phase C v1.8.2 (D-SPEC-56) per `rFP_titan_hcl_l2_separation_strategy.md §4.I`.
Maker greenlit Q1-Q6 inline 2026-05-15.

What this worker owns:
  1. Dream-state transition detection — closure-cell `_was_dreaming` pattern
     observing `DREAMING_STATE_UPDATED.payload.is_dreaming` for
     {dream_start, dream_end, dreaming, awake} edges (mirrors the proven
     pattern at `spirit_loop._publish_coord_subdomains:2131-2144`).
  2. `DREAM_STATE_CHANGED` canonical publisher — closes the latent
     Phase C silent-emit fleet-wide bug (sole emitter was dead
     `spirit_worker.py:3006/3007/3143/3144` under `l0_rust_enabled=true`
     since cognitive_worker drives the actual dream lifecycle via
     DreamingEngine but never emitted DREAM_STATE_CHANGED).
  3. `dream_state.bin` SHM slot writer (G21 single writer; sole writer
     under Phase C) — dual-trigger republish on every KERNEL_EPOCH_TICK
     (1.0 Hz adaptive) + on every DREAMING_STATE_UPDATED arrival per
     Maker Q6 greenlight.
  4. `_dream_inbox` chat-during-dream message queue — receives
     `DREAM_INBOX_ENQUEUE` events from chat handlers, drains on dream_end
     via `DREAM_INBOX_REPLAY` publication (sorted maker-first then FIFO).
     Volatile by design — worker crash forfeits queued messages, chat
     user sees standard 429 fallback (matches today's plugin._dream_inbox
     in-memory behavior).
  5. `DREAM_WAKE_REQUEST` routing hub — chat-API maker-fast-wake +
     spirit_worker world-observer-interrupt → worker forwards via
     `DREAM_WAKE_FORWARD` (dst="cognitive_worker" — broker uses registered
     ModuleSpec name) so cognitive_worker can call
     `coordinator.dreaming.request_wake(reason)` in-process (the
     DreamingEngine stays in cognitive_worker per §9.B canonical
     ownership).

Bus subscriptions (REQUIRED):
  • DREAMING_STATE_UPDATED  — cognitive_worker producer (2.5s coalesced
    via spirit_loop._publish_coord_subdomains)
  • DREAM_WAKE_REQUEST      — chat-API + world-observer; forwarded to cognitive
  • DREAM_INBOX_ENQUEUE     — chat handlers buffer messages during dream
  • KERNEL_EPOCH_TICK       — 1.0 Hz dual-trigger SHM republish cadence
  • MODULE_SHUTDOWN         — clean shutdown signal

Bus publications (non-blocking per §8.0.ter D-SPEC-48):
  • DREAM_STATE_CHANGED   — on transition only, dual-target dst="all" + dst="timechain"
  • DREAM_WAKE_FORWARD    — on DREAM_WAKE_REQUEST receipt, dst="cognitive_worker"
  • DREAM_INBOX_REPLAY    — on dream_end with non-empty queue, dst="all" (broadcast;
                              plugin's v4_bridge_loop subscribes via V4_EVENT_TYPES
                              filter + re-emits each message as a chat_handler QUERY)
  • MODULE_HEARTBEAT      — every 30s
  • (MODULE_READY retired — Phase 11 §11.I.2 SHM slot state=booted is the contract)

Persisted state: none — queue is volatile by design; dream_state.bin is
in /dev/shm.

External I/O: bus client only (no DB writes, no Solana RPC, no HTTP).

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • REQUIRED: cognitive_worker (DREAMING_STATE_UPDATED producer; transition
              detection source)
  • SOFT:     api_subprocess  (DREAM_INBOX_ENQUEUE producer +
              DREAM_INBOX_REPLAY consumer — boots independently; queue
              accumulates until api comes up)

See:
  - SPEC v1.8.2 §9.B `dream_state_worker` block
  - SPEC v1.8.2 §7.1 `dream_state.bin` SHM slot row
  - SPEC v1.8.2 §8.7 bus event rows (DREAM_STATE_CHANGED producer change +
    DREAM_WAKE_REQUEST dst change + 3 new events: DREAM_WAKE_FORWARD,
    DREAM_INBOX_ENQUEUE, DREAM_INBOX_REPLAY)
  - SPEC v1.8.2 §21 D-SPEC-56
  - PLAN_microkernel_phase_c_dream_state_worker_extraction.md (committed d4b6b37e)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Empty
from typing import Any, Optional

from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    DREAM_INBOX_MAX_ENTRIES,
    DREAM_INBOX_MAX_MESSAGE_CHARS,
    DREAM_STATE_REPUBLISH_CADENCE_S,
)
from titan_hcl.bus import (
    DREAM_INBOX_ENQUEUE,
    DREAM_INBOX_REPLAY,
    DREAM_STATE_CHANGED,
    DREAM_WAKE_FORWARD,
    DREAM_WAKE_REQUEST,
    DREAMING_STATE_UPDATED,
    KERNEL_EPOCH_TICK,
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    make_msg,
)
from titan_hcl.logic.dream_state_publisher import DreamStatePublisher
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0


# Phase 11 §11.I.3 / §11.I.5 (Chunk 11N) — module-level readiness sentinel
# mirrored to the per-process SHM slot (`module_dream_state_state.bin`) via
# ModuleStateWriter. Set False at import; flipped True after first SHM
# publish completes. Until True, the heartbeat thread skips
# `state_writer.heartbeat()` so the slot stays at state="starting"/"booted"
# rather than prematurely asserting "running".
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (mirrors api_subprocess._send_msg pattern
    for §8.0.ter D-SPEC-48 publish-non-blocking compliance)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[DreamStateWorker] _send %s → %s failed: %s",
            msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str, stop_event: threading.Event,
                    state_writer: Optional[Any] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s.

    Phase 11 §11.I.5 (Chunk 11N): also publishes ModuleStateWriter.heartbeat()
    on the SHM slot when `state_writer` is provided AND `_WORKER_READY` is
    True so guardian_HCL's SHM-staleness detector + observatory
    /v6/readiness see fresh data on the same cadence as the legacy bus path.
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
            try:
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001 — never crash heartbeat
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _validate_enqueue_payload(payload: Any) -> Optional[dict[str, Any]]:
    """Defensive validation + truncation for DREAM_INBOX_ENQUEUE payloads.

    Schema per SPEC v1.8.2 §8.7:
      {message: str (≤DREAM_INBOX_MAX_MESSAGE_CHARS), user_id: str,
       session_id: str, channel: str, priority: int (0=maker, 1=other),
       client_ts: float}
    """
    if not isinstance(payload, dict):
        return None
    try:
        message = str(payload.get("message", ""))[:DREAM_INBOX_MAX_MESSAGE_CHARS]
        return {
            "message": message,
            "user_id": str(payload.get("user_id", "anonymous")),
            "session_id": str(payload.get("session_id", "default")),
            "channel": str(payload.get("channel", "web")),
            "priority": int(payload.get("priority", 1)),
            "client_ts": float(payload.get("client_ts", time.time())),
        }
    except (TypeError, ValueError) as e:
        logger.warning(
            "[DreamStateWorker] malformed DREAM_INBOX_ENQUEUE payload: %s",
            e)
        return None


@with_error_envelope(module_name="dream_state", subsystem="entry", severity=_phase11_sev.FATAL)
def dream_state_worker_main(recv_queue, send_queue, name: str,
                            config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id, build DreamStatePublisher (lazy attach to SHM slot)
      2. Start heartbeat thread (daemon, 30s cadence)
      3. First SHM publish (cold defaults — is_dreaming=False, state="awake")
      4. Transition SHM slot starting → booted (Phase 11 §11.I.2 contract;
         readers can now trust the slot is initialized)
      5. Main loop: drain recv_queue, dispatch by msg_type:
           - DREAMING_STATE_UPDATED → update publisher, detect transition,
             on transition emit DREAM_STATE_CHANGED (dual-target all + timechain)
             + on dream_end flush inbox via DREAM_INBOX_REPLAY
           - DREAM_WAKE_REQUEST → forward via DREAM_WAKE_FORWARD (dst=cognitive)
           - DREAM_INBOX_ENQUEUE → append to inbox (cap at 50, queue-full silently
             drops since the chat handler already returned 429 to client)
           - KERNEL_EPOCH_TICK → republish SHM (1.0 Hz dual-trigger)
           - MODULE_SHUTDOWN → graceful exit
    """
    # Phase 11 §11.I.5 (Chunk 11N) — reset module-level readiness sentinel
    # on every entry (fork-mode re-entries inherit parent's True;
    # spawn-mode re-spawns get fresh False; explicit reset covers both).
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    # Resolve titan_id (per feedback_titan_id_canonical_resolve.md — SPEC §23.17 R-PORT-1).
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id(config.get("titan_id") if config else None)

    logger.info(
        "[DreamStateWorker] booting — titan_id=%s name=%s "
        "(SPEC v1.8.2 §9.B / D-SPEC-56 / rFP §4.I)",
        titan_id, name)

    # ── Phase 11 §11.I.5 / Chunk 11N — SHM state-slot writer (G21) ──
    # Built BEFORE the slow init so titan_hcl's 1Hz SHM poll sees the
    # worker is alive while it warms. Heartbeat thread (started below)
    # keeps `last_heartbeat` fresh during boot.
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
            "[DreamStateWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot disabled): %s", _sw_err)

    publisher = DreamStatePublisher(titan_id)

    # Inbox: thread-safe deque (only the main loop appends/drains, but the
    # heartbeat thread reads len() for observability — keep the lock).
    inbox: deque[dict[str, Any]] = deque(maxlen=DREAM_INBOX_MAX_ENTRIES)
    inbox_lock = threading.Lock()

    # ── Phase 11 §11.I.5 — heartbeat thread BEFORE slow init ──
    # The thread sends MODULE_HEARTBEAT on the bus (legacy Guardian path) AND
    # publishes _state_writer.heartbeat() on the SHM slot (Phase 11 contract)
    # once `_WORKER_READY` flips True.
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer),
        daemon=True, name=f"dream-state-hb-{name}")
    hb_thread.start()

    # Cold-boot first publish — readers can now mmap.
    publisher.publish()

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ─────────
    # SHM slot is initialized + first publish completed. The legacy
    # MODULE_READY bus emit is retired per locked D2 — the SHM slot
    # state=booted is the contract.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[DreamStateWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[DreamStateWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info(
        "[DreamStateWorker] boot complete — dream_state.bin SHM "
        "initialized with cold defaults (is_dreaming=False, state=awake)")

    # Counters for heartbeat observability.
    dreaming_state_updated_count = 0
    wake_request_count = 0
    inbox_enqueue_count = 0
    inbox_replay_count = 0
    transitions_count = 0
    kernel_tick_count = 0
    last_kernel_tick_republish_ts = time.time()

    try:
        while True:
            # Periodic SHM republish — MUST run every loop iteration, NOT only
            # inside `except Empty`. Under Phase C bus load recv_queue.get()
            # almost always returns a message (KERNEL_EPOCH_TICK + others), so
            # the `Empty` branch rarely fires; putting the republish there left
            # the slot uncreated when the boot publish (line ~200) lost the
            # race with SHM-dir creation and no KERNEL_EPOCH_TICK retry landed
            # (observed: dream_state.bin absent on T1+T2 post-restart while T3
            # won the boot race). This is the recv_queue-except-Empty periodic
            # trap (memory: feedback_recv_queue_except_empty_periodic_trap) —
            # gate on a 1Hz time-check OUTSIDE try/except so it always retries.
            now = time.time()
            if (now - last_kernel_tick_republish_ts) >= DREAM_STATE_REPUBLISH_CADENCE_S:
                publisher.publish()
                last_kernel_tick_republish_ts = now

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

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ────────
            if msg_type == MODULE_PROBE_REQUEST:
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
                        "[DreamStateWorker] MODULE_PROBE_REQUEST handler "
                        "failed: %s", _probe_err)
                continue

            if msg_type == DREAMING_STATE_UPDATED:
                dreaming_state_updated_count += 1
                transitioned = publisher.update_from_dreaming_state(payload)
                # Always republish SHM on DREAMING_STATE_UPDATED — dual-trigger
                # Q6 greenlight.
                publisher.publish()
                last_kernel_tick_republish_ts = time.time()

                if transitioned:
                    transitions_count += 1
                    snap = publisher.snapshot_for_emit()
                    # Canonical DREAM_STATE_CHANGED dual-target emit
                    # (preserves spirit_worker.py:3006/3007 + 3143/3144 dual-emit
                    # pattern for TimeChain dream-event commits).
                    _send(send_queue, DREAM_STATE_CHANGED, name, "all", snap)
                    _send(send_queue, DREAM_STATE_CHANGED, name, "timechain", snap)

                    logger.info(
                        "[DreamStateWorker] DREAM_STATE_CHANGED emitted — "
                        "is_dreaming=%s state=%s recovery_pct=%.1f "
                        "(transitions=%d)",
                        snap.get("is_dreaming"), snap.get("state"),
                        snap.get("recovery_pct", 0.0), transitions_count)

                    # On dream_end: flush inbox via DREAM_INBOX_REPLAY.
                    if not snap.get("is_dreaming"):
                        with inbox_lock:
                            if len(inbox) > 0:
                                # Sort maker-first (priority=0) then FIFO
                                # (stable sort preserves enqueue order within
                                # each priority bucket).
                                messages = sorted(
                                    list(inbox),
                                    key=lambda m: (m.get("priority", 1),
                                                   m.get("client_ts", 0.0)),
                                )
                                inbox.clear()
                                dream_duration = (
                                    snap.get("wake_ts", time.time())
                                    - snap.get("dream_started_ts", 0.0)
                                ) if snap.get("dream_started_ts") else 0.0
                                replay_payload = {
                                    "messages": messages,
                                    "dream_duration_s": float(max(0.0, dream_duration)),
                                    "replay_ts": time.time(),
                                }
                                # dst="all" broadcast — plugin's v4_bridge_loop
                                # receives it (in V4_EVENT_TYPES filter) +
                                # re-emits each message as a chat_handler QUERY.
                                # The "chat_api" name was aspirational in the
                                # SPEC; no subscriber registers that name today.
                                _send(send_queue, DREAM_INBOX_REPLAY,
                                      name, "all", replay_payload)
                                inbox_replay_count += 1
                                logger.info(
                                    "[DreamStateWorker] DREAM_INBOX_REPLAY emitted "
                                    "— %d messages, dream_duration=%.1fs "
                                    "(replays=%d)",
                                    len(messages), dream_duration,
                                    inbox_replay_count)

            elif msg_type == DREAM_WAKE_REQUEST:
                wake_request_count += 1
                forward_payload = {
                    "reason": str(payload.get("reason", "unspecified")),
                    "source": str(payload.get("source", "unknown")),
                    "original_client_ts": float(payload.get("client_ts",
                                                            time.time())),
                    "forwarded_ts": time.time(),
                }
                _send(send_queue, DREAM_WAKE_FORWARD, name, "cognitive_worker",
                      forward_payload)
                logger.info(
                    "[DreamStateWorker] DREAM_WAKE_REQUEST received from %s "
                    "(reason=%s) — forwarded to cognitive_worker (count=%d)",
                    forward_payload["source"], forward_payload["reason"],
                    wake_request_count)

            elif msg_type == DREAM_INBOX_ENQUEUE:
                validated = _validate_enqueue_payload(payload)
                if validated is None:
                    continue
                with inbox_lock:
                    inbox_enqueue_count += 1
                    if len(inbox) >= DREAM_INBOX_MAX_ENTRIES:
                        # Cap-full: chat handler should have returned 429
                        # already (they read dream_state.bin SHM + check
                        # queue depth via their own counter). Defensive
                        # drop here — log throttled.
                        if inbox_enqueue_count % 50 == 0:
                            logger.warning(
                                "[DreamStateWorker] inbox cap reached "
                                "(%d) — dropping enqueue from %s "
                                "(should not happen; chat handler check failed?)",
                                DREAM_INBOX_MAX_ENTRIES,
                                validated.get("user_id"))
                        continue
                    inbox.append(validated)
                logger.debug(
                    "[DreamStateWorker] DREAM_INBOX_ENQUEUE accepted — "
                    "user=%s session=%s priority=%d (inbox_size=%d, total=%d)",
                    validated.get("user_id"), validated.get("session_id"),
                    validated.get("priority"), len(inbox),
                    inbox_enqueue_count)

            elif msg_type == KERNEL_EPOCH_TICK:
                kernel_tick_count += 1
                # Dual-trigger Q6 republish on every kernel epoch tick.
                now = time.time()
                if (now - last_kernel_tick_republish_ts) >= DREAM_STATE_REPUBLISH_CADENCE_S:
                    publisher.publish()
                    last_kernel_tick_republish_ts = now

            elif msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[DreamStateWorker] MODULE_SHUTDOWN received — exiting "
                    "(stats: dreaming_state_updated=%d wake_requests=%d "
                    "transitions=%d inbox_enqueues=%d inbox_replays=%d "
                    "kernel_ticks=%d inbox_size_at_exit=%d)",
                    dreaming_state_updated_count, wake_request_count,
                    transitions_count, inbox_enqueue_count,
                    inbox_replay_count, kernel_tick_count, len(inbox))
                break

            # Periodic heartbeat log every 600 events for observability.
            total_events = (dreaming_state_updated_count
                            + wake_request_count + inbox_enqueue_count
                            + kernel_tick_count)
            if total_events > 0 and total_events % 600 == 0:
                logger.info(
                    "[DreamStateWorker] heartbeat — events=%d "
                    "(DREAMING_STATE_UPDATED=%d, DREAM_WAKE_REQUEST=%d, "
                    "DREAM_INBOX_ENQUEUE=%d, KERNEL_EPOCH_TICK=%d) "
                    "transitions=%d inbox_replays=%d inbox_size=%d",
                    total_events, dreaming_state_updated_count,
                    wake_request_count, inbox_enqueue_count,
                    kernel_tick_count, transitions_count,
                    inbox_replay_count, len(inbox))

    except KeyboardInterrupt:
        logger.info("[DreamStateWorker] KeyboardInterrupt — exiting")
    except Exception as e:
        logger.error(
            "[DreamStateWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        raise
    finally:
        stop_event.set()
        logger.info("[DreamStateWorker] shutdown complete")
