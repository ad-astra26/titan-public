"""
interface_advisor_worker — Python L2 module owning InterfaceAdvisor
(per-message-type sliding-window rate limiter) per
`rFP_titan_hcl_l2_separation_strategy.md §4.H` + SPEC v1.8.5 §9.B
(D-SPEC-59, 2026-05-15). Maker greenlit Path Y (SHM-rate-oracle) inline.

History: Pre-v1.8.5, InterfaceAdvisor was instantiated in the parent plugin
process at `plugin.py:1976-1984` with the in-code ADR "InterfaceAdvisor
stays in parent (cheap rate check before bus round-trip)" — a pre-Phase-C-
fleet-wide decision from A.8.6 agency_worker extraction. v1.8.5 §4.H closure
revisits per `feedback_phase_c_break_monolith_ethos.md`: every L2 carve
under Phase C earns its place via hot-reload + restart-isolation + own
§9.B block — perf is not the criterion.

What this worker owns:
  1. InterfaceAdvisor instance (logic/interface_advisor.py, 162 LOC) —
     per-msg-type sliding-window rate limiter with INITIAL_LIMITS for
     IMPULSE / INTERFACE_INPUT / BODY_STATE / MIND_STATE / SPIRIT_STATE /
     ACTION_RESULT / INTENT (60s window).
  2. interface_advisor_state.bin SHM slot writer (G21 single-writer;
     rate-throttled to INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S = 0.1s
     (10Hz cap) to avoid SHM thrash under burst).
  3. IMPULSE_RECEIVED bus event subscriber → records timestamp in deque
     via advisor.check() + republishes SHM + on rate-exceeded emits
     RATE_LIMIT bus event back to source.

Bus subscriptions (REQUIRED):
  • IMPULSE_RECEIVED — fired by parent `_handle_impulse` (and any future
                       caller emitting before processing); payload =
                       {msg_type: str, source: str, client_ts: float}
  • MODULE_SHUTDOWN  — clean shutdown signal

Bus publications (all non-blocking per §8.0.ter D-SPEC-48):
  • RATE_LIMIT       — on rate exceeded, dst=source from payload; payload =
                       advisor.check() feedback dict
  • MODULE_HEARTBEAT — every 30s (daemon thread)
  • (Phase 11 §11.I.2 D2: legacy boot-signal bus emit DELETED — SHM
     slot `module_<name>_state.bin` state=booted is the contract)

Persisted state: none — sliding-window deques are volatile by design
(60s rolling window — restart loses ≤60s of rate history; next IMPULSE
re-warms).

External I/O: bus client only (no DB, no Solana RPC, no HTTP).

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • REQUIRED: none — worker boots independently; cold SHM slot returns
              defaults (zero rates) until first IMPULSE_RECEIVED arrives.
              Parent's _handle_impulse can call InterfaceAdvisorStateReader
              even before worker boots — gets defaults, passes all checks,
              emits IMPULSE_RECEIVED which boot-buffers per §8.0.bis.

See:
  - SPEC v1.8.5 §9.B `interface_advisor_worker` block
  - SPEC v1.8.5 §7.1 `interface_advisor_state.bin` SHM slot row
  - SPEC v1.8.5 §8.7 `IMPULSE_RECEIVED` bus event row
  - SPEC v1.8.5 §21 D-SPEC-59
  - PLAN_microkernel_phase_c_l2_maybe_tier_cleanup.md §3
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty
from typing import Optional

from titan_hcl._phase_c_constants import (
    INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S,
)
from titan_hcl.bus import (
    IMPULSE_RECEIVED,
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    RATE_LIMIT,
    make_msg,
)
from titan_hcl.logic.interface_advisor_publisher import (
    InterfaceAdvisorStatePublisher,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0


# Phase 11 §11.I.5 (Chunk 11N) — module-level readiness sentinel; gates
# SHM-slot heartbeat() (legacy bus heartbeat fires unconditionally for
# the boot window so guardian_HCL's stale-heartbeat detector doesn't
# kill a slow boot).
_WORKER_READY: bool = False


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[InterfaceAdvisorWorker] _send %s → %s failed: %s",
            msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str,
                    stop_event: threading.Event,
                    state_writer: Optional[object] = None) -> None:
    """Daemon thread — MODULE_HEARTBEAT every 30s.

    Phase 11 §11.I.5: also publishes state_writer.heartbeat() on the SHM
    slot once `_WORKER_READY` is True so guardian_HCL's SHM-staleness
    detector + observatory `/v6/readiness` see fresh data on the same
    cadence as the legacy bus path. SHM writes are best-effort.
    """
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        if state_writer is not None and _WORKER_READY:
            try:
                state_writer.heartbeat()
            except Exception:  # noqa: BLE001 — never crash the heartbeat
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


@with_error_envelope(module_name="interface_advisor", subsystem="entry", severity=_phase11_sev.FATAL)
def interface_advisor_worker_main(recv_queue, send_queue, name: str,
                                  config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Resolve titan_id, build InterfaceAdvisorStatePublisher
         (composes InterfaceAdvisor)
      2. Start heartbeat thread (daemon, 30s cadence)
      3. Cold-boot SHM publish (empty rates dict + INITIAL_LIMITS)
      4. Phase 11 §11.I.2 — SHM slot state=booted (parent's
         InterfaceAdvisorStateReader can now SHM-read defaults)
      5. Main loop: drain recv_queue, dispatch by msg_type:
           - IMPULSE_RECEIVED  → advisor.check(msg_type, source);
                                 if rate exceeded, emit RATE_LIMIT to
                                 source; throttle SHM republish to
                                 INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S
           - MODULE_SHUTDOWN   → graceful exit
    """
    global _WORKER_READY
    _WORKER_READY = False

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id(config.get("titan_id") if config else None)

    logger.info(
        "[InterfaceAdvisorWorker] booting — titan_id=%s name=%s "
        "(SPEC v1.8.5 §9.B / D-SPEC-59 / rFP §4.H)",
        titan_id, name)

    # ── Phase 11 §11.I.5 (Chunk 11N) — SHM state-slot writer ──
    # Construct BEFORE slow init so the slot publishes state="starting"
    # immediately; titan_hcl's 1Hz SHM poll then sees liveness during the
    # cold-boot window. Heartbeat thread started immediately below so
    # last_heartbeat stays fresh during cold-boot window before slot=booted.
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
            "[InterfaceAdvisorWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer),
        daemon=True, name=f"interface-advisor-hb-{name}")
    hb_thread.start()

    publisher = InterfaceAdvisorStatePublisher(titan_id)

    # Cold-boot first SHM publish; readers can now mmap.
    publisher.publish()

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[InterfaceAdvisorWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    logger.info(
        "[InterfaceAdvisorWorker] boot complete — Phase 11 SHM slot=booted "
        "(awaiting MODULE_PROBE_REQUEST); interface_advisor_state.bin "
        "initialized with cold defaults")

    # Counters for heartbeat observability.
    impulse_count = 0
    rate_limit_count = 0
    last_publish_ts = time.time()
    pending_unpublished = False  # set when a record was throttled-skip-published

    try:
        while True:
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                # No activity → check if there's pending unpublished state
                # from a throttled record + the throttle window has elapsed.
                # This guarantees eventual SHM consistency under sparse
                # traffic (single IMPULSE_RECEIVED still reaches SHM
                # within ≤INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S + 1s
                # idle-wait).
                if pending_unpublished and (
                    time.time() - last_publish_ts
                    >= INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S
                ):
                    publisher.publish()
                    last_publish_ts = time.time()
                    pending_unpublished = False
                continue

            if msg is None:
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ──
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
                        "[InterfaceAdvisorWorker] MODULE_PROBE_REQUEST handler "
                        "failed: %s", _probe_err)
                continue

            if msg_type == IMPULSE_RECEIVED:
                impulse_count += 1
                # Defensive payload extraction.
                advisor_msg_type = str(payload.get("msg_type", "")) or "IMPULSE"
                source = str(payload.get("source", "unknown"))

                # Record + republish (rate-throttled).
                now = time.time()
                if (now - last_publish_ts) >= INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S:
                    feedback = publisher.record_and_publish(
                        advisor_msg_type, source=source)
                    last_publish_ts = now
                    pending_unpublished = False
                else:
                    # Throttle: just record in advisor (sliding window updates),
                    # skip the SHM write now. Mark pending so the next
                    # opportunity (next event OR idle Empty tick after the
                    # throttle window) will publish.
                    feedback = publisher.advisor.check(advisor_msg_type, source)
                    pending_unpublished = True

                if feedback is not None:
                    rate_limit_count += 1
                    _send(send_queue, RATE_LIMIT, name, source, feedback)
                    if rate_limit_count <= 5 or rate_limit_count % 50 == 0:
                        logger.info(
                            "[InterfaceAdvisorWorker] RATE_LIMIT emitted "
                            "(#%d) — msg_type=%s source=%s "
                            "current_rate=%d limit=%d",
                            rate_limit_count, advisor_msg_type, source,
                            feedback.get("current_rate"),
                            feedback.get("limit"))

            elif msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[InterfaceAdvisorWorker] MODULE_SHUTDOWN received — "
                    "exiting (impulses=%d rate_limits_emitted=%d)",
                    impulse_count, rate_limit_count)
                break

            # Periodic heartbeat log every 500 events for observability.
            if impulse_count > 0 and impulse_count % 500 == 0:
                stats = publisher.advisor.get_stats()
                logger.info(
                    "[InterfaceAdvisorWorker] heartbeat — impulses=%d "
                    "rate_limits=%d active_types=%d current_rates=%s",
                    impulse_count, rate_limit_count,
                    len(stats.get("current_rates", {})),
                    dict(stats.get("current_rates", {})))

    except KeyboardInterrupt:
        logger.info("[InterfaceAdvisorWorker] KeyboardInterrupt — exiting")
    except Exception as e:
        logger.error(
            "[InterfaceAdvisorWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        raise
    finally:
        stop_event.set()
        logger.info("[InterfaceAdvisorWorker] shutdown complete")
