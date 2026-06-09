"""
sovereignty_worker — Python L2 module owning the GreatCycleTracker
(M10 GREAT CYCLE convergence tracker) per `rFP_titan_hcl_l2_separation_strategy.md §4.L`.

Phase C v1.8.3 (D-SPEC-57). Maker greenlit 2026-05-15 inline:
  Q1.L (§9.B block + §8.7 rows + Changelog + glossary + D-SPEC + ModuleSpec).
  Scope-expansion greenlit 2026-05-15: SOVEREIGNTY_CONFIRM_MAKER +
  SOVEREIGNTY_INCREMENT_GREAT_CYCLE bus events added to close the latent
  Phase C api_subprocess kernel_rpc-serialization-gap that was silently
  no-opping both mutator paths today
  ("we'll add both, but we need to diagnose it and make them work").

What this worker owns:
  1. GreatCycleTracker (logic/sovereignty.py, 222 LOC) — rolling neuromod
     convergence windows (per-modulator deque maxlen=CONVERGENCE_WINDOW=5000),
     ENFORCING → ADVISORY transition criteria evaluation, 5-point test
     (great_cycle ≥1, dev_age >1000, no neuromod saturation/collapse,
     total_great_pulses >1000, maker_confirmed).
  2. data/sovereignty_state.json — sole writer (G21 trivially satisfied;
     parent + legacy_core mirror both retired in same commit set).
  3. 100-message soak log emission (every ~1000 epochs, transition-criteria
     snapshot — observers can see long-horizon convergence signal).

Bus subscriptions (REQUIRED):
  • SOVEREIGNTY_EPOCH                 — spirit_worker producer; every 10
                                         consciousness epochs; targeted
                                         dst="sovereignty"; payload =
                                         {epoch_id, neuromods, dev_age,
                                          great_pulse_fired, total_great_pulses}
  • SOVEREIGNTY_CONFIRM_MAKER         — api/webhook.py producer on verified
                                         Maker directive (Helius webhook);
                                         idempotent; payload = {tx_signature,
                                          ts}
  • SOVEREIGNTY_INCREMENT_GREAT_CYCLE — api/maker.py producer on Resurrection
                                         Protocol completion; payload =
                                         {ts, source}
  • MODULE_SHUTDOWN                   — clean shutdown signal; _save_state()
                                         before exit

Bus publications (all non-blocking per §8.0.ter D-SPEC-48):
  • MODULE_HEARTBEAT (every 30s)
  • (MODULE_READY retired Phase 11 §11.I.2 — SHM slot state=booted is the contract)

Persisted state: data/sovereignty_state.json (atomic-write via temp+rename
per §11.H.2; schema unchanged from pre-carve).

External I/O: bus client only (no DB, no Solana RPC, no HTTP, no /dev/shm).

Dependencies (boot order via guardian_HCL — see SPEC §10.A):
  • REQUIRED: spirit_worker (SOVEREIGNTY_EPOCH producer; worker stays idle
                              until first message arrives, but boot-buffer
                              per §8.0.bis ensures no drop during boot race)
  • SOFT:     none — JSON state self-bootstrapping (load on init, save on
                     every transition + every 500 epochs + shutdown)

See:
  - SPEC v1.8.3 §9.B `sovereignty_worker` block
  - SPEC v1.8.3 §8.7 bus event rows (SOVEREIGNTY_CONFIRM_MAKER NEW,
    SOVEREIGNTY_INCREMENT_GREAT_CYCLE NEW)
  - SPEC v1.8.3 §21 D-SPEC-57
  - PLAN_microkernel_phase_c_l2_maybe_tier_cleanup.md §1
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty
from typing import Any, Optional

from titan_hcl.bus import (
    MODULE_HEARTBEAT,
    MODULE_PROBE_REQUEST,
    MODULE_SHUTDOWN,
    SOVEREIGNTY_CONFIRM_MAKER,
    SOVEREIGNTY_EPOCH,
    SOVEREIGNTY_INCREMENT_GREAT_CYCLE,
    make_msg,
)
from titan_hcl.logic.sovereignty import GreatCycleTracker
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
CRITERIA_SNAPSHOT_EVERY_N_EPOCHS = 100  # every ~1000 actual epochs (10:1 sampled)
_STATE_CHECKPOINT_INTERVAL_S = 300.0    # periodic disk checkpoint — survives an
#                                         ungraceful crash (loses <= 1 interval)

# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel consumed by
# this worker's heartbeat thread (SHM heartbeat is suppressed until the
# worker has finished in-process scaffolding).
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


def _send(send_queue, msg_type: str, src: str, dst: str,
          payload: dict, rid: Optional[str] = None) -> None:
    """Non-blocking publish helper (§8.0.ter D-SPEC-48 publish-non-blocking
    compliance — must not block caller on socket I/O)."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning(
            "[SovereigntyWorker] _send %s → %s failed: %s",
            msg_type, dst, e)


def _heartbeat_loop(send_queue, name: str, stop_event: threading.Event,
                    state_writer: Optional[Any] = None) -> None:
    """Phase 11 §11.I.5 — bus MODULE_HEARTBEAT + SHM state-slot heartbeat.

    Bus heartbeat fires always (legacy Guardian path load-bears during boot).
    SHM heartbeat is suppressed until _WORKER_READY is True so the slot stays
    at state="starting"/"booted" until the worker is actually serving."""
    while not stop_event.is_set():
        _send(send_queue, MODULE_HEARTBEAT, name, "guardian", {})
        if state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
            try:
                state_writer.heartbeat()
            except Exception:
                pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


@with_error_envelope(module_name="sovereignty", subsystem="entry", severity=_phase11_sev.FATAL)
def sovereignty_worker_main(recv_queue, send_queue, name: str,
                            config: dict) -> None:
    """L2 module entry — Guardian supervised.

    Boot sequence:
      1. Instantiate GreatCycleTracker (loads data/sovereignty_state.json
         if present; sub-µs cold-bootstrap if not)
      2. Start heartbeat thread (daemon, 30s cadence)
      3. Write SHM slot state=booted (Phase 11 §11.I.2 — MODULE_READY retired)
      4. Main loop: drain recv_queue, dispatch by msg_type:
           - SOVEREIGNTY_EPOCH                 → tracker.record_epoch(...)
           - SOVEREIGNTY_CONFIRM_MAKER         → tracker.confirm_maker()
           - SOVEREIGNTY_INCREMENT_GREAT_CYCLE → tracker.increment_great_cycle()
           - MODULE_SHUTDOWN                   → tracker._save_state(); exit
    """
    logger.info(
        "[SovereigntyWorker] booting — name=%s "
        "(SPEC v1.8.3 §9.B / D-SPEC-57 / rFP §4.L)", name)

    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 per worker) ──
    # Created BEFORE the slow init so the slot publishes state="starting"
    # immediately and heartbeats keep last_heartbeat fresh during boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name="sovereignty",
            layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[SovereigntyWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    # Heartbeat thread — started BEFORE slow init so SHM `last_heartbeat`
    # stays fresh during boot (SHM publish gated by _WORKER_READY).
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(send_queue, name, stop_event, _state_writer),
        daemon=True, name=f"sovereignty-hb-{name}")
    hb_thread.start()

    tracker = GreatCycleTracker()
    logger.info(
        "[SovereigntyWorker] tracker loaded — mode=%s great_cycle=%d "
        "great_pulses=%d dev_age=%d maker_confirmed=%s",
        tracker._sovereignty_mode, tracker._great_cycle,
        tracker._total_great_pulses, tracker._developmental_age,
        tracker._maker_confirmed)

    # Phase 11 §11.I.2 — slot transition: starting → booted.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[SovereigntyWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[SovereigntyWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # Counters for heartbeat observability.
    epoch_count = 0
    confirm_maker_count = 0
    increment_great_cycle_count = 0
    last_state_checkpoint = time.time()  # first checkpoint ~5min after boot
    _ckpt_thread = [None]                 # single-slot non-blocking writer

    try:
        while True:
            # ── Periodic disk checkpoint (survives ANY crash) — NON-BLOCKING ──
            # tracker._save_state() otherwise fires only every 500 epochs / on
            # graceful shutdown, so an ungraceful death (shm_pid_dead / SIGKILL /
            # SIGSEGV) loses all sovereignty-convergence state since the last save
            # (observed frozen >22h). Loop-top placement fires every iteration
            # (recv timeout=1s) regardless of traffic. Offloaded to a daemon
            # thread so the disk write never blocks the heartbeat (esp. under
            # disk/swap pressure); single-slot guard avoids thread pile-up.
            _now_ck = time.time()
            if (_now_ck - last_state_checkpoint > _STATE_CHECKPOINT_INTERVAL_S
                    and (_ckpt_thread[0] is None
                         or not _ckpt_thread[0].is_alive())):
                last_state_checkpoint = _now_ck

                def _do_ckpt():
                    try:
                        tracker._save_state()
                    except Exception as _ckpt_err:  # noqa: BLE001
                        logger.warning(
                            "[SovereigntyWorker] periodic checkpoint failed: %s",
                            _ckpt_err)
                _ckpt_thread[0] = threading.Thread(
                    target=_do_ckpt, daemon=True, name="sovereignty-checkpoint")
                _ckpt_thread[0].start()
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

            # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
            if msg_type == MODULE_PROBE_REQUEST and _state_writer is not None:
                try:
                    from titan_hcl.core.probe_dispatcher import (
                        handle_module_probe_request,
                    )
                    handle_module_probe_request(
                        msg,
                        probe_fn=None,  # trivial pass-through per §11.I.2
                        send_queue=send_queue,
                        module_name=name,
                        state_writer=_state_writer,
                    )
                except Exception as _phb_err:  # noqa: BLE001
                    logger.warning(
                        "[SovereigntyWorker] Phase 11 probe handler raised: %s",
                        _phb_err)
                continue

            if msg_type == SOVEREIGNTY_EPOCH:
                # Payload schema (spirit_worker.py:3845-3851):
                #   {epoch_id, neuromods, dev_age, great_pulse_fired,
                #    total_great_pulses}
                try:
                    tracker.record_epoch(
                        epoch_id=int(payload.get("epoch_id", 0)),
                        neuromod_levels=payload.get("neuromods", {}) or {},
                        developmental_age=int(payload.get("dev_age", 0)),
                        great_pulse_fired=bool(payload.get(
                            "great_pulse_fired", False)),
                    )
                except Exception as e:
                    logger.debug(
                        "[SovereigntyWorker] record_epoch failed: %s", e)

                epoch_count += 1
                # Periodic transition-criteria snapshot — every 100 messages
                # (~1000 actual epochs; long-horizon convergence signal for
                # soak observers).
                if epoch_count % CRITERIA_SNAPSHOT_EVERY_N_EPOCHS == 0:
                    try:
                        criteria = tracker.check_transition_criteria()
                        logger.info(
                            "[SovereigntyWorker] Criteria snapshot: "
                            "mode=%s dev_age=%d great_pulses=%d "
                            "sat_violations=%d collapse_violations=%d "
                            "all_met=%s (after %d epoch messages)",
                            criteria.get("sovereignty_mode"),
                            criteria.get("developmental_age"),
                            criteria.get("total_great_pulses"),
                            criteria.get("saturation_violations"),
                            criteria.get("collapse_violations"),
                            criteria.get("all_met"),
                            epoch_count)
                    except Exception as e:
                        logger.debug(
                            "[SovereigntyWorker] check_transition_criteria "
                            "failed: %s", e)

            elif msg_type == SOVEREIGNTY_CONFIRM_MAKER:
                # Idempotent — repeated calls no-op once _maker_confirmed=True.
                tx_signature = str(payload.get("tx_signature", "<unknown>"))
                if not tracker._maker_confirmed:
                    try:
                        tracker.confirm_maker()
                        confirm_maker_count += 1
                        logger.info(
                            "[SovereigntyWorker] SOVEREIGNTY_CONFIRM_MAKER "
                            "received from tx=%s — _maker_confirmed=True "
                            "(persisted); criteria checked next epoch",
                            tx_signature[:16])
                    except Exception as e:
                        logger.warning(
                            "[SovereigntyWorker] confirm_maker failed: %s", e)
                else:
                    logger.debug(
                        "[SovereigntyWorker] SOVEREIGNTY_CONFIRM_MAKER "
                        "from tx=%s — already confirmed (idempotent no-op)",
                        tx_signature[:16])

            elif msg_type == SOVEREIGNTY_INCREMENT_GREAT_CYCLE:
                source = str(payload.get("source", "unknown"))
                try:
                    prev_cycle = tracker._great_cycle
                    tracker.increment_great_cycle()
                    increment_great_cycle_count += 1
                    logger.critical(
                        "[SovereigntyWorker] SOVEREIGNTY_INCREMENT_GREAT_CYCLE "
                        "received from source=%s — cycle %d → %d (persisted)",
                        source, prev_cycle, tracker._great_cycle)
                except Exception as e:
                    logger.warning(
                        "[SovereigntyWorker] increment_great_cycle failed: %s",
                        e)

            elif msg_type == MODULE_SHUTDOWN:
                logger.info(
                    "[SovereigntyWorker] MODULE_SHUTDOWN received — exiting "
                    "(epochs=%d, confirm_maker=%d, increment_great_cycle=%d, "
                    "final_mode=%s, final_great_cycle=%d, "
                    "final_great_pulses=%d)",
                    epoch_count, confirm_maker_count,
                    increment_great_cycle_count,
                    tracker._sovereignty_mode, tracker._great_cycle,
                    tracker._total_great_pulses)
                # Flush state to JSON before exit.
                try:
                    tracker._save_state()
                except Exception as e:
                    logger.warning(
                        "[SovereigntyWorker] shutdown _save_state failed: %s",
                        e)
                break

            # Periodic heartbeat log every 500 events for observability.
            total_events = (epoch_count + confirm_maker_count
                            + increment_great_cycle_count)
            if total_events > 0 and total_events % 500 == 0:
                logger.info(
                    "[SovereigntyWorker] heartbeat — events=%d "
                    "(epochs=%d, confirm_maker=%d, increment_great_cycle=%d) "
                    "mode=%s great_cycle=%d great_pulses=%d dev_age=%d",
                    total_events, epoch_count, confirm_maker_count,
                    increment_great_cycle_count,
                    tracker._sovereignty_mode, tracker._great_cycle,
                    tracker._total_great_pulses,
                    tracker._developmental_age)

    except KeyboardInterrupt:
        logger.info("[SovereigntyWorker] KeyboardInterrupt — exiting")
        try:
            tracker._save_state()
        except Exception:
            pass
    except Exception as e:
        logger.error(
            "[SovereigntyWorker] unhandled exception in main loop: %s",
            e, exc_info=True)
        try:
            tracker._save_state()
        except Exception:
            pass
        raise
    finally:
        stop_event.set()
        logger.info("[SovereigntyWorker] shutdown complete")
