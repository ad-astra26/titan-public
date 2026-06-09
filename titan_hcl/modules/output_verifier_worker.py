"""
Output Verifier Worker — L2 Subprocess (rFP_microkernel_phase_a8 §A.8.3).

Owns the OutputVerifier instance + responds to verify_and_sign queries
over the bus. Removes the ~5-15ms Ed25519 sign + regex-pattern blocking
work from the parent event loop, so the parent stays responsive during
shadow-swap adoption + general bus-traffic spikes.

Bus protocol:
  CONSUMES: QUERY(dst="output_verifier", payload.action="verify_and_sign"|"build_timechain_payload"|"stats")
  EMITS:    RESPONSE(dst=requester, payload=dataclass_dict)
            OUTPUT_VERIFIER_READY (titan_id, ts) — once on boot
            OUTPUT_VERIFIER_STATS (verified_count, rejected_count, sovereignty_score) — every 60s

When `microkernel.a8_output_verifier_subprocess_enabled=false` (default),
this worker is NOT autostarted by Guardian — the parent's local
OutputVerifier instance handles all calls (legacy behavior, byte-identical).

When the flag flips, parent's _output_verifier becomes an OutputVerifierProxy
that uses bus.request(...) to reach this worker.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.3
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from queue import Empty
from typing import Optional
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("output_verifier")

_HEARTBEAT_INTERVAL_S = 30.0
_STATS_PUBLISH_INTERVAL_S = 60.0

# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel; SHM heartbeat
# is suppressed until the worker has finished in-process scaffolding.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


@with_error_envelope(module_name="output_verifier", subsystem="entry", severity=_phase11_sev.FATAL)
def output_verifier_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the OutputVerifier worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal "output_verifier")
        config: full config dict — reads info_banner.titan_id +
                memory_and_storage.data_dir + network.wallet_keypair_path
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (full_config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    data_dir = (full_config.get("memory_and_storage", {}) or {}).get("data_dir", "./data")
    tc_dir = os.path.join(data_dir, "timechain")
    keypair_path = (full_config.get("network", {}) or {}).get(
        "wallet_keypair_path", "data/titan_identity_keypair.json")

    logger.info("[OutputVerifierWorker] Booting — titan_id=%s, tc_dir=%s",
                titan_id, tc_dir)

    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 per worker) ──
    # Created BEFORE the slow OutputVerifier init so the slot publishes
    # state="starting" immediately and the periodic heartbeat below keeps
    # last_heartbeat fresh during boot.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name="output_verifier",
            layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[OutputVerifierWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    try:
        from titan_hcl.logic.output_verifier import OutputVerifier
        verifier = OutputVerifier(
            titan_id=titan_id, data_dir=tc_dir, keypair_path=keypair_path)
    except Exception as e:
        logger.error("[OutputVerifierWorker] OutputVerifier init failed: %s — exiting", e)
        return

    # Phase 11 §11.I.2 — slot transition: starting → booted.
    # MODULE_READY bus-emit deleted per locked D2 (no shim).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[OutputVerifierWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[OutputVerifierWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    # OUTPUT_VERIFIER_READY remains — separate broadcast for parent-side proxy
    # to mark itself bus-routed (distinct from Phase 11 MODULE_READY contract).
    try:
        send_queue.put({
            "type": bus.OUTPUT_VERIFIER_READY, "src": name, "dst": "all",
            "payload": {"titan_id": titan_id, "ts": time.time()},
            "ts": time.time(),
        })
    except Exception:
        pass

    last_heartbeat = 0.0
    last_stats_publish = 0.0
    poll_interval_s = 0.2

    # Phase C Session 3 (rFP §4.B.11) — SHM-direct output_verifier_state.bin
    # publisher. Replaces the deadlock-prone sync bus.request path for
    # verifier stats. Cadence: 1 Hz.
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.output_verifier_state_publisher import (
            OutputVerifierStatePublisher)
        from titan_hcl.logic.worker_publisher_runner import (
            run_worker_publisher)
        _ov_state_publisher = OutputVerifierStatePublisher(
            titan_id=resolve_titan_id())
        run_worker_publisher(
            publisher=_ov_state_publisher,
            state_fetcher=lambda: verifier,
            worker_name="output_verifier_worker",
            cadence_s=1.0,
        )
    except Exception as _pub_init_err:
        logger.error(
            "[OutputVerifierWorker] SHM publisher BOOT FAILED — "
            "consumers fall back to sync bus.request path: %s",
            _pub_init_err, exc_info=True)

    while True:
        # Periodic heartbeat (Guardian liveness + Phase 11 SHM heartbeat)
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now},
                    "ts": now,
                })
            except Exception:
                pass
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:
                    pass
            last_heartbeat = now

        # Periodic stats publish (kernel snapshot consumes this)
        if now - last_stats_publish >= _STATS_PUBLISH_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.OUTPUT_VERIFIER_STATS, "src": name, "dst": "all",
                    "payload": {
                        # P3 (RFP_synthesis_decision_authority): the vestigial OVG
                        # `sovereignty_score` (always 0.0, no real consumer) is
                        # removed — sovereignty is the ONE metric S, in synthesis.
                        "verified_count": int(getattr(verifier, "verified_count", 0) or 0),
                        "rejected_count": int(getattr(verifier, "rejected_count", 0) or 0),
                        # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a — feeds
                        # outer_mind willing[13] protective_response. Rolling
                        # window rates from _rejection_timestamps deque.
                        "rejected_per_hour": float(getattr(verifier, "rejected_per_hour", 0.0) or 0.0),
                        "rejected_per_day": float(getattr(verifier, "rejected_per_day", 0.0) or 0.0),
                        "ts": now,
                    },
                    "ts": now,
                })
            except Exception:
                pass
            # AUDIT §C fix (rFP §P2): also flush cumulative counters to disk on
            # the 60s cadence so an ungraceful kill loses ≤60s, not everything.
            # (output_verifier is reply_only, so SAVE_NOW broadcasts never reach
            # it — periodic + MODULE_SHUTDOWN flush is the durable path.)
            try:
                verifier.save_counters()
            except Exception:  # noqa: BLE001
                pass
            last_stats_publish = now

        # Drain bus messages
        try:
            msg = recv_queue.get(timeout=poll_interval_s)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        # Spawn-mode workers outlive kernel swap via BUS_HANDOFF adoption.
        # Added 2026-04-28 PM during shadow swap E2E test (newer worker
        # missing original 15-worker wiring).
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
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
                    "[OutputVerifierWorker] Phase 11 probe handler raised: %s",
                    _phb_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[OutputVerifierWorker] Shutdown received — "
                        "persisting counters + exiting")
            # AUDIT §C fix (rFP §P2): flush cumulative verified/rejected counters
            # so they survive hot-reload / kill-respawn (were lost → reset to 0).
            try:
                verifier.save_counters()
            except Exception:  # noqa: BLE001
                pass
            return
        if msg_type != bus.QUERY:
            continue

        rid = msg.get("rid")
        src = msg.get("src", "unknown")
        payload = msg.get("payload") or {}
        action = payload.get("action")

        try:
            if action == "verify_and_sign":
                # D-SPEC-74 (SPEC v1.18.0) — legacy combined entry; kept for
                # back-compat during migration. New callers use verify_safety
                # + sign_and_commit explicitly per the split below.
                result = verifier.verify_and_sign(
                    output_text=payload.get("output_text", ""),
                    channel=payload.get("channel", "chat"),
                    injected_context=payload.get("injected_context", ""),
                    prompt_text=payload.get("prompt_text", ""),
                    chain_state=payload.get("chain_state"),
                )
                response_payload = dataclasses.asdict(result)
            elif action == "verify_safety":
                # D-SPEC-74 Phase 1 — deterministic truth-gate, no signing.
                # Returns SafetyResult including safety_verdict_token (HMAC)
                # for the paired sign_and_commit call.
                result = verifier.verify_safety(
                    output_text=payload.get("output_text", ""),
                    channel=payload.get("channel", "chat"),
                    injected_context=payload.get("injected_context", ""),
                    prompt_text=payload.get("prompt_text", ""),
                    chain_state=payload.get("chain_state"),
                )
                response_payload = dataclasses.asdict(result)
            elif action == "sign_and_commit":
                # D-SPEC-74 Phase 2 — Ed25519 sign + TimeChain commit.
                # Validates safety_verdict_token HMAC + TTL before signing.
                result = verifier.sign_and_commit(
                    output_text=payload.get("output_text", ""),
                    channel=payload.get("channel", "chat"),
                    prompt_text=payload.get("prompt_text", ""),
                    chain_state=payload.get("chain_state"),
                    safety_verdict_token=payload.get(
                        "safety_verdict_token", ""),
                    verdict_ts=payload.get("verdict_ts", 0.0),
                )
                response_payload = dataclasses.asdict(result)
            elif action == "build_timechain_payload":
                # Reconstruct OVGResult from dict, then call build_timechain_payload
                from titan_hcl.logic.output_verifier import OVGResult
                _result_dict = payload.get("result_dict") or {}
                _result = OVGResult(**_result_dict)
                tc_payload = verifier.build_timechain_payload(
                    _result, **payload.get("kwargs", {}),
                )
                response_payload = {"timechain_payload": tc_payload}
            # Phase C Session 5 (rFP §4.D.4): "stats" handler RETIRED —
            # output_verifier_proxy.get_stats now SHM-direct via
            # output_verifier_state.bin (Session 3 §4.B.11 publisher).
            else:
                response_payload = {"error": f"unknown action: {action}"}
        except Exception as e:
            logger.warning("[OutputVerifierWorker] action=%s failed: %s", action, e)
            response_payload = {"error": str(e)}

        try:
            send_queue.put({
                "type": bus.RESPONSE, "src": name, "dst": src, "rid": rid,
                "payload": response_payload, "ts": time.time(),
            })
        except Exception as e:
            logger.warning("[OutputVerifierWorker] response send failed: %s", e)
