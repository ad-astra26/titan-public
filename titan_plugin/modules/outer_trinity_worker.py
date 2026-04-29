"""
Outer Trinity Worker — L1 Subprocess (rFP_microkernel_phase_a8 §A.8.4).

Owns the OuterTrinityCollector instance and computes the 132D outer
trinity tensors (5D body + 15D mind + 45D spirit) on each
OUTER_TRINITY_COLLECT_REQUEST from the parent. Publishes the result
as OUTER_TRINITY_STATE for downstream consumers (state_register,
dashboard, spirit_worker — bus message unchanged from pre-A.8.4).

Design (A.8.4 Path A — parent pre-extracts):
  Source-gathering stays in parent because several inputs are
  parent-only (agency module, observatory_db reader, soul object).
  Parent flattens to plain types (including pre-extracted observatory
  expressive counts) and ships in the COLLECT_REQUEST payload. Worker
  is pure compute — no DB handles, no parent references.

Bus protocol:
  CONSUMES: OUTER_TRINITY_COLLECT_REQUEST(dst="outer_trinity",
              payload={"sources": <flat dict>})
  EMITS:    OUTER_TRINITY_STATE(dst="all", payload=<collector result>)
            OUTER_TRINITY_READY(dst="all") — once on boot
            MODULE_READY(dst="guardian") — Guardian state RUNNING
            MODULE_HEARTBEAT(dst="guardian") — every 30s, with counters

When `microkernel.a8_outer_trinity_subprocess_enabled=false` (default),
this worker is NOT autostarted by Guardian. Parent's
_outer_trinity_loop calls collector.collect() locally and publishes
OUTER_TRINITY_STATE itself (legacy behavior, byte-identical).

When the flag flips, parent stops local compute and instead publishes
COLLECT_REQUEST every interval; worker computes + publishes STATE.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.4
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty
from titan_plugin import bus

logger = logging.getLogger("outer_trinity")

_HEARTBEAT_INTERVAL_S = 30.0


def outer_trinity_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the OuterTrinity worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal "outer_trinity")
        config: full config dict — used only for titan_id at this stage
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    titan_id = (full_config.get("info_banner", {}) or {}).get("titan_id") or "T1"

    logger.info("[OuterTrinityWorker] Booting — titan_id=%s", titan_id)

    try:
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
    except Exception as e:
        logger.error("[OuterTrinityWorker] OuterTrinityCollector init failed: %s — exiting", e)
        return

    # Boot signals: MODULE_READY → Guardian flips state to RUNNING.
    # OUTER_TRINITY_READY is a separate broadcast for any consumer that
    # wants to know the worker is live (matches A.8.3 OV pattern).
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "ts": time.time()},
            "ts": time.time(),
        })
        send_queue.put({
            "type": bus.OUTER_TRINITY_READY, "src": name, "dst": "all",
            "payload": {"titan_id": titan_id, "ts": time.time()},
            "ts": time.time(),
        })
    except Exception:
        pass

    last_heartbeat = 0.0
    poll_interval_s = 0.5
    request_count = 0
    error_count = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {
                        "alive": True, "ts": now,
                        "request_count": request_count,
                        "error_count": error_count,
                    },
                    "ts": now,
                })
            except Exception:
                pass
            last_heartbeat = now

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
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[OuterTrinityWorker] Shutdown received — exiting")
            return
        if msg_type != bus.OUTER_TRINITY_COLLECT_REQUEST:
            continue

        payload = msg.get("payload") or {}
        sources = payload.get("sources") or {}

        request_count += 1
        try:
            result = collector.collect(sources)
        except Exception as e:
            error_count += 1
            logger.warning("[OuterTrinityWorker] collect failed (req=%d): %s",
                           request_count, e)
            continue

        try:
            send_queue.put({
                "type": bus.OUTER_TRINITY_STATE, "src": name, "dst": "all",
                "payload": result,
                "ts": time.time(),
            })
        except Exception as e:
            logger.warning("[OuterTrinityWorker] OUTER_TRINITY_STATE publish failed: %s", e)
