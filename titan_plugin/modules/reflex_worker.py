"""
Reflex Worker — L3 Subprocess (rFP_microkernel_phase_a8 §A.8.5).

Hosts a stateless ReflexCollector instance (no executors registered) that
performs the aggregation step of the reflex arc:
  group signals → guardian-shield fast-path → combined confidence
  → threshold + cooldown filter → top-N selection.

Cooldowns are owned by the parent's ReflexProxy (the parent runs the
executors and writes cooldowns on successful fire). Each request from
the proxy includes the current cooldowns snapshot.

Bus protocol:
  CONSUMES: QUERY(dst="reflex", payload.action="aggregate",
              payload.signals, payload.stimulus_features,
              payload.focus_magnitude, payload.cooldowns)
  EMITS:    RESPONSE(dst=requester, payload={
              "selected_serial": [{"reflex_type", "combined_confidence",
                                   "signals"}],
              "notices": []
            })
            REFLEX_READY (titan_id, ts) — once on boot
            MODULE_READY/MODULE_HEARTBEAT — Guardian state machine

When `microkernel.a8_reflex_subprocess_enabled=false` (default), this
worker is NOT autostarted by Guardian — the parent's reflex_collector
is the regular ReflexCollector (legacy behavior, byte-identical).

When the flag flips, parent's reflex_collector becomes a ReflexProxy
that bus-requests this worker for aggregation. Executors stay parent-
resident (they reference plugin.soul / plugin.metabolism / plugin.
memory_proxy etc. and cannot trivially move).

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.5
"""
from __future__ import annotations

import logging
import os
import sys
import time
from queue import Empty
from titan_plugin import bus

logger = logging.getLogger("reflex")

_HEARTBEAT_INTERVAL_S = 30.0


def reflex_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Reflex worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal "reflex")
        config: full config dict — used for titan_id + reflex params.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    titan_id = (full_config.get("info_banner", {}) or {}).get("titan_id") or "T1"

    logger.info("[ReflexWorker] Booting — titan_id=%s", titan_id)

    try:
        from titan_plugin.params import get_params
        from titan_plugin.logic.reflexes import ReflexCollector
        reflex_cfg = get_params("reflexes")
        # Stateless aggregator — no executors registered. Cooldowns are
        # synced from each request's payload (parent owns the truth).
        aggregator = ReflexCollector(reflex_cfg)
    except Exception as e:
        logger.error("[ReflexWorker] aggregator init failed: %s — exiting", e)
        return

    # Boot signals
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "ts": time.time()},
            "ts": time.time(),
        })
        send_queue.put({
            "type": bus.REFLEX_READY, "src": name, "dst": "all",
            "payload": {"titan_id": titan_id, "ts": time.time()},
            "ts": time.time(),
        })
    except Exception:
        pass

    last_heartbeat = 0.0
    poll_interval_s = 0.2
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
            logger.info("[ReflexWorker] Shutdown received — exiting")
            return
        if msg_type != bus.QUERY:
            continue

        rid = msg.get("rid")
        src = msg.get("src", "unknown")
        payload = msg.get("payload") or {}
        action = payload.get("action")

        if action != "aggregate":
            response_payload = {"error": f"unknown action: {action}"}
        else:
            try:
                signals = payload.get("signals") or []
                stimulus_features = payload.get("stimulus_features") or {}
                focus_magnitude = float(payload.get("focus_magnitude", 0.0))
                cooldowns = payload.get("cooldowns") or {}

                # Sync cooldowns from caller (parent owns the truth).
                aggregator._cooldowns = dict(cooldowns)

                request_count += 1
                selected = aggregator._aggregate_inline(
                    signals, stimulus_features, focus_magnitude,
                )

                # Serialize: ReflexType → str.value
                selected_serial = [
                    {
                        "reflex_type": rt.value,
                        "combined_confidence": float(conf),
                        "signals": list(sigs or []),
                    }
                    for rt, conf, sigs in selected
                ]
                response_payload = {
                    "selected_serial": selected_serial,
                    "notices": [],
                }
            except Exception as e:
                error_count += 1
                logger.warning("[ReflexWorker] aggregate failed (req=%d): %s",
                               request_count, e)
                response_payload = {
                    "selected_serial": [],
                    "notices": [f"aggregate_error: {e}"],
                    "error": str(e),
                }

        try:
            send_queue.put({
                "type": bus.RESPONSE, "src": name, "dst": src, "rid": rid,
                "payload": response_payload, "ts": time.time(),
            })
        except Exception as e:
            logger.warning("[ReflexWorker] response send failed: %s", e)
