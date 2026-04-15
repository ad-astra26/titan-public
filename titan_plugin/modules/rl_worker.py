"""
RL/Sage Module Worker — runs TorchRL IQL + SageScholar/Gatekeeper/Recorder
in its own supervised process.

This is the biggest memory win: TorchRL LazyMemmapStorage is ~2GB.
Isolating it saves Core from that entire footprint.

Entry point: rl_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def rl_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """
    Main loop for the RL module process.

    Args:
        recv_queue: receives messages from DivineBus (bus→worker)
        send_queue: sends messages back to DivineBus (worker→bus)
        name: module name ("rl")
        config: dict from [stealth_sage] config section
    """
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[RLWorker] Initializing Sage subsystems...")
    init_start = time.time()

    try:
        from titan_plugin.core.sage.recorder import SageRecorder
        from titan_plugin.logic.sage.scholar import SageScholar
        from titan_plugin.logic.sage.gatekeeper import SageGatekeeper

        recorder = SageRecorder()
        scholar = SageScholar(recorder)
        gatekeeper = SageGatekeeper(scholar, recorder)

        init_ms = (time.time() - init_start) * 1000
        logger.info("[RLWorker] Sage subsystems ready in %.0fms", init_ms)
    except Exception as e:
        logger.error("[RLWorker] Failed to init Sage: %s", e, exc_info=True)
        return

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    last_heartbeat = time.time()

    while True:
        # Heartbeat every iteration (throttled internally to 3s min).
        # MUST be at top — NOT only in the `except Empty` branch — because
        # broadcast messages (e.g. TITAN_SELF_STATE dst="all") arriving
        # within the get() timeout starve the Empty path, causing Guardian
        # heartbeat timeouts. See media_worker.py fix 2026-04-15 and audit.
        if time.time() - last_heartbeat > 3.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[RLWorker] Shutdown requested")
            break

        if msg_type == "QUERY":
            _handle_query(msg, recorder, scholar, gatekeeper, send_queue, name)

    logger.info("[RLWorker] Exiting")


def _handle_query(msg: dict, recorder, scholar, gatekeeper, send_queue, name: str) -> None:
    """Handle RL-related queries."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "evaluate":
            import torch
            state = payload.get("state", [0.0] * 128)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            result = gatekeeper.evaluate(state_tensor)
            mode, advantage = result if isinstance(result, tuple) else ("Shadow", 0.0)
            _send_response(send_queue, name, src, {
                "mode": mode,
                "advantage": float(advantage),
            }, rid)

        elif action == "record":
            obs = payload.get("observation", [])
            action_idx = payload.get("action_idx", 0)
            reward = payload.get("reward", 0.0)
            next_obs = payload.get("next_observation", [])
            done = payload.get("done", False)
            tid = recorder.record_transition(obs, action_idx, reward, next_obs, done)
            _send_response(send_queue, name, src, {"transition_id": tid}, rid)

        elif action == "stats":
            stats = {
                "total_transitions": recorder.total_transitions if hasattr(recorder, 'total_transitions') else 0,
                "buffer_size": len(recorder) if hasattr(recorder, '__len__') else 0,
            }
            _send_response(send_queue, name, src, stats, rid)

        else:
            logger.warning("[RLWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[RLWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    """Send a message via the send queue (worker→bus)."""
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})
