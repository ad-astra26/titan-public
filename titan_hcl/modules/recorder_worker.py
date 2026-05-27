"""
recorder_worker — Python L2 module owning the full Sage data hub
(SageRecorder + SageScholar + SageGatekeeper) per
`rFP_titan_hcl_l2_separation_strategy.md §4.N` + SPEC v1.8.4 §9.B
(D-SPEC-58, 2026-05-15).

History: This worker was operationally separated as `name="rl"` on
2026-04-28 (Microkernel v2 Layer 2 migration — closed
BUG-SAGE-INSTANTIATED-IN-PARENT). SageRecorder owns ~2GB
TorchRL LazyMemmapStorage; isolating it saved Core from that entire
footprint. §4.N closure renames the worker to its functionally-correct
canonical name "recorder" (was "rl") + adds the SPEC §9.B block that
was missing through the v1.7.x cascade.

Owns (unchanged from rl_worker scope):
  • SageRecorder         — LazyMemmapStorage (~2GB disk) + in-memory
                            buffer deque (transition records).
  • SageScholar          — IQL trainer (recorder.buffer.sample() in
                            hot dream() loop — in-proc reference).
  • SageGatekeeper       — execution-mode router (iterates
                            recorder.storage in decide_execution_mode).
  • recorder_state.bin   — SHM publisher (G21 single writer; 1Hz
                            cadence; mirrors SAGE_STATS broadcast
                            schema for SHM-direct consumers).

Entry point: recorder_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import sys
import time
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


@with_error_envelope(module_name="recorder", subsystem="entry", severity=_phase11_sev.FATAL)
def recorder_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """
    Main loop for the RL module process.

    Args:
        recv_queue: receives messages from DivineBus (bus→worker)
        send_queue: sends messages back to DivineBus (worker→bus)
        name: module name ("recorder")
        config: dict from [stealth_sage] config section
    """
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[RecorderWorker] Initializing Sage subsystems...")
    init_start = time.time()

    try:
        from titan_hcl.core.sage.recorder import SageRecorder
        from titan_hcl.logic.sage.scholar import SageScholar
        from titan_hcl.logic.sage.gatekeeper import SageGatekeeper

        recorder = SageRecorder()
        scholar = SageScholar(recorder)
        gatekeeper = SageGatekeeper(scholar, recorder)

        init_ms = (time.time() - init_start) * 1000
        logger.info("[RecorderWorker] Sage subsystems ready in %.0fms", init_ms)
    except Exception as e:
        logger.error("[RecorderWorker] Failed to init Sage: %s", e, exc_info=True)
        return

    # Signal ready — mirrors A.8.3 OutputVerifier dual-emit (commit 9406f13f):
    # MODULE_READY for Guardian state STARTING→RUNNING, SAGE_READY broadcast
    # for any subscriber waiting on recorder_worker availability.
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})
    _send_msg(send_queue, bus.SAGE_READY, name, "all", {
        "buffer_size": getattr(recorder, "buffer_size", 0),
        "iql_available": getattr(scholar, "iql_loss", None) is not None,
    })

    last_heartbeat = time.time()
    last_stats_broadcast = time.time()

    # Phase C Session 3 (rFP §4.B.5) — SHM-direct rl_state.bin publisher.
    # Replaces the deadlock-prone sync bus.request(action="stats") path.
    # Mirrors the existing SAGE_STATS broadcast schema so consumers see
    # the same fields via SHM. Cadence: 1 Hz.
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.rl_state_publisher import RLStatePublisher
        from titan_hcl.logic.worker_publisher_runner import (
            run_worker_publisher)
        _rl_state_publisher = RLStatePublisher(titan_id=resolve_titan_id())
        run_worker_publisher(
            publisher=_rl_state_publisher,
            state_fetcher=lambda: (recorder, gatekeeper),
            worker_name="recorder_worker",
            cadence_s=1.0,
            publish_args=lambda s: s,  # state is the (recorder, gatekeeper) tuple
        )
    except Exception as _pub_init_err:
        logger.error(
            "[RecorderWorker] SHM publisher BOOT FAILED — "
            "consumers fall back to sync bus.request path: %s",
            _pub_init_err, exc_info=True)

    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        # Heartbeat every iteration (throttled internally to 3s min).
        # MUST be at top — NOT only in the `except Empty` branch — because
        # broadcast messages (e.g. TITAN_SELF_STATE dst="all") arriving
        # within the get() timeout starve the Empty path, causing Guardian
        # heartbeat timeouts. See media_worker.py fix 2026-04-15 and audit.
        if time.time() - last_heartbeat > 3.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()

        # ── §A.8.7 — SAGE_STATS broadcast every 60s ─────────────────
        # Subscribers (RLProxy.sovereignty_score cache, dashboard) read this
        # without per-call bus round-trips. Pattern matches AGENCY_STATS /
        # OUTPUT_VERIFIER_STATS (60s broadcast cadence).
        if time.time() - last_stats_broadcast > 60.0:
            _broadcast_sage_stats(send_queue, name, recorder, gatekeeper)
            last_stats_broadcast = time.time()

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[RecorderWorker] Shutdown requested")
            break

        # ── Microkernel v2 Layer 2 (2026-04-28) — Sage subprocess migration ──
        # Parent publishes transition records via SAGE_RECORD_TRANSITION;
        # we own the LazyMemmapStorage in this subprocess.
        if msg_type == bus.SAGE_RECORD_TRANSITION:
            _handle_sage_record_transition(msg, recorder, send_queue, name)
            continue

        if msg_type == bus.QUERY:
            _handle_query(msg, recorder, scholar, gatekeeper, send_queue, name)

    logger.info("[RecorderWorker] Exiting")


def _handle_query(msg: dict, recorder, scholar, gatekeeper, send_queue, name: str) -> None:
    """Handle RL-related queries.

    §A.8.7 actions:
        decide_execution_mode — gatekeeper routing decision (sub-second)
        dream                 — IQL training step (LLM-time scale, 30-90s)
        evaluate              — back-compat alias for decide_execution_mode
        record                — back-compat (Layer 2 also routes via SAGE_RECORD_TRANSITION)
        stats                 — recorder + gatekeeper stats snapshot
    """
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action in ("evaluate", "decide_execution_mode"):
            # §A.8.7: Gatekeeper.decide_execution_mode returns (mode, advantage,
            # decoded_text). Caller passes 128-D state_tensor + raw_prompt.
            import torch
            state = payload.get("state_tensor", payload.get("state", [0.0] * 128))
            raw_prompt = payload.get("raw_prompt", "")
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mode, advantage, decoded_text = gatekeeper.decide_execution_mode(
                state_tensor, raw_prompt=raw_prompt,
            )
            _send_response(send_queue, name, src, {
                "mode": mode,
                "advantage": float(advantage),
                "decoded_text": decoded_text,
                "sovereignty_score": float(getattr(gatekeeper, "sovereignty_score", 0.0)),
            }, rid)

        elif action == "dream":
            # §A.8.7: SageScholar.dream is async — spin a fresh event loop per
            # call (mirrors Layer 2 SAGE_RECORD_TRANSITION asyncio.run pattern).
            # Caller's 120s timeout matches the typical 30-90s IQL training
            # window for epochs=50 batch_size=256.
            import asyncio
            epochs = int(payload.get("epochs", 50))
            batch_size = int(payload.get("batch_size", 256))
            dream_results = asyncio.run(scholar.dream(
                epochs=epochs, batch_size=batch_size,
            ))
            buffer_len = len(recorder.buffer) if (
                getattr(recorder, "buffer", None) is not None
            ) else 0
            _send_response(send_queue, name, src, {
                "loss_actor": float(dream_results.get("loss_actor", 0.0)),
                "loss_qvalue": float(dream_results.get("loss_qvalue", 0.0)),
                "loss_value": float(dream_results.get("loss_value", 0.0)),
                "buffer_len": int(buffer_len),
                "epochs": epochs,
            }, rid)

        elif action == "record":
            # Back-compat path. Layer 2 SAGE_RECORD_TRANSITION is the
            # canonical bus message for transition records — this query
            # action stays for callers that prefer request/response.
            obs = payload.get("observation", [])
            action_idx = payload.get("action_idx", 0)
            reward = payload.get("reward", 0.0)
            next_obs = payload.get("next_observation", [])
            done = payload.get("done", False)
            tid = recorder.record_transition(obs, action_idx, reward, next_obs, done)
            _send_response(send_queue, name, src, {"transition_id": tid}, rid)

        # Phase C Session 5 (rFP §4.D.4): "stats" handler RETIRED —
        # rl_proxy.refresh_stats now SHM-direct via rl_state.bin
        # (Session 3 §4.B.5 publisher; _build_sage_stats still used
        # internally by the publisher + the SAGE_STATS bus broadcast).

        else:
            logger.warning("[RecorderWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[RecorderWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


def _build_sage_stats(recorder, gatekeeper) -> dict:
    """Snapshot of Sage subsystem state — used by SAGE_STATS broadcast +
    `action="stats"` query response. Cheap (no DB / network)."""
    buf = getattr(recorder, "buffer", None)
    buffer_len = len(buf) if buf is not None else 0
    storage = getattr(recorder, "storage", None)
    storage_len = len(storage) if storage is not None else 0
    return {
        "buffer_len": int(buffer_len),
        "storage_len": int(storage_len),
        "buffer_size": int(getattr(recorder, "buffer_size", 0)),
        "sovereignty_score": float(
            getattr(gatekeeper, "sovereignty_score", 0.0)),
        "decision_history_len": int(
            len(getattr(gatekeeper, "_decision_history", []) or [])),
    }


def _broadcast_sage_stats(send_queue, name: str, recorder, gatekeeper) -> None:
    """Periodic SAGE_STATS broadcast (60s cadence). Subscribers (RLProxy
    cache, dashboard) receive without per-call bus round-trips."""
    try:
        stats = _build_sage_stats(recorder, gatekeeper)
        _send_msg(send_queue, bus.SAGE_STATS, name, "all", stats)
    except Exception as e:
        logger.warning("[RecorderWorker] SAGE_STATS broadcast failed: %s", e)


def _handle_sage_record_transition(msg: dict, recorder, send_queue, name: str) -> None:
    """Microkernel v2 Layer 2 (2026-04-28): receive parent's
    SAGE_RECORD_TRANSITION, call local recorder.record_transition with the
    kwargs payload. record_transition is async; spin a fresh event loop per
    call (no awaits inside the body — pure sync work — so ~5ms overhead).
    """
    payload = msg.get("payload", {})
    try:
        import asyncio
        coro = recorder.record_transition(
            observation_vector=payload.get("observation_vector", []),
            action=payload.get("action", ""),
            reward=float(payload.get("reward", 0.0)),
            trauma_metadata=payload.get("trauma_metadata"),
            research_metadata=payload.get("research_metadata"),
            session_id=payload.get("session_id", "default_session"),
        )
        asyncio.run(coro)
    except Exception as e:
        logger.error(
            "[RecorderWorker] SAGE_RECORD_TRANSITION failed: %s", e, exc_info=True)


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    """Send a message via the send queue (worker→bus)."""
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_hcl.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)


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
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", {"rss_mb": round(rss_mb, 1)})
