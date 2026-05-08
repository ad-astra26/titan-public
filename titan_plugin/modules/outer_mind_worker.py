"""
Outer Mind Module Worker — 15D outer creative/social tensor at Schumann ×3.

Runs in its own supervised process (microkernel v2 Phase A.S8). Mirrors
inner mind_worker.py shape, scaled to outer environmental tempo (×13).

Schumann cadence:
  - SHM write tick: 23.49 Hz (Schumann ×3, 42.6 ms period)
  - Bus publish:    15s ± 10% jitter (inner mind 1.15s × 13)

Observer Principle: reads outer_body_5d.bin from SHM on each tick.
Sources split:
  - Self-fetched: None (outer mind sources are all plugin-only)
  - Plugin-only:  art_count, audio_count, memory_status, uptime_seconds,
    agency_stats, assessment_stats, twin_state, anchor_state, bus_stats,
    social_perception_stats (via OUTER_SOURCES_SNAPSHOT every 10s)

Entry point: outer_mind_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import random
import sys
import threading
import time

from titan_plugin import bus

logger = logging.getLogger(__name__)

# ── Schumann constants (Phase A.S8 §1) ─────────────────────────────
_OUTER_MIND_SCHUMANN_HZ = 23.49
_OUTER_MIND_TICK_PERIOD_S = 1.0 / _OUTER_MIND_SCHUMANN_HZ   # ≈ 0.0426 s
_OUTER_MIND_PUBLISH_INTERVAL_S = 15.0

_HEARTBEAT_INTERVAL_S = 10.0


def outer_mind_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Outer Mind worker subprocess."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[OuterMindWorker] Initializing 15D outer creative tensor...")

    _plugin_cache: dict = {}
    _plugin_cache_lock = threading.Lock()

    severity_multipliers = [1.0] * 15

    fast_stop_event = threading.Event()

    shm_writer_thread = None
    refresh_threads = []
    mind_writer = None
    try:
        shm_writer_thread, mind_writer = _start_outer_mind_fast_path(
            config, fast_stop_event,
            _plugin_cache, _plugin_cache_lock,
            lambda: severity_multipliers,
        )
        logger.info("[OuterMindWorker] §L1 fast path ON: 23.49 Hz shm writer")
    except Exception as exc:
        logger.warning("[OuterMindWorker] §L1 fast-path init failed (%s); publish-only mode", exc)

    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})
    logger.info("[OuterMindWorker] outer mind online")

    last_publish = 0.0
    last_heartbeat = 0.0
    publish_count = 0

    from titan_plugin.core.readiness_reporter import trivial_reporter
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L1", send_queue=send_queue,
        save_state_cb=lambda: [],
    )

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            _send_heartbeat(send_queue, name)
            last_heartbeat = now

        msg = None
        try:
            msg = recv_queue.get(timeout=2.0)
        except Empty:
            now = time.time()
            jitter = random.uniform(-0.1, 0.1) * _OUTER_MIND_PUBLISH_INTERVAL_S
            if now - last_publish >= _OUTER_MIND_PUBLISH_INTERVAL_S + jitter:
                tensor_15d, tensor_5d = _collect_tick(
                    _plugin_cache, _plugin_cache_lock, severity_multipliers,
                )
                _publish_outer_mind_state(send_queue, name, tensor_15d, tensor_5d, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 60 == 0:
                    logger.info(
                        "[OuterMindWorker] publish #%d | 5d=[%s]",
                        publish_count,
                        ", ".join(f"{v:.3f}" for v in tensor_5d),
                    )
        except (KeyboardInterrupt, SystemExit):
            break

        if msg is None:
            continue

        msg_type = msg.get("type", "")

        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[OuterMindWorker] Shutdown received")
            if shm_writer_thread:
                from titan_plugin.core.sensor_cache import stop_threads
                _all = []
                if shm_writer_thread is not None:
                    _all.append(shm_writer_thread)
                stop_threads(fast_stop_event, _all, timeout_s=2.0)
            break

        elif msg_type == bus.OUTER_SOURCES_SNAPSHOT:
            payload = msg.get("payload") or {}
            with _plugin_cache_lock:
                _plugin_cache.update(payload)

        elif msg_type == bus.FILTER_DOWN:
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) >= 5:
                severity_multipliers = list(new_mult[:5]) + [1.0] * 10

        elif msg_type == bus.QUERY:
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")
            if action in ("get_tensor", "get_status"):
                tensor_15d, tensor_5d = _collect_tick(
                    _plugin_cache, _plugin_cache_lock, severity_multipliers,
                )
                _send_response(send_queue, name, src,
                               {"tensor_15d": tensor_15d, "tensor_5d": tensor_5d}, rid)

    logger.info("[OuterMindWorker] Exiting")


# ── Tick / Tensor Assembly ──────────────────────────────────────────

def _collect_tick(
    plugin_cache: dict, plugin_cache_lock: threading.Lock,
    severity_multipliers: list,
) -> tuple:
    """Compute outer mind 15D + 5D tensors from plugin snapshot + SHM body read."""
    from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d, collect_outer_mind_5d

    with plugin_cache_lock:
        snap = dict(plugin_cache)

    # Observer Principle: read outer_body from SHM (written by outer_body_worker)
    outer_body = _read_outer_body_shm()

    # 5D outer mind
    tensor_5d = collect_outer_mind_5d(
        art_count=int(snap.get("art_count_100") or 0),
        audio_count=int(snap.get("audio_count_100") or 0),
        memory_status=snap.get("memory_status") or {},
        uptime_seconds=float(snap.get("uptime_seconds") or 1.0),
    )

    # 15D outer mind (extended)
    agency = snap.get("agency_stats") or {}
    assessment = snap.get("assessment_stats") or {}
    total_actions = agency.get("total_actions", 0)
    failed_actions = agency.get("failed_actions", 0)
    success_rate = (total_actions - failed_actions) / max(1, total_actions)
    uptime = max(1.0, float(snap.get("uptime_seconds") or 1.0))
    actions_per_hour = total_actions / max(0.01, uptime / 3600.0)

    action_stats = {
        "total": total_actions,
        "success_count": total_actions - failed_actions,
        "success_rate": success_rate,
        "per_window": agency.get("actions_this_hour", 0),
        "per_hour": actions_per_hour,
        "failed_retry_rate": agency.get("failed_retry_rate", 0.0),
        "burst_frequency": agency.get("burst_frequency", 0.0),
        "error_rate": 1.0 - success_rate,
    }
    art_count = int(snap.get("art_count_500") or snap.get("art_count_100") or 0)
    audio_count = int(snap.get("audio_count_500") or snap.get("audio_count_100") or 0)
    creative_stats = {
        "total": art_count + audio_count,
        "art_count": art_count,
        "audio_count": audio_count,
        "per_window": agency.get("creative_this_hour", 0),
        "unique_types": min(2, (1 if art_count > 0 else 0) + (1 if audio_count > 0 else 0)),
        "mean_assessment": assessment.get("average_score", 0.5),
    }
    # Rejections + threats — Phase 1 (rFP §12.1, SPEC §23.8 willing[13]):
    # read from REAL producers (output_verifier + jailbreak_alerts), NOT
    # ghost agency keys. Sources:
    #   - output_verifier_stats.rejected_count (Titan rejecting own outputs)
    #   - jailbreak_alerts_stats.blocked_24h (Titan defending against attacks)
    #   - jailbreak_alerts_stats.threats_detected_24h (attacks faced)
    ov = snap.get("output_verifier_stats") or {}
    jb = snap.get("jailbreak_alerts_stats") or {}
    ov_rejected = int(ov.get("rejected_count", 0) or 0)
    jb_blocked_24h = int(jb.get("blocked_24h", 0) or 0)
    jb_threats_24h = int(jb.get("threats_detected_24h", 0) or 0)
    # Per-hour rate (rejections feed willing[13] protective_response).
    # output_verifier reports cumulative; subtract a 1h ago checkpoint
    # would require state — for Phase 1 use jb 24h / 24 + ov delta proxy.
    rejections_per_hour = (jb_blocked_24h / 24.0)  # smoothed 24h rate
    guardian_stats = {
        "threats_detected": jb_threats_24h,
        "rejections": ov_rejected + jb_blocked_24h,
        "severity_avg": float(jb.get("severity_avg_24h", 0.0) or 0.0),
        "rejections_per_window": rejections_per_hour,
    }
    _sp = snap.get("social_perception_stats") or {}
    mem_status = snap.get("memory_status") or {}
    social_stats = {
        "interactions_per_window": mem_status.get("unique_interactors", 0),
        "sentiment_avg": _sp.get("sentiment_ema", 0.5),
        "social_connection": _sp.get("connection_ema", 0.0),
        "social_events_count": _sp.get("events_count", 0),
        "last_contagion": _sp.get("last_contagion"),
        "mean_conversation_quality": assessment.get("average_score", 0.5),
    }
    # Phase 1 (rFP_trinity_130d_awakening §12.4): research_stats now uses
    # real seconds_since_last from events_teacher / language teacher
    # activity. Old hardcoded 300s/0.5 are gone — situational_awareness
    # reads events_teacher_stats directly via the new kwargs path.
    events = snap.get("events_teacher_stats") or {}
    last_event_seconds = (
        float(events.get("last_event_age_s", 3600.0))
        if isinstance(events, dict) and events.get("last_event_age_s") is not None
        else 3600.0
    )
    research_stats = {
        "queries": mem_status.get("research_nodes", 0),
        "seconds_since_last": last_event_seconds,
        "queries_per_window": 0,  # legacy — replaced by exploration_drive redesign
    }
    assessment_ext = {
        "mean_score": assessment.get("average_score", 0.5),
        "trend": assessment.get("trend", 0.0),
        "count": assessment.get("total_assessed", 0),
        "score_variance": assessment.get("score_variance", 0.3),
    }

    tensor_15d = collect_outer_mind_15d(
        current_5d=tensor_5d,
        action_stats=action_stats,
        creative_stats=creative_stats,
        guardian_stats=guardian_stats,
        social_stats=social_stats,
        research_stats=research_stats,
        assessment_stats=assessment_ext,
        body_state={"values": outer_body},
        twin_state=snap.get("twin_state"),
        anchor_state=snap.get("anchor_state"),
        bus_stats=snap.get("bus_stats"),
        # rFP §12.4 / SPEC §23.8 — rich producer kwargs
        cgn_stats=snap.get("cgn_stats"),
        meta_cgn_stats=snap.get("meta_cgn_stats"),
        language_stats=snap.get("language_stats"),
        memory_growth_metrics=snap.get("memory_growth_metrics"),
        events_teacher_stats=events if isinstance(events, dict) else None,
        knowledge_graph_stats=snap.get("knowledge_graph_stats"),
        social_x_gateway_stats=snap.get("social_x_gateway_stats"),
        uptime_seconds=float(snap.get("uptime_seconds") or 1.0),
    )

    return tensor_15d, tensor_5d


_OUTER_BODY_READER = None  # module-level cache; preserves StateRegistryReader._fallback_logged across ticks


def _read_outer_body_shm() -> list:
    """Read outer_body_5d SHM slot. Returns [0.5]*5 if unavailable (boot warmup)."""
    global _OUTER_BODY_READER
    try:
        if _OUTER_BODY_READER is None:
            from titan_plugin.core.state_registry import OUTER_BODY_5D, RegistryBank
            bank = RegistryBank(titan_id=None, config={})
            _OUTER_BODY_READER = bank.reader(OUTER_BODY_5D)
        arr = _OUTER_BODY_READER.read()
        if arr is not None and len(arr) == 5:
            return [float(v) for v in arr]
    except Exception:
        pass
    return [0.5] * 5


# ── §L1 Fast-path Setup ─────────────────────────────────────────────

def _start_outer_mind_fast_path(
    config: dict,
    stop_event: threading.Event,
    plugin_cache: dict,
    plugin_cache_lock: threading.Lock,
    get_severity_multipliers,
):
    """Start 23.49 Hz SHM writer thread. Returns (shm_writer_thread, mind_writer)."""
    from titan_plugin.core.sensor_cache import start_shm_writer_thread
    from titan_plugin.core.state_registry import OUTER_MIND_15D, RegistryBank

    shm_bank = RegistryBank(titan_id=None, config=config)
    mind_writer = None
    try:
        mind_writer = shm_bank.writer(OUTER_MIND_15D)
    except Exception:
        pass

    if mind_writer is None:
        return None, None

    def _tick():
        import numpy as np
        tensor_15d, _ = _collect_tick(plugin_cache, plugin_cache_lock,
                                       get_severity_multipliers())
        arr = np.asarray(tensor_15d, dtype=np.float32)
        if arr.shape == (15,):
            mind_writer.write(arr)

    shm_writer_thread = start_shm_writer_thread(
        _tick, _OUTER_MIND_TICK_PERIOD_S, stop_event, "outer_mind_shm_writer",
    )
    return shm_writer_thread, mind_writer


# ── Bus Messaging ───────────────────────────────────────────────────

def _publish_outer_mind_state(send_queue, name: str, tensor_15d: list,
                               tensor_5d: list, severity_multipliers: list) -> None:
    center_dist = sum((v - 0.5) ** 2 for v in tensor_5d) ** 0.5
    payload = {
        "dims": 15,
        "values": tensor_5d,
        "values_15d": tensor_15d,
        "outer_mind": tensor_5d,
        "outer_mind_15d": tensor_15d,
        "delta": [round(v - 0.5, 4) for v in tensor_5d],
        "center_dist": round(center_dist, 4),
        "filter_down_multipliers": [round(m, 4) for m in severity_multipliers[:5]],
    }
    _send_msg(send_queue, bus.OUTER_MIND_STATE, name, "all", payload)


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


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid=None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid) -> None:
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)
