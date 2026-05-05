"""
Outer Spirit Module Worker — 45D outer meta-awareness tensor at Schumann ×9.

Runs in its own supervised process (microkernel v2 Phase A.S8). Mirrors
inner spirit_worker §L1 fast-path shape, scaled to outer environmental tempo.

Schumann cadence:
  - SHM write tick: 70.47 Hz (Schumann ×9, 14.2 ms period)
  - Bus publish:    5s ± 10% jitter (inner spirit 0.383s × 13)

Observer Principle: reads outer_body_5d.bin + outer_mind_15d.bin from SHM
on each tick (body slowest, spirit fastest — G13 invariant preserved).

Sources: plugin snapshot only (soul_health, impulse_stats, assessment_stats,
sovereignty_ratio, uptime_ratio, social/memory stats from OUTER_SOURCES_SNAPSHOT).

Entry point: outer_spirit_worker_main(recv_queue, send_queue, name, config)
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
_OUTER_SPIRIT_SCHUMANN_HZ = 70.47
_OUTER_SPIRIT_TICK_PERIOD_S = 1.0 / _OUTER_SPIRIT_SCHUMANN_HZ  # ≈ 0.0142 s
_OUTER_SPIRIT_PUBLISH_INTERVAL_S = 5.0

_HEARTBEAT_INTERVAL_S = 10.0


def outer_spirit_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Outer Spirit worker subprocess."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[OuterSpiritWorker] Initializing 45D outer meta-awareness tensor...")

    _plugin_cache: dict = {}
    _plugin_cache_lock = threading.Lock()

    severity_multipliers = [1.0] * 5

    fast_stop_event = threading.Event()

    shm_writer_thread = None
    spirit_writer = None
    try:
        shm_writer_thread, spirit_writer = _start_outer_spirit_fast_path(
            config, fast_stop_event,
            _plugin_cache, _plugin_cache_lock,
            lambda: severity_multipliers,
        )
        logger.info("[OuterSpiritWorker] §L1 fast path ON: 70.47 Hz shm writer")
    except Exception as exc:
        logger.warning("[OuterSpiritWorker] §L1 fast-path init failed (%s); publish-only mode", exc)

    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})
    logger.info("[OuterSpiritWorker] outer spirit online")

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
            msg = recv_queue.get(timeout=1.0)
        except Empty:
            now = time.time()
            jitter = random.uniform(-0.1, 0.1) * _OUTER_SPIRIT_PUBLISH_INTERVAL_S
            if now - last_publish >= _OUTER_SPIRIT_PUBLISH_INTERVAL_S + jitter:
                tensor_45d, tensor_5d = _collect_tick(
                    _plugin_cache, _plugin_cache_lock, severity_multipliers,
                )
                _publish_outer_spirit_state(send_queue, name, tensor_45d, tensor_5d, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 120 == 0:
                    logger.info(
                        "[OuterSpiritWorker] publish #%d | 5d=[%s]",
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
            logger.info("[OuterSpiritWorker] Shutdown received")
            if shm_writer_thread:
                from titan_plugin.core.sensor_cache import stop_threads
                stop_threads(fast_stop_event, [shm_writer_thread], timeout_s=2.0)
            break

        elif msg_type == bus.OUTER_SOURCES_SNAPSHOT:
            payload = msg.get("payload") or {}
            with _plugin_cache_lock:
                _plugin_cache.update(payload)

        elif msg_type == bus.FILTER_DOWN:
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) >= 5:
                severity_multipliers = list(new_mult[:5])

        elif msg_type == bus.QUERY:
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")
            if action in ("get_tensor", "get_status"):
                tensor_45d, tensor_5d = _collect_tick(
                    _plugin_cache, _plugin_cache_lock, severity_multipliers,
                )
                _send_response(send_queue, name, src,
                               {"tensor_45d": tensor_45d, "tensor_5d": tensor_5d}, rid)

    logger.info("[OuterSpiritWorker] Exiting")


# ── Tick / Tensor Assembly ──────────────────────────────────────────

def _collect_tick(
    plugin_cache: dict, plugin_cache_lock: threading.Lock,
    severity_multipliers: list,
) -> tuple:
    """Compute outer spirit 45D + 5D tensors from SHM reads + plugin snapshot."""
    from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d, collect_outer_spirit_5d

    with plugin_cache_lock:
        snap = dict(plugin_cache)

    # Observer Principle: read body and mind from SHM
    outer_body = _read_outer_shm_5d("outer_body_5d")
    outer_mind = _read_outer_shm_15d("outer_mind_15d")

    soul_health = float(snap.get("soul_health") or 0.5)
    impulse = snap.get("impulse_stats") or {}
    assessment = snap.get("assessment_stats") or {}
    agency = snap.get("agency_stats") or {}
    mem_status = snap.get("memory_status") or {}

    total_impulses = impulse.get("total_fires", 0)
    total_assessed = assessment.get("total_assessed", 0)
    avg_score = assessment.get("average_score", 0.5)

    # 5D outer spirit
    tensor_5d = collect_outer_spirit_5d(
        outer_body=outer_body,
        outer_mind=outer_mind[:5] if len(outer_mind) >= 5 else outer_mind,
        soul_health=soul_health,
        total_impulses=total_impulses,
        total_assessed=total_assessed,
        avg_score=avg_score,
    )

    # 45D outer spirit (extended)
    total_actions = agency.get("total_actions", 0)
    failed_actions = agency.get("failed_actions", 0)
    success_rate = (total_actions - failed_actions) / max(1, total_actions)
    uptime = max(1.0, float(snap.get("uptime_seconds") or 1.0))
    uptime_ratio = min(1.0, uptime / max(1.0, uptime + 60.0))

    action_stats = {
        "total": total_actions,
        "success_count": total_actions - failed_actions,
        "success_rate": success_rate,
        "per_window": agency.get("actions_this_hour", 0),
        "per_hour": total_actions / max(0.01, uptime / 3600.0),
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
    guardian_stats = {
        "threats_detected": agency.get("threats_detected", 0),
        "rejections": agency.get("rejections", 0),
        "severity_avg": agency.get("threat_severity_avg", 0.0),
        "rejections_per_window": agency.get("rejections_this_hour", 0),
    }
    _sp = snap.get("social_perception_stats") or {}
    social_stats = {
        "interactions_per_window": mem_status.get("unique_interactors", 0),
        "sentiment_avg": _sp.get("sentiment_ema", 0.5),
        "social_connection": _sp.get("connection_ema", 0.0),
        "social_events_count": _sp.get("events_count", 0),
        "last_contagion": _sp.get("last_contagion"),
        "mean_conversation_quality": assessment.get("average_score", 0.5),
    }
    memory_stats = {
        "persistent_nodes": mem_status.get("persistent_count", 0),
        "growth_per_epoch": mem_status.get("growth_per_epoch", 0),
    }
    assessment_ext = {
        "mean_score": avg_score,
        "trend": assessment.get("trend", 0.0),
        "count": total_assessed,
        "score_variance": assessment.get("score_variance", 0.3),
    }

    tensor_45d = collect_outer_spirit_45d(
        current_5d=tensor_5d,
        outer_body=outer_body,
        outer_mind=outer_mind,
        action_stats=action_stats,
        creative_stats=creative_stats,
        guardian_stats=guardian_stats,
        sovereignty_ratio=agency.get("sovereignty_ratio", 0.0),
        uptime_ratio=uptime_ratio,
        social_stats=social_stats,
        memory_stats=memory_stats,
        assessment_stats=assessment_ext,
        history=snap.get("history"),
    )

    return [round(v, 4) for v in tensor_45d], tensor_5d


# Module-level reader cache; preserves StateRegistryReader._fallback_logged
# across the 70.47 Hz tick so a missing/torn slot logs once, not every tick.
_OUTER_BODY_READER = None
_OUTER_MIND_READER = None


def _read_outer_shm_5d(slot_name: str) -> list:
    """Read a 5D outer SHM slot. Returns [0.5]*5 on any failure (boot warmup)."""
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


def _read_outer_shm_15d(slot_name: str) -> list:
    """Read the outer_mind_15d SHM slot. Returns [0.5]*15 on any failure."""
    global _OUTER_MIND_READER
    try:
        if _OUTER_MIND_READER is None:
            from titan_plugin.core.state_registry import OUTER_MIND_15D, RegistryBank
            bank = RegistryBank(titan_id=None, config={})
            _OUTER_MIND_READER = bank.reader(OUTER_MIND_15D)
        arr = _OUTER_MIND_READER.read()
        if arr is not None and len(arr) == 15:
            return [float(v) for v in arr]
    except Exception:
        pass
    return [0.5] * 15


# ── §L1 Fast-path Setup ─────────────────────────────────────────────

def _start_outer_spirit_fast_path(
    config: dict,
    stop_event: threading.Event,
    plugin_cache: dict,
    plugin_cache_lock: threading.Lock,
    get_severity_multipliers,
):
    """Start 70.47 Hz SHM writer thread. Returns (shm_writer_thread, spirit_writer)."""
    from titan_plugin.core.sensor_cache import start_shm_writer_thread
    from titan_plugin.core.state_registry import OUTER_SPIRIT_45D, RegistryBank

    shm_bank = RegistryBank(titan_id=None, config=config)
    spirit_writer = None
    try:
        spirit_writer = shm_bank.writer(OUTER_SPIRIT_45D)
    except Exception:
        pass

    if spirit_writer is None:
        return None, None

    def _tick():
        import numpy as np
        tensor_45d, _ = _collect_tick(plugin_cache, plugin_cache_lock,
                                       get_severity_multipliers())
        arr = np.asarray(tensor_45d, dtype=np.float32)
        if arr.shape == (45,):
            spirit_writer.write(arr)

    shm_writer_thread = start_shm_writer_thread(
        _tick, _OUTER_SPIRIT_TICK_PERIOD_S, stop_event, "outer_spirit_shm_writer",
    )
    return shm_writer_thread, spirit_writer


# ── Bus Messaging ───────────────────────────────────────────────────

def _publish_outer_spirit_state(send_queue, name: str, tensor_45d: list,
                                 tensor_5d: list, severity_multipliers: list) -> None:
    center_dist = sum((v - 0.5) ** 2 for v in tensor_5d) ** 0.5
    payload = {
        "dims": 45,
        "values": tensor_5d,
        "values_45d": tensor_45d,
        "outer_spirit": tensor_5d,
        "outer_spirit_45d": tensor_45d,
        "delta": [round(v - 0.5, 4) for v in tensor_5d],
        "center_dist": round(center_dist, 4),
        "filter_down_multipliers": [round(m, 4) for m in severity_multipliers],
    }
    _send_msg(send_queue, bus.OUTER_SPIRIT_STATE, name, "all", payload)


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
