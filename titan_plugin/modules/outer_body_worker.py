"""
Outer Body Module Worker — 5D outer somatic tensor at Schumann fundamental.

Runs in its own supervised process (microkernel v2 Phase A.S8). Mirrors
inner body_worker.py shape exactly, scaled to outer environmental tempo (×13).

Schumann cadence:
  - SHM write tick: 7.83 Hz (Schumann ×1, 127.7 ms period)
  - Bus publish:    45s ± 10% jitter (inner body 3.45s × 13)

Sources split:
  - Self-fetched: anchor_state, sol_balance, system_sensor_stats,
    network_monitor_stats, tx_latency_stats, block_delta_stats, twin_state
  - Plugin-only:  agency_stats, assessment_stats, helper_statuses, bus_stats,
    memory_status, soul_health, impulse_stats, llm_avg_latency, etc.
    (delivered via OUTER_SOURCES_SNAPSHOT every 10s from plugin)

Entry point: outer_body_worker_main(recv_queue, send_queue, name, config)
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
_OUTER_BODY_SCHUMANN_HZ = 7.83
_OUTER_BODY_TICK_PERIOD_S = 1.0 / _OUTER_BODY_SCHUMANN_HZ   # ≈ 0.1277 s
_OUTER_BODY_PUBLISH_INTERVAL_S = 45.0

# ── Per-source refresh cadences (outer environmental tempo) ─────────
_OUTER_BODY_REFRESH_PERIODS_S = {
    "sol_balance":           30.0,  # file read; balance changes slowly
    "anchor_state":          30.0,  # file read; anchoring cadence ~minutes
    "system_sensor":          2.0,  # psutil — cheap, 2s for outer tempo
    "network_monitor":       30.0,  # socket probes; amortize cost
    "block_delta":           15.0,  # timechain_v2; block ~10-15s
    "tx_latency":            15.0,  # timechain_v2; latency window 15s
    "twin_state":            60.0,  # HTTP poll to twin; outer-tempo safe
}

# Heartbeat interval (parent must receive within 60s)
_HEARTBEAT_INTERVAL_S = 10.0


def outer_body_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Outer Body worker subprocess."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[OuterBodyWorker] Initializing 5D outer somatic tensor...")

    # Plugin-source cache (populated by OUTER_SOURCES_SNAPSHOT messages)
    _plugin_cache: dict = {}
    _plugin_cache_lock = threading.Lock()
    _plugin_cache_ts: float = 0.0

    # FILTER_DOWN severity multipliers from Spirit (1.0 = no modulation)
    severity_multipliers = [1.0] * 5

    # Self-fetched sensor cache (populated by background refresh threads)
    _self_cache: dict = {}
    _self_cache_lock = threading.Lock()

    # Last published outer_body for somatosensation self-reference
    _last_outer_body = [0.5] * 5

    fast_stop_event = threading.Event()

    # Start §L1 fast-path: per-source refresh threads + 7.83 Hz SHM writer
    shm_writer_thread = None
    refresh_threads = []
    body_5d_writer = None
    try:
        refresh_threads, shm_writer_thread, body_5d_writer = _start_outer_body_fast_path(
            config, fast_stop_event,
            _self_cache, _self_cache_lock,
            _plugin_cache, _plugin_cache_lock,
            lambda: _last_outer_body,
            lambda: severity_multipliers,
        )
        logger.info("[OuterBodyWorker] §L1 fast path ON: refresh threads + 7.83 Hz shm writer")
    except Exception as exc:
        logger.warning("[OuterBodyWorker] §L1 fast-path init failed (%s); running publish-only", exc)

    # Signal ready
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})
    logger.info("[OuterBodyWorker] outer body online")

    last_publish = 0.0
    last_heartbeat = 0.0
    publish_count = 0

    # Microkernel v2 Phase B.1 readiness reporter
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
            jitter = random.uniform(-0.1, 0.1) * _OUTER_BODY_PUBLISH_INTERVAL_S
            if now - last_publish >= _OUTER_BODY_PUBLISH_INTERVAL_S + jitter:
                tensor = _collect_tick(
                    _self_cache, _self_cache_lock,
                    _plugin_cache, _plugin_cache_lock,
                    _last_outer_body, severity_multipliers,
                )
                _last_outer_body = tensor
                _publish_outer_body_state(send_queue, name, tensor, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 20 == 0:
                    logger.info(
                        "[OuterBodyWorker] publish #%d | tensor=[%s]",
                        publish_count,
                        ", ".join(f"{v:.3f}" for v in tensor),
                    )
        except (KeyboardInterrupt, SystemExit):
            break

        if msg is None:
            continue

        msg_type = msg.get("type", "")

        # Phase B.1 shadow swap dispatch
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # Phase B.2.1 supervision-transfer dispatch
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[OuterBodyWorker] Shutdown received")
            if refresh_threads or shm_writer_thread:
                from titan_plugin.core.sensor_cache import stop_threads
                _all = list(refresh_threads)
                if shm_writer_thread is not None:
                    _all.append(shm_writer_thread)
                stop_threads(fast_stop_event, _all, timeout_s=2.0)
            break

        elif msg_type == bus.OUTER_SOURCES_SNAPSHOT:
            payload = msg.get("payload") or {}
            with _plugin_cache_lock:
                _plugin_cache.update(payload)
                _plugin_cache_ts = time.time()

        elif msg_type == bus.FILTER_DOWN:
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) >= 5:
                severity_multipliers = list(new_mult[:5])
                logger.debug("[OuterBodyWorker] FILTER_DOWN: %s",
                             [round(m, 2) for m in severity_multipliers])

        elif msg_type == bus.QUERY:
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")
            if action in ("get_tensor", "get_status"):
                tensor = _collect_tick(
                    _self_cache, _self_cache_lock,
                    _plugin_cache, _plugin_cache_lock,
                    _last_outer_body, severity_multipliers,
                )
                _send_response(send_queue, name, src, {"tensor": tensor}, rid)

    logger.info("[OuterBodyWorker] Exiting")


# ── Tick / Tensor Assembly ──────────────────────────────────────────

def _collect_tick(
    self_cache: dict, self_cache_lock: threading.Lock,
    plugin_cache: dict, plugin_cache_lock: threading.Lock,
    last_outer_body: list,
    severity_multipliers: list,
) -> list:
    """Assemble sources dict and compute outer body 5D tensor."""
    from titan_plugin.logic.outer_body_tensor import collect_outer_body_5d

    with self_cache_lock:
        self_snap = dict(self_cache)
    with plugin_cache_lock:
        plugin_snap = dict(plugin_cache)

    sources = {**self_snap, **plugin_snap}
    tensor = collect_outer_body_5d(sources, last_outer_body=last_outer_body)

    # Apply FILTER_DOWN multipliers (scale each dim toward center 0.5)
    modulated = []
    for i, val in enumerate(tensor):
        m = severity_multipliers[i] if i < len(severity_multipliers) else 1.0
        modulated.append(round(0.5 + (val - 0.5) * m, 4))
    return modulated


# ── §L1 Fast-path Setup ─────────────────────────────────────────────

def _start_outer_body_fast_path(
    config: dict,
    stop_event: threading.Event,
    self_cache: dict,
    self_cache_lock: threading.Lock,
    plugin_cache: dict,
    plugin_cache_lock: threading.Lock,
    get_last_body,
    get_severity_multipliers,
):
    """
    Start per-source refresh threads + 7.83 Hz SHM writer thread.

    Returns (refresh_threads, shm_writer_thread, body_5d_writer).
    """
    from titan_plugin.core.sensor_cache import (
        RefreshSpec, SensorCache, start_refresh_threads, start_shm_writer_thread,
    )
    from titan_plugin.core.state_registry import OUTER_BODY_5D, RegistryBank

    def _refresh_sol_balance():
        return {"value": _fetch_sol_balance()}

    def _refresh_anchor_state():
        return {"value": _fetch_anchor_state()}

    def _refresh_system_sensor():
        return {"value": _fetch_system_sensor()}

    def _refresh_network_monitor():
        return {"value": _fetch_network_monitor()}

    def _refresh_block_delta():
        return {"value": _fetch_block_delta()}

    def _refresh_tx_latency():
        return {"value": _fetch_tx_latency()}

    def _refresh_twin_state():
        return {"value": _fetch_twin_state(config)}

    # Synchronous warmup so SHM writer never reads cold cache
    initial = {
        "sol_balance":       {"value": _fetch_sol_balance()},
        "anchor_state":      {"value": _fetch_anchor_state()},
        "system_sensor":     {"value": _fetch_system_sensor()},
        "network_monitor":   {"value": _fetch_network_monitor()},
        "block_delta":       {"value": _fetch_block_delta()},
        "tx_latency":        {"value": _fetch_tx_latency()},
        "twin_state":        {"value": _fetch_twin_state(config)},
    }
    # Populate self_cache with initial values
    with self_cache_lock:
        for k, v in initial.items():
            self_cache[k] = v["value"]

    _sc = SensorCache(initial=initial)

    def _update_self_cache(key, val):
        with self_cache_lock:
            self_cache[key] = val

    specs = [
        RefreshSpec("sol_balance",     _refresh_sol_balance,
                    _OUTER_BODY_REFRESH_PERIODS_S["sol_balance"]),
        RefreshSpec("anchor_state",    _refresh_anchor_state,
                    _OUTER_BODY_REFRESH_PERIODS_S["anchor_state"]),
        RefreshSpec("system_sensor",   _refresh_system_sensor,
                    _OUTER_BODY_REFRESH_PERIODS_S["system_sensor"]),
        RefreshSpec("network_monitor", _refresh_network_monitor,
                    _OUTER_BODY_REFRESH_PERIODS_S["network_monitor"]),
        RefreshSpec("block_delta",     _refresh_block_delta,
                    _OUTER_BODY_REFRESH_PERIODS_S["block_delta"]),
        RefreshSpec("tx_latency",      _refresh_tx_latency,
                    _OUTER_BODY_REFRESH_PERIODS_S["tx_latency"]),
        RefreshSpec("twin_state",      _refresh_twin_state,
                    _OUTER_BODY_REFRESH_PERIODS_S["twin_state"]),
    ]

    # Wrap each refresh to also update the flat self_cache used by tick
    def _make_wrapped_spec(spec):
        orig_fn = spec.refresh_fn
        key = spec.name
        def wrapped():
            result = orig_fn()
            with self_cache_lock:
                self_cache[key] = result.get("value")
            return result
        return RefreshSpec(key, wrapped, spec.period_s)

    wrapped_specs = [_make_wrapped_spec(s) for s in specs]
    refresh_threads = start_refresh_threads(
        wrapped_specs, _sc, stop_event, thread_name_prefix="outer_body_refresh",
    )

    # SHM writer at 7.83 Hz
    shm_bank = RegistryBank(titan_id=None, config=config)
    body_5d_writer = None
    shm_writer_thread = None

    if shm_bank.is_enabled(OUTER_BODY_5D):
        body_5d_writer = shm_bank.writer(OUTER_BODY_5D)
    else:
        # Always-enabled; create unconditionally
        try:
            body_5d_writer = shm_bank.writer(OUTER_BODY_5D)
        except Exception:
            pass

    if body_5d_writer is not None:
        def _tick():
            from titan_plugin.logic.outer_body_tensor import collect_outer_body_5d
            import numpy as np
            with self_cache_lock:
                self_snap = dict(self_cache)
            with plugin_cache_lock:
                plugin_snap = dict(plugin_cache)
            sources = {**self_snap, **plugin_snap}
            mult = get_severity_multipliers()
            tensor = collect_outer_body_5d(sources, last_outer_body=get_last_body())
            modulated = [round(0.5 + (v - 0.5) * (mult[i] if i < len(mult) else 1.0), 4)
                         for i, v in enumerate(tensor)]
            arr = np.asarray(modulated, dtype=np.float32)
            if arr.shape == (5,):
                body_5d_writer.write(arr)

        shm_writer_thread = start_shm_writer_thread(
            _tick, _OUTER_BODY_TICK_PERIOD_S, stop_event, "outer_body_shm_writer",
        )

    return refresh_threads, shm_writer_thread, body_5d_writer


# ── Self-fetched Sensor Helpers ─────────────────────────────────────

def _fetch_sol_balance() -> float:
    try:
        balance_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "last_balance.txt")
        if os.path.exists(balance_file):
            with open(balance_file) as f:
                return float(f.read().strip())
    except Exception:
        pass
    return 0.5


def _fetch_anchor_state() -> dict:
    try:
        import json
        anchor_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
        if os.path.exists(anchor_file):
            with open(anchor_file) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _fetch_system_sensor() -> dict:
    try:
        from titan_plugin.utils.system_sensor import get_system_sensor_stats
        return get_system_sensor_stats()
    except Exception:
        return {}


def _fetch_network_monitor() -> dict:
    try:
        from titan_plugin.utils.network_monitor import get_network_monitor_stats
        return get_network_monitor_stats()
    except Exception:
        return {}


def _fetch_block_delta() -> dict:
    try:
        from titan_plugin.utils.timechain_v2 import get_block_delta_stats
        return get_block_delta_stats()
    except Exception:
        return {}


def _fetch_tx_latency() -> dict:
    try:
        from titan_plugin.utils.timechain_v2 import get_tx_latency_stats
        return get_tx_latency_stats()
    except Exception:
        return {}


def _fetch_twin_state(config: dict) -> dict:
    try:
        import httpx
        twin_host = (config.get("network", {}) or {}).get("twin_host", "")
        if not twin_host:
            return {}
        resp = httpx.get(f"http://{twin_host}:7777/v4/neuromod", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# ── Bus Messaging ───────────────────────────────────────────────────

def _publish_outer_body_state(send_queue, name: str, tensor: list,
                               severity_multipliers: list) -> None:
    center_dist = sum((v - 0.5) ** 2 for v in tensor) ** 0.5
    payload = {
        "dims": 5,
        "values": tensor,
        "delta": [round(v - 0.5, 4) for v in tensor],
        "center_dist": round(center_dist, 4),
        "filter_down_multipliers": [round(m, 4) for m in severity_multipliers],
    }
    _send_msg(send_queue, bus.OUTER_BODY_STATE, name, "all", payload)


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
