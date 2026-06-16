"""
Body Module Worker — 5DT somatic sensor array with urgency-aware tensor.

Runs in its own supervised process, collecting system telemetry and
producing a 5-dimensional Body tensor that reflects Titan's physical
state on the infrastructure he lives on.

Each sensor output is categorized as INFO/WARNING/CRITICAL with
exponential weighting. A velocity component detects sudden changes
(acute pain vs chronic stress).

5DT Body Senses (Physical + Digital Topology Synthesis):
  [0] Interoception — SOL balance / metabolic energy
  [1] Proprioception — BODY TOPOLOGY self-sensing (sphere radius, volume)
  [2] Somatosensation — system resources (CPU, RAM, disk, swap)
  [3] Entropy — disorder + network health (errors, connectivity)
  [4] Thermal — heat synthesis: physical (CPU load) × digital (topology activity)

S7 (microkernel v2 §A.7 / §L1, 2026-04-26): when
``microkernel.shm_body_fast_enabled`` is true, this worker runs the
3-layer Trinity Daemon Internal Design pattern: per-sense background
refresh threads at native cadences populate a SensorCache; the tick
layer reads cache only and writes the 5D tensor to /dev/shm at the
Schumann fundamental rate (7.83 Hz, 127.7 ms period). Pre-S7 inline
sensor calls (524 ms/call worst-case) drop to ~100 μs/tick.

Default flag-OFF preserves byte-identical pre-S7 behavior — no
threads start, sensor calls happen inline as before.

Entry point: body_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import os
import sys
import threading
import time
from collections import deque
from enum import IntEnum
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel; mirrored to
# `module_body_state.bin` via ModuleStateWriter so titan_hcl's 1Hz SHM
# poll + the orchestrator's MODULE_PROBE_REQUEST dispatcher see real
# liveness. Flipped True once sensors + (optional) fast-path are warm.
# Under `l0_rust_enabled=true` body_worker runs as a SHADOW providing
# sensor_cache_inner_body.bin input — the SHM slot tracks the Python
# shadow lifecycle, not the Rust daemon's.
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace

# Module-level state writer — populated at entry; consulted by the
# inline heartbeat helper so it can publish state_writer.heartbeat() on
# the same cadence as the bus MODULE_HEARTBEAT (§11.I.5 dual path).
_state_writer = None  # type: ignore[assignment]


# ── S7 Schumann shm writer cadence (microkernel v2 §A.7 / §L1) ─────
_BODY_SCHUMANN_HZ = 7.83  # Schumann fundamental
_BODY_TICK_PERIOD_S = 1.0 / _BODY_SCHUMANN_HZ  # ≈ 0.1277 s

# ── S7 per-sense refresh cadences (matches sensor's natural timescale) ──
# - interoception:  2 file reads (anchor + balance); 5s captures changes
# - proprioception: 1 file read (body_topology written by spirit clock);
#                   2s aligned with spirit clock tick (~1.15s)
# - somatosensation: psutil.cpu_percent(interval=0.5) — fastest sense; 1Hz
# - entropy:        4 socket connects (≤8s) + log tail; 10s amortizes cost
# - thermal:        loadavg + topology read + circadian math; 2s
_BODY_REFRESH_PERIODS_S = {
    "interoception": 5.0,
    "proprioception": 2.0,
    "somatosensation": 1.0,
    "entropy": 10.0,
    "thermal": 2.0,
}

# ── Category weights (exponential) ──────────────────────────────────

class Severity(IntEnum):
    INFO = 1
    WARNING = 3
    CRITICAL = 10


# ── Sensor history for velocity calculation ─────────────────────────

_HISTORY_SIZE = 30  # ~5 minutes at 10s intervals
_VELOCITY_WINDOW = 6  # compare last 6 readings (~1 min) for rate of change


_BODY_STATE_FILENAME = "body_state.json"


def _body_state_path(config: dict) -> str:
    # Same data_dir resolution mind_worker uses (config-level "data_dir").
    data_dir = config.get("data_dir", "./data")
    return os.path.join(data_dir, _BODY_STATE_FILENAME)


def _load_body_state(config: dict):
    """Restore (severity_multipliers, focus_nudges) from disk, else (None, None)."""
    import json
    try:
        with open(_body_state_path(config)) as f:
            d = json.load(f)
        sm = d.get("severity_multipliers")
        fn = d.get("focus_nudges")
        sm = [float(x) for x in sm] if isinstance(sm, list) and len(sm) == 5 else None
        fn = [float(x) for x in fn] if isinstance(fn, list) and len(fn) == 5 else None
        return sm, fn
    except Exception:
        return None, None


def _save_body_state(config: dict, severity_multipliers, focus_nudges) -> None:
    """Atomically persist Body's Spirit-learned FILTER_DOWN weights + focus
    nudges (§11.H.2 tmp+os.replace). AUDIT §C fix (rFP §P2): these were NEVER
    persisted (the only save fn, `_b1_save_state`, is the B1 readiness-reporter
    callback that returns [] — not the persistence path), so they reset to
    [1.0]*5 / [0.0]*5 on every respawn, losing Spirit-learned modulation until
    the next EVENT-ONLY FILTER_DOWN arrives. The sensor `history` deques
    re-accumulate from live sensors (~5min) → intentionally not persisted."""
    import json
    path = _body_state_path(config)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"severity_multipliers": list(severity_multipliers),
                       "focus_nudges": list(focus_nudges)}, f)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("[BodyWorker] state save failed: %s", e)


@with_error_envelope(module_name="body", subsystem="entry", severity=_phase11_sev.FATAL)
def body_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Body module process."""
    from queue import Empty

    # Phase 11 §11.I.5 — reset module-level readiness flags (fork inherits
    # parent's True; spawn gets fresh False; explicit reset covers both).
    global _WORKER_READY, _BOOT_DEADLINE, _state_writer
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()
    _state_writer = None

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[BodyWorker] Initializing 5DT somatic sensors...")

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 single-writer) ──
    # Constructed BEFORE the (possibly slow) fast-path thread startup so
    # the slot publishes state="starting" immediately. titan_hcl's 1Hz
    # SHM poll sees the worker is alive while it warms.
    try:
        from titan_hcl.core.module_state import (
            BootPriority, ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L1",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[BodyWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot will be absent): %s", _sw_err)

    # Sensor history for velocity tracking (deque per sense)
    history = {
        "interoception": deque(maxlen=_HISTORY_SIZE),
        "proprioception": deque(maxlen=_HISTORY_SIZE),
        "somatosensation": deque(maxlen=_HISTORY_SIZE),
        "entropy": deque(maxlen=_HISTORY_SIZE),
        "thermal": deque(maxlen=_HISTORY_SIZE),
    }

    # Thresholds from config (with sensible defaults)
    thresholds = _load_thresholds(config)

    # FILTER_DOWN severity multipliers (learned from Spirit via RL gradients)
    # Start at 1.0 = no modulation. Updated when FILTER_DOWN messages arrive.
    severity_multipliers = [1.0] * 5

    # FOCUS nudges from Spirit PID controller (suggestions, not overrides)
    focus_nudges = [0.0] * 5

    # AUDIT §C fix (rFP §P2): restore Spirit-learned FILTER_DOWN weights + focus
    # nudges from disk so they survive hot-reload / kill-respawn instead of
    # resetting to defaults until the next event-only FILTER_DOWN.
    _saved_sm, _saved_fn = _load_body_state(config)
    if _saved_sm is not None:
        severity_multipliers = _saved_sm
    if _saved_fn is not None:
        focus_nudges = _saved_fn

    # ── S7: §L1 fast-path setup (flag-gated) ──────────────────────
    # When microkernel.shm_body_fast_enabled is true, start per-sense
    # refresh threads + 7.83 Hz shm writer thread. Tick layer reads
    # from cache (no I/O on hot path).
    sensor_cache = None
    refresh_threads = []
    shm_writer_thread = None
    shm_bank = None
    body_5d_writer = None
    fast_stop_event = threading.Event()

    fast_enabled = _read_flag(config, "microkernel.shm_body_fast_enabled", False)
    l0_rust_enabled = _read_flag(config, "microkernel.l0_rust_enabled", False)
    # Phase C C-S5 closure 2026-05-08: under l0_rust_enabled=true the
    # Rust titan-inner-body-rs daemon owns inner_body_5d.bin output but
    # NEEDS sensor_cache_inner_body.bin as input. body_worker still
    # provides the sensor compute + 7.83 Hz cadence; _start_fast_path's
    # internal if/elif redirects the write target from inner_body_5d.bin
    # (Phase A+B) to sensor_cache_inner_body.bin (Phase C). Either flag
    # enables _start_fast_path; the slot identity is decided inside.
    if fast_enabled or l0_rust_enabled:
        try:
            sensor_cache, refresh_threads, shm_bank, body_5d_writer, shm_writer_thread = (
                _start_fast_path(thresholds, config, fast_stop_event,
                                 lambda: (severity_multipliers, focus_nudges))
            )
            if fast_enabled:
                logger.info(
                    "[BodyWorker] §L1 fast path ON: 5 refresh threads + "
                    "7.83 Hz inner_body_5d.bin writer (Phase A+B)"
                )
            else:
                logger.info(
                    "[BodyWorker] Phase C path ON: 5 refresh threads + "
                    "7.83 Hz sensor_cache_inner_body.bin writer "
                    "(l0_rust_enabled=true; Rust titan-inner-body-rs owns "
                    "inner_body_5d.bin output)"
                )
        except Exception as exc:
            logger.warning(
                "[BodyWorker] fast-path init failed (%s); falling back to inline senses",
                exc,
            )
            sensor_cache = None
            refresh_threads = []
            shm_writer_thread = None

    # ── Phase 11 §11.I.2 — SHM slot transition starting → booted ─────
    # MODULE_READY bus emit DELETED per D-SPEC-141 / v1.65.0 locked D2.
    # body_worker is a shadow under l0_rust_enabled=true (Rust owns
    # inner_body_5d.bin), but the SHM slot still tracks the Python
    # shadow's lifecycle so the orchestrator can probe it independently.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[BodyWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[BodyWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info("[BodyWorker] 5DT somatic sensors online (fast=%s)", bool(sensor_cache))

    # Phase C Session 4 (rFP §4.B.6) — SHM-direct body_state.bin publisher.
    # Closes the deadlock-prone sync bus.request path for body_proxy
    # method get_body_details. The Rust-owned inner_body_5d.bin slot
    # carries the bare 5D tensor; this slot carries the queryable
    # details/urgency/focus_nudges metadata.
    def _body_state_fetcher():
        # Cheap recompute on each tick — uses the same cache the QUERY
        # path uses; no new I/O on hot path beyond what _collect_body_tensor
        # already does (cache-backed when sensor_cache is populated).
        tensor, details = _collect_body_tensor(
            history, thresholds, severity_multipliers, focus_nudges,
            cache=sensor_cache)
        return {
            "tensor": tensor,
            "details": details,
            "history_size": {k: len(v) for k, v in history.items()},
            "severity_multipliers": list(severity_multipliers),
            "focus_nudges": list(focus_nudges),
        }

    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.body_state_publisher import BodyStatePublisher
        from titan_hcl.logic.worker_publisher_runner import (
            run_worker_publisher)
        _body_state_publisher = BodyStatePublisher(titan_id=resolve_titan_id())
        run_worker_publisher(
            publisher=_body_state_publisher,
            state_fetcher=_body_state_fetcher,
            worker_name="body_worker",
            cadence_s=1.0,
        )
    except Exception as _pub_init_err:
        logger.error(
            "[BodyWorker] body_state SHM publisher BOOT FAILED — "
            "consumers fall back to sync bus.request: %s",
            _pub_init_err, exc_info=True)

    last_publish = 0.0
    publish_interval = 3.45   # Body = Schumann/27 (0.29 Hz) — Earth resonance
    last_heartbeat = 0.0
    publish_count = 0  # observability — periodic summary cadence

    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L1", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        # Heartbeat on every iteration (not just Empty timeout)
        now = time.time()
        if now - last_heartbeat >= 10.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = now

        msg = None
        try:
            msg = recv_queue.get(timeout=2.0)
        except Empty:
            now = time.time()
            if now - last_publish >= publish_interval:
                tensor, details = _collect_body_tensor(history, thresholds,
                                                       severity_multipliers, focus_nudges,
                                                       cache=sensor_cache)
                _publish_body_state(send_queue, name, tensor, details, severity_multipliers)
                last_publish = now
                publish_count += 1
                if publish_count % 100 == 0:
                    logger.info(
                        "[BodyWorker] summary @publish=%d | tensor=[%s] | mults=[%s] | nudges=[%s]",
                        publish_count,
                        ", ".join(f"{t:.2f}" for t in tensor),
                        ", ".join(f"{m:.2f}" for m in severity_multipliers),
                        ", ".join(f"{n:+.2f}" for n in focus_nudges),
                    )
        except (KeyboardInterrupt, SystemExit):
            break

        if msg is None:
            continue

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

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────────
        if msg_type == "MODULE_PROBE_REQUEST":
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    state_writer=_state_writer, probe_fn=None,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[BodyWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[BodyWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            # AUDIT §C fix (rFP §P2): flush the current Spirit-learned weights +
            # focus nudges before exit so a hot-reload / kill-respawn restores
            # them. The locals here hold the live values (severity_multipliers is
            # reassigned by FILTER_DOWN; focus_nudges is mutated in place).
            _save_body_state(config, severity_multipliers, focus_nudges)
            # Stop S7 fast-path threads cleanly so they don't keep
            # writing shm during/after subprocess teardown.
            if refresh_threads or shm_writer_thread:
                from titan_hcl.core.sensor_cache import stop_threads
                _all = list(refresh_threads)
                if shm_writer_thread is not None:
                    _all.append(shm_writer_thread)
                stop_threads(fast_stop_event, _all, timeout_s=2.0)
            break

        # Receive FILTER_DOWN severity multipliers from Spirit
        if msg_type == bus.FILTER_DOWN:
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) == 5:
                severity_multipliers = new_mult
                logger.info("[BodyWorker] FILTER_DOWN received: %s",
                            [round(m, 2) for m in severity_multipliers])

        # Phase 10K (rFP §3G / audit §5.3): the FOCUS_NUDGE bus-consumer branch
        # was removed — its sole producer (spirit_loop._run_focus) is deleted and
        # the live focus path is focus_input.bin SHM (FocusPIDPublisher @7.83 Hz),
        # which the Rust trinity daemons read + apply as the cascade directly. The
        # bus message never arrives, so the handler was unreachable dead code.

        # Receive conversation stimulus → compute Body reflex Intuition
        elif msg_type == bus.CONVERSATION_STIMULUS:
            stimulus = msg.get("payload", {})
            tensor, _ = _collect_body_tensor(history, thresholds,
                                              severity_multipliers, focus_nudges,
                                              cache=sensor_cache)
            signals = _compute_body_reflex_intuition(stimulus, tensor)
            # REFLEX_SIGNAL broadcast removed — no consumer exists (audit 2026-03-26)

        # Receive Interface input signals (human conversation → somatic impact)
        elif msg_type == bus.INTERFACE_INPUT:
            iface = msg.get("payload", {})
            # Intensity maps to Thermal (load proxy): conversation energy = social load
            intensity = iface.get("intensity", 0.0)
            if intensity > 0.3:
                # Nudge thermal reading: high intensity = higher load feeling
                focus_nudges[4] = focus_nudges[4] + intensity * 0.05
                focus_nudges[4] = min(0.5, focus_nudges[4])
            logger.debug("[BodyWorker] INTERFACE_INPUT absorbed: intensity=%.2f", intensity)

        elif msg_type == bus.QUERY:
            from titan_hcl.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")

            # Phase C Session 5 (rFP §4.D.3): get_tensor / get_status /
            # get_details handlers RETIRED. Proxy consumers migrated to
            # SHM-direct via inner_body_5d.bin (Rust) + body_state.bin
            # (Session 4 §4.C.3). Static caller graph confirms zero
            # remaining callers.
            pass

    logger.info("[BodyWorker] Exiting")


# ── Sensor Collection ───────────────────────────────────────────────

def _collect_body_tensor(history: dict, thresholds: dict,
                         severity_multipliers: list | None = None,
                         focus_nudges: list | None = None,
                         cache=None) -> tuple[list, dict]:
    """
    Collect all 5 body senses, categorize, weight, compute velocity.

    FILTER_DOWN multipliers modulate the urgency formula:
      urgency = raw * category * multiplier / CRITICAL + velocity_contrib
    FOCUS nudges apply a gentle bias toward center after scoring.

    cache (S7 §L1):
      - None  → call _sense_X(thresholds) inline (pre-S7 byte-identical)
      - SensorCache → read from cache (populated by background refresh
        threads); fall back to inline call if a sense is missing from
        the cache (defensive — should only happen pre-warmup)

    Returns:
        (tensor, details) where tensor is [5 floats, 0.0-1.0 normalized]
        and details is a dict with per-sense breakdown.
    """
    if severity_multipliers is None:
        severity_multipliers = [1.0] * 5
    if focus_nudges is None:
        focus_nudges = [0.0] * 5

    readings = {}
    sense_fns = {
        "interoception": _sense_interoception,
        "proprioception": _sense_proprioception,
        "somatosensation": _sense_somatosensation,
        "entropy": _sense_entropy,
        "thermal": _sense_thermal,
    }

    if cache is not None:
        # §L1 fast path — read from cache populated by per-sense
        # refresh threads. Fallback to inline call only if cache is
        # cold for a particular sense (defensive — boot warmup is
        # done synchronously so this should never fire in steady state).
        for sense_name, sense_fn in sense_fns.items():
            cached = cache.get(sense_name)
            if cached is not None and "value" in cached and "severity" in cached:
                readings[sense_name] = {
                    "value": cached["value"], "severity": cached["severity"],
                }
            else:
                readings[sense_name] = sense_fn(thresholds)
    else:
        # Pre-S7 inline path — synchronous sense calls.
        for sense_name, sense_fn in sense_fns.items():
            readings[sense_name] = sense_fn(thresholds)

    # Build tensor with urgency weighting + velocity + FILTER_DOWN + FOCUS
    tensor = []
    details = {}

    sense_names = ["interoception", "proprioception", "somatosensation", "entropy", "thermal"]
    for idx, sense_name in enumerate(sense_names):
        reading = readings[sense_name]
        raw_value = reading["value"]
        severity = reading["severity"]
        category_weight = float(severity)
        multiplier = severity_multipliers[idx]

        # Record in history
        history[sense_name].append({"ts": time.time(), "value": raw_value, "severity": severity})

        # Calculate velocity (rate of change over recent window)
        velocity = _calculate_velocity(history[sense_name])

        # Urgency-weighted score with FILTER_DOWN multiplier:
        # Higher multiplier = this sense matters more (RL learned this hurts)
        # Formula: raw * category * multiplier / CRITICAL + velocity_contrib
        urgency = min(1.0, (raw_value * category_weight * multiplier / Severity.CRITICAL)
                       + abs(velocity) * 0.3)

        # Invert: 1.0 = healthy, 0.0 = critical distress
        health_score = max(0.0, 1.0 - urgency)

        # Apply FOCUS nudge (gentle bias toward center, clamped)
        nudge = focus_nudges[idx] * 0.1  # Scale down — nudges are suggestions
        health_score = max(0.0, min(1.0, health_score + nudge))

        tensor.append(round(health_score, 4))
        details[sense_name] = {
            "raw": round(raw_value, 4),
            "severity": severity.name,
            "velocity": round(velocity, 4),
            "health_score": round(health_score, 4),
            "category_weight": category_weight,
            "filter_down_multiplier": round(multiplier, 4),
        }

    # Phase 2.5.A — record firing for /v4/debug/dim-sources diagnostics.
    try:
        from titan_hcl.api.dim_registry import get_firing_tracker
        get_firing_tracker().record_block(
            "inner_body",
            tensor,
            {"body_state": history if history else None},
        )
    except Exception:
        pass
    return tensor, details


def _calculate_velocity(hist: deque) -> float:
    """
    Calculate rate of change over recent readings.
    Positive velocity = worsening (raw_value increasing).
    """
    if len(hist) < 2:
        return 0.0

    recent = list(hist)[-_VELOCITY_WINDOW:]
    if len(recent) < 2:
        return 0.0

    # Linear regression slope (simple: first vs last)
    dt = recent[-1]["ts"] - recent[0]["ts"]
    if dt < 1.0:
        return 0.0

    dv = recent[-1]["value"] - recent[0]["value"]
    return dv / (dt / 60.0)  # Change per minute


# ── Individual Sensors ──────────────────────────────────────────────

def _sense_interoception(thresholds: dict) -> dict:
    """
    SOL balance + anchor freshness — energy state AND physical-world connection.

    Blends: SOL balance (metabolic energy) + time since last on-chain anchor
    (how recently Titan "touched" the physical world). Both are REAL data.
    """
    try:
        import json as _json

        # SOL balance (from metabolism or anchor_state)
        balance = 1.0
        balance_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "last_balance.txt")
        if os.path.exists(balance_file):
            with open(balance_file) as f:
                balance = float(f.read().strip())

        # Anchor freshness (from anchor_state.json — written by spirit_worker)
        anchor_freshness = 0.5  # neutral if no anchors yet
        anchor_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
        if os.path.exists(anchor_file):
            with open(anchor_file) as _af:
                _anchor = _json.load(_af)
            _time_since = time.time() - _anchor.get("last_anchor_time", time.time())
            # Freshness decays: 0s=1.0, 300s=0.5, 600s=0.25
            anchor_freshness = max(0.05, 1.0 / (1.0 + _time_since / 300.0))

        # Blend: SOL energy (0.6) + anchor connection (0.4)
        sol_score = 0.1 if balance > thresholds.get("sol_warning", 0.5) else (0.5 if balance > thresholds.get("sol_critical", 0.1) else 0.9)
        combined = sol_score * 0.6 + (1.0 - anchor_freshness) * 0.4  # Fresh anchor = lower interoception (healthy)

        if balance < thresholds.get("sol_critical", 0.1):
            return {"value": max(combined, 0.8), "severity": Severity.CRITICAL}
        elif balance < thresholds.get("sol_warning", 0.5):
            return {"value": combined, "severity": Severity.WARNING}
        else:
            return {"value": combined, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


def _sense_proprioception(thresholds: dict) -> dict:
    """
    Body topology self-sensing — Titan feels his own body shape.

    Reads sphere clock mean radius from shared state file (written by
    spirit_worker). Lower radius = more contracted/balanced = healthier.

    This is TRUE proprioception: sensing one's own body position/shape,
    not external network latency (which is now in entropy).
    """
    try:
        # Read topology state from shared file (spirit_worker writes this)
        topo_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "body_topology.json")
        if os.path.exists(topo_file):
            import json
            with open(topo_file) as f:
                topo = json.load(f)
            # Mean inner sphere radius: 1.0 = fully expanded, 0.3 = minimum
            mean_radius = topo.get("mean_inner_radius", 0.8)
            volume = topo.get("volume", 1.0)
            # Health: lower radius = more balanced = healthier (inverted)
            # Map: radius 1.0→0.8 (unhealthy), radius 0.3→0.1 (very healthy)
            health_raw = max(0.05, (mean_radius - 0.3) / 0.7)
            if mean_radius < 0.5:
                return {"value": health_raw * 0.3, "severity": Severity.INFO}
            elif mean_radius < 0.8:
                return {"value": health_raw * 0.5, "severity": Severity.INFO}
            else:
                return {"value": health_raw * 0.7, "severity": Severity.WARNING}
        else:
            # No topology data yet — neutral
            return {"value": 0.3, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.INFO}


_soma_prev = {"cpu": 0.0, "ram": 0.0}  # Track previous readings for delta sensing

def _sense_somatosensation(thresholds: dict) -> dict:
    """System resources — CPU, RAM, swap, disk with DELTA sensing.

    Senses both absolute resource pressure AND rate of change.
    Delta component provides natural variability as workloads shift.
    """
    try:
        import psutil

        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")

        ram_pct = mem.percent
        swap_pct = swap.percent
        disk_pct = disk.percent

        # Delta sensing: rate of change since last reading
        cpu_delta = abs(cpu_pct - _soma_prev["cpu"]) / 100.0
        ram_delta = abs(ram_pct - _soma_prev["ram"]) / 100.0
        _soma_prev["cpu"] = cpu_pct
        _soma_prev["ram"] = ram_pct

        # Worst-case drives the severity (absolute pressure)
        worst = max(cpu_pct, ram_pct, swap_pct, disk_pct)

        if ram_pct > thresholds.get("ram_critical", 95) or disk_pct > 95 or swap_pct > 90:
            return {"value": 0.95, "severity": Severity.CRITICAL}
        elif worst > thresholds.get("resource_warning", 80):
            base = worst / 100.0 * 0.7
            return {"value": min(1.0, base + cpu_delta * 0.3), "severity": Severity.WARNING}
        else:
            # Blend: 70% absolute pressure + 30% rate of change
            base = worst / 100.0 * 0.3
            delta_component = min(0.2, (cpu_delta + ram_delta) * 0.5)
            return {"value": base + delta_component, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.5, "severity": Severity.WARNING}


def _sense_entropy(thresholds: dict) -> dict:
    """Disorder + network health — errors/chaos + connectivity."""
    try:
        log_path = "/tmp/titan_agent.log"
        if not os.path.exists(log_path):
            return {"value": 0.1, "severity": Severity.INFO}

        # Count ERROR/WARNING lines in last 1KB of log
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(max(0, size - read_size))
            tail = f.read().decode("utf-8", errors="replace")

        lines = tail.split("\n")
        recent_errors = sum(1 for l in lines if "ERROR" in l or "CRITICAL" in l)
        recent_warnings = sum(1 for l in lines if "WARNING" in l)

        # Multi-endpoint network sensing — Titan feels the "shape" of his network space
        # Real latency to multiple endpoints provides natural variability
        import socket
        net_entropy = 0.0
        latencies = []
        endpoints = [
            ("127.0.0.1", 7777),      # Self (local API)
            ("203.0.113.10", 7777),      # Twin (T2 via VPC)
            ("203.0.113.10", 443),     # Solana RPC (api.devnet.solana.com)
            ("203.0.113.10", 443),          # Cloudflare DNS — global internet health
        ]
        for host, port in endpoints:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                t0 = time.time()
                result = sock.connect_ex((host, port))
                latency = time.time() - t0
                sock.close()
                if result == 0:
                    latencies.append(latency)
                else:
                    latencies.append(2.0)  # Unreachable = max latency
                    net_entropy += 0.15
            except Exception:
                latencies.append(2.0)
                net_entropy += 0.1

        # Network entropy = variance in latencies (uneven = disordered) + failures
        if len(latencies) >= 2:
            _lat_mean = sum(latencies) / len(latencies)
            _lat_var = sum((l - _lat_mean) ** 2 for l in latencies) / len(latencies)
            net_entropy += min(0.3, _lat_var * 10)  # Scale variance to [0, 0.3]

        # Combine log errors + network space entropy
        error_score = min(0.7, recent_errors * 0.1 + recent_warnings * 0.02)
        combined = min(1.0, error_score + net_entropy)

        if combined > 0.7:
            return {"value": combined, "severity": Severity.CRITICAL}
        elif combined > 0.3:
            return {"value": combined, "severity": Severity.WARNING}
        else:
            return {"value": combined, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


def _sense_thermal(thresholds: dict) -> dict:
    """
    Heat synthesis — physical (CPU load) × digital (topology activity density).

    Heat is one of the most fundamental parameters in the universe.
    This dimension synthesizes both worlds Titan exists in:
    - Physical: CPU load, silicon heating up
    - Digital: topology activity density, mathematical space "heating"

    Both ARE the same phenomenon: energy being transformed.
    """
    try:
        load_1, load_5, load_15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1

        # Physical heat: CPU load normalized
        physical_heat = load_1 / max(1, cpu_count)

        # Digital heat: topology curvature magnitude (activity density)
        digital_heat = 0.0
        try:
            import json
            topo_file = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "body_topology.json")
            if os.path.exists(topo_file):
                with open(topo_file) as f:
                    topo = json.load(f)
                # Curvature magnitude = topology activity (absolute value)
                digital_heat = min(1.0, abs(topo.get("curvature", 0.0)))
        except Exception:
            pass

        # Circadian cycle: real UTC time → sinusoidal day/night rhythm
        # Titan feels Earth's rotation — all Earth beings share this rhythm
        import math
        hour = time.gmtime().tm_hour + time.gmtime().tm_min / 60.0
        circadian = 0.5 + 0.30 * math.sin(2 * math.pi * (hour - 6) / 24)  # Peak at noon, trough at midnight

        # Synthesis: physical heat × digital heat × circadian rhythm
        heat = 0.4 * physical_heat + 0.3 * digital_heat + 0.3 * circadian

        if heat > thresholds.get("load_critical", 4.0) / max(1, cpu_count):
            return {"value": min(0.95, heat), "severity": Severity.CRITICAL}
        elif heat > 0.4:
            return {"value": heat * 0.7, "severity": Severity.WARNING}
        else:
            return {"value": heat * 0.3, "severity": Severity.INFO}
    except Exception:
        return {"value": 0.3, "severity": Severity.WARNING}


# ── Config & Messaging Helpers ──────────────────────────────────────

def _load_thresholds(config: dict) -> dict:
    """Extract sensor thresholds from config, with defaults."""
    return {
        "sol_critical": config.get("sol_critical", 0.1),
        "sol_warning": config.get("sol_warning", 0.5),
        "api_port": config.get("api_port", 7777),
        "latency_warning_ms": config.get("latency_warning_ms", 2000),
        "ram_critical": config.get("ram_critical", 95),
        "resource_warning": config.get("resource_warning", 80),
        "errors_critical": config.get("errors_critical", 10),
        "warnings_warning": config.get("warnings_warning", 15),
        "load_critical": config.get("load_critical", 4.0),
        "load_warning": config.get("load_warning", 2.0),
    }


def _publish_body_state(send_queue, name: str, tensor: list, details: dict,
                        severity_multipliers: list | None = None) -> None:
    """Publish BODY_STATE to the bus (periodic broadcast)."""
    center = [0.5] * 5
    center_dist = sum((t - c) ** 2 for t, c in zip(tensor, center)) ** 0.5

    payload = {
        "dims": 5,
        "values": tensor,
        "delta": [round(t - 0.5, 4) for t in tensor],  # Delta from center
        "center_dist": round(center_dist, 4),
        "details": details,
    }
    if severity_multipliers:
        payload["filter_down_multipliers"] = [round(m, 4) for m in severity_multipliers]

    _send_msg(send_queue, bus.BODY_STATE, name, "all", payload)


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
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


# Phase 10J — Body reflex intuition relocated to logic/body_helpers.py (pure,
# torch/cgn-free) so agno_hooks (via logic/reflex_intuition) no longer imports
# from this worker module body per SPEC §11.B.4. body_worker self-imports it
# back for its own internal reflex callsite.
from titan_hcl.logic.body_helpers import _compute_body_reflex_intuition  # noqa: E402


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    """Send MODULE_HEARTBEAT on bus + Phase 11 SHM-slot heartbeat sidecar.

    Phase 11 §11.I.5 — when `_state_writer` is set AND `_WORKER_READY` is
    True, also publish `state_writer.heartbeat()` so guardian_hcl's
    SHM-staleness detector + observatory /v6/readiness see fresh data on
    the same cadence as the legacy bus path. SHM publish is best-effort.
    """
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
    if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
        try:
            _state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass


# ── S7 §L1 fast-path helpers ────────────────────────────────────────


def _read_flag(config: dict, dotted_path: str, default: bool) -> bool:
    """
    Resolve a dotted feature flag path through the worker's config dict.

    Workers receive the section of titan_params they need (config dict
    is built upstream in plugin.py / legacy_core.py module registration).
    The body worker config has been observed to contain either:
      - the [body] section directly (legacy)
      - the full titan_params (which includes [microkernel])

    We try both forms; default if neither resolves.
    """
    parts = dotted_path.split(".")
    node = config
    for part in parts:
        if not isinstance(node, dict):
            return default
        if part not in node:
            return default
        node = node[part]
    if isinstance(node, bool):
        return node
    return default


def _start_fast_path(thresholds: dict, config: dict, stop_event,
                     get_modulators) -> tuple:
    """
    Start the §L1 fast-path threads for body. Returns:
      (sensor_cache, refresh_threads, shm_bank, body_5d_writer, shm_writer_thread)

    get_modulators: callable returning current (severity_multipliers,
    focus_nudges) — closure over worker-local mutables so the shm
    writer always uses the live values.

    Synchronous warmup is run BEFORE refresh threads start, so the
    cache is fully populated when the writer thread fires its first
    tick (no chance of cold-cache reads).
    """
    from titan_hcl.core.sensor_cache import (
        RefreshSpec, SensorCache, start_refresh_threads, start_shm_writer_thread,
    )
    from titan_hcl.core.state_registry import INNER_BODY_5D, RegistryBank

    # Synchronous warmup so the tick path never reads cold cache.
    initial = {
        "interoception": _sense_interoception(thresholds),
        "proprioception": _sense_proprioception(thresholds),
        "somatosensation": _sense_somatosensation(thresholds),
        "entropy": _sense_entropy(thresholds),
        "thermal": _sense_thermal(thresholds),
    }
    cache = SensorCache(initial=initial)

    # Per-sense refresh threads at native cadences.
    specs = [
        RefreshSpec(
            name="interoception",
            refresh_fn=lambda: _sense_interoception(thresholds),
            period_s=_BODY_REFRESH_PERIODS_S["interoception"],
        ),
        RefreshSpec(
            name="proprioception",
            refresh_fn=lambda: _sense_proprioception(thresholds),
            period_s=_BODY_REFRESH_PERIODS_S["proprioception"],
        ),
        RefreshSpec(
            name="somatosensation",
            refresh_fn=lambda: _sense_somatosensation(thresholds),
            period_s=_BODY_REFRESH_PERIODS_S["somatosensation"],
        ),
        RefreshSpec(
            name="entropy",
            refresh_fn=lambda: _sense_entropy(thresholds),
            period_s=_BODY_REFRESH_PERIODS_S["entropy"],
        ),
        RefreshSpec(
            name="thermal",
            refresh_fn=lambda: _sense_thermal(thresholds),
            period_s=_BODY_REFRESH_PERIODS_S["thermal"],
        ),
    ]
    refresh_threads = start_refresh_threads(
        specs, cache, stop_event, thread_name_prefix="body_refresh",
    )

    # Schumann shm writer — two modes depending on Phase A+B vs Phase C:
    #
    # Phase A+B (microkernel.l0_rust_enabled = false):
    #   shm_body_fast_enabled gates body_worker's direct write to
    #   inner_body_5d.bin (the FINAL OUTPUT slot). No Rust daemon involved.
    #
    # Phase C (microkernel.l0_rust_enabled = true):
    #   shm_body_fast_enabled is false (per /root/.titan/microkernel_T<id>.toml
    #   override). Instead body_worker writes the SAME 5D tensor to
    #   sensor_cache_inner_body.bin (sensor INPUT slot for the Rust daemon).
    #   The Rust binary titan-inner-body-rs reads sensor_cache_inner_body.bin
    #   every Schumann tick, applies UNIFIED + LOCAL filter_down +
    #   GROUND_UP, and writes the FINAL OUTPUT to inner_body_5d.bin.
    #   See inner_body_sensor_refresh.py for the canonical writer class.
    #   SPEC §G1 + §23.4 + §9.A line 1014.
    shm_bank = RegistryBank(titan_id=None, config=config)
    body_5d_writer = None
    shm_writer_thread = None

    # History deque (separate from main-loop's history) for the writer's
    # velocity calculation — needs its own state since the writer runs at
    # 7.83 Hz vs main loop at 0.29 Hz. Used by both Phase A+B legacy
    # writer + Phase C sensor cache writer.
    writer_history = {
        "interoception": deque(maxlen=_HISTORY_SIZE),
        "proprioception": deque(maxlen=_HISTORY_SIZE),
        "somatosensation": deque(maxlen=_HISTORY_SIZE),
        "entropy": deque(maxlen=_HISTORY_SIZE),
        "thermal": deque(maxlen=_HISTORY_SIZE),
    }

    if shm_bank.is_enabled(INNER_BODY_5D):
        # Phase A+B path — direct write to inner_body_5d.bin.
        body_5d_writer = shm_bank.writer(INNER_BODY_5D)

        def tick():
            severity_multipliers, focus_nudges = get_modulators()
            tensor, _ = _collect_body_tensor(
                writer_history, thresholds,
                severity_multipliers, focus_nudges,
                cache=cache,
            )
            import numpy as np
            arr = np.asarray(tensor, dtype=np.float32)
            if arr.shape == (5,):
                body_5d_writer.write(arr)

        shm_writer_thread = start_shm_writer_thread(
            tick, _BODY_TICK_PERIOD_S, stop_event, "body_shm_writer",
        )
    elif (config or {}).get("microkernel", {}).get("l0_rust_enabled", False):
        # Phase C path — Step 7 §4.4 schema migration v1→v2:
        # Publish msgpack source dict (per-sense {value, severity, velocity})
        # to sensor_cache_inner_body.bin instead of pre-computed 5D tensor.
        # titan-inner-body-rs decodes msgpack + executes per-dim urgency-
        # weighted health-score formula in Rust L1 per SPEC §23.4 + G1.
        try:
            from titan_hcl.logic.inner_body_sensor_refresh import (
                InnerBodySensorRefresh)
            import msgpack as _msgpack

            # P0.6-B re-grounding (D-SPEC-104 / Maker call 2026-05-26):
            # inner_body[3] = entropy was std=0.016–0.020 fleet-wide per audit
            # 2026-05-25 because the raw composite (error_score + net_entropy)
            # is mostly low + stable. Re-ground the dim's `value` field via a
            # module-local ChangeBreathTracker so the dim reflects |Δlevel|/dt
            # rather than the level itself. Severity + velocity stay as-is
            # (Rust formula still applies its urgency aggregate).
            from titan_hcl.logic.expression_window_tracker import (
                ChangeBreathTracker as _CBT,
            )
            _entropy_change_tracker = _CBT()

            def _provide_body_source_dict() -> bytes:
                # Per-sense raw readings (value + severity + velocity from
                # rolling history). Rust applies urgency formula:
                #   urgency = raw * sev_value * mult / CRITICAL + |vel| * 0.3
                #   health = max(0, 1 - urgency)
                # The sense interpretation (_sense_*) stays Python because
                # OS reads (procfs etc.) are platform-specific.
                from titan_hcl.modules.body_worker import (
                    _sense_interoception, _sense_proprioception,
                    _sense_somatosensation, _sense_entropy, _sense_thermal,
                    _calculate_velocity, Severity,
                )
                sense_fns = [
                    ("interoception", _sense_interoception),
                    ("proprioception", _sense_proprioception),
                    ("somatosensation", _sense_somatosensation),
                    ("entropy", _sense_entropy),
                    ("thermal", _sense_thermal),
                ]
                senses = {}
                _now_for_entropy = time.time()
                for name, fn in sense_fns:
                    try:
                        reading = fn(thresholds)
                    except Exception:
                        # Sense raise → publish neutral defaults so Rust
                        # daemon doesn't starve on missing dim.
                        senses[name] = {
                            "value": 0.5, "severity": 1.0, "velocity": 0.0,
                        }
                        continue
                    history_dq = writer_history.get(name, deque())
                    history_dq.append({
                        "ts": time.time(),
                        "value": reading["value"],
                        "severity": reading["severity"],
                    })
                    velocity = _calculate_velocity(history_dq)
                    raw_value = float(reading["value"])
                    if name == "entropy":
                        # Re-ground to rate-of-change of raw entropy level.
                        _ch = _entropy_change_tracker.update(
                            _now_for_entropy, {"entropy": raw_value}
                        )
                        raw_value = float(_ch.get("entropy_change", 0.0))
                    senses[name] = {
                        "value": raw_value,
                        "severity": float(reading["severity"].value),
                        "velocity": float(velocity),
                    }
                payload = {
                    "senses": senses,
                    "critical_threshold": float(Severity.CRITICAL.value),
                }
                return _msgpack.packb(payload, use_bin_type=True)

            sensor_cache_writer = InnerBodySensorRefresh(
                tensor_provider=_provide_body_source_dict,
                titan_id=None,
            )
            shm_writer_thread = sensor_cache_writer.start_thread(stop_event)
            logger.info(
                "[BodyWorker] sensor_cache_inner_body.bin source-dict writer "
                "started (l0_rust_enabled=true; schema v2 msgpack; Rust "
                "titan-inner-body-rs computes 5D urgency/health formulas + "
                "owns inner_body_5d.bin output)")
        except Exception:
            logger.critical(
                "[BodyWorker] failed to start inner_body sensor cache writer — "
                "Rust inner-body-rs will starve on constant-zero input. "
                "Investigate before relying on inner_body_5d.bin downstream.",
                exc_info=True)

    return cache, refresh_threads, shm_bank, body_5d_writer, shm_writer_thread
