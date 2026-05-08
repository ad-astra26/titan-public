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

        # DIAG: count all messages received (Phase 1 wiring debug)
        _DIAG_MSG_TOTAL[0] += 1
        _DIAG_TYPE_HIST[msg_type] = _DIAG_TYPE_HIST.get(msg_type, 0) + 1
        if _DIAG_MSG_TOTAL[0] in (1, 100, 500, 1000, 5000):
            top = sorted(_DIAG_TYPE_HIST.items(), key=lambda kv: -kv[1])[:15]
            logger.info(
                "[OuterSpiritWorker] msg #%d type_histogram (top15): %s",
                _DIAG_MSG_TOTAL[0],
                ", ".join(f"{t}:{n}" for t, n in top))

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
                _cache_size = len(_plugin_cache)
                _payload_size = len(payload)
            # DIAG: log first + every 10th snapshot to confirm receipt + cache growth
            _DIAG_OSC_COUNT[0] += 1
            if _DIAG_OSC_COUNT[0] == 1 or _DIAG_OSC_COUNT[0] % 10 == 0:
                logger.info(
                    "[OuterSpiritWorker] OUTER_SOURCES_SNAPSHOT #%d "
                    "payload_keys=%d cache_keys=%d sample_keys=%s",
                    _DIAG_OSC_COUNT[0], _payload_size, _cache_size,
                    sorted(list(payload.keys())[:8]),
                )

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
    # rFP_trinity_130d_awakening §12 / SPEC §23.9 — guardian_stats now
    # sources from real producers (jailbreak_alerts, output_verifier).
    jb_stats = snap.get("jailbreak_alerts_stats") or {}
    ov_stats = snap.get("output_verifier_stats") or {}
    guardian_stats = {
        "threats_detected": int(jb_stats.get("threats_detected_24h", 0) or 0),
        "rejections": (int(ov_stats.get("rejected_count", 0) or 0)
                       + int(jb_stats.get("blocked_24h", 0) or 0)),
        "severity_avg": float(jb_stats.get("severity_avg_24h", 0.0) or 0.0),
        "rejections_per_window": float(jb_stats.get("blocked_24h", 0) or 0) / 24.0,
        "confirmed_threats": int(jb_stats.get("confirmed_threats_24h", 0) or 0),
    }
    _sp = snap.get("social_perception_stats") or {}
    # rFP_trinity_130d_awakening Phase 2 (SPEC §23.9 ANANDA[36,38]):
    # community_engagement_stats refreshed by plugin's heavy-stats thread
    # (60s cadence). distinct_handles_24h feeds ANANDA[36] community_connection
    # (tensor normalizes via /5.0); mean_engagement_per_post_7d feeds
    # ANANDA[38] expression_reach (the value is already normalized to [0,1]).
    _ce = snap.get("community_engagement_stats") or {}
    social_stats = {
        "interactions_per_window": mem_status.get("unique_interactors", 0),
        "sentiment_avg": _sp.get("sentiment_ema", 0.5),
        "social_connection": _sp.get("connection_ema", 0.0),
        "social_events_count": _sp.get("events_count", 0),
        "last_contagion": _sp.get("last_contagion"),
        "mean_conversation_quality": assessment.get("average_score", 0.5),
        # ANANDA[36] community_connection: distinct mention/reply handles
        # in last 24h (raw count; tensor clamps via min(1, x/5)).
        "new_connections_per_window": float(_ce.get("distinct_handles_24h", 0)),
        # ANANDA[38] expression_reach: pre-normalized [0,1] from
        # SocialXGateway.get_community_engagement_stats. Cold-start (no
        # posts in last 7d) → 0.0 (SPEC-correct: no reach yet).
        "creative_engagement": float(
            _ce.get("expression_reach_norm", 0.0) or 0.0),
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

    # rFP_trinity_130d_awakening §12 — sovereignty_ratio now comes from
    # expression_translator_stats (real producer) instead of ghost agency key.
    expr_translator = snap.get("expression_translator_stats") or {}
    sovereignty_ratio = float(expr_translator.get("sovereignty_ratio",
                                                    agency.get("sovereignty_ratio", 0.0)))

    # World footprint inputs (SPEC §23.3 weighted log-sum across all artifact streams)
    world_footprint_inputs = _build_world_footprint_inputs(snap, agency, lang_stats=snap.get("language_stats"))

    # 24h-delta tracking for capability_growth + knowledge_growth (Phase 1 minimal)
    deltas_24h = _record_and_compute_deltas(snap)

    # Inner-outer coherence: ratio of recent action_history entries where
    # the posture's expected hormone WAS dominant in trinity_before snapshot.
    # SPEC §23.9 SAT[1] expressive_authenticity. Worker reads from agency
    # _history via the agency_stats indirection (history not exposed today
    # — Phase 1 leaves at 0.5 unless agency exposes recent_actions detail).
    action_stats["inner_outer_coherence"] = _compute_inner_outer_coherence(snap)

    # rFP_trinity_130d_awakening Phase 2 — build the history dict feeding
    # SAT[11], CHIT[10,11,14], ANANDA[10,11]. Sources:
    #   * outer_spirit_history_stats (plugin-side, env_adapt + graceful_rest
    #     + circadian_alignment + dream_recall_ratio per SPEC §23.9)
    #   * inner_perception_stats.last_create_ts → seconds_since_last_create
    #     for ANANDA[11] creative_tension
    #   * worker-local _OUTER_SPIRIT_SNAPSHOTS for CHIT[29] self_trajectory
    #     (snapshot of prior tick's 45D; this tick's append happens after
    #     compute, see end of function)
    osh_stats = snap.get("outer_spirit_history_stats") or {}
    ip_stats = snap.get("inner_perception_stats") or {}
    last_create_ts = float(ip_stats.get("last_create_ts", 0.0) or 0.0)
    if last_create_ts > 0.0:
        seconds_since_create = max(0.0, time.time() - last_create_ts)
    else:
        # No create observed yet — use a long elapsed time so the ANANDA[11]
        # formula `CREATIVITY * min(1, dt/600)` saturates at the hormone
        # level (max creative tension when there's been no creative outlet).
        seconds_since_create = 600.0
    history_dict = {
        # SPEC §23.9 CHIT[10] dream_recall ratio (refined 2026-05-07 to use
        # experiential_memory.get_recall_ratio).
        "dream_recall_ratio": float(
            osh_stats.get("dream_recall_ratio", 0.0) or 0.0),
        # SPEC §23.9 CHIT[11] circadian_alignment (mean(sin) over last 200 ts).
        "circadian_alignment": float(
            osh_stats.get("circadian_alignment", 0.5) or 0.5),
        # SPEC §23.9 SAT[11] env_adaptation (1 - var(scores during high-thermal)).
        "environmental_adaptation": float(
            osh_stats.get("environmental_adaptation", 0.5) or 0.5),
        # SPEC §23.9 ANANDA[10] graceful_rest (min score during low-load+night).
        "rest_performance_floor": float(
            osh_stats.get("graceful_rest", 0.5) or 0.5),
        # SPEC §23.9 CHIT[29] self_trajectory (worker-local snapshot deque).
        "outer_spirit_trajectory": _compute_self_trajectory(),
        # SPEC §23.9 ANANDA[11] creative_tension (seconds since last create).
        "seconds_since_last_create": seconds_since_create,
    }
    # Phase 1 may have populated history elsewhere in snap; preserve those
    # keys for back-compat without overriding the Phase 2 producers.
    _phase1_hist = snap.get("history") or {}
    if isinstance(_phase1_hist, dict):
        for k, v in _phase1_hist.items():
            history_dict.setdefault(k, v)

    tensor_45d = collect_outer_spirit_45d(
        current_5d=tensor_5d,
        outer_body=outer_body,
        outer_mind=outer_mind,
        action_stats=action_stats,
        creative_stats=creative_stats,
        guardian_stats=guardian_stats,
        sovereignty_ratio=sovereignty_ratio,
        uptime_ratio=uptime_ratio,
        social_stats=social_stats,
        memory_stats=memory_stats,
        assessment_stats=assessment_ext,
        history=history_dict,
        # rFP §12 / SPEC §23.9 rich producer kwargs:
        anchor_state=snap.get("anchor_state"),
        bus_stats=snap.get("bus_stats"),
        cgn_stats=snap.get("cgn_stats"),
        meta_cgn_stats=snap.get("meta_cgn_stats"),
        language_stats=snap.get("language_stats"),
        memory_growth_metrics=snap.get("memory_growth_metrics"),
        knowledge_graph_stats=snap.get("knowledge_graph_stats"),
        inner_memory_stats=snap.get("inner_memory_stats"),
        jailbreak_alerts_stats=snap.get("jailbreak_alerts_stats"),
        output_verifier_stats=snap.get("output_verifier_stats"),
        solana_stats=snap.get("solana_local_stats"),
        hormone_levels=snap.get("hormone_levels"),
        world_footprint_inputs=world_footprint_inputs,
        deltas_24h=deltas_24h,
        llm_calls_this_hour=int(agency.get("llm_calls_this_hour", 0) or 0),
    )

    # rFP_trinity_130d_awakening Phase 2 (CHIT[29] self_trajectory).
    # Append snapshot to module-level deque, THROTTLED to once per
    # _SELF_TRAJ_SNAPSHOT_INTERVAL_S (30 s). The original Phase 2.3
    # implementation appended every tick, which sounds right but the
    # worker actually ticks at the SHM-writer cadence — 70.47 Hz Schumann
    # (≈14 ms), not the 30 s I assumed. Without throttling, deque(maxlen=120)
    # rotates fully in ~1.7 s, so self_trajectory measured 1.7 s of change
    # instead of 60 min and pinned at 0.0 for any stable system. The
    # _DELTA_HISTORY pattern below uses the same throttle approach for
    # 24h-delta tracking; we mirror it exactly.
    _now = time.time()
    if _now - _OUTER_SPIRIT_SNAPSHOT_LAST_TS[0] >= _SELF_TRAJ_SNAPSHOT_INTERVAL_S:
        _OUTER_SPIRIT_SNAPSHOTS.append((_now, list(tensor_45d)))
        _OUTER_SPIRIT_SNAPSHOT_LAST_TS[0] = _now

    return [round(v, 4) for v in tensor_45d], tensor_5d


# ── Phase 1 helpers: world_footprint, 24h deltas, inner-outer coherence ──
# rFP_trinity_130d_awakening §12 / SPEC §23.3 + §23.9.

# 24h-delta history per metric. Keyed by metric name → deque of (ts, value).
# Snapshot at most every 60s to bound memory at 1500 entries × handful of metrics.
import collections as _collections

_DELTA_HISTORY: dict = {}
_DELTA_LAST_RECORDED_TS: dict = {}
_DELTA_SNAPSHOT_INTERVAL_S = 60.0

# rFP_trinity_130d_awakening Phase 2 — worker-local 45D snapshot deque for
# CHIT[29] self_trajectory (SPEC §23.9). 120 snapshots × 30s throttled
# cadence = 60 min window. The append site (in _collect_tick above) gates
# on _SELF_TRAJ_SNAPSHOT_INTERVAL_S because the worker tick is 70.47 Hz
# Schumann — without the throttle, deque rotates in ~1.7 s and trajectory
# becomes pinned at ~0.0.
import math as _math
_OUTER_SPIRIT_SNAPSHOTS: _collections.deque = _collections.deque(maxlen=120)
_SELF_TRAJ_SNAPSHOT_INTERVAL_S = 30.0
# List wrapper so the append-time mutation in _collect_tick is visible to
# the module-level deque without needing `global`.
_OUTER_SPIRIT_SNAPSHOT_LAST_TS: list = [0.0]


def _compute_self_trajectory() -> float:
    """SPEC §23.9 CHIT[29]. L2 between most-recent prior snapshot and the
    oldest snapshot in the deque, normalized /5.0. Cold-start (n<2) → 0.0
    (no trajectory yet — SPEC-correct, not a default).
    """
    if len(_OUTER_SPIRIT_SNAPSHOTS) < 2:
        return 0.0
    old = _OUTER_SPIRIT_SNAPSHOTS[0][1]
    new = _OUTER_SPIRIT_SNAPSHOTS[-1][1]
    L = min(len(old), len(new))
    dist_sq = sum((old[i] - new[i]) ** 2 for i in range(L))
    return min(1.0, _math.sqrt(dist_sq) / 5.0)

# DIAG: counter for OUTER_SOURCES_SNAPSHOT receipts (Phase 1 wiring debug)
_DIAG_OSC_COUNT: list = [0]
_DIAG_MSG_TOTAL: list = [0]
_DIAG_TYPE_HIST: dict = {}


def _record_metric(name: str, value: float, now: float) -> None:
    """Throttled metric snapshot for 24h-delta computation."""
    last_ts = _DELTA_LAST_RECORDED_TS.get(name, 0)
    if now - last_ts < _DELTA_SNAPSHOT_INTERVAL_S:
        return
    dq = _DELTA_HISTORY.setdefault(name, _collections.deque(maxlen=1500))
    dq.append((now, value))
    _DELTA_LAST_RECORDED_TS[name] = now


def _compute_metric_delta_24h(name: str) -> float:
    """Current value minus value from ~24h ago. Returns 0.0 if no history."""
    dq = _DELTA_HISTORY.get(name)
    if not dq or len(dq) < 2:
        return 0.0
    import time as _time
    now = _time.time()
    target = now - 86400.0
    # Walk to find oldest sample within [now-25h, now-23h] window — falls
    # back to the oldest available if no sample is old enough yet.
    base_value = dq[0][1]
    for ts, val in dq:
        if ts >= target:
            base_value = val
            break
    return float(dq[-1][1]) - float(base_value)


def _record_and_compute_deltas(snap: dict) -> dict:
    """Record current values + return computed 24h deltas + per-day rates.

    Inputs sampled (SPEC §23.9 CHIT[21] knowledge_growth + ANANDA[37]
    capability_growth):
      - composition_level (language_stats.composition_level — "L1".."L9")
      - primitives_grounded (meta_cgn_stats.primitives_grounded)
      - vocab_producible (language_stats.vocab_producible)
      - reflex_distinct_fired_24h — Phase 2 (no per-reflex stats in snap yet)
      - compositions_computed (meta_cgn_stats.compositions_computed)
    """
    import time as _time
    now = _time.time()
    lang = snap.get("language_stats") or {}
    mcgn = snap.get("meta_cgn_stats") or {}

    # composition_level: parse "L3" → 3
    comp_str = lang.get("composition_level", "L1")
    try:
        comp_level = int(comp_str[1:]) if isinstance(comp_str, str) and comp_str.startswith("L") else 1
    except Exception:
        comp_level = 1
    _record_metric("composition_level", float(comp_level), now)

    primitives_grounded = float(mcgn.get("primitives_grounded", 0) or 0)
    _record_metric("primitives_grounded", primitives_grounded, now)

    vocab_producible = float(lang.get("vocab_producible", 0) or 0)
    _record_metric("vocab_producible", vocab_producible, now)

    compositions_computed = float(mcgn.get("compositions_computed", 0) or 0)
    _record_metric("compositions_computed", compositions_computed, now)

    return {
        "composition_level_24h": _compute_metric_delta_24h("composition_level"),
        "primitives_grounded_24h": _compute_metric_delta_24h("primitives_grounded"),
        "vocab_producible_24h": _compute_metric_delta_24h("vocab_producible"),
        "vocab_producible_per_day": _compute_metric_delta_24h("vocab_producible"),
        "compositions_computed_24h": _compute_metric_delta_24h("compositions_computed"),
        # Phase 2 producers (placeholders that the worker will populate later):
        "reflex_distinct_fired_24h": 0.0,
        "felt_experiences_to_action_rate": 0.5,
    }


def _build_world_footprint_inputs(snap: dict, agency: dict, lang_stats: dict | None = None) -> dict:
    """Compute world_footprint weighted log-sum (SPEC §23.3).

    Returns dict with `score_sum` (Σ log1p(N_i) * w_i) and `target_log`
    (calibration constant). outer_spirit_tensor SAT[7] divides them.
    """
    import math as _math
    lang = lang_stats or snap.get("language_stats") or {}
    sx = snap.get("social_x_gateway_stats") or {}
    inner_mem = snap.get("inner_memory_stats") or {}
    anc = snap.get("anchor_state") or {}
    jb = snap.get("jailbreak_alerts_stats") or {}
    extra = snap.get("world_footprint_extra_counts") or {}

    # (N_i, w_i) per SPEC §23.3
    streams: list[tuple[float, float]] = [
        (float(agency.get("total_actions", 0) or 0), 1.0),
        (float(snap.get("art_count_500", 0) or 0), 1.0),
        (float(snap.get("audio_count_500", 0) or 0), 1.0),
        (float(snap.get("text_count_500", 0) or 0), 0.7),
        (float(sx.get("posts_last_day", 0) or 0) * 30, 1.2),  # extrapolate to ~monthly
        (float(sx.get("posts_last_day", 0) or 0) * 0.5, 0.8),  # replies (proxy)
        (float(lang.get("vocab_total", 0) or 0), 0.7),
        (float(inner_mem.get("action_chains", 0) or 0), 0.7),
        (float(anc.get("anchor_count", 0) or 0), 1.5),
        (float(extra.get("arweave_inscriptions", 0) or 0), 1.5),
        (float(extra.get("meditation_memos", 0) or 0), 1.0),
        (float(jb.get("defended_all_time", 0) or 0), 0.7),
    ]

    score_sum = sum(_math.log1p(n) * w for n, w in streams if n >= 0)
    # Target: log1p of expected mature footprint. Calibrated so a Titan with
    # ~500 actions, ~50 art, ~50 audio, ~10 vocab, ~10k tx, ~10 anchor,
    # ~10 arweave, ~50 jailbreak, ~10 meditations reaches ~score 1.0.
    target_log = (
        _math.log1p(500) * 1.0     # actions
        + _math.log1p(50) * 1.0    # art
        + _math.log1p(50) * 1.0    # audio
        + _math.log1p(200) * 0.7   # text
        + _math.log1p(150) * 1.2   # X posts (monthly-extrapolated)
        + _math.log1p(75) * 0.8    # replies
        + _math.log1p(2000) * 0.7  # vocab
        + _math.log1p(500) * 0.7   # action chains
        + _math.log1p(100) * 1.5   # anchors
        + _math.log1p(50) * 1.5    # arweave
        + _math.log1p(50) * 1.0    # meditations
        + _math.log1p(50) * 0.7    # defended attacks
    )

    return {"score_sum": score_sum, "target_log": target_log}


def _compute_inner_outer_coherence(snap: dict) -> float:
    """SPEC §23.9 SAT[1] expressive_authenticity.

    Ratio of recent actions where the posture's target hormone WAS the
    dominant one in trinity_before.hormone_levels at action time.
    Reads agency_stats.recent_actions_detail (broadcast by AgencyModule
    via OUTER_SOURCES_SNAPSHOT). Each entry carries posture + per-action
    hormone snapshot.

    Posture → expected dominant hormone:
      research → CURIOSITY
      socialize → EMPATHY
      create → CREATIVITY
      rest → IMPULSE-low (any non-arousal — uses inverse check)
      meditate → VIGILANCE-low (any deep state — inverse check)
    """
    POSTURE_TO_HORMONE = {
        "research": ("CURIOSITY", True),
        "socialize": ("EMPATHY", True),
        "create": ("CREATIVITY", True),
        "rest": ("IMPULSE", False),  # inverse: low IMPULSE = rest-coherent
        "meditate": ("VIGILANCE", False),  # inverse: low VIGILANCE = meditate-coherent
    }
    agency = snap.get("agency_stats") or {}
    recent = agency.get("recent_actions_detail") or []
    if not isinstance(recent, list) or not recent:
        return 0.5

    matches = 0
    total = 0
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        posture = entry.get("posture")
        hormones = entry.get("hormones") or {}
        target = POSTURE_TO_HORMONE.get(posture)
        if not target or not hormones:
            continue
        hormone_name, expect_high = target
        level = float(hormones.get(hormone_name, 0.5) or 0.5)
        total += 1
        if expect_high and level > 0.5:
            matches += 1
        elif (not expect_high) and level < 0.5:
            matches += 1

    if total == 0:
        return 0.5
    return matches / total


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
