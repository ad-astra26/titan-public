"""
Mind Module Worker — 5DT cognitive/emotional sensor array.

Runs MoodEngine and SocialGraph in isolation. Produces a 5-dimensional
Mind tensor reflecting Titan's emotional, social, and perceptual state.

Each sense has two sub-senses:
  sub_a: Ambient metric (always active, lightweight)
  sub_b: Media digest (from MediaWorker via SENSE_VISUAL/SENSE_AUDIO bus messages)
  Combined: sense = sub_a * 0.5 + sub_b * 0.5 (weights become learnable via FILTER_DOWN)

5DT Mind Senses:
  [0] Vision — sub_a: research freshness | sub_b: image pattern digest
  [1] Hearing — sub_a: conversation quality | sub_b: audio pattern digest
  [2] Taste — social signal quality (SocialGraph)
  [3] Smell — environmental awareness (BonkPulse + WeatherVibe + circadian)
  [4] Touch — emotional state (MoodEngine valence)

S7 (microkernel v2 §A.7 / §L1, 2026-04-26): when
``microkernel.shm_mind_fast_enabled`` is true, this worker runs the
3-layer Trinity Daemon Internal Design pattern: per-sense background
refresh threads at native cadences populate a SensorCache; the tick
layer reads cache only and writes the 5D tensor to /dev/shm at
Schumann × 3 = 23.49 Hz (42.6 ms period). Pre-S7 inline sensor calls
(73 ms/call worst-case) drop to ~100 μs/tick.

The sub_a ambient senses (vision_ambient, hearing_ambient, smell)
are the I/O-heavy ones (sqlite3 connect, fs walk, http urlopen with
3s timeout); these are extracted into refresh threads. The sub_b
media digest is updated by SENSE_VISUAL/SENSE_AUDIO bus messages
on the main loop and read directly by the tick (no extra refresh
needed — bus arrival is already the natural cadence).

Default flag-OFF preserves byte-identical pre-S7 behavior.

Entry point: mind_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)


# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel; mirrored to
# `module_mind_state.bin` via ModuleStateWriter so titan_hcl's 1Hz SHM
# poll + the orchestrator's MODULE_PROBE_REQUEST dispatcher see real
# liveness. Under `l0_rust_enabled=true` mind_worker runs as a SHADOW
# providing sensor_cache_inner_mind.bin input — the SHM slot tracks the
# Python shadow lifecycle, not the Rust daemon's.
_WORKER_READY: bool = False
_state_writer = None  # type: ignore[assignment]

# Sub-sense decay: media digest features decay over time (half-life 30 min)
_DIGEST_HALFLIFE_S = 1800.0
_SUB_WEIGHT_A = 0.5
_SUB_WEIGHT_B = 0.5

# ── S7 Schumann shm writer cadence (microkernel v2 §A.7 / §L1) ─────
_MIND_SCHUMANN_HZ = 23.49  # Schumann × 3
_MIND_TICK_PERIOD_S = 1.0 / _MIND_SCHUMANN_HZ  # ≈ 0.0426 s

# ── S7 per-sense refresh cadences (matches sensor's natural timescale) ──
# - vision_ambient:  fs walk (kuzu dir) + 2 file mtimes; 5s captures research-job batches
# - hearing_ambient: sqlite3 connect + COUNT; 3s — INTERFACE_INPUT is the push path
# - taste:           in-process social_graph.get_stats; 2s
# - smell:           bonk + weather http (≤6s) + circadian; 30s, rate-limit friendly
# - touch:           in-process mood_engine attribute; 1s — but mood updates per
#                    epoch (10-30s) so the read is essentially free either way
_MIND_REFRESH_PERIODS_S = {
    "vision": 5.0,
    "hearing": 3.0,
    "taste": 2.0,
    "smell": 30.0,
    "touch": 1.0,
}


class _SocialGraphStatsShmReader:
    """Lightweight stats-only reader for social_graph_state.bin SHM slot.

    Replaces the in-process SocialGraph instance that mind_worker used to
    host (rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50, v1.7.1).
    SocialGraph is now hosted by social_graph_worker (G21 single-writer);
    mind_worker only needs `.get_stats()` for _sense_taste (~2s tick) +
    _compute_mind_reflex_intuition. Read via SHM per G18 (state via SHM,
    never bus).

    Exposes the same `.get_stats()` shape SocialGraph.get_stats() did so
    downstream callsites need no signature change.
    """

    def __init__(self, titan_id: str):
        from titan_hcl.core.state_registry import (
            StateRegistryReader, ensure_shm_root,
        )
        from titan_hcl.logic.social_graph_state_specs import (
            SOCIAL_GRAPH_STATE_SPEC,
        )
        self._titan_id = titan_id
        self._reader = StateRegistryReader(
            SOCIAL_GRAPH_STATE_SPEC, ensure_shm_root(titan_id))

    def get_stats(self) -> dict:
        try:
            import msgpack
            raw = self._reader.read_variable()
            if raw is None:
                return {
                    "users": 0, "edges": 0, "donations": 0,
                    "total_donated_sol": 0.0, "inspirations": 0,
                }
            decoded = msgpack.unpackb(raw, raw=False)
            if not isinstance(decoded, dict):
                return {
                    "users": 0, "edges": 0, "donations": 0,
                    "total_donated_sol": 0.0, "inspirations": 0,
                }
            return {
                "users": int(decoded.get("users", 0) or 0),
                "edges": int(decoded.get("edges", 0) or 0),
                "donations": int(decoded.get("donations", 0) or 0),
                "total_donated_sol": float(
                    decoded.get("total_donated_sol", 0.0) or 0.0),
                "inspirations": int(decoded.get("inspirations", 0) or 0),
            }
        except Exception:
            return {
                "users": 0, "edges": 0, "donations": 0,
                "total_donated_sol": 0.0, "inspirations": 0,
            }


_MIND_STATE_FILENAME = "mind_state.json"


def _mind_state_path(config: dict) -> str:
    data_dir = config.get("data_dir", "./data")
    return os.path.join(data_dir, _MIND_STATE_FILENAME)


def _load_mind_state(config: dict):
    """Restore (severity_multipliers, focus_nudges) from disk, else (None, None)."""
    import json
    try:
        with open(_mind_state_path(config)) as f:
            d = json.load(f)
        sm = d.get("severity_multipliers")
        fn = d.get("focus_nudges")
        sm = [float(x) for x in sm] if isinstance(sm, list) and len(sm) == 5 else None
        fn = [float(x) for x in fn] if isinstance(fn, list) and len(fn) == 5 else None
        return sm, fn
    except Exception:
        return None, None


def _save_mind_state(config: dict, severity_multipliers, focus_nudges) -> None:
    """Atomically persist Mind's Spirit-learned FILTER_DOWN weights + focus
    nudges (§11.H.2 tmp+os.replace). AUDIT §C fix (rFP §P2): like body_worker,
    these were never persisted (only `_b1_save_state`, the B1 readiness-reporter
    noop) → reset to [1.0]*5 / [0.0]*5 on every respawn until the next
    event-only FILTER_DOWN. media_state re-accumulates from SENSE_VISUAL/
    SENSE_AUDIO events → intentionally not persisted."""
    import json
    path = _mind_state_path(config)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"severity_multipliers": list(severity_multipliers),
                       "focus_nudges": list(focus_nudges)}, f)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("[MindWorker] state save failed: %s", e)


@with_error_envelope(module_name="mind", subsystem="entry", severity=_phase11_sev.FATAL)
def mind_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Mind module process."""
    from queue import Empty

    # Phase 11 §11.I.5 — reset module-level readiness flags.
    global _WORKER_READY, _state_writer
    _WORKER_READY = False
    _state_writer = None

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[MindWorker] Initializing 5DT cognitive sensors...")

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 single-writer) ──
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
            "[MindWorker] Phase 11 ModuleStateWriter init failed "
            "(continuing — SHM slot will be absent): %s", _sw_err)

    # Boot MoodEngine (needs a metabolism stub for standalone operation)
    mood_engine = None
    try:
        from titan_hcl.logic.mood.engine import MoodEngine
        mood_engine = MoodEngine(_MetabolismStub(), config_path="titan_hcl/config.toml")
        logger.info("[MindWorker] MoodEngine initialized")
    except Exception as e:
        logger.warning("[MindWorker] MoodEngine init failed: %s", e)

    # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1, 2026-05-14).
    # SocialGraph was previously instantiated here directly — that instance
    # is now hosted in social_graph_worker (its own subprocess); G21
    # single-writer of data/social_graph.db. mind_worker no longer touches
    # the DB; the 4 _handle_query SocialGraph action handlers (record_interaction,
    # get_or_create_user, should_engage, save_profile) + the get_social_stats
    # handler were REMOVED — callers route via SocialGraphProxy →
    # social_graph_worker now.
    #
    # The only mind_worker dependency on SocialGraph is `_sense_taste` (5DT
    # mind-tensor taste sense, ~2s tick cadence) which reads {users, edges}
    # counts. This now reads social_graph_state.bin SHM slot directly per
    # G18 (state via SHM, never bus) — closes the legacy G22 violation
    # (`get_social_stats` orphan-handler "full migration deferred").
    #
    # `social_graph` symbol below is a lightweight SHM-reading shim with
    # only `.get_stats()` defined. Keeps existing `_sense_taste` /
    # `_compute_mind_reflex_intuition` signatures intact (they only call
    # get_stats); the shim returns the same {users, edges, donations,
    # total_donated_sol, inspirations} shape the in-process SocialGraph
    # used to.
    social_graph = None
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        social_graph = _SocialGraphStatsShmReader(
            titan_id=resolve_titan_id())
        logger.info(
            "[MindWorker] SocialGraph stats reader initialized — SHM slot "
            "social_graph_state.bin (G18 path; replaces in-process SocialGraph "
            "instance per rFP §4.P + D-SPEC-50)")
    except Exception as e:
        logger.warning(
            "[MindWorker] SocialGraph stats reader init failed: %s — "
            "_sense_taste will fall back to neutral 0.5 default", e)

    # Media digest state (sub_b: latest features from MediaWorker)
    media_state = {
        "last_visual": None,       # [5 floats] from SENSE_VISUAL
        "last_visual_ts": 0.0,
        "last_audio": None,        # [5 floats] from SENSE_AUDIO
        "last_audio_ts": 0.0,
    }

    # FILTER_DOWN severity multipliers (learned from Spirit via RL gradients)
    # Modulate sub_a/sub_b blend weights per sense
    severity_multipliers = [1.0] * 5

    # FOCUS nudges from Spirit PID controller
    focus_nudges = [0.0] * 5

    # AUDIT §C fix (rFP §P2): restore Spirit-learned FILTER_DOWN weights + focus
    # nudges from disk so they survive hot-reload / kill-respawn instead of
    # resetting to defaults until the next event-only FILTER_DOWN.
    _saved_sm, _saved_fn = _load_mind_state(config)
    if _saved_sm is not None:
        severity_multipliers = _saved_sm
    if _saved_fn is not None:
        focus_nudges = _saved_fn

    # Outer-sources cache feeding collect_mind_15d's hormone_levels /
    # interaction_quality / assessment_quality / inner_perception args (inner_mind
    # feeling dims). Phase C dissolution (2026-05-22): refreshed SHM-direct via
    # the §9.F outer_source_assembly helper (was OUTER_SOURCES_SNAPSHOT bus
    # broadcast — a G18 state-over-bus violation, now retired).
    plugin_cache: dict = {}
    _MIND_PLUGIN_KEYS = {
        "social_perception_stats", "assessment_stats",
        "hormone_levels", "inner_perception_stats",
    }
    _osa_ctx_mind = None
    _last_plugin_refresh = 0.0
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank as _SRB
        from titan_hcl.logic.outer_source_assembly import OuterSourceContext as _OSC
        from titan_hcl.core.state_registry import resolve_titan_id as _rtid_m
        _osa_ctx_mind = _OSC(
            shm_bank=_SRB(titan_id=_rtid_m()), titan_id=_rtid_m(),
            data_dir=config.get("data_dir", "./data"), start_time=time.time())
    except Exception as _osc_err:
        logger.warning("[MindWorker] outer-source SHM ctx init failed "
                       "(plugin_cache stays empty → feeling dims default): %s",
                       _osc_err)

    def _refresh_plugin_cache_from_shm() -> None:
        if _osa_ctx_mind is None:
            return
        try:
            from titan_hcl.logic.outer_source_assembly import (
                assemble_outer_sources as _aos)
            plugin_cache.update(_aos(_MIND_PLUGIN_KEYS, _osa_ctx_mind))
        except Exception as _ref_err:
            logger.debug("[MindWorker] plugin_cache SHM refresh: %s", _ref_err)

    # Paths for ambient sensors
    data_dir = config.get("data_dir", "./data")
    session_db = os.path.join(data_dir, "agno_sessions.db")

    # ── S7: §L1 fast-path setup (flag-gated) ──────────────────────
    # When microkernel.shm_mind_fast_enabled is true, start per-sense
    # refresh threads + 23.49 Hz shm writer thread. Tick layer reads
    # from cache (no I/O on hot path).
    sensor_cache = None
    refresh_threads = []
    shm_writer_thread = None
    fast_stop_event = threading.Event()

    fast_enabled = _read_flag(config, "microkernel.shm_mind_fast_enabled", False)
    l0_rust_enabled = _read_flag(config, "microkernel.l0_rust_enabled", False)
    # Phase C C-S5 closure 2026-05-08: under l0_rust_enabled=true the
    # Rust titan-inner-mind-rs daemon owns inner_mind_15d.bin output but
    # NEEDS sensor_cache_inner_mind.bin as input. mind_worker still
    # provides the sensor compute + 23.49 Hz cadence; _start_fast_path's
    # internal if/elif redirects the write target from inner_mind_15d.bin
    # (Phase A+B) to sensor_cache_inner_mind.bin (Phase C). Either flag
    # enables _start_fast_path; the slot identity is decided inside.
    if fast_enabled or l0_rust_enabled:
        try:
            sensor_cache, refresh_threads, shm_writer_thread = _start_fast_path(
                mood_engine, social_graph, media_state, data_dir, session_db,
                config, fast_stop_event,
                lambda: (severity_multipliers, focus_nudges),
                plugin_cache=plugin_cache,
            )
            if fast_enabled:
                logger.info(
                    "[MindWorker] §L1 fast path ON: 5 refresh threads + "
                    "23.49 Hz inner_mind_15d.bin writer (Phase A+B)"
                )
            else:
                logger.info(
                    "[MindWorker] Phase C path ON: 5 refresh threads + "
                    "23.49 Hz sensor_cache_inner_mind.bin writer "
                    "(l0_rust_enabled=true; Rust titan-inner-mind-rs owns "
                    "inner_mind_15d.bin output)"
                )
        except Exception as exc:
            logger.warning(
                "[MindWorker] fast-path init failed (%s); falling back to inline senses",
                exc,
            )
            sensor_cache = None
            refresh_threads = []
            shm_writer_thread = None

    # ── Phase 11 §11.I.2 — SHM slot transition starting → booted ─────
    # MODULE_READY bus emit DELETED per D-SPEC-141 / v1.65.0 locked D2.
    # mind_worker is a shadow under l0_rust_enabled=true (Rust owns
    # inner_mind_15d.bin); the SHM slot tracks the Python shadow's
    # lifecycle so the orchestrator can probe it independently.
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[MindWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[MindWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info("[MindWorker] 5DT cognitive sensors online (fast=%s)",
                bool(sensor_cache))

    # Phase C Session 4 (rFP §4.B.6) — SHM-direct mind_state.bin publisher.
    # Closes the deadlock-prone sync bus.request paths for mind_proxy
    # methods get_mood_label/get_mood_valence/get_current_reward.
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.mind_state_publisher import MindStatePublisher
        from titan_hcl.logic.worker_publisher_runner import (
            run_worker_publisher)
        _mind_state_publisher = MindStatePublisher(titan_id=resolve_titan_id())
        run_worker_publisher(
            publisher=_mind_state_publisher,
            state_fetcher=lambda: mood_engine,
            worker_name="mind_worker",
            cadence_s=1.0,
        )
    except Exception as _pub_init_err:
        logger.error(
            "[MindWorker] mind_state SHM publisher BOOT FAILED — "
            "consumers fall back to sync bus.request: %s",
            _pub_init_err, exc_info=True)

    last_publish = 0.0
    publish_interval = 1.15   # Mind = Schumann/9 (0.87 Hz) — Earth resonance
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
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            now = time.time()
            # Refresh outer-source cache SHM-direct (≤10s staleness) — replaces
            # the retired OUTER_SOURCES_SNAPSHOT broadcast (Phase C / G18).
            if now - _last_plugin_refresh >= 10.0:
                _refresh_plugin_cache_from_shm()
                _last_plugin_refresh = now
            if now - last_publish >= publish_interval:
                tensor = _collect_mind_tensor(
                    mood_engine, social_graph, media_state, data_dir, session_db,
                    severity_multipliers, focus_nudges,
                    cache=sensor_cache,
                )
                _publish_mind_state(
                    send_queue, name, tensor, severity_multipliers,
                    plugin_cache=plugin_cache,
                    data_dir=data_dir, session_db=session_db,
                )
                last_publish = now
                publish_count += 1
                if publish_count % 200 == 0:  # ~3.8 min at 1.15s interval
                    logger.info(
                        "[MindWorker] summary @publish=%d | tensor=[%s] | mults=[%s] | nudges=[%s]",
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
                    "[MindWorker] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[MindWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            # AUDIT §C fix (rFP §P2): flush Spirit-learned weights + focus nudges
            # before exit so a hot-reload / kill-respawn restores them.
            _save_mind_state(config, severity_multipliers, focus_nudges)
            # Stop S7 fast-path threads cleanly so they don't keep
            # writing shm during/after subprocess teardown.
            if refresh_threads or shm_writer_thread:
                from titan_hcl.core.sensor_cache import stop_threads
                _all = list(refresh_threads)
                if shm_writer_thread is not None:
                    _all.append(shm_writer_thread)
                stop_threads(fast_stop_event, _all, timeout_s=2.0)
            break

        # OUTER_SOURCES_SNAPSHOT handler RETIRED (Phase C dissolution 2026-05-22):
        # outer-source cache is now refreshed SHM-direct in the publish branch
        # (_refresh_plugin_cache_from_shm) — the bus broadcast was a G18 violation.

        # Receive FILTER_DOWN severity multipliers from Spirit
        if msg_type == bus.FILTER_DOWN:
            new_mult = msg.get("payload", {}).get("multipliers")
            if new_mult and len(new_mult) == 5:
                severity_multipliers = new_mult
                logger.info("[MindWorker] FILTER_DOWN received: %s",
                            [round(m, 2) for m in severity_multipliers])

        # Phase 10K (rFP §3G / audit §5.3): the FOCUS_NUDGE bus-consumer branch
        # was removed — its sole producer (spirit_loop._run_focus) is deleted and
        # the live focus path is focus_input.bin SHM (FocusPIDPublisher @7.83 Hz),
        # which the Rust trinity daemons read + apply as the cascade directly. The
        # bus message never arrives, so the handler was unreachable dead code.

        # Receive conversation stimulus → compute Mind reflex Intuition
        elif msg_type == bus.CONVERSATION_STIMULUS:
            stimulus = msg.get("payload", {})
            tensor = _collect_mind_tensor(
                mood_engine, social_graph, media_state, data_dir, session_db,
                severity_multipliers, focus_nudges,
                cache=sensor_cache,
            )
            signals = _compute_mind_reflex_intuition(stimulus, tensor, mood_engine, social_graph)
            # REFLEX_SIGNAL broadcast removed — no consumer exists (audit 2026-03-26)

        # Receive Interface input signals (human conversation → cognitive impact)
        elif msg_type == bus.INTERFACE_INPUT:
            iface = msg.get("payload", {})
            # Valence → Touch (dim 4: emotional state)
            valence = iface.get("valence", 0.0)
            if abs(valence) > 0.2:
                # Nudge touch sense toward conversation valence
                focus_nudges[4] = focus_nudges[4] + valence * 0.1
                focus_nudges[4] = max(-0.5, min(0.5, focus_nudges[4]))

            # Engagement → Hearing (dim 1: conversation quality)
            engagement = iface.get("engagement", 0.0)
            if engagement > 0.2:
                focus_nudges[1] = focus_nudges[1] + engagement * 0.08
                focus_nudges[1] = min(0.5, focus_nudges[1])

            # Topic → Taste (dim 2: social signal quality) / Smell (dim 3: environmental)
            topic = iface.get("topic", "general")
            if topic == "social":
                focus_nudges[2] = focus_nudges[2] + 0.05
                focus_nudges[2] = min(0.5, focus_nudges[2])
            elif topic in ("crypto", "philosophy"):
                focus_nudges[3] = focus_nudges[3] + 0.03
                focus_nudges[3] = min(0.5, focus_nudges[3])

            logger.debug("[MindWorker] INTERFACE_INPUT absorbed: valence=%.2f engagement=%.2f topic=%s",
                        valence, engagement, topic)

        # Receive media digest from MediaWorker
        elif msg_type == bus.SENSE_VISUAL:
            features = msg.get("payload", {}).get("features")
            if features and len(features) == 5:
                media_state["last_visual"] = features
                media_state["last_visual_ts"] = time.time()
                logger.info("[MindWorker] Visual digest received: harmony=%.3f", features[4])

        elif msg_type == bus.SENSE_AUDIO:
            features = msg.get("payload", {}).get("features")
            if features and len(features) == 5:
                media_state["last_audio"] = features
                media_state["last_audio_ts"] = time.time()
                logger.info("[MindWorker] Audio digest received: harmony=%.3f", features[4])

        elif msg_type == bus.QUERY:
            from titan_hcl.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            _handle_query(msg, mood_engine, social_graph, media_state,
                         data_dir, session_db, send_queue, name,
                         severity_multipliers, focus_nudges,
                         cache=sensor_cache)

    logger.info("[MindWorker] Exiting")


def _handle_query(msg: dict, mood_engine, social_graph, media_state: dict,
                  data_dir: str, session_db: str, send_queue, name: str,
                  severity_multipliers: list | None = None,
                  focus_nudges: list | None = None,
                  cache=None) -> None:
    """Handle Mind-related queries."""
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        # Phase C Session 5 (rFP §4.D.2): get_tensor / get_mood /
        # get_valence / get_current_reward handlers RETIRED — proxy
        # consumers migrated to SHM-direct via mind_state.bin (Session 4
        # §4.C.2). Static caller graph (G-RPC-4) confirms zero remaining
        # callers.
        #
        # v1.7.1 (D-SPEC-50, rFP_titan_hcl_l2_separation_strategy §4.P):
        # SocialGraph action handlers RETIRED — `record_interaction`,
        # `get_or_create_user`, `should_engage`, `save_profile`, and the
        # legacy G22-violation `get_social_stats` are now served by
        # social_graph_worker (its own subprocess) via SocialGraphProxy.
        # mind_worker no longer hosts SocialGraph; stats read via
        # social_graph_state.bin SHM (G18 path) — see
        # _SocialGraphStatsShmReader above + _sense_taste.
        if action == "get_media_state":
            _send_response(send_queue, name, src, {
                "visual_features": media_state.get("last_visual"),
                "visual_age_s": time.time() - media_state["last_visual_ts"] if media_state["last_visual_ts"] else None,
                "audio_features": media_state.get("last_audio"),
                "audio_age_s": time.time() - media_state["last_audio_ts"] if media_state["last_audio_ts"] else None,
            }, rid)

        else:
            logger.warning("[MindWorker] Unknown action: %s", action)

    except Exception as e:
        logger.error("[MindWorker] Error handling %s: %s", action, e, exc_info=True)
        _send_response(send_queue, name, src, {"error": str(e)}, rid)


# ── Mind Tensor Collection ─────────────────────────────────────────

def _collect_mind_tensor(mood_engine, social_graph, media_state: dict,
                         data_dir: str, session_db: str,
                         severity_multipliers: list | None = None,
                         focus_nudges: list | None = None,
                         cache=None) -> list:
    """
    Collect 5DT Mind tensor with dual-layer perception.

    Each sense = sub_a * weight_a + sub_b * weight_b
    sub_b decays over time (half-life 30 min) when no new media arrives.

    FILTER_DOWN multipliers modulate final sense values:
      - multiplier > 1.0 = amplify deviations from center (more sensitive)
      - multiplier < 1.0 = dampen deviations from center (less sensitive)
    FOCUS nudges apply gentle bias toward center.

    cache (S7 §L1):
      - None  → call _sense_X(...) inline (pre-S7 byte-identical)
      - SensorCache → read sub_a values from cache (populated by
        background refresh threads); sub_b reads from media_state
        directly since SENSE_VISUAL/SENSE_AUDIO bus arrival drives
        media_state at its own natural cadence.
    """
    if severity_multipliers is None:
        severity_multipliers = [1.0] * 5
    if focus_nudges is None:
        focus_nudges = [0.0] * 5

    # Sub_a values — cached if cache present, else inline call.
    if cache is not None:
        vision_a = _read_cached_value(cache, "vision",
                                       lambda: _sense_vision_ambient(data_dir))
        hearing_a = _read_cached_value(cache, "hearing",
                                        lambda: _sense_hearing_ambient(session_db))
        taste = _read_cached_value(cache, "taste",
                                    lambda: _sense_taste(social_graph))
        smell = _read_cached_value(cache, "smell", _sense_smell)
        touch = _read_cached_value(cache, "touch",
                                    lambda: _sense_touch(mood_engine))
    else:
        vision_a = _sense_vision_ambient(data_dir)
        hearing_a = _sense_hearing_ambient(session_db)
        taste = _sense_taste(social_graph)
        smell = _sense_smell()
        touch = _sense_touch(mood_engine)

    # [0] Vision — sub_a (cached or inline) + sub_b (media_state, push-driven)
    vision_b = _get_decayed_feature(media_state, "last_visual", "last_visual_ts", index=4)
    vision = vision_a * _SUB_WEIGHT_A + vision_b * _SUB_WEIGHT_B

    # [1] Hearing — same pattern
    hearing_b = _get_decayed_feature(media_state, "last_audio", "last_audio_ts", index=4)
    hearing = hearing_a * _SUB_WEIGHT_A + hearing_b * _SUB_WEIGHT_B

    # [2] Taste, [3] Smell, [4] Touch — single-source senses (no sub_b yet)

    raw_senses = [vision, hearing, taste, smell, touch]

    # Apply FILTER_DOWN multipliers: amplify/dampen deviation from center
    # multiplier > 1 makes the sense MORE sensitive (deviation amplified)
    # multiplier < 1 makes the sense LESS sensitive (deviation dampened)
    modulated = []
    for i, val in enumerate(raw_senses):
        mult = severity_multipliers[i]
        deviation = val - 0.5
        adjusted = 0.5 + deviation * mult
        # Apply FOCUS nudge (gentle bias)
        nudge = focus_nudges[i] * 0.1
        adjusted += nudge
        modulated.append(max(0.0, min(1.0, adjusted)))

    return [round(v, 4) for v in modulated]


def _get_decayed_feature(media_state: dict, key: str, ts_key: str, index: int) -> float:
    """Get a media digest feature with exponential decay."""
    features = media_state.get(key)
    ts = media_state.get(ts_key, 0.0)

    if features is None or ts == 0.0:
        return 0.5  # Neutral when no media has been digested

    age_s = time.time() - ts
    # Exponential decay: value decays toward 0.5 (neutral) over time
    decay = math.exp(-0.693 * age_s / _DIGEST_HALFLIFE_S)  # 0.693 = ln(2)
    raw_value = features[index]
    # Decayed value approaches 0.5 as decay → 0
    return 0.5 + (raw_value - 0.5) * decay


# ── Individual Senses (sub_a: Ambient) ─────────────────────────────

def _sense_vision_ambient(data_dir: str) -> float:
    """
    Vision sub_a: How fresh is Titan's knowledge?

    Checks research-related file ages in data directory.
    Fresh research = high vision (sees clearly), stale = fading.
    Sigmoid decay with 12h midpoint.
    """
    try:
        research_indicators = [
            os.path.join(data_dir, "research_results.json"),
            os.path.join(data_dir, "last_research.txt"),
        ]

        newest_ts = 0.0
        for path in research_indicators:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                newest_ts = max(newest_ts, mtime)

        # Also check knowledge graph directory
        kuzu_dir = os.path.join(data_dir, "knowledge_graph.kuzu")
        if os.path.isdir(kuzu_dir):
            try:
                for f in os.listdir(kuzu_dir)[:20]:
                    fp = os.path.join(kuzu_dir, f)
                    if os.path.isfile(fp):
                        newest_ts = max(newest_ts, os.path.getmtime(fp))
            except Exception:
                pass

        if newest_ts == 0.0:
            return 0.3  # No research data at all — dim vision

        hours_since = (time.time() - newest_ts) / 3600.0
        # Sigmoid: 1.0 when fresh, 0.5 at 12h, ~0.1 at 36h
        midpoint = 12.0
        k = 0.3  # Steepness
        freshness = 1.0 / (1.0 + math.exp(k * (hours_since - midpoint)))
        return max(0.05, min(1.0, freshness))

    except Exception:
        return 0.5


def _sense_hearing_ambient(session_db: str) -> float:
    """
    Hearing sub_a: How well is Titan connecting with people?

    Checks recent chat session activity from Agno's SQLite sessions DB.
    Active conversations = good hearing, silence = hearing fading.
    """
    try:
        if not os.path.exists(session_db):
            return 0.4  # No session DB — quiet

        import sqlite3
        conn = sqlite3.connect(session_db, timeout=2.0)
        cursor = conn.cursor()

        # Count sessions with activity in last 6 hours
        cutoff = time.time() - 6 * 3600
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE updated_at > ?",
                (cutoff,)
            )
            recent_count = cursor.fetchone()[0]
        except Exception:
            # Table might not exist or have different schema
            recent_count = 0

        conn.close()

        # 0 sessions = 0.2, 5+ sessions = 0.9
        hearing = min(0.9, 0.2 + recent_count * 0.14)
        return max(0.1, hearing)

    except Exception:
        return 0.5


def _sense_taste(social_graph) -> float:
    """Taste: social interaction quality from SocialGraph."""
    if social_graph:
        try:
            stats = social_graph.get_stats()
            users = stats.get("users", 0)
            edges = stats.get("edges", 0)
            return min(1.0, (users / 20.0) * 0.6 + (edges / 10.0) * 0.4)
        except Exception:
            pass
    return 0.5


def _sense_smell() -> float:
    """
    Smell: environmental awareness.

    Combines BonkPulse (crypto sentiment) + WeatherVibe (weather mood)
    with circadian rhythm fallback.
    """
    bonk = _get_bonk_sentiment()
    weather = _get_weather_mood()
    circadian = _get_circadian_rhythm()

    # If both external sources available, use them; otherwise lean on circadian
    sources = []
    if bonk is not None:
        sources.append(bonk)
    if weather is not None:
        sources.append(weather)

    if sources:
        external = sum(sources) / len(sources)
        # 70% external, 30% circadian
        return external * 0.7 + circadian * 0.3
    else:
        return circadian


def _sense_touch(mood_engine) -> float:
    """Touch: emotional state from MoodEngine."""
    if mood_engine:
        try:
            return max(0.0, min(1.0, mood_engine.previous_mood / 100.0))
        except Exception:
            pass
    return 0.5


# ── Smell Sub-Sources ──────────────────────────────────────────────

def _get_bonk_sentiment() -> float | None:
    """Fetch BONK 24h change from CoinGecko. Returns 0-1 or None on failure."""
    try:
        import urllib.request
        import json
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bonk&vs_currencies=usd&include_24hr_change=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Titan/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        change = data.get("bonk", {}).get("usd_24h_change", 0.0)
        # Map -20% to +20% → 0.0 to 1.0
        return max(0.0, min(1.0, 0.5 + change / 40.0))
    except Exception:
        return None


def _get_weather_mood() -> float | None:
    """Fetch weather from Open-Meteo. Returns 0-1 or None on failure."""
    try:
        import urllib.request
        import json
        url = "https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current_weather=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Titan/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        code = data.get("current_weather", {}).get("weathercode", 0)
        if code == 0:
            return 1.0
        elif code <= 3:
            return 0.7
        elif 40 <= code <= 69:
            return 0.3
        else:
            return 0.1
    except Exception:
        return None


def _get_circadian_rhythm() -> float:
    """Time-of-day circadian fallback. Alert during day, mellow at night."""
    hour = datetime.now().hour
    # Peak alertness 8am-6pm, lowest 2am-5am
    if 8 <= hour <= 18:
        return 0.7 + 0.2 * math.sin(math.pi * (hour - 8) / 10)
    elif 2 <= hour <= 5:
        return 0.2
    else:
        # Evening/early morning transition
        return 0.4


# ── Publishing & Messaging ─────────────────────────────────────────

def _publish_mind_state(send_queue, name: str, tensor: list,
                        severity_multipliers: list | None = None,
                        hormone_levels: dict | None = None,
                        plugin_cache: dict | None = None,
                        data_dir: str | None = None,
                        session_db: str | None = None) -> None:
    """Publish MIND_STATE to the bus with 5D legacy + 15D extended.

    rFP_trinity_130d_awakening §12.4 / SPEC §23.5 — pulls
    interaction_quality and assessment_quality from plugin_cache (populated
    by OUTER_SOURCES_SNAPSHOT) so inner_mind feeling[6] inner_touch and
    feeling[8] inner_taste read real upstream values instead of defaulting
    to 0.5. hormone_levels for willing[10-14] also pulled from cache when
    not passed directly.
    """
    center = [0.5] * 5
    center_dist = sum((t - c) ** 2 for t, c in zip(tensor, center)) ** 0.5

    payload = {
        "dims": 5,
        "values": tensor,  # Legacy 5D (backward compatible)
        "delta": [round(t - 0.5, 4) for t in tensor],
        "center_dist": round(center_dist, 4),
    }
    if severity_multipliers:
        payload["filter_down_multipliers"] = [round(m, 4) for m in severity_multipliers]

    # DQ2: Extended 15D Mind tensor (Thinking + Feeling + Willing)
    try:
        from titan_hcl.logic.mind_tensor import collect_mind_15d

        # Pull rich-producer values from plugin_cache (Phase 1 wiring).
        cache = plugin_cache or {}
        # interaction_quality ← social_perception_stats.sentiment_ema
        soc_perc = cache.get("social_perception_stats") or {}
        interaction_quality = float(soc_perc.get("sentiment_ema", 0.5))
        # assessment_quality ← assessment_stats.average_score
        assess = cache.get("assessment_stats") or {}
        assessment_quality = float(assess.get("average_score", 0.5))
        # hormones: prefer explicit kwarg, fall back to cache (broadcast by
        # plugin every 10s via OUTER_SOURCES_SNAPSHOT.hormone_levels).
        hormones = hormone_levels if hormone_levels else cache.get("hormone_levels")

        # Phase 2 (rFP_trinity_130d_awakening §4.4 + SPEC §23.5):
        # Build audio_state / visual_state / ambient_change. Per SPEC:
        #   inner_hearing = 0.5*min(1,audio_creates_recent/5) + 0.5*sense_hearing_ambient
        #   inner_sight   = 0.5*min(1,art_creates_recent/5)   + 0.5*sense_vision_ambient
        #   inner_smell   = ambient_change (rolling stddev cpu+circ)
        # creates_recent + ambient_change come from plugin-side
        # InnerPerceptionState (broadcast via inner_perception_stats);
        # the *_ambient sub-signals come from mind_worker's own producers
        # since data_dir / session_db live in this worker's scope.
        ip_stats = cache.get("inner_perception_stats") or {}
        audio_state = {
            "creates_recent": int(
                (ip_stats.get("audio_state") or {}).get("creates_recent", 0)),
            "ambient": (
                _sense_hearing_ambient(session_db) if session_db else 0.5),
        }
        visual_state = {
            "creates_recent": int(
                (ip_stats.get("visual_state") or {}).get("creates_recent", 0)),
            "ambient": (
                _sense_vision_ambient(data_dir) if data_dir else 0.5),
        }
        ambient_change = float(ip_stats.get("ambient_change", 0.0) or 0.0)

        mind_15d = collect_mind_15d(
            current_5d=tensor,
            audio_state=audio_state,
            interaction_quality=interaction_quality,
            visual_state=visual_state,
            assessment_quality=assessment_quality,
            ambient_change=ambient_change,
            hormone_levels=hormones,
        )
        payload["values_15d"] = [round(v, 4) for v in mind_15d]
        payload["dims_extended"] = 15
    except Exception:
        pass

    _send_msg(send_queue, bus.MIND_STATE, name, "all", payload)


class _MetabolismStub:
    """Lightweight metabolism stub for standalone MoodEngine operation."""
    _last_balance = 1.0

    async def get_current_state(self):
        return "HIGH_ENERGY"

    async def get_learning_velocity(self):
        return 0.5

    async def get_social_density(self):
        return 0.5

    async def get_metabolic_health(self):
        return 1.0

    async def get_directive_alignment(self):
        return 0.5


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


# Phase 10J — Mind reflex intuition relocated to logic/mind_helpers.py (pure,
# torch/cgn-free) so agno_hooks (via logic/reflex_intuition) no longer imports
# from this worker module body per SPEC §11.B.4. mind_worker self-imports it
# back for its own internal reflex callsite.
from titan_hcl.logic.mind_helpers import _compute_mind_reflex_intuition  # noqa: E402


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
    if _state_writer is not None and _WORKER_READY:
        try:
            _state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass


# ── S7 §L1 fast-path helpers ────────────────────────────────────────


def _read_cached_value(cache, name: str, fallback_fn) -> float:
    """
    Read a single float sub_a value from cache. Cache stores
    {"value": float, ...}. Defensive fallback to inline sense fn if
    cache cold (boot warmup makes this rare in practice).
    """
    cached = cache.get(name)
    if cached is not None and "value" in cached:
        return float(cached["value"])
    return float(fallback_fn())


def _read_flag(config: dict, dotted_path: str, default: bool) -> bool:
    """Resolve a dotted feature flag path through the worker's config."""
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


def _start_fast_path(mood_engine, social_graph, media_state, data_dir,
                     session_db, config, stop_event, get_modulators,
                     plugin_cache: dict | None = None) -> tuple:
    """
    Start the §L1 fast-path threads for mind. Returns:
      (sensor_cache, refresh_threads, shm_writer_thread)

    get_modulators: callable returning current (severity_multipliers,
    focus_nudges) so the shm writer always uses live values.

    Synchronous warmup runs BEFORE refresh threads start.
    """
    from titan_hcl.core.sensor_cache import (
        RefreshSpec, SensorCache, start_refresh_threads, start_shm_writer_thread,
    )
    from titan_hcl.core.state_registry import INNER_MIND_15D, RegistryBank

    # Wrap each sense fn so the cache stores {"value": float} dicts
    # — uniform shape across body and mind, lets the same SensorCache
    # primitive serve both daemons.
    def _wrap(fn) -> dict:
        return {"value": float(fn())}

    initial = {
        "vision":  _wrap(lambda: _sense_vision_ambient(data_dir)),
        "hearing": _wrap(lambda: _sense_hearing_ambient(session_db)),
        "taste":   _wrap(lambda: _sense_taste(social_graph)),
        "smell":   _wrap(_sense_smell),
        "touch":   _wrap(lambda: _sense_touch(mood_engine)),
    }
    cache = SensorCache(initial=initial)

    specs = [
        RefreshSpec(
            name="vision",
            refresh_fn=lambda: _wrap(lambda: _sense_vision_ambient(data_dir)),
            period_s=_MIND_REFRESH_PERIODS_S["vision"],
        ),
        RefreshSpec(
            name="hearing",
            refresh_fn=lambda: _wrap(lambda: _sense_hearing_ambient(session_db)),
            period_s=_MIND_REFRESH_PERIODS_S["hearing"],
        ),
        RefreshSpec(
            name="taste",
            refresh_fn=lambda: _wrap(lambda: _sense_taste(social_graph)),
            period_s=_MIND_REFRESH_PERIODS_S["taste"],
        ),
        RefreshSpec(
            name="smell",
            refresh_fn=lambda: _wrap(_sense_smell),
            period_s=_MIND_REFRESH_PERIODS_S["smell"],
        ),
        RefreshSpec(
            name="touch",
            refresh_fn=lambda: _wrap(lambda: _sense_touch(mood_engine)),
            period_s=_MIND_REFRESH_PERIODS_S["touch"],
        ),
    ]
    refresh_threads = start_refresh_threads(
        specs, cache, stop_event, thread_name_prefix="mind_refresh",
    )

    # Two modes per Phase A+B vs Phase C — same architecture as body_worker
    # post-l0_rust flag flip. See body_worker.py inline comment for context.
    # SPEC §G1 + §23.5 + §9.A line 1034.
    shm_bank = RegistryBank(titan_id=None, config=config)
    shm_writer_thread = None

    def _compute_mind_15d(
        severity_multipliers: list, focus_nudges: list,
    ) -> "np.ndarray":
        """Single canonical compute fn — used by both Phase A+B writer +
        Phase C sensor cache writer so they produce byte-identical output.
        Reuses the SPEC §23.5 collect_mind_15d formula.
        """
        import numpy as np
        tensor_5d = _collect_mind_tensor(
            mood_engine, social_graph, media_state, data_dir, session_db,
            severity_multipliers, focus_nudges,
            cache=cache,
        )
        try:
            from titan_hcl.logic.mind_tensor import collect_mind_15d
            pc = plugin_cache or {}
            ip_stats = pc.get("inner_perception_stats") or {}
            soc_perc = pc.get("social_perception_stats") or {}
            assess = pc.get("assessment_stats") or {}
            _audio_state = {
                "creates_recent": int(
                    (ip_stats.get("audio_state") or {}).get("creates_recent", 0)),
                "ambient": (
                    _sense_hearing_ambient(session_db) if session_db else 0.5),
            }
            _visual_state = {
                "creates_recent": int(
                    (ip_stats.get("visual_state") or {}).get("creates_recent", 0)),
                "ambient": (
                    _sense_vision_ambient(data_dir) if data_dir else 0.5),
            }
            tensor_15d = collect_mind_15d(
                current_5d=tensor_5d,
                audio_state=_audio_state,
                interaction_quality=float(soc_perc.get("sentiment_ema", 0.5)),
                visual_state=_visual_state,
                assessment_quality=float(assess.get("average_score", 0.5)),
                ambient_change=float(ip_stats.get("ambient_change", 0.0) or 0.0),
                hormone_levels=pc.get("hormone_levels"),
            )
        except Exception:
            # Defensive — if the 15D extension fails, write the 5D base
            # padded with neutral 0.5s so consumers always see a valid
            # 15D vector.
            tensor_15d = list(tensor_5d) + [0.5] * 10
        return np.asarray(tensor_15d, dtype="<f4")

    if shm_bank.is_enabled(INNER_MIND_15D):
        # Phase A+B path — direct write to inner_mind_15d.bin.
        mind_15d_writer = shm_bank.writer(INNER_MIND_15D)

        def tick():
            severity_multipliers, focus_nudges = get_modulators()
            arr = _compute_mind_15d(severity_multipliers, focus_nudges)
            if arr.shape == (15,):
                mind_15d_writer.write(arr)

        shm_writer_thread = start_shm_writer_thread(
            tick, _MIND_TICK_PERIOD_S, stop_event, "mind_shm_writer",
        )
    elif (config or {}).get("microkernel", {}).get("l0_rust_enabled", False):
        # Phase C path — Sprint 6 §4.5 FULL Rust formula port:
        # Publish msgpack source dict of RAW INPUTS only. Rust
        # `project_inner_mind_15d` (titan-inner-mind-rs) decodes the
        # inputs and computes the 15D per SPEC §23.5 collect_mind_15d.
        # No tensor compute on the Python side.
        try:
            from titan_hcl.logic.inner_mind_sensor_refresh import (
                InnerMindSensorRefresh)
            import msgpack as _msgpack

            def _provide_mind_source_dict() -> bytes:
                # Raw 5D thinking tensor (pre-substrate). Severity = 1.0
                # + nudges = 0.0 so substrate-correct values flow from
                # _collect_mind_tensor without local modulation; Rust
                # daemon applies UNIFIED⊗LOCAL filter_down + ground_up
                # (willing only per SPEC §G10) downstream.
                tensor_5d = _collect_mind_tensor(
                    mood_engine, social_graph, media_state, data_dir,
                    session_db,
                    severity_multipliers=[1.0] * 15,
                    focus_nudges=[0.0] * 15,
                    cache=cache,
                )
                pc = plugin_cache or {}
                ip_stats = pc.get("inner_perception_stats") or {}
                soc_perc = pc.get("social_perception_stats") or {}
                assess = pc.get("assessment_stats") or {}
                audio_state = {
                    "creates_recent": int(
                        (ip_stats.get("audio_state") or {}).get(
                            "creates_recent", 0)),
                    "ambient": (
                        _sense_hearing_ambient(session_db)
                        if session_db else 0.5),
                }
                visual_state = {
                    "creates_recent": int(
                        (ip_stats.get("visual_state") or {}).get(
                            "creates_recent", 0)),
                    "ambient": (
                        _sense_vision_ambient(data_dir)
                        if data_dir else 0.5),
                }
                payload = {
                    "thinking_5d": [float(v) for v in tensor_5d[:5]],
                    "audio_state": audio_state,
                    "interaction_quality": float(
                        soc_perc.get("sentiment_ema", 0.5)),
                    "visual_state": visual_state,
                    "assessment_quality": float(
                        assess.get("average_score", 0.5)),
                    "ambient_change": float(
                        ip_stats.get("ambient_change", 0.0) or 0.0),
                    "hormone_levels": pc.get("hormone_levels") or {},
                }
                return _msgpack.packb(payload, use_bin_type=True)

            sensor_cache_writer = InnerMindSensorRefresh(
                tensor_provider=_provide_mind_source_dict,
                titan_id=None,
            )
            shm_writer_thread = sensor_cache_writer.start_thread(stop_event)
            logger.info(
                "[MindWorker] sensor_cache_inner_mind.bin raw-inputs writer "
                "started (l0_rust_enabled=true; Sprint 6 §4.5 full Rust "
                "formula port; Python publishes thinking_5d + audio_state + "
                "interaction_quality + visual_state + assessment_quality + "
                "ambient_change + hormone_levels; Rust titan-inner-mind-rs "
                "computes 15D via project_inner_mind_15d + substrate)")
        except Exception:
            logger.critical(
                "[MindWorker] failed to start inner_mind sensor cache writer — "
                "Rust inner-mind-rs will starve on constant-zero input. "
                "Investigate before relying on inner_mind_15d.bin downstream.",
                exc_info=True)

    return cache, refresh_threads, shm_writer_thread
