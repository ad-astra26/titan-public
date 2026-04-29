"""
EMOT-CGN Worker — Guardian-supervised L2 module for emotional grounding.

The 8th CGN consumer. Standalone subprocess per rFP_emot_cgn_v2.md §10 ADR
(2026-04-20: locked standalone-worker, no in-process fallback permitted) and
`memory/feedback_architectural_decisions_no_drift.md`. Scaffold follows the
exact structure of `titan_plugin/modules/language_worker.py`.

Owns (when Phase 1.6e lands): EmotCGNConsumer, EmotionClusterer, HAOV
hypotheses, β-posterior, cluster state, CGN worker registration +
transitions, `/dev/shm/titan/emot_state.bin` shm-mirror writes.

Bus protocol (full wire shipped across Phase 1.6 sub-phases):
  EMOT_CHAIN_EVIDENCE   (meta_reasoning → emot_cgn)  — per chain conclude
                                                        (Phase 1.6d)
  FELT_CLUSTER_UPDATE   (spirit → emot_cgn)          — per felt-tensor emit
                                                        (Phase 1.6d)
  EMOT_STATE_QUERY      (consumer → emot_cgn)        — optional bus fallback
                                                        when shm-mirror unavailable
                                                        (Phase 1.6d)
  EMOT_STATE_RESP       (emot_cgn → consumer)        — response to query
                                                        (Phase 1.6d)
  CGN_REGISTER          (emot_cgn → cgn)             — 8th consumer registration
                                                        (Phase 1.6e — from
                                                        existing EmotCGNConsumer)
  CGN_TRANSITION        (emot_cgn → cgn)             — per primitive update
                                                        (Phase 1.6e)
  CGN_CROSS_INSIGHT     (emot_cgn ↔ all)             — bidirectional cross-
                                                        consumer insight flow
                                                        (Phase 1.6e — from
                                                        existing EmotCGNConsumer)
  EMOT_CGN_SIGNAL       (emot_cgn → all)             — graduation / rollback /
                                                        cluster emergence events
                                                        (Phase 1.6e)

STATE reads by downstream consumers (narrator, social, dream, META-CGN,
dashboard) happen via `/dev/shm/titan/emot_state.bin` shm-mirror —
zero-copy mmap, no bus roundtrip. Bus is for EVENTS, shm is for STATE
(per rFP_microkernel_v2_shadow_core.md §State Registries).

Entry point: emot_cgn_worker_main(recv_queue, send_queue, name, config)

Phase 1.6a (this commit): SCAFFOLD ONLY — subprocess boot + heartbeat +
MODULE_SHUTDOWN handler + TBD dispatch for future bus messages. No
EmotCGNConsumer yet; that arrives in Phase 1.6e with the migration
from `meta_reasoning.py`. Shadow-mode EMOT-CGN continues to run in
meta_reasoning until 1.6e swaps ownership.
"""
import logging
import os
import sys
import threading
import time
from titan_plugin.utils.silent_swallow import swallow_warn
from titan_plugin import bus

logger = logging.getLogger(__name__)


def emot_cgn_worker_main(recv_queue, send_queue, name: str,
                         config: dict) -> None:
    """Main loop for the EMOT-CGN module process.

    Args:
        recv_queue: receives messages from DivineBus (bus→worker)
        send_queue: sends messages back to DivineBus (worker→bus)
        name: module name ("emot_cgn")
        config: dict from [emot_cgn] config section
    """
    from queue import Empty

    # Project root for imports (same pattern as language_worker)
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[EmotCGNWorker] Initializing emotion subsystem...")
    init_start = time.time()

    cfg = config or {}
    titan_id = cfg.get("titan_id", "T1")
    save_dir = cfg.get("emot_cgn_save_dir", "data/emot_cgn")

    # ── Instantiate EmotCGNConsumer (moved from meta_reasoning, Phase 1.6e.1)
    from titan_plugin.logic.emot_cgn import EmotCGNConsumer
    from titan_plugin.logic.emotion_cluster import EMOT_PRIMITIVE_INDEX
    from titan_plugin.logic.emot_shm_protocol import (
        ShmEmotWriter, DEFAULT_STATE_PATH, DEFAULT_GROUNDING_PATH)
    # v3 bundle — Titan's whole being, lossless (rFP §19).
    from titan_plugin.logic.emot_bundle_protocol import (
        BundleWriter, default_bundle_path, REGION_UNCLUSTERED, GRAD_SHADOW)
    from titan_plugin.logic.emot_thin_encoder import ThinEmotEncoder

    emot_cgn = EmotCGNConsumer(
        send_queue=send_queue,
        titan_id=titan_id,
        save_dir=save_dir,
        module_name=name,  # "emot_cgn" — dst for bus routing
    )
    # Shm paths config-driven so tests can redirect to tmp dirs without
    # touching /dev/shm/titan/ (which is shared with real running Titans).
    shm_state_path = cfg.get("shm_state_path", DEFAULT_STATE_PATH)
    shm_grounding_path = cfg.get("shm_grounding_path", DEFAULT_GROUNDING_PATH)
    shm_writer = ShmEmotWriter(state_path=shm_state_path,
                               grounding_path=shm_grounding_path)

    # v3 bundle writer (native-first 168D + side channels + reserved L5 slots).
    # Runs alongside shm_writer — legacy consumers keep reading emot_state.bin
    # via dominant_idx; v3-aware consumers read bundle via region_id.
    shm_bundle_path = cfg.get("shm_bundle_path", default_bundle_path(titan_id))
    bundle_writer = BundleWriter(path=shm_bundle_path, titan_id=titan_id)
    thin_encoder = ThinEmotEncoder(titan_id=titan_id)

    # v3 region clusterer — density-based emergent emotion over the 208D
    # state vector. Assigns region_id per observation; reclusters every
    # RECLUSTER_INTERVAL_S if buffer has enough data.
    from titan_plugin.logic.emot_region_clusterer import (
        RegionClusterer, assemble_state_vec)
    region_clusterer = RegionClusterer(save_dir=save_dir)
    RECLUSTER_INTERVAL_S = float(
        cfg.get("emot_recluster_interval_s", 900.0))  # 15 min default
    _last_recluster_ts = time.time()

    # BUG #10 fix (2026-04-24): legacy 8-primitive k-means recenter
    # invocation tick. Pre-Phase-1.6h, dreaming.py::end_dreaming() invoked
    # emot_cgn._clusterer.maybe_recenter(force=False) at dream-cycle end.
    # The Phase 1.6h cutover moved EMOT-CGN into this subprocess; the
    # dreaming.py wire still exists but `emot_cgn` param is always None
    # at both inner_coordinator.py call sites (:201, :296), so the inner
    # invocation is dead code. Consequence pre-fix: last_recenter_ts=0 on
    # all 3 Titans since v3 shipped 2026-04-20 → RNG-seeded legacy
    # primitive centroids NEVER drifted → WONDER monoculture locked in +
    # LOVE never fires. Fix: invoke maybe_recenter() periodically here
    # (internal 7-day gate in maybe_recenter() provides real throttling
    # per Q3 audit TRANSITIONAL directive — DO NOT tune the interval,
    # only restore as-designed invocation cadence).
    K_RECENTER_CHECK_INTERVAL_S = float(
        cfg.get("emot_k_recenter_check_interval_s", 3600.0))  # hourly check
    _last_k_recenter_check_ts = 0.0

    # F-phase routine emotional pulse — fires regardless of cluster
    # ambiguity so EmotMeta meets OBS-meta-service-session2-soak coverage
    # (≥1 META_REASON_REQUEST per consumer per 24h). Complementary to the
    # ambig-anchor branch in FELT_CLUSTER_UPDATE which only fires when
    # cluster_conf < 0.5. Maker decision 2026-04-28: 120s cadence.
    EMOT_META_PULSE_INTERVAL_S = float(
        cfg.get("emot_meta_pulse_interval_s", 120.0))
    _last_emot_meta_pulse_ts = 0.0

    # v3 kin protocol — Titan↔Titan emotion transmission (rFP §21).
    # Worker emits KIN_EMOT_STATE on region change (throttled) and
    # caches incoming peer states for MSL binding.
    from titan_plugin.logic.emot_kin_protocol import (
        KIN_EMOT_STATE_MSG_TYPE, build_kin_emot_state_payload,
        parse_kin_emot_state, compute_msl_activations,
        DEFAULT_EMIT_INTERVAL_S)
    KIN_EMIT_INTERVAL_S = float(
        cfg.get("emot_kin_emit_interval_s", DEFAULT_EMIT_INTERVAL_S))
    kin_state = {
        "last_emit_ts": 0.0,
        "last_emitted_signature": 0,
        "peer_states": {},  # titan_src → parsed payload
    }

    def _freshest_peer_state():
        """Return the freshest non-stale peer KIN_EMOT_STATE payload,
        or None. Freshness window already enforced at ingest time via
        parse_kin_emot_state — this just picks the most-recent of the
        cached peers (typically 1-2 kin Titans)."""
        peers = kin_state.get("peer_states", {})
        if not peers:
            return None
        freshest = None
        freshest_ts = 0
        for pid, p in peers.items():
            if p.get("peer_ts_ms", 0) > freshest_ts:
                freshest_ts = p["peer_ts_ms"]
                freshest = p
        return freshest

    def _maybe_emit_kin_state(region_id: int, region_sig: int,
                              region_conf: float, region_res_s: float,
                              valence: float, arousal: float,
                              novelty: float, legacy_idx: int,
                              regions_emerged: int):
        """Emit KIN_EMOT_STATE on region change or after KIN_EMIT_INTERVAL_S.
        Throttled — avoids chatter when Titan sits in the same region.
        Signature changes immediately force an emission (region edge).
        """
        now = time.time()
        time_ok = (now - kin_state["last_emit_ts"]) >= KIN_EMIT_INTERVAL_S
        edge_ok = (region_sig != 0
                   and region_sig != kin_state["last_emitted_signature"])
        if not (time_ok or edge_ok):
            return
        try:
            payload = build_kin_emot_state_payload(
                titan_src=titan_id,
                region_id=region_id,
                region_signature=region_sig,
                region_confidence=region_conf,
                region_residence_s=region_res_s,
                regions_emerged=regions_emerged,
                valence=valence, arousal=arousal, novelty=novelty,
                legacy_idx=legacy_idx,
                encoder_id=thin_encoder.encoder_id(),
                ts_ms=int(now * 1000),
            )
            _send_msg(send_queue, KIN_EMOT_STATE_MSG_TYPE, name, "all",
                      payload)
            kin_state["last_emit_ts"] = now
            kin_state["last_emitted_signature"] = region_sig
        except Exception as _ke:
            swallow_warn('[EmotCGNWorker] KIN_EMOT_STATE emit failed', _ke,
                         key="modules.emot_cgn_worker.kin_emot_state_emit_failed", throttle=100)

    def _prime_hormone_levels_from_history():
        """Warm-start hormone_levels EMA from the latest inner_memory
        snapshot so bundle writes don't see an all-zero hormone dim
        during the gap between worker boot and the first HORMONE_FIRED
        event. Read-only direct SQLite query — inner_memory's writer
        (IMW) can coexist. Silent on failure (still starts from zeros).
        """
        try:
            import sqlite3
            import json as _json
            from titan_plugin.logic.emot_bundle_protocol import NS_PROGRAMS
            db_path = cfg.get("inner_memory_db_path")
            if not db_path:
                db_path = os.path.join(project_root, "data", "inner_memory.db")
            if not os.path.exists(db_path):
                return
            conn = sqlite3.connect(f"file:{db_path}?mode=ro",
                                   uri=True, timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT levels FROM hormone_snapshots "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if not row or not row[0]:
                return
            levels_dict = _json.loads(row[0])
            primed = [float(levels_dict.get(p, 0.0)) for p in NS_PROGRAMS]
            worker_state["last_hormone_levels_11d"] = primed
            n_nonzero = sum(1 for v in primed if v > 0.0)
            logger.info("[EmotCGNWorker] Primed hormone_levels from "
                        "inner_memory snapshot (%d/%d programs non-zero)",
                        n_nonzero, len(primed))
        except Exception as _e:
            swallow_warn('[EmotCGNWorker] hormone prime skipped', _e,
                         key="modules.emot_cgn_worker.hormone_prime_skipped", throttle=100)

    # Per-worker closure state — caches of most recent inputs so
    # _write_state_to_bundle can assemble a full snapshot on any event.
    # Native-dim caches default to None (encoder pads zeros on miss).
    worker_state = {
        "last_felt_tensor_130d": None,
        "last_trajectory_2d": None,
        "last_space_topology_30d": None,
        "last_neuromod_6d": None,
        "last_hormone_levels_11d": None,
        "last_ns_urgencies_11d": None,
        "last_cgn_beta_states_8d": None,
        "last_msl_activations_6d": None,
        "last_pi_phase_6d": None,  # schema v2: all 6 sphere_clock phases
        "last_terminal_reward": None,
    }
    # Warm-start hormone levels from the latest snapshot so bundle isn't
    # all-zeros during the gap between boot and first HORMONE_FIRED event.
    _prime_hormone_levels_from_history()

    def _write_state_to_shm():
        """Update BOTH shm mirrors after any state-changing event.

        Legacy emot_state.bin: 64B, dominant_idx among 8 v2 primitives —
        kept for narrator/social/dashboard/meta_reasoning compatibility
        throughout v3 transition.

        v3 bundle: 2048B, Titan's whole native consciousness (168D) +
        side channels + reserved L5 latents + derived fields. Primary
        substrate for emergent-emotion HDBSCAN, kin transmission,
        consumer β-context plug-ins.
        """
        # ── Legacy shm (unchanged — keeps all current readers working).
        try:
            dom = emot_cgn.get_dominant_emotion()
            dom_idx = EMOT_PRIMITIVE_INDEX.get(dom, 0)
            prim = emot_cgn._primitives.get(dom)
            v_beta = float(prim.V) if prim else 0.5
            _, cluster_dist, cluster_conf = emot_cgn._last_cluster_assignment
            shm_writer.write_state(
                dominant_idx=dom_idx,
                is_active=emot_cgn.is_active(),
                cgn_registered=bool(emot_cgn._cgn_registered),
                V_beta=v_beta, V_blended=v_beta,
                cluster_confidence=float(cluster_conf),
                cluster_distance=float(cluster_dist),
                total_updates=int(emot_cgn._total_updates),
                cross_insights_sent=int(emot_cgn._cgn_cross_insights_sent),
                cross_insights_received=int(emot_cgn._cgn_cross_insights_received),
            )
        except Exception as e:
            swallow_warn('[EmotCGNWorker] shm write_state failed', e,
                         key="modules.emot_cgn_worker.shm_write_state_failed", throttle=100)
            dom_idx = 0  # fall through to bundle write with neutral idx

        # ── v3 bundle — native Titan state + derived emergence fields.
        try:
            # MSL binding: compute I/YOU/ME/WE/YES/NO from self + most
            # recent peer state (if any, non-stale). Updates
            # worker_state["last_msl_activations_6d"] BEFORE encoder
            # reads it so the bundle reflects current binding.
            # Uses previous tick's region info (self_region_signature
            # and self_region_confidence from last _write_state_to_shm).
            try:
                peer_state = _freshest_peer_state()
                msl_vec = compute_msl_activations(
                    self_region_confidence=float(
                        kin_state.get("self_region_confidence", 0.0)),
                    self_region_signature=int(
                        kin_state.get("self_region_signature", 0)),
                    self_valence=float(
                        kin_state.get("self_valence", 0.0)),
                    peer_state=peer_state,
                )
                worker_state["last_msl_activations_6d"] = msl_vec
            except Exception as _msl_err:
                swallow_warn('[EmotCGNWorker] MSL binding error', _msl_err,
                             key="modules.emot_cgn_worker.msl_binding_error", throttle=100)

            encoded = thin_encoder.encode(
                felt_tensor_130d=worker_state["last_felt_tensor_130d"],
                trajectory_2d=worker_state["last_trajectory_2d"],
                space_topology_30d=worker_state["last_space_topology_30d"],
                neuromod_state_6d=worker_state["last_neuromod_6d"],
                hormone_levels_11d=worker_state["last_hormone_levels_11d"],
                ns_urgencies_11d=worker_state["last_ns_urgencies_11d"],
                cgn_beta_states_8d=worker_state["last_cgn_beta_states_8d"],
                msl_activations_6d=worker_state["last_msl_activations_6d"],
                pi_phase_6d=worker_state["last_pi_phase_6d"],
                last_terminal_reward=worker_state["last_terminal_reward"],
            )
            # HDBSCAN region assignment over 210D state vector (schema v2).
            state_vec = assemble_state_vec(encoded)
            region_id, region_conf, region_residence_s, region_sig = (
                region_clusterer.observe(state_vec))
            # Cache for next tick's MSL binding (needs prior self-region info).
            kin_state["self_region_signature"] = region_sig
            kin_state["self_region_confidence"] = region_conf
            kin_state["self_valence"] = encoded["valence"]
            bundle_writer.write(
                encoder_id=thin_encoder.encoder_id(),
                felt_tensor_130d=encoded["felt_tensor_130d"],
                trajectory_2d=encoded["trajectory_2d"],
                space_topology_30d=encoded["space_topology_30d"],
                neuromod_state_6d=encoded["neuromod_state_6d"],
                hormone_levels_11d=encoded["hormone_levels_11d"],
                ns_urgencies_11d=encoded["ns_urgencies_11d"],
                cgn_beta_states_8d=encoded["cgn_beta_states_8d"],
                msl_activations_6d=encoded["msl_activations_6d"],
                pi_phase_6d=encoded["pi_phase_6d"],
                # L5 slots zero until Phase 0 encoder ships.
                region_id=region_id,
                legacy_idx=dom_idx,  # matches legacy state.bin dominant_idx
                # Phase B (rFP §23.5): graduation state machine driven by
                # region persistence across reclusters + observation age +
                # named-region gate. Falls back to GRAD_SHADOW if clusterer
                # query fails (defensive: never block a bundle write).
                graduation_status=(
                    region_clusterer.graduation_status()
                    if hasattr(region_clusterer, "graduation_status")
                    else GRAD_SHADOW),
                regions_emerged=region_clusterer.regions_count(),
                valence=encoded["valence"],
                arousal=encoded["arousal"],
                novelty=encoded["novelty"],
                region_confidence=region_conf,
                region_residence_s=region_residence_s,
                region_signature=region_sig,
            )
            # Emit KIN_EMOT_STATE (throttled / edge-triggered) so peer
            # Titans can ground their YOU/WE MSL concepts on our state.
            _maybe_emit_kin_state(
                region_id=region_id, region_sig=region_sig,
                region_conf=region_conf,
                region_res_s=region_residence_s,
                valence=encoded["valence"], arousal=encoded["arousal"],
                novelty=encoded["novelty"], legacy_idx=dom_idx,
                regions_emerged=region_clusterer.regions_count(),
            )
        except Exception as e:
            swallow_warn('[EmotCGNWorker] bundle write failed', e,
                         key="modules.emot_cgn_worker.bundle_write_failed", throttle=100)

    _write_state_to_shm()  # initial snapshot so consumers see valid data immediately

    init_ms = (time.time() - init_start) * 1000
    logger.info("[EmotCGNWorker] Ready in %.0fms (titan=%s, status=%s, "
                "primitives=%d, shm_state=%s bundle=%s)",
                init_ms, titan_id, emot_cgn._status,
                len(emot_cgn._primitives), shm_state_path, shm_bundle_path)

    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})

    # ── F-phase (rFP §16.7): Meta-Reasoning Consumer Service wire ────
    # Session 2 wire-now-gate-later. Meta returns not_yet_implemented;
    # emot_cgn falls back to existing centroid-distance assignment.
    _emot_meta_pending: dict = {}  # request_id → (t_sent, context_tag)
    try:
        from titan_plugin.logic.meta_service_client import (
            register_response_handler as _em_register_mrh,
        )

        def _em_meta_response_handler(payload: dict) -> None:
            req_id = payload.get("request_id", "")
            failure = payload.get("failure_mode")
            if failure:
                logger.info(
                    "[EmotMeta] response req_id=%s failure=%s "
                    "(dry-run expected)", req_id[:8], failure)
            else:
                insight = payload.get("insight") or {}
                logger.info(
                    "[EmotMeta] response req_id=%s sugg=%s",
                    req_id[:8],
                    insight.get("suggested_action") if insight else None)
            _emot_meta_pending.pop(req_id, None)

        _em_register_mrh("emotional", _em_meta_response_handler)
        logger.info("[EmotCGNWorker] F-phase meta response handler registered")
    except Exception as _emh_err:
        logger.warning(
            "[EmotCGNWorker] Meta response handler registration: %s",
            _emh_err)

    # ── Background heartbeat thread (prevents Guardian timeout) ──────
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            _hb_stop.wait(30.0)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                 name="emot-cgn-heartbeat")
    hb_thread.start()

    # ── Main loop ────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_plugin.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        _send_heartbeat(send_queue, name)

        # Periodic region re-clustering (rFP §19). HDBSCAN over the
        # rolling trajectory — naturally aligns with dream-cycle cadence
        # but decoupled from dreaming.py to keep subsystems independent.
        now = time.time()
        if now - _last_recluster_ts >= RECLUSTER_INTERVAL_S:
            _last_recluster_ts = now
            try:
                region_clusterer.recluster()
            except Exception as _re:
                swallow_warn('[EmotCGNWorker] recluster error', _re,
                             key="modules.emot_cgn_worker.recluster_error", throttle=100)

        # BUG #10 fix: legacy 8-primitive k-means recenter — check hourly,
        # internal 7-day gate in maybe_recenter() handles real throttling.
        # Restores as-designed cadence lost in Phase 1.6h cutover.
        if now - _last_k_recenter_check_ts >= K_RECENTER_CHECK_INTERVAL_S:
            _last_k_recenter_check_ts = now
            try:
                fired = emot_cgn._clusterer.maybe_recenter(force=False)
                if fired:
                    logger.info(
                        "[EmotCGNWorker] legacy k-means recenter fired "
                        "(BUG #10 fix path)")
            except Exception as _ke:
                swallow_warn('[EmotCGNWorker] k-recenter error', _ke,
                             key="modules.emot_cgn_worker.k_recenter_error", throttle=100)

        # ── F-phase routine emotional pulse (rFP §16.7, Maker 2026-04-28) ──
        # Fires every EMOT_META_PULSE_INTERVAL_S regardless of cluster
        # ambiguity. Closes the EmotMeta silence found in OBS-meta-service-
        # session2-soak. Dry-run pattern (Session 2): outcome closed
        # immediately with neutral 0.0 reward — Session 3 will swap to
        # real chain execution.
        if now - _last_emot_meta_pulse_ts >= EMOT_META_PULSE_INTERVAL_S:
            _last_emot_meta_pulse_ts = now
            try:
                from titan_plugin.logic.meta_service_client import (
                    send_meta_request as _em_p_send,
                    send_meta_outcome as _em_p_out,
                )
                from titan_plugin.logic.meta_consumer_contexts import (
                    build_emotional_meta_context_30d as _em_p_ctx_fn,
                )
                _em_p_anchors = {}
                try:
                    for _an_name, _an_blk in emot_cgn._primitives.items():
                        _em_p_anchors[_an_name] = {
                            "V": float(getattr(_an_blk, "V", 0.3)),
                        }
                except Exception as _swallow_exc:
                    swallow_warn(
                        '[EmotMeta] pulse anchors snapshot', _swallow_exc,
                        key="modules.emot_cgn_worker.pulse_anchors", throttle=100)
                _em_p_cluster_p, _em_p_cluster_d, _em_p_cluster_conf = (
                    emot_cgn._last_cluster_assignment)
                _em_p_req_id = _em_p_send(
                    consumer_id="emotional",
                    question_type="evaluate_trajectory",
                    context_vector=_em_p_ctx_fn(
                        anchors=_em_p_anchors,
                        cluster_stats={
                            "variance_1h": float(_em_p_cluster_d),
                        }),
                    time_budget_ms=200,
                    constraints={
                        "confidence_threshold": 0.4,
                        "allow_timechain_query": False,
                    },
                    payload_snippet=(
                        f"routine_pulse dom={_em_p_cluster_p} "
                        f"conf={_em_p_cluster_conf:.2f}"),
                    send_queue=send_queue, src=name)
                _emot_meta_pending[_em_p_req_id] = (now, "routine_pulse")
                _em_p_out(
                    request_id=_em_p_req_id,
                    consumer_id="emotional",
                    outcome_reward=0.0,
                    actual_primitive_used=None,
                    context=(
                        f"session_2_dry routine_pulse "
                        f"dom={_em_p_cluster_p}"),
                    send_queue=send_queue, src=name)
                _emot_meta_pending.pop(_em_p_req_id, None)
            except Exception as _em_p_err:
                swallow_warn('[EmotMeta] routine pulse skipped', _em_p_err,
                             key="modules.emot_cgn_worker.routine_pulse_skipped",
                             throttle=100)

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {}) or {}

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[EmotCGNWorker] Received MODULE_SHUTDOWN — "
                        "saving state + stopping heartbeat + exiting")
            try:
                emot_cgn.save_state()
            except Exception as _se:
                logger.warning("[EmotCGNWorker] save_state on shutdown failed: %s", _se)
            _hb_stop.set()
            return

        # ── EMOT_CHAIN_EVIDENCE (meta_reasoning → emot_cgn) ────
        elif msg_type == bus.EMOT_CHAIN_EVIDENCE:
            try:
                emot_cgn.observe_chain_evidence(
                    chain_id=int(payload.get("chain_id", 0)),
                    dominant_at_start=str(payload.get("dominant_at_start", "FLOW")),
                    dominant_at_end=str(payload.get("dominant_at_end", "FLOW")),
                    terminal_reward=float(payload.get("terminal_reward", 0.5)),
                    ctx=payload.get("ctx", {}),
                )
                # Also feed neuromod EMA from ctx (previously done inline)
                ctx = payload.get("ctx", {}) or {}
                emot_cgn.update_neuromod_ema({
                    k: float(ctx.get(k, 0.5))
                    for k in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
                    if k in ctx
                })
                # ── v3: cache native inputs for bundle assembly.
                _nm_order = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")
                worker_state["last_neuromod_6d"] = [
                    float(ctx.get(k, 0.5)) for k in _nm_order
                ]
                worker_state["last_terminal_reward"] = float(
                    payload.get("terminal_reward", 0.5))
                # Native consciousness + side-channel signals routed through
                # ctx by spirit_worker._attach_emot_producer_ctx. Any missing
                # key keeps the previous cached value (dead-dim detector in
                # A4 will WARN if a group stays zero post-warmup).
                if "trajectory_2d" in ctx:
                    worker_state["last_trajectory_2d"] = ctx["trajectory_2d"]
                if "space_topology_30d" in ctx:
                    worker_state["last_space_topology_30d"] = ctx[
                        "space_topology_30d"]
                if "ns_urgencies_11d" in ctx:
                    worker_state["last_ns_urgencies_11d"] = ctx[
                        "ns_urgencies_11d"]
                if "pi_phase_6d" in ctx:
                    worker_state["last_pi_phase_6d"] = ctx["pi_phase_6d"]
                _write_state_to_shm()
            except Exception as _e:
                logger.warning("[EmotCGNWorker] EMOT_CHAIN_EVIDENCE error: %s", _e)

        # ── FELT_CLUSTER_UPDATE (spirit → emot_cgn) ────────────
        elif msg_type == bus.FELT_CLUSTER_UPDATE:
            try:
                fv = payload.get("feature_vec_150d")
                if fv is None and "felt_tensor_130d" in payload:
                    # Build 150D from 130D (worker-local — cheaper than bus)
                    fv = emot_cgn.build_feature_vec(
                        felt_tensor_130d=payload["felt_tensor_130d"])
                # ── v3: cache native felt tensor for bundle assembly.
                if "felt_tensor_130d" in payload:
                    worker_state["last_felt_tensor_130d"] = payload[
                        "felt_tensor_130d"]
                if fv is not None:
                    emot_cgn.handle_felt_tensor(fv, emit_bus_signal=True)
                    _write_state_to_shm()

                # ── F-phase (rFP §16.7): consult meta on ambiguous
                # anchor assignment. Fire only when confidence is low
                # (top-2 anchors within 0.1 — genuinely ambiguous) to
                # stay within per_consumer_rpm=10. ──
                try:
                    cluster_p, cluster_d, cluster_conf = (
                        emot_cgn._last_cluster_assignment)
                    if cluster_conf < 0.5:  # ambiguous anchor
                        from titan_plugin.logic.meta_service_client import (
                            send_meta_request as _em_send,
                            send_meta_outcome as _em_out,
                        )
                        from titan_plugin.logic.meta_consumer_contexts import (
                            build_emotional_meta_context_30d as _em_ctx_fn,
                        )
                        _em_anchors = {}
                        try:
                            for _an_name, _an_blk in emot_cgn._primitives.items():
                                _em_anchors[_an_name] = {
                                    "V": float(getattr(_an_blk, "V", 0.3)),
                                }
                        except Exception as _swallow_exc:
                            swallow_warn('[modules.emot_cgn_worker] emot_cgn_worker_main: for _an_name, _an_blk in emot_cgn._primitives.items(): _e...', _swallow_exc,
                                         key='modules.emot_cgn_worker.emot_cgn_worker_main.line570', throttle=100)
                        _em_req_id = _em_send(
                            consumer_id="emotional",
                            question_type="spirit_self_nudge",
                            context_vector=_em_ctx_fn(
                                anchors=_em_anchors,
                                cluster_stats={
                                    "variance_1h": float(cluster_d),
                                }),
                            time_budget_ms=200,
                            constraints={
                                "confidence_threshold": 0.4,
                                "allow_timechain_query": False,
                            },
                            payload_snippet=(
                                f"ambig_anchor primary={cluster_p} "
                                f"conf={cluster_conf:.2f}"),
                            send_queue=send_queue, src=name)
                        _emot_meta_pending[_em_req_id] = (
                            time.time(), cluster_p)
                        _em_out(
                            request_id=_em_req_id,
                            consumer_id="emotional",
                            outcome_reward=0.0,
                            actual_primitive_used=None,
                            context=(
                                f"session_2_dry anchor={cluster_p}"),
                            send_queue=send_queue, src=name)
                        _emot_meta_pending.pop(_em_req_id, None)
                except Exception as _em_err:
                    swallow_warn('[EmotMeta] req skipped', _em_err,
                                 key="modules.emot_cgn_worker.req_skipped", throttle=100)
            except Exception as _e:
                logger.warning("[EmotCGNWorker] FELT_CLUSTER_UPDATE error: %s", _e)

        # ── META_REASON_RESPONSE (F-phase rFP §4.3) ────────────
        # Routed here per [meta_service_interface.consumer_home_worker]
        # emotional = "emot_cgn".
        elif msg_type == bus.META_REASON_RESPONSE:
            try:
                from titan_plugin.logic.meta_service_client import (
                    dispatch_meta_response as _em_dispatch,
                )
                _em_dispatch(msg, logger_obj=logger)
            except Exception as _em_disp_err:
                logger.warning(
                    "[EmotMeta] response dispatch error: %s", _em_disp_err)

        # ── CGN_HAOV_VERIFY_REQ (H.2 2026-04-28): emotional verifier ──
        # CGN Worker asks emot_cgn to verify hypotheses on the emotional
        # consumer (e.g., "emotional_impasse_stuck"). Delta check on
        # cluster_conf and dominant primitive's V — confirmed if
        # emotional state has moved since hypothesis was formed.
        elif msg_type == bus.CGN_HAOV_VERIFY_REQ:
            try:
                _haov_p = msg.get("payload", {})
                _haov_consumer = _haov_p.get("consumer", "")
                if _haov_consumer == "emotional":
                    _obs_b = _haov_p.get("obs_before", {})
                    if not isinstance(_obs_b, dict):
                        _obs_b = {}
                    _conf_b = float(_obs_b.get("cluster_conf", 0.5))
                    _v_dom_b = float(_obs_b.get("v_dominant", 0.3))
                    _conf_a = _conf_b
                    _v_dom_a = _v_dom_b
                    _dom_p = "FLOW"
                    try:
                        _, _, _conf_a = emot_cgn._last_cluster_assignment
                        _conf_a = float(_conf_a)
                        _dom_p = str(emot_cgn._last_cluster_assignment[0])
                        _dom_blk = emot_cgn._primitives.get(_dom_p)
                        if _dom_blk is not None:
                            _v_dom_a = float(getattr(_dom_blk, "V", _v_dom_b))
                    except Exception:
                        pass
                    # Confirmed if cluster moved (state diversified) OR
                    # dominant V shifted (rebalancing happening).
                    _confirmed = (abs(_conf_a - _conf_b) > 0.05) or (abs(_v_dom_a - _v_dom_b) > 0.02)
                    _error = abs(_conf_a - _conf_b) + abs(_v_dom_a - _v_dom_b)
                    _send_msg(send_queue, bus.CGN_HAOV_VERIFY_RSP, name, "cgn", {
                        "consumer": "emotional",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"cluster_conf": _conf_a,
                                      "v_dominant": _v_dom_a,
                                      "dominant_primitive": _dom_p},
                        "reward": _v_dom_a if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info(
                        "[HAOV] Emotional verify: conf %.3f→%.3f V_dom %.3f→%.3f confirmed=%s",
                        _conf_b, _conf_a, _v_dom_b, _v_dom_a, _confirmed)
                else:
                    logger.debug(
                        "[HAOV] emot_cgn received non-emotional consumer '%s'",
                        _haov_consumer)
            except Exception as _haov_err:
                swallow_warn('[EmotCGNWorker] HAOV verify error', _haov_err,
                             key="modules.emot_cgn_worker.haov_verify_error",
                             throttle=100)

        # ── CGN_CROSS_INSIGHT (incoming from META-CGN / others) ────
        elif msg_type == bus.CGN_CROSS_INSIGHT:
            try:
                origin = str(payload.get("origin_consumer", ""))
                if origin != "emotional":  # skip own emissions bouncing back
                    emot_cgn.handle_incoming_cross_insight(payload)
                    _write_state_to_shm()
            except Exception as _e:
                swallow_warn('[EmotCGNWorker] CGN_CROSS_INSIGHT error', _e,
                             key="modules.emot_cgn_worker.cgn_cross_insight_error", throttle=100)

        # ── KIN_EMOT_STATE (peer Titan → emot_cgn) — rFP §21 ───────
        # Peer Titan's emotional region + valence/arousal/novelty.
        # We cache the freshest per-peer payload; MSL binding at the
        # next bundle write uses it to update YOU/WE activations.
        # Own emissions are filtered by parse_kin_emot_state (same
        # titan_src → None), so the dst="all" bounce is safe.
        elif msg_type == KIN_EMOT_STATE_MSG_TYPE:
            try:
                parsed = parse_kin_emot_state(
                    payload, expected_self_id=titan_id)
                if parsed is not None:
                    kin_state["peer_states"][parsed["titan_src"]] = parsed
                    logger.debug(
                        "[EmotCGNWorker] kin state from %s: region=%d "
                        "valence=%.2f age=%.1fs",
                        parsed["titan_src"], parsed["region_id"],
                        parsed["valence"], parsed["age_s"])
            except Exception as _ke:
                swallow_warn('[EmotCGNWorker] KIN_EMOT_STATE error', _ke,
                             key="modules.emot_cgn_worker.kin_emot_state_error", throttle=100)

        # ── HORMONE_FIRED (spirit → emot_cgn) — rFP §4.3, §19 ──────
        # Per-fire event carrying (program, intensity, urgency). We
        # maintain a per-program EMA hormone-level accumulator in
        # worker_state and feed it into the bundle on each write so the
        # clusterer + HDBSCAN see hormonal regime as part of Titan's
        # native state. Rate is low (few fires/sec typical), no throttle.
        elif msg_type == bus.HORMONE_FIRED:
            try:
                from titan_plugin.logic.emot_bundle_protocol import (
                    NS_PROGRAM_INDEX, HORMONE_DIM)
                program = str(payload.get("program", ""))
                idx = NS_PROGRAM_INDEX.get(program)
                if idx is not None:
                    # No clamping — hormone levels in live NS system run
                    # well above 1.0 (e.g. VIGILANCE ~3.7). Clamping would
                    # flatten high-regime states that actually matter for
                    # emotion differentiation. Bundle f32 slot accepts any
                    # value; downstream consumers handle magnitude honestly.
                    intensity = float(payload.get("intensity", 0.0))
                    levels = worker_state["last_hormone_levels_11d"]
                    if levels is None:
                        levels = [0.0] * HORMONE_DIM
                    alpha = 0.3
                    levels[idx] = (1.0 - alpha) * float(levels[idx]) + alpha * intensity
                    worker_state["last_hormone_levels_11d"] = levels
                # No _write_state_to_shm() here — fires are frequent; the
                # next chain-evidence / felt-update tick will emit a bundle.
            except Exception as _e:
                swallow_warn('[EmotCGNWorker] HORMONE_FIRED error', _e,
                             key="modules.emot_cgn_worker.hormone_fired_error", throttle=100)

        # ── CGN_BETA_SNAPSHOT (cgn_worker → emot_cgn, §23.6a) ────────
        elif msg_type == bus.CGN_BETA_SNAPSHOT:
            try:
                from titan_plugin.logic.emot_bundle_protocol import (
                    CGN_CONSUMERS)
                v_by = payload.get("values_by_consumer", {}) or {}
                # Ordered array following CGN_CONSUMERS layout
                beta_8d = [
                    float(v_by.get(c, 0.5)) for c in CGN_CONSUMERS
                ]
                worker_state["last_cgn_beta_states_8d"] = beta_8d
                # No _write_state_to_shm() here — bundle-snapshot refresh
                # at next chain-evidence / felt-update tick pulls this in.
            except Exception as _e:
                swallow_warn('[EmotCGNWorker] CGN_BETA_SNAPSHOT error', _e,
                             key="modules.emot_cgn_worker.cgn_beta_snapshot_error", throttle=100)

        else:
            logger.debug("[EmotCGNWorker] unhandled msg_type=%s", msg_type)


# ── Helpers (lifted from language_worker pattern) ───────────────────────

def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict, rid: str = None) -> None:
    """Send a message via the send queue (worker→bus)."""
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


# Heartbeat throttle (shared pattern: Phase E Fix 2, 3s min interval).
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    """Send heartbeat to Guardian with RSS info (throttled to ≤1 per 3s)."""
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
    try:
        send_queue.put_nowait({
            "type": "MODULE_HEARTBEAT", "src": name, "dst": "guardian",
            "ts": now, "rid": None,
            "payload": {"rss_mb": round(rss_mb, 1)},
        })
    except Exception as _swallow_exc:
        swallow_warn("[modules.emot_cgn_worker] _send_heartbeat: send_queue.put_nowait({'type': 'MODULE_HEARTBEAT', 'src':...", _swallow_exc,
                     key='modules.emot_cgn_worker._send_heartbeat.line740', throttle=100)
