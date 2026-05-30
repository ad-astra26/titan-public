"""titan_hcl/logic/snapshot_builders.py — observatory snapshot builders.

rFP §3G Phase 10E — relocated out of the retiring ``modules/spirit_loop.py``.
Builds the coordinator observatory snapshot from the IN-PROCESS engine objects
passed via ``state_refs`` (coordinator.get_stats() etc.) — NOT pure SHM reads,
so it must run in the process that owns those engines. cognitive_worker (the
post-D8-3 consciousness + engine owner) drives it: it calls
``start_snapshot_builder_threads`` at boot (the coord thread keeps
``_COORD_SNAPSHOT_CACHE`` warm + publishes the ``*_UPDATED`` bus events via
``_publish_coord_subdomains`` + the SHM spirit-state publisher tick).

D-SPEC-143 profiling (2026-05-29): the former trinity + nervous-system 4 Hz
builders were REMOVED — they were orphaned (only reader was spirit_worker's
retired QueryThread; the dashboard reads trinity/NS SHM-direct via
TitanStateAccessor per G18) and were cognitive_worker's #1 CPU consumer. Only
the coordinator builder (2.5s, live bus fan-out) remains.

(The RFP originally named observatory_worker as the home, but that is a separate
process with no access to cognitive_worker's in-process engine objects — see
AUDIT_phase10_relocation_liveness_findings_20260528.md §10E. Maker chose the
cognitive_worker home, 2026-05-28.)

Logic is UNCHANGED from the spirit_loop original; only the home moved.
"""
from __future__ import annotations

import json
import logging
import time

from titan_hcl.logic.spirit_helpers import _send_msg

logger = logging.getLogger(__name__)


_COORD_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}
# _TRINITY_SNAPSHOT_CACHE / _NS_SNAPSHOT_CACHE removed (D-SPEC-143 profiling) —
# their 4 Hz builders were orphaned (reader = retired spirit_worker QueryThread;
# dashboard reads SHM-direct). See start_snapshot_builder_threads.

# Builder thread cadence: seconds of sleep BETWEEN builds (cycle ≈ build_time + interval).
# Coord build is heaviest (~1-1.5s observed on T1), so 2.5s gives ~4s cycle.
_COORD_SNAPSHOT_BUILDER_INTERVAL = 2.5
_SPIRIT_STATE_PUBLISHER_INTERVAL = 1.0  # SPEC §7.1 — 1 Hz for hormone_fires/impulse_engine_state/consciousness_state slots (rFP_phase_c_async_shm_consumer_migration §4.B.1)
_SNAPSHOT_BUILDER_ERROR_BACKOFF = 2.0  # sleep on exception (avoid CPU burn + log flood)

# Legacy TTL kept as compatibility shim for the coord QueryThread cold-boot
# window. (Trinity/NS TTLs removed with their orphaned builders — D-SPEC-143.)
_COORD_SNAPSHOT_TTL = 30.0


def build_coordinator_snapshot(state_refs: dict) -> dict | None:
    """Build the coordinator stats dict. Returns None if coordinator unavailable.

    Every subsystem's contribution is isolated via _safe_set below so a
    single buggy get_stats() (e.g. a tuple index mistake) can't blank
    the entire snapshot. Without this isolation, one subsystem's bug
    used to starve /v4/inner-trinity and every other coordinator-backed
    endpoint simultaneously, and made safe_restart.sh's dreaming-state
    check return `unknown` — blocking Titan restarts for unrelated
    reasons. See 2026-04-22 investigation session.
    """
    coordinator = state_refs.get("coordinator")
    if not coordinator:
        return None

    def _safe_set(stats_dict: dict, key: str, fn, default=None,
                  shm_snapshot_attr: str | None = None):
        """Call fn() and store in stats_dict[key]. On failure, store
        `{"error": str(exc)}` (or `default` if provided) and log WARN
        once so the snapshot still builds for other subsystems.

        chunk 8M.5 (2026-05-05): when ``shm_snapshot_attr`` is provided,
        first check ``coordinator.<shm_snapshot_attr>`` — if cognitive_worker
        injected a non-None shm-read snapshot (chunk 8M.4 step 1.6), use it
        directly instead of calling fn(). Closes
        rFP_phase_c_observatory_data_pipeline.md Gap H (§2.8): under
        l0_rust_enabled=true, the in-process Python engines that fn() would
        call have moved to Rust daemons; the only live data source is shm.
        """
        if shm_snapshot_attr:
            snap = getattr(coordinator, shm_snapshot_attr, None)
            if snap is not None:
                stats_dict[key] = snap
                return
        try:
            stats_dict[key] = fn()
        except Exception as _ss_err:
            stats_dict[key] = (default if default is not None
                               else {"error": str(_ss_err)})
            logger.warning(
                "[CoordSnapshot] %s.get_stats() failed: %s — "
                "partial snapshot continues", key, _ss_err)

    pi_monitor = state_refs.get("pi_monitor")
    e_mem = state_refs.get("e_mem")
    prediction_engine = state_refs.get("prediction_engine")
    exp_orchestrator = state_refs.get("exp_orchestrator")
    episodic_mem = state_refs.get("episodic_mem")
    working_mem = state_refs.get("working_mem")
    inner_lower_topo = state_refs.get("inner_lower_topo")
    outer_lower_topo = state_refs.get("outer_lower_topo")
    ground_up_enricher = state_refs.get("ground_up_enricher")
    neuromodulator_system = state_refs.get("neuromodulator_system")
    expression_manager = state_refs.get("expression_manager")
    life_force_engine = state_refs.get("life_force_engine")
    # meditation_tracker REMOVED — owned by meditation_worker (D-SPEC-57);
    # readers consume meditation_state.bin SHM via meditation_proxy.
    outer_interface = state_refs.get("outer_interface")
    reasoning_engine = state_refs.get("reasoning_engine")
    self_reasoning = state_refs.get("self_reasoning")
    coding_explorer = state_refs.get("coding_explorer")
    phase_tracker = state_refs.get("phase_tracker")
    inner_state = state_refs.get("inner_state")
    social_pressure_meter = state_refs.get("social_pressure_meter")
    msl = state_refs.get("msl")
    language_stats = state_refs.get("language_stats")

    # coordinator.get_stats() is the core — if IT fails, snapshot can't
    # be built meaningfully. Isolate it too so the error surfaces in the
    # snapshot rather than blanking everything.
    stats = {}
    try:
        stats = coordinator.get_stats() or {}
    except Exception as _cs_err:
        logger.warning(
            "[CoordSnapshot] coordinator.get_stats() failed: %s — "
            "building stats from isolated subsystems only", _cs_err)
        stats = {"coordinator_error": str(_cs_err)}
    if pi_monitor:
        _safe_set(stats, "pi_heartbeat", pi_monitor.get_stats)
    if e_mem:
        _safe_set(stats, "experiential_memory", e_mem.get_stats)
    if prediction_engine:
        _safe_set(stats, "prediction", prediction_engine.get_stats)
    if exp_orchestrator:
        # §3L Phase 15 chunk 15.1 — experience stats now sourced from the LIVE
        # ExperienceOrchestrator (incremental action_stats), not the retired
        # frozen ExperienceMemory.get_stats. In-proc read (cognitive_worker
        # owns the orchestrator); api reads experience_stats.bin via accessor.
        _safe_set(stats, "experience_memory",
                  exp_orchestrator.get_experience_stats_payload)
    if episodic_mem:
        # §3L chunk 15.2 stopgap — mtime-gated (frozen-DB bleed). See
        # RFP_phase_c_actr_memory_rehoming for the real fix.
        _safe_set(stats, "episodic_memory",
                  lambda: _episodic_stats_mtime_gated(episodic_mem))
    if working_mem:
        _safe_set(stats, "working_memory", working_mem.get_stats)
    if inner_lower_topo:
        _safe_set(stats, "inner_lower_topology", inner_lower_topo.get_stats)
    if outer_lower_topo:
        _safe_set(stats, "outer_lower_topology", outer_lower_topo.get_stats)
    if ground_up_enricher:
        _safe_set(stats, "ground_up", ground_up_enricher.get_stats)
    # Phase A+B path: legacy state_refs["neuromodulator_system"] (spirit_worker).
    # Phase C path: instance lives at coordinator.neuromodulator_system —
    # cognitive_worker's state_refs doesn't carry it because the system is
    # owned by the coordinator on l0_rust_enabled=true. Closes
    # BUG-T3-NEUROMODULATORS-EMPTY-PAYLOAD-20260512: pre-fix the elif
    # `_shm_neuromod_snapshot` branch was the ONLY Phase C escape hatch
    # and was never injected (chunk 8M.5 docstring said "future wiring"),
    # so /v4/neuromodulators returned `{}` and /status.lifetime.emotion
    # was "unknown" on T3.
    # §4.Q (2026-05-15): prefer the LIVE neuromod_worker-published payload
    # (NEUROMOD_STATS_UPDATED 2.5s coalesced) over coordinator's in-proc
    # `_coord_nm_sys` instance. Under Phase C the coordinator's instance has
    # its levels refreshed via `update_neuromodulators()` from SHM-read but
    # `total_evaluations` + `current_emotion` are NEVER updated there
    # (those increment only inside neuromod_worker's evaluate driver). The
    # cached publisher payload is the source of truth for these fields.
    _cached_neuromod_stats = state_refs.get("_last_neuromod_stats")
    if isinstance(_cached_neuromod_stats, dict) and _cached_neuromod_stats.get(
            "modulators"):
        stats["neuromodulators"] = {
            "modulators": _cached_neuromod_stats.get("modulators", {}),
            "modulation": _cached_neuromod_stats.get("modulation", {}),
            "current_emotion": _cached_neuromod_stats.get(
                "current_emotion", "neutral"),
            "emotion_confidence": float(_cached_neuromod_stats.get(
                "emotion_confidence", 0.0)),
            "total_evaluations": int(_cached_neuromod_stats.get(
                "total_evaluations", 0)),
        }
    else:
        # Pre-cache fallback to the legacy paths.
        if neuromodulator_system is None:
            _coord_nm_sys = getattr(coordinator, "neuromodulator_system", None)
            if _coord_nm_sys is not None:
                neuromodulator_system = _coord_nm_sys
        if neuromodulator_system:
            _safe_set(stats, "neuromodulators", neuromodulator_system.get_stats)
        elif getattr(coordinator, "_shm_neuromod_snapshot", None) is None:
            stats.setdefault("neuromodulators", {})
    if expression_manager:
        _safe_set(stats, "expression_composites", expression_manager.get_stats)
    # chunk 8M.5 — chi block prefers the cognitive_worker shm snapshot
    # (rFP §3.4). Under l0_rust_enabled=true `life_force_engine` is None and
    # `_chi_snapshot` carries the canonical Rust-owned chi_state.bin payload
    # injected by cognitive_worker._drive_one_epoch step 1.6.
    _chi_shm = getattr(coordinator, "_chi_snapshot", None)
    if _chi_shm is not None:
        stats["chi"] = _chi_shm
    elif life_force_engine:
        stats["chi"] = getattr(life_force_engine, '_latest_chi', {})
    # meditation snapshot REMOVED from coordinator stats — meditation_worker
    # is sole G21 writer of meditation_state.bin (D-SPEC-57); dashboard
    # /v4/meditation/health reads via meditation_proxy.get_state() SHM-direct.
    if outer_interface:
        _safe_set(stats, "outer_interface", outer_interface.get_stats)
    if reasoning_engine:
        _safe_set(stats, "reasoning", reasoning_engine.get_stats)
    # Meta-reasoning block — always emit the key (shape-stable for downstream)
    stats["meta_reasoning"] = {}
    _me = getattr(coordinator, '_meta_engine', None)
    if _me:
        try:
            stats["meta_reasoning"] = _me.get_stats()
        except Exception as _me_err:
            logger.warning(
                "[META] get_stats failed: %s — leaving "
                "meta_reasoning={} for this tick", _me_err)
            stats["meta_reasoning"] = {}
        try:
            stats["meta_reasoning_audit"] = _me.get_audit_stats()
        except Exception as _au_err:
            logger.warning("[META] get_audit_stats failed: %s", _au_err)
    else:
        logger.debug(
            "[META] coordinator._meta_engine is None at "
            "build_coordinator_snapshot — meta_reasoning={}")
    # F-phase (rFP §11.1): Meta-Reasoning Consumer Service status
    _ms = getattr(coordinator, '_meta_service', None)
    if _ms:
        try:
            stats["meta_service"] = _ms.get_status()
        except Exception as _ms_err:
            logger.warning("[MetaService] get_status failed: %s", _ms_err)
            stats["meta_service"] = {"error": str(_ms_err)}
    else:
        stats["meta_service"] = {}
    if self_reasoning:
        _safe_set(stats, "self_reasoning", self_reasoning.get_stats)
    if coding_explorer:
        _safe_set(stats, "coding_explorer", coding_explorer.get_stats)
    if phase_tracker:
        stats["phase_events"] = {
            "current_phase": phase_tracker.get("current_phase", "idle"),
            "recent_events": phase_tracker.get("events", [])[-20:],
            "total_events": len(phase_tracker.get("events", [])),
        }
    # Dreaming block (is_dreaming lives on inner_state, not DreamingEngine)
    if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
        _dr_is = False
        if inner_state and hasattr(inner_state, 'is_dreaming'):
            _dr_is = inner_state.is_dreaming
        _dr_dream_epochs = getattr(
            coordinator.dreaming, '_dream_epoch_count', 0)
        _dr_onset = getattr(
            coordinator.dreaming, '_dream_onset_fatigue', 0)
        _dr_fatigue = getattr(
            coordinator.dreaming, '_dream_fatigue', 0)
        _dr_wake_trans = getattr(
            coordinator.dreaming, '_wake_transition', False)
        _dr_recovery_pct = 0.0
        if _dr_is and _dr_onset > 0:
            _dr_recovery_pct = round(
                100.0 * (1.0 - max(0, _dr_fatigue) / _dr_onset), 1)
        _dr_remaining = max(0, round(_dr_fatigue / 3.0)) if _dr_is else 0
        stats["dreaming"] = {
            "is_dreaming": _dr_is,
            "fatigue": round(getattr(inner_state, 'fatigue', 0), 4)
                if inner_state else 0,
            "cycle_count": getattr(
                coordinator.dreaming, '_cycle_count', 0),
            "dream_epochs": _dr_dream_epochs,
            "recovery_pct": _dr_recovery_pct,
            "wake_transition": _dr_wake_trans,
            "remaining_epochs": _dr_remaining,
            "onset_fatigue": round(_dr_onset),
            "epochs_since_dream": getattr(
                coordinator.dreaming, '_epochs_since_dream', 0),
            "last_sleep_drive": round(float(getattr(
                coordinator.dreaming, 'last_sleep_drive', 0.0)), 4),
            "last_wake_drive": round(float(getattr(
                coordinator.dreaming, 'last_wake_drive', 0.0)), 4),
            "distilled_count": getattr(
                coordinator.dreaming, '_distilled_count', 0),
            "distill_threshold": getattr(
                coordinator.dreaming, '_distill_threshold', 0.02),
            "distill_attempts": getattr(
                coordinator.dreaming, '_distill_attempts', 0),
            "distill_passed": getattr(
                coordinator.dreaming, '_distill_passed', 0),
            "variance_samples_count": len(getattr(
                coordinator.dreaming, '_variance_samples', [])),
            "experience_buffer_size": len(getattr(
                coordinator.inner, '_experience_buffer', [])
                if coordinator.inner else []),
        }
    if social_pressure_meter:
        _safe_set(stats, "social_pressure", social_pressure_meter.get_stats)
    # rFP_observatory_data_loading_v1 §3.2 (2026-04-26): topology block
    # for the Trinity Architecture TopologyPanel.
    #
    # Batch D — legacy 3 fields (volume / curvature / cluster_count) from
    # TopologyEngine for backwards compatibility with the existing widget.
    #
    # Batch E (2026-04-26 follow-up, Maker-greenlit): the panel was
    # designed before the 30D space topology shipped. Now also expose
    # the rich state_register observables_30d (6 layers × 5 metrics —
    # coherence / magnitude / velocity / direction / polarity per
    # inner|outer × body|mind|spirit) so the frontend can render the
    # full space-topology view alongside the legacy summary.
    _topo_block = {
        "volume": 0.0, "curvature": 0.0,
        "cluster_count": 0, "cluster_threshold": 0.0,
        "observables_30d": [],
        "observables_dict": {},
    }
    if coordinator and hasattr(coordinator, "topology") and coordinator.topology:
        try:
            _topo_stats = coordinator.topology.get_stats() or {}
            _topo_block["volume"] = float(_topo_stats.get("current_volume", 0.0) or 0.0)
            _topo_block["curvature"] = float(_topo_stats.get("current_curvature", 0.0) or 0.0)
            _topo_block["cluster_count"] = int(_topo_stats.get("volume_history_size", 0) or 0)
            _topo_block["cluster_threshold"] = float(_topo_stats.get("cluster_threshold", 0.0) or 0.0)
        except Exception as _topo_err:
            logger.debug("[CoordSnapshot] topology read failed: %s", _topo_err)
    # Batch E — observables_dict is 6 layers × 5 metrics
    # (inner|outer × body|mind|spirit, each {coherence, magnitude,
    # velocity, direction, polarity}) = 30 metrics. InnerState.observables
    # carries the labelled dict; flatten it deterministically into a 30D
    # vector so the frontend can render either form.
    if inner_state is not None:
        try:
            _obs_dict = inner_state.observables if hasattr(inner_state, "observables") else None
            if isinstance(_obs_dict, dict) and _obs_dict:
                _topo_block["observables_dict"] = _obs_dict
                # Deterministic flatten: layer order matches state_register
                # observables_30d (inner_body, inner_mind, inner_spirit,
                # outer_body, outer_mind, outer_spirit), metric order:
                # coherence, magnitude, velocity, direction, polarity.
                _LAYERS = ("inner_body", "inner_mind", "inner_spirit",
                           "outer_body", "outer_mind", "outer_spirit")
                _METRICS = ("coherence", "magnitude", "velocity",
                            "direction", "polarity")
                _vec: list[float] = []
                for _l in _LAYERS:
                    _layer_vals = _obs_dict.get(_l, {}) if isinstance(
                        _obs_dict.get(_l), dict) else {}
                    for _m in _METRICS:
                        _v = _layer_vals.get(_m, 0.0)
                        _vec.append(round(float(_v), 4) if isinstance(
                            _v, (int, float)) else 0.0)
                if len(_vec) == 30:
                    _topo_block["observables_30d"] = _vec
        except Exception as _obs_err:
            logger.debug("[CoordSnapshot] observables read failed: %s", _obs_err)
    stats["topology"] = _topo_block

    # chunk 8M.5 — sphere_clocks / unified_spirit / hormonal / titanvm /
    # self_162d blocks. Under l0_rust_enabled=true these subsystems live in
    # Rust daemons (titan-trinity-rs / titan-unified-spirit-rs / hormonal-
    # rs); cognitive_worker._drive_one_epoch step 1.6 injects shm-read
    # snapshots onto the coordinator. Always emit the key (shape-stable
    # for downstream / JSON serialization) — `{}` when shm read failed or
    # legacy mode w/o legacy engine.
    _sc_shm = getattr(coordinator, "_sphere_clocks_snapshot", None)
    if _sc_shm is not None:
        stats["sphere_clocks"] = _sc_shm
    _us_shm = getattr(coordinator, "_self_162d_snapshot", None)
    if _us_shm is not None:
        # self_162d carries unified_spirit (130D) + topology (30D) + journey (2D);
        # expose under both `unified_spirit` and `self_162d` for downstream
        # consumers with different key conventions.
        stats["unified_spirit"] = _us_shm
        stats["self_162d"] = _us_shm
    _horm_shm = getattr(coordinator, "_hormonal_snapshot", None)
    if _horm_shm is not None:
        stats["hormonal"] = _horm_shm
    _tvm_shm = getattr(coordinator, "_titanvm_snapshot", None)
    if _tvm_shm is not None:
        stats["titanvm_registers"] = _tvm_shm

    # chunk 8M.5 — topology block can also pull from shm snapshot when
    # coordinator.topology engine is None (l0_rust mode). The legacy block
    # above already populated _topo_block.observables_30d from inner_state;
    # add labelled per-part view from shm topology_30d when available.
    _topo_shm = getattr(coordinator, "_topology_snapshot", None)
    if _topo_shm is not None:
        # _topo_shm = {"values": [30 floats], "parts": {...}, "age_seconds", "seq"}
        if not _topo_block.get("observables_30d") and _topo_shm.get("values"):
            _topo_block["observables_30d"] = _topo_shm["values"]
        if not _topo_block.get("observables_dict") and _topo_shm.get("parts"):
            _topo_block["observables_dict"] = _topo_shm["parts"]
        # Reflect freshness so frontend can show staleness if needed.
        _topo_block["age_seconds"] = _topo_shm.get("age_seconds")
        # Re-emit topology since the shm enrichment landed after the
        # _topo_block insertion at stats["topology"] above.
        stats["topology"] = _topo_block

    if msl:
        _msl_attn = msl.get_attention_weights_for_kin()
        _msl_entropy = 0.0
        if _msl_attn is not None:
            import numpy as _msl_np
            _vals = list(_msl_attn.values()) if isinstance(_msl_attn, dict) else _msl_attn
            _a = _msl_np.array(_vals, dtype=_msl_np.float32)
            _a_norm = _a / (_a.sum() + 1e-10)
            _msl_entropy = float(-(_a_norm * _msl_np.log(_a_norm + 1e-10)).sum())
        _depth_stats = msl.i_depth.get_stats() if hasattr(msl, 'i_depth') else {}
        _homeo_state = {}
        if (hasattr(msl, 'policy') and msl.policy
                and hasattr(msl.policy, 'homeostatic')):
            try:
                _homeo_state = msl.policy.homeostatic.get_state()
            except Exception:
                _homeo_state = {}
        stats["msl"] = {
            "i_confidence": msl.get_i_confidence(),
            "i_depth": _depth_stats.get("depth", 0.0),
            "i_depth_components": _depth_stats.get("components", {}),
            "convergence_count": msl.confidence._convergence_count,
            "concept_confidences": msl.concept_grounder.get_concept_confidences() if msl.concept_grounder else {},
            "attention_weights": _msl_attn,
            "attention_entropy": round(_msl_entropy, 3),
            "homeostatic": _homeo_state,
        }
    if language_stats:
        stats["language"] = language_stats
    from titan_hcl.logic.spirit_state import _sanitize_dict_keys
    return _sanitize_dict_keys(stats)


def start_snapshot_builder_threads(state_refs: dict, config: dict,
                                    send_queue=None, name: str = "spirit") -> None:
    """Launch 3 daemon threads that keep the heavy snapshot caches fresh.

    Called once from spirit_worker at boot, right after the query handler
    thread starts. Replaces the on-demand in-handler build pattern that
    blocked the QueryThread for 460-2144ms per coord build (cause of
    T1-COORD-QUERYTHREAD-BACKLOG). Each thread is daemon=True so it dies
    with the process.

    On builder exception: caught, logged rate-limited, cache keeps serving
    the last successful build. On loop exit (should never happen): FATAL
    log so investigators can correlate any stale cache with the crash.

    M1 phase C-E: when send_queue is provided, the coord-snapshot-builder
    additionally fans out per-domain *_UPDATED events for the
    api_subprocess BusSubscriber → CachedState pathway (pi_heartbeat,
    dreaming, meta_reasoning). chi has its own immediate publisher in
    spirit_worker (Phase B); this path covers domains whose only producer
    is the periodic snapshot.
    """
    import threading

    # Track 2 (v1.2.1) — closure cell for dream-cycle state-transition
    # detection. Per rFP_phase_c_self_improvement_subsystem_migration §2.B.5:
    # self_reflection_worker subscribes to DREAMING_STATE_UPDATED and dispatches
    # on payload.state ∈ {"dream_start","dream_end","dreaming","awake"}. The
    # existing periodic publisher in _publish_coord_subdomains gains an extra
    # `state` field computed from the (prev, current) is_dreaming pair. Single-
    # element list = mutable closure cell (Python closure pattern for in-place
    # state across nested function invocations).
    _dream_state_prev: list[bool] = [False]

    def _publish_coord_subdomains(snapshot: dict) -> None:
        """Fan out per-domain UPDATED events for api cache wiring."""
        if send_queue is None or not isinstance(snapshot, dict):
            return
        from titan_hcl.bus import (
            PI_HEARTBEAT_UPDATED, DREAMING_STATE_UPDATED,
            META_REASONING_STATS_UPDATED, REASONING_STATS_UPDATED,
            EXPRESSION_COMPOSITES_UPDATED, NEUROMOD_STATS_UPDATED,
            MSL_STATE_UPDATED, LANGUAGE_STATS_UPDATED,
            TOPOLOGY_STATE_UPDATED,
        )
        try:
            pi = snapshot.get("pi_heartbeat")
            if pi:
                _send_msg(send_queue, PI_HEARTBEAT_UPDATED, name, "all", pi)
            # Dreaming payload composed to match /v4/dreaming response
            # shape (is_dreaming + dreaming sub-dict + developmental_age
            # from pi_heartbeat). Frontend useDreaming hook reads these.
            dreaming = snapshot.get("dreaming") or {}
            dream_payload = dict(dreaming)
            _is_dreaming_now = bool(snapshot.get("is_dreaming", False))
            dream_payload["is_dreaming"] = _is_dreaming_now
            dream_payload["developmental_age"] = (
                (pi or {}).get("developmental_age", 0))
            # Track 2 (v1.2.1) — state-transition field for self_reflection_worker
            # subscriber per rFP §2.B.5. Computed from (prev, current) is_dreaming
            # pair so consumers can dispatch on dream_start / dream_end edges
            # without polling. self_reflection_worker handles dream_start →
            # _coding_explorer.on_dream_start(); dream_end →
            # _self_reasoning.consolidate_training() + _last_dream_profile set.
            _was_dreaming = _dream_state_prev[0]
            if _is_dreaming_now and not _was_dreaming:
                dream_payload["state"] = "dream_start"
            elif _was_dreaming and not _is_dreaming_now:
                dream_payload["state"] = "dream_end"
            elif _is_dreaming_now:
                dream_payload["state"] = "dreaming"
            else:
                dream_payload["state"] = "awake"
            # Surface the dream_profile if engine produced one (consumed by
            # self_reflection_worker on dream_end → _last_dream_profile).
            dream_payload.setdefault(
                "dream_profile", dreaming.get("dream_profile"))
            _dream_state_prev[0] = _is_dreaming_now
            _send_msg(send_queue, DREAMING_STATE_UPDATED, name, "all",
                      dream_payload)
            meta = snapshot.get("meta_reasoning")
            if meta:
                _send_msg(send_queue, META_REASONING_STATS_UPDATED, name,
                          "all", meta)
            # Reasoning engine stats (chains, commits, abandons, commit_rate)
            # — observed empty on /v4/reasoning until this publish was added
            # (2026-04-26 sweep). Endpoint reads from reasoning.state cache key.
            reasoning = snapshot.get("reasoning")
            if reasoning:
                _send_msg(send_queue, REASONING_STATS_UPDATED, name, "all",
                          reasoning)
            expr = snapshot.get("expression_composites")
            if expr:
                _send_msg(send_queue, EXPRESSION_COMPOSITES_UPDATED, name,
                          "all", expr)
            nm = snapshot.get("neuromodulators")
            if nm:
                _send_msg(send_queue, NEUROMOD_STATS_UPDATED, name, "all", nm)
            # rFP_observatory_data_loading_v1 Phase 4 — MSL state fan-out.
            # I-Depth tab consumes msl.state cache key for i_confidence /
            # i_depth / components / convergence_count / concept_confidences /
            # attention_weights. Coord snapshot already builds this at
            # build_coordinator_snapshot:1651 — fan it out here.
            msl_state = snapshot.get("msl")
            if msl_state:
                _send_msg(send_queue, MSL_STATE_UPDATED, name, "all", msl_state)
            # Language teacher periodic stats — vocab / prod / level / conf
            # / last_teach_at. Coord snapshot includes stats["language"] when
            # the worker passes language_stats; fan out so /v4/vocabulary +
            # related tabs render.
            lang = snapshot.get("language")
            if lang:
                _send_msg(send_queue, LANGUAGE_STATS_UPDATED, name, "all", lang)
            # Batch E (rFP §3.2 follow-up): topology block — legacy
            # volume/curvature/cluster_count + 30D space-topology
            # observables_dict (6 layers × 5 metrics). Frontend
            # TopologyPanel renders both forms. SpiritAccessor.get_coordinator()
            # overlay reads topology.state and merges into coord["topology"].
            topo = snapshot.get("topology")
            if topo:
                _send_msg(send_queue, TOPOLOGY_STATE_UPDATED, name, "all", topo)
        except Exception as pub_err:
            # Never let a publish glitch break the snapshot builder loop.
            logger.warning(
                "[SnapshotBuilder:coord] subdomain publish failed: %s",
                pub_err)

    def _builder_loop(kind: str, build_fn, cache: dict, interval: float):
        consecutive_errors = 0
        try:
            while True:
                _t0 = time.time()
                try:
                    result = build_fn()
                    if result is not None:
                        # Atomic pointer swap under GIL — readers see
                        # either the old dict or the new dict, never a
                        # partially-built one.
                        cache["data"] = result
                        cache["ts"] = time.time()
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                        # Include traceback so the dead-coordinator
                        # class of bugs (e.g. 2026-04-22 "tuple index
                        # out of range" that silently starved /v4/inner-
                        # trinity) diagnoses in one log line next time.
                        logger.warning(
                            "[SnapshotBuilder:%s] build failed "
                            "(#%d consecutive): %s",
                            kind, consecutive_errors, exc,
                            exc_info=True)
                build_ms = (time.time() - _t0) * 1000
                logger.debug(
                    "[SnapshotBuilder:%s] built in %.0fms",
                    kind, build_ms)
                sleep_s = (_SNAPSHOT_BUILDER_ERROR_BACKOFF
                           if consecutive_errors > 0 else interval)
                time.sleep(sleep_s)
        except BaseException as fatal:
            logger.error(
                "[SnapshotBuilder:%s] loop exited unexpectedly — "
                "cache will become stale: %s",
                kind, fatal, exc_info=True)

    def _coord_build_and_publish():
        snap = build_coordinator_snapshot(state_refs)
        if snap is not None:
            _publish_coord_subdomains(snap)
        return snap

    threading.Thread(
        target=_builder_loop,
        args=("coord",
              _coord_build_and_publish,
              _COORD_SNAPSHOT_CACHE,
              _COORD_SNAPSHOT_BUILDER_INTERVAL),
        daemon=True, name="coord-snapshot-builder",
    ).start()
    # D-SPEC-143 profiling (2026-05-29): the trinity + nervous-system snapshot
    # builders that ran here at 4 Hz (0.25s) were the #1 CPU consumer of
    # cognitive_worker (~44% via py-spy) — yet they were ORPHANED. Their only
    # ever reader was spirit_worker's `_handle_query` QueryThread (retired:
    # spirit_worker deleted D-SPEC-116, spirit_loop deleted Phase 10). The
    # dashboard now serves trinity + nervous-system SHM-direct via
    # TitanStateAccessor (`/v6/trinity/*`→trinity_state.bin,
    # `/v6/nervous-system`→titanvm_registers.bin, Preamble G18) — it never read
    # `_TRINITY_SNAPSHOT_CACHE` / `_NS_SNAPSHOT_CACHE`. So both builders rebuilt
    # caches nobody reads, 8 builds/sec. REMOVED (no shim per
    # `feedback_no_shim_old_path_must_be_deleted`) along with build_trinity_snapshot
    # / build_nervous_system_snapshot. The LIVE coord builder above (2.5s — its
    # `_publish_coord_subdomains` *_UPDATED bus fan-out is consumed by titan_hcl
    # parent / self_reflection / observatory) is UNAFFECTED. See
    # titan-docs/notes/profiling_findings_20260529.md F1.

    # Phase B.5 (rFP_phase_c_state_read_unification_l0_l1_canonical §B.5,
    # 2026-05-18) — SpiritStatePublisher + SpiritSupplementalStatePublisher
    # RETIRED fleet-wide. The 5 Python-wrapper slots they wrote
    # (hormone_fires / impulse_engine_state / consciousness_state /
    # resonance_state / spirit_supplemental_state) are superseded by the
    # canonical Rust L0+L1 slots shipped at B.0 (resonance_metadata,
    # unified_spirit_metadata, filter_down_state — all written by
    # titan-unified-spirit-rs MetadataPublisher) plus existing L1
    # composites (self_162d, hormonal_state, sphere_clocks, chi_state,
    # neuromod_state) per the Maker directive ("EVERYTHING IN OUR
    # CODEBASE THAT READS STATE MUST READ IT FROM L0+L1 rust layer").
    # spirit-state-publisher thread retired with its publishers.
    #
    # ReflexStatePublisher + SocialPerceptionStatePublisher kept — both
    # remain canonical Phase C-S3 L2 publishers per Maker greenlight
    # 2026-05-17 (legitimate worker-local state, NOT Cat-B wrapper drift).
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.reflex_state_publisher import (
            ReflexStatePublisher)
        from titan_hcl.logic.social_perception_state_publisher import (
            SocialPerceptionStatePublisher)

        _titan_id_resolved = resolve_titan_id()
        _reflex_state_publisher = ReflexStatePublisher(
            titan_id=_titan_id_resolved)
        _social_perception_publisher = SocialPerceptionStatePublisher(
            titan_id=_titan_id_resolved)

        def _spirit_state_publish_tick():
            # 2 surviving publishers — each per-publish failure is
            # isolated by BaseStatePublisher's internal try/except.
            _reflex_state_publisher.publish(
                state_refs.get("neural_nervous_system"))
            _social_perception_publisher.publish(state_refs)
            return None  # no cache; SHM is the canonical store

        threading.Thread(
            target=_builder_loop,
            args=("spirit_state_publisher",
                  _spirit_state_publish_tick,
                  {"data": None, "ts": 0.0},  # no-op cache (SHM is canonical)
                  _SPIRIT_STATE_PUBLISHER_INTERVAL),
            daemon=True, name="spirit-state-publisher",
        ).start()
        logger.info(
            "[SpiritLoop] Spirit state SHM publishers started — "
            "reflex_state (Session 3 §4.B.10) + social_perception_state "
            "(Session 3 §4.B.4) @ %.2fs cadence. SpiritStatePublisher + "
            "SpiritSupplementalStatePublisher RETIRED Phase B.5 — Rust "
            "L0+L1 canonical slots (resonance_metadata, "
            "unified_spirit_metadata, filter_down_state) supersede.",
            _SPIRIT_STATE_PUBLISHER_INTERVAL)
    except Exception as _publisher_boot_err:
        logger.error(
            "[SpiritLoop] Reflex/SocialPerception SHM publisher BOOT FAILED: %s",
            _publisher_boot_err, exc_info=True)

    logger.info(
        "[SpiritLoop] Coordinator snapshot builder thread started — "
        "rebuild every %.2fs (trinity/NS builders removed D-SPEC-143)",
        _COORD_SNAPSHOT_BUILDER_INTERVAL)


