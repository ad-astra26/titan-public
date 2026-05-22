"""
meta_reasoning_state_publisher — MetaReasoningStatePublisher writes
meta_reasoning_state.bin SHM slot.

Producer for meta_reasoning_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0).
G21 single-writer contract: only cognitive_worker publishes here.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.meta_reasoning_state_specs import (
    META_REASONING_STATE_SLOT,
    META_REASONING_STATE_SPEC,
)
from titan_hcl._phase_c_constants import META_REASONING_STATE_SCHEMA_VERSION


class MetaReasoningStatePublisher(BaseStatePublisher):
    slot_name = META_REASONING_STATE_SLOT
    slot_spec = META_REASONING_STATE_SPEC

    def _compute_payload(self, meta_engine: Any, nns: Any = None,
                         pi_monitor: Any = None,
                         neuromod_stats: Any = None) -> dict[str, Any]:
        if meta_engine is None:
            return self._stub()
        try:
            audit = meta_engine.get_audit_stats() if hasattr(
                meta_engine, "get_audit_stats") else {}
        except Exception:
            audit = {}
        # ── Additive: identity / lifetime telemetry the dashboard needs ──
        # `get_audit_stats()` doesn't carry the in-process accumulators
        # (`_total_eurekas`, `_total_meta_steps`, `_total_wisdom_saved`);
        # read them directly from the engine instance. The NS instance is
        # passed in by cognitive_worker so we can co-publish neural_maturity
        # (Hormonal._maturity) alongside meta_reasoning telemetry — both live
        # in cognitive_worker so this is the natural co-located surface for
        # the dashboard /status.lifetime composer + SovereigntyGauge bar.
        total_eurekas = int(getattr(meta_engine, "_total_eurekas", 0) or 0)
        total_meta_steps = int(getattr(meta_engine, "_total_meta_steps", 0) or 0)
        total_wisdom_saved = int(
            getattr(meta_engine, "_total_wisdom_saved", 0) or 0)
        neural_maturity = 0.0
        try:
            if nns is not None:
                horm = getattr(nns, "_hormonal", None)
                if horm is not None:
                    neural_maturity = float(getattr(horm, "maturity", 0.0) or 0.0)
        except Exception:
            pass
        # Rich π-heartbeat stats from the pi_monitor instance (PiHeartbeat
        # cluster detector — clusters/streaks/dev_age/heartbeat_ratio). Bounded
        # to scalar + small-list fields (recent_cluster_sizes ≤ 10 ints).
        _pi_stats: dict[str, Any] = {}
        try:
            if pi_monitor is not None and hasattr(pi_monitor, "get_stats"):
                _raw_pi = pi_monitor.get_stats() or {}
                if isinstance(_raw_pi, dict):
                    _pi_stats = {
                        k: v for k, v in _raw_pi.items()
                        if isinstance(v, (int, float, bool))
                        or (isinstance(v, list) and len(v) <= 12
                            and all(isinstance(x, (int, float)) for x in v))
                    }
        except Exception:
            pass
        monoc = audit.get("monoculture", {}) or {}
        if not isinstance(monoc, dict):
            monoc = {}
        prim_counts = monoc.get("primitive_counts_500", {}) or {}
        if not isinstance(prim_counts, dict):
            prim_counts = {}
        # Per-primitive distribution as shares (sum=1) over last 500 actions.
        total_a = sum(v for v in prim_counts.values()
                      if isinstance(v, (int, float))) or 1
        prim_dist = {
            str(k): round(float(v) / float(total_a), 4)
            for k, v in prim_counts.items()
            if isinstance(v, (int, float))
        }
        subsys = audit.get("subsystem_signals_status", {}) or {}
        if not isinstance(subsys, dict):
            subsys = {}
        introspect = audit.get("introspect_health", {}) or {}
        if not isinstance(introspect, dict):
            introspect = {}
        # D-SPEC-91 v1.30.0 schema v2 — slim meta_cgn + failsafe sub-blocks
        # so dashboard endpoints resolve against canonical SHM slot per G18.
        mcgn = audit.get("meta_cgn", {}) or {}
        if not isinstance(mcgn, dict):
            mcgn = {}
        mcgn_grad = mcgn.get("graduation", {}) or {}
        mcgn_haov = mcgn.get("haov", {}) or {}
        mcgn_haov_by_status = mcgn_haov.get("by_status", {}) or {}
        mcgn_pvs_full = mcgn.get("primitive_V_summary", {}) or {}
        # primitive_v_summary in SHM slot is the slim dict[str→float] (V only);
        # full per-primitive audit (α/β/ci/n) stays on disk in primitive_grounding.json
        mcgn_pvs_slim = {
            str(p_id): round(float(info.get("V", 0.0)), 4)
            for p_id, info in mcgn_pvs_full.items()
            if isinstance(info, dict) and isinstance(info.get("V"), (int, float))
        }
        failsafe = audit.get("failsafe", {}) or {}
        if not isinstance(failsafe, dict):
            failsafe = {}
        failsafe_payload = {
            "status": str(failsafe.get("status", "unknown") or "unknown"),
            "last_check_ts": float(failsafe.get("last_check_ts", 0.0) or 0.0),
            "recent_failures": int(failsafe.get("recent_failures", 0) or 0),
        }
        # Nested under meta_cgn to match MetaCGNConsumer.get_stats() native
        # shape + dashboard /v4/meta-cgn/failsafe-status drill path
        # (meta["meta_cgn"]["failsafe"]).
        meta_cgn_payload = {
            "status": str(mcgn.get("status", "unknown") or "unknown"),
            "graduation": {
                "progress": int(mcgn_grad.get("progress", 0) or 0),
                "rolled_back_count": int(
                    mcgn_grad.get("rolled_back_count", 0) or 0),
            },
            "primitives_well_sampled": int(
                mcgn.get("primitives_well_sampled", 0) or 0),
            "haov": {
                "by_status": {
                    "confirmed": int(
                        mcgn_haov_by_status.get("confirmed", 0) or 0),
                },
            },
            "updates_applied": int(mcgn.get("updates_applied", 0) or 0),
            "ready_to_graduate": bool(
                mcgn.get("ready_to_graduate", False)),
            "primitive_v_summary": mcgn_pvs_slim,
            "failsafe": failsafe_payload,
        }
        # Phase C dissolve (2026-05-22): the outer_mind/outer_spirit 130D dim
        # formulas read these RICH meta_cgn fields (knowledge_helpful_ratio,
        # knowledge_helpful_by_source, knowledge_responses_received, usage_gini,
        # eureka_accelerated_{updates,per_hour}, primitives_{total,grounded}).
        # They live in MetaCGNConsumer.get_stats() (the engine's rich block) —
        # NOT in get_audit_stats()["meta_cgn"] (the slim audit projection above).
        # Previously they reached the dim formulas ONLY via the
        # META_REASONING_STATS_UPDATED bus-cache (a G18 violation). Co-publish
        # the 8 bounded fields into the SHM meta_cgn block so the outer-source
        # sidecars read them SHM-direct (G18). Additive → no schema bump; the
        # outer-source helper projects to exactly these 8 to keep the
        # sensor_cache_outer_*.bin payload small (the full block is ~26 KB).
        try:
            _mcgn_engine = getattr(meta_engine, "_meta_cgn", None)
            if _mcgn_engine is not None and hasattr(_mcgn_engine, "get_stats"):
                _rich = _mcgn_engine.get_stats() or {}
                if isinstance(_rich, dict):
                    for _k in ("knowledge_helpful_ratio",
                               "knowledge_helpful_by_source",
                               "knowledge_responses_received", "usage_gini",
                               "eureka_accelerated_updates",
                               "eureka_accelerated_per_hour",
                               "primitives_total", "primitives_grounded"):
                        if _k in _rich:
                            meta_cgn_payload[_k] = _rich[_k]
        except Exception:
            pass
        return {
            "total_meta_chains": int(audit.get("total_meta_chains", 0) or 0),
            "total_introspect_picks": int(
                introspect.get("picks_lifetime", 0) or 0),
            "total_introspect_executions": int(
                introspect.get("executions_lifetime", 0) or 0),
            "monoculture_score": float(
                monoc.get("dominant_share_500", 0.0) or 0.0),
            "primitive_distribution": prim_dist,
            "last_chain_id": int(audit.get("last_chain_id", 0) or 0),
            "last_chain_reason": str(audit.get("last_chain_reason", "") or ""),
            "last_chain_succeeded": bool(audit.get("last_chain_succeeded", False)),
            "subsystem_signals_status": subsys,
            "meta_cgn": meta_cgn_payload,
            # Additive lifetime telemetry (D-SPEC-91 v1.30.0 schema v2
            # additive pattern; v1/v2 readers tolerate missing keys).
            "total_eurekas": total_eurekas,
            "total_meta_steps": total_meta_steps,
            "total_wisdom_saved": total_wisdom_saved,
            "neural_maturity": round(neural_maturity, 4),
            # Raw primitive counts (last 500 actions) for the ReasoningTab
            # PrimitiveBar table — the component reads meta.primitive_counts;
            # primitive_distribution above is the normalized share view.
            "primitive_counts": {str(k): int(v) for k, v in prim_counts.items()
                                 if isinstance(v, (int, float))},
            # Co-located rich π-heartbeat stats (PiHeartbeat.get_stats() from
            # the pi_monitor instance that lives in cognitive_worker). The
            # canonical pi_heartbeat.bin SHM slot is the lean Rust L0 oscillator
            # (phase + pulse_count only) and lacks the cluster/streak/dev_age/
            # heartbeat_ratio telemetry the Observatory PiHeartbeatStrip needs.
            # /v4/pi-heartbeat merges this rich block over the lean slot.
            "pi_heartbeat": _pi_stats,
            # Co-located neuromod emotion (NeuromodulatorSystem._current_emotion
            # via the cached NEUROMOD_STATS_UPDATED payload). The lean
            # neuromod_state.bin is a fixed (6,4) array (levels only) and can't
            # carry the emotion string; the classifier result rides here so
            # /status/mood + /status.lifetime.emotion can surface it instead of
            # the "unknown" default.
            "current_emotion": str(
                (neuromod_stats or {}).get("current_emotion", "neutral")
                if isinstance(neuromod_stats, dict) else "neutral"),
            "emotion_confidence": float(
                (neuromod_stats or {}).get("emotion_confidence", 0.0)
                if isinstance(neuromod_stats, dict) else 0.0),
            "schema_version": META_REASONING_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "total_meta_chains": 0,
            "total_introspect_picks": 0,
            "total_introspect_executions": 0,
            "monoculture_score": 0.0,
            "primitive_distribution": {},
            "last_chain_id": 0,
            "last_chain_reason": "",
            "last_chain_succeeded": False,
            "subsystem_signals_status": {},
            "meta_cgn": {},
            "total_eurekas": 0,
            "total_meta_steps": 0,
            "total_wisdom_saved": 0,
            "neural_maturity": 0.0,
            "primitive_counts": {},
            "pi_heartbeat": {},
            "current_emotion": "neutral",
            "emotion_confidence": 0.0,
            "schema_version": META_REASONING_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
