"""
msl_state_publisher — MSLStatePublisher writes msl_state.bin SHM slot.

Producer for msl_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only cognitive_worker publishes here (MSL engine
lives in cognitive_worker per SPEC §1 glossary preamble_extensions_pending
"L2 perception (MSL Multisensory Synthesis Layer + sensory wiring)").

Closes the msl.state bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.msl_state_specs import (
    MSL_STATE_SLOT,
    MSL_STATE_SPEC,
)
from titan_hcl._phase_c_constants import MSL_STATE_SCHEMA_VERSION


class MSLStatePublisher(BaseStatePublisher):
    slot_name = MSL_STATE_SLOT
    slot_spec = MSL_STATE_SPEC

    def _compute_payload(self, msl: Any) -> dict[str, Any]:
        if msl is None:
            return self._stub()
        try:
            stats = msl.get_stats() if hasattr(msl, "get_stats") else {}
        except Exception:
            stats = {}
        # ── Additive extension: i_confidence + i_depth + eureka_count ──
        # MultisensorySynthesisLayer doesn't define a top-level get_stats(),
        # so `stats` above is empty in practice; pull the values directly
        # from the engine's public attributes the way save_all() does (msl.py
        # L2520-2525). i_depth is a float in IDepthTracker.depth and the
        # eureka_count comes from IDepthTracker._eureka_count (canonical
        # accumulator for the dashboard Insights card).
        i_confidence = 0.0
        i_depth_value = 0.0
        eureka_count = 0
        wisdom_count = 0
        try:
            conf = getattr(msl, "confidence", None)
            if conf is not None:
                i_confidence = float(getattr(conf, "confidence", 0.0) or 0.0)
        except Exception:
            pass
        i_depth_components: dict[str, float] = {}
        try:
            idepth = getattr(msl, "i_depth", None)
            if idepth is not None:
                i_depth_value = float(getattr(idepth, "depth", 0.0) or 0.0)
                eureka_count = int(getattr(idepth, "_eureka_count", 0) or 0)
                wisdom_count = int(getattr(idepth, "_wisdom_count", 0) or 0)
                # 5-component breakdown for the IDepthTab "I-Depth Components"
                # bars — IDepthTracker.get_stats()["components"].
                try:
                    _id_stats = idepth.get_stats() if hasattr(idepth, "get_stats") else {}
                    _comp = _id_stats.get("components", {}) if isinstance(_id_stats, dict) else {}
                    if isinstance(_comp, dict):
                        i_depth_components = {
                            str(k): round(float(v), 4)
                            for k, v in _comp.items()
                            if isinstance(v, (int, float))
                        }
                except Exception:
                    pass
        except Exception:
            pass
        # ── Concept confidences for the IDepthTab "Concept Confidences /
        # MSL grounded concepts" grid. ConceptGrounder.get_concept_confidences()
        # → dict[concept→0-1]. Bounded to the top 30 by confidence to respect
        # the msl_state.bin payload cap.
        concept_confidences: dict[str, float] = {}
        try:
            cg = getattr(msl, "concept_grounder", None)
            if cg is not None and hasattr(cg, "get_concept_confidences"):
                _all = cg.get_concept_confidences() or {}
                if isinstance(_all, dict):
                    _top = sorted(
                        ((str(k), float(v)) for k, v in _all.items()
                         if isinstance(v, (int, float))),
                        key=lambda kv: kv[1], reverse=True)[:30]
                    concept_confidences = {k: round(v, 4) for k, v in _top}
        except Exception:
            pass
        return {
            "synthesis_count": int(stats.get("synthesis_count", 0) or 0),
            "novel_associations": int(
                stats.get("novel_associations", 0) or 0),
            "cross_modal_bindings": int(
                stats.get("cross_modal_bindings", 0) or 0),
            "decay_rate": float(stats.get("decay_rate", 0.0) or 0.0),
            "current_capacity": int(stats.get("current_capacity", 0) or 0),
            # Additive (D-SPEC-71 v1.10.0 → v2 additive): identity telemetry
            # that the dashboard /status.lifetime + Cognitive Depth card +
            # Insights card consume. Reader-tolerant (defaults to 0).
            "i_confidence": round(i_confidence, 4),
            "i_depth": round(i_depth_value, 4),
            "eureka_count": eureka_count,
            "wisdom_count": wisdom_count,
            # IDepthTab "I-Depth Components" + "Concept Confidences" panels.
            "i_depth_components": i_depth_components,
            "concept_confidences": concept_confidences,
            "schema_version": MSL_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "synthesis_count": 0,
            "novel_associations": 0,
            "cross_modal_bindings": 0,
            "decay_rate": 0.0,
            "current_capacity": 0,
            "i_confidence": 0.0,
            "i_depth": 0.0,
            "eureka_count": 0,
            "wisdom_count": 0,
            "i_depth_components": {},
            "concept_confidences": {},
            "schema_version": MSL_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
