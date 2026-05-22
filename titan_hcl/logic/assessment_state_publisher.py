"""
assessment_state_publisher — Phase C Session 3 §4.B.2.

Publishes assessment_state.bin from a ``SelfAssessment`` instance
(`titan_hcl/logic/agency/assessment.py`). Owned by agency_worker
(co-located with AgencyModule that holds the SelfAssessment instance).

Source: ``SelfAssessment.get_stats()`` returns
``{average_score, total, recent: [last 5 entries], trend, score_variance}``
(plus research_avg_score in some shapes). Publisher mirrors that shape +
``ts``.
"""
from __future__ import annotations

from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.session3_state_specs import (
    ASSESSMENT_STATE_SLOT,
    ASSESSMENT_STATE_SPEC,
)


def _json_clean_keys(obj: Any) -> Any:
    """Recursively coerce dict keys to str so the payload is JSON/msgpack-
    portable. Phase C root-fix (2026-05-22): assessment records carry trinity
    dim-index ENRICHMENT keyed by INT (`_compute_enrichment` → {layer:{dim_int:
    delta}}). Int map keys are valid msgpack but (a) rejected by the default
    strict_map_key=True decoder and (b) illegal in JSON (the api serves state as
    JSON) — so assessment_state.bin was unreadable fleet-wide whenever a
    non-neutral score produced a non-empty enrichment. We stringify ONLY at this
    SHM-serialization boundary; the in-process + bus enrichment (consumed
    int-keyed for dim-delta application at plugin.py:2964/3140) is untouched."""
    if isinstance(obj, dict):
        return {str(k): _json_clean_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean_keys(v) for v in obj]
    return obj


class AssessmentStatePublisher(BaseStatePublisher):
    slot_name = ASSESSMENT_STATE_SLOT
    slot_spec = ASSESSMENT_STATE_SPEC

    def _compute_payload(self, assessment: Any) -> dict[str, Any]:
        import time
        if assessment is None or not hasattr(assessment, "get_stats"):
            return {
                "average_score": 0.0,
                "total": 0,
                "recent": [],
                "trend": 0.0,
                "score_variance": 0.0,
                "research_avg_score": 0.0,
                "ts": time.time(),
            }
        stats = dict(assessment.get_stats() or {})
        # Coerce types defensively (msgpack stability)
        return {
            "average_score": float(stats.get("average_score", 0.0)),
            "total": int(stats.get("total", 0)),
            # _json_clean_keys: stringify int dim-index enrichment keys so the
            # slot is JSON/msgpack-portable (root-fix; see helper docstring).
            "recent": _json_clean_keys(list(stats.get("recent", []) or [])),
            "trend": float(stats.get("trend", 0.0)),
            "score_variance": float(stats.get("score_variance", 0.0)),
            "research_avg_score": float(
                stats.get("research_avg_score", 0.0)),
            "ts": time.time(),
        }
