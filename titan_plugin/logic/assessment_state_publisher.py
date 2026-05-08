"""
assessment_state_publisher — Phase C Session 3 §4.B.2.

Publishes assessment_state.bin from a ``SelfAssessment`` instance
(`titan_plugin/logic/agency/assessment.py`). Owned by agency_worker
(co-located with AgencyModule that holds the SelfAssessment instance).

Source: ``SelfAssessment.get_stats()`` returns
``{average_score, total, recent: [last 5 entries], trend, score_variance}``
(plus research_avg_score in some shapes). Publisher mirrors that shape +
``ts``.
"""
from __future__ import annotations

from typing import Any

from titan_plugin.logic.base_state_publisher import BaseStatePublisher
from titan_plugin.logic.session3_state_specs import (
    ASSESSMENT_STATE_SLOT,
    ASSESSMENT_STATE_SPEC,
)


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
            "recent": list(stats.get("recent", []) or []),
            "trend": float(stats.get("trend", 0.0)),
            "score_variance": float(stats.get("score_variance", 0.0)),
            "research_avg_score": float(
                stats.get("research_avg_score", 0.0)),
            "ts": time.time(),
        }
