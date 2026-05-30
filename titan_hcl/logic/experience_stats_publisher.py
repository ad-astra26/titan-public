"""
experience_stats_publisher — ExperienceStatsPublisher writes
experience_stats.bin SHM slot.

Producer for experience_stats slot per SPEC §7.1 (§3L Phase 15 chunk 15.1 /
D-SPEC-PHASE15). G21 single-writer contract: only cognitive_worker publishes
here (it owns the ExperienceOrchestrator instance).

Replaces the retired ExperienceMemory.get_stats recompute-on-read path per G18:
the aggregate is sourced from ExperienceOrchestrator's INCREMENTAL action_stats
(bounded table, O(1)-maintained), never a 142k-row GROUP BY.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.experience_stats_specs import (
    EXPERIENCE_STATS_SLOT,
    EXPERIENCE_STATS_SPEC,
)
from titan_hcl._phase_c_constants import EXPERIENCE_STATS_SCHEMA_VERSION


class ExperienceStatsPublisher(BaseStatePublisher):
    slot_name = EXPERIENCE_STATS_SLOT
    slot_spec = EXPERIENCE_STATS_SPEC

    def _compute_payload(self, exp_orchestrator: Any = None) -> dict[str, Any]:
        if exp_orchestrator is None or not hasattr(
                exp_orchestrator, "get_experience_stats_payload"):
            return self._stub()
        try:
            stats = exp_orchestrator.get_experience_stats_payload() or {}
        except Exception:
            return self._stub()
        by_domain = stats.get("by_domain", {}) or {}
        if not isinstance(by_domain, dict):
            by_domain = {}
        return {
            "total_records": int(stats.get("total_records", 0) or 0),
            "undistilled": int(stats.get("undistilled", 0) or 0),
            "total_wisdom": int(stats.get("total_wisdom", 0) or 0),
            "by_domain": {
                str(k): {
                    "count": int(v.get("count", 0) or 0),
                    "avg_score": float(v.get("avg_score", 0.0) or 0.0),
                    "success_rate": float(v.get("success_rate", 0.0) or 0.0),
                }
                for k, v in by_domain.items()
                if isinstance(v, dict)
            },
            "schema_version": EXPERIENCE_STATS_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "total_records": 0,
            "undistilled": 0,
            "total_wisdom": 0,
            "by_domain": {},
            "schema_version": EXPERIENCE_STATS_SCHEMA_VERSION,
            "ts": time.time(),
        }
