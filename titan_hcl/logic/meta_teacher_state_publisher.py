"""
meta_teacher_state_publisher — MetaTeacherStatePublisher writes
meta_teacher_state.bin SHM slot.

Producer for meta_teacher_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0).
G21 single-writer contract: only cognitive_worker publishes here.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.meta_teacher_state_specs import (
    META_TEACHER_STATE_SLOT,
    META_TEACHER_STATE_SPEC,
)
from titan_hcl._phase_c_constants import META_TEACHER_STATE_SCHEMA_VERSION


class MetaTeacherStatePublisher(BaseStatePublisher):
    slot_name = META_TEACHER_STATE_SLOT
    slot_spec = META_TEACHER_STATE_SPEC

    def _compute_payload(self, meta_teacher: Any) -> dict[str, Any]:
        if meta_teacher is None:
            return self._stub()
        try:
            stats = meta_teacher.get_stats() if hasattr(
                meta_teacher, "get_stats") else {}
        except Exception:
            stats = {}
        per_domain = stats.get("per_domain_critiques", {}) or {}
        if not isinstance(per_domain, dict):
            per_domain = {}
        return {
            "total_critiques": int(stats.get("total_critiques", 0) or 0),
            "voice_tuning_enabled": bool(
                stats.get("voice_tuning_enabled", False)),
            "peer_exchange_enabled": bool(
                stats.get("peer_exchange_enabled", False)),
            "last_critique_ts": float(
                stats.get("last_critique_ts", 0.0) or 0.0),
            "per_domain_critiques": {
                str(k): int(v) for k, v in per_domain.items()
                if isinstance(v, (int, float))
            },
            "adoption_rate": float(stats.get("adoption_rate", 0.0) or 0.0),
            "schema_version": META_TEACHER_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "total_critiques": 0,
            "voice_tuning_enabled": False,
            "peer_exchange_enabled": False,
            "last_critique_ts": 0.0,
            "per_domain_critiques": {},
            "adoption_rate": 0.0,
            "schema_version": META_TEACHER_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
