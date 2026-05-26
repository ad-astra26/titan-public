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

    def _compute_payload(
            self, meta_teacher: Any, memory: Any = None) -> dict[str, Any]:
        if meta_teacher is None:
            return self._stub()
        try:
            stats = meta_teacher.get_stats(memory=memory) if hasattr(
                meta_teacher, "get_stats") else {}
        except Exception:
            stats = {}
        per_domain = stats.get("per_domain_critiques", {}) or {}
        if not isinstance(per_domain, dict):
            per_domain = {}
        # rFP_teachers_update F5 (2026-05-26): additive msgpack fields per
        # the D-SPEC-103 dream_state precedent (no schema bump — variable
        # msgpack, backward-compat). Older readers ignore unknown keys.
        adoption_by_domain = stats.get("adoption_rate_by_domain", {}) or {}
        if not isinstance(adoption_by_domain, dict):
            adoption_by_domain = {}
        top_cats = stats.get("top_critique_categories", []) or []
        # Coerce to [[name, count], ...] for stable msgpack encoding.
        top_cats_norm = [
            [str(t[0]), int(t[1])] for t in top_cats
            if isinstance(t, (list, tuple)) and len(t) >= 2
        ]
        return {
            # Existing v1 fields (kept for backward compat).
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
            # F5 additive: full dashboard payload moved from API file-scan
            # to G21 in-worker computation.
            "enabled": bool(stats.get("enabled", False)),
            "sample_mode": str(
                stats.get("sample_mode", "uncertainty_plus_random")),
            "task_key": str(stats.get("task_key", "meta_teacher")),
            "max_critiques_per_hour": int(
                stats.get("max_critiques_per_hour", 30) or 30),
            "reward_weight_config": float(
                stats.get("reward_weight_config", 0.05) or 0.05),
            "reward_weight_cap": float(
                stats.get("reward_weight_cap", 0.30) or 0.30),
            "grounding_weight": float(
                stats.get("grounding_weight", 0.15) or 0.15),
            "critiques_24h": int(stats.get("critiques_24h", 0) or 0),
            "llm_ok_24h": int(stats.get("llm_ok_24h", 0) or 0),
            "llm_failed_24h": int(stats.get("llm_failed_24h", 0) or 0),
            "avg_quality_score_24h": float(
                stats.get("avg_quality_score_24h", 0.0) or 0.0),
            "top_critique_categories": top_cats_norm,
            "adoption_prompt_version": int(
                stats.get("adoption_prompt_version", 1) or 1),
            "adoption_rate_by_domain": {
                str(k): float(v) for k, v in adoption_by_domain.items()
                if isinstance(v, (int, float))
            },
            "adoption_rate_overall": float(
                stats.get("adoption_rate_overall", 0.0) or 0.0),
            "teaching_memory_enabled": bool(
                stats.get("teaching_memory_enabled", False)),
            "memory_cold_tier_topics": int(
                stats.get("memory_cold_tier_topics", 0) or 0),
            "memory_still_needs_push_count": int(
                stats.get("memory_still_needs_push_count", 0) or 0),
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
