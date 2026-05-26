"""
meta_teacher_state_specs — shared RegistrySpec for meta_teacher_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: cognitive_worker (G21 single-writer — MetaTeacherEngine
    instance hosted alongside MetaReasoning/Reasoning)
  - consumers:
      * api_subprocess StateAccessor.meta_teacher (replaces meta_teacher.stats
        bus-cache)
      * dashboard /v4/meta-teacher/* endpoints

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack — variable-size, additive-only per `feedback_implement_rfp_fully_no_simplifications_no_deferrals` + D-SPEC-103 dream_state precedent):
  {
    # v1 fields (D-SPEC-71 v1.10.0):
    "total_critiques":          int,
    "voice_tuning_enabled":     bool,
    "peer_exchange_enabled":    bool,
    "last_critique_ts":         float,
    "per_domain_critiques":     dict[str→int],
    "adoption_rate":            float,
    "schema_version":           int,
    "ts":                       float,

    # rFP_teachers_update F5 (2026-05-26): additive fields. Schema version
    # unchanged — variable msgpack, backward-compat (older readers ignore
    # unknown keys per the D-SPEC-103 dream_state precedent). Payload still
    # well under META_TEACHER_STATE_MAX_BYTES (4096); typical ≈ 1.5-2 KB.
    "enabled":                       bool,
    "sample_mode":                   str,
    "task_key":                      str,
    "max_critiques_per_hour":        int,
    "reward_weight_config":          float,
    "reward_weight_cap":             float,
    "grounding_weight":              float,
    "critiques_24h":                 int,       # 24h-windowed from in-mem deque
    "llm_ok_24h":                    int,
    "llm_failed_24h":                int,
    "avg_quality_score_24h":         float,
    "top_critique_categories":       list[[str, int]],  # top-5 24h
    "adoption_prompt_version":       int,
    "adoption_rate_by_domain":       dict[str→float],
    "adoption_rate_overall":         float,
    "teaching_memory_enabled":       bool,
    "memory_cold_tier_topics":       int,       # from teacher_memory.snapshot()
    "memory_still_needs_push_count": int,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    META_TEACHER_STATE_MAX_BYTES,
    META_TEACHER_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


META_TEACHER_STATE_SLOT = "meta_teacher_state"

META_TEACHER_STATE_SPEC = RegistrySpec(
    name=META_TEACHER_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(META_TEACHER_STATE_MAX_BYTES,),
    schema_version=META_TEACHER_STATE_SCHEMA_VERSION,
    variable_size=True,
)
