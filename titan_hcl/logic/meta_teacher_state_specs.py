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

Payload schema (msgpack):
  {
    "total_critiques":          int,
    "voice_tuning_enabled":     bool,
    "peer_exchange_enabled":    bool,
    "last_critique_ts":         float,
    "per_domain_critiques":     dict[str→int],
    "adoption_rate":            float,
    "schema_version":           int,
    "ts":                       float,
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
