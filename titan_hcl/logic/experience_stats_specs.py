"""
experience_stats_specs — shared RegistrySpec for experience_stats.bin SHM slot.

§3L Phase 15 chunk 15.1 (D-SPEC-PHASE15). Single source of truth shared by:

  - producer: cognitive_worker (G21 single-writer — ExperienceOrchestrator
    instance hosted alongside MetaReasoning/Reasoning/MetaTeacher)
  - consumers:
      * api_subprocess StateAccessor.experience (SHM-direct, INV-1/INV-2)
      * in-proc coord-snapshot build (snapshot_builders.build_coordinator_snapshot)
      * neuromod_inputs_builder._prediction_state (load-bearing cognition)

Slot is variable-size msgpack per the established Phase C Python L2 pattern
(mirrors meta_teacher_state.bin / D-SPEC-136).

Replaces the retired ExperienceMemory.get_stats recompute-on-read path: the
former store's sole writer (spirit_worker) was deleted in D8-3/72f95a6b, leaving
data/experience_memory.db frozen fleet-wide since 2026-05-14. ExperienceOrchestrator
is the live successor — it maintains INCREMENTAL per-(domain, action) aggregates in
its action_stats table (running average + success counts updated O(1) on each
distill), so this slot is computed from a small bounded table, never a 142k-row
GROUP BY.

Payload schema (msgpack — variable-size, additive-only per the D-SPEC-103
dream_state precedent — older readers ignore unknown keys):
  {
    "total_records":   int,                                   # experience_records count (live)
    "undistilled":     int,
    "total_wisdom":    int,                                   # distilled_wisdom count
    "by_domain":       dict[str→{count: int, avg_score: float, success_rate: float}],
    "schema_version":  int,
    "ts":              float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    EXPERIENCE_STATS_MAX_BYTES,
    EXPERIENCE_STATS_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


EXPERIENCE_STATS_SLOT = "experience_stats"

EXPERIENCE_STATS_SPEC = RegistrySpec(
    name=EXPERIENCE_STATS_SLOT,
    dtype=np.dtype("uint8"),
    shape=(EXPERIENCE_STATS_MAX_BYTES,),
    schema_version=EXPERIENCE_STATS_SCHEMA_VERSION,
    variable_size=True,
)
