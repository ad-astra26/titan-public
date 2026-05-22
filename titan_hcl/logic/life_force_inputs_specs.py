"""
life_force_inputs_specs — shared RegistrySpec for life_force_inputs.bin SHM slot.

Phase C v1.8.3 SPEC bump (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy
§4.G. Cross-process inputs bridge feeding life_force_worker.evaluate per
KERNEL_EPOCH_TICK. Mirrors the §4.Q `neuromod_inputs.bin` pattern (same
writer/reader roles, same cadence).

Roles:
  - producer: titan_hcl.logic.life_force_inputs_publisher.LifeForceInputsPublisher
    (invoked from cognitive_worker @ KERNEL_EPOCH_TICK; G21 single-writer)
  - consumer: titan_hcl.modules.life_force_worker.life_force_worker_main
    (reads via StateRegistryReader on every KERNEL_EPOCH_TICK arrival,
     feeds the 16 inputs to LifeForceEngine.evaluate)

Slot is variable-size msgpack per established pattern (matches
neuromod_inputs.bin / metabolism_state.bin / etc.).

Payload schema v1 (msgpack — 16 inputs total):
  Spirit inputs (4):
    "pi_heartbeat_ratio":         float ∈ [0,1]
    "developmental_age":          int   (epochs; ~0..MATURE_START=1000)
    "sovereignty_index":          int   (post-mint count; 0..10000)
    "spirit_coherence":           float ∈ [0,1]
  Mind inputs (6):
    "vocabulary_size":            int
    "learning_rate_gain":         float (1.0 = neutral; 0.5..2.0 healthy)
    "emotional_coherence":        float ∈ [0,1]
    "neuromodulator_homeostasis": float ∈ [0,1]
    "mind_coherence":             float ∈ [0,1]
    "expression_fire_rate":       float ∈ [0,1] (normalized)
  Body inputs (6):
    "sol_balance":                float (SOL units; healthy ≥ 0.5)
    "anchor_freshness":           float ∈ [0,1]
    "hormonal_vitality":          float ∈ [0,1]
    "body_coherence":             float ∈ [0,1]
    "topology_grounding":         float ∈ [0,1]
    "infrastructure_health":      float ∈ [0,1]
  Meta:
    "schema_version":             int   (= LIFE_FORCE_INPUTS_SCHEMA_VERSION)
    "ts":                         float (publisher wall-time at write)
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    LIFE_FORCE_INPUTS_MAX_BYTES,
    LIFE_FORCE_INPUTS_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


LIFE_FORCE_INPUTS_SLOT = "life_force_inputs"

LIFE_FORCE_INPUTS_SPEC = RegistrySpec(
    name=LIFE_FORCE_INPUTS_SLOT,
    dtype=np.dtype("uint8"),
    shape=(LIFE_FORCE_INPUTS_MAX_BYTES,),
    schema_version=LIFE_FORCE_INPUTS_SCHEMA_VERSION,
    variable_size=True,
)
