"""
meta_reasoning_state_specs — shared RegistrySpec for meta_reasoning_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: cognitive_worker (G21 single-writer — MetaReasoningEngine
    instance lives in cognitive_worker per SPEC §1 glossary line 321)
  - consumers:
      * api_subprocess StateAccessor.spirit.get_coordinator overlay
        (meta_reasoning key) — replaces meta_reasoning.state bus-cache
      * dashboard /v4/meta-reasoning endpoint

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "total_meta_chains":             int,
    "total_introspect_picks":        int,
    "total_introspect_executions":   int,
    "monoculture_score":             float,
    "primitive_distribution":        dict[str→float],  # share per primitive
    "last_chain_id":                 int,
    "last_chain_reason":             str,
    "last_chain_succeeded":          bool,
    "subsystem_signals_status":      dict,            # live=N, dead=14-N
    "schema_version":                int,
    "ts":                            float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    META_REASONING_STATE_MAX_BYTES,
    META_REASONING_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


META_REASONING_STATE_SLOT = "meta_reasoning_state"

META_REASONING_STATE_SPEC = RegistrySpec(
    name=META_REASONING_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(META_REASONING_STATE_MAX_BYTES,),
    schema_version=META_REASONING_STATE_SCHEMA_VERSION,
    variable_size=True,
)
