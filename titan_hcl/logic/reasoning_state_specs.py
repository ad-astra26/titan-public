"""
reasoning_state_specs — shared RegistrySpec for reasoning_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: cognitive_worker (G21 single-writer — ReasoningEngine instance
    lives in cognitive_worker per SPEC §1 glossary line 321)
  - consumers:
      * api_subprocess StateAccessor.reasoning (replaces reasoning.stats
        bus-cache) + StateAccessor.memory.get_reasoning_state (replaces
        memory.reasoning_state bus-cache)
      * dashboard /v4/reasoning endpoint
      * SpiritAccessor.get_coordinator overlay (reasoning key)

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "total_chains":          int,
    "total_commits":         int,
    "commit_rate":           float,
    "avg_chain_length":      float,
    "buffer_size":           int,
    "current_active":        bool,
    "last_action":           str,
    "last_outcome":          str,
    "action_distribution":   dict[str→int],
    "schema_version":        int,
    "ts":                    float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    REASONING_STATE_MAX_BYTES,
    REASONING_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


REASONING_STATE_SLOT = "reasoning_state"

REASONING_STATE_SPEC = RegistrySpec(
    name=REASONING_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(REASONING_STATE_MAX_BYTES,),
    schema_version=REASONING_STATE_SCHEMA_VERSION,
    variable_size=True,
)
