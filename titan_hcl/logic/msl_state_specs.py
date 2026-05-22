"""
msl_state_specs — shared RegistrySpec for msl_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: cognitive_worker (G21 single-writer — MSL engine lives in
    cognitive_worker per SPEC §1 glossary preamble_extensions_pending
    "L2 perception (MSL Multisensory Synthesis Layer + sensory wiring)")
  - consumers:
      * api_subprocess StateAccessor.spirit.get_coordinator overlay
        (msl key) — replaces msl.state bus-cache

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "synthesis_count":          int,
    "novel_associations":       int,
    "cross_modal_bindings":     int,
    "decay_rate":               float,
    "current_capacity":         int,
    "schema_version":           int,
    "ts":                       float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    MSL_STATE_MAX_BYTES,
    MSL_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


MSL_STATE_SLOT = "msl_state"

MSL_STATE_SPEC = RegistrySpec(
    name=MSL_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(MSL_STATE_MAX_BYTES,),
    schema_version=MSL_STATE_SCHEMA_VERSION,
    variable_size=True,
)
