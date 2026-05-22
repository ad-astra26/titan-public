"""
cgn_engine_state_specs — shared RegistrySpec for cgn_engine_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: cgn_worker (G21 single-writer; sibling to existing
    cgn_live_weights.bin tensor + cgn_beta_state.bin 8-float per-consumer
    reward EMA — this slot carries the engine-level stats)
  - consumers:
      * api_subprocess StateAccessor.cgn (replaces cgn.stats bus-cache)
      * dashboard /v4/meta-cgn + /v4/cgn-status endpoints

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "consumers":          dict[str→dict],   # per-consumer engine state
    "total_transitions":  int,
    "buffer_size":        int,
    "consolidations":     int,
    "anchor_count":       int,
    "sigma_updates":      int,
    "soar_impasses":      int,
    "haov_stats":         dict,             # per-consumer haov tracker stats
    "avg_reward":         float,            # Phase C dissolve — outer-trinity dim input
    "grounded_density":   float,            # Phase C dissolve — outer-trinity dim input
    "schema_version":     int,
    "ts":                 float,
  }

Phase C dissolve (2026-05-22): `avg_reward` + `grounded_density` added (additive,
no schema bump) so the outer-source sidecars read CGN dim inputs SHM-direct,
retiring the CGN_STATS_UPDATED bus-cache path (Preamble G18).

Closes the cgn.stats bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    CGN_ENGINE_STATE_MAX_BYTES,
    CGN_ENGINE_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


CGN_ENGINE_STATE_SLOT = "cgn_engine_state"

CGN_ENGINE_STATE_SPEC = RegistrySpec(
    name=CGN_ENGINE_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(CGN_ENGINE_STATE_MAX_BYTES,),
    schema_version=CGN_ENGINE_STATE_SCHEMA_VERSION,
    variable_size=True,
)
