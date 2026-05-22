"""
metabolism_state_specs — shared RegistrySpec for metabolism_state.bin SHM slot.

Phase C v1.7.2 SPEC bump (D-SPEC-51) per rFP_titan_hcl_l2_separation_strategy
§4.J. Single source of truth shared by:

  - producer: titan_hcl.logic.metabolism_state_publisher.MetabolismStatePublisher
    (invoked from metabolism_worker @ 1 Hz; G21 single-writer)
  - consumers:
      * titan_hcl.proxies.metabolism_proxy.MetabolismProxy
        (SHM-direct reads for get_metabolic_tier / get_gates_enforced /
         gates_enforced / get_tier_info / get_last_gate_decision_reason —
         hot-path sub-ms reads per G18+G20)
      * titan_hcl.core.soul.MetabolismShmReader
        (kernel-level Soul reads tier + gates_enforced for NFT mint gate
         after the v1.7.2 removal of `Soul.set_metabolism` reverse-injection
         — see SPEC §9.B metabolism_worker block "Soul migration" note)
      * dashboard /v4/metabolism/* endpoints (read once per request)

Slot is variable-size msgpack per the established pattern (matches
memory_state.bin / social_graph_state.bin / mind_state.bin / etc. from
Sessions 1-5 of rFP_phase_c_async_shm_consumer_migration).

Payload schema (msgpack):
  {
    "tier":                       str,    # one of HEALTHY/CONSERVATIVE/SURVIVAL/EMERGENCY/HIBERNATION/EXTINCT
    "balance_pct":                float,  # SOL balance as % of stable_balance
    "gates_enforced":             bool,   # kill-switch state from titan_params.toml [metabolism]
    "last_gate_decision_reason":  str,    # human-readable reason of last evaluate_gate decision
    "tier_info":                  dict,   # MetabolismController.get_tier_info() output —
                                          # {tier, gates_enforced, balance, last_balance_pct,
                                          #  emergency_duration_s, in_emergency, ...}
    "last_tier_change_ts":        float,  # wall-time of last tier transition
    "social_gravity_score":       float,  # cached gravity score (0..1) for /status mood ribbon
    "schema_version":             int,    # = METABOLISM_STATE_SCHEMA_VERSION
    "ts":                         float,  # publisher wall-time at write
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    METABOLISM_STATE_MAX_BYTES,
    METABOLISM_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


METABOLISM_STATE_SLOT = "metabolism_state"

METABOLISM_STATE_SPEC = RegistrySpec(
    name=METABOLISM_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(METABOLISM_STATE_MAX_BYTES,),
    schema_version=METABOLISM_STATE_SCHEMA_VERSION,
    variable_size=True,
)
