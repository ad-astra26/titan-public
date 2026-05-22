"""
life_force_state_specs — shared RegistrySpec for life_force_state.bin SHM slot.

Phase C v1.8.3 SPEC bump (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy
§4.G. Single source of truth shared by:

  - producer: titan_hcl.logic.life_force_state_publisher.LifeForceStatePublisher
    (invoked from life_force_worker @ 1 Hz; G21 single-writer)
  - consumers:
      * titan_hcl.proxies.life_force_proxy.LifeForceShmReader
        (SHM-direct reads for get_chi_total + get_metabolic_drain —
         hot-path sub-µs reads per G18+G20)
      * cognitive_worker (5 reader sites — MSL static_context chi_total,
        reasoning body_state, hormonal_pressure inputs, ground_up_enricher
        chi_overlay, NN modulation cap)
      * api_subprocess (/v4/chi route via chi.state cache key —
        populated by CHI_UPDATED bus event whose producer is now
        life_force_worker)
      * metabolism_worker (SOFT-dep tier weighting per
        metabolism_worker.py:93-95 NULL-safe subscriber, wired in
        v1.7.2 awaiting this slot)

Slot is variable-size msgpack per the established pattern (matches
metabolism_state.bin / social_graph_state.bin / dream_state.bin from
v1.7.x — v1.8.x carves).

Payload schema (msgpack):
  {
    "total":              float,  # ∈ [0,1] composite chi
    "spirit":             dict,   # ChiLayer: {raw, effective, weight,
                                  #            thinking, feeling, willing,
                                  #            components: dict}
    "mind":               dict,   # ChiLayer (same shape)
    "body":               dict,   # ChiLayer (same shape) — drain-adjusted
    "circulation":        float,  # chi flow rate (d_spirit + d_mind + d_body)
    "weights":            dict,   # {spirit, mind, body} — adaptive by age
    "state":              str,    # FLOURISHING/HEALTHY/CONSERVING/SURVIVAL/STARVATION
    "developmental_phase": str,   # BIRTH/YOUTH/MATURE
    "contemplation":      dict,   # {active, phase ∈ [0,4], phase_name,
                                  #  conviction, conviction_threshold,
                                  #  mature_enough, survival_mode?, action?}
    "metabolic_drain":    float,  # ∈ [0,0.8] adenosine-like accumulator
    "is_dreaming":        bool,   # cached from DREAM_STATE_CHANGED
    "schema_version":     int,    # = LIFE_FORCE_STATE_SCHEMA_VERSION
    "ts":                 float,  # publisher wall-time at write
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    LIFE_FORCE_STATE_MAX_BYTES,
    LIFE_FORCE_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


LIFE_FORCE_STATE_SLOT = "life_force_state"

LIFE_FORCE_STATE_SPEC = RegistrySpec(
    name=LIFE_FORCE_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(LIFE_FORCE_STATE_MAX_BYTES,),
    schema_version=LIFE_FORCE_STATE_SCHEMA_VERSION,
    variable_size=True,
)
