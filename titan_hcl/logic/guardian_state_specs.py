"""
guardian_state_specs — shared RegistrySpec for guardian_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: guardian (Python L1 supervisor — `titan_hcl/guardian.py`,
    G21 single-writer; guardian publishes its own status to SHM so api_subprocess
    + dashboard can read per-module liveness without a bus snapshot push)
  - consumers:
      * api_subprocess StateAccessor.guardian.get_status (replaces
        guardian.status bus-cache)
      * StateAccessor.guardian.get_modules_by_layer
      * dashboard /v4/state.guardian section (currently goes through
        request.app.state.titan_hcl.guardian.get_status which is the
        kernel_rpc proxy path — this slot lets api_subprocess read SHM
        directly, removing the cross-process call)

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "modules":              dict[str→dict],  # per-module {state, pid, rss_mb,
                                              # uptime, restart_count,
                                              # restarts_in_window,
                                              # last_heartbeat_age, layer,
                                              # start_method, adopted, adopt_ts}
    "total_modules":        int,
    "modules_by_layer":     dict[str→list[str]],  # {"L1": [...], "L2": [...], "L3": [...]}
    "escalation_count":     int,
    "schema_version":       int,
    "ts":                   float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    GUARDIAN_STATE_MAX_BYTES,
    GUARDIAN_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


GUARDIAN_STATE_SLOT = "guardian_state"

GUARDIAN_STATE_SPEC = RegistrySpec(
    name=GUARDIAN_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(GUARDIAN_STATE_MAX_BYTES,),
    schema_version=GUARDIAN_STATE_SCHEMA_VERSION,
    variable_size=True,
)
