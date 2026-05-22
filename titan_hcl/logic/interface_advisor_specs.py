"""
interface_advisor_specs — shared RegistrySpec for interface_advisor_state.bin
SHM slot.

Phase C v1.8.5 SPEC bump (D-SPEC-59) per `rFP_titan_hcl_l2_separation_strategy.md
§4.H`. Maker greenlit 2026-05-15 inline (SHM-rate-oracle pattern).

Single source of truth shared by:
  - producer: `titan_hcl.logic.interface_advisor_publisher.
    InterfaceAdvisorStatePublisher` (invoked from interface_advisor_worker
    on every IMPULSE_RECEIVED arrival — rate-throttled to
    INTERFACE_ADVISOR_RATE_REFRESH_CADENCE_S = 0.1s = 10Hz cap so SHM
    doesn't thrash under burst; G21 single-writer)
  - consumers:
      * `titan_hcl.logic.interface_advisor_reader.
        InterfaceAdvisorStateReader` (sub-µs G18 SHM-direct read with
        100ms cache; used by parent `_handle_impulse` rate check —
        replaces the in-proc `self._interface_advisor.check()` synchronous
        path retired in v1.8.5 §4.H)

Payload schema (msgpack, ≤INTERFACE_ADVISOR_STATE_MAX_BYTES=512 bytes):
  {
    "rates":             dict[str, int],  # msg_type → current_rate_in_window
                                          # (e.g. {"IMPULSE": 1, "INTERFACE_INPUT": 3})
    "limits":            dict[str, int],  # msg_type → limit (from INITIAL_LIMITS)
    "window_s":          float,           # window size (DEFAULT_WINDOW = 60.0)
    "rate_limit_count":  int,             # cumulative rate-limit-exceeded count
                                          # (mirrors InterfaceAdvisor._rate_limit_count)
    "schema_version":    int,             # = INTERFACE_ADVISOR_STATE_SCHEMA_VERSION
    "ts":                float,           # publisher wall-time at write
  }

Semantics shift from pre-carve InterfaceAdvisor:
  - Pre-carve: parent `_handle_impulse` called `self._interface_advisor.check(IMPULSE)`
    synchronously — atomic check+record in one µs-scale in-proc call.
  - Post-carve: parent reads `rates[msg_type]` from SHM (sub-µs G18) — if
    `rate >= limit`, parent skips + emits RATE_LIMIT feedback locally;
    else parent emits IMPULSE_RECEIVED bus event (fire-and-forget) and
    proceeds. Worker receives event, records timestamp in deque, publishes
    new SHM snapshot. Eventually-consistent within bus latency (~10-50ms).
    For IMPULSE limit=1/60s and ACTION_RESULT limit=3/60s this is well
    within tolerance.

Closes the in-code ADR at `plugin.py:1970-1984` ("InterfaceAdvisor stays
in parent — cheap rate check before bus round-trip") which was pre-Phase-C-
fleet-wide. Per `feedback_phase_c_break_monolith_ethos.md`: every L2 carve
under Phase C earns its place via hot-reload + restart-isolation + own
§9.B block — perf is not the criterion.
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    INTERFACE_ADVISOR_STATE_MAX_BYTES,
    INTERFACE_ADVISOR_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


INTERFACE_ADVISOR_STATE_SLOT = "interface_advisor_state"

INTERFACE_ADVISOR_STATE_SPEC = RegistrySpec(
    name=INTERFACE_ADVISOR_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(INTERFACE_ADVISOR_STATE_MAX_BYTES,),
    schema_version=INTERFACE_ADVISOR_STATE_SCHEMA_VERSION,
    variable_size=True,
)
