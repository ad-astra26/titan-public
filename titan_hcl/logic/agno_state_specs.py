"""
agno_state_specs — shared RegistrySpec for agno_state.bin SHM slot.

Phase C v1.17.0 SPEC bump (D-SPEC-72) per
`rFP_agno_worker_and_llm_libraries_extraction.md` + `rFP_titan_hcl_l2_separation_strategy.md §4.R`.

Single source of truth shared by:
  - producer: agno_worker (G21 single-writer; dual-trigger republish on
              KERNEL_EPOCH_TICK + immediately after every successful chat completion)
  - consumers (future): Observatory dashboard chat-volume widget, /v4/agno/stats
                        route, health_monitor_worker chat-latency plugin

Slot is variable-size msgpack per the established Python L2 slot family pattern
(matches metabolism_state.bin / social_graph_state.bin / dream_state.bin /
studio_state.bin from D-SPEC-50 + D-SPEC-51 + D-SPEC-56 + D-SPEC-57).

Payload schema (msgpack):
  {
    "schema_version":           int,    # = AGNO_STATE_SCHEMA_VERSION
    "session_count":            int,    # active Agno sessions (rows in titan_sessions table)
    "last_chat_ts":             float,  # unix epoch of last CHAT_RESPONSE (0 if none)
    "total_chats_24h":          int,    # rolling 24h chat count
    "provider_stats":           dict,   # {provider_name: {requests: int, errors: int, avg_latency_ms: float}}
    "dream_inbox_size":         int,    # buffered messages received during dream-state
    "ts":                       float,  # publisher wall-time at write
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    AGNO_STATE_MAX_BYTES,
    AGNO_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


AGNO_STATE_SLOT = "agno_state"

AGNO_STATE_SPEC = RegistrySpec(
    name=AGNO_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(AGNO_STATE_MAX_BYTES,),
    schema_version=AGNO_STATE_SCHEMA_VERSION,
    variable_size=True,
)
