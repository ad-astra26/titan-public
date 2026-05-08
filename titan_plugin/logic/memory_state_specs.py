"""
memory_state_specs — shared RegistrySpec for memory_state.bin SHM slot.

Phase C Session 2 of rFP_phase_c_async_shm_consumer_migration §4.B.8 +
§4.C.13. Single source of truth shared by:
  - producer: titan_plugin.logic.memory_state_publisher.MemoryStatePublisher
    (invoked from memory_worker @ 1 Hz)
  - consumer: titan_plugin.proxies.memory_proxy.MemoryProxy
    (SHM-direct reads for get_growth_metrics, get_memory_status,
    get_persistent_count — the 3 state-lookup methods that py-spy
    proved blocking sidecars on T3 2026-05-07 post-Session-1)

Slot is variable-size msgpack per the established pattern (matches
hormone_fires/impulse_engine_state/etc. from Session 1).
"""
from __future__ import annotations

import numpy as np

from titan_plugin._phase_c_constants import (
    MEMORY_STATE_MAX_BYTES,
    MEMORY_STATE_SCHEMA_VERSION,
)
from titan_plugin.core.state_registry import RegistrySpec


MEMORY_STATE_SLOT = "memory_state"

MEMORY_STATE_SPEC = RegistrySpec(
    name=MEMORY_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(MEMORY_STATE_MAX_BYTES,),
    schema_version=MEMORY_STATE_SCHEMA_VERSION,
    variable_size=True,
)
