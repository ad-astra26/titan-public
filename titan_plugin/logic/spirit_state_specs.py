"""
spirit_state_specs — shared RegistrySpec definitions for the 5 SHM slots
owned by spirit_worker (titan_HCL) per Phase C Session 1 of
rFP_phase_c_async_shm_consumer_migration.

Producer (spirit_state_publisher.SpiritStatePublisher) and consumer
(spirit_proxy.SpiritProxy) both import these specs so the shape contract
is single-source-of-truth. Schema versions + max_bytes come from
SPEC_titan_architecture_constants.toml via the generated
titan_plugin._phase_c_constants module — never hand-edited here.

Per Preamble G21 (one SHM slot, one writer): the producer writes; the
proxy + any other consumer reads. No second producer is allowed.

All 5 slots are variable-size msgpack payloads; readers use
``StateRegistryReader.read_variable() -> bytes`` and msgpack-decode.
Writers use ``StateRegistryWriter.write_variable(bytes)``.
"""
from __future__ import annotations

import numpy as np

from titan_plugin._phase_c_constants import (
    CONSCIOUSNESS_STATE_MAX_BYTES,
    CONSCIOUSNESS_STATE_SCHEMA_VERSION,
    HORMONE_FIRES_MAX_BYTES,
    HORMONE_FIRES_SCHEMA_VERSION,
    IMPULSE_ENGINE_STATE_MAX_BYTES,
    IMPULSE_ENGINE_STATE_SCHEMA_VERSION,
    RESONANCE_STATE_MAX_BYTES,
    RESONANCE_STATE_SCHEMA_VERSION,
    UNIFIED_SPIRIT_METADATA_MAX_BYTES,
    UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION,
)
from titan_plugin.core.state_registry import RegistrySpec


# Slot basenames (canonical per SPEC §7.1)

HORMONE_FIRES_SLOT = "hormone_fires"
IMPULSE_ENGINE_STATE_SLOT = "impulse_engine_state"
CONSCIOUSNESS_STATE_SLOT = "consciousness_state"
RESONANCE_STATE_SLOT = "resonance_state"
UNIFIED_SPIRIT_METADATA_SLOT = "unified_spirit_metadata"


# RegistrySpec instances (frozen at module load)
# All slots use uint8 dtype + variable_size=True per the standard
# msgpack-payload pattern (matches sensor_cache_outer_*.bin + cgn_live_weights).

HORMONE_FIRES_SPEC = RegistrySpec(
    name=HORMONE_FIRES_SLOT,
    dtype=np.dtype("uint8"),
    shape=(HORMONE_FIRES_MAX_BYTES,),
    schema_version=HORMONE_FIRES_SCHEMA_VERSION,
    variable_size=True,
)

IMPULSE_ENGINE_STATE_SPEC = RegistrySpec(
    name=IMPULSE_ENGINE_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(IMPULSE_ENGINE_STATE_MAX_BYTES,),
    schema_version=IMPULSE_ENGINE_STATE_SCHEMA_VERSION,
    variable_size=True,
)

CONSCIOUSNESS_STATE_SPEC = RegistrySpec(
    name=CONSCIOUSNESS_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(CONSCIOUSNESS_STATE_MAX_BYTES,),
    schema_version=CONSCIOUSNESS_STATE_SCHEMA_VERSION,
    variable_size=True,
)

RESONANCE_STATE_SPEC = RegistrySpec(
    name=RESONANCE_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(RESONANCE_STATE_MAX_BYTES,),
    schema_version=RESONANCE_STATE_SCHEMA_VERSION,
    variable_size=True,
)

UNIFIED_SPIRIT_METADATA_SPEC = RegistrySpec(
    name=UNIFIED_SPIRIT_METADATA_SLOT,
    dtype=np.dtype("uint8"),
    shape=(UNIFIED_SPIRIT_METADATA_MAX_BYTES,),
    schema_version=UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION,
    variable_size=True,
)


# Convenience aggregate (preserves insertion order for diagnostics)

ALL_SPIRIT_STATE_SPECS = (
    HORMONE_FIRES_SPEC,
    IMPULSE_ENGINE_STATE_SPEC,
    CONSCIOUSNESS_STATE_SPEC,
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SPEC,
)
