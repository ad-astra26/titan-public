"""
spirit_state_specs — shared RegistrySpec definitions for the spirit SHM
slots per SPEC §7.1.

Originally home of the 5 Phase C Session 1 Python-wrapper slots owned by
the legacy spirit_worker / SpiritStatePublisher per
rFP_phase_c_async_shm_consumer_migration. Phase B of
rFP_phase_c_state_read_unification_l0_l1_canonical flips ownership of
the three trinity metadata slots (resonance / unified_spirit / filter_down)
to Rust (titan-unified-spirit-rs); Python becomes reader-only under
``microkernel.l0_rust_enabled=true`` per G21 single-writer.

Producer + consumer import these specs so the shape contract is
single-source-of-truth. Schema versions + max_bytes come from
SPEC_titan_architecture_constants.toml via the generated
titan_hcl._phase_c_constants module — never hand-edited here.

Per Preamble G21 (one SHM slot, one writer): the producer writes; the
proxy + any other consumer reads. No second producer is allowed.

All slots are variable-size msgpack payloads; readers use
``StateRegistryReader.read_variable() -> bytes`` and msgpack-decode.
Writers use ``StateRegistryWriter.write_variable(bytes)``.
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    CONSCIOUSNESS_STATE_MAX_BYTES,
    CONSCIOUSNESS_STATE_SCHEMA_VERSION,
    FILTER_DOWN_STATE_MAX_BYTES,
    FILTER_DOWN_STATE_SCHEMA_VERSION,
    HORMONE_FIRES_MAX_BYTES,
    HORMONE_FIRES_SCHEMA_VERSION,
    IMPULSE_ENGINE_STATE_MAX_BYTES,
    IMPULSE_ENGINE_STATE_SCHEMA_VERSION,
    RESONANCE_METADATA_MAX_BYTES,
    RESONANCE_METADATA_SCHEMA_VERSION,
    RESONANCE_STATE_MAX_BYTES,
    RESONANCE_STATE_SCHEMA_VERSION,
    UNIFIED_SPIRIT_METADATA_MAX_BYTES,
    UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


# Spirit-proxy HORMONE_NAMES preserved as-is for inner_spirit_sidecar
# back-compat (it was historically mislabeled — these are NS-program
# names, NOT canonical hormones; the canonical 11-hormone roster lives
# at shm_reader_bank.HORMONE_NAMES). The sidecar uses this to label
# rows of hormonal_state.bin reads at decode time; renaming would be a
# runtime semantics change unrelated to Phase B retirement scope and is
# tracked separately.
SPIRIT_PROXY_LEGACY_HORMONE_NAMES = (
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",
)


# Slot basenames (canonical per SPEC §7.1)

HORMONE_FIRES_SLOT = "hormone_fires"
IMPULSE_ENGINE_STATE_SLOT = "impulse_engine_state"
CONSCIOUSNESS_STATE_SLOT = "consciousness_state"
RESONANCE_STATE_SLOT = "resonance_state"
UNIFIED_SPIRIT_METADATA_SLOT = "unified_spirit_metadata"
# Phase B (rFP_phase_c_state_read_unification_l0_l1_canonical) — 2 new
# Rust-owned slots (B.0 SHIPPED 2026-05-17). resonance_metadata replaces
# the Python-wrapper resonance_state slot under l0_rust_enabled=true.
# filter_down_state is brand-new; FilterDownV5Engine had no SHM publish
# pre-B.0. UNIFIED_SPIRIT_METADATA_SLOT name preserved — ownership flips
# Python→Rust per B.0 G21 gate. All three become G18-canonical reads.
RESONANCE_METADATA_SLOT = "resonance_metadata"
FILTER_DOWN_STATE_SLOT = "filter_down_state"


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

# Phase B (rFP §B.0 SHIPPED 2026-05-17) — Rust-owned slots.
RESONANCE_METADATA_SPEC = RegistrySpec(
    name=RESONANCE_METADATA_SLOT,
    dtype=np.dtype("uint8"),
    shape=(RESONANCE_METADATA_MAX_BYTES,),
    schema_version=RESONANCE_METADATA_SCHEMA_VERSION,
    variable_size=True,
)

FILTER_DOWN_STATE_SPEC = RegistrySpec(
    name=FILTER_DOWN_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(FILTER_DOWN_STATE_MAX_BYTES,),
    schema_version=FILTER_DOWN_STATE_SCHEMA_VERSION,
    variable_size=True,
)


# Convenience aggregate (preserves insertion order for diagnostics)

ALL_SPIRIT_STATE_SPECS = (
    HORMONE_FIRES_SPEC,
    IMPULSE_ENGINE_STATE_SPEC,
    CONSCIOUSNESS_STATE_SPEC,
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SPEC,
    RESONANCE_METADATA_SPEC,
    FILTER_DOWN_STATE_SPEC,
)
