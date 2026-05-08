"""
session3_state_specs — shared RegistrySpec for the 7 Session 3 SHM slots.

Phase C Session 3 of rFP_phase_c_async_shm_consumer_migration §4.B.2-B.5
+ §4.B.9-B.11. Slots:

  - assessment_state.bin       (publisher: agency_state_publisher in
                                 agency_worker — co-located with the
                                 SelfAssessment instance owned by
                                 AgencyModule)
  - agency_state.bin           (publisher: agency_state_publisher in
                                 agency_worker)
  - social_perception_state.bin (publisher: SpiritStatePublisher
                                 extension in spirit_worker, since
                                 social_perception state lives there)
  - rl_state.bin                (publisher: rl_state_publisher in rl_worker)
  - timechain_state.bin         (publisher: timechain_state_publisher
                                 in timechain_worker)
  - reflex_state.bin            (publisher: reflex_state_publisher
                                 in reflex_worker)
  - output_verifier_state.bin   (publisher: output_verifier_state_publisher
                                 in output_verifier_worker)

All 7 are variable-size msgpack per the established pattern (matches
hormone_fires/impulse_engine_state/memory_state/etc. from Sessions 1+2).

Single-source-of-truth for both producers and consumers (proxies). Per
G21 each slot has exactly one writer.
"""
from __future__ import annotations

import numpy as np

from titan_plugin._phase_c_constants import (
    AGENCY_STATE_MAX_BYTES,
    AGENCY_STATE_SCHEMA_VERSION,
    ASSESSMENT_STATE_MAX_BYTES,
    ASSESSMENT_STATE_SCHEMA_VERSION,
    OUTPUT_VERIFIER_STATE_MAX_BYTES,
    OUTPUT_VERIFIER_STATE_SCHEMA_VERSION,
    REFLEX_STATE_MAX_BYTES,
    REFLEX_STATE_SCHEMA_VERSION,
    RL_STATE_MAX_BYTES,
    RL_STATE_SCHEMA_VERSION,
    SOCIAL_PERCEPTION_STATE_MAX_BYTES,
    SOCIAL_PERCEPTION_STATE_SCHEMA_VERSION,
    TIMECHAIN_STATE_MAX_BYTES,
    TIMECHAIN_STATE_SCHEMA_VERSION,
)
from titan_plugin.core.state_registry import RegistrySpec


# Slot basenames (canonical per SPEC §7.1)

ASSESSMENT_STATE_SLOT = "assessment_state"
AGENCY_STATE_SLOT = "agency_state"
SOCIAL_PERCEPTION_STATE_SLOT = "social_perception_state"
RL_STATE_SLOT = "rl_state"
TIMECHAIN_STATE_SLOT = "timechain_state"
REFLEX_STATE_SLOT = "reflex_state"
OUTPUT_VERIFIER_STATE_SLOT = "output_verifier_state"


def _spec(name: str, max_bytes: int, schema_version: int) -> RegistrySpec:
    return RegistrySpec(
        name=name,
        dtype=np.dtype("uint8"),
        shape=(max_bytes,),
        schema_version=schema_version,
        variable_size=True,
    )


ASSESSMENT_STATE_SPEC = _spec(
    ASSESSMENT_STATE_SLOT, ASSESSMENT_STATE_MAX_BYTES,
    ASSESSMENT_STATE_SCHEMA_VERSION)
AGENCY_STATE_SPEC = _spec(
    AGENCY_STATE_SLOT, AGENCY_STATE_MAX_BYTES,
    AGENCY_STATE_SCHEMA_VERSION)
SOCIAL_PERCEPTION_STATE_SPEC = _spec(
    SOCIAL_PERCEPTION_STATE_SLOT, SOCIAL_PERCEPTION_STATE_MAX_BYTES,
    SOCIAL_PERCEPTION_STATE_SCHEMA_VERSION)
RL_STATE_SPEC = _spec(
    RL_STATE_SLOT, RL_STATE_MAX_BYTES, RL_STATE_SCHEMA_VERSION)
TIMECHAIN_STATE_SPEC = _spec(
    TIMECHAIN_STATE_SLOT, TIMECHAIN_STATE_MAX_BYTES,
    TIMECHAIN_STATE_SCHEMA_VERSION)
REFLEX_STATE_SPEC = _spec(
    REFLEX_STATE_SLOT, REFLEX_STATE_MAX_BYTES,
    REFLEX_STATE_SCHEMA_VERSION)
OUTPUT_VERIFIER_STATE_SPEC = _spec(
    OUTPUT_VERIFIER_STATE_SLOT, OUTPUT_VERIFIER_STATE_MAX_BYTES,
    OUTPUT_VERIFIER_STATE_SCHEMA_VERSION)


ALL_SESSION3_SPECS = (
    ASSESSMENT_STATE_SPEC,
    AGENCY_STATE_SPEC,
    SOCIAL_PERCEPTION_STATE_SPEC,
    RL_STATE_SPEC,
    TIMECHAIN_STATE_SPEC,
    REFLEX_STATE_SPEC,
    OUTPUT_VERIFIER_STATE_SPEC,
)
