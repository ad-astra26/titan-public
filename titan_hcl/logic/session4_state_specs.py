"""
session4_state_specs — shared RegistrySpec for the 5 Session 4 SHM slots.

Phase C Session 4 of rFP_phase_c_async_shm_consumer_migration §4.B.6+B.7
+ §4.C.1 expansion. Slots:

  - mind_state.bin                — MoodEngine + reward telemetry
                                     (publisher: MindStatePublisher in mind_worker)
  - body_state.bin                — body_details metadata
                                     (publisher: BodyStatePublisher in body_worker)
  - language_state.bin            — LanguageTeacher.get_stats()
                                     (publisher: LanguageStatePublisher in language_worker)
  - events_teacher_state.bin      — EventsTeacher curated-signal telemetry
                                     (publisher: EventsTeacherStatePublisher
                                      in language_worker — co-located with
                                      EventsTeacher instance)
  - spirit_supplemental_state.bin — 4-section spirit state covering
                                     filter_down_status/meditation_health/
                                     coordinator/nervous_system
                                     (publisher: SpiritSupplementalStatePublisher
                                      in spirit_worker — same producer family
                                      as Session 1's hormone_fires/
                                      impulse_engine_state/consciousness_state/
                                      resonance_state/unified_spirit_metadata)

All 5 are variable-size msgpack per the established Session 1-3 pattern.

Single-source-of-truth for both producers and consumers (proxies). Per
G21 each slot has exactly one writer.
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    BODY_STATE_MAX_BYTES,
    BODY_STATE_SCHEMA_VERSION,
    EVENTS_TEACHER_STATE_MAX_BYTES,
    EVENTS_TEACHER_STATE_SCHEMA_VERSION,
    LANGUAGE_STATE_MAX_BYTES,
    LANGUAGE_STATE_SCHEMA_VERSION,
    MIND_STATE_MAX_BYTES,
    MIND_STATE_SCHEMA_VERSION,
    SPIRIT_SUPPLEMENTAL_STATE_MAX_BYTES,
    SPIRIT_SUPPLEMENTAL_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


# Slot basenames (canonical per SPEC §7.1)

MIND_STATE_SLOT = "mind_state"
BODY_STATE_SLOT = "body_state"
LANGUAGE_STATE_SLOT = "language_state"
EVENTS_TEACHER_STATE_SLOT = "events_teacher_state"
SPIRIT_SUPPLEMENTAL_STATE_SLOT = "spirit_supplemental_state"


def _spec(name: str, max_bytes: int, schema_version: int) -> RegistrySpec:
    return RegistrySpec(
        name=name,
        dtype=np.dtype("uint8"),
        shape=(max_bytes,),
        schema_version=schema_version,
        variable_size=True,
    )


MIND_STATE_SPEC = _spec(
    MIND_STATE_SLOT, MIND_STATE_MAX_BYTES, MIND_STATE_SCHEMA_VERSION)
BODY_STATE_SPEC = _spec(
    BODY_STATE_SLOT, BODY_STATE_MAX_BYTES, BODY_STATE_SCHEMA_VERSION)
LANGUAGE_STATE_SPEC = _spec(
    LANGUAGE_STATE_SLOT, LANGUAGE_STATE_MAX_BYTES,
    LANGUAGE_STATE_SCHEMA_VERSION)
EVENTS_TEACHER_STATE_SPEC = _spec(
    EVENTS_TEACHER_STATE_SLOT, EVENTS_TEACHER_STATE_MAX_BYTES,
    EVENTS_TEACHER_STATE_SCHEMA_VERSION)
SPIRIT_SUPPLEMENTAL_STATE_SPEC = _spec(
    SPIRIT_SUPPLEMENTAL_STATE_SLOT, SPIRIT_SUPPLEMENTAL_STATE_MAX_BYTES,
    SPIRIT_SUPPLEMENTAL_STATE_SCHEMA_VERSION)


ALL_SESSION4_SPECS = (
    MIND_STATE_SPEC,
    BODY_STATE_SPEC,
    LANGUAGE_STATE_SPEC,
    EVENTS_TEACHER_STATE_SPEC,
    SPIRIT_SUPPLEMENTAL_STATE_SPEC,
)
