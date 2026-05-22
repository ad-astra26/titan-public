"""
inner_perception_state_specs — shared RegistrySpec for inner_perception_state.bin.

Phase C dissolution (2026-05-22, RFP_phase_c_titan_hcl_cleanup §2 Phase C / C.7).
``InnerPerception`` is a parent-resident hardware ambient sampler (audio/visual/
ambient-change), so its stats can't be read from an existing producer SHM slot.
Before the dissolution it reached mind_worker via the OUTER_SOURCES_SNAPSHOT bus
broadcast (a Preamble G18 state-over-bus violation). This slot replaces that
delivery: the main plugin publishes (G21 single-writer), mind_worker reads
SHM-direct.

Payload schema (msgpack), == InnerPerception.get_stats():
  {
    "audio_state":    dict,   # AudioPerception.get_state()
    "visual_state":   dict,   # VisualPerception.get_state()
    "ambient_change": float,  # rolling-stddev (mind_tensor reads directly)
    "last_create_ts": float,  # outer_spirit ANANDA[41] creative_tension
    "ts":             float,
  }

Consumers: mind_worker (inner_mind feeling[5] inner_hearing / [7] inner_sight /
[9] inner_smell) + the outer-source assembler (outer_spirit ANANDA[41]).
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    INNER_PERCEPTION_STATE_MAX_BYTES,
    INNER_PERCEPTION_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


INNER_PERCEPTION_STATE_SLOT = "inner_perception_state"

INNER_PERCEPTION_STATE_SPEC = RegistrySpec(
    name=INNER_PERCEPTION_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(INNER_PERCEPTION_STATE_MAX_BYTES,),
    schema_version=INNER_PERCEPTION_STATE_SCHEMA_VERSION,
    variable_size=True,
)
