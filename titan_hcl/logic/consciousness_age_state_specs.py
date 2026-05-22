"""
consciousness_age_state_specs — shared RegistrySpec for
consciousness_age.bin SHM slot.

Producer: cognitive_worker (Consciousness object lives in spirit_loop
under cognitive_worker per SPEC §1 glossary). G21 single-writer.

Carries the lifetime consciousness epoch count (``consciousness.get_epoch_count()``,
backed by ``consciousness.db`` row count). This is Titan's "main age" — the
fast cognitive self-observation tick counter, accumulating ~10s per epoch.
Distinct from ``unified_spirit_metadata.epoch_count`` (the slower GreatEpoch
counter, ~5min cycles).

Closes the post_dispatch gap: social_worker subprocess cannot reach
consciousness.db per Preamble G18 (state transport is SHM, never DB).

Per SPEC §7.1 row (D-SPEC-85 v1.25.0).

Payload schema (msgpack):
  {
    "age_epochs":      int,    # lifetime consciousness tick count
    "schema_version":  int,
    "ts":              float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    CONSCIOUSNESS_AGE_MAX_BYTES,
    CONSCIOUSNESS_AGE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


CONSCIOUSNESS_AGE_SLOT = "consciousness_age"

CONSCIOUSNESS_AGE_SPEC = RegistrySpec(
    name=CONSCIOUSNESS_AGE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(CONSCIOUSNESS_AGE_MAX_BYTES,),
    schema_version=CONSCIOUSNESS_AGE_SCHEMA_VERSION,
    variable_size=True,
)
