"""
media_state_specs — shared RegistrySpec for media_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: studio_worker (G21 single-writer; sibling to existing
    studio_state.bin — this slot carries media-pipeline counters consumed
    by api/state_accessor.py MediaAccessor specifically, while
    studio_state.bin remains the broader render-lifecycle slot)
  - consumers:
      * api_subprocess StateAccessor.media (replaces media.stats bus-cache)

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "meditation_render_count":  int,
    "epoch_render_count":       int,
    "eureka_render_count":      int,
    "last_render_ts":           float,
    "last_render_type":         str,
    "total_disk_mb":            float,
    "nft_composite_count":      int,
    "schema_version":           int,
    "ts":                       float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    MEDIA_STATE_MAX_BYTES,
    MEDIA_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


MEDIA_STATE_SLOT = "media_state"

MEDIA_STATE_SPEC = RegistrySpec(
    name=MEDIA_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(MEDIA_STATE_MAX_BYTES,),
    schema_version=MEDIA_STATE_SCHEMA_VERSION,
    variable_size=True,
)
