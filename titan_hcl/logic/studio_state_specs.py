"""
studio_state_specs — shared RegistrySpec for studio_state.bin SHM slot.

Phase C v1.8.3 SPEC bump (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md
§4.K`. Single source of truth shared by:

  - producer: `titan_hcl.logic.studio_state_publisher.StudioStatePublisher`
    (invoked from studio_worker on every KERNEL_EPOCH_TICK + immediately after
    every successful render per Maker Q3 greenlight — dual-trigger republish;
    G21 single-writer)
  - consumers:
      * `titan_hcl.proxies.studio_proxy.StudioStateShmReader`
        (sub-µs G18 SHM-direct reads for `/v4/studio/stats` Observatory route,
         future dashboards)

Slot is variable-size msgpack per the established Python L2 slot family pattern
(matches metabolism_state.bin / social_graph_state.bin / dream_state.bin /
memory_state.bin / mind_state.bin / body_state.bin from D-SPEC-50 + D-SPEC-51 +
D-SPEC-56 + Sessions 1-5 of rFP_phase_c_async_shm_consumer_migration).

Payload schema (msgpack):
  {
    "schema_version":           int,    # = STUDIO_STATE_SCHEMA_VERSION
    "meditation_count":         int,    # files in data/studio_exports/meditation/ (non-sidecar)
    "epoch_count":              int,    # files in data/studio_exports/epoch/ (non-sidecar)
    "eureka_count":             int,    # files in data/studio_exports/eureka/ (non-sidecar)
    "last_render_ts":           float,  # unix epoch of last successful render (0 if none)
    "last_render_type":         str,    # ∈ {"none", "meditation", "epoch", "eureka"}
    "output_root":              str,    # output_path config (readers can hash for config-drift)
    "default_resolution":       int,    # echoes config for Observatory display
    "highres_resolution":       int,    # echoes config for Observatory display
    "nft_composite_enabled":    bool,   # echoes config
    "ts":                       float,  # publisher wall-time at write (SeqLock write ts)
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    STUDIO_STATE_MAX_BYTES,
    STUDIO_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


STUDIO_STATE_SLOT = "studio_state"

STUDIO_STATE_SPEC = RegistrySpec(
    name=STUDIO_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(STUDIO_STATE_MAX_BYTES,),
    schema_version=STUDIO_STATE_SCHEMA_VERSION,
    variable_size=True,
)
