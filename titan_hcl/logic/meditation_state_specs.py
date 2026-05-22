"""
meditation_state_specs — shared RegistrySpec for meditation_state.bin SHM slot.

Phase C v1.8.3 SPEC bump (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md
§4.D`. Single source of truth shared by:

  - producer: `titan_hcl.logic.meditation_state_publisher.MeditationStatePublisher`
    (invoked from meditation_worker on every KERNEL_EPOCH_TICK + on every
    transition — in_meditation flip, phase change, watchdog alert, completion —
    per Maker Q1/Q3 greenlight; G21 single-writer)
  - consumers:
      * `titan_hcl.logic.meditation_state_reader.MeditationStateReader`
        (sub-µs G18 SHM-direct reads for dashboard `/v4/meditation/health`,
         daily_nft trigger, soul-NFT mint cron)
      * `titan_hcl.proxies.meditation_proxy.MeditationProxy` (wraps the
        reader for in-proc Python callers)

Slot is variable-size msgpack per the established pattern (matches
dream_state.bin / metabolism_state.bin / social_graph_state.bin from
D-SPEC-50 + D-SPEC-51 + D-SPEC-56).

Payload schema (msgpack):
  {
    "tracker": {
      "last_epoch":       int,    # epoch_id of last completed meditation
      "count":            int,    # total meditations since persistent boot
      "count_since_nft":  int,    # meditations since last MyDay NFT
      "last_ts":          float,  # wall-clock of last completion (0 = none)
      "in_meditation":    bool,   # currently meditating
      "current_phase":    str,    # ∈ {"idle", "entering", "deep", "exiting"}
    },
    "watchdog": {
      "last_check_ts":              float,
      "gap_samples":                int,
      "expected_interval_hours":    float,
      "in_meditation_since_ts":     float,
      "consecutive_zero_promoted":  int,
      "selftest_done":              bool,
      "selftest_pass":              bool,
    },
    "last_alert":                   {severity, failure_mode, detail, ts} | null,
    "last_completion":              {epoch, promoted, pruned, trigger,
                                     success, ts} | null,
    "schema_version":               int,    # = MEDITATION_STATE_SCHEMA_VERSION
    "ts":                           float,  # publisher wall-time at write
  }

Closes the cross-process `state_refs["meditation_tracker"]` direct dict
reference + `spirit_supplemental_state.bin` `meditation_health` section
indirection (G21 violation — meditation_worker would otherwise become 2nd
writer to a spirit-owned slot).
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    MEDITATION_STATE_MAX_BYTES,
    MEDITATION_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


MEDITATION_STATE_SLOT = "meditation_state"

MEDITATION_STATE_SPEC = RegistrySpec(
    name=MEDITATION_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(MEDITATION_STATE_MAX_BYTES,),
    schema_version=MEDITATION_STATE_SCHEMA_VERSION,
    variable_size=True,
)
