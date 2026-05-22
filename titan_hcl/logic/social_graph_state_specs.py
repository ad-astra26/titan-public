"""
social_graph_state_specs — shared RegistrySpec for social_graph_state.bin SHM slot.

Phase C v1.7.1 SPEC bump (D-SPEC-50) per rFP_titan_hcl_l2_separation_strategy
§4.P. Single source of truth shared by:

  - producer: titan_hcl.logic.social_graph_state_publisher.SocialGraphStatePublisher
    (invoked from social_graph_worker @ 1 Hz; G21 single-writer)
  - consumers: titan_hcl.proxies.social_graph_proxy.SocialGraphProxy
    (SHM-direct reads for get_stats — the stats action that pre-v1.7.1
    was served via mind_worker `get_social_stats` orphan-handler — a
    documented G22 violation; now SHM per G18) + mind_worker._sense_taste
    (5DT mind-tensor taste sense)

Slot is variable-size msgpack per the established pattern (matches
memory_state.bin / mind_state.bin / spirit_supplemental_state.bin from
Sessions 1+4 of rFP_phase_c_async_shm_consumer_migration).

Payload schema (msgpack):
  {
    "users":                    int,    # SELECT COUNT(*) FROM user_profiles
    "edges":                    int,    # SELECT COUNT(*) FROM social_edges
    "donations":                int,    # SELECT COUNT(*) FROM donations
    "total_donated_sol":        float,  # SELECT COALESCE(SUM(amount_sol), 0)
    "inspirations":             int,    # SELECT COUNT(*) FROM inspirations
    "engagement_ledger_today":  int,    # COUNT since midnight UTC
    "schema_version":           int,    # = SOCIAL_GRAPH_STATE_SCHEMA_VERSION
    "ts":                       float,  # publisher wall-time at write
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    SOCIAL_GRAPH_STATE_MAX_BYTES,
    SOCIAL_GRAPH_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


SOCIAL_GRAPH_STATE_SLOT = "social_graph_state"

SOCIAL_GRAPH_STATE_SPEC = RegistrySpec(
    name=SOCIAL_GRAPH_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(SOCIAL_GRAPH_STATE_MAX_BYTES,),
    schema_version=SOCIAL_GRAPH_STATE_SCHEMA_VERSION,
    variable_size=True,
)
