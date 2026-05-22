"""
network_state_specs — shared RegistrySpec for network_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: titan_HCL parent (TitanKernel monitor_tick loop alongside
    SoulStatePublisher + GuardianStatePublisher; G21 single-writer)
  - consumers:
      * api_subprocess StateAccessor.network (replaces network.balance /
        network.info / network.account.* bus-cache lookups)
      * dashboard /v4/wallet / /v4/network endpoints

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "balance_sol":            float,        # current SOL balance
    "pubkey":                 str,          # Ed25519 pubkey base58 (or "")
    "premium_rpc":            str | None,   # premium RPC URL (or None)
    "rpc_urls":               list[str],    # fallback RPC URL set
    "rpc_endpoint":           str,          # active RPC endpoint
    "recent_account_data":    dict[str→dict], # PDA → cached account data
    "last_balance_update_ts": float,
    "last_info_update_ts":    float,
    "network_available":      bool,
    "schema_version":         int,
    "ts":                     float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    NETWORK_STATE_MAX_BYTES,
    NETWORK_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


NETWORK_STATE_SLOT = "network_state"

NETWORK_STATE_SPEC = RegistrySpec(
    name=NETWORK_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(NETWORK_STATE_MAX_BYTES,),
    schema_version=NETWORK_STATE_SCHEMA_VERSION,
    variable_size=True,
)
