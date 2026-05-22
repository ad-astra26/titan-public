"""
soul_state_specs — shared RegistrySpec for soul_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: sovereignty_worker (G21 single-writer; chosen owner because
    Soul governance is sovereignty-domain per SPEC §1 glossary;
    sovereignty_worker already manages GREAT CYCLE convergence + ENFORCING
    ⇄ ADVISORY transition)
  - consumers:
      * api_subprocess StateAccessor.soul (replaces soul.state bus-cache)
      * api_subprocess StateAccessor.identity.maker_pubkey (fallback path)
      * dashboard /v4/soul-state + /v4/governance endpoints

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "maker_pubkey":          str,        # Ed25519 pubkey base58 (or "" if not set)
    "nft_address":           str,        # Soul NFT mint address (or "" pre-mint)
    "current_gen":           int,        # generation counter (resurrection increments)
    "active_directives":     list[dict], # [{id, text, signed_by, ts, ...}]
    "directives_count":      int,        # convenience cached len(active_directives)
    "last_directive_ts":     float,      # most recent directive issuance
    "soul_initialized":      bool,       # true after first Soul.initialize()
    "schema_version":        int,        # = SOUL_STATE_SCHEMA_VERSION
    "ts":                    float,      # publisher wall-time at write
  }

Closes the soul.state bus-cache state-lookup (~5 sites in api/state_accessor.py
SoulAccessor + IdentityAccessor) per Preamble G18.
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    SOUL_STATE_MAX_BYTES,
    SOUL_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


SOUL_STATE_SLOT = "soul_state"

SOUL_STATE_SPEC = RegistrySpec(
    name=SOUL_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(SOUL_STATE_MAX_BYTES,),
    schema_version=SOUL_STATE_SCHEMA_VERSION,
    variable_size=True,
)
