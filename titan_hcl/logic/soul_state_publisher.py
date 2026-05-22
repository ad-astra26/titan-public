"""
soul_state_publisher — SoulStatePublisher writes soul_state.bin SHM slot.

Producer for the soul_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0) /
rFP_phase_c_state_read_unification_l0_l1_canonical Phase A.4. G21 single-writer
contract: only sovereignty_worker publishes here.

Source: titan_hcl.core.soul.Soul instance — reads maker_pubkey,
nft_address, current_gen, active_directives via Soul's public surface.

Closes the soul.state bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.soul_state_specs import (
    SOUL_STATE_SLOT,
    SOUL_STATE_SPEC,
)
from titan_hcl._phase_c_constants import SOUL_STATE_SCHEMA_VERSION


class SoulStatePublisher(BaseStatePublisher):
    slot_name = SOUL_STATE_SLOT
    slot_spec = SOUL_STATE_SPEC

    def _compute_payload(self, soul: Any) -> dict[str, Any]:
        if soul is None:
            return {
                "maker_pubkey": "",
                "nft_address": "",
                "current_gen": 0,
                "active_directives": [],
                "directives_count": 0,
                "last_directive_ts": 0.0,
                "soul_initialized": False,
                "schema_version": SOUL_STATE_SCHEMA_VERSION,
                "ts": time.time(),
            }
        maker_pubkey = getattr(soul, "maker_pubkey", None) or ""
        nft_address = getattr(soul, "nft_address", None) or ""
        current_gen = int(getattr(soul, "current_gen", 0) or 0)
        active = list(getattr(soul, "active_directives", []) or [])
        last_directive_ts = 0.0
        if active:
            ts_candidates = [d.get("ts", 0.0) for d in active
                             if isinstance(d, dict)]
            if ts_candidates:
                last_directive_ts = float(max(ts_candidates))
        return {
            "maker_pubkey": str(maker_pubkey),
            "nft_address": str(nft_address),
            "current_gen": current_gen,
            "active_directives": active,
            "directives_count": len(active),
            "last_directive_ts": last_directive_ts,
            "soul_initialized": bool(maker_pubkey),
            "schema_version": SOUL_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
