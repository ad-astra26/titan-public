"""
cgn_engine_state_publisher — CGNEngineStatePublisher writes
cgn_engine_state.bin SHM slot.

Producer for the cgn_engine_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0).
G21 single-writer contract: only cgn_worker publishes here. Sibling to
existing cgn_live_weights.bin (tensor) + cgn_beta_state.bin (8-float
per-consumer reward EMA) — this slot carries the engine-level stats
previously surfaced via the cgn.stats bus-cache key.

Closes the cgn.stats bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.cgn_engine_state_specs import (
    CGN_ENGINE_STATE_SLOT,
    CGN_ENGINE_STATE_SPEC,
)
from titan_hcl._phase_c_constants import CGN_ENGINE_STATE_SCHEMA_VERSION


class CGNEngineStatePublisher(BaseStatePublisher):
    slot_name = CGN_ENGINE_STATE_SLOT
    slot_spec = CGN_ENGINE_STATE_SPEC

    def _compute_payload(self, cgn: Any) -> dict[str, Any]:
        if cgn is None:
            return self._stub()
        try:
            stats = cgn.get_stats() if hasattr(cgn, "get_stats") else {}
        except Exception:
            stats = {}
        # Lightweight slice — leave full weights/buffer out of SHM payload.
        # `consumers` shape varies by CGN engine version: dict[name→state],
        # list[dict], or list[str] (just names). The outer-trinity dims need ONLY
        # the top-level avg_reward/grounded_density/consolidations below, so this
        # slice is best-effort for the dashboard — wrapped to NEVER fail the
        # publish (live T2 2026-05-22 hit both `list.items()` and `str.get()`,
        # each leaving cgn_engine_state empty fleet-wide).
        _KEEP = ("transitions", "outcomes", "buffer_len", "registered",
                 "actions", "anchors", "soar_impasses", "haov_hypotheses")
        consumers: dict = {}
        try:
            _raw_consumers = stats.get("consumers") or {}
            if isinstance(_raw_consumers, dict):
                for name, cstate in _raw_consumers.items():
                    if isinstance(cstate, dict):
                        consumers[str(name)] = {
                            k: v for k, v in cstate.items() if k in _KEEP}
            elif isinstance(_raw_consumers, list):
                for i, c in enumerate(_raw_consumers):
                    if isinstance(c, dict):
                        consumers[str(c.get("name", i))] = {
                            k: v for k, v in c.items() if k in _KEEP}
                    elif isinstance(c, str):
                        consumers[c] = {}
        except Exception:
            consumers = {}
        # Phase C dissolve (2026-05-22): the outer_mind/outer_spirit 130D dim
        # formulas read cgn_stats {avg_reward, grounded_density, consolidations}
        # — previously via the CGN_STATS_UPDATED bus-cache (G18 violation). Mirror
        # the bus-path computation (cgn_worker.py:577-585) into this slot so the
        # outer-source sidecars read it SHM-direct. additive msgpack fields →
        # no schema bump (consumers use .get()).
        grounded_density = 0.0
        try:
            _vm = cgn.get_vm_snapshot() if hasattr(cgn, "get_vm_snapshot") else {}
            if isinstance(_vm, dict):
                grounded_density = float(_vm.get("grounded_density", 0.0) or 0.0)
        except Exception:
            pass
        return {
            "consumers": consumers,
            "total_transitions": int(stats.get("total_transitions", 0) or 0),
            "buffer_size": int(stats.get("buffer_size", 0) or 0),
            "consolidations": int(stats.get("consolidations", 0) or 0),
            "anchor_count": int(stats.get("anchor_count", 0) or 0),
            "sigma_updates": int(stats.get("sigma_updates", 0) or 0),
            "soar_impasses": int(stats.get("soar_impasses", 0) or 0),
            "haov_stats": stats.get("haov", {}) or {},
            # outer-trinity dim inputs (SHM-direct replacement for CGN_STATS_UPDATED)
            "avg_reward": float(stats.get("avg_reward", 0.0) or 0.0),
            "grounded_density": grounded_density,
            "schema_version": CGN_ENGINE_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "consumers": {},
            "total_transitions": 0,
            "buffer_size": 0,
            "consolidations": 0,
            "anchor_count": 0,
            "sigma_updates": 0,
            "soar_impasses": 0,
            "haov_stats": {},
            "avg_reward": 0.0,
            "grounded_density": 0.0,
            "schema_version": CGN_ENGINE_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
