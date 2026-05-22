"""
reasoning_state_publisher — ReasoningStatePublisher writes
reasoning_state.bin SHM slot.

Producer for reasoning_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only cognitive_worker publishes here (ReasoningEngine
lives in cognitive_worker per SPEC §1 glossary line 321).

Closes the reasoning.stats bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.reasoning_state_specs import (
    REASONING_STATE_SLOT,
    REASONING_STATE_SPEC,
)
from titan_hcl._phase_c_constants import REASONING_STATE_SCHEMA_VERSION


class ReasoningStatePublisher(BaseStatePublisher):
    slot_name = REASONING_STATE_SLOT
    slot_spec = REASONING_STATE_SPEC

    def _compute_payload(self, reasoning_engine: Any) -> dict[str, Any]:
        if reasoning_engine is None:
            return self._stub()
        try:
            stats = reasoning_engine.get_stats() if hasattr(
                reasoning_engine, "get_stats") else {}
        except Exception:
            stats = {}
        action_dist = stats.get("action_distribution", {}) or {}
        if not isinstance(action_dist, dict):
            action_dist = {}
        # mind_neuromods — small dict[str→float] for the ReasoningTab
        # NeuromodPanel; bound to scalar floats.
        _mn = stats.get("mind_neuromods", {}) or {}
        mind_neuromods = ({str(k): float(v) for k, v in _mn.items()
                           if isinstance(v, (int, float))}
                          if isinstance(_mn, dict) else {})
        _so = stats.get("spirit_observer", {}) or {}
        spirit_observer = ({str(k): (float(v) if isinstance(v, float) else int(v))
                            for k, v in _so.items()
                            if isinstance(v, (int, float))}
                           if isinstance(_so, dict) else {})
        return {
            "total_chains": int(stats.get("total_chains", 0) or 0),
            "total_commits": int(stats.get("total_commits", 0) or 0),
            "commit_rate": float(stats.get("commit_rate", 0.0) or 0.0),
            "avg_chain_length": float(stats.get("avg_chain_length", 0.0) or 0.0),
            "buffer_size": int(stats.get("buffer_size", 0) or 0),
            "current_active": bool(stats.get("current_active", False)),
            "last_action": str(stats.get("last_action", "") or ""),
            "last_outcome": str(stats.get("last_outcome", "") or ""),
            "action_distribution": {str(k): int(v) for k, v in action_dist.items()
                                    if isinstance(v, (int, float))},
            # ── Additive: live reasoning telemetry the Observatory
            # ReasoningTab consumes (ConfidenceGauge + NeuromodPanel +
            # Technical panel). ReasoningEngine.get_stats() produces all of
            # these; the publisher previously dropped them. Additive msgpack
            # extension — v1 readers tolerate missing keys.
            "total_conclusions": int(stats.get("total_conclusions", 0) or 0),
            "total_reasoning_steps": int(stats.get("total_reasoning_steps", 0) or 0),
            "is_active": bool(stats.get("is_active", False)),
            "chain_length": int(stats.get("chain_length", 0) or 0),
            "confidence": float(stats.get("confidence", 0.0) or 0.0),
            "gut_agreement": float(stats.get("gut_agreement", 0.0) or 0.0),
            "spirit_nudge": float(stats.get("spirit_nudge", 0.0) or 0.0),
            "persistence": float(stats.get("persistence", 0.0) or 0.0),
            "policy_updates": int(stats.get("policy_updates", 0) or 0),
            "policy_loss": float(stats.get("policy_loss", 0.0) or 0.0),
            "spirit_observer": spirit_observer,
            "mind_neuromods": mind_neuromods,
            "schema_version": REASONING_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "total_chains": 0,
            "total_commits": 0,
            "commit_rate": 0.0,
            "avg_chain_length": 0.0,
            "buffer_size": 0,
            "current_active": False,
            "last_action": "",
            "last_outcome": "",
            "action_distribution": {},
            "schema_version": REASONING_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
