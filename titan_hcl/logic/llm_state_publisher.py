"""
llm_state_publisher — LLMStatePublisher writes llm_state.bin SHM slot.

Producer for llm_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only llm_worker publishes here.

Closes the llm.stats bus-cache state-lookup per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.llm_state_specs import (
    LLM_STATE_SLOT,
    LLM_STATE_SPEC,
)
from titan_hcl._phase_c_constants import LLM_STATE_SCHEMA_VERSION


class LLMStatePublisher(BaseStatePublisher):
    slot_name = LLM_STATE_SLOT
    slot_spec = LLM_STATE_SPEC

    def _compute_payload(self, llm_state: Any) -> dict[str, Any]:
        if llm_state is None:
            return self._stub()
        # llm_state can be a dict (from llm_worker tally) or an object with
        # get_stats() — handle both.
        if hasattr(llm_state, "get_stats"):
            try:
                stats = llm_state.get_stats() or {}
            except Exception:
                stats = {}
        elif isinstance(llm_state, dict):
            stats = llm_state
        else:
            stats = {}
        return {
            "provider": str(stats.get("provider", "") or ""),
            "model": str(stats.get("model", "") or ""),
            "total_completions": int(stats.get("total_completions", 0) or 0),
            "completions_this_hour": int(
                stats.get("completions_this_hour", 0) or 0),
            "avg_latency_ms": float(stats.get("avg_latency_ms", 0.0) or 0.0),
            "p99_latency_ms": float(stats.get("p99_latency_ms", 0.0) or 0.0),
            "total_input_tokens": int(stats.get("total_input_tokens", 0) or 0),
            "total_output_tokens": int(
                stats.get("total_output_tokens", 0) or 0),
            "last_completion_ts": float(
                stats.get("last_completion_ts", 0.0) or 0.0),
            "last_error": str(stats.get("last_error", "") or ""),
            "error_rate": float(stats.get("error_rate", 0.0) or 0.0),
            "schema_version": LLM_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "provider": "",
            "model": "",
            "total_completions": 0,
            "completions_this_hour": 0,
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "last_completion_ts": 0.0,
            "last_error": "",
            "error_rate": 0.0,
            "schema_version": LLM_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
