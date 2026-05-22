"""
llm_state_specs — shared RegistrySpec for llm_state.bin SHM slot.

Phase C v1.10.0 SPEC bump (D-SPEC-71) per rFP_phase_c_state_read_unification
Phase A.4. Single source of truth shared by:

  - producer: llm_worker (G21 single-writer)
  - consumers:
      * api_subprocess StateAccessor.llm (replaces llm.stats bus-cache)
      * dashboard /v4/llm-status endpoints

Slot is variable-size msgpack per the established Phase C Python L2 pattern.

Payload schema (msgpack):
  {
    "provider":             str,        # "venice", "openrouter", "ollama_cloud", ...
    "model":                str,        # active model id
    "total_completions":    int,
    "completions_this_hour": int,
    "avg_latency_ms":       float,
    "p99_latency_ms":       float,
    "total_input_tokens":   int,
    "total_output_tokens":  int,
    "last_completion_ts":   float,
    "last_error":           str,
    "error_rate":           float,
    "schema_version":       int,
    "ts":                   float,
  }
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    LLM_STATE_MAX_BYTES,
    LLM_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


LLM_STATE_SLOT = "llm_state"

LLM_STATE_SPEC = RegistrySpec(
    name=LLM_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(LLM_STATE_MAX_BYTES,),
    schema_version=LLM_STATE_SCHEMA_VERSION,
    variable_size=True,
)
