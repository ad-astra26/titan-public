"""dim_firing_state_specs — Phase 2.5.A.2 SHM slot specs for the
producer-firing tracker.

Six per-block slots, one per tensor block. Single-writer per slot
per G21 — the worker process that owns each tensor function (e.g.
body_worker for inner_body) is the sole writer. Reader is the
api_subprocess (`/v4/debug/dim-sources` endpoint).

Filed 2026-05-08 by `rFP_trinity_130d_phase2_5_closure §2` after the
deploy revealed `api_process_separation_enabled=true` makes a
process-singleton in-memory tracker invisible to the API process.

Per slot payload shape (msgpack-encoded dict):

    {
      "block": "inner_body" | "inner_mind" | ...,
      "block_calls_total": int,
      "block_last_call_ts": float | None,
      "inputs_state": {input_name: "real" | "default" | "absent"},
      "dims": [
        {"v": float | None, "ts": float | None},
        ...  # length = block size (5 / 15 / 45)
      ],
      "ts": float,  # publish wall-clock
    }
"""
from __future__ import annotations

import numpy as np

from titan_plugin._phase_c_constants import (
    INNER_BODY_FIRING_MAX_BYTES,
    INNER_BODY_FIRING_SCHEMA_VERSION,
    INNER_MIND_FIRING_MAX_BYTES,
    INNER_MIND_FIRING_SCHEMA_VERSION,
    INNER_SPIRIT_FIRING_MAX_BYTES,
    INNER_SPIRIT_FIRING_SCHEMA_VERSION,
    OUTER_BODY_FIRING_MAX_BYTES,
    OUTER_BODY_FIRING_SCHEMA_VERSION,
    OUTER_MIND_FIRING_MAX_BYTES,
    OUTER_MIND_FIRING_SCHEMA_VERSION,
    OUTER_SPIRIT_FIRING_MAX_BYTES,
    OUTER_SPIRIT_FIRING_SCHEMA_VERSION,
)
from titan_plugin.core.state_registry import RegistrySpec


# Slot names — used for SHM file path and producer→slot audit.
INNER_BODY_FIRING_SLOT = "inner_body_firing"
INNER_MIND_FIRING_SLOT = "inner_mind_firing"
INNER_SPIRIT_FIRING_SLOT = "inner_spirit_firing"
OUTER_BODY_FIRING_SLOT = "outer_body_firing"
OUTER_MIND_FIRING_SLOT = "outer_mind_firing"
OUTER_SPIRIT_FIRING_SLOT = "outer_spirit_firing"


def _spec(name: str, max_bytes: int, schema_version: int) -> RegistrySpec:
    return RegistrySpec(
        name=name,
        dtype=np.dtype("uint8"),
        shape=(max_bytes,),
        schema_version=schema_version,
        variable_size=True,
    )


# Per-block specs — single source of truth for both writer (DimFiringTracker
# in each tensor producer's worker process) and reader (api_subprocess
# /v4/debug/dim-sources endpoint). G21 single-writer-per-slot is the contract.
INNER_BODY_FIRING_SPEC = _spec(
    INNER_BODY_FIRING_SLOT,
    INNER_BODY_FIRING_MAX_BYTES,
    INNER_BODY_FIRING_SCHEMA_VERSION,
)
INNER_MIND_FIRING_SPEC = _spec(
    INNER_MIND_FIRING_SLOT,
    INNER_MIND_FIRING_MAX_BYTES,
    INNER_MIND_FIRING_SCHEMA_VERSION,
)
INNER_SPIRIT_FIRING_SPEC = _spec(
    INNER_SPIRIT_FIRING_SLOT,
    INNER_SPIRIT_FIRING_MAX_BYTES,
    INNER_SPIRIT_FIRING_SCHEMA_VERSION,
)
OUTER_BODY_FIRING_SPEC = _spec(
    OUTER_BODY_FIRING_SLOT,
    OUTER_BODY_FIRING_MAX_BYTES,
    OUTER_BODY_FIRING_SCHEMA_VERSION,
)
OUTER_MIND_FIRING_SPEC = _spec(
    OUTER_MIND_FIRING_SLOT,
    OUTER_MIND_FIRING_MAX_BYTES,
    OUTER_MIND_FIRING_SCHEMA_VERSION,
)
OUTER_SPIRIT_FIRING_SPEC = _spec(
    OUTER_SPIRIT_FIRING_SLOT,
    OUTER_SPIRIT_FIRING_MAX_BYTES,
    OUTER_SPIRIT_FIRING_SCHEMA_VERSION,
)


# Block name → spec lookup, used by DimFiringTracker for lazy writer init.
DIM_FIRING_SPEC_BY_BLOCK: dict[str, RegistrySpec] = {
    "inner_body": INNER_BODY_FIRING_SPEC,
    "inner_mind": INNER_MIND_FIRING_SPEC,
    "inner_spirit": INNER_SPIRIT_FIRING_SPEC,
    "outer_body": OUTER_BODY_FIRING_SPEC,
    "outer_mind": OUTER_MIND_FIRING_SPEC,
    "outer_spirit": OUTER_SPIRIT_FIRING_SPEC,
}


__all__ = [
    "INNER_BODY_FIRING_SLOT", "INNER_BODY_FIRING_SPEC",
    "INNER_MIND_FIRING_SLOT", "INNER_MIND_FIRING_SPEC",
    "INNER_SPIRIT_FIRING_SLOT", "INNER_SPIRIT_FIRING_SPEC",
    "OUTER_BODY_FIRING_SLOT", "OUTER_BODY_FIRING_SPEC",
    "OUTER_MIND_FIRING_SLOT", "OUTER_MIND_FIRING_SPEC",
    "OUTER_SPIRIT_FIRING_SLOT", "OUTER_SPIRIT_FIRING_SPEC",
    "DIM_FIRING_SPEC_BY_BLOCK",
]
