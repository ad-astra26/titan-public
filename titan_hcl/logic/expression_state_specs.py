"""
expression_state_specs — shared RegistrySpec for expression_state.bin SHM slot.

Sprint 7 §4.6 closure of rFP_phase_c_130d_rust_l1_port: ExpressionTranslator
+ ExpressionManager stats are published to SHM at 1 Hz so cross-process
consumers (spirit_worker subprocess in particular) can read sovereignty +
expression intensity without sync bus.request (G19).

Single source of truth shared by:
  - producer: titan_hcl.logic.expression_state_publisher.ExpressionStatePublisher
    (invoked from main plugin periodic loop @ 1 Hz)
  - consumer: titan_hcl.logic.inner_spirit_sidecar (started by cognitive_worker
    under l0_rust_enabled=true; D-SPEC-116 — was spirit_worker pre-retirement) —
    writes the inner_spirit sensor cache feeding Rust titan-inner-spirit-rs
    project_inner_spirit_45d for SAT[2] sovereignty + CHIT[28]
    causal_understanding + ANANDA[8] expression_quality.

Slot is variable-size msgpack per the established pattern (matches
memory_state / mind_state / etc. from earlier sessions).
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    EXPRESSION_STATE_MAX_BYTES,
    EXPRESSION_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


EXPRESSION_STATE_SLOT = "expression_state"

EXPRESSION_STATE_SPEC = RegistrySpec(
    name=EXPRESSION_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(EXPRESSION_STATE_MAX_BYTES,),
    schema_version=EXPRESSION_STATE_SCHEMA_VERSION,
    variable_size=True,
)
