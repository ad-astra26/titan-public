"""
dream_state_specs — shared RegistrySpec for dream_state.bin SHM slot.

Phase C v1.8.2 SPEC bump (D-SPEC-56) per `rFP_titan_hcl_l2_separation_strategy.md
§4.I`. Single source of truth shared by:

  - producer: `titan_hcl.logic.dream_state_publisher.DreamStatePublisher`
    (invoked from dream_state_worker on every KERNEL_EPOCH_TICK + on every
    DREAMING_STATE_UPDATED arrival per Maker Q6 greenlight — dual-trigger
    republish so readers detect staleness via `last_transition_ts` freshness
    probe; G21 single-writer)
  - consumers:
      * `titan_hcl.logic.dream_state_reader.DreamStateReader`
        (sub-µs G18 SHM-direct reads for plugin chat-during-dream buffer
         decision, api_subprocess chat-bridge buffer decision, spirit_worker
         `_read_is_dreaming_from_shm()` helper replacing the deleted
         `_shared_is_dreaming` module-level flag + 20+ readers,
         expression_worker tick-gate cache, timechain_worker dream-hook)
      * dashboard `/v4/dreaming` legacy compat (reads via CachedState which
        is filled by api_subprocess's BusSubscriber on DREAM_STATE_CHANGED)

Slot is variable-size msgpack per the established pattern (matches
metabolism_state.bin / social_graph_state.bin / memory_state.bin / mind_state.bin
/ body_state.bin from D-SPEC-50 + D-SPEC-51 + Sessions 1-5 of
rFP_phase_c_async_shm_consumer_migration).

Payload schema (msgpack):
  {
    "is_dreaming":          bool,    # current dream state
    "state":                str,     # ∈ {"awake", "dreaming", "dream_start", "dream_end"}
    "recovery_pct":         float,   # 0.0–100.0 (100.0 immediately after wake)
    "remaining_epochs":     int,     # countdown of expected dream epochs (0 when awake)
    "wake_transition":      bool,    # true during brief post-wake transition window
    "just_woke":            bool,    # true on the tick immediately after dream end
    "wake_ts":              float,   # unix ts of last wake (0 if N/A)
    "dream_started_ts":     float,   # unix ts of current/last dream start (0 if N/A)
    "last_transition_ts":   float,   # unix ts of last state change — FRESHNESS PROBE:
                                     # readers detect staleness if
                                     # (time.time() - last_transition_ts)
                                     # > DREAM_STATE_REPUBLISH_CADENCE_S × 5
    # Additive circadian telemetry (no schema bump — additive msgpack fields per
    # this slot's own precedent; payload stays < DREAM_STATE_MAX_BYTES=512):
    "cycle_count":          int,     # Dreaming.get_stats cycle_count
    "fatigue":              float,   # last_fatigue
    "developmental_age":    int,
    "epochs_since_dream":   int,
    # Additive distillation telemetry (rFP_experience_distillation_phase_c Bug B,
    # 2026-05-21) — surfaces real distill state to the health check via
    # /v4/inner-trinity so it no longer false-positives "DISTILLATION
    # DISCONNECTED". Sourced from the DREAMING_STATE_UPDATED payload (full
    # Dreaming.get_stats() output). Additive — no schema bump:
    "distill_attempts":        int,
    "distill_passed":          int,
    "distilled_count":         int,
    "distill_threshold":       float,
    "experience_buffer_size":  int,
    "schema_version":       int,     # = DREAM_STATE_SCHEMA_VERSION
    "ts":                   float,   # publisher wall-time at write
  }

Closes the latent fleet-wide Phase C `DREAM_STATE_CHANGED` silent-emit bug
(sole emitter was dead `spirit_worker.py:3006/3007/3143/3144` under
`l0_rust_enabled=true` since cognitive_worker drives the actual dream
lifecycle via DreamingEngine but never emitted DREAM_STATE_CHANGED;
3 downstream subscribers — plugin chat-buffer, expression_worker tick-gate,
timechain_worker dream-hook — received nothing for the entire Phase C era).
"""
from __future__ import annotations

import numpy as np

from titan_hcl._phase_c_constants import (
    DREAM_STATE_MAX_BYTES,
    DREAM_STATE_SCHEMA_VERSION,
)
from titan_hcl.core.state_registry import RegistrySpec


DREAM_STATE_SLOT = "dream_state"

DREAM_STATE_SPEC = RegistrySpec(
    name=DREAM_STATE_SLOT,
    dtype=np.dtype("uint8"),
    shape=(DREAM_STATE_MAX_BYTES,),
    schema_version=DREAM_STATE_SCHEMA_VERSION,
    variable_size=True,
)
