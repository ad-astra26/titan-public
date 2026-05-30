"""
titan_hcl/api/shm_reader_bank.py — typed shm-registry readers for the
api_subprocess.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

Wraps `titan_hcl.core.state_registry.StateRegistryReader` with per-
registry typed accessors. Each method returns a structured dict (or None
if the registry is unavailable/disabled), including `age_seconds` for
freshness checks.

The api_subprocess uses these readers to serve endpoint state without
issuing any RPC to the kernel. The mmap handles are owned per-reader
and never closed; reads are zero-copy + defensive-copy + SeqLock-
validated.

Design notes:
- All methods are sync (no async). Reads are sub-microsecond.
- Missing/disabled registries return None — endpoint code should fall
  back to bus-cached values (set by BusSubscriber).
- The bank is constructed once at api_subprocess boot; readers attach
  lazily on first read.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from titan_hcl.core.state_registry import (
    CHI_STATE,
    EPOCH_COUNTER,
    HORMONAL_STATE,
    IDENTITY,
    INNER_BODY_5D,
    INNER_MIND_15D,
    INNER_SELF_INSIGHT,
    INNER_SPIRIT_45D,
    NEUROMOD_STATE,
    OUTER_BODY_5D,
    OUTER_MIND_15D,
    OUTER_SPIRIT_45D,
    PI_HEARTBEAT_STATE,
    SPHERE_CLOCKS_STATE,
    CGN_BETA_STATE,
    NS_PROGRAM_URGENCIES_INPUT,
    StateRegistryReader,
    TITANVM_REGISTERS,
    TOPOLOGY_30D,
    TRAJECTORY_STATE,
    TRINITY_STATE,
    resolve_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic.session3_state_specs import (
    AGENCY_STATE_SPEC,
    ASSESSMENT_STATE_SPEC,
    OUTPUT_VERIFIER_STATE_SPEC,
    REFLEX_STATE_SPEC,
    RL_STATE_SPEC,
    SOCIAL_PERCEPTION_STATE_SPEC,
    TIMECHAIN_STATE_SPEC,
)
from titan_hcl.logic.session4_state_specs import (
    BODY_STATE_SPEC,
    EVENTS_TEACHER_STATE_SPEC,
    LANGUAGE_STATE_SPEC,
    MIND_STATE_SPEC,
    SPIRIT_SUPPLEMENTAL_STATE_SPEC,
)
from titan_hcl.logic.spirit_state_specs import (
    FILTER_DOWN_STATE_SPEC,
    RESONANCE_METADATA_SPEC,
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SPEC,
)
from titan_hcl.logic.memory_state_specs import MEMORY_STATE_SPEC
from titan_hcl.logic.social_graph_state_specs import SOCIAL_GRAPH_STATE_SPEC
from titan_hcl.logic.dream_state_specs import DREAM_STATE_SPEC
from titan_hcl.logic.metabolism_state_specs import METABOLISM_STATE_SPEC
from titan_hcl.logic.life_force_state_specs import LIFE_FORCE_STATE_SPEC
from titan_hcl.logic.meditation_state_specs import MEDITATION_STATE_SPEC
from titan_hcl.logic.studio_state_specs import STUDIO_STATE_SPEC
from titan_hcl.logic.expression_state_specs import EXPRESSION_STATE_SPEC
from titan_hcl.logic.inner_perception_state_specs import (
    INNER_PERCEPTION_STATE_SPEC)
from titan_hcl.logic.interface_advisor_specs import (
    INTERFACE_ADVISOR_STATE_SPEC,
)
# Phase A.4 (rFP_phase_c_state_read_unification_l0_l1_canonical v1.10.0 / D-SPEC-70):
# 9 new Python L2 SHM slots — full SHM-direct migration of api_subprocess
# state-lookups per Preamble G18.
from titan_hcl.logic.soul_state_specs import SOUL_STATE_SPEC
from titan_hcl.logic.cgn_engine_state_specs import CGN_ENGINE_STATE_SPEC
from titan_hcl.logic.consciousness_age_state_specs import (
    CONSCIOUSNESS_AGE_SPEC,
)
# ARCH-MAP-HEALTH-OBSERVABILITY Class B (2026-05-26): consciousness_state.bin
# slot reader. Writer: cognitive_worker (G21 single-writer, see
# cognitive_worker.py:3002 — Phase 3.A D-SPEC-86 v1.26.0). Surfaces
# {epoch_count, epoch_id, density, curvature, dream_quality, fatigue,
# trajectory_magnitude, ts} so /v4/inner-trinity can overlay the live epoch
# counter onto the stale `unified_spirit.epoch_count` field that
# `_get_cached_coordinator_async` returns. Per SPEC §10.E telemetry
# write-then-publish (SHM-canonical, LOCKED 2026-05-07 per Preamble G18).
from titan_hcl.logic.spirit_state_specs import (
    CONSCIOUSNESS_STATE_SPEC,
)
from titan_hcl.logic.reasoning_state_specs import REASONING_STATE_SPEC
from titan_hcl.logic.meta_reasoning_state_specs import META_REASONING_STATE_SPEC
from titan_hcl.logic.meta_teacher_state_specs import META_TEACHER_STATE_SPEC
from titan_hcl.logic.experience_stats_specs import EXPERIENCE_STATS_SPEC
from titan_hcl.logic.guardian_state_specs import GUARDIAN_STATE_SPEC
from titan_hcl.logic.llm_state_specs import LLM_STATE_SPEC
from titan_hcl.logic.media_state_specs import MEDIA_STATE_SPEC
from titan_hcl.logic.msl_state_specs import MSL_STATE_SPEC
from titan_hcl.logic.network_state_specs import NETWORK_STATE_SPEC

logger = logging.getLogger(__name__)


# ── Schema labels ─────────────────────────────────────────────────────
# Index → name lookup so Python-side reads return labelled dicts instead
# of raw arrays. Endpoint code stays readable (no magic offsets).

NEUROMOD_NAMES = ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]

CHI_FIELD_NAMES = ["total", "spirit", "mind", "body", "coherence", "urgency"]

SPHERE_CLOCK_NAMES = [
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
]
SPHERE_CLOCK_FIELDS = [
    "radius", "scalar_position", "phase", "contraction_velocity",
    "pulse_count", "consecutive_balanced", "last_pulse_age_s",
]

NS_PROGRAM_NAMES = [
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",
]
NS_PROGRAM_FIELDS = ["urgency", "fire_count", "total_updates", "last_loss"]

# 11-hormone schema (HormonalSystem per C-S5). Each row = (level, target,
# acceleration, decay) per hormone. Names mirror HormonalSystem.hormone_names.
HORMONE_NAMES = [
    "cortisol", "adrenaline", "oxytocin", "dopamine", "serotonin",
    "endorphin", "melatonin", "growth_hormone", "insulin",
    "thyroid", "estrogen",
]
HORMONE_FIELDS = ["level", "target", "acceleration", "decay"]

# 30D space-topology layout per SPEC §G4: [0:10] outer_lower + [10:20]
# inner_lower + [20:30] whole. The `parts` view (see read_topology_30d)
# derives 6 per-LAYER LayerObservables from the 6 daemon-tensor slots
# (inner_body_5d, inner_mind_15d, inner_spirit_45d, outer_body_5d,
# outer_mind_15d, outer_spirit_45d) mirroring titan-trinity-rs::topology::
# TopologyEngine::derive_layer_observables (rFP_phase_c_substrate_observable_
# closure §2.1 / D-SPEC-80). Each layer produces a 5-tuple {coherence,
# magnitude, velocity, direction, polarity}. Pre-rFP this slot misused
# body-part labels (head/torso/limbs) to slice the 30D vector as 6×5 — that
# was always architecturally wrong (real layout is 3×10) but invisible while
# WHOLE was hardcoded zero; the body-part labels are now retired.
TOPOLOGY_PART_NAMES = [
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
]
TOPOLOGY_FIELDS = ["coherence", "magnitude", "velocity", "direction", "polarity"]

# Numerical-zero threshold matching titan-rust/.../topology.rs:MIN_MAGNITUDE
# (1e-10) — guards cosine_sim and polarity from zero-magnitude divergence.
_LAYER_OBS_MIN_MAGNITUDE = 1e-10


def _derive_layer_observable(tensor: list[float]) -> dict[str, float]:
    """Python mirror of Rust `topology::TopologyEngine::derive_layer_observables`
    per-layer derivation. Stateless (no prev-tick history), so velocity +
    direction degrade to 0.0. The 3 stateless fields (coherence, magnitude,
    polarity) are exact 1:1 with the Rust impl.

    Coherence uses `middle_path.layer_coherence` (1 - variance/0.25) per SPEC
    §G4 + §G11 + D-SPEC-84 — single source of truth shared with sphere_clock
    coherence input + observables.BodyPartObserver. Previous cosine-vs-uniform
    formula was a port-time drift (D-SPEC-80 v1.21.0) — restored 2026-05-18.
    """
    from titan_hcl.logic.middle_path import layer_coherence
    n = len(tensor)
    if n == 0:
        return {f: 0.0 for f in TOPOLOGY_FIELDS}
    # Magnitude = l2_norm / sqrt(N) clamped [0, 1] (normalized).
    mag_sq = sum(v * v for v in tensor)
    mag_norm = mag_sq ** 0.5
    magnitude = max(0.0, min(1.0, mag_norm / (n ** 0.5))) if n > 0 else 0.0
    # Coherence = `1 - variance/0.25` (canonical Python middle_path formula).
    coherence = layer_coherence(tensor)
    # Polarity = (mean - 0.5) * 2 clamped [-1, 1]; 0 for zero-magnitude
    # (parity with Rust topology.rs:tensor_polarity).
    if mag_sq < _LAYER_OBS_MIN_MAGNITUDE:
        polarity = 0.0
    else:
        mean = sum(tensor) / n
        polarity = max(-1.0, min(1.0, (mean - 0.5) * 2.0))
    return {
        "coherence": round(float(coherence), 6),
        "magnitude": round(float(magnitude), 6),
        "velocity": 0.0,    # stateless reader; Rust tracks prev_magnitudes
        "direction": 0.0,   # same — sign(Δmag) requires prev state
        "polarity": round(float(polarity), 6),
    }

# 45D inner spirit slot labels (per spirit_tensor.collect_spirit_45d).
INNER_SPIRIT_GROUPS = {
    "SAT":    slice(0, 15),
    "CHIT":   slice(15, 30),
    "ANANDA": slice(30, 45),
}

# 162D Trinity layout (per state_registry.py:606-609).
TRINITY_GROUPS = {
    "full_130dt":         slice(0, 130),
    "full_30d_topology":  slice(130, 160),
    "journey":            slice(160, 162),  # (curvature, density)
}


# ── Helper: read metadata + payload, return structured dict ───────────

def _payload_with_meta(
    reader: StateRegistryReader,
    name: str,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """Returns (payload, age_seconds, seq) — None tuple on read failure."""
    payload = reader.read()
    if payload is None:
        return (None, None, None)
    meta = reader.read_meta()
    if meta is None:
        # Should not happen if read() succeeded, but defensive.
        return (payload, None, None)
    return (payload, meta.get("age_seconds"), meta.get("seq"))


# ── Bank ──────────────────────────────────────────────────────────────

class ShmReaderBank:
    """Aggregate owner of per-registry readers.

    Constructed once at api_subprocess boot. Each registry has its own
    reader; readers attach lazily on first read.

    All methods return a structured dict with `age_seconds` and `seq`,
    or None if the registry is missing/disabled/torn.
    """

    __slots__ = (
        "titan_id", "shm_root",
        "_trinity", "_neuromod", "_epoch", "_inner_spirit", "_pi_heartbeat",
        "_sphere_clocks", "_chi", "_titanvm", "_identity",
        # chunk 8M.4 (2026-05-05): cognitive_worker shm-direct-read bank
        # extended to all SPEC §1096 slots so the cognitive epoch driver
        # can populate /v4/inner-trinity / /v4/sphere-clocks / etc.
        "_topology_30d", "_hormonal",
        "_inner_body", "_inner_mind",
        "_outer_body", "_outer_mind", "_outer_spirit",
        # rFP_worker_broadcast_topics_completion §4.D abstraction-completion
        # (2026-05-10): variable-size msgpack-encoded composite slots that
        # SpiritAccessor's get_nervous_system / get_resonance /
        # get_unified_spirit need for SHM-first migration (matching the
        # get_sphere_clocks pattern at state_accessor.py:242-247).
        "_spirit_supplemental", "_resonance_state", "_unified_spirit_metadata",
        # rFP_phase_c_state_read_unification_l0_l1_canonical Phase B.0
        # SHIPPED 2026-05-17: 2 Rust-owned trinity metadata slots
        # (titan-unified-spirit-rs writer). resonance_metadata replaces
        # the Python-wrapper resonance_state slot under l0_rust_enabled=true
        # (B.5 deletes the wrapper); filter_down_state is brand-new.
        # unified_spirit_metadata reader already declared above —
        # ownership flipped Python→Rust in B.0 (same slot name + schema).
        "_resonance_metadata", "_filter_down_state",
        # rFP_dead_dim_wiring_fix §2.C / SPEC §7.1 + D-SPEC-68 v1.13.0:
        # cross-process NS-program urgency feed from cognitive_worker
        # to ns_worker (closes the load-bearing wire-up gap from the
        # ns_worker L2 carve-out).
        "_ns_program_urgencies_input",
        # rFP_dead_dim_wiring_fix §2.E + §2.F / SPEC §7.1 + D-SPEC-69 v1.14.0:
        # G18-pure SHM replacement for retired TRAJECTORY_UPDATE +
        # CGN_BETA_SNAPSHOT bus events. Writers = cognitive_worker
        # (trajectory_state) + cgn_worker (cgn_beta_state). Reader =
        # emot_cgn_worker (substrate trajectory_2d + cgn_beta_states keys).
        "_trajectory_state", "_cgn_beta_state",
        # rFP_phase_c_state_read_unification_l0_l1_canonical Phase A.2
        # (2026-05-17): expanded reader inventory so the api_subprocess
        # StateAccessor can read every Python L2 slot SHM-direct (closes
        # the ~50 _cache.get(...) bus-cache state-lookups per G18).
        "_memory_state", "_social_graph_state", "_dream_state",
        "_metabolism_state", "_life_force_state", "_meditation_state",
        "_studio_state", "_expression_state", "_inner_perception_state",
        "_interface_advisor_state",
        "_assessment_state", "_agency_state", "_social_perception_state",
        "_recorder_state", "_timechain_state", "_reflex_state",
        "_output_verifier_state",
        "_body_state", "_mind_state", "_language_state",
        "_events_teacher_state",
        # rFP_phase_c_state_read_unification_l0_l1_canonical Phase A.4
        # (2026-05-17 / v1.10.0 / D-SPEC-71 — renumbered at merge from
        # D-SPEC-70 to resolve collision with parallel session
        # rFP_meta_reasoning_self_reasoning_resolver_migration that landed
        # D-SPEC-70/v1.15.0 first): 10 new Python L2 slots that close the
        # api_subprocess bus-cache state-lookup drift per Preamble G18.
        # Each slot has one canonical producer (G21).
        "_soul_state", "_cgn_engine_state",
        "_reasoning_state", "_meta_reasoning_state", "_meta_teacher_state",
        "_experience_stats",
        "_guardian_state", "_llm_state", "_media_state", "_msl_state",
        "_consciousness_age",
        # ARCH-MAP-HEALTH-OBSERVABILITY Class B (2026-05-26): consciousness_state.bin
        # reader for D-SPEC-86 v1.26.0 slot — closes the
        # `unified_spirit.epoch_count` staleness gap on /v4/inner-trinity.
        "_consciousness_state",
        "_network_state",
        # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §7.1
        # + D-SPEC-70 v1.15.0 (parallel session, landed first): variable
        # msgpack SHM slot carrying the latest SelfReasoningEngine.introspect()
        # output. Writer = self_reflection_worker (G21 single-writer); reader
        # = cognitive_worker (_prim_introspect pre-warmed-cache read per G20).
        "_inner_self_insight",
        # rFP_phase_c_substrate_observable_closure §2.1 follow-on (2026-05-18):
        # per-layer previous-tick magnitude cache for derive_layer_observable
        # velocity + direction computation. Stateless API reads return 0 for
        # vel/dir on cold-boot; subsequent reads compute Δ from this cache.
        # Mirrors Rust topology::TopologyEngine::prev_magnitudes (per-tick
        # tracking in titan-trinity-rs).
        "_prev_layer_magnitudes",
    )

    def __init__(self, titan_id: str | None = None) -> None:
        self.titan_id = resolve_titan_id(titan_id)
        self.shm_root: Path = resolve_shm_root(self.titan_id)
        # Lazy readers — attach on first read.
        self._trinity = StateRegistryReader(TRINITY_STATE, self.shm_root)
        self._neuromod = StateRegistryReader(NEUROMOD_STATE, self.shm_root)
        self._epoch = StateRegistryReader(EPOCH_COUNTER, self.shm_root)
        # Phase A.3 reader gap closure (2026-05-18) — pi_heartbeat overlay
        # in SpiritAccessor.get_coordinator was previously TODO; closes
        # the last deferral in Phase A per the Maker no-deferral rule.
        self._pi_heartbeat = StateRegistryReader(
            PI_HEARTBEAT_STATE, self.shm_root)
        self._inner_spirit = StateRegistryReader(INNER_SPIRIT_45D, self.shm_root)
        self._sphere_clocks = StateRegistryReader(SPHERE_CLOCKS_STATE, self.shm_root)
        self._chi = StateRegistryReader(CHI_STATE, self.shm_root)
        self._titanvm = StateRegistryReader(TITANVM_REGISTERS, self.shm_root)
        # rFP_dead_dim_wiring_fix §2.C / SPEC §7.1 + D-SPEC-68 v1.13.0:
        # cross-process NS-program urgency feed from cognitive_worker
        # (writer) to ns_worker (reader). Mirrors NEUROMOD_INPUTS reader
        # pattern (§4.Q D-SPEC-57).
        self._ns_program_urgencies_input = StateRegistryReader(
            NS_PROGRAM_URGENCIES_INPUT, self.shm_root)
        # rFP_dead_dim_wiring_fix §2.E + §2.F / SPEC §7.1 + D-SPEC-69 v1.14.0:
        # G18-pure SHM replacement for retired TRAJECTORY_UPDATE +
        # CGN_BETA_SNAPSHOT bus events. emot_cgn_worker reads both each
        # bundle-write cycle.
        self._trajectory_state = StateRegistryReader(
            TRAJECTORY_STATE, self.shm_root)
        self._cgn_beta_state = StateRegistryReader(
            CGN_BETA_STATE, self.shm_root)
        # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §7.1
        # + D-SPEC-70 v1.15.0: cross-process bridge from self_reflection_worker
        # (writer) to cognitive_worker (_prim_introspect reader). Closes F-8.
        self._inner_self_insight = StateRegistryReader(
            INNER_SELF_INSIGHT, self.shm_root)
        self._identity = StateRegistryReader(IDENTITY, self.shm_root)
        # rFP_phase_c_substrate_observable_closure §2.1 follow-on — per-layer
        # prev-tick normalized magnitude cache for stateful Δmagnitude
        # (velocity + direction) computation in read_topology_30d.parts.
        # Keys: 6 layer names. None = first tick (vel/dir → 0).
        self._prev_layer_magnitudes: dict[str, float] = {}
        # chunk 8M.4 — additional SPEC §1096 cognitive_worker readers.
        self._topology_30d = StateRegistryReader(TOPOLOGY_30D, self.shm_root)
        self._hormonal = StateRegistryReader(HORMONAL_STATE, self.shm_root)
        self._inner_body = StateRegistryReader(INNER_BODY_5D, self.shm_root)
        self._inner_mind = StateRegistryReader(INNER_MIND_15D, self.shm_root)
        self._outer_body = StateRegistryReader(OUTER_BODY_5D, self.shm_root)
        self._outer_mind = StateRegistryReader(OUTER_MIND_15D, self.shm_root)
        self._outer_spirit = StateRegistryReader(OUTER_SPIRIT_45D, self.shm_root)
        # rFP_worker_broadcast_topics_completion §4.D — variable-size
        # msgpack-encoded composite slots (SHM-first migration for
        # SpiritAccessor.get_nervous_system / get_resonance /
        # get_unified_spirit, completing the abstraction get_sphere_clocks
        # has used since chunk 8M.4).
        self._spirit_supplemental = StateRegistryReader(
            SPIRIT_SUPPLEMENTAL_STATE_SPEC, self.shm_root)
        self._resonance_state = StateRegistryReader(
            RESONANCE_STATE_SPEC, self.shm_root)
        self._unified_spirit_metadata = StateRegistryReader(
            UNIFIED_SPIRIT_METADATA_SPEC, self.shm_root)
        # rFP_phase_c_state_read_unification_l0_l1_canonical Phase B.0
        # SHIPPED 2026-05-17 — 2 Rust-owned slots (titan-unified-spirit-rs
        # MetadataPublisher; G21 single-writer):
        # - resonance_metadata.bin: ResonanceDetector::get_stats() Rust output,
        #   schema-parity with Python wrapper (r4/r6 precision rounded).
        #   B.5 retires the wrapper Python slot.
        # - filter_down_state.bin: FilterDownV5Engine::get_stats() Rust output,
        #   brand-new SHM publish (pre-B.0 was disk-JSON only).
        self._resonance_metadata = StateRegistryReader(
            RESONANCE_METADATA_SPEC, self.shm_root)
        self._filter_down_state = StateRegistryReader(
            FILTER_DOWN_STATE_SPEC, self.shm_root)
        # rFP_phase_c_state_read_unification_l0_l1_canonical Phase A.2
        # (2026-05-17): expanded reader inventory — every Python L2 slot
        # with an existing publisher gets a dedicated reader so the
        # api_subprocess StateAccessor can route _cache.get(...) onto
        # canonical SHM per Preamble G18. Lazy attach on first read.
        self._memory_state = StateRegistryReader(
            MEMORY_STATE_SPEC, self.shm_root)
        self._social_graph_state = StateRegistryReader(
            SOCIAL_GRAPH_STATE_SPEC, self.shm_root)
        self._dream_state = StateRegistryReader(
            DREAM_STATE_SPEC, self.shm_root)
        self._metabolism_state = StateRegistryReader(
            METABOLISM_STATE_SPEC, self.shm_root)
        self._life_force_state = StateRegistryReader(
            LIFE_FORCE_STATE_SPEC, self.shm_root)
        self._meditation_state = StateRegistryReader(
            MEDITATION_STATE_SPEC, self.shm_root)
        self._studio_state = StateRegistryReader(
            STUDIO_STATE_SPEC, self.shm_root)
        self._expression_state = StateRegistryReader(
            EXPRESSION_STATE_SPEC, self.shm_root)
        self._inner_perception_state = StateRegistryReader(
            INNER_PERCEPTION_STATE_SPEC, self.shm_root)
        self._interface_advisor_state = StateRegistryReader(
            INTERFACE_ADVISOR_STATE_SPEC, self.shm_root)
        # Session 3 (rFP_phase_c_async_shm_consumer_migration §4.B.2-B.5
        # + B.9-B.11) — agency_worker / rl→recorder_worker / timechain_worker /
        # reflex_worker / output_verifier_worker / SpiritStatePublisher
        # social_perception extension.
        self._assessment_state = StateRegistryReader(
            ASSESSMENT_STATE_SPEC, self.shm_root)
        self._agency_state = StateRegistryReader(
            AGENCY_STATE_SPEC, self.shm_root)
        self._social_perception_state = StateRegistryReader(
            SOCIAL_PERCEPTION_STATE_SPEC, self.shm_root)
        self._recorder_state = StateRegistryReader(
            RL_STATE_SPEC, self.shm_root)
        self._timechain_state = StateRegistryReader(
            TIMECHAIN_STATE_SPEC, self.shm_root)
        self._reflex_state = StateRegistryReader(
            REFLEX_STATE_SPEC, self.shm_root)
        self._output_verifier_state = StateRegistryReader(
            OUTPUT_VERIFIER_STATE_SPEC, self.shm_root)
        # Session 4 (rFP_phase_c_async_shm_consumer_migration §4.C.2-3 +
        # §4.B.7) — body_worker / mind_worker / language_worker (the
        # last co-located events_teacher_state per memory note in
        # §7.1 row 792).
        self._body_state = StateRegistryReader(
            BODY_STATE_SPEC, self.shm_root)
        self._mind_state = StateRegistryReader(
            MIND_STATE_SPEC, self.shm_root)
        self._language_state = StateRegistryReader(
            LANGUAGE_STATE_SPEC, self.shm_root)
        self._events_teacher_state = StateRegistryReader(
            EVENTS_TEACHER_STATE_SPEC, self.shm_root)
        # Phase A.4 readers (rFP_phase_c_state_read_unification_l0_l1_canonical
        # v1.10.0 / D-SPEC-70).
        self._soul_state = StateRegistryReader(
            SOUL_STATE_SPEC, self.shm_root)
        self._cgn_engine_state = StateRegistryReader(
            CGN_ENGINE_STATE_SPEC, self.shm_root)
        self._reasoning_state = StateRegistryReader(
            REASONING_STATE_SPEC, self.shm_root)
        self._meta_reasoning_state = StateRegistryReader(
            META_REASONING_STATE_SPEC, self.shm_root)
        self._meta_teacher_state = StateRegistryReader(
            META_TEACHER_STATE_SPEC, self.shm_root)
        # §3L Phase 15 chunk 15.1 (D-SPEC-PHASE15) — experience_stats.bin
        # reader. Writer is cognitive_worker (G21 single-writer;
        # ExperienceOrchestrator instance). Replaces the retired
        # ExperienceMemory.get_stats recompute-on-read path per G18.
        self._experience_stats = StateRegistryReader(
            EXPERIENCE_STATS_SPEC, self.shm_root)
        self._guardian_state = StateRegistryReader(
            GUARDIAN_STATE_SPEC, self.shm_root)
        self._llm_state = StateRegistryReader(
            LLM_STATE_SPEC, self.shm_root)
        self._media_state = StateRegistryReader(
            MEDIA_STATE_SPEC, self.shm_root)
        self._msl_state = StateRegistryReader(
            MSL_STATE_SPEC, self.shm_root)
        self._network_state = StateRegistryReader(
            NETWORK_STATE_SPEC, self.shm_root)
        # D-SPEC-85 v1.25.0 (2026-05-18) — consciousness_age slot
        # surfaces Titan's "main age" (lifetime self-observation tick
        # counter from consciousness.db) to post_dispatch which cannot
        # reach the DB per G18. G21 single-writer = cognitive_worker.
        self._consciousness_age = StateRegistryReader(
            CONSCIOUSNESS_AGE_SPEC, self.shm_root)
        # ARCH-MAP-HEALTH-OBSERVABILITY Class B (2026-05-26) — D-SPEC-86
        # v1.26.0 consciousness_state.bin reader. Writer is cognitive_worker
        # (G21 single-writer at cognitive_worker.py:3002, Phase 3.A). The
        # payload carries the current `epoch_count`/`epoch_id` that the
        # /v4/inner-trinity API must overlay on top of the stale
        # cached-snapshot `unified_spirit.epoch_count` per the
        # ARCH-MAP-HEALTH-OBSERVABILITY rFP.
        self._consciousness_state = StateRegistryReader(
            CONSCIOUSNESS_STATE_SPEC, self.shm_root)
        logger.info(
            "[ShmReaderBank] initialized for titan_id=%s root=%s",
            self.titan_id, self.shm_root,
        )

    # -- Trinity (162D) ------------------------------------------------

    def read_trinity(self) -> dict[str, Any] | None:
        """Return Trinity state with subgroups + metadata, or None."""
        payload, age, seq = _payload_with_meta(self._trinity, "trinity_state")
        if payload is None:
            return None
        return {
            "full_130dt": payload[TRINITY_GROUPS["full_130dt"]].tolist(),
            "full_30d_topology": payload[TRINITY_GROUPS["full_30d_topology"]].tolist(),
            "journey": {
                "curvature": float(payload[160]),
                "density": float(payload[161]),
            },
            "age_seconds": age,
            "seq": seq,
        }

    # -- Neuromodulators (6) -------------------------------------------

    def read_neuromod(self) -> dict[str, Any] | None:
        """Return neuromodulator levels by name + metadata, or None.

        SHM `neuromod_state` is shape (6, 4) per state_registry.NEUROMOD_STATE
        — 6 modulators × (level, gain, phasic, tonic). `payload[i]` is the
        4-element row for modulator i; level is field 0.
        rFP_dead_dim_wiring_fix §2.D bonus — previous `float(payload[i])`
        raised TypeError (0-dim conversion of a 1-D array).
        """
        payload, age, seq = _payload_with_meta(self._neuromod, "neuromod_state")
        if payload is None:
            return None
        return {
            "modulators": {
                name: {"level": float(payload[i][0])}
                for i, name in enumerate(NEUROMOD_NAMES)
            },
            "age_seconds": age,
            "seq": seq,
        }

    # -- Epoch counter -------------------------------------------------

    def read_pi_heartbeat(self) -> dict[str, Any] | None:
        """π-heartbeat state — Rust L0 owned by titan-kernel-rs. Returns
        {phase, pulse_count, age_seconds, seq} or None on SHM unavailability.
        Closes Phase A.3 reader gap 2026-05-18 (overlay was TODO)."""
        payload, age, seq = _payload_with_meta(self._pi_heartbeat, "pi_heartbeat")
        if payload is None:
            return None
        record = payload[0]
        return {
            "phase": float(record["phase"]),
            "pulse_count": int(record["pulse_count"]),
            "age_seconds": age,
            "seq": seq,
        }

    def read_epoch(self) -> dict[str, Any] | None:
        """Return current consciousness epoch counter + metadata, or None."""
        payload, age, seq = _payload_with_meta(self._epoch, "epoch_counter")
        if payload is None:
            return None
        return {
            "epoch": int(payload[0]),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Inner Spirit 45D ---------------------------------------------

    def read_inner_spirit_45d(self) -> dict[str, Any] | None:
        """Return SAT/CHIT/ANANDA groups + metadata, or None.

        S3b shm-spirit-fast — written at 70.47 Hz when flag enabled.
        """
        payload, age, seq = _payload_with_meta(
            self._inner_spirit, "inner_spirit_45d")
        if payload is None:
            return None
        # NEW 2026-05-18: also expose the flat 45D `values` array so that
        # consumers reading per-layer LayerObservables + the v3/trinity
        # spirit panel can access the canonical Rust L0+L1 tensor without
        # SAT/CHIT/ANANDA-specific knowledge. Matches the shape returned
        # by read_inner_body_5d / read_inner_mind_15d / etc.
        flat = payload.tolist() if hasattr(payload, "tolist") else list(payload)
        return {
            "values": [float(x) for x in flat],
            "SAT": payload[INNER_SPIRIT_GROUPS["SAT"]].tolist(),
            "CHIT": payload[INNER_SPIRIT_GROUPS["CHIT"]].tolist(),
            "ANANDA": payload[INNER_SPIRIT_GROUPS["ANANDA"]].tolist(),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Sphere clocks (S4) -------------------------------------------

    def read_sphere_clocks(self) -> dict[str, Any] | None:
        """Return per-clock state by name + metadata, or None.

        S4 — flag-gated by `microkernel.shm_sphere_clocks_enabled`.
        """
        payload, age, seq = _payload_with_meta(
            self._sphere_clocks, "sphere_clocks")
        if payload is None:
            return None
        clocks = {}
        for i, name in enumerate(SPHERE_CLOCK_NAMES):
            row = payload[i]
            clocks[name] = {
                field: float(row[j])
                for j, field in enumerate(SPHERE_CLOCK_FIELDS)
            }
        return {
            "clocks": clocks,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Chi state (S4) ------------------------------------------------

    def read_chi(self) -> dict[str, Any] | None:
        """Return chi circulation fields by name + metadata, or None.

        S4 — flag-gated by `microkernel.shm_chi_enabled`.
        """
        payload, age, seq = _payload_with_meta(self._chi, "chi_state")
        if payload is None:
            return None
        return {
            **{name: float(payload[i]) for i, name in enumerate(CHI_FIELD_NAMES)},
            "age_seconds": age,
            "seq": seq,
        }

    # -- TitanVM registers (S4) ---------------------------------------

    def read_titanvm_registers(self) -> dict[str, Any] | None:
        """Return per-NS-program register state + metadata, or None.

        S4 — flag-gated by `microkernel.shm_titanvm_enabled`.
        """
        payload, age, seq = _payload_with_meta(self._titanvm, "titanvm_registers")
        if payload is None:
            return None
        programs = {}
        for i, name in enumerate(NS_PROGRAM_NAMES):
            row = payload[i]
            programs[name] = {
                field: float(row[j])
                for j, field in enumerate(NS_PROGRAM_FIELDS)
            }
        return {
            "programs": programs,
            "age_seconds": age,
            "seq": seq,
        }

    def read_trajectory_state(self) -> dict[str, Any] | None:
        """Return [curvature, density] + metadata, or None.

        rFP_dead_dim_wiring_fix §2.E / SPEC §7.1 + D-SPEC-69 v1.14.0.
        Slot `trajectory_state.bin` is the G18-pure SHM replacement for
        the retired TRAJECTORY_UPDATE bus event (PART B §8 D-SPEC-65).
        cognitive_worker writes the 2 floats directly from coordinator's
        freshly-computed values per `_run_consciousness_epoch` (bypassing
        the broken `consciousness["latest_epoch"].state_vector` snapshot
        pipe that reads empty `[]` post-Phase-C api_subprocess migration).
        """
        payload, age, seq = _payload_with_meta(
            self._trajectory_state, "trajectory_state")
        if payload is None:
            return None
        return {
            "trajectory_2d": [float(payload[0]), float(payload[1])],
            "age_seconds": age,
            "seq": seq,
        }

    def read_cgn_beta_state(self) -> dict[str, Any] | None:
        """Return per-consumer reward_ema dict + metadata, or None.

        rFP_dead_dim_wiring_fix §2.F / SPEC §7.1 + D-SPEC-69 v1.14.0.
        Slot `cgn_beta_state.bin` is the G18-pure SHM replacement for
        the retired CGN_BETA_SNAPSHOT bus event (§23.6a). cgn_worker
        writes the 8 floats directly from live cgn_consumer._reward_ema
        (NOT the snapshot that defaults to 0.5 → degenerate values).
        Returns dict keyed by CGN_CONSUMERS order (language, social,
        knowledge, reasoning, coding, self_model, reasoning_strategy,
        meta from emot_bundle_protocol.py:172-175).
        """
        payload, age, seq = _payload_with_meta(
            self._cgn_beta_state, "cgn_beta_state")
        if payload is None:
            return None
        # Lazy import to avoid circular deps at module level.
        from titan_hcl.logic.emot_bundle_protocol import CGN_CONSUMERS
        return {
            "values_by_consumer": {
                name: float(payload[i])
                for i, name in enumerate(CGN_CONSUMERS)
            },
            "age_seconds": age,
            "seq": seq,
        }

    def read_ns_program_urgencies_input(self) -> dict[str, Any] | None:
        """Return canonical per-NS-program urgency snapshot or None.

        rFP_dead_dim_wiring_fix §2.C / SPEC §7.1 + D-SPEC-68 v1.13.0.
        Slot `ns_program_urgencies_input.bin` is the cross-process bridge
        from cognitive_worker (G21 single-writer; sources from
        `coordinator._last_nervous_signals` per consciousness epoch tick)
        to ns_worker (downstream titanvm_registers.bin urgency column
        writer + NS_URGENCIES_UPDATE emitter for emot_cgn `ns_urgencies`
        substrate cache). Pattern mirrors NEUROMOD_INPUTS (§4.Q D-SPEC-57).

        Returns ``{"urgencies_by_program": {NAME: float, ...},
                   "age_seconds": float, "seq": int}`` — same key shape
        as the legacy NS_URGENCIES_UPDATE bus payload so ns_worker's
        downstream emit path stays unchanged.
        """
        payload, age, seq = _payload_with_meta(
            self._ns_program_urgencies_input, "ns_program_urgencies_input")
        if payload is None:
            return None
        return {
            "urgencies_by_program": {
                name: float(payload[i])
                for i, name in enumerate(NS_PROGRAM_NAMES)
            },
            "age_seconds": age,
            "seq": seq,
        }

    # -- Identity (S4) -------------------------------------------------

    def read_identity(self) -> dict[str, Any] | None:
        """Return titan_id + maker_pubkey + kernel_instance_nonce, or None.

        S4 — flag-gated by `microkernel.shm_identity_enabled`. Until
        flag flipped, falls back to None and endpoint code reads from
        bus-cached "identity.*" keys.
        """
        payload, age, seq = _payload_with_meta(self._identity, "identity")
        if payload is None:
            return None
        # Decode per IDENTITY layout (state_registry.py:740-744):
        #   [0:32]   titan_id (UTF-8, NUL-padded)
        #   [32:64]  maker_pubkey (raw 32-byte Ed25519, zero if absent)
        #   [64:96]  kernel_instance_nonce (random per boot)
        titan_id_bytes = bytes(payload[0:32]).rstrip(b"\x00")
        maker_pubkey_bytes = bytes(payload[32:64])
        nonce_bytes = bytes(payload[64:96])
        return {
            "titan_id": titan_id_bytes.decode("utf-8", errors="replace"),
            "maker_pubkey": maker_pubkey_bytes.hex() if any(maker_pubkey_bytes) else "",
            "kernel_instance_nonce": nonce_bytes.hex(),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Topology 30D (chunk 8M.4) ------------------------------------

    def read_topology_30d(self) -> dict[str, Any] | None:
        """Return 30D space-topology vector + structured 6-LAYER view + meta.

        Written by titan-trinity-rs per SPEC §9.A. Cognitive_worker reads
        per epoch and injects into coordinator.topology snapshot.

        `parts` is a per-tick derivation of 6 LayerObservables from the 6
        daemon-tensor slots, NOT a slice of the 30D output (per
        rFP_phase_c_substrate_observable_closure §2.1 / D-SPEC-80; mirror of
        titan-trinity-rs::topology::TopologyEngine::derive_layer_observables).
        Each layer's tensor T (5D, 15D, or 45D) yields a 5-tuple:
          - coherence  = `1 - variance(T) / 0.25` clamped [0, 1]
                         (canonical `middle_path.layer_coherence` per SPEC §G4 + D-SPEC-84)
          - magnitude  = l2_norm(T) / sqrt(N) clamped [0, 1]
          - velocity   = 0.0 (Python-side reader is stateless; per-tick
                              prev-magnitude tracking lives in Rust)
          - direction  = 0.0 (same stateless caveat)
          - polarity   = clamp((mean(T) - 0.5) * 2, -1, 1); 0 for zero-mag
        """
        payload, age, seq = _payload_with_meta(self._topology_30d, "topology_30d")
        if payload is None:
            return None
        flat = payload.tolist()
        # Derive per-LAYER observables from the 6 daemon tensor slots.
        # Each reader returns {"values": [floats]} or None.
        layer_specs = (
            ("inner_body", self.read_inner_body_5d),
            ("inner_mind", self.read_inner_mind_15d),
            ("inner_spirit", self.read_inner_spirit_45d),
            ("outer_body", self.read_outer_body_5d),
            ("outer_mind", self.read_outer_mind_15d),
            ("outer_spirit", self.read_outer_spirit_45d),
        )
        parts: dict[str, dict[str, float]] = {}
        for name, reader in layer_specs:
            tensor: list[float] = []
            try:
                src = reader() or {}
                src_vals = src.get("values")
                if isinstance(src_vals, list):
                    tensor = [float(v) for v in src_vals]
            except Exception:
                tensor = []
            obs = _derive_layer_observable(tensor)
            # Stateful velocity + direction (mirror Rust
            # TopologyEngine::prev_magnitudes per-tick tracking).
            n = len(tensor)
            if n > 0:
                mag_sq = sum(v * v for v in tensor)
                cur_mag = (mag_sq ** 0.5 / (n ** 0.5)) if n > 0 else 0.0
                cur_mag = max(0.0, min(1.0, cur_mag))
                prev = self._prev_layer_magnitudes.get(name)
                if prev is not None:
                    dm = cur_mag - prev
                    obs["velocity"] = round(min(1.0, abs(dm)), 6)
                    obs["direction"] = 1.0 if dm > 0 else (-1.0 if dm < 0 else 0.0)
                # Update cache for next tick.
                self._prev_layer_magnitudes[name] = cur_mag
            parts[name] = obs
        return {
            "values": flat,
            "parts": parts,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Hormonal (11×4) (chunk 8M.4) ----------------------------------

    def read_hormonal(self) -> dict[str, Any] | None:
        """Return per-hormone 4-field state by name + metadata, or None.

        Written by hormonal_worker (registered in chunk 8M.1) via
        HORMONAL_STATE shm slot. Read by cognitive_worker per epoch +
        injected into coordinator.hormonal snapshot.
        """
        payload, age, seq = _payload_with_meta(self._hormonal, "hormonal_state")
        if payload is None:
            return None
        hormones = {}
        for i, name in enumerate(HORMONE_NAMES):
            row = payload[i]
            hormones[name] = {
                field: float(row[j])
                for j, field in enumerate(HORMONE_FIELDS)
            }
        return {
            "hormones": hormones,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Inner / Outer trinity tensors (5/15/45) (chunk 8M.4) ---------

    def read_inner_body_5d(self) -> dict[str, Any] | None:
        """5D inner-body tensor — written by titan-inner-body-rs."""
        payload, age, seq = _payload_with_meta(self._inner_body, "inner_body_5d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_inner_mind_15d(self) -> dict[str, Any] | None:
        """15D inner-mind tensor — written by titan-inner-mind-rs."""
        payload, age, seq = _payload_with_meta(self._inner_mind, "inner_mind_15d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_body_5d(self) -> dict[str, Any] | None:
        """5D outer-body tensor — written by titan-outer-body-rs."""
        payload, age, seq = _payload_with_meta(self._outer_body, "outer_body_5d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_mind_15d(self) -> dict[str, Any] | None:
        """15D outer-mind tensor — written by titan-outer-mind-rs."""
        payload, age, seq = _payload_with_meta(self._outer_mind, "outer_mind_15d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_spirit_45d(self) -> dict[str, Any] | None:
        """45D outer-spirit tensor — written by titan-outer-spirit-rs."""
        payload, age, seq = _payload_with_meta(
            self._outer_spirit, "outer_spirit_45d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    # -- Variable-size composite slots (rFP_worker_broadcast_topics §4.D) ---
    #
    # spirit_supplemental_state.bin / resonance_state.bin /
    # unified_spirit_metadata.bin are msgpack-encoded variable-size slots
    # produced by SpiritStatePublisher + SpiritSupplementalStatePublisher
    # in spirit_loop's snapshot-builder threads. Read via reader.read_variable()
    # (NOT read() which is fixed-size), then msgpack-decode.
    #
    # Closes the bug surfaced 2026-05-10: SpiritAccessor.get_nervous_system /
    # get_resonance / get_unified_spirit returned `{}` because they read
    # bus-cache only (`self._cache.get("spirit.X", {}) or {}`) — and on T3
    # under l0_rust_enabled=true, spirit_worker is heartbeat-only so the
    # cache key is never populated. The publisher writes SHM correctly
    # (verified: spirit_supplemental_state.bin has 53,934-byte payload with
    # 4 sections including nervous_system with 10 keys); the api_subprocess
    # accessor just wasn't reading SHM. This completes the abstraction
    # get_sphere_clocks has used since chunk 8M.4 (state_accessor.py:242-247).

    def _read_msgpack_variable(
        self,
        reader: StateRegistryReader,
        slot_name: str,
    ) -> dict[str, Any] | None:
        """Read variable-size SHM slot, msgpack-decode, return dict.

        Returns None on cold-boot / missing / torn / decode-failure.
        Mirrors the proven pattern in titan_hcl/proxies/spirit_proxy.py:
        _read_msgpack — same byte-format contract, same failure modes.
        """
        try:
            raw = reader.read_variable()
        except Exception as e:
            logger.warning(
                "[ShmReaderBank] %s read_variable raised: %s",
                slot_name, e, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            import msgpack
            # strict_map_key=False: some publishers (e.g. assessment_state's
            # `recent`/`trend` sub-maps) carry int map keys; the msgpack default
            # (strict_map_key=True) rejects them and silently returns None →
            # the slot reads as "absent" fleet-wide (api StateAccessor + the
            # Phase C outer-source assembler both hit this). Permissive decode
            # matches the writer's intent. (Phase C dissolution fix 2026-05-22.)
            decoded = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception as e:
            logger.warning(
                "[ShmReaderBank] %s msgpack decode failed: %s",
                slot_name, e)
            return None
        if not isinstance(decoded, dict):
            logger.warning(
                "[ShmReaderBank] %s decoded to non-dict: %s",
                slot_name, type(decoded).__name__)
            return None
        return decoded

    def read_spirit_supplemental(self) -> dict[str, Any] | None:
        """Return full spirit_supplemental_state dict (4 sections:
        filter_down_status / meditation_health / coordinator / nervous_system
        + ts), or None if SHM unavailable. Per Session 4 §4.C.1 expansion
        producer in spirit_loop snapshot-builder.

        Caller extracts the section it needs (e.g., for /v4/nervous-system,
        SpiritAccessor.get_nervous_system reads `dict["nervous_system"]`).
        """
        return self._read_msgpack_variable(
            self._spirit_supplemental, "spirit_supplemental_state")

    def read_resonance_state(self) -> dict[str, Any] | None:
        """Return ResonanceDetector.get_stats() output from the LEGACY
        Python-wrapper slot ``resonance_state.bin``, or None if SHM
        unavailable. Producer: SpiritStatePublisher (retired by Phase B.5).
        Prefer ``read_resonance_metadata`` (Rust-owned canonical slot)
        for new callers per rFP_phase_c_state_read_unification §B."""
        return self._read_msgpack_variable(
            self._resonance_state, "resonance_state")

    def read_unified_spirit_metadata(self) -> dict[str, Any] | None:
        """Return UnifiedSpirit::get_stats() output (epoch_count, velocity,
        is_stale, tensor_magnitude, latest_epoch dict, etc.), or None if
        SHM unavailable. Phase B.0 SHIPPED 2026-05-17 flipped this slot
        from Python→Rust ownership (titan-unified-spirit-rs
        MetadataPublisher; G21 single-writer under l0_rust_enabled=true);
        schema preserved 1:1 with the retired Python publisher (r4/r6
        precision rounded for byte-identical msgpack)."""
        return self._read_msgpack_variable(
            self._unified_spirit_metadata, "unified_spirit_metadata")

    def read_resonance_metadata(self) -> dict[str, Any] | None:
        """Return ResonanceDetector::get_stats() output (pairs,
        resonant_count, all_resonant, great_pulse_count,
        last_great_pulse_ts, config) from the Rust-owned canonical slot
        ``resonance_metadata.bin``, or None if SHM unavailable.
        Phase B.0 SHIPPED 2026-05-17 — titan-unified-spirit-rs
        MetadataPublisher; G21 single-writer. Replaces the legacy
        Python-wrapper ``resonance_state.bin`` slot under
        l0_rust_enabled=true (B.5 deletes the wrapper). Schema-parity
        preserved with the retired Python publisher (r4/r6 precision
        rounded so msgpack is byte-identical pre/post-flip)."""
        return self._read_msgpack_variable(
            self._resonance_metadata, "resonance_metadata")

    def read_filter_down_state(self) -> dict[str, Any] | None:
        """Return FilterDownV5Engine::get_stats() output (version,
        input_dim, output_dim, buffer_size, total_train_steps, last_loss,
        publish_enabled, spirit_filter_strength, cold_start_floor,
        multipliers_mean, multipliers) from the Rust-owned canonical slot
        ``filter_down_state.bin``, or None if SHM unavailable.
        Phase B.0 SHIPPED 2026-05-17 — brand-new SHM publish path
        (pre-B.0 was disk-JSON only via ``data/filter_down_v5_state.json``).
        titan-unified-spirit-rs MetadataPublisher; G21 single-writer."""
        return self._read_msgpack_variable(
            self._filter_down_state, "filter_down_state")

    # -- Phase B.4 composite helpers (rFP_phase_c_state_read_unification §B.4)
    #
    # ``compose_trinity`` and ``compose_v4_state`` mirror the response shape
    # of SpiritProxy.get_trinity / get_v4_state but read PURELY from
    # Rust L0+L1 canonical slots — closing the Maker directive
    # ("EVERYTHING IN OUR CODEBASE THAT READS STATE MUST READ IT FROM
    # L0+L1 rust layer"). Replaces SpiritProxy as the trinity composer
    # so spirit_proxy.py can be retired at B.5.
    #
    # Slot sources (all Rust-owned post-B.0 except as noted):
    #   - body_values            ← inner_body_5d.bin (Rust)
    #   - mind_values            ← inner_mind_15d.bin[0:5] (Rust)
    #   - spirit_tensor          ← unified_spirit_metadata.full_130dt[0:5]
    #   - consciousness          ← unified_spirit_metadata.latest_epoch (Rust)
    #   - sphere_clock           ← sphere_clocks.bin (Rust L1)
    #   - resonance              ← resonance_metadata.bin (Rust B.0)
    #   - unified_spirit         ← unified_spirit_metadata.bin (Rust B.0)
    #   - filter_down            ← filter_down_state.bin (Rust B.0)
    #   - hormone_levels         ← hormonal_state.bin (Rust L1)
    #   - middle_path_loss       ← computed inline (middle_path module)
    #   - body/mind_center_dist  ← 0.0 (B.0 metric; no Rust producer yet —
    #                              non-load-bearing observatory metric)
    # Dropped (no Rust home; observatory-only; callers use .get() defaults):
    #   - hormone_fires (Python L2 metric)
    #   - impulse_engine (Python L2 metric)
    #   - impulse_engine_hormones (Python L2 metric)

    def compose_trinity(self) -> dict[str, Any]:
        """Compose the Trinity snapshot dict from Rust L0+L1 canonical SHM
        slots. Output shape matches SpiritProxy.get_trinity() so callers
        can swap in place. Returns a populated dict even on cold-boot
        (missing slots → empty/default fields)."""
        body_pl = self.read_inner_body_5d() or {}
        mind_pl = self.read_inner_mind_15d() or {}
        unified_spirit_pl = self.read_unified_spirit_metadata() or {}
        body_values = [float(v) for v in (body_pl.get("values") or [0.5] * 5)][:5]
        mind_values = [float(v) for v in (mind_pl.get("values") or [0.5] * 15)][:5]
        full_130dt = unified_spirit_pl.get("full_130dt") or []
        if isinstance(full_130dt, list) and len(full_130dt) >= 5:
            spirit_tensor = [float(v) for v in full_130dt[:5]]
        else:
            spirit_tensor = [0.5] * 5

        response: dict[str, Any] = {
            "spirit_tensor": spirit_tensor,
            "body_values": body_values,
            "mind_values": mind_values,
            "body_center_dist": 0.0,
            "mind_center_dist": 0.0,
        }

        latest_epoch = unified_spirit_pl.get("latest_epoch")
        if isinstance(latest_epoch, dict) and latest_epoch:
            response["consciousness"] = latest_epoch

        try:
            from titan_hcl.logic.middle_path import middle_path_loss
            response["middle_path_loss"] = round(
                middle_path_loss(body_values, mind_values, spirit_tensor), 4)
        except Exception as e:
            logger.warning(
                "[ShmReaderBank] middle_path_loss compute failed: %s",
                e, exc_info=True)

        hormonal_pl = self.read_hormonal()
        if hormonal_pl is not None:
            hormones = hormonal_pl.get("hormones") or {}
            response["hormone_levels"] = {
                name: float(payload.get("level", 0.0))
                for name, payload in hormones.items()
            }

        sphere_pl = self.read_sphere_clocks()
        if sphere_pl is not None:
            clocks = sphere_pl.get("clocks")
            if isinstance(clocks, dict) and clocks:
                response["sphere_clock"] = clocks
            else:
                response["sphere_clock"] = {
                    k: v for k, v in sphere_pl.items()
                    if k not in ("age_seconds", "seq", "clocks")
                }

        resonance_pl = self.read_resonance_metadata()
        if resonance_pl is not None:
            response["resonance"] = {
                k: v for k, v in resonance_pl.items()
                if k not in ("ts", "schema_version")
            }

        unified_strip = {
            k: v for k, v in unified_spirit_pl.items()
            if k not in ("ts", "schema_version")
        }
        if unified_strip:
            response["unified_spirit"] = unified_strip

        filter_down_pl = self.read_filter_down_state()
        if filter_down_pl is not None:
            response["filter_down"] = {
                k: v for k, v in filter_down_pl.items()
                if k not in ("ts", "schema_version")
            }

        return response

    def compose_v4_state(self) -> dict[str, Any]:
        """Compose the V4 Time-Awareness state dict from Rust L0+L1
        canonical SHM slots. Output shape matches SpiritProxy.get_v4_state()."""
        trinity = self.compose_trinity()
        return {
            "sphere_clock": trinity.get("sphere_clock", {}),
            "resonance": trinity.get("resonance", {}),
            "unified_spirit": trinity.get("unified_spirit", {}),
            "filter_down": trinity.get("filter_down", {}),
            "consciousness": trinity.get("consciousness", {}),
            "middle_path_loss": trinity.get("middle_path_loss"),
        }

    # -- Python L2 worker state slots (Phase A.2 expansion 2026-05-17) ---
    # Producers per SPEC §7.1 / §9.B. All variable-size msgpack.

    def read_memory_state(self) -> dict[str, Any] | None:
        """MemoryStatePublisher payload — persistent_count, mempool_size,
        learning_velocity, directive_alignment, effective_nodes_24h,
        high_quality_count, kg_node_count, kg_edge_count,
        topology_clusters_summary, ts. Producer: memory_worker."""
        return self._read_msgpack_variable(
            self._memory_state, "memory_state")

    def read_social_graph_state(self) -> dict[str, Any] | None:
        """SocialGraphStatePublisher payload — users, edges, donations,
        total_donated_sol, inspirations, engagement_ledger_today,
        schema_version, ts. Producer: social_graph_worker."""
        return self._read_msgpack_variable(
            self._social_graph_state, "social_graph_state")

    def read_dream_state(self) -> dict[str, Any] | None:
        """DreamStatePublisher payload — is_dreaming, state,
        recovery_pct, remaining_epochs, wake_transition, just_woke,
        wake_ts, dream_started_ts, last_transition_ts, ts.
        Producer: dream_state_worker."""
        return self._read_msgpack_variable(
            self._dream_state, "dream_state")

    def read_metabolism_state(self) -> dict[str, Any] | None:
        """MetabolismStatePublisher payload — tier, balance_pct,
        gates_enforced, last_gate_decision_reason, tier_info,
        last_tier_change_ts, social_gravity_score, ts.
        Producer: metabolism_worker."""
        return self._read_msgpack_variable(
            self._metabolism_state, "metabolism_state")

    def read_life_force_state(self) -> dict[str, Any] | None:
        """LifeForceStatePublisher payload — total, spirit/mind/body
        sub-states, circulation, weights, state, developmental_phase,
        contemplation, metabolic_drain, is_dreaming, ts.
        Producer: life_force_worker."""
        return self._read_msgpack_variable(
            self._life_force_state, "life_force_state")

    def read_meditation_state(self) -> dict[str, Any] | None:
        """MeditationStatePublisher payload — tracker, watchdog,
        last_alert, last_completion, schema_version, ts.
        Producer: meditation_worker."""
        return self._read_msgpack_variable(
            self._meditation_state, "meditation_state")

    def read_studio_state(self) -> dict[str, Any] | None:
        """StudioStatePublisher payload — meditation/epoch/eureka counts,
        last_render_ts, last_render_type, output_root, resolution,
        nft_composite_enabled, ts. Producer: studio_worker."""
        return self._read_msgpack_variable(
            self._studio_state, "studio_state")

    def read_expression_state(self) -> dict[str, Any] | None:
        """ExpressionStatePublisher payload — 6 composites (SPEAK/ART/
        MUSIC/SOCIAL/KIN_SENSE/LONGING) + ledger + ts.
        Producer: expression_worker."""
        return self._read_msgpack_variable(
            self._expression_state, "expression_state")

    def read_inner_perception_state(self) -> dict[str, Any] | None:
        """InnerPerceptionStatePublisher payload — audio_state, visual_state,
        ambient_change, last_create_ts, ts. Producer: main plugin (parent-
        resident hardware sampler). Phase C dissolution 2026-05-22."""
        return self._read_msgpack_variable(
            self._inner_perception_state, "inner_perception_state")

    def read_interface_advisor_state(self) -> dict[str, Any] | None:
        """InterfaceAdvisorStatePublisher payload — rates, limits,
        window_s, rate_limit_count, schema_version, ts.
        Producer: interface_advisor_worker."""
        return self._read_msgpack_variable(
            self._interface_advisor_state, "interface_advisor_state")

    # -- Session 3 publishers (agency / recorder / timechain / reflex /
    # output_verifier / social_perception / assessment) -----------------

    def read_assessment_state(self) -> dict[str, Any] | None:
        """AssessmentStatePublisher payload — average_score, total,
        recent, trend, score_variance, research_avg_score, ts.
        Producer: agency_worker (assessment co-located)."""
        return self._read_msgpack_variable(
            self._assessment_state, "assessment_state")

    def read_agency_state(self) -> dict[str, Any] | None:
        """AgencyStatePublisher payload — total_actions, actions_this_hour,
        success_rate, llm_calls_this_hour, helper_statuses, last_action_ts,
        posture_history_digest, ts. Producer: agency_worker."""
        return self._read_msgpack_variable(
            self._agency_state, "agency_state")

    def read_social_perception_state(self) -> dict[str, Any] | None:
        """SocialPerceptionStatePublisher payload — sentiment_ema,
        interaction_rate, social_activity, last_interaction_ts, ts.
        Producer: spirit_worker (SpiritStatePublisher extension)."""
        return self._read_msgpack_variable(
            self._social_perception_state, "social_perception_state")

    def read_recorder_state(self) -> dict[str, Any] | None:
        """RecorderStatePublisher payload — programs, current_program_id,
        dream_quality, training_loss_ema, transitions, last_train_ts, ts.
        Producer: recorder_worker (formerly rl_worker; slot path is
        recorder_state.bin per RL_STATE_SLOT = 'recorder_state')."""
        return self._read_msgpack_variable(
            self._recorder_state, "recorder_state")

    def read_timechain_state(self) -> dict[str, Any] | None:
        """TimechainStatePublisher payload — tx_latency_norm,
        block_delta_norm, recent_anchor_age_s, fork_summary,
        integrity_status, total_blocks, chi_spent_total, ts.
        Producer: timechain_worker."""
        return self._read_msgpack_variable(
            self._timechain_state, "timechain_state")

    def read_reflex_state(self) -> dict[str, Any] | None:
        """ReflexStatePublisher payload — per-reflex stats + ts.
        Producer: reflex_worker."""
        return self._read_msgpack_variable(
            self._reflex_state, "reflex_state")

    def read_output_verifier_state(self) -> dict[str, Any] | None:
        """OutputVerifierStatePublisher payload — verified_count,
        rejected_count, sovereignty_score, threats_24h,
        recent_rejections_digest, ts. Producer: output_verifier_worker."""
        return self._read_msgpack_variable(
            self._output_verifier_state, "output_verifier_state")

    # -- Session 4 publishers (body / mind / language / events_teacher) ---

    def read_body_state(self) -> dict[str, Any] | None:
        """BodyStatePublisher payload — interoception, proprioception,
        somatosensation, entropy, thermal, sol_balance, sol_norm,
        block_delta_norm, anchor_fresh, body_health, body_details, ts.
        Producer: body_worker."""
        return self._read_msgpack_variable(
            self._body_state, "body_state")

    def read_mind_state(self) -> dict[str, Any] | None:
        """MindStatePublisher payload — mood_label, mood_valence,
        mood_intensity, current_reward, info_gain_ema, mood_history_digest,
        ts. Producer: mind_worker."""
        return self._read_msgpack_variable(
            self._mind_state, "mind_state")

    def read_language_state(self) -> dict[str, Any] | None:
        """LanguageStatePublisher payload — vocab_total, vocab_producible,
        vocab_contextual, avg_confidence, max_confidence, recent_words,
        teacher_sessions_last_hour, composition_level,
        teacher_compositions_since, teacher_last_fire_time, ts.
        Producer: language_worker."""
        return self._read_msgpack_variable(
            self._language_state, "language_state")

    def read_events_teacher_state(self) -> dict[str, Any] | None:
        """EventsTeacherStatePublisher payload — fingerprints_count,
        last_run_time, window_count, perception_buffer_size,
        follower_rotation_idx, mode_stats, felt_experiences,
        followers_tracked, windows_completed, ts.
        Producer: language_worker (events_teacher co-located)."""
        return self._read_msgpack_variable(
            self._events_teacher_state, "events_teacher_state")

    # -- Phase A.4 readers (rFP_phase_c_state_read_unification_l0_l1_canonical
    #    v1.10.0 / D-SPEC-70) -----------------------------------------------

    def read_soul_state(self) -> dict[str, Any] | None:
        """SoulStatePublisher payload — maker_pubkey, nft_address, current_gen,
        active_directives, directives_count, last_directive_ts,
        soul_initialized, schema_version, ts. Producer: sovereignty_worker."""
        return self._read_msgpack_variable(self._soul_state, "soul_state")

    def read_cgn_engine_state(self) -> dict[str, Any] | None:
        """CGNEngineStatePublisher payload — engine-level CGN stats (sibling
        to cgn_live_weights.bin tensor + cgn_beta_state.bin per-consumer
        EMA). Producer: cgn_worker."""
        return self._read_msgpack_variable(
            self._cgn_engine_state, "cgn_engine_state")

    def read_reasoning_state(self) -> dict[str, Any] | None:
        """ReasoningStatePublisher payload — ReasoningEngine.get_stats() output.
        Producer: cognitive_worker."""
        return self._read_msgpack_variable(
            self._reasoning_state, "reasoning_state")

    def read_meta_reasoning_state(self) -> dict[str, Any] | None:
        """MetaReasoningStatePublisher payload — MetaReasoningEngine.get_audit()
        output. Producer: cognitive_worker."""
        return self._read_msgpack_variable(
            self._meta_reasoning_state, "meta_reasoning_state")

    def read_consciousness_age(self) -> dict[str, Any] | None:
        """ConsciousnessAgePublisher payload — lifetime consciousness epoch
        count (Titan's "main age" — fast self-observation tick counter from
        consciousness.db). Producer: cognitive_worker (Consciousness lives
        there). Per D-SPEC-85 v1.25.0. Schema: {age_epochs, schema_version, ts}.
        """
        return self._read_msgpack_variable(
            self._consciousness_age, "consciousness_age")

    def read_consciousness_state(self) -> dict[str, Any] | None:
        """Consciousness-loop latest-epoch state — Phase 3.A D-SPEC-86 v1.26.0
        consciousness_state.bin slot. Producer: cognitive_worker (G21
        single-writer at cognitive_worker.py:3002). Schema:
        ``{epoch_count, epoch_id, density, curvature, dream_quality, fatigue,
        trajectory, trajectory_magnitude, ts}``.

        Closes ARCH-MAP-HEALTH-OBSERVABILITY Class B field 1 (2026-05-26):
        `/v4/inner-trinity.unified_spirit.epoch_count` was being served from
        the cached coordinator snapshot, which lags the live consciousness
        loop. This reader is the SPEC §10.E canonical source — the live
        loop writes per-epoch, the API now overlays the SHM value onto the
        cached snapshot.
        """
        return self._read_msgpack_variable(
            self._consciousness_state, "consciousness_state")

    def read_meta_teacher_state(self) -> dict[str, Any] | None:
        """MetaTeacherStatePublisher payload — MetaTeacherEngine.get_stats()
        output. Producer: cognitive_worker."""
        return self._read_msgpack_variable(
            self._meta_teacher_state, "meta_teacher_state")

    def read_experience_stats(self) -> dict[str, Any] | None:
        """ExperienceStatsPublisher payload — total_records, undistilled,
        total_wisdom, by_domain{domain→{count, avg_score, success_rate}},
        schema_version, ts. Producer: cognitive_worker (ExperienceOrchestrator).
        §3L Phase 15 chunk 15.1 — replaces frozen ExperienceMemory.get_stats."""
        return self._read_msgpack_variable(
            self._experience_stats, "experience_stats")

    def read_guardian_state(self) -> dict[str, Any] | None:
        """GuardianStatePublisher payload — per-module status (state, pid,
        rss_mb, uptime, layer, etc.). Producer: guardian (Python L1)."""
        return self._read_msgpack_variable(
            self._guardian_state, "guardian_state")

    def read_llm_state(self) -> dict[str, Any] | None:
        """LLMStatePublisher payload — provider, model, completion counters,
        latency, token counts. Producer: llm_worker."""
        return self._read_msgpack_variable(self._llm_state, "llm_state")

    def read_media_state(self) -> dict[str, Any] | None:
        """MediaStatePublisher payload — render counts + last render type/ts.
        Producer: studio_worker."""
        return self._read_msgpack_variable(self._media_state, "media_state")

    def read_msl_state(self) -> dict[str, Any] | None:
        """MSLStatePublisher payload — Multisensory Synthesis Layer engine
        stats. Producer: cognitive_worker."""
        return self._read_msgpack_variable(self._msl_state, "msl_state")

    def read_network_state(self) -> dict[str, Any] | None:
        """NetworkStatePublisher payload — Solana balance + pubkey + RPC
        endpoint + cached account data. Producer: titan_HCL kernel
        monitor_tick loop."""
        return self._read_msgpack_variable(
            self._network_state, "network_state")

    def read_inner_self_insight(self) -> dict[str, Any] | None:
        """Return latest SelfReasoningEngine.introspect() output + metadata,
        or None if SHM unavailable (cold-start grace period).

        rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §7.1
        + D-SPEC-70 v1.15.0 (parallel session, landed first). Slot
        `inner_self_insight.bin` is the cross-process bridge replacing the
        legacy in-process `meta_reasoning.set_self_reasoning(engine)`
        attachment that broke fleet-wide post-D8-3 spirit_worker retirement.
        Read by cognitive_worker's `_prim_introspect()` per G20 pre-warmed
        cache. Producer = self_reflection_worker (G21 single-writer) on
        each `META_INTROSPECT_REQUEST` handler completion.

        Schema v1: `{primitive, sub_mode, effective_sub_mode, confidence,
        mode_trigger, inner_avg, outer_avg, neuromods, chi_coh, epoch,
        ts, cold_start}`. Caller (meta_reasoning._prim_introspect) returns
        the dict directly; cold_start=True on placeholder writes (used to
        surface the boot-init grace period in logs).
        """
        return self._read_msgpack_variable(
            self._inner_self_insight, "inner_self_insight")

    # -- Diagnostic ----------------------------------------------------

    def availability_report(self) -> dict[str, bool]:
        """Per-registry attached/readable status — used by /v4/api-status."""
        report = {}
        for name, reader in [
            ("trinity", self._trinity),
            ("neuromod", self._neuromod),
            ("epoch", self._epoch),
            ("inner_spirit_45d", self._inner_spirit),
            ("sphere_clocks", self._sphere_clocks),
            ("chi", self._chi),
            ("titanvm_registers", self._titanvm),
            ("identity", self._identity),
            # chunk 8M.4 additions
            ("topology_30d", self._topology_30d),
            ("hormonal", self._hormonal),
            ("inner_body_5d", self._inner_body),
            ("inner_mind_15d", self._inner_mind),
            ("outer_body_5d", self._outer_body),
            ("outer_mind_15d", self._outer_mind),
            ("outer_spirit_45d", self._outer_spirit),
            # rFP_worker_broadcast_topics_completion §4.D
            ("spirit_supplemental", self._spirit_supplemental),
            ("resonance_state", self._resonance_state),
            ("unified_spirit_metadata", self._unified_spirit_metadata),
            ("resonance_metadata", self._resonance_metadata),
            ("filter_down_state", self._filter_down_state),
            # rFP_dead_dim_wiring_fix
            ("ns_program_urgencies_input", self._ns_program_urgencies_input),
            ("trajectory_state", self._trajectory_state),
            ("cgn_beta_state", self._cgn_beta_state),
            # Phase A.2 expansion (rFP_phase_c_state_read_unification_l0_l1_canonical)
            ("memory_state", self._memory_state),
            ("social_graph_state", self._social_graph_state),
            ("dream_state", self._dream_state),
            ("metabolism_state", self._metabolism_state),
            ("life_force_state", self._life_force_state),
            ("meditation_state", self._meditation_state),
            ("studio_state", self._studio_state),
            ("expression_state", self._expression_state),
            ("interface_advisor_state", self._interface_advisor_state),
            ("assessment_state", self._assessment_state),
            ("agency_state", self._agency_state),
            ("social_perception_state", self._social_perception_state),
            ("recorder_state", self._recorder_state),
            ("timechain_state", self._timechain_state),
            ("reflex_state", self._reflex_state),
            ("output_verifier_state", self._output_verifier_state),
            ("body_state", self._body_state),
            ("mind_state", self._mind_state),
            ("language_state", self._language_state),
            ("events_teacher_state", self._events_teacher_state),
            # Phase A.4
            ("soul_state", self._soul_state),
            ("cgn_engine_state", self._cgn_engine_state),
            ("reasoning_state", self._reasoning_state),
            ("meta_reasoning_state", self._meta_reasoning_state),
            ("meta_teacher_state", self._meta_teacher_state),
            ("guardian_state", self._guardian_state),
            ("llm_state", self._llm_state),
            ("media_state", self._media_state),
            ("msl_state", self._msl_state),
            ("network_state", self._network_state),
            ("inner_self_insight", self._inner_self_insight),
        ]:
            meta = reader.read_meta()
            report[name] = meta is not None
        return report
