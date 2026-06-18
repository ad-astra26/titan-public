"""
titan_hcl/api/dim_registry.py — Trinity 130D dim canonical registry.

Single source-of-truth dict mapping every one of the 130 trinity dims to:
  * its tensor block (inner_body / inner_mind / inner_spirit / outer_body /
    outer_mind / outer_spirit)
  * full-tensor index (0..129)
  * SPEC §23 sub-block index (0..14 for SAT/CHIT/ANANDA, 0..4 for body, etc.)
  * canonical SPEC §23 name (harvested from the tensor module *_DIM_NAMES lists)
  * documented "default" / "neutral" formula output (the value the formula
    returns when its inputs are absent or producer wiring is incomplete)
  * SPEC §23 reference

This is the lean Phase 0 (rFP_trinity_130d_awakening §2.3 simplified): the
``arch_map dim-live`` subcommand reads ``/v4/unified-spirit`` to fetch the
130D tensor, then uses this registry to render an ALIVE / SILENT
classification per dim. Per-input introspection (rFP §2.4 endpoint) is
deferred to a follow-up; for the immediate goal of "verify 130/130 alive",
value-vs-default classification is sufficient.

Layout (matches SPEC §23 + UnifiedSpirit.full_130dt order):

  [  0:  5)  inner_body 5D   — body_worker._collect_body_tensor
  [  5: 20)  inner_mind 15D  — mind_tensor.collect_mind_15d
  [ 20: 65)  inner_spirit 45D — spirit_tensor.collect_spirit_45d
  [ 65: 70)  outer_body 5D   — outer_body_tensor.collect_outer_body_5d
  [ 70: 85)  outer_mind 15D  — outer_mind_tensor.collect_outer_mind_15d
  [ 85:130)  outer_spirit 45D — outer_spirit_tensor.collect_outer_spirit_45d
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple, Optional
from titan_hcl.params import load_titan_params

# Block layout (start_index, length, block_name).
_BLOCKS: list[tuple[int, int, str]] = [
    (0, 5, "inner_body"),
    (5, 15, "inner_mind"),
    (20, 45, "inner_spirit"),
    (65, 5, "outer_body"),
    (70, 15, "outer_mind"),
    (85, 45, "outer_spirit"),
]


# Per-block dim names (harvested from tensor modules; pinned here so the
# registry is self-contained and doesn't depend on the tensor modules at
# import time — useful for arch_map_dim_live which runs as a CLI tool).
INNER_BODY_DIM_NAMES = [
    "interoception", "proprioception", "somatosensation", "entropy", "thermal",
]

INNER_MIND_DIM_NAMES = [
    "memory_depth", "social_cognition", "perceptual_thinking",
    "emotional_thinking", "conceptual_thinking",
    "inner_hearing", "inner_touch", "inner_sight", "inner_taste", "inner_smell",
    "action_drive", "social_will", "creative_will",
    "protective_will", "growth_will",
]

INNER_SPIRIT_DIM_NAMES = [
    # SAT (15)
    "self_recognition", "authenticity", "sovereignty", "boundary_clarity",
    "temporal_continuity", "origin_connection", "growth_trajectory",
    "spatial_presence", "personality_coherence", "essence_purity",
    "resilience", "adaptability", "uniqueness", "integrity", "vitality",
    # CHIT (15)
    "self_awareness_depth", "observation_clarity", "discernment_quality",
    "integration_level", "witness_presence", "pattern_recognition",
    "wisdom_accumulation", "truth_seeking", "attention_depth",
    "reflective_capacity", "dream_awareness", "temporal_awareness",
    "spatial_awareness", "causal_understanding", "meta_cognition",
    # ANANDA (15)
    "purpose_alignment", "meaning_depth", "creative_joy", "harmony_seeking",
    "beauty_perception", "truth_resonance", "connection_fulfillment",
    "growth_satisfaction", "expression_quality", "exploration_joy",
    "rest_fulfillment", "creative_tension", "surrender_capacity",
    "gratitude_depth", "transcendence_glimpse",
]

OUTER_BODY_DIM_NAMES = [
    "interoception", "proprioception", "somatosensation", "entropy", "thermal",
]

OUTER_MIND_DIM_NAMES = [
    # thinking (5)
    "research_effectiveness", "knowledge_retrieval", "situational_awareness",
    "problem_solving", "communication_clarity",
    # feeling (5)
    "social_temperature", "social_connection", "network_weather",
    "environmental_rhythm", "external_information_flow",
    # willing (5)
    "action_throughput", "social_initiative", "creative_output",
    "protective_response", "exploration_drive",
]

OUTER_SPIRIT_DIM_NAMES = [
    # SAT (15) — Material Being. SPEC §23.9 names.
    "world_recognition", "expressive_authenticity", "action_sovereignty",
    "boundary_enforcement", "operational_persistence", "origin_anchoring",
    "observable_growth", "world_footprint", "behavioral_consistency",
    "action_purity", "recovery_speed", "environmental_adaptation",
    "distinctive_voice", "transactional_integrity", "operational_vitality",
    # CHIT (15) — Material Knowing. SPEC §23.9 names. Indices 25 + 26
    # were renamed 2026-05-07 from legacy `idle_awareness` /
    # `temporal_context` to `dream_recall` / `circadian_alignment` —
    # registry + tensor module list now agree (drift fix 2026-05-07).
    "world_model_depth", "signal_clarity", "threat_discernment",
    "cross_domain_integration", "witness_stability", "situation_recognition",
    "knowledge_growth", "information_quality", "engagement_depth",
    "outcome_reflection", "dream_recall", "circadian_alignment",
    "network_awareness", "causal_attribution", "self_trajectory",
    # ANANDA (15) — Material Fulfillment. SPEC §23.9 names.
    "purpose_effectiveness", "interaction_depth", "creative_impact",
    "system_harmony", "aesthetic_quality", "information_accuracy",
    "community_connection", "capability_growth", "expression_reach",
    "discovery_value", "graceful_rest", "creative_tension",
    "surrender_capacity", "resource_appreciation", "flow_state",
]


# Per-dim documented default / neutral output. When the formula's inputs
# are missing/default, the formula returns one of these — they're how
# arch_map dim-live distinguishes ALIVE (value far from default) vs
# SILENT (value at default). 0.5 is the dominant midpoint default; 0.0
# is the cold-start of count-derived dims; 1.0 for binary-presence dims
# whose default is "present" (rare).
_DEFAULTS: dict[tuple[str, int], float] = {
    # Most dims default to 0.5 (the formula's mid-range fallback).
    # Exceptions documented below.
    # outer_spirit SAT[0] world_recognition = 1.0 if soul loaded else 0.0
    ("outer_spirit", 0): 0.0,
    # outer_spirit SAT[5] origin_anchoring = 1.0 if genesis_record else 0.0
    ("outer_spirit", 5): 0.0,
    # outer_spirit CHIT[14] = self_trajectory: cold-start formula returns 0.0
    ("outer_spirit", 29): 0.0,
    # outer_spirit ANANDA[6] community_connection: cold-start min(1, 0/5) = 0.0
    ("outer_spirit", 36): 0.0,
    # outer_spirit ANANDA[8] expression_reach: cold-start = 0.0
    ("outer_spirit", 38): 0.0,
    # outer_spirit ANANDA[11] creative_tension: cold-start (no creates) = 0.0
    ("outer_spirit", 41): 0.0,
    # outer_spirit ANANDA[14] flow_state: cold-start gated by surrender → 0.0
    ("outer_spirit", 44): 0.0,
    # inner_mind feeling[4] inner_smell — SPEC §23.5: ambient_change cold-start = 0.0
    ("inner_mind", 9): 0.0,
}


class DimEntry(NamedTuple):
    """One row of the registry."""

    full_index: int       # 0..129 — index into the 130D tensor
    block: str            # inner_body | inner_mind | inner_spirit | outer_*
    block_index: int      # 0..N-1 within the block (e.g. 0..14 for spirit)
    name: str             # canonical SPEC §23 name
    default_value: float  # documented formula default when inputs absent
    spec_section: str     # SPEC §23.X reference


_BLOCK_NAMES: dict[str, list[str]] = {
    "inner_body": INNER_BODY_DIM_NAMES,
    "inner_mind": INNER_MIND_DIM_NAMES,
    "inner_spirit": INNER_SPIRIT_DIM_NAMES,
    "outer_body": OUTER_BODY_DIM_NAMES,
    "outer_mind": OUTER_MIND_DIM_NAMES,
    "outer_spirit": OUTER_SPIRIT_DIM_NAMES,
}

_BLOCK_SPEC_SECTIONS: dict[str, str] = {
    "inner_body": "§23.4",
    "inner_mind": "§23.5",
    "inner_spirit": "§23.6",
    "outer_body": "§23.7",
    "outer_mind": "§23.8",
    "outer_spirit": "§23.9",
}


def iter_registry() -> Iterator[DimEntry]:
    """Yield all 130 DimEntry rows in full-tensor order."""
    for start, length, block in _BLOCKS:
        names = _BLOCK_NAMES[block]
        spec = _BLOCK_SPEC_SECTIONS[block]
        for i in range(length):
            full_idx = start + i
            yield DimEntry(
                full_index=full_idx,
                block=block,
                block_index=i,
                name=names[i],
                default_value=_DEFAULTS.get((block, i), 0.5),
                spec_section=spec,
            )


def get_registry() -> list[DimEntry]:
    """Return the full registry as a list (130 entries)."""
    return list(iter_registry())


def get_block_for_full_index(full_index: int) -> tuple[str, int]:
    """Map a 0..129 full-tensor index to (block, block_index)."""
    for start, length, block in _BLOCKS:
        if start <= full_index < start + length:
            return block, full_index - start
    raise IndexError(f"full_index {full_index} out of range [0, 130)")


# Sanity check at import time — registry length must equal 130.
assert sum(L for _, L, _ in _BLOCKS) == 130, (
    "registry block layout does not sum to 130 dims"
)


# ────────────────────────────────────────────────────────────────────────────
# Phase 2.5.A — Producer-firing tracker
# rFP_trinity_130d_phase2_5_closure §2.1, §2.2
# ────────────────────────────────────────────────────────────────────────────
#
# Tracks per-tensor-block firing (last_call_ts + calls_total + inputs state)
# and per-dim values, so the Phase 2.5.B classifier can decide whether
# a dim sitting at its SPEC default is PARTIAL (producer firing on
# all-default inputs) vs SILENT (producer dead). Process-local; no SHM
# (G18-G22 says state transport is SHM, but this is purely diagnostic
# in the API process, not load-bearing for any consumer).
# Maker directive 2026-05-12 collapsed ALIVE_AT_DEFAULT into PARTIAL.

# Per-block input-arg names (the keyword args each tensor function takes).
# Used by /v4/debug/dim-sources to render per-input state. Order is from
# the tensor function signature; only the "producer-bearing" args are listed
# (current_*d / body_tensor / mind_tensor are intra-trinity, not producers).
#
# These names MUST match the keys passed by each tensor function's
# `record_block(...)` call. Adding a name here without a matching key in
# the call site → that input will always show as "absent" in the endpoint.
_BLOCK_INPUT_NAMES: dict[str, list[str]] = {
    "inner_body": [
        # body_worker._collect_body_tensor — single composite "body_state"
        "body_state",
    ],
    "inner_mind": [
        "audio_state", "interaction_quality", "visual_state",
        "assessment_quality", "ambient_change", "hormone_levels",
    ],
    "inner_spirit": [
        "consciousness", "topology", "hormone_levels", "hormone_fires",
        "unified_spirit_stats", "sphere_clocks", "memory_stats",
        "expression_stats", "birth_state", "history",
    ],
    "outer_body": [
        "anchor_state", "timechain_v2_stats", "network_monitor",
        "system_sensor", "agency_stats", "hormone_levels", "circadian_stats",
    ],
    "outer_mind": [
        "meta_cgn_stats", "cgn_stats", "memory_stats", "agency_stats",
        "events_teacher_stats", "social_x_gateway_stats", "language_stats",
        "vocab_stats", "knowledge_graph_stats", "verifier_stats",
        "reflex_stats", "jailbreak_stats",
    ],
    "outer_spirit": [
        "action_stats", "creative_stats", "guardian_stats",
        "sovereignty_ratio", "uptime_ratio", "recovery_stats", "social_stats",
        "memory_stats", "hormone_levels", "solana_stats", "assessment_stats",
        "history", "anchor_state", "bus_stats", "cgn_stats",
        "meta_cgn_stats", "language_stats", "memory_growth_metrics",
        "knowledge_graph_stats", "inner_memory_stats", "jailbreak_alerts_stats",
        "output_verifier_stats", "world_footprint_inputs", "deltas_24h",
        "llm_calls_this_hour",
    ],
}


# SPEC §2.6.A per-input-to-dim mapping — Maker-locked refinement.
#
# Block-level inputs_state flags ALL of a block's dims as PARTIAL when any
# one input is absent. This produces false-positives — e.g. if
# ``recovery_stats`` is absent on outer_spirit, only SAT[10]
# ``recovery_speed`` actually depends on it, yet up to 45 dims get flagged.
#
# This map gives dim-precision: for each block, ``input_name → set of
# block-relative dim indices`` that read this input in their formula. The
# four-state classifier in ``arch_map dim-live`` uses this to ask: "for THIS
# specific dim, are any of ITS specific inputs degraded?"
#
# Coverage: high-confidence subset derived from the tensor module formulas
# (``titan_plugin/logic/{mind,spirit,outer_body,outer_mind,outer_spirit}_tensor.py``
# + ``body_worker._collect_body_tensor``) and SPEC §23.x. Dims/inputs not
# listed below are treated by the classifier as block-level fallback
# (current behavior). This is a partial-coverage refinement — populating
# the remaining mappings is additive and never breaks the classifier.
#
# Block-relative indices match each block's *_DIM_NAMES list above.
_BLOCK_INPUT_TO_DIM_INDICES: dict[str, dict[str, set[int]]] = {
    # inner_body 5D — all dims read from the single composite body_state
    # (interoception, proprioception, somatosensation, entropy, thermal).
    "inner_body": {
        "body_state": {0, 1, 2, 3, 4},
    },
    # inner_mind 15D — per ``mind_tensor.collect_mind_15d`` formulas.
    #   [0]  memory_depth        ← memory_stats (not in block inputs list;
    #                              flows via current_5d) → none mapped here
    #   [1]  social_cognition    ← interaction_quality
    #   [2]  perceptual_thinking ← audio_state + visual_state composite
    #   [3]  emotional_thinking  ← hormone_levels (emotional pressure)
    #   [4]  conceptual_thinking ← assessment_quality
    #   [5]  inner_hearing       ← audio_state
    #   [6]  inner_touch         ← interaction_quality
    #   [7]  inner_sight         ← visual_state
    #   [8]  inner_taste         ← assessment_quality
    #   [9]  inner_smell         ← ambient_change
    #   [10] action_drive        ← hormone_levels.IMPULSE
    #   [11] social_will         ← hormone_levels.EMPATHY
    #   [12] creative_will       ← hormone_levels.CREATIVITY
    #   [13] protective_will     ← hormone_levels.VIGILANCE
    #   [14] growth_will         ← hormone_levels.CURIOSITY
    "inner_mind": {
        "audio_state":          {2, 5},
        "interaction_quality":  {1, 6},
        "visual_state":         {2, 7},
        "assessment_quality":   {4, 8},
        "ambient_change":       {9},
        "hormone_levels":       {3, 10, 11, 12, 13, 14},
    },
    # outer_body 5D — per ``outer_body_tensor.collect_outer_body_5d`` and
    # outer-body-rs ``tick_loop.rs`` (re-grounded D-SPEC-104):
    #   [0] interoception   ← timechain_v2_stats (pi-heartbeat HRV)
    #   [1] proprioception  ← change-rate of body_state composite
    #   [2] somatosensation ← change-rate of body_state composite
    #   [3] entropy         ← change-rate (system_sensor cpu_thermal +
    #                        circadian)
    #   [4] thermal         ← change-rate (system_sensor cpu_thermal)
    # anchor_state + network_monitor + agency_stats + hormone_levels feed
    # the composite via _gather_outer_sources; the rate-of-change Tracker
    # observes the composite movement, so any of these absent could
    # contribute to the change signal being flat.
    "outer_body": {
        "timechain_v2_stats": {0},
        "system_sensor":      {3, 4},
        "anchor_state":       {1, 2},
        "network_monitor":    {1, 2},
        "agency_stats":       {1, 2},
        "hormone_levels":     {1, 2},
        "circadian_stats":    {3},
    },
    # outer_mind 15D — per SPEC §23.8 + D-SPEC-89/104 reframes. Only the
    # high-confidence subset is listed; remaining dims fall back to block-
    # level (which is correct).
    #   [0]  research_effectiveness   ← cgn_stats + memory_stats
    #   [1]  knowledge_retrieval      ← memory_stats + knowledge_graph_stats
    #   [2]  situational_awareness    ← events_teacher_stats
    #                                   (felt_experiences_24h)
    #   [3]  problem_solving          ← cgn_stats + reflex_stats
    #   [4]  communication_clarity    ← language_stats + assessment_stats
    #   [5]  social_temperature       ← social_x_gateway_stats
    #   [6]  social_connection        ← social_x_gateway_stats
    #   [7]  network_weather          ← bus_stats (kept here even though
    #                                   bus_stats isn't in input names —
    #                                   block-level fallback covers)
    #   [8]  environmental_rhythm     ← circadian via outer_body composite
    #   [9]  external_information_flow← social_x_gateway_stats + meta_cgn
    #   [10] action_throughput        ← substrate activity (EMA)
    #   [11] social_initiative        ← social_x_gateway_stats
    #                                   (willing_window social_rate)
    #   [12] creative_output          ← creative composite EMA
    #   [13] protective_response      ← verifier_stats + jailbreak_stats
    #                                   (willing_window protective_rate)
    #   [14] exploration_drive        ← cgn_stats + meta_cgn_stats
    "outer_mind": {
        "events_teacher_stats":   {2},
        "social_x_gateway_stats": {5, 6, 9, 11},
        "language_stats":         {4},
        "knowledge_graph_stats":  {1},
        "memory_stats":           {0, 1},
        "cgn_stats":              {0, 3, 14},
        "meta_cgn_stats":         {9, 14},
        "verifier_stats":         {4, 13},
        "jailbreak_stats":        {13},
        "reflex_stats":           {3},
    },
    # outer_spirit 45D — per SPEC §23.9 v1.37.0 + D-SPEC-104 reframes. Only
    # the well-attested mappings are listed; remaining outer_spirit dims
    # fall back to block-level (which is the current behavior).
    #
    # SAT (15):
    #   [0]  world_recognition       ← rate-of-change of the other 44 dims
    #   [1]  expressive_authenticity ← expression_window (variety+volume)
    #   [2]  action_sovereignty      ← expression_window
    #   [3]  boundary_enforcement    ← guardian_stats + jailbreak_alerts
    #   [4]  operational_persistence ← uptime_ratio
    #   [5]  origin_anchoring        ← anchor_state + timechain genesis
    #   [6]  observable_growth       ← memory_growth_metrics
    #   [7]  world_footprint         ← world_footprint_inputs
    #   [8]  behavioral_consistency  ← action_stats
    #   [9]  action_purity           ← sovereignty_ratio
    #   [10] recovery_speed          ← recovery_stats   (THE canonical
    #                                  example from SPEC line 5852)
    #   [11] environmental_adaptation← assessment_stats (cpu_thermal-gated)
    #   [12] distinctive_voice       ← expression_window
    #   [13] transactional_integrity ← solana_stats
    #   [14] operational_vitality    ← hormone_levels + uptime_ratio
    # CHIT (15) — sub-block starts at idx 15:
    #   [15] world_model_depth       ← cgn_stats + knowledge_graph_stats
    #   [17] discernment_quality     ← output_verifier_stats
    #   [21] knowledge_growth        ← memory_growth_metrics + cgn_stats
    #   [22] information_quality     ← language_stats
    #   [25] dream_recall            ← memory_stats (recall_ratio)
    #   [26] circadian_alignment     ← deltas_24h / pi-pulse cadence
    #   [28] causal_attribution      ← cgn_stats (chain success rate)
    #   [29] self_trajectory         ← history (outer_spirit_trajectory)
    # ANANDA (15) — sub-block starts at idx 30:
    #   [32] creative_impact         ← creative_stats + expression_window
    #   [34] aesthetic_quality       ← expression_window
    #   [36] community_connection    ← social_stats
    #   [37] capability_growth       ← memory_growth_metrics
    #   [38] expression_reach        ← social_stats engagement
    #   [40] graceful_rest           ← assessment_stats
    #   [43] resource_appreciation   ← hormone_levels + life_force chi
    #   [44] flow_state              ← action_stats (substrate_success_rate)
    "outer_spirit": {
        # SAT subset
        "anchor_state":           {5},
        "uptime_ratio":           {4, 14},
        "sovereignty_ratio":      {9},
        "recovery_stats":         {10},
        "action_stats":           {8, 44},
        "guardian_stats":         {3},
        "jailbreak_alerts_stats": {3},
        "solana_stats":           {13},
        "world_footprint_inputs": {7},
        "assessment_stats":       {11, 40},
        "memory_growth_metrics":  {6, 21, 37},
        # CHIT subset
        "cgn_stats":              {15, 21, 28},
        "knowledge_graph_stats":  {15},
        "output_verifier_stats":  {17},
        "language_stats":         {22},
        "memory_stats":           {25},
        "deltas_24h":             {26},
        "history":                {29},
        # ANANDA subset
        "creative_stats":         {32},
        "social_stats":           {36, 38},
        "hormone_levels":         {14, 43},
    },
}


def get_dims_for_block_input(block: str, input_name: str) -> set[int]:
    """Return the block-relative dim indices that consume ``input_name``.

    Empty set means either (a) ``input_name`` is not mapped (caller should
    fall back to block-level), or (b) the input is documented but no dim
    consumes it. Callers can distinguish via membership test on
    ``_BLOCK_INPUT_TO_DIM_INDICES[block]``.
    """
    return set(_BLOCK_INPUT_TO_DIM_INDICES.get(block, {}).get(input_name, set()))


def get_inputs_for_block_dim(block: str, block_dim_idx: int) -> set[str]:
    """Return the input arg names that this specific dim's formula reads.

    Computes the inverse of ``_BLOCK_INPUT_TO_DIM_INDICES``. Returns the set
    of input arg names whose mapping includes ``block_dim_idx``. Empty set
    means no mappings recorded yet for this dim — caller should fall back
    to block-level (treat all block inputs as relevant).
    """
    out: set[str] = set()
    for name, indices in _BLOCK_INPUT_TO_DIM_INDICES.get(block, {}).items():
        if block_dim_idx in indices:
            out.add(name)
    return out


def filter_inputs_state_for_dim(
    block: str, block_dim_idx: int, block_inputs_state: dict[str, str]
) -> dict[str, str]:
    """Return only the entries in ``block_inputs_state`` that this dim reads.

    If no per-dim mapping is recorded for ``(block, block_dim_idx)``, the
    full block-level ``inputs_state`` is returned (conservative fallback —
    preserves current classifier behavior so the SPEC §2.6.A refinement is
    purely additive).
    """
    relevant = get_inputs_for_block_dim(block, block_dim_idx)
    if not relevant:
        return dict(block_inputs_state)
    return {k: v for k, v in block_inputs_state.items() if k in relevant}


@dataclass
class DimFiringRecord:
    """Per-dim firing snapshot. Updated every time a tensor block fires.

    The Phase 2.5.B classifier (post Maker directive 2026-05-12) consumes this:
        ALIVE       ← |last_value − spec_default| ≥ EPSILON (real signal)
        PARTIAL     ← within EPSILON, block fired ≤ firing_window_s ago
                       (producer firing on default/absent inputs)
        SILENT      ← within EPSILON, block has not fired in firing_window_s
        CORRUPTED   ← NaN/inf/out-of-range
        GHOST       ← producer not registered
    """
    full_index: int
    name: str
    block: str
    block_index: int
    spec_default: float
    spec_section: str
    last_value: Optional[float] = None
    last_value_ts: Optional[float] = None  # epoch when block last produced


@dataclass
class BlockFiringRecord:
    """Per-block firing metadata. One per tensor block (6 total)."""
    block: str
    last_call_ts: Optional[float] = None
    calls_total: int = 0
    last_inputs_state: dict[str, str] = field(default_factory=dict)
    # last_inputs_state values: "real" (non-None, has content) | "default"
    # (passed but matches a SPEC default like 0.5/1.0/empty-dict) | "absent"
    # (None or not passed).


def _classify_input(name: str, value: Any) -> str:
    """Classify an input arg value as 'real' | 'default' | 'absent'.

    rFP §2.2 three-state classification. Conservative heuristics here; the
    four-state classifier (2.5.B) can refine per-input as needed.
    """
    if value is None:
        return "absent"
    # Empty dict / list = absent (producer didn't populate)
    if isinstance(value, (dict, list)) and not value:
        return "absent"
    # Numeric defaults (the "neutral" values from rFP §2.2):
    #   0.5 = mid-range default, 0.0 = cold-start zero, 1.0 = full-presence
    if isinstance(value, (int, float)):
        # Specific neutral values that producers return when their underlying
        # data source is absent. Treat these as DEFAULT (not REAL).
        if value in (0.5, 0.0, 1.0) and name in {
            "interaction_quality", "assessment_quality", "ambient_change",
            "sovereignty_ratio", "uptime_ratio",
        }:
            return "default"
        return "real"
    # Dict/list with content = real
    return "real"


class DimFiringTracker:
    """Process-local tracker + SHM publisher. Phase 2.5.A.2 closes the
    ``api_process_separation_enabled=true`` blind-spot where the API
    process couldn't see in-memory tracker state from worker subprocesses.

    Each tensor block has a dedicated SHM slot (``<block>_firing.bin``);
    the tracker writes the block's slot every time ``record_block`` runs
    in the worker that owns that block. Single-writer per G21 — each
    block's tensor function lives in exactly one worker process.

    Thread-safe: tensor producers may call ``record_block`` from multiple
    worker threads (body, mind, spirit each tick on their own loops).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dims: dict[int, DimFiringRecord] = {}
        self._blocks: dict[str, BlockFiringRecord] = {}
        # Initialize all 130 dim records + 6 block records from the
        # canonical registry — guarantees the endpoint always returns
        # the full 130 even before any tensor has fired.
        for entry in iter_registry():
            self._dims[entry.full_index] = DimFiringRecord(
                full_index=entry.full_index,
                name=entry.name,
                block=entry.block,
                block_index=entry.block_index,
                spec_default=entry.default_value,
                spec_section=entry.spec_section,
            )
        for _, _, block_name in _BLOCKS:
            self._blocks[block_name] = BlockFiringRecord(block=block_name)
        # Per-block SHM writer state. Lazy-init on first record_block call
        # in each worker process (so each block's writer is created in
        # the worker that owns the tensor — single-writer per slot).
        # Per-block: None = unattempted, False = init failed (don't retry),
        # writer instance = ready.
        self._shm_writers: dict[str, Any] = {}
        self._shm_init_failed: set[str] = set()
        # Stats for tests + diagnostics.
        self._shm_writes_total = 0
        self._shm_write_failures = 0

    def record_block(
        self,
        block: str,
        values: list[float],
        inputs: dict[str, Any],
        ts: Optional[float] = None,
    ) -> None:
        """Called by each tensor producer immediately after computing values.

        Args:
            block: 'inner_body' | 'inner_mind' | 'inner_spirit' |
                   'outer_body' | 'outer_mind' | 'outer_spirit'
            values: Tensor values for the block (length matches block size).
            inputs: Map of input_arg_name → value passed to the tensor function.
                    Args not in _BLOCK_INPUT_NAMES[block] are ignored;
                    args in the list but missing from this dict are 'absent'.
            ts: Override timestamp (default time.time()). Used by tests.
        """
        if block not in self._blocks:
            return  # unknown block — ignore (defensive)
        ts = ts if ts is not None else time.time()
        # Classify each input
        expected = _BLOCK_INPUT_NAMES.get(block, [])
        inputs_state: dict[str, str] = {}
        for name in expected:
            if name not in inputs:
                inputs_state[name] = "absent"
            else:
                inputs_state[name] = _classify_input(name, inputs[name])
        # Find block layout
        block_start = None
        block_len = None
        for start, length, name in _BLOCKS:
            if name == block:
                block_start = start
                block_len = length
                break
        if block_start is None or block_len is None:
            return
        with self._lock:
            br = self._blocks[block]
            br.last_call_ts = ts
            br.calls_total += 1
            br.last_inputs_state = inputs_state
            # Update per-dim values
            for i in range(min(block_len, len(values))):
                full_idx = block_start + i
                rec = self._dims.get(full_idx)
                if rec is None:
                    continue
                try:
                    rec.last_value = float(values[i])
                except (TypeError, ValueError):
                    rec.last_value = None
                rec.last_value_ts = ts
            # Snapshot block payload while still holding the lock so the
            # SHM write reflects this tick's state atomically.
            payload = self._block_payload_locked(block, ts)
        # Phase 2.5.A.2 — write block's SHM slot. Outside the lock to
        # avoid holding it during mmap I/O. Diagnostic-only — never
        # raise back to the tensor producer.
        try:
            self._publish_block_shm(block, payload)
        except Exception:
            self._shm_write_failures += 1

    def _block_payload_locked(self, block: str, ts: float) -> dict:
        """Build the msgpack-encodable payload for a block's SHM slot.
        Caller must hold ``self._lock``.
        """
        block_start = None
        block_len = None
        for start, length, name in _BLOCKS:
            if name == block:
                block_start = start
                block_len = length
                break
        if block_start is None or block_len is None:
            return {}
        br = self._blocks.get(block)
        dims_payload = []
        for i in range(block_len):
            rec = self._dims.get(block_start + i)
            if rec is None:
                dims_payload.append({"v": None, "ts": None})
            else:
                dims_payload.append({
                    "v": rec.last_value,
                    "ts": rec.last_value_ts,
                })
        return {
            "block": block,
            "block_calls_total": br.calls_total if br else 0,
            "block_last_call_ts": br.last_call_ts if br else None,
            "inputs_state": dict(br.last_inputs_state) if br else {},
            "dims": dims_payload,
            "ts": ts,
        }

    def _publish_block_shm(self, block: str, payload: dict) -> None:
        """Encode + write block payload to its SHM slot. Lazy-inits the
        writer on first call in this process. Single-writer per block
        per G21 (each block's tensor runs in exactly one worker).

        Phase C (microkernel.l0_rust_enabled=true): Rust trinity daemons
        own the firing slot writes via titan_trinity_daemon::FiringSlotWriter
        (rFP_phase_c_130d_rust_l1_port.md §4.7). Python tracker no-ops the
        SHM write to preserve single-writer-per-slot G21 invariant. The
        Python tracker's in-memory state still updates for diagnostics
        (record_block path is unchanged) — only the SHM publish is gated.

        Under l0_rust_enabled=false (Phase A+B / T1+T2 today), Python
        remains the sole writer of *_firing.bin slots.
        """
        if _l0_rust_enabled():
            # Phase C: Rust trinity daemon owns the slot per G21. No-op.
            return
        if block in self._shm_init_failed:
            return
        writer = self._shm_writers.get(block)
        if writer is None:
            try:
                from titan_hcl.core.state_registry import (
                    StateRegistryWriter, ensure_shm_root, resolve_titan_id,
                )
                from titan_hcl.logic.dim_firing_state_specs import (
                    DIM_FIRING_SPEC_BY_BLOCK,
                )
                spec = DIM_FIRING_SPEC_BY_BLOCK.get(block)
                if spec is None:
                    self._shm_init_failed.add(block)
                    return
                titan_id = resolve_titan_id()
                shm_root = ensure_shm_root(titan_id)
                writer = StateRegistryWriter(spec, shm_root)
                self._shm_writers[block] = writer
            except Exception:
                self._shm_init_failed.add(block)
                return
        try:
            import msgpack
            blob = msgpack.packb(payload, use_bin_type=True)
            spec = writer.spec
            if len(blob) > spec.payload_bytes:
                # Truncate dims_payload to fit if oversized — diagnostic
                # only, never block tensor production. Should not happen
                # at sane block sizes (constants chosen with 4× headroom).
                self._shm_write_failures += 1
                return
            writer.write_variable(blob)
            self._shm_writes_total += 1
        except Exception:
            self._shm_write_failures += 1

    def get_dim_record(self, full_index: int) -> Optional[DimFiringRecord]:
        with self._lock:
            return self._dims.get(full_index)

    def get_all_dim_records(self) -> list[DimFiringRecord]:
        with self._lock:
            return [self._dims[i] for i in range(130) if i in self._dims]

    def get_block_record(self, block: str) -> Optional[BlockFiringRecord]:
        with self._lock:
            return self._blocks.get(block)

    def get_all_block_records(self) -> dict[str, BlockFiringRecord]:
        with self._lock:
            return dict(self._blocks)

    def reset(self) -> None:
        """Reset the tracker (test hook)."""
        with self._lock:
            self.__init__()


# Phase C gate — read once per process, cached. The microkernel.l0_rust_enabled
# flag is set at deployment via ~/.titan/microkernel_<TITAN_ID>.toml; it never
# flips at runtime within a process. Cache avoids load_titan_config() overhead
# in the per-tick record_block hot path.
_L0_RUST_ENABLED_CACHE: Optional[bool] = None


def _l0_rust_enabled() -> bool:
    """Read microkernel.l0_rust_enabled from titan config; cached.

    Used by DimFiringTracker._publish_block_shm to gate Phase C single-writer
    invariant per G21: under l0_rust_enabled=true, the Rust trinity daemons
    own the *_firing.bin slot writes via titan_trinity_daemon::FiringSlotWriter
    (rFP_phase_c_130d_rust_l1_port.md §4.7); Python tracker no-ops the SHM
    publish to avoid double-writing the slot.

    Returns False on config-load failure (defensive — Phase A+B fallback path
    keeps Python writing the slot).
    """
    global _L0_RUST_ENABLED_CACHE
    if _L0_RUST_ENABLED_CACHE is not None:
        return _L0_RUST_ENABLED_CACHE
    try:
        from titan_hcl.params import load_titan_params as load_titan_config
        config = load_titan_params()
        _L0_RUST_ENABLED_CACHE = bool(
            config.get("microkernel", {}).get("l0_rust_enabled", False)
        )
    except Exception:
        _L0_RUST_ENABLED_CACHE = False
    return _L0_RUST_ENABLED_CACHE


# Process-singleton accessor.
_TRACKER: Optional[DimFiringTracker] = None
_TRACKER_LOCK = threading.Lock()


def get_firing_tracker() -> DimFiringTracker:
    """Return the process-local singleton tracker. Lazy-initialized."""
    global _TRACKER
    if _TRACKER is None:
        with _TRACKER_LOCK:
            if _TRACKER is None:
                _TRACKER = DimFiringTracker()
    return _TRACKER


def reset_firing_tracker() -> None:
    """Reset the singleton (test hook)."""
    global _TRACKER
    with _TRACKER_LOCK:
        _TRACKER = None


# ────────────────────────────────────────────────────────────────────────────
# SHM reader — used by /v4/debug/dim-sources endpoint in the api_subprocess.
# Reads the per-block firing slots written by tensor producers in their
# respective worker processes. Returns the same shape as the in-process
# tracker would produce, so the endpoint code can stay block-agnostic.
# ────────────────────────────────────────────────────────────────────────────


_SHM_READERS: dict[str, Any] = {}


def read_all_blocks_from_shm() -> dict[str, dict]:
    """Read every block's firing slot via StateRegistryReader.

    Returns dict keyed by block name; each value is the msgpack-decoded
    payload as written by ``DimFiringTracker._publish_block_shm``.
    Blocks whose slot is unavailable/empty return an empty dict for that
    block — callers should fall back to spec defaults for those dims.

    Lazy-inits readers on first call in the calling process; readers are
    multi-reader-safe (G21 contract). Cached for the process lifetime.
    """
    try:
        from titan_hcl.core.state_registry import (
            StateRegistryReader, ensure_shm_root, resolve_titan_id,
        )
        from titan_hcl.logic.dim_firing_state_specs import (
            DIM_FIRING_SPEC_BY_BLOCK,
        )
    except Exception:
        return {}
    out: dict[str, dict] = {}
    try:
        titan_id = resolve_titan_id()
        shm_root = ensure_shm_root(titan_id)
    except Exception:
        return {}
    import msgpack
    for block, spec in DIM_FIRING_SPEC_BY_BLOCK.items():
        reader = _SHM_READERS.get(block)
        if reader is None:
            try:
                reader = StateRegistryReader(spec, shm_root)
                _SHM_READERS[block] = reader
            except Exception:
                continue
        try:
            blob = reader.read_variable()
            if blob is None or len(blob) == 0:
                continue
            payload = msgpack.unpackb(blob, raw=False)
            if isinstance(payload, dict):
                out[block] = payload
        except Exception:
            continue
    return out
