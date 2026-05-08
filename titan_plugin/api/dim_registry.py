"""
titan_plugin/api/dim_registry.py — Trinity 130D dim canonical registry.

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

from typing import Iterator, NamedTuple

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
