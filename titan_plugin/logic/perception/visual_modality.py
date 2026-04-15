"""
Visual modality declaration for SensoryHub.

30D features from SpatialPerception → Trinity targets:
  Physical (5D)  → outer_body (interoception, proprioception, entropy, thermal)
  Pattern (5D)   → outer_mind Feeling[5:10] (social_temp, community_res, etc.)
  Semantic (5D)  → outer_mind Thinking[0:5] (complexity, beauty, warmth → cognitive)
  Spatial features modulate outer_body.
  Journey (5D) + Resonance (5D) available for post-processing (neuromod nudge, memory).

Index remaps preserve the original wiring from the G1+GP handler:
  physical[4] (harmony)     → outer_body[0] (interoception)
  physical[1] (edge_density) → outer_body[1] (proprioception)
  physical[3] (spatial_freq) → outer_body[3] (entropy)
  physical[0] (color_entropy)→ outer_body[4] (thermal)
  outer_body[2] (somatosensation) gets flat nudge only (creation felt)
"""
from . import SensoryModality, TargetSpec, ModulationSpec

MODALITY = SensoryModality(
    name="visual",
    msg_types={"SENSE_VISUAL"},
    payload_key="features_30d",
    feature_groups={
        "physical": 5, "pattern": 5, "spatial": 5,
        "semantic": 5, "journey": 5, "resonance": 5,
    },
    target_map={
        "physical": TargetSpec(
            target="outer_body",
            index_remap={0: 4, 1: 1, 3: 3, 4: 0},  # target_dim: source_idx
            clamp=0.05,
            strength_multiplier=1.0,
        ),
        "pattern": TargetSpec(
            target="outer_mind_feeling",
            index_remap={5: 0, 6: 1, 7: 2, 8: 3, 9: 4},  # Feeling[5:10] ← pattern[0:5]
            clamp=0.04,
            strength_multiplier=0.8,
        ),
        "semantic": TargetSpec(
            target="outer_mind_feeling",
            index_remap={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},  # Thinking[0:5] ← semantic[0:5]
            clamp=0.03,
            strength_multiplier=0.6,
        ),
    },
    modulations=[
        ModulationSpec("spatial", 2, "outer_body", 3, 0.5, 0.03),  # dispersion → entropy
        ModulationSpec("spatial", 3, "outer_body", 1, 0.5, 0.03),  # direction → proprioception
    ],
    self_strength=0.04,
    external_strength=0.025,
    creation_nudge_self=0.03,
    creation_nudge_external=0.015,
)
