"""
Audio modality declaration for SensoryHub.

15D features from AudioPerception → outer_body (Physical) + oMind Feeling (Pattern).
Temporal features modulate outer_body.

Index remaps preserve the original wiring from the 15D handler:
  physical[4] (harmony)           → outer_body[0] (interoception)
  physical[1] (harmonic_ratio)    → outer_body[1] (proprioception)
  physical[2] (rhythmic_regularity)→ outer_body[3] (entropy)  ← NOTE: [2] not [3]
  physical[0] (spectral_centroid) → outer_body[4] (thermal)
  outer_body[2] (somatosensation) gets flat nudge only
"""
from . import SensoryModality, TargetSpec, ModulationSpec

MODALITY = SensoryModality(
    name="audio",
    msg_types={"SENSE_AUDIO"},
    payload_key="features_15d",
    feature_groups={
        "physical": 5, "pattern": 5, "temporal": 5,
    },
    target_map={
        "physical": TargetSpec(
            target="outer_body",
            index_remap={0: 4, 1: 1, 3: 2, 4: 0},  # NOTE: dim3 ← source[2] (rhythm, not [3])
            clamp=0.05,
            strength_multiplier=1.0,
        ),
        "pattern": TargetSpec(
            target="outer_mind_feeling",
            index_remap={5: 0, 6: 1, 7: 2, 8: 3, 9: 4},  # Feeling[5:10] ← pattern[0:5]
            clamp=0.04,
            strength_multiplier=0.8,
        ),
    },
    modulations=[
        ModulationSpec("temporal", 3, "outer_body", 3, 0.5, 0.03),  # spectral_flux → entropy
        ModulationSpec("temporal", 0, "outer_body", 1, 0.5, 0.03),  # attack → proprioception
    ],
    self_strength=0.04,
    external_strength=0.025,
    creation_nudge_self=0.03,
    creation_nudge_external=0.015,
)
