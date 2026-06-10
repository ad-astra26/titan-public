"""rFP_x_voice_enrichment Phase 1 — 9 archetypes.

Each archetype subclasses ``ArchetypeBase`` and implements ``find_candidate``.
The gateway dispatcher constructs all 9 once and probes them in priority
order each post-attempt; the first one whose ``find_candidate`` returns a
non-None ``ArchetypeCandidate`` wins (with cross-archetype 4 h spacing
enforced inside ``ArchetypeBase``).
"""
from .base import ArchetypeBase, ArchetypeCandidate
from .proof_day import ProofDayArchetype, PROOF_DAY_POST_TYPE
from .soul_diary import SoulDiaryArchetype, SOUL_DIARY_POST_TYPE
from .world_mirror import WorldMirrorArchetype, WORLD_MIRROR_POST_TYPE
from .outer_rumination import (
    OuterRuminationArchetype, OUTER_RUMINATION_POST_TYPE,
)
from .outer_inner_bridge import OuterInnerBridgeArchetype, OINB_POST_TYPE
from .grounded_today import (
    GroundedTodayArchetype, GROUNDED_TODAY_POST_TYPE,
)
from .practiced_response import (
    PracticedResponseArchetype, PRACTICED_RESPONSE_POST_TYPE,
)
from .reflection import ReflectionArchetype, REFLECTION_POST_TYPE
from .composed_thought import (
    ComposedThoughtArchetype, COMPOSED_THOUGHT_POST_TYPE,
)
from .self_watching import SelfWatchingArchetype, SELF_WATCHING_POST_TYPE
from .amplify import AmplifyArchetype, AMPLIFY_POST_TYPE

ALL_ARCHETYPES = (
    ProofDayArchetype,
    SoulDiaryArchetype,
    WorldMirrorArchetype,
    OuterRuminationArchetype,
    OuterInnerBridgeArchetype,
    GroundedTodayArchetype,
    PracticedResponseArchetype,
    ReflectionArchetype,
    ComposedThoughtArchetype,
    SelfWatchingArchetype,
    AmplifyArchetype,
)

ARCHETYPE_POST_TYPES = (
    PROOF_DAY_POST_TYPE,
    SOUL_DIARY_POST_TYPE,
    WORLD_MIRROR_POST_TYPE,
    OUTER_RUMINATION_POST_TYPE,
    OINB_POST_TYPE,
    GROUNDED_TODAY_POST_TYPE,
    PRACTICED_RESPONSE_POST_TYPE,
    REFLECTION_POST_TYPE,
    COMPOSED_THOUGHT_POST_TYPE,
    SELF_WATCHING_POST_TYPE,
    AMPLIFY_POST_TYPE,
)

__all__ = (
    "ArchetypeBase",
    "ArchetypeCandidate",
    "ALL_ARCHETYPES",
    "ARCHETYPE_POST_TYPES",
    "ProofDayArchetype",
    "SoulDiaryArchetype",
    "WorldMirrorArchetype",
    "OuterRuminationArchetype",
    "OuterInnerBridgeArchetype",
    "GroundedTodayArchetype",
    "PracticedResponseArchetype",
    "ReflectionArchetype",
    "ComposedThoughtArchetype",
    "SelfWatchingArchetype",
    "AmplifyArchetype",
    "PROOF_DAY_POST_TYPE",
    "SOUL_DIARY_POST_TYPE",
    "WORLD_MIRROR_POST_TYPE",
    "OUTER_RUMINATION_POST_TYPE",
    "OINB_POST_TYPE",
    "GROUNDED_TODAY_POST_TYPE",
    "PRACTICED_RESPONSE_POST_TYPE",
    "REFLECTION_POST_TYPE",
    "COMPOSED_THOUGHT_POST_TYPE",
    "SELF_WATCHING_POST_TYPE",
    "AMPLIFY_POST_TYPE",
)
