"""
titan_plugin/logic/mind_tensor.py — 15D Mind Tensor (DQ2).

Expands Mind from 5D to 15D following the Rosicrucian trinity:
  Thinking (5D) — What Mind KNOWS (cognitive, current sensors refined)
  Feeling  (5D) — What Mind SENSES (Pancha Tanmatra — 5 subtle senses)
  Willing  (5D) — What Mind WANTS TO DO (creative force, maps to hormones)

The Willing dimensions directly map to hormonal pressure levels:
  Mind's WILL creates the PRESSURE that the nervous system translates into ACTION.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Mind 15D dimension names for documentation and API
MIND_DIM_NAMES = [
    # Thinking (0-4)
    "memory_depth", "social_cognition", "perceptual_thinking",
    "emotional_thinking", "conceptual_thinking",
    # Feeling (5-9)
    "inner_hearing", "inner_touch", "inner_sight",
    "inner_taste", "inner_smell",
    # Willing (10-14)
    "action_drive", "social_will", "creative_will",
    "protective_will", "growth_will",
]


def collect_mind_15d(
    # Existing 5D sources (become Thinking)
    current_5d: list,
    # Feeling sources
    audio_state: Optional[dict] = None,
    interaction_quality: float = 0.5,
    visual_state: Optional[dict] = None,
    assessment_quality: float = 0.5,
    ambient_change: float = 0.0,
    # Willing sources (hormone levels)
    hormone_levels: Optional[dict] = None,
) -> list:
    """
    Collect 15D Mind tensor from all sources.

    Args:
        current_5d: Existing 5D mind tensor [memory, social, media, mood, knowledge]
        audio_state: Audio/music processing state
        interaction_quality: Quality of recent interactions [0-1]
        visual_state: Visual/image processing state
        assessment_quality: Mean assessment score of recent actions [0-1]
        ambient_change: Rate of environmental change [0-1]
        hormone_levels: Current hormone pressure levels from HormonalSystem

    Returns:
        15D tensor [thinking(5) + feeling(5) + willing(5)]
    """
    # ── THINKING (5D) — from current mind sensors ──
    # Use existing 5D as the cognitive base
    thinking = list(current_5d[:5]) if len(current_5d) >= 5 else [0.5] * 5

    # ── FEELING (5D) — subtle sense dimensions ──
    feeling = [0.5] * 5

    # [5] Inner hearing — audio/vibration sensitivity
    if audio_state:
        feeling[0] = _clamp(audio_state.get("sensitivity", 0.5))
    else:
        # Baseline hearing: gentle ambient awareness
        feeling[0] = 0.4

    # [6] Inner touch — interaction responsiveness
    feeling[1] = _clamp(interaction_quality)

    # [7] Inner sight — visual pattern sensitivity
    if visual_state:
        feeling[2] = _clamp(visual_state.get("sensitivity", 0.5))
    else:
        feeling[2] = 0.4

    # [8] Inner taste — quality discrimination (beauty/ugliness)
    feeling[3] = _clamp(assessment_quality)

    # [9] Inner smell — ambient awareness (environmental change)
    feeling[4] = _clamp(0.3 + ambient_change * 0.7)

    # ── WILLING (5D) — from hormonal pressure levels ──
    # Mind's WILL creates the PRESSURE that drives ACTION
    willing = [0.5] * 5
    if hormone_levels:
        # [10] Action drive — IMPULSE hormone level
        willing[0] = _clamp(hormone_levels.get("IMPULSE", 0.0))
        # [11] Social will — EMPATHY hormone level
        willing[1] = _clamp(hormone_levels.get("EMPATHY", 0.0))
        # [12] Creative will — CREATIVITY hormone level
        willing[2] = _clamp(hormone_levels.get("CREATIVITY", 0.0))
        # [13] Protective will — VIGILANCE hormone level
        willing[3] = _clamp(hormone_levels.get("VIGILANCE", 0.0))
        # [14] Growth will — CURIOSITY hormone level
        willing[4] = _clamp(hormone_levels.get("CURIOSITY", 0.0))

    return thinking + feeling + willing


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, float(v)))
