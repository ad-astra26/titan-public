"""
scripts/learning/smart_scheduler.py — Smart Scheduler for Learning TestSuite.

Observes Titan's state via API and decides:
  1. WHEN to teach (emotional readiness, Chi, dreaming state)
  2. WHAT to teach (module type matched to emotional state)

Never fights the hormonal state — rides it.
Never teaches during dreaming — consolidation in progress.
Never teaches when Chi is too low — let self-exploration build energy first.
"""
import logging
from typing import Optional

logger = logging.getLogger("testsuite.scheduler")

# Emotional state → module type mapping
# Based on neuroscience: attention (NE), reward (DA), inhibition (GABA)
MODULE_AFFINITY = {
    # (emotion, ne_range, da_range, gaba_range) → module_type
    "flow": "puzzle",           # Peak cognitive performance
    "curiosity": "language",    # Maximum attention for learning
    "love": "conversation",     # Social warmth
    "joy": "conversation",      # Social engagement
    "peace": "music",           # Receptive, contemplative
    "fear": None,               # Don't teach during stress
}


class SmartScheduler:
    """Watches Titan's state and selects optimal learning modules."""

    def __init__(self):
        self._consecutive_skips = 0
        self._max_consecutive_skips = 20  # After 20 skips (~20 epochs), lower threshold

    def should_teach_now(self, state: dict) -> bool:
        """Is Titan in a good state for a learning module?

        Checks:
        1. Not dreaming (consolidation in progress)
        2. Sufficient Chi (energy for learning)
        3. Not too inhibited (GABA not too high)
        4. Alert enough (NE above minimum)
        5. Not mid-external-interaction
        """
        # Never during dreaming
        if state.get("is_dreaming", False):
            logger.info("[Scheduler] Skip: dreaming (letting consolidation complete)")
            self._consecutive_skips += 1
            return False

        # Need minimum Chi (energy)
        chi = state.get("chi_total", 0.5)
        if chi < 0.35:
            logger.info("[Scheduler] Skip: low Chi (%.2f) — letting self-exploration build energy", chi)
            self._consecutive_skips += 1
            return False

        # Don't interrupt external interaction
        se_mode = state.get("se_mode", "SELF_EXPLORE")
        if se_mode == "EXTERNAL_PASSTHROUGH":
            logger.info("[Scheduler] Skip: external passthrough — letting Titan finish interaction")
            self._consecutive_skips += 1
            return False

        gaba = state.get("gaba", 0.5)
        ne = state.get("ne", 0.5)

        # Adaptive threshold: after many skips, lower the bar
        gaba_threshold = 0.85
        ne_threshold = 0.15
        if self._consecutive_skips > self._max_consecutive_skips:
            gaba_threshold = 0.90  # More lenient
            ne_threshold = 0.05  # Low enough to break bliss-lock catch-22
            logger.debug("[Scheduler] Adaptive: lowered thresholds after %d skips",
                        self._consecutive_skips)

        # Too inhibited → let self-exploration activate first
        if gaba > gaba_threshold:
            logger.info("[Scheduler] Skip: GABA too high (%.2f > %.2f) — letting exploration lower inhibition",
                        gaba, gaba_threshold)
            self._consecutive_skips += 1
            return False

        # Need minimum alertness
        if ne < ne_threshold:
            logger.info("[Scheduler] Skip: NE too low (%.2f < %.2f) — waiting for alertness",
                        ne, ne_threshold)
            self._consecutive_skips += 1
            return False

        # All checks passed — ready to teach
        self._consecutive_skips = 0
        logger.info("[Scheduler] Ready to teach: emotion=%s NE=%.2f GABA=%.2f Chi=%.2f",
                    state.get("emotion", "?"), ne, gaba, chi)
        return True

    def select_module_type(self, state: dict, curriculum_suggestion: str = None) -> str:
        """Match module type to Titan's current emotional state.

        Priority:
        1. Curriculum suggestion (if specific module is next in queue)
        2. Emotional state matching
        3. Default: language (always beneficial)
        """
        # Curriculum override
        if curriculum_suggestion and curriculum_suggestion != "auto":
            return curriculum_suggestion

        emotion = state.get("emotion", "peace")
        da = state.get("da", 0.5)
        ne = state.get("ne", 0.5)
        gaba = state.get("gaba", 0.5)

        # Flow state → puzzle (peak cognitive)
        if da > 0.7 and gaba < 0.4:
            return "puzzle"

        # High alertness → language (attention is maximal)
        if ne > 0.5:
            return "language"

        # Emotion-based selection
        affinity = MODULE_AFFINITY.get(emotion)
        if affinity:
            return affinity

        # Warm emotions → conversation
        if emotion in ("love", "joy"):
            return "conversation"

        # Calm state → music or art narration
        if gaba > 0.6:
            return "music"

        # Default
        return "language"

    def select_module_count(self, state: dict) -> int:
        """How many modules to run in this teaching window?

        High energy + alertness → 2-3 modules
        Moderate → 1 module
        """
        chi = state.get("chi_total", 0.5)
        ne = state.get("ne", 0.5)

        if chi > 0.7 and ne > 0.5:
            return 3  # Excellent state — extended session
        if chi > 0.5 and ne > 0.3:
            return 2  # Good state — double session
        return 1  # Standard — single module

    def get_stats(self) -> dict:
        return {
            "consecutive_skips": self._consecutive_skips,
            "adaptive_active": self._consecutive_skips > self._max_consecutive_skips,
        }
