"""
titan_hcl/logic/expression_translator.py — Expression Translation Layer.

Learns to translate internal hormone-driven program fires into optimal
external actions. Replaces LLM-based helper selection with experience-driven
mapping that improves autonomously from outcomes.

Key principles:
- NO hardcoded mappings — all scores learned from experience
- Default affinities only SEED initial bias (like DNA), not fix outcomes
- IQL-style EMA learning from action outcomes
- LLM fallback for novel situations (confidence < threshold)
- Sovereignty tracking: learned_actions / total_actions ratio

The FeedbackRouter closes the loop:
  hormone fires → expression translates → action executes →
  outcome scored → feedback to hormone (satisfaction) + translator (learning)
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum observations before trusting a learned mapping
MIN_CONFIDENCE = 3

# Default affinities — SEED values only (DNA-like initial bias)
# All start at 0.5 (neutral) and learn from there
# Programs with action_helpers get slightly higher seeds for their natural helpers
DEFAULT_AFFINITIES = {
    "CURIOSITY":   {"web_search": 0.55, "code_knowledge": 0.52},
    "CREATIVITY":  {"art_generate": 0.55, "audio_generate": 0.53},
    "EMPATHY":     {"web_search": 0.52},  # social posting via SocialPressureMeter only
    "INSPIRATION": {"art_generate": 0.53, "audio_generate": 0.53,
                    "web_search": 0.51, "coding_sandbox": 0.51},
    "VIGILANCE":   {"infra_inspect": 0.55},
    "REFLECTION":  {"code_knowledge": 0.53, "infra_inspect": 0.51},
    "IMPULSE":     {"infra_inspect": 0.52, "web_search": 0.51},
    "REFLEX":      {"infra_inspect": 0.53},
    "FOCUS":       {"infra_inspect": 0.52, "code_knowledge": 0.51},
    "INTUITION":   {"web_search": 0.52, "code_knowledge": 0.51},
}


class ExpressionTranslator:
    """
    Learns optimal program→action mappings from experience.

    For each (program, helper) pair, maintains a quality score
    that adapts from action outcomes via IQL-style EMA learning.
    Scores start near 0.5 (neutral) and diverge purely from
    Titan's own experience — no hardcoded preferences.
    """

    def __init__(self, all_helpers: list[str] = None):
        # Per-program quality scores: {program: {helper: score}}
        self._scores: dict[str, dict[str, float]] = {}
        # Confidence: observations count per mapping
        self._confidence: dict[str, dict[str, int]] = {}
        # All available helper names
        self._all_helpers = all_helpers or []

        # Sovereignty tracking — Wave 4b Cat B.1 persistence (D-SPEC-89,
        # 2026-05-18): cumulative counters persisted across restart so
        # sovereignty_ratio = learned/total reflects lifetime ratio, not
        # session-only. Pre-fix: every restart reset all 3 → 0, sovereignty=0
        # forever on T1 (low actor) → cascaded to inner_spirit[2] sovereignty +
        # outer_spirit[2] action_sovereignty + outer_spirit[9] action_purity
        # dead fleet-wide.
        self._learned_actions: int = 0
        self._llm_actions: int = 0
        self._total_actions: int = 0
        try:
            from titan_hcl.core.rolling_state_persistence import (
                RollingStateStore)
            # Single-entry "store" — keep a 1-row JSON file with the latest
            # cumulative counters. max_entries=1, no age cap (lifetime data).
            self._sovereignty_counter_store = RollingStateStore(
                name="expression_translator_sovereignty",
                max_entries=1,
                max_age_s=None,  # lifetime cumulative — never age out
                save_every_n=10,
                save_every_s=60.0,
            )
            _loaded = self._sovereignty_counter_store.load()
            if _loaded:
                _row = _loaded[-1] if isinstance(_loaded, list) else _loaded
                if isinstance(_row, dict):
                    self._learned_actions = int(_row.get("learned", 0) or 0)
                    self._llm_actions = int(_row.get("llm", 0) or 0)
                    self._total_actions = int(_row.get("total", 0) or 0)
                    logger.info(
                        "[ExpressionTranslator] restored sovereignty counters: "
                        "learned=%d llm=%d total=%d",
                        self._learned_actions, self._llm_actions,
                        self._total_actions)
        except Exception as _e:
            logger.warning(
                "[ExpressionTranslator] sovereignty-counter persistence "
                "init failed (%s) — in-memory only", _e)
            self._sovereignty_counter_store = None

        # Sprint 7 §4.6 / rFP §4.3.1 row 1 — posture-authenticity tracker.
        # Ring buffer of the last 30 actions, each carrying:
        #   - posture (str)
        #   - target hormone for that posture (str, from POSTURE_HORMONE
        #     mapping below)
        #   - whether the target hormone was DOMINANT in the hormone
        #     levels snapshot captured AT action time (bool)
        # Authenticity ratio = fraction of last 30 actions where the
        # target hormone was dominant. Feeds outer_spirit SAT[1]
        # `expressive_authenticity` + inner_spirit via expression_state
        # SHM slot.
        from collections import deque
        self._posture_authenticity_log: deque = deque(maxlen=30)
        # rolling_state_persistence: survives Titan restarts so SAT[1]
        # `expressive_authenticity` warms from history at boot rather
        # than needing 30 fresh actions each restart. Per-Titan file
        # `<titan_data_dir>/dim_history/posture_authenticity.json`.
        try:
            from titan_hcl.core.rolling_state_persistence import (
                RollingStateStore)
            self._posture_authenticity_store = RollingStateStore(
                name="posture_authenticity",
                max_entries=30,
                max_age_s=24 * 3600,  # 24h rolling window
                save_every_n=1,
                save_every_s=30.0,
                ts_key="ts",
            )
            loaded = self._posture_authenticity_store.load()
            if loaded:
                self._posture_authenticity_log.extend(loaded[-30:])
                logger.info(
                    "[ExpressionTranslator] loaded %d posture-authenticity "
                    "entries", len(loaded))
        except Exception as _e:
            logger.warning(
                "[ExpressionTranslator] posture_authenticity persistence "
                "boot failed (%s) — in-memory only", _e)
            self._posture_authenticity_store = None

        # Initialize from default affinities (DNA seed)
        self._init_default_scores()

    # POSTURE → target hormone mapping. Encodes the SPEC §23.9 SAT[1]
    # premise: a "creative" action expresses CREATIVITY pressure, an
    # "art" action expresses CREATIVITY, "kin_sense" expresses EMPATHY,
    # etc. The dominant-hormone-at-action-time check measures whether
    # the action emerged from the matching internal pressure.
    _POSTURE_HORMONE: dict[str, str] = {
        "art": "CREATIVITY",
        "music": "CREATIVITY",
        "creative": "CREATIVITY",
        "kin_sense": "EMPATHY",
        "social": "EMPATHY",
        "speak": "EMPATHY",
        "meditate": "REFLECTION",
        "reflect": "REFLECTION",
        "research": "CURIOSITY",
        "explore": "CURIOSITY",
        "code_knowledge": "CURIOSITY",
        "web_search": "CURIOSITY",
        "boundary_block": "VIGILANCE",
        "defend": "VIGILANCE",
        "act": "IMPULSE",
        "action": "IMPULSE",
    }

    def record_posture_authenticity(
        self,
        posture: str,
        hormone_levels_at_fire: dict[str, float],
    ) -> None:
        """
        Record a single action's posture + hormone snapshot. Computes
        whether the posture's target hormone was DOMINANT (i.e. had the
        highest level among all hormones) at action time.

        Called from the action-firing path with the hormone levels
        captured just before the action ran.
        """
        import time as _time
        target = self._POSTURE_HORMONE.get(str(posture).lower())
        if not target or not isinstance(hormone_levels_at_fire, dict):
            return
        if not hormone_levels_at_fire:
            entry = {"posture": posture, "target": target,
                     "authentic": False, "ts": _time.time()}
            self._posture_authenticity_log.append(entry)
            if self._posture_authenticity_store is not None:
                self._posture_authenticity_store.append_and_save(
                    entry, list(self._posture_authenticity_log))
            return
        try:
            max_hormone = max(
                hormone_levels_at_fire.items(), key=lambda kv: float(kv[1]))[0]
        except (TypeError, ValueError):
            return
        authentic = (max_hormone == target)
        entry = {"posture": posture, "target": target,
                 "authentic": authentic, "ts": _time.time()}
        self._posture_authenticity_log.append(entry)
        if self._posture_authenticity_store is not None:
            self._posture_authenticity_store.append_and_save(
                entry, list(self._posture_authenticity_log))

    def posture_authenticity_ratio_30(self) -> float:
        """
        Fraction of last 30 actions where the posture's target hormone
        was dominant in hormone_levels at action time. Returns 0.0 when
        no actions tracked yet (cold-start).
        Feeds outer_spirit SAT[1] expressive_authenticity.
        """
        if not self._posture_authenticity_log:
            return 0.0
        hits = sum(
            1 for rec in self._posture_authenticity_log
            if rec.get("authentic"))
        return hits / len(self._posture_authenticity_log)

    def _init_default_scores(self) -> None:
        """Seed scores from default affinities. All helpers get 0.5 base."""
        for program, helpers in DEFAULT_AFFINITIES.items():
            if program not in self._scores:
                self._scores[program] = {}
            if program not in self._confidence:
                self._confidence[program] = {}
            # Seed natural affinities slightly above neutral
            for helper, score in helpers.items():
                self._scores[program][helper] = score
                self._confidence[program][helper] = 0  # Not yet experienced

    def translate(
        self,
        program: str,
        intensity: float,
        posture: str,
        available_helpers: list[str],
        trinity_snapshot: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Translate a program fire into a specific action selection.

        Returns dict with helper + params if confident enough,
        or None to signal LLM fallback.
        """
        scores = self._scores.get(program, {})
        confidences = self._confidence.get(program, {})

        if not scores:
            return None

        # Filter to available helpers and score them
        candidates = []
        for h in available_helpers:
            if h in scores:
                candidates.append((h, scores[h], confidences.get(h, 0)))
            else:
                # Unknown helper for this program — neutral score
                candidates.append((h, 0.5, 0))

        if not candidates:
            return None

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_helper, best_score, best_conf = candidates[0]

        # Need minimum observations to be confident
        if best_conf < MIN_CONFIDENCE:
            return None  # Not enough experience — LLM fallback

        # Intensity modulation: higher intensity → bolder action selection
        # (could select 2nd-ranked helper if intensity is very high — exploration)

        return {
            "helper": best_helper,
            "params": self._build_params(program, best_helper, intensity,
                                          posture, trinity_snapshot),
            "confidence": min(1.0, best_conf / 10.0),
            "reasoning": (f"Learned: {program}→{best_helper} "
                         f"(score={best_score:.2f}, observations={best_conf})"),
        }

    def _build_params(self, program: str, helper: str, intensity: float,
                      posture: str, trinity_snapshot: Optional[dict]) -> dict:
        """Build helper-specific params from context."""
        params = {
            "posture": posture,
            "trinity_snapshot": trinity_snapshot or {},
            "intensity": intensity,
            "triggering_program": program,
        }
        return params

    def record_outcome(
        self,
        program: str,
        helper: str,
        score: float,
        learning_rate: float = 0.15,
    ) -> None:
        """
        Learn from action outcome — update quality score via IQL-style EMA.

        Recent outcomes matter more than old ones (exponential moving average).
        Score converges toward the actual outcome quality over time.
        """
        if program not in self._scores:
            self._scores[program] = {}
        if program not in self._confidence:
            self._confidence[program] = {}

        old = self._scores[program].get(helper, 0.5)
        # EMA update: score moves toward outcome
        self._scores[program][helper] = old + learning_rate * (score - old)
        self._confidence[program][helper] = \
            self._confidence[program].get(helper, 0) + 1

    def record_action_type(self, was_learned: bool) -> None:
        """Track whether this action used learned mapping or LLM fallback.

        Wave 4b Cat B.1 (D-SPEC-89) — also persist to JSON snapshot so
        cumulative counters survive Titan restart. save_every_n=10 batches
        disk writes; append_and_save is O(1) plus rare fsync.
        """
        self._total_actions += 1
        if was_learned:
            self._learned_actions += 1
        else:
            self._llm_actions += 1
        # Persist cumulative state (best-effort)
        if getattr(self, "_sovereignty_counter_store", None) is not None:
            try:
                row = {
                    "learned": self._learned_actions,
                    "llm": self._llm_actions,
                    "total": self._total_actions,
                    "ts": time.time(),
                }
                self._sovereignty_counter_store.append_and_save(row, [row])
            except Exception:
                pass  # best-effort; never block action recording

    @property
    def sovereignty_ratio(self) -> float:
        """Ratio of self-learned actions to total. 1.0 = fully autonomous."""
        if self._total_actions == 0:
            return 0.0
        return self._learned_actions / self._total_actions

    def get_stats(self) -> dict:
        """Full stats for API — includes sovereignty tracking."""
        # Find highest-scoring mappings
        top_mappings = []
        for program, helpers in self._scores.items():
            for helper, score in helpers.items():
                conf = self._confidence.get(program, {}).get(helper, 0)
                if conf > 0:
                    top_mappings.append({
                        "program": program, "helper": helper,
                        "score": round(score, 3), "observations": conf,
                    })
        top_mappings.sort(key=lambda x: x["observations"], reverse=True)

        return {
            "sovereignty_ratio": round(self.sovereignty_ratio, 4),
            "learned_actions": self._learned_actions,
            "llm_actions": self._llm_actions,
            "total_actions": self._total_actions,
            "top_mappings": top_mappings[:15],
            "total_learned_pairs": sum(
                1 for p in self._confidence.values()
                for c in p.values() if c >= MIN_CONFIDENCE
            ),
            # Sprint 7 §4.6 / rFP §4.3.1 row 1: outer_spirit SAT[1]
            # `expressive_authenticity` source. Ratio of last-30 actions
            # where posture's target hormone was dominant at action time.
            "posture_authenticity_ratio_30": round(
                self.posture_authenticity_ratio_30(), 4),
        }

    def save(self, path: str) -> None:
        import os
        data = {
            "scores": self._scores,
            "confidence": self._confidence,
            "learned_actions": self._learned_actions,
            "llm_actions": self._llm_actions,
            "total_actions": self._total_actions,
        }
        tmp = path + ".tmp"
        Path(tmp).write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self._scores = data.get("scores", self._scores)
                self._confidence = data.get("confidence", self._confidence)
                self._learned_actions = data.get("learned_actions", 0)
                self._llm_actions = data.get("llm_actions", 0)
                self._total_actions = data.get("total_actions", 0)
                logger.info("[ExpressionTranslator] Loaded: %d total actions, "
                            "sovereignty=%.1f%%",
                            self._total_actions, self.sovereignty_ratio * 100)
            except Exception as e:
                logger.warning("[ExpressionTranslator] Load error: %s", e)


class FeedbackRouter:
    """Routes action outcomes to hormone + translator for learning."""

    def __init__(self, hormonal_system, translator: ExpressionTranslator):
        self._hormonal = hormonal_system
        self._translator = translator

    def route(self, action_result: dict) -> None:
        """
        Route outcome to:
        1. Originating hormone (pressure satisfaction/frustration)
        2. Expression translator (learn better mappings)
        """
        program = action_result.get("triggering_program", "")
        helper = action_result.get("helper", "")
        score = action_result.get("score", 0.5)
        success = action_result.get("success", False)

        if not program or not helper:
            return

        # 1. Feed hormone — satisfy or frustrate the urge
        if self._hormonal:
            hormone = self._hormonal.get_hormone(program)
            if hormone:
                if score > 0.6 and success:
                    # Good outcome: satisfy the urge
                    hormone.level *= 0.3
                    hormone.adapt_threshold(reward=score - 0.5)
                    logger.debug("[Feedback] %s satisfied by %s (score=%.2f)",
                                 program, helper, score)
                elif score < 0.3:
                    # Bad outcome: partial frustration
                    hormone.level *= 0.7
                    hormone.adapt_threshold(reward=score - 0.5)
                else:
                    # Neutral outcome
                    hormone.level *= 0.6

        # 2. Feed translator — learn from this outcome
        if self._translator:
            self._translator.record_outcome(program, helper, score)
