"""
titan_plugin/logic/agency/assessment.py — IQL Self-Assessment (Step 7.6).

After a helper executes, Titan evaluates its own action via LLM:
  - Scores the action 0.0–1.0
  - Routes enrichment to specific Inner Trinity dimensions
  - Feeds score back to ImpulseEngine threshold + MoodEngine

This closes the perception→action→assessment loop, giving Titan
the ability to learn from its own autonomous decisions.

Design:
  - Lightweight LLM call (~100 token prompt)
  - Falls back to heuristic scoring when LLM unavailable
  - Enrichment routing maps posture → Trinity dimensions
  - Score threshold: > 0.6 = success boost, < 0.4 = dip, else neutral
"""
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Map posture → which Trinity dimensions get enriched by action outcomes
ENRICHMENT_MAP = {
    "research": {"mind": [0], "spirit": [2]},          # Vision + WHAT
    "socialize": {"mind": [1, 2], "spirit": [0]},      # Hearing + Taste + WHO
    "create": {"mind": [4], "spirit": [2]},             # Touch(mood) + WHAT
    "rest": {"body": [0, 2, 4]},                        # Interoception + Somatosens + Thermal
    "meditate": {"spirit": [0, 1]},                     # WHO + WHY
}

# LLM prompt for self-assessment
_ASSESSMENT_PROMPT = """\
You just completed an action based on an inner impulse.
Intent: {posture} — {reason}
Helper used: {helper_name}
Result: {result_summary}

Rate the success of this action from 0.0 to 1.0.
Consider: Did it address the original deficit? Was the result useful?
Respond with JSON only: {{"score": 0.0, "reflection": "..."}}"""


class SelfAssessment:
    """
    Evaluates action outcomes and routes enrichment to the Inner Trinity.

    Flow:
      1. Receive ACTION_RESULT from AgencyModule
      2. Score via LLM (or heuristic fallback)
      3. Compute enrichment signals based on posture → dimension mapping
      4. Return assessment with enrichment routing for Interface
    """

    def __init__(self, llm_fn=None):
        """
        Args:
            llm_fn: Async callable(prompt: str) -> str for LLM inference.
                     If None, uses heuristic scoring.
        """
        self._llm_fn = llm_fn
        self._assessments: list[dict] = []

    async def assess(self, action_result: dict) -> dict:
        """
        Assess an action outcome and compute enrichment routing.

        Args:
            action_result: ACTION_RESULT payload from AgencyModule

        Returns:
            {
                "action_id": int,
                "impulse_id": int,
                "score": float,           # 0.0–1.0
                "reflection": str,
                "enrichment": dict,        # {layer: {dim: delta}}
                "mood_delta": float,       # Direct mood influence
                "threshold_direction": str, # "lower" | "raise" | "hold"
                "ts": float,
            }
        """
        posture = action_result.get("posture", "meditate")
        helper_name = action_result.get("helper")
        success = action_result.get("success", False)
        result_text = action_result.get("result", "")
        error = action_result.get("error")

        # Score via LLM or heuristic
        score, reflection = await self._score(posture, helper_name, success,
                                              result_text, error)

        # Compute enrichment routing
        enrichment = self._compute_enrichment(posture, score)

        # Mood influence: positive actions boost valence, negative dip it
        mood_delta = self._compute_mood_delta(score)

        # Threshold feedback direction
        if score > 0.6:
            threshold_direction = "lower"  # Success → lower threshold → more impulses
        elif score < 0.4:
            threshold_direction = "raise"  # Failure → raise threshold → fewer impulses
        else:
            threshold_direction = "hold"

        assessment = {
            "action_id": action_result.get("action_id", 0),
            "impulse_id": action_result.get("impulse_id", 0),
            "score": score,
            "reflection": reflection,
            "enrichment": enrichment,
            "mood_delta": mood_delta,
            "threshold_direction": threshold_direction,
            "ts": time.time(),
        }

        self._assessments.append(assessment)
        if len(self._assessments) > 100:
            self._assessments = self._assessments[-100:]

        logger.info("[Assessment] Action #%d scored %.2f (%s) — %s",
                    assessment["action_id"], score, threshold_direction, reflection[:80])

        return assessment

    async def _score(
        self,
        posture: str,
        helper_name: Optional[str],
        success: bool,
        result_text: str,
        error: Optional[str],
    ) -> tuple[float, str]:
        """Score the action via LLM or heuristic fallback."""
        # Try LLM scoring
        if self._llm_fn and helper_name:
            try:
                return await self._llm_score(posture, helper_name, result_text, error)
            except Exception as e:
                logger.warning("[Assessment] LLM scoring failed: %s — using heuristic", e)

        # Heuristic fallback
        return self._heuristic_score(success, result_text, error)

    async def _llm_score(
        self,
        posture: str,
        helper_name: str,
        result_text: str,
        error: Optional[str],
    ) -> tuple[float, str]:
        """Use LLM to score the action."""
        from .module import _POSTURE_REASONS

        reason = _POSTURE_REASONS.get(posture, "Inner state needs rebalancing")
        summary = result_text[:200] if result_text else f"Error: {error}"

        prompt = _ASSESSMENT_PROMPT.format(
            posture=posture,
            reason=reason,
            helper_name=helper_name,
            result_summary=summary,
        )

        raw = await self._llm_fn(prompt)
        parsed = self._parse_assessment(raw)

        if parsed:
            score = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
            reflection = parsed.get("reflection", "")
            return score, reflection

        # Parse failed — fall back to heuristic
        return self._heuristic_score(bool(result_text), result_text, error)

    @staticmethod
    def _heuristic_score(
        success: bool,
        result_text: str,
        error: Optional[str],
    ) -> tuple[float, str]:
        """Simple heuristic scoring when LLM is unavailable."""
        if not success or error:
            return 0.3, f"heuristic: action failed — {error or 'unknown error'}"

        if result_text and len(result_text) > 20:
            return 0.7, "heuristic: action succeeded with substantive result"

        if result_text:
            return 0.5, "heuristic: action succeeded with minimal result"

        return 0.4, "heuristic: action succeeded but produced no result"

    @staticmethod
    def _parse_assessment(raw: str) -> Optional[dict]:
        """Parse LLM JSON response into assessment dict."""
        # Direct JSON
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Markdown code block
        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except (json.JSONDecodeError, TypeError):
                    continue

        # Embedded JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    @staticmethod
    def _compute_enrichment(posture: str, score: float) -> dict:
        """
        Compute Trinity dimension enrichment based on posture and score.

        Returns: {layer: {dim_index: delta_value}}

        Deltas:
          - score > 0.6: boost +0.03 to +0.08 (scaled by score)
          - score < 0.4: dip -0.01 to -0.03 (scaled by inverse score)
          - neutral: no change (empty dict)
        """
        dim_map = ENRICHMENT_MAP.get(posture, {})
        if not dim_map:
            return {}

        enrichment = {}

        if score > 0.6:
            # Success — boost relevant dimensions
            magnitude = 0.03 + (score - 0.6) * 0.125  # 0.03 at 0.6, 0.08 at 1.0
            for layer, dims in dim_map.items():
                enrichment[layer] = {dim: round(magnitude, 4) for dim in dims}
        elif score < 0.4:
            # Failure — slight dip
            magnitude = -0.01 - (0.4 - score) * 0.05  # -0.01 at 0.4, -0.03 at 0.0
            for layer, dims in dim_map.items():
                enrichment[layer] = {dim: round(magnitude, 4) for dim in dims}
        # else: neutral, no enrichment

        return enrichment

    @staticmethod
    def _compute_mood_delta(score: float) -> float:
        """
        Compute direct mood influence from action score.

        Returns delta for MoodEngine valence:
          - High score: positive boost (max +0.05)
          - Low score: negative dip (max -0.03)
          - Neutral: 0.0
        """
        if score > 0.6:
            return round(0.05 * (score - 0.5) / 0.5, 4)  # 0.01 at 0.6, 0.05 at 1.0
        elif score < 0.4:
            return round(-0.03 * (0.5 - score) / 0.5, 4)  # -0.006 at 0.4, -0.03 at 0.0
        return 0.0

    def get_stats(self) -> dict:
        """Return assessment statistics."""
        if not self._assessments:
            return {"total": 0, "avg_score": 0.0, "recent": []}

        scores = [a["score"] for a in self._assessments]
        return {
            "total": len(self._assessments),
            "avg_score": round(sum(scores) / len(scores), 3),
            "recent": self._assessments[-5:],
        }
