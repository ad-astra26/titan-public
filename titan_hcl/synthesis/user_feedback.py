"""UserFeedbackOverride — Tier-2 explicit-feedback override (Synthesis Engine Phase 9).

rFP §11.3 success-signal tiers: Tier-0 (implicit pre-filter), Tier-1 (objective
oracle), Tier-1-LLM (fuzzy-domain fallback), **Tier-2 (explicit user feedback —
overrides when present)**. INV-Syn-24: when an explicit `USER_FEEDBACK_SIGNAL`
exists for a tool-call TX, `scored_by="user"` is canonical for that TX regardless
of any prior oracle/llm verdict.

The override is **provenance-preserving** (INV-5): the prior verdict TX stays
on-chain, readable, marked superseded — the override is a new meta-fork patch that
supersedes without erasing. Explicit feedback applies a larger utility delta
(`user_feedback_delta`, default 0.15) than the 0.05 oracle/invocation delta,
because a human signal is the strongest tier.

v1 is explicit-only (a thumbs-up/down or recognized explicit-feedback message,
emitted agno-side as `USER_FEEDBACK_SIGNAL{..., source:"explicit"}`); implicit
sentiment is out of scope. Sole writer is synthesis_worker.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["UserFeedbackOverride", "VALID_VERDICTS"]

VALID_VERDICTS = ("positive", "negative")


class UserFeedbackOverride:
    """Applies Tier-2 explicit user feedback to a tool-call TX + (optionally) a skill.

    `skill_store` may be None (feedback on a tool call that didn't resolve through a
    compiled skill). `outer_memory_writer` anchors the superseding meta-fork patch.
    """

    def __init__(
        self,
        *,
        outer_memory_writer,
        skill_store=None,
        user_feedback_delta: float = 0.15,
    ) -> None:
        self._omw = outer_memory_writer
        self._skill_store = skill_store
        self._delta = float(user_feedback_delta)

    def apply(
        self,
        *,
        tool_call_tx: str,
        verdict: str,
        skill_id: Optional[str] = None,
        source: str = "explicit",
    ) -> Optional[dict]:
        """Apply Tier-2 feedback. Returns a summary dict, or None on bad input.

        - Anchors a `scored_by="user"` patch on the meta fork (supersedes prior
          oracle/llm verdict without erasing it; OracleCoverage reads it).
        - When `skill_id` is given, adjusts that skill's utility by ±delta.
        Soft, total: never raises (anchoring/utility failures log + degrade).
        """
        if source != "explicit":
            # v1 is explicit-only; implicit sentiment is ignored.
            logger.debug(
                "[UserFeedbackOverride] non-explicit feedback ignored "
                "(source=%s, v1 explicit-only)", source,
            )
            return None
        if not tool_call_tx or verdict not in VALID_VERDICTS:
            logger.warning(
                "[UserFeedbackOverride] bad input tool_call_tx=%r verdict=%r",
                tool_call_tx, verdict,
            )
            return None

        new_utility: Optional[float] = None
        if skill_id and self._skill_store is not None:
            delta = self._delta if verdict == "positive" else -self._delta
            try:
                new_utility = self._skill_store.apply_utility_delta(skill_id, delta)
            except Exception as exc:
                logger.warning(
                    "[UserFeedbackOverride] utility adjust failed skill=%s: %s",
                    skill_id, exc,
                )

        patch_tx: Optional[str] = None
        try:
            patch_tx = self._omw.write_scored_by_patch(entries=[{
                "parent_tool_call_tx": tool_call_tx,
                "scored_by": "user",
                "verdict": verdict,
                "source": source,
                "skill_id": skill_id,
                "supersedes": "oracle_or_llm",
            }])
        except Exception as exc:
            logger.error(
                "[UserFeedbackOverride] scored_by patch anchor failed for "
                "tx=%s: %s", tool_call_tx, exc, exc_info=True,
            )

        logger.info(
            "[UserFeedbackOverride] Tier-2 override: tx=%s verdict=%s "
            "scored_by=user skill=%s new_utility=%s patch_tx=%s",
            tool_call_tx, verdict, skill_id, new_utility, patch_tx,
        )
        return {
            "tool_call_tx": tool_call_tx,
            "verdict": verdict,
            "scored_by": "user",
            "skill_id": skill_id,
            "new_utility": new_utility,
            "patch_tx": patch_tx,
        }
