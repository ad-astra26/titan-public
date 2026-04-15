"""NarrativeChannel — Tier 3 dispatch layer for Maker-Titan narrative understanding.

Emits MAKER_NARRATION_REQUEST via DivineBus so language_worker can:
  1. Ground Maker's words somatically (CGN MAKER_RESPONSE_LINKED)
  2. Generate LLM narration (Titan's first-person reflection)
  3. Write back via MAKER_NARRATION_RESULT → spirit_worker → ProposalStore

The iron rule: Titan FEELS the response (Tier 2 somatic) BEFORE he
UNDERSTANDS it (Tier 3 narrative). This channel fires AFTER somatic.

Bus pattern: same as SomaticChannel — uses plugin.bus.publish(make_msg(...)).
Fire-and-forget (non-blocking). LLM call happens asynchronously in
language_worker subprocess.
"""
import logging
from typing import Optional

logger = logging.getLogger("NarrativeChannel")


class NarrativeChannel:
    """Dispatches MAKER_NARRATION_REQUEST for language worker processing."""

    def __init__(self, bus, src_module: str = "titan_maker"):
        self._bus = bus
        self._src = src_module

    def queue_narration(
        self,
        *,
        proposal_id: str,
        proposal_type,
        title: str,
        response: str,
        reason: str,
    ) -> None:
        """Emit MAKER_NARRATION_REQUEST. Non-blocking fire-and-forget.

        Args:
            proposal_id: which proposal was responded to
            proposal_type: ProposalType enum or string
            title: proposal title (for LLM context)
            response: "approve" or "decline"
            reason: Maker's written reason (≥10 chars)
        """
        try:
            from titan_plugin.bus import make_msg
            ptype = (proposal_type.value
                     if hasattr(proposal_type, "value")
                     else str(proposal_type))
            self._bus.publish(make_msg(
                msg_type="MAKER_NARRATION_REQUEST",
                src=self._src,
                dst="language",
                payload={
                    "proposal_id": proposal_id,
                    "proposal_type": ptype,
                    "title": title,
                    "response": response,
                    "reason": reason,
                },
            ))
            logger.info(
                "[NarrativeChannel] Narration queued: proposal=%s response=%s",
                proposal_id[:8], response)
        except Exception as e:
            logger.warning(
                "[NarrativeChannel] Failed to queue narration: %s", e)
