"""SomaticChannel — Tier 2 bus dispatch for Maker response somatic processing.

Spirit_worker subscribes to MAKER_RESPONSE_RECEIVED and applies neuromod
nudges + felt-state updates + timechain meta-fork commits. This class is
purely the dispatch layer — the actual mod application lives in
spirit_worker so it has access to the live neuromodulator_system.

The iron rule: every approve OR decline propagates through this channel
when wired. Approval: felt validation (DA/Endorphin/5HT). Decline: felt
friction (NE/ACh, 5HT dip). Both are dialogic — both teach Titan.

Architecture note: TitanMaker lives in the MAIN TitanCore process where
dashboard endpoints serve. The DivineBus instance (plugin.bus from v5_core)
is the publish path. Workers (spirit_worker, language_worker) subscribe
to MAKER_RESPONSE_RECEIVED via their own subprocess queues — DivineBus
routes the message to all subscribers via dst="all".
"""
import logging

from .schemas import ProposalRecord, ProposalType

logger = logging.getLogger("SomaticChannel")


class SomaticChannel:
    """Bus dispatcher for MAKER_PROPOSAL_CREATED + MAKER_RESPONSE_RECEIVED.

    Held by TitanMaker via composition. Workers consume the messages
    via DivineBus subscriptions (no direct coupling to this class).
    """

    def __init__(self, bus, src_module: str = "titan_maker"):
        """Args:
            bus: a DivineBus instance with .publish() and a make_msg helper.
            src_module: the source name for emitted messages (default 'titan_maker')
        """
        self._bus = bus
        self._src = src_module

    def emit_proposal_created(self, record: ProposalRecord) -> None:
        """Notify the bus that a new pending proposal exists.

        Used by spirit_worker to log + by chat_api (Tier 2) to surface a
        system message in the conversation stream when isMaker connects.
        """
        try:
            from titan_plugin.bus import make_msg
            self._bus.publish(make_msg(
                "MAKER_PROPOSAL_CREATED", self._src, "all",
                {
                    "proposal_id": record.proposal_id,
                    "proposal_type": record.proposal_type.value,
                    "title": record.title,
                    "requires_signature": record.requires_signature,
                },
            ))
        except Exception as e:
            logger.warning("[SomaticChannel] proposal_created emit failed: %s", e)

    def emit_response_received(
        self, *, proposal_id: str, proposal_type: ProposalType,
        response: str, reason: str
    ) -> None:
        """Emit a MAKER_RESPONSE_RECEIVED for somatic processing.

        Args:
            proposal_id: the proposal that was approved/declined
            proposal_type: the ProposalType enum value (used for tagging)
            response: "approve" or "decline"
            reason: Maker's written reason (≥10 chars, validated upstream)
        """
        try:
            from titan_plugin.bus import make_msg
            self._bus.publish(make_msg(
                "MAKER_RESPONSE_RECEIVED", self._src, "all",
                {
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type.value,
                    "response": response,           # "approve" | "decline"
                    "reason": reason,
                },
            ))
        except Exception as e:
            logger.warning("[SomaticChannel] response_received emit failed: %s", e)
