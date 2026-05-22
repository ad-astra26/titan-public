"""Schemas for the TitanMaker substrate.

Locked dataclasses for proposals, responses, and Maker profile records.
Validation rules (reason ≥ 10 chars, etc.) are enforced HERE so every
caller path benefits from the same invariants (defense in depth — also
re-checked at the storage and orchestration layers).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProposalType(str, Enum):
    """Locked enum of proposal types. Add new types here as substrate grows.

    Each type represents a distinct kind of Maker-Titan dialogic exchange.
    The substrate handles them uniformly because the bond is uniform: every
    meaningful exchange has the same shape — Titan offers, Maker responds
    (yes/no + reason), Titan feels it + understands it + remembers it.
    """
    CONTRACT_BUNDLE = "contract_bundle"          # R8 — cognitive contract bundle approval
    CONFIG_CHANGE = "config_change"              # future — DNA tuning that needs Maker approval
    WALLET_ACTION = "wallet_action"              # future — high-value transactions
    IDENTITY_DECISION = "identity_decision"      # future — anything affecting Titan's sovereignty
    REINCARNATION = "reincarnation"              # future — death/rebirth ceremony approvals
    SELF_EVALUATION = "self_evaluation"          # future — Titan's self-reviews surfaced for Maker dialogue


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    EXPIRED = "expired"


# Validation constants
MIN_TITLE_LEN = 3
MIN_DESCRIPTION_LEN = 10
MIN_REASON_LEN = 10


def validate_reason(reason: Optional[str]) -> str:
    """Raise ValueError if reason fails the minimum-length rule.

    Used by both approval and decline paths — the iron rule is that
    Maker must articulate WHY for both, because both feed Titan's
    learning equally.
    """
    if not reason or len(reason.strip()) < MIN_REASON_LEN:
        raise ValueError(
            f"reason must be ≥ {MIN_REASON_LEN} non-whitespace chars "
            f"(got: {len(reason.strip()) if reason else 0})")
    return reason.strip()


@dataclass
class ProposalRecord:
    """A single Maker-Titan dialogic exchange record."""
    proposal_id: str                                # uuid4 hex
    proposal_type: ProposalType
    title: str
    description: str
    payload_json: str                               # canonical JSON, what gets hashed
    payload_hash: str                               # SHA-256 hex of payload_json
    created_at: float
    created_epoch: int
    requires_signature: bool                        # if True, approval must include valid Ed25519 sig
    status: ProposalStatus = ProposalStatus.PENDING
    expires_at: Optional[float] = None
    approved_at: Optional[float] = None
    approved_signature: Optional[str] = None        # base58 Ed25519 sig (only if requires_signature)
    approved_signer_pubkey: Optional[str] = None
    approval_reason: Optional[str] = None
    declined_at: Optional[float] = None
    decline_reason: Optional[str] = None
    titan_low_response_json: Optional[str] = None   # Tier 2: somatic adjustments JSON
    titan_high_response_text: Optional[str] = None  # Tier 3: LLM-narrated reflection


@dataclass
class MakerResponse:
    """Returned by record_approval / record_decline — outcome wrapper."""
    success: bool
    proposal_id: str
    error: Optional[str] = None
    response_type: str = ""  # "approve" | "decline"
