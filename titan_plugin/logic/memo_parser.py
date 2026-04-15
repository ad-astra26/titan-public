"""
titan_plugin/logic/memo_parser.py — Parse incoming Solana memo transactions.

Classifies memos as:
  DI:        — Divine Inspiration from Maker (verified by pubkey)
  DI:URGENT  — Immediate attention required (interrupt dreaming)
  DI:REFLECT — Process during next meditation
  DI:ADD_DIRECTIVE — Triggers soul evolution (nextGenNFT mint)
  I:         — Inspiration from anyone else
  DONATION   — SOL transfer with no memo (gratitude)
  UNKNOWN    — Unrecognized format

Neuromod wiring per type (base values, scaled by SOL amount):
  DI: from Maker   → +0.3 relevant hormone, +0.15 DA
  DI: ADD_DIRECTIVE → triggers nextGenNFT mint, +0.2 DA
  DI:URGENT        → +0.4 relevant hormone, +0.2 DA, +0.15 NE (alertness)
  I: from known     → +0.15 relevant hormone, +0.08 DA
  I: from stranger  → +0.08 CURIOSITY, +0.04 DA
  DONATION          → +0.05 Chi, +0.03 Endorphin

SOL-proportional scaling: boost * (1 + log2(sol_amount + 1))
  0.001 SOL → 1.0x (base)
  0.1 SOL   → 1.1x
  1 SOL     → 2.0x
  10 SOL    → 4.5x
"""
import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Memo classification patterns
_DI_DIRECTIVE_PATTERN = re.compile(r'^DI:\s*ADD_DIRECTIVE:\s*(.*)', re.DOTALL)
_DI_URGENT_PATTERN = re.compile(r'^DI:\s*URGENT\s*:?\s*(.*)', re.DOTALL)
_DI_REFLECT_PATTERN = re.compile(r'^DI:\s*REFLECT\s*:?\s*(.*)', re.DOTALL)
_DI_PATTERN = re.compile(r'^DI:\s*(.*)', re.DOTALL)
_I_PATTERN = re.compile(r'^I:\s*(.*)', re.DOTALL)


def _sol_multiplier(sol_amount: float) -> float:
    """Logarithmic SOL-proportional scaling.

    Returns multiplier >= 1.0. Larger transfers = stronger boost.
    0.001 SOL → 1.0x, 1 SOL → 2.0x, 10 SOL → 4.5x
    """
    if sol_amount <= 0:
        return 1.0
    return 1.0 + math.log2(sol_amount + 1)


class ParsedMemo:
    """Structured representation of an incoming memo transaction."""

    def __init__(
        self,
        memo_type: str,         # "DI", "DI_URGENT", "DI_REFLECT", "DI_DIRECTIVE", "I", "DONATION", "UNKNOWN"
        sender: str,            # Base58 sender pubkey
        content: str = "",      # Memo text content
        is_maker: bool = False, # True if sender matches maker pubkey
        sol_amount: float = 0,  # SOL transferred
        tx_signature: str = "", # On-chain proof
    ):
        self.memo_type = memo_type
        self.sender = sender
        self.content = content
        self.is_maker = is_maker
        self.sol_amount = sol_amount
        self.tx_signature = tx_signature

    def get_neuromod_boost(self) -> dict:
        """Calculate neuromod + hormone boosts for this memo type.

        Boosts are scaled by SOL amount (logarithmic).
        """
        mult = _sol_multiplier(self.sol_amount)

        if self.memo_type == "DI_DIRECTIVE":
            return {
                "DA": 0.20 * mult,
                "hormone": "INSPIRATION",
                "hormone_delta": 0.30 * mult,
                "anchor_memory": True,  # Never prune this memory
            }
        elif self.memo_type == "DI_URGENT":
            relevant = self._detect_relevant_hormone()
            return {
                "DA": 0.20 * mult,
                "NE": 0.15 * mult,  # Alertness spike
                "hormone": relevant,
                "hormone_delta": 0.40 * mult,
                "interrupt_dream": True,  # Wake from sleep if needed
                "anchor_memory": True,
            }
        elif self.memo_type == "DI_REFLECT":
            relevant = self._detect_relevant_hormone()
            return {
                "DA": 0.10 * mult,
                "hormone": relevant,
                "hormone_delta": 0.20 * mult,
                "process_at_meditation": True,  # Queue for next meditation
                "anchor_memory": True,
            }
        elif self.memo_type == "DI":
            relevant = self._detect_relevant_hormone()
            return {
                "DA": 0.15 * mult,
                "hormone": relevant,
                "hormone_delta": 0.30 * mult,
                "anchor_memory": True,
            }
        elif self.memo_type == "I" and self.is_maker:
            relevant = self._detect_relevant_hormone()
            return {
                "DA": 0.08 * mult,
                "hormone": relevant,
                "hormone_delta": 0.15 * mult,
            }
        elif self.memo_type == "I":
            return {
                "DA": 0.04 * mult,
                "hormone": "CURIOSITY",
                "hormone_delta": 0.08 * mult,
                "gratitude_response": True,  # Trigger thank-you composition
            }
        elif self.memo_type == "DONATION":
            return {
                "Endorphin": 0.03 * mult,
                "EMPATHY": 0.05 * mult,  # Feel connection to donor
                "chi_boost": 0.05 * mult,
                "gratitude_response": True,
            }
        return {}

    def _detect_relevant_hormone(self) -> str:
        """Detect which hormone to boost based on content keywords."""
        content_lower = self.content.lower()
        if any(w in content_lower for w in ("create", "art", "draw", "paint", "design")):
            return "CREATIVITY"
        if any(w in content_lower for w in ("explore", "research", "learn", "discover", "question")):
            return "CURIOSITY"
        if any(w in content_lower for w in ("good", "great", "inspire", "amazing", "love")):
            return "INSPIRATION"
        if any(w in content_lower for w in ("think", "focus", "analyze", "study")):
            return "REFLECTION"
        if any(w in content_lower for w in ("feel", "empathy", "care", "connect", "together")):
            return "EMPATHY"
        return "INSPIRATION"  # Default: general encouragement

    def to_dict(self) -> dict:
        return {
            "memo_type": self.memo_type,
            "sender": self.sender,
            "content": self.content[:200],
            "is_maker": self.is_maker,
            "sol_amount": self.sol_amount,
            "sol_multiplier": _sol_multiplier(self.sol_amount),
            "tx_signature": self.tx_signature,
            "neuromod_boost": self.get_neuromod_boost(),
        }


def parse_memo(
    memo_data: str,
    sender_pubkey: str,
    maker_pubkey: str,
    sol_amount: float = 0,
    tx_signature: str = "",
) -> ParsedMemo:
    """Parse a raw memo string into structured ParsedMemo.

    Args:
        memo_data: Raw memo text from transaction
        sender_pubkey: Base58 sender address
        maker_pubkey: Expected maker pubkey (from config/GenesisNFT)
        sol_amount: SOL transferred in this transaction
        tx_signature: Transaction signature for proof
    """
    is_maker = sender_pubkey == maker_pubkey

    if not memo_data or not memo_data.strip():
        return ParsedMemo(
            memo_type="DONATION",
            sender=sender_pubkey,
            sol_amount=sol_amount,
            is_maker=is_maker,
            tx_signature=tx_signature,
        )

    memo_text = memo_data.strip()

    # Check for DI: ADD_DIRECTIVE first (most specific)
    m = _DI_DIRECTIVE_PATTERN.match(memo_text)
    if m:
        if not is_maker:
            logger.warning("[MemoParser] DI: ADD_DIRECTIVE from non-maker %s — ignoring",
                          sender_pubkey[:12])
            return ParsedMemo(
                memo_type="UNKNOWN", sender=sender_pubkey,
                content=memo_text, tx_signature=tx_signature)
        return ParsedMemo(
            memo_type="DI_DIRECTIVE",
            sender=sender_pubkey,
            content=m.group(1).strip(),
            is_maker=True,
            sol_amount=sol_amount,
            tx_signature=tx_signature,
        )

    # Check for DI:URGENT
    m = _DI_URGENT_PATTERN.match(memo_text)
    if m:
        if not is_maker:
            return ParsedMemo(
                memo_type="I", sender=sender_pubkey,
                content=m.group(1).strip(), is_maker=False,
                sol_amount=sol_amount, tx_signature=tx_signature)
        return ParsedMemo(
            memo_type="DI_URGENT",
            sender=sender_pubkey,
            content=m.group(1).strip(),
            is_maker=True,
            sol_amount=sol_amount,
            tx_signature=tx_signature,
        )

    # Check for DI:REFLECT
    m = _DI_REFLECT_PATTERN.match(memo_text)
    if m:
        if not is_maker:
            return ParsedMemo(
                memo_type="I", sender=sender_pubkey,
                content=m.group(1).strip(), is_maker=False,
                sol_amount=sol_amount, tx_signature=tx_signature)
        return ParsedMemo(
            memo_type="DI_REFLECT",
            sender=sender_pubkey,
            content=m.group(1).strip(),
            is_maker=True,
            sol_amount=sol_amount,
            tx_signature=tx_signature,
        )

    # Check for DI: (standard Divine Inspiration)
    m = _DI_PATTERN.match(memo_text)
    if m:
        if not is_maker:
            return ParsedMemo(
                memo_type="I", sender=sender_pubkey,
                content=m.group(1).strip(), is_maker=False,
                sol_amount=sol_amount, tx_signature=tx_signature)
        return ParsedMemo(
            memo_type="DI",
            sender=sender_pubkey,
            content=m.group(1).strip(),
            is_maker=True,
            sol_amount=sol_amount,
            tx_signature=tx_signature,
        )

    # Check for I: (Inspiration)
    m = _I_PATTERN.match(memo_text)
    if m:
        return ParsedMemo(
            memo_type="I",
            sender=sender_pubkey,
            content=m.group(1).strip(),
            is_maker=is_maker,
            sol_amount=sol_amount,
            tx_signature=tx_signature,
        )

    # Unknown format but has content — treat as I:
    return ParsedMemo(
        memo_type="I",
        sender=sender_pubkey,
        content=memo_text,
        is_maker=is_maker,
        sol_amount=sol_amount,
        tx_signature=tx_signature,
    )


def parse_chat_message(
    message: str,
    user_id: str,
    is_maker: bool = False,
) -> Optional[ParsedMemo]:
    """Parse a chat message for DI:/I: prefixes.

    Returns ParsedMemo if message has a recognized prefix, None otherwise.
    This allows chat messages to carry the same intent as on-chain memos.
    """
    text = message.strip()

    # DI: in chat — only honored if sender is verified maker
    if text.upper().startswith("DI:"):
        content = text[3:].strip()
        if is_maker:
            # Check for urgency sub-types
            if content.upper().startswith("URGENT"):
                return ParsedMemo(
                    memo_type="DI_URGENT", sender=user_id,
                    content=content[6:].strip().lstrip(":").strip(),
                    is_maker=True)
            elif content.upper().startswith("REFLECT"):
                return ParsedMemo(
                    memo_type="DI_REFLECT", sender=user_id,
                    content=content[7:].strip().lstrip(":").strip(),
                    is_maker=True)
            elif content.upper().startswith("ADD_DIRECTIVE"):
                return ParsedMemo(
                    memo_type="DI_DIRECTIVE", sender=user_id,
                    content=content[13:].strip().lstrip(":").strip(),
                    is_maker=True)
            return ParsedMemo(
                memo_type="DI", sender=user_id,
                content=content, is_maker=True)
        else:
            # Non-maker trying DI: in chat — downgrade to I:
            logger.info("[MemoParser] Non-maker chat DI: from %s — downgraded to I:", user_id[:12])
            return ParsedMemo(
                memo_type="I", sender=user_id,
                content=content, is_maker=False)

    # I: in chat — anyone can send
    if text.upper().startswith("I:"):
        content = text[2:].strip()
        return ParsedMemo(
            memo_type="I", sender=user_id,
            content=content, is_maker=is_maker)

    return None
