"""
Outbound PII Sanitizer for Titan.
Strips personally identifiable information from text before sending to external LLM APIs.
Pure regex — no ML, no new dependencies.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Solana data patterns to NEVER strip (allowlist)
_SOLANA_BASE58 = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')

# PII patterns
_PATTERNS = {
    "email": (
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "[EMAIL_REDACTED]"
    ),
    "phone": (
        re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "[PHONE_REDACTED]"
    ),
    "ssn": (
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "[SSN_REDACTED]"
    ),
    "credit_card": (
        re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        "[CARD_REDACTED]"
    ),
    "ip_address": (
        re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
        "[IP_REDACTED]"
    ),
}


def sanitize_outbound(text: str, active_patterns: list = None) -> tuple[str, int]:
    """
    Strip PII from text before sending to external APIs.

    Args:
        text: The text to sanitize.
        active_patterns: List of pattern names to apply. None = all patterns.

    Returns:
        Tuple of (sanitized_text, redaction_count).
    """
    if not text:
        return text, 0

    patterns_to_use = active_patterns or list(_PATTERNS.keys())
    redaction_count = 0

    for name in patterns_to_use:
        if name not in _PATTERNS:
            continue
        pattern, replacement = _PATTERNS[name]

        # Find matches but skip Solana base58 addresses
        def _replace_if_not_solana(match, _replacement=replacement):
            nonlocal redaction_count
            matched = match.group()
            # Check if this looks like a Solana address (base58, 32-44 chars)
            if _SOLANA_BASE58.fullmatch(matched):
                return matched  # Don't redact
            redaction_count += 1
            return _replacement

        text = pattern.sub(_replace_if_not_solana, text)

    if redaction_count > 0:
        logger.info("[Privacy] Redacted %d PII item(s) from outbound text.", redaction_count)

    return text, redaction_count
