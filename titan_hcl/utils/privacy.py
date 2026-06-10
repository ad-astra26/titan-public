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


# ── Public-surface sanitizer (RFP_titan_authored_soul_diary §7.P6 / INV-SD-3) ──
# A FAIL-CLOSED denylist for any PUBLIC soul-diary surface (X post, NFT excerpts,
# archive page): no raw path / IP / hostname / key / PID / topology may leave the
# private record. It reuses the PII patterns above + the Solana-pubkey allowlist
# (public chain identity is KEPT), and adds the infra/topology classes that reach
# an entry through the §P5 self-inspection summary (journal error lines and
# `file.py:NN` code refs — the realistic leak vector). The verification hash is
# taken over the *private* entry; this only shapes what is shown publicly. The
# private chronicle / ledger / self-memory are NEVER altered.
_PUBLIC_PATTERNS = {
    # absolute filesystem paths under known roots (+ optional :line). Anchored to
    # real roots so ordinary prose ("and/or", "duck/faiss") is not eaten and the
    # public example.com URL (no leading FS root) survives.
    "fs_path": (
        re.compile(r'/(?:home|tmp|var|etc|usr|opt|root|mnt|proc|sys|data|srv|run)'
                   r'(?:/[\w.\-]+)*(?::\d+)?'),
        "[PATH_REDACTED]"),
    # home-relative paths — ~/.titan/microkernel_T1.toml
    "home_path": (re.compile(r'~/[\w./\-]+(?::\d+)?'), "[PATH_REDACTED]"),
    # known relative source/data roots — titan_hcl/core/soul.py, data/foo.json
    "repo_path": (
        re.compile(r'\b(?:titan_hcl|titan3|titan-observatory|scripts|programs|tests'
                   r'|data)/[\w./\-]+(?::\d+)?'),
        "[PATH_REDACTED]"),
    # bare python code refs — infra_inspect.py:16, soul.py
    "code_ref": (re.compile(r'\b[\w\-]+\.py(?::\d+)?\b'), "[CODE_REDACTED]"),
    # systemd unit names — titan-T1.service
    "service_unit": (re.compile(r'\btitan-[\w\-]+\.service\b', re.IGNORECASE),
                     "[SERVICE_REDACTED]"),
    # the VPS username (host identity)
    "username": (re.compile(r'\bantigravity\b', re.IGNORECASE), "[USER_REDACTED]"),
    # process ids — pid 1234 / PID: 1234 / pid=1234
    "pid": (re.compile(r'\b[Pp][Ii][Dd]s?\b[\s:=#]*\d+'), "[PID_REDACTED]"),
    # long hex (sha256 / hex secret); base58 Solana pubkeys are non-hex so they
    # never match here (and the allowlist guards the rare all-[1-9a-f] case).
    "hex_secret": (re.compile(r'\b[0-9a-fA-F]{32,}\b'), "[HASH_REDACTED]"),
    # labeled secrets/keys — sk-..., api_key=..., internal_key: ...
    "labeled_secret": (
        re.compile(r'\b(?:sk|api[_-]?key|secret|token|internal[_-]?key)'
                   r'[\s:=\-]+[A-Za-z0-9_\-]{12,}\b', re.IGNORECASE),
        "[KEY_REDACTED]"),
}

# Application order: paths (incl. the optional :line) BEFORE the bare `.py`
# code-ref, so a full `titan_hcl/core/soul.py:16` collapses to ONE
# [PATH_REDACTED] rather than splitting into PATH + CODE.
_PUBLIC_ORDER = ("fs_path", "home_path", "repo_path", "code_ref", "service_unit",
                 "username", "pid", "hex_secret", "labeled_secret")
# PII classes a public surface must also strip (email/phone/ssn/card/ip).
_PUBLIC_PII_ORDER = ("email", "phone", "ssn", "credit_card", "ip_address")


def sanitize_for_public(text: str) -> tuple[str, int]:
    """Fail-closed sanitize ``text`` for a PUBLIC soul-diary surface (INV-SD-3).

    Default-denies paths / IPs / hostnames / keys / PIDs / topology while
    preserving reflective prose; public Solana pubkeys (chain identity) are kept
    via the base58 allowlist. Returns ``(sanitized, redaction_count)`` — a
    nonzero count on any public payload is the G9 tripwire.
    """
    if not text:
        return text, 0

    count = 0

    def _apply(pattern, replacement, s):
        def _replace(match, _rep=replacement):
            nonlocal count
            matched = match.group()
            if _SOLANA_BASE58.fullmatch(matched):
                return matched          # keep public Solana pubkeys (chain identity)
            count += 1
            return _rep
        return pattern.sub(_replace, s)

    for name in _PUBLIC_PII_ORDER:
        pattern, replacement = _PATTERNS[name]
        text = _apply(pattern, replacement, text)
    for name in _PUBLIC_ORDER:
        pattern, replacement = _PUBLIC_PATTERNS[name]
        text = _apply(pattern, replacement, text)

    if count > 0:
        logger.info("[Privacy] sanitize_for_public redacted %d token(s) "
                    "from a public surface.", count)
    return text, count
