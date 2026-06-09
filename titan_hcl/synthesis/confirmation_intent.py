"""titan_hcl.synthesis.confirmation_intent — deterministic confirm/dispute detector.

EEL Pillar A / Phase A2 (RFP_emergent_experience_learning §7.A2; INV-EEL-2/6):
the *next* user turn after a STATE_NEED_RESEARCH answer is classified — cheaply,
deterministically — into {confirm, dispute, neutral}. This classification IS the
research oracle (INV-EEL-2): it gates whether a researched fact promotes to
durable memory. It is NEVER the LLM's self-claim — only the user's own reaction
counts.

Same cheap-regex pattern as `synthesis/tool_intent.py`: pure regex, no model
call, runs only when a pending research confirmation exists for the user → no
measurable cost.
"""
from __future__ import annotations

import re

# Explicit dispute of / request to re-do the prior researched answer.
_DISPUTE_RE = re.compile(
    r"\b(wrong|incorrect|inaccurate|not (?:right|correct|true|it|quite)|"
    r"that'?s (?:wrong|incorrect|not (?:right|it))|search (?:more|again)|"
    r"look (?:it up )?again|try again|doesn'?t (?:seem|look) right|"
    r"out[ -]?dated|out[ -]of[ -]date|stale|disagree|"
    r"that'?s not (?:it|right|true))\b",
    re.IGNORECASE,
)
# Leading "no"/"nope" that is a dispute (excludes benign "no problem/worries/thanks").
_LEADING_NO_RE = re.compile(
    r"^\s*(?:no|nope)\b(?!\s+(?:problem|worries|thanks|thank|big deal))",
    re.IGNORECASE,
)

# Explicit positive acknowledgement of the prior answer.
_CONFIRM_RE = re.compile(
    r"\b(thanks|thank you|thx|cheers|correct|exactly|perfect|great|"
    r"that'?s (?:it|right|correct|helpful)|yep|yeah|"
    r"got it|makes sense|helpful|appreciate|spot on|accurate|"
    r"good (?:answer|to know)|that helps|nice one)\b",
    re.IGNORECASE,
)
_BARE_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)


def detect_confirmation(user_turn: str) -> str:
    """Classify the user's reaction to the prior researched answer.

    Returns "dispute" | "confirm" | "neutral". Dispute is tested FIRST —
    an explicit "no, that's wrong / search again" must outweigh an incidental
    "thanks" in the same message (don't promote a fact the user is rejecting).
    """
    text = user_turn or ""
    if _DISPUTE_RE.search(text) or _LEADING_NO_RE.search(text):
        return "dispute"
    if _CONFIRM_RE.search(text) or _BARE_YES_RE.search(text):
        return "confirm"
    return "neutral"
