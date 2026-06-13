"""research_volatility — DK Axis-1 deterministic durability classifier.

RFP_synthesis_self_learning_meta_reasoning §7.D-knowledge (RESEARCH LIFECYCLE,
Maker-locked 2026-06-13). Splits a confirmed research finding into:

- **volatile** — a fresh value of a queried parameter ("current SOL price",
  "today's news", "how many X now"). Must NOT become a permanent anchored
  declarative concept (anchoring a price quote as a permanent Idea violates
  FC-3 / INV-OML-6). Stays as a decaying memory node, pruned when stale.
- **durable** — stable conceptual knowledge ("what *is* spirit", "how Solana
  consensus works", definitions, history). Becomes an anchored declarative
  `Engram` (DK.1).

DETERMINISTIC by design (markers + domain + the INV-EEL-8 "current value of
<entity-param>" shape) — NOT an LLM judgment, which would re-open the
librarian-vs-author line and add nondeterminism. The research SKILL (which
source answered it, DK.5) is captured for BOTH classes — only the *result*
durability differs here.

Pure + side-effect-free → unit-testable in isolation; importable by both
memory_worker (the discern gate) and synthesis (DK.3 lint) with no heavy deps.
"""
from __future__ import annotations

import re

# Temporal volatility markers — the answer is "as of now" and goes stale.
_TEMPORAL_MARKERS = (
    "current", "currently", "latest", "now", "today", "tonight", "as of",
    "live", "right now", "at the moment", "this week", "this month",
    "this year", "recent", "recently", "up to date", "up-to-date", "ongoing",
    "breaking", "just announced", "so far",
)

# Quantitative-volatile markers — a number/measure that moves over time.
_QUANTITATIVE_MARKERS = (
    "price", "prices", "rate", "rates", "count", "how many", "how much",
    "balance", "market cap", "marketcap", "ranking", "rank", "trending",
    "score", "stock", "exchange rate", "valuation", "volume", "supply",
    "percentage of", "number of", "population of", "temperature", "weather",
    "forecast", "odds", "standings",
)

# Domain overrides — inherently volatile (real-time) vs inherently evergreen.
# `domain_hint` (§6.2.4) is the consolidation-LLM's coarse advisory label.
_VOLATILE_DOMAINS = frozenset((
    "market", "markets", "finance", "crypto", "cryptocurrency", "news",
    "weather", "sports", "prices", "trading", "stocks",
))
_EVERGREEN_DOMAINS = frozenset((
    "mathematics", "math", "philosophy", "philosophy_of_mind", "history",
    "physics", "chemistry", "biology", "neuroscience", "self", "definition",
    "definitions", "theory", "logic",
))


def classify_volatility(text: str, domain_hint: str = "") -> str:
    """Return ``"volatile"`` or ``"durable"`` for a research finding.

    Order of precedence (most decisive first):
      1. Domain override — a volatile domain → volatile; an evergreen domain →
         durable (the curated domain label is a stronger signal than lexical).
      2. Lexical markers — any temporal or quantitative-volatile marker in the
         text → volatile.
      3. Default → durable (no volatility signal = a stable "what is X" fact).

    Deterministic + cheap (lowercase substring scan). `text` = the question +
    finding (user_prompt + agent_response is the natural input)."""
    dom = (domain_hint or "").strip().lower()
    if dom:
        if dom in _VOLATILE_DOMAINS:
            return "volatile"
        if dom in _EVERGREEN_DOMAINS:
            return "durable"

    blob = (text or "").lower()
    if not blob.strip():
        return "durable"  # nothing to judge → keep (conservative; durable)

    # Word-boundary-ish scan for single tokens; substring for phrases.
    for marker in _TEMPORAL_MARKERS + _QUANTITATIVE_MARKERS:
        if " " in marker:
            if marker in blob:
                return "volatile"
        elif re.search(r"\b" + re.escape(marker) + r"\b", blob):
            return "volatile"
    return "durable"


def is_volatile(text: str, domain_hint: str = "") -> bool:
    return classify_volatility(text, domain_hint) == "volatile"


__all__ = ("classify_volatility", "is_volatile")
