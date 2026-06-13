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


# Default volatile half-life in Titan emergent epochs (~417 ≈ 1 human-hour at the
# ~10k-epochs/day cognitive tick). Config `[synthesis.research].volatile_lifetime
# _epochs` overrides per-Titan; per-category overrides ride the classifier marker
# class. RFP §7.D-knowledge M1 / the RESEARCH LIFECYCLE Axis-1 decay.
DEFAULT_VOLATILE_LIFETIME_EPOCHS = 417


def age_epochs(created_epoch: float, now_epochs: float) -> float:
    """Age in Titan emergent epochs = ``now_epochs − created_epoch``. A finding/
    concept with `created_epoch` ≤ 0 (legacy / unstamped) returns ``0.0`` — the
    M0 grandfather signal (uncomputable age → treated as fresh, never TTL'd).
    Clamped at 0 (a clock that went backwards never yields a negative age)."""
    try:
        ce = float(created_epoch or 0.0)
        ne = float(now_epochs or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if ce <= 0.0 or ne <= 0.0:
        return 0.0
    return max(0.0, ne - ce)


def is_stale(created_epoch: float, now_epochs: float,
             lifetime_epochs: float = DEFAULT_VOLATILE_LIFETIME_EPOCHS) -> bool:
    """A volatile finding/concept is STALE once its emergent age reaches its
    `lifetime_epochs` half-life. Grandfathered rows (`created_epoch` ≤ 0) are
    NEVER stale (age_epochs → 0). Shared by the DK.3 idle-pass TTL (M2) AND the
    M1 mempool epoch-prune (and Phase E.2's hot-path cache when it lands)."""
    lt = float(lifetime_epochs or DEFAULT_VOLATILE_LIFETIME_EPOCHS)
    if lt <= 0.0:
        return False
    return age_epochs(created_epoch, now_epochs) >= lt


def freshness_weight(created_epoch: float, now_epochs: float,
                     lifetime_epochs: float = DEFAULT_VOLATILE_LIFETIME_EPOCHS
                     ) -> float:
    """Gradual decay multiplier ``max(0, 1 − age/lifetime)`` ∈ [0,1] for a
    volatile node's recall weight (M1). 1.0 at birth → 0.0 at the half-life
    (where the epoch-prune gate fires). Grandfathered rows → 1.0 (never decays)."""
    lt = float(lifetime_epochs or DEFAULT_VOLATILE_LIFETIME_EPOCHS)
    if lt <= 0.0:
        return 1.0
    a = age_epochs(created_epoch, now_epochs)
    if a <= 0.0:
        return 1.0
    return max(0.0, 1.0 - a / lt)


__all__ = (
    "classify_volatility", "is_volatile",
    "DEFAULT_VOLATILE_LIFETIME_EPOCHS",
    "age_epochs", "is_stale", "freshness_weight",
)
