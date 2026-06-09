"""goal_class — deterministic outcome/task-shape labeler for EEL Pillar B1.

`RFP_emergent_experience_learning.md §7.B1 / §1.0` (INV-EEL-8). A skill is keyed
on its OUTCOME `(oracle_id, goal_class)` and is a policy over the TASK-SHAPES
that reach it. This module computes the two free-text-derived labels of those
keys, deterministically and cheaply (regex/keyword only — NO LLM, NO embedding;
INV-EEL-1), so the same goal always maps to the same key (a stable key is what
lets one skill generalize across entities; INV-EEL-8).

  goal_class(goal)            → the parameterized goal-CLASS slug (e.g. "defi-lookup")
  make_task_shape(...)        → the means signature "(task_type|tool_id|domain_hint)"

**Outcome vs means (the §1.0 orthogonality):** `goal_class` is the WHAT (the
goal-class); the entity/args ride as parameters and are NEVER part of it — so
"TVL of Jupiter" and "TVL of Uniswap" share one `goal_class` ("defi-lookup") and
therefore one skill. Keying on the literal goal string is the anti-pattern that
re-creates Problem 2 (skills never form) — this module exists to prevent it.

**`oracle_id` already carries the verification domain** (`coding_sandbox` /
`web_api_oracle` / `solana_oracle` …, the first half of the outcome key), so
`goal_class` deliberately stays a coarse `domain-action` intent label rather than
a fine-grained topic — **broad on purpose** (the RFP's "bootstrap broad; let the
synthesis-reduction over {B_i,c,time_cost} refine"; INV-EEL-4). The domain vocab
is the LIVE tool/oracle surface (defi/onchain/web/market/code/social/time/
security), supplemented by the Engram-content classifier `derive_domain_hint`
(which is tuned for thought domains, not tool goals — verified 2026-06-09) and
folded through the SHARED `_normalize_domain_hint` so labels stay byte-consistent
with the rest of synthesis. This is the one residual fuzzy edge of the outcome
key (RFP §5) — pinned here, deterministic + reproducible.
"""
from __future__ import annotations

import re
from typing import Optional

from titan_hcl.synthesis.consolidation_defaults import (
    _normalize_domain_hint,
    derive_domain_hint,
)

# ── Action vocabulary (the verb half of goal_class) ──────────────────────────
# Ordered, first-match-wins, most-specific → general. Each entry maps a set of
# surface keywords to ONE canonical action. The action is the intent ("what is
# being done to the goal"), orthogonal to the tool used (the task-shape).
_ACTION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("verify", "validate", "confirm", "check that", "is it true", "double-check",
      "double check"), "verify"),
    (("compute", "calculate", "what is the result", "evaluate", "hash of",
      "factorial", "how many", "sum of", "multiply"), "compute"),
    (("convert", "translate", "in usd", "to usd", "exchange rate"), "convert"),
    (("summarize", "summary", "tl;dr", "tldr", "recap"), "summarize"),
    (("analyze", "analysis", "compare", "diff between", "trend", "correlate"),
     "analyze"),
    (("post", "tweet", "reply to", "publish", "share on"), "post"),
    (("search", "research", "look into", "find out about", "dig into"), "search"),
    (("fetch", "get the", "retrieve", "pull the", "download"), "fetch"),
    # The broadest informational bucket — questions / "what is X" / value reads.
    (("tvl", "price", "balance", "how much", "what is", "what's", "what are",
      "who is", "when", "where", "lookup", "look up", "value of", "current"),
     "lookup"),
)
_DEFAULT_ACTION = "query"

# ── Tool-goal domain vocabulary (supplements derive_domain_hint, which is tuned
# for Engram CONTENT, not tool goals — verified 2026-06-09). First-match-wins,
# most-specific → general. Kept small + grounded in Titan's live oracle/tool
# surface; unknown → "" → goal_class falls back to "general". ────────────────
_TOOL_DOMAIN_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tvl", "defi", "liquidity", "yield", "apy", "protocol", "dex",
      "jupiter", "uniswap", "aave"), "defi"),
    (("solana", "onchain", "on-chain", "blockchain", "transaction", "tx hash",
      "signature", "wallet", "pubkey", "devnet", "mainnet", "rpc"), "onchain"),
    (("price", "usd", "market cap", "marketcap", "ticker", "stock", "token price",
      "worth"), "market"),
    (("code", "python", "function", "script", "compile", "algorithm", "regex",
      "sandbox", "stdout"), "code"),
    (("weather", "time", "date", "utc", "timezone", "today's", "current time"),
     "time"),
    (("tweet", "twitter", " x ", "follower", "mention", "social", "reply"),
     "social"),
    (("news", "headline", "article", "latest on", "recent events"), "web"),
)

# Preserve underscores so real identifiers (tool_ids like `coding_sandbox`,
# domain labels like `philosophy_of_mind`) survive verbatim in the signature.
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slug(text: str) -> str:
    """Lowercase, collapse runs of non-[a-z0-9_] to single dashes, strip ends."""
    return _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")


def _match_first(blob: str, rules: tuple[tuple[tuple[str, ...], str], ...]) -> str:
    for keywords, label in rules:
        if any(kw in blob for kw in keywords):
            return label
    return ""


def _action_of(blob: str) -> str:
    return _match_first(blob, _ACTION_RULES) or _DEFAULT_ACTION


def _domain_of(blob: str) -> str:
    """Domain for a tool-call goal. derive_domain_hint first (it confidently
    covers code/security/the thought domains), then the tool-goal supplement,
    then "" (→ caller falls back to "general"). All folded identically."""
    d = derive_domain_hint(blob)  # already _normalize_domain_hint-folded
    if d:
        return d
    return _normalize_domain_hint(_match_first(blob, _TOOL_DOMAIN_RULES))


def goal_class(goal: str) -> str:
    """Map a free-text goal → a stable, parameterized goal-CLASS slug.

    Deterministic + reproducible (same goal → same class), cheap (keyword only,
    no LLM/embedding — INV-EEL-1). The entity/args are NOT extracted — they ride
    as parameters, never in the key (INV-EEL-8). Returns "{domain}-{action}"
    (e.g. "defi-lookup", "code-compute") or "general-{action}" when no domain
    keyword fires. Empty goal → "general-query" (a valid catch-all class).
    """
    blob = (goal or "").strip().lower()
    if not blob:
        return "general-query"
    domain = _domain_of(blob) or "general"
    action = _action_of(blob)
    return _slug(f"{domain}-{action}")


def make_task_shape(
    task_type: str,
    tool_id: str,
    domain_hint: str = "",
) -> str:
    """The MEANS signature of a task — a skill cell's column key (§1.0).

    `(task_type|tool_id|domain_hint)`: the how, individuated by the tool-path,
    scored skill-relative. Cheap O(1), no embedding (INV-EEL-1). domain_hint is
    folded for consistency; empty parts collapse cleanly.
    """
    parts = [
        _slug(task_type or "unknown"),
        _slug(tool_id or "unknown"),
        _normalize_domain_hint(domain_hint or ""),
    ]
    return "|".join(p for p in parts if p)


def task_shape_for_goal(task_type: str, tool_id: str, goal: str) -> str:
    """Convenience: derive the task-shape directly from a goal, reusing the same
    domain classifier `goal_class` uses (so the task-shape's domain_hint and the
    outcome's goal_class draw from one source of truth)."""
    blob = (goal or "").strip().lower()
    return make_task_shape(task_type, tool_id, _domain_of(blob) if blob else "")
