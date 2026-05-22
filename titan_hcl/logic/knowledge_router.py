"""
knowledge_router — single canonical entry for all external-knowledge queries.

Classifies a query into one of seven types and resolves a per-type preferred
backend chain. Caching, budgets, error taxonomy, decision logging and smart
routing plug in on top in later KP phases.

See: titan-docs/rFP_knowledge_pipeline_v2.md §3.1.

KP-0 scope: pure classification + normalization + hash + backend-chain
resolution. No network, no cache, no fetch — those land in KP-1..KP-8.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import List, Optional


class QueryType(str, Enum):
    """Seven mutually-exclusive query types per rFP §3.1.

    Values are stable strings so they survive serialization to the cache DB
    and the decision log without coupling to integer ordinals.
    """
    DICTIONARY = "dictionary"              # single word lookup
    DICTIONARY_PHRASE = "dictionary_phrase"  # "{word} meaning" / "{word} definition"
    WIKIPEDIA_LIKE = "wikipedia_like"      # 2-4 word encyclopedic noun phrase
    CONCEPTUAL = "conceptual"              # 3+ word abstract/strategy query
    TECHNICAL = "technical"                # programming / system keywords
    NEWS = "news"                          # current events / time-sensitive
    INTERNAL_REJECTED = "internal_rejected"  # Titan-internal names, skip externally


# ── Classification lexicons ──────────────────────────────────────────

# Order matters: classify_query runs detection top-down, first match wins.

_NEWS_MARKERS = frozenset({
    "today", "tonight", "news", "latest", "breaking", "recent", "recently",
    "now", "current", "headlines", "live", "yesterday",
})

_TECH_MARKERS = frozenset({
    "python", "javascript", "typescript", "async", "await", "asyncio",
    "sql", "sqlite", "postgres", "mysql", "redis", "mongodb", "neo4j",
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
    "rust", "golang", "go", "java", "kotlin", "swift", "c++", "c#",
    "react", "vue", "svelte", "angular", "nextjs", "next.js",
    "api", "http", "https", "rest", "graphql", "websocket", "grpc",
    "json", "yaml", "xml", "csv", "protobuf",
    "bug", "error", "exception", "stacktrace", "traceback", "syntax",
    "compile", "linker", "deadlock", "race condition", "memory leak",
    "regex", "regexp", "unicode", "utf-8",
})

_DEFINITION_MARKERS = frozenset({
    "meaning", "definition", "define", "etymology", "definition of",
    "meaning of",
})

_ABSTRACT_MARKERS = frozenset({
    "how", "why", "what", "when", "where", "who",  # interrogatives
    "strategy", "strategies", "approach", "method", "methodology",
    "problem", "solution", "hypothesis", "theory", "framework",
    "critical", "cognitive", "emotional", "psychological",
    "generation", "thinking", "reasoning", "analysis", "synthesis",
})

# Stopwords that shouldn't themselves count toward content-word analysis.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "for", "with", "and", "or",
    "but", "to", "from", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "it",
    "this", "that", "these", "those",
})


# ── Public helpers ───────────────────────────────────────────────────

def normalize_query(topic: str) -> str:
    """Canonicalize a query string for hashing + classification.

    Lower-cases, strips, collapses internal whitespace to single spaces.
    Keeps meaningful characters (letters, digits, hyphens, underscores,
    apostrophes). Empty/None → "".
    """
    if not topic:
        return ""
    s = topic.strip().lower()
    # Collapse any run of whitespace to a single space
    s = re.sub(r"\s+", " ", s)
    return s


def query_hash(normalized: str, qt: QueryType, backend: str) -> str:
    """Stable sha256 hash used as cache key.

    Combines normalized query + query type + backend so the same topic
    routed to different backends caches independently (rFP §3.2 schema).
    """
    payload = f"{normalized}\x1f{qt.value}\x1f{backend}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Query-type classifier ────────────────────────────────────────────

_INTERNAL_DOT_RE = re.compile(r"^[A-Za-z][\w]*\.[A-Za-z_]+$")


def _is_titan_internal(topic_raw: str, normalized: str) -> bool:
    """Stage-A internal-name heuristics (now owned by router entry).

    Catches the specific Titan-internal code patterns that leak from
    primitive-output pipelines and waste bandwidth externally:

      * underscore-no-space:   inner_spirit, outer_perception, self_reasoning
      * primitive.submode:     FORMULATE.load_wisdom, META.state_audit
      * ≤ 2 chars:             DA, NE (neuromod codes; too short for search)
      * 3 chars mixed-case:    ACh (neuromod), 5HT (has digit)
      * 3 chars all-upper:     CGN, MSL, DNA, API (acronyms/subsystem codes)
      * 3 chars with digit:    5HT, 6PO

    3-char all-lowercase-alpha words (own, chi, ...) PASS — they are valid
    dictionary queries even if "chi" is ambiguous with the internal chi
    concept. Internal code emits "chi_total"/"chi_coherence" underscored
    which fall under the underscore-no-space rule.
    """
    if not normalized:
        return False
    # ≤ 2 chars → too short for any external query
    if len(normalized) <= 2:
        return True
    # underscore-with-no-space
    if "_" in topic_raw and " " not in topic_raw:
        return True
    # primitive.submode (single dotted identifier)
    if _INTERNAL_DOT_RE.match(topic_raw.strip()):
        return True
    # 3-char tokens: reject if not all-lowercase-alpha (catches acronym
    # codes like CGN/MSL and mixed-case like ACh and digit-containing 5HT)
    if len(normalized) == 3 and " " not in normalized:
        tr = topic_raw.strip()
        if tr != tr.lower() or any(c.isdigit() for c in tr):
            return True
    return False


def _tokens(normalized: str) -> List[str]:
    """Split on whitespace; normalize already lowercased + collapsed."""
    return [t for t in normalized.split(" ") if t]


def _is_single_alpha_word(tokens: List[str]) -> bool:
    return (len(tokens) == 1
            and 3 <= len(tokens[0]) <= 20
            and tokens[0].isalpha())


def _has_marker(tokens: List[str], markers: frozenset) -> bool:
    """True if any token (or bigram) appears in the marker set."""
    token_set = set(tokens)
    if token_set & markers:
        return True
    # Check 2-word markers (e.g. "definition of")
    if any(" " in m for m in markers):
        joined = " ".join(tokens)
        for m in markers:
            if " " in m and m in joined:
                return True
    return False


def classify_query(topic: str) -> QueryType:
    """Map a raw topic string to a QueryType (rFP §3.1, Layer 1).

    Detection precedence (first match wins):
      1. INTERNAL_REJECTED — Titan-internal names
      2. NEWS — time-sensitive markers
      3. TECHNICAL — programming / system markers
      4. DICTIONARY_PHRASE — "{word} meaning/definition"
      5. DICTIONARY — single alpha word, 3-20 chars
      6. WIKIPEDIA_LIKE — 2-4 noun-phrase words without abstract markers
      7. CONCEPTUAL — default for any multi-word query
    """
    topic_raw = topic or ""
    normalized = normalize_query(topic_raw)
    if not normalized:
        return QueryType.INTERNAL_REJECTED

    # Stage 1 — Titan-internal guard (exact Stage A heuristics)
    if _is_titan_internal(topic_raw, normalized):
        return QueryType.INTERNAL_REJECTED

    tokens = _tokens(normalized)

    # Stage 2 — news markers before anything else so "latest python async"
    # classifies as news rather than technical. The news backend still has
    # the full query to work with.
    if _has_marker(tokens, _NEWS_MARKERS):
        return QueryType.NEWS

    # Stage 3 — technical markers
    if _has_marker(tokens, _TECH_MARKERS):
        return QueryType.TECHNICAL

    # Stage 4 — dictionary phrase: "{word} meaning" / "{word} definition"
    if (2 <= len(tokens) <= 3
            and tokens[-1] in _DEFINITION_MARKERS):
        return QueryType.DICTIONARY_PHRASE

    # Stage 5 — single-word dictionary lookup
    if _is_single_alpha_word(tokens):
        return QueryType.DICTIONARY

    # Stage 6 — wikipedia-like: 2-4 tokens, no stopwords, no abstract markers
    if 2 <= len(tokens) <= 4:
        has_stop = any(t in _STOPWORDS for t in tokens)
        has_abstract = any(t in _ABSTRACT_MARKERS for t in tokens)
        if not has_stop and not has_abstract:
            return QueryType.WIKIPEDIA_LIKE

    # Stage 7 — default: conceptual multi-word
    return QueryType.CONCEPTUAL


# ── Backend-chain resolver ───────────────────────────────────────────

# Per rFP §3.1 table: ordered preference chain per query type.
# Backend names are stable strings; actual implementations land in KP-1.
# Last entry in each chain is the fallback of last resort.

_BACKEND_CHAINS = {
    QueryType.DICTIONARY: [
        "wiktionary", "free_dictionary", "wikipedia_direct",
    ],
    QueryType.DICTIONARY_PHRASE: [
        "wiktionary", "wikipedia_direct",
    ],
    QueryType.WIKIPEDIA_LIKE: [
        "wikipedia_direct", "searxng_wikipedia",
    ],
    QueryType.CONCEPTUAL: [
        "searxng_ddg_brave_wiki", "searxng_google_bing",
    ],
    QueryType.TECHNICAL: [
        "searxng_ddg_stackoverflow", "searxng_ddg_brave_wiki",
    ],
    QueryType.NEWS: [
        "news_api", "searxng_news",
    ],
    QueryType.INTERNAL_REJECTED: [],  # caller should short-circuit
}


def route(topic: str, qt: Optional[QueryType] = None) -> List[str]:
    """Return the ordered backend preference chain for a query.

    If qt is omitted, classify_query() determines it. Returned list is the
    caller's fallback order: try backend[0] first, backend[1] if that
    fails, and so on. Empty list for INTERNAL_REJECTED signals "do not
    fetch externally".

    KP-3+ consumers wrap this with actual fetch + cache + budget logic.
    """
    if qt is None:
        qt = classify_query(topic)
    return list(_BACKEND_CHAINS.get(qt, []))


__all__ = [
    "QueryType",
    "normalize_query",
    "query_hash",
    "classify_query",
    "route",
]
