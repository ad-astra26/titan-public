"""
Knowledge gate — shared "what does Titan know about topic X?" utility.

Single source of truth for grounded-knowledge confidence lookups, extracted
from the /chat agno_hooks section [24] so the X-post path and any future
consumer can reuse the exact same semantics.

Design doc: titan-docs/rFP_phase5_narrator_evolution.md §9.3
"""
from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB = "data/inner_memory.db"

# Same stopword set as agno_hooks.py section [24]. Kept in sync here so
# both the /chat path and the X path extract topics identically.
STOPWORDS = frozenset({
    "i", "a", "the", "is", "are", "was", "were", "do", "does",
    "what", "how", "why", "when", "where", "who", "which",
    "can", "could", "would", "should", "will", "to", "of",
    "in", "on", "at", "for", "with", "and", "or", "but",
    "not", "this", "that", "it", "my", "your", "me", "you",
    "be", "have", "has", "had", "been", "being", "am",
    "tell", "about", "please", "hi", "hello", "hey", "thanks",
})

_WORD_RE = re.compile(r"[a-zA-Z]+")


def extract_topic_words(text: str, max_words: int = 5) -> list[str]:
    """Extract candidate topic words from free-form text.

    Filters: alphabetic only, length > 2, not in stopword set. Returns
    up to `max_words` first survivors (ordered as in source text).
    """
    if not text:
        return []
    out: list[str] = []
    for w in _WORD_RE.findall(text.lower()):
        if len(w) <= 2:
            continue
        if w in STOPWORDS:
            continue
        out.append(w)
        if len(out) >= max_words:
            break
    return out


def check_topic_confidence(topics: list[str],
                           db_path: str = _DEFAULT_DB) -> float:
    """Return max confidence across all knowledge_concepts rows whose
    topic LIKE-matches any of the given keywords.

    Returns a float in [0.0, 1.0]. 0.0 means "no grounded knowledge
    found." Callers treat <threshold as "don't speak confidently on this."

    Never raises — DB errors log at debug and return 0.0 (the
    conservative default — treat unknown state as ungrounded).
    """
    conf, _topic = check_topic_confidence_with_match(topics, db_path=db_path)
    return conf


def check_topic_confidence_with_match(
        topics: list[str],
        db_path: str = _DEFAULT_DB) -> tuple[float, str]:
    """Like `check_topic_confidence` but also returns the matched topic.

    Returns (best_confidence, matched_topic). `matched_topic` is the
    knowledge_concepts.topic row that provided `best_confidence`, or ""
    if no row matched. Callers use this to emit CGN_KNOWLEDGE_USAGE with
    the correct topic attribution so RoutingLearner.record_usage() feeds
    the right (query_type, backend) reputation — the Definition-3 reward-
    attribution loop from rFP_knowledge_pipeline_v2 §3.4.

    Never raises — DB errors log at debug and return (0.0, "").
    """
    if not topics:
        return 0.0, ""
    if not Path(db_path).exists():
        return 0.0, ""
    try:
        db = sqlite3.connect(db_path, timeout=2.0)
        try:
            best_conf = 0.0
            best_topic = ""
            for kw in topics:
                row = db.execute(
                    "SELECT topic, confidence FROM knowledge_concepts "
                    "WHERE topic LIKE ? "
                    "ORDER BY confidence DESC LIMIT 1",
                    (f"%{kw}%",)
                ).fetchone()
                if row and row[1] is not None:
                    try:
                        v = float(row[1])
                    except (TypeError, ValueError):
                        continue
                    if v > best_conf:
                        best_conf = v
                        best_topic = str(row[0] or "")
            return best_conf, best_topic
        finally:
            db.close()
    except sqlite3.Error as e:
        logger.debug("[knowledge_gate] DB error on topics=%s: %s", topics, e)
        return 0.0, ""


def assess_text(text: str, db_path: str = _DEFAULT_DB,
                max_words: int = 5) -> tuple[list[str], float]:
    """Convenience: extract topic words AND look up confidence.

    Returns (topic_words, best_confidence). Empty topic list → 0.0.
    """
    topics = extract_topic_words(text, max_words=max_words)
    return topics, check_topic_confidence(topics, db_path=db_path)
