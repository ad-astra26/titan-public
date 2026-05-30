"""Topic-tag extractor — arch §7 / §13.2 topic-driven sidechain feed.

Phase 3 (rFP §18 Phase 3 — episode model, D-SPEC-127).

**Placement rationale (arch §1.4 + §7 + INV-Syn-2 + G19):** topic_tags
must arrive ON the conversation-fork TX at write time so the existing
auto-sidechain mechanic (`timechain.py:1119`, tag-count ≥3 in 24h fires
a `topic:<TAG>` sidechain) takes effect. The synthesis_worker cannot
be queried inline by agno (G19 forbids sync bus.request) — so this
extractor runs **in the agno_worker process**, called from
`llm_pipeline.verify_post_async` between safety-pass and sign.

**Algorithm:** deterministic case-folded substring match against the
existing `inner_memory.db.knowledge_concepts.topic` column (151+ rows
of Titan's earned-knowledge topic strings). No LLM call; no
external dependency; sub-millisecond hot path.

**Why this is spec-correct:**
  - §3.2 CGN is the meaning oracle; `knowledge_concepts` is CGN's
    surface (it is populated by knowledge_worker via
    `knowledge_gate.best_confidence_topic` matching against CGN
    grounding requests).
  - §7 spec is silent on extractor implementation; using the
    already-grounded topic list keeps the "earned, verifiable"
    discipline (we only tag with topics Titan has actually encountered
    via knowledge_worker, not LLM-imagined ones).
  - §13.2 standing contracts will materialize per-topic bundles
    (P3.E `actr_topic_conversation_bundle`) — populating them with
    the same topic universe as `knowledge_concepts` makes recall +
    storage symmetric.

**Cache:** `_topic_cache` is reloaded every `_RELOAD_TTL_S` seconds
(default 300s = 5min). Knowledge_concepts grows slowly via
knowledge_worker (event-driven, NOT per chat turn) so 5min freshness
is more than adequate. First call after process start blocks ~10ms
on the SQLite scan; thereafter sub-millisecond.

**Filters:**
  - `MIN_TOPIC_CHARS` (4) — skip "and", "the", "you" — too noisy.
    Trims the knowledge_concepts row count for matching but does
    NOT delete data (filter happens at match time, not load time).
  - `MAX_TAGS_PER_TURN` (10) — cap the per-TX tag list (avoids
    runaway tag bloat on long turns about many concepts).
  - Sort by topic length DESCENDING then alphabetical — more
    specific phrases ("metaplex nft minting") get a tag slot before
    sub-phrases ("nft") when capping.

**Surfaced concerns (revisit after live soak):**
  - Substring match has false-positive risk: "are" matches "areas",
    "stark" matches "starkly". For v1 we accept this — the standing
    bundle is forgiving (msgpack ring-buffer eviction caps churn).
    Phase 4 (Kuzu Concept spine) gives us a proper word-grounded
    extractor via CGN's tokenizer.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# Minimum chars to qualify as a "real" topic — filters out stop-words
# that match too liberally via substring (the/and/you/...). Knowledge
# topics that fall below this are kept in the DB (preserved data!) but
# excluded from chat-TX tag extraction.
MIN_TOPIC_CHARS = 4

# Hard cap on tags per TX — keeps the tag list lean even if the user
# turn mentions many known topics. arch §7 tag list is unbounded in
# the spec, but standing-bundle key churn benefits from a cap.
MAX_TAGS_PER_TURN = 10

# Topic-cache TTL — re-read knowledge_concepts every N seconds.
_RELOAD_TTL_S = 300.0

# Default knowledge_concepts DB path — overridable for tests.
_DEFAULT_DB_PATH = os.path.join("data", "inner_memory.db")


# ── Module-level cache (process-local) ───────────────────────────────

_cache_lock = threading.Lock()
_topic_cache: list[str] = []        # sorted by len desc, then alpha
_cache_loaded_at: float = 0.0
_db_path: str = _DEFAULT_DB_PATH


def set_db_path(path: str) -> None:
    """Override the knowledge_concepts DB path (test injection).

    Forces a cache reload on next call. Safe to invoke multiple times.
    """
    global _db_path, _cache_loaded_at
    with _cache_lock:
        _db_path = path
        _cache_loaded_at = 0.0
        _topic_cache.clear()


def _load_topics() -> list[str]:
    """Read `knowledge_concepts.topic` column, filtered + sorted."""
    if not os.path.exists(_db_path):
        return []
    try:
        # read-only open (uri=true) — extractor must never lock the
        # writer (knowledge_worker). sqlite handles concurrent reads
        # cleanly under WAL.
        conn = sqlite3.connect(  # noqa: async-block — TTL-gated topic-cache refresh (rare); short cached sqlite read
            f"file:{_db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            cur = conn.execute("SELECT topic FROM knowledge_concepts")
            raw = [r[0] for r in cur.fetchall() if r and r[0]]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning(
            "[topic_extractor] knowledge_concepts read failed: %s", exc)
        return []
    # Filter + dedupe (case-folded) + sort by length desc / alpha asc.
    seen_lower: set[str] = set()
    filtered: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if len(s) < MIN_TOPIC_CHARS:
            continue
        lower = s.casefold()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        filtered.append(s)
    filtered.sort(key=lambda s: (-len(s), s.casefold()))
    return filtered


def _ensure_cache() -> list[str]:
    """Lazy-load + TTL-refresh the topic cache. Returns the current list."""
    global _cache_loaded_at, _topic_cache
    now = time.time()
    with _cache_lock:
        if (now - _cache_loaded_at) > _RELOAD_TTL_S or not _topic_cache:
            _topic_cache = _load_topics()
            _cache_loaded_at = now
        return list(_topic_cache)


def extract_topic_tags(
    user_prompt: Optional[str],
    agent_response: Optional[str],
    *,
    extra: Optional[list[str]] = None,
) -> list[str]:
    """Extract `topic:<topic>` tags for arch §7 chat-TX tag list.

    Args:
        user_prompt:    Originating user message (full text).
        agent_response: Final agent reply text (post-OVG-pass).
        extra:          Optional caller-supplied raw topic tags (already
                        prefixed with `topic:` if intended as topic
                        tags, or as free-form). Merged + deduped with
                        the extracted list. Used by tests + future
                        callers that have richer extraction context.

    Returns:
        List of `topic:<topic>` strings, capped at MAX_TAGS_PER_TURN,
        deterministic order (length desc, alpha asc within length).
        Empty list when no topics match — caller passes through to
        verify_post_async which omits tags cleanly.

    NEVER raises — DB unavailability + bad inputs all return [].
    """
    haystack_parts: list[str] = []
    if user_prompt:
        haystack_parts.append(str(user_prompt))
    if agent_response:
        haystack_parts.append(str(agent_response))
    if not haystack_parts:
        return [] if not extra else _normalize_extra(extra)
    haystack = " ".join(haystack_parts).casefold()

    matched: list[str] = []
    try:
        for topic in _ensure_cache():
            if topic.casefold() in haystack:
                matched.append(f"topic:{topic}")
                if len(matched) >= MAX_TAGS_PER_TURN:
                    break
    except Exception as exc:
        logger.warning(
            "[topic_extractor] match phase raised %s — returning empty", exc)
        matched = []

    # Merge caller extras (deduped, preserving cap).
    if extra:
        seen = set(matched)
        for e in _normalize_extra(extra):
            if e not in seen:
                matched.append(e)
                seen.add(e)
                if len(matched) >= MAX_TAGS_PER_TURN:
                    break
    return matched


def _normalize_extra(extra: list[str]) -> list[str]:
    """Coerce caller-supplied extras into the canonical `topic:<X>` shape.

    Strings already starting with `topic:` pass through. Others get
    auto-prefixed. Empty/None entries dropped. Used by tests.
    """
    out: list[str] = []
    for e in extra:
        if not e:
            continue
        s = str(e).strip()
        if not s:
            continue
        out.append(s if s.startswith("topic:") else f"topic:{s}")
    return out


def clear_cache_for_test() -> None:
    """Reset module cache — test fixtures only."""
    global _cache_loaded_at
    with _cache_lock:
        _topic_cache.clear()
        _cache_loaded_at = 0.0


__all__ = [
    "extract_topic_tags",
    "set_db_path",
    "clear_cache_for_test",
    "MIN_TOPIC_CHARS",
    "MAX_TAGS_PER_TURN",
]
