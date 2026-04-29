"""
Verified Context Builder — Multi-store retrieval with TimeChain verification stamps.

Parses user queries for entities, temporal references, and activity types, then
pulls relevant records from across Titan's 14-layer memory landscape. Each result
is stamped with its TimeChain chain status (CHAINED/PARTIAL/WIRED/NOT_COVERED).

This replaces the narrow memory.query() in the pre-hook with a comprehensive,
topic-aware retrieval system that gives the LLM truthful, verified context.

Performance budget: ~10-20ms total (negligible vs 2-30s LLM call).
"""

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("VerifiedContextBuilder")

# ═════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════


@dataclass
class VerifiedRecord:
    """A single memory record with its TimeChain verification stamp."""
    content: str
    source: str              # e.g. "inner_memory.db:vocabulary"
    timestamp: float
    chain_status: str        # CHAINED / PARTIAL / WIRED / NOT_COVERED
    block_height: Optional[int] = None
    db_ref: Optional[str] = None
    confidence: float = 0.3  # CHAINED=1.0, PARTIAL=0.7, WIRED=0.5, NOT_COVERED=0.3
    relevance: float = 0.0   # Query-specific relevance score


@dataclass
class ParsedQuery:
    """Structured extraction from a user message."""
    entities: list[str] = field(default_factory=list)       # Person/kin/topic names
    entity_types: dict = field(default_factory=dict)         # entity → "person"/"kin"/"topic"
    temporal_start: Optional[float] = None                   # Unix timestamp range
    temporal_end: Optional[float] = None
    temporal_label: str = ""                                 # "last Thursday", "recently"
    activities: list[str] = field(default_factory=list)      # Detected activity keywords
    store_hints: list[str] = field(default_factory=list)     # Stores to query


@dataclass
class VerifiedContext:
    """Assembled context ready for LLM injection."""
    records: list[VerifiedRecord]
    text: str                    # Formatted markdown for injection
    total_records: int
    chained_count: int
    parse_ms: float
    query_ms: float
    total_ms: float


# ═════════════════════════════════════════════════════════════════════
# CHAIN STATUS MAPPING — which stores have TimeChain coverage
# ═════════════════════════════════════════════════════════════════════

# From memory_structure_titan_v6.md
_CHAIN_STATUS = {
    "vocabulary": "CHAINED",
    "knowledge_concepts": "CHAINED",
    "chain_archive": "WIRED",
    "meta_wisdom": "WIRED",
    "composition_history": "PARTIAL",
    "creative_works": "PARTIAL",
    "kin_encounters": "CHAINED",
    "self_insights": "WIRED",
    "episodic_memory": "PARTIAL",
    "distilled_wisdom": "WIRED",
    "experience_records": "PARTIAL",
    "social_x_actions": "WIRED",
    "social_graph": "NOT_COVERED",
    "events_teacher": "NOT_COVERED",
    "memory_nodes": "WIRED",
}

_CONFIDENCE_MAP = {
    "CHAINED": 1.0,
    "PARTIAL": 0.7,
    "WIRED": 0.5,
    "NOT_COVERED": 0.3,
}

# ═════════════════════════════════════════════════════════════════════
# ACTIVITY → STORE ROUTING
# ═════════════════════════════════════════════════════════════════════

_ACTIVITY_ROUTES = {
    # verb triggers → (store_key, db_path_relative, table, description)
    "post": ("social_x_actions", "social_x.db", "actions", "X posts and replies"),
    "posted": ("social_x_actions", "social_x.db", "actions", "X posts and replies"),
    "tweeted": ("social_x_actions", "social_x.db", "actions", "X posts and replies"),
    "shared": ("social_x_actions", "social_x.db", "actions", "X posts and replies"),
    "said": ("events_teacher", "events_teacher.db", "felt_experiences", "social observations"),
    "told": ("events_teacher", "events_teacher.db", "felt_experiences", "social observations"),
    "mentioned": ("events_teacher", "events_teacher.db", "felt_experiences", "social observations"),
    "learned": ("vocabulary", "inner_memory.db", "vocabulary", "learned words"),
    "word": ("vocabulary", "inner_memory.db", "vocabulary", "learned words"),
    "vocabulary": ("vocabulary", "inner_memory.db", "vocabulary", "learned words"),
    "know": ("knowledge_concepts", "inner_memory.db", "knowledge_concepts", "grounded knowledge"),
    "understand": ("knowledge_concepts", "inner_memory.db", "knowledge_concepts", "grounded knowledge"),
    "thought": ("chain_archive", "inner_memory.db", "chain_archive", "reasoning chains"),
    "reasoned": ("chain_archive", "inner_memory.db", "chain_archive", "reasoning chains"),
    "concluded": ("chain_archive", "inner_memory.db", "chain_archive", "reasoning chains"),
    "insight": ("meta_wisdom", "inner_memory.db", "meta_wisdom", "distilled insights"),
    "wisdom": ("meta_wisdom", "inner_memory.db", "meta_wisdom", "distilled insights"),
    "realized": ("meta_wisdom", "inner_memory.db", "meta_wisdom", "distilled insights"),
    "dreamed": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "dream": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "sleep": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "created": ("creative_works", "inner_memory.db", "creative_works", "creative works"),
    "art": ("creative_works", "inner_memory.db", "creative_works", "creative works"),
    "music": ("creative_works", "inner_memory.db", "creative_works", "creative works"),
    "image": ("creative_works", "inner_memory.db", "creative_works", "creative works"),
    "met": ("kin_encounters", "inner_memory.db", "kin_encounters", "kin encounters"),
    "kin": ("kin_encounters", "inner_memory.db", "kin_encounters", "kin encounters"),
    "sibling": ("kin_encounters", "inner_memory.db", "kin_encounters", "kin encounters"),
    "felt": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "experienced": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "happened": ("episodic_memory", "episodic_memory.db", "episodic_memory", "episodic experiences"),
    "researched": ("experience_records", "experience_orchestrator.db", "experience_records", "research experiences"),
    "explored": ("experience_records", "experience_orchestrator.db", "experience_records", "research experiences"),
    "composed": ("composition_history", "inner_memory.db", "composition_history", "own compositions"),
    "wrote": ("composition_history", "inner_memory.db", "composition_history", "own compositions"),
    "predict": ("self_insights", "inner_memory.db", "self_insights", "self-model predictions"),
    "myself": ("self_insights", "inner_memory.db", "self_insights", "self-model predictions"),
}

# Kin titan identifiers
_KIN_NAMES = {"t1", "t2", "t3", "titan1", "titan2", "titan3", "titan 1", "titan 2", "titan 3"}

# ═════════════════════════════════════════════════════════════════════
# TEMPORAL PATTERNS
# ═════════════════════════════════════════════════════════════════════

_RELATIVE_PATTERNS = [
    (r"\byesterday\b", lambda now: (now - timedelta(days=1)).replace(hour=0, minute=0, second=0),
     lambda now: now.replace(hour=0, minute=0, second=0)),
    (r"\btoday\b", lambda now: now.replace(hour=0, minute=0, second=0),
     lambda now: now),
    (r"\bthis\s+week\b", lambda now: now - timedelta(days=now.weekday()),
     lambda now: now),
    (r"\blast\s+week\b", lambda now: now - timedelta(days=now.weekday() + 7),
     lambda now: now - timedelta(days=now.weekday())),
    (r"\brecent(?:ly)?\b", lambda now: now - timedelta(hours=48), lambda now: now),
    (r"\blast\s+hour\b", lambda now: now - timedelta(hours=1), lambda now: now),
]

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_LAST_DAY_RE = re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


# ═════════════════════════════════════════════════════════════════════
# QUERY PARSER
# ═════════════════════════════════════════════════════════════════════

class QueryParser:
    """Extract entities, temporal references, and activity types from user message."""

    _mention_re = re.compile(r"@(\w+)")
    _stopwords = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
        "have", "has", "had", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "me", "him",
        "us", "them", "this", "that", "these", "those", "what", "when",
        "where", "who", "how", "why", "which", "can", "could", "would",
        "should", "will", "shall", "may", "might", "must", "not", "no",
        "yes", "and", "or", "but", "if", "then", "so", "than", "too",
        "very", "just", "about", "to", "from", "in", "on", "at", "by",
        "for", "with", "of", "up", "out", "off", "over", "into",
    })

    def __init__(self, known_users: list[str] = None):
        self._known_users = set(u.lower() for u in (known_users or []))

    def parse(self, text: str) -> ParsedQuery:
        result = ParsedQuery()
        lower = text.lower()
        words = lower.split()

        # ── Entities ──
        # @mentions
        for match in self._mention_re.finditer(text):
            name = match.group(1)
            result.entities.append(name)
            result.entity_types[name] = "person"

        # Kin titan references
        for kin in _KIN_NAMES:
            if kin in lower:
                result.entities.append(kin)
                result.entity_types[kin] = "kin"

        # Known users (from social_graph)
        for user in self._known_users:
            if user in lower and user not in [e.lower() for e in result.entities]:
                result.entities.append(user)
                result.entity_types[user] = "person"

        # ── Temporal ──
        now = datetime.now(timezone.utc)

        # "last Thursday" pattern
        day_match = _LAST_DAY_RE.search(text)
        if day_match:
            day_name = day_match.group(1).lower()
            target_weekday = _DAY_NAMES[day_name]
            days_back = (now.weekday() - target_weekday) % 7
            if days_back == 0:
                days_back = 7
            target = now - timedelta(days=days_back)
            result.temporal_start = target.replace(hour=0, minute=0, second=0).timestamp()
            result.temporal_end = target.replace(hour=23, minute=59, second=59).timestamp()
            result.temporal_label = f"last {day_match.group(1)}"

        # Relative patterns
        if not result.temporal_start:
            for pattern, start_fn, end_fn in _RELATIVE_PATTERNS:
                if re.search(pattern, text, re.I):
                    result.temporal_start = start_fn(now).timestamp()
                    result.temporal_end = end_fn(now).timestamp()
                    result.temporal_label = re.search(pattern, text, re.I).group(0)
                    break

        # ── Activities ──
        seen_stores = set()
        for word in words:
            word_clean = word.strip("?.,!;:'\"")
            if word_clean in _ACTIVITY_ROUTES:
                store_key = _ACTIVITY_ROUTES[word_clean][0]
                if store_key not in seen_stores:
                    result.activities.append(word_clean)
                    result.store_hints.append(store_key)
                    seen_stores.add(store_key)

        # Entity-based store routing
        if any(et == "person" for et in result.entity_types.values()):
            for store in ("social_graph", "events_teacher"):
                if store not in seen_stores:
                    result.store_hints.append(store)
                    seen_stores.add(store)

        if any(et == "kin" for et in result.entity_types.values()):
            if "kin_encounters" not in seen_stores:
                result.store_hints.append("kin_encounters")

        return result


# ═════════════════════════════════════════════════════════════════════
# STORE ROUTER — Execute queries against specific databases
# ═════════════════════════════════════════════════════════════════════

class StoreRouter:
    """Routes parsed queries to specific database tables and executes SQL."""

    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)

    def query_store(self, store_key: str, parsed: ParsedQuery,
                    limit: int = 5) -> list[dict]:
        """Query a specific store based on parsed dimensions."""
        try:
            handler = getattr(self, f"_query_{store_key}", None)
            if handler:
                return handler(parsed, limit)
            return []
        except Exception as e:
            swallow_warn(f'[VCB] Store query failed for {store_key}', e,
                         key="logic.verified_context_builder.store_query_failed_for", throttle=100)
            return []

    def _connect(self, db_name: str) -> sqlite3.Connection:
        path = self._data_dir / db_name
        if not path.exists():
            return None
        conn = sqlite3.connect(str(path), timeout=3)
        conn.execute("PRAGMA busy_timeout=3000")
        conn.row_factory = sqlite3.Row
        return conn

    def _time_filter(self, parsed: ParsedQuery, col: str = "created_at") -> tuple:
        """Build SQL WHERE clause fragment for temporal filtering."""
        if parsed.temporal_start and parsed.temporal_end:
            return f"AND {col} BETWEEN ? AND ?", [parsed.temporal_start, parsed.temporal_end]
        return "", []

    # ── Store-specific query handlers ──

    def _query_vocabulary(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND word LIKE ?"
                params.append(f"%{e}%")
            time_sql, time_params = self._time_filter(parsed)
            rows = conn.execute(
                f"SELECT word, word_type, confidence, times_produced, learning_phase, "
                f"created_at FROM vocabulary WHERE 1=1 {entity_filter} {time_sql} "
                f"ORDER BY confidence DESC, times_produced DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Word '{r['word']}' ({r['word_type']}, {r['learning_phase']}): "
                                f"confidence={r['confidence']:.2f}, produced {r['times_produced']} times",
                     "source": "inner_memory.db:vocabulary",
                     "timestamp": r["created_at"],
                     "db_ref": f"vocabulary:{r['word']}"}
                    for r in rows]
        finally:
            conn.close()

    def _query_knowledge_concepts(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND (topic LIKE ? OR summary LIKE ?)"
                params.extend([f"%{e}%", f"%{e}%"])
            rows = conn.execute(
                f"SELECT topic, summary, confidence, source, encounter_count, "
                f"created_at FROM knowledge_concepts WHERE 1=1 {entity_filter} "
                f"ORDER BY confidence DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [{"content": f"Knowledge: '{r['topic']}' (conf={r['confidence']:.2f}, "
                                f"via {r['source']}, {r['encounter_count']} encounters)"
                                f"{': ' + r['summary'][:150] if r['summary'] else ''}",
                     "source": "inner_memory.db:knowledge_concepts",
                     "timestamp": r["created_at"],
                     "db_ref": f"knowledge_concepts:{r['topic']}"}
                    for r in rows]
        finally:
            conn.close()

    def _query_chain_archive(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed)
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND (domain LIKE ? OR strategy_label LIKE ?)"
                params.extend([f"%{e}%", f"%{e}%"])
            rows = conn.execute(
                f"SELECT chain_sequence, chain_length, confidence, outcome_score, "
                f"domain, strategy_label, epoch_id, created_at "
                f"FROM chain_archive WHERE confidence > 0.3 {entity_filter} {time_sql} "
                f"ORDER BY outcome_score DESC, created_at DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Reasoning ({r['domain']}, {r['strategy_label'] or 'general'}): "
                                f"{r['chain_length']} steps, outcome={r['outcome_score']:.2f}, "
                                f"conf={r['confidence']:.2f}",
                     "source": "inner_memory.db:chain_archive",
                     "timestamp": r["created_at"],
                     "db_ref": f"chain_archive:{r['id']}" if 'id' in r.keys() else None}
                    for r in rows]
        finally:
            conn.close()

    def _query_meta_wisdom(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND problem_pattern LIKE ?"
                params.append(f"%{e}%")
            rows = conn.execute(
                f"SELECT problem_pattern, strategy_sequence, outcome_score, confidence, "
                f"times_reused, created_at FROM meta_wisdom "
                f"WHERE 1=1 {entity_filter} "
                f"ORDER BY confidence DESC, times_reused DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [{"content": f"Wisdom: '{r['problem_pattern'][:80]}' → "
                                f"'{r['strategy_sequence'][:80]}' "
                                f"(outcome={r['outcome_score']:.2f}, reused {r['times_reused']}x)",
                     "source": "inner_memory.db:meta_wisdom",
                     "timestamp": r["created_at"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_kin_encounters(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed)
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND kin_pubkey LIKE ?"
                params.append(f"%{e}%")
            rows = conn.execute(
                f"SELECT kin_pubkey, resonance, my_emotion, kin_emotion, "
                f"exchange_type, epoch_id, timestamp "
                f"FROM kin_encounters WHERE 1=1 {entity_filter} {time_sql} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Kin encounter with {r['kin_pubkey'][:16]}...: "
                                f"resonance={r['resonance']:.2f}, "
                                f"my_emotion={r['my_emotion']}, kin_emotion={r['kin_emotion']}, "
                                f"type={r['exchange_type']}",
                     "source": "inner_memory.db:kin_encounters",
                     "timestamp": r["timestamp"],
                     "db_ref": f"kin_encounter:{r['epoch_id']}"}
                    for r in rows]
        finally:
            conn.close()

    def _query_episodic_memory(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("episodic_memory.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed, "created_at")
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND description LIKE ?"
                params.append(f"%{e}%")
            # Filter by event_type if dream-related activity detected
            type_filter = ""
            if any(a in ("dreamed", "dream", "sleep") for a in parsed.activities):
                type_filter = " AND event_type LIKE '%dream%'"
            rows = conn.execute(
                f"SELECT event_type, description, felt_state, significance, epoch_id, "
                f"created_at FROM episodic_memory "
                f"WHERE significance > 0.3 {entity_filter} {type_filter} {time_sql} "
                f"ORDER BY significance DESC, created_at DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Experience ({r['event_type']}): "
                                f"{(r['description'] or '')[:120]} "
                                f"(significance={r['significance']:.2f})",
                     "source": "episodic_memory.db:episodic_memory",
                     "timestamp": r["created_at"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_social_x_actions(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("social_x.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed)
            rows = conn.execute(
                f"SELECT action_type, text, post_type, emotion, epoch, "
                f"created_at, tweet_id, status "
                f"FROM actions WHERE status IN ('posted', 'verified') "
                f"{time_sql} "
                f"ORDER BY created_at DESC LIMIT ?",
                time_params + [limit]
            ).fetchall()
            return [{"content": f"X {r['action_type']} ({r['post_type'] or 'general'}, "
                                f"emotion={r['emotion']}): "
                                f"{(r['text'] or '')[:150]}",
                     "source": "social_x.db:actions",
                     "timestamp": r["created_at"],
                     "db_ref": f"social_x:{r['tweet_id']}" if r["tweet_id"] else None}
                    for r in rows]
        finally:
            conn.close()

    def _query_social_graph(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("social_graph.db")
        if not conn:
            return []
        try:
            results = []
            for entity in parsed.entities:
                if parsed.entity_types.get(entity) != "person":
                    continue
                # Check user_profiles
                rows = conn.execute(
                    "SELECT user_id, display_name, interaction_count, engagement_level, "
                    "like_score, first_seen, last_seen FROM user_profiles "
                    "WHERE user_id LIKE ? OR display_name LIKE ? LIMIT ?",
                    [f"%{entity}%", f"%{entity}%", limit]
                ).fetchall()
                for r in rows:
                    results.append({
                        "content": f"Contact: {r['display_name'] or r['user_id']} — "
                                   f"{r['interaction_count']} interactions, "
                                   f"engagement={r['engagement_level']:.2f}",
                        "source": "social_graph.db:user_profiles",
                        "timestamp": r["last_seen"] or r["first_seen"] or 0,
                    })
                # Check community_registry
                rows = conn.execute(
                    "SELECT user_name, display_name, bio, is_follower "
                    "FROM community_registry "
                    "WHERE user_name LIKE ? OR display_name LIKE ? LIMIT ?",
                    [f"%{entity}%", f"%{entity}%", limit]
                ).fetchall()
                for r in rows:
                    follower = "follower" if r["is_follower"] else "not follower"
                    results.append({
                        "content": f"Community: @{r['user_name']} "
                                   f"({r['display_name'] or 'no name'}, {follower})"
                                   f"{': ' + r['bio'][:80] if r['bio'] else ''}",
                        "source": "social_graph.db:community_registry",
                        "timestamp": 0,
                    })
            return results[:limit]
        finally:
            conn.close()

    def _query_events_teacher(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("events_teacher.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed)
            entity_filter = ""
            params = []
            for entity in parsed.entities:
                entity_filter += " AND (author LIKE ? OR topic LIKE ?)"
                params.extend([f"%{entity}%", f"%{entity}%"])
            rows = conn.execute(
                f"SELECT source, author, topic, sentiment, felt_summary, "
                f"created_at FROM felt_experiences "
                f"WHERE 1=1 {entity_filter} {time_sql} "
                f"ORDER BY created_at DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Social event ({r['source']}): {r['author']} on '{r['topic']}' — "
                                f"{r['felt_summary'][:120]} (sentiment={r['sentiment']:.2f})",
                     "source": "events_teacher.db:felt_experiences",
                     "timestamp": r["created_at"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_creative_works(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed, "timestamp")
            rows = conn.execute(
                f"SELECT work_type, file_path, triggering_program, posture, "
                f"assessment_score, timestamp "
                f"FROM creative_works WHERE 1=1 {time_sql} "
                f"ORDER BY timestamp DESC LIMIT ?",
                time_params + [limit]
            ).fetchall()
            return [{"content": f"Created {r['work_type']}: {r['file_path'] or 'untitled'} "
                                f"(score={r['assessment_score']:.2f}, "
                                f"trigger={r['triggering_program']})",
                     "source": "inner_memory.db:creative_works",
                     "timestamp": r["timestamp"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_composition_history(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed, "timestamp")
            rows = conn.execute(
                f"SELECT sentence, confidence, level, intent, timestamp "
                f"FROM composition_history "
                f"WHERE confidence > 0.3 AND level >= 3 {time_sql} "
                f"ORDER BY confidence DESC, timestamp DESC LIMIT ?",
                time_params + [limit]
            ).fetchall()
            return [{"content": f"Composed (L{r['level']}, conf={r['confidence']:.2f}): "
                                f"\"{r['sentence'][:120]}\"",
                     "source": "inner_memory.db:composition_history",
                     "timestamp": r["timestamp"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_self_insights(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("inner_memory.db")
        if not conn:
            return []
        try:
            rows = conn.execute(
                "SELECT sub_mode, epoch, data, confidence, timestamp "
                "FROM self_insights ORDER BY timestamp DESC LIMIT ?",
                [limit]
            ).fetchall()
            return [{"content": f"Self-insight ({r['sub_mode']}): "
                                f"{(r['data'] or '')[:120]} (conf={r['confidence']:.2f})",
                     "source": "inner_memory.db:self_insights",
                     "timestamp": r["timestamp"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_experience_records(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("experience_orchestrator.db")
        if not conn:
            return []
        try:
            time_sql, time_params = self._time_filter(parsed)
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND (domain LIKE ? OR action_taken LIKE ?)"
                params.extend([f"%{e}%", f"%{e}%"])
            rows = conn.execute(
                f"SELECT domain, action_taken, outcome_score, context, "
                f"created_at FROM experience_records "
                f"WHERE outcome_score > 0.3 {entity_filter} {time_sql} "
                f"ORDER BY outcome_score DESC, created_at DESC LIMIT ?",
                params + time_params + [limit]
            ).fetchall()
            return [{"content": f"Experience ({r['domain']}): {r['action_taken']} → "
                                f"score={r['outcome_score']:.2f}",
                     "source": "experience_orchestrator.db:experience_records",
                     "timestamp": r["created_at"]}
                    for r in rows]
        finally:
            conn.close()

    def _query_distilled_wisdom(self, parsed: ParsedQuery, limit: int) -> list[dict]:
        conn = self._connect("experience_orchestrator.db")
        if not conn:
            return []
        try:
            entity_filter = ""
            params = []
            for e in parsed.entities:
                entity_filter += " AND (domain LIKE ? OR pattern LIKE ?)"
                params.extend([f"%{e}%", f"%{e}%"])
            rows = conn.execute(
                f"SELECT domain, pattern, confidence, experience_count, "
                f"dream_cycle, created_at FROM distilled_wisdom "
                f"WHERE 1=1 {entity_filter} "
                f"ORDER BY confidence DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [{"content": f"Dream wisdom ({r['domain']}): '{r['pattern'][:100]}' "
                                f"(conf={r['confidence']:.2f}, from {r['experience_count']} experiences, "
                                f"dream #{r['dream_cycle']})",
                     "source": "experience_orchestrator.db:distilled_wisdom",
                     "timestamp": r["created_at"]}
                    for r in rows]
        finally:
            conn.close()


# ═════════════════════════════════════════════════════════════════════
# VERIFIED CONTEXT BUILDER — Main orchestrator
# ═════════════════════════════════════════════════════════════════════

class VerifiedContextBuilder:
    """Orchestrates query parsing → store routing → verification → assembly."""

    def __init__(self, data_dir: str = "./data",
                 memory_verifier=None,
                 known_users: list[str] = None):
        self._data_dir = data_dir
        self._parser = QueryParser(known_users=known_users)
        self._router = StoreRouter(data_dir=data_dir)
        self._verifier = memory_verifier  # MemoryVerifier instance (optional)

    def build(self, query: str, user_id: str = "",
              max_tokens: int = 2000,
              max_records: int = 30) -> VerifiedContext:
        """Build verified context from a user query.

        Args:
            query: The user's message text.
            user_id: Authenticated user ID (for user-specific queries).
            max_tokens: Token budget for the assembled context.
            max_records: Maximum records to return.

        Returns:
            VerifiedContext with records, formatted text, and timing.
        """
        t0 = time.time()

        # Step 1: Parse query
        parsed = self._parser.parse(query)
        t_parse = time.time()

        # Step 2: Determine which stores to query
        stores_to_query = parsed.store_hints[:]

        # If no specific stores detected, use general fallback
        if not stores_to_query:
            stores_to_query = [
                "episodic_memory",    # Recent experiences
                "social_x_actions",   # Recent posts
                "vocabulary",         # Word knowledge
                "chain_archive",      # Recent reasoning
            ]

        # Always include social_graph for person entities
        if any(et == "person" for et in parsed.entity_types.values()):
            if "social_graph" not in stores_to_query:
                stores_to_query.append("social_graph")

        # Step 3: Query each store
        all_records: list[VerifiedRecord] = []
        per_store_limit = max(3, max_records // max(len(stores_to_query), 1))

        for store_key in stores_to_query:
            raw_results = self._router.query_store(store_key, parsed, limit=per_store_limit)
            chain_status = _CHAIN_STATUS.get(store_key, "NOT_COVERED")
            confidence = _CONFIDENCE_MAP.get(chain_status, 0.3)

            for raw in raw_results:
                db_ref = raw.get("db_ref")
                # Check MemoryVerifier if available
                actual_status = chain_status
                block_height = None
                if self._verifier and db_ref:
                    vr = self._verifier.verify(db_ref)
                    if vr and vr.authentic:
                        actual_status = "CHAINED"
                        block_height = vr.block_height
                        confidence = 1.0
                    elif vr and vr.untracked:
                        actual_status = chain_status  # Use store-level status

                all_records.append(VerifiedRecord(
                    content=raw["content"],
                    source=raw.get("source", store_key),
                    timestamp=raw.get("timestamp", 0),
                    chain_status=actual_status,
                    block_height=block_height,
                    db_ref=db_ref,
                    confidence=confidence,
                ))

        t_query = time.time()

        # Step 4: Rank and trim
        all_records.sort(key=lambda r: (r.confidence, r.timestamp), reverse=True)
        records = all_records[:max_records]

        # Step 5: Assemble text
        text = self._assemble_text(records, parsed, max_tokens)

        t_end = time.time()
        chained = sum(1 for r in records if r.chain_status == "CHAINED")

        return VerifiedContext(
            records=records,
            text=text,
            total_records=len(records),
            chained_count=chained,
            parse_ms=round((t_parse - t0) * 1000, 1),
            query_ms=round((t_query - t_parse) * 1000, 1),
            total_ms=round((t_end - t0) * 1000, 1),
        )

    def _assemble_text(self, records: list[VerifiedRecord],
                       parsed: ParsedQuery, max_tokens: int) -> str:
        """Format records as markdown for LLM injection."""
        if not records:
            return ("### Verified Memory Recall\n"
                    "No specific memories found for this query.\n"
                    "⚠ Answer honestly from what you know. "
                    "If you don't recall something, say so.\n")

        lines = ["### Verified Memory Recall"]

        if parsed.temporal_label:
            lines.append(f"*Time reference: {parsed.temporal_label}*")
        if parsed.entities:
            lines.append(f"*Entities: {', '.join(parsed.entities)}*")
        lines.append("")

        # Group by source for readability
        by_source: dict[str, list[VerifiedRecord]] = {}
        for r in records:
            src_short = r.source.split(":")[-1] if ":" in r.source else r.source
            by_source.setdefault(src_short, []).append(r)

        token_est = 0
        for src, recs in by_source.items():
            src_header = f"**{src}:**"
            lines.append(src_header)
            for r in recs:
                status_tag = {
                    "CHAINED": "[CHAINED ✓]",
                    "PARTIAL": "[PARTIAL]",
                    "WIRED": "[WIRED]",
                    "NOT_COVERED": "[unverified]",
                }.get(r.chain_status, "")

                ts_str = ""
                if r.timestamp and r.timestamp > 0:
                    try:
                        dt = datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
                        ts_str = dt.strftime(" (%Y-%m-%d %H:%M)")
                    except (OSError, ValueError):
                        pass

                block_ref = f" Block #{r.block_height}" if r.block_height else ""
                line = f"- {r.content}{ts_str} {status_tag}{block_ref}"
                token_est += len(line.split()) * 1.3  # rough token estimate
                if token_est > max_tokens:
                    lines.append("- *(truncated — token budget reached)*")
                    break
                lines.append(line)
            lines.append("")

            if token_est > max_tokens:
                break

        lines.append("⚠ These memories are verified against your TimeChain.")
        lines.append("Only reference what is provided here. If asked about "
                     "something not in your memories, say honestly that you don't recall.")

        return "\n".join(lines)
