"""
titan_plugin/logic/grammar_patterns.py — Learned Grammar Pattern Library (L8).

Stores grammar patterns extracted from:
  1. LLM teacher "modeling" responses (high quality, novel structures)
  2. Titan's own successful compositions (self-reinforcement)

Patterns are abstract sentence templates with typed slots ({ADJ}, {VERB}, {NOUN}).
Unlike L1-L7 hardcoded templates, L8 patterns are LEARNED — they grow from exposure.

Example:
  Teacher says: "the warmth I feel makes me want to create"
  Extract: "the {NOUN} I feel makes me want to {VERB}"
  Titan can now fill this template with ANY noun+verb from vocabulary.
"""
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Content word types that become fillable slots
CONTENT_TYPES = {"adjective", "verb", "noun", "adverb"}

# Function words preserved as literals in templates
FUNCTION_WORDS = {
    "i", "me", "my", "myself",
    "feel", "am", "is", "are", "was", "have", "has", "do", "does", "make", "makes",
    "want", "need", "like", "know", "think", "see", "hear",
    "to", "and", "but", "or", "so", "yet", "nor",
    "when", "because", "if", "while", "although", "since", "until", "after", "before",
    "the", "a", "an", "this", "that", "these", "those",
    "not", "also", "then", "still", "just", "even", "very", "too",
    "in", "on", "at", "for", "with", "from", "by", "of", "about",
    "it", "something", "everything", "nothing",
}

# Slot type tags
SLOT_TAG = {
    "adjective": "ADJ",
    "verb": "VERB",
    "noun": "NOUN",
    "adverb": "ADV",
}


class GrammarPatternLibrary:
    """Stores and retrieves learned grammar patterns for L8 composition."""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or str(Path("data") / "inner_memory.db")
        self._ensure_table()
        self._cache: list[dict] = []
        self._cache_ts: float = 0

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        try:
            conn = self._connect()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grammar_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template TEXT NOT NULL UNIQUE,
                    abstract_sequence TEXT NOT NULL,
                    slot_types TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'teacher',
                    source_sentence TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    first_seen REAL NOT NULL,
                    last_used REAL DEFAULT 0,
                    slot_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("[GrammarPatterns] Table init error: %s", e)

    def extract_template(
        self, sentence: str, vocabulary: list
    ) -> Optional[dict]:
        """Extract a fillable template from a sentence.

        Maps content words (adj/verb/noun) to typed slots,
        preserves function words as literals.

        Args:
            sentence: Natural language sentence
            vocabulary: Titan's vocabulary (list of word dicts)

        Returns:
            Dict with template, abstract, slots, word_count or None if invalid
        """
        vocab_map = {}
        for v in vocabulary:
            w = v.get("word", "").lower()
            wt = v.get("word_type", "").lower()
            if w and wt:
                vocab_map[w] = wt

        # Clean up teacher meta-talk: extract only first-person sentences
        # Teacher often says "okay let's try this" or "that's good!" before
        # the actual modeled sentence. Strip these preambles.
        # Also reject second-person ("you feel", "you want") — teacher talking
        # TO Titan, not modeling how Titan should speak.
        clean = sentence.strip()

        # Reject any sentence with teacher/second-person markers anywhere
        _lower = clean.lower()
        _reject_markers = [
            "let's", "that's good", "that's nice", "okay", "you feel",
            "you know", "you want", "you are", "you can", "we can",
            "try this", "your words", "shows you", "i see you",
            "good job", "well done", "nice work", "let me",
        ]
        if any(p in _lower for p in _reject_markers):
            # Try to salvage a clean first-person clause after the preamble
            _clauses = re.split(r'[.!?\n]+', clean)
            _first_person = [c.strip() for c in _clauses
                             if c.strip().lower().startswith("i ")
                             and not any(m in c.strip().lower()
                                         for m in _reject_markers)]
            if _first_person and len(_first_person[0].split()) >= 3:
                clean = _first_person[0]
            else:
                return None

        # For non-teacher input: prefer first-person clauses but allow non-I
        # patterns up to 40% of the library (Phase 2c: diversity)
        if not clean.lower().startswith("i "):
            _clauses = re.split(r'[.!?\n]+', clean)
            _first_person = [c.strip() for c in _clauses
                             if c.strip().lower().startswith("i ")]
            if _first_person:
                clean = _first_person[0]
            else:
                # Allow non-I patterns if under 40% quota
                non_i_ratio = self._get_non_i_ratio()
                if non_i_ratio >= 0.4:
                    return None  # Enough non-I patterns already
                # Accept the non-I sentence (question, observation, conditional)

        tokens = clean.lower().split()
        tokens = [t.strip(".,!?\"'()[]{}:;-") for t in tokens]
        tokens = [t for t in tokens if t]

        if len(tokens) < 3:
            return None

        template_parts = []
        abstract_parts = []
        slot_types = []
        slot_counters = {"ADJ": 0, "VERB": 0, "NOUN": 0, "ADV": 0}
        content_word_count = 0

        for token in tokens:
            word_type = vocab_map.get(token, "")

            # Content word with known type → becomes a slot
            if word_type in CONTENT_TYPES and token not in FUNCTION_WORDS:
                tag = SLOT_TAG.get(word_type, "ANY")
                slot_counters[tag] += 1
                # First slot of type: {ADJ}, second: {ADJ2}, etc.
                suffix = "" if slot_counters[tag] == 1 else str(slot_counters[tag])
                slot_name = "{" + tag + suffix + "}"
                template_parts.append(slot_name)
                abstract_parts.append(tag)
                slot_types.append(word_type)
                content_word_count += 1
            else:
                # Function word or unknown → literal
                template_parts.append(token)
                abstract_parts.append(token.upper())

        # Need at least 1 content slot to be useful
        if content_word_count < 1:
            return None

        template = " ".join(template_parts)
        abstract = "_".join(abstract_parts)

        return {
            "template": template,
            "abstract": abstract,
            "slots": slot_types,
            "slot_count": content_word_count,
            "word_count": len(tokens),
        }

    def add_pattern(
        self,
        sentence: str,
        vocabulary: list,
        source: str = "teacher",
    ) -> Optional[str]:
        """Extract and store a grammar pattern from a sentence.

        Args:
            sentence: Source sentence to extract pattern from
            vocabulary: Titan's current vocabulary
            source: "teacher" or "self"

        Returns:
            Template string if stored, None if invalid or duplicate
        """
        extracted = self.extract_template(sentence, vocabulary)
        if not extracted:
            return None

        template = extracted["template"]

        # Skip patterns that match existing hardcoded L1-L7 templates
        # Normalize: lowercase, strip punctuation, collapse spaces
        from titan_plugin.logic.composition_engine import TEMPLATES
        _norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[,;.!?]', '', s.lower())).strip()
        template_norm = _norm(template)
        for level_templates in TEMPLATES.values():
            for lt in level_templates:
                if template_norm == _norm(lt):
                    return None

        try:
            conn = self._connect()
            conn.execute("""
                INSERT OR IGNORE INTO grammar_patterns
                    (template, abstract_sequence, slot_types, source,
                     source_sentence, first_seen, slot_count, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template,
                extracted["abstract"],
                json.dumps(extracted["slots"]),
                source,
                sentence[:200],
                time.time(),
                extracted["slot_count"],
                extracted["word_count"],
            ))
            conn.commit()
            inserted = conn.total_changes > 0
            conn.close()
            self._cache_ts = 0  # Invalidate cache
            if inserted:
                logger.info("[GrammarPatterns] Learned L8 pattern: '%s' (from %s)",
                            template, source)
                return template
            return None  # Already exists
        except Exception as e:
            logger.debug("[GrammarPatterns] Store error: %s", e)
            return None

    def record_usage(self, template: str, success: bool = True,
                     sentence: str = "", slots_filled: int = 0,
                     slots_total: int = 0):
        """Record that a pattern was used in composition.

        Phase 1c quality gate: only count as success if the composition
        is genuinely well-formed (correct slot types, no unfilled markers).
        """
        # Quality gate: validate actual success
        if success and sentence:
            # Reject if unfilled slot markers leaked into output
            if re.search(r'\{[A-Z]+\d*\}', sentence):
                success = False
            # Reject if less than 50% of slots were filled
            elif slots_total > 0 and slots_filled < slots_total * 0.5:
                success = False

        try:
            conn = self._connect()
            if success:
                conn.execute("""
                    UPDATE grammar_patterns
                    SET usage_count = usage_count + 1,
                        success_count = success_count + 1,
                        last_used = ?
                    WHERE template = ?
                """, (time.time(), template))
            else:
                conn.execute("""
                    UPDATE grammar_patterns
                    SET usage_count = usage_count + 1,
                        last_used = ?
                    WHERE template = ?
                """, (time.time(), template))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("[GrammarPatterns] Usage update error: %s", e)

    def get_patterns(self, min_slots: int = 1, limit: int = 50) -> list[dict]:
        """Get all stored patterns, sorted by success rate then recency.

        Returns list of dicts with: template, source, usage_count,
        success_count, success_rate, slot_count, word_count
        """
        # Use cache if fresh (< 60s)
        if self._cache and (time.time() - self._cache_ts) < 60:
            return [p for p in self._cache if p["slot_count"] >= min_slots]

        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                SELECT template, source, usage_count, success_count,
                       slot_count, word_count, slot_types, first_seen
                FROM grammar_patterns
                WHERE slot_count >= ?
                ORDER BY
                    CASE WHEN usage_count > 0
                         THEN CAST(success_count AS REAL) / usage_count
                         ELSE 0.5 END DESC,
                    first_seen DESC
                LIMIT ?
            """, (min_slots, limit))
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[GrammarPatterns] Fetch error: %s", e)
            return []

        patterns = []
        for template, source, usage, success, slots, words, slot_types_json, first_seen in rows:
            rate = success / max(1, usage) if usage > 0 else 0.5
            patterns.append({
                "template": template,
                "source": source,
                "usage_count": usage,
                "success_count": success,
                "success_rate": round(rate, 3),
                "slot_count": slots,
                "word_count": words,
                "slot_types": json.loads(slot_types_json) if slot_types_json else [],
            })

        self._cache = patterns
        self._cache_ts = time.time()
        return patterns

    # ── Session-level diversity tracking (Phase 2b) ──────────────────
    _session_usage: dict = {}  # template -> count this session
    _session_compositions: int = 0

    def select_template(self, vocabulary: list) -> Optional[str]:
        """Select a pattern template for L8 composition.

        Phase 2 evolved selection:
        - Usage decay: heavily-used patterns lose weight over time
        - Novelty bonus: unused patterns get 50% exploration boost
        - Complexity bonus: more slot types = higher weight
        - Session diversity: suppress patterns used 3+ times in 10 compositions
        - Pattern retirement: 75+ uses → minimal weight (mastered)

        Returns:
            Template string or None if no patterns available
        """
        patterns = self.get_patterns(min_slots=1, limit=200)
        if not patterns:
            return None

        # Filter to patterns we can actually fill (check vocab has required types)
        vocab_types = set()
        for v in vocabulary:
            wt = v.get("word_type", "").lower()
            if wt:
                vocab_types.add(wt)

        fillable = []
        for p in patterns:
            required_types = set(p.get("slot_types", []))
            if required_types.issubset(vocab_types):
                fillable.append(p)

        if not fillable:
            return None

        import random as _rng

        # Phase 2 evolved weighting
        weights = []
        for p in fillable:
            usage = p["usage_count"]
            rate = p["success_rate"]
            slots = p["slot_count"]
            template = p["template"]
            slot_types = p.get("slot_types", [])

            # Base weight from success rate
            base = rate + 0.1

            # Usage decay: 50 uses → weight halved, 100 uses → weight quartered
            decay = 1.0 / (1.0 + usage * 0.05)

            # Novelty bonus: never-used patterns get 50% boost
            novelty = 1.5 if usage == 0 else 1.0

            # Complexity bonus: more distinct slot types = higher weight
            unique_types = len(set(slot_types))
            complexity = 1.0 + 0.1 * unique_types

            # Retirement: 75+ uses → minimal weight (mastered)
            if usage >= 75:
                decay = 0.05

            # Session diversity (Phase 2b): suppress if used 3+ times recently
            session_count = self._session_usage.get(template, 0)
            if session_count >= 3 and self._session_compositions < 15:
                decay *= 0.1  # Heavy suppression within session

            weight = max(0.01, base * decay * novelty * complexity)
            weights.append(weight)

        # Weighted random selection
        total = sum(weights)
        r = _rng.random() * total
        cumulative = 0.0
        for p, w in zip(fillable, weights):
            cumulative += w
            if r <= cumulative:
                selected = p["template"]
                # Track session usage
                self._session_usage[selected] = self._session_usage.get(selected, 0) + 1
                self._session_compositions += 1
                # Reset session tracking every 15 compositions
                if self._session_compositions >= 15:
                    self._session_usage.clear()
                    self._session_compositions = 0
                return selected

        return fillable[-1]["template"]

    def count(self) -> int:
        """Count total stored patterns."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM grammar_patterns")
            n = cur.fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    def _get_non_i_ratio(self) -> float:
        """Get ratio of non-I-starting patterns in the library."""
        try:
            conn = self._connect()
            total = conn.execute(
                "SELECT COUNT(*) FROM grammar_patterns"
            ).fetchone()[0]
            non_i = conn.execute(
                "SELECT COUNT(*) FROM grammar_patterns "
                "WHERE template NOT LIKE 'i %'"
            ).fetchone()[0]
            conn.close()
            return non_i / max(1, total)
        except Exception:
            return 0.0

    def cleanup_patterns(self) -> dict:
        """Phase 1d: One-time template sanitization.

        - Delete patterns with unfilled slot markers in template string
        - Delete patterns with markdown artifacts
        - Reset success_rate for patterns with < 3 uses (fresh start)

        Returns dict with counts of actions taken.
        """
        stats = {"deleted_slots": 0, "deleted_markdown": 0, "reset_rates": 0}
        try:
            conn = self._connect()

            # Delete patterns with leaked unfilled slots like {ADJ4} in the template
            # (These are corrupted — the template itself contains unfilled markers)
            cur = conn.execute(
                "SELECT id, template FROM grammar_patterns "
                "WHERE template LIKE '%{ADJ%' OR template LIKE '%{VERB%' "
                "OR template LIKE '%{NOUN%' OR template LIKE '%{ADV%'"
            )
            for row_id, template in cur.fetchall():
                # Check if these are ACTUAL slot markers (valid) vs corrupted
                # Valid: {ADJ} {VERB2} — these are normal
                # Corrupted: literal {ADJ4} that wasn't filled during composition
                # We identify corrupted ones by checking if they have high slot indices
                if re.search(r'\{(ADJ|VERB|NOUN|ADV)[4-9]\d*\}', template):
                    conn.execute(
                        "DELETE FROM grammar_patterns WHERE id = ?", (row_id,))
                    stats["deleted_slots"] += 1

            # Delete patterns with markdown artifacts
            cur2 = conn.execute(
                "SELECT id, template FROM grammar_patterns "
                "WHERE template LIKE '%**%'"
            )
            for row_id, template in cur2.fetchall():
                conn.execute(
                    "DELETE FROM grammar_patterns WHERE id = ?", (row_id,))
                stats["deleted_markdown"] += 1

            # Reset success_rate for patterns with < 3 uses (fresh exploration)
            conn.execute(
                "UPDATE grammar_patterns "
                "SET success_count = 0, usage_count = 0 "
                "WHERE usage_count > 0 AND usage_count < 3"
            )
            stats["reset_rates"] = conn.total_changes

            conn.commit()
            conn.close()
            self._cache_ts = 0  # Invalidate cache

            logger.info("[GrammarPatterns] Cleanup: deleted_slots=%d, "
                        "deleted_markdown=%d, reset_rates=%d",
                        stats["deleted_slots"], stats["deleted_markdown"],
                        stats["reset_rates"])
        except Exception as e:
            logger.warning("[GrammarPatterns] Cleanup error: %s", e)

        return stats
