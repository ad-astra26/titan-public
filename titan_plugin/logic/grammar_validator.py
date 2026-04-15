"""
titan_plugin/logic/grammar_validator.py — Learned Grammar Correction Rules.

Accumulates grammar rules from LLM spot-check corrections.
Applied AFTER composition, BEFORE output. No LLM at runtime — only learned rules.

Rules are persistent (SQLite) and grow over time as the LLM spot-checker
identifies patterns. This is Titan's Broca's area learning grammar from
exposure, not from pre-programming.

Example rules learned:
  - "want" → "want to" (before verb)
  - After "feel", next word must be adjective
  - "I" always capitalized
  - Double adjective needs "and" between them
"""
import json
import logging
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GrammarRule:
    """A single learned grammar correction rule."""

    def __init__(
        self,
        rule_id: int,
        pattern: str,
        replacement: str,
        context: str = "",
        confidence: float = 1.0,
        times_applied: int = 0,
    ):
        self.rule_id = rule_id
        self.pattern = pattern
        self.replacement = replacement
        self.context = context  # e.g., "before_verb", "after_feel"
        self.confidence = confidence
        self.times_applied = times_applied

    def apply(self, sentence: str) -> str:
        """Apply this rule to a sentence. Returns corrected sentence."""
        if self.pattern in sentence:
            # Replace all occurrences for whitespace/formatting rules
            if self.context == "whitespace":
                while self.pattern in sentence:
                    sentence = sentence.replace(self.pattern, self.replacement)
            else:
                sentence = sentence.replace(self.pattern, self.replacement, 1)
            self.times_applied += 1
        return sentence


# Built-in bootstrap rules (always present, not learned)
BOOTSTRAP_RULES = [
    {"pattern": "i ", "replacement": "I ", "context": "capitalization"},
    {"pattern": "i'm", "replacement": "I'm", "context": "capitalization"},
    {"pattern": "  ", "replacement": " ", "context": "whitespace"},
]


class GrammarValidator:
    """Learns and applies grammar correction rules."""

    def __init__(self, db_path: str = "./data/grammar_rules.db"):
        self._db_path = db_path
        self._rules: list[GrammarRule] = []
        self._total_corrections = 0
        self._total_validations = 0
        self._init_db()
        self._load_rules()
        logger.info("[GrammarValidator] Initialized with %d rules (incl. %d bootstrap)",
                    len(self._rules), len(BOOTSTRAP_RULES))

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grammar_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                replacement TEXT NOT NULL,
                context TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                times_applied INTEGER DEFAULT 0,
                source TEXT DEFAULT 'llm_correction',
                created_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _load_rules(self):
        """Load rules from DB + bootstrap."""
        self._rules = []

        # Bootstrap rules (always present)
        for i, br in enumerate(BOOTSTRAP_RULES):
            self._rules.append(GrammarRule(
                rule_id=-(i + 1),
                pattern=br["pattern"],
                replacement=br["replacement"],
                context=br.get("context", ""),
            ))

        # Learned rules from DB
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute(
                "SELECT id, pattern, replacement, context, confidence, times_applied "
                "FROM grammar_rules ORDER BY confidence DESC"
            )
            for row in cur.fetchall():
                self._rules.append(GrammarRule(
                    rule_id=row[0],
                    pattern=row[1],
                    replacement=row[2],
                    context=row[3] or "",
                    confidence=row[4],
                    times_applied=row[5],
                ))
            conn.close()
        except Exception as e:
            logger.warning("[GrammarValidator] Failed to load rules: %s", e)

    def validate(self, sentence: str) -> str:
        """Apply all grammar rules to a sentence.

        Returns corrected sentence.
        """
        self._total_validations += 1
        original = sentence

        for rule in self._rules:
            sentence = rule.apply(sentence)

        if sentence != original:
            self._total_corrections += 1
            logger.debug("[GrammarValidator] Corrected: '%s' → '%s'", original, sentence)

        return sentence

    def learn_from_correction(
        self,
        original: str,
        corrected: str,
        source: str = "llm_spot_check",
    ) -> Optional[GrammarRule]:
        """Learn a new grammar rule from a correction.

        Extracts the pattern→replacement from comparing original and corrected.
        Simple heuristic: find the first difference and create a local rule.

        Args:
            original: The sentence before correction
            corrected: The corrected sentence
            source: Where the correction came from

        Returns:
            The new GrammarRule if one was created, None otherwise
        """
        if original == corrected:
            return None

        # Find the differing segment
        # Simple approach: split into words, find first diff
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()

        # Find first difference
        pattern = None
        replacement = None
        for i in range(min(len(orig_words), len(corr_words))):
            if orig_words[i] != corr_words[i]:
                # Found difference — create rule for this word and its context
                # Include 1 word before for context
                start = max(0, i - 1)
                end = min(len(orig_words), i + 2)
                pattern = " ".join(orig_words[start:end])
                replacement = " ".join(corr_words[start:min(len(corr_words), i + 2)])
                break

        # Handle length difference (insertion/deletion)
        if pattern is None and len(orig_words) != len(corr_words):
            # Whole sentence replacement as a fallback
            pattern = original.lower().strip()
            replacement = corrected.lower().strip()

        if not pattern or not replacement or pattern == replacement:
            return None

        # Check if rule already exists
        for r in self._rules:
            if r.pattern == pattern:
                r.confidence = min(1.0, r.confidence + 0.1)
                return r

        # Create new rule
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO grammar_rules (pattern, replacement, context, confidence, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pattern, replacement, "", 0.5, source, time.time()),
            )
            conn.commit()
            rule_id = cur.lastrowid
            conn.close()

            rule = GrammarRule(
                rule_id=rule_id,
                pattern=pattern,
                replacement=replacement,
                confidence=0.5,
            )
            self._rules.append(rule)
            logger.info("[GrammarValidator] Learned rule #%d: '%s' → '%s'",
                        rule_id, pattern, replacement)
            return rule
        except Exception as e:
            logger.warning("[GrammarValidator] Failed to save rule: %s", e)
            return None

    def get_stats(self) -> dict:
        return {
            "total_rules": len(self._rules),
            "bootstrap_rules": len(BOOTSTRAP_RULES),
            "learned_rules": len(self._rules) - len(BOOTSTRAP_RULES),
            "total_validations": self._total_validations,
            "total_corrections": self._total_corrections,
            "correction_rate": round(
                self._total_corrections / max(1, self._total_validations), 3),
        }
