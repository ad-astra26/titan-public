"""
titan_plugin/logic/sentence_pattern.py — Sentence pattern analysis from composition history.

Reads Titan's composition_history table and extracts:
- Level distribution (which template levels are used most)
- Word frequency and success rates
- Template preferences
- Slot fill rates

Used by LanguageLearningPlugin.summarize_for_distillation() during dreaming
to compress raw language experience into actionable patterns.
"""

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("titan.sentence_pattern")


class SentencePatternExtractor:
    """Analyze Titan's composition history for language learning patterns."""

    # Fixed word type mapping for function words (not in vocabulary)
    FIXED_WORDS = {
        "i": "SELF", "me": "SELF", "my": "SELF",
        "feel": "VERB", "am": "VERB", "want": "VERB", "need": "VERB",
        "is": "VERB", "are": "VERB", "was": "VERB", "have": "VERB",
        "to": "PREP", "and": "CONJ", "but": "CONJ", "or": "CONJ",
        "when": "COND", "because": "CAUSE", "so": "CAUSE",
        "the": "DET", "a": "DET", "an": "DET",
        "not": "NEG", "do": "AUX", "does": "AUX",
        "that": "PRON", "this": "PRON", "it": "PRON",
    }

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path("data") / "inner_memory.db")
        self._db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def extract_patterns(self, since_ts: float = 0) -> dict:
        """Extract composition patterns from history.

        Returns dict with:
            level_distribution: {level: count}
            avg_confidence: float
            word_frequency: {word: count}
            template_preferences: {template: count}
            intent_distribution: {intent: count}
            slot_fill_rate: float
            total_compositions: int
        """
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT level, template, words_used, confidence, "
                "slots_filled, slots_total, intent "
                "FROM composition_history WHERE timestamp > ?",
                (since_ts,),
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[SentencePattern] DB error: %s", e)
            return {"total_compositions": 0}

        if not rows:
            return {"total_compositions": 0}

        level_dist = Counter()
        template_prefs = Counter()
        intent_dist = Counter()
        word_freq = Counter()
        total_conf = 0.0
        total_fill = 0.0
        fill_count = 0

        for level, template, words_json, conf, filled, total, intent in rows:
            level_dist[level or 0] += 1
            template_prefs[template or "?"] += 1
            intent_dist[intent or "default"] += 1
            total_conf += conf or 0.0

            if filled is not None and total and total > 0:
                total_fill += filled / total
                fill_count += 1

            if words_json:
                try:
                    words = json.loads(words_json)
                    if isinstance(words, list):
                        word_freq.update(words)
                except (json.JSONDecodeError, TypeError):
                    pass

        n = len(rows)
        return {
            "level_distribution": dict(level_dist),
            "avg_confidence": round(total_conf / n, 4) if n else 0.0,
            "word_frequency": dict(word_freq.most_common(30)),
            "template_preferences": dict(template_prefs.most_common(10)),
            "intent_distribution": dict(intent_dist),
            "slot_fill_rate": round(total_fill / fill_count, 4) if fill_count else 0.0,
            "total_compositions": n,
        }

    def get_word_success_rates(self, since_ts: float = 0) -> dict:
        """Compute per-word success rate (ratio of high-confidence appearances).

        Returns {word: success_rate} where success_rate is fraction of
        compositions using that word with confidence > 0.6.
        """
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT words_used, confidence FROM composition_history "
                "WHERE timestamp > ?",
                (since_ts,),
            )
            rows = cur.fetchall()
            conn.close()
        except Exception:
            return {}

        total_uses = Counter()
        success_uses = Counter()

        for words_json, conf in rows:
            if not words_json:
                continue
            try:
                words = json.loads(words_json)
                if not isinstance(words, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for w in words:
                total_uses[w] += 1
                if conf and conf > 0.6:
                    success_uses[w] += 1

        return {
            w: round(success_uses.get(w, 0) / total_uses[w], 4)
            for w in total_uses
            if total_uses[w] >= 3  # minimum 3 uses for meaningful rate
        }

    def extract_sentence_pattern(self, sentence: str, vocabulary: list) -> dict:
        """Extract structural pattern from a sentence using Titan's vocabulary.

        Maps each word to its type: known words use word_type from vocabulary,
        function words use FIXED_WORDS, unknown words become UNK.

        Args:
            sentence: The sentence to analyze
            vocabulary: List of dicts with word, word_type, confidence keys

        Returns:
            Dict with sequence, words, hash, length, known_ratio keys
        """
        vocab_map = {v.get("word", "").lower(): v.get("word_type", "UNK").upper()
                     for v in vocabulary if v.get("word")}

        tokens = sentence.lower().split()
        sequence = []
        clean_words = []
        known = 0

        for w in tokens:
            w = w.strip(".,!?\"'()[]{}:;")
            if not w:
                continue
            clean_words.append(w)
            if w in self.FIXED_WORDS:
                sequence.append(self.FIXED_WORDS[w])
                known += 1
            elif w in vocab_map:
                wtype = vocab_map[w]
                # Normalize common types
                if wtype in ("ADJECTIVE",):
                    wtype = "ADJ"
                elif wtype in ("ADVERB",):
                    wtype = "ADV"
                sequence.append(wtype)
                known += 1
            else:
                sequence.append("UNK")

        pattern_hash = "_".join(sequence) if sequence else ""
        return {
            "sequence": sequence,
            "words": clean_words,
            "hash": pattern_hash,
            "length": len(sequence),
            "known_ratio": known / max(1, len(clean_words)),
        }

    def get_preferred_templates(self, top_k: int = 3, since_ts: float = 0) -> list:
        """Return top-K most-used templates."""
        patterns = self.extract_patterns(since_ts)
        prefs = patterns.get("template_preferences", {})
        return sorted(prefs.keys(), key=lambda k: prefs[k], reverse=True)[:top_k]
