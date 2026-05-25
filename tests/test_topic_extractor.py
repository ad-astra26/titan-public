"""Tests — llm_pipeline.topic_extractor (Phase 3 P3.C).

Deterministic in-process topic-tag derivation from
inner_memory.db.knowledge_concepts.topic. Used by verify_post_async
to populate the conversation-fork TX `topic_tags` carry + arch §7
`topic:<X>` tag list.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest

from titan_hcl.llm_pipeline import topic_extractor as te


def _build_knowledge_db(path: str, topics: list[str]) -> None:
    """Create a knowledge_concepts table with the given topics."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE knowledge_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                UNIQUE(topic)
            )
        """)
        for t in topics:
            conn.execute(
                "INSERT INTO knowledge_concepts (topic, confidence, created_at) "
                "VALUES (?, ?, ?)",
                (t, 0.9, time.time()))
        conn.commit()
    finally:
        conn.close()


class TestTopicExtractor(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "inner.db")
        _build_knowledge_db(self.db_path, [
            "metaplex nft minting",
            "solana",
            "kuzu",
            "trinity homeostasis",
            "the",        # under MIN_TOPIC_CHARS — should be filtered
            "and",        # under MIN_TOPIC_CHARS — should be filtered
            "Prague Castle",
        ])
        te.set_db_path(self.db_path)

    def tearDown(self):
        te.clear_cache_for_test()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── Match correctness ────────────────────────────────────────

    def test_extracts_single_known_topic(self):
        tags = te.extract_topic_tags("hey can we discuss solana?", "")
        self.assertIn("topic:solana", tags)

    def test_extracts_multi_word_topic(self):
        tags = te.extract_topic_tags(
            "I'm debugging Metaplex NFT minting flow.", "")
        self.assertIn("topic:metaplex nft minting", tags)

    def test_case_insensitive_match(self):
        tags = te.extract_topic_tags("Prague CASTLE is cool", "")
        self.assertIn("topic:Prague Castle", tags)

    def test_matches_across_prompt_and_response(self):
        tags = te.extract_topic_tags(
            "ping?", "here we discuss kuzu graphs.")
        self.assertIn("topic:kuzu", tags)

    def test_no_match_returns_empty(self):
        tags = te.extract_topic_tags(
            "totally unrelated chitchat about weather", "")
        self.assertEqual(tags, [])

    def test_empty_input_returns_empty(self):
        self.assertEqual(te.extract_topic_tags("", ""), [])
        self.assertEqual(te.extract_topic_tags(None, None), [])

    # ── Filter behaviors ─────────────────────────────────────────

    def test_stopwords_below_min_chars_not_returned(self):
        """`the` + `and` are in the DB but below MIN_TOPIC_CHARS=4."""
        tags = te.extract_topic_tags("the cat and the dog", "")
        for t in tags:
            self.assertNotIn(t, ["topic:the", "topic:and"])

    def test_tag_cap_enforced(self):
        # Build a tiny DB with many topics that all match.
        big_topics = [f"longtopic-{i:03d}" for i in range(50)]
        _build_knowledge_db(
            os.path.join(self.tmpdir, "big.db"), big_topics)
        te.set_db_path(os.path.join(self.tmpdir, "big.db"))
        text = " ".join(big_topics)
        tags = te.extract_topic_tags(text, "")
        self.assertLessEqual(len(tags), te.MAX_TAGS_PER_TURN)

    def test_extras_merged_and_prefixed(self):
        tags = te.extract_topic_tags(
            "discussing kuzu", "",
            extra=["custom-topic", "topic:already-prefixed"])
        self.assertIn("topic:kuzu", tags)
        self.assertIn("topic:custom-topic", tags)
        self.assertIn("topic:already-prefixed", tags)

    def test_extras_deduped_with_extracted(self):
        tags = te.extract_topic_tags(
            "discussing solana", "",
            extra=["solana"])  # dup with extracted
        # Should appear exactly once.
        self.assertEqual(tags.count("topic:solana"), 1)

    # ── Cache + reload ───────────────────────────────────────────

    def test_cache_reloads_after_ttl(self):
        # First call loads.
        tags_1 = te.extract_topic_tags("solana", "")
        self.assertIn("topic:solana", tags_1)
        # Add a new topic to the DB.
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO knowledge_concepts (topic, confidence, created_at) "
            "VALUES (?, ?, ?)",
            ("brand-new-topic", 0.9, time.time()))
        conn.commit()
        conn.close()
        # Without TTL expiry, new topic is NOT in cache → not matched.
        tags_2 = te.extract_topic_tags("brand-new-topic mention", "")
        self.assertNotIn("topic:brand-new-topic", tags_2)
        # Force TTL expiry — clear cache + re-set path triggers reload.
        te.clear_cache_for_test()
        te.set_db_path(self.db_path)
        tags_3 = te.extract_topic_tags("brand-new-topic mention", "")
        self.assertIn("topic:brand-new-topic", tags_3)

    # ── Soft-failure ─────────────────────────────────────────────

    def test_missing_db_returns_empty(self):
        te.set_db_path("/nonexistent/path/to.db")
        tags = te.extract_topic_tags("anything", "anything else")
        self.assertEqual(tags, [])

    def test_corrupt_db_returns_empty_no_raise(self):
        bad_path = os.path.join(self.tmpdir, "bad.db")
        with open(bad_path, "wb") as f:
            f.write(b"not-a-sqlite-file")
        te.set_db_path(bad_path)
        tags = te.extract_topic_tags("solana", "")
        self.assertEqual(tags, [])


if __name__ == "__main__":
    unittest.main()
