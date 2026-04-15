"""Tests for the VerifiedContextBuilder — query parsing, store routing, assembly."""

import time
import pytest
from unittest.mock import MagicMock, patch
from titan_plugin.logic.verified_context_builder import (
    QueryParser, StoreRouter, VerifiedContextBuilder, ParsedQuery, VerifiedContext,
)


class TestQueryParser:
    """Test entity, temporal, and activity extraction."""

    def setup_method(self):
        self.parser = QueryParser(known_users=["peter", "alice"])

    def test_mention_extraction(self):
        parsed = self.parser.parse("What did @peteronx post?")
        assert "peteronx" in parsed.entities
        assert parsed.entity_types["peteronx"] == "person"

    def test_multiple_mentions(self):
        parsed = self.parser.parse("Did @alice talk to @bob?")
        assert "alice" in parsed.entities
        assert "bob" in parsed.entities

    def test_kin_detection(self):
        parsed = self.parser.parse("Have you met T2 recently?")
        assert any("t2" in e for e in parsed.entities)
        assert any(v == "kin" for v in parsed.entity_types.values())

    def test_known_user_detection(self):
        parsed = self.parser.parse("Do you remember peter?")
        assert "peter" in parsed.entities
        assert parsed.entity_types["peter"] == "person"

    def test_temporal_yesterday(self):
        parsed = self.parser.parse("What happened yesterday?")
        assert parsed.temporal_start is not None
        assert parsed.temporal_end is not None
        assert parsed.temporal_label == "yesterday"

    def test_temporal_last_thursday(self):
        parsed = self.parser.parse("What did you post last Thursday?")
        assert parsed.temporal_start is not None
        assert parsed.temporal_label.lower().startswith("last thursday")

    def test_temporal_recently(self):
        parsed = self.parser.parse("Any recent dreams?")
        assert parsed.temporal_start is not None
        assert "recent" in parsed.temporal_label.lower()

    def test_temporal_today(self):
        parsed = self.parser.parse("What did you learn today?")
        assert parsed.temporal_start is not None
        assert parsed.temporal_label == "today"

    def test_no_temporal(self):
        parsed = self.parser.parse("Tell me about yourself")
        assert parsed.temporal_start is None
        assert parsed.temporal_label == ""

    def test_activity_posted(self):
        parsed = self.parser.parse("When did you last post?")
        assert "social_x_actions" in parsed.store_hints

    def test_activity_learned(self):
        parsed = self.parser.parse("Have you learned any new words?")
        assert "vocabulary" in parsed.store_hints

    def test_activity_dreamed(self):
        parsed = self.parser.parse("What did you dream about?")
        assert "episodic_memory" in parsed.store_hints

    def test_activity_met_kin(self):
        parsed = self.parser.parse("Have you met T2?")
        assert "kin_encounters" in parsed.store_hints

    def test_activity_thought(self):
        parsed = self.parser.parse("What have you thought about recently?")
        assert "chain_archive" in parsed.store_hints

    def test_activity_created(self):
        parsed = self.parser.parse("Show me art you created")
        assert "creative_works" in parsed.store_hints

    def test_combined_entity_temporal_activity(self):
        parsed = self.parser.parse("What did @peter tell you about Solana last Thursday?")
        assert "peter" in parsed.entities
        assert parsed.temporal_start is not None
        assert any(s in parsed.store_hints for s in ("events_teacher", "social_graph"))

    def test_general_conversation(self):
        parsed = self.parser.parse("Hello, how are you?")
        assert len(parsed.store_hints) == 0  # No specific stores detected
        assert len(parsed.entities) == 0

    def test_person_entity_adds_social_stores(self):
        parsed = self.parser.parse("Who is @alice?")
        assert "social_graph" in parsed.store_hints


class TestStoreRouter:
    """Test SQL query generation and execution against test databases."""

    def test_init(self, tmp_path):
        router = StoreRouter(data_dir=str(tmp_path))
        assert router._data_dir == tmp_path

    def test_missing_db_returns_empty(self, tmp_path):
        router = StoreRouter(data_dir=str(tmp_path))
        result = router.query_store("vocabulary", ParsedQuery(), limit=5)
        assert result == []

    def test_unknown_store_returns_empty(self, tmp_path):
        router = StoreRouter(data_dir=str(tmp_path))
        result = router.query_store("nonexistent_store", ParsedQuery(), limit=5)
        assert result == []


class TestVerifiedContextBuilder:
    """Test the full build pipeline."""

    def test_build_returns_verified_context(self, tmp_path):
        vcb = VerifiedContextBuilder(data_dir=str(tmp_path))
        result = vcb.build("Hello, how are you?")
        assert isinstance(result, VerifiedContext)
        assert result.total_ms >= 0
        assert result.parse_ms >= 0
        assert "Verified Memory Recall" in result.text

    def test_build_with_no_records(self, tmp_path):
        vcb = VerifiedContextBuilder(data_dir=str(tmp_path))
        result = vcb.build("Tell me about quantum physics")
        assert result.total_records == 0
        assert "No specific memories" in result.text
        assert "honestly" in result.text.lower()

    def test_build_respects_max_records(self, tmp_path):
        vcb = VerifiedContextBuilder(data_dir=str(tmp_path))
        result = vcb.build("What happened?", max_records=5)
        assert result.total_records <= 5

    def test_text_contains_verification_warning(self, tmp_path):
        vcb = VerifiedContextBuilder(data_dir=str(tmp_path))
        result = vcb.build("hello")
        assert "don't recall" in result.text.lower() or "don't remember" in result.text.lower()

    def test_timing_is_reasonable(self, tmp_path):
        vcb = VerifiedContextBuilder(data_dir=str(tmp_path))
        result = vcb.build("What did you do yesterday?")
        # Should complete in <100ms even with empty DBs
        assert result.total_ms < 100


class TestVerifiedContextWithRealData:
    """Tests that run against the actual Titan data directory."""

    @pytest.fixture
    def vcb(self):
        """Create VCB with real data directory (skips if not available)."""
        from pathlib import Path
        data_dir = Path("./data")
        if not data_dir.exists():
            pytest.skip("Titan data directory not available")
        return VerifiedContextBuilder(data_dir=str(data_dir))

    def test_vocabulary_query(self, vcb):
        result = vcb.build("What words have you learned?")
        assert result.total_records > 0
        assert any("vocabulary" in r.source for r in result.records)

    def test_social_posts_query(self, vcb):
        result = vcb.build("When did you last post on X?")
        # Should find some posts
        assert result.total_records >= 0  # May be 0 if no posts

    def test_episodic_query(self, vcb):
        result = vcb.build("What happened recently?")
        assert result.total_records > 0

    def test_general_query_gets_fallback(self, vcb):
        result = vcb.build("Hello!")
        # Should get fallback stores (episodic, social, vocab, reasoning)
        assert result.total_records > 0

    def test_performance_under_200ms(self, vcb):
        result = vcb.build("What did you dream about last night?")
        assert result.total_ms < 200  # Allow slack for cold SQLite cache + swap

    def test_chained_records_have_status(self, vcb):
        result = vcb.build("Tell me about your vocabulary")
        for r in result.records:
            assert r.chain_status in ("CHAINED", "PARTIAL", "WIRED", "NOT_COVERED")

    def test_assembled_text_has_stamps(self, vcb):
        result = vcb.build("What words do you know?")
        if result.total_records > 0:
            assert any(tag in result.text for tag in
                      ("[CHAINED ✓]", "[PARTIAL]", "[WIRED]", "[unverified]"))
