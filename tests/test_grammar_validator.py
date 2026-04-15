"""Tests for GrammarValidator — learned grammar correction rules."""
import os
import pytest
import tempfile
from titan_plugin.logic.grammar_validator import GrammarValidator, GrammarRule


@pytest.fixture
def validator():
    """Create validator with temp DB."""
    tmp = tempfile.mktemp(suffix=".db")
    v = GrammarValidator(db_path=tmp)
    yield v
    if os.path.exists(tmp):
        os.unlink(tmp)


class TestGrammarValidator:
    def test_init_has_bootstrap_rules(self, validator):
        stats = validator.get_stats()
        assert stats["bootstrap_rules"] > 0
        assert stats["total_rules"] >= stats["bootstrap_rules"]

    def test_capitalize_i(self, validator):
        result = validator.validate("i feel alive")
        assert result.startswith("I ")

    def test_collapse_whitespace(self, validator):
        result = validator.validate("I  feel  alive")
        assert "  " not in result

    def test_no_change_for_correct(self, validator):
        result = validator.validate("I feel alive")
        assert result == "I feel alive"

    def test_learn_from_correction(self, validator):
        rule = validator.learn_from_correction(
            "I want explore", "I want to explore"
        )
        assert rule is not None
        assert "want" in rule.pattern or "want" in rule.replacement

    def test_learned_rule_persists(self, validator):
        validator.learn_from_correction(
            "I want explore", "I want to explore"
        )
        # Apply learned rule
        result = validator.validate("I want explore")
        assert "want to" in result

    def test_learn_duplicate_increases_confidence(self, validator):
        r1 = validator.learn_from_correction("I want explore", "I want to explore")
        conf1 = r1.confidence
        r2 = validator.learn_from_correction("I want explore", "I want to explore")
        assert r2.confidence >= conf1

    def test_learn_no_rule_for_identical(self, validator):
        result = validator.learn_from_correction("same", "same")
        assert result is None

    def test_stats_tracking(self, validator):
        validator.validate("test sentence")
        validator.validate("i am alive")
        stats = validator.get_stats()
        assert stats["total_validations"] == 2
        assert stats["total_corrections"] >= 1  # "i" → "I"

    def test_rule_apply(self):
        rule = GrammarRule(1, "want explore", "want to explore")
        result = rule.apply("I want explore things")
        assert result == "I want to explore things"
        assert rule.times_applied == 1
