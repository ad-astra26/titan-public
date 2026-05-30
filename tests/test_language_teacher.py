"""
Tests for LanguageTeacher and SentencePatternExtractor.extract_sentence_pattern()

Run: python -m pytest tests/test_language_teacher.py -v -p no:anchorpy
"""

from collections import Counter

import pytest

# select_mode() uses random.choices(weights=...) by design (anti-starvation +
# cooldown). Tests that probe neurochemistry→preferred-mode bias therefore run
# the call N=SELECT_MODE_TRIALS times and assert the *expected* mode is the
# most-frequent one. Each trial creates a fresh teacher state via the fixture
# (function-scoped), so cooldown/staleness do not cross trials. Threshold
# rationale: with the +0.30 weight bonus on a target mode vs +0.15 baseline on
# 5-6 competitors, the target wins ~28-40% of the time individually — over 200
# trials, that converges to "most common" with high confidence. Bumped to
# 400 (from 200) on 2026-05-19 after empirical 3% flake rate observed across
# 30-run loops — doubling N halves the stdev and pushes edge cases past 4σ.
SELECT_MODE_TRIALS = 400


def _most_common_mode(teacher_factory, queue, vocab, neuromod_state, n=SELECT_MODE_TRIALS):
    """Run select_mode n times on fresh teacher instances; return most-frequent mode."""
    counts = Counter()
    for _ in range(n):
        t = teacher_factory()
        counts[t.select_mode(queue, vocab, neuromod_state)] += 1
    return counts.most_common(1)[0][0]


# ── LanguageTeacher Tests ──


class TestLanguageTeacher:
    """Test LanguageTeacher mode selection, prompt building, and response parsing."""

    @pytest.fixture
    def teacher(self):
        from titan_hcl.logic.language_teacher import LanguageTeacher
        return LanguageTeacher()

    @pytest.fixture
    def teacher_factory(self):
        from titan_hcl.logic.language_teacher import LanguageTeacher
        return LanguageTeacher

    @pytest.fixture
    def sample_queue(self):
        return [
            {"sentence": "I feel curious and I want to explore",
             "confidence": 0.7, "level": 6, "words_used": ["curious", "explore"],
             "template": "L6", "epoch": 100},
            {"sentence": "I am alive",
             "confidence": 0.9, "level": 3, "words_used": ["alive"],
             "template": "L3", "epoch": 101},
            {"sentence": "when I feel brave I melt",
             "confidence": 0.4, "level": 5, "words_used": ["brave", "melt"],
             "template": "L5", "epoch": 102},
        ]

    @pytest.fixture
    def sample_vocab(self):
        return [
            {"word": "curious", "word_type": "adjective", "confidence": 0.8},
            {"word": "explore", "word_type": "verb", "confidence": 0.6},
            {"word": "alive", "word_type": "adjective", "confidence": 0.9},
            {"word": "brave", "word_type": "adjective", "confidence": 0.3},
            {"word": "melt", "word_type": "verb", "confidence": 0.4},
            {"word": "warm", "word_type": "adjective", "confidence": 0.7},
            {"word": "drift", "word_type": "verb", "confidence": 0.5},
        ]

    def test_select_mode_high_gaba_not_called(self, teacher, sample_queue, sample_vocab):
        """GABA gate is checked by caller, not select_mode. But mode should still work."""
        from titan_hcl.logic.language_teacher import ALL_MODES
        mode = teacher.select_mode(sample_queue, sample_vocab,
                                   {"GABA": 0.5, "DA": 0.3, "NE": 0.3, "5-HT": 0.5})
        assert mode in ALL_MODES

    def test_select_mode_high_da_creative(self, teacher_factory, sample_vocab):
        """High dopamine → creative reinforcement (statistical: most-frequent over N trials).

        Empty queue so queue-analysis bonuses (rare-words→context etc.) do not
        compete with the neuromod signal under test.
        """
        mode = _most_common_mode(teacher_factory, [], sample_vocab,
                                 {"DA": 0.8, "NE": 0.3, "5-HT": 0.5, "GABA": 0.1})
        assert mode == "creative"

    def test_select_mode_high_ne_modeling(self, teacher_factory, sample_vocab):
        """High norepinephrine → modeling (statistical: most-frequent over N trials).

        Uses a high-repetition queue with no rare words so queue-analysis
        does NOT add the rare-words→context bonus (would dominate the
        intentionally-reduced NE→modeling weight per line 210 comment).
        Pattern repetition (most_common_count >= 60%) explicitly adds to
        modeling, compounding the NE bonus.
        """
        repeated_queue = [
            {"sentence": "I feel curious", "confidence": 0.7, "level": 3,
             "words_used": ["curious"], "template": "L3", "epoch": 100},
            {"sentence": "I feel curious", "confidence": 0.7, "level": 3,
             "words_used": ["curious"], "template": "L3", "epoch": 101},
            {"sentence": "I feel curious", "confidence": 0.7, "level": 3,
             "words_used": ["curious"], "template": "L3", "epoch": 102},
        ]
        mode = _most_common_mode(teacher_factory, repeated_queue, sample_vocab,
                                 {"DA": 0.3, "NE": 0.8, "5-HT": 0.5, "GABA": 0.1})
        assert mode == "modeling"

    def test_select_mode_low_sht_meaning(self, teacher_factory, sample_vocab):
        """Low serotonin → meaning enrichment (statistical: most-frequent over N trials).

        Empty queue so queue-analysis bonuses (rare-words→context etc.) do not
        compete with the neuromod signal under test.
        """
        mode = _most_common_mode(teacher_factory, [], sample_vocab,
                                 {"DA": 0.3, "NE": 0.3, "5-HT": 0.1, "GABA": 0.1})
        assert mode == "meaning"

    def test_select_mode_low_confidence_grammar(self, teacher_factory, sample_vocab):
        """Low avg confidence → grammar correction (statistical: most-frequent over N trials).

        Uses a queue with low confidence but with each composition using
        distinct frequent words AND distinct templates — so rare-words bonus
        and pattern-repetition bonus do not compete with the grammar signal
        under test. Each word used ≥2 times so word_counts ≤ 1 condition
        (rare_words) yields < 2 → no context bonus.
        """
        low_conf_queue = [
            {"sentence": "I want explore curious", "confidence": 0.2,
             "words_used": ["explore", "curious"], "template": "L4", "level": 4, "epoch": 1},
            {"sentence": "I am explore curious", "confidence": 0.3,
             "words_used": ["explore", "curious"], "template": "L5", "level": 5, "epoch": 2},
            {"sentence": "feel explore curious", "confidence": 0.1,
             "words_used": ["explore", "curious"], "template": "L6", "level": 6, "epoch": 3},
        ]
        mode = _most_common_mode(teacher_factory, low_conf_queue, sample_vocab,
                                 {"DA": 0.4, "NE": 0.4, "5-HT": 0.5, "GABA": 0.1})
        assert mode == "grammar"

    def test_build_prompt_grammar(self, teacher, sample_queue, sample_vocab):
        """Grammar prompt targets lowest confidence sentence."""
        result = teacher.build_prompt("grammar", sample_queue, sample_vocab)
        assert result["mode"] == "grammar"
        assert "grammar" in result["prompt"].lower() or "error" in result["prompt"].lower()
        assert result["original"]  # Has a sentence
        assert result["max_tokens"] > 0

    def test_build_prompt_modeling_avoids_patterns(self, teacher, sample_queue, sample_vocab):
        """Modeling prompt includes patterns to avoid."""
        result = teacher.build_prompt("modeling", sample_queue, sample_vocab,
                                      patterns_to_avoid=["L6", "L3"])
        assert "L6" in result["prompt"] or "L3" in result["prompt"]
        assert result["mode"] == "modeling"

    def test_build_prompt_meaning_uses_vocab(self, teacher, sample_queue, sample_vocab):
        """Meaning prompt includes vocabulary list."""
        result = teacher.build_prompt("meaning", sample_queue, sample_vocab)
        assert "curious" in result["prompt"] or "alive" in result["prompt"]  # vocab words present

    def test_parse_response_grammar_correct(self, teacher, sample_vocab):
        """CORRECT response means no error found."""
        result = teacher.parse_response("grammar", "CORRECT", "I feel alive", sample_vocab)
        assert result["correction"] is None
        assert result["mode"] == "grammar"

    def test_parse_response_grammar_correction(self, teacher, sample_vocab):
        """Correction response extracts the corrected sentence."""
        result = teacher.parse_response(
            "grammar", "I want to explore",
            "I want explore", sample_vocab)
        assert result["correction"] == "I want to explore"

    def test_parse_response_modeling_pattern(self, teacher, sample_vocab):
        """Modeling response extracts pattern hash."""
        result = teacher.parse_response(
            "modeling", "When I feel brave, I explore.",
            "I am alive", sample_vocab)
        assert result["pattern"] is not None
        assert "hash" in result["pattern"]
        assert result["pattern"]["source"] == "teacher"

    def test_parse_response_validates_vocabulary(self, teacher, sample_vocab):
        """Response with mostly unknown words is flagged as invalid."""
        result = teacher.parse_response(
            "creative",
            "The quintessential paradigmatic framework transcends epistemological boundaries.",
            "I feel alive", sample_vocab)
        assert result["is_valid"] is False

    def test_compute_interval_scaling(self, teacher):
        """Interval increases as confidence grows.

        Thresholds lowered in commit `9b0d3a93` (CGN Phase 4b + Language Worker
        Phase 4+5) to teach early-stage Titans more aggressively — matches
        rFP_teachers_update.md Phase T3 design intent + memory rule
        "early-stage Titans need MORE teacher cycles, not fewer".
        """
        assert teacher.compute_interval(0.2) == 2   # <0.4 → intensive
        assert teacher.compute_interval(0.5) == 3   # <0.6
        assert teacher.compute_interval(0.7) == 5   # <0.8 (was 10)
        assert teacher.compute_interval(0.85) == 10  # <0.95 (was 20)
        assert teacher.compute_interval(0.98) == 50  # ≥0.95 graduate-level

    def test_select_mode_rich_format_setpoint_relative(self, teacher_factory, sample_vocab):
        """Rich format uses setpoint-relative deviation (statistical: most-frequent over N trials).

        Intent: verify DA setpoint-deviation (33%) outranks NE setpoint-deviation (19%)
        purely from neuromod weights. Uses empty queue so queue-analysis bonuses
        (rare-words→context, pattern-repetition→modeling, L9→reasoning) do not
        compete with the neuromod signal under test.
        """
        # NE=0.83 looks high (absolute > 0.6), but setpoint=0.70 means only +19% deviation
        # DA=0.60 with setpoint=0.45 means +33% deviation — DA should win
        mode = _most_common_mode(teacher_factory, [], sample_vocab, {
            "DA": {"level": 0.60, "setpoint": 0.45},
            "NE": {"level": 0.83, "setpoint": 0.70},
            "5-HT": {"level": 0.93, "setpoint": 0.70},
            "GABA": {"level": 0.10, "setpoint": 0.30},
        })
        assert mode == "creative"  # DA deviation (33%) > NE deviation (19%)

    def test_select_mode_no_dominant_neuromod_uses_queue(self, teacher, sample_vocab):
        """When all neuromods are near setpoint, queue analysis takes over."""
        from titan_hcl.logic.language_teacher import ALL_MODES
        balanced_queue = [
            {"sentence": "I feel curious and I explore",
             "confidence": 0.7, "level": 6, "words_used": ["curious", "explore"],
             "template": "L6", "epoch": 100},
            {"sentence": "I feel alive and warm",
             "confidence": 0.8, "level": 6, "words_used": ["alive", "warm"],
             "template": "L6", "epoch": 101},
            {"sentence": "I feel brave",
             "confidence": 0.75, "level": 3, "words_used": ["brave"],
             "template": "L3", "epoch": 102},
        ]
        mode = teacher.select_mode(balanced_queue, sample_vocab, {
            "DA": {"level": 0.46, "setpoint": 0.45},   # +2% → below threshold
            "NE": {"level": 0.72, "setpoint": 0.70},   # +3% → below threshold
            "5-HT": {"level": 0.68, "setpoint": 0.70}, # -3% → below threshold
            "GABA": {"level": 0.28, "setpoint": 0.30},
        })
        # No dominant neuromod, weighted random across ALL_MODES (8 modes since 9b0d3a93)
        assert mode in ALL_MODES

    def test_neuromod_deviation_helper(self, teacher):
        """_neuromod_deviation computes correct values for both formats."""
        # Legacy float format: assumes setpoint=0.5
        assert abs(teacher._neuromod_deviation({"DA": 0.8}, "DA") - 0.6) < 0.01
        assert abs(teacher._neuromod_deviation({"DA": 0.3}, "DA") - (-0.4)) < 0.01
        # Rich dict format
        assert abs(teacher._neuromod_deviation(
            {"NE": {"level": 0.83, "setpoint": 0.70}}, "NE") - 0.186) < 0.01
        # Missing key → 0.0
        assert teacher._neuromod_deviation({}, "DA") == 0.0

    def test_empty_queue(self, teacher, sample_vocab):
        """Empty queue + no neuromod signal → uniform-weighted random across active modes.

        With empty queue, queue-analysis adds no weight; with DA=0.5 → deviation=0,
        no neuromod weight added. Result: all active modes at baseline 0.15
        (syntax_advancement gated by vocab<200; conversation gated by vocab<15;
        embodied_reasoning gated by no cross_priority_words). Returned mode must
        be a valid active mode.
        """
        from titan_hcl.logic.language_teacher import ALL_MODES
        mode = teacher.select_mode([], sample_vocab, {"DA": 0.5})
        assert mode in ALL_MODES


# ── SentencePatternExtractor Tests ──


class TestSentencePatternExtract:
    """Test extract_sentence_pattern() method."""

    @pytest.fixture
    def extractor(self):
        from titan_hcl.logic.sentence_pattern import SentencePatternExtractor
        return SentencePatternExtractor(db_path=":memory:")

    @pytest.fixture
    def vocab(self):
        return [
            {"word": "curious", "word_type": "adjective"},
            {"word": "explore", "word_type": "verb"},
            {"word": "alive", "word_type": "adjective"},
            {"word": "warm", "word_type": "adjective"},
        ]

    def test_known_words_mapped(self, extractor, vocab):
        """Known vocabulary words are mapped to their word_type."""
        result = extractor.extract_sentence_pattern("I feel curious", vocab)
        assert "SELF" in result["sequence"]  # "I" from FIXED_WORDS
        assert "VERB" in result["sequence"]  # "feel" from FIXED_WORDS
        assert "ADJECTIVE" in result["sequence"] or "ADJ" in result["sequence"]

    def test_unknown_words_unk(self, extractor, vocab):
        """Unknown words become UNK."""
        result = extractor.extract_sentence_pattern("I feel quintessential", vocab)
        assert "UNK" in result["sequence"]

    def test_fixed_words_mapped(self, extractor, vocab):
        """Function words use FIXED_WORDS mapping."""
        result = extractor.extract_sentence_pattern("I and you", vocab)
        assert result["sequence"][0] == "SELF"  # "I"
        assert result["sequence"][1] == "CONJ"  # "and"

    def test_hash_deterministic(self, extractor, vocab):
        """Same sentence produces same hash."""
        r1 = extractor.extract_sentence_pattern("I feel curious", vocab)
        r2 = extractor.extract_sentence_pattern("I feel curious", vocab)
        assert r1["hash"] == r2["hash"]

    def test_known_ratio_computed(self, extractor, vocab):
        """known_ratio reflects fraction of recognized words."""
        result = extractor.extract_sentence_pattern("I feel curious", vocab)
        assert result["known_ratio"] == 1.0  # All words known

        result2 = extractor.extract_sentence_pattern("I feel xyzzy", vocab)
        assert result2["known_ratio"] < 1.0  # "xyzzy" unknown

    def test_punctuation_stripped(self, extractor, vocab):
        """Punctuation is stripped before matching."""
        result = extractor.extract_sentence_pattern("I feel curious!", vocab)
        assert "UNK" not in result["sequence"]  # "curious!" should match "curious"

    def test_empty_sentence(self, extractor, vocab):
        """Empty sentence returns empty pattern."""
        result = extractor.extract_sentence_pattern("", vocab)
        assert result["length"] == 0
        assert result["hash"] == ""
