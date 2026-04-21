"""Unit tests for titan_plugin.logic.knowledge_router (KP-0).

Covers: normalize_query, query_hash, classify_query across all 7 types,
and route() backend-chain resolution. Pure-function tests — no network,
no DB, no mocks.

See: titan-docs/rFP_knowledge_pipeline_v2.md §3.1.
"""

import pytest

from titan_plugin.logic.knowledge_router import (
    QueryType,
    classify_query,
    normalize_query,
    query_hash,
    route,
)


# ── normalize_query ──────────────────────────────────────────────────

class TestNormalizeQuery:
    def test_empty_and_none(self):
        assert normalize_query("") == ""
        assert normalize_query(None) == ""  # type: ignore[arg-type]

    def test_lowercases(self):
        assert normalize_query("Python") == "python"
        assert normalize_query("PYTHON") == "python"

    def test_strips_outer_whitespace(self):
        assert normalize_query("  hypothesis  ") == "hypothesis"
        assert normalize_query("\t\npython\n") == "python"

    def test_collapses_internal_whitespace(self):
        assert normalize_query("foo   bar") == "foo bar"
        assert normalize_query("foo\tbar") == "foo bar"
        assert normalize_query("foo \n  bar") == "foo bar"

    def test_idempotent(self):
        s = "hypothesis generation"
        assert normalize_query(normalize_query(s)) == normalize_query(s)


# ── query_hash ───────────────────────────────────────────────────────

class TestQueryHash:
    def test_deterministic(self):
        h1 = query_hash("python", QueryType.TECHNICAL, "searxng")
        h2 = query_hash("python", QueryType.TECHNICAL, "searxng")
        assert h1 == h2

    def test_differs_by_type(self):
        h1 = query_hash("python", QueryType.TECHNICAL, "searxng")
        h2 = query_hash("python", QueryType.DICTIONARY, "searxng")
        assert h1 != h2

    def test_differs_by_backend(self):
        h1 = query_hash("python", QueryType.DICTIONARY, "wiktionary")
        h2 = query_hash("python", QueryType.DICTIONARY, "wikipedia_direct")
        assert h1 != h2

    def test_returns_sha256_length(self):
        h = query_hash("hello", QueryType.DICTIONARY, "wiktionary")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── classify_query: internal rejection ───────────────────────────────

class TestInternalRejected:
    @pytest.mark.parametrize("topic", [
        "inner_spirit",
        "outer_perception",
        "chi_total",
        "self_reasoning",
    ])
    def test_underscore_no_space(self, topic):
        assert classify_query(topic) == QueryType.INTERNAL_REJECTED

    @pytest.mark.parametrize("topic", [
        "FORMULATE.load_wisdom",
        "META.state_audit",
        "INTROSPECT.coherence_check",
    ])
    def test_primitive_submode(self, topic):
        assert classify_query(topic) == QueryType.INTERNAL_REJECTED

    @pytest.mark.parametrize("topic", [
        "DA",      # 2-char neuromod code
        "NE",      # 2-char neuromod code
        "5HT",     # 3-char, has digit
        "ACh",     # 3-char, mixed case
        "CGN",     # 3-char, all uppercase (subsystem acronym)
        "MSL",     # 3-char, all uppercase (subsystem acronym)
        "API",     # 3-char, all uppercase (acronym)
    ])
    def test_too_short_or_coded(self, topic):
        assert classify_query(topic) == QueryType.INTERNAL_REJECTED

    @pytest.mark.parametrize("topic", [
        "own",     # 3-char lowercase alpha — valid dictionary query
        "chi",     # 3-char lowercase alpha — Greek letter (ambiguous with
                   # Titan chi but Titan uses chi_energy/chi_total form)
        "dna",     # 3-char lowercase alpha — real word
    ])
    def test_3char_lowercase_alpha_is_dictionary_not_rejected(self, topic):
        # Bugfix regression: "< 4 chars" was over-rejecting valid short
        # dictionary queries like "own" (rFP §3.1 table example). Router
        # now only rejects 3-char tokens that are all-upper / mixed-case /
        # contain digits.
        assert classify_query(topic) == QueryType.DICTIONARY

    def test_empty(self):
        assert classify_query("") == QueryType.INTERNAL_REJECTED
        assert classify_query(None) == QueryType.INTERNAL_REJECTED


# ── classify_query: news ─────────────────────────────────────────────

class TestNewsQueries:
    @pytest.mark.parametrize("topic", [
        "today in technology",
        "latest AI breakthroughs",
        "breaking news about space",
        "python news this week",  # tech marker AND news — news wins
        "recent developments in physics",
    ])
    def test_news_markers_fire(self, topic):
        assert classify_query(topic) == QueryType.NEWS


# ── classify_query: technical ────────────────────────────────────────

class TestTechnicalQueries:
    @pytest.mark.parametrize("topic", [
        "python async await example",
        "docker kubernetes deploy",
        "sql query optimization",
        "rust golang comparison",
        "json api design",
        "memory leak debugging",
        "regex unicode escape",
    ])
    def test_tech_markers_fire(self, topic):
        assert classify_query(topic) == QueryType.TECHNICAL


# ── classify_query: dictionary variants ──────────────────────────────

class TestDictionaryQueries:
    @pytest.mark.parametrize("topic", [
        "hypothesis",
        "ontology",
        "grammar",
        "noun",
        "own",
    ])
    def test_single_word_is_dictionary(self, topic):
        assert classify_query(topic) == QueryType.DICTIONARY

    @pytest.mark.parametrize("topic", [
        "own meaning",
        "hypothesis definition",
        "noun etymology",
        "mitochondrion definition",
    ])
    def test_word_plus_definition_is_phrase(self, topic):
        assert classify_query(topic) == QueryType.DICTIONARY_PHRASE


# ── classify_query: wikipedia-like ───────────────────────────────────

class TestWikipediaLike:
    @pytest.mark.parametrize("topic", [
        "mitochondrial biogenesis",
        "french revolution",
        "quantum entanglement",
        "brownian motion",
        "fermats last theorem",  # 3-word noun phrase without abstract markers
    ])
    def test_short_noun_phrase_is_wikipedia(self, topic):
        assert classify_query(topic) == QueryType.WIKIPEDIA_LIKE


# ── classify_query: conceptual ───────────────────────────────────────

class TestConceptualQueries:
    @pytest.mark.parametrize("topic", [
        "hypothesis generation critical thinking",
        "cognitive strategies for problem solving",
        "how to learn a new language effectively",
        "why do we dream",
        "metacognition and self-reflection methods",
    ])
    def test_abstract_multi_word_is_conceptual(self, topic):
        assert classify_query(topic) == QueryType.CONCEPTUAL

    def test_stopword_triggers_conceptual(self):
        # "the french revolution" — stopword kicks out of wikipedia_like
        assert classify_query("the french revolution") == QueryType.CONCEPTUAL


# ── route() backend-chain resolution ─────────────────────────────────

class TestRoute:
    def test_dictionary_chain(self):
        chain = route("hypothesis")
        assert chain == ["wiktionary", "free_dictionary", "wikipedia_direct"]

    def test_dictionary_phrase_chain(self):
        chain = route("own meaning")
        assert chain == ["wiktionary", "wikipedia_direct"]

    def test_wikipedia_like_chain(self):
        chain = route("mitochondrial biogenesis")
        assert chain[0] == "wikipedia_direct"

    def test_conceptual_chain(self):
        chain = route("hypothesis generation critical thinking")
        assert chain[0] == "searxng_ddg_brave_wiki"

    def test_technical_chain(self):
        chain = route("python async deadlock")
        assert chain[0] == "searxng_ddg_stackoverflow"

    def test_news_chain(self):
        chain = route("today in technology")
        assert chain[0] == "news_api"

    def test_internal_rejected_empty_chain(self):
        chain = route("inner_spirit")
        assert chain == []

    def test_explicit_query_type_override(self):
        # Caller forces the type — skips classification
        chain = route("python", qt=QueryType.DICTIONARY)
        assert chain == ["wiktionary", "free_dictionary", "wikipedia_direct"]

    def test_returns_copy_not_reference(self):
        c1 = route("hypothesis")
        c1.append("tampered")
        c2 = route("hypothesis")
        assert "tampered" not in c2


# ── QueryType enum stability ─────────────────────────────────────────

class TestQueryTypeEnum:
    def test_values_are_strings(self):
        for qt in QueryType:
            assert isinstance(qt.value, str)

    def test_seven_types(self):
        assert len(list(QueryType)) == 7

    def test_stable_values(self):
        # These strings are serialized to cache DB + decision log; changing
        # them breaks on-disk state. Fail loudly if someone renames.
        assert QueryType.DICTIONARY.value == "dictionary"
        assert QueryType.DICTIONARY_PHRASE.value == "dictionary_phrase"
        assert QueryType.WIKIPEDIA_LIKE.value == "wikipedia_like"
        assert QueryType.CONCEPTUAL.value == "conceptual"
        assert QueryType.TECHNICAL.value == "technical"
        assert QueryType.NEWS.value == "news"
        assert QueryType.INTERNAL_REJECTED.value == "internal_rejected"
