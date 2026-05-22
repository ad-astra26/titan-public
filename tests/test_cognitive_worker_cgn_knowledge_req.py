"""Chunk B.3 (RFP_meta-reasoning_CGN_FIX.md §4.2 rows 1/2/5) — verify the
cognitive_worker CGN_KNOWLEDGE_REQ handler dispatches by `kind` and
produces the expected response shape for the 3 Session 3 categories:
reasoning, pattern_primitives, chain_archive.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from titan_hcl import bus
from titan_hcl.modules.cognitive_worker import (
    _COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
    _build_chain_archive_response,
    _build_pattern_response,
    _build_reasoning_response,
)


# ──────────────────────────────────────────────────────────────────────
# Subscribe-topic registration
# ──────────────────────────────────────────────────────────────────────


def test_cgn_knowledge_req_in_subscribe_topics():
    """cognitive_worker must subscribe to CGN_KNOWLEDGE_REQ so Session 3
    meta_service resolvers can dispatch to it."""
    assert bus.CGN_KNOWLEDGE_REQ in _COGNITIVE_WORKER_SUBSCRIBE_TOPICS


# ──────────────────────────────────────────────────────────────────────
# _build_reasoning_response
# ──────────────────────────────────────────────────────────────────────


class _FakeReasoningEngine:
    def get_stats(self):
        return {
            "total_chains": 12345,
            "commit_rate": 0.89,
            "avg_chain_length": 3.7,
            "buffer_size": 1024,
        }


def test_reasoning_response_with_engine_returns_real_stats():
    rsp = _build_reasoning_response(
        _FakeReasoningEngine(), "DECOMPOSE",
        {"question_type": "formulate_strategy", "consumer_id": "reasoning"})
    assert rsp["engine"] == "reasoning"
    assert rsp["name"] == "DECOMPOSE"
    assert rsp["chains_total"] == 12345
    assert rsp["chains_commit_rate"] == 0.89
    assert rsp["avg_chain_length"] == 3.7
    assert rsp["buffer_size"] == 1024
    assert rsp["question_type"] == "formulate_strategy"
    assert rsp["consumer_id"] == "reasoning"


def test_reasoning_response_with_none_engine_graceful():
    rsp = _build_reasoning_response(None, "DECOMPOSE", {})
    assert rsp["engine"] == "unavailable"
    assert rsp["name"] == "DECOMPOSE"


def test_reasoning_response_with_broken_stats_returns_zeros():
    class _Broken:
        def get_stats(self):
            raise RuntimeError("boom")
    rsp = _build_reasoning_response(_Broken(), "COMPARE", {})
    assert rsp["engine"] == "reasoning"
    assert rsp["chains_total"] == 0
    assert rsp["chains_commit_rate"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# _build_pattern_response
# ──────────────────────────────────────────────────────────────────────


class _FakeChainArchive:
    def __init__(self, scoring_rows=None, domain_rows=None):
        self._scoring = scoring_rows or []
        self._domain = domain_rows or []

    def query_high_scoring(self, limit=5):
        return self._scoring[:limit]

    def query_by_domain(self, domain, limit=5):
        return [r for r in self._domain if r.get("domain") == domain][:limit]


def test_pattern_response_returns_top_chain_ids():
    fake = _FakeChainArchive(scoring_rows=[
        {"chain_id": 100, "score": 0.9},
        {"chain_id": 200, "score": 0.85},
    ])
    rsp = _build_pattern_response(
        {"chain_archive": fake}, "extract_structure",
        {"question_type": "formulate_strategy"})
    assert rsp["engine"] == "pattern_primitives"
    assert rsp["top_chain_count"] == 2
    assert rsp["top_chain_ids"] == [100, 200]


def test_pattern_response_unavailable_when_no_archive():
    rsp = _build_pattern_response({}, "merge", {})
    assert rsp["engine"] == "unavailable"


# ──────────────────────────────────────────────────────────────────────
# _build_chain_archive_response
# ──────────────────────────────────────────────────────────────────────


def test_chain_archive_response_maps_consumer_to_domain():
    fake = _FakeChainArchive(domain_rows=[
        {"chain_id": 11, "domain": "language"},
        {"chain_id": 22, "domain": "language"},
        {"chain_id": 33, "domain": "knowledge"},  # filtered out
    ])
    rsp = _build_chain_archive_response(
        fake, "query", {"consumer_id": "language"})
    assert rsp["engine"] == "chain_archive"
    assert rsp["domain"] == "language"
    assert rsp["matched_count"] == 2
    assert rsp["top_chain_ids"] == [11, 22]


def test_chain_archive_response_unknown_consumer_uses_general_domain():
    fake = _FakeChainArchive(domain_rows=[
        {"chain_id": 7, "domain": "general"},
    ])
    rsp = _build_chain_archive_response(
        fake, "query", {"consumer_id": "unmapped_consumer"})
    assert rsp["domain"] == "general"
    assert rsp["matched_count"] == 1


def test_chain_archive_response_none_archive_graceful():
    rsp = _build_chain_archive_response(None, "query", {"consumer_id": "x"})
    assert rsp["engine"] == "unavailable"


# ──────────────────────────────────────────────────────────────────────
# Consumer → domain mapping coverage
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("consumer,expected_domain", [
    ("language", "language"),
    ("knowledge", "knowledge"),
    ("social", "outer_perception"),
    ("reasoning", "inner_spirit"),
    ("emotional", "emot"),
    ("self_model", "introspect"),
    ("coding", "knowledge"),
    ("dreaming", "inner_spirit"),
    ("reflection", "introspect"),
])
def test_chain_archive_consumer_domain_mapping(consumer, expected_domain):
    """All 9 KNOWN_CONSUMERS have an explicit domain mapping (else they fall
    through to 'general' which is a less-targeted query)."""
    fake = _FakeChainArchive(domain_rows=[])
    rsp = _build_chain_archive_response(
        fake, "query", {"consumer_id": consumer})
    assert rsp["domain"] == expected_domain
