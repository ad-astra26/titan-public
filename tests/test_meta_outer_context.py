"""
tests/test_meta_outer_context.py — Unit tests for rFP_titan_meta_outer_layer.

Covers:
  - _LRUTTLCache hit/miss/expiry + eviction
  - PeerCGNReader: missing files graceful, mtime-based cache
  - activation flag file lifecycle (is_active / set_active)
  - OuterContextReader.compose_recall_query: shape, budget, partial timeout
  - OuterContextReader.get_* convenience readers graceful on missing DBs
  - Regression: reader with is_active=False still usable (no error)
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from titan_plugin.logic.meta_outer_context import (
    _LRUTTLCache,
    FeltExperiencesReader,
    OuterContextConfig,
    OuterContextReader,
    PeerCGNReader,
    SocialGraphReader,
    is_active,
    set_active,
)


# ── _LRUTTLCache ──────────────────────────────────────────────────────────

def test_cache_basic_hit_miss():
    c = _LRUTTLCache(max_size=3, ttl_s=60.0)
    assert c.get(("k", 1)) is None
    assert c.misses == 1
    c.set(("k", 1), "v1")
    assert c.get(("k", 1)) == "v1"
    assert c.hits == 1


def test_cache_eviction_lru():
    c = _LRUTTLCache(max_size=2, ttl_s=60.0)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("a") is None  # evicted as LRU
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.evictions >= 1


def test_cache_ttl_expiry():
    c = _LRUTTLCache(max_size=10, ttl_s=0.05)
    c.set("k", "v")
    assert c.get("k") == "v"
    time.sleep(0.06)
    assert c.get("k") is None  # expired


def test_cache_stats_shape():
    c = _LRUTTLCache(max_size=10, ttl_s=60.0)
    c.set("x", 1)
    s = c.stats()
    assert set(s.keys()) == {"size", "max_size", "ttl_s", "hits",
                              "misses", "evictions", "hit_rate"}
    assert s["size"] == 1


# ── PeerCGNReader ─────────────────────────────────────────────────────────

def test_peer_cgn_missing_files_graceful():
    reader = PeerCGNReader(paths={
        "fake1": "/nonexistent/path1.json",
        "fake2": "/nonexistent/path2.json",
    })
    assert reader.peer_cgn_beta("fake1", "CONCEPT") is None
    assert reader.peer_cgn_alpha("fake2", "CONCEPT") is None
    summary = reader.peer_cgn_summary()
    assert summary["fake1"]["available"] is False
    assert summary["fake2"]["available"] is False


def test_peer_cgn_parse_meta_schema():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False) as f:
        json.dump({
            "primitives": {
                "RECALL": {"beta": 0.72, "alpha": 0.28},
                "FORMULATE": {"beta": 0.65, "alpha": 0.35},
            }
        }, f)
        path = f.name
    try:
        reader = PeerCGNReader(paths={"meta": path})
        assert abs(reader.peer_cgn_beta("meta", "RECALL") - 0.72) < 1e-6
        assert abs(reader.peer_cgn_alpha("meta", "FORMULATE") - 0.35) < 1e-6
        summary = reader.peer_cgn_summary()
        assert summary["meta"]["available"] is True
        assert summary["meta"]["concepts"] == 2
    finally:
        os.unlink(path)


def test_peer_cgn_generic_schema():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False) as f:
        json.dump({"concept_grounding": {"LIBERTY": {"beta": 0.5}}}, f)
        path = f.name
    try:
        reader = PeerCGNReader(paths={"language": path})
        assert reader.peer_cgn_beta("language", "LIBERTY") == 0.5
    finally:
        os.unlink(path)


# ── is_active / set_active flag file ──────────────────────────────────────

def test_activation_flag_lifecycle():
    # Ensure clean start
    set_active(False)
    assert is_active() is False
    set_active(True)
    assert is_active() is True
    set_active(False)
    assert is_active() is False


# ── SocialGraphReader / FeltExperiencesReader graceful ────────────────────

def test_social_reader_missing_db():
    r = SocialGraphReader("/nonexistent/social.db")
    assert r.get_person_profile("anyone") is None
    assert r.get_recent_engagements() == []


def test_felt_reader_missing_db():
    r = FeltExperiencesReader("/nonexistent/felt.db")
    assert r.get_for_person("anyone") == []
    assert r.get_for_topic("anything") == []
    assert r.get_recent() == []


# ── OuterContextReader composed fetch ─────────────────────────────────────

def test_reader_compose_shape_no_dbs():
    """With no DBs available, compose returns graceful empty dict with
    full shape keys (never raises, never partial-keyed)."""
    cfg = OuterContextConfig(
        social_graph_path="/nonexistent/s.db",
        events_teacher_path="/nonexistent/e.db",
        inner_memory_path="/nonexistent/i.db",
        fetch_budget_ms=200,
        peer_cgn_enabled=False,
    )
    reader = OuterContextReader(config=cfg)
    try:
        set_active(True)  # gate on so paths execute
        fut = reader.compose_recall_query({
            "primary_person": "@test", "current_topic": "test_topic"
        })
        ctx = fut.result(timeout=1.0)
        # Full shape
        for key in ("person", "topic", "felt_history", "recent_events",
                     "inner_narrative", "titan_self_snapshot",
                     "peer_cgn", "sources_queried",
                     "sources_failed", "sources_timed_out", "fetch_ms"):
            assert key in ctx
        assert isinstance(ctx["sources_queried"], list)
        assert ctx["fetch_ms"] >= 0
    finally:
        set_active(False)
        reader.shutdown()


def test_reader_inactive_still_usable():
    """Regression: inactive reader still constructs + stats-queries cleanly."""
    set_active(False)
    reader = OuterContextReader()
    try:
        assert reader.is_active() is False
        s = reader.stats()
        assert s["active"] is False
        assert "cache" in s
        # Peer CGN reading works regardless of activation flag
        v = reader.peer_cgn_beta("nonexistent_consumer", "anything")
        assert v is None
    finally:
        reader.shutdown()


def test_reader_config_defaults():
    cfg = OuterContextConfig()
    assert cfg.fetch_budget_ms == 200
    assert cfg.per_read_timeout_ms == 50
    assert cfg.cache_ttl_s == 60
    assert cfg.max_workers == 4
    assert cfg.active_search_knowledge is True
    assert cfg.active_search_x is False
    assert cfg.active_search_events is False
    assert cfg.peer_cgn_enabled is True
    assert cfg.reward_weight == 0.0
