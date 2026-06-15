"""Phase E (RFP_research_resilience §7.E) — agno's SHARED knowledge cache+learner.

agno co-mines the same WAL SQLite cache+learner as knowledge_worker (the
sanctioned INV-RR-1 2-writer exception). Verifies: the config-flag gate, lazy
single-build, and that agno's Phase-2 sage research is cached under the chain's
first backend so the next lookup (cross-call / cross-worker) hits it.

Run isolated: python -m pytest tests/test_phase_e_shared_cache.py -v -p no:anchorpy
"""
import os
import tempfile

import pytest

from titan_hcl.modules.agno_hooks import (
    _get_shared_cache_learner, _cache_phase2_research)
from titan_hcl.logic.knowledge_router import (
    classify_query, normalize_query, query_hash, route)


class _Plugin:
    def __init__(self, kcfg):
        self._full_config = {"knowledge": kcfg}


def _enabled_cfg(tmp):
    return {
        "agno_shared_cache_enabled": True,
        "search_cache_path": os.path.join(tmp, "search_cache.db"),
        "routing_stats_path": os.path.join(tmp, "routing.db"),
    }


# ── flag gate (DEFAULT ON — Maker rule: features ship enabled) ───────────────

def test_explicit_false_is_the_kill_switch(tmp_path):
    # an explicit =false still disables (emergency kill-switch)
    cfg = _enabled_cfg(str(tmp_path))
    cfg["agno_shared_cache_enabled"] = False
    assert _get_shared_cache_learner(_Plugin(cfg)) == (None, None)


def test_default_on_when_flag_absent(tmp_path):
    # Maker rule 2026-06-15: NO forgotten default-OFF flags — absent ⇒ ENABLED
    cfg = _enabled_cfg(str(tmp_path))
    del cfg["agno_shared_cache_enabled"]      # key absent entirely
    cache, learner = _get_shared_cache_learner(_Plugin(cfg))
    assert cache is not None and learner is not None


def test_enabled_builds_cache_and_learner(tmp_path):
    p = _Plugin(_enabled_cfg(str(tmp_path)))
    cache, learner = _get_shared_cache_learner(p)
    assert cache is not None and learner is not None
    assert cache.db_path == os.path.join(str(tmp_path), "search_cache.db")


def test_lazy_single_build(tmp_path):
    p = _Plugin(_enabled_cfg(str(tmp_path)))
    c1, l1 = _get_shared_cache_learner(p)
    c2, l2 = _get_shared_cache_learner(p)
    assert c1 is c2 and l1 is l2     # built once, reused


# ── Phase-2 caching round-trip (the "don't lose agno research" path) ─────────

class _DR:
    def __init__(self, gap):
        self.query_type = classify_query(gap)
        self.normalized = normalize_query(gap)


def test_phase2_research_cached_and_retrievable(tmp_path):
    p = _Plugin(_enabled_cfg(str(tmp_path)))
    cache, learner = _get_shared_cache_learner(p)
    gap = "what is the current weather in Berlin"
    dr = _DR(gap)
    _cache_phase2_research(cache, learner, gap, dr, "Berlin is 14C and cloudy.")
    # the NEXT dispatch checks cache.get(query_hash(normalized, qt, chain[0]))
    backend = route(gap, dr.query_type)[0]
    hit = cache.get(query_hash(dr.normalized, dr.query_type, backend))
    assert hit is not None
    import json
    assert "Berlin" in json.loads(hit.result_json).get("raw_text", "")


def test_phase2_noop_when_cache_none():
    # OFF path: cache=None → no-op, never raises
    _cache_phase2_research(None, None, "x", _DR("x"), "findings")   # no exception


def test_phase2_noop_when_empty_findings(tmp_path):
    p = _Plugin(_enabled_cfg(str(tmp_path)))
    cache, learner = _get_shared_cache_learner(p)
    gap = "quantum entanglement"
    dr = _DR(gap)
    _cache_phase2_research(cache, learner, gap, dr, "")    # empty → no write
    backend = route(gap, dr.query_type)[0]
    assert cache.get(query_hash(dr.normalized, dr.query_type, backend)) is None


def test_init_failure_is_failsafe(monkeypatch):
    # a bad path that can't be created → fail-safe (None, None), never raises
    p = _Plugin({"agno_shared_cache_enabled": True,
                 "search_cache_path": "/proc/cannot/create/here.db"})
    cache, learner = _get_shared_cache_learner(p)
    assert (cache, learner) == (None, None)
