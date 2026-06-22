"""RFP_synthesis_agno_hot_path_optimization §1.2 P1 — enrichment cache correctness.

The P1 fix wraps the two expensive enrichment DB reads (own_language [17],
experience_sources [22]) in the existing PreHook TTL cache. Two non-obvious
behaviors the edits rely on, verified here (gate GO3 — cached == fresh):

  1. `_pre_hook_cache_get` returns None ONLY when absent/expired — a cached
     empty string ("" own_language for a Titan with no compositions) round-trips
     as "", NOT None. (If it returned None for "", every empty turn would re-hit
     the DB — the cache would be useless for the common case.)
  2. The experience read caches a (lines, topics) TUPLE; on a hit the topics
     must round-trip intact so the per-turn CGN_KNOWLEDGE_USAGE emission still
     fires from cache (INV-OPT-1 — behavior preserved, only the DB read skipped).

Run isolated: python -m pytest tests/test_enrichment_cache.py -v -p no:anchorpy
"""
import time

from titan_hcl.modules import agno_hooks
from titan_hcl.modules.agno_hooks import (
    _pre_hook_cache_get, _pre_hook_cache_set, _pre_hook_cache_set_with_ttl)


def _clear():
    agno_hooks._pre_hook_cache.clear()


def test_absent_returns_none():
    _clear()
    assert _pre_hook_cache_get("enrich:own_language") is None   # never written


def test_cached_empty_string_roundtrips_not_none():
    # The own_language [17] path caches "" for a Titan with no compositions so
    # empty turns don't re-hit inner_memory.db. A cached "" must read back as ""
    # (falsy but NOT None) — None means "recompute".
    _clear()
    _pre_hook_cache_set_with_ttl("enrich:own_language", "", ttl_s=90.0)
    v = _pre_hook_cache_get("enrich:own_language")
    assert v == "" and v is not None


def test_nonempty_string_roundtrips_identical():
    # cached == fresh: the stored value is the exact computed string.
    _clear()
    ctx = "### My Own Words\n- \"glim\" (L2)\nMy vocabulary: 412 words.\n\n"
    _pre_hook_cache_set_with_ttl("enrich:own_language", ctx, ttl_s=90.0)
    assert _pre_hook_cache_get("enrich:own_language") == ctx


def test_experience_tuple_roundtrips_with_topics():
    # [22] caches (lines, topics); the topics must survive so the per-turn
    # CGN_KNOWLEDGE_USAGE emission fires identically from cache (INV-OPT-1).
    _clear()
    lines = ["In the last hour: 2 INSIGHT(s).", "Recently acquired: \"CRISPR\"."]
    topics = ["CRISPR", "aurora"]
    _pre_hook_cache_set_with_ttl(
        "enrich:experience_sources", (lines, list(topics)), ttl_s=90.0)
    cached = _pre_hook_cache_get("enrich:experience_sources")
    assert cached is not None
    _lines, _topics = cached[0], list(cached[1])
    assert _lines == lines
    assert _topics == topics            # emission topics preserved through cache


def test_ttl_expiry_recomputes():
    # After the TTL elapses, get returns None → the turn recomputes (fresh read).
    _clear()
    _pre_hook_cache_set_with_ttl("enrich:own_language", "x", ttl_s=0.05)
    assert _pre_hook_cache_get("enrich:own_language") == "x"
    time.sleep(0.08)
    assert _pre_hook_cache_get("enrich:own_language") is None   # expired → recompute


def test_default_ttl_helper_still_works():
    # The 30s-default helper (used by the pre-existing CGN cache) is untouched.
    _clear()
    _pre_hook_cache_set("cgn_grounding", "g")
    assert _pre_hook_cache_get("cgn_grounding") == "g"


# ── INV-OPT-4 kill-switch: chat.enrichment_cache_enabled ──────────────────────
def test_enrichment_cache_enabled_default_on(monkeypatch):
    # No config / key absent → default ON (Maker flag rule).
    monkeypatch.setattr(agno_hooks, "get_params", lambda s: {} if s == "chat" else {})
    assert agno_hooks._enrichment_cache_enabled() is True


def test_enrichment_cache_kill_switch_off(monkeypatch):
    # =false forces a fresh read each turn (the get-site passes None on a hit).
    monkeypatch.setattr(
        agno_hooks, "get_params",
        lambda s: {"enrichment_cache_enabled": False} if s == "chat" else {})
    assert agno_hooks._enrichment_cache_enabled() is False


def test_enrichment_cache_enabled_soft_fails_on(monkeypatch):
    # A config glitch must never disable the cache silently — soft-fail ON.
    def _boom(_s):
        raise RuntimeError("shm read failed")
    monkeypatch.setattr(agno_hooks, "get_params", _boom)
    assert agno_hooks._enrichment_cache_enabled() is True


# ── CGN port-7777 hardcode fix: _local_api_base resolves the configured port ──
def test_local_api_base_default():
    # Bare default matches config.toml [api] (127.0.0.1:7777).
    assert agno_hooks._local_api_base() in (
        "http://127.0.0.1:7777",)  # no monkeypatch → real get_params, T1 default


def test_local_api_base_uses_configured_port(monkeypatch):
    # T3 runs on 7778 — the call MUST follow config, not a hardcoded 7777.
    monkeypatch.setattr(
        agno_hooks, "get_params",
        lambda s: {"host": "127.0.0.1", "port": 7778} if s == "api" else {})
    assert agno_hooks._local_api_base() == "http://127.0.0.1:7778"


def test_local_api_base_bindall_host_loops_back(monkeypatch):
    # host 0.0.0.0 (bind-all) is not dialable → loop back to 127.0.0.1.
    monkeypatch.setattr(
        agno_hooks, "get_params",
        lambda s: {"host": "0.0.0.0", "port": 7778} if s == "api" else {})
    assert agno_hooks._local_api_base() == "http://127.0.0.1:7778"


def test_local_api_base_soft_fails_to_default(monkeypatch):
    def _boom(_s):
        raise RuntimeError("config unavailable")
    monkeypatch.setattr(agno_hooks, "get_params", _boom)
    assert agno_hooks._local_api_base() == "http://127.0.0.1:7777"
