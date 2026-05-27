"""Phase 7 integration tests — bus-command → persist → snapshot chain
+ goal-hook helper + agno-tool surface assertions.

Covers:
- BufferCache.set → SYNTHESIS_BUFFER_COMMAND payload → ActrBufferStore.persist
  end-to-end (with the same payload shape the bus carries on the wire).
- ActrBufferStore.buffer_entities returns the agno-set concept_ids — proves
  the spreading-activation source rewire is wired all the way through
  (INV-Syn-18).
- _ground_for_goal_hook helper grounds tokens via plugin.cgn_lexicon when
  set; soft-fails to [] when missing.
- agno_tools.create_tools returns the 4 P7 tools when synthesis_buffer_cache
  is present on the plugin.
- BufferStub import path is gone (INV-Syn-18 — no shim).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import duckdb
import pytest

from titan_hcl.synthesis.buffer_cache import BufferCache
from titan_hcl.synthesis.buffer_store import ActrBufferStore


@pytest.fixture()
def chain(tmp_path):
    """A full bus-command → persist chain: agno cache emits payloads, the
    store consumes them inline (no real bus needed for unit-level reproduction
    of the wire contract)."""
    conn = duckdb.connect(":memory:")
    snap = str(tmp_path / "buffers_snapshot.json")
    store = ActrBufferStore(duckdb_conn=conn, snapshot_path=snap)

    def _route(payload: dict):
        """Mirrors synthesis_worker's SYNTHESIS_BUFFER_COMMAND handler."""
        op = payload["op"]
        if op == "set":
            store.persist(
                chat_id=payload["chat_id"],
                buffer_name=payload["buffer_name"],
                content=payload.get("content", "") or "",
                concept_ids=payload.get("concept_ids", []) or [],
                ts=payload.get("ts"),
            )
        elif op == "clear":
            store.clear(
                chat_id=payload["chat_id"],
                buffer_name=payload["buffer_name"],
            )

    cache = BufferCache(bus_emit=_route, snapshot_path=snap)
    return cache, store, snap


def test_agno_cache_set_persists_via_store(chain):
    cache, store, _ = chain
    cache.set("alice:s1", "goal", content="debug rust", concept_ids=["rust"])
    rows = store.read_all_for_chat("alice:s1")
    assert rows["goal"]["content"] == "debug rust"
    assert rows["goal"]["concept_ids"] == ["rust"]


def test_buffer_entities_picks_up_agno_writes(chain):
    """The full P7 round-trip that the spreading-activation source needs:
    agno LLM writes a buffer → bus event persists → buffer_entities reads
    the concept_ids back. INV-Syn-18 in one test."""
    cache, store, _ = chain
    cache.set("alice:s1", "goal", content="g", concept_ids=["rust", "panic"])
    cache.set("alice:s1", "retrieval", content="r", concept_ids=["past_fix"])
    entities = store.buffer_entities("alice:s1")
    # Goal first per the precedence order.
    assert entities[:2] == ["rust", "panic"]
    assert "past_fix" in entities


def test_clear_round_trip(chain):
    cache, store, _ = chain
    cache.set("alice:s1", "goal", content="x", concept_ids=[])
    cache.clear("alice:s1", "goal")
    assert store.read_all_for_chat("alice:s1") == {}


# ── _ground_for_goal_hook ─────────────────────────────────────────────


def test_ground_helper_uses_plugin_lexicon():
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace(cgn_lexicon={
        "rust": "concept_rust",
        "panic": "concept_panic",
    })
    out = _ground_for_goal_hook(plugin, "I have a rust panic in my code")
    assert "concept_rust" in out
    assert "concept_panic" in out


def test_ground_helper_returns_empty_without_lexicon():
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace()   # no cgn_lexicon attr
    assert _ground_for_goal_hook(plugin, "any text here") == []


def test_ground_helper_soft_fails_on_bad_lexicon():
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace(cgn_lexicon="not-a-dict")
    assert _ground_for_goal_hook(plugin, "any text") == []


def test_ground_helper_empty_text_returns_empty():
    from titan_hcl.modules.agno_worker import _ground_for_goal_hook
    plugin = SimpleNamespace(cgn_lexicon={"rust": "concept_rust"})
    assert _ground_for_goal_hook(plugin, "") == []


# ── agno_tools — P7 tool surface ──────────────────────────────────────


def test_agno_tools_register_4_buffer_tools(chain):
    """Confirms `create_tools(plugin)` returns the 4 P7 tools when the
    plugin has synthesis_buffer_cache wired."""
    from titan_hcl.modules.agno_tools import create_tools
    cache, _, _ = chain
    plugin = SimpleNamespace(
        synthesis_buffer_cache=cache,
        synthesis_tool_plugs={},
        _current_user_id="alice",
        _current_session_id="default",
    )
    tools = create_tools(plugin)
    names = {getattr(t, "__name__", "") for t in tools}
    assert {"read_buffer", "write_buffer", "clear_buffer",
            "query_retrieval"} <= names


# ── INV-Syn-18: BufferStub gone ───────────────────────────────────────


def test_buffer_stub_module_is_gone():
    """INV-Syn-18 enforcement: importing the old module path must fail."""
    # Drop any cached partial import from earlier tests.
    sys.modules.pop("titan_hcl.synthesis.buffer_stub", None)
    with pytest.raises(ImportError):
        import titan_hcl.synthesis.buffer_stub  # noqa: F401
