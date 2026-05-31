"""Regression: memory recall work-RPC payloads stay bounded + lean (§3.1).

2026-05-31 incident: TieredMemoryGraph._local_keyword_search was UNBOUNDED —
it appended EVERY persistent/mempool node sharing >=1 meaningful word with the
prompt. For a common greeting this matched ~7,125 of ~7,734 nodes, and the
`query` work-RPC then shipped all of them (full text) as the reply → a
19,720,170-byte bus frame that exceeded MAX_FRAME_SIZE (16 MB, core/_frame.py)
→ frame dropped in _flush_outbound_buffer → agno MemoryProxy.query timed out
(5s G19 cap) → recall silently returned NOTHING on the chat hot path.

rFP_bus_payload_contracts §3.1: work-RPC replies stay lean + bounded. FAISS
(layer 1) and mempool semantic (layer 2) already cap to top_k; the keyword
fallback (layer 3) must too — scored by overlap so the strongest matches
survive the cap.
"""
from __future__ import annotations

from titan_hcl.core.memory import TieredMemoryGraph
from titan_hcl.modules.memory_worker import (
    _lean_memory_item, _lean_mempool_item, _MEMPOOL_WIRE_CAP,
)


def _bare_graph() -> TieredMemoryGraph:
    """A TieredMemoryGraph with only _node_store wired — no DB/FAISS setup."""
    g = TieredMemoryGraph.__new__(TieredMemoryGraph)
    g._node_store = {}
    return g


def test_local_keyword_search_bounded_to_top_k():
    """5,000 nodes all share the prompt's words → still only top_k returned."""
    g = _bare_graph()
    g._node_store = {
        i: {"id": i, "type": "MemoryNode", "status": "persistent",
            "user_prompt": "hello my friend",
            "agent_response": f"response {i} about sitting by the window"}
        for i in range(5000)
    }
    hits = g._local_keyword_search("hello friend sitting", top_k=5)
    assert len(hits) == 5, (
        "keyword fallback must cap to top_k — the unbounded dump was the "
        "19.7 MB bus-frame bug")


def test_local_keyword_search_ranks_by_overlap():
    """Higher meaningful-word overlap ranks first within the cap."""
    g = _bare_graph()
    g._node_store = {
        1: {"id": 1, "type": "MemoryNode", "status": "persistent",
            "user_prompt": "alpha beta gamma", "agent_response": ""},
        2: {"id": 2, "type": "MemoryNode", "status": "persistent",
            "user_prompt": "alpha", "agent_response": ""},
    }
    ranked = g._local_keyword_search("alpha beta gamma", top_k=2)
    assert [n["id"] for n in ranked] == [1, 2]


def test_local_keyword_search_stopword_only_prompt_returns_empty():
    """A prompt of only stopwords must not dump the whole store."""
    g = _bare_graph()
    g._node_store = {
        1: {"id": 1, "type": "MemoryNode", "status": "persistent",
            "user_prompt": "the a an", "agent_response": "it is in to"},
    }
    assert g._local_keyword_search("the a an it", top_k=5) == []


def test_lean_memory_item_strips_internal_fields():
    """The wire shape carries only consumer-read fields — no embeddings,
    neuromod_context, embedding_id, or other internal bloat."""
    raw = {
        "id": 42, "user_prompt": "hi", "agent_response": "yo",
        "effective_weight": 2.5, "emotional_intensity": 3,
        "reinforcement_count": 7, "created_at": 1234.5, "status": "persistent",
        # bloat that must NOT cross the wire:
        "embedding": [0.1] * 768, "embedding_id": 99,
        "neuromod_context": "x" * 100000, "base_weight": 1.0,
    }
    lean = _lean_memory_item(raw)
    assert set(lean) == {
        "id", "user_prompt", "agent_response", "effective_weight",
        "emotional_intensity", "reinforcement_count", "created_at", "status"}
    assert "embedding" not in lean and "neuromod_context" not in lean
    assert lean["id"] == "42" and lean["effective_weight"] == 2.5


def test_lean_mempool_item_shape():
    raw = {"id": 1, "user_prompt": "p", "agent_response": "r",
           "mempool_weight": 0.8, "mempool_reinforcements": 2,
           "created_at": 9.0, "embedding": [0.0] * 384}
    lean = _lean_mempool_item(raw)
    assert set(lean) == {
        "id", "user_prompt", "agent_response", "mempool_weight",
        "mempool_reinforcements", "created_at"}
    assert "embedding" not in lean


def test_mempool_wire_cap_is_bounded():
    """The cap constant exists and is a sane finite bound."""
    assert isinstance(_MEMPOOL_WIRE_CAP, int)
    assert 0 < _MEMPOOL_WIRE_CAP <= 5000
