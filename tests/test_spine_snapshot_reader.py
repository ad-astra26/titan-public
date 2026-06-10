"""SnapshotSpineReader — the chat-path self/concept recall surface (RFP §7.P4).

Proves the chat path (agno) can now run concept- AND self-granularity recall
WITHOUT the live Kuzu lock, by reading the atomic `spine_snapshot.json`. The
self-engram set is derived from `domain_hint == "self"` — the SAME predicate the
live `SELF_HAS_ENGRAM` linker uses (direct_memory.py:828) — so it is faithful to
the hub. This closes the "chat resolves from the SELF node" gap.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from titan_hcl.synthesis.recall import EngineRecall
from titan_hcl.synthesis.spine_snapshot_reader import SnapshotSpineReader


def _write_snapshot(tmp_path, concepts):
    p = tmp_path / "spine_snapshot.json"
    p.write_text(json.dumps({"version": 1, "concepts": concepts}))
    return p


_CONCEPTS = [
    {"concept_id": "self_a", "version": 1, "name": "Old Self Reflection",
     "memory_type": "episodic", "groundedness": 0.4, "anchor_tx": "tx_a1",
     "created_at": 1000.0, "domain_hint": "self"},
    {"concept_id": "self_a", "version": 2, "name": "Daily Self-Reflection",
     "memory_type": "episodic", "groundedness": 0.6, "anchor_tx": "tx_a2",
     "created_at": 3000.0, "domain_hint": "self"},          # latest version wins
    {"concept_id": "self_b", "version": 1, "name": "Sovereignty Arc",
     "memory_type": "episodic", "groundedness": 0.2, "anchor_tx": "tx_b",
     "created_at": 2000.0, "domain_hint": "self"},
    {"concept_id": "coding_c", "version": 1, "name": "Solana RPC",
     "memory_type": "procedural", "groundedness": 0.9, "anchor_tx": "tx_c",
     "created_at": 2500.0, "domain_hint": "coding"},        # non-self → excluded
]


def _engine(reader):
    return EngineRecall(rule_evaluator=MagicMock(), activation_lookup=lambda _: {},
                        embedder=None, kuzu_reader=reader)


def test_spine_list_concepts_latest_version_and_groundedness_order(tmp_path):
    r = SnapshotSpineReader(snapshot_path=str(_write_snapshot(tmp_path, _CONCEPTS)))
    rows = r.spine_list_concepts(limit=10)
    by_id = {c["concept_id"]: c for c in rows}
    assert by_id["self_a"]["version"] == 2                  # latest version
    assert by_id["self_a"]["name"] == "Daily Self-Reflection"
    # ordered by groundedness DESC (Solana RPC 0.9 first).
    assert rows[0]["name"] == "Solana RPC"


def test_spine_self_recall_filters_domain_self_newest_first(tmp_path):
    r = SnapshotSpineReader(snapshot_path=str(_write_snapshot(tmp_path, _CONCEPTS)))
    hub = r.spine_self_recall()
    names = [e["name"] for e in hub["engrams"]]
    assert names == ["Daily Self-Reflection", "Sovereignty Arc"]  # newest-first, self only
    assert "Solana RPC" not in names                        # non-self excluded
    # carries the deref handle + groundedness EngineRecall needs.
    assert hub["engrams"][0]["anchor_tx"] == "tx_a2"
    assert hub["engrams"][0]["groundedness"] == 0.6
    assert hub["skills"] == []                              # Production not in snapshot


def test_missing_snapshot_soft_fails(tmp_path):
    r = SnapshotSpineReader(snapshot_path=str(tmp_path / "nope.json"))
    assert r.spine_list_concepts() == []
    assert r.spine_self_recall() == {"engrams": [], "skills": []}


def test_corrupt_snapshot_soft_fails(tmp_path):
    p = tmp_path / "spine_snapshot.json"
    p.write_text("{not valid json")
    r = SnapshotSpineReader(snapshot_path=str(p))
    assert r.spine_list_concepts() == []
    assert r.spine_self_recall()["engrams"] == []


# ── the chat-path proof: EngineRecall over the snapshot reader ──────────────

def test_chat_path_self_recall_surfaces_self_hub(tmp_path):
    """The exact chat-path wiring: EngineRecall(kuzu_reader=SnapshotSpineReader)
    → recall(granularity="self") surfaces the self-engrams (and excludes non-self).
    This is what makes 'what have I learned about myself' resolve from the SELF
    node in chat (G5)."""
    reader = SnapshotSpineReader(snapshot_path=str(_write_snapshot(tmp_path, _CONCEPTS)))
    results = _engine(reader).recall("what have I learned about myself",
                                     granularity="self", k=10)
    assert results is not None
    summaries = {r.summary for r in results}
    assert "Daily Self-Reflection" in summaries
    assert "Sovereignty Arc" in summaries
    assert "Solana RPC" not in summaries
    diary = next(r for r in results if r.summary == "Daily Self-Reflection")
    assert diary.fork == "self_hub" and diary.source == "synthesis_self_recall"
    assert diary.tx_hash == "tx_a2"                         # anchor_tx deref handle


def test_chat_path_concept_recall_now_works(tmp_path):
    """The same snapshot reader also un-breaks concept-granularity recall in chat
    (it was silently None-gated when kuzu_reader=None)."""
    reader = SnapshotSpineReader(snapshot_path=str(_write_snapshot(tmp_path, _CONCEPTS)))
    results = _engine(reader).recall("solana", granularity="concept", k=5)
    assert results is not None
    assert any(r.summary == "Solana RPC" for r in results)


def test_chat_path_self_recall_empty_snapshot_falls_back(tmp_path):
    """No self-engrams yet → recall returns None (caller falls back), never errors."""
    reader = SnapshotSpineReader(snapshot_path=str(_write_snapshot(tmp_path, [
        {"concept_id": "coding_c", "version": 1, "name": "Solana RPC",
         "memory_type": "procedural", "groundedness": 0.9, "anchor_tx": "tx_c",
         "created_at": 2500.0, "domain_hint": "coding"}])))
    assert _engine(reader).recall("who am i", granularity="self") is None
