"""Tests for ReasoningStore — the per-use graphed Reasoning record (RFP v1.1 / C1).

Covers DuckDB scalars + FAISS signature + SC-search DEREF round-trip (G9), with
graph=None (the Kuzu node path is soft + exercised live on T3). No torch, no
network. A deterministic fake embedder makes the FAISS round-trip exact.
"""
import hashlib

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.reasoning_store import EMBEDDING_DIM, ReasoningStore


def _fake_embed(text: str):
    # deterministic per-text 384-d vector → identical text embeds identically
    # (so an exact write→query round-trip returns dist ~0).
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _store(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    return ReasoningStore(conn, faiss_path=str(tmp_path / "reasoning_vectors.faiss"),
                          graph=None, embedder=_fake_embed, writer=None)


def _feat():
    return [1.0, 0.4, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_record_tool_use_persists_and_derefs(tmp_path):
    s = _store(tmp_path)
    ok = s.record_tool_use(
        reasoning_id="tx_abc123", goal_class="combinatorics", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="how many ways to order 8 climbing routes")
    assert ok is True
    assert s.count() == 1
    rec = s.get_record("tx_abc123")
    assert rec is not None
    assert rec["kind"] == "tool_use"
    assert rec["goal_class"] == "combinatorics"
    assert rec["action"] == "tool"
    assert rec["verdict"] == "true"
    assert rec["reward"] == 1.0
    assert rec["features"] == _feat()          # the real recallable context
    assert rec["anchor_tx"] == "tx_abc123"      # chain pointer = reasoning_id


def test_sc_search_finds_and_derefs(tmp_path):
    s = _store(tmp_path)
    s.record_tool_use(
        reasoning_id="tx_perm", goal_class="combinatorics", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="permutations of 8 items")
    s.record_tool_use(
        reasoning_id="tx_prime", goal_class="primality", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="is 9973 a prime number")
    # SC-search by the SAME signature text → DEREFs the matching record (G9).
    hits = s.search("permutations of 8 items", k=1)
    assert len(hits) == 1
    assert hits[0]["reasoning_id"] == "tx_perm"
    assert hits[0]["goal_class"] == "combinatorics"


def test_record_idempotent_on_reasoning_id(tmp_path):
    s = _store(tmp_path)
    s.record_tool_use(
        reasoning_id="tx_dup", goal_class="x", action="tool", oracle_id="o",
        verdict="true", reward=1.0, features=_feat(), signature_text="dup")
    s.record_tool_use(
        reasoning_id="tx_dup", goal_class="x", action="tool", oracle_id="o",
        verdict="false", reward=-1.0, features=_feat(), signature_text="dup")
    assert s.count() == 1  # ON CONFLICT DO NOTHING — one row


def test_write_macro(tmp_path):
    s = _store(tmp_path)
    s.record_tool_use(
        reasoning_id="tx_leaf1", goal_class="combinatorics", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="leaf 1")
    ok = s.write_macro(
        reasoning_id="macro_combinatorics", goal_class="combinatorics", action="tool",
        signature=_feat(), b_i=5, c=1.0, time_cost=1.0, use_count=5,
        composed_from=["tx_leaf1"])
    assert ok is True
    rec = s.get_record("macro_combinatorics")
    assert rec is not None and rec["kind"] == "macro_strategy"
    assert s.macros_written == 1


def test_snapshot_export_readable(tmp_path):
    # the snapshot is the read-path past the writer-lock (the /v6 endpoint reads it).
    import json
    s = _store(tmp_path)
    s.record_tool_use(
        reasoning_id="tx_snap", goal_class="combinatorics", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="snap")
    snap_path = tmp_path / "reasoning_snapshot.json"
    assert snap_path.exists()  # written on the write
    snap = json.loads(snap_path.read_text())
    assert snap["count"] == 1
    assert snap["records_written"] == 1
    assert snap["by_kind"].get("tool_use") == 1
    assert any(g["goal_class"] == "combinatorics" and g["wins"] == 1
               for g in snap["by_goal_class"])
    assert snap["recent"][0]["reasoning_id"] == "tx_snap"


def test_no_embedder_still_persists_scalars(tmp_path):
    # no embedder → no FAISS signature, but the DuckDB record still lands (deref ok).
    conn = duckdb.connect(str(tmp_path / "synth2.duckdb"))
    s = ReasoningStore(conn, faiss_path=str(tmp_path / "rv.faiss"),
                       graph=None, embedder=None, writer=None)
    ok = s.record_tool_use(
        reasoning_id="tx_noembed", goal_class="x", action="tool", oracle_id="o",
        verdict="true", reward=1.0, features=_feat(), signature_text="x")
    assert ok is True
    assert s.get_record("tx_noembed") is not None
    assert s.search("x", k=1) == []  # no FAISS → no search hits (graceful)
