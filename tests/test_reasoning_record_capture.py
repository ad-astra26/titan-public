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


def test_recipe_json_round_trips_on_tool_use(tmp_path):
    """§7.E (E1.1) — the executable recipe captured at the verdict seam persists on
    the leaf and derefs back, so E.1 can symbolically replay a matched composite."""
    import json
    s = _store(tmp_path)
    recipe = json.dumps({"tool_id": "coding_sandbox",
                         "args": {"code": "math.factorial(8)"}}, separators=(",", ":"))
    ok = s.record_tool_use(
        reasoning_id="tx_recipe", goal_class="combinatorics", action="tool",
        oracle_id="coding_sandbox", verdict="true", reward=1.0, features=_feat(),
        signature_text="order 8 routes", recipe_json=recipe)
    assert ok is True
    rec = s.get_record("tx_recipe")
    assert rec is not None
    assert rec["recipe_json"] == recipe
    parsed = json.loads(rec["recipe_json"])
    assert parsed["tool_id"] == "coding_sandbox"
    assert parsed["args"]["code"] == "math.factorial(8)"


def test_recipe_json_defaults_empty_when_not_captured(tmp_path):
    """Backward-compat: a record written without a recipe (cold composite / the
    LLM-fallback lane) has recipe_json='' — no silent None, no break."""
    s = _store(tmp_path)
    s.record_tool_use(
        reasoning_id="tx_norecipe", goal_class="x", action="tool", oracle_id="o",
        verdict="true", reward=1.0, features=_feat(), signature_text="x")
    rec = s.get_record("tx_norecipe")
    assert rec is not None and rec["recipe_json"] == ""


def test_recipe_json_carries_onto_macro(tmp_path):
    """§7.E (E1.1) — write_macro carries the modal leaf recipe so the composite
    itself is replayable (not just its leaves)."""
    import json
    s = _store(tmp_path)
    recipe = json.dumps({"tool_id": "coding_sandbox",
                         "args": {"code": "math.factorial({n})"}}, separators=(",", ":"))
    s.write_macro(
        reasoning_id="macro_perm", goal_class="combinatorics", action="tool",
        signature=_feat(), b_i=5, c=1.0, time_cost=1.0, use_count=5,
        composed_from=["tx_leaf1"], recipe_json=recipe)
    rec = s.get_record("macro_perm")
    assert rec is not None and rec["recipe_json"] == recipe


def test_modal_leaf_recipe_propagates_to_macro(tmp_path):
    """§7.E (E1.1) — the macro-recipe propagation fix (gap caught live 2026-06-13):
    a composite distilled from leaves that carry templatized recipes gets the modal
    (shared) template → E.1 can replay it. Without this, macros had recipe_json='' →
    E.1 could never fire."""
    import json
    s = _store(tmp_path)
    tmpl = json.dumps({"tool_id": "coding_sandbox",
                       "code_template": "math.factorial({p0})",
                       "param_kinds": ["number"], "captured_params": ["8"]},
                      separators=(",", ":"))
    # three combinatorics wins, same template (different captured params)
    for i, rid in enumerate(["lf1", "lf2", "lf3"]):
        s.record_tool_use(reasoning_id=rid, goal_class="combinatorics", action="tool",
                          oracle_id="coding_sandbox", verdict="true", reward=1.0,
                          features=_feat(), signature_text=f"order {8+i} routes",
                          recipe_json=tmpl)
    # one stray leaf with no recipe (must not win the modal)
    s.record_tool_use(reasoning_id="lf4", goal_class="combinatorics", action="tool",
                      oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                      signature_text="x")
    modal = s.modal_leaf_recipe(["lf1", "lf2", "lf3", "lf4"])
    assert modal == tmpl                      # the shared template is the modal
    s.write_macro(reasoning_id="macro_comb", goal_class="combinatorics", action="tool",
                  signature=_feat(), b_i=3, c=1.0, time_cost=1.0, use_count=3,
                  composed_from=["lf1", "lf2", "lf3"], recipe_json=modal)
    assert s.get_record("macro_comb")["recipe_json"] == tmpl   # E.1 can now replay it


def test_modal_leaf_recipe_empty_when_no_leaf_recipe(tmp_path):
    s = _store(tmp_path)
    s.record_tool_use(reasoning_id="nr1", goal_class="g", action="tool", oracle_id="o",
                      verdict="true", reward=1.0, features=_feat(), signature_text="x")
    assert s.modal_leaf_recipe(["nr1"]) == ""   # no recipe → '' → E.1 LLM fallback
    assert s.modal_leaf_recipe([]) == ""


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
