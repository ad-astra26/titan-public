"""§7.D D-strategy — the Reasoning-composite library (D.1 provenance · D.2 anchor ·
D.3 idea_type · D.4b macro-of-macros provenance).

Real Kuzu graph so the macro node's `idea_type`, the REASONING_COMPOSED_FROM
provenance edges, and the verified-only anchor are asserted end-to-end. No torch,
no network; a deterministic fake embedder makes the FAISS round-trip exact.
"""
import hashlib
import os
import queue
import tempfile

import duckdb
import numpy as np
import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.kuzu_spine_schema import bootstrap_spine_schema
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.reasoning_store import EMBEDDING_DIM, ReasoningStore


def _fake_embed(text: str):
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _feat():
    return [1.0, 0.4, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture()
def store_with_graph():
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "d.kuzu"))
        bootstrap_spine_schema(g)
        conn = duckdb.connect(os.path.join(tmp, "synth.duckdb"))
        s = ReasoningStore(conn, faiss_path=os.path.join(tmp, "rv.faiss"),
                           graph=g, embedder=_fake_embed, writer=None)
        try:
            yield s, g
        finally:
            g.close()


def _node_idea_type(g, rid):
    qr = g._conn.execute(
        "MATCH (r:Reasoning {reasoning_id: $rid}) RETURN r.idea_type", {"rid": rid})
    return qr.get_next()[0] if qr.has_next() else None


def _composed_leaves(g, macro_id):
    qr = g._conn.execute(
        "MATCH (m:Reasoning {reasoning_id: $m})-[:REASONING_COMPOSED_FROM]->"
        "(l:Reasoning) RETURN l.reasoning_id", {"m": macro_id})
    out = []
    while qr.has_next():
        out.append(qr.get_next()[0])
    return sorted(out)


# ── D.1 — the macro→leaf provenance join (synthesis-side) ───────────────
def test_d1_leaf_join_selects_verified_leaves_of_class_action(store_with_graph):
    s, _g = store_with_graph
    s.record_tool_use(reasoning_id="tx1", goal_class="combinatorics", action="tool",
                      oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                      signature_text="a")
    s.record_tool_use(reasoning_id="tx2", goal_class="combinatorics", action="tool",
                      oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                      signature_text="b")
    s.record_tool_use(reasoning_id="tx_fail", goal_class="combinatorics", action="tool",
                      oracle_id="o", verdict="false", reward=-1.0, features=_feat(),
                      signature_text="c")           # not verified → excluded
    s.record_tool_use(reasoning_id="tx_other", goal_class="primality", action="tool",
                      oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                      signature_text="d")            # other class → excluded
    leaves = s.leaf_reasoning_ids("combinatorics", "tool")
    assert set(leaves) == {"tx1", "tx2"}
    assert s.leaf_reasoning_ids("nope") == []
    # action filter narrows
    assert s.leaf_reasoning_ids("combinatorics", "research") == []


# ── D.2 — verified-only Idea-tier anchor (OuterMemoryWriter) ────────────
def test_d2_composite_anchor_is_deterministic_procedural_fork():
    q = queue.Queue()
    w = OuterMemoryWriter(send_queue=q, src="d_test")
    tx1 = w.write_reasoning_composite(reasoning_id="macro::combinatorics::tool",
                                      goal_class="combinatorics", action="tool",
                                      use_count=5, composed_from=["tx1", "tx2"])
    assert isinstance(tx1, str) and len(tx1) == 64    # sha256 hex content-hash
    # an event was emitted on the procedural fork
    evt = q.get_nowait()
    payload = evt["payload"] if isinstance(evt, dict) and "payload" in evt else evt
    # OuterMemoryEvent → the bus event carries fork=procedural + the composite tag
    assert "procedural" in str(evt).lower()
    assert "reasoning_composite" in str(evt).lower()


def test_d2_write_macro_stores_anchor_tx(store_with_graph):
    s, _g = store_with_graph
    s.record_tool_use(reasoning_id="lf1", goal_class="x", action="tool", oracle_id="o",
                      verdict="true", reward=1.0, features=_feat(), signature_text="lf1")
    s.write_macro(reasoning_id="macro::x", goal_class="x", action="tool",
                  signature=_feat(), b_i=5, c=1.0, time_cost=1.0, use_count=5,
                  anchor_tx="deadbeef" * 8, composed_from=["lf1"])
    rec = s.get_record("macro::x")
    assert rec["anchor_tx"] == "deadbeef" * 8       # verified composite is anchored


# ── D.3 — idea_type on the Kuzu node (FC-8) ─────────────────────────────
def test_d3_macro_node_carries_idea_type_procedural(store_with_graph):
    s, g = store_with_graph
    s.record_tool_use(reasoning_id="leaf1", goal_class="combinatorics", action="tool",
                      oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                      signature_text="leaf1")
    s.write_macro(reasoning_id="macro::combinatorics", goal_class="combinatorics",
                  action="tool", signature=_feat(), b_i=5, c=1.0, time_cost=1.0,
                  use_count=5, composed_from=["leaf1"])
    assert _node_idea_type(g, "macro::combinatorics") == "procedural"
    assert _node_idea_type(g, "leaf1") == ""        # a tool_use episode is not an Idea


def test_d3_schema_migration_adds_column_to_existing_table():
    # an existing graph whose Reasoning table predates idea_type → bootstrap ALTERs it.
    import kuzu

    from titan_hcl.synthesis.kuzu_spine_schema import _column_exists
    with tempfile.TemporaryDirectory() as tmp:
        db = kuzu.Database(os.path.join(tmp, "old.kuzu"))
        conn = kuzu.Connection(db)
        # the pre-D.3 Reasoning table (no idea_type column)
        conn.execute(
            "CREATE NODE TABLE Reasoning(reasoning_id STRING, kind STRING, "
            "goal_class STRING, action STRING, oracle_id STRING, verdict STRING, "
            "anchor_tx STRING, created_at DOUBLE, PRIMARY KEY(reasoning_id))")
        assert _column_exists(conn, "Reasoning", "idea_type") is False

        class _Shim:           # bootstrap_spine_schema only needs ._conn
            _conn = conn
        out = bootstrap_spine_schema(_Shim())        # migrates the existing table
        assert "Reasoning.idea_type" in out.get("migrated_columns", [])
        assert _column_exists(conn, "Reasoning", "idea_type") is True


# ── D.1+D.3 — provenance edges exist down to the evidence ───────────────
def test_d1_composed_from_edges_deref_to_leaves(store_with_graph):
    s, g = store_with_graph
    for lid in ("e1", "e2", "e3"):
        s.record_tool_use(reasoning_id=lid, goal_class="combinatorics", action="tool",
                          oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                          signature_text=lid)
    s.write_macro(reasoning_id="M", goal_class="combinatorics", action="tool",
                  signature=_feat(), b_i=3, c=1.0, time_cost=1.0, use_count=3,
                  composed_from=["e1", "e2", "e3"])
    assert _composed_leaves(g, "M") == ["e1", "e2", "e3"]


# ── D.4b — macro-of-macros: a composite composed from CHILD composites ──
def test_d4b_macro_of_macros_links_child_composites(store_with_graph):
    s, g = store_with_graph
    # two child composites
    for cid in ("child_a", "child_b"):
        s.record_tool_use(reasoning_id=f"{cid}_leaf", goal_class=cid, action="tool",
                          oracle_id="o", verdict="true", reward=1.0, features=_feat(),
                          signature_text=f"{cid}_leaf")
        s.write_macro(reasoning_id=cid, goal_class=cid, action="tool",
                      signature=_feat(), b_i=2, c=1.0, time_cost=1.0, use_count=2,
                      composed_from=[f"{cid}_leaf"])
    # the parent composite composes from the two child COMPOSITES (not leaves)
    s.write_macro(reasoning_id="parent", goal_class="multi_step", action="skill_delegate",
                  signature=_feat(), b_i=2, c=1.0, time_cost=1.0, use_count=2,
                  composed_from=["child_a", "child_b"])
    assert _composed_leaves(g, "parent") == ["child_a", "child_b"]
    assert _node_idea_type(g, "parent") == "procedural"
