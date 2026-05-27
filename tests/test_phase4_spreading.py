"""Phase 4 — spreading-activation `w_s` enablement tests (§P4.F).

After Phase 7 (D-SPEC-PHASE7) the buffer-entity source is the real
`ActrBufferStore.buffer_entities(chat_id)` per INV-Syn-18 — BufferStub
was deleted per `feedback_no_shim_old_path_must_be_deleted.md`. The
spreading + composite_score machinery itself is unchanged from P4 so
this file retains its `make_kuzu_spreading_lookup` coverage; the
BufferStub-specific tests are replaced with equivalent
ActrBufferStore.buffer_entities tests below.

Covers:
- ActrBufferStore.buffer_entities precedence + dedup + cap (P7 successor
  to BufferStub.current_entities)
- _candidate_id_to_concept_id parses concept-keyed item_ids correctly
- make_kuzu_spreading_lookup pre-computes the reachable map
- spreading contribution formula `S - ln(fan_j)` matches arch §5.3
- candidates that aren't concept-keyed get spreading 0 in P4
- dense buffer nodes (ln(fan_j) >= S) contribute 0 (saturation)
- composite_score with the real kuzu spreading_lookup outranks the same
  candidates with spreading_lookup=None (regression vs P1)
- Kuzu reader exceptions don't break the lookup (soft-fail per INV-Syn-4)
"""
from __future__ import annotations

import math
import os
import tempfile

import duckdb
import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.buffer_store import ActrBufferStore
from titan_hcl.synthesis.composite_score import (
    Candidate,
    _candidate_id_to_concept_id,
    composite_score,
    make_kuzu_spreading_lookup,
)
from titan_hcl.synthesis.concept_store import ConceptStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


# ── ActrBufferStore.buffer_entities (P7 successor to BufferStub) ──


@pytest.fixture()
def actr_store(tmp_path):
    """In-memory ActrBufferStore for buffer-entities tests."""
    conn = duckdb.connect(":memory:")
    return ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "buffers_snapshot.json"),
    )


def test_buffer_entities_goal_takes_precedence(actr_store):
    """INV-Syn-18: goal → perception → retrieval → imaginal precedence.

    The ordering matches ACT-R's intuition: explicit goal beats incidental
    observation; retrieved memories beat freeform scratchpad."""
    chat_id = "alice:s1"
    actr_store.persist(
        chat_id=chat_id, buffer_name="imaginal",
        content="thinking", concept_ids=["draft_idea"],
    )
    actr_store.persist(
        chat_id=chat_id, buffer_name="goal",
        content="debug", concept_ids=["rust_panic", "debugging"],
    )
    actr_store.persist(
        chat_id=chat_id, buffer_name="perception",
        content="msg", concept_ids=["user_input"],
    )
    actr_store.persist(
        chat_id=chat_id, buffer_name="retrieval",
        content="recall", concept_ids=["past_fix"],
    )
    entities = actr_store.buffer_entities(chat_id)
    # First two slots come from `goal`, in their persisted order.
    assert entities[:2] == ["rust_panic", "debugging"]
    # Then perception, then retrieval, then imaginal.
    assert "user_input" in entities
    assert entities.index("user_input") < entities.index("past_fix")
    assert entities.index("past_fix") < entities.index("draft_idea")


def test_buffer_entities_dedup_preserves_first_occurrence(actr_store):
    chat_id = "alice:s2"
    actr_store.persist(
        chat_id=chat_id, buffer_name="goal",
        content="g", concept_ids=["a", "b"],
    )
    actr_store.persist(
        chat_id=chat_id, buffer_name="retrieval",
        content="r", concept_ids=["a", "c"],   # "a" already in goal
    )
    entities = actr_store.buffer_entities(chat_id)
    assert entities == ["a", "b", "c"]


def test_buffer_entities_cap_respected(actr_store):
    chat_id = "alice:s3"
    actr_store.persist(
        chat_id=chat_id, buffer_name="goal",
        content="g", concept_ids=[f"c{i}" for i in range(10)],
    )
    entities = actr_store.buffer_entities(chat_id, cap=3)
    assert entities == ["c0", "c1", "c2"]


def test_buffer_entities_empty_chat_returns_empty(actr_store):
    assert actr_store.buffer_entities("ghost:never_existed") == []


def test_buffer_entities_skips_malformed_concept_ids(actr_store):
    """If concept_ids in storage are corrupted to non-strings, skip them
    instead of crashing the spreading-activation lookup."""
    chat_id = "alice:s4"
    actr_store.persist(
        chat_id=chat_id, buffer_name="goal",
        content="g", concept_ids=["valid_id"],
    )
    # Manually inject a malformed row (simulates upstream corruption).
    import json
    actr_store._db.execute(
        "INSERT OR REPLACE INTO actr_buffers "
        "(chat_id, buffer_name, content, concept_ids, embedding_hash, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [chat_id, "imaginal", "x", "not-a-json-array", "h", 100.0],
    )
    entities = actr_store.buffer_entities(chat_id)
    # The valid goal concept survives; the corrupted imaginal row degrades silently.
    assert entities == ["valid_id"]


# ── _candidate_id_to_concept_id ─────────────────────────────────────


def test_candidate_id_parsing():
    assert _candidate_id_to_concept_id("concept:x") == "x"
    assert _candidate_id_to_concept_id("concept:x:v3") == "x"
    assert _candidate_id_to_concept_id("mem:1234") is None
    assert _candidate_id_to_concept_id("tc:abcd") is None
    assert _candidate_id_to_concept_id("") is None
    assert _candidate_id_to_concept_id("concept:") is None  # empty after prefix


# ── make_kuzu_spreading_lookup ──────────────────────────────────────


@pytest.fixture()
def graph_with_spine():
    """A TitanKnowledgeGraph with a small spine: ssh + linux_terminal both
    COMPOSED_FROM 'unix_shell'; cosmetic_business_website COMPOSED_FROM
    'linux_terminal' only. So 'linux_terminal' has fan_j = 2 (ssh sibling
    via unix_shell? no — we test more concretely below)."""
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "test_spread.kuzu"))
        import queue
        q = queue.Queue()
        w = OuterMemoryWriter(send_queue=q, src="test_spreading")
        store = ConceptStore(g, w, clock=lambda: 1000.0)
        store.create_concept("unix_shell", "Unix shell", memory_type="declarative")
        store.create_concept(
            "linux_terminal", "Linux terminal", memory_type="declarative",
            composed_from=[("unix_shell", 1)],
        )
        store.create_concept(
            "ssh", "SSH", memory_type="declarative",
            composed_from=[("unix_shell", 1)],
        )
        store.create_concept(
            "cosmetic_business_website", "Cosmetic business website",
            memory_type="declarative",
            composed_from=[("linux_terminal", 1)],
        )
        try:
            yield g
        finally:
            g.close()


def test_spreading_formula_matches_arch_5_3(graph_with_spine):
    """`spreading(i, buf) = Σ (S - ln(fan_j)) · 1[edge(j,i)]`. Verify the
    exact value for a known graph state."""
    # buffer_entities = ["unix_shell"]. fan_j for unix_shell:
    # COMPOSED_FROM neighbors: 0 (nothing references unix_shell as a base)
    # COMPOSED_INTO neighbors: 2 (linux_terminal v1 + ssh v1 both compose INTO from it)
    # → fan_j = 2.
    #
    # Spreading contribution for any candidate that's a neighbor of unix_shell:
    # S - ln(2) = 2.0 - 0.693... = 1.306...
    lookup = make_kuzu_spreading_lookup(
        graph_with_spine, buffer_entities=["unix_shell"], S=2.0,
    )
    scores = lookup(["concept:linux_terminal", "concept:ssh"])
    expected = 2.0 - math.log(2)
    assert scores["concept:linux_terminal"] == pytest.approx(expected)
    assert scores["concept:ssh"] == pytest.approx(expected)


def test_non_concept_candidates_get_zero_spreading(graph_with_spine):
    """Non-concept-keyed items (mem:/tc:/skill:) contribute 0 in P4."""
    lookup = make_kuzu_spreading_lookup(
        graph_with_spine, buffer_entities=["unix_shell"],
    )
    scores = lookup(["mem:9999", "tc:abcdef", "skill:research_topic"])
    # No entries → 0 contribution downstream (absent from dict).
    assert scores == {}


def test_unreachable_candidate_concept_gets_no_entry(graph_with_spine):
    """A concept_id with no spine edge from any buffer_entity is absent
    from the returned dict (composite_score defaults to 0 for absent)."""
    lookup = make_kuzu_spreading_lookup(
        graph_with_spine, buffer_entities=["unix_shell"],
    )
    # 'cosmetic_business_website' is reachable from 'linux_terminal' (one
    # hop further), NOT directly from 'unix_shell'. P4 spreading is 1-hop
    # only — multi-hop walks would change the formula semantics.
    scores = lookup(["concept:cosmetic_business_website"])
    assert "concept:cosmetic_business_website" not in scores


def test_empty_buffer_entities_returns_zero_for_all(graph_with_spine):
    """No buffer entities → no spreading contribution possible."""
    lookup = make_kuzu_spreading_lookup(graph_with_spine, buffer_entities=[])
    scores = lookup(["concept:linux_terminal", "concept:ssh"])
    assert scores == {}


def test_dense_buffer_node_saturates(graph_with_spine):
    """When ln(fan_j) >= S, the contribution is 0 (saturation). With S=0.5
    and fan_j=2 (unix_shell has 2 neighbors), contribution = 0.5 - ln(2)
    ≈ -0.19 → clamped to 0. Verifies the saturation guard."""
    lookup = make_kuzu_spreading_lookup(
        graph_with_spine, buffer_entities=["unix_shell"], S=0.5,
    )
    scores = lookup(["concept:linux_terminal"])
    assert scores == {}


def test_kuzu_reader_exception_soft_fails():
    """spine_concept_neighbors raising must not crash the lookup."""

    class FlakyReader:
        def spine_concept_neighbors(self, *_a, **_kw):
            raise RuntimeError("kuzu down")

    lookup = make_kuzu_spreading_lookup(
        FlakyReader(), buffer_entities=["anything"],
    )
    assert lookup(["concept:x"]) == {}


# ── composite_score integration ────────────────────────────────────


def test_composite_score_with_kuzu_spreading_outranks_p1_baseline(graph_with_spine):
    """End-to-end: same candidate set, w_s>0 with kuzu spreading promotes
    concept-keyed candidates that share neighbors with buffer entities.
    This is the B.1 acceptance gate becoming measurable."""
    cands = [
        Candidate(item_id="concept:linux_terminal", cosine=0.5, importance=0.5),
        Candidate(item_id="concept:ssh",            cosine=0.5, importance=0.5),
        Candidate(item_id="mem:99999",              cosine=0.5, importance=0.5),
    ]
    activations = {}  # all cold-start
    activation_lookup = lambda ids: activations

    spreading = make_kuzu_spreading_lookup(
        graph_with_spine, buffer_entities=["unix_shell"], S=2.0,
    )

    # P1 baseline: spreading_lookup=None → all three candidates have same score.
    baseline = composite_score(
        cands, activation_lookup, spreading_lookup=None,
        w_b=1.0, w_s=1.0, w_r=1.0, w_p=1.0,
    )
    base_scores = {sc.candidate.item_id: sc.score for sc in baseline}
    assert base_scores["concept:linux_terminal"] == pytest.approx(
        base_scores["mem:99999"],
    )

    # P4: spreading_lookup wired → concept candidates outrank mem:99999.
    p4 = composite_score(
        cands, activation_lookup, spreading_lookup=spreading,
        w_b=1.0, w_s=1.0, w_r=1.0, w_p=1.0,
    )
    p4_scores = {sc.candidate.item_id: sc.score for sc in p4}
    assert p4_scores["concept:linux_terminal"] > p4_scores["mem:99999"]
    assert p4_scores["concept:ssh"] > p4_scores["mem:99999"]
    # mem candidate's score is unchanged from baseline (no spreading
    # contribution).
    assert p4_scores["mem:99999"] == pytest.approx(base_scores["mem:99999"])


def test_actr_buffer_store_to_spreading_lookup_pipeline(graph_with_spine, tmp_path):
    """P7 glue test: ActrBufferStore.buffer_entities → make_kuzu_spreading_lookup
    → composite_score. Replaces the pre-P7 BufferStub pipeline test."""
    conn = duckdb.connect(":memory:")
    store = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "snap.json"),
    )
    chat_id = "alice:pipeline"
    store.persist(
        chat_id=chat_id, buffer_name="goal",
        content="unix question", concept_ids=["unix_shell"],
    )
    buf = store.buffer_entities(chat_id)
    assert buf == ["unix_shell"]

    spreading = make_kuzu_spreading_lookup(graph_with_spine, buf, S=2.0)
    scores = spreading(["concept:linux_terminal", "concept:ssh"])
    expected = 2.0 - math.log(2)
    assert scores["concept:linux_terminal"] == pytest.approx(expected)
    assert scores["concept:ssh"] == pytest.approx(expected)
