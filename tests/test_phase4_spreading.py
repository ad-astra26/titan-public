"""Phase 4 — spreading-activation `w_s` enablement tests (§P4.F).

Covers:
- buffer_stub source: topic_tags + extras + CGN history with proper
  precedence, dedup, and cap
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

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.buffer_stub import BufferStub, _strip_tag_prefix
from titan_hcl.synthesis.composite_score import (
    Candidate,
    _candidate_id_to_concept_id,
    composite_score,
    make_kuzu_spreading_lookup,
)
from titan_hcl.synthesis.concept_store import ConceptStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


# ── BufferStub ──────────────────────────────────────────────────────


def test_strip_tag_prefix_handles_known_prefixes():
    assert _strip_tag_prefix("topic:linux_terminal") == "linux_terminal"
    assert _strip_tag_prefix("concept:metaplex_nft_minting") == "metaplex_nft_minting"
    assert _strip_tag_prefix("concept:metaplex_nft_minting:v5") == "metaplex_nft_minting"
    assert _strip_tag_prefix("bare_concept") == "bare_concept"


def test_buffer_stub_topic_tags_take_precedence():
    """Topic tags from the in-flight chat TX come first (most recently
    relevant). Order is preserved."""
    s = BufferStub(cgn_handle=None, max_entities=20)
    entities = s.current_entities(
        topic_tags=["topic:linux_terminal", "topic:ssh"],
        extra_concepts=["solana_rpc"],
    )
    assert entities[:3] == ["linux_terminal", "ssh", "solana_rpc"]


def test_buffer_stub_dedup_preserves_first_occurrence():
    s = BufferStub(cgn_handle=None, max_entities=20)
    entities = s.current_entities(
        topic_tags=["topic:a", "topic:b"],
        extra_concepts=["a", "c"],   # "a" already present
    )
    assert entities == ["a", "b", "c"]


def test_buffer_stub_cap_respected():
    s = BufferStub(cgn_handle=None, max_entities=3)
    entities = s.current_entities(
        topic_tags=[f"topic:c{i}" for i in range(10)],
    )
    assert entities == ["c0", "c1", "c2"]


def test_buffer_stub_cgn_history_fallback():
    """A CGN-shaped handle's _concept_journeys is sampled when supplied."""

    class FakeCGN:
        _concept_journeys = {
            "old_concept":   {"last_seen": 10.0},
            "fresh_concept": {"last_seen": 100.0},
            "mid_concept":   {"last_seen": 50.0},
        }

    s = BufferStub(cgn_handle=FakeCGN(), history_window_turns=1, max_entities=20)
    entities = s.current_entities()
    # Sorted by last_seen DESC.
    assert entities[0] == "fresh_concept"
    assert entities[1] == "mid_concept"
    assert entities[2] == "old_concept"


def test_buffer_stub_cgn_handle_with_bad_shape_silently_degrades():
    """If CGN's internal shape isn't what we expect, the buffer-stub must
    not raise — it just returns whatever it can without the CGN contribution."""

    class BrokenCGN:
        _concept_journeys = ["not", "a", "dict"]  # wrong shape

    s = BufferStub(cgn_handle=BrokenCGN(), max_entities=10)
    entities = s.current_entities(topic_tags=["topic:safe"])
    assert entities == ["safe"]


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


def test_buffer_stub_to_spreading_lookup_pipeline(graph_with_spine):
    """Glue test: BufferStub → make_kuzu_spreading_lookup → composite_score.
    The full P4.F pipeline that EngineRecall will run."""
    bs = BufferStub(cgn_handle=None, max_entities=20)
    buf = bs.current_entities(topic_tags=["topic:unix_shell"])
    assert buf == ["unix_shell"]

    spreading = make_kuzu_spreading_lookup(graph_with_spine, buf, S=2.0)
    scores = spreading(["concept:linux_terminal", "concept:ssh"])
    expected = 2.0 - math.log(2)
    assert scores["concept:linux_terminal"] == pytest.approx(expected)
    assert scores["concept:ssh"] == pytest.approx(expected)
