"""Phase 4 — dream-boundary consolidation pass tests (§P4.G).

Covers `titan_hcl/synthesis/consolidation.py` against PLAN §P4.G + arch §10.4:

- pass mines + clusters + applies proposals + anchors summary TX
- empty mine result → pass skipped + summary TX still anchored
- clustering: cosine + jaccard gates; below-threshold doesn't cluster
- min_cluster_size enforced (size 2 cluster rejected)
- max_concepts_per_pass cap respected
- llm_calls_max cap respected
- new_concept proposal: registers CGN, creates concept, anchors TX,
  maintains composition edges, recomputes groundedness
- version_bump proposal: bumps existing concept; parent v(n) untouched
- reject proposal: cluster counted as rejected, no spine writes
- writer-side failures don't crash the pass (logged + counted)
- C.5 invariant: ConsolidationPass NEVER deletes a Concept row
- pass_id is unique across passes (timestamp-based)
- summary TX content includes all required fields
"""
from __future__ import annotations

import os
import queue
import tempfile

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.consolidation import (
    Cluster,
    ConsolidationPass,
    ConsolidationResult,
    LLMProposal,
    TxCandidate,
    _default_cosine,
    _jaccard,
)
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def env():
    """Real graph + engram_store + cgn_bridge + writer; collaborators
    are stubbed below per-test."""
    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "p4g.kuzu"))
        q = queue.Queue()
        w = OuterMemoryWriter(send_queue=q, src="consolidation_test")
        bridge = CGNRegistrationBridge(
            os.path.join(tmp, "spine_concepts.json"),
        )
        store = EngramStore(g, w, clock=lambda: 1000.0)
        try:
            yield {
                "graph": g, "queue": q, "writer": w,
                "bridge": bridge, "store": store, "tmp": tmp,
            }
        finally:
            g.close()


# Convenience: make TXs with shared tags + similar embeddings.
def _mk_tx(tx_hash, tags, embedding=None, fork="declarative"):
    return TxCandidate(
        tx_hash=tx_hash, fork=fork, tags=tuple(tags),
        embedding=tuple(embedding) if embedding else None,
    )


# ── Cosine + jaccard primitives ─────────────────────────────────────


def test_jaccard_basic():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_default_cosine_orthogonal_zero():
    assert _default_cosine((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_default_cosine_parallel_one():
    assert _default_cosine((1.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)


def test_default_cosine_zero_norm_returns_zero():
    assert _default_cosine((0.0, 0.0), (1.0, 1.0)) == 0.0


# ── Clustering ──────────────────────────────────────────────────────


def _make_pass(env, txs, proposals=None, **kw):
    """Build a ConsolidationPass with stubbed mine + propose."""
    proposals = proposals or []
    propose_iter = iter(proposals)

    def mine_fn(**_kw):
        return txs

    def propose_fn(_cluster):
        try:
            return next(propose_iter)
        except StopIteration:
            return LLMProposal(action="reject", reason="exhausted")

    return ConsolidationPass(
        engram_store=env["store"],
        cgn_bridge=env["bridge"],
        outer_memory_writer=env["writer"],
        mine_recent_txs_fn=mine_fn,
        llm_propose_fn=propose_fn,
        **kw,
    )


def test_run_with_empty_mine_skips_pass_anchors_summary(env):
    cp = _make_pass(env, txs=[])
    result = cp.run()
    assert result.skipped is True
    assert "insufficient_txs" in result.skip_reason
    # Summary TX still anchored (auditable per §10.4).
    msgs = []
    while not env["queue"].empty():
        msgs.append(env["queue"].get_nowait())
    assert len(msgs) == 1
    assert msgs[0]["payload"]["thought_type"] == "consolidation_pass"
    assert msgs[0]["payload"]["content"]["skipped"] is True


def test_run_with_below_min_cluster_size_yields_no_proposals(env):
    """3 TXs but all tag-disjoint → no cluster reaches min_cluster_size."""
    txs = [
        _mk_tx("tx1", tags=["unrelated_a"]),
        _mk_tx("tx2", tags=["unrelated_b"]),
        _mk_tx("tx3", tags=["unrelated_c"]),
    ]
    cp = _make_pass(env, txs=txs, min_cluster_size=3)
    result = cp.run()
    assert result.clusters_considered == 0
    assert result.llm_calls == 0
    assert result.concepts_created == []


def test_clustering_with_shared_tags_and_similar_embeddings(env):
    """3 TXs sharing tags + similar embeddings → 1 cluster of 3."""
    emb = [1.0, 0.0, 0.0]
    txs = [
        _mk_tx("t1", tags=["topic:linux"], embedding=emb),
        _mk_tx("t2", tags=["topic:linux"], embedding=[0.95, 0.05, 0.0]),
        _mk_tx("t3", tags=["topic:linux"], embedding=[0.97, 0.0, 0.05]),
    ]
    cp = _make_pass(env, txs=txs, min_cluster_size=3)
    clusters = cp._cluster_txs(txs)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 3


def test_clustering_below_cosine_threshold_separates(env):
    """Even with shared tags, orthogonal embeddings split clusters."""
    txs = [
        _mk_tx("t1", tags=["topic:a"], embedding=[1.0, 0.0, 0.0]),
        _mk_tx("t2", tags=["topic:a"], embedding=[0.0, 1.0, 0.0]),
        _mk_tx("t3", tags=["topic:a"], embedding=[0.0, 0.0, 1.0]),
    ]
    cp = _make_pass(env, txs=txs, min_cluster_size=2,
                    cluster_cosine_threshold=0.85)
    clusters = cp._cluster_txs(txs)
    # All three start their own clusters; none reaches min_cluster_size=2.
    assert clusters == []


# ── Proposal application ────────────────────────────────────────────


def test_new_concept_proposal_creates_spine_concept(env):
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:linux"], embedding=emb) for i in range(3)]
    proposals = [LLMProposal(
        action="new_concept", concept_id="linux_basics",
        proposed_name="Linux basics", memory_type="declarative",
        reason="emergent from cluster",
    )]
    cp = _make_pass(env, txs=txs, proposals=proposals)
    result = cp.run()
    assert result.concepts_created == [("linux_basics", 1)]
    # Spine row exists with anchor_tx.
    row = env["graph"].spine_get_concept_version("linux_basics", 1)
    assert row is not None
    # CGN bridge registered the concept.
    assert env["bridge"].is_registered("linux_basics")
    # Groundedness recomputed from cluster stats > 0.
    assert row["groundedness"] > 0.0


def test_version_bump_proposal_bumps_existing(env):
    # Pre-seed v=1.
    env["bridge"].register_spine_concept("solana_rpc", "Solana RPC")
    env["store"].create_concept(
        "solana_rpc", "Solana RPC", memory_type="procedural",
    )
    v1_anchor_before = env["graph"].spine_get_concept_version(
        "solana_rpc", 1)["anchor_tx"]

    emb = [0.0, 1.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:solana"], embedding=emb) for i in range(3)]
    proposals = [LLMProposal(
        action="version_bump", concept_id="solana_rpc",
        reason="enrichment",
    )]
    cp = _make_pass(env, txs=txs, proposals=proposals)
    result = cp.run()

    assert result.concepts_bumped == [("solana_rpc", 2)]
    # INV-3: v=1 untouched (anchor_tx + all identity fields preserved).
    v1_after = env["graph"].spine_get_concept_version("solana_rpc", 1)
    assert v1_after["anchor_tx"] == v1_anchor_before
    # v=2 row exists.
    assert env["graph"].spine_get_concept_version("solana_rpc", 2) is not None


def test_version_bump_for_missing_concept_treated_as_rejected(env):
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:a"], embedding=emb) for i in range(3)]
    proposals = [LLMProposal(
        action="version_bump", concept_id="never_existed",
    )]
    cp = _make_pass(env, txs=txs, proposals=proposals)
    result = cp.run()
    # ParentVersionMissing caught + counted as rejected.
    assert result.concepts_bumped == []
    assert result.rejected_clusters >= 1


def test_reject_proposal_counted_as_rejected(env):
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:a"], embedding=emb) for i in range(3)]
    cp = _make_pass(env, txs=txs, proposals=[LLMProposal(action="reject")])
    result = cp.run()
    assert result.rejected_clusters == 1
    assert result.concepts_created == []
    assert result.concepts_bumped == []


def test_invalid_action_treated_as_rejected(env):
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:a"], embedding=emb) for i in range(3)]
    proposals = [LLMProposal(action="something_else")]  # type: ignore
    cp = _make_pass(env, txs=txs, proposals=proposals)
    result = cp.run()
    assert result.concepts_created == []
    assert result.rejected_clusters >= 1


# ── Caps ────────────────────────────────────────────────────────────


def test_max_concepts_per_pass_cap_respected(env):
    # 3 separate viable clusters (each 3 TXs).
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    emb_c = [0.0, 0.0, 1.0]
    txs = (
        [_mk_tx(f"a{i}", tags=["topic:a"], embedding=emb_a) for i in range(3)]
        + [_mk_tx(f"b{i}", tags=["topic:b"], embedding=emb_b) for i in range(3)]
        + [_mk_tx(f"c{i}", tags=["topic:c"], embedding=emb_c) for i in range(3)]
    )
    proposals = [
        LLMProposal(action="new_concept", concept_id="ca",
                    proposed_name="CA", memory_type="declarative"),
        LLMProposal(action="new_concept", concept_id="cb",
                    proposed_name="CB", memory_type="declarative"),
        LLMProposal(action="new_concept", concept_id="cc",
                    proposed_name="CC", memory_type="declarative"),
    ]
    cp = _make_pass(env, txs=txs, proposals=proposals,
                    max_concepts_per_pass=2)
    result = cp.run()
    assert len(result.concepts_created) == 2


def test_llm_calls_max_cap_respected(env):
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    emb_c = [0.0, 0.0, 1.0]
    txs = (
        [_mk_tx(f"a{i}", tags=["topic:a"], embedding=emb_a) for i in range(3)]
        + [_mk_tx(f"b{i}", tags=["topic:b"], embedding=emb_b) for i in range(3)]
        + [_mk_tx(f"c{i}", tags=["topic:c"], embedding=emb_c) for i in range(3)]
    )
    proposals = [LLMProposal(action="reject")] * 10  # all reject
    cp = _make_pass(env, txs=txs, proposals=proposals, llm_calls_max=2)
    result = cp.run()
    assert result.llm_calls <= 2


# ── Robustness ──────────────────────────────────────────────────────


def test_llm_propose_exception_counted_as_rejected(env):
    """LLM raising for a cluster doesn't crash the pass; cluster counted
    as rejected, pass continues."""
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:a"], embedding=emb) for i in range(3)]

    def mine_fn(**_kw):
        return txs

    def propose_fn(_c):
        raise RuntimeError("ollama down")

    cp = ConsolidationPass(
        engram_store=env["store"], cgn_bridge=env["bridge"],
        outer_memory_writer=env["writer"],
        mine_recent_txs_fn=mine_fn, llm_propose_fn=propose_fn,
    )
    result = cp.run()
    assert result.rejected_clusters >= 1
    assert result.concepts_created == []
    # Pass summary still anchored.
    assert result.pass_tx_hash is not None


def test_mine_exception_aborts_pass_anchors_summary(env):
    def bad_mine(**_kw):
        raise RuntimeError("chain unavailable")

    cp = ConsolidationPass(
        engram_store=env["store"], cgn_bridge=env["bridge"],
        outer_memory_writer=env["writer"],
        mine_recent_txs_fn=bad_mine,
        llm_propose_fn=lambda _c: LLMProposal(action="reject"),
    )
    result = cp.run()
    assert result.skipped is True
    assert "mine_failed" in result.skip_reason


def test_c5_invariant_no_kuzu_deletion(env):
    """C.5: consolidation must NEVER delete a Concept row. We seed v=1 +
    let consolidation propose a version_bump → both versions exist
    afterwards."""
    env["bridge"].register_spine_concept("base", "Base")
    env["store"].create_concept("base", "Base", memory_type="declarative")
    assert env["graph"].spine_count_concepts() == 1

    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:base"], embedding=emb) for i in range(3)]
    cp = _make_pass(env, txs=txs, proposals=[
        LLMProposal(action="version_bump", concept_id="base"),
    ])
    cp.run()

    assert env["graph"].spine_count_concepts() == 2
    # Both v=1 and v=2 exist.
    assert env["graph"].spine_get_concept_version("base", 1) is not None
    assert env["graph"].spine_get_concept_version("base", 2) is not None


def test_pass_id_unique_across_passes(env):
    cp = _make_pass(env, txs=[])
    ids = set()
    for _ in range(3):
        # Bump the clock between passes so pass_ids differ.
        old_clock = cp._clock
        offset = [0]
        def _c(o=offset, base=old_clock):
            t = base() + o[0]
            o[0] += 0.01
            return t
        cp._clock = _c
        result = cp.run()
        ids.add(result.pass_id)
    assert len(ids) == 3


def test_summary_tx_content_complete(env):
    """The consolidation_pass TX content carries every observability field."""
    emb = [1.0, 0.0]
    txs = [_mk_tx(f"t{i}", tags=["topic:a"], embedding=emb) for i in range(3)]
    cp = _make_pass(env, txs=txs, proposals=[
        LLMProposal(action="new_concept", concept_id="c1",
                    proposed_name="C1", memory_type="declarative"),
    ])
    cp.run()
    msgs = []
    while not env["queue"].empty():
        msgs.append(env["queue"].get_nowait())
    # Last message is the summary; earlier ones are the per-concept anchors.
    summary = next(
        m for m in msgs if m["payload"]["thought_type"] == "consolidation_pass"
    )
    c = summary["payload"]["content"]
    for k in ("pass_id", "started_at", "finished_at", "duration_ms",
              "txs_mined", "clusters_considered", "concepts_created",
              "concepts_bumped", "rejected_clusters", "llm_calls",
              "skipped", "skip_reason"):
        assert k in c, f"missing field {k!r} in summary content"
    assert c["concepts_created"] == [{"concept_id": "c1", "version": 1}]
    # Summary TX tags carry §P4.G discriminators.
    assert summary["payload"]["tags"] == [
        "consolidation_pass", "synthesis_worker", "dream_boundary",
    ]
