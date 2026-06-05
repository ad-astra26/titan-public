"""Phase 4 — EngramStore tests (P4.B).

Covers `titan_hcl/synthesis/engram_store.py` against PLAN §P4.B + the
INV-3 / INV-4 / INV-10 invariants:

- create_concept anchors a TX + inserts the Kuzu row + maintains composition edges
- bump_version inserts v(n+1) without touching v(n)  (INV-3)
- bump_version raises ParentVersionMissing for nonexistent concepts  (INV-10)
- writer failure aborts the Kuzu insert (rollback semantics)         (INV-4)
- recompute_groundedness updates only the derived column
- recompute_groundedness_batch handles partial failures cleanly
- composition edges land in BOTH directions on bump_version
- groundedness formula behaves per §P4.E (log-norm, weighted sum, clamped)

The tests use a FakeWriter that records calls + can be configured to raise;
the real OuterMemoryWriter integration is verified in test_phase4_concept_version_tx.py
(P4.D).
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import (
    EngramStore,
    Engram,
    ParentVersionMissing,
    WriterFailure,
    _GroundednessParams,
    compute_groundedness,
)


# ── Fake writer ─────────────────────────────────────────────────────


@dataclass
class _Call:
    concept_id: str
    version: int
    name: str
    memory_type: str
    parent_version_tx: Optional[str]
    composed_from: list
    derivation_evidence: list
    groundedness: float
    derivation_merkle_root: Optional[str]


class FakeWriter:
    """Records every write_concept_version call. `raise_on_call` can be set
    to an exception instance to simulate writer failure."""

    def __init__(self, raise_on_call: Optional[Exception] = None):
        self.calls: list[_Call] = []
        self._raise = raise_on_call
        self._counter = 0

    def write_concept_version(self, **kwargs) -> str:
        self.calls.append(_Call(
            concept_id=kwargs["concept_id"],
            version=kwargs["version"],
            name=kwargs["name"],
            memory_type=kwargs["memory_type"],
            parent_version_tx=kwargs["parent_version_tx"],
            composed_from=list(kwargs["composed_from"]),
            derivation_evidence=list(kwargs["derivation_evidence"]),
            groundedness=kwargs["groundedness"],
            derivation_merkle_root=kwargs.get("derivation_merkle_root"),
        ))
        if self._raise is not None:
            raise self._raise
        self._counter += 1
        return f"tx_{kwargs['concept_id']}_v{kwargs['version']}_{self._counter}"


@pytest.fixture()
def graph():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_cs.kuzu")
        g = TitanKnowledgeGraph(path)
        try:
            yield g
        finally:
            g.close()


@pytest.fixture()
def store(graph):
    return EngramStore(graph, FakeWriter(), clock=lambda: 1000.0)


# ── Groundedness formula ────────────────────────────────────────────


def test_groundedness_zero_for_all_zero_inputs():
    assert compute_groundedness(
        episodic_encounters=0, distinct_contexts=0, procedural_links=0,
    ) == 0.0


def test_groundedness_clamped_to_unit_interval():
    # Hammer all terms way above the log-norm saturation point.
    g = compute_groundedness(
        episodic_encounters=10_000, distinct_contexts=10_000,
        procedural_links=10_000, felt_coverage=1.0,
        params=_GroundednessParams(w_e=1.0, w_c=1.0, w_p=1.0, w_f=1.0),
    )
    assert 0.0 <= g <= 1.0
    assert g == pytest.approx(1.0)  # all terms saturate, weights = 1 each


def test_groundedness_default_weights_sum_to_one_when_felt_disabled():
    # P4 defaults: w_e=0.3, w_c=0.3, w_p=0.4, w_f=0.0 → sum to 1.0 without
    # the felt strand. Saturating each count gives g ~ 1.0.
    g = compute_groundedness(
        episodic_encounters=50, distinct_contexts=50, procedural_links=50,
    )
    assert g == pytest.approx(1.0, abs=1e-6)


def test_groundedness_procedural_links_dominate_in_p4():
    """w_p > w_e ≈ w_c by default since P4 has no felt strand yet."""
    only_proc = compute_groundedness(
        episodic_encounters=0, distinct_contexts=0, procedural_links=50,
    )
    only_epi = compute_groundedness(
        episodic_encounters=50, distinct_contexts=0, procedural_links=0,
    )
    assert only_proc > only_epi


def test_groundedness_felt_coverage_inert_with_default_w_f():
    """felt_coverage is in the formula but w_f=0.0 in P4, so it can't
    contribute. Confirms the bridge phase only has to flip a parameter."""
    base = compute_groundedness(
        episodic_encounters=10, distinct_contexts=5, procedural_links=3,
        felt_coverage=0.0,
    )
    saturated_felt = compute_groundedness(
        episodic_encounters=10, distinct_contexts=5, procedural_links=3,
        felt_coverage=1.0,
    )
    assert base == saturated_felt


# ── create_concept ──────────────────────────────────────────────────


def test_create_concept_anchors_tx_inserts_row(graph, store):
    cv = store.create_concept(
        concept_id="metaplex_nft_minting",
        name="Metaplex NFT minting",
        memory_type="procedural",
    )
    assert isinstance(cv, Engram)
    assert cv.version == 1
    assert cv.anchor_tx.startswith("tx_metaplex_nft_minting_v1_")

    # Row in Kuzu reflects the writer-returned tx hash.
    row = graph.spine_get_concept_version("metaplex_nft_minting", 1)
    assert row is not None
    assert row["anchor_tx"] == cv.anchor_tx
    assert row["memory_type"] == "procedural"


def test_create_concept_rejects_invalid_memory_type(store):
    with pytest.raises(ValueError):
        store.create_concept("x", "X", memory_type="philosophical")


def test_create_concept_writer_failure_aborts_kuzu_insert(graph):
    """INV-4: if writer raises, the Kuzu row must NOT exist."""
    failing_writer = FakeWriter(raise_on_call=RuntimeError("chain down"))
    s = EngramStore(graph, failing_writer, clock=lambda: 1000.0)
    with pytest.raises(WriterFailure):
        s.create_concept("orphan", "Orphan", memory_type="declarative")
    # The Kuzu row must NOT have been inserted.
    assert graph.spine_get_concept_version("orphan", 1) is None
    assert graph.spine_count_concepts() == 0
    # Writer was attempted exactly once.
    assert len(failing_writer.calls) == 1


def test_create_concept_with_composed_from_lays_edges(graph, store):
    # Lay base concepts first.
    store.create_concept(
        "domain_dns", "Domain + DNS", memory_type="declarative",
    )
    store.create_concept(
        "ssl_cert", "SSL certificate", memory_type="declarative",
    )
    # Now create the composition.
    store.create_concept(
        "cosmetic_business_website", "Cosmetic business website",
        memory_type="declarative",
        composed_from=[("domain_dns", 1), ("ssl_cert", 1)],
    )

    # COMPOSED_FROM edges: cosmetic_business_website v1 → domain_dns v1, ssl_cert v1
    neighbors = graph.spine_concept_neighbors(
        "cosmetic_business_website", version=1, limit=20,
    )
    assert ("domain_dns", 1) in neighbors
    assert ("ssl_cert", 1) in neighbors

    # COMPOSED_INTO reciprocal: domain_dns v1 → cosmetic_business_website v1
    parent_neighbors = graph.spine_concept_neighbors("domain_dns", version=1)
    assert ("cosmetic_business_website", 1) in parent_neighbors


# ── bump_version ────────────────────────────────────────────────────


def test_bump_version_inserts_new_row_without_mutating_parent(graph, store):
    """INV-3 hard test: v=1 row must be byte-identical after bump."""
    store.create_concept(
        "metaplex_nft_minting", "Metaplex NFT minting",
        memory_type="procedural",
    )
    v1_before = graph.spine_get_concept_version("metaplex_nft_minting", 1)
    assert v1_before is not None

    store.bump_version("metaplex_nft_minting")

    v1_after = graph.spine_get_concept_version("metaplex_nft_minting", 1)
    v2 = graph.spine_get_concept_version("metaplex_nft_minting", 2)
    assert v1_after is not None and v2 is not None
    # v=1 unchanged in every field that defines identity / lineage.
    assert v1_after["anchor_tx"] == v1_before["anchor_tx"]
    assert v1_after["created_at"] == v1_before["created_at"]
    assert v1_after["name"] == v1_before["name"]
    assert v1_after["memory_type"] == v1_before["memory_type"]
    # v=2 carries parent_version_tx referencing v=1's hash (via writer call).


def test_bump_version_raises_for_missing_parent(graph, store):
    """INV-10: bump_version on a nonexistent concept_id raises rather than
    quietly creating a v=1 ghost."""
    with pytest.raises(ParentVersionMissing):
        store.bump_version("ghost_concept")


def test_bump_version_passes_parent_tx_to_writer(graph):
    """The writer must receive the v(n-1) anchor_tx as parent_version_tx so
    the chain links the version-bump TX back to its predecessor."""
    writer = FakeWriter()
    s = EngramStore(graph, writer, clock=lambda: 1000.0)
    s.create_concept("solana_rpc", "Solana RPC", memory_type="procedural")
    s.bump_version("solana_rpc")
    assert len(writer.calls) == 2
    v1_call, v2_call = writer.calls
    assert v1_call.parent_version_tx is None
    # v2's parent_version_tx is the tx hash returned by the writer for v1.
    expected_parent = f"tx_solana_rpc_v1_1"
    assert v2_call.parent_version_tx == expected_parent


def test_bump_version_maintains_bidirectional_composition_edges(graph, store):
    """Per §10, a version bump that consumes base concepts produces BOTH
    COMPOSED_FROM and COMPOSED_INTO edges so decompile + recompile traversal
    both work."""
    store.create_concept(
        "linux_terminal", "Linux terminal", memory_type="declarative",
    )
    store.create_concept(
        "ssh", "SSH", memory_type="declarative",
    )
    # bump cosmetic_business_website v1 → v2, consuming ssh v1 as a new base.
    store.create_concept(
        "cosmetic_business_website", "Cosmetic business website",
        memory_type="declarative",
        composed_from=[("linux_terminal", 1)],
    )
    store.bump_version(
        "cosmetic_business_website",
        composed_from=[("linux_terminal", 1), ("ssh", 1)],
    )

    # v=2 neighbors include both bases.
    v2_neighbors = graph.spine_concept_neighbors(
        "cosmetic_business_website", version=2, limit=20,
    )
    assert ("linux_terminal", 1) in v2_neighbors
    assert ("ssh", 1) in v2_neighbors

    # Reciprocal COMPOSED_INTO edges from bases to v=2.
    ssh_neighbors = graph.spine_concept_neighbors("ssh", version=1)
    assert ("cosmetic_business_website", 2) in ssh_neighbors


# ── recompute_groundedness ──────────────────────────────────────────


def test_recompute_groundedness_updates_persisted_value(graph, store):
    store.create_concept(
        "x", "X", memory_type="declarative",
    )
    g0 = graph.spine_get_concept_version("x", 1)["groundedness"]

    new_g = store.recompute_groundedness(
        "x", 1, episodic_encounters=50, distinct_contexts=50,
        procedural_links=50,
    )
    assert new_g == pytest.approx(1.0, abs=1e-6)
    g1 = graph.spine_get_concept_version("x", 1)["groundedness"]
    assert g1 == pytest.approx(new_g)
    assert g1 > g0


def test_recompute_groundedness_missing_row_returns_zero(graph, store):
    """Missing row returns 0.0 without raising — used by consolidation's
    batch path which may race against GC."""
    assert store.recompute_groundedness("missing", 99) == 0.0


def test_recompute_groundedness_batch_handles_partial_failures(graph, store):
    store.create_concept("a", "A", memory_type="declarative")
    store.create_concept("b", "B", memory_type="declarative")

    rows = [
        {"concept_id": "a", "version": 1, "episodic_encounters": 10},
        {"concept_id": "missing", "version": 1, "episodic_encounters": 5},
        {"concept_id": "b", "version": 1, "procedural_links": 3},
        # Bad row — missing required key. Should be skipped, not crash.
        {"concept_id": "broken"},
    ]
    n = store.recompute_groundedness_batch(rows)
    # `a` updated, `missing` returns 0 (NOT counted), `b` updated, `broken` raises KeyError.
    # The 'missing' row's recompute returned 0.0 but didn't raise — n counts every row
    # that didn't raise. So: a + missing + b succeed (3); broken raises → 3.
    assert n == 3


# ── Sole-writer contract ────────────────────────────────────────────


def test_export_snapshot_writes_atomic_json(graph, store):
    """FU-1: EngramStore.export_snapshot writes a valid JSON snapshot
    with the documented schema (version + exported_at + concepts +
    composition_edges)."""
    import json
    import os
    import tempfile

    store.create_concept("a", "A", memory_type="declarative")
    store.create_concept(
        "b", "B", memory_type="declarative",
        composed_from=[("a", 1)],
    )
    store.bump_version("b")  # B v=2

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "spine.json")
        n = store.export_snapshot(snap_path)
        assert n == 3  # a/v1, b/v1, b/v2

        with open(snap_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "exported_at" in data
        concept_ids = sorted({c["concept_id"] for c in data["concepts"]})
        assert concept_ids == ["a", "b"]
        b_versions = sorted(
            c["version"] for c in data["concepts"]
            if c["concept_id"] == "b"
        )
        assert b_versions == [1, 2]
        edges_from = data["composition_edges"]["from"]
        edges_into = data["composition_edges"]["into"]
        assert [["b", 1], ["a", 1]] in edges_from
        assert [["a", 1], ["b", 1]] in edges_into


def test_export_snapshot_empty_spine_returns_zero(graph, store):
    """Brand-new spine → exports 0 concepts but still writes a valid
    JSON file (frontend can distinguish 'empty spine' from 'missing
    snapshot')."""
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "spine.json")
        n = store.export_snapshot(snap_path)
        assert n == 0
        with open(snap_path) as f:
            data = json.load(f)
        assert data["concepts"] == []
        assert data["composition_edges"] == {"from": [], "into": []}


def test_writer_called_once_per_create_or_bump(graph):
    """Sanity: exactly one outer_memory_writer call per EngramStore op."""
    writer = FakeWriter()
    s = EngramStore(graph, writer, clock=lambda: 1000.0)
    s.create_concept("c1", "C1", memory_type="declarative")
    s.create_concept("c2", "C2", memory_type="procedural")
    s.bump_version("c1")
    assert len(writer.calls) == 3
    assert writer.calls[0].version == 1
    assert writer.calls[1].version == 1
    assert writer.calls[2].version == 2
    assert writer.calls[2].concept_id == "c1"
