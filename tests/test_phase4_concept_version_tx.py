"""Phase 4 — concept-version TX shape tests (P4.D).

Covers `titan_hcl/synthesis/outer_memory_writer.py::write_concept_version`
against PLAN §P4.D + arch §10.

- TX content shape exactly matches the §P4.D spec.
- Tags list: ["concept_version", "concept:<id>", "v:<n>", memory_type].
- Fork routing by memory_type.
- v=1 → parent_version_tx must be None; v>1 → must be set.
- Invalid memory_type rejected.
- Returned anchor_tx is a deterministic content-hash (same content →
  same hash; mutated content → different hash).
- composed_from cap respected (large list trimmed on TX, full list still
  passed by caller).
- EngramStore + real OuterMemoryWriter integration (INV-14 parity:
  EngramStore's FakeWriter-shape vs real writer produce a TX of the
  expected shape — proves the §P4.B mock contract matches reality).
"""
from __future__ import annotations

import queue

import pytest

from titan_hcl import bus
from titan_hcl.synthesis.outer_memory_writer import (
    OuterMemoryWriter,
    _CONCEPT_VERSION_FORK_BY_MEMORY_TYPE,
    _COMPOSED_FROM_CAP_ON_TX,
    _canonical_concept_content_hash,
)


@pytest.fixture()
def writer():
    q = queue.Queue()
    w = OuterMemoryWriter(send_queue=q, src="synthesis_worker_test")
    return w, q


def _drain(q: "queue.Queue") -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ── TX shape ────────────────────────────────────────────────────────


def test_v1_tx_shape_matches_p4d_spec(writer):
    w, q = writer
    anchor = w.write_concept_version(
        concept_id="metaplex_nft_minting",
        version=1,
        name="Metaplex NFT minting",
        memory_type="procedural",
        parent_version_tx=None,
        composed_from=[("solana_rpc", 1)],
        derivation_evidence=["tx_abc"],
        groundedness=0.3,
    )
    assert isinstance(anchor, str) and len(anchor) == 64  # sha256 hex

    msgs = _drain(q)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["type"] == bus.TIMECHAIN_COMMIT
    assert msg["src"] == "synthesis_worker_test"
    assert msg["dst"] == "timechain"

    p = msg["payload"]
    assert p["thought_type"] == "concept_version"
    # Fork routing: procedural memory_type → procedural fork.
    assert p["fork"] == "procedural"
    # Tags list matches §P4.D order/contents.
    assert p["tags"] == [
        "concept_version", "concept:metaplex_nft_minting", "v:1", "procedural",
    ]
    c = p["content"]
    assert c["concept_id"] == "metaplex_nft_minting"
    assert c["version"] == 1
    assert c["name"] == "Metaplex NFT minting"
    assert c["memory_type"] == "procedural"
    assert c["parent_version_tx"] is None
    assert c["composed_from"] == [{"concept_id": "solana_rpc", "version": 1}]
    assert c["derivation_evidence"] == ["tx_abc"]
    assert c["groundedness"] == 0.3
    assert c["derivation_merkle_root"] is None
    assert "created_at" in c


def test_v2_tx_carries_parent_version_tx(writer):
    w, q = writer
    parent_hash = "deadbeef" * 8  # arbitrary 64-char hash
    w.write_concept_version(
        concept_id="x", version=2, name="X",
        memory_type="declarative",
        parent_version_tx=parent_hash,
        composed_from=[],
        derivation_evidence=[],
        groundedness=0.5,
    )
    p = _drain(q)[0]["payload"]
    assert p["content"]["parent_version_tx"] == parent_hash
    assert p["tags"] == ["concept_version", "concept:x", "v:2", "declarative"]
    assert p["fork"] == "declarative"


def test_fork_routing_covers_all_memory_types(writer):
    """All 4 memory_types must map to a fork; assert the map is exhaustive."""
    w, q = writer
    for mt in ("declarative", "procedural", "episodic", "meta"):
        # v=1 to satisfy parent_version_tx invariant.
        w.write_concept_version(
            concept_id=f"c_{mt}", version=1, name=mt.title(),
            memory_type=mt,
            parent_version_tx=None,
            composed_from=[],
            derivation_evidence=[],
            groundedness=0.0,
        )
    msgs = _drain(q)
    forks = [m["payload"]["fork"] for m in msgs]
    assert forks == [_CONCEPT_VERSION_FORK_BY_MEMORY_TYPE[mt] for mt in
                     ("declarative", "procedural", "episodic", "meta")]


def test_invalid_memory_type_raises(writer):
    w, _ = writer
    with pytest.raises(ValueError):
        w.write_concept_version(
            concept_id="x", version=1, name="X",
            memory_type="philosophical",
            parent_version_tx=None,
            composed_from=[], derivation_evidence=[], groundedness=0.0,
        )


def test_v1_with_parent_raises(writer):
    """INV-3 inverse: v=1 cannot have a parent."""
    w, _ = writer
    with pytest.raises(ValueError):
        w.write_concept_version(
            concept_id="x", version=1, name="X", memory_type="declarative",
            parent_version_tx="some_hash",
            composed_from=[], derivation_evidence=[], groundedness=0.0,
        )


def test_v2_without_parent_raises(writer):
    """Non-genesis version MUST link to predecessor."""
    w, _ = writer
    with pytest.raises(ValueError):
        w.write_concept_version(
            concept_id="x", version=2, name="X", memory_type="declarative",
            parent_version_tx=None,
            composed_from=[], derivation_evidence=[], groundedness=0.0,
        )


def test_zero_version_raises(writer):
    w, _ = writer
    with pytest.raises(ValueError):
        w.write_concept_version(
            concept_id="x", version=0, name="X", memory_type="declarative",
            parent_version_tx=None,
            composed_from=[], derivation_evidence=[], groundedness=0.0,
        )


# ── Anchor hash determinism + content-addressing ────────────────────


def test_anchor_hash_is_deterministic_for_same_content():
    """Same canonical content → same hash. This is the content-addressing
    property that lets the spine row write synchronously without waiting
    for chain commit (arch §16.2)."""
    content = {
        "concept_id": "x", "version": 1, "name": "X",
        "memory_type": "declarative", "parent_version_tx": None,
        "composed_from": [], "derivation_evidence": [],
        "groundedness": 0.5, "derivation_merkle_root": None,
        "created_at": 1000.0,
    }
    h1 = _canonical_concept_content_hash(content)
    h2 = _canonical_concept_content_hash(dict(content))
    assert h1 == h2
    assert len(h1) == 64


def test_anchor_hash_differs_for_different_content():
    base = {
        "concept_id": "x", "version": 1, "name": "X",
        "memory_type": "declarative", "parent_version_tx": None,
        "composed_from": [], "derivation_evidence": [],
        "groundedness": 0.5, "derivation_merkle_root": None,
        "created_at": 1000.0,
    }
    h_base = _canonical_concept_content_hash(base)

    # Change groundedness → different hash.
    h_g = _canonical_concept_content_hash({**base, "groundedness": 0.6})
    assert h_g != h_base

    # Add a composed_from entry → different hash.
    h_c = _canonical_concept_content_hash({
        **base, "composed_from": [{"concept_id": "y", "version": 1}],
    })
    assert h_c != h_base


# ── composed_from cap on TX ─────────────────────────────────────────


def test_composed_from_cap_respected_on_tx(writer):
    """Excess composed_from parents trimmed at TX-write boundary so block
    sizes stay bounded under tiered anchoring (§16.1). Kuzu spine still
    carries the full neighborhood — chain payload is the lean projection."""
    w, q = writer
    big_list = [(f"base_{i}", 1) for i in range(_COMPOSED_FROM_CAP_ON_TX + 20)]
    w.write_concept_version(
        concept_id="composite", version=1, name="Composite",
        memory_type="meta",
        parent_version_tx=None,
        composed_from=big_list,
        derivation_evidence=[],
        groundedness=0.0,
    )
    p = _drain(q)[0]["payload"]
    on_tx = p["content"]["composed_from"]
    assert len(on_tx) == _COMPOSED_FROM_CAP_ON_TX
    assert on_tx[0] == {"concept_id": "base_0", "version": 1}
    # Last preserved entry is at index cap-1; entries beyond that are dropped.
    assert on_tx[-1] == {
        "concept_id": f"base_{_COMPOSED_FROM_CAP_ON_TX - 1}", "version": 1,
    }


# ── EngramStore integration ────────────────────────────────────────


def test_engram_store_with_real_writer_end_to_end():
    """INV-14 parity: EngramStore + real OuterMemoryWriter produces the
    expected TX on the queue + the expected Kuzu row. Proves the P4.B
    FakeWriter mock contract matches the real writer."""
    import os
    import tempfile
    from titan_hcl.core.direct_memory import TitanKnowledgeGraph
    from titan_hcl.synthesis.engram_store import EngramStore

    with tempfile.TemporaryDirectory() as tmp:
        g = TitanKnowledgeGraph(os.path.join(tmp, "p4d_e2e.kuzu"))
        q = queue.Queue()
        w = OuterMemoryWriter(send_queue=q, src="synthesis_worker_e2e_test")
        store = EngramStore(g, w, clock=lambda: 12345.6)

        cv = store.create_concept(
            "linux_terminal", "Linux terminal",
            memory_type="declarative",
        )

        # Spine row carries the content-hash returned by the real writer.
        row = g.spine_get_concept_version("linux_terminal", 1)
        assert row is not None
        assert row["anchor_tx"] == cv.anchor_tx
        assert len(row["anchor_tx"]) == 64

        # Exactly one TIMECHAIN_COMMIT on the queue with the right shape.
        msgs = _drain(q)
        assert len(msgs) == 1
        p = msgs[0]["payload"]
        assert p["thought_type"] == "concept_version"
        assert p["content"]["concept_id"] == "linux_terminal"
        assert p["content"]["version"] == 1
        assert p["tags"] == [
            "concept_version", "concept:linux_terminal", "v:1", "declarative",
        ]
        g.close()
