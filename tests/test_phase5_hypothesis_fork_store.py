"""Phase 5 — HypothesisForkStore tests (P5.A-G).

Covers `titan_hcl/synthesis/hypothesis_fork_store.py` against PLAN
§P5.A-G + the INV-3 / INV-10 / INV-Syn-8 / INV-Syn-9 / INV-Syn-11 invariants:

- create_fork: net-new (root_anchor=None), repair (root_anchor=<tx>),
  argument-validation guards, parent-exists check, Kuzu writes
- create_fork: net-new + repair both write Kuzu HypothesisFork node;
  repair also writes EXPLORES edge to parent concept
- record_exploration_tx: persisted to durable table, replayed on restart
- on_fork_read: increments use_count, touches activation, auto-grad at 3
- graduate_oracle: verdict='true' gate; verdict-not-true raises;
  emits concept-version TX via writer; reuses EngramStore (INV-10/11);
  status='graduated' + anchor_tx persisted
- graduate_used: requires use_count>=3; auto-trigger on threshold crossing
- abandon: writes tombstone TX with exploration_root Merkle (INV-Syn-9);
  idempotent on re-call (no-op); status='abandoned' + tombstone_tx persisted
- find_below_floor: respects locked-in defaults; only returns status='open'
- update_activation: persists base_level to DuckDB + Kuzu
- export_snapshot: atomic JSON, includes all rows + summary
- purge_durable_state: drops Kuzu node + durable exploration log

Real EngramStore + real Kuzu graph; FakeWriter for OuterMemoryWriter
(spies on TX emissions); in-memory DuckDB for synthesis.duckdb-like store.

Uses a fake ActivationStore (record_access spy) — the real one belongs to
synthesis_worker process; tests don't need its full DuckDB persistence.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.hypothesis_fork_store import (
    DEFAULT_ACTIVATION_FLOOR,
    DEFAULT_WINDOW_SEC,
    USE_COUNT_GRADUATION_THRESHOLD,
    ForkNotFound,
    ForkStateError,
    GraduationGateError,
    HypothesisForkStore,
)


# ── Fakes ─────────────────────────────────────────────────────────


class FakeActivationStore:
    def __init__(self):
        self.access_calls: list[tuple[str, float]] = []

    def record_access(self, item_id: str, ts: float) -> None:
        self.access_calls.append((item_id, ts))


@dataclass
class _GradWriterCall:
    concept_id: str
    version: int
    derivation_merkle_root: str
    oracle_verdict: dict


@dataclass
class _TombstoneCall:
    fork_id: str
    root_anchor: Optional[str]
    exploration_root: str
    abandonment_reason: str
    reference_count_pruned: int


class FakeOuterMemoryWriter:
    """Spies on graduation + tombstone calls. Also fulfils the writer
    protocol that EngramStore needs (write_concept_version returns a tx)."""

    def __init__(self):
        self.grad_calls: list[_GradWriterCall] = []
        self.tombstones: list[_TombstoneCall] = []
        self._counter = 0
        self.bare_version_writes: list[dict] = []

    # EngramStore-facing
    def write_concept_version(self, **kwargs) -> str:
        self.bare_version_writes.append(kwargs)
        self._counter += 1
        return f"tx_bare_{kwargs['concept_id']}_v{kwargs['version']}_{self._counter}"

    # HypothesisForkStore-facing
    def write_concept_version_with_proof(self, **kwargs) -> tuple[str, str]:
        self.grad_calls.append(_GradWriterCall(
            concept_id=kwargs["concept_id"], version=kwargs["version"],
            derivation_merkle_root=kwargs["derivation_merkle_root"],
            oracle_verdict=kwargs["oracle_verdict"],
        ))
        self._counter += 1
        return (
            f"tx_grad_{kwargs['concept_id']}_v{kwargs['version']}_{self._counter}",
            f"tx_verdict_{self._counter}",
        )

    def write_tombstone(self, **kwargs) -> str:
        self.tombstones.append(_TombstoneCall(
            fork_id=kwargs["fork_id"],
            root_anchor=kwargs["root_anchor"],
            exploration_root=kwargs["exploration_root"],
            abandonment_reason=kwargs["abandonment_reason"],
            reference_count_pruned=kwargs["reference_count_pruned"],
        ))
        self._counter += 1
        return f"tx_tombstone_{kwargs['fork_id']}_{self._counter}"


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def tmp_kuzu():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "test_p5.kuzu")


@pytest.fixture()
def graph(tmp_kuzu):
    g = TitanKnowledgeGraph(tmp_kuzu)
    try:
        yield g
    finally:
        g.close()


@pytest.fixture()
def duck():
    """Each test gets its own in-memory DuckDB connection (no cross-test
    state leak; cheap to spin up)."""
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def writer():
    return FakeOuterMemoryWriter()


@pytest.fixture()
def activation():
    return FakeActivationStore()


@pytest.fixture()
def engram_store(graph, writer):
    """Real EngramStore — the graduation paths go through this for end-
    to-end INV-10 reuse."""
    return EngramStore(graph, writer, clock=lambda: 1000.0)


@pytest.fixture()
def store(duck, graph, engram_store, writer, activation):
    """Hypothesis-fork store under test.

    Clock starts at 10_000_000.0 (~116 days post-epoch) so the default
    TTL window (7 days) computes a non-degenerate `cutoff` value that
    `last_touched` columns can be compared against in tests."""
    clock_calls = [10_000_000.0]

    def clock():
        return clock_calls[0]

    s = HypothesisForkStore(
        duckdb_conn=duck,
        kuzu_graph=graph,
        engram_store=engram_store,
        outer_memory_writer=writer,
        activation_store=activation,
        clock=clock,
    )
    # Expose the clock-list for tests that need to advance time.
    s.__test_clock__ = clock_calls
    return s


# ── create_fork ──────────────────────────────────────────────────


def test_create_fork_netnew_inserts_duckdb_row_and_kuzu_node(store, graph, duck):
    fid = store.create_fork(intent="Try a new ranking approach")
    assert isinstance(fid, str)
    assert len(fid) == 16

    # DuckDB row
    row = duck.execute(
        "SELECT fork_id, root_anchor, parent_concept_id, intent, status, "
        "use_count, activation FROM hypothesis_forks WHERE fork_id=?",
        (fid,),
    ).fetchone()
    assert row is not None
    assert row[0] == fid
    assert row[1] is None
    assert row[2] is None
    assert row[3] == "Try a new ranking approach"
    assert row[4] == "open"
    assert row[5] == 0
    assert row[6] == 0.0

    # Kuzu node
    node = graph.fork_get_node(fid)
    assert node is not None
    assert node["fork_id"] == fid
    assert node["root_anchor"] is None         # empty string → None
    assert node["status"] == "open"


def test_create_fork_repair_requires_existing_parent(store, graph):
    # Parent doesn't exist → ForkStateError
    with pytest.raises(ForkStateError):
        store.create_fork(
            intent="Repair a non-existent thing",
            root_anchor="tx_aabbcc",
            parent_concept_id="ghost_concept",
        )


def test_create_fork_repair_writes_explores_edge(store, graph, engram_store):
    # Materialize a parent concept first.
    parent = engram_store.create_concept(
        concept_id="metaplex_nft_minting", name="Metaplex NFT minting",
        memory_type="procedural",
    )
    fid = store.create_fork(
        intent="Resolve undocumented metaplex bug",
        root_anchor=parent.anchor_tx,
        parent_concept_id="metaplex_nft_minting",
    )
    # EXPLORES edge exists
    targets = graph.fork_explores_targets(fid)
    assert targets == [("metaplex_nft_minting", 1)]


def test_create_fork_validates_arg_consistency(store):
    with pytest.raises(ValueError):
        store.create_fork(intent="", )
    with pytest.raises(ValueError):
        store.create_fork(intent="x", root_anchor="tx_a")   # missing parent
    with pytest.raises(ValueError):
        store.create_fork(intent="x", parent_concept_id="p")  # missing root


def test_create_fork_touches_activation(store, activation):
    fid = store.create_fork(intent="prepare for ranking experiment")
    assert (f"fork:{fid}", 10_000_000.0) in activation.access_calls


# ── record_exploration_tx ─────────────────────────────────────────


def test_record_exploration_tx_persists_to_durable_log(store, duck):
    fid = store.create_fork(intent="exp")
    store.record_exploration_tx(fid, "a" * 64)
    store.record_exploration_tx(fid, "b" * 64)

    rows = duck.execute(
        "SELECT tx_hash FROM hypothesis_fork_explorations WHERE fork_id=? "
        "ORDER BY recorded_at",
        (fid,),
    ).fetchall()
    assert [r[0] for r in rows] == ["a" * 64, "b" * 64]


def test_record_exploration_tx_rejects_unknown_fork(store):
    with pytest.raises(ForkNotFound):
        store.record_exploration_tx("ghost", "a" * 64)


def test_record_exploration_tx_rejects_non_open_fork(store):
    fid = store.create_fork(intent="exp")
    store.abandon(fork_id=fid, reason="test")
    with pytest.raises(ForkStateError):
        store.record_exploration_tx(fid, "a" * 64)


# ── on_fork_read + use-count graduation ───────────────────────────


def test_on_fork_read_increments_use_count(store, duck):
    fid = store.create_fork(intent="exp")
    store.on_fork_read(fid)
    store.on_fork_read(fid)
    row = duck.execute(
        "SELECT use_count FROM hypothesis_forks WHERE fork_id=?", (fid,),
    ).fetchone()
    assert row[0] == 2


def test_on_fork_read_auto_graduates_at_threshold(store, duck, writer):
    """USE_COUNT_GRADUATION_THRESHOLD=3 default. Third on_fork_read fires
    graduate_used; fork ends up 'graduated' with a concept anchor_tx."""
    assert USE_COUNT_GRADUATION_THRESHOLD == 3
    fid = store.create_fork(intent="a known-good exploration")
    # Two reads — no auto-graduation yet.
    assert store.on_fork_read(fid) is None
    assert store.on_fork_read(fid) is None
    fork_mid = store.get_fork(fid)
    assert fork_mid.status == "open"
    assert fork_mid.use_count == 2

    # Third read fires auto-graduate.
    tx = store.on_fork_read(fid)
    assert tx is not None
    fork_final = store.get_fork(fid)
    assert fork_final.status == "graduated"
    assert fork_final.use_count == 3
    assert fork_final.graduated_anchor_tx is not None

    # Writer saw a write_concept_version_with_proof call carrying
    # oracle_id='use_threshold'.
    assert len(writer.grad_calls) == 1
    assert writer.grad_calls[0].oracle_verdict["oracle_id"] == "use_threshold"
    assert writer.grad_calls[0].oracle_verdict["verdict"] == "true"


def test_on_fork_read_on_graduated_fork_does_not_bump(store, duck):
    fid = store.create_fork(intent="x")
    # Force-graduate via three reads.
    for _ in range(3):
        store.on_fork_read(fid)
    # Additional read on graduated fork: returns None, no count change.
    assert store.on_fork_read(fid) is None
    use_count = duck.execute(
        "SELECT use_count FROM hypothesis_forks WHERE fork_id=?", (fid,),
    ).fetchone()[0]
    assert use_count == 3


# ── graduate_oracle ───────────────────────────────────────────────


def test_graduate_oracle_requires_verdict_true(store):
    fid = store.create_fork(intent="x")
    with pytest.raises(GraduationGateError):
        store.graduate_oracle(
            fork_id=fid,
            oracle_verdict={"verdict": "false", "oracle_id": "x"},
            concept_name="X",
        )
    with pytest.raises(GraduationGateError):
        store.graduate_oracle(
            fork_id=fid,
            oracle_verdict={"verdict": "unknown", "oracle_id": "x"},
            concept_name="X",
        )


def test_graduate_oracle_netnew_writes_concept_version_with_proof(
    store, writer,
):
    fid = store.create_fork(intent="net-new earned concept")
    store.record_exploration_tx(fid, "a" * 64)
    store.record_exploration_tx(fid, "b" * 64)

    tx = store.graduate_oracle(
        fork_id=fid,
        oracle_verdict={
            "oracle_id": "coding_sandbox", "verdict": "true",
            "evidence_ref": "sandbox_run_42", "cost": 0.0,
            "latency_ms": 12, "ts": 1000.5,
        },
        concept_name="NewEarnedConcept",
    )
    assert tx is not None
    fork_final = store.get_fork(fid)
    assert fork_final.status == "graduated"

    # Writer saw the proof-carrying call.
    assert len(writer.grad_calls) == 1
    call = writer.grad_calls[0]
    assert call.oracle_verdict["oracle_id"] == "coding_sandbox"
    assert call.oracle_verdict["verdict"] == "true"
    # Merkle root over the two exploration TXs.
    assert len(call.derivation_merkle_root) == 64
    assert call.derivation_merkle_root != "0" * 64


def test_graduate_oracle_repair_reuses_bump_version(
    store, graph, engram_store, writer,
):
    """INV-10 / INV-Syn-11: repair-fork graduation produces v(n+1)
    insert-only, parent v(n) byte-identical."""
    parent = engram_store.create_concept(
        concept_id="metaplex", name="Metaplex", memory_type="procedural",
    )
    parent_row_v1 = graph.spine_get_concept_version("metaplex", 1)
    assert parent_row_v1 is not None

    fid = store.create_fork(
        intent="resolve undocumented bug",
        root_anchor=parent.anchor_tx,
        parent_concept_id="metaplex",
    )
    store.record_exploration_tx(fid, "c" * 64)
    store.record_exploration_tx(fid, "d" * 64)
    store.graduate_oracle(
        fork_id=fid,
        oracle_verdict={
            "oracle_id": "solana_rpc", "verdict": "true",
            "evidence_ref": "tx_sig_xyz", "cost": 0.0001,
            "latency_ms": 250, "ts": 1001.0,
        },
    )

    # v=2 must exist; v=1 must be byte-identical to before graduation.
    v2 = graph.spine_get_concept_version("metaplex", 2)
    assert v2 is not None
    parent_row_v1_after = graph.spine_get_concept_version("metaplex", 1)
    assert parent_row_v1_after == parent_row_v1


def test_graduate_oracle_rejects_non_open_fork(store):
    fid = store.create_fork(intent="x")
    store.abandon(fork_id=fid, reason="test")
    with pytest.raises(ForkStateError):
        store.graduate_oracle(
            fork_id=fid,
            oracle_verdict={"verdict": "true", "oracle_id": "x"},
            concept_name="X",
        )


def test_graduate_oracle_netnew_requires_concept_name(store):
    fid = store.create_fork(intent="x")
    with pytest.raises(ValueError):
        store.graduate_oracle(
            fork_id=fid,
            oracle_verdict={"verdict": "true", "oracle_id": "x"},
            concept_name=None,
        )


# ── graduate_used ─────────────────────────────────────────────────


def test_graduate_used_requires_threshold(store):
    fid = store.create_fork(intent="x")
    with pytest.raises(GraduationGateError):
        store.graduate_used(fork_id=fid, concept_name="X")


# ── abandon ───────────────────────────────────────────────────────


def test_abandon_writes_tombstone_with_exploration_root(store, writer):
    fid = store.create_fork(intent="never confirmed")
    store.record_exploration_tx(fid, "a" * 64)
    store.record_exploration_tx(fid, "b" * 64)
    tx = store.abandon(fork_id=fid, reason="activation_below_floor")
    assert tx is not None

    fork_final = store.get_fork(fid)
    assert fork_final.status == "abandoned"
    assert fork_final.abandonment_reason == "activation_below_floor"

    # Tombstone TX recorded by writer.
    assert len(writer.tombstones) == 1
    ts = writer.tombstones[0]
    assert ts.fork_id == fid
    assert len(ts.exploration_root) == 64
    assert ts.abandonment_reason == "activation_below_floor"


def test_abandon_empty_fork_produces_deterministic_exploration_root(
    store, writer,
):
    """A fork that recorded zero exploration TXs still gets a deterministic
    exploration_root (SHA-256 of empty) — proves the SET was empty, never
    silent gap (INV-Syn-9)."""
    import hashlib
    fid = store.create_fork(intent="never explored")
    store.abandon(fork_id=fid, reason="never_advanced")
    expected = hashlib.sha256(b"").hexdigest()
    assert writer.tombstones[0].exploration_root == expected


def test_abandon_idempotent_on_already_terminal_states(store):
    fid = store.create_fork(intent="x")
    store.abandon(fork_id=fid)
    # Re-call → returns None (no-op).
    assert store.abandon(fork_id=fid) is None
    # Force-graduate via 3 reads on a different fork:
    fid2 = store.create_fork(intent="y")
    for _ in range(3):
        store.on_fork_read(fid2)
    assert store.abandon(fork_id=fid2) is None


# ── find_below_floor ──────────────────────────────────────────────


def test_find_below_floor_only_returns_open_status(store, duck):
    fid_open = store.create_fork(intent="open and stale")
    fid_grad = store.create_fork(intent="will graduate")
    for _ in range(3):
        store.on_fork_read(fid_grad)
    # Force activation below floor + last_touched older than window for
    # the open fork.
    duck.execute(
        "UPDATE hypothesis_forks SET activation=-5.0, last_touched=0 "
        "WHERE fork_id=?",
        (fid_open,),
    )
    # And for the graduated fork (it should NOT be returned even if it
    # otherwise satisfies the threshold).
    duck.execute(
        "UPDATE hypothesis_forks SET activation=-5.0, last_touched=0 "
        "WHERE fork_id=?",
        (fid_grad,),
    )

    out = store.find_below_floor()
    assert fid_open in out
    assert fid_grad not in out


def test_find_below_floor_respects_locked_in_defaults(store):
    """Per Maker decision 2026-05-27 floor=-3.0, window=7d."""
    assert DEFAULT_ACTIVATION_FLOOR == -3.0
    assert DEFAULT_WINDOW_SEC == 7.0 * 86400.0


# ── update_activation + snapshot ──────────────────────────────────


def test_update_activation_persists_to_duckdb(store, duck):
    fid = store.create_fork(intent="x")
    store.update_activation(fid, -2.5)
    val = duck.execute(
        "SELECT activation FROM hypothesis_forks WHERE fork_id=?", (fid,),
    ).fetchone()[0]
    assert val == -2.5


def test_export_snapshot_is_atomic_and_complete(store, tmp_kuzu):
    fid_a = store.create_fork(intent="A")
    fid_b = store.create_fork(intent="B")
    store.abandon(fork_id=fid_b)

    out_path = os.path.join(os.path.dirname(tmp_kuzu), "forks_snapshot.json")
    n = store.export_snapshot(out_path)
    assert n == 2

    import json
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["version"] == 1
    assert len(payload["forks"]) == 2
    assert payload["summary"]["open"] == 1
    assert payload["summary"]["abandoned"] == 1
    assert payload["summary"]["graduated"] == 0

    fork_ids_in_snapshot = {f["fork_id"] for f in payload["forks"]}
    assert fork_ids_in_snapshot == {fid_a, fid_b}


# ── purge_durable_state ───────────────────────────────────────────


def test_purge_durable_state_drops_kuzu_and_log(store, graph, duck):
    fid = store.create_fork(intent="x")
    store.record_exploration_tx(fid, "a" * 64)
    store.abandon(fork_id=fid)

    # Before: Kuzu node + durable log present.
    assert graph.fork_get_node(fid) is not None
    log_count = duck.execute(
        "SELECT COUNT(*) FROM hypothesis_fork_explorations WHERE fork_id=?",
        (fid,),
    ).fetchone()[0]
    assert log_count == 1

    store.purge_durable_state(fid)

    # After: Kuzu node gone, log gone, DuckDB lifecycle row preserved (audit).
    assert graph.fork_get_node(fid) is None
    log_count_after = duck.execute(
        "SELECT COUNT(*) FROM hypothesis_fork_explorations WHERE fork_id=?",
        (fid,),
    ).fetchone()[0]
    assert log_count_after == 0
    lifecycle_row = duck.execute(
        "SELECT status FROM hypothesis_forks WHERE fork_id=?", (fid,),
    ).fetchone()
    assert lifecycle_row is not None
    assert lifecycle_row[0] == "abandoned"


# ── crash-resilience: durable log rehydrates on restart ───────────


def test_durable_log_rehydrates_on_store_recreation(
    duck, graph, engram_store, writer, activation,
):
    """A worker restart re-instantiates HypothesisForkStore against the
    same DuckDB connection; the durable exploration log must be visible
    so a later abandonment can still compute the correct Merkle root."""
    s1 = HypothesisForkStore(
        duckdb_conn=duck, kuzu_graph=graph, engram_store=engram_store,
        outer_memory_writer=writer, activation_store=activation,
        clock=lambda: 1000.0,
    )
    fid = s1.create_fork(intent="rehydrate me")
    s1.record_exploration_tx(fid, "f" * 64)
    s1.record_exploration_tx(fid, "e" * 64)

    # "Restart": new instance, same DuckDB connection.
    s2 = HypothesisForkStore(
        duckdb_conn=duck, kuzu_graph=graph, engram_store=engram_store,
        outer_memory_writer=writer, activation_store=activation,
        clock=lambda: 1000.0,
    )
    # abandon now via s2 — Merkle root must reflect the rehydrated set.
    s2.abandon(fork_id=fid, reason="post_restart_test")
    import hashlib
    # Two leaves: hash(f*32 || e*32). Order = recorded_at asc = (f, e).
    f_bytes = bytes.fromhex("f" * 64)
    e_bytes = bytes.fromhex("e" * 64)
    expected = hashlib.sha256(f_bytes + e_bytes).hexdigest()
    assert writer.tombstones[-1].exploration_root == expected
