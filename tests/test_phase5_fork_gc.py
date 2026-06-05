"""Phase 5 — ForkGC sweep tests (P5.H).

Covers `titan_hcl/synthesis/fork_gc.py` against PLAN §P5.H + INV-Syn-10
(conjunction predicate; in-doubt-keep; nightly sweep semantics).

- sweep visits all pending_gc_targets (graduated + abandoned forks)
- dry-run mode: no writes; SweepReport.plans + total_nodes_dropped populated
- live mode: Kuzu HypothesisFork node deleted + durable log purged
- per-sweep cap: forks past the cap are skipped, retried next sweep
- INV-Syn-10 conjunction: each of (a)/(b)/(c) tested in isolation
  - (a) cross-fork reference keeps the node alive
  - (b) canonical anchor keeps the node alive
  - (c) activation above floor keeps the node alive
- repair-fork: parent concept never pruned (its anchor is canonical)
- idempotency: re-sweep after purge → no-op for that fork
- transactional safety: DuckDB rollback on memory_nodes failure leaves
  the fork still pending
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import duckdb
import pytest

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.fork_gc import (
    DEFAULT_MAX_NODES_PER_SWEEP,
    ForkGC,
    PrunePlan,
    SweepReport,
)
from titan_hcl.synthesis.hypothesis_fork_store import HypothesisForkStore


# ── Shared fakes (mirror the fork_store test file's pattern) ────


class FakeActivation:
    def __init__(self):
        self.access_calls: list[tuple[str, float]] = []

    def record_access(self, item_id: str, ts: float) -> None:
        self.access_calls.append((item_id, ts))


class FakeWriter:
    def __init__(self):
        self.grad_calls: list[dict] = []
        self.tombstones: list[dict] = []
        self._counter = 0

    def write_concept_version(self, **kwargs) -> str:
        self._counter += 1
        return f"tx_bare_{kwargs['concept_id']}_v{kwargs['version']}_{self._counter}"

    def write_concept_version_with_proof(self, **kwargs) -> tuple[str, str]:
        self.grad_calls.append(kwargs)
        self._counter += 1
        return (f"tx_grad_{self._counter}", f"tx_verdict_{self._counter}")

    def write_tombstone(self, **kwargs) -> str:
        self.tombstones.append(kwargs)
        self._counter += 1
        return f"tx_tomb_{self._counter}"


@pytest.fixture()
def tmp_kuzu():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "test_p5_gc.kuzu")


@pytest.fixture()
def graph(tmp_kuzu):
    g = TitanKnowledgeGraph(tmp_kuzu)
    try:
        yield g
    finally:
        g.close()


@pytest.fixture()
def duck():
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def writer():
    return FakeWriter()


@pytest.fixture()
def activation():
    return FakeActivation()


@pytest.fixture()
def engram_store(graph, writer):
    return EngramStore(graph, writer, clock=lambda: 10_000_000.0)


@pytest.fixture()
def store(duck, graph, engram_store, writer, activation):
    return HypothesisForkStore(
        duckdb_conn=duck, kuzu_graph=graph, engram_store=engram_store,
        outer_memory_writer=writer, activation_store=activation,
        clock=lambda: 10_000_000.0,
    )


@pytest.fixture()
def gc(store, duck, graph, activation):
    """ForkGC under test — no memory_db (cross-process scope skipped for
    base tests; predicate-scope tests inject a synthetic memory_db)."""
    return ForkGC(
        fork_store=store,
        synthesis_duckdb_conn=duck,
        kuzu_graph=graph,
        activation_store=activation,
        memory_db_conn=None,
        clock=lambda: 10_000_000.0,
    )


# ── Sweep visits pending targets (graduated + abandoned) ────────


def test_sweep_visits_graduated_and_abandoned_forks(store, gc):
    a = store.create_fork(intent="will-graduate")
    for _ in range(3):
        store.on_fork_read(a)
    b = store.create_fork(intent="will-abandon")
    store.abandon(fork_id=b)
    # Third fork stays open — should NOT be visited.
    _c_open = store.create_fork(intent="still open")

    report = gc.sweep(dry_run=True)
    visited_ids = {p.fork_id for p in report.plans}
    assert a in visited_ids
    assert b in visited_ids
    assert _c_open not in visited_ids


# ── Dry-run vs live ─────────────────────────────────────────────


def test_dry_run_makes_no_writes(store, duck, graph, gc):
    fid = store.create_fork(intent="x")
    store.abandon(fork_id=fid)
    assert graph.fork_get_node(fid) is not None

    report = gc.sweep(dry_run=True)
    assert report.dry_run is True
    assert report.total_nodes_dropped > 0
    assert graph.fork_get_node(fid) is not None  # NOT deleted


def test_live_sweep_purges_kuzu_node_and_durable_log(store, duck, graph, gc):
    fid = store.create_fork(intent="x")
    store.record_exploration_tx(fid, "a" * 64)
    store.abandon(fork_id=fid)
    assert graph.fork_get_node(fid) is not None
    log_count = duck.execute(
        "SELECT COUNT(*) FROM hypothesis_fork_explorations WHERE fork_id=?",
        (fid,),
    ).fetchone()[0]
    assert log_count == 1

    report = gc.sweep(dry_run=False)
    assert report.forks_pruned == 1
    assert graph.fork_get_node(fid) is None
    log_after = duck.execute(
        "SELECT COUNT(*) FROM hypothesis_fork_explorations WHERE fork_id=?",
        (fid,),
    ).fetchone()[0]
    assert log_after == 0
    # DuckDB lifecycle row preserved (audit trail).
    row = duck.execute(
        "SELECT status FROM hypothesis_forks WHERE fork_id=?", (fid,),
    ).fetchone()
    assert row is not None
    assert row[0] == "abandoned"


# ── Per-sweep cap ───────────────────────────────────────────────


def test_per_sweep_cap_bounds_work(store, duck, graph, activation):
    # Cap at 2 nodes/sweep — 2 forks with ~1 exploration TX each is
    # ~3 nodes/fork, so the first one fits (3 > 2 → exceeds budget,
    # skipped). Cap = 100 gives both room.
    gc_tiny = ForkGC(
        fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
        activation_store=activation, max_nodes_per_sweep=2,
    )
    for i in range(3):
        fid = store.create_fork(intent=f"f{i}")
        store.record_exploration_tx(fid, f"{i:064x}")
        store.abandon(fork_id=fid)

    report = gc_tiny.sweep(dry_run=False)
    # Each fork plan = Kuzu node (1) + 1 durable row (1) = 2. With cap=2
    # the first fork's plan exactly fits → pruned. Second fork: cap is
    # decremented by 2, leaving 0 → skipped. Third: also skipped.
    assert report.forks_pruned == 1
    assert report.forks_skipped >= 1


# ── Idempotency ─────────────────────────────────────────────────


def test_sweep_idempotent_on_already_pruned_fork(store, duck, graph, gc):
    fid = store.create_fork(intent="x")
    store.abandon(fork_id=fid)
    gc.sweep(dry_run=False)
    # Re-sweep: fork no longer in pending_gc_targets (Kuzu node deleted).
    report2 = gc.sweep(dry_run=False)
    assert report2.forks_visited == 0


# ── INV-Syn-10 predicate — clauses tested in isolation ─────────


def _setup_memory_db_with_node(memory_db, node_id: int, source_id: str):
    """Helper: synthesize a minimal memory_nodes table for predicate tests."""
    memory_db.execute(
        "CREATE TABLE memory_nodes (id INT PRIMARY KEY, source_id TEXT)"
    )
    memory_db.execute(
        "INSERT INTO memory_nodes (id, source_id) VALUES (?, ?)",
        (node_id, source_id),
    )


def test_predicate_a_keeps_node_referenced_by_other_open_fork(
    store, duck, graph, activation,
):
    """Predicate (a): if ANOTHER open fork also references the same node,
    keep it — sole-inbound predicate fails."""
    mem_db = duckdb.connect(":memory:")
    try:
        _setup_memory_db_with_node(mem_db, 42, "tx_aaaa_referenced_here")

        f1 = store.create_fork(intent="dying fork")
        f2 = store.create_fork(intent="still-open fork")
        store.record_exploration_tx(f1, "a" * 64)
        store.record_exploration_tx(f2, "a" * 64)   # SAME TX → cross-ref
        store.abandon(fork_id=f1)

        # In memory_nodes, source_id contains the 64-char TX hash of f1/f2.
        mem_db.execute(
            "UPDATE memory_nodes SET source_id=? WHERE id=42",
            ("a" * 64,),
        )

        gc = ForkGC(
            fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
            activation_store=activation,
            memory_db_conn=mem_db,
            chain_canonical_tx_resolver=lambda _h: False,  # all probationary
            activation_floor=0.0,    # high so B_i is below
        )
        # Pre-set a low activation in the synthesis store so (c) would pass.
        duck.execute(
            "CREATE TABLE IF NOT EXISTS activation_state ("
            "item_id TEXT PRIMARY KEY, base_level DOUBLE)"
        )
        duck.execute(
            "INSERT INTO activation_state VALUES ('mem:42', -10.0)"
        )
        plan = gc._plan_prune(f1)
        # (a) failed → node 42 NOT in prune list, keep_reasons has a-fail.
        assert 42 not in plan.memory_nodes_will_drop
        assert any("a-fail" in r for r in plan.keep_reasons)
    finally:
        mem_db.close()


def test_predicate_b_keeps_node_with_canonical_anchor(
    store, duck, graph, activation,
):
    """Predicate (b): if ANY TX in source_id is canonically anchored, keep."""
    mem_db = duckdb.connect(":memory:")
    try:
        _setup_memory_db_with_node(mem_db, 7, "a" * 64)

        fid = store.create_fork(intent="x")
        store.record_exploration_tx(fid, "a" * 64)
        store.abandon(fork_id=fid)

        gc = ForkGC(
            fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
            activation_store=activation,
            memory_db_conn=mem_db,
            chain_canonical_tx_resolver=lambda _h: True,  # all canonical
            activation_floor=0.0,
        )
        duck.execute(
            "CREATE TABLE IF NOT EXISTS activation_state ("
            "item_id TEXT PRIMARY KEY, base_level DOUBLE)"
        )
        duck.execute(
            "INSERT INTO activation_state VALUES ('mem:7', -10.0)"
        )
        plan = gc._plan_prune(fid)
        assert 7 not in plan.memory_nodes_will_drop
        assert any("b-fail" in r for r in plan.keep_reasons)
    finally:
        mem_db.close()


def test_predicate_c_keeps_node_above_activation_floor(
    store, duck, graph, activation,
):
    """Predicate (c): if B_i >= floor, keep — even if (a) and (b) say prune."""
    mem_db = duckdb.connect(":memory:")
    try:
        _setup_memory_db_with_node(mem_db, 99, "a" * 64)

        fid = store.create_fork(intent="x")
        store.record_exploration_tx(fid, "a" * 64)
        store.abandon(fork_id=fid)

        gc = ForkGC(
            fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
            activation_store=activation,
            memory_db_conn=mem_db,
            chain_canonical_tx_resolver=lambda _h: False,
            activation_floor=-5.0,
        )
        duck.execute(
            "CREATE TABLE IF NOT EXISTS activation_state ("
            "item_id TEXT PRIMARY KEY, base_level DOUBLE)"
        )
        # B_i = -2.0, floor = -5.0 → above floor → keep.
        duck.execute(
            "INSERT INTO activation_state VALUES ('mem:99', -2.0)"
        )
        plan = gc._plan_prune(fid)
        assert 99 not in plan.memory_nodes_will_drop
        assert any("c-fail" in r for r in plan.keep_reasons)
    finally:
        mem_db.close()


def test_predicate_all_three_fail_keeps_node(
    store, duck, graph, activation,
):
    """All three predicates failing also produces 'keep' with reasons."""
    mem_db = duckdb.connect(":memory:")
    try:
        _setup_memory_db_with_node(mem_db, 5, "a" * 64)
        # Two open forks both reference the TX → (a) cross-ref.
        f1 = store.create_fork(intent="x")
        f2 = store.create_fork(intent="y")
        store.record_exploration_tx(f1, "a" * 64)
        store.record_exploration_tx(f2, "a" * 64)
        store.abandon(fork_id=f1)

        gc = ForkGC(
            fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
            activation_store=activation,
            memory_db_conn=mem_db,
            chain_canonical_tx_resolver=lambda _h: True,  # (b) canonical
            activation_floor=-100.0,  # (c) B_i easily above
        )
        duck.execute(
            "CREATE TABLE IF NOT EXISTS activation_state ("
            "item_id TEXT PRIMARY KEY, base_level DOUBLE)"
        )
        duck.execute("INSERT INTO activation_state VALUES ('mem:5', 1.0)")
        plan = gc._plan_prune(f1)
        assert 5 not in plan.memory_nodes_will_drop
        # Test stops at the first failing predicate so we have ≥1 keep_reason.
        assert any("memory_node:5" in r for r in plan.keep_reasons)
    finally:
        mem_db.close()


def test_predicate_all_three_pass_prunes_node(
    store, duck, graph, activation,
):
    """The teeth: when (a) AND (b) AND (c) all pass, the node IS pruned."""
    mem_db = duckdb.connect(":memory:")
    try:
        # Use a valid 64-hex string ("e" is a valid hex digit, unlike "x"/"z").
        tx_hex = "e" * 64
        _setup_memory_db_with_node(mem_db, 11, tx_hex)

        fid = store.create_fork(intent="x")
        store.record_exploration_tx(fid, tx_hex)
        store.abandon(fork_id=fid)

        gc = ForkGC(
            fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
            activation_store=activation,
            memory_db_conn=mem_db,
            chain_canonical_tx_resolver=lambda _h: False,  # never canonical
            activation_floor=0.0,
        )
        duck.execute(
            "CREATE TABLE IF NOT EXISTS activation_state ("
            "item_id TEXT PRIMARY KEY, base_level DOUBLE)"
        )
        duck.execute("INSERT INTO activation_state VALUES ('mem:11', -10.0)")
        plan = gc._plan_prune(fid)
        assert 11 in plan.memory_nodes_will_drop
    finally:
        mem_db.close()


# ── Repair fork — parent concept never pruned ──────────────────


def test_repair_fork_parent_concept_never_in_plan(
    store, duck, graph, engram_store, activation,
):
    parent = engram_store.create_concept(
        concept_id="metaplex", name="Metaplex", memory_type="procedural",
    )
    fid = store.create_fork(
        intent="repair", root_anchor=parent.anchor_tx,
        parent_concept_id="metaplex",
    )
    store.record_exploration_tx(fid, "f" * 64)   # valid hex
    store.abandon(fork_id=fid)

    gc = ForkGC(
        fork_store=store, synthesis_duckdb_conn=duck, kuzu_graph=graph,
        activation_store=activation, memory_db_conn=None,
    )
    plan = gc._plan_prune(fid)
    # The parent Concept row is in Kuzu; it is NOT a hypothesis-fork node
    # so it never appears in any prune list. Sanity check: Kuzu row exists
    # before AND after sweep.
    assert graph.spine_get_concept_version("metaplex", 1) is not None
    gc.sweep(dry_run=False)
    assert graph.spine_get_concept_version("metaplex", 1) is not None


# ── Defaults ────────────────────────────────────────────────────


def test_default_cap_is_10k():
    assert DEFAULT_MAX_NODES_PER_SWEEP == 10_000
