"""Phase 6 — OracleRouter + OracleSpendStore + OuterMemoryWriter extension tests (§P6.F).

Covers SPEC §25.1 INV-Syn-12 + INV-Syn-13 + INV-4:

- OracleSpendStore: idempotent table create, UPSERT spend, snapshot export
- OracleRouter registration + can_handle dispatch
- Standalone path: claim → plug → verdict → anchor on fork per
  CLAIM_DOMAIN_TO_FORK (hard INV-Syn-12 assertion across all 12 domains)
- Companion path: parent_tool_call_tx set → buffer → flush_companion_batches
  produces one OracleVerdictBatch per fork with Merkle root over leaves
- Gate denial path: still anchors unknown verdict (auditable)
- Plug.verify raises → unknown verdict with evidence_ref="plug_exception"
- No-plug-for-domain → unknown verdict with evidence_ref="no_plug_for_domain"
- OuterMemoryWriter.write_oracle_verdict_standalone + _batch emit correct
  TIMECHAIN_COMMIT shape (queue captures)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

import duckdb
import pytest

from titan_hcl.synthesis.merkle import merkle_root_hex
from titan_hcl.synthesis.oracle_gate import OracleGate, OracleGateConfig
from titan_hcl.synthesis.oracle_router import (
    CLAIM_DOMAIN_TO_FORK,
    OracleRouter,
    OracleSpendStore,
)
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def duckdb_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    yield conn
    conn.close()


@pytest.fixture
def spend_store(duckdb_conn):
    return OracleSpendStore(duckdb_conn)


@pytest.fixture
def writer():
    """OuterMemoryWriter wrapping a queue.Queue so tests can capture emits."""
    q = Queue()
    w = OuterMemoryWriter(q, src="test_p6f")
    w._test_queue = q  # type: ignore[attr-defined]
    return w


@pytest.fixture
def default_gate():
    return OracleGate(
        OracleGateConfig(
            balance_sol_baseline=1.0,
            admit_threshold=0.15,
            default_daily_sol_budget=0.1,
            daily_sol_budget={},
        )
    )


# ─────────────────────────────────────────────────────────────────────────
# Fake plug
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FakePlug:
    oracle_id: str
    cost_class: str
    domains: frozenset
    next_verdict: OracleVerdict = field(default=None)  # type: ignore[assignment]
    raise_on_verify: bool = False
    verify_calls: list = field(default_factory=list)

    def can_handle(self, domain: str) -> bool:
        return domain in self.domains

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        self.verify_calls.append(claim)
        if self.raise_on_verify:
            raise RuntimeError("plug exploded")
        return self.next_verdict or OracleVerdict(
            oracle_id=self.oracle_id,
            verdict="true",
            evidence_ref="evidence",
            cost=0.0,
            latency_ms=1,
            ts=time.time(),
        )


def _make_router(
    *,
    gate,
    spend_store,
    writer,
    balance: float = 1.0,
    now: float = None,
) -> OracleRouter:
    return OracleRouter(
        gate=gate,
        spend_store=spend_store,
        outer_memory_writer=writer,
        balance_provider=lambda: balance,
        now_fn=lambda: (now if now is not None else time.time()),
    )


# ─────────────────────────────────────────────────────────────────────────
# OracleSpendStore
# ─────────────────────────────────────────────────────────────────────────


def test_spend_store_zero_when_no_row(spend_store):
    assert spend_store.spent_today("helius_rpc") == 0.0


def test_spend_store_record_then_read(spend_store):
    total = spend_store.record_spend("helius_rpc", 0.001)
    assert total == 0.001
    assert spend_store.spent_today("helius_rpc") == 0.001


def test_spend_store_records_accumulate(spend_store):
    spend_store.record_spend("helius_rpc", 0.001)
    spend_store.record_spend("helius_rpc", 0.002)
    spend_store.record_spend("helius_rpc", 0.003)
    assert spend_store.spent_today("helius_rpc") == pytest.approx(0.006)


def test_spend_store_zero_cost_still_bumps_n_calls(spend_store):
    """Free oracles + gate-denied paths still record the attempt."""
    spend_store.record_spend("coding_sandbox", 0.0)
    spend_store.record_spend("coding_sandbox", 0.0)
    snapshot = spend_store.export_snapshot()
    per = {p["oracle_id"]: p for p in snapshot["per_oracle"]}
    assert per["coding_sandbox"]["spent_sol"] == 0.0
    assert per["coding_sandbox"]["n_calls"] == 2


def test_spend_store_separate_oracles_isolated(spend_store):
    spend_store.record_spend("helius_rpc", 0.005)
    spend_store.record_spend("web_api", 0.003)
    assert spend_store.spent_today("helius_rpc") == 0.005
    assert spend_store.spent_today("web_api") == 0.003


def test_spend_store_snapshot_shape(spend_store):
    spend_store.record_spend("helius_rpc", 0.001)
    snap = spend_store.export_snapshot()
    assert "as_of" in snap
    assert "date" in snap
    assert isinstance(snap["per_oracle"], list)
    assert len(snap["per_oracle"]) == 1
    assert snap["per_oracle"][0]["oracle_id"] == "helius_rpc"


# ─────────────────────────────────────────────────────────────────────────
# OracleRouter registration
# ─────────────────────────────────────────────────────────────────────────


def test_router_register_and_list(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("a", "free", frozenset({"x"})))
    r.register(FakePlug("b", "metered", frozenset({"y"})))
    listed = r.registered_oracles()
    assert {p["oracle_id"] for p in listed} == {"a", "b"}
    assert {p["cost_class"] for p in listed} == {"free", "metered"}


def test_router_can_handle_dispatches_to_first_matching_plug(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    pa = FakePlug("a", "free", frozenset({"code_correctness"}))
    pb = FakePlug("b", "free", frozenset({"web_fact"}))
    r.register(pa)
    r.register(pb)

    res = r.verify(OracleClaim(domain="code_correctness", payload={"code": "x"}))
    assert len(pa.verify_calls) == 1
    assert len(pb.verify_calls) == 0
    assert res.verdict.verdict == "true"


# ─────────────────────────────────────────────────────────────────────────
# Standalone path — INV-Syn-12 fork routing (HARD assertion)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "domain,expected_fork",
    list(CLAIM_DOMAIN_TO_FORK.items()),
)
def test_inv_syn_12_standalone_fork_routing_hard(default_gate, spend_store, writer, domain, expected_fork):
    """Every claim domain in CLAIM_DOMAIN_TO_FORK rides the right fork."""
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    plug = FakePlug("test_plug", "free", frozenset({domain}))
    r.register(plug)

    res = r.verify(OracleClaim(domain=domain, payload={}))
    assert res.fork == expected_fork, (
        f"INV-Syn-12 violation: domain {domain!r} routed to {res.fork}, "
        f"expected {expected_fork}"
    )
    assert res.anchor_tx is not None
    assert len(res.anchor_tx) == 64  # sha256 hex


def test_unsupported_domain_rides_meta_fork(default_gate, spend_store, writer):
    """No plug, unknown domain → anchor on `meta` (default per
    INV-Syn-9 symmetry with tombstone routing)."""
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    res = r.verify(OracleClaim(domain="never_seen_domain", payload={}))
    assert res.fork == "meta"
    assert res.verdict.evidence_ref == "no_plug_for_domain"
    assert res.anchor_tx is not None


def test_standalone_path_emits_oracle_verdict_standalone_tx(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("p", "free", frozenset({"code_correctness"})))

    r.verify(OracleClaim(domain="code_correctness", payload={}))

    msg = writer._test_queue.get_nowait()
    assert msg["dst"] == "timechain"
    payload = msg["payload"]
    assert payload["thought_type"] == "oracle_verdict"
    assert payload["fork"] == "procedural"  # per CLAIM_DOMAIN_TO_FORK
    assert "oracle_verdict_standalone" in payload["tags"]
    assert any(t.startswith("domain:code_correctness") for t in payload["tags"])
    assert any(t.startswith("verdict:") for t in payload["tags"])


# ─────────────────────────────────────────────────────────────────────────
# Gate denial path
# ─────────────────────────────────────────────────────────────────────────


def test_gate_denied_metered_plug_still_anchors_unknown_verdict(default_gate, spend_store, writer):
    """INV-Syn-13: gate-denied claims still produce auditable on-chain verdicts."""
    r = _make_router(
        gate=default_gate, spend_store=spend_store, writer=writer,
        balance=0.0,  # broke — admit_score = 0 < threshold
    )
    plug = FakePlug("helius_rpc", "metered", frozenset({"solana_tx_confirmed"}))
    r.register(plug)

    res = r.verify(
        OracleClaim(domain="solana_tx_confirmed", payload={"signature": "sig"}, importance=0.5)
    )
    assert res.verdict.verdict == "unknown"
    assert res.verdict.evidence_ref == "metabolic_gate_denied"
    # Plug.verify was NOT called
    assert plug.verify_calls == []
    # Verdict was still anchored
    assert res.anchor_tx is not None
    # Verdict TX shape correct
    msg = writer._test_queue.get_nowait()
    assert msg["payload"]["thought_type"] == "oracle_verdict"
    assert msg["payload"]["fork"] == "procedural"


def test_gate_denied_budget_exhausted(default_gate, spend_store, writer):
    # Pre-spend the daily budget.
    spend_store.record_spend("helius_rpc", 0.1)  # full default budget

    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer, balance=10.0)
    r.register(FakePlug("helius_rpc", "metered", frozenset({"solana_tx_confirmed"})))

    res = r.verify(
        OracleClaim(
            domain="solana_tx_confirmed",
            payload={"cost_estimate_sol": 0.01},
            importance=0.9,
        )
    )
    assert res.verdict.verdict == "unknown"
    assert res.verdict.evidence_ref == "daily_budget_exhausted"


# ─────────────────────────────────────────────────────────────────────────
# Plug exception handling
# ─────────────────────────────────────────────────────────────────────────


def test_plug_exception_returns_unknown_verdict(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    plug = FakePlug("p", "free", frozenset({"code_correctness"}), raise_on_verify=True)
    r.register(plug)

    res = r.verify(OracleClaim(domain="code_correctness", payload={}))
    assert res.verdict.verdict == "unknown"
    assert res.verdict.evidence_ref == "plug_exception"
    # Anchored anyway (INV-4 single canonical write path)
    assert res.anchor_tx is not None


def test_plug_can_handle_exception_skipped(default_gate, spend_store, writer):
    """A misbehaving plug whose can_handle raises must not block other plugs."""

    class BadCanHandle(FakePlug):
        def can_handle(self, domain):
            raise RuntimeError("can_handle broken")

    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(BadCanHandle("bad", "free", frozenset({"x"})))
    r.register(FakePlug("good", "free", frozenset({"code_correctness"})))

    res = r.verify(OracleClaim(domain="code_correctness", payload={}))
    assert res.verdict.oracle_id == "good"
    assert res.verdict.verdict == "true"


# ─────────────────────────────────────────────────────────────────────────
# Companion path — Merkle-batched at flush
# ─────────────────────────────────────────────────────────────────────────


def test_companion_buffered_not_anchored_immediately(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("p", "free", frozenset({"code_correctness"})))

    res = r.verify(
        OracleClaim(domain="code_correctness", payload={}),
        parent_tool_call_tx="parent_tx_hash",
        parent_tool_call_fork="procedural",
    )
    # Buffered — no immediate anchor TX
    assert res.anchor_tx is None
    assert res.fork == "procedural"
    assert r.companion_buffer_size() == 1
    # NO oracle_verdict TX queued yet
    assert writer._test_queue.empty()


def test_companion_flush_emits_single_batch_per_fork(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("p", "free", frozenset({"code_correctness"})))

    # Buffer 3 verdicts on procedural
    for i in range(3):
        r.verify(
            OracleClaim(domain="code_correctness", payload={"i": i}),
            parent_tool_call_tx=f"tx_{i}",
            parent_tool_call_fork="procedural",
        )

    anchored = r.flush_companion_batches()
    assert set(anchored.keys()) == {"procedural"}
    assert r.companion_buffer_size() == 0

    msg = writer._test_queue.get_nowait()
    assert msg["payload"]["thought_type"] == "oracle_verdict_batch"
    assert msg["payload"]["fork"] == "procedural"
    assert msg["payload"]["content"]["n_entries"] == 3
    assert len(msg["payload"]["content"]["entries"]) == 3
    # Merkle root present and stable
    root = msg["payload"]["content"]["merkle_root"]
    assert len(root) == 64
    # Should match what merkle_root_hex computes over the leaves
    import hashlib
    leaves = [
        hashlib.sha256((f"tx_{i}" + ":evidence").encode()).hexdigest()
        for i in range(3)
    ]
    assert root == merkle_root_hex(leaves)


def test_companion_flush_handles_multiple_forks(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("p", "free", frozenset({"code_correctness", "web_fact"})))

    # 2 verdicts on procedural
    for i in range(2):
        r.verify(
            OracleClaim(domain="code_correctness", payload={"i": i}),
            parent_tool_call_tx=f"proc_tx_{i}",
            parent_tool_call_fork="procedural",
        )
    # 1 on declarative
    r.verify(
        OracleClaim(domain="web_fact", payload={}),
        parent_tool_call_tx="decl_tx",
        parent_tool_call_fork="declarative",
    )

    anchored = r.flush_companion_batches()
    assert set(anchored.keys()) == {"procedural", "declarative"}

    forks_in_queue = set()
    while not writer._test_queue.empty():
        msg = writer._test_queue.get_nowait()
        forks_in_queue.add(msg["payload"]["fork"])
    assert forks_in_queue == {"procedural", "declarative"}


def test_companion_flush_empty_buffer_returns_empty_dict(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    assert r.flush_companion_batches() == {}


def test_companion_default_fork_is_procedural_when_not_supplied(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer)
    r.register(FakePlug("p", "free", frozenset({"code_correctness"})))

    res = r.verify(
        OracleClaim(domain="code_correctness", payload={}),
        parent_tool_call_tx="parent_tx",
        # parent_tool_call_fork NOT supplied
    )
    assert res.fork == "procedural"


# ─────────────────────────────────────────────────────────────────────────
# OuterMemoryWriter extensions
# ─────────────────────────────────────────────────────────────────────────


def test_write_oracle_verdict_standalone_emits_correct_shape(writer):
    v = OracleVerdict(
        oracle_id="helius_rpc",
        verdict="true",
        evidence_ref="sig_hash",
        cost=0.0001,
        latency_ms=120,
        ts=1234567890.0,
    )
    tx = writer.write_oracle_verdict_standalone(
        verdict=v, claim_domain="solana_tx_confirmed", fork="procedural"
    )
    assert isinstance(tx, str)
    assert len(tx) == 64

    msg = writer._test_queue.get_nowait()
    payload = msg["payload"]
    assert payload["fork"] == "procedural"
    assert payload["thought_type"] == "oracle_verdict"
    assert "oracle_verdict_standalone" in payload["tags"]
    assert "oracle:helius_rpc" in payload["tags"]
    assert "domain:solana_tx_confirmed" in payload["tags"]
    assert "verdict:true" in payload["tags"]
    content = payload["content"]
    assert content["oracle_id"] == "helius_rpc"
    assert content["verdict"] == "true"
    assert content["evidence_ref"] == "sig_hash"
    assert content["cost"] == 0.0001
    assert content["latency_ms"] == 120
    assert content["claim_domain"] == "solana_tx_confirmed"
    assert content["ts"] == 1234567890.0


def test_write_oracle_verdict_batch_emits_correct_shape(writer):
    entries = [
        {
            "parent_tool_call_tx": "tx_a",
            "oracle_id": "p",
            "verdict": "true",
            "evidence_ref": "e1",
            "cost": 0.0,
            "ts": 1.0,
        },
        {
            "parent_tool_call_tx": "tx_b",
            "oracle_id": "p",
            "verdict": "false",
            "evidence_ref": "e2",
            "cost": 0.0,
            "ts": 2.0,
        },
    ]
    tx = writer.write_oracle_verdict_batch(
        fork="procedural",
        merkle_root="a" * 64,
        entries=entries,
    )
    assert isinstance(tx, str)
    assert len(tx) == 64

    msg = writer._test_queue.get_nowait()
    payload = msg["payload"]
    assert payload["fork"] == "procedural"
    assert payload["thought_type"] == "oracle_verdict_batch"
    assert "oracle_verdict_batch" in payload["tags"]
    assert any(t.startswith("merkle:") for t in payload["tags"])
    content = payload["content"]
    assert content["merkle_root"] == "a" * 64
    assert content["n_entries"] == 2
    assert content["entries"] == entries


# ─────────────────────────────────────────────────────────────────────────
# Spend recording side-effect of verify()
# ─────────────────────────────────────────────────────────────────────────


def test_verify_records_spend_for_admitted_metered_call(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer, balance=1.0)
    plug = FakePlug(
        "helius_rpc",
        "metered",
        frozenset({"solana_tx_confirmed"}),
        next_verdict=OracleVerdict(
            oracle_id="helius_rpc",
            verdict="true",
            evidence_ref="sig",
            cost=0.0001,
            latency_ms=50,
            ts=time.time(),
        ),
    )
    r.register(plug)

    r.verify(OracleClaim(domain="solana_tx_confirmed", payload={}, importance=0.5))
    assert spend_store.spent_today("helius_rpc") == pytest.approx(0.0001)


def test_verify_records_attempt_even_when_gate_denied(default_gate, spend_store, writer):
    r = _make_router(gate=default_gate, spend_store=spend_store, writer=writer, balance=0.0)
    r.register(FakePlug("helius_rpc", "metered", frozenset({"solana_tx_confirmed"})))

    r.verify(
        OracleClaim(domain="solana_tx_confirmed", payload={}, importance=0.5)
    )
    snap = spend_store.export_snapshot()
    per = {p["oracle_id"]: p for p in snap["per_oracle"]}
    assert per["helius_rpc"]["n_calls"] == 1
    assert per["helius_rpc"]["spent_sol"] == 0.0
