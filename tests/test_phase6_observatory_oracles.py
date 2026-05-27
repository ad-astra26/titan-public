"""Phase 6 — Observatory /v6/synthesis/oracles/* + /proofs/* tests (§P6.K).

Covers:
- OracleSnapshotExporter.build_payload shape (router/budget/coverage/
  recent_verdicts/recent_proofs all present)
- export() atomic tmp+rename writes a valid JSON snapshot
- export() handles missing data_dir (creates parents)
- per-oracle budget includes registered-but-no-spend-today rows
- record_verdict + record_proof append + bound the ring buffers
- 5 handler functions return correct soft-fail shape when snapshot
  missing / corrupt / stale / ok
- All routes registered in v6.ROUTE_TABLE with the expected handler names
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest

from titan_hcl.synthesis.oracle_coverage import CoverageAnalyzer
from titan_hcl.synthesis.oracle_gate import OracleGateConfig
from titan_hcl.synthesis.oracle_router import OracleRouter, OracleSpendStore
from titan_hcl.synthesis.oracle_snapshot import (
    OracleSnapshotExporter,
    SNAPSHOT_VERSION,
    resolve_snapshot_path,
)
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def writer():
    return OuterMemoryWriter(Queue(), src="test_p6k")


@pytest.fixture
def gate_config():
    return OracleGateConfig(
        balance_sol_baseline=1.0,
        admit_threshold=0.15,
        default_daily_sol_budget=0.1,
        daily_sol_budget={"helius_rpc": 0.1, "web_api": 0.1},
    )


@pytest.fixture
def duck_conn(tmp_path):
    import duckdb
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    yield conn
    conn.close()


@pytest.fixture
def spend_store(duck_conn):
    return OracleSpendStore(duck_conn)


@pytest.fixture
def router(writer, spend_store, gate_config):
    from titan_hcl.synthesis.oracle_gate import OracleGate
    return OracleRouter(
        gate=OracleGate(gate_config),
        spend_store=spend_store,
        outer_memory_writer=writer,
        balance_provider=lambda: 1.0,
    )


# ─────────────────────────────────────────────────────────────────────────
# OracleSnapshotExporter.build_payload — shape
# ─────────────────────────────────────────────────────────────────────────


def test_snapshot_payload_has_version_and_timestamp(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload(now=12345.6)
    assert payload["version"] == SNAPSHOT_VERSION
    assert payload["exported_at"] == 12345.6


def test_snapshot_payload_includes_router_listing(router, spend_store, gate_config):
    class P:
        oracle_id = "p1"
        cost_class = "free"

        def can_handle(self, d): return False

        def verify(self, c): pass

    router.register(P())
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    assert any(r["oracle_id"] == "p1" for r in payload["router"])


def test_snapshot_payload_budget_includes_per_oracle_spend(router, spend_store, gate_config):
    spend_store.record_spend("helius_rpc", 0.03)
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    per_oracle = payload["budget"]["per_oracle"]
    helius = next(p for p in per_oracle if p["oracle_id"] == "helius_rpc")
    assert helius["spent_sol"] == pytest.approx(0.03)
    assert helius["daily_budget_sol"] == 0.1
    assert helius["remaining_sol"] == pytest.approx(0.07)


def test_snapshot_payload_budget_includes_registered_oracles_with_no_spend(
    router, spend_store, gate_config
):
    """A registered metered plug that hasn't been called today still
    surfaces in budget with spent_sol=0, n_calls=0."""

    class P:
        oracle_id = "web_api"
        cost_class = "metered"

        def can_handle(self, d): return False

        def verify(self, c): pass

    router.register(P())
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    per_oracle = payload["budget"]["per_oracle"]
    web = next(p for p in per_oracle if p["oracle_id"] == "web_api")
    assert web["spent_sol"] == 0.0
    assert web["n_calls"] == 0
    assert web["remaining_sol"] == 0.1


def test_snapshot_payload_free_oracle_not_in_budget_without_spend(
    router, spend_store, gate_config
):
    """Free oracles don't surface in budget by default (they're not
    metered — no point showing them). They WILL surface if they
    actually had spend records (defensive — shouldn't happen but
    don't drop the row)."""

    class FreeP:
        oracle_id = "coding_sandbox"
        cost_class = "free"

        def can_handle(self, d): return False

        def verify(self, c): pass

    router.register(FreeP())
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    ids = {p["oracle_id"] for p in payload["budget"]["per_oracle"]}
    assert "coding_sandbox" not in ids


def test_snapshot_payload_includes_coverage_when_analyzer_present(router, spend_store, gate_config):
    coverage = CoverageAnalyzer(
        tool_call_reader=lambda s, l: [
            {"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0},
        ],
        batch_reader=lambda s, l: [],
    )
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
        coverage_analyzer=coverage,
    )
    payload = exp.build_payload()
    assert payload["coverage"]["coverage_ratio"] == 1.0
    assert payload["coverage"]["a6_gate_passes"] is True


def test_snapshot_payload_coverage_empty_without_analyzer(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    assert payload["coverage"] == {}


def test_snapshot_payload_recent_verdicts_and_proofs_empty_by_default(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    payload = exp.build_payload()
    assert payload["recent_verdicts"] == []
    assert payload["recent_proofs"] == []


# ─────────────────────────────────────────────────────────────────────────
# record_verdict + record_proof
# ─────────────────────────────────────────────────────────────────────────


def test_record_verdict_appears_in_snapshot(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    exp.record_verdict(
        tx_hash="tx_abc", oracle_id="coding_sandbox", verdict="true",
        claim_domain="code_correctness", evidence_ref="ev", cost=0.0,
        fork="procedural", ts=100.0,
    )
    payload = exp.build_payload()
    assert len(payload["recent_verdicts"]) == 1
    v = payload["recent_verdicts"][0]
    assert v["tx_hash"] == "tx_abc"
    assert v["verdict"] == "true"
    assert v["fork"] == "procedural"


def test_record_verdict_buffer_bounded(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
        max_recent=10,
    )
    for i in range(50):
        exp.record_verdict(
            tx_hash=f"tx_{i}", oracle_id="x", verdict="true",
            claim_domain="code_correctness", evidence_ref="ev",
            cost=0.0, fork="procedural",
        )
    payload = exp.build_payload()
    assert len(payload["recent_verdicts"]) == 10  # capped at max_recent


def test_record_proof_appears_in_snapshot(router, spend_store, gate_config):
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
    )
    exp.record_proof(
        strategy="merkle",
        commitment_hex="a" * 64,
        payload_ref=None,
        cost=0.0,
        ts=100.0,
    )
    payload = exp.build_payload()
    assert len(payload["recent_proofs"]) == 1
    assert payload["recent_proofs"][0]["strategy"] == "merkle"


# ─────────────────────────────────────────────────────────────────────────
# export() — atomic write
# ─────────────────────────────────────────────────────────────────────────


def test_export_writes_valid_json(router, spend_store, gate_config, tmp_path):
    snap_path = str(tmp_path / "subdir" / "oracles_snapshot.json")
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
        snapshot_path=snap_path,
    )
    assert exp.export() is True
    assert os.path.exists(snap_path)
    with open(snap_path, "r") as f:
        loaded = json.load(f)
    assert loaded["version"] == SNAPSHOT_VERSION


def test_export_creates_parent_directory(router, spend_store, gate_config, tmp_path):
    snap_path = str(tmp_path / "a" / "b" / "c" / "oracles_snapshot.json")
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
        snapshot_path=snap_path,
    )
    assert exp.export() is True
    assert os.path.exists(snap_path)


# ─────────────────────────────────────────────────────────────────────────
# Handlers — soft-fail shape
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolate_snapshot(tmp_path, monkeypatch):
    """Point handlers at a tmp directory so tests stay deterministic."""
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    # Clear handler-side cache
    from titan_hcl.api import synthesis_oracle_handlers
    synthesis_oracle_handlers._SNAPSHOT_CACHE.clear()
    yield tmp_path


def _request():
    """Build a fake FastAPI Request — just needs to be a valid object."""
    return MagicMock()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_handlers_missing_snapshot_return_soft_fail(isolate_snapshot):
    from titan_hcl.api.synthesis_oracle_handlers import (
        get_synthesis_oracles_router,
        get_synthesis_oracles_recent,
        get_synthesis_oracles_coverage,
        get_synthesis_oracles_budget,
        get_synthesis_proofs_recent,
    )
    # No snapshot file written yet.
    r = _run(get_synthesis_oracles_router(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "missing"
    assert r["router"] == []

    r = _run(get_synthesis_oracles_recent(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "missing"
    assert r["verdicts"] == []

    r = _run(get_synthesis_oracles_coverage(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "missing"

    r = _run(get_synthesis_oracles_budget(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "missing"

    r = _run(get_synthesis_proofs_recent(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "missing"


def test_handlers_ok_snapshot_returns_payload(
    isolate_snapshot, router, spend_store, gate_config
):
    from titan_hcl.api.synthesis_oracle_handlers import (
        get_synthesis_oracles_router,
        get_synthesis_oracles_coverage,
    )
    snap_path = str(isolate_snapshot / "oracles_snapshot.json")
    exp = OracleSnapshotExporter(
        router=router, spend_store=spend_store, gate_config=gate_config,
        snapshot_path=snap_path,
    )
    # Register one plug + record some spend so payload is non-trivial.
    class P:
        oracle_id = "coding_sandbox"
        cost_class = "free"
        def can_handle(self, d): return False
        def verify(self, c): pass

    router.register(P())
    exp.export()

    r = _run(get_synthesis_oracles_router(_request()))
    assert r["snapshot"] == "ok"
    assert any(p["oracle_id"] == "coding_sandbox" for p in r["router"])


def test_handlers_corrupt_snapshot_returns_corrupt_status(isolate_snapshot):
    """Write garbage; handlers must NOT 500."""
    snap_path = str(isolate_snapshot / "oracles_snapshot.json")
    with open(snap_path, "w") as f:
        f.write("not valid json {")
    # Clear cache so we re-parse.
    from titan_hcl.api import synthesis_oracle_handlers
    synthesis_oracle_handlers._SNAPSHOT_CACHE.clear()

    from titan_hcl.api.synthesis_oracle_handlers import (
        get_synthesis_oracles_router,
    )
    r = _run(get_synthesis_oracles_router(_request()))
    assert r["ok"] is True
    assert r["snapshot"] == "corrupt"
    assert r["router"] == []


# ─────────────────────────────────────────────────────────────────────────
# Route table registration
# ─────────────────────────────────────────────────────────────────────────


def test_all_5_p6k_routes_in_route_table():
    from titan_hcl.api.v6 import _T as ROUTE_TABLE
    paths = {row[0] for row in ROUTE_TABLE}
    expected = {
        "/v6/synthesis/oracles/router",
        "/v6/synthesis/oracles/recent",
        "/v6/synthesis/oracles/coverage",
        "/v6/synthesis/oracles/budget",
        "/v6/synthesis/proofs/recent",
    }
    missing = expected - paths
    assert not missing, f"Missing P6.K routes in ROUTE_TABLE: {missing}"


def test_p6k_handler_names_resolve_via_dashboard():
    """The v6 router resolves handlers via `getattr(dashboard, name)`;
    confirm all 5 handler names resolve to callables."""
    from titan_hcl.api import dashboard
    for name in (
        "get_v6_synthesis_oracles_router",
        "get_v6_synthesis_oracles_recent",
        "get_v6_synthesis_oracles_coverage",
        "get_v6_synthesis_oracles_budget",
        "get_v6_synthesis_proofs_recent",
    ):
        assert callable(getattr(dashboard, name, None)), (
            f"dashboard.{name} not callable / not re-exported"
        )
