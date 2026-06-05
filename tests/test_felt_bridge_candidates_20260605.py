"""RFP_inner_outer_felt_teaching_bridge §7.3 — Phase 3: producer (gap → queue + emit).

Covers:
  • FeltBridge.queue_candidate — §3.4 Object-shaped row, idempotent, returns
    True-on-new (so the producer emits the bus handoff once)
  • ConsolidationPass._queue_felt_gaps — ungrounded Object → candidate + emit;
    grounded Object → skipped (mismatch/frame_dependent deferral); felt seeded from
    agg_felt; emit fires once (not every dream)
  • PROPOSE-ONLY: the producer touches synthesis.duckdb + the emit fn ONLY — there is
    no CGN handle in the path, so zero CGN writes by construction (INV-Syn-ENG-4)
"""
from __future__ import annotations

import json
import types

import duckdb
import pytest

from titan_hcl.synthesis.felt_bridge import FeltBridge
from titan_hcl.synthesis.consolidation import (
    ConsolidationPass,
    Cluster,
    LLMProposal,
    TxCandidate,
    PROBATION_C,
    agg_felt,
)


class _DirectWriter:
    def submit(self, fn):
        return fn()

    def submit_sync(self, fn):
        return fn()


@pytest.fixture()
def bridge():
    conn = duckdb.connect(":memory:")
    fb = FeltBridge(conn, _DirectWriter())
    assert fb.ensure_schema() is True
    return fb, conn


def _make_pass(fb, emit=None, decompose=None):
    return ConsolidationPass(
        engram_store=object(),
        cgn_bridge=object(),
        outer_memory_writer=object(),
        mine_recent_txs_fn=lambda **_k: [],
        llm_propose_fn=lambda _c: LLMProposal(action="reject"),
        decompose_fn=decompose or (lambda _n, _s: ["glacier", "microbe"]),
        felt_bridge=fb,
        emit_candidate_fn=emit,
    )


def _cv(concept_id="glacier_x", version=2):
    return types.SimpleNamespace(concept_id=concept_id, version=version)


def _proposal():
    return LLMProposal(action="new_concept", concept_id="glacier_x",
                       proposed_name="Glacier Microbes", domain_hint="biology")


def _cluster():
    return Cluster(members=[
        TxCandidate("t1", "declarative", (), None, "glaciers host microbes",
                    felt='{"curiosity": 0.7, "awe": 0.4}'),
        TxCandidate("t2", "declarative", (), None, "altitude effects",
                    felt='{"curiosity": 0.5}'),
    ])


# ── queue_candidate (the §3.4 row) ────────────────────────────────────────
def test_queue_candidate_inserts_idempotent(bridge):
    fb, conn = bridge
    new1 = fb.queue_candidate(
        object_label="Microbe", felt_state_json='{"awe": 0.4}', c=PROBATION_C,
        source_engram="glacier_x", source_version=2, domain_hint="biology")
    assert new1 is True
    new2 = fb.queue_candidate(  # same PK → no re-insert
        object_label="microbe", felt_state_json='{"awe": 0.4}', c=PROBATION_C,
        source_engram="glacier_x", source_version=2, domain_hint="biology")
    assert new2 is False
    row = conn.execute(
        "SELECT object_label, c, source_engram, source_version, provenance, "
        "domain_hint, status FROM engram_felt_candidates").fetchall()
    assert row == [("microbe", PROBATION_C, "glacier_x", 2, "engram_felt_gap",
                    "biology", "candidate")]


def test_queue_candidate_soft_fail_no_raise():
    class _BrokenWriter:
        def submit(self, fn):
            raise RuntimeError("boom")

        def submit_sync(self, fn):
            raise RuntimeError("boom")

    fb = FeltBridge(duckdb.connect(":memory:"), _BrokenWriter())
    fb.ensure_schema()
    assert fb.queue_candidate(object_label="x", felt_state_json="{}", c=0.05,
                              source_engram="e", source_version=1) is False


# ── producer: gap → queue + emit ──────────────────────────────────────────
def test_gap_queues_and_emits(bridge):
    fb, conn = bridge
    emitted = []
    cp = _make_pass(fb, emit=emitted.append)
    out = cp._maybe_decompose(_cv(), _proposal(), _cluster())
    assert out == ["glacier", "microbe"]
    # Both Objects ungrounded → 2 candidate rows + 2 emits.
    rows = conn.execute(
        "SELECT object_label FROM engram_felt_candidates ORDER BY object_label"
    ).fetchall()
    assert rows == [("glacier",), ("microbe",)]
    assert {e["object_label"] for e in emitted} == {"glacier", "microbe"}
    # Payload shape + felt seeded from agg_felt(members).
    e0 = emitted[0]
    assert set(e0) == {"object_label", "felt_state", "source_engram",
                       "source_version", "domain_hint"}
    assert e0["source_engram"] == "glacier_x" and e0["source_version"] == 2
    assert e0["domain_hint"] == "biology"
    assert e0["felt_state"] == agg_felt(_cluster().members)


def test_felt_state_json_persisted(bridge):
    fb, conn = bridge
    cp = _make_pass(fb, emit=lambda _p: None)
    cp._maybe_decompose(_cv(), _proposal(), _cluster())
    fj = conn.execute(
        "SELECT felt_state_json FROM engram_felt_candidates LIMIT 1").fetchone()[0]
    assert json.loads(fj) == agg_felt(_cluster().members)


def test_grounded_object_skipped_no_candidate(bridge):
    # A grounded Object is skipped (no candidate, no emit) — and this is where a
    # mismatch→frame_dependent WOULD be detected, but cannot be (label-only
    # grounded-set; deferred). Proves grounded → no synthesis write for it.
    fb, conn = bridge
    fb.record_grounded("glacier")  # now grounded
    emitted = []
    cp = _make_pass(fb, emit=emitted.append)
    cp._maybe_decompose(_cv(), _proposal(), _cluster())
    rows = conn.execute(
        "SELECT object_label FROM engram_felt_candidates").fetchall()
    assert rows == [("microbe",)]                      # glacier skipped
    assert [e["object_label"] for e in emitted] == ["microbe"]


def test_emit_fires_once_not_every_dream(bridge):
    fb, _ = bridge
    emitted = []
    cp = _make_pass(fb, emit=emitted.append)
    cp._maybe_decompose(_cv(), _proposal(), _cluster())   # 2 fresh → 2 emits
    cp._maybe_decompose(_cv(), _proposal(), _cluster())   # all exist → 0 emits
    assert len(emitted) == 2


def test_no_emit_fn_still_queues(bridge):
    fb, conn = bridge
    cp = _make_pass(fb, emit=None)   # durable rows still written, no live event
    cp._maybe_decompose(_cv(), _proposal(), _cluster())
    n = conn.execute(
        "SELECT count(*) FROM engram_felt_candidates").fetchone()[0]
    assert n == 2
