"""Phase C — true felt-vector frame_dependent (RFP_cgn_felt_state_exposure §7.C).

The producer (`consolidation._queue_felt_gaps`) no longer blindly skips a grounded
Object. It reads CGN's per-concept felt centroid (event-sourced via felt_bridge,
G18 — no RPC) and compares it to the cluster's lived felt:

  • ungrounded                       → status="candidate"  (unchanged)
  • grounded + divergent (>thresh)   → status="frame_dependent"  (a NEW felt frame)
  • grounded + aligned (≤thresh)     → skip  (genuinely redundant)
  • grounded + no centroid yet       → skip-safe  (nothing to compare)

Still propose-only — no CGN handle in the path (INV-Syn-ENG-4).

Run: python -m pytest tests/test_felt_bridge_frame_dependent_conflict_20260605.py -v -p no:anchorpy
"""
from __future__ import annotations

import types

import duckdb
import pytest

from titan_hcl.synthesis.felt_bridge import FeltBridge
from titan_hcl.synthesis.consolidation import (
    ConsolidationPass, Cluster, LLMProposal, TxCandidate, agg_felt,
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


def _make_pass(fb, emit=None, frame_divergence=0.15):
    # decompose → exactly ["glacier", "microbe"]; we ground "glacier" per-test.
    return ConsolidationPass(
        engram_store=object(),
        cgn_bridge=object(),
        outer_memory_writer=object(),
        mine_recent_txs_fn=lambda **_k: [],
        llm_propose_fn=lambda _c: LLMProposal(action="reject"),
        decompose_fn=lambda _n, _s: ["glacier", "microbe"],
        felt_bridge=fb,
        emit_candidate_fn=emit,
        frame_divergence=frame_divergence,
    )


def _cv(concept_id="glacier_x", version=2):
    return types.SimpleNamespace(concept_id=concept_id, version=version)


def _proposal():
    return LLMProposal(action="new_concept", concept_id="glacier_x",
                       proposed_name="Glacier Microbes", domain_hint="biology")


def _cluster():
    # Lived felt agg_felt(members) == {"DA": 0.80, "NE": 0.75} (high arousal/novelty).
    return Cluster(members=[
        TxCandidate("t1", "declarative", (), None, "thrilling microbe discovery",
                    felt='{"DA": 0.80, "NE": 0.75}'),
        TxCandidate("t2", "declarative", (), None, "more of the same",
                    felt='{"DA": 0.80, "NE": 0.75}'),
    ])


def _rows(conn):
    return conn.execute(
        "SELECT object_label, status, provenance FROM engram_felt_candidates "
        "ORDER BY object_label").fetchall()


# ── lived felt sanity ───────────────────────────────────────────────────────
def test_agg_felt_is_expected():
    assert agg_felt(_cluster().members) == {"DA": 0.80, "NE": 0.75}


# ── grounded + divergent → frame_dependent ──────────────────────────────────
def test_grounded_divergent_queues_frame_dependent(bridge):
    fb, conn = bridge
    # grounded under a CALM centroid → far from the THRILLING lived felt.
    fb.record_grounded("glacier", felt_centroid={"DA": 0.55, "NE": 0.45})
    emitted = []
    _make_pass(fb, emit=emitted.append)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    # glacier → frame_dependent (divergent); microbe → candidate (ungrounded).
    assert _rows(conn) == [
        ("glacier", "frame_dependent", "engram_felt_frame"),
        ("microbe", "candidate", "engram_felt_gap"),
    ]
    by = {e["object_label"]: e for e in emitted}
    assert by["glacier"]["status"] == "frame_dependent"
    assert by["microbe"]["status"] == "candidate"
    # propose-only: the lived felt is carried, base grounding untouched (no CGN handle).
    assert by["glacier"]["felt_state"] == {"DA": 0.80, "NE": 0.75}


# ── grounded + aligned → skip ───────────────────────────────────────────────
def test_grounded_aligned_is_skipped(bridge):
    fb, conn = bridge
    # grounded under (nearly) the SAME felt → redundant → skip.
    fb.record_grounded("glacier", felt_centroid={"DA": 0.80, "NE": 0.75})
    emitted = []
    _make_pass(fb, emit=emitted.append)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    assert _rows(conn) == [("microbe", "candidate", "engram_felt_gap")]
    assert [e["object_label"] for e in emitted] == ["microbe"]


def test_grounded_minor_jitter_still_aligned(bridge):
    fb, conn = bridge
    # ±0.05 jitter → RMS ≈ 0.05 < 0.15 → still redundant.
    fb.record_grounded("glacier", felt_centroid={"DA": 0.78, "NE": 0.72})
    _make_pass(fb, emit=lambda _p: None)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    assert _rows(conn) == [("microbe", "candidate", "engram_felt_gap")]


# ── grounded + no centroid → skip-safe ──────────────────────────────────────
def test_grounded_without_centroid_skip_safe(bridge):
    fb, conn = bridge
    fb.record_grounded("glacier")  # label-only, no centroid
    _make_pass(fb, emit=lambda _p: None)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    assert _rows(conn) == [("microbe", "candidate", "engram_felt_gap")]


# ── threshold is tunable (emergence over a hardcode) ────────────────────────
def test_frame_divergence_threshold_tunable(bridge):
    fb, conn = bridge
    fb.record_grounded("glacier", felt_centroid={"DA": 0.55, "NE": 0.45})  # ~0.276 apart
    # A high threshold suppresses the frame (treats the divergence as still-redundant).
    _make_pass(fb, emit=lambda _p: None, frame_divergence=0.99)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    assert _rows(conn) == [("microbe", "candidate", "engram_felt_gap")]  # glacier skipped


def test_frame_divergence_zero_treats_any_gap_as_frame(bridge):
    fb, conn = bridge
    fb.record_grounded("glacier", felt_centroid={"DA": 0.55, "NE": 0.45})
    _make_pass(fb, emit=lambda _p: None, frame_divergence=0.0)._maybe_decompose(
        _cv(), _proposal(), _cluster())
    assert ("glacier", "frame_dependent", "engram_felt_frame") in _rows(conn)


# ── idempotent: the frame_dependent emit fires once, not every dream ────────
def test_frame_dependent_emits_once(bridge):
    fb, _ = bridge
    fb.record_grounded("glacier", felt_centroid={"DA": 0.55, "NE": 0.45})
    emitted = []
    cp = _make_pass(fb, emit=emitted.append)
    cp._maybe_decompose(_cv(), _proposal(), _cluster())  # glacier+microbe fresh → 2
    cp._maybe_decompose(_cv(), _proposal(), _cluster())  # all exist → 0
    assert len(emitted) == 2
    assert {e["object_label"] for e in emitted} == {"glacier", "microbe"}
