"""Phase 1 plug-protocol contract tests — D-SPEC-123 (SPEC v1.56.0 §25).

Validates the 5 Protocol shapes via runtime_checkable isinstance + a minimal
fake plug per protocol that satisfies the contract.
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

from titan_hcl.synthesis.plugs import (
    ConceptRef,
    FeltContext,
    Grounding,
    MeaningOraclePlug,
    MeaningStrand,
    OracleClaim,
    OracleVerdict,
    Proof,
    ProofStrategyPlug,
    Record,
    SubstrateHealth,
    SubstratePlug,
    SubstrateQuery,
    ToolCall,
    ToolPlug,
    ToolResult,
    TruthOraclePlug,
    WriteResult,
)


# ─────────────────────────────────────────────────────────────────────────
# Substrate plug
# ─────────────────────────────────────────────────────────────────────────

class _FakeSubstrate:
    name = "duckdb"

    def read(self, q: SubstrateQuery) -> list[Record]:
        if q.op in ("search", "fork_read", "diff", "cross_ref"):
            return [Record(kind="unsupported_op", data={"op": q.op})]
        return [Record(kind="row", data={"x": 1})]

    def write(self, recs: list[Record]) -> WriteResult:
        return WriteResult(accepted=len(recs), rejected=0)

    def health(self) -> SubstrateHealth:
        return SubstrateHealth(name="duckdb", last_consistent_event_ts=time.time(), lag_s=0.0)


def test_substrate_plug_runtime_checkable():
    plug = _FakeSubstrate()
    assert isinstance(plug, SubstratePlug)


def test_substrate_unsupported_op_for_phase2_sc_ops():
    # Phase 1 plugs return Record(kind="unsupported_op") for SC ops that
    # land in Phase 2 — proves the interface accepts the op + the Phase-2
    # gate is operative.
    plug = _FakeSubstrate()
    for op in ("search", "fork_read", "diff", "cross_ref"):
        out = plug.read(SubstrateQuery(op=op))
        assert len(out) == 1
        assert out[0].kind == "unsupported_op"
        assert out[0].data["op"] == op


def test_substrate_query_defaults():
    q = SubstrateQuery(op="get")
    assert q.args == {}
    assert q.watermark_ts is None
    assert q.limit == 50


def test_record_shapes():
    r = Record(kind="row", data={"id": 7})
    assert r.kind == "row"
    assert r.data == {"id": 7}

    r2 = Record(kind="stale")
    assert r2.data == {}


def test_write_result_default_rejected_zero():
    wr = WriteResult(accepted=3)
    assert wr.accepted == 3
    assert wr.rejected == 0
    assert wr.evidence_ref is None


def test_substrate_health_healthy_default_true():
    h = SubstrateHealth(name="kuzu", last_consistent_event_ts=0.0, lag_s=0.0)
    assert h.healthy is True
    assert h.error is None


# ─────────────────────────────────────────────────────────────────────────
# Truth oracle
# ─────────────────────────────────────────────────────────────────────────

class _FakeXOracle:
    oracle_id = "x_oracle"
    cost_class = "metered"

    def can_handle(self, domain: str) -> bool:
        return domain in ("x_event_real", "x_topic_trending")

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        return OracleVerdict(
            oracle_id=self.oracle_id,
            verdict="true",
            evidence_ref=f"tweet:{claim.payload.get('tweet_id')}",
            cost=0.0001,
            latency_ms=120,
            ts=time.time(),
        )


def test_truth_oracle_plug_runtime_checkable():
    plug = _FakeXOracle()
    assert isinstance(plug, TruthOraclePlug)


def test_truth_oracle_x_dual_surface_supported_by_protocol():
    # X-oracle expansion per D-SPEC-123: X serves as TruthOraclePlug for
    # current-events verification (second-source companion to web).
    plug = _FakeXOracle()
    assert plug.can_handle("x_event_real")
    assert not plug.can_handle("solana_tx")
    verdict = plug.verify(OracleClaim(domain="x_event_real",
                                      payload={"tweet_id": "12345"}))
    assert verdict.verdict == "true"
    assert verdict.evidence_ref == "tweet:12345"
    assert verdict.oracle_id == "x_oracle"


# ─────────────────────────────────────────────────────────────────────────
# Meaning oracle (CGN — sole grounding authority per INV-Syn-1)
# ─────────────────────────────────────────────────────────────────────────

class _FakeCGN:
    def meaning_of(self, concept: ConceptRef) -> MeaningStrand:
        return MeaningStrand(concept=concept)

    def ground(self, concept: ConceptRef, felt: FeltContext) -> Grounding:
        return Grounding(concept=concept, grounding_id=f"g:{concept.concept_id}",
                         strength=0.8, ts=time.time())


def test_meaning_oracle_plug_runtime_checkable():
    plug = _FakeCGN()
    assert isinstance(plug, MeaningOraclePlug)


def test_concept_ref_default_version_zero():
    c = ConceptRef(concept_id="metaplex_nft_minting")
    assert c.version == 0


def test_meaning_strand_four_anchor_lists():
    strand = MeaningStrand(concept=ConceptRef(concept_id="x"))
    assert strand.declarative_anchors == []
    assert strand.procedural_anchors == []
    assert strand.episodic_anchors == []
    assert strand.felt_anchors == []


# ─────────────────────────────────────────────────────────────────────────
# Proof strategy
# ─────────────────────────────────────────────────────────────────────────

class _FakeMerkle:
    strategy = "merkle"

    def commit(self, payload: bytes) -> Proof:
        import hashlib
        return Proof(strategy="merkle",
                     commitment=hashlib.sha256(payload).digest(),
                     cost=0.0)

    def verify(self, proof: Proof, payload: Optional[bytes] = None) -> bool:
        import hashlib
        if payload is None:
            return False
        return hashlib.sha256(payload).digest() == proof.commitment


def test_proof_strategy_plug_runtime_checkable():
    plug = _FakeMerkle()
    assert isinstance(plug, ProofStrategyPlug)


def test_proof_merkle_commit_verify_roundtrip():
    plug = _FakeMerkle()
    proof = plug.commit(b"hello")
    assert plug.verify(proof, b"hello") is True
    assert plug.verify(proof, b"goodbye") is False


# ─────────────────────────────────────────────────────────────────────────
# Tool (with dual surface — x_research as ToolPlug whose .oracle() returns
# the x_oracle TruthOraclePlug; per D-SPEC-123 X serves both surfaces).
# ─────────────────────────────────────────────────────────────────────────

class _FakeXResearch:
    tool_id = "x_research"

    def __init__(self) -> None:
        self._x_oracle = _FakeXOracle()

    def capabilities(self) -> list[str]:
        return ["post", "fetch_thread", "fetch_topic", "fetch_account"]

    def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(tool_id=self.tool_id, success=True,
                          result_summary=f"fetched {call.args.get('topic')}")

    def oracle(self) -> Optional[TruthOraclePlug]:
        return self._x_oracle


def test_tool_plug_runtime_checkable():
    plug = _FakeXResearch()
    assert isinstance(plug, ToolPlug)


def test_x_research_tool_dual_surface_returns_x_oracle():
    plug = _FakeXResearch()
    assert "fetch_topic" in plug.capabilities()
    result = plug.invoke(ToolCall(tool_id="x_research",
                                  args={"topic": "phase-c"}))
    assert result.success
    assert result.result_summary == "fetched phase-c"
    oracle = plug.oracle()
    assert oracle is not None
    assert isinstance(oracle, TruthOraclePlug)
    assert oracle.oracle_id == "x_oracle"


def test_tool_call_optional_links_default_none():
    call = ToolCall(tool_id="t", args={})
    assert call.parent_chat_tx is None
    assert call.parent_goal is None
    assert call.parent_skill_id is None


# ─────────────────────────────────────────────────────────────────────────
# Public export surface
# ─────────────────────────────────────────────────────────────────────────

def test_all_protocols_exported():
    from titan_hcl.synthesis import plugs as P
    for name in (
        "SubstratePlug", "TruthOraclePlug", "MeaningOraclePlug",
        "ProofStrategyPlug", "ToolPlug",
    ):
        assert name in P.__all__
        assert hasattr(P, name)
