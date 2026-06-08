"""EngineRecall — Phase 2 (PLAN_synthesis_engine_Phase2.md §2D) end-to-end tests.

Drives the contract-driven recall pipeline through the real
RuleEvaluator + Phase 2 SC ops + actr_episodic_recall_helper JSON +
composite_score, with mocked embedder + faiss_reader + index.db.

Validates:
  - Happy path: text → embed → contract eval → rank → top-K dicts.
  - Fallback paths: no embedder, no contract, all-empty sources,
    chi-budget exhausted, contract-eval error.
  - Composite weights from the contract override defaults.
  - Candidate merge: dedup by tx_hash across $base/$semantic/$threaded.
  - Cold-start activation handling.
  - k_override caps the contract's `limit`.
  - Hot-reload via mtime cache invalidation.

PLAN_synthesis_engine_Phase2.md §2D acceptance gate.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest

from titan_hcl.logic.timechain_v2 import FORK_IDS, RuleEvaluator
from titan_hcl.synthesis.recall import (
    DEFAULT_CANDIDATES_FROM,
    DEFAULT_K,
    EngineRecall,
    HELPER_CONTRACT_ID,
    RecallResult,
)


# Real helper contract (the JSON shipped in 2C).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER_CONTRACT_PATH = os.path.join(
    REPO_ROOT, "titan_hcl", "contracts", "meta_cognitive",
    "actr_episodic_recall_helper.json")


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

class _StubFaiss:
    def __init__(self, results=None):
        self._results = results or []

    def knn(self, fork, vec, k, min_similarity):
        return list(self._results)


def _make_conv_index_db(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE block_index ("
        "block_hash BLOB PRIMARY KEY, fork_id INTEGER NOT NULL, "
        "block_height INTEGER NOT NULL, timestamp REAL NOT NULL, "
        "epoch_id INTEGER NOT NULL, thought_type TEXT, source TEXT, "
        "significance REAL, chi_spent REAL, neuromod_da REAL, "
        "neuromod_ach REAL, neuromod_ne REAL, tags TEXT, cross_refs TEXT, "
        "db_ref TEXT, compacted INTEGER DEFAULT 0, file_offset INTEGER NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO block_index VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return conn


def _conv_row(byte: int, epoch: int) -> tuple:
    return (
        bytes([byte]) * 32, FORK_IDS["conversation"], byte, 0.0, epoch,
        "chat_turn", "chat", 0.5, 0.001, 0, 0, 0,
        "chat,user:hash1", "", "", 0, byte * 100,
    )


def _build_engine(
    faiss_results=None, db_rows=None,
    activation_map=None, embedder_vec=None,
    contracts_dir=None,
) -> tuple[EngineRecall, sqlite3.Connection]:
    """Build an EngineRecall wired with stubs. Returns (engine, conn) so
    tests can close the conn in tearDown."""
    conn = _make_conv_index_db(db_rows if db_rows is not None else [])
    evaluator = RuleEvaluator(
        faiss_reader=_StubFaiss(faiss_results or []),
        index_db=conn,
    )
    activations = activation_map or {}
    er = EngineRecall(
        rule_evaluator=evaluator,
        activation_lookup=lambda ids: {
            i: activations[i] for i in ids if i in activations},
        embedder=(lambda t: list(embedder_vec)) if embedder_vec else None,
        contracts_dir=contracts_dir,
    )
    return er, conn


# ─────────────────────────────────────────────────────────────────────────
# Fallback paths
# ─────────────────────────────────────────────────────────────────────────

class TestFallbacks(unittest.TestCase):

    def test_no_embedder_returns_none(self) -> None:
        er, conn = _build_engine(embedder_vec=None)
        try:
            assert er.recall("hello") is None
            assert er.get_stats()["total_fallbacks"] == 1
        finally:
            conn.close()

    def test_no_helper_contract_returns_none(self) -> None:
        # Point at an empty tmp dir → no helper contract present.
        with tempfile.TemporaryDirectory() as tmp:
            er, conn = _build_engine(
                embedder_vec=[0.1] * 8, contracts_dir=tmp)
            try:
                assert er.recall("hello") is None
                assert er.get_stats()["total_fallbacks"] == 1
            finally:
                conn.close()

    def test_all_sources_empty_returns_none(self) -> None:
        # No db rows, no faiss results, no cross-ref hits → OR-gate
        # falls through → action=None → engine returns None.
        er, conn = _build_engine(embedder_vec=[0.1] * 8)
        try:
            assert er.recall("hello") is None
            stats = er.get_stats()
            assert stats["total_fallbacks"] == 1
            assert stats["total_contract_hits"] == 0
        finally:
            conn.close()

    def test_inactive_contract_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Copy the real helper contract but flip status to "disabled".
            with open(HELPER_CONTRACT_PATH) as f:
                c = json.load(f)
            c["status"] = "disabled"
            with open(os.path.join(tmp, "actr_episodic_recall_helper.json"), "w") as f:
                json.dump(c, f)
            er, conn = _build_engine(
                embedder_vec=[0.1] * 8, contracts_dir=tmp,
                db_rows=[_conv_row(1, 100)])
            try:
                assert er.recall("hello") is None
            finally:
                conn.close()

    def test_embedder_exception_returns_none(self) -> None:
        evaluator = RuleEvaluator()

        def _bad_embedder(_text):
            raise RuntimeError("boom")

        er = EngineRecall(
            rule_evaluator=evaluator,
            activation_lookup=lambda ids: {},
            embedder=_bad_embedder,
        )
        assert er.recall("anything") is None


# ─────────────────────────────────────────────────────────────────────────
# Happy path: FAISS-only result set
# ─────────────────────────────────────────────────────────────────────────

class TestRecallHappyPath(unittest.TestCase):

    def test_returns_ranked_results_from_faiss_hits(self) -> None:
        faiss_results = [
            {"tx_hash": "sem1", "score": 0.95, "fork": "conversation"},
            {"tx_hash": "sem2", "score": 0.85, "fork": "conversation"},
            {"tx_hash": "sem3", "score": 0.75, "fork": "conversation"},
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8, faiss_results=faiss_results)
        try:
            result = er.recall("show me recent conversations")
            assert result is not None
            assert len(result) == 3
            assert all(isinstance(r, RecallResult) for r in result)
            tx_hashes = {r.tx_hash for r in result}
            assert tx_hashes == {"sem1", "sem2", "sem3"}
        finally:
            conn.close()

    def test_results_sorted_by_descending_score(self) -> None:
        faiss_results = [
            {"tx_hash": "lo", "score": 0.2, "fork": "conversation"},
            {"tx_hash": "hi", "score": 0.99, "fork": "conversation"},
            {"tx_hash": "mid", "score": 0.5, "fork": "conversation"},
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8, faiss_results=faiss_results)
        try:
            result = er.recall("query")
            # All have equal activation (cold-start) so cosine drives ranking.
            assert result[0].tx_hash == "hi"
            assert result[-1].tx_hash == "lo"
        finally:
            conn.close()

    def test_dedup_across_sources(self) -> None:
        """When the same tx_hash appears in BOTH $base (FORK_READ) and
        $semantic (FAISS), the merged candidate list dedups to one entry —
        the first-seen wins (iteration order of candidates_from).
        """
        # block_hash for byte 1 = bytes([1]) * 32 → hex "01" * 32 = 64 chars.
        shared_hex = (bytes([1]) * 32).hex()
        faiss_results = [
            {"tx_hash": shared_hex, "score": 0.9, "fork": "conversation"},
            {"tx_hash": "sem2", "score": 0.8, "fork": "conversation"},
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8,
            faiss_results=faiss_results,
            db_rows=[_conv_row(1, 100)],   # same block_hash hex
        )
        try:
            result = er.recall("q")
            tx_hashes = {r.tx_hash for r in result}
            # Two unique tx_hashes: the shared block_hash (dedupped to
            # one entry) + sem2.
            assert tx_hashes == {shared_hex, "sem2"}
            assert len(result) == 2
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Contract weights override + activation lookup integration
# ─────────────────────────────────────────────────────────────────────────

class TestWeightsAndActivation(unittest.TestCase):

    def test_activation_lookup_consulted(self) -> None:
        """Recall results include base_level + norm_base_level from
        activation_lookup."""
        faiss_results = [
            {"tx_hash": "a", "score": 0.9, "fork": "conversation"},
            {"tx_hash": "b", "score": 0.5, "fork": "conversation"},
        ]
        activation_map = {"tc:a": 5.0, "tc:b": 2.0}
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8,
            faiss_results=faiss_results,
            activation_map=activation_map,
        )
        try:
            result = er.recall("q")
            by_hash = {r.tx_hash: r for r in result}
            # Higher activation → higher norm_base_level.
            assert by_hash["a"].base_level == 5.0
            assert by_hash["b"].base_level == 2.0
            # Activation contributes to composite — `a` should outrank `b`
            # despite tied cosine differences being small relative to
            # activation's z-score swing.
            assert by_hash["a"].score > by_hash["b"].score
        finally:
            conn.close()

    def test_cold_start_handled(self) -> None:
        """Items absent from activation_lookup get cold-start default and
        still appear in results (not dropped)."""
        faiss_results = [
            {"tx_hash": "cold", "score": 0.8, "fork": "conversation"},
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8,
            faiss_results=faiss_results,
            activation_map={},   # nothing — all cold
        )
        try:
            result = er.recall("q")
            assert len(result) == 1
            assert result[0].tx_hash == "cold"
            # Cold-start sentinel was substituted before z-scoring.
            import math
            assert math.isinf(result[0].base_level) or \
                   result[0].base_level == 0.5
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────
# k_override + contract `limit`
# ─────────────────────────────────────────────────────────────────────────

class TestKOverride(unittest.TestCase):

    def test_k_truncates_below_contract_limit(self) -> None:
        # Contract ships limit=8; we ask k=3.
        faiss_results = [
            {"tx_hash": f"s{i}", "score": 1.0 - i * 0.1,
             "fork": "conversation"} for i in range(10)
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8, faiss_results=faiss_results)
        try:
            result = er.recall("q", k=3)
            assert len(result) == 3
        finally:
            conn.close()

    def test_k_clamps_to_contract_limit(self) -> None:
        # Asking for more than contract.limit → bounded by contract.
        faiss_results = [
            {"tx_hash": f"s{i}", "score": 1.0 - i * 0.05,
             "fork": "conversation"} for i in range(15)
        ]
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8, faiss_results=faiss_results)
        try:
            result = er.recall("q", k=100)
            # Contract caps at 8; we got 8 (we have 15 candidates).
            assert len(result) == 8
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Chi-budget exhaustion fallback
# ─────────────────────────────────────────────────────────────────────────

class TestChiExhaustionFallback(unittest.TestCase):

    def test_chi_exhaustion_returns_none(self) -> None:
        # Build a RuleEvaluator with a chi_cap below the helper contract's
        # required cost (FORK_READ=0.001 + SEARCH=0.002 + CROSS_REF=0.002).
        # cap=0.001 → SEARCH overflows.
        from titan_hcl.synthesis.recall import EngineRecall as _ER
        conn = _make_conv_index_db([])
        evaluator = RuleEvaluator(
            faiss_reader=_StubFaiss([]),
            index_db=conn,
            chi_cap=0.0005,    # below even FORK_READ (0.001) — overflows on first SC op? actually:
        )
        # First SC op is FORK_READ (cost 0.001). 0.001 > 0.0005 cap → exhausted.
        er = _ER(
            rule_evaluator=evaluator,
            activation_lookup=lambda ids: {},
            embedder=lambda t: [0.1] * 8,
        )
        result = er.recall("q")
        conn.close()
        assert result is None


# ─────────────────────────────────────────────────────────────────────────
# Contract hot-reload (mtime cache)
# ─────────────────────────────────────────────────────────────────────────

class TestContractHotReload(unittest.TestCase):

    def test_contract_reloads_on_mtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Stage 1: ship a contract that requires $semantic.length >= 5
            # (so 3 results trigger fallback).
            with open(HELPER_CONTRACT_PATH) as f:
                c = json.load(f)
            high_threshold = json.loads(json.dumps(c))
            # Replace the OR rule to require >= 5 results.
            for clause in high_threshold["rules"][-1]["clauses"]:
                if "$semantic" in clause.get("field", ""):
                    clause["value"] = 5
            path = os.path.join(tmp, "actr_episodic_recall_helper.json")
            with open(path, "w") as f:
                json.dump(high_threshold, f)

            faiss_results = [
                {"tx_hash": f"s{i}", "score": 0.9, "fork": "conversation"}
                for i in range(3)
            ]
            er, conn = _build_engine(
                embedder_vec=[0.1] * 8,
                faiss_results=faiss_results,
                contracts_dir=tmp,
            )
            try:
                # 3 results, but ALL OR clauses require >= 5 → falls through.
                assert er.recall("q") is None

                # Bump mtime + lower threshold to >=1.
                time.sleep(0.05)
                lower = json.loads(json.dumps(c))   # original (threshold=1)
                with open(path, "w") as f:
                    json.dump(lower, f)
                os.utime(path, None)

                # Now recall succeeds — contract reloaded.
                result = er.recall("q")
                assert result is not None
                assert len(result) == 3
            finally:
                conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Stats surface
# ─────────────────────────────────────────────────────────────────────────

class TestStats(unittest.TestCase):

    def test_stats_count_hits_and_fallbacks(self) -> None:
        # Two calls: one hit, one fallback (no embedder swap mid-flight,
        # so simulate by calling with vs without faiss results).
        er, conn = _build_engine(
            embedder_vec=[0.1] * 8,
            faiss_results=[{"tx_hash": "a", "score": 0.9}],
        )
        try:
            assert er.recall("q1") is not None
            # Now create a second engine with no sources → fallback.
            er2, conn2 = _build_engine(embedder_vec=[0.1] * 8)
            try:
                assert er2.recall("q2") is None
                s1 = er.get_stats()
                s2 = er2.get_stats()
                assert s1["total_contract_hits"] == 1
                assert s1["total_fallbacks"] == 0
                assert s2["total_contract_hits"] == 0
                assert s2["total_fallbacks"] == 1
            finally:
                conn2.close()
        finally:
            conn.close()


class TestEmbedOnce(unittest.TestCase):
    """P4 embed-once (RFP_synthesis_decision_authority): a caller-supplied
    ``query_vec`` (the shared get_text_embedder() vector threaded from the agno
    PreHook) is reused verbatim — the injected embedder is NOT invoked (G9)."""

    def test_query_vec_works_without_an_embedder(self) -> None:
        # No embedder at all, but the shared vector is supplied → recall still
        # runs the SEARCH (the embed-once path does not require an embedder).
        faiss_results = [{"tx_hash": "v1", "score": 0.9, "fork": "conversation"}]
        er, conn = _build_engine(embedder_vec=None, faiss_results=faiss_results)
        try:
            result = er.recall("anything", query_vec=[0.1] * 8)
            assert result is not None
            assert {r.tx_hash for r in result} == {"v1"}
        finally:
            conn.close()

    def test_query_vec_short_circuits_the_embedder(self) -> None:
        # An embedder that would RAISE must never be called when query_vec is
        # supplied — proving the shared vector is used, not a fresh embed.
        calls: list = []

        def _boom(_t):
            calls.append(_t)
            raise RuntimeError("embedder must not be called under embed-once")

        faiss_results = [{"tx_hash": "v2", "score": 0.8, "fork": "conversation"}]
        conn = _make_conv_index_db([])
        evaluator = RuleEvaluator(
            faiss_reader=_StubFaiss(faiss_results), index_db=conn)
        er = EngineRecall(
            rule_evaluator=evaluator,
            activation_lookup=lambda ids: {},
            embedder=_boom,
        )
        try:
            result = er.recall("anything", query_vec=[0.2] * 8)
            assert result is not None
            assert {r.tx_hash for r in result} == {"v2"}
            assert calls == []   # embedder never invoked
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
