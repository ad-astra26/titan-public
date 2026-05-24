"""Phase 2 chi-budget compliance — B.5 acceptance gate (rFP §19).

Per-evaluate() chi accumulator + hard cap; on overflow, remaining rules
short-circuit and the evaluator returns the `chi_budget_exhausted` action.
Costs match titan_params.toml [synthesis.chi].

PLAN_synthesis_engine_Phase2.md 2A.4.
"""
from __future__ import annotations

import sqlite3
import unittest

from titan_hcl.logic.timechain_v2 import (
    RuleEvaluator,
    DEFAULT_CHI_COSTS,
    DEFAULT_CHI_CAP,
    FORK_IDS,
)


class _FaissStub:
    def knn(self, fork, vec, k, min_similarity):
        return []


def _empty_index_db() -> sqlite3.Connection:
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
    conn.commit()
    return conn


class TestChiCosts(unittest.TestCase):
    """Per-op cost matches DEFAULT_CHI_COSTS / titan_params.toml."""

    def setUp(self) -> None:
        self.conn = _empty_index_db()
        self.ev = RuleEvaluator(
            faiss_reader=_FaissStub(),
            index_db=self.conn,
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_search_op_spends_search_cost(self) -> None:
        rules = [{"op": "SEARCH", "fork": "conversation",
                  "query_embedding": [0.1] * 4, "store": "$h"}]
        self.ev.evaluate(rules, {})
        assert self.ev._chi_spent == DEFAULT_CHI_COSTS["search"]

    def test_fork_read_op_spends_fork_read_cost(self) -> None:
        rules = [{"op": "FORK_READ", "fork": "conversation", "store": "$r"}]
        self.ev.evaluate(rules, {})
        assert self.ev._chi_spent == DEFAULT_CHI_COSTS["fork_read"]

    def test_diff_op_spends_diff_cost(self) -> None:
        rules = [{"op": "DIFF", "field_a": "a", "field_b": "b",
                  "cmp": "GT", "value": 0, "then": {"action": "x"}}]
        self.ev.evaluate(rules, {"a": 5, "b": 1})
        assert self.ev._chi_spent == DEFAULT_CHI_COSTS["diff"]

    def test_cross_ref_op_spends_cross_ref_cost(self) -> None:
        rules = [{"op": "CROSS_REF", "tx_hash": "abc", "store": "$r"}]
        self.ev.evaluate(rules, {})
        assert self.ev._chi_spent == DEFAULT_CHI_COSTS["cross_ref"]

    def test_group_by_op_spends_group_by_cost(self) -> None:
        # Pre-seed via FORK_READ (cost = fork_read) then GROUP_BY (cost = group_by).
        rules = [
            {"op": "FORK_READ", "fork": "conversation", "store": "$rows"},
            {"op": "GROUP_BY", "source": "$rows", "group_by": "source",
             "store": "$groups"},
        ]
        self.ev.evaluate(rules, {})
        assert self.ev._chi_spent == (
            DEFAULT_CHI_COSTS["fork_read"] + DEFAULT_CHI_COSTS["group_by"]
        )

    def test_zero_cost_for_pure_cmp_rules(self) -> None:
        rules = [{"op": "IF", "field": "x", "cmp": "GT", "value": 0,
                  "then": {"action": "match"}}]
        self.ev.evaluate(rules, {"x": 5})
        assert self.ev._chi_spent == 0.0


class TestChiCap(unittest.TestCase):
    """Cap is enforced; overflow returns chi_budget_exhausted; counter resets."""

    def setUp(self) -> None:
        self.conn = _empty_index_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_chi_cap_enforced_returns_exhausted_action(self) -> None:
        # Tiny cap so a single SEARCH overflows.
        ev = RuleEvaluator(
            faiss_reader=_FaissStub(), index_db=self.conn,
            chi_cap=0.0001,  # below search cost (0.002)
        )
        rules = [
            {"op": "SEARCH", "fork": "conversation",
             "query_embedding": [0.1] * 4, "store": "$h"},
            {"op": "IF", "field": "always", "cmp": "EQ", "value": "yes",
             "then": {"action": "should_not_reach"}},
        ]
        result = ev.evaluate(rules, {"always": "yes"})
        assert result is not None
        assert result["action"] == "chi_budget_exhausted"
        assert result["cap"] == 0.0001
        assert result["spent"] > 0.0
        # Stats reflect the exhaustion.
        stats = ev.get_stats()
        assert stats["total_chi_exhausted"] == 1

    def test_chi_resets_per_evaluate_call(self) -> None:
        ev = RuleEvaluator(index_db=self.conn)
        rules = [{"op": "FORK_READ", "fork": "conversation", "store": "$r"}]
        ev.evaluate(rules, {})
        first_spent = ev._chi_spent
        assert first_spent > 0.0
        ev.evaluate(rules, {})
        # Reset on entry → ends at same single-call spend, not 2×.
        assert ev._chi_spent == first_spent
        # But total stat accumulates.
        assert ev.get_stats()["total_chi_spent"] == first_spent * 2

    def test_max_queries_cap_independent_of_chi_cap(self) -> None:
        # Issue MAX_QUERIES + 2 binding ops; only the first MAX_QUERIES execute.
        ev = RuleEvaluator(index_db=self.conn)
        many = [
            {"op": "FORK_READ", "fork": "conversation", "store": f"$r{i}"}
            for i in range(ev.MAX_QUERIES + 2)
        ]
        ev.evaluate(many, {})
        # Spent = MAX_QUERIES × fork_read cost (the rest are skipped by query cap).
        assert ev._chi_spent == ev.MAX_QUERIES * DEFAULT_CHI_COSTS["fork_read"]


class TestChiCostsOverride(unittest.TestCase):
    """Caller-supplied chi_costs override defaults (titan_params injection path)."""

    def test_custom_costs_applied(self) -> None:
        ev = RuleEvaluator(
            index_db=_empty_index_db(),
            chi_costs={"fork_read": 0.005},  # 5× default
        )
        rules = [{"op": "FORK_READ", "fork": "conversation", "store": "$r"}]
        ev.evaluate(rules, {})
        assert ev._chi_spent == 0.005

    def test_uppercase_op_name_normalized(self) -> None:
        ev = RuleEvaluator(
            index_db=_empty_index_db(),
            chi_costs={"FORK_READ": 0.005},
        )
        rules = [{"op": "FORK_READ", "fork": "conversation", "store": "$r"}]
        ev.evaluate(rules, {})
        assert ev._chi_spent == 0.005

    def test_default_cap_is_0_01(self) -> None:
        ev = RuleEvaluator()
        assert ev._chi_cap == DEFAULT_CHI_CAP == 0.01


if __name__ == "__main__":
    unittest.main()
