"""Phase 6 — OracleGate tests (§P6.A — INV-Syn-13 metabolic gate).

Covers `titan_hcl/synthesis/oracle_gate.py` against SPEC §25.1 INV-Syn-13:

    free oracles (cost_class="free"): always admit, no spend bookkeeping
    metered oracles: admit IFF
        admit_score = importance × (balance/baseline) ≥ admit_threshold
        AND  remaining_daily_sol ≥ claim.cost_estimate_sol
    gate-denied → GateDecision(admit=False, reason ∈ {metabolic_gate_denied,
                                                       daily_budget_exhausted})

Also covers:
- build_gate_config defaults + numeric-coercion safety
- zk_privacy_domains pulls from [synthesis.oracle.zk] (INV-Syn-14)
- DDL helper is idempotent
- Pathological config (baseline=0) falls back to module default
- gate-denied verdicts still carry admit_score + latency_ms for auditability
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import pytest

from titan_hcl.synthesis.oracle_gate import (
    DEFAULT_ADMIT_THRESHOLD,
    DEFAULT_BALANCE_SOL_BASELINE,
    DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL,
    DENY_REASON_BUDGET,
    DENY_REASON_THRESHOLD,
    GateDecision,
    OracleGate,
    OracleGateConfig,
    build_gate_config,
    ensure_oracle_daily_spend_table,
    zk_privacy_domains,
)


# ─────────────────────────────────────────────────────────────────────────
# Lightweight fakes (the real plugs land in P6.B–E; this test stays at
# the gate's pure-functional surface).
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeClaim:
    domain: str
    payload: dict[str, Any]
    importance: float = 0.5


@dataclass
class _FakeOracle:
    oracle_id: str
    cost_class: str  # "free" | "metered"


# ─────────────────────────────────────────────────────────────────────────
# build_gate_config
# ─────────────────────────────────────────────────────────────────────────


def test_build_gate_config_defaults_when_block_missing():
    cfg = build_gate_config({})
    assert cfg.balance_sol_baseline == DEFAULT_BALANCE_SOL_BASELINE
    assert cfg.admit_threshold == DEFAULT_ADMIT_THRESHOLD
    assert cfg.default_daily_sol_budget == DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL
    assert cfg.daily_sol_budget == {}


def test_build_gate_config_reads_titan_params_block():
    cfg = build_gate_config(
        {
            "synthesis": {
                "oracle": {
                    "balance_sol_baseline": 2.5,
                    "admit_threshold": 0.42,
                    "daily_sol_budget": {
                        "helius_rpc": 0.25,
                        "web_api": 0.05,
                    },
                }
            }
        }
    )
    assert cfg.balance_sol_baseline == 2.5
    assert cfg.admit_threshold == 0.42
    assert cfg.daily_budget_for("helius_rpc") == 0.25
    assert cfg.daily_budget_for("web_api") == 0.05
    # Unknown oracle → fall back to module default.
    assert cfg.daily_budget_for("nonexistent_oracle") == DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL


def test_build_gate_config_drops_non_numeric_daily_budget_entries():
    """A typo'd string entry must not crash float() at admit() time."""
    cfg = build_gate_config(
        {
            "synthesis": {
                "oracle": {
                    "daily_sol_budget": {
                        "helius_rpc": 0.1,
                        "bad_entry": "not-a-number",  # silently dropped
                    }
                }
            }
        }
    )
    assert cfg.daily_budget_for("helius_rpc") == 0.1
    assert cfg.daily_budget_for("bad_entry") == DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL


def test_build_gate_config_none_input_returns_defaults():
    cfg = build_gate_config(None)
    assert cfg.balance_sol_baseline == DEFAULT_BALANCE_SOL_BASELINE
    assert cfg.admit_threshold == DEFAULT_ADMIT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────
# zk_privacy_domains (INV-Syn-14 — privacy whitelist surface)
# ─────────────────────────────────────────────────────────────────────────


def test_zk_privacy_domains_default_empty():
    assert zk_privacy_domains({}) == frozenset()
    assert zk_privacy_domains(None) == frozenset()


def test_zk_privacy_domains_reads_block():
    s = zk_privacy_domains(
        {
            "synthesis": {
                "oracle": {
                    "zk": {
                        "privacy_domains": [
                            "private_user_data",
                            "user_pii",
                            "private_transaction",
                        ]
                    }
                }
            }
        }
    )
    assert s == frozenset(
        {"private_user_data", "user_pii", "private_transaction"}
    )


# ─────────────────────────────────────────────────────────────────────────
# OracleGate.admit — INV-Syn-13
# ─────────────────────────────────────────────────────────────────────────


def _default_gate() -> OracleGate:
    return OracleGate(build_gate_config({}))


def test_free_oracle_always_admits_no_matter_what():
    """INV-Syn-13: free oracles bypass the gate; balance/budget irrelevant."""
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(domain="code_correctness", payload={}, importance=0.0),
        _FakeOracle(oracle_id="coding_sandbox", cost_class="free"),
        balance_sol=0.0,                 # broke!
        remaining_daily_sol=0.0,         # exhausted!
    )
    assert decision.admit is True
    assert decision.reason == ""
    assert decision.admit_score == 1.0
    assert decision.latency_ms >= 0


def test_metered_oracle_admits_when_score_above_threshold_and_budget_available():
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(domain="solana_tx_confirmed", payload={}, importance=0.5),
        _FakeOracle(oracle_id="helius_rpc", cost_class="metered"),
        balance_sol=1.0,                 # = baseline → admit_score = 0.5
        remaining_daily_sol=0.1,
    )
    assert decision.admit is True
    assert decision.reason == ""
    assert pytest.approx(decision.admit_score, abs=1e-9) == 0.5


def test_metered_oracle_denies_when_score_below_threshold():
    """admit_score = 0.05 × 1.0 = 0.05 < 0.15 → deny with metabolic_gate_denied."""
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(domain="solana_tx_confirmed", payload={}, importance=0.05),
        _FakeOracle(oracle_id="helius_rpc", cost_class="metered"),
        balance_sol=1.0,
        remaining_daily_sol=0.1,
    )
    assert decision.admit is False
    assert decision.reason == DENY_REASON_THRESHOLD
    assert decision.admit_score < DEFAULT_ADMIT_THRESHOLD


def test_metered_oracle_denies_when_daily_budget_exhausted():
    """admit_score passes but daily_remaining=0 → daily_budget_exhausted."""
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(domain="web_fact", payload={}, importance=0.9),
        _FakeOracle(oracle_id="web_api", cost_class="metered"),
        balance_sol=1.0,
        remaining_daily_sol=0.0,
    )
    assert decision.admit is False
    assert decision.reason == DENY_REASON_BUDGET


def test_metered_oracle_denies_when_remaining_below_cost_estimate():
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(
            domain="web_fact",
            payload={"cost_estimate_sol": 0.05},
            importance=0.9,
        ),
        _FakeOracle(oracle_id="web_api", cost_class="metered"),
        balance_sol=1.0,
        remaining_daily_sol=0.01,  # less than cost_estimate_sol
    )
    assert decision.admit is False
    assert decision.reason == DENY_REASON_BUDGET


def test_balance_below_baseline_tightens_gate_linearly():
    """At balance = baseline/4, admit_score scales 1/4 — high-importance claims still get through."""
    gate = _default_gate()
    # importance=0.9, balance=0.25*baseline → admit_score = 0.225 (above 0.15 threshold)
    decision_pass = gate.admit(
        _FakeClaim(domain="solana_tx_confirmed", payload={}, importance=0.9),
        _FakeOracle(oracle_id="helius_rpc", cost_class="metered"),
        balance_sol=0.25,
        remaining_daily_sol=0.1,
    )
    assert decision_pass.admit is True

    # Same balance but importance=0.3 → admit_score = 0.075 < 0.15 → deny
    decision_deny = gate.admit(
        _FakeClaim(domain="solana_tx_confirmed", payload={}, importance=0.3),
        _FakeOracle(oracle_id="helius_rpc", cost_class="metered"),
        balance_sol=0.25,
        remaining_daily_sol=0.1,
    )
    assert decision_deny.admit is False
    assert decision_deny.reason == DENY_REASON_THRESHOLD


def test_pathological_zero_baseline_falls_back_to_default():
    """Maker typo `balance_sol_baseline = 0` must not crash or auto-admit everything."""
    gate = OracleGate(
        OracleGateConfig(
            balance_sol_baseline=0.0,
            admit_threshold=0.15,
            default_daily_sol_budget=0.1,
            daily_sol_budget={},
        )
    )
    decision = gate.admit(
        _FakeClaim(domain="web_fact", payload={}, importance=0.5),
        _FakeOracle(oracle_id="web_api", cost_class="metered"),
        balance_sol=1.0,
        remaining_daily_sol=0.1,
    )
    # With fallback baseline=1.0: admit_score = 0.5 × 1.0 = 0.5 > 0.15 → admit
    assert decision.admit is True


def test_denied_verdict_still_carries_admit_score_and_latency():
    """INV-Syn-13: gate-denied claims must still be auditable on-chain."""
    gate = _default_gate()
    decision = gate.admit(
        _FakeClaim(domain="solana_tx_confirmed", payload={}, importance=0.0),
        _FakeOracle(oracle_id="helius_rpc", cost_class="metered"),
        balance_sol=1.0,
        remaining_daily_sol=0.1,
    )
    assert decision.admit is False
    assert decision.admit_score == 0.0  # importance was 0
    assert decision.latency_ms >= 0     # latency captured even on deny


# ─────────────────────────────────────────────────────────────────────────
# DuckDB DDL helper
# ─────────────────────────────────────────────────────────────────────────


def test_ensure_oracle_daily_spend_table_creates_idempotently(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    try:
        # First call creates.
        ensure_oracle_daily_spend_table(conn)
        cols = conn.execute(
            "DESCRIBE oracle_daily_spend"
        ).fetchall()
        col_names = {row[0] for row in cols}
        assert col_names >= {
            "oracle_id",
            "spend_date",
            "spent_sol",
            "n_calls",
            "updated_at",
        }

        # Second call is a no-op (CREATE TABLE IF NOT EXISTS).
        ensure_oracle_daily_spend_table(conn)
        # Insert + read survives the idempotent call.
        conn.execute(
            "INSERT INTO oracle_daily_spend (oracle_id, spend_date, spent_sol, n_calls) "
            "VALUES ('helius_rpc', '2026-05-27', 0.03, 5)"
        )
        row = conn.execute(
            "SELECT oracle_id, spent_sol, n_calls FROM oracle_daily_spend"
        ).fetchone()
        assert row == ("helius_rpc", 0.03, 5)
    finally:
        conn.close()


def test_ensure_oracle_daily_spend_primary_key_prevents_duplicates(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    try:
        ensure_oracle_daily_spend_table(conn)
        conn.execute(
            "INSERT INTO oracle_daily_spend (oracle_id, spend_date, spent_sol, n_calls) "
            "VALUES ('helius_rpc', '2026-05-27', 0.03, 5)"
        )
        # Same (oracle_id, spend_date) is the primary key; duplicate raises.
        with pytest.raises(duckdb.ConstraintException):
            conn.execute(
                "INSERT INTO oracle_daily_spend (oracle_id, spend_date, spent_sol, n_calls) "
                "VALUES ('helius_rpc', '2026-05-27', 0.05, 7)"
            )
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
# GateDecision shape
# ─────────────────────────────────────────────────────────────────────────


def test_gate_decision_is_a_dataclass_with_known_fields():
    """Pin the shape — downstream OracleRouter consumes these fields verbatim."""
    d = GateDecision(admit=True, reason="", admit_score=0.5, latency_ms=1)
    assert d.admit is True
    assert d.reason == ""
    assert d.admit_score == 0.5
    assert d.latency_ms == 1
