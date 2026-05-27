"""Phase 6 — CoverageAnalyzer tests (§P6.J; INV-Syn-15 / §A.6 gate).

Covers `titan_hcl/synthesis/oracle_coverage.py`:

- Empty chain → 0/0/0 with coverage_ratio=0
- All tool calls scored at write time via "oracle" → 100% coverage
- All tool calls unscored at write time but covered retrospectively
  via OracleVerdictBatch entries → 100% coverage
- Mix of write-time + retrospective oracle scoring counts each once
- Tool call referenced by batch entry with verdict="unknown" does NOT
  count as oracle-scored (verdict must be true/false to be evidence)
- Write-time "llm" classification counted in scored_by_llm
- Window filtering — only TXs since (now - window) considered
- Reader exception defenses
- report_dict shape matches what Observatory route returns
- a6_gate_passes flag flips at 0.95 boundary
"""
from __future__ import annotations

import time

import pytest

from titan_hcl.synthesis.oracle_coverage import (
    CoverageAnalyzer,
    CoverageReport,
)


# ─────────────────────────────────────────────────────────────────────────
# Empty chain
# ─────────────────────────────────────────────────────────────────────────


def test_empty_chain_returns_zero_coverage():
    a = CoverageAnalyzer()
    r = a.analyze()
    assert r.total_tool_call_txs == 0
    assert r.scored_by_oracle == 0
    assert r.scored_by_llm == 0
    assert r.unscored == 0
    assert r.coverage_ratio == 0.0


# ─────────────────────────────────────────────────────────────────────────
# Write-time scored_by
# ─────────────────────────────────────────────────────────────────────────


def test_all_tool_calls_scored_oracle_at_write_time():
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0},
        {"tx_hash": "tc2", "scored_by": "oracle", "ts": 2.0},
        {"tx_hash": "tc3", "scored_by": "oracle", "ts": 3.0},
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: [],
    )
    r = a.analyze()
    assert r.total_tool_call_txs == 3
    assert r.scored_by_oracle == 3
    assert r.scored_by_llm == 0
    assert r.unscored == 0
    assert r.coverage_ratio == 1.0


def test_all_unscored_returns_zero_coverage():
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": None, "ts": 1.0},
        {"tx_hash": "tc2", "scored_by": None, "ts": 2.0},
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: [],
    )
    r = a.analyze()
    assert r.total_tool_call_txs == 2
    assert r.scored_by_oracle == 0
    assert r.scored_by_llm == 0
    assert r.unscored == 2
    assert r.coverage_ratio == 0.0


def test_write_time_llm_counted_separately():
    """Phase 8 will write scored_by="llm" via skill-miner follow-up TXs;
    P6.J only counts what's on the chain. Validates the classification
    works when it's present."""
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0},
        {"tx_hash": "tc2", "scored_by": "llm", "ts": 2.0},
        {"tx_hash": "tc3", "scored_by": None, "ts": 3.0},
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: [],
    )
    r = a.analyze()
    assert r.scored_by_oracle == 1
    assert r.scored_by_llm == 1
    assert r.unscored == 1
    assert r.coverage_ratio == pytest.approx(2 / 3)


# ─────────────────────────────────────────────────────────────────────────
# Retrospective oracle scoring via batch joins
# ─────────────────────────────────────────────────────────────────────────


def test_retrospective_oracle_scoring_via_batch_join():
    """write_tool_call defaults scored_by=None at write time;
    OracleVerdictBatch carries parent_tool_call_tx → retrospective
    scoring promotes the tool-call to oracle-scored at read time."""
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": None, "ts": 1.0},
        {"tx_hash": "tc2", "scored_by": None, "ts": 2.0},
        {"tx_hash": "tc3", "scored_by": None, "ts": 3.0},
    ]
    batches = [
        {
            "tx_hash": "batch_a",
            "ts": 4.0,
            "entries": [
                {"parent_tool_call_tx": "tc1", "verdict": "true"},
                {"parent_tool_call_tx": "tc2", "verdict": "false"},
            ],
        }
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: batches,
    )
    r = a.analyze()
    assert r.total_tool_call_txs == 3
    assert r.scored_by_oracle == 2  # tc1, tc2
    assert r.unscored == 1           # tc3
    assert r.coverage_ratio == pytest.approx(2 / 3)


def test_retrospective_and_write_time_oracle_each_counted_once():
    """A tool call referenced by both a write-time scored_by="oracle"
    and a batch entry must count ONCE (not double)."""
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0},
    ]
    batches = [
        {
            "tx_hash": "b",
            "entries": [{"parent_tool_call_tx": "tc1", "verdict": "true"}],
        }
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: batches,
    )
    r = a.analyze()
    assert r.total_tool_call_txs == 1
    assert r.scored_by_oracle == 1     # not 2
    assert r.coverage_ratio == 1.0


def test_batch_verdict_unknown_does_not_count_as_oracle_scored():
    """Per arch §11.1: verdict ∈ {true, false, unknown}. Only true/false
    is evidence; unknown verdicts (e.g. metabolic_gate_denied,
    no_evidence) do NOT mark the tool call as oracle-scored."""
    tool_calls = [{"tx_hash": "tc1", "scored_by": None, "ts": 1.0}]
    batches = [
        {
            "entries": [{"parent_tool_call_tx": "tc1", "verdict": "unknown"}],
        }
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: batches,
    )
    r = a.analyze()
    assert r.scored_by_oracle == 0
    assert r.unscored == 1


def test_batch_with_missing_parent_tool_call_tx_skipped():
    tool_calls = [{"tx_hash": "tc1", "scored_by": None, "ts": 1.0}]
    batches = [
        {"entries": [{"verdict": "true"}]}  # no parent_tool_call_tx
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=lambda since, lim: batches,
    )
    r = a.analyze()
    assert r.scored_by_oracle == 0
    assert r.unscored == 1


# ─────────────────────────────────────────────────────────────────────────
# Reader exception defenses
# ─────────────────────────────────────────────────────────────────────────


def test_tool_call_reader_raises_returns_zero_coverage():
    def bad(since, lim):
        raise RuntimeError("chain down")

    a = CoverageAnalyzer(tool_call_reader=bad)
    r = a.analyze()
    assert r.total_tool_call_txs == 0
    assert r.coverage_ratio == 0.0


def test_batch_reader_raises_still_counts_write_time_scoring():
    """If batch_reader raises, we lose retrospective scoring but
    write-time scored_by still counts (so chain partial-read doesn't
    falsely zero the gate)."""
    tool_calls = [{"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0}]

    def bad(since, lim):
        raise RuntimeError("batch reader down")

    a = CoverageAnalyzer(
        tool_call_reader=lambda since, lim: tool_calls,
        batch_reader=bad,
    )
    r = a.analyze()
    assert r.total_tool_call_txs == 1
    assert r.scored_by_oracle == 1
    assert r.coverage_ratio == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Window + limit
# ─────────────────────────────────────────────────────────────────────────


def test_window_seconds_passed_through_to_readers():
    captured = {}

    def reader(since, lim):
        captured["since"] = since
        captured["lim"] = lim
        return []

    now = 1000.0
    a = CoverageAnalyzer(
        tool_call_reader=reader,
        batch_reader=lambda s, l: [],
        now_fn=lambda: now,
        default_window_seconds=600.0,
        default_limit=42,
    )
    a.analyze()
    assert captured["since"] == 400.0   # now - default window
    assert captured["lim"] == 42

    captured.clear()
    a.analyze(window_seconds=60.0, limit=5)
    assert captured["since"] == 940.0
    assert captured["lim"] == 5


# ─────────────────────────────────────────────────────────────────────────
# report_dict shape
# ─────────────────────────────────────────────────────────────────────────


def test_report_dict_includes_a6_gate_flag_true():
    tool_calls = [
        {"tx_hash": "tc1", "scored_by": "oracle", "ts": 1.0},
        {"tx_hash": "tc2", "scored_by": "oracle", "ts": 2.0},
    ]
    a = CoverageAnalyzer(
        tool_call_reader=lambda s, l: tool_calls,
        batch_reader=lambda s, l: [],
    )
    d = a.report_dict()
    assert d["total_tool_call_txs"] == 2
    assert d["coverage_ratio"] == 1.0
    assert d["a6_gate_passes"] is True


def test_report_dict_a6_gate_below_threshold():
    """Need 95% — 94/100 should fail the gate."""
    tool_calls = (
        [{"tx_hash": f"tc{i}", "scored_by": "oracle", "ts": 1.0} for i in range(94)]
        + [{"tx_hash": f"tc{i}", "scored_by": None, "ts": 1.0} for i in range(94, 100)]
    )
    a = CoverageAnalyzer(
        tool_call_reader=lambda s, l: tool_calls,
        batch_reader=lambda s, l: [],
    )
    d = a.report_dict()
    assert d["total_tool_call_txs"] == 100
    assert d["coverage_ratio"] == pytest.approx(0.94)
    assert d["a6_gate_passes"] is False


def test_report_dict_a6_gate_at_exact_threshold():
    """0.95 exactly passes the gate."""
    tool_calls = (
        [{"tx_hash": f"tc{i}", "scored_by": "oracle", "ts": 1.0} for i in range(95)]
        + [{"tx_hash": f"tc{i}", "scored_by": None, "ts": 1.0} for i in range(95, 100)]
    )
    a = CoverageAnalyzer(
        tool_call_reader=lambda s, l: tool_calls,
        batch_reader=lambda s, l: [],
    )
    d = a.report_dict()
    assert d["coverage_ratio"] == pytest.approx(0.95)
    assert d["a6_gate_passes"] is True
