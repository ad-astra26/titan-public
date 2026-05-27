"""Phase 8 — actr_procedural_skill_proposer SC contract tests (D-SPEC-PHASE8).

Covers:
- JSON loads and parses
- version bumped to 2 (P8 graduation from P2 TIMER stub)
- scored_by filter ∈ {oracle, llm} present
- Threshold lowered from 10 → 6 (min_seq_len × min_occurrences)
- triggers include dream_boundary
- status active
- emit event is META_SKILL_COMPILATION_CANDIDATE
- emit data carries min_seq_len + min_occurrences for the miner
"""
from __future__ import annotations

import json
import os

import pytest


CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "titan_hcl", "contracts",
    "meta_cognitive", "actr_procedural_skill_proposer.json",
)


@pytest.fixture()
def contract() -> dict:
    with open(CONTRACT_PATH) as f:
        return json.load(f)


def test_contract_loads(contract):
    assert contract["contract_id"] == "actr_procedural_skill_proposer"


def test_version_bumped_to_p8(contract):
    assert contract["version"] == 2


def test_status_active(contract):
    assert contract["status"] == "active"


def test_triggers_include_dream_boundary(contract):
    assert "dream_boundary" in contract["triggers"]


def test_fork_read_filters_scored_by(contract):
    fork_read = next(r for r in contract["rules"] if r["op"] == "FORK_READ")
    assert fork_read["filter"]["thought_type"] == "tool_call"
    assert set(fork_read["filter"]["scored_by_in"]) == {"oracle", "llm"}


def test_fork_read_window_168h(contract):
    fork_read = next(r for r in contract["rules"] if r["op"] == "FORK_READ")
    assert fork_read["since_hours"] == 168


def test_if_threshold_lowered_from_stub(contract):
    """P2 stub threshold was 10; P8 lowers to miner_min_seq_len × miner_min_occurrences = 6."""
    if_rule = next(r for r in contract["rules"] if r["op"] == "IF")
    assert if_rule["cmp"] == "GTE"
    assert if_rule["value"] == 6
    assert if_rule["field"] == "$tier1_traces.length"


def test_emit_event_name(contract):
    if_rule = next(r for r in contract["rules"] if r["op"] == "IF")
    emit_action = if_rule["then"]
    assert emit_action["action"] == "emit"
    assert emit_action["event"] == "META_SKILL_COMPILATION_CANDIDATE"


def test_emit_data_carries_miner_params(contract):
    if_rule = next(r for r in contract["rules"] if r["op"] == "IF")
    data = if_rule["then"]["data"]
    assert data["min_seq_len"] == 2
    assert data["min_occurrences"] == 3
    assert data["phase"].startswith("p8_")


def test_phase_marker_p8(contract):
    assert contract["phase"].startswith("p8_")
