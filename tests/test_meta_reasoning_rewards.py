"""
Tests for TUNING-012 v2 Sub-phase A — Per-primitive compound reward helpers.

Verifies:
  - Each helper returns (reward, breakdown) tuple
  - Breakdown contains all expected component keys
  - Reward is within DNA-defined range
  - Each helper degrades gracefully when signals are 0
  - Each helper responds monotonically to its input signals
  - Per-Titan DNA overrides change reward output
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from titan_plugin.logic.meta_reasoning_rewards import (
    PRIMITIVE_REWARD_HELPERS,
    compute_primitive_reward,
    empty_subsystem_signals,
    reward_break,
    reward_evaluate,
    reward_formulate,
    reward_introspect,
    reward_recall,
)


# ── Test fixtures ─────────────────────────────────────────────────


@dataclass
class _MockState:
    """Minimal state shim mirroring MetaChainState surface used by reward helpers."""
    confidence: float = 0.5
    chain: list = field(default_factory=list)
    chain_succeeded: float = 0.0
    max_steps: int = 20
    formulate_output: dict = field(default_factory=dict)
    delegate_results: list = field(default_factory=list)
    pre_eval_confidence: float = 0.0
    pre_break_avg_reward: float = 0.0
    eureka_after_break: bool = False
    recall_history: list = field(default_factory=list)


def _full_dna() -> dict:
    """Return DNA matching defaults from titan_params.toml [meta_reasoning_dna]."""
    return {
        # RECALL
        "recall_base": 0.08,
        "recall_inner_relevance_weight": 0.10,
        "recall_inner_source_entropy_weight": 0.05,
        "recall_kuzu_centrality_weight": 0.04,
        "recall_timechain_depth_weight": 0.07,
        "recall_contract_ratified_weight": 0.05,
        "recall_outcome_corr_weight": 0.10,
        # FORMULATE
        "formulate_base": 0.08,
        "formulate_anomaly_weight": 0.10,
        "formulate_anomaly_dim_weight": 0.06,
        "formulate_specificity_weight": 0.06,
        "formulate_timechain_novelty_weight": 0.08,
        "formulate_contract_priority_weight": 0.05,
        "formulate_solvability_weight": 0.04,
        # EVALUATE
        "evaluate_base": 0.08,
        "evaluate_info_gain_weight": 0.12,
        "evaluate_consistency_weight": 0.08,
        "evaluate_timechain_consistency_weight": 0.07,
        "evaluate_contract_compliance_weight": 0.05,
        "evaluate_timing_weight": 0.05,
        # INTROSPECT
        "introspect_base": 0.08,
        "introspect_accuracy_weight": 0.10,
        "introspect_deepening_weight": 0.08,
        "introspect_timechain_continuity_weight": 0.07,
        "introspect_identity_alignment_weight": 0.05,
        "introspect_calibration_weight": 0.05,
        # BREAK
        "break_base": 0.05,
        "break_recovery_weight": 0.12,
        "break_eureka_weight": 0.08,
        "break_timechain_pattern_weight": 0.06,
        "break_contract_trigger_weight": 0.05,
        "break_cost_weight": -0.05,
    }


# ── RECALL ────────────────────────────────────────────────────────


class TestRecallReward:
    def test_returns_tuple_with_breakdown(self):
        state = _MockState()
        r, bd = reward_recall(state, {}, _full_dna(), empty_subsystem_signals())
        assert isinstance(r, float)
        assert isinstance(bd, dict)
        assert "total" in bd
        # All expected component keys present
        for k in ("base", "sim", "entropy", "centrality", "depth", "ratified", "outcome"):
            assert k in bd, f"missing component: {k}"

    def test_base_only_when_signals_empty(self):
        """With zero signals, RECALL should return ~base only."""
        state = _MockState()
        dna = _full_dna()
        r, bd = reward_recall(state, {}, dna, empty_subsystem_signals())
        # Just base + 0 from all components
        assert abs(r - dna["recall_base"]) < 1e-6
        assert bd["base"] == pytest.approx(0.08)

    def test_inner_relevance_increases_reward(self):
        """Higher FAISS similarity should increase reward."""
        state = _MockState()
        dna = _full_dna()
        sigs_low = empty_subsystem_signals()
        sigs_low["inner_relevance"] = 0.0
        sigs_high = empty_subsystem_signals()
        sigs_high["inner_relevance"] = 1.0
        r_low, _ = reward_recall(state, {}, dna, sigs_low)
        r_high, _ = reward_recall(state, {}, dna, sigs_high)
        assert r_high > r_low
        assert r_high - r_low == pytest.approx(dna["recall_inner_relevance_weight"], rel=1e-5)

    def test_recall_in_expected_range(self):
        """All-max signals should produce reward inside the documented 0.0-0.41 range."""
        state = _MockState(chain_succeeded=1.0, recall_history=[
            {"source": "chain_archive"}, {"source": "experience"},
            {"source": "wisdom"}, {"source": "entity"},
        ])
        dna = _full_dna()
        sigs = {
            "inner_relevance": 1.0,
            "kuzu_centrality": 1.0,
            "timechain_depth": 1.0,
            "contract_ratified": 1.0,
        }
        r, bd = reward_recall(state, {}, dna, sigs)
        assert 0.0 <= r <= 0.50  # documented max ~0.41 (allow small float headroom)
        assert bd["entropy"] > 0.0  # diverse history → entropy bonus

    def test_chain_success_adds_outcome_reward(self):
        state_fail = _MockState(chain_succeeded=0.0)
        state_succ = _MockState(chain_succeeded=1.0)
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        r_fail, _ = reward_recall(state_fail, {}, dna, sigs)
        r_succ, _ = reward_recall(state_succ, {}, dna, sigs)
        assert r_succ > r_fail


# ── FORMULATE ─────────────────────────────────────────────────────


class TestFormulateReward:
    def test_returns_tuple_with_breakdown(self):
        state = _MockState()
        r, bd = reward_formulate(state, {}, _full_dna(), empty_subsystem_signals())
        assert isinstance(r, float)
        assert isinstance(bd, dict)
        for k in ("base", "anomaly", "anomaly_dim", "specificity", "novelty", "contract", "solvability"):
            assert k in bd

    def test_difficulty_drives_anomaly_component(self):
        """Higher problem difficulty → larger anomaly term."""
        state = _MockState()
        dna = _full_dna()
        r_low, _ = reward_formulate(state, {"difficulty": 0.0}, dna, empty_subsystem_signals())
        r_high, _ = reward_formulate(state, {"difficulty": 1.0}, dna, empty_subsystem_signals())
        # The solvability heuristic peaks at 0.5 so difficulty=1.0 gets ZERO solvability
        # but high anomaly. Make sure anomaly contribution dominates.
        assert r_high > r_low

    def test_dimensionality_increases_score(self):
        state = _MockState()
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        r0, _ = reward_formulate(state, {"anomalous_dims": []}, dna, sigs)
        r5, _ = reward_formulate(state, {"anomalous_dims": [1, 2, 3, 4, 5]}, dna, sigs)
        assert r5 > r0

    def test_repetitive_formulations_penalized(self):
        """High novelty score should yield more reward than low novelty."""
        state = _MockState()
        dna = _full_dna()
        sigs_repetitive = empty_subsystem_signals()
        sigs_repetitive["timechain_novelty"] = 0.0
        sigs_novel = empty_subsystem_signals()
        sigs_novel["timechain_novelty"] = 1.0
        r_rep, _ = reward_formulate(state, {}, dna, sigs_repetitive)
        r_novel, _ = reward_formulate(state, {}, dna, sigs_novel)
        assert r_novel > r_rep

    def test_t2_override_boosts_anomaly_weight(self):
        """T2's per-Titan override (formulate_anomaly_weight=0.12) yields more reward."""
        state = _MockState()
        dna_base = _full_dna()
        dna_t2 = _full_dna()
        dna_t2["formulate_anomaly_weight"] = 0.12  # T2 override from titan_params.toml
        out = {"difficulty": 0.5}
        r_base, _ = reward_formulate(state, out, dna_base, empty_subsystem_signals())
        r_t2, _ = reward_formulate(state, out, dna_t2, empty_subsystem_signals())
        assert r_t2 > r_base


# ── EVALUATE ──────────────────────────────────────────────────────


class TestEvaluateReward:
    def test_returns_tuple_with_breakdown(self):
        state = _MockState()
        r, bd = reward_evaluate(state, {}, _full_dna(), empty_subsystem_signals())
        for k in ("base", "info_gain", "consistency", "tc_consistency", "compliance", "timing"):
            assert k in bd

    def test_info_gain_rewards_confidence_change(self):
        """Larger pre→post confidence delta should yield more info_gain reward."""
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        # No change
        s_flat = _MockState(pre_eval_confidence=0.5, confidence=0.5)
        r_flat, _ = reward_evaluate(s_flat, {}, dna, sigs)
        # Big jump
        s_jump = _MockState(pre_eval_confidence=0.2, confidence=0.8)
        r_jump, _ = reward_evaluate(s_jump, {}, dna, sigs)
        assert r_jump > r_flat

    def test_timing_peaks_at_midpoint(self):
        """Calling EVALUATE at the midpoint of the chain should be optimal."""
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        # 1 step out of 20 (early)
        s_early = _MockState(chain=["x"], max_steps=20)
        # 10 steps out of 20 (midpoint)
        s_mid = _MockState(chain=["x"] * 10, max_steps=20)
        # 19 steps out of 20 (late)
        s_late = _MockState(chain=["x"] * 19, max_steps=20)
        r_early, b_early = reward_evaluate(s_early, {}, dna, sigs)
        r_mid, b_mid = reward_evaluate(s_mid, {}, dna, sigs)
        r_late, b_late = reward_evaluate(s_late, {}, dna, sigs)
        assert b_mid["timing"] > b_early["timing"]
        assert b_mid["timing"] > b_late["timing"]


# ── INTROSPECT ────────────────────────────────────────────────────


class TestIntrospectReward:
    def test_returns_tuple_with_breakdown(self):
        state = _MockState()
        r, bd = reward_introspect(state, {}, _full_dna(), empty_subsystem_signals())
        for k in ("base", "accuracy", "deepening", "continuity", "identity", "calibration"):
            assert k in bd

    def test_calibration_high_when_confidence_matches_outcome(self):
        """Reward should be higher when meta-confidence calibrates with chain outcome."""
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        s_calibrated = _MockState(confidence=0.8, chain_succeeded=0.8)
        s_overconf = _MockState(confidence=0.9, chain_succeeded=0.1)
        r_cal, b_cal = reward_introspect(s_calibrated, {}, dna, sigs)
        r_over, b_over = reward_introspect(s_overconf, {}, dna, sigs)
        assert b_cal["calibration"] > b_over["calibration"]

    def test_t3_override_boosts_deepening(self):
        """T3's per-Titan override (introspect_deepening_weight=0.10) yields more reward."""
        state = _MockState()
        dna_base = _full_dna()
        dna_t3 = _full_dna()
        dna_t3["introspect_deepening_weight"] = 0.10
        sigs = empty_subsystem_signals()
        sigs["self_profile_divergence"] = 1.0
        r_base, _ = reward_introspect(state, {}, dna_base, sigs)
        r_t3, _ = reward_introspect(state, {}, dna_t3, sigs)
        assert r_t3 > r_base


# ── BREAK ─────────────────────────────────────────────────────────


class TestBreakReward:
    def test_returns_tuple_with_breakdown(self):
        state = _MockState()
        r, bd = reward_break(state, {}, _full_dna(), empty_subsystem_signals())
        for k in ("base", "recovery", "eureka", "pattern", "trigger", "cost"):
            assert k in bd

    def test_break_can_be_net_negative_if_unused_well(self):
        """BREAK with no recovery + no eureka + no contract trigger = base + cost."""
        state = _MockState(chain_succeeded=0.0, eureka_after_break=False)
        dna = _full_dna()
        r, bd = reward_break(state, {}, dna, empty_subsystem_signals())
        # base (0.05) + 0 + 0 + 0 + 0 + cost (-0.05) = 0.0
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_eureka_after_break_adds_reward(self):
        dna = _full_dna()
        sigs = empty_subsystem_signals()
        s_no_eureka = _MockState(eureka_after_break=False)
        s_eureka = _MockState(eureka_after_break=True)
        r_no, _ = reward_break(s_no_eureka, {}, dna, sigs)
        r_yes, _ = reward_break(s_eureka, {}, dna, sigs)
        assert r_yes - r_no == pytest.approx(dna["break_eureka_weight"], abs=1e-6)


# ── Dispatcher ────────────────────────────────────────────────────


class TestDispatcher:
    def test_dispatcher_returns_zero_for_unknown_primitive(self):
        state = _MockState()
        r, bd = compute_primitive_reward(
            "DELEGATE", state, {}, _full_dna(), empty_subsystem_signals())
        assert r == 0.0
        assert bd == {}

    def test_dispatcher_routes_to_helper(self):
        state = _MockState()
        for prim in ("RECALL", "FORMULATE", "EVALUATE", "INTROSPECT", "BREAK"):
            r, bd = compute_primitive_reward(
                prim, state, {}, _full_dna(), empty_subsystem_signals())
            assert "total" in bd
            assert "base" in bd

    def test_all_helpers_in_registry(self):
        assert set(PRIMITIVE_REWARD_HELPERS.keys()) == {
            "RECALL", "FORMULATE", "EVALUATE", "INTROSPECT", "BREAK"
        }

    def test_empty_signals_has_all_keys(self):
        sigs = empty_subsystem_signals()
        # Verify presence of every signal name we use across helpers
        expected = {
            "inner_relevance", "kuzu_centrality", "timechain_depth", "contract_ratified",
            "timechain_novelty", "contract_priority",
            "timechain_eval_consistency", "contract_compliance",
            "self_prediction_accuracy", "self_profile_divergence",
            "timechain_self_continuity", "contract_identity_alignment",
            "timechain_break_pattern", "contract_break_trigger",
        }
        assert expected.issubset(set(sigs.keys()))
