"""Tests for Outer Trinity Dimensional Symmetry (OT1 + OT2)."""
import pytest


class TestOuterMindTensor15D:
    """OT1: Outer Mind 15D tensor tests."""

    def test_output_is_15d(self):
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        result = collect_outer_mind_15d([0.5] * 5)
        assert len(result) == 15

    def test_thinking_from_assessment(self):
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        result = collect_outer_mind_15d(
            [0.5] * 5,
            assessment_stats={"mean_score": 0.9},
        )
        # Knowledge retrieval [1] and communication clarity [4] use assessment
        assert result[1] > 0.7
        assert result[4] > 0.7

    def test_feeling_threat_sensing(self):
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        # High threats = low comfort
        result_safe = collect_outer_mind_15d(
            [0.5] * 5,
            guardian_stats={"threats_detected": 0, "severity_avg": 0.0},
        )
        result_danger = collect_outer_mind_15d(
            [0.5] * 5,
            guardian_stats={"threats_detected": 10, "severity_avg": 0.8},
        )
        assert result_safe[8] > result_danger[8]  # Threat sensing dim

    def test_willing_from_actions(self):
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        result = collect_outer_mind_15d(
            [0.5] * 5,
            action_stats={"per_window": 8},
            creative_stats={"per_window": 4},
            research_stats={"queries_per_window": 3},
        )
        assert result[10] > 0.5  # Action throughput
        assert result[12] > 0.5  # Creative output
        assert result[14] > 0.5  # Exploration drive

    def test_all_bounded(self):
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        result = collect_outer_mind_15d(
            [0.5] * 5,
            action_stats={"per_window": 100, "success_rate": 1.5},
            creative_stats={"per_window": 100},
            guardian_stats={"threats_detected": 100, "severity_avg": 2.0},
        )
        for i, v in enumerate(result):
            assert 0.0 <= v <= 1.0, f"Dim {i} = {v} out of bounds"

    def test_dim_names_count(self):
        from titan_plugin.logic.outer_mind_tensor import OUTER_MIND_DIM_NAMES
        assert len(OUTER_MIND_DIM_NAMES) == 15


class TestOuterSpiritTensor45D:
    """OT2: Outer Spirit 45D tensor tests."""

    def test_output_is_45d(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        result = collect_outer_spirit_45d([0.5] * 5, [0.5] * 5, [0.5] * 15)
        assert len(result) == 45

    def test_sat_is_first_15(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        result = collect_outer_spirit_45d([0.5] * 5, [0.5] * 5, [0.5] * 15)
        sat = result[0:15]
        assert len(sat) == 15
        assert all(0.0 <= v <= 1.0 for v in sat)

    def test_chit_is_middle_15(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        result = collect_outer_spirit_45d([0.5] * 5, [0.5] * 5, [0.5] * 15)
        chit = result[15:30]
        assert len(chit) == 15
        assert all(0.0 <= v <= 1.0 for v in chit)

    def test_ananda_is_last_15(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        result = collect_outer_spirit_45d([0.5] * 5, [0.5] * 5, [0.5] * 15)
        ananda = result[30:45]
        assert len(ananda) == 15
        assert all(0.0 <= v <= 1.0 for v in ananda)

    def test_observer_principle_body_affects_spirit(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        # High body coherence should affect Spirit dimensions
        result_low = collect_outer_spirit_45d(
            [0.5] * 5, [0.2] * 5, [0.5] * 15)
        result_high = collect_outer_spirit_45d(
            [0.5] * 5, [0.9] * 5, [0.5] * 15)
        # SAT-3 boundary_enforcement uses body_coh indirectly
        # CHIT-4 witness_stability = body_coh × mind_coh
        assert result_high[15 + 4] > result_low[15 + 4]

    def test_action_purity(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        # High assessment + high success = high purity
        result_pure = collect_outer_spirit_45d(
            [0.5] * 5, [0.5] * 5, [0.5] * 15,
            action_stats={"success_rate": 0.9},
            assessment_stats={"mean_score": 0.9},
        )
        result_impure = collect_outer_spirit_45d(
            [0.5] * 5, [0.5] * 5, [0.5] * 15,
            action_stats={"success_rate": 0.2},
            assessment_stats={"mean_score": 0.2},
        )
        assert result_pure[9] > result_impure[9]  # SAT-9 action purity

    def test_surrender_capacity(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        # Low retry rate + healthy body = high surrender
        result_surrendered = collect_outer_spirit_45d(
            [0.5] * 5, [0.9] * 5, [0.5] * 15,
            action_stats={"failed_retry_rate": 0.0, "burst_frequency": 0.0},
        )
        # High retry rate + depleted body = low surrender
        result_pushing = collect_outer_spirit_45d(
            [0.5] * 5, [0.1] * 5, [0.5] * 15,
            action_stats={"failed_retry_rate": 0.8, "burst_frequency": 0.7},
        )
        assert result_surrendered[42] > result_pushing[42]  # ANANDA-12

    def test_flow_state_depends_on_surrender(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        # High coherence BUT low surrender → limited flow
        result_no_surrender = collect_outer_spirit_45d(
            [0.5] * 5, [0.1] * 5, [0.9] * 15,  # Low body = low surrender
            action_stats={"failed_retry_rate": 0.9, "burst_frequency": 0.9,
                          "error_rate": 0.0},
            assessment_stats={"mean_score": 0.9},
        )
        # High coherence AND high surrender → flow possible
        result_with_surrender = collect_outer_spirit_45d(
            [0.5] * 5, [0.9] * 5, [0.9] * 15,  # High body = high surrender
            action_stats={"failed_retry_rate": 0.0, "burst_frequency": 0.0,
                          "error_rate": 0.0},
            assessment_stats={"mean_score": 0.9},
        )
        assert result_with_surrender[44] > result_no_surrender[44]  # ANANDA-14

    def test_all_bounded(self):
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d
        result = collect_outer_spirit_45d(
            [0.5] * 5, [0.5] * 5, [0.5] * 15,
            action_stats={"total": 999, "success_rate": 1.5, "per_hour": 100},
            creative_stats={"total": 999, "mean_assessment": 1.5},
            guardian_stats={"threats_detected": 999, "rejections": 999},
        )
        for i, v in enumerate(result):
            assert 0.0 <= v <= 1.0, f"Dim {i} = {v} out of bounds"

    def test_dim_names_count(self):
        from titan_plugin.logic.outer_spirit_tensor import OUTER_SPIRIT_DIM_NAMES
        assert len(OUTER_SPIRIT_DIM_NAMES) == 45


class TestSymmetryPrinciples:
    """Verify architectural symmetry between Inner and Outer."""

    def test_mind_both_15d(self):
        from titan_plugin.logic.mind_tensor import MIND_DIM_NAMES
        from titan_plugin.logic.outer_mind_tensor import OUTER_MIND_DIM_NAMES
        assert len(MIND_DIM_NAMES) == 15
        assert len(OUTER_MIND_DIM_NAMES) == 15

    def test_spirit_both_45d(self):
        from titan_plugin.logic.spirit_tensor import SPIRIT_DIM_NAMES
        from titan_plugin.logic.outer_spirit_tensor import OUTER_SPIRIT_DIM_NAMES
        assert len(SPIRIT_DIM_NAMES) == 45
        assert len(OUTER_SPIRIT_DIM_NAMES) == 45

    def test_total_130d(self):
        """Inner 65D + Outer 65D = 130D total."""
        inner = 5 + 15 + 45  # Body + Mind + Spirit
        outer = 5 + 15 + 45
        assert inner == 65
        assert outer == 65
        assert inner + outer == 130
