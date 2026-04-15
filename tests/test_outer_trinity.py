"""
Tests for V4 Time Awareness — OuterTrinityCollector.

Tests computation of Outer Trinity 3×5DT tensors from live data sources,
normalization, graceful degradation, and integration with mock sources.
"""
import pytest


def _make_sources(**overrides) -> dict:
    """Create a complete sources dict with sensible defaults."""
    defaults = {
        "agency_stats": {
            "budget_per_hour": 10,
            "actions_this_hour": 3,
            "total_actions": 20,
            "failed_actions": 2,
        },
        "assessment_stats": {
            "total_assessed": 15,
            "average_score": 0.72,
        },
        "helper_statuses": {
            "infra_inspect": "available",
            "web_search": "available",
            "social_post": "unavailable",
            "art_generate": "available",
            "audio_generate": "available",
            "coding_sandbox": "available",
            "code_knowledge": "available",
        },
        "bus_stats": {
            "published": 500,
            "routed": 450,
            "dropped": 5,
        },
        "impulse_stats": {
            "total_fires": 10,
        },
        "observatory_db": None,
        "memory_status": {
            "persistent_count": 50,
            "total_nodes": 100,
            "research_nodes": 3,
            "unique_interactors": 8,
        },
        "soul_health": 0.85,
        "llm_avg_latency": 5.0,
        "uptime_seconds": 3600.0,
    }
    defaults.update(overrides)
    return defaults


class TestOuterTrinityCollector:
    """Tests for OuterTrinityCollector."""

    def test_collect_returns_correct_structure(self):
        """Collector returns dict with 3 keys, each a 5-element list."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
        result = collector.collect(_make_sources())

        assert "outer_body" in result
        assert "outer_mind" in result
        assert "outer_spirit" in result
        assert len(result["outer_body"]) == 5
        assert len(result["outer_mind"]) == 5
        assert len(result["outer_spirit"]) == 5

    def test_all_values_normalized_0_1(self):
        """All output values are within [0.0, 1.0]."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()
        result = collector.collect(_make_sources())

        for key in ["outer_body", "outer_mind", "outer_spirit"]:
            for i, v in enumerate(result[key]):
                assert 0.0 <= v <= 1.0, f"{key}[{i}] = {v} out of range"

    def test_outer_body_action_energy(self):
        """action_energy decreases as budget is consumed."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        # Low usage → high energy
        result_low = collector.collect(_make_sources(
            agency_stats={"budget_per_hour": 10, "actions_this_hour": 1,
                         "total_actions": 5, "failed_actions": 0}
        ))

        # High usage → low energy
        result_high = collector.collect(_make_sources(
            agency_stats={"budget_per_hour": 10, "actions_this_hour": 9,
                         "total_actions": 50, "failed_actions": 5}
        ))

        assert result_low["outer_body"][0] > result_high["outer_body"][0]

    def test_outer_body_helper_health(self):
        """helper_health reflects available/total ratio."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources(
            helper_statuses={
                "h1": "available",
                "h2": "available",
                "h3": "unavailable",
                "h4": "available",
            }
        ))
        # 3/4 = 0.75
        assert result["outer_body"][1] == pytest.approx(0.75, abs=0.01)

    def test_outer_body_from_bus_stats(self):
        """bus_throughput is derived from bus routing stats."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources(
            bus_stats={"routed": 3600, "published": 4000, "dropped": 10},
            uptime_seconds=3600.0,
        ))

        # 3600 routed / 3600 expected ≈ 1.0
        assert result["outer_body"][2] >= 0.9

    def test_outer_mind_memory_quality(self):
        """memory_quality reflects persistent/total ratio."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources(
            memory_status={
                "persistent_count": 80,
                "total_nodes": 100,
                "research_nodes": 5,
                "unique_interactors": 10,
            }
        ))
        # 80/100 = 0.8
        assert result["outer_mind"][2] == pytest.approx(0.8, abs=0.01)

    def test_outer_spirit_identity_coherence(self):
        """identity_coherence uses soul_health directly."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources(soul_health=0.92))
        assert result["outer_spirit"][0] == pytest.approx(0.92, abs=0.01)

    def test_outer_spirit_purpose_clarity(self):
        """purpose_clarity from assessment average score."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources(
            assessment_stats={"total_assessed": 20, "average_score": 0.85},
            impulse_stats={"total_fires": 15},
        ))
        assert result["outer_spirit"][1] == pytest.approx(0.85, abs=0.01)

    def test_outer_spirit_scalars_are_means(self):
        """outer_body_scalar and outer_mind_scalar are means of respective tensors."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        result = collector.collect(_make_sources())
        outer_body = result["outer_body"]
        outer_mind = result["outer_mind"]
        outer_spirit = result["outer_spirit"]

        expected_body_scalar = sum(outer_body) / len(outer_body)
        expected_mind_scalar = sum(outer_mind) / len(outer_mind)

        assert outer_spirit[3] == pytest.approx(expected_body_scalar, abs=0.01)
        assert outer_spirit[4] == pytest.approx(expected_mind_scalar, abs=0.01)

    def test_missing_sources_degrade_gracefully(self):
        """Missing or None sources return 0.5 (neutral) values."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        # All sources missing/empty
        result = collector.collect({})

        for key in ["outer_body", "outer_mind", "outer_spirit"]:
            for i, v in enumerate(result[key]):
                assert 0.0 <= v <= 1.0, f"{key}[{i}] = {v} out of range"

    def test_full_integration_with_all_sources(self):
        """Full collection with all sources populated."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        sources = _make_sources(
            soul_health=0.9,
            llm_avg_latency=3.0,
            uptime_seconds=86400.0,
        )
        result = collector.collect(sources)

        # Verify structure completeness
        assert collector._collect_count == 1
        assert collector._last_collect_ts > 0

        stats = collector.get_stats()
        assert stats["collect_count"] == 1
        assert len(stats["outer_body"]) == 5

    def test_get_last_tensors(self):
        """get_last_tensors returns most recent collection."""
        from titan_plugin.logic.outer_trinity import OuterTrinityCollector
        collector = OuterTrinityCollector()

        collector.collect(_make_sources())
        last = collector.get_last_tensors()

        assert "outer_body" in last
        assert "outer_mind" in last
        assert "outer_spirit" in last


class TestSafeClamp:
    """Test the _safe_clamp utility."""

    def test_normal_values(self):
        from titan_plugin.logic.outer_trinity import _safe_clamp
        assert _safe_clamp(0.5) == 0.5
        assert _safe_clamp(0.0) == 0.0
        assert _safe_clamp(1.0) == 1.0

    def test_out_of_range(self):
        from titan_plugin.logic.outer_trinity import _safe_clamp
        assert _safe_clamp(-0.5) == 0.0
        assert _safe_clamp(1.5) == 1.0

    def test_nan_returns_default(self):
        from titan_plugin.logic.outer_trinity import _safe_clamp
        assert _safe_clamp(float('nan')) == 0.5

    def test_none_returns_default(self):
        from titan_plugin.logic.outer_trinity import _safe_clamp
        assert _safe_clamp(None) == 0.5

    def test_inf_returns_default(self):
        from titan_plugin.logic.outer_trinity import _safe_clamp
        assert _safe_clamp(float('inf')) == 0.5
