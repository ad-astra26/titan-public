"""Tests for Step 7.6 — SelfAssessment (IQL self-scoring and enrichment routing)."""
import pytest
from titan_plugin.logic.agency.assessment import SelfAssessment, ENRICHMENT_MAP


def _make_action_result(
    success=True, posture="research", helper="web_search",
    result_text="Found 5 results", error=None, impulse_id=1, action_id=1,
):
    return {
        "action_id": action_id,
        "impulse_id": impulse_id,
        "posture": posture,
        "helper": helper,
        "success": success,
        "result": result_text,
        "enrichment_data": {},
        "error": error,
        "reasoning": "test",
        "trinity_before": {},
        "ts": 0,
    }


class TestHeuristicScoring:
    """Heuristic fallback scoring (no LLM)."""

    @pytest.mark.asyncio
    async def test_successful_action_scores_high(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=True, result_text="Found 5 results about Solana"))
        assert result["score"] == 0.7
        assert "succeeded" in result["reflection"]

    @pytest.mark.asyncio
    async def test_failed_action_scores_low(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=False, error="timeout"))
        assert result["score"] == 0.3
        assert "failed" in result["reflection"]

    @pytest.mark.asyncio
    async def test_success_minimal_result(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=True, result_text="ok"))
        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_success_no_result(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=True, result_text=""))
        assert result["score"] == 0.4


class TestLLMScoring:
    """LLM-based scoring."""

    @pytest.mark.asyncio
    async def test_llm_score_parsed(self):
        async def mock_llm(prompt):
            return '{"score": 0.85, "reflection": "great research results"}'

        sa = SelfAssessment(llm_fn=mock_llm)
        result = await sa.assess(_make_action_result())
        assert result["score"] == 0.85
        assert result["reflection"] == "great research results"

    @pytest.mark.asyncio
    async def test_llm_score_clamped(self):
        async def mock_llm(prompt):
            return '{"score": 1.5, "reflection": "over max"}'

        sa = SelfAssessment(llm_fn=mock_llm)
        result = await sa.assess(_make_action_result())
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        async def mock_llm(prompt):
            raise RuntimeError("LLM unavailable")

        sa = SelfAssessment(llm_fn=mock_llm)
        result = await sa.assess(_make_action_result(success=True, result_text="Found 5 results about Solana ecosystem"))
        # Falls back to heuristic (substantive result > 20 chars)
        assert result["score"] == 0.7


class TestEnrichmentRouting:
    """Enrichment computation from posture + score."""

    def test_research_success_enriches_mind_spirit(self):
        enrichment = SelfAssessment._compute_enrichment("research", 0.8)
        assert "mind" in enrichment
        assert 0 in enrichment["mind"]
        assert enrichment["mind"][0] > 0

        assert "spirit" in enrichment
        assert 2 in enrichment["spirit"]
        assert enrichment["spirit"][2] > 0

    def test_socialize_success_enriches_mind_spirit(self):
        enrichment = SelfAssessment._compute_enrichment("socialize", 0.9)
        assert "mind" in enrichment
        assert 1 in enrichment["mind"]
        assert 2 in enrichment["mind"]
        assert "spirit" in enrichment
        assert 0 in enrichment["spirit"]

    def test_rest_success_enriches_body(self):
        enrichment = SelfAssessment._compute_enrichment("rest", 0.7)
        assert "body" in enrichment
        assert 0 in enrichment["body"]
        assert 2 in enrichment["body"]
        assert 4 in enrichment["body"]

    def test_failure_gives_negative_enrichment(self):
        enrichment = SelfAssessment._compute_enrichment("research", 0.2)
        assert "mind" in enrichment
        assert enrichment["mind"][0] < 0

    def test_neutral_gives_no_enrichment(self):
        enrichment = SelfAssessment._compute_enrichment("research", 0.5)
        assert enrichment == {}

    def test_unknown_posture_gives_no_enrichment(self):
        enrichment = SelfAssessment._compute_enrichment("unknown_posture", 0.9)
        assert enrichment == {}


class TestMoodDelta:
    """Mood influence from action score."""

    def test_high_score_positive_mood(self):
        delta = SelfAssessment._compute_mood_delta(0.9)
        assert delta > 0

    def test_low_score_negative_mood(self):
        delta = SelfAssessment._compute_mood_delta(0.2)
        assert delta < 0

    def test_neutral_score_zero_mood(self):
        delta = SelfAssessment._compute_mood_delta(0.5)
        assert delta == 0.0


class TestThresholdDirection:
    """Threshold feedback from assessment."""

    @pytest.mark.asyncio
    async def test_success_lowers_threshold(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=True, result_text="Found 5 results about Solana"))
        assert result["threshold_direction"] == "lower"

    @pytest.mark.asyncio
    async def test_failure_raises_threshold(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=False, error="failed"))
        assert result["threshold_direction"] == "raise"

    @pytest.mark.asyncio
    async def test_neutral_holds_threshold(self):
        sa = SelfAssessment()
        result = await sa.assess(_make_action_result(success=True, result_text="ok"))
        assert result["threshold_direction"] == "hold"


class TestAssessmentStats:
    """Stats and history tracking."""

    @pytest.mark.asyncio
    async def test_stats_accumulate(self):
        sa = SelfAssessment()
        await sa.assess(_make_action_result(success=True, result_text="Found good results here"))
        await sa.assess(_make_action_result(success=False, error="oops"))
        stats = sa.get_stats()
        assert stats["total"] == 2
        assert 0 < stats["avg_score"] < 1.0
        assert len(stats["recent"]) == 2


class TestAssessmentParsing:
    """JSON parsing from LLM responses."""

    def test_parse_plain_json(self):
        result = SelfAssessment._parse_assessment('{"score": 0.8, "reflection": "good"}')
        assert result["score"] == 0.8

    def test_parse_markdown_block(self):
        raw = 'Here is my assessment:\n```json\n{"score": 0.6, "reflection": "ok"}\n```'
        result = SelfAssessment._parse_assessment(raw)
        assert result["score"] == 0.6

    def test_parse_embedded_json(self):
        raw = 'I rate this {"score": 0.9, "reflection": "excellent"} because reasons.'
        result = SelfAssessment._parse_assessment(raw)
        assert result["score"] == 0.9

    def test_parse_garbage_returns_none(self):
        result = SelfAssessment._parse_assessment("not json at all")
        assert result is None
