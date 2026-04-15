"""Tests for Step 7.4 — AgencyModule orchestrator."""
import pytest


class MockHelper:
    """Helper that succeeds."""
    def __init__(self, name="test_helper", status_val="available"):
        self._name = name
        self._status = status_val

    @property
    def name(self): return self._name

    @property
    def description(self): return "A test helper"

    @property
    def capabilities(self): return ["test"]

    @property
    def resource_cost(self): return "low"

    @property
    def latency(self): return "fast"

    @property
    def enriches(self): return ["mind"]

    @property
    def requires_sandbox(self): return False

    async def execute(self, params):
        return {"success": True, "result": "done", "enrichment_data": {"mind": [0]}, "error": None}

    def status(self): return self._status


class FailingHelper(MockHelper):
    """Helper that raises on execute."""
    async def execute(self, params):
        raise RuntimeError("boom")


def _make_intent(posture="research", impulse_id=1, urgency=0.5):
    return {
        "impulse_id": impulse_id,
        "posture": posture,
        "source_layer": "mind",
        "source_dims": [("mind", 0)],
        "deficit_values": [0.4],
        "urgency": urgency,
        "trinity_snapshot": {"body": [0.5]*5, "mind": [0.3]*5, "spirit": [0.5]*5},
    }


class TestAgencyModuleHandleIntent:
    """Core intent handling and helper execution."""

    @pytest.mark.asyncio
    async def test_handle_intent_no_helpers(self):
        from titan_plugin.logic.agency.module import AgencyModule
        agency = AgencyModule()
        result = await agency.handle_intent(_make_intent())
        assert result is None  # No helpers registered → skip

    @pytest.mark.asyncio
    async def test_handle_intent_rule_based(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("web_search"))
        agency = AgencyModule(registry=registry)

        result = await agency.handle_intent(_make_intent("research"))
        assert result is not None
        assert result["success"] is True
        assert result["helper"] == "web_search"
        assert result["posture"] == "research"
        assert result["impulse_id"] == 1

    @pytest.mark.asyncio
    async def test_handle_intent_fallback_helper(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("only_helper"))
        agency = AgencyModule(registry=registry)

        # No rule for "research" → "only_helper" (not "web_search"), but fallback finds it
        result = await agency.handle_intent(_make_intent("research"))
        assert result is not None
        assert result["helper"] == "only_helper"

    @pytest.mark.asyncio
    async def test_handle_intent_helper_error(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(FailingHelper("web_search"))
        agency = AgencyModule(registry=registry)

        result = await agency.handle_intent(_make_intent("research"))
        assert result is not None
        assert result["success"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_action_counter_increments(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("web_search"))
        agency = AgencyModule(registry=registry)

        await agency.handle_intent(_make_intent("research", impulse_id=1))
        await agency.handle_intent(_make_intent("research", impulse_id=2))
        assert agency.get_stats()["action_count"] == 2


class TestAgencyModuleBudget:
    """LLM budget management."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_skips(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry

        async def mock_llm(prompt):
            return '{"helper": "web_search", "params": {}, "reasoning": "test"}'

        registry = HelperRegistry()
        registry.register(MockHelper("web_search"))
        agency = AgencyModule(registry=registry, llm_fn=mock_llm, budget_per_hour=1)

        # First call uses budget
        r1 = await agency.handle_intent(_make_intent())
        assert r1 is not None
        # Second call blocked by budget
        r2 = await agency.handle_intent(_make_intent())
        assert r2 is None

    @pytest.mark.asyncio
    async def test_budget_resets_after_hour(self):
        import time
        from titan_plugin.logic.agency.module import AgencyModule
        agency = AgencyModule(budget_per_hour=1)
        agency._llm_calls_this_hour = 1
        agency._hour_start = time.time() - 3601  # Force hour reset
        agency._check_budget_reset()
        assert agency._llm_calls_this_hour == 0


class TestAgencyModuleLLMSelection:
    """LLM-based helper selection."""

    @pytest.mark.asyncio
    async def test_llm_selects_helper(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry

        async def mock_llm(prompt):
            return '{"helper": "web_search", "params": {"query": "solana"}, "reasoning": "need research"}'

        registry = HelperRegistry()
        registry.register(MockHelper("web_search"))
        agency = AgencyModule(registry=registry, llm_fn=mock_llm, budget_per_hour=10)

        result = await agency.handle_intent(_make_intent("research"))
        assert result["success"] is True
        assert result["helper"] == "web_search"

    @pytest.mark.asyncio
    async def test_llm_returns_none(self):
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.registry import HelperRegistry

        async def mock_llm(prompt):
            return '{"helper": "none", "reasoning": "nothing fits"}'

        registry = HelperRegistry()
        registry.register(MockHelper("web_search"))
        agency = AgencyModule(registry=registry, llm_fn=mock_llm, budget_per_hour=10)

        result = await agency.handle_intent(_make_intent("research"))
        assert result["success"] is False
        assert result["helper"] is None


class TestAgencyModuleParsing:
    """JSON parsing from LLM responses."""

    def test_parse_plain_json(self):
        from titan_plugin.logic.agency.module import AgencyModule
        result = AgencyModule._parse_selection('{"helper": "web_search", "params": {}}')
        assert result["helper"] == "web_search"

    def test_parse_markdown_block(self):
        from titan_plugin.logic.agency.module import AgencyModule
        raw = 'Here is my choice:\n```json\n{"helper": "web_search", "params": {}}\n```'
        result = AgencyModule._parse_selection(raw)
        assert result["helper"] == "web_search"

    def test_parse_embedded_json(self):
        from titan_plugin.logic.agency.module import AgencyModule
        raw = 'I think the best option is {"helper": "art_generate", "params": {}} because it fits.'
        result = AgencyModule._parse_selection(raw)
        assert result["helper"] == "art_generate"

    def test_parse_garbage_returns_none(self):
        from titan_plugin.logic.agency.module import AgencyModule
        result = AgencyModule._parse_selection("this is not json at all")
        assert result is None


class TestAgencyModuleStats:
    """Stats and introspection."""

    def test_stats_structure(self):
        from titan_plugin.logic.agency.module import AgencyModule
        agency = AgencyModule()
        stats = agency.get_stats()
        assert "action_count" in stats
        assert "llm_calls_this_hour" in stats
        assert "budget_per_hour" in stats
        assert "budget_remaining" in stats
        assert "registered_helpers" in stats
        assert "helper_statuses" in stats
