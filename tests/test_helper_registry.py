"""Tests for Step 7.3 — HelperRegistry and BaseHelper protocol."""
import pytest


class MockHelper:
    """Test helper implementing BaseHelper protocol."""

    def __init__(self, name: str = "test_helper", status_val: str = "available"):
        self._name = name
        self._status = status_val

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A test helper for unit tests"

    @property
    def capabilities(self) -> list[str]:
        return ["test", "mock"]

    @property
    def resource_cost(self) -> str:
        return "low"

    @property
    def latency(self) -> str:
        return "fast"

    @property
    def enriches(self) -> list[str]:
        return ["mind"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        return {"success": True, "result": "test done", "enrichment_data": {}, "error": None}

    def status(self) -> str:
        return self._status


class TestHelperRegistryCRUD:
    """Registration, lookup, and removal."""

    def test_register_and_get(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        helper = MockHelper("web_search")
        registry.register(helper)
        assert registry.get_helper("web_search") is helper

    def test_get_nonexistent_returns_none(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        assert registry.get_helper("nonexistent") is None

    def test_unregister(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        helper = MockHelper("test")
        registry.register(helper)
        assert registry.unregister("test") is True
        assert registry.get_helper("test") is None
        assert registry.unregister("test") is False  # Already removed

    def test_list_all_names(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("alpha"))
        registry.register(MockHelper("beta"))
        names = registry.list_all_names()
        assert "alpha" in names
        assert "beta" in names


class TestHelperRegistryManifest:
    """LLM-readable manifest generation."""

    def test_manifest_includes_available(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("web_search", "available"))
        manifest = registry.list_available()
        assert "web_search" in manifest
        assert "test, mock" in manifest  # capabilities
        assert "mind" in manifest  # enriches

    def test_manifest_excludes_unavailable(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("broken_helper", "unavailable"))
        manifest = registry.list_available()
        assert "broken_helper" not in manifest

    def test_manifest_marks_degraded(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("slow_helper", "degraded"))
        manifest = registry.list_available()
        assert "slow_helper" in manifest
        assert "[degraded]" in manifest

    def test_empty_registry(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        assert registry.list_available() == "No helpers available."


class TestHelperRegistryStatus:
    """Status tracking and caching."""

    def test_status_cached(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        helper = MockHelper("test", "available")
        registry.register(helper)
        # First call queries helper
        assert registry.get_status("test") == "available"
        # Change helper status (simulating degradation)
        helper._status = "degraded"
        # Cached — still returns old value
        assert registry.get_status("test") == "available"

    def test_stats_structure(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        registry.register(MockHelper("a", "available"))
        registry.register(MockHelper("b", "unavailable"))
        stats = registry.get_stats()
        assert stats["total"] == 2
        assert stats["available"] == 1
        assert stats["unavailable"] == 1
        assert "a" in stats["helpers"]
        assert "b" in stats["helpers"]


class TestBaseHelperProtocol:
    """Verify protocol compliance checking."""

    def test_mock_helper_implements_protocol(self):
        from titan_plugin.logic.agency.registry import BaseHelper
        helper = MockHelper()
        assert isinstance(helper, BaseHelper)
