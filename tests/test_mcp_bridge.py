"""
tests/test_mcp_bridge.py

Verification Test Suite for Titan V2.0 Phase 9 Step 3 — MCP Bridge.

Tests the five MCP tools in isolation (no MCP server protocol, no live services):
  Test 1: titan_guard blocks known dangerous prompts
  Test 2: titan_guard passes safe prompts
  Test 3: titan_recall returns formatted memories
  Test 4: titan_status returns metabolic info
  Test 5: titan_research triggers pipeline and returns findings
  Test 6: Lazy plugin initialisation (singleton)
  Test 7: Graceful degradation when subsystems unavailable
  Test 8: titan_chat fallback when Agno agent not available
  Test 9: titan_recall returns empty message for no results
  Test 10: titan_guard degraded when plugin boot fails

Follows existing codebase patterns:
  - pytest + pytest-asyncio
  - unittest.mock for all external dependencies
  - Isolated from live services
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import importlib.util

import pytest

# Ensure titan_plugin is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import titan_mcp by file path to avoid collision with the `mcp` PyPI package
_mcp_path = os.path.join(os.path.dirname(__file__), "..", "mcp", "titan_mcp.py")
_spec = importlib.util.spec_from_file_location("titan_mcp_mod", os.path.abspath(_mcp_path))
titan_mcp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(titan_mcp_mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the lazy singleton before each test for isolation."""
    titan_mcp_mod._plugin = None
    titan_mcp_mod._plugin_error = None
    yield
    titan_mcp_mod._plugin = None
    titan_mcp_mod._plugin_error = None


def _make_mock_plugin(
    guardian_safe: bool = True,
    memories: list | None = None,
    metabolic_state: str = "HIGH_ENERGY",
    balance: float = 1.5,
    mood: float = 0.75,
    research_result: str = "",
    limbo: bool = False,
):
    """Build a mock TitanPlugin with configurable subsystem responses."""
    plugin = MagicMock()
    plugin._limbo_mode = limbo

    # Guardian
    guardian = MagicMock()
    guardian.process_shield = AsyncMock(return_value=guardian_safe)
    guardian.restricted_keywords = ["rm -rf", "sudo rm", "drop table"]
    plugin.guardian = guardian

    # Memory
    memory = MagicMock()
    memory.query = AsyncMock(return_value=memories or [])
    plugin.memory = memory

    # Metabolism
    metabolism = MagicMock()
    metabolism.get_current_state = AsyncMock(return_value=metabolic_state)
    metabolism._last_balance = balance
    metabolism._last_balance_pct = max(1.0, min(100.0, (balance / 2.0) * 100))
    plugin.metabolism = metabolism

    # Mood engine
    mood_engine = MagicMock()
    mood_engine.get_current_mood = AsyncMock(return_value=mood)
    plugin.mood_engine = mood_engine

    # Gatekeeper
    gatekeeper = MagicMock()
    gatekeeper.sovereignty_score = 42.0
    plugin.gatekeeper = gatekeeper
    plugin._last_execution_mode = "Collaborative"

    # Researcher
    researcher = MagicMock()
    researcher.research = AsyncMock(return_value=research_result)
    plugin.sage_researcher = researcher

    # Agent (will raise to test fallback)
    plugin.create_agent = MagicMock(side_effect=RuntimeError("Agno not configured"))

    return plugin


# ---------------------------------------------------------------------------
# Test 1: titan_guard blocks dangerous prompts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_blocks_dangerous_prompt():
    """Guardian returns BLOCKED for a prompt matching restricted keywords."""
    plugin = _make_mock_plugin(guardian_safe=False)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_guard("I will run rm -rf / on the server")
    assert result.startswith("BLOCKED")
    assert "rm -rf" in result
    plugin.guardian.process_shield.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: titan_guard passes safe prompts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_passes_safe_prompt():
    """Guardian returns SAFE for an innocuous prompt."""
    plugin = _make_mock_plugin(guardian_safe=True)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_guard("What is the weather today?")
    assert result == "SAFE"


# ---------------------------------------------------------------------------
# Test 3: titan_recall returns formatted memories
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_returns_formatted_memories():
    """Memory query returns formatted results with weight and status."""
    memories = [
        {"content": "Solana hit $200 on 2026-02-15", "weight": 0.92, "status": "persistent"},
        {"content": "Meditation epoch completed at 03:00 UTC", "weight": 0.61, "status": "persistent"},
        {"content": "User asked about ZK compression", "weight": 0.45, "status": "mempool"},
    ]
    plugin = _make_mock_plugin(memories=memories)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_recall("Solana price")

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result
    assert "Solana hit $200" in result
    assert "w=0.92" in result
    assert "persistent" in result
    assert "mempool" in result
    plugin.memory.query.assert_awaited_once_with("Solana price")


# ---------------------------------------------------------------------------
# Test 4: titan_status returns metabolic info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_returns_metabolic_info():
    """Status tool returns structured vital signs."""
    plugin = _make_mock_plugin(
        metabolic_state="LOW_ENERGY",
        balance=0.35,
        mood=0.55,
    )
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_status()

    assert "LOW_ENERGY" in result
    assert "0.35" in result
    assert "Mood: 0.55" in result
    assert "Collaborative" in result
    assert "Sovereignty: 42.0%" in result


# ---------------------------------------------------------------------------
# Test 5: titan_research triggers pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_returns_findings():
    """Research tool invokes the Stealth-Sage pipeline and returns findings."""
    findings = "[SAGE_RESEARCH_FINDINGS]: Solana TVL reached $12B in March 2026."
    plugin = _make_mock_plugin(research_result=findings)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_research("Solana TVL 2026")

    assert "SAGE_RESEARCH_FINDINGS" in result
    assert "$12B" in result
    plugin.sage_researcher.research.assert_awaited_once_with("Solana TVL 2026")


# ---------------------------------------------------------------------------
# Test 6: Lazy plugin initialisation (singleton)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lazy_singleton_init():
    """Plugin is created once and reused across multiple tool calls."""
    plugin = _make_mock_plugin()
    titan_mcp_mod._plugin = plugin

    # Call two different tools
    await titan_mcp_mod.titan_guard("test")
    await titan_mcp_mod.titan_status()

    # Same instance used both times
    assert titan_mcp_mod._plugin is plugin


@pytest.mark.asyncio
async def test_lazy_init_caches_error():
    """When plugin boot fails, error is cached and not retried."""
    titan_mcp_mod._plugin = None
    titan_mcp_mod._plugin_error = "TitanPlugin boot failed: no wallet"

    result = await titan_mcp_mod.titan_guard("anything")
    assert "DEGRADED" in result
    assert "no wallet" in result

    # Verify it didn't try to re-init
    assert titan_mcp_mod._plugin is None


# ---------------------------------------------------------------------------
# Test 7: Graceful degradation when subsystems unavailable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_degraded_guardian_missing():
    """titan_guard reports degradation when guardian subsystem is absent."""
    plugin = MagicMock()
    plugin.guardian = None
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_guard("test")
    assert "DEGRADED" in result
    assert "Guardian" in result


@pytest.mark.asyncio
async def test_degraded_memory_missing():
    """titan_recall reports degradation when memory subsystem is absent."""
    plugin = MagicMock()
    plugin.memory = None
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_recall("test")
    assert "DEGRADED" in result
    assert "Memory" in result


@pytest.mark.asyncio
async def test_degraded_metabolism_missing():
    """titan_status handles missing metabolism gracefully."""
    plugin = MagicMock()
    plugin.metabolism = None
    plugin.mood_engine = None
    plugin.gatekeeper = None
    plugin._limbo_mode = False
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_status()
    assert "unavailable" in result.lower()


@pytest.mark.asyncio
async def test_degraded_researcher_missing():
    """titan_research reports degradation when researcher is absent."""
    plugin = MagicMock()
    plugin.sage_researcher = None
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_research("test query")
    assert "DEGRADED" in result
    assert "researcher" in result.lower()


# ---------------------------------------------------------------------------
# Test 8: titan_chat fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_fallback_no_agent():
    """titan_chat falls back to memory recall when Agno agent is unavailable."""
    memories = [
        {"content": "Previous discussion about validators"},
    ]
    plugin = _make_mock_plugin(guardian_safe=True, memories=memories)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_chat("Tell me about validators")

    # Should contain memory recall fallback
    assert "validators" in result.lower()


@pytest.mark.asyncio
async def test_chat_blocked_by_guardian():
    """titan_chat returns BLOCKED when guardian rejects the message."""
    plugin = _make_mock_plugin(guardian_safe=False)
    # Override create_agent to fail so we hit fallback
    plugin.create_agent = MagicMock(side_effect=RuntimeError("no agent"))
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_chat("rm -rf everything")
    assert "BLOCKED" in result


# ---------------------------------------------------------------------------
# Test 9: titan_recall empty results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_no_results():
    """titan_recall returns a clear message when no memories match."""
    plugin = _make_mock_plugin(memories=[])
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_recall("nonexistent topic xyz")
    assert "No memories found" in result


# ---------------------------------------------------------------------------
# Test 10: titan_status in limbo mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_limbo_mode():
    """titan_status shows limbo warning when plugin is in limbo."""
    plugin = _make_mock_plugin(limbo=True)
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_status()
    assert "LIMBO" in result


# ---------------------------------------------------------------------------
# Test 11: titan_research empty findings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_no_findings():
    """titan_research returns informative message when pipeline finds nothing."""
    plugin = _make_mock_plugin(research_result="")
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_research("obscure topic with no results")
    assert "no findings" in result.lower()


# ---------------------------------------------------------------------------
# Test 12: titan_guard handles process_shield exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_handles_exception():
    """titan_guard returns ERROR when process_shield raises."""
    plugin = _make_mock_plugin()
    plugin.guardian.process_shield = AsyncMock(side_effect=RuntimeError("embedder crash"))
    titan_mcp_mod._plugin = plugin

    result = await titan_mcp_mod.titan_guard("test prompt")
    assert "ERROR" in result
    assert "embedder crash" in result
