"""
tests/test_inference_module.py — Chunk A regression tests for the
new titan_hcl.inference library (D-SPEC-72).

Covers:
  - get_provider() factory: all 5 registered providers + unknown error
  - InferenceProvider ABC contract: id / name / base_url / chat / stream_chat
  - Convenience wrappers: complete() + distill() route through chat()
  - get_agno_model() returns the right agno class per provider
  - Stats surface returns provider-specific shape
  - Mocked HTTP backends — no live network calls in this test file
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from titan_hcl import inference
from titan_hcl.inference import (
    CustomProvider,
    InferenceProvider,
    OllamaCloudProvider,
    OpenRouterProvider,
    VeniceProvider,
    VeniceSessionProvider,
    get_model_for_task,
    get_provider,
)


# ────────────────────────────────────────────────────────────────────
# get_provider() factory
# ────────────────────────────────────────────────────────────────────

class TestGetProviderFactory:
    """Factory function — name dispatch + unknown error."""

    def test_returns_venice_provider(self):
        p = get_provider("venice", {"venice_api_key": "test"})
        assert isinstance(p, VeniceProvider)
        assert p.name == "VeniceAI"

    def test_returns_venice_session_provider(self):
        p = get_provider("venice_session", {"venice_session_token": "test"})
        assert isinstance(p, VeniceSessionProvider)
        assert p.name == "VeniceSession"

    def test_returns_openrouter_provider(self):
        p = get_provider("openrouter", {"openrouter_api_key": "test"})
        assert isinstance(p, OpenRouterProvider)
        assert p.name == "OpenRouter"

    def test_returns_ollama_cloud_provider(self):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "test"})
        assert isinstance(p, OllamaCloudProvider)
        assert p.name == "OllamaCloud"

    def test_returns_custom_provider(self):
        p = get_provider("custom", {"custom_llm_api_key": "test"})
        assert isinstance(p, CustomProvider)
        assert p.name == "CustomLLM"

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent", {})
        msg = str(excinfo.value)
        assert "nonexistent" in msg
        assert "Known providers" in msg

    def test_unknown_provider_lists_all_known(self):
        try:
            get_provider("does-not-exist", {})
        except ValueError as e:
            msg = str(e)
            for known in [
                "venice", "venice_session", "openrouter",
                "ollama_cloud", "custom",
            ]:
                assert known in msg


# ────────────────────────────────────────────────────────────────────
# ABC contract — identity properties
# ────────────────────────────────────────────────────────────────────

class TestIdentityProperties:
    """Every provider exposes id / name / base_url without network calls."""

    @pytest.mark.parametrize("name,cfg,expected_name", [
        ("venice", {"venice_api_key": "k"}, "VeniceAI"),
        ("venice_session", {"venice_session_token": "t"}, "VeniceSession"),
        ("openrouter", {"openrouter_api_key": "k"}, "OpenRouter"),
        ("ollama_cloud", {"ollama_cloud_api_key": "k"}, "OllamaCloud"),
        ("custom", {"custom_llm_api_key": "k"}, "CustomLLM"),
    ])
    def test_name_property(self, name, cfg, expected_name):
        p = get_provider(name, cfg)
        assert p.name == expected_name

    @pytest.mark.parametrize("name,cfg", [
        ("venice", {"venice_api_key": "k"}),
        ("venice_session", {"venice_session_token": "t"}),
        ("openrouter", {"openrouter_api_key": "k"}),
        ("ollama_cloud", {"ollama_cloud_api_key": "k"}),
        ("custom", {"custom_llm_api_key": "k"}),
    ])
    def test_id_property_non_empty(self, name, cfg):
        p = get_provider(name, cfg)
        assert isinstance(p.id, str)
        assert len(p.id) > 0

    @pytest.mark.parametrize("name,cfg", [
        ("venice", {"venice_api_key": "k"}),
        ("venice_session", {"venice_session_token": "t"}),
        ("openrouter", {"openrouter_api_key": "k"}),
        ("ollama_cloud", {"ollama_cloud_api_key": "k"}),
        ("custom", {"custom_llm_api_key": "k"}),
    ])
    def test_base_url_no_trailing_slash(self, name, cfg):
        p = get_provider(name, cfg)
        assert not p.base_url.endswith("/")

    def test_ollama_cloud_normalizes_api_subdomain(self):
        """api.ollama.com → ollama.com (HTTP 301 redirect)."""
        p = get_provider(
            "ollama_cloud",
            {
                "ollama_cloud_api_key": "k",
                "ollama_cloud_base_url": "https://api.ollama.com/v1",
            },
        )
        assert p.base_url == "https://ollama.com/v1"

    def test_custom_uses_custom_provider_name(self):
        p = get_provider(
            "custom",
            {
                "custom_llm_api_key": "k",
                "custom_provider_name": "MyVLLM",
            },
        )
        assert p.name == "MyVLLM"


# ────────────────────────────────────────────────────────────────────
# Construction safety — no network calls in __init__
# ────────────────────────────────────────────────────────────────────

class TestConstructionCheap:
    """All providers must be construction-cheap (no HTTP at __init__)."""

    def test_venice_session_no_token_still_constructs(self):
        # Missing session_token logs a warning but does not raise.
        p = VeniceSessionProvider({})
        assert isinstance(p, InferenceProvider)

    def test_all_providers_construct_with_empty_cfg(self):
        # Each provider must tolerate empty cfg (returns "" from chat()
        # on bad config rather than raising at construction time).
        for name in ["venice", "venice_session", "openrouter",
                     "ollama_cloud", "custom"]:
            p = get_provider(name, {})
            assert isinstance(p, InferenceProvider)


# ────────────────────────────────────────────────────────────────────
# get_agno_model() — returns the right agno class
# ────────────────────────────────────────────────────────────────────

class TestGetAgnoModel:
    """Each provider returns an agno model wrapper for Agent use."""

    def test_venice_returns_openai_like(self):
        from agno.models.openai.like import OpenAILike
        p = get_provider("venice", {"venice_api_key": "k"})
        m = p.get_agno_model()
        assert isinstance(m, OpenAILike)
        assert m.name == "VeniceAI"

    def test_openrouter_returns_native_openrouter(self):
        from agno.models.openrouter import OpenRouter
        p = get_provider(
            "openrouter",
            {"openrouter_api_key": "k", "max_tokens": 2048},
        )
        m = p.get_agno_model()
        assert isinstance(m, OpenRouter)

    def test_ollama_cloud_returns_openai_like(self):
        from agno.models.openai.like import OpenAILike
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})
        m = p.get_agno_model()
        assert isinstance(m, OpenAILike)
        assert m.name == "OllamaCloud"

    def test_custom_returns_openai_like(self):
        from agno.models.openai.like import OpenAILike
        p = get_provider("custom", {"custom_llm_api_key": "k"})
        m = p.get_agno_model()
        assert isinstance(m, OpenAILike)

    def test_venice_session_returns_openai_like_with_refresher_attached(self):
        from agno.models.openai.like import OpenAILike
        p = get_provider(
            "venice_session",
            {"venice_session_token": "t", "venice_client_cookie": "c"},
        )
        m = p.get_agno_model()
        assert isinstance(m, OpenAILike)
        assert hasattr(m, "_venice_refresher")

    def test_venice_session_caches_agno_model(self):
        """get_agno_model() returns the same instance on repeat calls."""
        p = get_provider(
            "venice_session", {"venice_session_token": "t"},
        )
        m1 = p.get_agno_model()
        m2 = p.get_agno_model()
        assert m1 is m2


# ────────────────────────────────────────────────────────────────────
# Convenience wrappers — complete() + distill() routes via chat()
# ────────────────────────────────────────────────────────────────────

class _ChatStub:
    """Captures chat() call args + returns canned text."""

    def __init__(self, return_text: str = "stub response"):
        self.return_text = return_text
        self.last_messages: list[dict] = []
        self.last_kwargs: dict = {}
        self.call_count = 0

    async def __call__(
        self, messages, *, model=None, temperature=0.7,
        max_tokens=4096, timeout=60.0,
    ):
        self.last_messages = messages
        self.last_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        self.call_count += 1
        return self.return_text


class TestConvenienceWrappers:
    """complete() and distill() route through chat() with correct shape."""

    @pytest.mark.parametrize("provider_name,cfg", [
        ("venice", {"venice_api_key": "k"}),
        ("openrouter", {"openrouter_api_key": "k"}),
        ("ollama_cloud", {"ollama_cloud_api_key": "k"}),
        ("custom", {"custom_llm_api_key": "k"}),
    ])
    def test_complete_calls_chat_with_user_message(
        self, provider_name, cfg, monkeypatch,
    ):
        p = get_provider(provider_name, cfg)
        stub = _ChatStub("hello back")
        monkeypatch.setattr(p, "chat", stub)
        result = asyncio.run(p.complete("hello"))
        assert result == "hello back"
        assert stub.last_messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.parametrize("provider_name,cfg", [
        ("venice", {"venice_api_key": "k"}),
        ("openrouter", {"openrouter_api_key": "k"}),
        ("ollama_cloud", {"ollama_cloud_api_key": "k"}),
        ("custom", {"custom_llm_api_key": "k"}),
    ])
    def test_complete_with_system_includes_system_message(
        self, provider_name, cfg, monkeypatch,
    ):
        p = get_provider(provider_name, cfg)
        stub = _ChatStub()
        monkeypatch.setattr(p, "chat", stub)
        asyncio.run(p.complete("hi", system="be terse"))
        assert stub.last_messages[0] == {"role": "system", "content": "be terse"}
        assert stub.last_messages[1] == {"role": "user", "content": "hi"}

    def test_distill_uses_instruction_in_prompt(self, monkeypatch):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})
        stub = _ChatStub("distilled")
        monkeypatch.setattr(p, "chat", stub)
        result = asyncio.run(
            p.distill("a long document",
                      instruction="Extract the key claims")
        )
        assert result == "distilled"
        user_msg = stub.last_messages[0]
        assert "Extract the key claims" in user_msg["content"]
        assert "a long document" in user_msg["content"]


# ────────────────────────────────────────────────────────────────────
# OllamaCloud-specific: model routing + score() + backoff
# ────────────────────────────────────────────────────────────────────

class TestOllamaCloudModelRouting:
    """TASK_MODEL_MAP preserved verbatim from legacy utils/ollama_cloud.py."""

    def test_get_model_for_task_known(self):
        # Tasks known in the map return the configured model
        m = get_model_for_task("haiku")
        assert isinstance(m, str)
        assert len(m) > 0

    def test_get_model_for_task_unknown_returns_default(self):
        m_unknown = get_model_for_task("totally-made-up-task")
        m_default = get_model_for_task("haiku")
        assert m_unknown == m_default  # both fall to light tier

    def test_score_clamps_to_unit_range(self, monkeypatch):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})

        async def fake_complete(*args, **kwargs):
            return "0.42"

        monkeypatch.setattr(p, "complete", fake_complete)
        result = asyncio.run(p.score("Rate this 0-1"))
        assert result == 0.42

    def test_score_out_of_range_clamps(self, monkeypatch):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})

        async def fake_complete(*args, **kwargs):
            return "1.7"

        monkeypatch.setattr(p, "complete", fake_complete)
        result = asyncio.run(p.score("Rate this"))
        assert result == 1.0

    def test_score_unparseable_returns_default_half(self, monkeypatch):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})

        async def fake_complete(*args, **kwargs):
            return "I don't know"

        monkeypatch.setattr(p, "complete", fake_complete)
        result = asyncio.run(p.score("Rate this"))
        assert result == 0.5


# ────────────────────────────────────────────────────────────────────
# Stats surface
# ────────────────────────────────────────────────────────────────────

class TestStats:
    """get_stats() returns provider-shaped dict; safe to call at boot."""

    def test_ollama_cloud_stats_initial(self):
        p = get_provider("ollama_cloud", {"ollama_cloud_api_key": "k"})
        s = p.get_stats()
        assert s["provider"] == "OllamaCloud"
        assert s["total_requests"] == 0
        assert s["consecutive_failures"] == 0
        assert s["in_backoff"] is False
        assert isinstance(s["request_counts"], dict)
        assert isinstance(s["total_tokens"], dict)

    def test_venice_stats_initial(self):
        p = get_provider("venice", {"venice_api_key": "k"})
        s = p.get_stats()
        assert s["provider"] == "VeniceAI"
        assert s["requests"] == 0
        assert s["errors"] == 0
        assert s["last_error"] is None

    def test_openrouter_stats_initial(self):
        p = get_provider("openrouter", {"openrouter_api_key": "k"})
        s = p.get_stats()
        assert s["provider"] == "OpenRouter"
        assert s["requests"] == 0

    def test_custom_stats_initial(self):
        p = get_provider(
            "custom",
            {"custom_llm_api_key": "k", "custom_provider_name": "MyLLM"},
        )
        s = p.get_stats()
        assert s["provider"] == "MyLLM"
        assert s["requests"] == 0

    def test_venice_session_stats_includes_session_metadata(self):
        p = get_provider("venice_session", {"venice_session_token": "t"})
        s = p.get_stats()
        assert "configured" in s
        assert "auto_refresh" in s
        assert "requests" in s


# ────────────────────────────────────────────────────────────────────
# Public surface — __all__ exports the right names
# ────────────────────────────────────────────────────────────────────

class TestPublicSurface:
    """The library's public exports should be stable and complete."""

    def test_all_lists_canonical_exports(self):
        for name in [
            "InferenceProvider",
            "VeniceProvider", "VeniceSessionProvider",
            "OpenRouterProvider", "OllamaCloudProvider", "CustomProvider",
            "get_provider", "TASK_MODEL_MAP", "get_model_for_task",
        ]:
            assert name in inference.__all__, f"missing: {name}"

    def test_get_provider_callable_from_module(self):
        assert callable(inference.get_provider)
        assert callable(inference.get_model_for_task)
        assert isinstance(inference.TASK_MODEL_MAP, dict)
