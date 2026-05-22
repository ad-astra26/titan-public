"""
titan_hcl.inference.custom — Generic OpenAI-compatible provider.

Escape hatch for any OpenAI-compatible endpoint (self-hosted vLLM,
LM Studio, llama.cpp server, Together, Anyscale, Fireworks, etc.)
that wasn't worth a dedicated provider file. Lifts `agent.py:_build_custom`
into the provider abstraction.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from .base import InferenceProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class CustomProvider(InferenceProvider):
    """Generic OpenAI-compatible provider for endpoints without dedicated impl."""

    def __init__(self, cfg: dict[str, Any]):
        """Build from [inference] config block.

        Required: custom_llm_api_key (or api_key).
        Optional: custom_model_id (default 'gpt-4o'),
                 custom_base_url (default OpenAI endpoint).
        """
        self._api_key = cfg.get("custom_llm_api_key", cfg.get("api_key", ""))
        self._model = cfg.get("custom_model_id", _DEFAULT_MODEL)
        self._base_url = cfg.get(
            "custom_base_url", _DEFAULT_BASE_URL
        ).rstrip("/")
        self._provider_name = cfg.get("custom_provider_name", "CustomLLM")
        # Phase 2 Chunk ζ.0 (D-SPEC-79, 2026-05-18) — model-class registry.
        self._model_class_map = {
            "fast":  cfg.get("custom_fast_model",  self._model),
            "light": cfg.get("custom_light_model", self._model),
            "heavy": cfg.get("custom_heavy_model", self._model),
        }
        self._request_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._cfg = cfg

    @property
    def id(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def base_url(self) -> str:
        return self._base_url

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> str:
        use_model = model or self._model
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            self._request_count += 1
            self._last_error = None
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.warning(
                "[%s] chat() failed (model=%s): %s",
                self._provider_name, use_model, e,
            )
            return ""

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> AsyncIterator[str]:
        use_model = model or self._model
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        if not data_str:
                            continue
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            self._request_count += 1
            self._last_error = None
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.warning(
                "[%s] stream_chat() failed: %s",
                self._provider_name, e,
            )

    def get_agno_model(self) -> Any:
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=self._model,
            name=self._provider_name,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self._provider_name,
            "base_url": self._base_url,
            "model": self._model,
            "requests": self._request_count,
            "errors": self._error_count,
            "last_error": self._last_error,
        }
