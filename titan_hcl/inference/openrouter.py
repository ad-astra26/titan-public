"""
titan_hcl.inference.openrouter — OpenRouter inference provider.

Lifts `agent.py:_build_openrouter` into the provider abstraction. Used
for Agno chat agent paths when configured; direct HTTP inference is
also exposed via the standard chat() / stream_chat() surface.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from .base import InferenceProvider

logger = logging.getLogger(__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterProvider(InferenceProvider):
    """OpenRouter inference provider — multi-model gateway."""

    def __init__(self, cfg: dict[str, Any]):
        """Build from [inference] config block.

        Required: openrouter_api_key (or api_key).
        Optional: openrouter_model_id (default 'meta-llama/llama-3.3-70b-instruct'),
                 max_tokens (default 4096).
        """
        self._api_key = cfg.get("openrouter_api_key", cfg.get("api_key", ""))
        self._model = cfg.get("openrouter_model_id", _DEFAULT_MODEL)
        self._max_tokens = int(cfg.get("max_tokens", 4096))
        # Phase 2 Chunk ζ.0 (D-SPEC-79, 2026-05-18) — model-class registry.
        self._model_class_map = {
            "fast":  cfg.get("openrouter_fast_model",  self._model),
            "light": cfg.get("openrouter_light_model", self._model),
            "heavy": cfg.get("openrouter_heavy_model", self._model),
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
        return "OpenRouter"

    @property
    def base_url(self) -> str:
        return OPENROUTER_API_BASE

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
                    f"{OPENROUTER_API_BASE}/chat/completions",
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
            logger.warning("[OpenRouter] chat() failed (model=%s): %s",
                           use_model, e)
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
                    f"{OPENROUTER_API_BASE}/chat/completions",
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
            logger.warning("[OpenRouter] stream_chat() failed (model=%s): %s",
                           use_model, e)

    def get_agno_model(self) -> Any:
        from agno.models.openrouter import OpenRouter
        return OpenRouter(
            id=self._model,
            api_key=self._api_key,
            max_tokens=self._max_tokens,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": OPENROUTER_API_BASE,
            "model": self._model,
            "requests": self._request_count,
            "errors": self._error_count,
            "last_error": self._last_error,
        }
