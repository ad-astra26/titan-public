"""
titan_hcl.inference.venice — Venice AI provider (API key + session cookie variants).

Two subclasses share the OpenAI-compatible Venice API surface:
  - VeniceProvider          → static API key authentication (paid credits)
  - VeniceSessionProvider   → Pro plan session cookie (rotates JWT via Clerk)

VeniceSessionProvider wraps `inference/venice_session.py:VeniceSessionClient`
for direct HTTP inference. For Agno Agent use (chat agent path), Venice
returns an `OpenAILike` model wrapper; the Pro-plan variant keeps a
background token refresher alive that updates the wrapper's api_key.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from .base import InferenceProvider
from .venice_session import VeniceSessionClient

logger = logging.getLogger(__name__)

VENICE_API_BASE = "https://api.venice.ai/api/v1"
_DEFAULT_MODEL = "llama-3.3-70b"


class VeniceProvider(InferenceProvider):
    """Venice AI inference via static API key (paid credits)."""

    def __init__(self, cfg: dict[str, Any]):
        """Build from [inference] config block.

        Required: venice_api_key (or api_key).
        Optional: venice_model_id (default 'llama-3.3-70b'),
                 venice_base_url (default Venice API endpoint).
        """
        self._api_key = cfg.get("venice_api_key", cfg.get("api_key", ""))
        self._model = cfg.get("venice_model_id", _DEFAULT_MODEL)
        self._base_url = cfg.get("venice_base_url", VENICE_API_BASE).rstrip("/")
        # Phase 2 Chunk ζ.0 (D-SPEC-79, 2026-05-18) — model-class registry.
        self._model_class_map = {
            "fast":  cfg.get("venice_fast_model",  self._model),
            "light": cfg.get("venice_light_model", self._model),
            "heavy": cfg.get("venice_heavy_model", self._model),
        }
        self._request_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._cfg = cfg  # retained for get_agno_model()

    @property
    def id(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return "VeniceAI"

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
        import httpx
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
            logger.warning("[Venice] chat() failed (model=%s): %s", use_model, e)
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
        import json
        import httpx
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
            logger.warning("[Venice] stream_chat() failed (model=%s): %s",
                           use_model, e)

    def get_agno_model(self) -> Any:
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=self._model,
            name=self.name,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self._base_url,
            "model": self._model,
            "requests": self._request_count,
            "errors": self._error_count,
            "last_error": self._last_error,
        }


class VeniceSessionProvider(InferenceProvider):
    """Venice Pro plan via Clerk session cookie (no API credits charged).

    Uses `VeniceSessionClient` for both direct HTTP inference (chat/distill)
    and as the underlying refresher when get_agno_model() is called.
    Background refresh loop keeps the JWT alive (~60s TTL).
    """

    def __init__(self, cfg: dict[str, Any]):
        """Build from [inference] config block.

        Required: venice_session_token (the __session JWT).
        Optional: venice_client_cookie (the __client cookie for auto-refresh),
                 venice_model_id (default 'llama-3.3-70b').
        """
        self._session_token = cfg.get("venice_session_token", "")
        self._client_cookie = cfg.get("venice_client_cookie", "")
        self._model = cfg.get("venice_model_id", _DEFAULT_MODEL)
        # Phase 2 Chunk ζ.0 (D-SPEC-79, 2026-05-18) — model-class registry.
        self._model_class_map = {
            "fast":  cfg.get("venice_fast_model",  self._model),
            "light": cfg.get("venice_light_model", self._model),
            "heavy": cfg.get("venice_heavy_model", self._model),
        }
        self._cfg = cfg  # retained for get_agno_model()

        if not self._session_token:
            logger.warning(
                "[VeniceSession] No venice_session_token in config — "
                "provider will return empty responses until configured"
            )

        # Construction-cheap — VeniceSessionClient does not make network calls.
        self._client = VeniceSessionClient(
            session_token=self._session_token,
            client_cookie=self._client_cookie,
            model=self._model,
        )
        # Agno model wrapper + refresher task are created lazily in
        # get_agno_model() — workers that only use direct HTTP inference
        # never pay the agno wrapper cost.
        self._agno_model: Optional[Any] = None
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def id(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return "VeniceSession"

    @property
    def base_url(self) -> str:
        return VENICE_API_BASE

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> str:
        result = await self._client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if "error" in result:
            return ""
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
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
        # VeniceSessionClient does not currently expose a stream API.
        # Fall back to non-streaming and yield the entire response as one
        # chunk so callers using `async for` still work. When Venice
        # streaming is wired into the session client, this delegates.
        full = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if full:
            yield full

    def get_agno_model(self) -> Any:
        """Return Agno OpenAILike model + start background token refresher.

        The refresher updates `model.api_key` every ~45s by calling
        VeniceSessionClient._refresh_token. Agno Agent picks up the new
        key on its next request (we clear cached async_client + client
        attrs so the connection is rebuilt with the fresh JWT).
        """
        if self._agno_model is not None:
            return self._agno_model

        from agno.models.openai.like import OpenAILike
        model = OpenAILike(
            id=self._model,
            name=self.name,
            api_key=self._session_token,
            base_url=VENICE_API_BASE,
        )
        # Stash refresher reference on the model object for stats endpoint
        # parity with the legacy agent.py:_build_venice_session behaviour.
        model._venice_refresher = self._client  # type: ignore[attr-defined]
        self._agno_model = model

        if self._session_token and self._client_cookie:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._refresh_task = asyncio.ensure_future(
                        self._refresh_loop(model)
                    )
                else:
                    # Will be started by plugin boot sequence; stash coroutine
                    model._venice_refresh_coro = self._refresh_loop  # type: ignore[attr-defined]
            except RuntimeError:
                model._venice_refresh_coro = self._refresh_loop  # type: ignore[attr-defined]
            logger.info(
                "[VeniceSession] Auto-refresh configured (client_cookie set)"
            )
        elif self._session_token:
            logger.warning(
                "[VeniceSession] Session token set but no client_cookie — "
                "no auto-refresh"
            )

        return model

    async def _refresh_loop(self, model: Any) -> None:
        """Background loop refreshing the session token every 45s."""
        while True:
            await asyncio.sleep(45)
            try:
                if self._client._is_token_expired():
                    ok = await self._client._refresh_token()
                    if ok:
                        model.api_key = self._client._session_token
                        # Clear cached clients so next request uses new key
                        if (
                            hasattr(model, "async_client")
                            and model.async_client is not None
                        ):
                            try:
                                await model.async_client.close()
                            except Exception:
                                pass
                            model.async_client = None
                        if hasattr(model, "client") and model.client is not None:
                            try:
                                model.client.close()
                            except Exception:
                                pass
                            model.client = None
                        logger.info(
                            "[VeniceSession] Token refreshed for Agno model"
                        )
            except Exception as e:
                logger.error("[VeniceSession] Token refresh error: %s", e)

    def get_stats(self) -> dict[str, Any]:
        return self._client.stats
