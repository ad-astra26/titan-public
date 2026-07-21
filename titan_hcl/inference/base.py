"""
titan_hcl.inference.base — abstract InferenceProvider contract.

Stable surface that every provider implementation must satisfy. Workers
import this contract via `titan_hcl.inference.get_provider(name, cfg)`
and program against `InferenceProvider`, never against a concrete subclass.

Adding a new provider:
    1. Drop `titan_hcl/inference/<name>.py` with a subclass implementing
       at least `chat()`, `stream_chat()`, `id`, `name`, `base_url`.
    2. Register it in `titan_hcl/inference/__init__.py:_PROVIDER_MAP`.
    3. (Optional) Implement `get_agno_model()` if the provider should be
       usable as an Agno Agent's `model=` parameter (needed for agno_worker).

Documented in SPEC §9.C.1 (D-SPEC-72).
"""
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Optional


class InferenceProvider(abc.ABC):
    """Abstract base class for all LLM inference providers.

    Concrete subclasses live as one file per provider under
    `titan_hcl/inference/`. Subclasses MUST be construction-cheap
    (no network calls in __init__) so workers can instantiate them
    eagerly at boot without blocking on remote endpoints.
    """

    # ── Identity (subclasses populate via @property or set in __init__) ──

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Model identifier (e.g. 'llama-3.3-70b', 'gemma4:31b')."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'VeniceAI', 'OllamaCloud')."""

    @property
    @abc.abstractmethod
    def base_url(self) -> str:
        """Base URL of the upstream API (with no trailing slash)."""

    # ── Core inference surface (subclasses MUST implement) ──

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> str:
        """OpenAI-style chat completion.

        Args:
            messages: list of {"role": str, "content": str} per OpenAI schema.
            model: override provider's default model. None → provider default.
            temperature: sampling temperature 0.0-2.0.
            max_tokens: maximum tokens in response.
            timeout: HTTP request timeout in seconds.

        Returns:
            Response content string. Empty string on provider error
            (subclasses log the error; callers MAY check via get_stats()).
        """

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> AsyncIterator[str]:
        """Streaming chat completion — async iterator yielding content chunks.

        Subclasses implement as `async def` with `yield` (async generator).
        Callers iterate with `async for chunk in provider.stream_chat(...)`.

        Returns chunks of generated text in arrival order. Final chunk may
        contain end-of-stream sentinel (provider-specific); iterator
        terminates naturally when the upstream stream closes.
        """

    # ── Convenience wrappers (subclasses MAY override for efficiency) ──

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
        timeout: float = 60.0,
    ) -> str:
        """Single-turn convenience wrapper over chat().

        Used by workers needing scoring / classification / haiku generation
        where a full messages-list construction is overkill.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def distill(
        self,
        text: str,
        *,
        instruction: str = "Summarize concisely",
        model: Optional[str] = None,
        max_tokens: int = 500,
        timeout: float = 60.0,
    ) -> str:
        """Document distillation convenience wrapper over complete().

        Used by studio_worker haiku Tier-1, language_teacher, research
        distillation, etc. Routes via `bus.QUERY action="distill"` against
        llm_worker today; agno_worker may call directly post-D-SPEC-72.
        """
        return await self.complete(
            prompt=f"{instruction}:\n\n{text}",
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # ── Model-class resolution (provider-agnostic tier routing) ──

    def resolve_model_class(self, model_class: str) -> str:
        """Map abstract model-class name → concrete provider model ID.

        Used by `[chat.tiers]` so the chat layer can specify abstract roles
        (e.g. "fast" for greetings, "heavy" for reasoning) without hard-coding
        provider-specific model IDs. Each provider populates `_model_class_map`
        from its own `[inference]` config keys (e.g. `ollama_cloud_fast_model`,
        `venice_heavy_model`).

        Args:
            model_class: abstract role name. Conventional values are "fast",
                "light", "heavy" but providers MAY support arbitrary classes.

        Returns:
            Concrete model ID string passable to `chat(model=...)`. Falls
            back to the provider's default model (`self.id`) when the class
            is not configured.
        """
        return getattr(self, "_model_class_map", {}).get(model_class, self.id)

    # ── Agno integration (optional — only providers used by agno_worker) ──

    def get_agno_model(self) -> Any:
        """Return an Agno-compatible model instance.

        agno_worker constructs its Agent with `model=provider.get_agno_model()`.
        Providers that don't implement this raise NotImplementedError; only
        providers configured for the chat agent path need to implement it.

        Returns:
            An object satisfying Agno's model interface (typically
            `agno.models.openai.like.OpenAILike` or a native class like
            `agno.models.openrouter.OpenRouter`).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide an Agno model wrapper. "
            f"Override get_agno_model() in the subclass to enable agno_worker use."
        )

    # ── Stats (subclasses SHOULD override) ──

    def get_stats(self) -> dict[str, Any]:
        """Return per-provider usage stats for Observatory + health monitor.

        Default: empty dict. Subclasses tracking request counts / token
        usage / error rates / backoff state override to expose them.
        """
        return {}
