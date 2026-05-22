"""
titan_hcl.inference — Pluggable LLM inference provider abstraction.

Canonical home for ALL provider plumbing in the Titan fleet. Workers that
need LLM inference import a provider via `get_provider(name, cfg)` and
program against the `InferenceProvider` ABC — never against a concrete
provider class.

Adding a new provider (Anthropic, Mistral, local vLLM, etc.):
    1. Drop `titan_hcl/inference/<name>.py` with a `InferenceProvider`
       subclass implementing chat() + stream_chat() + id/name/base_url.
    2. Add one row to _PROVIDER_MAP below.
    3. Done — no worker code changes.

See SPEC §9.C.1 (D-SPEC-72) for the canonical library contract.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import InferenceProvider
from .venice import VeniceProvider, VeniceSessionProvider
from .openrouter import OpenRouterProvider
from .ollama_cloud import OllamaCloudProvider, TASK_MODEL_MAP, get_model_for_task
from .custom import CustomProvider

__all__ = [
    "InferenceProvider",
    "VeniceProvider",
    "VeniceSessionProvider",
    "OpenRouterProvider",
    "OllamaCloudProvider",
    "CustomProvider",
    "get_provider",
    "TASK_MODEL_MAP",
    "get_model_for_task",
]


# Registry — name → constructor callable taking (cfg: dict) → InferenceProvider
_PROVIDER_MAP: dict[str, Callable[[dict[str, Any]], InferenceProvider]] = {
    "venice": VeniceProvider,
    "venice_session": VeniceSessionProvider,
    "openrouter": OpenRouterProvider,
    "ollama_cloud": OllamaCloudProvider,
    "custom": CustomProvider,
}


def get_provider(name: str, cfg: dict[str, Any]) -> InferenceProvider:
    """Factory — construct a provider instance by name.

    Args:
        name: provider key ('venice', 'venice_session', 'openrouter',
              'ollama_cloud', 'custom').
        cfg:  config dict (typically the merged [inference] + [agent] block
              from config.toml). Each provider reads only the keys it needs;
              missing keys fall back to documented defaults.

    Returns:
        Constructed InferenceProvider subclass instance, ready for chat()
        / stream_chat() / get_agno_model() calls.

    Raises:
        ValueError: if `name` is not a registered provider. Adding a new
                    provider is one file + one _PROVIDER_MAP row — see
                    titan_hcl/inference/__init__.py module docstring.
    """
    if name not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown inference provider: '{name}'. "
            f"Known providers: {sorted(_PROVIDER_MAP.keys())}. "
            f"To add a new provider, see titan_hcl/inference/__init__.py."
        )
    return _PROVIDER_MAP[name](cfg)
