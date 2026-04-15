"""
utils/ollama_cloud.py
Thin async client for Ollama Cloud API (OpenAI-compatible).

Replaces local Ollama (phi3:mini) for all internal operations:
- Meditation memory scoring
- Guardian Tier 3 LLM veto
- Haiku / reflection generation
- Document distillation
- Skill validation LLM analysis
- Research distillation fallback

Uses smart model routing: task complexity → model tier.
"""
import json
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart Model Routing — map task complexity to model tier
# ---------------------------------------------------------------------------
# Model names read from config.toml [inference] section:
#   ollama_cloud_light_model  — fast tasks (scoring, haiku, teacher)
#   ollama_cloud_heavy_model  — complex tasks (research, code gen)
# Fallback defaults if config not available.

_LIGHT_MODEL = "gemma4:31b"
_MEDIUM_MODEL = "ministral-3:8b"
_HEAVY_MODEL = "deepseek-v3.1:671b"

# Load from config.toml if available
try:
    import tomllib
    from pathlib import Path
    _cfg_path = Path(__file__).parent.parent / "config.toml"
    if _cfg_path.exists():
        with open(_cfg_path, "rb") as _f:
            _cfg = tomllib.load(_f)
        _inf = _cfg.get("inference", {})
        _LIGHT_MODEL = _inf.get("ollama_cloud_light_model", _LIGHT_MODEL)
        _HEAVY_MODEL = _inf.get("ollama_cloud_heavy_model", _HEAVY_MODEL)
except Exception:
    pass  # Use defaults

# Task → model tier mapping
TASK_MODEL_MAP: dict[str, str] = {
    # Light — scoring, policy checks, haiku
    "meditation_scoring": _LIGHT_MODEL,
    "policy_check": _LIGHT_MODEL,
    "haiku": _LIGHT_MODEL,
    "skill_validation": _LIGHT_MODEL,
    "art_title": _LIGHT_MODEL,
    "language_teacher": _LIGHT_MODEL,
    # Medium — guardian analysis, social synthesis
    "guardian_veto": _MEDIUM_MODEL,
    "social_synthesis": _MEDIUM_MODEL,
    "document_distill": _MEDIUM_MODEL,
    # Heavy — research distillation, cognify, code gen
    "research_distill": _HEAVY_MODEL,
    "cognee_cognify": _HEAVY_MODEL,
    "agency_code_gen": _HEAVY_MODEL,
}

_DEFAULT_MODEL = _LIGHT_MODEL


def get_model_for_task(task_name: str) -> str:
    """Return the appropriate model for a given task name."""
    return TASK_MODEL_MAP.get(task_name, _DEFAULT_MODEL)


class OllamaCloudClient:
    """
    Async client for Ollama Cloud API (OpenAI-compatible chat/completions).

    Tracks per-model request counts and token usage for observability.
    """

    # Exponential backoff: 2→4→8→16→30s max, reset on success
    _BACKOFF_BASE = 2.0
    _BACKOFF_MAX = 30.0
    _BACKOFF_MAX_FAILURES = 8  # cap at 8 consecutive failures

    def __init__(self, api_key: str, base_url: str = "https://ollama.com/v1"):
        self._api_key = api_key
        # Normalize: api.ollama.com redirects to ollama.com (301)
        self._base_url = base_url.rstrip("/").replace("://api.ollama.com", "://ollama.com")
        self._request_counts: dict[str, int] = {}   # model -> count
        self._total_tokens: dict[str, int] = {}      # model -> tokens
        # Backoff state
        self._consecutive_failures = 0
        self._backoff_until = 0.0  # timestamp when backoff expires

    @property
    def request_counts(self) -> dict[str, int]:
        return dict(self._request_counts)

    @property
    def total_tokens(self) -> dict[str, int]:
        return dict(self._total_tokens)

    def _track(self, model: str, tokens: int = 0):
        """Increment request counter and token usage for a model."""
        self._request_counts[model] = self._request_counts.get(model, 0) + 1
        if tokens > 0:
            self._total_tokens[model] = self._total_tokens.get(model, 0) + tokens

    async def complete(
        self,
        prompt: str,
        model: str = "gemma4:31b",
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 500,
        timeout: float = 60.0,
    ) -> str:
        """
        Single-turn chat completion. Returns the response text.

        Args:
            prompt: User message content.
            model: Ollama Cloud model name.
            system: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: HTTP request timeout in seconds.

        Returns:
            Response text string, or "" on failure.
        """
        # Exponential backoff: skip if in cooldown
        now = time.time()
        if now < self._backoff_until:
            remaining = self._backoff_until - now
            logger.debug(
                "[OllamaCloud] Backoff active — %.1fs remaining (%d failures)",
                remaining, self._consecutive_failures)
            return ""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
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

            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            total_tok = usage.get("total_tokens", 0)
            self._track(model, total_tok)

            # Success — reset backoff
            if self._consecutive_failures > 0:
                logger.info("[OllamaCloud] Recovered after %d failures",
                            self._consecutive_failures)
            self._consecutive_failures = 0
            self._backoff_until = 0.0

            logger.debug(
                "[OllamaCloud] %s completed (%d tokens, %d chars)",
                model, total_tok, len(content),
            )
            return content

        except Exception as e:
            self._consecutive_failures += 1
            # Exponential backoff: 2, 4, 8, 16, 30s max
            delay = min(
                self._BACKOFF_BASE ** min(self._consecutive_failures,
                                          self._BACKOFF_MAX_FAILURES),
                self._BACKOFF_MAX)
            self._backoff_until = time.time() + delay
            logger.warning(
                "[OllamaCloud] complete() failed (model=%s, attempt=%d, "
                "backoff=%.0fs): %s",
                model, self._consecutive_failures, delay, e)
            self._track(model, 0)
            return ""

    async def score(
        self,
        prompt: str,
        model: str = "gemma4:31b",
        timeout: float = 30.0,
    ) -> float:
        """
        Ask the LLM to rate something 0.0-1.0. Parses float from response.

        Args:
            prompt: Scoring prompt that instructs the LLM to return a number.
            model: Ollama Cloud model name.
            timeout: HTTP request timeout.

        Returns:
            Float score between 0.0 and 1.0, or 0.5 as fallback.
        """
        response = await self.complete(
            prompt=prompt,
            model=model,
            system="You are a scoring system. Respond with ONLY a single decimal number between 0.0 and 1.0.",
            temperature=0.1,
            max_tokens=20,
            timeout=timeout,
        )

        if not response:
            return 0.5

        # Parse the first float found in the response
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))

        return 0.5

    def get_stats(self) -> dict:
        """Return usage statistics for the Observatory API."""
        return {
            "base_url": self._base_url,
            "request_counts": self.request_counts,
            "total_tokens": self.total_tokens,
            "total_requests": sum(self._request_counts.values()),
        }
