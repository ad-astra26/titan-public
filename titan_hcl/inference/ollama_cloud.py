"""
titan_hcl.inference.ollama_cloud — Ollama Cloud provider.

Async client for Ollama Cloud API (OpenAI-compatible). Used by
llm_worker (canonical inference RPC service) + agno_worker (when
configured as the chat provider).

Consolidates the previous `titan_hcl/utils/ollama_cloud.py` into
the new provider abstraction. Adds:
  - OpenAI-style chat() (messages list) — was missing in the previous
    OllamaCloudClient class; llm_worker:130 was calling .chat() against
    a class that only had .complete(), so the action="chat" QUERY path
    was dead. Standardising on OpenAI messages-list shape fixes that.
  - stream_chat() — server-sent events streaming for SSE relay.
  - get_agno_model() — returns an `OpenAILike` instance for Agno Agent.
  - get_stats() — surfaces request counts + token usage + backoff state.

Smart task→model routing (TASK_MODEL_MAP) is preserved verbatim.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator, Any, Optional

import httpx

from .base import InferenceProvider
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)

# ── Smart Model Routing — task complexity → model tier ──
# Model names read from config.toml [inference] section:
#   ollama_cloud_light_model  — fast tasks (scoring, haiku, teacher)
#   ollama_cloud_heavy_model  — complex tasks (research, code gen)
# Fallback defaults if config not available at import time.

_LIGHT_MODEL = "gemma4:31b"
_MEDIUM_MODEL = "ministral-3:8b"
_HEAVY_MODEL = "deepseek-v3.1:671b"
# Dedicated language-teacher model (2026-05-30). qwen3-next:80b was tried but the
# 80B is too slow on Ollama Cloud — chat() times out at >30s / returns 0 chars,
# breaking the teacher. gemma4:31b is fast + reliable, and PATH 3 (giving the model
# a specific new word to use, language_teacher._pick_new_word_to_teach) removes the
# need for a stronger instruction-follower — the model only has to use the given
# word in a sentence. Config-overridable via [inference] ollama_cloud_teacher_model;
# gpt-oss:20b is a faster strong-instruction alternative if richer teaching is wanted.
_TEACHER_MODEL = "gemma4:31b"

try:
    _inf = get_params("inference")
    _LIGHT_MODEL = _inf.get("ollama_cloud_light_model", _LIGHT_MODEL)
    _HEAVY_MODEL = _inf.get("ollama_cloud_heavy_model", _HEAVY_MODEL)
    _TEACHER_MODEL = _inf.get("ollama_cloud_teacher_model", _TEACHER_MODEL)
except Exception:
    pass  # Use defaults

TASK_MODEL_MAP: dict[str, str] = {
    # Light — scoring, policy checks, haiku
    "meditation_scoring": _LIGHT_MODEL,
    "policy_check": _LIGHT_MODEL,
    "haiku": _LIGHT_MODEL,
    "skill_validation": _LIGHT_MODEL,
    "art_title": _LIGHT_MODEL,
    "language_teacher": _TEACHER_MODEL,   # dedicated — stronger new-word instruction-following
    "meta_teacher": _LIGHT_MODEL,
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
    """Return the appropriate model for a given task name.

    Public — used by callers across the codebase (language_teacher,
    studio Tier-1 haiku, meditation scoring, etc.) to pick the right
    model tier without hardcoding model strings.
    """
    return TASK_MODEL_MAP.get(task_name, _DEFAULT_MODEL)


class OllamaCloudProvider(InferenceProvider):
    """Ollama Cloud inference provider — OpenAI-compatible chat/completions API.

    Tracks per-model request counts + token usage for observability.
    Implements exponential backoff (2→4→8→16→30s max, reset on success)
    to absorb transient failures without amplifying load on the upstream.

    Construction is cheap: no network calls in __init__, so workers can
    boot eagerly. Connection only opens at first chat()/stream_chat() call.
    """

    # Exponential backoff: 2→4→8→16→30s max, reset on success
    _BACKOFF_BASE = 2.0
    _BACKOFF_MAX = 30.0
    _BACKOFF_MAX_FAILURES = 8  # cap at 8 consecutive failures

    def __init__(self, cfg: dict[str, Any]):
        """Build provider from [inference] config section.

        Args:
            cfg: dict — typically the merged inference+agent config block.
                Required: ollama_cloud_api_key OR api_key.
                Optional: ollama_cloud_base_url (default 'https://ollama.com/v1'),
                         ollama_cloud_chat_model (default 'deepseek-v3.1:671b').
        """
        self._api_key = cfg.get("ollama_cloud_api_key", cfg.get("api_key", ""))
        base_url = cfg.get(
            "ollama_cloud_base_url", "https://ollama.com/v1"
        )
        # Normalize: api.ollama.com redirects to ollama.com (301)
        self._base_url = base_url.rstrip("/").replace(
            "://api.ollama.com", "://ollama.com"
        )
        # Agno chat path uses deepseek-v3.1:671b (highest quality available);
        # internal RPC callers pass model= explicitly via get_model_for_task.
        self._chat_model = cfg.get(
            "ollama_cloud_chat_model", _HEAVY_MODEL
        )
        # Phase 2 Chunk ζ.0 (D-SPEC-79, 2026-05-18) — provider-agnostic model
        # class registry. [chat.tiers] specifies abstract roles ("fast" / "light"
        # / "heavy"); this map resolves them to provider-concrete model IDs.
        self._model_class_map = {
            "fast":  cfg.get("ollama_cloud_fast_model",  _LIGHT_MODEL),
            "light": cfg.get("ollama_cloud_light_model", _LIGHT_MODEL),
            "heavy": cfg.get("ollama_cloud_heavy_model", _HEAVY_MODEL),
        }
        self._cfg = cfg  # retained for get_agno_model()
        self._request_counts: dict[str, int] = {}
        self._total_tokens: dict[str, int] = {}
        # Backoff state
        self._consecutive_failures = 0
        self._backoff_until = 0.0
        # Phase 2 Chunk ε (D-SPEC-78, 2026-05-18) — shared httpx.AsyncClient
        # for connection-pool reuse. Default httpx.AsyncClient() creates a
        # fresh TCP+TLS handshake per call (~200-500ms tax). With a shared
        # client, the keepalive connection is reused — first call pays the
        # handshake, subsequent calls reuse the connection.
        #
        # Lazy-init on first call so non-async-loop contexts (test imports
        # without `asyncio.run`) don't trip up. Owned per provider instance.
        # Closed when the provider is garbage-collected (httpx handles its
        # own cleanup).
        self._shared_client: Optional["httpx.AsyncClient"] = None
        # The shared AsyncClient binds to the event loop that creates it. Track
        # that loop so sync-worker bridges (asyncio.run per call, which closes
        # the loop each time) get a fresh client instead of one bound to a dead
        # loop. See chat() for the rebuild logic. (2026-06-02)
        self._shared_client_loop: Optional[Any] = None

    # ── Identity ──

    @property
    def id(self) -> str:
        return self._chat_model

    @property
    def name(self) -> str:
        return "OllamaCloud"

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── Internal: track stats ──

    def _track(self, model: str, tokens: int = 0) -> None:
        self._request_counts[model] = self._request_counts.get(model, 0) + 1
        if tokens > 0:
            self._total_tokens[model] = self._total_tokens.get(model, 0) + tokens

    def _in_backoff(self) -> bool:
        return time.time() < self._backoff_until

    def _record_failure(self, model: str) -> None:
        self._consecutive_failures += 1
        delay = min(
            self._BACKOFF_BASE ** min(self._consecutive_failures,
                                      self._BACKOFF_MAX_FAILURES),
            self._BACKOFF_MAX,
        )
        self._backoff_until = time.time() + delay
        self._track(model, 0)

    def _record_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info(
                "[OllamaCloud] Recovered after %d failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._backoff_until = 0.0

    # ── Core surface ──

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> str:
        """OpenAI-style chat completion. Returns content string or "" on failure."""
        if self._in_backoff():
            remaining = self._backoff_until - time.time()
            logger.debug(
                "[OllamaCloud] Backoff active — %.1fs remaining (%d failures)",
                remaining, self._consecutive_failures,
            )
            return ""

        use_model = model or self._chat_model
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Phase 2 Chunk ε (D-SPEC-78, 2026-05-18) — per-stage latency
        # instrumentation. Production observation: Maker measured bare
        # Ollama Cloud at 1-2s, but the wrapped path consumed 15-18s.
        # Need to find which stage adds the wrap overhead. Stages:
        #   t_pool_acquire  — httpx.AsyncClient ctor (new TLS connection!)
        #   t_network       — POST request + response
        #   t_parse         — JSON decode + content extraction
        #   t_post          — usage tracking + success record
        _t0 = time.perf_counter()
        # Ensure shared client (lazy-init on first call from an async loop).
        # LOOP-AWARE: an httpx.AsyncClient binds to the event loop that creates
        # it. Sync-worker bridges (synthesis_worker's ConsolidationPass /
        # ProceduralMiner) call provider.complete() via asyncio.run(), which
        # opens a FRESH loop per call and CLOSES it on return — orphaning a
        # shared client bound to the now-dead loop ("Event loop is closed" on
        # every call after the first). Detect a loop change and rebuild the
        # client on the current loop. The API's persistent loop (agno/chat)
        # keeps full pooling; cross-loop callers get a correct fresh client.
        # 2026-06-02 — root cause of ProceduralMiner 7/7 abstraction failures:
        # consolidation's asyncio.run closed the loop before the miner's calls.
        try:
            _cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            _cur_loop = None
        if (self._shared_client is not None
                and self._shared_client_loop is not _cur_loop):
            # Old client is bound to a different / closed loop — drop it (httpx
            # releases its transport on GC; the old loop is gone so we cannot
            # await aclose() here) and rebuild below on the current loop.
            self._shared_client = None
            self._shared_client_loop = None
        if self._shared_client is None:
            try:
                self._shared_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout, connect=10.0),
                    limits=httpx.Limits(
                        max_keepalive_connections=4,
                        max_connections=8,
                        keepalive_expiry=300.0,  # 5 min keepalive
                    ),
                )
                self._shared_client_loop = _cur_loop
                logger.info(
                    "[OllamaCloud] shared httpx client initialized "
                    "(keepalive=300s, pool=4 idle / 8 max)")
            except Exception as _shared_err:
                logger.warning(
                    "[OllamaCloud] shared client init failed (%s) — "
                    "falling back to per-call client", _shared_err)
                self._shared_client = None
                self._shared_client_loop = None
        try:
            # Phase 2 Chunk ε — reuse the shared client (set up once at
            # provider construct time). New AsyncClient per call costs
            # ~200-500ms in TLS handshake + connection setup; for a hot
            # caller like agno_worker every chat pays that tax.
            client = self._shared_client
            if client is None:
                # Fallback to per-call client (cold path / shared not init)
                async with httpx.AsyncClient(timeout=timeout) as fallback_client:
                    _t_pool = time.perf_counter()
                    resp = await fallback_client.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    _t_net = time.perf_counter()
                    resp.raise_for_status()
                    data = resp.json()
                    _t_parse = time.perf_counter()
            else:
                _t_pool = time.perf_counter()
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                _t_net = time.perf_counter()
                resp.raise_for_status()
                data = resp.json()
                _t_parse = time.perf_counter()

            content = data["choices"][0]["message"]["content"].strip()
            # finish_reason="length" ⇒ the model hit max_tokens and was HARD-CUT
            # mid-reply (the silent truncation behind "Titan got cut off"). Surface
            # it loudly so it's never invisible again (directive_error_visibility).
            _finish = (data["choices"][0].get("finish_reason") or "").strip()
            usage = data.get("usage", {})
            self._track(use_model, usage.get("total_tokens", 0))
            self._record_success()
            _t_done = time.perf_counter()
            if _finish == "length":
                logger.warning(
                    "[OllamaCloud] reply TRUNCATED by max_tokens (model=%s "
                    "tokens=%d) — the tier's max_tokens ceiling cut the reply "
                    "mid-sentence; raise it for this lane if replies trail off",
                    use_model, usage.get("total_tokens", 0))

            # Stage timings (debug-level for routine, info-level when slow)
            stage_pool_ms = (_t_pool - _t0) * 1000
            stage_net_ms = (_t_net - _t_pool) * 1000
            stage_parse_ms = (_t_parse - _t_net) * 1000
            stage_post_ms = (_t_done - _t_parse) * 1000
            total_ms = (_t_done - _t0) * 1000
            log_fn = logger.info if total_ms > 3000 else logger.debug
            log_fn(
                "[OllamaCloud] chat %s done in %.0fms "
                "(pool=%.0f net=%.0f parse=%.0f post=%.0f) "
                "tokens=%d chars=%d finish=%s shared_client=%s",
                use_model, total_ms,
                stage_pool_ms, stage_net_ms, stage_parse_ms, stage_post_ms,
                usage.get("total_tokens", 0), len(content), _finish or "?",
                "yes" if self._shared_client is not None else "no",
            )
            return content
        except Exception as e:
            self._record_failure(use_model)
            logger.warning(
                "[OllamaCloud] chat() failed (model=%s, attempt=%d, "
                "backoff=%.0fs, elapsed=%.0fms): %s",
                use_model, self._consecutive_failures,
                self._backoff_until - time.time(),
                (time.perf_counter() - _t0) * 1000, e,
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
        """Server-sent-events streaming chat. Yields content chunks as they arrive."""
        if self._in_backoff():
            return

        use_model = model or self._chat_model
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
            self._track(use_model, 0)
            self._record_success()
        except Exception as e:
            self._record_failure(use_model)
            logger.warning(
                "[OllamaCloud] stream_chat() failed (model=%s, attempt=%d): %s",
                use_model, self._consecutive_failures, e,
            )

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 500,
        timeout: float = 60.0,
    ) -> str:
        """Single-turn completion — signature absorbs legacy OllamaCloudClient
        positional API (model 2nd / system 3rd) so 20+ migrated callsites work
        without keyword-only friction. D-SPEC-72: utils/ollama_cloud.py was
        deleted, NOT shimmed — this method IS the canonical path."""
        # Use the canonical model default (light tier — gemma4:31b legacy)
        # NOT the heavy chat model. Preserves scoring + haiku + teacher
        # latency profiles.
        use_model = model or _DEFAULT_MODEL
        return await super().complete(
            prompt,
            system=system,
            model=use_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def score(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> float:
        """Ask the LLM to rate something 0.0-1.0. Returns 0.5 fallback on failure.

        Preserves the legacy `OllamaCloudClient.score()` surface — still used
        by language_teacher meditation scoring + policy checks.
        """
        response = await self.complete(
            prompt=prompt,
            system=(
                "You are a scoring system. Respond with ONLY a single "
                "decimal number between 0.0 and 1.0."
            ),
            model=model or _DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=20,
            timeout=timeout,
        )
        if not response:
            return 0.5
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return 0.5

    # ── Agno integration ──

    def get_agno_model(self) -> Any:
        """Return an Agno OpenAILike instance for the chat model.

        Used by agno_worker._init_agent. The Agno Agent uses this object
        directly for its inference — NOT this provider's chat() method —
        because Agno owns the streaming + tool-calling + multi-turn
        history orchestration. Future: collapse this dual path by having
        Agno call the provider via a custom client adapter.
        """
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=self._chat_model,
            name=self.name,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    # ── Legacy back-compat properties (absorbed from deleted utils/ollama_cloud.py shim) ──
    # Some callsites access these directly rather than via get_stats(); preserve
    # the legacy public surface so all 20+ migrated callers continue to work.

    @property
    def request_counts(self) -> dict[str, int]:
        return dict(self._request_counts)

    @property
    def total_tokens(self) -> dict[str, int]:
        return dict(self._total_tokens)

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self._base_url,
            "chat_model": self._chat_model,
            "request_counts": dict(self._request_counts),
            "total_tokens": dict(self._total_tokens),
            "total_requests": sum(self._request_counts.values()),
            "consecutive_failures": self._consecutive_failures,
            "in_backoff": self._in_backoff(),
            "backoff_remaining_s": max(0.0, self._backoff_until - time.time()),
        }
