"""
titan_hcl.api.llm_proxy_endpoints — HTTP proxy for the LLM_DISTILL_REQUEST
+ LLM_SCORE_REQUEST bus topics.

D-SPEC-88 (Phase 3 Chunk ω, 2026-05-18). Out-of-kernel callers (cron scripts:
events_teacher_run.py, persona_endurance.py, persona_social_v2.py, any future
tooling) cannot publish to the bus directly — they run as standalone processes
without a bus client. This module exposes:

    POST /v4/llm-distill   →  plugin.llm_proxy.distill(...)
    POST /v4/llm-score     →  plugin.llm_proxy.score(...)

Both endpoints internally publish the corresponding bus topic, so every
external LLM call appears in `llm_state.bin` and `arch_map llm-stats`
regardless of caller type.

Auth: `X-Titan-Internal-Key` header matching `[api] internal_key` in
config.toml (same pattern as other internal-script endpoints — see
api/auth.py:127 verify_maker_auth).

Architectural rule (codified in Chunk Ω SPEC §9.B v1.20.0):
  • In-kernel hot path (chat)         → inference.get_provider() direct
  • In-kernel background workers      → LLMProxy (bus)
  • Out-of-kernel cron / scripts      → these HTTP endpoints
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from titan_hcl.api.auth import verify_maker_auth


async def _await_if_coroutine(result):
    """Return `result` unwrapped if it's a coroutine, else as-is.

    In api_subprocess context, plugin._proxies access goes through
    kernel_rpc — which already awaits async methods on the kernel loop
    and ships the resolved value back. So `llm_proxy.distill(...)`
    returns a `str` directly there.

    In in-process / test context, the proxy method IS a coroutine and
    must be awaited locally. This helper handles both transparently.
    """
    if inspect.iscoroutine(result):
        return await result
    return result

logger = logging.getLogger(__name__)
router = APIRouter(tags=["LLM Proxy"])


# ── Request / response shapes ───────────────────────────────────────


class DistillRequest(BaseModel):
    """Payload for POST /v4/llm-distill."""
    text: str = Field(..., min_length=1, max_length=200_000,
                      description="The content to distill. Often a user message, "
                                  "document, or items_text block.")
    instruction: str = Field("Summarize concisely", max_length=20_000,
                             description="The instruction/system-style prompt "
                                         "telling the LLM what to do with the text.")
    model: Optional[str] = Field(None,
                                 description="Provider model override (e.g. "
                                             "'gemma4:31b'). None → "
                                             "provider default.")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192,
                                      description="Cap response length. None → "
                                                  "provider default (500).")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0,
                                         description="Sampling temperature. None → "
                                                     "provider default (0.3 for distill).")
    pre_hook: bool = Field(False,
                           description="If True, llm_worker wraps the prompt with "
                                       "llm_pipeline.compose_pre (felt-state + "
                                       "primedirectives). For external scripts, "
                                       "default False — the caller usually has "
                                       "its own framing.")
    post_hook: bool = Field(False,
                            description="If True, llm_worker runs verify_post_async "
                                        "(OVG truth gate) on the output. Returns the "
                                        "verified text + ovg dict.")
    channel: str = Field("agent", max_length=64,
                         description="OVG channel when post_hook=True.")
    timeout_s: float = Field(30.0, ge=1.0, le=120.0,
                             description="Bus-RPC timeout. Phase C exemption: "
                                         "30s > §1.B 5s default (allowlisted as "
                                         "LLM work-RPC).")
    consumer: str = Field("external", max_length=64,
                          description="Caller identifier for telemetry / "
                                      "llm_state.bin counters (e.g. "
                                      "'events_teacher', 'persona_endurance').")


class DistillResponse(BaseModel):
    """Response for POST /v4/llm-distill."""
    status: str  # "ok" | "timeout" | "no_proxy" | "error"
    text: str = ""
    latency_ms: float = 0.0
    consumer: str = "external"
    error: Optional[str] = None


class ScoreRequest(BaseModel):
    """Payload for POST /v4/llm-score."""
    prompt: str = Field(..., min_length=1, max_length=20_000,
                        description="The scoring prompt — usually a quality / "
                                    "relevance / classification question with a "
                                    "constrained answer space (e.g. '0.0-1.0' or "
                                    "'yes/no').")
    model: Optional[str] = Field(None,
                                 description="Provider model override.")
    timeout_s: float = Field(15.0, ge=1.0, le=60.0,
                             description="Bus-RPC timeout (default 15s — score "
                                         "prompts are typically short).")
    consumer: str = Field("external", max_length=64,
                          description="Caller identifier.")


class ScoreResponse(BaseModel):
    """Response for POST /v4/llm-score."""
    status: str  # "ok" | "timeout" | "no_proxy" | "error"
    text: str = ""
    latency_ms: float = 0.0
    consumer: str = "external"
    error: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/v4/llm-distill", response_model=DistillResponse,
             dependencies=[Depends(verify_maker_auth)])
async def llm_distill_endpoint(req: DistillRequest, request: Request) -> DistillResponse:
    """Proxy external HTTP request to the in-kernel `LLM_DISTILL_REQUEST` bus topic.

    See module docstring for the architectural rule. Internal-key auth via
    `X-Titan-Internal-Key` (same as other internal-script endpoints).
    """
    plugin = getattr(request.app.state, "titan_hcl", None)
    # Use _proxies["llm"] access — this matches kernel_rpc._is_path_allowed's
    # chained-proxy escape (`_proxies.<known>.<method>` allowed when <known>
    # is in plugin._proxies). Direct `plugin.llm_proxy` would generate path
    # `llm_proxy.distill` which would need explicit allowlisting.
    llm_proxy = None
    if plugin is not None:
        # Prefer dict-style access on _proxies (works in both in-process
        # and api_subprocess RPC modes).
        proxies = getattr(plugin, "_proxies", None)
        if proxies is not None:
            try:
                llm_proxy = proxies.get("llm")
            except Exception:
                llm_proxy = None
        # Fallback to attribute access (in-process tests, future direct attr)
        if llm_proxy is None:
            llm_proxy = getattr(plugin, "llm_proxy", None)
    if llm_proxy is None:
        return DistillResponse(
            status="no_proxy",
            error="plugin._proxies['llm'] / plugin.llm_proxy not available — "
                  "kernel may be booting",
            consumer=req.consumer,
        )

    t0 = time.perf_counter()
    try:
        # Two contexts:
        #   • api_subprocess (production): kernel_rpc resolves the awaitable
        #     on the kernel loop and ships back a string. result is a str.
        #   • in-process (tests, fallback): result is a coroutine that
        #     must be awaited locally.
        # _await_if_coroutine handles both.
        raw = llm_proxy.distill(
            text=req.text,
            instruction=req.instruction,
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            pre_hook=req.pre_hook,
            post_hook=req.post_hook,
            channel=req.channel,
            timeout=req.timeout_s,
        )
        result = await _await_if_coroutine(raw)
    except Exception as e:
        logger.warning("[llm-distill] proxy raised: %s", e)
        return DistillResponse(
            status="error",
            error=str(e),
            latency_ms=(time.perf_counter() - t0) * 1000,
            consumer=req.consumer,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    if not result:
        logger.warning("[llm-distill] empty result (timeout or worker down) "
                       "consumer=%s after %.0fms", req.consumer, latency_ms)
        return DistillResponse(
            status="timeout",
            error="empty result from llm_worker (timeout or worker unavailable)",
            latency_ms=latency_ms,
            consumer=req.consumer,
        )

    logger.info("[llm-distill] consumer=%s ok latency=%.0fms result_len=%d",
                req.consumer, latency_ms, len(result))
    return DistillResponse(
        status="ok",
        text=result,
        latency_ms=latency_ms,
        consumer=req.consumer,
    )


@router.post("/v4/llm-score", response_model=ScoreResponse,
             dependencies=[Depends(verify_maker_auth)])
async def llm_score_endpoint(req: ScoreRequest, request: Request) -> ScoreResponse:
    """Proxy external HTTP request to the in-kernel `LLM_SCORE_REQUEST` bus topic.

    Same auth + architectural-rule semantics as `/v4/llm-distill`.
    Used by persona scoring + future caller patterns that need a constrained
    1-token-style answer (0.0-1.0, yes/no, classification).
    """
    plugin = getattr(request.app.state, "titan_hcl", None)
    # Same _proxies["llm"] preference as /v4/llm-distill — see comment there.
    llm_proxy = None
    if plugin is not None:
        proxies = getattr(plugin, "_proxies", None)
        if proxies is not None:
            try:
                llm_proxy = proxies.get("llm")
            except Exception:
                llm_proxy = None
        if llm_proxy is None:
            llm_proxy = getattr(plugin, "llm_proxy", None)
    if llm_proxy is None:
        return ScoreResponse(
            status="no_proxy",
            error="plugin._proxies['llm'] / plugin.llm_proxy not available",
            consumer=req.consumer,
        )

    t0 = time.perf_counter()
    try:
        # Same dual-context handling as /v4/llm-distill.
        raw = llm_proxy.score(
            prompt=req.prompt,
            model=req.model,
            timeout=req.timeout_s,
        )
        result = await _await_if_coroutine(raw)
    except Exception as e:
        logger.warning("[llm-score] proxy raised: %s", e)
        return ScoreResponse(
            status="error",
            error=str(e),
            latency_ms=(time.perf_counter() - t0) * 1000,
            consumer=req.consumer,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    if not result:
        return ScoreResponse(
            status="timeout",
            error="empty result from llm_worker",
            latency_ms=latency_ms,
            consumer=req.consumer,
        )

    logger.info("[llm-score] consumer=%s ok latency=%.0fms result_len=%d",
                req.consumer, latency_ms, len(result))
    return ScoreResponse(
        status="ok",
        text=result,
        latency_ms=latency_ms,
        consumer=req.consumer,
    )
