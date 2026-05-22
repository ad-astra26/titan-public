"""
titan_hcl.logic.llm_distill_client — shared HTTP client for /v4/llm-distill.

D-SPEC-88 Phase 3 Chunk χ (2026-05-18). Multiple in-kernel callers (meditation
scoring, TitanCognify entity extraction) need to invoke the centralized LLM
worker via the HTTP proxy endpoint. This helper consolidates the boilerplate:

  • async + sync variants
  • internal-key auth header
  • payload assembly
  • response unwrap
  • error / timeout handling

Per the architectural rule (rFP §10.2 Chunk Ω SPEC v1.20.0):

    In-kernel hot path (chat)         → inference.get_provider() direct
    In-kernel background workers      → LLMProxy (bus)
    Out-of-kernel cron / scripts      → POST /v4/llm-distill (HTTP)
    Helpers running in subprocess     → POST /v4/llm-distill (HTTP) ← this module

The pattern matches Chunk ω-bis's `PostDispatchOrchestrator._compose_post_text`
+ Chunk ω's `events_teacher._distill_content`. All external LLM traffic
appears in `llm_state.bin` regardless of caller process.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def distill_via_http_sync(
    *,
    text: str,
    instruction: str,
    api_base: str,
    internal_key: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    consumer: str = "internal",
    timeout_s: float = 30.0,
) -> str:
    """Synchronous HTTP call to {api_base}/v4/llm-distill.

    For use from sync code paths (worker tick loops, subprocess scoring,
    sync TitanCognify call sites). The LLM round-trip itself blocks but
    these callers are not on user-facing latency paths.

    Returns the distilled text, or "" on timeout / error / auth failure.
    """
    if not internal_key:
        logger.debug("[llm_distill_http] no internal_key configured — skipping")
        return ""
    try:
        import httpx
        payload: dict = {
            "text": text,
            "instruction": instruction,
            "consumer": consumer,
            "timeout_s": timeout_s,
        }
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        resp = httpx.post(
            f"{api_base.rstrip('/')}/v4/llm-distill",
            headers={"X-Titan-Internal-Key": internal_key,
                     "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_s + 5.0,
        )
        body = resp.json()
        if body.get("status") != "ok":
            logger.warning(
                "[llm_distill_http] consumer=%s status=%s error=%s",
                consumer, body.get("status"), body.get("error"))
            return ""
        return (body.get("text") or "").strip()
    except Exception as exc:
        logger.warning("[llm_distill_http] consumer=%s raised: %s",
                       consumer, exc)
        return ""


async def distill_via_http_async(
    *,
    text: str,
    instruction: str,
    api_base: str,
    internal_key: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    consumer: str = "internal",
    timeout_s: float = 30.0,
) -> str:
    """Async sibling of `distill_via_http_sync`. Same semantics, uses
    `httpx.AsyncClient` so callers in async contexts don't block the
    event loop.

    Returns the distilled text, or "" on timeout / error / auth failure.
    """
    if not internal_key:
        logger.debug("[llm_distill_http] no internal_key configured — skipping")
        return ""
    try:
        import httpx
        payload: dict = {
            "text": text,
            "instruction": instruction,
            "consumer": consumer,
            "timeout_s": timeout_s,
        }
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        async with httpx.AsyncClient(timeout=timeout_s + 5.0) as client:
            resp = await client.post(
                f"{api_base.rstrip('/')}/v4/llm-distill",
                headers={"X-Titan-Internal-Key": internal_key,
                         "Content-Type": "application/json"},
                json=payload,
            )
        body = resp.json()
        if body.get("status") != "ok":
            logger.warning(
                "[llm_distill_http] consumer=%s status=%s error=%s",
                consumer, body.get("status"), body.get("error"))
            return ""
        return (body.get("text") or "").strip()
    except Exception as exc:
        logger.warning("[llm_distill_http] consumer=%s raised: %s",
                       consumer, exc)
        return ""
