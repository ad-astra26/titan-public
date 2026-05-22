"""
titan_hcl.logic.social_x_composer — LLM composition for X posts.

D-SPEC-88 Phase 3 Chunk ω-bis (2026-05-18). The gateway no longer composes:
its responsibility is gates + archetype dispatcher + grounding + transport.
This module is the bridge between `SocialXGateway.prepare_post()` (which
returns a `PostDescriptor`) and `SocialXGateway.post()` (which sends the
pre-composed text to X).

Per Maker direction 2026-05-18: "post() just send it through". And per
the architectural rule codified for SPEC v1.20.0:

    In-kernel callers (social_worker) → llm_proxy.distill() (bus)
    Out-of-kernel callers (cron)      → POST /v4/llm-distill (HTTP)

Both arrive at the same `llm_worker` LLM_DISTILL_REQUEST handler, so the
result is observable in `llm_state.bin` regardless of caller process.

Usage (in-kernel):

    err, desc = gateway.prepare_post(ctx, consumer="spirit_worker")
    if err is not None:
        return err
    ctx.composed_text = await compose_post_text(desc, llm_proxy)
    result = gateway.post(ctx, descriptor=desc, consumer="spirit_worker")
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from titan_hcl.logic.social_x_gateway import PostDescriptor
    from titan_hcl.proxies.llm_proxy import LLMProxy

logger = logging.getLogger(__name__)


async def compose_post_text(descriptor: "PostDescriptor",
                            llm_proxy: "LLMProxy",
                            *,
                            timeout_s: float = 45.0) -> Optional[str]:
    """Run the LLM round-trip for an X post and return the raw text.

    Uses the in-kernel `LLMProxy.distill()` (bus-routed to llm_worker).
    Out-of-kernel callers should use the HTTP `/v4/llm-distill` endpoint
    instead — both arrive at the same `LLM_DISTILL_REQUEST` handler.

    Returns:
        The generated text, or None on timeout / error / empty response.
        Caller should treat None as `ActionResult(status="generation_failed")`
        when invoking `gateway.post()`.
    """
    if llm_proxy is None:
        logger.warning("[social_x_composer] llm_proxy is None — cannot compose")
        return None
    try:
        # Same instruction/text split as events_teacher (Chunk ω) — the
        # llm_worker's distill() builds `{instruction}\n\n{text}` for the
        # provider. The X-post system prompt is heavy on rules + style;
        # the user_prompt carries the catalyst + layers. Both work in
        # combined form because the model is instruction-following.
        result = await llm_proxy.distill(
            text=descriptor.user_prompt,
            instruction=descriptor.system_prompt,
            max_tokens=descriptor.max_tokens,
            temperature=descriptor.temperature,
            pre_hook=False,    # gateway already enforces its own
                               # grounding gate + voice nudge in
                               # _build_prompts; composer skips OVG
                               # pre-hook to avoid double-framing
            post_hook=False,   # OVG post-hook still runs INSIDE
                               # gateway.post() at step 8b for X-post
                               # security gate (publish_timechain=False)
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.warning("[social_x_composer] distill raised: %s", exc)
        return None

    if not result:
        logger.warning("[social_x_composer] distill returned empty result "
                       "(timeout or worker down)")
        return None

    text = result.strip()
    # Strip wrapping quotes (matches the old _generate_text behavior).
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text
