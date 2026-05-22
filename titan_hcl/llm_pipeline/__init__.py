"""
titan_hcl.llm_pipeline — shared pre/post wrappers for any LLM-touching context.

Companion library to `titan_hcl.inference` (provider abstraction).
Together the two libraries unify the chat pipeline across every
LLM-using code path in the fleet:

    /chat, /v4/pitch-chat   → agno_worker (D-SPEC-72)
    llm_worker              → cross-worker bus.QUERY action=chat/distill
    social_x_gateway        → X post verification (existing 2 callsites)
    autonomous_language_pipeline → script-driven composition
    dashboard /v4/compose-reply → test endpoint
    future avatars/voice    → same facade

Usage:

    from titan_hcl import inference, llm_pipeline

    provider = inference.get_provider(cfg["provider"], cfg)
    pre = await llm_pipeline.compose_pre(message, user_id, channel="chat")
    effective = (pre.pre_text + "\\n\\n" if pre.pre_text else "") + message
    raw_response = await provider.chat(
        [{"role": "user", "content": effective}],
    )
    verified = llm_pipeline.verify_post(
        raw_response,
        channel="chat",
        prompt=message,
        output_verifier=plugin._output_verifier,
        bus=plugin.bus,
    )
    # verified.text is final string for the user
    # verified.ovg_data → ChatResponse.ovg field

Or the convenience all-in-one:

    result = await llm_pipeline.wrap_llm_call(
        provider, message, channel="chat",
        user_id=uid, output_verifier=plugin._output_verifier, bus=plugin.bus,
    )
    return result.text

Documented in SPEC §9.C.2 (D-SPEC-72).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from . import state_gather
from .composer import ComposeResult, compose_pre, reset_singletons
from .verifier import VerifiedResult, verify_post

__all__ = [
    "ComposeResult",
    "VerifiedResult",
    "compose_pre",
    "verify_post",
    "wrap_llm_call",
    "state_gather",
    "reset_singletons",  # test helper
]

logger = logging.getLogger(__name__)


async def wrap_llm_call(
    provider: Any,
    message: str,
    *,
    channel: str,
    user_id: str = "",
    output_verifier: Any = None,
    bus: Any = None,
    system: str = "",
    coordinator_snapshot: Optional[dict] = None,
    injected_context: str = "",
    chain_state: Optional[dict] = None,
    min_confidence: float = 0.3,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> VerifiedResult:
    """Convenience: compose_pre + provider.chat + verify_post in one call.

    Returns a VerifiedResult whose .text is the final user-facing string
    (DialogueComposer prefix if composed, LLM response, OVG guard
    message if blocked).

    Workers and call sites that want fine-grained control over the steps
    (e.g. agno_worker uses `agent.arun()` inside an Agno Agent rather
    than direct provider.chat) should compose `compose_pre()` and
    `verify_post()` themselves.

    Args:
        provider:      An `InferenceProvider` instance (from titan_hcl.inference).
        message:       User message.
        channel:       OVG channel ("chat", "x_post", etc.).
        user_id:       User identifier for InputExtractor.
        output_verifier: OutputVerifier or OutputVerifierProxy instance.
        bus:           Bus client for TIMECHAIN_COMMIT publish.
        system:        Optional system prompt.
        coordinator_snapshot / injected_context / chain_state: see
                       verify_post() documentation.
        min_confidence: Pre-LLM composition threshold (default 0.3).
        temperature / max_tokens: provider.chat kwargs.

    Returns:
        VerifiedResult with .text containing the assembled final string.
    """
    # 1. Pre-LLM composition
    pre = await compose_pre(
        message,
        user_id=user_id,
        channel=channel,
        min_confidence=min_confidence,
    )

    effective = message
    if pre.composed and pre.pre_text:
        effective = pre.pre_text + "\n\n" + message

    # 2. LLM call
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": effective})

    try:
        raw_response = await provider.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error("[llm_pipeline.wrap_llm_call] provider.chat raised: %s", e)
        raw_response = ""

    # 3. Post-LLM verification + TimeChain commit
    verified = verify_post(
        raw_response,
        channel=channel,
        prompt=message,
        injected_context=injected_context,
        chain_state=chain_state,
        coordinator_snapshot=coordinator_snapshot,
        output_verifier=output_verifier,
        bus=bus,
    )

    # 4. If pre-LLM composition produced a sentence, prepend italic-marked
    #    to the verified output. Matches the existing /chat behavior at
    #    api/chat.py:300 ("*{_self_composed}*\\n\\n{response_text}").
    if pre.composed and pre.pre_text and verified.text:
        verified.text = f"*{pre.pre_text}*\n\n{verified.text}"

    return verified
