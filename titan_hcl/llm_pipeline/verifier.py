"""
titan_hcl.llm_pipeline.verifier — OutputVerifierFacade.

Wraps `OutputVerifierProxy.verify_safety + sign_and_commit +
build_timechain_payload + bus.publish(TIMECHAIN_COMMIT)` into a single
facade call. Single implementation of the post-LLM verification block
currently duplicated 4× today:
  - agno_hooks.py:1588-1632 (PostHook in chat agent path)
  - core/plugin.py:2795-2850 (deprecated inline chat handler)
  - chat_pipeline.py (run_chat inline)
  - social_x_gateway.py:2935 + 3222 (X post verification)

After D-SPEC-72 + D-SPEC-74, ALL output-emitting code paths go through
verify_post() — no more drift between callsites, identical semantics for
guard_message appending + TimeChain commit + sovereignty score updates.

D-SPEC-74 (SPEC v1.18.0):
  - `verify_post` (sync) — back-compat for sync callsites (studio,
    social_x_gateway). Uses combined OVG.verify_and_sign path; signing
    blocks the caller.
  - `verify_post_async` (NEW, recommended) — splits safety verification
    (blocks until clear) from signing (spawned as asyncio.Task; result
    attached to VerifiedResult.signature when caller awaits .sign_task).
    Used by agno_hooks PostHook (rFP Chunk C) and async callers that
    want concurrent-sign latency win.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from . import state_gather

logger = logging.getLogger(__name__)


class VerifiedResult:
    """Result of verify_post() — caller uses this to assemble final response.

    Attributes:
        text:             Final output text. May be the original `text` with
                          guard_message appended (verified path), or the
                          guard_message ALONE (blocked path).
        passed:           True iff OVG returned passed=True (verified).
        blocked:          True iff OVG returned passed=False (blocked).
        soft_alert:       Non-empty string if a soft alert fired (verified
                          but with warning) — informational only.
        violation_type:   On blocked path, OVG's violation_type ("directive",
                          "injection", "consistency", "identity", "qualia").
        violations:       List of human-readable violation descriptions.
        signature:        Ed25519 signature (hex) for verified outputs.
        merkle_root:      TimeChain Merkle root for verified outputs.
        block_height:     TimeChain block height the commit references.
        ovg_data:         Wire-shaped OVG dict for ChatResponse.ovg field.
        timechain_committed: True iff TIMECHAIN_COMMIT was published.
        raw_result:       The underlying OVGResult instance (for advanced
                          callers needing original fields).
    """

    __slots__ = (
        "text", "passed", "blocked", "soft_alert", "violation_type",
        "violations", "signature", "merkle_root", "block_height",
        "ovg_data", "timechain_committed", "raw_result",
        "sign_task",
    )

    def __init__(
        self,
        text: str,
        *,
        passed: bool = False,
        blocked: bool = False,
        soft_alert: Optional[str] = None,
        violation_type: Optional[str] = None,
        violations: Optional[list] = None,
        signature: Optional[str] = None,
        merkle_root: Optional[str] = None,
        block_height: int = 0,
        ovg_data: Optional[dict] = None,
        timechain_committed: bool = False,
        raw_result: Optional[Any] = None,
        sign_task: Optional[Any] = None,
    ):
        self.text = text
        self.passed = passed
        self.blocked = blocked
        self.soft_alert = soft_alert
        self.violation_type = violation_type
        self.violations = violations or []
        self.signature = signature
        self.merkle_root = merkle_root
        self.block_height = block_height
        self.ovg_data = ovg_data or {}
        self.timechain_committed = timechain_committed
        self.raw_result = raw_result
        # D-SPEC-74: when verify_post_async is called with concurrent_sign=True
        # and safety PASSED, this points to the in-flight sign_and_commit task.
        # Caller (agno_worker, SSE writer) awaits it to attach the signature
        # to the final ovg_headers frame. None on blocked / sync paths.
        self.sign_task = sign_task

    @classmethod
    def unverified(cls, text: str) -> "VerifiedResult":
        """Return-as-is when verifier is unavailable. Logs a WARN-class event."""
        return cls(text=text, passed=False, blocked=False)


def _build_ovg_data(ovg_result: Any) -> dict[str, Any]:
    """Construct the ChatResponse.ovg dict from an OVGResult.

    Mirror of the inline block at chat.py:386-404. Stable across callsites.
    """
    return {
        "verified": bool(getattr(ovg_result, "passed", False)),
        "guard_alert": getattr(ovg_result, "guard_alert", None),
        "guard_message": getattr(ovg_result, "guard_message", "") or "",
        "block_height": int(getattr(ovg_result, "block_height", 0) or 0),
        "merkle_root": getattr(ovg_result, "merkle_root", "") or "",
        "signature": getattr(ovg_result, "signature", None),
    }


def verify_post(
    text: str,
    *,
    channel: str,
    prompt: str = "",
    injected_context: str = "",
    chain_state: Optional[dict] = None,
    coordinator_snapshot: Optional[dict] = None,
    output_verifier: Any = None,
    bus: Any = None,
    publish_timechain: bool = True,
    append_guard_on_pass: bool = True,
    src_for_timechain: str = "ovg",
    # Phase 2 closure (2026-05-25) — arch §7 chat-TX shape. None / "" /
    # "anonymous" passes through unchanged (build_timechain_payload
    # short-circuits the user_id hash → no `user:` tag added). Existing
    # non-chat callers (social_x_gateway, studio_worker, etc.) pass
    # nothing and get pre-Phase-2-closure behavior byte-identical.
    user_id: str = "",
    chat_id: str = "",
    turn_index: int = 0,
    topic_tags: Optional[list] = None,
) -> VerifiedResult:
    """Verify, sign, and TimeChain-commit an LLM response.

    Args:
        text:                  Raw LLM output to verify.
        channel:               "chat" / "x_post" / "x_reply" / "agent" / "telegram"
                               — passed through to OVG for channel-specific gates.
        prompt:                The originating prompt (for OVG's
                               injection-detection cross-check).
        injected_context:      Any extra context provided to the LLM
                               (e.g. agent.additional_context — for OVG's
                               consistency check).
        chain_state:           Pre-gathered chain_state dict. If None, the
                               facade will call state_gather.gather_chain_state
                               with coordinator_snapshot.
        coordinator_snapshot:  In-parent contexts pass plugin's cached
                               coordinator dict here. Worker contexts
                               supply chain_state directly.
        output_verifier:       OutputVerifier or OutputVerifierProxy instance.
                               Required — pass plugin._output_verifier or
                               worker-side OutputVerifierProxy. None →
                               returns VerifiedResult.unverified(text).
        bus:                   Bus client for TIMECHAIN_COMMIT publish.
                               None → skip publish (timechain_committed=False).
        publish_timechain:     False → build payload but skip publish (test
                               OR X-post path which is publish-gate-only).
        append_guard_on_pass:  True (default — chat behavior) appends OVG
                               `guard_message` footer to verified text +
                               soft-alert text. False (X-post / external
                               post path) returns raw text unchanged on
                               pass — appending "[VERIFIED]" to a tweet
                               body would be poor UX. Blocked path ALWAYS
                               replaces with guard_message regardless of
                               this flag (sovereignty enforcement is
                               non-negotiable per OVG contract).
        src_for_timechain:     Source name on the TIMECHAIN_COMMIT message.
                               Default "ovg" matches existing chat.py emit.

    Returns:
        VerifiedResult. `text` is the final assembled string callers should
        return to the user. Behavior matrix:
          - blocked      → replaced with guard_message (always)
          - soft alert + append_guard_on_pass=True  → text + "\\n\\n" + guard_message
          - soft alert + append_guard_on_pass=False → raw text
          - clean      + append_guard_on_pass=True  → text + "\\n\\n" + guard_message
          - clean      + append_guard_on_pass=False → raw text unchanged
    """
    if not text:
        return VerifiedResult.unverified("")

    if output_verifier is None:
        logger.debug(
            "[llm_pipeline.verify_post] no output_verifier supplied — "
            "returning text unverified (channel=%s)", channel,
        )
        return VerifiedResult.unverified(text)

    # 1. Assemble chain_state
    final_chain_state = state_gather.gather_chain_state(
        coordinator_snapshot=coordinator_snapshot,
        override=chain_state,
    )

    # 2. Call OVG verify_and_sign (sync entry point — proxy handles async
    #    work-RPC underneath; OutputVerifierProxy is the live wrapper).
    try:
        ovg_result = output_verifier.verify_and_sign(
            output_text=text,
            channel=channel,
            injected_context=injected_context,
            prompt_text=prompt,
            chain_state=final_chain_state,
        )
    except Exception as e:
        logger.warning(
            "[llm_pipeline.verify_post] verify_and_sign raised: %s", e
        )
        return VerifiedResult.unverified(text)

    # 3. Compose final text (verified-with-guard / blocked / soft-alert)
    passed = bool(getattr(ovg_result, "passed", False))
    guard_message = getattr(ovg_result, "guard_message", "") or ""
    guard_alert = getattr(ovg_result, "guard_alert", None)
    violations = getattr(ovg_result, "violations", []) or []
    violation_type = getattr(ovg_result, "violation_type", None)
    signature = getattr(ovg_result, "signature", None)
    merkle_root = getattr(ovg_result, "merkle_root", "") or ""
    block_height = int(getattr(ovg_result, "block_height", 0) or 0)

    if not passed:
        logger.warning(
            "[llm_pipeline.verify_post:%s] BLOCKED (%s): %s",
            channel, violation_type, violations[:2],
        )
        # Blocked path ALWAYS replaces with guard_message — sovereignty
        # enforcement is non-negotiable regardless of append_guard_on_pass.
        final_text = guard_message
    elif guard_alert:
        logger.info(
            "[llm_pipeline.verify_post:%s] Soft alert: %s",
            channel, guard_alert,
        )
        if append_guard_on_pass:
            final_text = text.rstrip() + "\n\n" + guard_message
        else:
            # X-post path — caller decides whether to drop on soft alert;
            # we surface the alert via VerifiedResult.soft_alert but keep
            # text clean for downstream publishing.
            final_text = text
    else:
        logger.info(
            "[llm_pipeline.verify_post:%s] Verified + signed (sig=%s)",
            channel, signature[:16] if signature else "none",
        )
        if append_guard_on_pass:
            final_text = text.rstrip() + "\n\n" + guard_message
        else:
            # X-post / external-publish path — return raw verified text
            # (no footer in the tweet body; signature still attached via
            # ovg_data for the caller to log / audit / use).
            final_text = text

    # 4. Build + publish TimeChain commit
    committed = False
    if publish_timechain and bus is not None:
        try:
            tc_payload = output_verifier.build_timechain_payload(
                ovg_result, prompt_text=prompt,
                user_id=user_id, chat_id=chat_id, turn_index=turn_index,
                topic_tags=topic_tags,
            )
            if tc_payload:
                from titan_hcl.bus import TIMECHAIN_COMMIT, make_msg
                bus.publish(
                    make_msg(
                        TIMECHAIN_COMMIT, src_for_timechain, "timechain",
                        tc_payload,
                    )
                )
                committed = True
        except Exception as e:
            logger.warning(
                "[llm_pipeline.verify_post] TimeChain commit failed: %s", e
            )

    return VerifiedResult(
        text=final_text,
        passed=passed,
        blocked=not passed,
        soft_alert=guard_alert,
        violation_type=violation_type,
        violations=violations,
        signature=signature,
        merkle_root=merkle_root,
        block_height=block_height,
        ovg_data=_build_ovg_data(ovg_result),
        timechain_committed=committed,
        raw_result=ovg_result,
    )


# ═══════════════════════════════════════════════════════════════════════
# D-SPEC-74 (SPEC v1.18.0) — async verify_post with safety/sign split
# ═══════════════════════════════════════════════════════════════════════

async def verify_post_async(
    text: str,
    *,
    channel: str,
    prompt: str = "",
    injected_context: str = "",
    chain_state: Optional[dict] = None,
    coordinator_snapshot: Optional[dict] = None,
    output_verifier: Any = None,
    bus: Any = None,
    publish_timechain: bool = True,
    append_guard_on_pass: bool = True,
    src_for_timechain: str = "ovg",
    concurrent_sign: bool = True,
    # Phase 2 closure (2026-05-25) — arch §7 chat-TX shape. See verify_post
    # docstring for semantics; passes through verbatim to
    # build_timechain_payload via _sign_and_attach / _assemble_signed_result.
    user_id: str = "",
    chat_id: str = "",
    turn_index: int = 0,
    topic_tags: Optional[list] = None,
) -> VerifiedResult:
    """Async verify_post with split safety / signing.

    D-SPEC-74 rationale:
      - Safety verification BLOCKS — truth-gate must clear before any byte
        leaves the producer (Maker 2026-05-17). If failed, the response
        is replaced with the guard_message and the function returns.
      - Signing is OPTIONALLY spawned as asyncio.create_task() when
        concurrent_sign=True. The VerifiedResult has `sign_task` attribute
        pointing to the in-flight task. Caller (agno_worker / chat SSE
        writer) can:
          * await result.sign_task before publishing the final SSE frame
            with ovg_headers — the recommended flow for buffered chat
          * NOT await it, letting the task finish in background — flush
            CHAT_STREAM_CHUNK with empty ovg_headers, attach headers on a
            follow-up final-frame when task resolves (rFP Chunk C SSE)

    Returns:
        VerifiedResult with `sign_task` set when concurrent_sign=True and
        safety PASSED. When concurrent_sign=False (legacy semantics),
        signing runs in-line and the result reflects the final signed state.

    For sync callsites (social_x_gateway, studio_worker) use `verify_post()`.
    """
    if not text:
        return VerifiedResult.unverified("")

    if output_verifier is None:
        logger.debug(
            "[llm_pipeline.verify_post_async] no output_verifier supplied — "
            "returning text unverified (channel=%s)", channel,
        )
        return VerifiedResult.unverified(text)

    # 1. Assemble chain_state (same as sync path)
    final_chain_state = state_gather.gather_chain_state(
        coordinator_snapshot=coordinator_snapshot,
        override=chain_state,
    )

    # 2. Phase 1: verify_safety (blocking)
    has_async_safety = hasattr(output_verifier, "verify_safety_async")
    try:
        if has_async_safety:
            safety = await output_verifier.verify_safety_async(
                output_text=text,
                channel=channel,
                injected_context=injected_context,
                prompt_text=prompt,
                chain_state=final_chain_state,
            )
        else:
            # Local OutputVerifier instance — has sync verify_safety
            safety = output_verifier.verify_safety(
                output_text=text,
                channel=channel,
                injected_context=injected_context,
                prompt_text=prompt,
                chain_state=final_chain_state,
            )
    except Exception as e:
        logger.warning(
            "[llm_pipeline.verify_post_async] verify_safety raised: %s", e
        )
        return VerifiedResult.unverified(text)

    passed = bool(getattr(safety, "passed", False))
    guard_message = getattr(safety, "guard_message", "") or ""
    guard_alert = getattr(safety, "guard_alert", None)
    violations = getattr(safety, "violations", []) or []
    violation_type = getattr(safety, "violation_type", None)
    safety_verdict_token = getattr(safety, "safety_verdict_token", "")
    verdict_ts = getattr(safety, "verdict_ts", 0.0)

    # 3. Compose final text (verified-with-guard / blocked / soft-alert)
    if not passed:
        logger.warning(
            "[llm_pipeline.verify_post_async:%s] BLOCKED (%s): %s",
            channel, violation_type, violations[:2],
        )
        # Blocked path NEVER signs (no token issued either). Build
        # an unsigned ovg_data with verified=false so the caller surfaces
        # the violation cleanly.
        return VerifiedResult(
            text=guard_message,
            passed=False,
            blocked=True,
            soft_alert=guard_alert,
            violation_type=violation_type,
            violations=violations,
            signature=None,
            ovg_data={
                "verified": False,
                "guard_alert": guard_alert,
                "guard_message": guard_message,
                "block_height": 0,
                "merkle_root": "",
                "signature": None,
            },
            timechain_committed=False,
            raw_result=safety,
        )

    # 4. Phase 2: sign_and_commit — concurrent OR inline
    has_async_sign = hasattr(output_verifier, "sign_and_commit_async")

    async def _sign_coroutine():
        try:
            if has_async_sign:
                return await output_verifier.sign_and_commit_async(
                    output_text=text,
                    channel=channel,
                    prompt_text=prompt,
                    chain_state=final_chain_state,
                    safety_verdict_token=safety_verdict_token,
                    verdict_ts=verdict_ts,
                )
            else:
                return output_verifier.sign_and_commit(
                    output_text=text,
                    channel=channel,
                    prompt_text=prompt,
                    chain_state=final_chain_state,
                    safety_verdict_token=safety_verdict_token,
                    verdict_ts=verdict_ts,
                )
        except Exception as e:
            logger.warning(
                "[llm_pipeline.verify_post_async] sign_and_commit raised: %s",
                e)
            return None

    # Build user-visible text now (safety PASSED) — does not wait for sign
    if guard_alert:
        logger.info(
            "[llm_pipeline.verify_post_async:%s] Soft alert: %s",
            channel, guard_alert)
        final_text = (text.rstrip() + "\n\n" + guard_message
                      if append_guard_on_pass else text)
    else:
        final_text = (text.rstrip() + "\n\n" + guard_message
                      if append_guard_on_pass else text)

    if concurrent_sign:
        # Spawn signing as a Task — caller awaits or fires-and-forgets
        sign_task = asyncio.create_task(
            _sign_and_attach(
                _sign_coroutine(),
                output_verifier=output_verifier,
                safety_result=safety,
                prompt=prompt,
                bus=bus,
                publish_timechain=publish_timechain,
                src_for_timechain=src_for_timechain,
                user_id=user_id,
                chat_id=chat_id,
                turn_index=turn_index,
                topic_tags=topic_tags,
            )
        )
        return VerifiedResult(
            text=final_text,
            passed=True,
            blocked=False,
            soft_alert=guard_alert,
            violation_type=violation_type,
            violations=violations,
            signature=None,            # filled when task resolves
            merkle_root="",
            block_height=0,
            ovg_data={
                "verified": True,
                "guard_alert": guard_alert,
                "guard_message": guard_message,
                "block_height": 0,
                "merkle_root": "",
                "signature": None,
            },
            timechain_committed=False,  # filled when task resolves
            raw_result=safety,
            sign_task=sign_task,
        )

    # Inline sign (legacy semantics — caller waits)
    signed = await _sign_coroutine()
    return _assemble_signed_result(
        final_text, safety, signed, output_verifier, prompt, bus,
        publish_timechain, src_for_timechain,
        guard_alert, violations, violation_type, append_guard_on_pass,
        user_id=user_id, chat_id=chat_id, turn_index=turn_index,
        topic_tags=topic_tags,
    )


async def _sign_and_attach(sign_coro, *, output_verifier, safety_result,
                           prompt: str, bus: Any, publish_timechain: bool,
                           src_for_timechain: str,
                           user_id: str = "", chat_id: str = "",
                           turn_index: int = 0,
                           topic_tags: Optional[list] = None):
    """Run the sign coroutine and publish TimeChain commit. Returns the
    SignedResult (or None on failure)."""
    signed = await sign_coro
    if signed is None or not getattr(signed, "signed", False):
        return signed

    # TimeChain publish
    if publish_timechain and bus is not None:
        try:
            # Reconstruct OVGResult-shaped object for build_timechain_payload
            from titan_hcl.logic.output_verifier import OVGResult
            ovg_compat = OVGResult(
                passed=True,
                output_text=safety_result.output_text,
                signature=signed.signature,
                block_height=signed.block_height,
                merkle_root=signed.merkle_root,
                genesis_hash=signed.genesis_hash,
                checks=safety_result.checks,
                violations=safety_result.violations,
                violation_type=safety_result.violation_type,
                channel=safety_result.channel,
                timestamp=signed.timestamp,
                guard_alert=safety_result.guard_alert,
                guard_message=safety_result.guard_message,
            )
            tc_payload = output_verifier.build_timechain_payload(
                ovg_compat, prompt_text=prompt,
                user_id=user_id, chat_id=chat_id, turn_index=turn_index,
                topic_tags=topic_tags)
            if tc_payload:
                from titan_hcl.bus import TIMECHAIN_COMMIT, make_msg
                bus.publish(
                    make_msg(TIMECHAIN_COMMIT, src_for_timechain, "timechain",
                             tc_payload))
        except Exception as e:
            logger.warning(
                "[llm_pipeline.verify_post_async] TimeChain commit failed: %s",
                e)
    return signed


def _assemble_signed_result(final_text: str, safety, signed,
                            output_verifier, prompt: str, bus: Any,
                            publish_timechain: bool, src_for_timechain: str,
                            guard_alert, violations, violation_type: str,
                            append_guard_on_pass: bool,
                            *,
                            user_id: str = "", chat_id: str = "",
                            turn_index: int = 0,
                            topic_tags: Optional[list] = None) -> VerifiedResult:
    """For concurrent_sign=False legacy path — assemble final result inline."""
    signature = getattr(signed, "signature", None) if signed else None
    merkle_root = getattr(signed, "merkle_root", "") if signed else ""
    block_height = int(getattr(signed, "block_height", 0) or 0) if signed else 0
    committed = False
    if publish_timechain and bus is not None and signed and signed.signed:
        try:
            from titan_hcl.logic.output_verifier import OVGResult
            ovg_compat = OVGResult(
                passed=True, output_text=safety.output_text,
                signature=signature, block_height=block_height,
                merkle_root=merkle_root,
                genesis_hash=getattr(signed, "genesis_hash", ""),
                checks=safety.checks, violations=safety.violations,
                violation_type=safety.violation_type,
                channel=safety.channel, timestamp=signed.timestamp,
                guard_alert=safety.guard_alert,
                guard_message=safety.guard_message,
            )
            tc_payload = output_verifier.build_timechain_payload(
                ovg_compat, prompt_text=prompt,
                user_id=user_id, chat_id=chat_id, turn_index=turn_index,
                topic_tags=topic_tags)
            if tc_payload:
                from titan_hcl.bus import TIMECHAIN_COMMIT, make_msg
                bus.publish(
                    make_msg(TIMECHAIN_COMMIT, src_for_timechain, "timechain",
                             tc_payload))
                committed = True
        except Exception as e:
            logger.warning(
                "[llm_pipeline.verify_post_async/inline] TimeChain commit "
                "failed: %s", e)

    return VerifiedResult(
        text=final_text, passed=True, blocked=False,
        soft_alert=guard_alert,
        violation_type=violation_type, violations=violations,
        signature=signature, merkle_root=merkle_root,
        block_height=block_height,
        ovg_data={
            "verified": True,
            "guard_alert": guard_alert,
            "guard_message": safety.guard_message,
            "block_height": block_height,
            "merkle_root": merkle_root,
            "signature": signature,
        },
        timechain_committed=committed,
        raw_result=signed if signed else safety,
    )
