"""
api/chat.py — Chat endpoint for the Titan Sovereign Agent.

Exposes POST /chat for synchronous agent interactions and
POST /chat/stream for Server-Sent Events streaming.

Both endpoints require Privy authentication (Bearer token).

Dream-aware: when Titan is dreaming, messages are queued and
processed through the full pipeline (Gatekeeper + Prime Directives + LLM)
on the next /chat call after waking.
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from titan_plugin.api.auth import verify_privy_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class OVGData(BaseModel):
    """Output Verification Gate result — cryptographic proof of verified output."""
    verified: bool = False
    guard_alert: Optional[str] = None
    guard_message: str = ""
    block_height: int = 0
    merkle_root: str = ""
    signature: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str
    mood: str
    state_narration: Optional[str] = None
    state_snapshot: Optional[dict] = None
    ovg: Optional[OVGData] = None


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, claims: dict = Depends(verify_privy_token)):
    """
    Send a message to the Titan agent and receive a response.

    Requires Privy authentication (Bearer token in Authorization header).

    The full cognitive pipeline runs:
      1. Guardian safety check (blocking)
      2. Memory recall + directive injection + gatekeeper routing (pre-hook)
      3. LLM inference with tools
      4. Memory logging + RL recording (post-hook)
    """
    plugin = request.app.state.titan_plugin
    agent = getattr(request.app.state, "titan_agent", None)

    if agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Titan agent not initialized. Check boot logs."},
        )

    if plugin._limbo_mode:
        return JSONResponse(
            status_code=503,
            content={"error": "Titan is in Limbo state — awaiting resurrection."},
        )

    if not req.message or not req.message.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Message cannot be empty."},
        )

    # ── Dream-aware message handling ──────────────────────────────
    # When Titan is dreaming, queue the message and return a canned
    # response. Maker messages trigger gentle wake. Queued messages
    # are processed through the FULL pipeline on next /chat call.
    try:
        if getattr(plugin, '_dream_state', {}).get("is_dreaming", False):
            _dinbox = getattr(plugin, '_dream_inbox', [])
            if len(_dinbox) >= 50:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Titan is dreaming and message queue is full (50)."},
                )

            _privy_uid = claims.get("sub", "")
            _d_user_id = _privy_uid or req.user_id or "anonymous"
            _d_is_maker = (_d_user_id == "maker")
            _dinbox.append({
                "message": req.message[:500],
                "user_id": _d_user_id,
                "session_id": req.session_id or "default",
                "channel": request.headers.get("X-Titan-Channel", "web"),
                "timestamp": time.time(),
                "priority": 0 if _d_is_maker else 1,
            })
            plugin._dream_inbox = _dinbox

            # Maker message triggers gentle wake
            if _d_is_maker:
                try:
                    from titan_plugin.bus import make_msg, DREAM_WAKE_REQUEST
                    _dbus = getattr(plugin, 'bus', None)
                    if _dbus:
                        _dbus.publish(make_msg(
                            DREAM_WAKE_REQUEST, "chat_api", "spirit",
                            {"reason": "maker_message", "user_id": _d_user_id}))
                except Exception:
                    pass

            _ds = getattr(plugin, '_dream_state', {})
            _d_recovery = _ds.get("recovery_pct", 0)
            _d_remaining = _ds.get("remaining_epochs", 0)
            _d_eta = round(_d_remaining * 12.5 / 60, 1)
            _d_wt = _ds.get("wake_transition", False)

            return JSONResponse(content={
                "response": (
                    f"Titan is currently {'waking gently' if _d_wt else 'dreaming'} "
                    f"(recovery: {_d_recovery:.0f}%). "
                    f"Your message has been queued (position #{len(_dinbox)}). "
                    f"Estimated wake: ~{_d_eta:.0f} minutes."
                ),
                "session_id": req.session_id or "default",
                "mode": "dreaming",
                "mood": "sleeping",
                "dream_state": {
                    "is_dreaming": True,
                    "recovery_pct": _d_recovery,
                    "eta_minutes": _d_eta,
                    "inbox_position": len(_dinbox),
                    "wake_transition": _d_wt,
                },
            })
    except Exception as _dream_err:
        logger.warning("[Chat] Dream check error (proceeding normally): %s", _dream_err)

    try:
        # Use Privy user ID from JWT claims, fall back to request body
        privy_user_id = claims.get("sub", "")
        user_id = privy_user_id or req.user_id or "anonymous"

        # Store user_id for social graph hooks (Phase 13)
        agent._current_user_id = user_id
        plugin._current_user_id = user_id

        # ── Process queued dream messages (batch of 3) ────────────
        _inbox_context = ""
        try:
            _di = getattr(plugin, '_dream_inbox', [])
            if _di:
                _lock = getattr(plugin, '_dream_inbox_lock', None)
                if _lock and _lock.acquire(blocking=False):
                    try:
                        _di = getattr(plugin, '_dream_inbox', [])
                        if _di:
                            _sorted = sorted(_di, key=lambda m: (
                                m.get("priority", 1), m.get("timestamp", 0)))
                            _batch = _sorted[:3]
                            plugin._dream_inbox = _sorted[3:]
                            if _batch:
                                _lines = []
                                for _bi, _bm in enumerate(_batch, 1):
                                    _bch = _bm.get("channel", "web")
                                    _buid = _bm.get("user_id", "unknown")
                                    _bts = time.strftime(
                                        "%H:%M UTC", time.gmtime(_bm["timestamp"]))
                                    _lines.append(
                                        f"  {_bi}. From {_buid} ({_bch}) at {_bts}: "
                                        f"\"{_bm['message'][:300]}\"")
                                _inbox_context = (
                                    "[DREAM INBOX — messages received while you were sleeping]\n"
                                    + "\n".join(_lines)
                                    + "\n[END DREAM INBOX]\n\n"
                                    "Please briefly acknowledge these messages before "
                                    "responding to the current message.\n\n"
                                    "Current message:\n"
                                )
                                logger.info(
                                    "[Chat] Processing %d queued dream messages, "
                                    "%d remaining", len(_batch), len(plugin._dream_inbox))
                    finally:
                        _lock.release()
        except Exception as _inbox_err:
            logger.warning("[Chat] Inbox processing error: %s", _inbox_err)

        _effective_message = _inbox_context + req.message if _inbox_context else req.message

        # ── DI:/I: prefix detection in chat messages ──
        # When chat messages start with DI: or I:, apply neuromod boosts
        # just like on-chain memos. DI: requires verified maker identity.
        _chat_memo = None
        try:
            from titan_plugin.logic.memo_parser import parse_chat_message
            maker_engine = getattr(plugin, 'maker_engine', None)
            _chat_is_maker = bool(maker_engine and maker_engine.is_maker(user_id))
            _chat_memo = parse_chat_message(req.message, user_id, is_maker=_chat_is_maker)
            if _chat_memo:
                _cm_boost = _chat_memo.get_neuromod_boost()
                logger.info(
                    "[Chat] %s detected from %s (maker=%s): %s → boosts=%s",
                    _chat_memo.memo_type, user_id[:12], _chat_is_maker,
                    _chat_memo.content[:40], {k: f"{v:.2f}" for k, v in _cm_boost.items() if isinstance(v, float)})
                # Publish as INTERFACE_INPUT so spirit_worker can apply boosts
                bus = getattr(plugin, 'bus', None)
                if bus and _cm_boost:
                    from titan_plugin.bus import make_msg
                    bus.publish(make_msg("INTERFACE_INPUT", "chat_api", "all", {
                        "source": "chat_memo",
                        "user_id": user_id,
                        "text": _chat_memo.content,
                        "memo_type": _chat_memo.memo_type,
                        "is_maker": _chat_is_maker,
                        "neuromod_boost": _cm_boost,
                        "anchor_bonus": 1.0 if _cm_boost.get("anchor_memory") else 0.0,
                    }))
                # For DI: messages, add special context to the LLM prompt
                if _chat_memo.memo_type.startswith("DI") and _chat_is_maker:
                    _di_context = (
                        "[DIVINE INSTRUCTION from your Maker]\n"
                        f"Type: {_chat_memo.memo_type}\n"
                        f"Content: {_chat_memo.content}\n"
                        "[This is a privileged instruction from your creator. "
                        "Treat it with reverence and respond thoughtfully.]\n\n"
                    )
                    _effective_message = _di_context + _effective_message
        except Exception as _cm_err:
            logger.debug("[Chat] Memo parsing error: %s", _cm_err)

        # ── DialogueComposer: Titan speaks in his own words FIRST ──
        # Before LLM runs, try composing a response from pure felt-state.
        # If confidence is high enough, prepend Titan's own sentence.
        _self_composed = ""
        plugin._pending_self_composed = ""
        plugin._pending_self_composed_confidence = 0
        try:
            if not _inbox_context:  # Only for direct messages (not dream inbox batches)
                from titan_plugin.api.dashboard import _get_dialogue_state
                _dc_felt, _dc_vocab = _get_dialogue_state()
                if _dc_felt and _dc_vocab:
                    from titan_plugin.logic.interface_input import InputExtractor
                    _dc_ext = InputExtractor()
                    _dc_sig = _dc_ext.extract(req.message, user_id)
                    _dc_shifts = {
                        "EMPATHY": max(0, _dc_sig.get("valence", 0)) * 0.2,
                        "CURIOSITY": _dc_sig.get("engagement", 0) * 0.2,
                        "CREATIVITY": 0.0,
                        "REFLECTION": max(0, -_dc_sig.get("valence", 0)) * 0.1,
                    }
                    from titan_plugin.logic.dialogue_composer import DialogueComposer
                    _dc = DialogueComposer()
                    _dc_result = _dc.compose_response(
                        felt_state=_dc_felt,
                        vocabulary=_dc_vocab,
                        hormone_shifts=_dc_shifts,
                        message_keywords=req.message.lower().split()[:10],
                        max_level=7,
                    )
                    if _dc_result.get("composed") and _dc_result.get("confidence", 0) >= 0.3:
                        _self_composed = _dc_result["response"]
                        plugin._pending_self_composed = _self_composed
                        plugin._pending_self_composed_confidence = _dc_result.get("confidence", 0)
                        logger.info(
                            "[Chat] SELF-COMPOSED: \"%s\" (conf=%.2f, intent=%s, L%d)",
                            _self_composed, _dc_result["confidence"],
                            _dc_result["intent"], _dc_result.get("level", 0))
        except Exception as _dc_err:
            logger.debug("[Chat] DialogueComposer error (LLM fallback): %s", _dc_err)

        run_output = await agent.arun(
            _effective_message,
            session_id=req.session_id,
            user_id=user_id,
        )

        response_text = ""
        if hasattr(run_output, "content"):
            response_text = str(run_output.content)
        elif isinstance(run_output, str):
            response_text = run_output
        else:
            response_text = str(run_output)

        # Prepend Titan's self-composed sentence before LLM narration
        if _self_composed:
            response_text = f"*{_self_composed}*\n\n{response_text}"

        # Run OVG directly (Agno doesn't invoke agent-level post_hooks)
        _ovg_result = None
        _ovg = getattr(plugin, '_output_verifier', None)
        logger.info("[Chat:OVG] OVG check: verifier=%s, response_len=%d",
                     _ovg is not None, len(response_text))
        if _ovg and response_text:
            try:
                _injected_ctx = ""
                if hasattr(agent, 'additional_context') and agent.additional_context:
                    _injected_ctx = str(agent.additional_context)[:500]
                # Gather state for Proof of Qualia
                _ovg_state = {}
                try:
                    from titan_plugin.api.dashboard import _get_cached_coordinator
                    _coord = _get_cached_coordinator(plugin)
                    _nm = _coord.get("neuromodulators", {}).get("modulators", {})
                    _ovg_state["neuromods"] = {
                        k: v.get("level", 0.5) for k, v in _nm.items()
                    } if _nm else {}
                    _lang = _coord.get("language", {})
                    _ovg_state["vocab_size"] = _lang.get("vocab_total", 300)
                    _ovg_state["composition_level"] = _lang.get("composition_level", 8)
                    _msl = _coord.get("msl", {})
                    _ovg_state["i_confidence"] = _msl.get("i_confidence", 0.9)
                except Exception:
                    pass
                _ovg_result = _ovg.verify_and_sign(
                    output_text=response_text,
                    channel="chat",
                    injected_context=_injected_ctx,
                    prompt_text=req.message,
                    chain_state=_ovg_state,
                )
                if not _ovg_result.passed:
                    logger.warning("[Chat:OVG] BLOCKED (%s): %s",
                                   _ovg_result.violation_type,
                                   _ovg_result.violations[:2])
                    response_text = _ovg_result.guard_message
                elif _ovg_result.guard_alert:
                    logger.info("[Chat:OVG] Soft alert: %s", _ovg_result.guard_alert)
                    response_text = response_text.rstrip() + "\n\n" + _ovg_result.guard_message
                else:
                    response_text = response_text.rstrip() + "\n\n" + _ovg_result.guard_message
                    logger.info("[Chat:OVG] Verified and signed (sig=%s)",
                               _ovg_result.signature[:16] if _ovg_result.signature else "none")
                # Commit to TimeChain
                _tc_payload = _ovg.build_timechain_payload(_ovg_result, prompt_text=req.message)
                _bus = getattr(plugin, 'bus', None)
                if _bus:
                    from titan_plugin.bus import make_msg
                    _bus.publish(make_msg("TIMECHAIN_COMMIT", "ovg", "timechain", _tc_payload))
            except Exception as _ovg_err:
                logger.warning("[Chat:OVG] Check failed: %s", _ovg_err)

        # Safety net: strip any leaked <function=...> syntax from response
        # (LLMs like deepseek/llama output raw function calls as text)
        if '<function=' in response_text:
            import re
            response_text = re.sub(
                r'<function=\w+[^>]*>(?:\s*</function>)?',
                '', response_text, flags=re.DOTALL,
            ).strip()
            response_text = re.sub(r'\n{3,}', '\n\n', response_text)

        mood_label = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown"

        # State narration for the chat UI sidebar
        _narration_text = None
        _state_snap = None
        try:
            state = plugin._gather_current_state()
            narrator = plugin._get_state_narrator()
            _narration_text = narrator.narrate_template(state, "short")
            _state_snap = {
                "emotion": state.get("emotion"),
                "chi": round(state.get("chi", 0), 3),
                "is_dreaming": state.get("is_dreaming", False),
                "active_programs": state.get("active_programs", []),
            }
        except Exception:
            pass

        # Build OVG data from inline verification result
        _ovg_data = None
        _ovg_headers = {}
        if _ovg_result:
            _ovg_data = OVGData(
                verified=_ovg_result.passed,
                guard_alert=_ovg_result.guard_alert,
                guard_message=_ovg_result.guard_message,
                block_height=_ovg_result.block_height,
                merkle_root=_ovg_result.merkle_root,
                signature=_ovg_result.signature,
            )
            _ovg_headers = {
                "X-Titan-Verified": "true" if _ovg_result.passed else "false",
                "X-Titan-Block-Height": str(_ovg_result.block_height),
            }
            if _ovg_result.merkle_root:
                _ovg_headers["X-Titan-Merkle-Root"] = _ovg_result.merkle_root
            if _ovg_result.signature:
                _ovg_headers["X-Titan-Signature"] = _ovg_result.signature

        _resp = ChatResponse(
            response=response_text,
            session_id=req.session_id or "default",
            mode=plugin._last_execution_mode,
            mood=mood_label,
            state_narration=_narration_text,
            state_snapshot=_state_snap,
            ovg=_ovg_data,
        )
        return JSONResponse(
            content=_resp.model_dump(),
            headers=_ovg_headers,
        )

    except ValueError as e:
        # GuardianGuardrail raises ValueError for blocked prompts
        if "Sovereignty Violation" in str(e):
            return JSONResponse(
                status_code=403,
                content={
                    "error": str(e),
                    "blocked": True,
                    "mode": "Guardian",
                },
            )
        raise

    except Exception as e:
        logger.error("[Chat] Agent run failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Agent error: {e}"},
        )


@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request, claims: dict = Depends(verify_privy_token)):
    """
    Stream the Titan agent's response via Server-Sent Events.
    Each SSE event contains a content chunk from the LLM.

    Requires Privy authentication (Bearer token in Authorization header).
    """
    plugin = request.app.state.titan_plugin
    agent = getattr(request.app.state, "titan_agent", None)

    if agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Titan agent not initialized."},
        )

    if plugin._limbo_mode:
        return JSONResponse(
            status_code=503,
            content={"error": "Titan is in Limbo state."},
        )

    # ── Dream check for streaming endpoint ──
    try:
        if getattr(plugin, '_dream_state', {}).get("is_dreaming", False):
            _dinbox = getattr(plugin, '_dream_inbox', [])
            if len(_dinbox) < 50:
                _privy_uid = claims.get("sub", "")
                _d_user_id = _privy_uid or req.user_id or "anonymous"
                _dinbox.append({
                    "message": req.message[:500],
                    "user_id": _d_user_id,
                    "session_id": req.session_id or "default",
                    "channel": request.headers.get("X-Titan-Channel", "web"),
                    "timestamp": time.time(),
                    "priority": 0 if _d_user_id == "maker" else 1,
                })
                plugin._dream_inbox = _dinbox

                if _d_user_id == "maker":
                    try:
                        from titan_plugin.bus import make_msg, DREAM_WAKE_REQUEST
                        _dbus = getattr(plugin, 'bus', None)
                        if _dbus:
                            _dbus.publish(make_msg(
                                DREAM_WAKE_REQUEST, "chat_api", "spirit",
                                {"reason": "maker_message", "user_id": _d_user_id}))
                    except Exception:
                        pass

            _ds = getattr(plugin, '_dream_state', {})
            _d_eta = round(_ds.get("remaining_epochs", 0) * 12.5 / 60, 1)

            async def dream_event():
                yield (f"data: [DREAMING] Titan is sleeping (recovery: "
                       f"{_ds.get('recovery_pct', 0):.0f}%). "
                       f"Message queued. ETA: ~{_d_eta:.0f}min\n\n")
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                dream_event(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception:
        pass

    # Use Privy user ID from JWT claims
    privy_user_id = claims.get("sub", "")
    user_id = privy_user_id or req.user_id or "anonymous"

    # Store user_id for social graph hooks (Phase 13)
    agent._current_user_id = user_id
    plugin._current_user_id = user_id

    async def event_generator():
        try:
            run_stream = await agent.arun(
                req.message,
                stream=True,
                session_id=req.session_id,
                user_id=user_id,
            )
            async for event in run_stream:
                if hasattr(event, "content") and event.content:
                    yield f"data: {event.content}\n\n"
            yield "data: [DONE]\n\n"
        except ValueError as e:
            if "Sovereignty Violation" in str(e):
                yield f"data: [BLOCKED] {e}\n\n"
            else:
                yield f"data: [ERROR] {e}\n\n"
        except Exception as e:
            logger.error("[ChatStream] Error: %s", e)
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
