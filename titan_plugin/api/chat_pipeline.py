"""
api/chat_pipeline.py — shared /chat pipeline implementation.

Single source of truth for the chat run logic, called from both:
  - titan_plugin/legacy_core.py:TitanCore   (legacy monolithic runtime,
    kernel_plugin_split_enabled=false; current default on T1)
  - titan_plugin/core/plugin.py:TitanPlugin (microkernel V6 runtime,
    kernel_plugin_split_enabled=true)

Both classes have a thin `async def run_chat(self, payload, claims, headers)`
wrapper that calls `await run_chat(self, payload, claims, headers)` here.

The function uses a `core` parameter (the runtime instance — TitanCore or
TitanPlugin); attribute access goes through `core` only and is purely
read-or-defensive (`getattr(core, "X", default)` patterns), so any
attribute that one runtime has but the other doesn't degrades gracefully.

Lives on the runtime side (parent process) because the Agno agent +
gatekeeper + memory + OVG state cannot be safely re-constructed in the
api subprocess (would double LLM clients + drift state + lose tool
wiring — see BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS).

Returns a serializable dict — chat.py wraps in JSONResponse:
    {"status_code": int,
     "body":        dict,           # ChatResponse-shaped or {"error": str}
     "extra_headers": dict | None}  # X-Titan-* OVG headers

Callers from two places:
  1. In-process mode (api_process_separation_enabled=false):
     chat.py endpoint calls `await core.run_chat(...)` directly from
     the api server's event loop (which IS the parent loop).
  2. Subprocess mode (api_process_separation_enabled=true):
     chat.py endpoint forwards via `bus.request_async(dst="chat_handler",
     ...)`. Parent's _chat_handler_loop receives the QUERY, calls
     `await core.run_chat(...)`, replies with RESPONSE rid-routed.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from titan_plugin import bus

logger = logging.getLogger(__name__)


async def run_chat(
    core,
    payload: dict,
    claims: dict,
    headers: Optional[dict] = None,
) -> dict:
    """Full chat pipeline — agent.arun + dream inbox + memo parsing +
    DialogueComposer + OVG verification + state narration.

    Args:
        core:    runtime instance (TitanCore or TitanPlugin) — must expose
                 `_agent` (Agno Agent) and the various attributes touched
                 by the chat path (bus, mood_engine, _output_verifier,
                 maker_engine, _gather_current_state, _get_state_narrator,
                 _dream_state, _dream_inbox, _dream_inbox_lock,
                 _last_execution_mode). Missing attributes degrade
                 gracefully (defensive `getattr` everywhere).
        payload: {"message": str, "session_id": str|None, "user_id": str|None}
        claims:  Privy JWT claims dict (has "sub" = privy user id)
        headers: HTTP headers dict (uses "X-Titan-Channel" only)

    Faithful port of the previous chat() body in api/chat.py. Logic
    unchanged; only access patterns are remapped (request.app.state
    replaced with core; req.X with payload[X]; JSONResponse with the
    return-dict shape).
    """
    headers = headers or {}
    plugin = core
    agent = core._agent

    if agent is None:
        return {
            "status_code": 503,
            "body": {"error": "Titan agent not initialized. Check boot logs."},
            "extra_headers": None,
        }

    if getattr(plugin, "_limbo_mode", False):
        return {
            "status_code": 503,
            "body": {"error": "Titan is in Limbo state — awaiting resurrection."},
            "extra_headers": None,
        }

    message = payload.get("message", "")
    if not message or not message.strip():
        return {
            "status_code": 400,
            "body": {"error": "Message cannot be empty."},
            "extra_headers": None,
        }

    session_id_in = payload.get("session_id")
    user_id_in = payload.get("user_id")
    channel = headers.get("X-Titan-Channel", "web")

    # ── Dream-aware message handling ──────────────────────────────
    try:
        if getattr(plugin, "_dream_state", {}).get("is_dreaming", False):
            _dinbox = getattr(plugin, "_dream_inbox", [])
            if len(_dinbox) >= 50:
                return {
                    "status_code": 429,
                    "body": {"error": "Titan is dreaming and message queue is full (50)."},
                    "extra_headers": None,
                }
            _privy_uid = claims.get("sub", "")
            _d_user_id = _privy_uid or user_id_in or "anonymous"
            _d_is_maker = (_d_user_id == "maker")
            _dinbox.append({
                "message": message[:500],
                "user_id": _d_user_id,
                "session_id": session_id_in or "default",
                "channel": channel,
                "timestamp": time.time(),
                "priority": 0 if _d_is_maker else 1,
            })
            plugin._dream_inbox = _dinbox

            if _d_is_maker:
                try:
                    from titan_plugin.bus import make_msg, DREAM_WAKE_REQUEST
                    _dbus = getattr(plugin, "bus", None)
                    if _dbus:
                        _dbus.publish(make_msg(
                            DREAM_WAKE_REQUEST, "chat_api", "spirit",
                            {"reason": "maker_message", "user_id": _d_user_id}))
                except Exception:
                    pass

            _ds = getattr(plugin, "_dream_state", {})
            _d_recovery = _ds.get("recovery_pct", 0)
            _d_remaining = _ds.get("remaining_epochs", 0)
            _d_eta = round(_d_remaining * 12.5 / 60, 1)
            _d_wt = _ds.get("wake_transition", False)

            return {
                "status_code": 200,
                "body": {
                    "response": (
                        f"Titan is currently {'waking gently' if _d_wt else 'dreaming'} "
                        f"(recovery: {_d_recovery:.0f}%). "
                        f"Your message has been queued (position #{len(_dinbox)}). "
                        f"Estimated wake: ~{_d_eta:.0f} minutes."
                    ),
                    "session_id": session_id_in or "default",
                    "mode": "dreaming",
                    "mood": "sleeping",
                    "dream_state": {
                        "is_dreaming": True,
                        "recovery_pct": _d_recovery,
                        "eta_minutes": _d_eta,
                        "inbox_position": len(_dinbox),
                        "wake_transition": _d_wt,
                    },
                },
                "extra_headers": None,
            }
    except Exception as _dream_err:
        logger.warning("[Chat] Dream check error (proceeding normally): %s", _dream_err)

    try:
        privy_user_id = claims.get("sub", "")
        user_id = privy_user_id or user_id_in or "anonymous"

        agent._current_user_id = user_id
        plugin._current_user_id = user_id

        # ── Process queued dream messages (batch of 3) ────────────
        _inbox_context = ""
        try:
            _di = getattr(plugin, "_dream_inbox", [])
            if _di:
                _lock = getattr(plugin, "_dream_inbox_lock", None)
                if _lock and _lock.acquire(blocking=False):
                    try:
                        _di = getattr(plugin, "_dream_inbox", [])
                        if _di:
                            _sorted = sorted(_di, key=lambda m: (
                                m.get("priority", 1), m.get("timestamp", 0)))
                            _batch = _sorted[:3]
                            # L3 Phase A.8.1: preserve deque(maxlen=256) via
                            # clear+extend (slice assignment would replace
                            # the bounded deque with an unbounded list).
                            plugin._dream_inbox.clear()
                            plugin._dream_inbox.extend(_sorted[3:])
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
                                    "%d remaining", len(_batch),
                                    len(getattr(plugin, "_dream_inbox", []) or []))
                    finally:
                        _lock.release()
        except Exception as _inbox_err:
            logger.warning("[Chat] Inbox processing error: %s", _inbox_err)

        _effective_message = _inbox_context + message if _inbox_context else message

        # ── DI:/I: prefix detection in chat messages ──
        _chat_memo = None
        try:
            from titan_plugin.logic.memo_parser import parse_chat_message
            maker_engine = getattr(plugin, "maker_engine", None)
            _chat_is_maker = bool(maker_engine and maker_engine.is_maker(user_id))
            _chat_memo = parse_chat_message(message, user_id, is_maker=_chat_is_maker)
            if _chat_memo:
                _cm_boost = _chat_memo.get_neuromod_boost()
                logger.info(
                    "[Chat] %s detected from %s (maker=%s): %s → boosts=%s",
                    _chat_memo.memo_type, user_id[:12], _chat_is_maker,
                    _chat_memo.content[:40],
                    {k: f"{v:.2f}" for k, v in _cm_boost.items()
                     if isinstance(v, float)})
                _pbus = getattr(plugin, "bus", None)
                if _pbus and _cm_boost:
                    from titan_plugin.bus import make_msg
                    _pbus.publish(make_msg(bus.INTERFACE_INPUT, "chat_api", "all", {
                        "source": "chat_memo",
                        "user_id": user_id,
                        "text": _chat_memo.content,
                        "memo_type": _chat_memo.memo_type,
                        "is_maker": _chat_is_maker,
                        "neuromod_boost": _cm_boost,
                        "anchor_bonus": 1.0 if _cm_boost.get("anchor_memory") else 0.0,
                    }))
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
        _self_composed = ""
        plugin._pending_self_composed = ""
        plugin._pending_self_composed_confidence = 0
        try:
            if not _inbox_context:
                from titan_plugin.api.dashboard import _get_dialogue_state
                _dc_felt, _dc_vocab = _get_dialogue_state()
                if _dc_felt and _dc_vocab:
                    from titan_plugin.logic.interface_input import InputExtractor
                    _dc_ext = InputExtractor()
                    _dc_sig = _dc_ext.extract(message, user_id)
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
                        message_keywords=message.lower().split()[:10],
                        max_level=7,
                    )
                    if (_dc_result.get("composed")
                            and _dc_result.get("confidence", 0) >= 0.3):
                        _self_composed = _dc_result["response"]
                        plugin._pending_self_composed = _self_composed
                        plugin._pending_self_composed_confidence = (
                            _dc_result.get("confidence", 0))
                        logger.info(
                            "[Chat] SELF-COMPOSED: \"%s\" (conf=%.2f, intent=%s, L%d)",
                            _self_composed, _dc_result["confidence"],
                            _dc_result["intent"], _dc_result.get("level", 0))
        except Exception as _dc_err:
            logger.debug("[Chat] DialogueComposer error (LLM fallback): %s", _dc_err)

        run_output = await agent.arun(
            _effective_message,
            session_id=session_id_in,
            user_id=user_id,
        )

        response_text = ""
        if hasattr(run_output, "content"):
            response_text = str(run_output.content)
        elif isinstance(run_output, str):
            response_text = run_output
        else:
            response_text = str(run_output)

        if _self_composed:
            response_text = f"*{_self_composed}*\n\n{response_text}"

        # ── OVG check ────────────────────────────────────────────
        _ovg_result = None
        _ovg = getattr(plugin, "_output_verifier", None)
        logger.info("[Chat:OVG] OVG check: verifier=%s, response_len=%d",
                    _ovg is not None, len(response_text))
        if _ovg and response_text:
            try:
                _injected_ctx = ""
                if hasattr(agent, "additional_context") and agent.additional_context:
                    _injected_ctx = str(agent.additional_context)[:500]
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
                    prompt_text=message,
                    chain_state=_ovg_state,
                )
                if not _ovg_result.passed:
                    logger.warning("[Chat:OVG] BLOCKED (%s): %s",
                                   _ovg_result.violation_type,
                                   _ovg_result.violations[:2])
                    response_text = _ovg_result.guard_message
                elif _ovg_result.guard_alert:
                    logger.info("[Chat:OVG] Soft alert: %s", _ovg_result.guard_alert)
                    response_text = (response_text.rstrip() + "\n\n"
                                     + _ovg_result.guard_message)
                else:
                    response_text = (response_text.rstrip() + "\n\n"
                                     + _ovg_result.guard_message)
                    logger.info("[Chat:OVG] Verified and signed (sig=%s)",
                                _ovg_result.signature[:16]
                                if _ovg_result.signature else "none")
                _tc_payload = _ovg.build_timechain_payload(
                    _ovg_result, prompt_text=message)
                _bus = getattr(plugin, "bus", None)
                if _bus:
                    from titan_plugin.bus import make_msg
                    _bus.publish(make_msg(
                        bus.TIMECHAIN_COMMIT, "ovg", "timechain", _tc_payload))
            except Exception as _ovg_err:
                logger.warning("[Chat:OVG] Check failed: %s", _ovg_err)

        # Safety net: strip any leaked <function=...> syntax
        if "<function=" in response_text:
            response_text = re.sub(
                r"<function=\w+[^>]*>(?:\s*</function>)?",
                "", response_text, flags=re.DOTALL,
            ).strip()
            response_text = re.sub(r"\n{3,}", "\n\n", response_text)

        # Mood label
        mood_label = "Unknown"
        try:
            _ml = (plugin.mood_engine.get_mood_label()
                   if getattr(plugin, "mood_engine", None) else None)
            if isinstance(_ml, str) and _ml:
                mood_label = _ml
        except Exception:
            pass

        # State narration for chat UI sidebar
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

        # Build OVG data dict (matches OVGData pydantic shape)
        _ovg_data = None
        _ovg_headers = {}
        if _ovg_result:
            _ovg_data = {
                "verified": bool(_ovg_result.passed),
                "guard_alert": _ovg_result.guard_alert,
                "guard_message": _ovg_result.guard_message,
                "block_height": int(_ovg_result.block_height),
                "merkle_root": _ovg_result.merkle_root,
                "signature": _ovg_result.signature,
            }
            _ovg_headers = {
                "X-Titan-Verified": "true" if _ovg_result.passed else "false",
                "X-Titan-Block-Height": str(_ovg_result.block_height),
            }
            if _ovg_result.merkle_root:
                _ovg_headers["X-Titan-Merkle-Root"] = _ovg_result.merkle_root
            if _ovg_result.signature:
                _ovg_headers["X-Titan-Signature"] = _ovg_result.signature

        # ChatResponse-shaped body (matches ChatResponse pydantic dump)
        _resp_body = {
            "response": response_text,
            "session_id": session_id_in or "default",
            "mode": getattr(plugin, "_last_execution_mode", "") or "",
            "mood": mood_label,
            "state_narration": _narration_text,
            "state_snapshot": _state_snap,
            "ovg": _ovg_data,
        }
        return {
            "status_code": 200,
            "body": _resp_body,
            "extra_headers": _ovg_headers or None,
        }

    except ValueError as e:
        # GuardianGuardrail raises ValueError for blocked prompts
        if "Sovereignty Violation" in str(e):
            return {
                "status_code": 403,
                "body": {
                    "error": str(e),
                    "blocked": True,
                    "mode": "Guardian",
                },
                "extra_headers": None,
            }
        logger.error("[Chat] Agent ValueError: %s", e, exc_info=True)
        return {
            "status_code": 500,
            "body": {"error": f"Agent error: {e}"},
            "extra_headers": None,
        }
    except Exception as e:
        logger.error("[Chat] Agent run failed: %s", e, exc_info=True)
        return {
            "status_code": 500,
            "body": {"error": f"Agent error: {e}"},
            "extra_headers": None,
        }
