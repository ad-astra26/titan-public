"""
api/chat.py — Chat endpoint for the Titan Sovereign Agent.

Exposes POST /chat for synchronous agent interactions and
POST /chat/stream for Server-Sent Events streaming.

Both endpoints require Privy authentication (Bearer token).

Phase C v1.17.0 (D-SPEC-72 / SPEC §4.R — see SPEC §9.B agno_worker block):
this file is now a THIN ENVELOPE over `agno_proxy.chat()` / `chat_stream()`.
All chat-pipeline business logic (dream-state gate, DialogueComposer pre,
Agno agent.arun, OVG verify+sign+TimeChain commit, memory recall, social
graph hooks) lives in `agno_worker` subprocess; this endpoint only:

  1. Validates the request (auth + payload shape)
  2. Forwards via CHAT_REQUEST bus event through AgnoProxy
  3. Maps the run_chat-shaped response → FastAPI JSONResponse / SSE

Replaces the pre-v1.17.0 path `chat.py → plugin.run_chat() → chat_pipeline.run_chat() →
agent.arun() in-parent`. chat_pipeline.py was DELETED in Chunk H (Q5 LOCKED).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from titan_hcl.api.auth import verify_privy_token, resolve_maker_pubkey
from titan_hcl.api.maker_presence_session import (
    session_key_from_claims, emit_maker_presence)
from titan_hcl.logic.maker_engine import resolve_maker_presence, MakerPresence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ─────────────────────────────────────────────────────────────────────
# RFP_affective_grounding_loop §7.D (D.0/D.1/D.2) — verified-Maker presence
# ─────────────────────────────────────────────────────────────────────

def _resolve_chat_presence(request: Request, claims: Optional[dict],
                           user_id: str, channel: str) -> MakerPresence:
    """Resolve verified-Maker presence for a chat turn (RFP §7.D D.0/D.1).

    • web  — cryptographic proof = a nonce-signed verified-Maker session marker
             (D.0), bound to this Privy session.
    • app  — the co-located Console (internal-key authenticated) relays its
             device pairing-binding (Ed25519) verification via X-Titan-Maker-
             Verified; a bare internal-key chat-test script does NOT set it.
    Honest: an unverified "maker" claim (incl. the internal-key chat-test bypass,
    user_id=="maker") yields is_maker=True (behaviour preserved) but verified=
    False → no maker_bond (INV-AFF-HONEST / GD3)."""
    ch = (channel or "web").strip().lower()
    claimed = (user_id == "maker")
    crypto_verified = False
    if ch in ("web", "chat", "observatory"):
        store = getattr(request.app.state, "maker_presence_sessions", None)
        if store is not None:
            crypto_verified = store.is_verified(session_key_from_claims(claims))
    elif ch == "app":
        is_internal = bool(claims and claims.get("iss") == "titan-internal")
        relayed = request.headers.get("X-Titan-Maker-Verified", "") == "1"
        crypto_verified = is_internal and relayed
    return resolve_maker_presence(ch, claimed_maker=claimed,
                                  crypto_verified=crypto_verified)


def _emit_maker_presence(request: Request, presence: MakerPresence) -> None:
    """Fire the cross-platform maker_bond tap on a cryptographically-verified
    Maker chat turn (RFP §7.D D.2). Only verified presence emits (GD3)."""
    if not presence.verified:
        return
    emit_maker_presence(request, presence.channel)


# ─────────────────────────────────────────────────────────────────────
# Pydantic models (API contract — unchanged)
# ─────────────────────────────────────────────────────────────────────

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
    # Non-null when a deterministic tool ran this turn (PreHook force / OVG
    # PostHook salvage): {tool, phase, executed, success, verdict, summary,
    # salvaged}. Lets the UI show "Titan verified this via its sandbox" + explain
    # the extra latency. (2026-06-01 tool-backstop.)
    tool_activity: Optional[dict] = None
    # §7.B (B.4): non-null on a NON-verifiable turn (direct/research/IDK). The UI
    # returns it to POST /v6/synthesis/turn_feedback so a user/Maker rating attaches
    # to this turn's stashed decision (the teaching loop). None otherwise.
    reasoning_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────
# AgnoProxy resolver
# ─────────────────────────────────────────────────────────────────────

def _get_agno_proxy(request: Request):
    """Resolve the AgnoProxy instance for this request.

    D-SPEC-73 (SPEC v1.18.0) resolution order:
      1. `app.state.agno_proxy` — installed by api factory at boot with
         the correct backend (AgnoBridgeClient in api_subprocess, or
         DivineBus in parent process).
      2. `app.state.agno_bridge` — present in api_subprocess context;
         lazy-construct an AgnoProxy(bridge=...) for the first-call
         transition window.
      3. `plugin._proxies["agno"]` — parent-process fallback if installed.
      4. None — no chat path available; /chat returns 503.

    The pre-D-SPEC-73 path that lazy-constructed `AgnoProxy(plugin.bus)`
    DELETED — that path called `bus.subscribe` via kernel_rpc and 500'd
    fleet-wide (CHAT-500). Replaced by bridge backend (no shim per
    feedback_no_shim_old_path_must_be_deleted.md).
    """
    cached = getattr(request.app.state, "agno_proxy", None)
    if cached is not None:
        return cached

    bridge = getattr(request.app.state, "agno_bridge", None)
    if bridge is not None:
        from titan_hcl.proxies.agno_proxy import AgnoProxy
        proxy = AgnoProxy(bridge=bridge)
        request.app.state.agno_proxy = proxy
        return proxy

    plugin = getattr(request.app.state, "titan_hcl", None)
    if plugin is not None:
        proxies = getattr(plugin, "_proxies", None)
        if isinstance(proxies, dict):
            from_dict = proxies.get("agno")
            if from_dict is not None:
                request.app.state.agno_proxy = from_dict
                return from_dict

    return None


def _result_to_response(result: dict) -> JSONResponse:
    """Map AgnoProxy.chat result dict → FastAPI JSONResponse with OVG headers."""
    return JSONResponse(
        status_code=int(result.get("status_code", 500)),
        content=result.get("body") or {"error": "empty body"},
        headers=result.get("extra_headers") or {},
    )


# ─────────────────────────────────────────────────────────────────────
# POST /chat — synchronous chat
# ─────────────────────────────────────────────────────────────────────


def _compute_ip_hash(request: Request) -> str:
    """Presence Memory §7.F (F.2) — hash the client IP at the edge so the RAW IP
    never leaves this process (never onto the bus, never into a store). Titan is
    behind its OWN nginx → the left-most X-Forwarded-For hop is the real client.
    Same per-Titan salt (api.internal_key) as the agno subprocess → consistent
    with the did_hash computed there. Soft: any failure → '' (no signal)."""
    try:
        from titan_hcl.params import get_params
        from titan_hcl.utils.identity_hash import (
            client_ip_from_xff, derive_salt, identity_hash,
        )
        salt = derive_salt((get_params("api") or {}).get("internal_key", "") or "")
        ip = client_ip_from_xff(
            request.headers.get("X-Forwarded-For", ""),
            request.client.host if request.client else "")
        return identity_hash(ip, salt)
    except Exception:
        return ""


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request,
               claims: dict = Depends(verify_privy_token)):
    """Send a message to the Titan agent and receive a response.

    Requires Privy authentication (Bearer token in Authorization header).

    Pipeline (lives in agno_worker subprocess per D-SPEC-72):
      1. Dream-state gate (SHM read) — if dreaming, buffer + return dream-mode
      2. Pre-hook chain (Guardian / memory recall / directive injection /
         gatekeeper routing / DialogueComposer felt-state sentence)
      3. agent.arun() — Agno-driven LLM inference with tools
      4. Post-hook chain (memory logging / RL recording / social graph)
      5. OVG verify+sign+TimeChain commit (via llm_pipeline.verify_post)
    """
    if not req.message or not req.message.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Message cannot be empty."},
        )

    agno_proxy = _get_agno_proxy(request)
    if agno_proxy is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Chat service unavailable (agno_proxy not installed)."},
        )

    # Resolve user_id from Privy claims, fall back to request body
    privy_user_id = claims.get("sub", "") if claims else ""
    user_id = privy_user_id or req.user_id or "anonymous"
    # RFP §7.D D.0/D.1 — is_maker is now "verified-Maker present OR asserted",
    # closing the plaintext gap: a real Privy-logged-in Maker (DID ≠ "maker") is
    # recognized via his nonce-signed session marker, not just claims.sub=="maker".
    channel = request.headers.get("X-Titan-Channel", "web")
    presence = _resolve_chat_presence(request, claims, user_id, channel)
    is_maker = presence.is_maker
    _emit_maker_presence(request, presence)   # D.2 — verified turn → maker_bond

    try:
        result = await agno_proxy.chat(
            message=req.message,
            user_id=user_id,
            session_id=req.session_id or "default",
            channel=channel,
            is_maker=is_maker,
            claims_sub=privy_user_id,
            ip_hash=_compute_ip_hash(request),
        )
    except Exception as e:
        logger.error("[Chat] agno_proxy.chat raised: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat pipeline error: {e}"},
        )

    return _result_to_response(result)


# ─────────────────────────────────────────────────────────────────────
# POST /chat/stream — SSE streaming chat
# ─────────────────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request,
                      claims: dict = Depends(verify_privy_token)):
    """Stream the Titan agent's response via Server-Sent Events.

    Each SSE event contains a content chunk from the LLM. Final marker is
    `data: [DONE]\\n\\n` (D-SPEC-72 — agno_worker emits done=True chunk
    on stream end; this handler maps that to the SSE [DONE] sentinel).

    Requires Privy authentication.
    """
    agno_proxy = _get_agno_proxy(request)
    if agno_proxy is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Chat stream service unavailable (agno_proxy not installed)."},
        )

    privy_user_id = claims.get("sub", "") if claims else ""
    user_id = privy_user_id or req.user_id or "anonymous"
    session_id = req.session_id or "default"
    channel = request.headers.get("X-Titan-Channel", "web")
    # RFP §7.D D.0/D.1/D.2 — verified-Maker presence (see /chat).
    presence = _resolve_chat_presence(request, claims, user_id, channel)
    is_maker = presence.is_maker
    _emit_maker_presence(request, presence)

    async def _sse_relay():
        """SSE relay over CHAT_STREAM_CHUNK payloads.

        D-SPEC-73 (SPEC v1.18.0): AgnoProxy.chat_stream now yields full
        payload dicts (not bare strings) per the bus contract — each dict
        contains {chunk: str?, done: bool, ovg_headers: dict?, error: str?}.
        We emit `data: <chunk>` for content frames and `event: ovg-headers`
        for the final frame's provenance attachment (D-SPEC-75 will wire
        the agno_worker side to populate ovg_headers; for Chunk A this is
        defensive — empty ovg_headers means non-streamed buffered reply).
        """
        import json as _json
        try:
            async for payload in agno_proxy.chat_stream(
                message=req.message,
                user_id=user_id,
                session_id=session_id,
                channel=channel,
                is_maker=is_maker,
                claims_sub=privy_user_id,
                ip_hash=_compute_ip_hash(request),
            ):
                if not isinstance(payload, dict):
                    continue
                err = payload.get("error")
                if err:
                    yield f"event: error\ndata: {_json.dumps({'error': err, 'detail': payload.get('detail', '')})}\n\n"
                    break
                # §7.B (B.4) — live progress phase (our-logic metadata, not content):
                # "thinking" / "reasoning" / "researching" / "running-tool" /
                # "using-skill" / "writing-reply". The client renders it + a live
                # timer + an op icon. Carries no chunk → forward + move on.
                phase = payload.get("phase")
                if phase:
                    yield (f"event: progress\ndata: "
                           f"{_json.dumps({'phase': phase, 'detail': payload.get('detail', '')})}\n\n")
                    continue
                chunk_text = payload.get("chunk", "")
                if chunk_text:
                    yield f"data: {_json.dumps({'chunk': chunk_text})}\n\n"
                ovg_headers = payload.get("ovg_headers") or {}
                if ovg_headers:
                    yield f"event: ovg-headers\ndata: {_json.dumps(ovg_headers)}\n\n"
                if payload.get("done"):
                    # §7.B (B.4) — closing meta frame: the non-verifiable turn's
                    # reasoning_id (→ the voluntary rating footer posts against it)
                    # + the execution mode (→ the client picks the per-lane rating
                    # scale: research3 for the research lane, else stars5). Only
                    # the done-frame carries these (agno_worker stream path).
                    meta = {}
                    _rid = payload.get("reasoning_id")
                    if _rid:
                        meta["reasoning_id"] = _rid
                    _mode = payload.get("mode")
                    if _mode:
                        meta["mode"] = _mode
                    if meta:
                        yield f"event: meta\ndata: {_json.dumps(meta)}\n\n"
                    break
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("[ChatStream] relay error: %s", e, exc_info=True)
            yield f"event: error\ndata: {{\"error\": \"relay\", \"detail\": {repr(str(e))}}}\n\n"

    return StreamingResponse(
        _sse_relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────
# RFP_affective_grounding_loop §7.D (D.0) — verified-Maker presence endpoints
#
# The Maker (logged in via Privy) proves control of maker_pubkey by Ed25519-
# signing a one-time server nonce with the same Solana wallet his MakerPanel
# already uses for proposals. On success a short-lived verified-Maker marker is
# minted for his Privy session → /chat reads it for is_maker=verified + the
# maker_bond tap. NON-BREAKING: additive; the MakerPanel proposal sign-flow is
# untouched. Sovereign: Titan verifies against the maker_pubkey HE holds.
# ─────────────────────────────────────────────────────────────────────

class VerifyPresenceRequest(BaseModel):
    nonce: str
    signature: str   # Base58 Ed25519 signature of the nonce by the Maker wallet


@router.post("/maker/presence-nonce")
async def maker_presence_nonce(request: Request,
                               claims: dict = Depends(verify_privy_token)):
    """Issue a single-use, short-lived signing challenge bound to this Privy
    session. The MakerPanel signs the returned `nonce` verbatim and POSTs it to
    /chat/maker/verify-presence. Any logged-in user may request a nonce; only a
    signature from the real maker_pubkey mints a marker (the proof, not the ask,
    is what gates the bond)."""
    store = getattr(request.app.state, "maker_presence_sessions", None)
    if store is None:
        return JSONResponse(status_code=503,
                            content={"error": "presence store unavailable"})
    session_key = session_key_from_claims(claims)
    if not session_key:
        return JSONResponse(status_code=401,
                            content={"error": "unauthenticated"})
    nonce = store.issue_nonce(session_key)
    return JSONResponse(status_code=200, content={"nonce": nonce})


@router.post("/maker/verify-presence")
async def maker_verify_presence(req: VerifyPresenceRequest, request: Request,
                                claims: dict = Depends(verify_privy_token)):
    """Verify the Maker's wallet signature over the issued nonce and, on success,
    mint a verified-Maker session marker. Returns {verified: bool}. Honest-fail
    (verified=False) on any bad nonce/signature — no marker, no bond."""
    store = getattr(request.app.state, "maker_presence_sessions", None)
    if store is None:
        return JSONResponse(status_code=503,
                            content={"error": "presence store unavailable"})
    session_key = session_key_from_claims(claims)
    if not session_key:
        return JSONResponse(status_code=401,
                            content={"error": "unauthenticated"})
    maker_pubkey = resolve_maker_pubkey(request)
    if not maker_pubkey:
        return JSONResponse(status_code=503,
                            content={"error": "maker_pubkey unavailable"})
    verified = store.verify_and_mint(
        session_key, req.nonce, req.signature, maker_pubkey)
    return JSONResponse(status_code=200, content={"verified": bool(verified)})
