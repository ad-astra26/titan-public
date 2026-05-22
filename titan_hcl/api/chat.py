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

from titan_hcl.api.auth import verify_privy_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


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
    is_maker = (user_id == "maker")

    try:
        result = await agno_proxy.chat(
            message=req.message,
            user_id=user_id,
            session_id=req.session_id or "default",
            channel=request.headers.get("X-Titan-Channel", "web"),
            is_maker=is_maker,
            claims_sub=privy_user_id,
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
    is_maker = (user_id == "maker")
    session_id = req.session_id or "default"
    channel = request.headers.get("X-Titan-Channel", "web")

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
            ):
                if not isinstance(payload, dict):
                    continue
                err = payload.get("error")
                if err:
                    yield f"event: error\ndata: {_json.dumps({'error': err, 'detail': payload.get('detail', '')})}\n\n"
                    break
                chunk_text = payload.get("chunk", "")
                if chunk_text:
                    yield f"data: {_json.dumps({'chunk': chunk_text})}\n\n"
                ovg_headers = payload.get("ovg_headers") or {}
                if ovg_headers:
                    yield f"event: ovg-headers\ndata: {_json.dumps(ovg_headers)}\n\n"
                if payload.get("done"):
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
