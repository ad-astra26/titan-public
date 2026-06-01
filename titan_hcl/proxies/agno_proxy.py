"""
agno_proxy — Public surface for /chat + /v4/pitch-chat → agno_worker.

D-SPEC-72 (initial carve, SPEC v1.17.0) + D-SPEC-73 (api_subprocess
backend, SPEC v1.18.0).

Two backends behind one public surface:

  Context              Backend                     Why
  ───────────────────  ──────────────────────────  ─────────────────────
  parent process       DivineBus.request_async     real DivineBus exists;
                       (in-process bus.subscribe   subscribe is local; rid
                       to set up reply queue)      routed via the queue
  api_subprocess       AgnoBridgeClient            kernel_rpc-proxied bus
                       (send_queue + rid Future)   has NO subscribe; the
                                                   subprocess uses its own
                                                   send_queue/recv_queue
                                                   already wired to broker

The backend is selected at construction time. No runtime branching in the
hot path.

D-SPEC-73 RETIREMENT (no shim, per feedback_no_shim_old_path_must_be_deleted):
  - Removed `_ensure_reply_queue()` and `_reply_queue` field — the broken
    `bus.subscribe` call that caused fleet-wide /chat 500 errors when
    invoked in the api_subprocess context (kernel_rpc-proxied bus). The
    parent-context backend now uses bus.request_async directly (it owns
    its own reply queue internally on the in-process DivineBus path).
  - Removed legacy `__init__(bus_client=...)` signature alias — callers
    must pass either `bus_client=` (parent) or `bridge=` (subprocess)
    explicitly. Construction error if both / neither.

Timeout: 90s ceiling matches the existing `_AGENT_ARUN_TIMEOUT_S` Layer-1
closure of BUG-CHAT-AGENT-ARUN-HANG-T3-PHASE-C. Allowlisted in
`phase_c_rpc_exemptions.yaml` as `agno_proxy → agno_worker` work-RPC.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional, Union

from ..bus import (
    CHAT_REQUEST,
    CHAT_STREAM_CHUNK,
    CHAT_STREAM_REQUEST,
    DivineBus,
    make_msg,
)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_S = 90.0
STREAM_TICK_S = 0.05


class AgnoProxy:
    """Public surface wrapping the agno_worker chat contract.

    Backend dispatch is selected at construction:

        # Parent process: real DivineBus available
        proxy = AgnoProxy(bus_client=plugin.bus)

        # api_subprocess: use the bridge that owns the send_queue rid routing
        proxy = AgnoProxy(bridge=app.state.agno_bridge)

    Public methods unchanged from D-SPEC-72: `chat()`, `chat_stream()`,
    `get_stats()`. Backends are private implementation detail.
    """

    def __init__(
        self,
        bus_client: Optional[DivineBus] = None,
        bridge: Optional[Any] = None,  # AgnoBridgeClient (forward decl)
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ):
        if (bus_client is None) == (bridge is None):
            raise ValueError(
                "AgnoProxy requires EXACTLY ONE of bus_client= (parent "
                "process DivineBus) or bridge= (api_subprocess "
                "AgnoBridgeClient). Got both or neither."
            )
        self._bus = bus_client
        self._bridge = bridge
        self._timeout = float(request_timeout_s)
        self._request_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

    # ── Public methods ────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        session_id: str = "default",
        channel: str = "web",
        is_maker: bool = False,
        claims_sub: str = "",
    ) -> dict[str, Any]:
        """Synchronous chat — round-trips CHAT_REQUEST/CHAT_RESPONSE.

        Returns:
            {status_code: int, body: dict, extra_headers: dict | None}
        """
        self._request_count += 1
        if self._bridge is not None:
            try:
                return await self._bridge.chat(
                    message,
                    user_id=user_id, session_id=session_id, channel=channel,
                    is_maker=is_maker, claims_sub=claims_sub,
                )
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.exception("[AgnoProxy] bridge.chat raised: %s", e)
                return _envelope_error(500, "agno_proxy_bridge_error",
                                       str(e))

        # Parent-process path — direct bus.request_async via DivineBus
        return await self._chat_via_bus(
            message,
            user_id=user_id, session_id=session_id, channel=channel,
            is_maker=is_maker, claims_sub=claims_sub,
        )

    async def chat_stream(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        session_id: str = "default",
        channel: str = "web",
        is_maker: bool = False,
        claims_sub: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """SSE relay — yields chunk dicts until agno_worker emits done=true.

        Each yielded dict matches the CHAT_STREAM_CHUNK payload shape:
            {chunk: str, done: bool, ovg_headers: dict?, error: str?}

        Caller (chat.py SSE writer) is responsible for SSE-encoding and
        emitting an `event: ovg-headers` SSE event when the final frame
        carries verified provenance.
        """
        self._request_count += 1
        if self._bridge is not None:
            try:
                async for chunk in self._bridge.chat_stream(
                    message,
                    user_id=user_id, session_id=session_id, channel=channel,
                    is_maker=is_maker, claims_sub=claims_sub,
                ):
                    yield chunk
                return
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.exception("[AgnoProxy] bridge.chat_stream raised: %s",
                                 e)
                yield {"error": "agno_proxy_stream_error", "detail": str(e),
                       "done": True}
                return

        # Parent-process path
        async for chunk in self._stream_via_bus(
            message,
            user_id=user_id, session_id=session_id, channel=channel,
            is_maker=is_maker, claims_sub=claims_sub,
        ):
            yield chunk

    def get_stats(self) -> dict[str, Any]:
        """Proxy-side stats for Observatory + health monitor."""
        if self._bridge is not None:
            inner = self._bridge.get_stats()
            inner["proxy"] = "agno_proxy"
            inner["backend"] = "bridge"
            return inner
        return {
            "proxy": "agno_proxy",
            "backend": "bus",
            "requests": self._request_count,
            "errors": self._error_count,
            "last_error": self._last_error,
            "timeout_s": self._timeout,
        }

    # ── Parent-process backend (DivineBus) ────────────────────────────
    #
    # In the parent process the in-process DivineBus supports subscribe;
    # request_async sets up its own reply queue internally (DivineBus
    # primitive — does NOT cross kernel_rpc, so the api_subprocess
    # MethodNotExposed issue does not apply here).

    async def _chat_via_bus(
        self, message: str, *,
        user_id: str, session_id: str, channel: str,
        is_maker: bool, claims_sub: str,
    ) -> dict[str, Any]:
        """Publish CHAT_REQUEST + await CHAT_RESPONSE by rid on a reply
        queue. Uses in-process DivineBus subscribe (works in parent;
        the kernel_rpc-blocked code path was deleted per D-SPEC-73).

        Note: `bus.request_async` is the QUERY/RESPONSE work-RPC primitive
        and routes msg.type=QUERY internally — it does NOT accept a custom
        msg_type. For CHAT_REQUEST we use publish + manual rid + dedicated
        reply queue, equivalent to request_async but with explicit type.
        """
        request_id = uuid.uuid4().hex
        reply_queue = self._bus.subscribe("agno_proxy", reply_only=True)

        request_payload = {
            "request_id": request_id,
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
            "is_maker": is_maker,
            "claims_sub": claims_sub,
            "prefer_streaming": False,
            "ts": time.time(),
        }
        try:
            self._bus.publish(make_msg(
                CHAT_REQUEST,
                "agno_proxy", "agno_worker",
                request_payload,
                rid=request_id,
            ))
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.exception(
                "[AgnoProxy] CHAT_REQUEST publish raised: %s", e)
            return _envelope_error(500, "agno_proxy_publish_error", str(e))

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, reply_queue.get, True, 0.1,
                    ),
                    timeout=0.5,
                )
            except (asyncio.TimeoutError, Exception):
                continue
            if not msg or msg.get("type") != "CHAT_RESPONSE":
                continue
            if msg.get("rid") != request_id and \
               (msg.get("payload") or {}).get("request_id") != request_id:
                continue
            return _translate_reply(msg, session_id)

        self._error_count += 1
        self._last_error = "agno_timeout"
        logger.warning(
            "[AgnoProxy] CHAT_RESPONSE timeout (%.0fs)", self._timeout)
        return _envelope_error(
            504, "agno_timeout",
            f"agno_worker did not respond within {self._timeout}s")

    async def _stream_via_bus(
        self, message: str, *,
        user_id: str, session_id: str, channel: str,
        is_maker: bool, claims_sub: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parent-process streaming — uses DivineBus subscribe (works in
        the parent because subscribe is in-process, NOT kernel_rpc).
        """
        request_id = uuid.uuid4().hex
        reply_queue = self._bus.subscribe("agno_proxy", reply_only=True)

        request_payload = {
            "request_id": request_id,
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
            "is_maker": is_maker,
            "claims_sub": claims_sub,
            "stream": True,
            "prefer_streaming": True,
            "ts": time.time(),
        }

        try:
            self._bus.publish(make_msg(
                CHAT_STREAM_REQUEST,
                "agno_proxy", "agno_worker",
                request_payload,
            ))
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.exception(
                "[AgnoProxy] CHAT_STREAM_REQUEST publish raised: %s", e)
            yield {"error": "publish_failed", "detail": str(e),
                   "done": True}
            return

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, reply_queue.get, True, STREAM_TICK_S,
                    ),
                    timeout=STREAM_TICK_S * 4,
                )
            except (asyncio.TimeoutError, Exception):
                continue

            if not msg:
                continue
            if msg.get("type") != CHAT_STREAM_CHUNK:
                continue
            payload = msg.get("payload") or {}
            if payload.get("request_id") != request_id:
                continue

            yield payload
            if payload.get("done"):
                return

        self._error_count += 1
        self._last_error = "agno_stream_timeout"
        logger.warning(
            "[AgnoProxy] CHAT_STREAM_REQUEST timeout (%.0fs)", self._timeout)
        yield {"error": "agno_stream_timeout", "done": True}


# ── helpers ────────────────────────────────────────────────────────────

def _envelope_error(status: int, error: str, detail: str) -> dict[str, Any]:
    return {
        "status_code": status,
        "body": {"error": error, "detail": detail},
        "extra_headers": None,
    }


def _translate_reply(reply: Optional[dict],
                     session_id: str) -> dict[str, Any]:
    body = (reply or {}).get("payload") or {}
    error = body.get("error")
    if error:
        return {
            "status_code": 500,
            "body": {"error": "agno_worker_error", "detail": error},
            "extra_headers": None,
        }
    chat_response_body = {
        "response": body.get("response", ""),
        "session_id": body.get("session_id", session_id),
        "mode": body.get("mode", ""),
        "mood": body.get("mood", ""),
        "state_narration": body.get("state_narration"),
        "state_snapshot": body.get("state_snapshot"),
        "ovg": body.get("ovg_data"),
        # Tool-backstop activity (2026-06-01) — non-null when a deterministic
        # tool ran this turn; lets the frontend/comma show "verified via sandbox".
        "tool_activity": body.get("tool_activity"),
    }
    extra_headers = _build_ovg_headers(body.get("ovg_data") or {})
    return {
        "status_code": 200,
        "body": chat_response_body,
        "extra_headers": extra_headers,
    }


def _build_ovg_headers(ovg_data: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not ovg_data:
        return headers
    headers["X-Titan-Verified"] = (
        "true" if ovg_data.get("verified") else "false"
    )
    block_height = ovg_data.get("block_height")
    if block_height is not None:
        headers["X-Titan-Block-Height"] = str(int(block_height))
    merkle_root = ovg_data.get("merkle_root") or ""
    if merkle_root:
        headers["X-Titan-Merkle-Root"] = merkle_root
    signature = ovg_data.get("signature") or ""
    if signature:
        headers["X-Titan-Signature"] = signature
    return headers
