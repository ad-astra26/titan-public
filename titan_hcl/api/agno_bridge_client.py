"""
api/agno_bridge_client.py — subprocess-side backend for AgnoProxy.

D-SPEC-73 (SPEC v1.18.0). Closes BUG-CHAT-500-AGNO-PROXY-SUBSCRIBE-IN-
API-SUBPROCESS surfaced on T3 by runtime test B1 (2026-05-17): AgnoProxy
called `bus.subscribe` to set up its reply queue, which fails fleet-wide
in the api_subprocess context because `bus.subscribe` is not in
KERNEL_RPC_EXPOSED_METHODS (security: subscribe is a broad capability).

Architecture: mirrors `ChatBridgeClient` (the chat_handler bridge that
ran before D-SPEC-72 retired chat_handler), but routes the new
agno-worker topics:

      send side  (api_subprocess  → agno_worker)
        CHAT_REQUEST          rid + payload
        CHAT_STREAM_REQUEST   rid + payload

      recv side  (agno_worker → api_subprocess)
        CHAT_RESPONSE         rid-matched → asyncio.Future resolution
        CHAT_STREAM_CHUNK     rid-matched → asyncio.Queue draining

Why this is the right primitive in the api_subprocess context:
- The api_subprocess already has `send_queue` (outbound, kernel-forwarded
  → Rust broker) and `_bus_listener_loop` (inbound). These ARE the Phase C
  contract for any worker-to-broker messaging from this process.
- Direct `bus.subscribe` would require either (a) a separate
  BusSocketClient connection from api_subprocess to the broker
  (significant scope expansion, deferred per rFP §6) or (b) exposing
  bus.subscribe in KERNEL_RPC_EXPOSED_METHODS (security hole).
- `ChatBridgeClient` proved this exact pattern (rid + send_queue + Future
  registry, dispatched via _bus_listener_loop) works at production scale
  for /chat throughput. Reusing the shape is correctness-by-precedent.

Concurrency: a single AgnoBridgeClient handles many concurrent /chat
requests AND concurrent /chat-stream requests. Each gets its own rid;
sync chat uses Future, streaming uses bounded Queue. No locking on the
pending dicts because all mutations happen on either:
  - the event loop thread (request_async creates entries; futures resolve
    via call_soon_threadsafe), or
  - the bus listener thread, which only POPS via call_soon_threadsafe.

Forward-compat: if the api_subprocess ever migrates to a direct
BusSocketClient (rFP §6 deferral), the AgnoProxy public surface is
preserved — only the backend swaps. Same rid-matching contract.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from titan_hcl import bus

logger = logging.getLogger(__name__)

# Per-rid CHAT_STREAM_CHUNK queue capacity. Bounded so a slow SSE client
# can't unbounded-buffer agno_worker output. On overflow, the queue's
# overflow sentinel terminates the stream with a clean event.
_STREAM_QUEUE_MAX = 64

# Stream tick — async generator wakes this often to check the deadline
# in addition to the per-chunk event-loop signal.
_STREAM_TICK_S = 0.05


class _StreamOverflow:
    """Sentinel placed on a per-rid queue when bounded capacity is hit."""
    __slots__ = ()


_STREAM_OVERFLOW = _StreamOverflow()


class AgnoBridgeClient:
    """Subprocess-side backend for `AgnoProxy` in api_subprocess context.

    Public methods:
      - chat(...) → dict — sync round-trip via CHAT_REQUEST/CHAT_RESPONSE
      - chat_stream(...) → AsyncIterator[dict] — drain CHAT_STREAM_CHUNK
      - handle_response(msg) → bool — called by _bus_listener_loop; returns
        True if this client owns the message by rid match.
      - pending_count() → int — diagnostics

    Stats are exposed via get_stats() for Observatory + health monitor.
    """

    def __init__(
        self,
        send_queue: Any,
        request_timeout_s: float = 90.0,
        name: str = "api",
    ) -> None:
        """
        Args:
            send_queue: api_subprocess send_queue (mp.Queue → kernel-
                forwarded → Rust broker). Required.
            request_timeout_s: matches AgnoProxy DEFAULT_REQUEST_TIMEOUT_S
                + the phase_c_rpc_exemptions.yaml allowlist for
                agno_proxy → agno_worker (90s).
            name: bus src name. MUST match the api_subprocess's registered
                module name ("api" per core/plugin.py:1878) so agno_worker's
                CHAT_RESPONSE reply (with dst=msg.src) is routed back here
                via name-based dst routing (per plugin.py:1904-1906 — RESPONSE
                messages bypass broadcast filter automatically).
        """
        self._send_queue = send_queue
        self._timeout = float(request_timeout_s)
        self._name = name

        # rid → Future. Sync chat path. Mutations always on event-loop
        # thread (request_async creates; call_soon_threadsafe resolves).
        self._pending_responses: dict[str, asyncio.Future] = {}
        # rid → bounded Queue. Streaming path. handle_response enqueues
        # via call_soon_threadsafe; chat_stream drains via Queue.get.
        self._pending_streams: dict[str, asyncio.Queue] = {}

        # Captured on first async call. Used by call_soon_threadsafe to
        # schedule future-set / queue-put on the owning loop.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self._request_count = 0
        self._stream_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

    # ── Receive side (called from bus listener thread) ────────────────

    def handle_response(self, msg: dict) -> bool:
        """Dispatch CHAT_RESPONSE / CHAT_STREAM_CHUNK to pending registry.

        Called from `_bus_listener_loop` on every recv_queue message.
        Returns True if this client claimed the message; False otherwise
        (so the loop falls through to other handlers).

        Thread-safe: `call_soon_threadsafe` schedules the future-set /
        queue-put on the owning asyncio event loop (the listener runs on
        a separate thread; futures + queues must be mutated on their
        owning loop).
        """
        if not isinstance(msg, dict):
            return False
        mt = msg.get("type")

        # Sync chat — match on CHAT_RESPONSE + rid
        if mt == bus.CHAT_RESPONSE:
            rid = msg.get("rid")
            if not rid or rid not in self._pending_responses:
                return False
            future = self._pending_responses.get(rid)
            if future is None or future.done():
                self._pending_responses.pop(rid, None)
                return True
            loop = self._loop
            if loop is None:
                logger.warning(
                    "[AgnoBridge] no event loop captured but CHAT_RESPONSE "
                    "arrived rid=%s — dropping", rid[:8])
                self._pending_responses.pop(rid, None)
                return True
            try:
                loop.call_soon_threadsafe(future.set_result, msg)
            except RuntimeError:
                # Loop closed — server shutting down.
                self._pending_responses.pop(rid, None)
            return True

        # Streaming — match on CHAT_STREAM_CHUNK + rid
        if mt == bus.CHAT_STREAM_CHUNK:
            payload = msg.get("payload") or {}
            rid = payload.get("request_id")
            if not rid or rid not in self._pending_streams:
                return False
            queue = self._pending_streams.get(rid)
            if queue is None:
                return True  # registry says we own it; dropped
            loop = self._loop
            if loop is None:
                return True
            # Bounded enqueue. If queue is full, push overflow sentinel
            # (idempotent — only first overflow matters) and stop trying.
            def _enqueue():
                if queue.full():
                    if not queue._unfinished_tasks:  # avoid double-overflow
                        return
                    try:
                        queue.put_nowait(_STREAM_OVERFLOW)
                    except asyncio.QueueFull:
                        pass
                    return
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        queue.put_nowait(_STREAM_OVERFLOW)
                    except asyncio.QueueFull:
                        pass
            try:
                loop.call_soon_threadsafe(_enqueue)
            except RuntimeError:
                pass
            return True

        return False

    # ── Send side: sync chat ──────────────────────────────────────────

    async def chat(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        session_id: str = "default",
        channel: str = "web",
        is_maker: bool = False,
        claims_sub: str = "",
        ip_hash: str = "",
    ) -> dict[str, Any]:
        """Sync round-trip: CHAT_REQUEST → CHAT_RESPONSE (rid-matched).

        Returns AgnoProxy-shaped dict:
            {status_code, body, extra_headers}

        Timeout returns 504 envelope so the api caller can classify and
        surface a clean message.
        """
        if self._send_queue is None:
            logger.warning("[AgnoBridge] no send_queue — bridge disabled")
            return _envelope_error(503, "agno_bridge_disabled",
                                   "AgnoBridgeClient has no send_queue")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._request_count += 1

        rid = uuid.uuid4().hex
        future: asyncio.Future = self._loop.create_future()
        self._pending_responses[rid] = future

        payload = {
            "request_id": rid,
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
            "is_maker": is_maker,
            "claims_sub": claims_sub,
            "ip_hash": ip_hash,
            "prefer_streaming": False,
            "ts": time.time(),
        }
        if not self._send_request(rid, bus.CHAT_REQUEST, payload):
            self._pending_responses.pop(rid, None)
            self._error_count += 1
            return _envelope_error(503, "send_queue_full",
                                   "send_queue rejected CHAT_REQUEST")

        try:
            reply = await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._error_count += 1
            self._last_error = "agno_timeout"
            logger.warning(
                "[AgnoBridge] CHAT_REQUEST timeout (rid=%s, %.0fs)",
                rid[:8], self._timeout)
            return _envelope_error(
                504, "agno_timeout",
                f"agno_worker did not respond within {self._timeout}s")
        finally:
            self._pending_responses.pop(rid, None)

        return _translate_agno_reply(reply, session_id)

    # ── Send side: streaming ──────────────────────────────────────────

    async def chat_stream(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        session_id: str = "default",
        channel: str = "web",
        is_maker: bool = False,
        claims_sub: str = "",
        ip_hash: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """SSE relay: CHAT_STREAM_REQUEST → CHAT_STREAM_CHUNK frames.

        Yields chunk dicts as they arrive (each contains keys like
        `chunk: str`, `done: bool`, `ovg_headers: dict?`). Caller is
        responsible for SSE-encoding and emitting `event: ovg-headers`
        when the final frame carries verified provenance.

        Stops on:
          - chunk with done=True
          - timeout (rFP §2.3 — 90s default)
          - stream-overflow sentinel (bounded queue exceeded)
        """
        if self._send_queue is None:
            yield {"error": "agno_bridge_disabled", "done": True}
            return
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._stream_count += 1

        rid = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAX)
        self._pending_streams[rid] = queue

        payload = {
            "request_id": rid,
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
            "is_maker": is_maker,
            "claims_sub": claims_sub,
            "ip_hash": ip_hash,
            "stream": True,
            "prefer_streaming": True,
            "ts": time.time(),
        }
        if not self._send_request(rid, bus.CHAT_STREAM_REQUEST, payload):
            self._pending_streams.pop(rid, None)
            self._error_count += 1
            yield {"error": "send_queue_full", "done": True}
            return

        deadline = time.time() + self._timeout
        try:
            while time.time() < deadline:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=min(_STREAM_TICK_S * 4,
                                    max(0.0, deadline - time.time())),
                    )
                except asyncio.TimeoutError:
                    continue

                if item is _STREAM_OVERFLOW:
                    self._error_count += 1
                    self._last_error = "stream_overflow"
                    yield {"error": "stream_overflow", "done": True}
                    return

                yield item
                if item.get("done"):
                    return

            # Loop exit on deadline
            self._error_count += 1
            self._last_error = "agno_stream_timeout"
            logger.warning(
                "[AgnoBridge] CHAT_STREAM_REQUEST timeout (rid=%s, %.0fs)",
                rid[:8], self._timeout)
            yield {"error": "agno_stream_timeout", "done": True}
        finally:
            self._pending_streams.pop(rid, None)

    # ── Internals ─────────────────────────────────────────────────────

    def _send_request(self, rid: str, msg_type: str, payload: dict) -> bool:
        """Enqueue the bus message onto the api_subprocess send_queue.

        Returns True on success, False if send_queue rejected (full / closed).
        """
        try:
            self._send_queue.put_nowait({
                "type": msg_type,
                "src": self._name,
                "dst": "agno_worker",
                "rid": rid,
                "payload": payload,
                "ts": time.time(),
            })
            return True
        except Exception as e:
            logger.warning(
                "[AgnoBridge] send_queue.put_nowait failed (%s): %s",
                msg_type, e)
            self._last_error = str(e)
            return False

    # ── Diagnostics ───────────────────────────────────────────────────

    def pending_count(self) -> int:
        """In-flight requests (sync + stream)."""
        return len(self._pending_responses) + len(self._pending_streams)

    def get_stats(self) -> dict[str, Any]:
        """For Observatory + health monitor."""
        return {
            "bridge": "agno",
            "requests": self._request_count,
            "streams": self._stream_count,
            "errors": self._error_count,
            "last_error": self._last_error,
            "pending_responses": len(self._pending_responses),
            "pending_streams": len(self._pending_streams),
            "timeout_s": self._timeout,
        }


# ── helpers ────────────────────────────────────────────────────────────

def _envelope_error(status: int, error: str, detail: str) -> dict[str, Any]:
    """Build an AgnoProxy-shaped error envelope."""
    return {
        "status_code": status,
        "body": {"error": error, "detail": detail},
        "extra_headers": None,
    }


def _translate_agno_reply(reply: Optional[dict],
                          session_id: str) -> dict[str, Any]:
    """Translate the agno_worker CHAT_RESPONSE envelope → AgnoProxy shape.

    Matches AgnoProxy.chat() return shape exactly so chat.py + pitch_chat.py
    don't change.
    """
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
        # §7.B (B.4) — reasoning_id of a NON-verifiable turn (direct/research/IDK;
        # None otherwise). The in-process bridge is the live chat path, so this is
        # what /v6/pitch/chat reads to surface the id for the wallet-less rating
        # footer (→ /v6/pitch/rate). Mirrors agno_proxy._translate_reply (bus path).
        "reasoning_id": body.get("reasoning_id"),
    }
    extra_headers = _build_ovg_headers(body.get("ovg_data") or {})
    return {
        "status_code": 200,
        "body": chat_response_body,
        "extra_headers": extra_headers,
    }


def _build_ovg_headers(ovg_data: dict[str, Any]) -> dict[str, str]:
    """X-Titan-* response headers from ovg_data (matches AgnoProxy)."""
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
