"""
api/chat_bridge_client.py — subprocess-side client for the CHAT_REQUEST
bus bridge.

Microkernel v2: when `microkernel.api_process_separation_enabled=true`,
the /chat FastAPI endpoint lives in the api subprocess but the Agno
agent + gatekeeper + memory + OVG state live in the parent process and
cannot be safely re-constructed (would double LLM clients + drift state
+ lose tool wiring — see BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS).

This client forwards /chat requests from subprocess to parent over the
standard worker IPC pipe (send_queue / recv_queue) using rid-routed
QUERY/RESPONSE semantics — same protocol as bus.request, just
implemented over the worker pipe instead of in-process queues.

Architecture:
  api_subprocess /chat handler
    ↓ chat_bridge_client.request_async("chat_subproc", "chat_handler",
                                       payload={action:"chat", body, claims, headers},
                                       timeout=60.0)
    ↓ register pending Future keyed by rid
    ↓ send_queue.put({type=QUERY, src="chat_subproc", dst="chat_handler",
                      rid, payload, ts})
    ↓ kernel forwards via bus → parent's chat_handler subscriber
    ↓ parent's _chat_handler_loop calls plugin.run_chat() → publishes RESPONSE
    ↓ kernel routes RESPONSE → api_subprocess recv_queue
    ↓ api_subprocess _bus_listener_loop dispatches RESPONSE to
      chat_bridge_client.handle_response(msg)
    ↓ pending Future resolves with the reply envelope
    ↓ /chat handler unwraps reply.payload (run_chat result dict) and
      returns JSONResponse

Concurrency: a single ChatBridgeClient handles many concurrent /chat
requests. Each request gets its own rid + Future; replies are
dispatched by rid match. No locking on the pending dict because all
mutations happen on the asyncio event loop thread (request_async +
the `_loop.call_soon_threadsafe(future.set_result)` path).

Forward-compat: same QUERY/RESPONSE wire shape that bus.request uses,
so a future migration from worker-IPC pipes to in-process bus
subscription (e.g., via bus_ipc_socket broker) requires only changing
the transport layer — the rid-matching contract is preserved.

See: titan_plugin/core/plugin.py:_chat_handler_loop +
     titan_plugin/api/chat.py /chat endpoint Mode 2 +
     titan-docs/BUGS.md BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from titan_plugin import bus

logger = logging.getLogger(__name__)


class ChatBridgeClient:
    """Subprocess-side client for the parent's chat_handler bus consumer.

    Exposes `request_async(src, dst, payload, timeout)` matching
    DivineBus.request_async semantics (returns the full reply envelope
    dict on success, None on timeout). The /chat endpoint reads
    `app.state.chat_bridge_bus` and calls request_async when the
    in-process agent is None.

    Send path: send_queue.put_nowait(QUERY message). Kernel-side
    Guardian forwarding routes the message to the parent's
    DivineBus, which delivers to the chat_handler subscriber.

    Receive path: handle_response(msg) — called by api_subprocess's
    _bus_listener_loop on every recv_queue message. Matches by
    msg.type == bus.RESPONSE and msg.rid in self._pending. Returns
    True if the message was claimed (so the loop skips further
    handlers).
    """

    def __init__(self, send_queue: Any) -> None:
        """
        Args:
            send_queue: api_subprocess send_queue. Required (None
                disables bridging — equivalent to no bridge wired).
        """
        self._send_queue = send_queue
        # rid → asyncio.Future. Mutations happen on the event loop
        # thread (request_async creates entries; _on_response_threadsafe
        # via call_soon_threadsafe resolves them). No explicit lock.
        self._pending: dict[str, asyncio.Future] = {}
        # Captured on first request_async call. Used by
        # call_soon_threadsafe to schedule the future-set on the
        # endpoint-event-loop thread (the bus listener runs in its
        # own thread).
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Receive side (called from bus listener thread) ────────────

    def handle_response(self, msg: dict) -> bool:
        """Dispatch RESPONSE messages to pending Futures by rid match.

        Called from `_bus_listener_loop` in api_subprocess.py on every
        recv_queue message. Returns True if this client claimed the
        message; False otherwise (so the loop tries other handlers).

        Thread-safe: `call_soon_threadsafe` schedules the future-set
        on the asyncio event loop (the listener runs on a separate
        thread; futures must be mutated on their owning loop).
        """
        if not isinstance(msg, dict):
            return False
        if msg.get("type") != bus.RESPONSE:
            return False
        rid = msg.get("rid")
        if not rid or rid not in self._pending:
            return False
        future = self._pending.get(rid)
        if future is None or future.done():
            # Already resolved — request_async timed out or was cancelled.
            # Pop the entry so memory doesn't grow.
            self._pending.pop(rid, None)
            return True  # We claimed it (silently dropped)
        loop = self._loop
        if loop is None:
            # Should not happen — request_async sets _loop before sending.
            logger.warning("[ChatBridgeClient] no event loop captured yet "
                           "but RESPONSE arrived; dropping rid=%s", rid[:8])
            self._pending.pop(rid, None)
            return True
        try:
            loop.call_soon_threadsafe(future.set_result, msg)
        except RuntimeError:
            # Loop closed — server shutting down.
            self._pending.pop(rid, None)
        return True

    # ── Send side (called from endpoint event loop) ────────────────

    async def request_async(
        self,
        src: str,
        dst: str,
        payload: dict,
        timeout: float = 60.0,
    ) -> Optional[dict]:
        """Forward a chat request to the parent's chat_handler over the
        worker IPC pipe. Returns the full reply envelope dict on success,
        or None on timeout / send failure.

        Mirrors DivineBus.request_async semantics so chat.py's Mode 2
        path stays transport-agnostic (parent in-process bus or
        subprocess pipe — same call, same return shape).
        """
        if self._send_queue is None:
            logger.warning("[ChatBridgeClient] no send_queue — bridge disabled")
            return None
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        rid = str(uuid.uuid4())
        future: asyncio.Future = self._loop.create_future()
        self._pending[rid] = future
        try:
            self._send_queue.put_nowait({
                "type": bus.QUERY,
                "src":  src,
                "dst":  dst,
                "rid":  rid,
                "payload": payload,
                "ts":   time.time(),
            })
        except Exception as e:
            self._pending.pop(rid, None)
            logger.warning("[ChatBridgeClient] send_queue.put_nowait failed: %s", e)
            return None
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[ChatBridgeClient] request_async timeout (rid=%s, %.1fs, dst=%s)",
                rid[:8], timeout, dst)
            return None
        finally:
            self._pending.pop(rid, None)

    # ── Diagnostics ────────────────────────────────────────────────

    def pending_count(self) -> int:
        """Return number of in-flight requests. Useful for /v4/admin
        endpoints + debugging stuck requests."""
        return len(self._pending)
