"""
titan_plugin/api/command_sender.py — fire-and-forget command dispatcher
for the api_subprocess.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

Per Q5 (PLAN v2): write-side commands from endpoint code (e.g.,
`plugin.guardian.start("module")`, `plugin.reload_api()`) become
bus.publish calls. The kernel subscribes on the other end, executes
the command, and publishes a result event that the BusSubscriber
caches. Endpoint code that needs the result polls the cache or
subscribes to the result event explicitly.

For commands where waiting for a result IS required by the endpoint
contract (e.g., HTTP POST that returns the operation outcome), the
caller awaits a `RequestFuture` that resolves when the matching
result event arrives. This is rare (~5 callsites total).

Bus message contract:
  - command request: type=`<COMMAND>_REQUEST`, src=`api`, dst=`<owner>`,
    payload={`request_id`: str (uuid), ...args}
  - command result: type=`<COMMAND>_RESPONSE`, src=`<owner>`, dst=`api`,
    payload={`request_id`: str, `ok`: bool, `result`: any, `err`: str?}

Commands implemented:
  - guardian_start(module)         → GUARDIAN_START_REQUEST
  - guardian_stop(module)          → GUARDIAN_STOP_REQUEST
  - guardian_restart(module)       → GUARDIAN_RESTART_REQUEST
  - reload_api()                   → RELOAD_API_REQUEST
  - force_dream()                  → FORCE_DREAM_REQUEST
  - inject_memory(record)          → MEMORY_INJECT_REQUEST
  - solana_balance_refresh()       → SOLANA_BALANCE_REFRESH_REQUEST
  - publish(type, dst, payload)    → raw escape hatch (Maker module et al.)

Most are fire-and-forget; result lands in CachedState if the kernel
publishes the matching *_UPDATED event.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any

logger = logging.getLogger(__name__)


class CommandSender:
    """Fire-and-forget command dispatcher. Wraps the api_subprocess
    send_queue with typed methods.

    Endpoints call `state.commands.<verb>(args)` instead of
    `plugin.X.method(args)`. For write-side semantics, this matches
    the kernel-as-authority pattern: api never mutates kernel state
    directly, only sends requests.
    """

    def __init__(self, send_queue: Any | None) -> None:
        """
        Args:
          send_queue: api_subprocess send_queue. None disables (tests
                      may construct without a queue).
        """
        self._send_queue = send_queue

    # -- generic publish ----------------------------------------------

    def publish(
        self,
        msg_type: str,
        dst: str,
        payload: dict | None = None,
        src: str = "api",
    ) -> str:
        """Raw bus.publish — escape hatch for callers that need direct
        bus access (Maker module, custom command types). Returns the
        request_id assigned to the message."""
        if self._send_queue is None:
            logger.warning(
                "[CommandSender] publish skipped (no send_queue): %s → %s",
                msg_type, dst)
            return ""
        request_id = secrets.token_hex(8)
        full_payload = dict(payload or {})
        full_payload.setdefault("request_id", request_id)
        full_payload.setdefault("requested_at_ns", time.time_ns())
        try:
            from titan_plugin.bus import make_msg
            msg = make_msg(msg_type, src, dst, full_payload)
            self._send_queue.put(msg)
        except Exception as e:
            logger.warning(
                "[CommandSender] publish failed (%s → %s): %s",
                msg_type, dst, e)
            return ""
        return request_id

    # -- typed commands ------------------------------------------------

    def guardian_start(self, module: str) -> str:
        return self.publish(
            "GUARDIAN_START_REQUEST", "guardian",
            {"module": module})

    def guardian_stop(self, module: str) -> str:
        return self.publish(
            "GUARDIAN_STOP_REQUEST", "guardian",
            {"module": module})

    def guardian_restart(self, module: str) -> str:
        return self.publish(
            "GUARDIAN_RESTART_REQUEST", "guardian",
            {"module": module})

    def reload_api(self) -> str:
        return self.publish(
            "RELOAD_API_REQUEST", "api",
            {})

    def force_dream(self) -> str:
        return self.publish(
            "FORCE_DREAM_REQUEST", "spirit",
            {})

    def inject_memory(self, record: dict) -> str:
        return self.publish(
            "MEMORY_INJECT_REQUEST", "memory",
            {"record": record})

    def solana_balance_refresh(self) -> str:
        return self.publish(
            "SOLANA_BALANCE_REFRESH_REQUEST", "core",
            {})

    # -- websocket / SSE bridge --------------------------------------

    def emit(self, event_type: str, payload: dict | None = None) -> str:
        """Publish a websocket-bridged event. Microkernel v2 D2 amendment
        (2026-04-26) — replaces legacy `plugin.event_bus.emit(type, payload)`
        callsites in maker.py and webhook.py. Routes via OBSERVATORY_EVENT
        bus type → BusSubscriber → SSE/WebSocket subscribers.

        Wire shape compatible with Phase B (workers persist across kernel
        swaps; event delivery survives) and Phase C (Rust L0 routes the
        same OBSERVATORY_EVENT msgpack payload).
        """
        return self.publish(
            "OBSERVATORY_EVENT", "all",
            {"event_type": event_type, "data": payload or {}})
