"""
api_subprocess.py — Guardian-supervised L3 API subprocess entry.

Microkernel v2 Phase A §A.4 (S5) — when
`microkernel.api_process_separation_enabled=true`, the Observatory FastAPI
app runs in a separate process spawned by Guardian instead of as a coroutine
on the main TitanPlugin event loop. Solves the 2026-04-17 T2 16-minute
unresponsive-API-during-boot incident: uvicorn's accept loop is no longer
blocked by spirit_worker boot or other heavy plugin coroutines.

Process layout when flag on:
  - Main process: TitanKernel + TitanPlugin + DivineBus + KernelRPCServer
  - This process (api): uvicorn + FastAPI + WebSocket subscribers + bus
                        client + KernelRPCClient (transparent plugin proxy)

Inter-process communication:
  - DivineBus (recv/send queues) — bus events + cross-worker messages.
    The api subprocess subscribes to OBSERVATORY_EVENT messages (kernel→api
    websocket bridge) and translates them to local event_bus.emit() for
    WebSocket clients.
  - kernel_rpc Unix socket — transparent plugin attribute access.
    Endpoint code reads `plugin.guardian.get_status()` and the
    _RPCRemoteRef proxy routes the call over /tmp/titan_kernel_{id}.sock.
  - /dev/shm — shm registries (Trinity/Neuromod/Epoch/CGN/etc.) read
    directly without IPC overhead. Same path as in-process API.

Heartbeat: standard MODULE_HEARTBEAT every 30s via send_queue (Guardian
heartbeat_timeout default 60s catches stalls).

See:
  - PLAN_microkernel_phase_a_s5.md §2.1 + §4.2
  - titan_plugin/core/kernel_rpc.py (the transport layer this connects to)
  - titan_plugin/api/__init__.py:create_app (factory accepts proxy or plugin)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty
from typing import Any

from titan_plugin.bus import (
    MODULE_HEARTBEAT,
    MODULE_READY,
    OBSERVATORY_EVENT,
    make_msg,
)
from titan_plugin import bus

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
KERNEL_CONNECT_TIMEOUT_S = 30.0


def _send_msg(send_queue, msg_type: str, src: str, dst: str,
              payload: dict, rid: str | None = None) -> None:
    """Standard worker → bus message helper. Mirrors emot_cgn_worker / etc."""
    try:
        send_queue.put_nowait(make_msg(msg_type, src, dst, payload, rid=rid))
    except Exception as e:
        logger.warning("[ApiSubprocess] _send_msg failed (%s): %s", msg_type, e)


def _send_heartbeat(send_queue, name: str) -> None:
    """Standard MODULE_HEARTBEAT — mirrors guardian's expected pattern."""
    _send_msg(send_queue, MODULE_HEARTBEAT, name, "guardian", {})


# ── Subprocess main ─────────────────────────────────────────────────


def api_subprocess_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """L3 module entry — Guardian supervised. Runs uvicorn in this process.

    Boot sequence:
      1. Connect to KernelRPCServer (with retry; Guardian restarts us if hung)
      2. Build FastAPI app with the remote plugin proxy
      3. Subscribe to OBSERVATORY_EVENT bus messages → translate to local
         EventBus.emit() for WebSocket clients
      4. Start uvicorn server in this process's event loop
      5. Heartbeat thread keeps Guardian happy

    Crashes are caught by Guardian's standard restart loop. Crashes inside
    uvicorn endpoint handlers do not propagate beyond this process — the
    kernel + workers continue uninterrupted.
    """
    import asyncio
    import os

    # 1. Resolve titan_id (canonical chain: data/titan_identity.json → env → "T1")
    from titan_plugin.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id()

    logger.info("[ApiSubprocess] starting (titan_id=%s, pid=%d)",
                titan_id, os.getpid())

    # 2. Connect to kernel RPC
    from titan_plugin.core.kernel_rpc import KernelRPCClient
    rpc_client = KernelRPCClient(
        titan_id=titan_id, connect_timeout_s=KERNEL_CONNECT_TIMEOUT_S)
    try:
        rpc_client.connect()
    except Exception as e:
        logger.error(
            "[ApiSubprocess] kernel_rpc connect failed: %s", e, exc_info=True)
        return  # Guardian will restart us
    plugin_proxy = rpc_client.get_plugin_proxy()
    logger.info("[ApiSubprocess] connected to kernel_rpc")

    # S5 amendment (2026-04-25): build the TitanStateAccessor — this is what
    # endpoint code actually reads from now (post-codemod). The legacy
    # plugin_proxy is kept as a backward-compat fallback for the remaining
    # ~135 Category C callsites that still touch plugin.bus / hasattr etc.
    from titan_plugin.api.cached_state import CachedState
    from titan_plugin.api.shm_reader_bank import ShmReaderBank
    from titan_plugin.api.bus_subscriber import BusSubscriber
    from titan_plugin.api.command_sender import CommandSender
    from titan_plugin.api.state_accessor import TitanStateAccessor

    cached_state = CachedState()
    shm_bank = ShmReaderBank(titan_id=titan_id)
    command_sender = CommandSender(send_queue=send_queue)
    titan_state = TitanStateAccessor(
        shm=shm_bank,
        cache=cached_state,
        commands=command_sender,
        full_config=config,
    )
    bus_subscriber = BusSubscriber(
        cached_state=cached_state, send_queue=send_queue)
    bus_subscriber.request_snapshot()  # bootstrap — kernel publishes back

    # 2026-04-29 — chat bus bridge client (BUG-CHAT-AGENT-NOT-INITIALIZED-
    # API-SUBPROCESS architectural fix). Owns the QUERY/RESPONSE rid-
    # routing for /chat forwarding. Hooked into _bus_listener_loop below
    # so RESPONSE messages reach pending Futures; passed to create_app
    # so chat.py's Mode 2 path can call request_async on it.
    from titan_plugin.api.chat_bridge_client import ChatBridgeClient
    chat_bridge_client = ChatBridgeClient(send_queue=send_queue)

    # 3. Build local EventBus (websocket subscribers live in THIS process)
    from titan_plugin.api.events import EventBus
    event_bus = EventBus()

    # 4. Heartbeat thread (sends MODULE_HEARTBEAT every 30s)
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name)
            _hb_stop.wait(HEARTBEAT_INTERVAL_S)

    hb_thread = threading.Thread(
        target=_heartbeat_loop, daemon=True, name="api-heartbeat")
    hb_thread.start()

    # 5. Bus listener thread — translate OBSERVATORY_EVENT → event_bus.emit()
    _bus_stop = threading.Event()
    _ws_loop_holder: dict[str, Any] = {"loop": None}

    def _bus_listener_loop():
        """Translate kernel-side OBSERVATORY_EVENT bus msgs into local
        event_bus.emit() calls for WebSocket subscribers.

        EventBus.emit() is async — we schedule it on the uvicorn event
        loop via run_coroutine_threadsafe. The loop reference is set
        once uvicorn has started (Step 7).
        """
        while not _bus_stop.is_set():
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                continue
            except Exception as e:
                logger.warning("[ApiSubprocess] bus recv error: %s", e)
                continue

            msg_type = msg.get("type")
            # S5 amendment: route bus messages through BusSubscriber first
            # so cache stays current. Returns True if it owns this msg_type.
            if bus_subscriber.handle_message(msg):
                continue
            # 2026-04-29 — chat bus bridge: dispatch RESPONSE messages to
            # pending /chat Futures by rid match. Returns True if claimed.
            if chat_bridge_client.handle_response(msg):
                continue
            if msg_type == OBSERVATORY_EVENT:
                payload = msg.get("payload", {})
                ev_type = payload.get("event_type", "unknown")
                ev_data = payload.get("data", {})
                loop = _ws_loop_holder.get("loop")
                if loop is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            event_bus.emit(ev_type, ev_data), loop)
                    except Exception as e:
                        # Per directive_error_visibility: WARNING+ in critical
                        # paths so silent dispatch failures surface in logs.
                        logger.warning(
                            "[ApiSubprocess] event_bus dispatch err (%s): %s",
                            ev_type, e)
            elif msg_type == bus.RELOAD:
                # Hot-reload API routes (Layer 2 hot-reload — preserved across
                # subprocess boundary). Triggered via /v4/reload-api.
                logger.info("[ApiSubprocess] RELOAD received — rebuilding app")
                # Note: in subprocess mode reload-api is a kernel→api signal,
                # not the legacy in-process swap. We just exit; Guardian
                # restarts us with fresh code on next CONFIG_RELOAD or restart.
                _bus_stop.set()
                return
            # Other bus types ignored — not for the API.

    bus_thread = threading.Thread(
        target=_bus_listener_loop, daemon=True, name="api-bus-listener")
    bus_thread.start()

    # 6. Build FastAPI app — pass the StateAccessor as the primary state ref.
    # create_app stores it on app.state.titan_state; endpoint code reads
    # `request.app.state.titan_state.X.Y` (post-S5-amendment). The legacy
    # plugin_proxy is also stored on app.state.titan_plugin for the residual
    # Category C callsites that still reference plugin.X directly (these
    # will fail gracefully with RPC errors if hit; gradually migrated in
    # follow-up sessions).
    from titan_plugin.api import create_app
    api_cfg = config.get("api", {})
    app = create_app(plugin_proxy, event_bus, api_cfg, agent=None,
                     titan_state=titan_state,
                     chat_bridge_bus=chat_bridge_client)

    # 7. Start uvicorn — this is where we publish MODULE_READY (right
    # before starting serve, since serve() blocks).
    _send_msg(send_queue, MODULE_READY, name, "guardian", {})

    import uvicorn
    host = api_cfg.get("host", "0.0.0.0")
    # B.1 §7 — TITAN_API_PORT env var override for shadow kernel boot
    # (titan_main.py sets it from --shadow-port). Falls back to config
    # api.port for normal boots.
    _env_port = os.environ.get("TITAN_API_PORT")
    if _env_port:
        try:
            port = int(_env_port)
            logger.info("[api_subprocess] B.1 shadow port override: TITAN_API_PORT=%d", port)
        except ValueError:
            logger.warning("[api_subprocess] invalid TITAN_API_PORT=%s — using config", _env_port)
            port = int(api_cfg.get("port", 7777))
    else:
        port = int(api_cfg.get("port", 7777))
    uvi_config = uvicorn.Config(
        app=app, host=host, port=port, log_level="info", access_log=False,
    )
    server = uvicorn.Server(uvi_config)

    # Capture the event loop reference so the bus listener can schedule
    # event_bus.emit() coroutines on it.
    async def _set_loop_ref():
        _ws_loop_holder["loop"] = asyncio.get_running_loop()

    async def _serve_with_loop_capture():
        await _set_loop_ref()
        await server.serve()

    try:
        logger.info(
            "[ApiSubprocess] uvicorn starting on %s:%d", host, port)
        asyncio.run(_serve_with_loop_capture())
    except Exception as e:
        logger.error("[ApiSubprocess] uvicorn failed: %s", e, exc_info=True)
    finally:
        _hb_stop.set()
        _bus_stop.set()
        try:
            rpc_client.close()
        except Exception:
            pass
        logger.info("[ApiSubprocess] shutdown complete")
