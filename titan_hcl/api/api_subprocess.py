"""
api_subprocess.py — Guardian-supervised L3 API subprocess entry.

Microkernel v2 Phase A §A.4 (S5) — when
`microkernel.api_process_separation_enabled=true`, the Observatory FastAPI
app runs in a separate process spawned by Guardian instead of as a coroutine
on the main TitanHCL event loop. Solves the 2026-04-17 T2 16-minute
unresponsive-API-during-boot incident: uvicorn's accept loop is no longer
blocked by spirit_worker boot or other heavy plugin coroutines.

Process layout when flag on:
  - Main process: TitanKernel + TitanHCL + DivineBus + KernelRPCServer
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
  - titan_hcl/core/kernel_rpc.py (the transport layer this connects to)
  - titan_hcl/api/__init__.py:create_app (factory accepts proxy or plugin)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty
from typing import Any

from titan_hcl.bus import (
    MODULE_HEARTBEAT,
    OBSERVATORY_EVENT,
    make_msg,
)
from titan_hcl import bus

logger = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_S = 30.0
KERNEL_CONNECT_TIMEOUT_S = 30.0


def _make_reuseport_socket(host: str, port: int):
    """SPEC §11.B.5 / rFP_kernel_zero_downtime_api_reload P1 — build the api's
    listen socket with SO_REUSEPORT so OLD + NEW api processes can co-bind the
    same port during a kernel-driven zero-downtime reload (the OS load-balances
    new connections across both; OLD drains its in-flight set on SIGTERM).

    uvicorn 0.41's Config exposes no reuse_port param and its bind_socket() sets
    only SO_REUSEADDR — so we bind the socket ourselves and hand it to
    server.serve(sockets=[...]), which skips uvicorn's internal bind. INV-9 — a
    pre-bound socket adds no sync-blocking to the api event loop.
    """
    import socket as _socket
    fam = _socket.AF_INET6 if (host and ":" in host) else _socket.AF_INET
    sock = _socket.socket(fam, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


def _await_health_ready(plugin, kernel_proxy, timeout_s: float) -> None:
    """SPEC §11.B.5 — reload-child readiness gate.

    A zero-downtime reload child must NOT co-bind port 7777 until it can
    actually serve 200s — otherwise SO_REUSEPORT routes OLD's live traffic to
    us mid-warmup and we 503 it (the premature-routing gap). `/health` returns
    200 off a warm in-process cache (dashboard `_health_summary_cache`,
    refreshed by a background warmer). So we prime that warmer and block until
    the cache is populated (or `timeout_s`), THEN the caller binds the port —
    so OLD keeps 100% of traffic until we're genuinely ready. Bounded under the
    kernel's API_RELOAD_HEALTH_TIMEOUT_S so the swap never stalls.
    """
    import time as _t
    try:
        from titan_hcl.api.dashboard import (
            _get_health_summary_cached, _start_health_warmer)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[api_subprocess] §11.B.5 readiness-gate import failed (%s) — "
            "binding immediately", e)
        return
    try:
        _start_health_warmer(plugin, kernel_proxy)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[api_subprocess] §11.B.5 health-warmer start failed (%s) — "
            "binding immediately", e)
        return
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        try:
            if _get_health_summary_cached() is not None:
                logger.info(
                    "[api_subprocess] §11.B.5 reload child health-warm — "
                    "binding SO_REUSEPORT port now (OLD kept all traffic during warmup)")
                return
        except Exception:  # noqa: BLE001
            pass
        _t.sleep(0.2)
    logger.warning(
        "[api_subprocess] §11.B.5 reload-child readiness gate timed out after "
        "%.0fs — binding anyway (may briefly 503 until warm)", timeout_s)


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
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = resolve_titan_id()

    logger.info("[ApiSubprocess] starting (titan_id=%s, pid=%d)",
                titan_id, os.getpid())

    # 2. kernel_rpc — LAZY, non-blocking (Phase 11 §11.I.1 standalone L3).
    #
    # The api is a true standalone L3 (INV-PROC-5): it binds uvicorn and
    # serves SHM-direct reads (Preamble G18) WITHOUT waiting for L2. The
    # kernel_rpc connection is the L3→L2 bridge used ONLY by mutating /
    # residual `plugin_proxy` endpoints; per INV-PROC-5 those transiently
    # fail until L2 (titan_hcl) is up, while state reads stay 200.
    #
    # Why this MUST be lazy under kernel-rs peer-spawn: kernel_rpc binds
    # inside TitanHCL.boot() which runs AFTER titan_hcl's orchestrator
    # start_all (minutes of worker dependency waves). The api is spawned
    # by kernel-rs at T+0, so a blocking connect would either time out and
    # exit (the old `return # Guardian will restart us` is dead — the api
    # is a kernel-rs peer now, nothing respawns it) or pin the whole API
    # offline for the entire L2 boot. `get_plugin_proxy()` does NOT touch
    # the socket — the proxy only dials on first `.call()`, and we
    # establish the connection in a background thread that retries until
    # L2's kernel_rpc server binds.
    from titan_hcl.core.kernel_rpc import KernelRPCClient
    rpc_client = KernelRPCClient(
        titan_id=titan_id, connect_timeout_s=KERNEL_CONNECT_TIMEOUT_S)
    plugin_proxy = rpc_client.get_plugin_proxy()

    _rpc_connect_stop = threading.Event()

    def _kernel_rpc_connect_loop():
        attempt = 0
        while not _rpc_connect_stop.is_set():
            attempt += 1
            try:
                rpc_client.connect()
                logger.info(
                    "[ApiSubprocess] kernel_rpc connected (attempt %d) — "
                    "mutating/plugin-proxy endpoints now live", attempt)
                return
            except Exception as e:  # noqa: BLE001
                logger.info(
                    "[ApiSubprocess] kernel_rpc not ready yet (attempt %d): "
                    "%s — serving SHM-direct reads meanwhile, retrying in 5s",
                    attempt, e)
                _rpc_connect_stop.wait(5.0)

    _rpc_connect_thread = threading.Thread(
        target=_kernel_rpc_connect_loop, daemon=True,
        name="api-kernel-rpc-connect")
    _rpc_connect_thread.start()
    logger.info(
        "[ApiSubprocess] kernel_rpc connect deferred to background thread "
        "(standalone L3 — uvicorn + SHM reads do not block on L2)")

    # TitanStateAccessor — primary state read object. All sub-accessors
    # are SHM-direct per Preamble G18 (D-SPEC-71 Phase A + D-SPEC-78 Phase B
    # + D-SPEC-79 Phase C closed bus-cache + Python-wrapper + sync-RPC
    # state-read paths fleet-wide). The bus-cache → CachedState pipeline
    # is RETIRED per Phase D — no BusSubscriber, no cache writes, no
    # kernel snapshot push. The legacy plugin_proxy is kept as a
    # backward-compat fallback for the residual Category C callsites
    # that still touch plugin.bus / hasattr etc.
    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    from titan_hcl.api.command_sender import CommandSender
    from titan_hcl.api.state_accessor import TitanStateAccessor

    shm_bank = ShmReaderBank(titan_id=titan_id)
    command_sender = CommandSender(send_queue=send_queue)
    titan_state = TitanStateAccessor(
        shm=shm_bank,
        commands=command_sender,
        full_config=config,
    )

    # 2026-04-29 — chat bus bridge client (BUG-CHAT-AGENT-NOT-INITIALIZED-
    # API-SUBPROCESS architectural fix). Owns the QUERY/RESPONSE rid-
    # routing for /chat forwarding. Hooked into _bus_listener_loop below
    # so RESPONSE messages reach pending Futures; passed to create_app
    # so chat.py's Mode 2 path can call request_async on it.
    from titan_hcl.api.chat_bridge_client import ChatBridgeClient
    chat_bridge_client = ChatBridgeClient(send_queue=send_queue)

    # D-SPEC-73 (SPEC v1.18.0) — subprocess-side backend for AgnoProxy.
    # Owns CHAT_REQUEST / CHAT_STREAM_REQUEST → CHAT_RESPONSE /
    # CHAT_STREAM_CHUNK rid-routing for /chat + /v4/pitch-chat against the
    # NEW agno_worker target (replaces the retired chat_handler bridge).
    # Hooked into _bus_listener_loop below so CHAT_RESPONSE and
    # CHAT_STREAM_CHUNK messages reach pending Futures / Queues; passed
    # to create_app so AgnoProxy resolves to its bridge backend (no
    # bus.subscribe via kernel_rpc — closes CHAT-500 fleet-wide).
    from titan_hcl.api.agno_bridge_client import AgnoBridgeClient
    agno_bridge_client = AgnoBridgeClient(send_queue=send_queue)

    # 3. Build local EventBus (websocket subscribers live in THIS process)
    from titan_hcl.api.events import EventBus
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

    # 4b. Memory hygiene daemon — periodic gc.collect() + glibc malloc_trim(0).
    # Keeps api subprocess RSS bounded. Mirrors kernel._memory_hygiene_loop().
    # 2026-05-01 — interim measure ahead of Phase C C-S7 Rust kernel swap.
    # Configurable via [microkernel] memory_hygiene_interval_s (set 0 = off).
    _mh_stop = threading.Event()
    _mh_interval_s = float(config.get("microkernel", {}).get(
        "memory_hygiene_interval_s", 60.0))

    def _memory_hygiene_loop():
        if _mh_interval_s <= 0:
            logger.info("[MemHygiene:api] disabled (interval_s=%s)",
                        _mh_interval_s)
            return
        logger.info("[MemHygiene:api] starting (interval_s=%.1f)",
                    _mh_interval_s)
        libc = None
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError as e:
            logger.warning("[MemHygiene:api] libc.so.6 unavailable (%s) — "
                           "running gc.collect() only", e)
        import gc
        while not _mh_stop.is_set():
            if _mh_stop.wait(_mh_interval_s):
                return
            try:
                t0 = time.time()
                n_collected = gc.collect()
                t1 = time.time()
                trim_result = -1
                if libc is not None:
                    try:
                        trim_result = libc.malloc_trim(0)
                    except Exception as trim_err:  # noqa: BLE001
                        logger.debug(
                            "[MemHygiene:api] malloc_trim failed: %s",
                            trim_err)
                t2 = time.time()
                logger.info(
                    "[MemHygiene:api] gc=%d freed (%.1fms) trim=%d (%.1fms)",
                    n_collected, (t1 - t0) * 1000,
                    trim_result, (t2 - t1) * 1000)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemHygiene:api] cycle failed: %s", e)

    mh_thread = threading.Thread(
        target=_memory_hygiene_loop, daemon=True, name="api-mem-hygiene")
    mh_thread.start()

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
            # 2026-04-29 — chat bus bridge: dispatch RESPONSE messages to
            # pending /chat Futures by rid match. Returns True if claimed.
            if chat_bridge_client.handle_response(msg):
                continue
            # D-SPEC-73 (SPEC v1.18.0) — agno bridge: dispatch
            # CHAT_RESPONSE + CHAT_STREAM_CHUNK to pending Futures /
            # Queues by rid match. Falls through (False) if msg.type not
            # in {CHAT_RESPONSE, CHAT_STREAM_CHUNK} or rid not pending.
            if agno_bridge_client.handle_response(msg):
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
    # plugin_proxy is also stored on app.state.titan_hcl for the residual
    # Category C callsites that still reference plugin.X directly (these
    # will fail gracefully with RPC errors if hit; gradually migrated in
    # follow-up sessions).
    from titan_hcl.api import create_app
    api_cfg = config.get("api", {})
    app = create_app(plugin_proxy, event_bus, api_cfg, agent=None,
                     titan_state=titan_state,
                     chat_bridge_bus=chat_bridge_client,
                     agno_bridge=agno_bridge_client)

    # 7. Start uvicorn (serve() blocks). Phase 11 §11.I.2 (D1/D2): legacy
    # MODULE_READY emit DELETED — api readiness is its SHM slot (api_main
    # ModuleStateWriter), not a bus broadcast.
    import uvicorn
    host = api_cfg.get("host", "0.0.0.0")
    # B.1 §7 — TITAN_API_PORT env var override for shadow kernel boot
    # (titan_hcl.py sets it from --shadow-port). Falls back to config
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

    # SPEC §11.B.5 / rFP P1 — SO_REUSEPORT, gated on TITAN_API_REUSEPORT=1
    # (kernel sets it on EVERY api spawn so the running api is always
    # swap-ready). See _make_reuseport_socket above.
    #
    # P4 refinement: a RELOAD CHILD (TITAN_API_RELOAD_CHILD=1) defers the bind
    # until its /health is warm — otherwise SO_REUSEPORT would route OLD's live
    # traffic to us mid-warmup and 503 it. A normal boot binds immediately
    # (no OLD serving → nothing to protect).
    _reuse_sock = None
    if os.environ.get("TITAN_API_REUSEPORT") == "1":
        if os.environ.get("TITAN_API_RELOAD_CHILD") == "1":
            _await_health_ready(titan_state, plugin_proxy, timeout_s=25.0)
        _reuse_sock = _make_reuseport_socket(host, port)
        logger.info(
            "[api_subprocess] §11.B.5 SO_REUSEPORT enabled (TITAN_API_REUSEPORT=1) "
            "— co-bound %s:%d", host, port)

    # Capture the event loop reference so the bus listener can schedule
    # event_bus.emit() coroutines on it.
    async def _set_loop_ref():
        _ws_loop_holder["loop"] = asyncio.get_running_loop()

    async def _serve_with_loop_capture():
        await _set_loop_ref()
        if _reuse_sock is not None:
            # P1: serve on the SO_REUSEPORT-bound socket (uvicorn skips its own bind).
            await server.serve(sockets=[_reuse_sock])
        else:
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
        _rpc_connect_stop.set()
        try:
            rpc_client.close()
        except Exception:
            pass
        logger.info("[ApiSubprocess] shutdown complete")
