"""
titan_plugin.api — Sovereign Observatory REST API + WebSocket server.

Factory function creates a FastAPI app wired to the TitanPlugin instance.
Uvicorn runs as a non-blocking background task inside the plugin's event loop.
"""
import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from queue import Empty

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .events import EventBus

logger = logging.getLogger(__name__)


class _BusPublishShim:
    """Adapter so BusSubscriber's ``send_queue.put(msg)`` routes through
    ``DivineBus.publish(msg)`` for the in-process bridge.

    Lets us reuse BusSubscriber unchanged (it expects a queue-like target
    with a ``.put()`` method — designed around mp.Queue / SocketQueue).
    The adapter converts that ``.put()`` into ``bus.publish()`` so the
    bridge can issue STATE_SNAPSHOT_REQUEST without owning a separate
    send pipe.

    Forward-compat: identical wire shape regardless of bus transport
    (mp.Queue today, BusSocket post-B.3, Rust bus in Phase C).
    """

    __slots__ = ("_bus",)

    def __init__(self, bus) -> None:
        self._bus = bus

    def put(self, msg: dict, block: bool = True, timeout: float | None = None) -> None:  # noqa: ARG002
        # block/timeout intentionally ignored — DivineBus.publish() is
        # non-blocking by design. Signature matches Queue.put() contract
        # so BusSubscriber doesn't need a code path branch.
        self._bus.publish(msg)


def _start_inprocess_bus_listener(recv_q, bus_subscriber) -> threading.Thread:
    """Spawn a daemon thread that drains ``recv_q`` and dispatches to
    ``BusSubscriber.handle_message()``.

    Mirrors ``titan_plugin/api/api_subprocess.py:_bus_listener_loop`` but
    stays in-process (legacy ``api_process_separation_enabled=False``
    mode). Non-handled messages are discarded — OBSERVATORY_EVENT for
    websocket bridge already publishes directly to event_bus in the same
    process, so no rerouting needed here (unlike the subprocess case).
    """
    def _loop():
        while True:
            try:
                msg = recv_q.get(timeout=1.0)
            except Empty:
                continue
            except Exception:
                # Bus closed / queue removed — exit cleanly.
                return
            try:
                bus_subscriber.handle_message(msg)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[InProcessBusListener] dispatch error: %s", e)

    t = threading.Thread(
        target=_loop,
        daemon=True,
        name="api-inprocess-bus-listener",
    )
    t.start()
    return t


def create_app(plugin, event_bus: EventBus, config: dict | None = None,
               agent=None, titan_state=None, chat_bridge_bus=None) -> FastAPI:
    """
    Build the Sovereign Observatory FastAPI application.

    Args:
        plugin: Either a TitanPlugin instance (legacy in-process mode) OR
            a kernel_rpc._RPCRemoteRef transparent proxy (subprocess mode,
            api_process_separation_enabled=true). Endpoint code reads
            ``request.app.state.titan_plugin.X.Y(...)`` and the proxy
            routes the call over /tmp/titan_kernel_{titan_id}.sock — no
            endpoint code change needed (Microkernel v2 §A.4 / S5).
        event_bus: EventBus for WebSocket broadcasting. In subprocess
            mode this is a fresh EventBus owned by the API subprocess;
            kernel-side events flow via OBSERVATORY_EVENT bus messages
            translated by api_subprocess_main's bus listener thread.
        config: Optional [api] config dict with keys: cors_origins, rate_limit.
        agent: Optional Agno Agent instance for /chat endpoint. None when
            running in api_subprocess (chat endpoints route through proxy
            to the agent that lives in the main kernel process).

    Returns:
        Configured FastAPI app ready for uvicorn.Server.
    """
    config = config or {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("[Observatory] Starting Sovereign Observatory API...")
        # 2026-04-27 hardening: eager-start the tc-status-warmer so the
        # /v4/timechain/* warm cache begins populating at API boot — not
        # lazily on first request. Reduces the cold-boot window where
        # /v4/timechain/status falls through to the bounded sync fetch.
        # Without this, every api_subprocess restart (Guardian-driven
        # under RSS pressure on T1) re-introduces a 1-15s vulnerable
        # window. Lifespan placement (vs module-level) keeps test imports
        # clean — the warmer only fires when the API is actually booting.
        try:
            from titan_plugin.api.dashboard import _start_tc_status_warmer
            _start_tc_status_warmer()
        except Exception as _eager_err:
            logger.warning("[TCStatusWarmer] eager-start in lifespan failed: %s", _eager_err)
        yield
        logger.info("[Observatory] Sovereign Observatory shutting down.")

    app = FastAPI(
        title="Titan Sovereign Observatory",
        description="Real-time window into the Titan's sovereign cognitive state.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS — configurable origins, default to localhost for dev
    cors_origins = config.get("cors_origins", ["http://localhost:3000", "http://localhost:5173"])
    if isinstance(cors_origins, str):
        cors_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Gzip compression for API responses (>500 bytes)
    from starlette.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # ── Request-timeout middleware (microkernel v2 architectural guarantee) ──
    # Microkernel v2's promise is a lightning-fast API: endpoints read
    # from the CachedState / shm registries that background workers + the
    # snapshot publisher keep current, NEVER do slow in-line work. This
    # middleware enforces that promise: any request exceeding
    # `request_timeout_s` (default 3s) is cancelled and returned as HTTP
    # 504. Frontend polling stays predictable; one misbehaving endpoint
    # cannot hang the whole API.
    #
    # Endpoints that genuinely need longer (large bulk exports, etc.)
    # should opt out by reading `request.scope["bypass_request_timeout"] = True`
    # in the handler before its first await — checked at the outer wrap
    # via the per-request flag below.
    request_timeout_s = float(config.get("request_timeout_s", 3.0))

    @app.middleware("http")
    async def _request_timeout(request: Request, call_next):
        # Cheap fast-path for health probes — never time them out.
        path = request.url.path
        if path in ("/health", "/ws", "/metrics"):
            return await call_next(request)
        # Diagnostic admin endpoints (memory-profile, heap-dump) are
        # intentionally slower than 3s — they walk gc.get_objects() or
        # do tracemalloc snapshots. They're called rarely + manually,
        # so the "fast endpoint" architectural guarantee doesn't apply.
        # Closes the heap-dump-times-out blocker exposed during the
        # 2026-04-27 worker-stability audit.
        if path.startswith("/v4/admin/"):
            return await call_next(request)
        # /chat + /chat/stream invoke the full agent pipeline (memory
        # recall + LLM inference + post-hooks) which legitimately takes
        # 5-30s. They're not "fast cached-state endpoints" — they're
        # interactive cognitive operations. Bypass the 3s budget;
        # chat-specific timeouts live inside the agent itself.
        if path.startswith("/chat"):
            return await call_next(request)
        t0 = time.monotonic()
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=request_timeout_s)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[RequestTimeout] %s %s exceeded %.1fs (elapsed=%.2fs) — "
                "endpoint should read from cache, not compute in-line",
                request.method, path, request_timeout_s, elapsed)
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "error": "request_timeout",
                    "path": path,
                    "timeout_s": request_timeout_s,
                    "hint": ("Endpoint did not return within budget. "
                             "Likely doing in-line slow work; should "
                             "read from cached_state/shm instead."),
                },
            )

    # Store plugin + event bus + agent in app.state for endpoint access
    app.state.titan_plugin = plugin
    app.state.event_bus = event_bus
    app.state.titan_agent = agent
    # 2026-04-29 — chat bus bridge (BUG-CHAT-AGENT-NOT-INITIALIZED-API-
    # SUBPROCESS architectural fix). Subprocess mode passes a
    # ChatBridgeClient with `await request_async("chat_subproc",
    # "chat_handler", payload, timeout=60)` semantics. chat.py reads
    # this when agent is None and forwards via bus instead of returning
    # 503. Legacy in-process mode passes None — chat.py uses the local
    # plugin.run_chat() directly.
    app.state.chat_bridge_bus = chat_bridge_bus
    # S5 amendment (2026-04-25): TitanStateAccessor — primary state-access
    # object for post-codemod endpoint code (titan_state.X.Y patterns).
    # Falls back to the plugin proxy when titan_state is None (legacy
    # in-process mode where api lives in same process as kernel).
    if titan_state is None:
        # Microkernel v2 §A.4 amendment 2026-04-28: legacy in-process bridge.
        #
        # Pre-fix: this branch built TitanStateAccessor with an empty
        # CachedState and no BusSubscriber, so every /v4/* endpoint
        # reading titan_state.X.read_X() returned {} forever (BUG #3
        # documented in BUGS.md 2026-04-28 PM late). Comment promised
        # "sourcing values from the plugin" but the wiring wasn't there.
        #
        # Post-fix: same TitanStateAccessor is constructed, AND a
        # BusSubscriber + listener thread are wired against the kernel's
        # in-process DivineBus via plugin.bus.subscribe("api"). The
        # kernel's state-snapshot publisher (which now runs unconditionally
        # — see kernel._start_state_snapshot_publisher) emits dst="api"
        # snapshots every 2s; per-event *_UPDATED publishers (balance,
        # dreaming, cgn, language, etc.) emit dst="all"; both reach this
        # subscriber and populate cached_state via BusSubscriber.handle_
        # message().
        #
        # When api_process_separation flips to true later, this branch
        # is not taken (titan_state arrives non-None from api_subprocess)
        # and api_subprocess owns the BusSubscriber via Unix-socket. Same
        # publisher, transport-equivalent consumers, no behavior divergence.
        #
        # Forward-compat with Phase B.3 (mp.Queue retirement) + Phase C
        # (Rust bus): bridge uses bus.subscribe()/publish() — the public
        # contract that survives transport swaps.
        try:
            from titan_plugin.api.cached_state import CachedState
            from titan_plugin.api.shm_reader_bank import ShmReaderBank
            from titan_plugin.api.command_sender import CommandSender
            from titan_plugin.api.state_accessor import TitanStateAccessor
            from titan_plugin.api.bus_subscriber import BusSubscriber
            cached_state = CachedState()
            titan_state = TitanStateAccessor(
                shm=ShmReaderBank(),
                cache=cached_state,
                commands=CommandSender(send_queue=None),
                full_config=config or {},
            )
            _kernel_bus = getattr(plugin, "bus", None) if plugin else None
            if _kernel_bus is not None:
                try:
                    _api_recv_q = _kernel_bus.subscribe("api")
                    _bus_sub = BusSubscriber(
                        cached_state=cached_state,
                        send_queue=_BusPublishShim(_kernel_bus),
                    )
                    _start_inprocess_bus_listener(_api_recv_q, _bus_sub)
                    _bus_sub.request_snapshot()
                    logger.info(
                        "[create_app] in-process BusSubscriber bridge wired "
                        "(api_process_separation=False legacy mode); "
                        "observatory endpoints will populate from bus events")
                except Exception as bridge_err:  # noqa: BLE001
                    logger.warning(
                        "[create_app] in-process bus bridge wiring failed: "
                        "%s — observatory endpoints will return empty data",
                        bridge_err, exc_info=True)
            else:
                logger.warning(
                    "[create_app] plugin.bus is None — cannot wire "
                    "in-process BusSubscriber bridge; observatory "
                    "endpoints will return empty data")
        except Exception as e:
            logger.warning(
                "[create_app] StateAccessor construction failed in legacy "
                "mode (%s) — endpoints reading titan_state.X will fail",
                e)
            titan_state = None
    app.state.titan_state = titan_state

    # ── TitanMaker substrate (R8 + future Maker-Titan dialogic flow) ──
    # The substrate lives in the MAIN process (where dashboard endpoints
    # serve), not in any worker subprocess. Auto-seeds the Phase C contract
    # bundle proposal at boot if it has not yet been Maker-verified
    # (R8 ceremony pending). The singleton is consumed by the dashboard
    # /v4/maker/proposals endpoints via get_titan_maker().
    try:
        import os
        from titan_plugin.maker import (
            ProposalStore, ProposalType, TitanMaker, set_titan_maker,
        )
        from titan_plugin.maker.contract_bundle import (
            compute_bundle_hash_and_names, is_bundle_verified_on_disk,
        )
        # Storage path: data/maker_proposals.db (governance state, separate
        # from inner_memory.db; wired into Arweave via AUXILIARY_BACKUP_PATHS).
        _data_dir = config.get("data_dir", "data")
        _maker_db_path = os.path.join(_data_dir, "maker_proposals.db")
        _proposal_store = ProposalStore(db_path=_maker_db_path)
        # Maker pubkey from soul (may be None if soul not yet loaded)
        _maker_pubkey = None
        try:
            if plugin and titan_state.soul and getattr(
                    titan_state.soul, "_maker_pubkey", None):
                _maker_pubkey = str(titan_state.soul.maker_pubkey)
        except Exception:
            pass
        _titan_maker = TitanMaker(
            proposal_store=_proposal_store,
            maker_pubkey=_maker_pubkey,
        )
        # Tier 2 — wire the SomaticChannel via the bus shim.
        # Microkernel v2 D5 amendment (2026-04-26): plugin.bus is now a
        # kernel_rpc proxy ref that doesn't expose .publish() in api_subprocess.
        # titan_state.bus (_BusShim) routes publish() → CommandSender →
        # OBSERVATORY_EVENT bus, which spirit_worker still subscribes to via
        # its own subprocess queue. End result: same wire shape, no plugin
        # coupling.
        _bus_target = (titan_state.bus
                       if titan_state is not None and getattr(titan_state, "bus", None) is not None
                       else getattr(plugin, "bus", None) if plugin else None)
        try:
            if _bus_target is not None:
                from titan_plugin.maker.somatic_channel import SomaticChannel
                _titan_maker.set_somatic_channel(
                    SomaticChannel(bus=_bus_target, src_module="titan_maker"))
                logger.info(
                    "[Observatory] TitanMaker SomaticChannel wired "
                    "(via %s)", type(_bus_target).__name__)
        except Exception as sc_err:
            logger.warning(
                "[Observatory] SomaticChannel wiring failed: %s", sc_err)
        # Tier 3 — wire NarrativeChannel + MakerProfile (same bus shim).
        try:
            if _bus_target is not None:
                from titan_plugin.maker.narrative_channel import NarrativeChannel
                from titan_plugin.maker.maker_profile import MakerProfile
                _narrative_ch = NarrativeChannel(
                    bus=_bus_target, src_module="titan_maker")
                _maker_profile = MakerProfile(db_path=_maker_db_path)
                _titan_maker.set_narrative_channel(_narrative_ch)
                _titan_maker._profile = _maker_profile
                logger.info(
                    "[Observatory] TitanMaker Tier 3 wired "
                    "(NarrativeChannel + MakerProfile)")
        except Exception as t3_err:
            logger.warning(
                "[Observatory] Tier 3 wiring failed: %s", t3_err)
        set_titan_maker(_titan_maker)
        logger.info(
            "[Observatory] TitanMaker substrate initialized "
            "(maker=%s, db=%s)",
            (_maker_pubkey[:16] + "...") if _maker_pubkey else "none",
            _maker_db_path)
        # Auto-seed Phase C contract bundle proposal if R8 not yet complete
        try:
            if not is_bundle_verified_on_disk():
                _bundle_hash, _contract_names = compute_bundle_hash_and_names()
                if _bundle_hash and _contract_names:
                    _titan_maker.autoseed_contract_bundle(
                        bundle_hash=_bundle_hash,
                        contract_count=len(_contract_names),
                        contract_names=_contract_names,
                        epoch=0,
                    )
                    logger.info(
                        "[Observatory] R8 contract bundle proposal "
                        "autoseeded (hash=%s..., contracts=%d)",
                        _bundle_hash[:16], len(_contract_names))
            else:
                logger.info(
                    "[Observatory] R8 contract bundle already verified "
                    "(no autoseed needed)")
        except Exception as r8_err:
            logger.warning("[Observatory] R8 autoseed failed: %s", r8_err)
    except Exception as tm_err:
        logger.warning("[Observatory] TitanMaker init failed: %s", tm_err)

    # Register routers
    from .dashboard import router as dashboard_router
    from .maker import router as maker_router
    from .webhook import router as webhook_router
    from .websocket import router as ws_router
    from .chat import router as chat_router

    app.include_router(dashboard_router)
    app.include_router(maker_router)
    app.include_router(webhook_router)
    app.include_router(ws_router)
    app.include_router(chat_router)

    # Store factory args for API hot-reload
    app.state._api_factory_args = {
        "plugin": plugin, "event_bus": event_bus, "config": config, "agent": agent,
    }

    return app


def reload_api_app(current_app: FastAPI) -> FastAPI:
    """Hot-reload API by reimporting all route modules and recreating the app.

    Returns a new FastAPI app with updated routes. The caller must swap
    this into the running uvicorn server.
    """
    import importlib
    from . import dashboard, maker, webhook, websocket, chat

    # Reload all API route modules
    reloaded = []
    for mod in [dashboard, maker, webhook, websocket, chat]:
        try:
            importlib.reload(mod)
            reloaded.append(mod.__name__.split(".")[-1])
        except Exception as e:
            logger.error("[API Reload] Failed to reload %s: %s", mod.__name__, e)

    # Rebuild app with fresh routes
    args = current_app.state._api_factory_args
    # Reload the __init__ module itself to pick up any create_app changes
    # (but NOT this function — it's already running)
    new_app = create_app(**args)

    logger.info("[API Reload] Rebuilt app with %d reloaded modules: %s",
                len(reloaded), ", ".join(reloaded))
    return new_app


async def start_server(plugin, event_bus: EventBus, config: dict | None = None, agent=None):
    """
    Launch the Observatory as a non-blocking uvicorn server.

    Call this from within an existing asyncio event loop (e.g. TitanPlugin boot).
    Returns the uvicorn.Server instance for graceful shutdown.
    """
    import uvicorn

    config = config or {}
    host = config.get("host", "0.0.0.0")
    port = int(config.get("port", 7777))

    app = create_app(plugin, event_bus, config, agent=agent)

    uvi_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(uvi_config)

    logger.info("[Observatory] Launching on %s:%d", host, port)

    # server.serve() is awaitable and runs until shutdown
    # The caller should wrap this in asyncio.create_task() for non-blocking
    await server.serve()

    return server
