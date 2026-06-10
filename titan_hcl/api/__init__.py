"""
titan_hcl.api — Sovereign Observatory REST API + WebSocket server.

Factory function creates a FastAPI app wired to the TitanHCL instance.
Uvicorn runs as a non-blocking background task inside the plugin's event loop.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .events import EventBus

logger = logging.getLogger(__name__)


def create_app(plugin, event_bus: EventBus, config: dict | None = None,
               agent=None, titan_state=None, chat_bridge_bus=None,
               agno_bridge=None) -> FastAPI:
    """
    Build the Sovereign Observatory FastAPI application.

    Args:
        plugin: Either a TitanHCL instance (legacy in-process mode) OR
            a kernel_rpc._RPCRemoteRef transparent proxy (subprocess mode,
            api_process_separation_enabled=true). Endpoint code reads
            ``request.app.state.titan_hcl.X.Y(...)`` and the proxy
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
        # Eager-start of warm-cache builders so heavy /v4/* endpoints
        # are warm by the time the frontend hits them on first connect.
        # Without this, every api_subprocess restart (Guardian-driven
        # under RSS pressure on T1) re-introduces a 1-15s vulnerable
        # window where the first request falls through to the bounded
        # sync fetch. Lifespan placement (vs module-level) keeps test
        # imports clean — warmers only fire when the API is actually
        # booting.
        #
        # 2026-04-27: tc-status-warmer (BUG-TIMECHAIN-STATUS-INLINE-COMPUTE)
        # 2026-05-05: vocabulary + tc-verify + v4-history warmers
        #             (OBSERVATORY-API-LATENCY-AUDIT closure +
        #              BUG-TIMECHAIN-VERIFY-INLINE-COMPUTE-20260505)
        for warmer_attr in (
            "_start_tc_status_warmer",
            "_start_tc_verify_warmer",
            "_start_vocabulary_warmer",
            "_start_v4_history_warmer",
        ):
            try:
                from titan_hcl.api import dashboard as _dash
                fn = getattr(_dash, warmer_attr, None)
                if fn is not None:
                    fn()
            except Exception as _eager_err:
                logger.warning(
                    "[Observatory] %s eager-start failed: %s",
                    warmer_attr, _eager_err)
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
    # from SHM slots (Phase A/B/C/D state-read unification, per Preamble
    # G18) — sub-µs latency, NEVER slow in-line work. This middleware
    # enforces that promise: any request exceeding `request_timeout_s`
    # (default 3s) is cancelled and returned as HTTP 504. Frontend polling
    # stays predictable; one misbehaving endpoint cannot hang the whole API.
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
        # /v6/pitch/* — bypass the 3s "fast cached-state endpoint" budget
        # for the entire pitch group:
        #   - /chat invokes the same agent pipeline as /chat (wallet-less
        #     pitch chat for VC + hackathon route, rFP §5; internal 60s
        #     timeout lives inside pitch_chat.run_chat()).
        #   - /witness-tail, /thinking-tail read observatory.db.event_log
        #     which can be multi-GB on long-running Titans (T1 ~4.2 GB on
        #     2026-05-26 — the sqlite SELECT, even wrapped in
        #     asyncio.to_thread, can exceed 3s on first cold hit).
        #   - /sessions, /sessions/{thread_id} walk data/pitch_sessions/
        #     filesystem, also potentially >3s on slow disk.
        # Legacy /v4/pitch-chat path also bypassed: 308-redirect itself is
        # sub-3s but we keep the prefix so the post-redirect retry lands
        # here too.
        if path.startswith("/v6/pitch/") or path.startswith("/v4/pitch-chat"):
            return await call_next(request)
        # Spirit-RPC work endpoints — /v4/signal-concept, /v4/signal-co-
        # occurrence, /v4/social-relief — publish QUERY dst="spirit" via
        # chat_bridge_bus.request_async and wait for RESPONSE rid-routed
        # back. Total round-trip is 1-5s depending on spirit_worker load
        # (signal_concept walks MSL retrieval; social_relief touches
        # SocialPressureMeter; signal_co_occurrence reinforces the
        # interaction matrix). They're work-RPC not cached-state reads,
        # so the 3s "fast endpoint" budget genuinely doesn't apply. Same
        # bypass rationale as /chat — an internal 5-15s timeout lives in
        # `_spirit_query_async`. Closes BUG-DASHBOARD-BUS-ATTR-ERRORS
        # Phase 2 deploy regression on T2/T3 (T1 was lucky-fast first try).
        if path in ("/v4/signal-concept", "/v4/signal-co-occurrence",
                    "/v4/social-relief"):
            return await call_next(request)
        # /v4/llm-distill + /v4/llm-score (D-SPEC-88, Phase 3 Chunk ω) —
        # HTTP proxy to LLM_DISTILL_REQUEST / LLM_SCORE_REQUEST bus topics
        # for out-of-kernel cron callers (events_teacher, persona_endurance,
        # persona_social_v2). The bus round-trip + Ollama Cloud generation
        # legitimately takes 1-45s (deepseek-v3.1:671b heavy distill on T1
        # peaks ~40s; gemma3:4b on greeting-style ~1s). Internal timeout
        # lives in the request payload (timeout_s, default 30s).
        if path in ("/v4/llm-distill", "/v4/llm-score"):
            return await call_next(request)
        # /v4/mood-narrative + /v4/art-narration (D-SPEC-88, Phase 3 Chunk
        # ω-cleanup) — wrap an internal /v4/llm-distill round-trip
        # (gemma4:31b, 60-80 tokens). Total round-trip 5-20s. Each has its
        # own 15s internal timeout in `distill_via_http_async`.
        if path in ("/v4/mood-narrative", "/v4/art-narration"):
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
                             "read from SHM slots instead."),
                },
            )

    # Store plugin + event bus + agent in app.state for endpoint access
    app.state.titan_hcl = plugin
    app.state.event_bus = event_bus
    app.state.titan_agent = agent
    # D-SPEC-73 (SPEC v1.18.0) — install the AgnoProxy with backend-
    # selection: in api_subprocess context, use the AgnoBridgeClient
    # backend (send_queue + rid Future/Queue routing) so /chat does NOT
    # call bus.subscribe via kernel_rpc (which is blocked, CHAT-500).
    # In parent-process / in-process contexts, fall back to the proxy
    # already installed at plugin._proxies["agno"] (DivineBus backend).
    app.state.agno_bridge = agno_bridge
    if agno_bridge is not None:
        try:
            from titan_hcl.proxies.agno_proxy import AgnoProxy
            app.state.agno_proxy = AgnoProxy(bridge=agno_bridge)
            logger.info("[create_app] AgnoProxy installed with "
                        "AgnoBridgeClient backend (api_subprocess context)")
        except Exception as e:
            logger.warning(
                "[create_app] AgnoProxy bridge-backend install failed: %s — "
                "chat.py will lazy-construct on first request", e)
    else:
        # Parent / in-process — read pre-installed proxy with bus backend
        try:
            _proxies = getattr(plugin, "_proxies", None)
            if isinstance(_proxies, dict):
                _agno = _proxies.get("agno")
                if _agno is not None:
                    app.state.agno_proxy = _agno
        except Exception:
            pass
    # 2026-04-29 — chat bus bridge (BUG-CHAT-AGENT-NOT-INITIALIZED-API-
    # SUBPROCESS architectural fix). Subprocess mode passes a
    # ChatBridgeClient with `await request_async("chat_subproc",
    # "chat_handler", payload, timeout=60)` semantics. chat.py reads
    # this when agent is None and forwards via bus instead of returning
    # 503. Legacy in-process mode passes None — chat.py uses the local
    # plugin.run_chat() directly.
    app.state.chat_bridge_bus = chat_bridge_bus
    # TitanStateAccessor — primary state-read object for post-codemod
    # endpoint code (titan_state.X.Y patterns). Falls back to a fresh
    # SHM-direct instance when titan_state is None (legacy in-process
    # mode where api lives in same process as kernel). Phase D retired
    # the bus-cache → CachedState bootstrap path; every sub-accessor is
    # now SHM-direct per Preamble G18 (D-SPEC-71 + D-SPEC-78 + D-SPEC-79).
    if titan_state is None:
        try:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            from titan_hcl.api.command_sender import CommandSender
            from titan_hcl.api.state_accessor import TitanStateAccessor
            titan_state = TitanStateAccessor(
                shm=ShmReaderBank(),
                commands=CommandSender(send_queue=None),
                full_config=config or {},
            )
            logger.info(
                "[create_app] TitanStateAccessor constructed SHM-direct "
                "(in-process mode; Phase D — no BusSubscriber bridge)")
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
        from titan_hcl.maker import (
            ProposalStore, ProposalType, TitanMaker, set_titan_maker,
        )
        from titan_hcl.maker.contract_bundle import (
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
                from titan_hcl.maker.somatic_channel import SomaticChannel
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
                from titan_hcl.maker.narrative_channel import NarrativeChannel
                from titan_hcl.maker.maker_profile import MakerProfile
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
    from .pitch_chat import router as pitch_chat_router
    from .llm_proxy_endpoints import router as llm_proxy_router
    from .v6 import router as v6_router
    from .v6_deprecation import router as v6_deprecation_router

    app.include_router(dashboard_router)
    app.include_router(maker_router)
    app.include_router(webhook_router)
    app.include_router(ws_router)
    app.include_router(chat_router)
    app.include_router(pitch_chat_router)
    app.include_router(llm_proxy_router)
    # Phase E — api/v6 single readout roof (RFP §2 Phase E). The legacy /v3,/v4
    # route bodies are removed from dashboard.py (no-shim); their handler
    # functions are re-mounted under /v6 by v6_router, and the legacy paths are
    # served only as 301/308 redirects to /v6 by v6_deprecation_router.
    app.include_router(v6_router)
    app.include_router(v6_deprecation_router)

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
    from . import dashboard, maker, webhook, websocket, chat, pitch_chat
    from . import v6_manifest, v6, v6_deprecation

    # Reload all API route modules. v6_manifest before v6 so the manifest
    # REGISTRY is freshly cleared+repopulated; v6 before pitch_chat so
    # pitch_chat's manifest rows (Phase E v6/pitch group) don't get wiped
    # when v6_manifest reloads; v6_deprecation LAST so it builds the legacy
    # /v3,/v4 → /v6 redirects from the fully-populated REGISTRY.
    reloaded = []
    for mod in [dashboard, maker, webhook, websocket, chat,
                v6_manifest, v6, pitch_chat, v6_deprecation]:
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

    Call this from within an existing asyncio event loop (e.g. TitanHCL boot).
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
