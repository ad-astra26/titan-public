"""
titan_plugin.api — Sovereign Observatory REST API + WebSocket server.

Factory function creates a FastAPI app wired to the TitanPlugin instance.
Uvicorn runs as a non-blocking background task inside the plugin's event loop.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .events import EventBus

logger = logging.getLogger(__name__)


def create_app(plugin, event_bus: EventBus, config: dict | None = None, agent=None) -> FastAPI:
    """
    Build the Sovereign Observatory FastAPI application.

    Args:
        plugin: TitanPlugin instance (stored in app.state for endpoint access).
        event_bus: Shared EventBus for WebSocket broadcasting.
        config: Optional [api] config dict with keys: cors_origins, rate_limit.
        agent: Optional Agno Agent instance for /chat endpoint.

    Returns:
        Configured FastAPI app ready for uvicorn.Server.
    """
    config = config or {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("[Observatory] Starting Sovereign Observatory API...")
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

    # Store plugin + event bus + agent in app.state for endpoint access
    app.state.titan_plugin = plugin
    app.state.event_bus = event_bus
    app.state.titan_agent = agent

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
            if plugin and plugin.soul and getattr(
                    plugin.soul, "_maker_pubkey", None):
                _maker_pubkey = str(plugin.soul._maker_pubkey)
        except Exception:
            pass
        _titan_maker = TitanMaker(
            proposal_store=_proposal_store,
            maker_pubkey=_maker_pubkey,
        )
        # Tier 2 — wire the SomaticChannel via the TitanCore DivineBus.
        # plugin.bus is the DivineBus instance from v5_core.TitanCore;
        # spirit_worker subscribes to MAKER_RESPONSE_RECEIVED via its
        # own subprocess queue and DivineBus routes dst="all" to it.
        try:
            if plugin and hasattr(plugin, "bus") and plugin.bus is not None:
                from titan_plugin.maker.somatic_channel import SomaticChannel
                _titan_maker.set_somatic_channel(
                    SomaticChannel(bus=plugin.bus, src_module="titan_maker"))
                logger.info(
                    "[Observatory] TitanMaker SomaticChannel wired to DivineBus")
        except Exception as sc_err:
            logger.warning(
                "[Observatory] SomaticChannel wiring failed: %s", sc_err)
        # Tier 3 — wire NarrativeChannel + MakerProfile
        try:
            if plugin and hasattr(plugin, "bus") and plugin.bus is not None:
                from titan_plugin.maker.narrative_channel import NarrativeChannel
                from titan_plugin.maker.maker_profile import MakerProfile
                _narrative_ch = NarrativeChannel(
                    bus=plugin.bus, src_module="titan_maker")
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
