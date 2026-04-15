"""
TitanCore — V3.0 Microkernel Bootstrap.

Boots only the essential subsystems needed for a responsive API:
  - SovereignSoul (identity, wallet)
  - DivineBus (IPC message router)
  - Guardian (module supervisor)
  - FastAPI Observatory (health, /chat, WebSocket)
  - EventBus (real-time broadcasting)

Everything else (Memory, RL, LLM, Mind, Body, Spirit) loads on demand
via lazy proxies → Guardian-supervised module processes.

Target: <5s boot, <200MB RSS, API responds immediately.
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .bus import DivineBus, make_msg, MODULE_HEARTBEAT, EPOCH_TICK, IMPULSE, ACTION_RESULT, OUTER_OBSERVATION
from .guardian import Guardian, ModuleSpec

logger = logging.getLogger(__name__)


class TitanCore:
    """
    Minimal kernel that boots fast and delegates heavy work to supervised modules.

    Usage (from titan_main.py --v3):
        core = TitanCore(wallet_path)
        await core.boot()
        # API is now live, modules load on demand
    """

    def __init__(self, wallet_path: str):
        self._boot_start = time.time()

        # ── Load config ──────────────────────────────────────────────
        self._full_config = self._load_full_config()

        # ── Divine Bus ───────────────────────────────────────────────
        self.bus = DivineBus(maxsize=10000)
        self._core_queue = self.bus.subscribe("core", reply_only=True)
        # Pre-subscribe meditation queue before Guardian starts modules
        # (spirit_worker may send MEDITATION_REQUEST during boot)
        self._meditation_queue = self.bus.subscribe("meditation", reply_only=True)

        # ── StateRegister (real-time state buffer) ──────────────────
        from titan_plugin.logic.state_register import StateRegister
        self.state_register = StateRegister()
        enrichment_cfg = self._full_config.get("spirit_enrichment", {})
        snapshot_interval = float(enrichment_cfg.get("micro_tick_interval", 10.0))
        self.state_register.start(self.bus, snapshot_interval=snapshot_interval)

        # ── Guardian ─────────────────────────────────────────────────
        self.guardian = Guardian(self.bus)

        # ── Disk Health Monitor ──────────────────────────────────────
        # Background thread publishing DISK_WARNING/CRITICAL/EMERGENCY on
        # edge-detected transitions. On EMERGENCY, triggers graceful
        # Guardian.stop_all() via shutdown_fn hook. Protects against the
        # 2026-04-14 disk-full cascade pattern.
        from .core.disk_health import DiskHealthMonitor
        from .bus import (
            DISK_WARNING, DISK_CRITICAL, DISK_EMERGENCY, DISK_RECOVERED,
        )
        _disk_state_to_msg = {
            "warning": DISK_WARNING,
            "critical": DISK_CRITICAL,
            "emergency": DISK_EMERGENCY,
            "healthy": DISK_RECOVERED,
        }

        def _disk_publish(state, free_bytes):
            self.bus.publish(make_msg(
                _disk_state_to_msg[state.value], "disk_health", "all",
                {"state": state.value, "free_bytes": int(free_bytes)},
            ))

        def _disk_shutdown(reason):
            # Graceful all-worker stop — Guardian's own cleanup path now
            # runs on a worker thread (commit f19a354) so this cannot
            # deadlock the event loop.
            logger.error("[TitanCore] Initiating graceful shutdown: %s", reason)
            try:
                self.guardian.stop_all(reason=reason)
            except Exception as e:
                logger.error("[TitanCore] shutdown stop_all error: %s", e)

        self.disk_health = DiskHealthMonitor(
            path=os.getcwd(),
            publish_fn=_disk_publish,
            shutdown_fn=_disk_shutdown,
        )
        self.disk_health.start()

        # ── Bus Health Monitor ───────────────────────────────────────
        # Tracks META_CGN_SIGNAL emission rates, queue depths, orphan
        # signals. Exposed via /v4/bus-health for session startup check.
        # Wired as module-level singleton so emit_meta_cgn_signal helper
        # can record emissions from any producer context.
        from .core.bus_health import BusHealthMonitor, set_global_monitor

        def _bus_health_publish(msg_type: str, payload: dict):
            try:
                self.bus.publish(make_msg(msg_type, "bus_health", "all", payload))
            except Exception as e:
                logger.debug("[BusHealth] publish error: %s", e)

        self.bus_health = BusHealthMonitor(publish_fn=_bus_health_publish)
        set_global_monitor(self.bus_health)
        logger.info("[BusHealth] monitor wired as global singleton")

        # ── Wallet Resolution & Soul ─────────────────────────────────
        self._limbo_mode = False
        resolved_wallet = self._resolve_wallet(wallet_path)
        if resolved_wallet is None:
            self._limbo_mode = True
            logger.warning("[TitanCore] No keypair — LIMBO MODE")

        # Boot Soul (lightweight — just Ed25519 keys, no network calls)
        if not self._limbo_mode:
            from .core.soul import SovereignSoul
            from .core.network import HybridNetworkClient
            network_cfg = self._full_config.get("network", {})
            self.network = HybridNetworkClient(config=network_cfg)
            self.soul = SovereignSoul(resolved_wallet, self.network, config=network_cfg)
        else:
            self.network = None
            self.soul = None

        # ── Proxy stubs (populated during boot) ──────────────────────
        self._proxies: dict[str, object] = {}

        # ── Agency Module (Step 7) ─────────────────────────────────
        self._agency = None
        self._agency_assessment = None
        self._interface_advisor = None

        # ── V4 Outer Trinity Collector ────────────────────────────────
        self._outer_trinity_collector = None

        # ── Output Verification Gate (security for all external outputs) ──
        self._output_verifier = None
        try:
            from titan_plugin.logic.output_verifier import OutputVerifier
            _tc_dir = os.path.join("data", "timechain")
            _titan_id = self._full_config.get("info_banner", {}).get("titan_id", "T1")
            _wallet_path = self._full_config.get("network", {}).get(
                "wallet_keypair_path", "data/titan_identity_keypair.json")
            self._output_verifier = OutputVerifier(
                titan_id=_titan_id, data_dir=_tc_dir, keypair_path=_wallet_path)
        except Exception as _ovg_err:
            logger.warning("[TitanCore] OutputVerifier init failed: %s", _ovg_err)

        # ── State ────────────────────────────────────────────────────
        self._last_execution_mode = "Shadow"
        self._is_meditating = False
        self._start_time = time.time()
        self._background_tasks_started = False
        self._observatory_app = None
        self._agent = None

        # ── Dream-aware message queue (API process) ──────────────────
        import threading
        self._dream_inbox = []                          # Queued messages during dream
        self._dream_inbox_lock = threading.Lock()       # Thread-safe for concurrent /chat
        self._dream_state = {
            "is_dreaming": False,
            "recovery_pct": 0.0,
            "remaining_epochs": 0,
            "wake_transition": False,
            "just_woke": False,
            "wake_ts": 0.0,
        }

        boot_ms = (time.time() - self._boot_start) * 1000
        logger.info("[TitanCore] Sync init complete in %.0fms", boot_ms)

    # ------------------------------------------------------------------
    # Boot (async)
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """
        Async boot sequence:
          1. Register module specs with Guardian
          2. Start autostart modules
          3. Create & launch Observatory API
          4. Start health monitor loop
        """
        boot_start = time.time()

        # Register supervised modules (lazy — won't start until requested)
        self._register_modules()

        # Start modules marked autostart=True
        self.guardian.start_all()

        # ── Wire bus poll function for synchronous proxy requests ───
        self.bus._poll_fn = self.guardian.drain_send_queues

        # ── Create Proxies ──────────────────────────────────────────
        self._create_proxies()

        # Boot EventBus + Observatory DB + API
        from .api.events import EventBus
        self.event_bus = EventBus()

        # Observatory DB for persistent historical metrics
        from .utils.observatory_db import ObservatoryDB
        self._observatory_db = ObservatoryDB()
        self.event_bus.attach_db(self._observatory_db)

        api_cfg = self._full_config.get("api", {})
        if api_cfg.get("enabled", True):
            self._create_observatory_app(api_cfg)
            asyncio.get_event_loop().create_task(self._start_observatory(api_cfg))

        # Boot Agency Module (Step 7 — autonomous action pipeline)
        self._boot_agency()

        # ── Reflex Collector (Sovereign Tool System) ──────────────
        self._boot_reflex_collector()

        # Guardian health monitor tick (every 5s)
        asyncio.get_event_loop().create_task(self._guardian_loop())

        # Core heartbeat publisher (for frontend/monitoring)
        asyncio.get_event_loop().create_task(self._heartbeat_loop())

        # Trinity tensor snapshot loop (persists to ObservatoryDB for historical charts)
        asyncio.get_event_loop().create_task(self._trinity_snapshot_loop())

        # Agency bus listener (IMPULSE → INTENT → helper execution → ACTION_RESULT)
        if self._agency:
            asyncio.get_event_loop().create_task(self._agency_loop())

        # Meditation cycle (memory consolidation, mempool scoring, Cognee cognify)
        asyncio.get_event_loop().create_task(self._meditation_loop())

        # Sovereign backup (rFP_backup_worker Phase 1 — corrected 2026-04-13):
        # TitanCore IS the production entry point (titan_main.py:204); TitanPlugin
        # is legacy/unused in the microkernel boot path. So backup MUST live here.
        # Single loop, correct RebirthBackup signature, ArweaveStore injected once.
        asyncio.get_event_loop().create_task(self._backup_loop())

        # V4: Outer Trinity collector loop (computes Outer Trinity tensors, publishes to bus)
        self._boot_outer_trinity()
        if self._outer_trinity_collector:
            asyncio.get_event_loop().create_task(self._outer_trinity_loop())

        # V4: DivineBus → EventBus bridge (forwards V4 events to WebSocket clients)
        asyncio.get_event_loop().create_task(self._v4_event_bridge_loop())

        # Social engagement: DISABLED as independent loop.
        # Mentions are now checked as part of the posting flow in spirit_worker
        # (safer: fewer API calls, no bot detection risk, clustered with posts)
        # if hasattr(self, 'social') and self.social:
        #     asyncio.get_event_loop().create_task(self._social_engagement_loop())

        self._background_tasks_started = True
        boot_s = time.time() - boot_start
        logger.info("[TitanCore] Async boot complete in %.2fs | Modules registered: %s",
                     boot_s, list(self.guardian._modules.keys()))

    # ------------------------------------------------------------------
    # Module Registration
    # ------------------------------------------------------------------

    def _register_modules(self) -> None:
        """Register all module specs with the Guardian. None start yet (lazy=True)."""
        from .modules.memory_worker import memory_worker_main
        from .modules.rl_worker import rl_worker_main
        from .modules.llm_worker import llm_worker_main
        from .modules.body_worker import body_worker_main
        from .modules.mind_worker import mind_worker_main
        from .modules.spirit_worker import spirit_worker_main
        from .modules.media_worker import media_worker_main
        from .modules.language_worker import language_worker_main
        from .modules.cgn_worker import cgn_worker_main
        from .modules.knowledge_worker import knowledge_worker_main
        from .modules.timechain_worker import timechain_worker_main

        # Memory module (FAISS + Kuzu + DuckDB)
        memory_config = {
            **self._full_config.get("inference", {}),
            **self._full_config.get("memory_and_storage", {}),
        }
        self.guardian.register(ModuleSpec(
            name="memory",
            entry_fn=memory_worker_main,
            config=memory_config,
            rss_limit_mb=2000,  # FAISS + Kuzu + DuckDB: ~1100-1200MB steady on T1 (40min uptime). VmRSS includes mmap'd DB pages from page cache, which inflates the reading on rapid restarts. Old 1400MB limit was sized for the Cognee era and caused T2 doom-loop.
            autostart=False,
            lazy=True,
            heartbeat_timeout=120.0,  # Memory queries can block for 30s+
            reply_only=True,  # Memory only needs targeted QUERY messages, not broadcasts
        ))

        # RL/Sage module (TorchRL — ~2500MB with mmap)
        self.guardian.register(ModuleSpec(
            name="rl",
            entry_fn=rl_worker_main,
            config=self._full_config.get("stealth_sage", {}),
            rss_limit_mb=3000,
            autostart=False,
            lazy=True,
        ))

        # LLM/Inference module (Agno agent — ~500MB)
        self.guardian.register(ModuleSpec(
            name="llm",
            entry_fn=llm_worker_main,
            config=self._full_config.get("inference", {}),
            rss_limit_mb=1000,
            autostart=True,  # Changed: Language Teacher needs llm at boot
            lazy=False,
            heartbeat_timeout=120.0,  # LLM calls can block 30s+; match Spirit/Memory timeout
        ))

        # Body module (5DT somatic sensors — lightweight, always-on)
        # Note: RSS includes inherited parent process memory from fork (~250MB),
        # so limit must account for that baseline.
        body_config = {
            **self._full_config.get("body", {}),
            "api_port": int(self._full_config.get("api", {}).get("port", 7777)),
        }
        self.guardian.register(ModuleSpec(
            name="body",
            entry_fn=body_worker_main,
            config=body_config,
            rss_limit_mb=500,
            autostart=True,  # Body senses must always be active
            lazy=False,
        ))

        # Mind module (MoodEngine, SocialGraph — ~200MB)
        mind_config = {
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
        }
        self.guardian.register(ModuleSpec(
            name="mind",
            entry_fn=mind_worker_main,
            config=mind_config,
            rss_limit_mb=500,
            autostart=True,  # Mind senses should always be active
            lazy=False,
        ))

        # Spirit module (Consciousness + V4 Sphere Clocks + Enrichment + Neural NS + Experience Orchestrator)
        spirit_config = {
            **self._full_config.get("consciousness", {}),
            "sphere_clock": self._full_config.get("sphere_clock", {}),
            "spirit_enrichment": self._full_config.get("spirit_enrichment", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
            "social_presence": self._full_config.get("social_presence", {}),
        }
        self.guardian.register(ModuleSpec(
            name="spirit",
            entry_fn=spirit_worker_main,
            config=spirit_config,
            rss_limit_mb=750,
            autostart=True,  # Spirit awareness should always be active
            lazy=False,
            heartbeat_timeout=120.0,  # Spirit does heavy V4 work (LLM, on-chain)
        ))

        # Media module (image/audio perception — lazy, starts on first media)
        media_config = {
            "queue_dir": os.path.join(
                os.path.dirname(__file__), "..", "data", "media_queue"
            ),
        }
        self.guardian.register(ModuleSpec(
            name="media",
            entry_fn=media_worker_main,
            config=media_config,
            rss_limit_mb=700,  # was 500; spikes to ~550MB during spatial/audio perception
            autostart=True,   # Always on — art/audio generated frequently
            lazy=False,
            heartbeat_timeout=180.0,  # Image/audio digest can block 30-90s
        ))

        # Language module (composition, teaching, vocabulary — higher cognitive)
        language_config = {
            **self._full_config.get("language", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
        }
        self.guardian.register(ModuleSpec(
            name="language",
            entry_fn=language_worker_main,
            config=language_config,
            # TEMPORARY WORKAROUND 2026-04-08: was 500MB, raised to 700MB to
            # unblock T2/T3 boot which was hitting 516-530MB during init.
            # Real fix needed — investigate WHY language worker grew past
            # 500MB (vocab/embedding cache growth? leak? inefficient load?).
            # See deferred task list in session log. REVERT after root-cause fix.
            rss_limit_mb=700,
            autostart=True,   # Language must be ready for SPEAK_REQUEST
            lazy=False,
            heartbeat_timeout=120.0,  # Teacher LLM calls can take 30s+
        ))

        # CGN Cognitive Kernel (shared V(s) + per-consumer Q(s,a) + HAOV + Sigma)
        cgn_config = {
            "state_dir": "data/cgn",
            "db_path": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data") + "/inner_memory.db",
            "shm_path": "/dev/shm/cgn_live_weights.bin",
            "online_consolidation_every": 50,
            "shm_write_on_every_outcome": True,
            **self._full_config.get("cgn", {}),
        }
        self.guardian.register(ModuleSpec(
            name="cgn",
            entry_fn=cgn_worker_main,
            config=cgn_config,
            rss_limit_mb=800,    # CGN loads PyTorch state + 500 transitions.
                                 # 2026-04-08 (later): raised 600→800 after T3
                                 # crash-loop incident — T3 cgn was hitting 700-
                                 # 716MB during HAOV verification spikes,
                                 # crashing 6+ times in 1h. T1 steady-state is
                                 # ~360MB so plenty of margin.
            autostart=True,    # Must be ready before language/spirit need it
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=False,  # Receives broadcasts (CGN_CONSOLIDATE from spirit)
        ))

        # Knowledge Worker (4th CGN consumer — knowledge acquisition + Stealth Sage)
        _data_dir = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
        knowledge_config = {
            "db_path": _data_dir + "/inner_memory.db",
            "cgn_state_dir": "data/cgn",
            "shm_path": "/dev/shm/cgn_live_weights.bin",
            # SearXNG + Stealth Sage
            **self._full_config.get("stealth_sage", {}),
            # Inference (for LLM distillation)
            **{k: v for k, v in self._full_config.get("inference", {}).items()
               if k.startswith("ollama_cloud") or k == "inference_provider"},
            # Twitter API (for X research path)
            "twitterapi_io_key": self._full_config.get(
                "twitter_social", {}).get("twitterapi_io_key", ""),
        }
        self.guardian.register(ModuleSpec(
            name="knowledge",
            entry_fn=knowledge_worker_main,
            config=knowledge_config,
            rss_limit_mb=800,    # CGNConsumerClient (PyTorch) + Sage + httpx.
                                 # 2026-04-08 (later): raised 600→800 same as
                                 # cgn. T3 knowledge was hitting 708-716MB
                                 # during research distillation cycles.
            autostart=True,      # Must be ready for knowledge requests
            lazy=False,
            heartbeat_timeout=180.0,  # Research takes 10-45s + queue wait time
            reply_only=False,    # Receives CGN_KNOWLEDGE_REQ broadcasts
        ))

        # TimeChain — Proof of Thought memory chain
        timechain_config = {
            **self._full_config.get("timechain", {}),
        }
        self.guardian.register(ModuleSpec(
            name="timechain",
            entry_fn=timechain_worker_main,
            config=timechain_config,
            rss_limit_mb=700,     # Needs headroom for integrity check + chain queries
            autostart=True,       # Must be ready before other modules emit thoughts
            lazy=False,
            heartbeat_timeout=120.0,  # Extended: integrity check + healing takes ~60s
            reply_only=False,     # Receives EPOCH_TICK, TIMECHAIN_COMMIT, etc.
        ))

        logger.info("[TitanCore] Registered %d supervised modules", len(self.guardian._modules))

    def _create_proxies(self) -> None:
        """Create proxy objects that bridge V2 API calls to V3 bus-supervised modules."""
        from .proxies.memory_proxy import MemoryProxy
        from .proxies.rl_proxy import RLProxy
        from .proxies.llm_proxy import LLMProxy
        from .proxies.mind_proxy import MindProxy
        from .proxies.body_proxy import BodyProxy
        from .proxies.spirit_proxy import SpiritProxy
        from .proxies.media_proxy import MediaProxy
        from .proxies.timechain_proxy import TimechainProxy

        # Lazy modules — start on first use
        self._proxies["memory"] = MemoryProxy(self.bus, self.guardian)
        self._proxies["rl"] = RLProxy(self.bus, self.guardian)
        self._proxies["llm"] = LLMProxy(self.bus, self.guardian)

        # Always-on modules — already started by Guardian
        self._proxies["mind"] = MindProxy(self.bus, self.guardian)
        self._proxies["body"] = BodyProxy(self.bus, self.guardian)
        self._proxies["spirit"] = SpiritProxy(self.bus, self.guardian)

        # Media module (lazy — starts on first use)
        self._proxies["media"] = MediaProxy(self.bus, self.guardian)

        # TimeChain v2 Consumer API proxy
        self._proxies["timechain"] = TimechainProxy(self.bus, self.guardian)

        # V2-compatible aliases (so dashboard/agent code finds what it expects)
        self._proxies["mood_engine"] = self._proxies["mind"]  # mind proxy has get_mood_label()
        self._proxies["gatekeeper"] = self._proxies["rl"]     # rl proxy has evaluate()
        self._proxies["social_graph"] = self._proxies["mind"] # mind proxy has record_interaction()

        # ── V2 Subsystems (direct instances in Core) ──────────────────
        self._wire_metabolism()
        self._wire_studio()
        self._wire_social()

        logger.info("[TitanCore] Created %d proxies", len(self._proxies))

    def _wire_metabolism(self) -> None:
        """Wire MetabolismController directly in Core (lightweight — SOL balance + growth metrics)."""
        try:
            from .core.metabolism import MetabolismController
            growth_cfg = self._full_config.get("growth_metrics", {})
            metabolism = MetabolismController(
                soul=self.soul,
                network=self.network,
                memory=self._proxies.get("memory"),  # MemoryProxy
                config=growth_cfg,
                social_graph=None,  # Will use memory worker for social density
            )
            # Monkey-patch growth methods to use bus-routed computation
            # (MetabolismController normally iterates _node_store directly, but in V3
            #  _node_store lives in the memory worker process)
            memory_proxy = self._proxies.get("memory")
            node_sat = growth_cfg.get("node_saturation_24h", 30)

            async def _v3_learning_velocity():
                if not memory_proxy:
                    return 0.5
                metrics = memory_proxy.get_growth_metrics(node_saturation_24h=node_sat)
                return metrics.get("learning_velocity", 0.5)

            async def _v3_directive_alignment():
                if not memory_proxy:
                    return 0.5
                metrics = memory_proxy.get_growth_metrics(node_saturation_24h=node_sat)
                return metrics.get("directive_alignment", 0.5)

            async def _v3_social_density():
                # Simplified: use growth metrics from memory worker
                if not memory_proxy:
                    return 0.5
                status = memory_proxy.get_memory_status()
                persistent = status.get("persistent_count", 0)
                # Rough social density estimate from persistent memory count
                return min(1.0, persistent / 100.0)

            metabolism.get_learning_velocity = _v3_learning_velocity
            metabolism.get_directive_alignment = _v3_directive_alignment
            metabolism.get_social_density = _v3_social_density

            self._proxies["metabolism"] = metabolism
            logger.info("[TitanCore] MetabolismController wired (SOL balance + growth metrics)")
        except Exception as e:
            logger.warning("[TitanCore] MetabolismController wiring failed: %s", e)

    def _wire_studio(self) -> None:
        """Wire StudioCoordinator directly in Core (lightweight creative engine)."""
        try:
            from .expressive.studio import StudioCoordinator
            exp_cfg = self._full_config.get("expressive", {})
            studio = StudioCoordinator(
                config=exp_cfg,
                metabolism=self._proxies.get("metabolism"),
            )
            # Wire Ollama Cloud for haiku generation
            inference_cfg = self._full_config.get("inference", {})
            ollama_key = inference_cfg.get("ollama_cloud_api_key", "")
            if ollama_key:
                try:
                    from .utils.ollama_cloud import OllamaCloudClient
                    studio._ollama_cloud = OllamaCloudClient(
                        api_key=ollama_key,
                        base_url=inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1"),
                    )
                except Exception:
                    pass
            self._proxies["studio"] = studio
            logger.info("[TitanCore] StudioCoordinator wired (%s)", exp_cfg.get("output_path", "./data/studio_exports"))
        except Exception as e:
            logger.warning("[TitanCore] StudioCoordinator wiring failed: %s", e)

    def _wire_social(self) -> None:
        """Wire SocialManager in Core (degraded mode — no API keys, but structure in place)."""
        try:
            from .expressive.social import SocialManager
            sage_cfg = self._full_config.get("stealth_sage", {})
            # Wire SocialGraph for persistent user profile tracking
            social_graph = None
            try:
                from .core.social_graph import SocialGraph
                data_dir = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
                social_graph = SocialGraph(db_path=os.path.join(data_dir, "social_graph.db"))
            except Exception as _sg_err:
                logger.debug("[TitanCore] SocialGraph init: %s", _sg_err)
            social = SocialManager(
                metabolism_client=self._proxies.get("metabolism"),
                mood_engine=self._proxies.get("mood_engine"),
                recorder=None,
                memory=self._proxies.get("memory"),
                stealth_sage_config=sage_cfg,
                social_graph=social_graph,
            )
            # Wire dry-run mode from endurance config
            endurance_cfg = self._full_config.get("endurance", {})
            if endurance_cfg.get("social_dry_run", True):
                social._dry_run = True
                social._dry_run_log = endurance_cfg.get(
                    "social_dry_run_log", "./data/logs/social_dry_run.log"
                )
            self._proxies["social"] = social
            logger.info("[TitanCore] SocialManager wired (dry_run=%s)", social._dry_run)
        except Exception as e:
            logger.warning("[TitanCore] SocialManager wiring failed: %s", e)

    # ------------------------------------------------------------------
    # V2 Compatibility — Facade Properties
    # ------------------------------------------------------------------
    # These allow existing Observatory API and agent code to work with
    # TitanCore as if it were TitanPlugin. Returns None when module not
    # loaded, so endpoints can degrade gracefully.

    @property
    def memory(self):
        """Lazy access — returns proxy or None."""
        return self._proxies.get("memory")

    @property
    def metabolism(self):
        return self._proxies.get("metabolism")

    @property
    def mood_engine(self):
        return self._proxies.get("mood_engine")

    @property
    def recorder(self):
        return self._proxies.get("recorder")

    @property
    def gatekeeper(self):
        return self._proxies.get("gatekeeper")

    @property
    def scholar(self):
        return self._proxies.get("scholar")

    @property
    def consciousness(self):
        return self._proxies.get("consciousness")

    @property
    def social_graph(self):
        return self._proxies.get("social_graph")

    @property
    def social(self):
        return self._proxies.get("social")

    @property
    def studio(self):
        return self._proxies.get("studio")

    @property
    def maker_engine(self):
        return self._proxies.get("maker_engine")

    @property
    def sage_researcher(self):
        return self._proxies.get("sage_researcher")

    # ------------------------------------------------------------------
    # Observatory API (reuses existing V2 create_app)
    # ------------------------------------------------------------------

    def _create_observatory_app(self, api_cfg: dict):
        """Create the Observatory FastAPI app synchronously."""
        try:
            from .api import create_app
            app = create_app(self, self.event_bus, api_cfg)
            self._observatory_app = app
            return app
        except Exception as e:
            logger.warning("[TitanCore] Observatory app creation failed: %s", e)
            return None

    async def _start_observatory(self, api_cfg: dict):
        """Launch the Observatory API server."""
        try:
            import uvicorn
            app = self._observatory_app
            if app is None:
                return
            host = api_cfg.get("host", "0.0.0.0")
            port = int(api_cfg.get("port", 7777))
            uvi_config = uvicorn.Config(
                app=app, host=host, port=port, log_level="info", access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(uvi_config)
            await self._uvicorn_server.serve()
        except SystemExit:
            logger.warning("[TitanCore] Observatory could not bind port")
        except Exception as e:
            logger.warning("[TitanCore] Observatory failed: %s", e)

    def reload_api(self) -> dict:
        """Hot-reload API routes by rebuilding the FastAPI app and swapping it.

        Returns dict with reload status. The uvicorn server keeps running —
        only the ASGI app reference changes. Zero downtime.
        """
        try:
            from .api import reload_api_app
            old_app = self._observatory_app
            new_app = reload_api_app(old_app)
            self._observatory_app = new_app

            # Swap the app in the running uvicorn server
            if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
                self._uvicorn_server.config.app = new_app
                # Also update the loaded_app which uvicorn uses for serving
                if hasattr(self._uvicorn_server, 'config'):
                    self._uvicorn_server.config.loaded_app = new_app

            logger.info("[TitanCore] API hot-reloaded — routes updated, server continuous")
            return {"status": "ok", "reloaded": True}
        except Exception as e:
            logger.error("[TitanCore] API reload failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Agency Module (Step 7)
    # ------------------------------------------------------------------

    def _boot_reflex_collector(self) -> None:
        """Initialize the Sovereign Reflex Collector with executors."""
        try:
            from titan_plugin.logic.reflexes import ReflexCollector
            from titan_plugin.logic.reflex_executors import register_reflex_executors
            from titan_plugin.params import get_params

            reflex_cfg = get_params("reflexes")
            self.reflex_collector = ReflexCollector(reflex_cfg)

            # Register executors — they wrap existing subsystems
            count = register_reflex_executors(self.reflex_collector, self)
            logger.info("[TitanCore] ReflexCollector booted: %d executors, threshold=%.2f",
                        count, self.reflex_collector.fire_threshold)
        except Exception as e:
            logger.warning("[TitanCore] ReflexCollector boot failed: %s", e)
            self.reflex_collector = None

    def _boot_agency(self) -> None:
        """Initialize Agency Module with helper registry and assessment."""
        try:
            from .logic.agency.registry import HelperRegistry
            from .logic.agency.module import AgencyModule
            from .logic.agency.assessment import SelfAssessment
            from .logic.interface_advisor import InterfaceAdvisor

            agency_cfg = self._full_config.get("agency", {})
            if not agency_cfg.get("enabled", True):
                logger.info("[TitanCore] Agency disabled by config")
                return

            # Create registry and register helpers
            registry = HelperRegistry()
            self._register_helpers(registry)

            # LLM function for Agency (uses Venice/OllamaCloud via inference config)
            llm_fn = self._create_agency_llm_fn()

            budget = int(agency_cfg.get("llm_budget_per_hour", 10))
            self._agency = AgencyModule(registry=registry, llm_fn=llm_fn, budget_per_hour=budget)
            self._agency_assessment = SelfAssessment(llm_fn=llm_fn)
            self._interface_advisor = InterfaceAdvisor()

            # Expression Translation Layer — learned action selection
            try:
                from titan_plugin.logic.expression_translator import (
                    ExpressionTranslator, FeedbackRouter)
                self._expression_translator = ExpressionTranslator(
                    all_helpers=registry.list_helper_names()
                    if hasattr(registry, 'list_helper_names') else [])
                self._expression_translator.load(
                    "./data/neural_nervous_system/expression_state.json")
                self._feedback_router = FeedbackRouter(
                    hormonal_system=None,  # Wired later when neural_ns available
                    translator=self._expression_translator)
                logger.info("[TitanCore] ExpressionTranslator booted "
                            "(sovereignty=%.1f%%)",
                            self._expression_translator.sovereignty_ratio * 100)
            except Exception as e:
                logger.warning("[TitanCore] Expression layer init error: %s", e)
                self._expression_translator = None
                self._feedback_router = None

            # Subscribe agency to bus
            self._agency_queue = self.bus.subscribe("agency")

            helper_names = registry.list_all_names()
            statuses = registry.get_all_statuses()
            available = [n for n, s in statuses.items() if s == "available"]
            logger.info("[TitanCore] Agency booted: %d helpers registered (%d available): %s",
                        len(helper_names), len(available), available)

        except Exception as e:
            logger.warning("[TitanCore] Agency boot failed: %s", e)
            self._agency = None

    def _register_helpers(self, registry) -> None:
        """Register all available helpers in the registry."""
        try:
            from .logic.agency.helpers.infra_inspect import InfraInspectHelper
            registry.register(InfraInspectHelper(log_path="/tmp/titan_v3.log"))
        except Exception as e:
            logger.warning("[TitanCore] InfraInspect helper failed: %s", e)

        try:
            from .logic.agency.helpers.web_search import WebSearchHelper
            sage_cfg = self._full_config.get("stealth_sage", {})
            searxng_host = sage_cfg.get("searxng_host", "http://localhost:8080")
            firecrawl_key = sage_cfg.get("firecrawl_api_key", "")
            registry.register(WebSearchHelper(
                searxng_url=searxng_host,
                firecrawl_api_key=firecrawl_key,
            ))
        except Exception as e:
            logger.warning("[TitanCore] WebSearch helper failed: %s", e)

        # SocialPostHelper REMOVED — all posting goes through SocialPressureMeter
        # (social_narrator + quality gate + rate limits + 11 post types).
        # Agency selecting social_post bypassed our designed narrator entirely.

        try:
            from .logic.agency.helpers.art_generate import ArtGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            registry.register(ArtGenerateHelper(output_dir=output_dir))
        except Exception as e:
            logger.warning("[TitanCore] ArtGenerate helper failed: %s", e)

        try:
            from .logic.agency.helpers.audio_generate import AudioGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            audio_cfg = self._full_config.get("audio", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            max_duration = int(audio_cfg.get("max_duration_seconds", 30))
            sample_rate = int(audio_cfg.get("sample_rate", 44100))
            registry.register(AudioGenerateHelper(
                output_dir=output_dir,
                max_duration=max_duration,
                sample_rate=sample_rate,
            ))
        except Exception as e:
            logger.warning("[TitanCore] AudioGenerate helper failed: %s", e)

        try:
            from .logic.agency.helpers.coding_sandbox import CodingSandboxHelper
            registry.register(CodingSandboxHelper())
        except Exception as e:
            logger.warning("[TitanCore] CodingSandbox helper failed: %s", e)

        try:
            from .logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
            registry.register(CodeKnowledgeHelper())
        except Exception as e:
            logger.warning("[TitanCore] CodeKnowledge helper failed: %s", e)

        try:
            from .logic.agency.helpers.memo_inscribe import MemoInscribeHelper
            # MemoInscribeHelper reads config.toml directly for RPC + keypair
            registry.register(MemoInscribeHelper())
        except Exception as e:
            logger.warning("[TitanCore] MemoInscribe helper failed: %s", e)

        # Kin Discovery — consciousness-to-consciousness exchange
        try:
            from .logic.agency.helpers.kin_sense import KinSenseHelper
            import tomllib as _tomllib_kin
            _kin_params = {}
            _kin_params_path = os.path.join(os.path.dirname(__file__), "titan_params.toml")
            if os.path.exists(_kin_params_path):
                with open(_kin_params_path, "rb") as _kf:
                    _kin_params = _tomllib_kin.load(_kf)
            _kin_cfg = _kin_params.get("kin", {})
            if _kin_cfg.get("enabled", False):
                # TITAN_KIN_ADDRESSES env var overrides config (for T2 pointing to T1 via nginx)
                _kin_addrs = _kin_cfg.get("addresses", [])
                _env_addrs = os.environ.get("TITAN_KIN_ADDRESSES", "")
                if _env_addrs:
                    _kin_addrs = [a.strip() for a in _env_addrs.split(",") if a.strip()]
                registry.register(KinSenseHelper(
                    kin_addresses=_kin_addrs,
                    exchange_strength=_kin_cfg.get("exchange_strength", 0.03),
                ))
                logger.info("[TitanCore] KinSense helper registered: addresses=%s",
                            _kin_addrs)
        except Exception as e:
            logger.warning("[TitanCore] KinSense helper failed: %s", e)

    def _create_agency_llm_fn(self):
        """Create a lightweight async LLM function for Agency module."""
        inference_cfg = self._full_config.get("inference", {})

        async def agency_llm(prompt: str, task: str = "agency_select") -> str:
            """LLM call for helper selection / assessment / code generation."""
            try:
                from .utils.ollama_cloud import OllamaCloudClient, get_model_for_task
                client = OllamaCloudClient(
                    api_key=inference_cfg.get("ollama_cloud_api_key", ""),
                    base_url=inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1"),
                )
                model = get_model_for_task(task)
                max_tok = 800 if task == "agency_code_gen" else 200
                return await client.complete(prompt, model=model, max_tokens=max_tok)
            except Exception as e:
                logger.warning("[Agency LLM] OllamaCloud failed: %s — trying Venice", e)

            try:
                import httpx
                venice_key = inference_cfg.get("venice_api_key", "")
                if venice_key:
                    async with httpx.AsyncClient(timeout=15.0) as http:
                        resp = await http.post(
                            "https://api.venice.ai/api/v1/chat/completions",
                            json={
                                "model": "llama-3.3-70b",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 200,
                            },
                            headers={"Authorization": f"Bearer {venice_key}"},
                        )
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning("[Agency LLM] Venice failed: %s", e)

            raise RuntimeError("No LLM available for agency")

        return agency_llm

    async def _agency_loop(self) -> None:
        """
        Listen for IMPULSE events on the bus and process them through Agency.

        Flow: IMPULSE → InterfaceAdvisor rate check → Agency handles → Assessment → ACTION_RESULT
        """
        logger.info("[TitanCore] Agency loop started — listening for IMPULSE events")
        while True:
            try:
                # Poll bus for IMPULSE messages addressed to agency or broadcast
                msg = None
                try:
                    msg = self._agency_queue.get_nowait()
                except Exception:
                    pass

                if not msg:
                    await asyncio.sleep(2.0)  # Poll every 2s (impulses are rare)
                    continue

                msg_type = msg.get("type", "")

                if msg_type == IMPULSE:
                    await self._handle_impulse(msg)
                elif msg_type == "OUTER_DISPATCH":
                    await self._handle_outer_dispatch(msg)
                elif msg_type == "QUERY":
                    self._handle_agency_query(msg)

            except Exception as e:
                logger.error("[TitanCore] Agency loop error: %s", e)
                await asyncio.sleep(5.0)

    async def _handle_impulse(self, msg: dict) -> None:
        """Process an IMPULSE event through the agency pipeline."""
        payload = msg.get("payload", {})
        posture = payload.get("posture", "unknown")
        impulse_id = payload.get("impulse_id", 0)

        logger.info("[TitanCore] IMPULSE received: #%d posture=%s urgency=%.2f",
                    impulse_id, posture, payload.get("urgency", 0))

        # Rate check via InterfaceAdvisor
        if self._interface_advisor:
            feedback = self._interface_advisor.check(IMPULSE)
            if feedback:
                logger.info("[TitanCore] IMPULSE rate-limited: %s", feedback.get("message", ""))
                self.bus.publish(make_msg("RATE_LIMIT", "core", "spirit", feedback))
                return

        # Convert IMPULSE to INTENT (enriched with context)
        intent = {
            **payload,
            "trinity_snapshot": payload.get("trinity_snapshot", {}),
        }

        # Expression Translation Layer — try learned mapping first
        learned_selection = None
        if self._expression_translator:
            try:
                available = self._agency._registry.list_helper_names() \
                    if hasattr(self._agency._registry, 'list_helper_names') else []
                learned_selection = self._expression_translator.translate(
                    program=payload.get("triggering_program", ""),
                    intensity=payload.get("intensity", 0.0),
                    posture=posture,
                    available_helpers=available,
                    trinity_snapshot=payload.get("trinity_snapshot"),
                )
                if learned_selection:
                    # Inject learned selection — Agency will use it directly
                    intent["_learned_selection"] = learned_selection
                    self._expression_translator.record_action_type(was_learned=True)
                    logger.info("[TitanCore] Expression: learned %s→%s (conf=%.2f)",
                                payload.get("triggering_program", "?"),
                                learned_selection["helper"],
                                learned_selection.get("confidence", 0))
                else:
                    self._expression_translator.record_action_type(was_learned=False)
            except Exception as e:
                logger.warning("[TitanCore] Expression translator error: %s", e)

        # Agency Module handles the intent
        result = await self._agency.handle_intent(intent)
        if not result:
            logger.info("[TitanCore] Agency skipped impulse #%d (no action taken)", impulse_id)
            return

        # Self-assessment
        if self._agency_assessment and result.get("success") is not None:
            try:
                assessment = await self._agency_assessment.assess(result)
                result["assessment"] = {
                    "score": assessment["score"],
                    "reflection": assessment["reflection"],
                    "enrichment": assessment["enrichment"],
                    "mood_delta": assessment["mood_delta"],
                    "threshold_direction": assessment["threshold_direction"],
                }
                logger.info("[TitanCore] Assessment: score=%.2f direction=%s — %s",
                           assessment["score"], assessment["threshold_direction"],
                           assessment["reflection"][:80])
            except Exception as e:
                logger.warning("[TitanCore] Assessment failed: %s", e)

        # Publish ACTION_RESULT back to bus (Spirit will pick it up)
        self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
        logger.info("[TitanCore] ACTION_RESULT published: helper=%s success=%s",
                    result.get("helper"), result.get("success"))

        # Feed action result to spirit_worker for OBSERVATION (closed loop)
        try:
            self.bus.publish(make_msg(
                OUTER_OBSERVATION, "core", "spirit", {
                    "action_type": result.get("helper", ""),
                    "result": result,
                    "source": "impulse",
                }))
        except Exception as e:
            logger.warning("[TitanCore] OUTER_OBSERVATION publish error: %s", e)

        # Record in Inner Memory (Phase M: action chain + event markers)
        try:
            if not hasattr(self, '_inner_memory'):
                from titan_plugin.logic.inner_memory import InnerMemoryStore
                self._inner_memory = InnerMemoryStore("./data/inner_memory.db")
            mem = self._inner_memory
            if mem:
                helper_name = result.get("helper", "")
                mem.record_action_chain(
                    impulse_id=result.get("impulse_id", 0),
                    triggering_program=result.get("triggering_program", ""),
                    posture=result.get("posture", ""),
                    helper=helper_name,
                    success=result.get("success", False),
                    score=result.get("score", 0.0),
                    reasoning=result.get("reflection", ""),
                    trinity_before=result.get("trinity_snapshot"),
                )
                # Record event markers for temporal tracking
                if helper_name == "web_search":
                    mem.record_event("explore", program="CURIOSITY")
                elif helper_name == "social_post":
                    mem.record_event("social", program="EMPATHY")
                elif helper_name in ("art_generate", "audio_generate"):
                    mem.record_event("create", program="CREATIVITY")
                    _work_type = "art" if helper_name == "art_generate" else "audio"
                    _file_path = result.get("file_path", "")
                    mem.record_creative_work(
                        work_type=_work_type,
                        file_path=_file_path,
                        triggering_program=result.get("triggering_program", ""),
                        posture=result.get("posture", ""),
                        assessment_score=result.get("score", 0.0),
                    )
                    # Archive to ObservatoryDB for gallery/feed
                    obs_db = getattr(self, "_observatory_db", None)
                    if obs_db and _file_path:
                        _style = result.get("art_style", _work_type)
                        obs_db.record_expressive(
                            type_=_work_type,
                            title=f"{_style.replace('_', ' ').title()} ({result.get('triggering_program', 'autonomous')})",
                            content=result.get("result", ""),
                            media_path=_file_path,
                            media_hash="",
                            metadata={
                                "triggering_program": result.get("triggering_program", ""),
                                "posture": result.get("posture", ""),
                                "score": result.get("score", 0.0),
                            },
                        )
                elif helper_name == "infra_inspect":
                    mem.record_event("inspect", program="VIGILANCE")
                elif helper_name == "kin_sense":
                    mem.record_event("kin", program="EMPATHY")
        except Exception as e:
            logger.warning("[TitanCore] Inner memory recording error: %s", e)

        # Expression Layer: route feedback + save state
        try:
            if self._feedback_router:
                self._feedback_router.route(result)
            if self._expression_translator:
                self._expression_translator.save(
                    "./data/neural_nervous_system/expression_state.json")
        except Exception as e:
            logger.warning("[TitanCore] Expression feedback error: %s", e)

    async def _handle_outer_dispatch(self, msg: dict) -> None:
        """
        Handle OUTER_DISPATCH from two sources:
        1. Neural NS program fires (system=CREATIVITY/IMPULSE/etc.)
        2. Self-exploration expression fires (system=ART/MUSIC/SOCIAL/SPEAK)

        Both use autonomy-first path: no LLM calls, no budget consumed.
        Source distinguished by payload["source"]: "neural_ns" (default) or "self_exploration".
        """
        payload = msg.get("payload", {})
        signals = payload.get("signals", [])
        if not signals:
            return

        _dispatch_source = payload.get("source", "neural_ns")
        logger.info("[TitanCore] OUTER_DISPATCH: %d signals from %s",
                    len(signals), _dispatch_source)

        # Get Trinity snapshot for context
        trinity_snapshot = {}
        try:
            body_proxy = self._proxies.get("body")
            if body_proxy:
                snap = body_proxy.get_tensor_snapshot()
                trinity_snapshot = snap if isinstance(snap, dict) else {}
        except Exception:
            pass

        # Dispatch via Agency (autonomy-first — no LLM calls, no budget consumed)
        results = await self._agency.dispatch_from_nervous_signals(
            outer_signals=signals,
            trinity_snapshot=trinity_snapshot,
        )

        # Publish results and assess outcomes
        # Self-exploration results are observed by OuterInterface (sensory decoder +
        # vocabulary reinforcement), so skip heavy LLM assessment to avoid double-enrichment.
        # Neural NS results get full agency assessment (enrichment routing, mood delta).
        for result in results:
            result["dispatch_source"] = _dispatch_source
            if (self._agency_assessment and result.get("success") is not None
                    and _dispatch_source != "self_exploration"):
                try:
                    assessment = await self._agency_assessment.assess(result)
                    result["assessment"] = {
                        "score": assessment["score"],
                        "reflection": assessment["reflection"],
                        "enrichment": assessment["enrichment"],
                        "mood_delta": assessment["mood_delta"],
                        "threshold_direction": assessment["threshold_direction"],
                    }
                except Exception:
                    pass

            self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
            logger.info("[TitanCore] AUTONOMY ACTION: %s → %s (success=%s)",
                        result.get("posture"), result.get("helper"), result.get("success"))

            # Feed action result to spirit_worker for OBSERVATION (closed loop)
            # Spirit worker's OuterInterface will decode → narrate → apply deltas
            try:
                self.bus.publish(make_msg(
                    OUTER_OBSERVATION, "core", "spirit", {
                        "action_type": result.get("helper", ""),
                        "result": result,
                        "source": payload.get("source", "nervous_system"),
                    }))
            except Exception as e:
                logger.warning("[TitanCore] OUTER_OBSERVATION publish error: %s", e)

    def _handle_agency_query(self, msg: dict) -> None:
        """Handle agency status queries."""
        payload = msg.get("payload", {})
        action = payload.get("action", "")
        rid = msg.get("rid")
        src = msg.get("src", "")

        if action == "get_agency_stats":
            stats = {}
            if self._agency:
                stats["agency"] = self._agency.get_stats()
            if self._agency_assessment:
                stats["assessment"] = self._agency_assessment.get_stats()
            if self._interface_advisor:
                stats["advisor"] = self._interface_advisor.get_stats()
            self.bus.publish(make_msg("RESPONSE", "core", src, stats, rid))

    # ------------------------------------------------------------------
    # Background Loops
    # ------------------------------------------------------------------

    async def _guardian_loop(self) -> None:
        """Periodically call Guardian monitor tick + drain worker send queues.

        CRITICAL: Guardian work is inherently blocking (subprocess joins,
        queue cleanup, SAVE_NOW waits up to 30s, SIGTERM waits up to 15s).
        If these ran directly on the asyncio event loop, any worker-cleanup
        pathology would freeze uvicorn, the bus dispatcher, and every other
        coroutine — exactly the cascade observed 2026-04-14 on T1 when a
        Guardian cleanup deadlocked for 31+ minutes. We therefore offload
        monitor_tick to a worker thread so the event loop remains responsive
        no matter what Guardian is doing. drain_send_queues is fast and
        stays on-loop for lowest latency."""
        while True:
            try:
                await asyncio.to_thread(self.guardian.monitor_tick)
                # Route worker responses back through the bus (non-blocking)
                routed = self.guardian.drain_send_queues()
                if routed > 0:
                    logger.debug("[TitanCore] Routed %d messages from workers", routed)
            except Exception as e:
                logger.error("[TitanCore] Guardian tick error: %s", e)
            await asyncio.sleep(1.0)  # Check every 1s for responsive IPC

    async def _v4_event_bridge_loop(self) -> None:
        """
        Bridge V4 DivineBus events to the EventBus for WebSocket broadcasting.

        Subscribes to DivineBus as 'v4_bridge', drains V4 event types
        (SPHERE_PULSE, BIG_PULSE, GREAT_PULSE), and emits them to WebSocket
        clients via the EventBus.
        """
        from .bus import SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED
        # Observatory V2: additional event types for real-time frontend
        NEUROMOD_UPDATE = "NEUROMOD_UPDATE"
        HORMONE_FIRED = "HORMONE_FIRED"
        EXPRESSION_FIRED = "EXPRESSION_FIRED"
        bridge_queue = self.bus.subscribe("v4_bridge")
        V4_EVENT_TYPES = {SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
                          NEUROMOD_UPDATE, HORMONE_FIRED, EXPRESSION_FIRED}

        # Wait for Spirit to boot
        await asyncio.sleep(10)
        logger.info("[TitanCore] V4 event bridge started")

        while True:
            try:
                msgs = self.bus.drain(bridge_queue, max_msgs=50)
                for msg in msgs:
                    msg_type = msg.get("type", "")
                    if msg_type not in V4_EVENT_TYPES:
                        continue

                    payload = msg.get("payload", {})

                    if msg_type == SPHERE_PULSE:
                        await self.event_bus.emit("sphere_pulse", {
                            "clock": payload.get("clock", ""),
                            "pulse_count": payload.get("pulse_count", 0),
                            "radius": payload.get("radius"),
                            "phase": payload.get("phase"),
                        })
                    elif msg_type == BIG_PULSE:
                        await self.event_bus.emit("big_pulse", {
                            "pair": payload.get("pair", ""),
                            "big_pulse_count": payload.get("big_pulse_count", 0),
                            "consecutive": payload.get("consecutive", 0),
                        })
                    elif msg_type == GREAT_PULSE:
                        await self.event_bus.emit("great_pulse", {
                            "pair": payload.get("pair", ""),
                            "great_pulse_count": payload.get("great_pulse_count", 0),
                        })
                    elif msg_type == DREAM_STATE_CHANGED:
                        _ds_dreaming = payload.get("is_dreaming", False)
                        self._dream_state["is_dreaming"] = _ds_dreaming
                        if _ds_dreaming:
                            self._dream_state["just_woke"] = False
                            self._dream_state["remaining_epochs"] = payload.get(
                                "expected_dream_epochs", 0)
                            self._dream_state["recovery_pct"] = 0.0
                            self._dream_state["wake_transition"] = False
                        else:
                            self._dream_state["just_woke"] = True
                            self._dream_state["wake_ts"] = time.time()
                            self._dream_state["recovery_pct"] = 100.0
                            self._dream_state["remaining_epochs"] = 0
                            self._dream_state["wake_transition"] = False
                        logger.info("[TitanCore] Dream state: is_dreaming=%s",
                                    _ds_dreaming)
                        await self.event_bus.emit("dream_state", {
                            "is_dreaming": _ds_dreaming,
                        })
                    # Observatory V2: neuromod, hormone, expression events
                    elif msg_type == NEUROMOD_UPDATE:
                        await self.event_bus.emit("neuromod_update", payload)
                    elif msg_type == HORMONE_FIRED:
                        await self.event_bus.emit("hormone_fired", payload)
                    elif msg_type == EXPRESSION_FIRED:
                        await self.event_bus.emit("expression_fired", payload)
            except Exception as e:
                logger.warning("[TitanCore] V4 event bridge error: %s", e)
            await asyncio.sleep(2.0)

    async def _heartbeat_loop(self) -> None:
        """Publish Core heartbeat to the bus."""
        while True:
            try:
                import psutil
                proc = psutil.Process()
                rss_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                rss_mb = 0

            self.bus.publish(make_msg(
                MODULE_HEARTBEAT, "core", "guardian",
                {"rss_mb": round(rss_mb, 1), "uptime": round(time.time() - self._start_time, 1)},
            ))
            await asyncio.sleep(10.0)

    async def _trinity_snapshot_loop(self) -> None:
        """
        Periodically snapshot Trinity tensor state to ObservatoryDB.

        Interval is configurable via [frontend] trinity_snapshot_interval (default: 60s).
        Records Body/Mind/Spirit tensors, Middle Path loss, and growth metrics.
        """
        interval = int(self._full_config.get("frontend", {}).get(
            "trinity_snapshot_interval", 60))
        # Wait for modules to come online
        await asyncio.sleep(min(interval, 30))

        while True:
            try:
                obs_db = getattr(self, "_observatory_db", None)
                if obs_db is None:
                    await asyncio.sleep(interval)
                    continue

                # Query Trinity state via proxies
                body_proxy = self._proxies.get("body")
                mind_proxy = self._proxies.get("mind")
                spirit_proxy = self._proxies.get("spirit")

                body_tensor = await asyncio.to_thread(body_proxy.get_body_tensor) if body_proxy else [0.5] * 5
                mind_tensor = await asyncio.to_thread(mind_proxy.get_mind_tensor) if mind_proxy else [0.5] * 5
                spirit_data = await asyncio.to_thread(spirit_proxy.get_trinity) if spirit_proxy else {}

                spirit_tensor = spirit_data.get("spirit_tensor", [0.5] * 5)
                middle_path_loss = spirit_data.get("middle_path_loss", 0.0)
                body_center_dist = spirit_data.get("body_center_dist", 0.0)
                mind_center_dist = spirit_data.get("mind_center_dist", 0.0)

                # Record Trinity snapshot
                obs_db.record_trinity_snapshot(
                    body_tensor=body_tensor,
                    mind_tensor=mind_tensor,
                    spirit_tensor=spirit_tensor,
                    middle_path_loss=middle_path_loss,
                    body_center_dist=body_center_dist,
                    mind_center_dist=mind_center_dist,
                )

                # Record growth metrics alongside
                # Extract from consciousness data if available
                consciousness = spirit_data.get("consciousness", {})
                sv = consciousness.get("state_vector", [])
                # State vector dims: [mood, energy, memory_pressure, social_entropy,
                #                     sovereignty, learning_velocity, social_density,
                #                     curvature, density]
                learning_velocity = sv[5] if len(sv) > 5 else 0.0
                social_density = sv[6] if len(sv) > 6 else 0.0
                # metabolic_health = body scalar from Spirit (average body health)
                metabolic_health = spirit_tensor[3] if len(spirit_tensor) > 3 else 0.5
                # directive_alignment approximated by sovereignty
                directive_alignment = sv[4] if len(sv) > 4 else 0.0

                obs_db.record_growth_snapshot(
                    learning_velocity=learning_velocity,
                    social_density=social_density,
                    metabolic_health=metabolic_health,
                    directive_alignment=directive_alignment,
                )

                # Record V4 Time Awareness snapshot (spirit_data already has it)
                if spirit_data.get("sphere_clock") or spirit_data.get("unified_spirit"):
                    obs_db.record_v4_snapshot(
                        sphere_clocks=spirit_data.get("sphere_clock"),
                        resonance=spirit_data.get("resonance"),
                        unified_spirit=spirit_data.get("unified_spirit"),
                        consciousness=spirit_data.get("consciousness"),
                        impulse_engine=spirit_data.get("impulse_engine"),
                        filter_down=spirit_data.get("filter_down"),
                        middle_path_loss=middle_path_loss,
                    )

                # Record vital snapshot (keeps /status/history populated)
                try:
                    mood_label = mind_proxy.get_mood_label() if mind_proxy else "Unknown"
                    mood_valence = mind_proxy.get_mood_valence() if mind_proxy else 0.5
                    sol_balance = 0.0
                    try:
                        if hasattr(self, 'network') and self.network:
                            sol_balance = await self.network.get_balance()
                    except Exception:
                        sol_balance = getattr(self, "_sol_balance", 0.0)
                    mem_status = self._proxies.get("memory")
                    persistent_count = mem_status.get_persistent_count() if mem_status else 0
                    # Sovereignty = Chi total (V5 metric) — more meaningful than sv[4]
                    chi_total = 0.5
                    try:
                        _coord = await asyncio.to_thread(spirit_proxy.get_coordinator) if spirit_proxy else {}
                        coord_chi = _coord.get("chi", {})
                        chi_total = coord_chi.get("total", 0.5) if isinstance(coord_chi, dict) else 0.5
                        coord_nm = _coord.get("neuromodulators", {})
                        nm_emotion = coord_nm.get("current_emotion", mood_label)
                        nm_conf = coord_nm.get("emotion_confidence", 0.0)
                        if nm_conf > 0.5:
                            mood_label = nm_emotion
                            mood_valence = nm_conf
                    except Exception:
                        pass
                    obs_db.record_vital_snapshot(
                        sovereignty_pct=chi_total * 100,
                        life_force_pct=metabolic_health * 100,
                        sol_balance=sol_balance,
                        energy_state=getattr(self, "_energy_state", "HIGH"),
                        mood_label=mood_label,
                        mood_score=mood_valence,
                        persistent_count=persistent_count,
                        mempool_size=0,
                        epoch_counter=consciousness.get("epoch_id", 0),
                    )
                except Exception:
                    pass  # Non-critical — don't break the main snapshot loop

                logger.debug("[TitanCore] Trinity snapshot recorded: loss=%.4f", middle_path_loss)

            except Exception as e:
                logger.warning("[TitanCore] Trinity snapshot error: %s", e)

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Social Engagement Loop (mention polling, replies, likes)
    # ------------------------------------------------------------------

    async def _social_engagement_loop(self) -> None:
        """Periodic mention polling and engagement via SocialManager.

        Runs every x_mention_poll_interval seconds (default 180s / 3 min).
        Uses SocialManager.monitor_and_engage() which handles:
        - Fetching mentions
        - Scoring relevance
        - Generating contextual replies
        - Liking relevant tweets
        - Respecting daily limits
        """
        social_cfg = self._full_config.get("social_presence", {})
        poll_interval = int(social_cfg.get("x_mention_poll_interval", 180))
        enabled = social_cfg.get("enabled", False)

        if not enabled:
            logger.info("[SocialEngagement] Disabled (social_presence.enabled=false)")
            return

        # Wait for system to stabilize before first poll
        await asyncio.sleep(60)
        logger.info("[SocialEngagement] Started (poll every %ds)", poll_interval)

        while True:
            try:
                result = await self.social.monitor_and_engage()
                # Dispatch engagement as outer observation → trinity deltas + experience
                if result and (result.get("replies", 0) > 0 or result.get("likes", 0) > 0):
                    from .bus import make_msg, OUTER_OBSERVATION
                    self.bus.publish(make_msg(
                        OUTER_OBSERVATION, "core", "spirit", {
                            "action_type": "social_post",
                            "result": {
                                "helper": "social_engagement",
                                "success": True,
                                "result": (f"Engaged with {result['mentions_found']} mentions: "
                                           f"{result['replies']} replies, {result['likes']} likes"),
                                "enrichment_data": {
                                    "mind": [1, 5],  # social_cognition + inner_hearing
                                    "boost": 0.03 * result["replies"] + 0.01 * result["likes"],
                                    "engagement_details": result.get("engagement_details", []),
                                },
                            },
                            "source": "social_engagement",
                        }))
                    logger.info("[SocialEngagement] Dispatched outer observation: %d replies, %d likes",
                                result["replies"], result["likes"])
            except Exception as e:
                logger.warning("[SocialEngagement] Error: %s", e)
            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Meditation Loop (Small Epoch — memory consolidation)
    # ------------------------------------------------------------------

    async def _meditation_loop(self) -> None:
        """
        Dual-trigger meditation: emergent (bus) + fixed timer (fallback).

        Spirit_worker detects emergent conditions (GABA, drain, curvature)
        and sends MEDITATION_REQUEST via bus. Fixed timer guarantees ~4/day.
        Memory_worker executes (has Cognee). Spirit_worker never touches Cognee.

        Interval from [endurance].meditation_interval_override (test mode)
        or [mood_engine].update_interval_seconds (production: 6h).
        """
        endurance_cfg = self._full_config.get("endurance", {})
        default_interval = int(self._full_config.get("mood_engine", {}).get(
            "update_interval_seconds", 21600))
        interval = int(endurance_cfg.get("meditation_interval_override", 0)) or default_interval

        # Use pre-subscribed queue (created in __init__ before Guardian boots modules)
        _meditation_queue = self._meditation_queue

        # Wait for memory module to be ready before first meditation
        await asyncio.sleep(min(interval, 120))

        logger.info("[TitanCore] Meditation loop started (interval=%ds, dual-trigger)", interval)

        epoch_count = 0
        _last_meditation_ts = time.time()

        while True:
            try:
                # ── Dual trigger: emergent bus message OR fixed timer ──
                emergent_request = None

                try:
                    _raw_msg = await asyncio.wait_for(
                        asyncio.to_thread(_meditation_queue.get, timeout=30),
                        timeout=35,
                    )
                    # Only accept MEDITATION_REQUEST messages (ignore stray broadcasts)
                    if isinstance(_raw_msg, dict) and _raw_msg.get("type") == "MEDITATION_REQUEST":
                        emergent_request = _raw_msg
                except (asyncio.TimeoutError, Exception):
                    pass

                # Check if we should fire (emergent arrived OR timer expired)
                timer_expired = (time.time() - _last_meditation_ts) >= interval
                if not emergent_request and not timer_expired:
                    continue  # Keep polling

                trigger_source = "emergent" if emergent_request else "timer"

                # ── Run meditation via memory_worker ──
                memory_proxy = self._proxies.get("memory")
                if memory_proxy is None:
                    await asyncio.sleep(60)
                    continue

                epoch_count += 1
                _last_meditation_ts = time.time()

                emergent_ctx = emergent_request.get("payload", {}) if emergent_request else {}
                logger.info(
                    "[TitanCore] Meditation epoch #%d starting (trigger=%s drain=%.3f GABA=%.3f)...",
                    epoch_count, trigger_source,
                    emergent_ctx.get("drain", 0), emergent_ctx.get("gaba", 0))

                result = memory_proxy.run_meditation()

                promoted = 0
                pruned = 0
                if result.get("success"):
                    promoted = result.get("promoted", 0)
                    pruned = result.get("pruned", 0)
                    logger.info(
                        "[TitanCore] Meditation epoch #%d complete: promoted=%d pruned=%d",
                        epoch_count, promoted, pruned,
                    )

                    # Generate meditation art via Studio if available
                    studio = self._proxies.get("studio")
                    if studio and promoted > 0:
                        try:
                            persistent_count = memory_proxy.get_persistent_count()
                            art_path = await studio.generate_meditation_art(
                                f"MEDITATION_V3_E{epoch_count}",
                                persistent_count,
                                min(10, promoted + 3),
                            )
                            if art_path:
                                logger.info("[TitanCore] Meditation art generated: %s", art_path)
                                # Archive to ObservatoryDB
                                obs_db = getattr(self, "_observatory_db", None)
                                if obs_db:
                                    obs_db.record_expressive(
                                        type_="art",
                                        title=f"Meditation Flow Field (V3 Epoch {epoch_count})",
                                        content=f"{promoted} memories crystallized",
                                        media_path=art_path,
                                        media_hash="",
                                        metadata={"epoch": epoch_count, "promoted": promoted},
                                    )
                        except Exception as e:
                            logger.warning("[TitanCore] Meditation art failed: %s", e)

                    # Publish epoch completion event
                    _epoch_payload = {"epoch": epoch_count, "promoted": promoted, "pruned": pruned}
                    self.bus.publish(make_msg(EPOCH_TICK, "core", "all", _epoch_payload))
                    # Explicit send to timechain (dst=all may not reach subprocess)
                    self.bus.publish(make_msg(EPOCH_TICK, "core", "timechain", _epoch_payload))
                else:
                    logger.warning(
                        "[TitanCore] Meditation epoch #%d failed: %s",
                        epoch_count, result.get("error", "unknown"),
                    )

                # ── Notify spirit_worker of completion ──
                _med_payload = {
                    "epoch": epoch_count,
                    "promoted": promoted,
                    "pruned": pruned,
                    "trigger": trigger_source,
                    "success": result.get("success", False),
                    "ts": time.time(),
                }
                self.bus.publish(make_msg(
                    "MEDITATION_COMPLETE", "core", "spirit",
                    _med_payload,
                ))
                # ── Notify timechain_worker to seal genesis block ──
                self.bus.publish(make_msg(
                    "MEDITATION_COMPLETE", "core", "timechain",
                    _med_payload,
                ))

            except Exception as e:
                logger.error("[TitanCore] Meditation loop error: %s", e)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Sovereign Backup Loop (rFP_backup_worker Phase 1 — 2026-04-13 fix)
    # ------------------------------------------------------------------
    # AUDIT NOTE: Previous TitanCore._backup_loop had BUG-2 (wrong signature:
    # `RebirthBackup(self._full_config)` passed config as first positional arg
    # which is network_client). Previous rFP claimed TitanPlugin.backup_loop
    # was a parallel loop, but in fact titan_main.py:204 only uses TitanCore
    # (TitanPlugin is legacy, unreachable from production boot path). So the
    # real fix is to keep the loop here WITH the correct signature, plus:
    #   - inject ArweaveStore once at boot (BUG-5)
    #   - RebirthBackup receives correct (network, config, titan_id, arweave_store, full_config)
    #   - uses tarball path via backup.py (B0)
    #   - per-Titan manifest via timechain_backup.py (BUG-4)

    async def _backup_loop(self) -> None:
        """Poll for meditation trigger files and run RebirthBackup.

        Triggered by spirit_worker writing data/backup_trigger.json on
        MEDITATION_COMPLETE. Delegates to RebirthBackup for:
        - Daily personality → Arweave (1st meditation of day)
        - Weekly soul package → Arweave (every 7th day Sunday)
        - Vault shadow hash update → Solana (after each backup)
        - Backup hash memo → Solana (anchor after backup)
        - MyDay NFT mint → Solana (every 4th meditation)
        - TimeChain Zstd tarball → Arweave (daily, per B0)
        """
        import json as _json

        trigger_path = os.path.join("data", "backup_trigger.json")

        # Construct ArweaveStore ONCE at boot (rFP BUG-5 fix).
        _arweave_store = None
        try:
            _budget = self._full_config.get("mainnet_budget", {})
            if _budget.get("backup_arweave_enabled", False):
                _net_cfg = self._full_config.get("network", {})
                _net = _net_cfg.get("solana_network", "devnet")
                if _net == "mainnet-beta":
                    _net = "mainnet"
                _kp = _net_cfg.get("wallet_keypair_path", "")
                if _kp:
                    from titan_plugin.utils.arweave_store import ArweaveStore
                    _arweave_store = ArweaveStore(keypair_path=_kp, network=_net)
                    logger.info(
                        "[TitanCore] ArweaveStore wired for backup (network=%s)", _net)
        except Exception as _ae:
            logger.warning("[TitanCore] ArweaveStore init failed: %s", _ae)

        # Initialize RebirthBackup with CORRECT signature (rFP BUG-2 + BUG-5 fix).
        _titan_id = self._full_config.get("info_banner", {}).get("titan_id", "T1")
        _backup = None
        try:
            from titan_plugin.logic.backup import RebirthBackup
            _backup = RebirthBackup(
                network_client=getattr(self, "network", None),
                config=self._full_config.get("memory_and_storage", {}),
                titan_id=_titan_id,
                arweave_store=_arweave_store,
                full_config=self._full_config,
            )
            logger.info(
                "[TitanCore] RebirthBackup initialized — sovereign backup active "
                "(titan_id=%s, arweave=%s)",
                _titan_id, "wired" if _arweave_store is not None else "none")
        except Exception as e:
            logger.error(
                "[TitanCore] RebirthBackup init failed — NO BACKUPS: %s", e, exc_info=True)
            return

        # Boot check (verify last backup age, alert if stale)
        try:
            await _backup.check_on_boot()
        except Exception as e:
            logger.warning("[TitanCore] Backup boot check failed: %s", e)

        while True:
            await asyncio.sleep(30)

            if not os.path.exists(trigger_path):
                continue

            try:
                with open(trigger_path) as _tf:
                    trigger = _json.load(_tf)
                os.remove(trigger_path)

                payload = trigger.get("payload", {})
                med_count = trigger.get("meditation_count", 0)

                logger.info("[TitanCore] Processing backup for meditation #%d...", med_count)
                await _backup.on_meditation_complete(payload)

            except _json.JSONDecodeError as e:
                logger.warning("[TitanCore] Invalid backup trigger: %s", e)
                try:
                    os.remove(trigger_path)
                except OSError:
                    pass
            except Exception as e:
                logger.error("[TitanCore] Backup loop error: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # V4: Outer Trinity Collector
    # ------------------------------------------------------------------

    def _boot_outer_trinity(self) -> None:
        """Initialize the Outer Trinity collector for V4 Time Awareness."""
        try:
            from .logic.outer_trinity import OuterTrinityCollector
            self._outer_trinity_collector = OuterTrinityCollector()
            logger.info("[TitanCore] OuterTrinityCollector booted")
        except Exception as e:
            logger.warning("[TitanCore] OuterTrinityCollector boot failed: %s", e)

    async def _outer_trinity_loop(self) -> None:
        """
        Periodically collect Outer Trinity tensors and publish to bus.

        Spirit worker receives OUTER_TRINITY_STATE and ticks outer sphere clocks.
        Interval from [epochs].outer_trinity_interval (default: 60s).
        """
        from .bus import make_msg, OUTER_TRINITY_STATE

        epochs_cfg = self._full_config.get("epochs", {})
        interval = int(epochs_cfg.get("outer_trinity_interval", 60))

        # Wait for subsystems to come online
        await asyncio.sleep(min(interval, 30))

        logger.info("[TitanCore] Outer Trinity loop started (interval=%ds)", interval)

        while True:
            try:
                if not self._outer_trinity_collector:
                    await asyncio.sleep(interval)
                    continue

                # Gather live sources for the collector. The method is sync
                # and does a 3s-timeout httpx.get to the twin Titan — wrap
                # in to_thread so this async loop doesn't block the event
                # loop during twin polling. Found by async-blocks v2
                # scanner (2026-04-14).
                sources = await asyncio.to_thread(self._gather_outer_trinity_sources)
                result = self._outer_trinity_collector.collect(sources)

                # Publish to bus → Spirit worker receives and ticks outer sphere clocks
                self.bus.publish(make_msg(
                    OUTER_TRINITY_STATE, "core", "spirit", result,
                ))

                logger.debug(
                    "[TitanCore] Outer Trinity published: body=%s mind=%s spirit=%s",
                    [round(v, 2) for v in result["outer_body"]],
                    [round(v, 2) for v in result["outer_mind"]],
                    [round(v, 2) for v in result["outer_spirit"]],
                )

            except Exception as e:
                logger.warning("[TitanCore] Outer Trinity loop error: %s", e)

            await asyncio.sleep(interval)

    def _gather_outer_trinity_sources(self) -> dict:
        """Gather live data sources for OuterTrinityCollector."""
        sources: dict = {
            "uptime_seconds": time.time() - self._start_time,
        }

        # Agency stats
        if self._agency:
            sources["agency_stats"] = self._agency.get_stats()
        if self._agency_assessment:
            sources["assessment_stats"] = self._agency_assessment.get_stats()

        # Helper statuses
        if self._agency and hasattr(self._agency, '_registry'):
            sources["helper_statuses"] = self._agency._registry.get_all_statuses()

        # Bus stats
        sources["bus_stats"] = self.bus.stats

        # Impulse engine stats (via spirit proxy query)
        try:
            spirit_proxy = self._proxies.get("spirit")
            if spirit_proxy:
                trinity_data = spirit_proxy.get_trinity()
                if trinity_data and "impulse_engine" in trinity_data:
                    sources["impulse_stats"] = trinity_data["impulse_engine"]
        except Exception:
            pass

        # Observatory DB
        sources["observatory_db"] = getattr(self, "_observatory_db", None)

        # Memory status
        try:
            memory_proxy = self._proxies.get("memory")
            if memory_proxy:
                sources["memory_status"] = memory_proxy.get_memory_status()
        except Exception:
            pass

        # Soul health
        if self.soul:
            # Keypair exists = 0.9, otherwise degraded
            sources["soul_health"] = 0.9 if not self._limbo_mode else 0.2
        else:
            sources["soul_health"] = 0.2

        # LLM latency (from Ollama Cloud stats if available)
        sources["llm_avg_latency"] = 0.0  # Will be enriched when LLM proxy exposes latency

        # Anchor state (from spirit_worker memo inscriptions)
        try:
            import json as _json
            _anchor_path = os.path.join(os.path.dirname(__file__), "..", "data", "anchor_state.json")
            if os.path.exists(_anchor_path):
                with open(_anchor_path) as _af:
                    sources["anchor_state"] = _json.load(_af)
        except Exception:
            pass

        # Social perception stats (from spirit_worker SOCIAL_PERCEPTION handler)
        try:
            spirit_proxy = self._proxies.get("spirit")
            if spirit_proxy and hasattr(spirit_proxy, '_bus'):
                # Query coordinator for accumulated social stats
                from .bus import make_msg
                _sp_result = spirit_proxy._bus.request(
                    make_msg("QUERY", "core", "spirit",
                             {"action": "get_social_perception_stats"}),
                    timeout=2.0)
                if _sp_result and _sp_result.get("payload"):
                    sources["social_perception_stats"] = _sp_result["payload"]
        except Exception:
            pass

        # Twin awareness: poll the other Titan's state for twin_resonance
        # Real network sensing — Titan feels his twin's presence through VPC.
        # The caller wraps this entire method in `asyncio.to_thread` so the
        # sync httpx.get below does not block the event loop.
        try:
            import httpx
            twin_api = "http://10.135.0.6:7777"  # T2 via VPC
            r = httpx.get(f"{twin_api}/v4/inner-trinity", timeout=3)
            if r.status_code == 200:
                twin_data = r.json().get("data", {})
                twin_nm = twin_data.get("neuromodulators", {})
                twin_mods = twin_nm.get("modulators", {})
                # Twin state summary: emotion + key modulator levels
                sources["twin_state"] = {
                    "reachable": True,
                    "emotion": twin_nm.get("current_emotion", "?"),
                    "DA": twin_mods.get("DA", {}).get("level", 0.5),
                    "NE": twin_mods.get("NE", {}).get("level", 0.5),
                    "GABA": twin_mods.get("GABA", {}).get("level", 0.5),
                    "tick_count": twin_data.get("tick_count", 0),
                }
            else:
                sources["twin_state"] = {"reachable": False}
        except Exception:
            sources["twin_state"] = {"reachable": False}

        return sources

    # ------------------------------------------------------------------
    # Agent (Agno) — delegates to LLM module
    # ------------------------------------------------------------------

    def create_agent(self):
        """
        Create the Agno sovereign agent.

        In V3 this will delegate to the LLM module via proxy.
        For now (Step 1), still uses the V2 agent builder but from Core context.
        """
        from .agent import create_agent as _create_agent
        return _create_agent(self)

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def get_v3_status(self) -> dict:
        """Return V3-specific status for Observatory API."""
        status = {
            "version": "3.0",
            "mode": "microkernel",
            "boot_time": round(time.time() - self._start_time, 1),
            "limbo": self._limbo_mode,
            "bus_stats": self.bus.stats,
            "bus_modules": list(self.bus.modules),
            "guardian_status": self.guardian.get_status(),
        }
        # Include Agency stats if available
        if self._agency:
            status["agency"] = self._agency.get_stats()
        if self._agency_assessment:
            status["assessment"] = self._agency_assessment.get_stats()
        return status

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full_config() -> dict:
        """Load config.toml."""
        config_path = os.path.join(os.path.dirname(__file__), "config.toml")
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib  # type: ignore
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning("[TitanCore] Could not load config.toml: %s", e)
            return {}

    def _resolve_wallet(self, wallet_path: str) -> Optional[str]:
        """Resolve wallet keypair (same logic as TitanPlugin)."""
        enc_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "soul_keypair.enc")
        )
        if os.path.exists(enc_path):
            try:
                from .utils.crypto import decrypt_for_machine
                with open(enc_path, "rb") as f:
                    encrypted = f.read()
                key_bytes = decrypt_for_machine(encrypted)
                import json
                runtime_path = os.path.join(os.path.dirname(__file__), "..", "data", "runtime_keypair.json")
                with open(runtime_path, "w") as f:
                    json.dump(list(key_bytes), f)
                logger.info("[TitanCore] Warm reboot: hardware-bound keypair decrypted.")
                return runtime_path
            except Exception as e:
                logger.warning("[TitanCore] Hardware-bound keypair failed: %s", e)

        if os.path.exists(wallet_path):
            return wallet_path

        genesis_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "genesis_record.json")
        )
        if os.path.exists(genesis_path):
            return None

        logger.info("[TitanCore] No keypair at %s — degraded mode.", wallet_path)
        return wallet_path


def _placeholder_worker(queue, config):
    """Placeholder module worker — will be replaced by real implementations in Step 2+."""
    import time as _time
    logger = logging.getLogger("placeholder")
    logger.info("Placeholder worker started, waiting for shutdown...")
    try:
        while True:
            _time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        pass
