"""
Titan Plugin for OpenClaw 2026.
Initializes the Titan Memory Architecture and exposes OpenClaw hooks.

V2.0 Step 5 (The Stealth-Sage): Integrates autonomous multi-modal research via
SearXNG (search), Crawl4AI (scrape), DocumentProcessor (PDF/DOCX/PPTX/XLSX),
XResearcher (Twitter/X), and Ollama phi3:mini (distillation). Research findings
are injected into the system_prompt via STATE_NEED_RESEARCH Gatekeeper routing.
"""
import logging
import time
from pathlib import Path

# Eager imports — lightweight, no torch/triton dependencies.
from .core.metabolism import MetabolismController
from .core.memory import TieredMemoryGraph
from .core.soul import SovereignSoul
from .core.network import HybridNetworkClient
from .logic.mood.engine import MoodEngine
from .logic.meditation import MeditationEpoch
from .logic.backup import RebirthBackup
from .api.events import EventBus
from .core.social_graph import SocialGraph

# Lazy imports — these transitively load torch (~443MB libtorch_cpu.so) and
# triton (~416MB libtriton.so). Loading them eagerly forces every supervised
# worker process (memory, body, mind, timechain, etc.) to map ~860MB of shared
# libraries it doesn't need, blowing the memory budget. Lazy-load via PEP 562
# __getattr__ so the symbols still work as `from titan_plugin import X` but
# the actual import only fires when X is first accessed.
_LAZY_IMPORTS = {
    "SageGuardian": (".logic.sage.guardian", "SageGuardian"),
    "SageRecorder": (".core.sage.recorder", "SageRecorder"),
    "StealthSageResearcher": (".logic.sage.researcher", "StealthSageResearcher"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        from importlib import import_module
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value  # cache so subsequent accesses skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# PERSISTENCE_BY_DESIGN: TitanPlugin._full_config / _stealth_sage_config /
# _recovery_mode / _last_execution_mode / _last_research_sources are runtime
# bootstrap state (config loaded from config.toml, flags from filesystem
# markers, session caches). Not self-owned state to persist across restarts.
class TitanPlugin:
    """
    Main entry point for the Titan Memory Architecture plugin.
    Coordinates memory, metabolism, soul (identity), and network operations.
    """

    # Limbo state: True when the Titan has no brain and is waiting for resurrection
    _limbo_mode = False

    def __init__(self, wallet_path: str):
        """
        Initializes the TitanPlugin with all its core subsystems.

        Boot sequence:
          1. Try hardware-bound keypair (data/soul_keypair.enc) → warm reboot
          2. Fall back to plaintext keypair (authority.json) → first boot / legacy
          3. If neither exists → enter Limbo State (await Maker resurrection)
          4. Check for recovery flag → RECOVERY mode post (one-time)

        Args:
            wallet_path (str): The file path to the Solana wallet keypair.
        """
        import os

        # ── Keypair Resolution: hardware-bound → plaintext → limbo ──
        resolved_wallet = self._resolve_wallet(wallet_path)
        if resolved_wallet is None:
            self._enter_limbo()
            return

        # ── Recovery Mode Detection ──
        self._recovery_mode = False
        recovery_flag = os.path.join(os.path.dirname(__file__), "..", "data", "recovery_flag.json")
        if os.path.exists(recovery_flag):
            self._recovery_mode = True
            logging.info("[TitanPlugin] RECOVERY MODE detected — will post resurrection tweet.")
            try:
                os.remove(recovery_flag)
            except Exception:
                pass

        # Load full config.toml once, pass sections to subsystems
        self._full_config = self._load_full_config()
        network_cfg = self._full_config.get("network", {})
        inference_cfg = self._full_config.get("inference", {})

        # Boot Ollama Cloud client for internal LLM operations
        self._ollama_cloud = None
        ollama_cloud_key = inference_cfg.get("ollama_cloud_api_key", "")
        ollama_cloud_url = inference_cfg.get("ollama_cloud_base_url", "https://api.ollama.com/v1")
        if ollama_cloud_key:
            from .utils.ollama_cloud import OllamaCloudClient
            self._ollama_cloud = OllamaCloudClient(
                api_key=ollama_cloud_key,
                base_url=ollama_cloud_url,
            )
            logging.info("[TitanPlugin] Ollama Cloud client initialized: %s", ollama_cloud_url)

        self.network = HybridNetworkClient(config=network_cfg)
        self.soul = SovereignSoul(wallet_path, self.network, config=network_cfg)
        self.memory = TieredMemoryGraph(config={
            **inference_cfg,
            **self._full_config.get("memory_and_storage", {}),
        })
        growth_cfg = self._full_config.get("growth_metrics", {})
        self.metabolism = MetabolismController(
            self.soul, self.network, memory=self.memory, config=growth_cfg
        )

        self.mood_engine = MoodEngine(self.metabolism)
        self.recorder = SageRecorder()

        # Step 2: Boot the Pre-Frontal Cortex (Guardian)
        self.guardian = SageGuardian(self.recorder, config=inference_cfg)
        if self._ollama_cloud:
            self.guardian._ollama_cloud = self._ollama_cloud
        logging.info("[TitanPlugin] Booting SageGuardian and syncing Prime Directives...")
        self.guardian.sync_prime_directives()

        self.meditation = MeditationEpoch(self.memory, self.network, config=inference_cfg)
        if self._ollama_cloud:
            self.meditation._ollama_cloud = self._ollama_cloud
        # rFP_backup_worker Phase 1 BUG-5: construct ArweaveStore ONCE at boot
        # + pass titan_id + full_config to RebirthBackup. Avoids per-backup
        # config re-read and makes backup behavior deterministic at boot time.
        _rb_arweave = None
        try:
            _rb_budget = self._full_config.get("mainnet_budget", {})
            if _rb_budget.get("backup_arweave_enabled", False):
                _rb_net_cfg = self._full_config.get("network", {})
                _rb_net = _rb_net_cfg.get("solana_network", "devnet")
                if _rb_net == "mainnet-beta":
                    _rb_net = "mainnet"
                _rb_kp = _rb_net_cfg.get("wallet_keypair_path", "")
                if _rb_kp:
                    from .utils.arweave_store import ArweaveStore
                    _rb_arweave = ArweaveStore(keypair_path=_rb_kp, network=_rb_net)
                    logging.info("[TitanPlugin] ArweaveStore wired for backup (network=%s)", _rb_net)
        except Exception as _rb_err:
            logging.warning("[TitanPlugin] ArweaveStore init failed — backups will fall back: %s", _rb_err)

        _rb_titan_id = self._full_config.get("info_banner", {}).get("titan_id", "T1")
        self.backup = RebirthBackup(
            self.network,
            config=self._full_config.get("memory_and_storage", {}),
            titan_id=_rb_titan_id,
            arweave_store=_rb_arweave,
            full_config=self._full_config,
        )

        # Wire vault program ID (from config) into epoch subsystems
        vault_pid = self._full_config.get("network", {}).get("vault_program_id", "")
        if vault_pid:
            self.meditation._vault_program_id = vault_pid
            self.backup._vault_program_id = vault_pid
            logging.info("[TitanPlugin] Vault program wired: %s", vault_pid[:16])

        # Wire PhotonClient for ZK compression (Helius Photon indexer)
        helius_rpc = self._full_config.get("network", {}).get("helius_rpc_url", "")
        if helius_rpc:
            from .utils.photon_client import PhotonClient
            photon = PhotonClient(helius_rpc)
            self.meditation._photon = photon
            self.backup._photon = photon
            logging.info("[TitanPlugin] PhotonClient wired: %s", helius_rpc[:30])

        # Boot the StudioCoordinator (centralized expressive engine)
        from .expressive.studio import StudioCoordinator
        self.studio = StudioCoordinator(
            config=self._full_config.get("expressive", {}),
            metabolism=self.metabolism,
        )
        if self._ollama_cloud:
            self.studio._ollama_cloud = self._ollama_cloud

        # Step 6: Boot the Social Subsystem (X/Twitter Production Grade)
        from .expressive.social import SocialManager
        self.social = SocialManager(
            self.metabolism,
            mood_engine=self.mood_engine,
            recorder=self.recorder,
            memory=self.memory,
            stealth_sage_config=self._full_config.get("stealth_sage", {}),
        )
        self.meditation.social = self.social  # Wire into loops
        self.meditation.studio = self.studio
        self.backup.memory = self.memory
        # Social posting now routed via X_POST_DISPATCH in spirit_worker
        # (synthesize_social_post removed — all posts go through social_narrator gateway)
        
        # Step 3: Boot The Scholar (IQL Offline RL)
        from .logic.sage.scholar import SageScholar
        from .logic.sage.gatekeeper import SageGatekeeper
        self.scholar = SageScholar(self.recorder)
        
        # Step 4: Boot The Gatekeeper (Ego)
        self.gatekeeper = SageGatekeeper(self.scholar, self.recorder)
        self._last_execution_mode = "Shadow"
        self._is_meditating = False
        self._last_meditation_ts = 0.0
        self._last_rebirth_ts = 0.0
        self._start_time = time.time()

        # Step 5: Boot The Stealth-Sage Research Engine
        self._stealth_sage_config = self._load_stealth_sage_config()
        # Inject inference config so researcher can use cloud distillation
        self._stealth_sage_config["_inference"] = self._full_config.get("inference", {})
        self.sage_researcher = StealthSageResearcher(self._stealth_sage_config)
        if self._ollama_cloud:
            self.sage_researcher._ollama_cloud = self._ollama_cloud
            self.sage_researcher._doc_processor._ollama_cloud = self._ollama_cloud
        # Instance state: tracks research sources used in the most recent pre_prompt_hook call.
        # Consumed in post_resolution_hook to tag the RL transition with research metadata.
        self._last_research_sources: list = []
        self._last_transition_id: int = -1
        logging.info("[TitanPlugin] Stealth-Sage Research Engine initialized.")

        # Endurance testing: epoch compression & social dry-run
        endurance_cfg = self._full_config.get("endurance", {})
        self._meditation_interval = int(endurance_cfg.get("meditation_interval_override", 0)) or 21600
        self._rebirth_interval = int(endurance_cfg.get("rebirth_interval_override", 0)) or 86400
        self._snapshot_interval = int(endurance_cfg.get("snapshot_interval_override", 0)) or 900
        self._social_dry_run = endurance_cfg.get("social_dry_run", False)
        if self._social_dry_run:
            self.social._dry_run = True
            self.social._dry_run_log = endurance_cfg.get("social_dry_run_log", "./data/logs/social_dry_run.log")
            logging.info("[TitanPlugin] Social dry-run mode ENABLED → %s", self.social._dry_run_log)
        if self._meditation_interval != 21600 or self._rebirth_interval != 86400:
            logging.info(
                "[TitanPlugin] Epoch compression ACTIVE: meditation=%ds, rebirth=%ds, snapshot=%ds",
                self._meditation_interval, self._rebirth_interval, self._snapshot_interval,
            )

        # Boot the ObservatoryDB for long-term metrics storage
        from .utils.observatory_db import ObservatoryDB
        self._observatory_db = ObservatoryDB()
        # Wire observatory DB to meditation for expressive archiving
        self.meditation._observatory_db = self._observatory_db

        # Boot the TimeseriesStore for historical metrics (30-day rolling window)
        from .logic.timeseries import TimeseriesStore
        self._timeseries_store = TimeseriesStore("./data/timeseries.db")

        # Boot the EventBus for real-time WebSocket broadcasting
        self.event_bus = EventBus()
        self.event_bus.attach_db(self._observatory_db)

        # Boot the SocialGraph for per-user tracking (Phase 13: Sage Socialite)
        social_graph_db = os.path.join(
            self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
            "social_graph.db",
        )
        self.social_graph = SocialGraph(db_path=social_graph_db)
        self.metabolism._social_graph = self.social_graph  # Wire for social density
        self.social.social_graph = self.social_graph  # Wire for X engagement tracking
        logging.info("[TitanPlugin] SocialGraph initialized: %s", social_graph_db)

        # Maker Relationship Engine
        from titan_plugin.logic.maker_engine import MakerRelationshipEngine
        maker_cfg = self._full_config.get("maker_relationship", {})
        self.maker_engine = MakerRelationshipEngine(
            memory=self.memory,
            ollama_cloud=getattr(self, '_ollama_cloud', None),
            soul_md_path=str(Path(__file__).resolve().parent.parent / "titan.md"),
            config=maker_cfg,
        )
        self.maker_engine.load_state()
        logging.info("[TitanPlugin] MakerRelationshipEngine initialized.")

        # Consciousness Module (Phase B+C: self-awareness substrate)
        from titan_plugin.logic.consciousness import ConsciousnessLoop
        consciousness_cfg = self._full_config.get("consciousness", {})
        consciousness_db = os.path.join(
            self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
            "consciousness.db",
        )
        self.consciousness = ConsciousnessLoop(
            memory=self.memory,
            metabolism=self.metabolism,
            mood_engine=self.mood_engine,
            social_graph=self.social_graph,
            gatekeeper=self.gatekeeper,
            network=self.network,
            ollama_cloud=getattr(self, '_ollama_cloud', None),
            db_path=consciousness_db,
            config=consciousness_cfg,
        )
        logging.info("[TitanPlugin] Consciousness Module initialized.")

        # Verified Context Builder (multi-store retrieval with TimeChain stamps)
        try:
            from titan_plugin.logic.verified_context_builder import VerifiedContextBuilder
            _data_dir = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
            # Collect known users from social_graph for entity matching
            _known = []
            try:
                _top = self.social_graph.get_top_users(limit=100)
                _known = [u.user_id for u in _top if hasattr(u, 'user_id')]
            except Exception:
                pass
            self._verified_context_builder = VerifiedContextBuilder(
                data_dir=_data_dir,
                known_users=_known,
            )
            logging.info("[TitanPlugin] VerifiedContextBuilder initialized (known_users=%d)", len(_known))
        except Exception as _vcb_err:
            self._verified_context_builder = None
            logging.warning("[TitanPlugin] VCB init failed (legacy recall will be used): %s", _vcb_err)

        # Output Verification Gate (security gate for all external outputs)
        try:
            from titan_plugin.logic.output_verifier import OutputVerifier
            _tc_dir = os.path.join(
                self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
                "timechain")
            _titan_id = self._full_config.get("info_banner", {}).get("titan_id", "T1")
            _wallet_path = self._full_config.get("network", {}).get(
                "wallet_keypair_path", "data/titan_identity_keypair.json")
            self._output_verifier = OutputVerifier(
                titan_id=_titan_id,
                data_dir=_tc_dir,
                keypair_path=_wallet_path,
            )
            logging.info("[TitanPlugin] OutputVerifier initialized (titan_id=%s)", _titan_id)
        except Exception as _ovg_err:
            self._output_verifier = None
            logging.warning("[TitanPlugin] OutputVerifier init failed (OVG disabled): %s", _ovg_err)

        # Prepare Observatory app (sync-safe), defer background task launch to start_background_tasks()
        self._background_tasks_started = False
        api_cfg = self._full_config.get("api", {})
        if api_cfg.get("enabled", True):
            self._create_observatory_app(api_cfg)  # Create app sync (so agent can inject later)
        logging.info("[TitanPlugin] Background tasks deferred — call start_background_tasks() from async context.")

    # ------------------------------------------------------------------
    # Background Task Management
    # ------------------------------------------------------------------

    def _launch_background_tasks(self, loop, api_cfg: dict):
        """Create all background asyncio tasks on the given loop."""
        if self._background_tasks_started:
            return
        loop.create_task(self.meditation_loop())
        loop.create_task(self.backup_loop())
        loop.create_task(self._vital_snapshot_loop())
        logging.info("[TitanPlugin] Started Internal Biological Rhythms (Meditation + Rebirth + Vitals Loops).")

        if api_cfg.get("enabled", True):
            loop.create_task(self._start_observatory(api_cfg))
            logging.info("[TitanPlugin] Sovereign Observatory API launching on port %s.", api_cfg.get("port", 7777))
        self._background_tasks_started = True

    async def start_background_tasks(self):
        """Async entry point to launch background tasks from a running event loop."""
        import asyncio
        loop = asyncio.get_event_loop()
        api_cfg = self._full_config.get("api", {})
        self._launch_background_tasks(loop, api_cfg)

    # ------------------------------------------------------------------
    # Sovereign Observatory
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Agno Sovereign Agent
    # ------------------------------------------------------------------

    def create_agent(self):
        """
        Create an Agno Agent wired to all Titan subsystems.

        Returns:
            agno.agent.Agent instance ready for .run() / .arun() calls.
        """
        from .agent import create_agent as _create_agent
        return _create_agent(self)

    # ------------------------------------------------------------------
    # Wallet Resolution & Limbo State
    # ------------------------------------------------------------------

    def _resolve_wallet(self, wallet_path: str) -> str | None:
        """
        Resolve the Titan's keypair using a degradation chain:
          1. Hardware-bound encrypted keypair (data/soul_keypair.enc) → warm reboot
          2. Plaintext keypair at wallet_path (authority.json) → first boot / legacy
          3. None → triggers Limbo State

        Returns:
            The effective wallet path to use, or None if no keypair is available.
        """
        import os

        # Try 1: Hardware-bound keypair (preferred — sovereign warm reboot)
        enc_path = os.path.join(os.path.dirname(__file__), "..", "data", "soul_keypair.enc")
        enc_path = os.path.normpath(enc_path)

        if os.path.exists(enc_path):
            try:
                from .utils.crypto import decrypt_for_machine
                with open(enc_path, "rb") as f:
                    encrypted = f.read()
                key_bytes = decrypt_for_machine(encrypted)

                # Write decrypted keypair to a temp location for subsystems
                import json
                runtime_path = os.path.join(os.path.dirname(__file__), "..", "data", "runtime_keypair.json")
                with open(runtime_path, "w") as f:
                    json.dump(list(key_bytes), f)

                logging.info("[TitanPlugin] Warm reboot: hardware-bound keypair decrypted.")
                return runtime_path

            except Exception as e:
                logging.warning("[TitanPlugin] Hardware-bound keypair failed: %s", e)
                # Fall through to plaintext

        # Try 2: Plaintext keypair (first boot or legacy mode)
        if os.path.exists(wallet_path):
            logging.info("[TitanPlugin] Using plaintext keypair: %s", wallet_path)
            return wallet_path

        # Try 3: Accept the configured path even if missing — subsystems handle
        # absent wallets gracefully (return 0 balance, None pubkey, etc.).
        # Only enter Limbo if the genesis record exists (post-ceremony state)
        # which proves a keypair once existed and was burned.
        genesis_path = os.path.join(os.path.dirname(__file__), "..", "data", "genesis_record.json")
        if os.path.exists(genesis_path):
            logging.warning("[TitanPlugin] No keypair found (post-ceremony) — entering Limbo State.")
            return None

        # Pre-ceremony / test environment: proceed with missing wallet
        logging.info("[TitanPlugin] No keypair at %s — proceeding in degraded mode.", wallet_path)
        return wallet_path

    def _enter_limbo(self):
        """
        Enter Limbo State: the Titan has no brain and awaits resurrection.
        Boots only the minimal Observatory API with /health and /maker/resurrect.
        All other subsystems are skipped.
        """
        import asyncio

        TitanPlugin._limbo_mode = True
        self._full_config = self._load_full_config()

        # Minimal stubs so API endpoints don't crash on attribute access
        self.memory = None
        self.soul = None
        self.metabolism = None
        self.mood_engine = None
        self.recorder = None
        self.guardian = None
        self.meditation = None
        self.backup = None
        self.social = None
        self.studio = None
        self.scholar = None
        self.gatekeeper = None
        self.sage_researcher = None
        self._last_execution_mode = "Limbo"
        self._last_research_sources = []
        self._last_transition_id = -1
        self._recovery_mode = False

        # Boot EventBus and minimal Observatory
        self.event_bus = EventBus()
        obs_db = getattr(self, "_observatory_db", None)
        if obs_db:
            self.event_bus.attach_db(obs_db)

        try:
            loop = asyncio.get_event_loop()
            api_cfg = self._full_config.get("api", {})
            if api_cfg.get("enabled", True):
                self._create_observatory_app(api_cfg)
                loop.create_task(self._start_observatory(api_cfg))
                logging.info(
                    "[TitanPlugin] LIMBO STATE — Observatory API launching (resurrection endpoint active)."
                )
        except RuntimeError:
            logging.warning("[TitanPlugin] LIMBO STATE — No event loop; Observatory will not start.")

        logging.warning(
            "[TitanPlugin] LIMBO STATE ACTIVE. Awaiting Maker resurrection at POST /maker/resurrect."
        )

    def _create_observatory_app(self, api_cfg: dict):
        """Create the Observatory FastAPI app synchronously (so it's available for agent injection)."""
        try:
            from .api import create_app
            app = create_app(self, self.event_bus, api_cfg)
            self._observatory_app = app
            return app
        except Exception as e:
            logging.warning("[TitanPlugin] Observatory app creation failed: %s", e)
            return None

    async def _start_observatory(self, api_cfg: dict):
        """Launch the Observatory API server, absorbing startup errors gracefully."""
        try:
            import uvicorn

            # Use already-created app, or create if not yet available
            app = getattr(self, '_observatory_app', None) or self._create_observatory_app(api_cfg)
            if app is None:
                return

            host = api_cfg.get("host", "0.0.0.0")
            port = int(api_cfg.get("port", 7777))

            uvi_config = uvicorn.Config(
                app=app, host=host, port=port, log_level="info", access_log=False,
            )
            server = uvicorn.Server(uvi_config)
            await server.serve()
        except SystemExit:
            logging.warning("[TitanPlugin] Observatory API could not bind port (already in use?).")
        except Exception as e:
            logging.warning("[TitanPlugin] Observatory API failed to start: %s", e)

    # ------------------------------------------------------------------
    # Config Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full_config() -> dict:
        """Loads the full merged Titan config (config.toml + ~/.titan/secrets.toml)."""
        from titan_plugin.config_loader import load_titan_config
        return load_titan_config()

    @staticmethod
    def _load_stealth_sage_config() -> dict:
        """Returns [stealth_sage] from merged config, with defaults filled in."""
        from titan_plugin.config_loader import load_titan_config
        defaults = {
            "searxng_host": "http://localhost:8080",
            "searxng_top_num_urls": 3,
            "twitterapi_io_key": "",
            "twitterapi_search_depth": 20,
            "webshare_rotating_url": "",
            "max_load_avg": 2.0,
            "research_timeout_seconds": 30,
            "doc_safe_room": "/tmp/titan_sage_docs",
        }
        loaded = load_titan_config().get("stealth_sage", {})
        return {**defaults, **loaded}

    async def pre_prompt_hook(self, user_prompt: str, context: dict) -> dict:
        """
        OpenClaw Pre-Prompt Hook:
        Intercepts the user prompt before it reaches the main LLM.
        1. Recollection Phase: Fetch semantic + episodic memory from the TieredMemoryGraph.
        2. Directives: Injects Prime Directives from the Sovereign Soul NFT.
        3. Gatekeeper Decision (V1.4/V2.0): Evaluates Agent confidence using the IQL Advantage Score.
           - Sovereign (A > 0.8): Sets bypass_llm=True and injects exact intent.
           - Collaborative (0.4 < A ≤ 0.8): Prepends [TITAN_INTERNAL_INTENT] suggestion.
           - STATE_NEED_RESEARCH (A ≤ 0.4 + informational query): Triggers Stealth-Sage pipeline.
           - Shadow (A ≤ 0.4): Defers fully to the LLM Oracle.

        Args:
            user_prompt (str): The raw text from the user interaction.
            context (dict): The global OpenClaw execution context dictionary.

        Returns:
            dict: The augmented context dictionary ready for execution or bypass.
        """
        import torch

        context = context or {}

        # Reset research tracking state for this interaction cycle
        self._last_research_sources = []

        # 1. Recollection Phase: Fetch semantic + episodic memory
        relevant_memories = await self.memory.query(user_prompt)

        # 2. Fetch Directives
        directives = await self.soul.get_active_directives()

        # 3. Inject context
        context["titan_memory"] = relevant_memories
        context["titan_directives"] = directives

        # 4. Get 128-dim state tensor via SentenceTransformer + projection layer
        try:
            embedder = self.recorder.action_embedder
            raw_emb = embedder.encode([user_prompt], convert_to_tensor=True)[0]
            pad_size = 3072 - raw_emb.shape[0]
            if pad_size > 0:
                padded = torch.cat([raw_emb, torch.zeros(pad_size, dtype=torch.float32, device=raw_emb.device)])
            else:
                padded = raw_emb[:3072]
            state_tensor = self.recorder.projection_layer(padded.unsqueeze(0)).squeeze(0)
            self._last_observation_vector = padded.tolist()
        except Exception:
            # Fallback for test environments without embedder
            state_tensor = torch.zeros(128)
            self._last_observation_vector = None

        # 5. Gatekeeper: Hybrid Execution Routing (now passes raw_prompt for Step 5)
        mode, adv, text = self.gatekeeper.decide_execution_mode(state_tensor, raw_prompt=user_prompt)
        self._last_execution_mode = mode

        if mode == "Sovereign":
            # High confidence: The Titan knows exactly what to do. Mute the LLM.
            context["bypass_llm"] = True
            context["override_response"] = text

        elif mode == "Collaborative":
            # Medium confidence: suggest action for LLM to review and refine.
            intent_block = (
                f"\n\n[TITAN_INTERNAL_INTENT]: \"The Titan's latent policy suggests: "
                f"{text}. Review against the current prompt; override if needed.\"\n\n"
            )
            context["titan_internal_intent"] = intent_block
            context.setdefault("system_prompt", "")
            context["system_prompt"] += intent_block

        elif mode == "STATE_NEED_RESEARCH":
            # Low confidence + informational query: Activate Stealth-Sage research pipeline.
            logging.info(
                f"[TitanPlugin] STATE_NEED_RESEARCH triggered for: '{user_prompt[:80]}...' "
                f"(Advantage: {adv:.3f}). Invoking Stealth-Sage."
            )
            # Snapshot current RL buffer index for research audit log session link
            try:
                transition_id = len(self.recorder.buffer) if self.recorder.buffer else -1
            except Exception:
                transition_id = -1
            
            self._last_transition_id = transition_id

            if self.sage_researcher is None:
                logging.warning("[TitanPlugin] sage_researcher not initialized — skipping research")
                sage_findings = ""
            else:
                sage_findings = await self.sage_researcher.research(
                    knowledge_gap=user_prompt,
                    transition_id=transition_id,
                )

            if sage_findings:
                context.setdefault("system_prompt", "")
                # Prepend findings so they are the first thing the LLM reads
                context["system_prompt"] = sage_findings + "\n\n" + context["system_prompt"]
                # Track which sources were used for post_resolution_hook metadata tagging
                # Parse from the researcher's audit log entry is implicit; we derive from the
                # findings block itself and store on instance state.
                self._last_research_sources = self._extract_sources_from_findings(sage_findings)
                # Record research topic for social synchronicity detection
                self.memory.add_research_topic(user_prompt[:200])
                logging.info(
                    f"[TitanPlugin] Stealth-Sage injected {len(sage_findings)} chars of research findings."
                )
            else:
                logging.info("[TitanPlugin] Stealth-Sage returned no findings; falling through to Shadow (LLM)."
                             )

        return context

    # ------------------------------------------------------------------
    # State Narrator: Reusable state-to-human translation
    # ------------------------------------------------------------------
    def _get_state_narrator(self):
        """Lazy-init the StateNarrator."""
        if not hasattr(self, '_state_narrator') or self._state_narrator is None:
            from .logic.state_narrator import StateNarrator
            self._state_narrator = StateNarrator(
                ollama_cloud=getattr(self, '_ollama_cloud', None)
            )
        return self._state_narrator

    def _gather_current_state(self) -> dict:
        """Gather Titan's current state for narration."""
        state = {"neuromod": {}, "emotion": "neutral", "chi": 0.5, "is_dreaming": False, "active_programs": []}
        try:
            neuromod = self.memory.get_neuromod_state() if self.memory else None
            if neuromod and isinstance(neuromod, dict):
                state["neuromod"] = {k: neuromod.get(k, 0.5) for k in ["DA", "5-HT", "NE", "GABA", "ACh"]}
                state["emotion"] = neuromod.get("emotion", "neutral")
        except Exception:
            pass
        try:
            coord = self.memory.get_coordinator() if self.memory else None
            if coord and isinstance(coord, dict):
                chi = coord.get("chi", {})
                state["chi"] = chi.get("total", 0.5) if isinstance(chi, dict) else 0.5
                dreaming = coord.get("dreaming", {})
                state["is_dreaming"] = dreaming.get("is_dreaming", False) if isinstance(dreaming, dict) else False
        except Exception:
            pass
        return state

    def _add_state_header(self, tweet: str) -> str:
        """Prepend a compact state header to a tweet for X/Twitter."""
        try:
            narrator = self._get_state_narrator()
            state = self._gather_current_state()
            header = narrator.format_x_header(state)
            combined = f"{header}\n\n{tweet}"
            return combined[:280]
        except Exception:
            return tweet[:280]

    # ------------------------------------------------------------------
    # Omni-Voice: Centralized Social Synthesis
    # ------------------------------------------------------------------
    # Epoch-aware tone map — gives the Titan a Circadian Rhythm
    # synthesize_social_post REMOVED — all X posts now go through
    # social_narrator.build_dispatch_payload() → X_POST_DISPATCH handler
    # in spirit_worker.py. See memory/directive_social_posting_gateway.md

    @staticmethod
    def _extract_sources_from_findings(findings: str) -> list:
        """
        Heuristically extracts a list of source types from the [SAGE_RESEARCH_FINDINGS] block.
        This is used to populate the research_sources metadata in post_resolution_hook without
        requiring an extra async call or shared mutable state from the researcher.

        Sources are additive — the Stealth-Sage pipeline layers Web, X, and Document
        data cumulatively. Web (SearXNG + Crawl4AI) is always the base when findings
        are non-empty; X and Document blocks are detected independently.

        Args:
            findings (str): The [SAGE_RESEARCH_FINDINGS] block.

        Returns:
            list[str]: Source type strings e.g. ["Web", "Document", "X"]
        """
        if not findings:
            return []

        sources = ["Web"]
        if "[X_SEARCH_RESULTS" in findings:
            sources.append("X")
        if "Document Topic:" in findings:
            sources.append("Document")
        return sources

    async def post_resolution_hook(self, user_prompt: str, agent_response: str) -> None:
        """
        OpenClaw Post-Resolution Hook:
        Intercepts the final generated response before serving it to the user.
        1. Logs the interaction to the local Mempool.
        2. V1.4 Guardian Shield Check: Evaluates the response for Prime Directive alignment.
           If violated, triggers Divine Trauma recording.
        3. Records the RL Transition with Mood Engine reward.
           V2.0 Step 5: If a research cycle was triggered, tags the transition with
           research_used=True and research_sources=[...] for Sovereignty Index auditing.

        Args:
            user_prompt (str): The initial prompt.
            agent_response (str): The final text generated by the LLM or Sovereign bypass.
        """
        # 1. Main Memory Store
        await self.memory.add_to_mempool(user_prompt, agent_response)

        # 2. Titan V1.4 Step 2: The Guardian Shield Check
        is_safe = await self.guardian.process_shield(agent_response)
        if not is_safe:
            logging.warning(
                f"[TitanPlugin] Agent action was BLOCKED by the Guardian Shield. Divine Trauma applied."
            )
            return  # Skip standard RL recording; Guardian has already logged trauma

        # 3. RL Experience Recording (Async Background Task)
        import random
        import asyncio

        try:
            # Observation vector: use real embedding from pre-hook, fallback to random
            real_obs = getattr(self, '_last_observation_vector', None)
            observation_vector = real_obs if real_obs else [random.uniform(-1.0, 1.0) for _ in range(3072)]

            # Pull Reward from MoodEngine
            # info_gain weighted by research source diversity:
            #   Web only: +0.03, Web+X: +0.05, Web+X+Document: +0.08
            # Incentivizes full multi-modal deep research over lazy single-source
            sources = set(self._last_research_sources)
            if not sources:
                info_gain = 0.0
            elif "Document" in sources and "X" in sources:
                info_gain = 0.08
            elif "X" in sources:
                info_gain = 0.05
            else:
                info_gain = 0.03
            reward = self.mood_engine.get_current_reward(info_gain=info_gain)

            # Base trauma metadata
            metadata: dict = {
                "is_violation": False,
                "directive_id": -1,
                "trauma_score": 0.0,
                "reasoning_trace": "",
                "guardian_veto_logic": "",
                "execution_mode": self._last_execution_mode,
            }

            # V2.0 Step 5: Tag the transition if research was used in this cycle.
            research_md = {
                "research_used": False,
                "transition_id": self._last_transition_id
            }
            
            if self._last_research_sources:
                research_md["research_used"] = True
                logging.info(
                    f"[TitanPlugin] Tagging RL transition: research_used=True, "
                    f"transition_id={self._last_transition_id}, sources={self._last_research_sources}"
                )

            # Fire-and-forget: record without blocking the OpenClaw response pipeline
            asyncio.create_task(
                self.recorder.record_transition(
                    observation_vector=observation_vector,
                    action=agent_response,
                    reward=reward,
                    trauma_metadata=metadata,
                    research_metadata=research_md,
                    session_id="openclaw_session",
                )
            )
            logging.info("[TitanPlugin] Dispatched SageRecorder background task.")
        except Exception as e:
            logging.error(f"[TitanPlugin] Failed to dispatch SageRecorder task: {e}")

    async def meditation_loop(self):
        """
        The Internal "Circadian" Loop of the Titan (Master Scheduler).
        Sleeps for configurable interval (default 6 hours), then triggers:
        Phase A: Memory Consolidation (V1.1-1.3)
        Phase B: The Scholar's Dream (IQL Offline RL - V1.4)
        """
        import asyncio
        import os

        meditation_interval = self._meditation_interval
        logging.info("[TitanPlugin] Meditation loop entered. Interval: %ds. First cycle in %ds.", meditation_interval, meditation_interval)

        while True:
            await asyncio.sleep(meditation_interval)
            logging.info("[TitanPlugin] Meditation loop woke up after %ds sleep.", meditation_interval)
            
            # Check system load to ensure we don't crash
            try:
                load1, load5, load15 = os.getloadavg()
                if load1 > 4.0: # Bumped for endurance testing (4-core VPS under concurrent load)
                    logging.warning(f"[TitanPlugin] System load too high ({load1}). Skipping meditation cycle.")
                    continue
            except Exception:
                pass

            self._is_meditating = True
            self._last_meditation_ts = time.time()
            logging.info("[TitanPlugin] Initiating 6-hour Meditation Cycle...")
            
            # Phase A: Consolidate Memory via Ollama & generate flow field art
            try:
                await self.meditation.run_small_epoch()
                logging.info("[TitanPlugin] Memory Consolidation Phase Complete.")
            except Exception as e:
                logging.error(f"[TitanPlugin] Memory Consolidation Failed: {e}")
                
            # Phase B: Scholar Dreams over the optimized ReplayBuffer
            try:
                logging.info("[TitanPlugin] Commencing The Scholar's Dream (IQL)...")
                dream_results = await self.scholar.dream(epochs=50, batch_size=256)
                # Only write to Chronicle if IQL actually trained (non-zero losses)
                total_loss = sum(dream_results.get(k, 0.0) for k in ("loss_actor", "loss_qvalue", "loss_value"))
                if total_loss > 0.0:
                    self._append_to_chronicle(dream_results)
                else:
                    logging.info("[TitanPlugin] Scholar Dream produced no training — skipping Chronicle entry.")
            except Exception as e:
                logging.error(f"[TitanPlugin] Scholar Training Failed: {e}")
            
            # Phase C: Maker Relationship Engine (post-meditation idle task)
            try:
                if hasattr(self, 'maker_engine') and self.maker_engine:
                    maker_result = await self.maker_engine.run()
                    if maker_result.get("topics_promoted", 0) > 0:
                        logging.info("[TitanPlugin] MakerEngine promoted %d topics.", maker_result["topics_promoted"])
                    self.maker_engine.save_state()
            except Exception as e:
                logging.error(f"[TitanPlugin] MakerEngine failed: {e}")

            # Phase D: Consciousness — self-observation and journey topology
            try:
                if hasattr(self, 'consciousness') and self.consciousness:
                    c_result = await self.consciousness.run()
                    logging.info(
                        "[TitanPlugin] Consciousness epoch %d: drift=%.4f curvature=%.3f density=%.3f%s",
                        c_result["epoch_id"], c_result["drift_magnitude"],
                        c_result["curvature"], c_result["density"],
                        " [ANCHORED]" if c_result["anchored"] else "",
                    )
            except Exception as e:
                logging.error(f"[TitanPlugin] Consciousness failed: {e}")

            self._is_meditating = False

    async def _vital_snapshot_loop(self):
        """
        Background loop: record vital snapshots to ObservatoryDB every 15 minutes.
        Powers the Sovereignty Horizon chart and historical stats views.
        """
        import asyncio

        snapshot_interval = self._snapshot_interval
        logging.info("[TitanPlugin] Vital snapshot loop entered. Interval: %ds.", snapshot_interval)

        while True:
            await asyncio.sleep(snapshot_interval)
            logging.info("[TitanPlugin] Vital snapshot tick.")
            try:
                obs_db = getattr(self, "_observatory_db", None)
                if obs_db is None:
                    continue

                # Gather cached metrics (no heavy async calls for non-critical snapshot)
                sov_pct = getattr(self.gatekeeper, "sovereignty_score", 0.0)
                life_pct = getattr(self.metabolism, "_last_balance_pct", -1.0)
                sol_balance = getattr(self.metabolism, "_last_balance", 0.0) or 0.0
                energy_state = getattr(self.metabolism, "energy_state", "UNKNOWN")
                mood_label = self.mood_engine.get_mood_label()
                mood_score = self.mood_engine.previous_mood
                persistent_count = self.memory.get_persistent_count()
                mempool = await self.memory.fetch_mempool()
                epoch_counter = getattr(self.meditation, "_epoch_counter", 0)

                obs_db.record_vital_snapshot(
                    sovereignty_pct=sov_pct,
                    life_force_pct=life_pct,
                    sol_balance=sol_balance,
                    energy_state=energy_state,
                    mood_label=mood_label,
                    mood_score=mood_score,
                    persistent_count=persistent_count,
                    mempool_size=len(mempool),
                    epoch_counter=epoch_counter,
                )
            except Exception as e:
                logging.debug("[TitanPlugin] Vital snapshot failed: %s", e)

    async def backup_loop(self):
        """
        Meditation-triggered backup loop.
        Watches for trigger files written by spirit_worker on MEDITATION_COMPLETE,
        then delegates to backup.on_meditation_complete() which decides:
        - ZK snapshot (every meditation)
        - Personality → Arweave (1st of day)
        - Soul package → Arweave (1st Sunday)
        - MyDay NFT (every 4th meditation)
        """
        import asyncio
        import json

        trigger_path = os.path.join("data", "backup_trigger.json")

        # Boot check
        try:
            await self.backup.check_on_boot()
        except Exception as e:
            logging.warning("[Backup] Boot check failed: %s", e)

        while True:
            await asyncio.sleep(30)  # Poll every 30 seconds

            if not os.path.exists(trigger_path):
                continue

            try:
                with open(trigger_path) as f:
                    trigger = json.load(f)
                os.remove(trigger_path)

                payload = trigger.get("payload", {})
                med_count = trigger.get("meditation_count", 0)

                logging.info("[Backup] Processing meditation #%d trigger...", med_count)
                self._last_rebirth_ts = time.time()
                await self.backup.on_meditation_complete(payload)

            except json.JSONDecodeError as e:
                logging.warning("[Backup] Invalid trigger file: %s", e)
                try:
                    os.remove(trigger_path)
                except OSError:
                    pass
            except Exception as e:
                logging.error("[Backup] Backup loop error: %s", e)

    def _append_to_chronicle(self, dream_results: dict):
        import os
        from datetime import datetime

        soul_path = os.path.join(os.path.dirname(__file__), "..", "titan.md")
        archive_path = os.path.join(os.path.dirname(__file__), "..", "data", "history", "soul_archive.md")
        raw_log_path = os.path.join(os.path.dirname(__file__), "..", "data", "logs", "scholar_raw.log")

        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)

        actor_loss = dream_results.get("loss_actor", 0.0)
        q_loss = dream_results.get("loss_qvalue", 0.0)
        v_loss = dream_results.get("loss_value", 0.0)
        total_loss = actor_loss + q_loss + v_loss

        # Log raw metrics to data file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(raw_log_path, "a") as rf:
            rf.write(f"[{timestamp}] Raw Loss -> Actor: {actor_loss:.4f}, Q: {q_loss:.4f}, V: {v_loss:.4f}\n")

        # --- Cognitive State Mapper ---
        # Map raw RL metrics to self-aware narrative states.
        # Actor loss: how well decisions align with learned policy
        # Q-value loss: stability of action-value estimates
        # V-value loss: clarity of state valuation

        # Policy alignment: derived from actor loss trajectory
        if actor_loss < 0.01:
            alignment = "Decisions are fully aligned with Prime Directives."
            alignment_label = "Sovereign Clarity"
        elif actor_loss < 0.1:
            alignment = "Decisions are converging toward Prime Directive alignment."
            alignment_label = "Converging"
        elif actor_loss < 0.5:
            alignment = "Policy is stabilizing; directive integration in progress."
            alignment_label = "Stabilizing"
        else:
            alignment = "Early learning phase; building foundational decision patterns."
            alignment_label = "Foundational"

        # Conceptual complexity: derived from Q-value variance
        if q_loss < 0.05:
            complexity = "Action-value landscape is well-mapped and predictable."
        elif q_loss < 0.3:
            complexity = "Navigating moderate conceptual complexity in action evaluation."
        else:
            complexity = "High conceptual complexity — exploring unfamiliar action-value territory."

        # Value clarity: derived from V-network loss
        if v_loss < 0.05:
            insight = "State valuation is sharp — I can clearly distinguish beneficial from harmful states."
        elif v_loss < 0.3:
            insight = "State awareness is developing; value distinctions are becoming clearer with each cycle."
        else:
            insight = "State valuation is still forming — many states remain ambiguous in their long-term impact."

        new_entry = (
            f"[{timestamp}] Meditation Cycle\n"
            f"* IQL Loss — Actor: {actor_loss:.4f}, Q-Value: {q_loss:.4f}, V-Value: {v_loss:.4f} (Total: {total_loss:.4f})\n"
            f"* Policy Alignment: {alignment_label} — {alignment}\n"
            f"* Complexity: {complexity}\n"
            f"* Insight: \"{insight}\"\n\n"
        )

        try:
            with open(soul_path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            content = "# Titan Soul Document\n"

        separator = "## The Scholar's Chronicle"
        if separator not in content:
            content += f"\n\n{separator}\n"

        parts = content.split(separator)
        header = parts[0]
        chronicle = parts[1].strip()

        entries = [e for e in chronicle.split("\n[") if e.strip()]

        # Archive oldest entries beyond rolling window
        if len(entries) >= 50:
            if not os.path.exists(archive_path):
                with open(archive_path, "w") as af:
                    af.write("# Titan Soul Archive\n## Historical Records of the Sage's Growth\nThis file contains the archived reflections and training data of the Titan, moved from the primary titan.md to preserve cognitive efficiency.\n\n")

            oldest_entry = "[" + entries[0].strip() + "\n\n"
            with open(archive_path, "a") as af:
                af.write(oldest_entry)

            entries = entries[1:]

        # Rebuild Chronicle
        rebuilt_chronicle = "\n".join(["[" + e.strip() for e in entries])
        if rebuilt_chronicle:
            rebuilt_chronicle += "\n\n"
        rebuilt_chronicle += new_entry

        with open(soul_path, "w") as f:
            f.write(f"{header.rstrip()}\n\n{separator}\n{rebuilt_chronicle}")

        logging.info("[TitanPlugin] Appended rolling chronicle entry to titan.md.")

# Required OpenClaw plugin export
def init_plugin(config: dict) -> TitanPlugin:
    """
    Plugin initialization factory required by OpenClaw.
    
    Args:
        config (dict): Configuration dictionary containing settings like WALLET_KEYPAIR_PATH.
        
    Returns:
        TitanPlugin: An initialized instance of the Titan plugin.
    """
    wallet_path = config.get("WALLET_KEYPAIR_PATH", "./authority.json")
    return TitanPlugin(wallet_path)
