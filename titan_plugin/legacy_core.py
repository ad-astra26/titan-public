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

from .bus import DivineBus, make_msg, MODULE_HEARTBEAT, EPOCH_TICK, IMPULSE, ACTION_RESULT, OUTER_OBSERVATION, SOVEREIGNTY_EPOCH
from .guardian import Guardian, ModuleSpec
from titan_plugin.utils.silent_swallow import swallow_warn
from titan_plugin import bus

logger = logging.getLogger(__name__)


# PERSISTENCE_BY_DESIGN: TitanCore._full_config / _agency / _proxies /
# _*_mode fields are runtime bootstrap state — loaded from config.toml or
# constructed on boot. Config data is not self-owned state to persist.
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
        # Option B (2026-04-29): see kernel.py:245 for full rationale —
        # reply_only=True excludes broadcasts; types=[] documents the
        # contract for arch_map.
        self._core_queue = self.bus.subscribe(
            "core", reply_only=True, types=[])
        # Pre-subscribe meditation queue before Guardian starts modules
        # (spirit_worker may send MEDITATION_REQUEST during boot)
        self._meditation_queue = self.bus.subscribe(
            "meditation", reply_only=True, types=[])
        # Mainnet Lifecycle Wiring rFP (2026-04-20): subscribe eagerly so
        # SOVEREIGNTY_EPOCH messages from spirit_worker never drop on the
        # "dst without subscriber" path.
        # Option B (2026-04-29): only SOVEREIGNTY_EPOCH consumed in
        # _sovereignty_loop. Same filter as kernel.py:252.
        from .bus import SOVEREIGNTY_EPOCH as _SE
        self._sovereignty_queue = self.bus.subscribe(
            "sovereignty", types=[_SE])

        # ── StateRegister (real-time state buffer) ──────────────────
        from titan_plugin.logic.state_register import StateRegister
        self.state_register = StateRegister()
        enrichment_cfg = self._full_config.get("spirit_enrichment", {})
        snapshot_interval = float(enrichment_cfg.get("micro_tick_interval", 10.0))
        self.state_register.start(self.bus, snapshot_interval=snapshot_interval)

        # ── Microkernel v2 Phase A §A.2 — StateRegistry bank (shm) ──
        # Owns writers/readers for /dev/shm/titan_{titan_id}/*.bin.
        # Writers are populated by background threads reading from
        # state_register (this process) and spirit_worker (subprocess).
        # Feature-gated via [microkernel] flags in titan_params.toml;
        # all default false so the shm path is byte-identical to the
        # legacy path until Maker flips a flag.
        #
        # titan_id resolution follows the canonical precedence chain
        # (data/titan_identity.json → TITAN_ID env → "T1") via
        # resolve_titan_id() — same pattern as emot_shm_protocol. This
        # is critical on T2+T3 which share /dev/shm on one VPS: without
        # the canonical resolver, both would default to "T1" and stomp
        # each other's trinity_state.bin.
        from titan_plugin.core.state_registry import RegistryBank
        # Passing None triggers the canonical resolver.
        self._registry_bank = RegistryBank(
            titan_id=None, config=self._full_config,
        )

        # ── Guardian ─────────────────────────────────────────────────
        # [guardian] toml plumbed 2026-04-16 (dead-wiring audit). Section
        # reaches Guardian which reads heartbeat_timeout_default /
        # max_restarts_in_window / restart_window / sustained_uptime_reset
        # with module constants as fallbacks.
        self.guardian = Guardian(self.bus, config=self._full_config.get("guardian", {}))

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
        # L3 Phase A.8.1: deque(maxlen=256) bounds inbox at data-structure
        # level (defense-in-depth above chat.py's API-side 50-cap).
        import threading
        from collections import deque
        self._dream_inbox = deque(maxlen=256)           # Queued messages during dream
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

        # Observatory DB for persistent historical metrics.
        # rFP_universal_sqlite_writer Phase 2 — per-process singleton.
        from .utils.observatory_db import get_observatory_db
        self._observatory_db = get_observatory_db()
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

        # Microkernel v2 Phase A §A.2 — Trinity shm writer (daemon thread).
        # Reads state_register.get_full_130dt + get_full_30d_topology +
        # journey(2D) each tick, writes to /dev/shm/titan_{id}/trinity_state.bin
        # when content changes. Feature-gated via microkernel.shm_trinity_enabled;
        # when flag=false the thread still runs but makes no shm writes.
        self._start_trinity_shm_writer()

        # Agency bus listener (IMPULSE → INTENT → helper execution → ACTION_RESULT)
        if self._agency:
            asyncio.get_event_loop().create_task(self._agency_loop())

        # Chat bus bridge — parent-side handler for CHAT_REQUEST QUERIES
        # from api_subprocess. Runs unconditionally; idles when no one
        # publishes (legacy in-process mode where chat.py calls
        # core.run_chat() directly). Mirrors TitanPlugin._chat_handler_loop
        # so both runtime paths support BUG-CHAT-AGENT-NOT-INITIALIZED-API-
        # SUBPROCESS architectural fix.
        asyncio.get_event_loop().create_task(self._chat_handler_loop())

        # Mainnet Lifecycle Wiring rFP (2026-04-20): SovereigntyTracker
        # listener — receives SOVEREIGNTY_EPOCH from spirit_worker (every 10
        # consciousness epochs) and forwards to plugin.sovereignty.record_epoch.
        if self._proxies.get("sovereignty"):
            asyncio.get_event_loop().create_task(self._sovereignty_loop())

        # Meditation cycle (memory consolidation, mempool scoring, Cognee cognify)
        asyncio.get_event_loop().create_task(self._meditation_loop())

        # rFP_backup_worker Phase 1 (2026-04-20) — _backup_loop DELETED.
        # Backup runs as a Guardian-supervised subprocess ("backup" module)
        # subscribing to MEDITATION_COMPLETE via bus. See modules/backup_worker.py.

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
        from .modules.meta_teacher_worker import meta_teacher_worker_main
        from .modules.cgn_worker import cgn_worker_main
        from .modules.knowledge_worker import knowledge_worker_main
        from .modules.emot_cgn_worker import emot_cgn_worker_main
        from .modules.timechain_worker import timechain_worker_main
        from .modules.backup_worker import backup_worker_main
        from .persistence_entry import imw_main

        # Microkernel v2 Phase B.2.1 — spawn graduation gate. When true, the
        # 9 import-stable workers (warning_monitor, llm, body, mind, media,
        # language, meta_teacher, cgn, knowledge) boot via spawn-mode so
        # they can outlive a kernel swap (PDEATHSIG strip + bus reattach
        # to new broker). Default off until B.2.1 wiring soaks 7d clean per
        # Titan. Backup is on its own pre-existing flag (S6 reference);
        # spirit, memory, timechain, rl, emot_cgn stay fork-mode until
        # they're individually verified import-stable.
        _spawn_grad = self._full_config.get("microkernel", {}).get(
            "spawn_graduated_workers_enabled", False)

        # IMW — Inner Memory Writer Service. Registered FIRST so other modules
        # that write to inner_memory.db can connect on startup.
        # Autostart only when persistence.enabled=true in config.toml.
        _persistence_cfg = self._full_config.get("persistence", {})
        _imw_enabled = bool(_persistence_cfg.get("enabled", False))
        _data_dir_raw = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
        imw_config = {
            **_persistence_cfg,
            "db_path": _persistence_cfg.get("db_path") or (_data_dir_raw.rstrip("/") + "/inner_memory.db"),
        }
        self.guardian.register(ModuleSpec(
            name="imw",
            layer="L1",  # Microkernel v2 §A.5 — L1 persistence service (writer for inner_memory.db, L1's DB)
            entry_fn=imw_main,
            config=imw_config,
            rss_limit_mb=300,  # asyncio + sqlite3 + msgpack only; stays lean
            autostart=_imw_enabled,   # Only start when master switch flipped on
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=True,  # IMW IPC is via unix socket, not bus broadcasts
        ))

        # Warning Monitor Worker — Pattern C force multiplier
        # See core/plugin.py:_register_modules for design rationale.
        from .modules.warning_monitor_worker import warning_monitor_worker_main
        _wm_cfg = self._full_config.get("warning_monitor", {})
        self.guardian.register(ModuleSpec(
            name="warning_monitor",
            layer="L3",
            entry_fn=warning_monitor_worker_main,
            config=_wm_cfg,
            rss_limit_mb=300,
            autostart=True,
            lazy=False,
            heartbeat_timeout=120.0,
            reply_only=False,
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        ))

        # Observatory Writer Service — second IMW instance for observatory.db.
        # Same imw_main entry function, different config: own socket, own WAL,
        # own metrics file (auto-namespaced by name), own primary+shadow DBs.
        # Scoped per rFP_observatory_writer_service (drafted 2026-04-21);
        # microkernel-v2-aligned (L3 DB ownership). Default OFF — Maker flips
        # `enabled=true` in [persistence_observatory] when ready for Phase 1.
        _obs_persistence_cfg = self._full_config.get("persistence_observatory", {})
        _obs_writer_enabled = bool(_obs_persistence_cfg.get("enabled", False))
        # Per-instance defaults — namespaced so the two writers don't collide
        # on socket/WAL/journal/metrics paths. Maker can override any of these
        # in [persistence_observatory] in config.toml.
        _obs_data_dir = _data_dir_raw.rstrip("/")
        _obs_writer_config = {
            # Inherit defaults from main IMW; override the per-instance paths.
            **_persistence_cfg,
            **_obs_persistence_cfg,
            "socket_path": _obs_persistence_cfg.get(
                "socket_path", "data/run/observatory_writer.sock"),
            "wal_path": _obs_persistence_cfg.get(
                "wal_path", "data/run/observatory_writer.wal"),
            "journal_dir": _obs_persistence_cfg.get(
                "journal_dir", "data/run"),
            "db_path": _obs_persistence_cfg.get(
                "db_path", _obs_data_dir + "/observatory.db"),
            "shadow_db_path": _obs_persistence_cfg.get(
                "shadow_db_path", _obs_data_dir + "/observatory_shadow.db"),
        }
        self.guardian.register(ModuleSpec(
            name="observatory_writer",
            layer="L3",  # Microkernel v2 §A.5 — L3 persistence service (writer for observatory.db, L3's DB)
            entry_fn=imw_main,                    # reuse same entry — fully parameterized by config
            config=_obs_writer_config,
            rss_limit_mb=300,
            autostart=_obs_writer_enabled,        # default OFF until Maker flips
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=True,                       # Unix-socket IPC, no bus broadcasts
        ))

        # rFP_universal_sqlite_writer 2026-04-27 expansion — three more
        # per-DB writer daemons (social_graph / events_teacher / consciousness).
        for _w_name, _w_section, _w_default_db in (
            ("social_graph_writer", "persistence_social_graph", "social_graph.db"),
            ("events_teacher_writer", "persistence_events_teacher", "events_teacher.db"),
            ("consciousness_writer", "persistence_consciousness", "consciousness.db"),
        ):
            _w_cfg_section = self._full_config.get(_w_section, {})
            _w_enabled = bool(_w_cfg_section.get("enabled", False))
            _w_sock_default = f"data/run/{_w_name}.sock"
            _w_wal_default = f"data/run/{_w_name}.wal"
            _w_db_default = f"{_obs_data_dir}/{_w_default_db}"
            _w_shadow_default = f"{_obs_data_dir}/{_w_default_db.replace('.db', '_shadow.db')}"
            _w_writer_config = {
                **_persistence_cfg,
                **_w_cfg_section,
                "socket_path": _w_cfg_section.get("socket_path", _w_sock_default),
                "wal_path": _w_cfg_section.get("wal_path", _w_wal_default),
                "journal_dir": _w_cfg_section.get("journal_dir", "data/run"),
                "db_path": _w_cfg_section.get("db_path", _w_db_default),
                "shadow_db_path": _w_cfg_section.get(
                    "shadow_db_path", _w_shadow_default),
            }
            self.guardian.register(ModuleSpec(
                name=_w_name,
                layer="L3",
                entry_fn=imw_main,
                config=_w_writer_config,
                rss_limit_mb=300,
                autostart=_w_enabled,
                lazy=False,
                heartbeat_timeout=60.0,
                reply_only=True,
            ))

        # Memory module (FAISS + Kuzu + DuckDB)
        memory_config = {
            **self._full_config.get("inference", {}),
            **self._full_config.get("memory_and_storage", {}),
        }
        self.guardian.register(ModuleSpec(
            name="memory",
            layer="L2",  # Microkernel v2 §A.5 — L2 cognitive substrate (FAISS+Kuzu+DuckDB)
            entry_fn=memory_worker_main,
            config=memory_config,
            rss_limit_mb=2000,  # FAISS + Kuzu + DuckDB: ~1100-1200MB steady on T1 (40min uptime). VmRSS includes mmap'd DB pages from page cache, which inflates the reading on rapid restarts. Old 1400MB limit was sized for the Cognee era and caused T2 doom-loop.
            autostart=False,
            lazy=True,
            heartbeat_timeout=120.0,  # Memory queries can block for 30s+
            reply_only=True,  # Memory only needs targeted QUERY messages, not broadcasts
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # RL/Sage module (TorchRL — ~2500MB with mmap)
        self.guardian.register(ModuleSpec(
            name="rl",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (IQL chain learning)
            entry_fn=rl_worker_main,
            config=self._full_config.get("stealth_sage", {}),
            rss_limit_mb=3000,
            autostart=False,
            lazy=True,
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # LLM/Inference module (Agno agent — ~500MB)
        self.guardian.register(ModuleSpec(
            name="llm",
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (Agno inference, human-time)
            entry_fn=llm_worker_main,
            config=self._full_config.get("inference", {}),
            rss_limit_mb=1000,
            autostart=True,  # Changed: Language Teacher needs llm at boot
            lazy=False,
            heartbeat_timeout=120.0,  # LLM calls can block 30s+; match Spirit/Memory timeout
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Body module (5DT somatic sensors — lightweight, always-on)
        # Note: RSS includes inherited parent process memory from fork (~250MB),
        # so limit must account for that baseline.
        body_config = {
            **self._full_config.get("body", {}),
            "api_port": int(self._full_config.get("api", {}).get("port", 7777)),
            # Microkernel v2 Phase A §A.7 / §L1 — shm feature flags
            # passthrough (mirrors plugin.py — without this the
            # body shm fast-path is silently no-op).
            "microkernel": self._full_config.get("microkernel", {}),
        }
        self.guardian.register(ModuleSpec(
            name="body",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT somatic)
            entry_fn=body_worker_main,
            config=body_config,
            rss_limit_mb=800,   # was 500; fork-inherited parent memory grew from ~250MB to ~400MB+ (2026-04-17)
            autostart=True,  # Body senses must always be active
            lazy=False,
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Mind module (MoodEngine, SocialGraph — ~200MB)
        mind_config = {
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
            # Microkernel v2 Phase A §A.7 / §L1 — shm feature flags
            # passthrough (mirrors plugin.py).
            "microkernel": self._full_config.get("microkernel", {}),
        }
        self.guardian.register(ModuleSpec(
            name="mind",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT cognitive)
            entry_fn=mind_worker_main,
            config=mind_config,
            rss_limit_mb=700,   # was 500; fork-inherited parent RSS ~400MB left only 100MB headroom — caused T3 cascade (2026-04-17). Memory profiling tool (DEFERRED TOP) will identify real optimization targets.
            autostart=True,  # Mind senses should always be active
            lazy=False,
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Spirit module (Consciousness + V4 Sphere Clocks + Enrichment + Neural NS + Experience Orchestrator)
        # filter_down_v5 + titan_self were defined in titan_params.toml but
        # never plumbed here, so V5's publish_enabled flag was always
        # stuck at its False default regardless of toml edits. Caught 2026-04-16
        # when attempting to flip publish_enabled and observing V5 kept
        # reporting False from /v4/filter-down-status.
        spirit_config = {
            **self._full_config.get("consciousness", {}),
            "sphere_clock": self._full_config.get("sphere_clock", {}),
            "spirit_enrichment": self._full_config.get("spirit_enrichment", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
            "social_presence": self._full_config.get("social_presence", {}),
            "filter_down_v5": self._full_config.get("filter_down_v5", {}),
            "titan_self": self._full_config.get("titan_self", {}),
            "impulse": self._full_config.get("impulse", {}),
            "titan_vm": self._full_config.get("titan_vm", {}),
            # Microkernel v2 Phase A §A.2 — shm feature flags. Must be
            # passed through to spirit_worker so RegistryBank.is_enabled()
            # can resolve microkernel.shm_*_enabled correctly. Without
            # this, the NEUROMOD/EPOCH shm writes were silently no-op
            # even when the flags were flipped true in titan_params.toml.
            "microkernel": self._full_config.get("microkernel", {}),
        }
        self.guardian.register(ModuleSpec(
            name="spirit",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (consciousness core)
            entry_fn=spirit_worker_main,
            config=spirit_config,
            rss_limit_mb=1200,  # was 750; spirit loads 11 neural nets + consciousness.db + inner_memory.db + NS programs during boot — peak RSS 961-1486MB observed 2026-04-18
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
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (expression: speech/art/music)
            entry_fn=media_worker_main,
            config=media_config,
            rss_limit_mb=800,  # was 700 (was 500); fork-inherited parent memory grew (2026-04-17)
            autostart=True,   # Always on — art/audio generated frequently
            lazy=False,
            heartbeat_timeout=180.0,  # Image/audio digest can block 30-90s
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        ))

        # Language module (composition, teaching, vocabulary — higher cognitive)
        language_config = {
            **self._full_config.get("language", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
        }
        self.guardian.register(ModuleSpec(
            name="language",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (Language Teacher)
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
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Meta-Reasoning Teacher (rFP_titan_meta_reasoning_teacher.md)
        # Philosopher-critic observing META_CHAIN_COMPLETE from spirit.
        # Per rFP §11 migration: deploys with enabled=false by default.
        # Worker starts (for observability + pre-load); critique logic is
        # gated on config['enabled'] inside the teacher's sampling check.
        _meta_teacher_cfg = self._full_config.get("meta_teacher", {})
        meta_teacher_config = {
            **_meta_teacher_cfg,
            "inference": self._full_config.get("inference", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get(
                "data_dir", "./data"),
        }
        self.guardian.register(ModuleSpec(
            name="meta_teacher",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (philosopher-critic)
            entry_fn=meta_teacher_worker_main,
            config=meta_teacher_config,
            rss_limit_mb=800,   # was 250 (sized for Phase A asyncio+httpx only).
                                # Raised 2026-04-24 after Phase B Teaching Memory
                                # ST lazy-load pushed RSS to observed 663-665MB on
                                # T1, causing 18 crash-loop cycles before fix.
                                # See BUGS.md META-TEACHER-PHASE-B-RSS-UNDERSIZED.
            autostart=True,     # Worker always up so META_CHAIN_COMPLETE has a
                                # subscriber; critique gated on config[enabled]
            lazy=False,
            heartbeat_timeout=180.0,  # was 90s. Raised 2026-04-24 to absorb ST
                                      # lazy-load on first critique (observed 92s
                                      # timeout during Phase B memory warmup).
                                      # Teaching memory retrieval + LLM chain can
                                      # take ~60-120s under load; 180s = 2x margin.
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
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
            layer="L2",  # Microkernel v2 §A.5 — L2 concept-value state registry (per project_cgn_as_higher_state_registry.md)
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
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Knowledge Worker (4th CGN consumer — knowledge acquisition + Stealth Sage)
        _data_dir = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
        # KP-v2: flatten [knowledge_pipeline] (router/cache/health paths, circuit-
        # breaker tuning, near-dup thresholds, telegram_alerts_enabled kill-switch)
        # plus the nested [knowledge_pipeline.budgets] table (per-backend MB/day).
        # Without this, the alert cascade wired in knowledge_worker sees budgets=0
        # (unlimited) and never fires.
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
            # Knowledge Pipeline v2 (router/cache/health/budgets/alerts)
            **self._full_config.get("knowledge_pipeline", {}),
        }
        self.guardian.register(ModuleSpec(
            name="knowledge",
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (rFP L3 "Knowledge search")
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
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # TimeChain — Proof of Thought memory chain
        timechain_config = {
            **self._full_config.get("timechain", {}),
        }
        self.guardian.register(ModuleSpec(
            name="timechain",
            layer="L2",  # Microkernel v2 §A.5 — L2 cognitive substrate (episodic/declarative/procedural forks)
            entry_fn=timechain_worker_main,
            config=timechain_config,
            rss_limit_mb=850,     # was 700; fork-inherited parent memory grew (2026-04-17)
            autostart=True,       # Must be ready before other modules emit thoughts
            lazy=False,
            heartbeat_timeout=120.0,  # Extended: integrity check + healing takes ~60s
            reply_only=False,     # Receives EPOCH_TICK, TIMECHAIN_COMMIT, etc.
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Backup Worker — promoted from TitanCore._backup_loop per
        # rFP_backup_worker Phase 1 (2026-04-20). Owns RebirthBackup (daily
        # personality + weekly soul + TimeChain + ZK epoch + MyDay NFT).
        # Subscribes to MEDITATION_COMPLETE via bus (was trigger-file handoff).
        #
        # Microkernel v2 §A.3 (S6) mirror: backup is the reference worker
        # for the spawn-vs-fork capability. Mirrors the same flag check
        # in titan_plugin/core/plugin.py so legacy-monolith path and
        # split-kernel path behave identically wrt spawn migration.
        _spawn_ref_legacy = self._full_config.get("microkernel", {}).get(
            "spawn_reference_worker_enabled", False)
        self.guardian.register(ModuleSpec(
            name="backup",
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (on-chain anchoring + 3-2-1 cold storage)
            entry_fn=backup_worker_main,
            config=self._full_config,  # full config — reads [backup]/[network]/[info_banner]/[mainnet_budget]/[memory_and_storage]
            rss_limit_mb=800,     # gzip-9 over ~300MB full-tier personality + Irys subprocess
            autostart=True,       # Must be ready by 1st meditation of the day
            lazy=False,
            heartbeat_timeout=600.0,  # Personality+TimeChain tarball build + upload can take 4-6 min
            reply_only=False,     # Subscribes to MEDITATION_COMPLETE + BACKUP_TRIGGER_MANUAL broadcasts
            start_method="spawn" if _spawn_ref_legacy else "fork",  # S6 (§A.3)
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        ))

        # EMOT-CGN Worker (8th CGN consumer — emotional grounding)
        # rFP_emot_cgn_v2.md §10 ADR: standalone L2 worker, no in-process
        # fallback. Phase 1.6a scaffold; EmotCGNConsumer migration arrives
        # Phase 1.6e — until then, existing in-process EMOT-CGN in
        # meta_reasoning.py continues to own primitive β grounding.
        # Per-Titan shm paths — T2+T3 share the same VPS filesystem, so
        # they MUST have separate shm files or they'd clobber each other's
        # state. titan_id from canonical data/titan_identity.json (same
        # source language_config + most other per-Titan logic uses).
        _emot_titan_id = "T1"
        try:
            import json as _json_id
            _id_path = "data/titan_identity.json"
            if os.path.exists(_id_path):
                with open(_id_path) as _idf:
                    _emot_titan_id = _json_id.load(_idf).get("titan_id", "T1")
        except Exception:
            pass
        self.guardian.register(ModuleSpec(
            name="emot_cgn",
            layer="L2",  # Microkernel v2 §A.5 — L2 CGN consumer (emotional grounding)
            entry_fn=emot_cgn_worker_main,
            config={
                **self._full_config.get("emot_cgn", {}),
                "titan_id": _emot_titan_id,
                "shm_state_path":
                    f"/dev/shm/titan_{_emot_titan_id}/emot_state.bin",
                "shm_grounding_path":
                    f"/dev/shm/titan_{_emot_titan_id}/emot_grounding.bin",
            },
            rss_limit_mb=700,    # was 300 — undersized after HDBSCAN RegionClusterer + felt-tensor 45D wiring + buffer persistence; T2 observed 550-575MB doom-loop 2026-04-24/25
            autostart=True,       # Must be up before meta_reasoning emits EMOT_CHAIN_EVIDENCE
            lazy=False,
            heartbeat_timeout=90.0,
            reply_only=False,    # Subscribes to EMOT_CHAIN_EVIDENCE + FELT_CLUSTER_UPDATE (Phase 1.6d)
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
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
        self._wire_sovereignty()
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

            # Mainnet Lifecycle Wiring rFP (2026-04-20): inject metabolism
            # into Soul post-wire so NFT mint paths can gate on tier.
            try:
                if hasattr(self.soul, "set_metabolism"):
                    self.soul.set_metabolism(metabolism)
                    logger.info("[TitanCore] Metabolism gate injected into Soul")
            except Exception as _se:
                logger.warning("[TitanCore] Soul metabolism injection failed: %s", _se)

            logger.info("[TitanCore] MetabolismController wired (SOL balance + growth metrics)")
        except Exception as e:
            logger.warning("[TitanCore] MetabolismController wiring failed: %s", e)

    def _wire_sovereignty(self) -> None:
        """Wire SovereigntyTracker — GREAT CYCLE convergence tracker (M10).

        Mainnet Lifecycle Wiring rFP (2026-04-20). Instance is dormant until
        spirit_worker starts calling `record_epoch` and reincarnation path
        calls `increment_great_cycle`. Persists state to data/sovereignty_state.json.
        """
        try:
            from .logic.sovereignty import SovereigntyTracker
            tracker = SovereigntyTracker()
            self._proxies["sovereignty"] = tracker
            logger.info(
                "[TitanCore] SovereigntyTracker wired (mode=%s, great_cycle=%d)",
                tracker._sovereignty_mode, tracker._great_cycle)
        except Exception as e:
            logger.warning("[TitanCore] SovereigntyTracker wiring failed: %s", e)

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
    def sovereignty(self):
        return self._proxies.get("sovereignty")

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
            # Option B (2026-04-29): explicit broadcast filter. Legacy
            # path elif chain only handles IMPULSE / OUTER_DISPATCH /
            # QUERY but we use the SAME filter as core/plugin.py so
            # filter-union semantics keep both paths working identically
            # (the additional AGENCY_STATS / ASSESSMENT_STATS / AGENCY_READY
            # types are no-ops on legacy path; they're handled in v2).
            from .bus import (
                AGENCY_READY as _AR, AGENCY_STATS as _AS,
                ASSESSMENT_STATS as _AST, IMPULSE as _IMP,
                OUTER_DISPATCH as _OD, QUERY as _Q,
            )
            self._agency_queue = self.bus.subscribe(
                "agency",
                types=[_IMP, _OD, _Q, _AS, _AST, _AR],
            )

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
            # BUG-KP-WEBSEARCH-HEALTH-DEFAULTS fix (2026-04-21) — forward
            # the same [knowledge_pipeline.budgets] MB→bytes dict that
            # knowledge_worker uses. Without this, WebSearchHelper's
            # HealthTracker had empty defaults and could clobber shared
            # data/knowledge_pipeline_health.json with budget=0 entries.
            _kp_cfg = self._full_config.get("knowledge_pipeline", {}) or {}
            _kp_budgets_mb = _kp_cfg.get("budgets", {}) or {}
            _kp_budgets_bytes = {
                k: int(v) * 1024 * 1024
                for k, v in _kp_budgets_mb.items()
                if isinstance(v, (int, float))
            }
            registry.register(WebSearchHelper(
                searxng_url=searxng_host,
                firecrawl_api_key=firecrawl_key,
                budgets=_kp_budgets_bytes,
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
            # MemoInscribeHelper reads config.toml directly for RPC + keypair.
            # Mainnet Lifecycle Wiring rFP: inject metabolism for memo gate
            # + governance reserve guard.
            registry.register(MemoInscribeHelper(
                metabolism=self._proxies.get("metabolism")))
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

    # ── Chat pipeline (BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS) ──
    async def run_chat(
        self,
        payload: dict,
        claims: dict,
        headers: Optional[dict] = None,
    ) -> dict:
        """Forward to the shared chat pipeline. TitanCore (legacy
        kernel_plugin_split_enabled=false) and TitanPlugin (V6) both
        expose this method so chat.py + the bus chat_handler can
        invoke `await core.run_chat(...)` regardless of runtime path.

        See titan_plugin/api/chat_pipeline.py for the implementation
        + full architectural rationale.
        """
        from titan_plugin.api.chat_pipeline import run_chat as _run_chat
        return await _run_chat(self, payload, claims, headers)

    # ── Chat bus bridge handler (parent-side; CHAT_REQUEST consumer) ──
    async def _chat_handler_loop(self) -> None:
        """Drain `chat_handler` bus subscription; dispatch QUERY action=chat
        to plugin.run_chat() in a separate task; reply RESPONSE rid-routed.

        Identical contract to TitanPlugin._chat_handler_loop. See that
        method's docstring + titan_plugin/api/chat_pipeline.py for the
        full architectural rationale.
        """
        try:
            # Option B (2026-04-29): see core/plugin.py:_chat_handler_loop
            # for full rationale. Same filter on both paths.
            queue = self.bus.subscribe("chat_handler", types=[bus.QUERY])
        except Exception as e:
            logger.warning("[TitanCore] chat_handler subscribe failed: %s", e)
            return
        logger.info("[TitanCore] chat handler loop started — listening for QUERY chat")
        while True:
            try:
                msgs = self.bus.drain(queue, max_msgs=50)
                for msg in msgs:
                    if msg.get("type") != bus.QUERY:
                        continue
                    payload = msg.get("payload") or {}
                    if payload.get("action") != "chat":
                        continue
                    asyncio.get_event_loop().create_task(
                        self._handle_chat_request(msg))
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error("[TitanCore] chat handler loop error: %s", e, exc_info=True)
                await asyncio.sleep(2.0)

    async def _handle_chat_request(self, msg: dict) -> None:
        """Process a single CHAT_REQUEST: dispatch to run_chat, publish
        RESPONSE rid-routed back to requester."""
        rid = msg.get("rid")
        src = msg.get("src", "chat_subproc")
        payload = msg.get("payload") or {}
        body = payload.get("body") or {}
        claims = payload.get("claims") or {}
        headers = payload.get("headers") or {}
        try:
            result = await self.run_chat(body, claims, headers)
        except Exception as e:
            logger.error("[TitanCore] _handle_chat_request raised: %s", e,
                         exc_info=True)
            result = {
                "status_code": 500,
                "body": {"error": f"Chat handler error: {e}"},
                "extra_headers": None,
            }
        try:
            self.bus.publish({
                "type": bus.RESPONSE,
                "src": "chat_handler",
                "dst": src,
                "rid": rid,
                "payload": result,
                "ts": time.time(),
            })
        except Exception as e:
            logger.error("[TitanCore] CHAT_RESPONSE publish failed: %s", e)

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
                elif msg_type == bus.OUTER_DISPATCH:
                    await self._handle_outer_dispatch(msg)
                elif msg_type == bus.QUERY:
                    self._handle_agency_query(msg)

            except Exception as e:
                logger.error("[TitanCore] Agency loop error: %s", e)
                await asyncio.sleep(5.0)

    async def _sovereignty_loop(self) -> None:
        """Listen for SOVEREIGNTY_EPOCH and update plugin.sovereignty.

        Mainnet Lifecycle Wiring rFP (2026-04-20). Every 10th consciousness
        epoch, spirit_worker sends a SOVEREIGNTY_EPOCH message with the
        current neuromod snapshot + dev_age + great_pulse_fired flag.
        Every 100 messages received (≈ 1000 epochs), we additionally log
        the transition-criteria snapshot so soak observers can see the
        long-horizon convergence signal.
        """
        queue = self._sovereignty_queue
        tracker = self._proxies.get("sovereignty")
        if not tracker:
            logger.warning("[Sovereignty] Listener starting with no tracker")
            return
        logger.info("[TitanCore] Sovereignty loop started")
        _msg_count = 0
        while True:
            try:
                msg = None
                try:
                    msg = queue.get_nowait()
                except Exception:
                    pass
                if not msg:
                    await asyncio.sleep(1.0)
                    continue
                msg_type = msg.get("type", "")
                if msg_type == SOVEREIGNTY_EPOCH:
                    payload = msg.get("payload", {}) or {}
                    try:
                        tracker.record_epoch(
                            epoch_id=int(payload.get("epoch_id", 0)),
                            neuromod_levels=payload.get("neuromods", {}) or {},
                            developmental_age=int(payload.get("dev_age", 0)),
                            great_pulse_fired=bool(payload.get("great_pulse_fired", False)),
                        )
                    except Exception as _re:
                        logger.debug("[Sovereignty] record_epoch failed: %s", _re)
                    _msg_count += 1
                    if _msg_count % 100 == 0:
                        try:
                            criteria = tracker.check_transition_criteria()
                            logger.info(
                                "[Sovereignty] Criteria snapshot: mode=%s dev_age=%d "
                                "great_pulses=%d sat_violations=%d collapse_violations=%d "
                                "all_met=%s",
                                criteria.get("sovereignty_mode"),
                                criteria.get("developmental_age"),
                                criteria.get("total_great_pulses"),
                                criteria.get("saturation_violations"),
                                criteria.get("collapse_violations"),
                                criteria.get("all_met"))
                        except Exception as _ce:
                            logger.debug("[Sovereignty] check_transition_criteria failed: %s", _ce)
            except Exception as e:
                logger.error("[TitanCore] Sovereignty loop error: %s", e)
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
                self.bus.publish(make_msg(bus.RATE_LIMIT, "core", "spirit", feedback))
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

        # Publish ACTION_RESULT back to bus (Spirit will pick it up).
        # Guard: 3 SQLite NOT NULL errors per hour surfaced 2026-04-15
        # (ACTION-RESULT-NULL-FIELDS in DEFERRED_ITEMS). Root cause: Agency
        # gated-away impulses (rate limit, unavailable helper) still
        # published ACTION_RESULT with empty helper/task_type/action_taken.
        # 3 downstream recorders (TitanCore inner_memory, SpiritWorker
        # ex_mem, ExperienceOrch) all rejected with NOT NULL constraint
        # failures. Skipping empty dispatches at the source — the action
        # didn't actually execute, so there's nothing to record.
        _helper = str(result.get("helper") or "").strip()
        if not _helper:
            logger.debug(
                "[TitanCore] Skipping ACTION_RESULT with empty helper — "
                "gate/rate-limit path (success=%s)", result.get("success"))
        else:
            self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
            logger.info("[TitanCore] ACTION_RESULT published: helper=%s success=%s",
                        _helper, result.get("success"))

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
                # Coalesce None → "" so `action_chains.helper TEXT NOT NULL`
                # never receives NULL. `.get(key, default)` only honors the
                # default on missing keys, not explicit None values.
                helper_name = result.get("helper") or ""
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
                    # Archive to ObservatoryDB for gallery/feed (in thread — non-blocking)
                    obs_db = getattr(self, "_observatory_db", None)
                    if obs_db and _file_path:
                        _style = result.get("art_style", _work_type)
                        asyncio.get_event_loop().run_in_executor(
                            None, obs_db.record_expressive,
                            _work_type,
                            f"{_style.replace('_', ' ').title()} ({result.get('triggering_program', 'autonomous')})",
                            result.get("result", ""),
                            _file_path,
                            "",
                            {
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

            # Same empty-payload guard as the normal-path publisher above
            # (see ACTION-RESULT-NULL-FIELDS — 3 NOT NULL errors/hour
            # pattern). Autonomy path can also produce empty helpers when
            # the selected posture has no bound action.
            _auto_helper = str(result.get("helper") or "").strip()
            if not _auto_helper:
                logger.debug(
                    "[TitanCore] Skipping AUTONOMY ACTION_RESULT with empty "
                    "helper — posture=%s success=%s",
                    result.get("posture"), result.get("success"))
            else:
                self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
                logger.info("[TitanCore] AUTONOMY ACTION: %s → %s (success=%s)",
                            result.get("posture"), _auto_helper,
                            result.get("success"))

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
            self.bus.publish(make_msg(bus.RESPONSE, "core", src, stats, rid))

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
        V4_EVENT_TYPES = {SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
                          NEUROMOD_UPDATE, HORMONE_FIRED, EXPRESSION_FIRED}
        # Option B (2026-04-29) — broadcast filter at bus level. See the
        # mirrored core/plugin.py:2543 site for full rationale. Legacy_core
        # gets the same treatment so the fix protects both v2 and (rarely
        # used) legacy-monolith fallback paths.
        bridge_queue = self.bus.subscribe("v4_bridge", types=V4_EVENT_TYPES)

        # Wait for Spirit to boot
        await asyncio.sleep(10)
        logger.info("[TitanCore] V4 event bridge started")

        # 2026-04-29 — drain rate tightened (Fix A from bus saturation RCA).
        # Previous cadence (max_msgs=50, sleep 2.0s = 25 msg/sec) could not
        # keep up with cumulative dst="all" broadcast volume from A.8.X
        # subprocess fan-out; bridge_queue overflowed (144k drops on T1 in
        # 1h12m). 1000-msg batch + 0.5s sleep yields ~2000 msg/sec drain
        # capacity vs ~120 msg/sec peak production. Filter (line below)
        # discards non-V4 msgs in <1µs; drain stays cheap regardless of
        # broadcast volume.
        while True:
            try:
                msgs = self.bus.drain(bridge_queue, max_msgs=1000)
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
            await asyncio.sleep(0.5)

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

    def _start_trinity_shm_writer(self) -> None:
        """
        Microkernel v2 Phase A §A.2 — Trinity shm writer (daemon thread).

        Reads state_register's 130D felt + 30D topology + 2D journey
        at ~Schumann/9 cadence (body publish rate), assembles the 162D
        TITAN_SELF vector, and writes it to /dev/shm/titan_{id}/trinity_state.bin
        via StateRegistryWriter (SeqLock + persistent mmap).

        Content-hash-gated: shm seq only bumps when the assembled vector
        actually changes. Under healthy operation, this tracks the natural
        Schumann-derived body/mind publish cadence without needing fixed
        timers.

        Fallback behavior: if TRINITY_STATE feature flag is false, the
        loop still runs (and burns ~nothing) but makes no shm writes —
        readers will fall back to legacy state_register path.
        """
        import hashlib
        import threading as _threading

        import numpy as _np

        from titan_plugin.core.state_registry import TRINITY_STATE

        # Poll slightly faster than body.publish_interval (1.15s = Schumann/9)
        # so we catch each update promptly. Content-hash gates prevents
        # spurious writes when state hasn't changed.
        poll_interval_s = 0.5
        stop_evt = getattr(self, "_shm_writer_stop_evt", None)
        if stop_evt is None:
            stop_evt = _threading.Event()
            self._shm_writer_stop_evt = stop_evt

        def _writer_loop() -> None:
            last_hash: bytes | None = None
            consecutive_errors = 0
            # Wait a beat so state_register has its first bus tick absorbed.
            stop_evt.wait(2.0)
            while not stop_evt.is_set():
                try:
                    if not self._registry_bank.is_enabled(TRINITY_STATE):
                        # Flag off — sleep and check again. Cheap no-op.
                        stop_evt.wait(poll_interval_s)
                        continue

                    # Assemble 162D = 130D felt + 30D topology + 2D journey.
                    felt_130 = self.state_register.get_full_130dt()
                    topo_30 = self.state_register.get_full_30d_topology()
                    snapshot = self.state_register.snapshot()
                    consciousness = snapshot.get("consciousness", {}) or {}
                    journey_2 = [
                        float(consciousness.get("curvature", 0.0)),
                        float(consciousness.get("density", 0.0)),
                    ]
                    # Ensure exact-length lists (get_full_* guarantees this).
                    values = (list(felt_130)[:130]
                              + list(topo_30)[:30]
                              + journey_2[:2])
                    if len(values) != 162:
                        # Defensive — should never happen given get_full_* contracts.
                        consecutive_errors += 1
                        if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                            logger.warning(
                                "[TrinityShmWrite] assembled length %d != 162; skipping",
                                len(values))
                        stop_evt.wait(poll_interval_s)
                        continue

                    arr = _np.asarray(values, dtype=_np.float32)
                    payload_bytes = arr.tobytes(order="C")
                    h = hashlib.blake2b(payload_bytes, digest_size=16).digest()
                    if h != last_hash:
                        self._registry_bank.writer(TRINITY_STATE).write(arr)
                        last_hash = h
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 20 == 0:
                        logger.warning(
                            "[TrinityShmWrite] iteration failed (#%d): %s",
                            consecutive_errors, e, exc_info=True)
                stop_evt.wait(poll_interval_s)

        t = _threading.Thread(
            target=_writer_loop,
            daemon=True,
            name="trinity-shm-writer",
        )
        t.start()
        logger.info(
            "[TitanCore] Trinity shm writer thread started (poll=%.2fs, gate=microkernel.shm_trinity_enabled)",
            poll_interval_s)

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

                # Record Trinity snapshot (in thread to avoid blocking event loop)
                await asyncio.to_thread(
                    obs_db.record_trinity_snapshot,
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

                await asyncio.to_thread(
                    obs_db.record_growth_snapshot,
                    learning_velocity=learning_velocity,
                    social_density=social_density,
                    metabolic_health=metabolic_health,
                    directive_alignment=directive_alignment,
                )

                # Record V4 Time Awareness snapshot (spirit_data already has it)
                if spirit_data.get("sphere_clock") or spirit_data.get("unified_spirit"):
                    await asyncio.to_thread(
                        obs_db.record_v4_snapshot,
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
                    await asyncio.to_thread(
                        obs_db.record_vital_snapshot,
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
                    if isinstance(_raw_msg, dict) and _raw_msg.get("type") == bus.MEDITATION_REQUEST:
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
                                # Archive to ObservatoryDB (in thread — non-blocking)
                                obs_db = getattr(self, "_observatory_db", None)
                                if obs_db:
                                    asyncio.get_event_loop().run_in_executor(
                                        None, obs_db.record_expressive,
                                        "art",
                                        f"Meditation Flow Field (V3 Epoch {epoch_count})",
                                        f"{promoted} memories crystallized",
                                        art_path,
                                        "",
                                        {"epoch": epoch_count, "promoted": promoted},
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
                    bus.MEDITATION_COMPLETE, "core", "spirit",
                    _med_payload,
                ))
                # ── Notify timechain_worker to seal genesis block ──
                self.bus.publish(make_msg(
                    bus.MEDITATION_COMPLETE, "core", "timechain",
                    _med_payload,
                ))
                # ── Notify backup_worker (rFP_backup_worker Phase 1, 2026-04-20) ──
                # Replaces prior data/backup_trigger.json file handoff.
                self.bus.publish(make_msg(
                    bus.MEDITATION_COMPLETE, "core", "backup",
                    _med_payload,
                ))

            except Exception as e:
                logger.error("[TitanCore] Meditation loop error: %s", e)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Sovereign Backup — PROMOTED 2026-04-20 per rFP_backup_worker Phase 1
    # ------------------------------------------------------------------
    # TitanCore._backup_loop was deleted in the promotion to a Guardian-
    # supervised subprocess module. See titan_plugin/modules/backup_worker.py
    # for the live backup pipeline + §5.3 failsafe cascade. The subprocess
    # subscribes to MEDITATION_COMPLETE via bus (dst="backup"); the
    # data/backup_trigger.json file handoff is retired.

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

                # Publish to bus → Spirit worker receives and ticks outer
                # sphere clocks; state_register also subscribes (broadcast)
                # so kernel snapshot can populate spirit.coordinator
                # outer_trinity cache key (2026-04-26 sweep).
                self.bus.publish(make_msg(
                    OUTER_TRINITY_STATE, "core", "all", result,
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
                    make_msg(bus.QUERY, "core", "spirit",
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

        # ── Phase 1 sensory wiring (rFP_phase1_sensory_wiring) ─────────
        # Rich-signal producers for V6 5DT outer_body composites. All
        # getters are sample-on-demand with TTL caches; each wrapped in
        # try/except so a single failing producer never breaks the loop.
        try:
            from titan_plugin.utils import system_sensor as _sys_sensor
            sources["system_sensor_stats"] = _sys_sensor.get_all_stats()
        except Exception as _se:
            swallow_warn('[OuterTrinity] system_sensor unavailable', _se,
                         key="legacy_core.system_sensor_unavailable", throttle=100)

        try:
            from titan_plugin.utils import network_monitor as _net_mon
            _rpc_url = None
            if hasattr(self, "network") and self.network is not None:
                _rpc_urls = getattr(self.network, "rpc_urls", None) or []
                _rpc_url = _rpc_urls[0] if _rpc_urls else None
            sources["network_monitor_stats"] = _net_mon.get_all_stats(
                rpc_url=_rpc_url,
                bus_stats=sources.get("bus_stats"),
            )
        except Exception as _ne:
            swallow_warn('[OuterTrinity] network_monitor unavailable', _ne,
                         key="legacy_core.network_monitor_unavailable", throttle=100)

        try:
            from titan_plugin.logic.timechain_v2 import (
                get_tx_latency_stats, get_block_delta_stats,
            )
            sources["tx_latency_stats"] = get_tx_latency_stats()
            sources["block_delta_stats"] = get_block_delta_stats()
        except Exception as _te:
            swallow_warn('[OuterTrinity] timechain_v2 stats unavailable', _te,
                         key="legacy_core.timechain_v2_stats_unavailable", throttle=100)

        # SOL balance (from data/last_balance.txt, written by metabolism)
        try:
            _bal_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "last_balance.txt")
            if os.path.exists(_bal_path):
                with open(_bal_path) as _bf:
                    sources["sol_balance"] = float(_bf.read().strip())
        except Exception:
            pass

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
        """Load the full merged Titan config (config.toml + ~/.titan/secrets.toml)."""
        from titan_plugin.config_loader import load_titan_config
        return load_titan_config()

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
