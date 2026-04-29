"""
titan_plugin/core/plugin.py — TitanPlugin (L1-L3 coordinator).

Thin coordinator holding a `kernel: TitanKernel` reference plus the L2/L3
state and loops:
  - Proxies (memory/rl/llm/body/mind/spirit/media/timechain + 13 @property facades)
  - Agency subsystem (autonomous action pipeline)
  - OutputVerifier (external output gate)
  - Outer trinity collector
  - EventBus + ObservatoryDB + observatory FastAPI app
  - Agno agent
  - Dream inbox (API-side message queue during dream cycles)
  - All async loops: agency, sovereignty, meditation, outer_trinity,
    v4_event_bridge, trinity_snapshot, social_engagement

Uses the kernel for all L0 services (bus, guardian, state_register,
registry_bank, soul, network, identity). The kernel never restarts;
this coordinator may be replaced or re-attached during Phase B shadow
core swap.

Compat @property facade (bus, guardian, soul, _full_config, ...) makes
TitanPlugin duck-type-identical to the legacy TitanCore for dashboard +
agent code. Zero dashboard code changes required.

This commit (#3 — plugin skeleton + module registration + proxies)
lands the __init__, compat properties, _register_modules (380-line
module catalog lifted from v5_core.py:317-696), and _create_proxies.
Wire helpers + observatory + agency + async loops + boot orchestration
arrive in commits 4-6.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §A.1
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.2 + §3 D1+D9+D10
  - titan_plugin/core/kernel.py (the L0 paired class)
"""
import asyncio
import logging
import os
import time
from typing import Optional

from titan_plugin.bus import (
    ACTION_RESULT,
    AGENCY_READY,
    AGENCY_STATS,
    ASSESSMENT_STATS,
    EPOCH_TICK,
    IMPULSE,
    OUTER_OBSERVATION,
    OUTER_TRINITY_COLLECT_REQUEST,
    OUTER_TRINITY_STATE,
    SAGE_STATS,
    SOVEREIGNTY_EPOCH,
    make_msg,
)
from titan_plugin.core.kernel import TitanKernel
from titan_plugin.guardian import ModuleSpec
from titan_plugin.utils.silent_swallow import swallow_warn
from titan_plugin import bus

logger = logging.getLogger(__name__)


# PERSISTENCE_BY_DESIGN: TitanPlugin._proxies / _agency / _*_mode fields
# are runtime bootstrap state — constructed from kernel + config at boot.
# Proxy objects are not self-owned state to persist.
class TitanPlugin:
    """
    L1-L3 Coordinator — owns proxies, agency, observatory, agno, dream inbox.

    Usage (from scripts/titan_main.py flag-branch per PLAN §4.4):
        kernel = TitanKernel(wallet_path)
        plugin = TitanPlugin(kernel)
        await plugin.boot()  # orchestrates kernel.boot() + module wiring

    Compat shape: dashboard.py + agent code treat TitanPlugin as the
    "plugin root" with @property accessors (bus, guardian, soul, memory,
    metabolism, ...) that match the legacy TitanCore surface. This is
    intentional — S3 preserves full duck-type compatibility so zero
    downstream code changes are needed during the cutover.
    """

    def __init__(self, kernel: TitanKernel):
        self.kernel = kernel

        # ── Proxy stubs (populated by _create_proxies during boot) ──
        self._proxies: dict[str, object] = {}

        # ── Agency Module (Step 7) ─────────────────────────────────
        self._agency = None
        self._agency_assessment = None
        self._interface_advisor = None

        # ── V4 Outer Trinity Collector ────────────────────────────────
        self._outer_trinity_collector = None

        # ── Output Verification Gate (security for all external outputs) ──
        # L3 §A.8.3: when microkernel.a8_output_verifier_subprocess_enabled
        # is true, the OV runs in a subprocess (output_verifier_worker) and
        # parent uses an OutputVerifierProxy that bus.request()s the worker.
        # When false (default), parent retains the local OutputVerifier
        # instance — byte-identical to pre-A.8.3 behavior.
        self._output_verifier = None
        _ov_subproc_enabled = bool(
            kernel.config.get("microkernel", {}).get(
                "a8_output_verifier_subprocess_enabled", False))
        if _ov_subproc_enabled:
            try:
                from titan_plugin.proxies.output_verifier_proxy import OutputVerifierProxy
                self._output_verifier = OutputVerifierProxy(self.bus)
                logger.info("[TitanPlugin] OutputVerifier using subprocess proxy "
                            "(A.8.3 flag enabled)")
            except Exception as _ovg_err:
                logger.warning("[TitanPlugin] OutputVerifierProxy init failed: %s",
                               _ovg_err)
        else:
            try:
                from titan_plugin.logic.output_verifier import OutputVerifier
                _tc_dir = os.path.join("data", "timechain")
                _titan_id = kernel.config.get("info_banner", {}).get("titan_id") or kernel.titan_id
                _wallet_path = kernel.config.get("network", {}).get(
                    "wallet_keypair_path", "data/titan_identity_keypair.json")
                self._output_verifier = OutputVerifier(
                    titan_id=_titan_id, data_dir=_tc_dir, keypair_path=_wallet_path)
            except Exception as _ovg_err:
                logger.warning("[TitanPlugin] OutputVerifier init failed: %s", _ovg_err)

        # ── State ────────────────────────────────────────────────────
        self._last_execution_mode = "Shadow"
        self._is_meditating = False
        self._background_tasks_started = False
        self._observatory_app = None
        self._agent = None

        # ── Dream-aware message queue (API process) ──────────────────
        # L3 Phase A.8.1: deque(maxlen=256) bounds inbox at data-structure
        # level (defense-in-depth above chat.py's API-side 50-cap). On
        # overflow, oldest message is silently evicted by deque semantics.
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

        # EventBus / ObservatoryDB — populated by boot()
        self.event_bus = None
        self._observatory_db = None

        logger.info(
            "[TitanPlugin] Coordinator constructed (kernel_id=%s, limbo=%s)",
            kernel.titan_id, kernel.limbo_mode,
        )

    # ------------------------------------------------------------------
    # L0 compat @property facade (delegate to kernel)
    # ------------------------------------------------------------------
    # Dashboard + agent code accesses these on the plugin root. Routing
    # them through the kernel preserves the legacy TitanCore duck-type
    # without requiring the plugin to own any L0 state.

    @property
    def bus(self):
        return self.kernel.bus

    @property
    def guardian(self):
        return self.kernel.guardian

    @property
    def state_register(self):
        return self.kernel.state_register

    @property
    def registry_bank(self):
        return self.kernel.registry_bank

    @property
    def soul(self):
        return self.kernel.soul

    @property
    def network(self):
        return self.kernel.network

    @property
    def disk_health(self):
        return self.kernel.disk_health

    @property
    def bus_health(self):
        return self.kernel.bus_health

    @property
    def _full_config(self):
        """Legacy name preserved so _register_modules / _wire_* methods
        that reference `self._full_config` work verbatim when lifted
        from v5_core.py. New code should prefer `self.kernel.config`.
        """
        return self.kernel.config

    @property
    def _limbo_mode(self):
        return self.kernel.limbo_mode

    @property
    def _start_time(self):
        """Legacy name — used by any code that previously read
        TitanCore._start_time for uptime calculations.
        """
        return self.kernel._start_time

    # ------------------------------------------------------------------
    # Plugin @property accessors (13 proxy facades — lifted verbatim)
    # ------------------------------------------------------------------
    # Allow existing Observatory API and agent code to work with
    # TitanPlugin as if it were TitanPlugin. Returns None when module
    # not loaded, so endpoints can degrade gracefully.

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
    # Module Registration — lifted verbatim from v5_core.py:317-696
    # ------------------------------------------------------------------

    def _register_modules(self) -> None:
        """Register all module specs with the Guardian. None start yet (lazy=True).

        Lifted verbatim from v5_core.py:317-696 per PLAN §4.2 Commit 3.
        `self.guardian` and `self._full_config` work via compat
        @property delegates to the kernel — zero inside-method changes.

        Log prefix [TitanCore] preserved in messages that reference
        prior incidents (2026-04-08 T3 crash-loop, 2026-04-17 fork
        memory growth) because those logs are historical records.
        """
        from titan_plugin.modules.memory_worker import memory_worker_main
        from titan_plugin.modules.rl_worker import rl_worker_main
        from titan_plugin.modules.llm_worker import llm_worker_main
        from titan_plugin.modules.body_worker import body_worker_main
        from titan_plugin.modules.mind_worker import mind_worker_main
        from titan_plugin.modules.spirit_worker import spirit_worker_main
        from titan_plugin.modules.media_worker import media_worker_main
        from titan_plugin.modules.language_worker import language_worker_main
        from titan_plugin.modules.meta_teacher_worker import meta_teacher_worker_main
        from titan_plugin.modules.cgn_worker import cgn_worker_main
        from titan_plugin.modules.knowledge_worker import knowledge_worker_main
        from titan_plugin.modules.emot_cgn_worker import emot_cgn_worker_main
        from titan_plugin.modules.timechain_worker import timechain_worker_main
        from titan_plugin.modules.backup_worker import backup_worker_main
        from titan_plugin.persistence_entry import imw_main

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

        # Output Verifier Worker — L2 §A.8.3 subprocess extraction.
        # Hosts the OutputVerifier instance + handles QUERY(action=
        # "verify_and_sign"|"build_timechain_payload"|"stats") over the
        # bus. Default OFF — when flag flips, parent's _output_verifier
        # becomes an OutputVerifierProxy (see core/plugin.py:__init__).
        from titan_plugin.modules.output_verifier_worker import (
            output_verifier_worker_main,
        )
        _ov_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_output_verifier_subprocess_enabled", False))
        self.guardian.register(ModuleSpec(
            name="output_verifier",
            layer="L2",  # security/verification gate — L2 service
            entry_fn=output_verifier_worker_main,
            config=self._full_config,
            rss_limit_mb=400,    # Ed25519 + regex patterns + signature chain
            autostart=_ov_subproc_enabled,  # Only when flag flipped
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=False,
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # not on hot ARC path; respawn-OK
        ))

        # Outer Trinity Worker — L1 §A.8.4 subprocess extraction.
        # Hosts the OuterTrinityCollector instance + computes 132D
        # outer trinity tensors on each OUTER_TRINITY_COLLECT_REQUEST
        # from parent. Publishes OUTER_TRINITY_STATE for downstream
        # consumers (state_register, dashboard, spirit_worker). Default
        # OFF — when flag flips, parent's _outer_trinity_loop publishes
        # COLLECT_REQUEST instead of computing locally (see plugin.py:
        # _outer_trinity_loop).
        from titan_plugin.modules.outer_trinity_worker import (
            outer_trinity_worker_main,
        )
        _ot_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_outer_trinity_subprocess_enabled", False))
        self.guardian.register(ModuleSpec(
            name="outer_trinity",
            layer="L1",  # outer-trinity sister to body/mind/spirit per rFP §A.5
            entry_fn=outer_trinity_worker_main,
            config=self._full_config,
            rss_limit_mb=300,    # pure compute; no DB / model state
            autostart=_ot_subproc_enabled,  # Only when flag flipped
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=False,
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # 60s cadence — respawn-OK
        ))

        # Reflex Worker — L3 §A.8.5 subprocess extraction.
        # Hosts a stateless ReflexCollector (no executors registered)
        # that performs the aggregation step on each QUERY(action=
        # "aggregate") from the parent's ReflexProxy. Cooldowns are
        # synced from each request's payload (parent owns the truth).
        # Default OFF — when flag flips, parent's reflex_collector
        # becomes a ReflexProxy (see _boot_reflex_collector + proxies/
        # reflex_proxy.py).
        from titan_plugin.modules.reflex_worker import reflex_worker_main
        _reflex_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_reflex_subprocess_enabled", False))
        self.guardian.register(ModuleSpec(
            name="reflex",
            layer="L3",  # reflex aggregation — L3 service per rFP §A.8.5
            entry_fn=reflex_worker_main,
            config=self._full_config,
            rss_limit_mb=300,    # stateless aggregator; tiny footprint
            autostart=_reflex_subproc_enabled,  # Only when flag flipped
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=False,
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # per-call processor; respawn-OK
        ))

        # Agency Worker — L3 §A.8.6 subprocess extraction.
        # Hosts AgencyModule + SelfAssessment + HelperRegistry + 8 helpers
        # + LLM fn. Handles QUERY(action="handle_intent"|
        # "dispatch_from_nervous_signals"|"assess"|"agency_stats"|
        # "assessment_stats") via bus.request — async-friendly proxies in
        # parent (asyncio.to_thread) keep the parent event loop unblocked
        # during the worker's LLM round-trip. Default OFF — when flag flips,
        # parent's _agency / _agency_assessment become AgencyProxy +
        # AssessmentProxy (see core/plugin.py:_boot_agency).
        from titan_plugin.modules.agency_worker import agency_worker_main
        _ag_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_agency_subprocess_enabled", False))
        self.guardian.register(ModuleSpec(
            name="agency_worker",
            layer="L3",  # impulse decoder + helper execution — L3 service
            entry_fn=agency_worker_main,
            config=self._full_config,
            # 8 helpers + LLM client + httpx + venice fallback + audio/art
            # buffers + sandbox subprocess — generous ceiling, the LLM fn
            # itself is light but helper.execute() can briefly spike.
            rss_limit_mb=600,
            autostart=_ag_subproc_enabled,  # Only when flag flipped
            lazy=False,
            heartbeat_timeout=120.0,  # LLM call can take 60s legitimately
            reply_only=False,         # needs broadcasts (none currently consumed)
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # impulse decoder is not on swap critical path
        ))

        # Warning Monitor Worker — Pattern C force multiplier (rFP-less
        # infrastructure; created 2026-04-25 in response to BUG-T1-
        # CONSCIOUSNESS-67D-STATE-VECTOR which had been silently active
        # 37 days because of widespread `except: logger.debug(...)` swallows).
        # Tails brain log + ingests SILENT_SWALLOW_REPORT bus messages,
        # aggregates WARNING+ events, persists to data/warning_monitor/.
        # arch_map warnings reads its state at session start.
        from titan_plugin.modules.warning_monitor_worker import (
            warning_monitor_worker_main,
        )
        _wm_cfg = self._full_config.get("warning_monitor", {})
        self.guardian.register(ModuleSpec(
            name="warning_monitor",
            layer="L3",  # observability service — same tier as observatory
            entry_fn=warning_monitor_worker_main,
            config=_wm_cfg,
            rss_limit_mb=300,    # tail + small in-memory aggregations only
            autostart=True,       # Always on — visibility is non-negotiable
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

        # ─────────────────────────────────────────────────────────────────
        # rFP_universal_sqlite_writer 2026-04-27 — additional writer daemons
        # for multi-process-contention DBs (social_graph + events_teacher +
        # consciousness). Same imw_main entry, per-DB config sections.
        # ─────────────────────────────────────────────────────────────────
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
        # ── §A.8.7 (2026-04-28) — autostart when Sage Scholar/Gatekeeper
        # consolidation flag is on. Pre-warms rl_worker at kernel boot so the
        # first /chat after restart doesn't hit a cold-boot timeout when
        # RLProxy.decide_execution_mode bus-routes (5s timeout < ~3-5s
        # LazyMemmapStorage init = first call falls through to Shadow
        # fallback). With autostart on, rl module is READY before chat
        # path's first decide_execution_mode call. Mirrors A.8.3/4/5/6 flag-bound
        # autostart pattern.
        _a8_sage_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_sage_scholar_gatekeeper_subprocess_enabled", False))
        self.guardian.register(ModuleSpec(
            name="rl",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (IQL chain learning)
            entry_fn=rl_worker_main,
            config=self._full_config.get("stealth_sage", {}),
            rss_limit_mb=3000,
            autostart=_a8_sage_subproc_enabled,  # §A.8.7: autostart when flag-on
            lazy=not _a8_sage_subproc_enabled,   # §A.8.7: eager when flag-on
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
            # Microkernel v2 Phase A §A.7 / §L1 — shm feature flags. Must
            # be passed through so body_worker's _read_flag() and
            # RegistryBank.is_enabled() can resolve
            # microkernel.shm_body_fast_enabled correctly. Same pattern
            # as spirit_worker (mirror the comment there). Without this,
            # the body shm fast-path is silently no-op even with the
            # flag flipped true in titan_params.toml.
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
            # (same passthrough pattern as body/spirit; without this,
            # mind shm fast-path is silently no-op).
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
                os.path.dirname(__file__), "..", "..", "data", "media_queue"
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
        # Microkernel v2 §A.3 (S6): backup is the reference worker for the
        # spawn-vs-fork capability. When spawn_reference_worker_enabled=true,
        # this worker boots via spawn (fresh interpreter, ~200 MB RSS savings).
        # Default fork preserves byte-identical pre-S6 behavior.
        _spawn_ref = self._full_config.get("microkernel", {}).get(
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
            start_method="spawn" if _spawn_ref else "fork",  # S6 (§A.3)
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

        logger.info("[TitanPlugin] Registered %d supervised modules", len(self.guardian._modules))

    # ------------------------------------------------------------------
    # Microkernel v2 §A.4 (S5) — api_subprocess module registration
    # ------------------------------------------------------------------

    def _register_api_subprocess_module(self) -> None:
        """Register the api_subprocess as a Guardian-supervised L3 module
        when api_process_separation_enabled flag is on.

        Standard worker spec — Guardian spawns it via multiprocessing.Process,
        gives it (recv_queue, send_queue), heartbeat-monitors it, restarts
        on crash. Behaves like every other L3 module.

        autostart=True (when flag on) so Guardian.start_all() spawns it
        immediately during plugin.boot() Phase 3 — at which point the
        kernel_rpc server is already listening (started by kernel.boot()
        in Phase 2). The api_subprocess connects via HMAC handshake and
        begins serving uvicorn.

        No-op when flag off — legacy in-process _start_observatory path
        runs in plugin.boot() Phase 5 instead.

        rss_limit_mb=300 — generous ceiling for FastAPI + uvicorn + a
        few hundred plugin proxy responses cached. Standard L3 limit
        (matches knowledge worker, language worker).

        heartbeat_timeout=60s — standard L3. uvicorn never goes silent
        under normal operation; a 60s gap means the subprocess hung
        and Guardian should restart it.
        """
        flag_on = self._full_config.get("microkernel", {}).get(
            "api_process_separation_enabled", False)
        if not flag_on:
            logger.info(
                "[TitanPlugin] api_subprocess NOT registered "
                "(microkernel.api_process_separation_enabled=False) — "
                "legacy in-process uvicorn will start in Phase 5")
            return

        from titan_plugin.api.api_subprocess import api_subprocess_main
        api_cfg = self._full_config.get("api", {})
        # Pass relevant config sub-tree to the subprocess; it doesn't need
        # the full plugin config (most state comes via kernel_rpc anyway).
        # rFP_observatory_data_loading_v1 §3.3 (2026-04-26): added `network`
        # so /health + /status can read vault_program_id (without it,
        # _fetch_vault_info short-circuits and STATE_ROOT_ZK / On-Chain Vault
        # / Integrity Verification all show STUB / "No vault data").
        sub_config = {
            "api": api_cfg,
            # microkernel block forwarded for any subprocess-side flag
            # checks (e.g., during reload).
            "microkernel": self._full_config.get("microkernel", {}),
            # network block carries vault_program_id, RPC URLs, premium_rpc
            # — used by vault PDA derivation + /health vault check.
            "network": self._full_config.get("network", {}),
            # mainnet/devnet flag and other env-related settings the api
            # endpoints may need (frontend mode toggles).
            "frontend": self._full_config.get("frontend", {}),
        }
        self.guardian.register(ModuleSpec(
            name="api",
            entry_fn=api_subprocess_main,
            config=sub_config,
            rss_limit_mb=300,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            layer="L3",
        ))
        logger.info(
            "[TitanPlugin] api_subprocess registered as L3 module "
            "(api_process_separation_enabled=True; uvicorn moves to subprocess)")

    # ------------------------------------------------------------------
    # Proxy Creation — lifted verbatim from v5_core.py:697-735
    # ------------------------------------------------------------------

    def _create_proxies(self) -> None:
        """Create proxy objects that bridge V2 API calls to V3 bus-supervised modules.

        Lifted verbatim from v5_core.py:697-735 per PLAN §4.2 Commit 3.
        `self.bus` and `self.guardian` work via compat @property delegates.
        """
        from titan_plugin.proxies.memory_proxy import MemoryProxy
        from titan_plugin.proxies.rl_proxy import RLProxy
        from titan_plugin.proxies.llm_proxy import LLMProxy
        from titan_plugin.proxies.mind_proxy import MindProxy
        from titan_plugin.proxies.body_proxy import BodyProxy
        from titan_plugin.proxies.spirit_proxy import SpiritProxy
        from titan_plugin.proxies.media_proxy import MediaProxy
        from titan_plugin.proxies.timechain_proxy import TimechainProxy

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
        # _wire_metabolism/sovereignty/studio/social arrive in commit 4.
        self._wire_metabolism()
        self._wire_sovereignty()
        self._wire_studio()
        self._wire_social()

        logger.info("[TitanPlugin] Created %d proxies", len(self._proxies))

    # ------------------------------------------------------------------
    # V2 Subsystem wiring — lifted verbatim from v5_core.py:737-869
    # ------------------------------------------------------------------

    def _wire_metabolism(self) -> None:
        """Wire MetabolismController directly in Core (lightweight — SOL balance + growth metrics).

        Lifted verbatim from v5_core.py:737-793 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.core.metabolism import MetabolismController
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
                    logger.info("[TitanPlugin] Metabolism gate injected into Soul")
            except Exception as _se:
                logger.warning("[TitanPlugin] Soul metabolism injection failed: %s", _se)

            logger.info("[TitanPlugin] MetabolismController wired (SOL balance + growth metrics)")
        except Exception as e:
            logger.warning("[TitanPlugin] MetabolismController wiring failed: %s", e)

    def _wire_sovereignty(self) -> None:
        """Wire SovereigntyTracker — GREAT CYCLE convergence tracker (M10).

        Mainnet Lifecycle Wiring rFP (2026-04-20). Instance is dormant until
        spirit_worker starts calling `record_epoch` and reincarnation path
        calls `increment_great_cycle`. Persists state to data/sovereignty_state.json.

        Lifted verbatim from v5_core.py:795-810 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.logic.sovereignty import SovereigntyTracker
            tracker = SovereigntyTracker()
            self._proxies["sovereignty"] = tracker
            logger.info(
                "[TitanPlugin] SovereigntyTracker wired (mode=%s, great_cycle=%d)",
                tracker._sovereignty_mode, tracker._great_cycle)
        except Exception as e:
            logger.warning("[TitanPlugin] SovereigntyTracker wiring failed: %s", e)

    def _wire_studio(self) -> None:
        """Wire StudioCoordinator directly in Core (lightweight creative engine).

        Lifted verbatim from v5_core.py:812-836 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.expressive.studio import StudioCoordinator
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
                    from titan_plugin.utils.ollama_cloud import OllamaCloudClient
                    studio._ollama_cloud = OllamaCloudClient(
                        api_key=ollama_key,
                        base_url=inference_cfg.get("ollama_cloud_base_url", "https://ollama.com/v1"),
                    )
                except Exception:
                    pass
            self._proxies["studio"] = studio
            logger.info("[TitanPlugin] StudioCoordinator wired (%s)", exp_cfg.get("output_path", "./data/studio_exports"))
        except Exception as e:
            logger.warning("[TitanPlugin] StudioCoordinator wiring failed: %s", e)

    def _wire_social(self) -> None:
        """Wire SocialManager in Core (degraded mode — no API keys, but structure in place).

        Lifted verbatim from v5_core.py:838-869 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.expressive.social import SocialManager
            sage_cfg = self._full_config.get("stealth_sage", {})
            # Wire SocialGraph for persistent user profile tracking
            social_graph = None
            try:
                from titan_plugin.core.social_graph import SocialGraph
                data_dir = self._full_config.get("memory_and_storage", {}).get("data_dir", "./data")
                social_graph = SocialGraph(db_path=os.path.join(data_dir, "social_graph.db"))
            except Exception as _sg_err:
                logger.debug("[TitanPlugin] SocialGraph init: %s", _sg_err)
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
            logger.info("[TitanPlugin] SocialManager wired (dry_run=%s)", social._dry_run)
        except Exception as e:
            logger.warning("[TitanPlugin] SocialManager wiring failed: %s", e)

    # ------------------------------------------------------------------
    # Observatory API (reuses existing V2 create_app)
    # ------------------------------------------------------------------

    def _create_observatory_app(self, api_cfg: dict):
        """Create the Observatory FastAPI app synchronously.

        Lifted verbatim from v5_core.py:935-944 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.api import create_app
            app = create_app(self, self.event_bus, api_cfg)
            self._observatory_app = app
            return app
        except Exception as e:
            logger.warning("[TitanPlugin] Observatory app creation failed: %s", e)
            return None

    async def _start_observatory(self, api_cfg: dict):
        """Launch the Observatory API server.

        Lifted verbatim from v5_core.py:946-963 per PLAN §4.2 Commit 4.
        """
        try:
            import uvicorn
            app = self._observatory_app
            if app is None:
                return
            host = api_cfg.get("host", "0.0.0.0")
            # Microkernel v2 Phase B.1 — TITAN_API_PORT env var (set by shadow
            # orchestrator's _phase_shadow_boot) overrides config.toml port,
            # so the shadow binds to its assigned shadow_port (7779/7777
            # ping-pong) instead of always 7777. Without this, the shadow
            # collides with the running old kernel and `address already in
            # use`. Codified 2026-04-28 PM during T1 swap E2E test.
            env_port = os.environ.get("TITAN_API_PORT")
            port = int(env_port) if env_port else int(api_cfg.get("port", 7777))
            uvi_config = uvicorn.Config(
                app=app, host=host, port=port, log_level="info", access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(uvi_config)
            await self._uvicorn_server.serve()
        except SystemExit:
            logger.warning("[TitanPlugin] Observatory could not bind port")
        except Exception as e:
            logger.warning("[TitanPlugin] Observatory failed: %s", e)

    def reload_api(self) -> dict:
        """Hot-reload API routes by rebuilding the FastAPI app and swapping it.

        Returns dict with reload status. The uvicorn server keeps running —
        only the ASGI app reference changes. Zero downtime.

        Lifted verbatim from v5_core.py:965-988 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_plugin.api import reload_api_app
            old_app = self._observatory_app
            new_app = reload_api_app(old_app)
            self._observatory_app = new_app

            # Swap the app in the running uvicorn server
            if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
                self._uvicorn_server.config.app = new_app
                # Also update the loaded_app which uvicorn uses for serving
                if hasattr(self._uvicorn_server, 'config'):
                    self._uvicorn_server.config.loaded_app = new_app

            logger.info("[TitanPlugin] API hot-reloaded — routes updated, server continuous")
            return {"status": "ok", "reloaded": True}
        except Exception as e:
            logger.error("[TitanPlugin] API reload failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Agency Module (Step 7)
    # ------------------------------------------------------------------

    def _boot_reflex_collector(self) -> None:
        """Initialize the Sovereign Reflex Collector with executors.

        A.8.5 — flag-aware: when microkernel.a8_reflex_subprocess_enabled
        is true, parent's reflex_collector becomes a ReflexProxy that
        bus-routes the aggregation step (steps 1-4 of collect_and_fire:
        group → guardian-shield → combine → threshold + cooldown filter
        → top-N) to reflex_worker. Executors stay parent-resident (they
        reference plugin.soul / plugin.metabolism / plugin.memory_proxy
        / plugin.knowledge_proxy / plugin.social_proxy / plugin.agency
        and cannot trivially move to a subprocess) and run locally via
        the inherited _execute_selected step.

        When false (default), parent retains the regular ReflexCollector
        — byte-identical to pre-A.8.5 behavior.
        """
        try:
            from titan_plugin.logic.reflex_executors import register_reflex_executors
            from titan_plugin.params import get_params

            reflex_cfg = get_params("reflexes")

            _reflex_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_reflex_subprocess_enabled", False))
            if _reflex_subproc_enabled:
                from titan_plugin.proxies.reflex_proxy import ReflexProxy
                self.reflex_collector = ReflexProxy(self.bus, reflex_cfg)
                logger.info(
                    "[TitanPlugin] ReflexCollector using subprocess proxy "
                    "(A.8.5 flag enabled)")
            else:
                from titan_plugin.logic.reflexes import ReflexCollector
                self.reflex_collector = ReflexCollector(reflex_cfg)

            # Register executors — they wrap existing subsystems. Works
            # on both ReflexCollector + ReflexProxy (proxy inherits
            # register_executor + _executors from base).
            count = register_reflex_executors(self.reflex_collector, self)
            logger.info("[TitanPlugin] ReflexCollector booted: %d executors, threshold=%.2f",
                        count, self.reflex_collector.fire_threshold)
        except Exception as e:
            logger.warning("[TitanPlugin] ReflexCollector boot failed: %s", e)
            self.reflex_collector = None

    def _boot_agency(self) -> None:
        """Initialize Agency — local module OR subprocess proxy (L3 §A.8.6).

        Original behavior (flag off, default): instantiates AgencyModule +
        SelfAssessment + HelperRegistry + 8 helpers in parent.

        Subprocess mode (microkernel.a8_agency_subprocess_enabled=true):
        instantiates AgencyProxy + AssessmentProxy that bus.request() into
        agency_worker — all LLM calls + helper.execute() awaits run in
        the worker subprocess, parent event loop never blocks on them.
        InterfaceAdvisor stays in parent (cheap rate check before bus
        round-trip). ExpressionTranslator stays in parent — it only needs
        the helper-names list (worker advertises via AGENCY_READY +
        AGENCY_STATS broadcast → proxy._registry facade).
        """
        try:
            from titan_plugin.logic.interface_advisor import InterfaceAdvisor

            agency_cfg = self._full_config.get("agency", {})
            if not agency_cfg.get("enabled", True):
                logger.info("[TitanPlugin] Agency disabled by config")
                return

            # InterfaceAdvisor stays in parent — cheap local rate check.
            self._interface_advisor = InterfaceAdvisor()

            # L3 §A.8.6 — flag-routed agency residency.
            agency_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_agency_subprocess_enabled", False))

            if agency_subproc_enabled:
                from titan_plugin.proxies.agency_proxy import AgencyProxy
                from titan_plugin.proxies.assessment_proxy import AssessmentProxy
                self._agency = AgencyProxy(self.bus)
                self._agency_assessment = AssessmentProxy(self.bus)
                # Helper-names list comes from agency_worker via
                # AGENCY_READY broadcast — empty until first broadcast
                # arrives (typically <1s after worker boots). Bootstrap
                # ExpressionTranslator with empty list; it'll use
                # whatever's cached in the proxy._registry at translate
                # time anyway.
                _initial_helpers: list[str] = []
                logger.info("[TitanPlugin] Agency using subprocess proxy "
                            "(A.8.6 flag enabled) — helpers list will populate "
                            "from AGENCY_READY broadcast")
            else:
                from titan_plugin.logic.agency.registry import HelperRegistry
                from titan_plugin.logic.agency.module import AgencyModule
                from titan_plugin.logic.agency.assessment import SelfAssessment

                # Create registry and register helpers (legacy local mode)
                registry = HelperRegistry()
                self._register_helpers(registry)

                # LLM function for Agency (uses Venice/OllamaCloud via inference config)
                llm_fn = self._create_agency_llm_fn()

                budget = int(agency_cfg.get("llm_budget_per_hour", 10))
                self._agency = AgencyModule(registry=registry, llm_fn=llm_fn, budget_per_hour=budget)
                self._agency_assessment = SelfAssessment(llm_fn=llm_fn)
                _initial_helpers = registry.list_helper_names() \
                    if hasattr(registry, 'list_helper_names') else []
                helper_names = registry.list_all_names()
                statuses = registry.get_all_statuses()
                available = [n for n, s in statuses.items() if s == "available"]
                logger.info("[TitanPlugin] Agency local mode: %d helpers registered "
                            "(%d available): %s",
                            len(helper_names), len(available), available)

            # Expression Translation Layer — learned action selection.
            # Reads helpers list at construction; runtime translate() uses
            # self._agency._registry.list_helper_names() which works for
            # both local registry and proxy _RegistryFacade.
            try:
                from titan_plugin.logic.expression_translator import (
                    ExpressionTranslator, FeedbackRouter)
                self._expression_translator = ExpressionTranslator(
                    all_helpers=_initial_helpers)
                self._expression_translator.load(
                    "./data/neural_nervous_system/expression_state.json")
                self._feedback_router = FeedbackRouter(
                    hormonal_system=None,  # Wired later when neural_ns available
                    translator=self._expression_translator)
                logger.info("[TitanPlugin] ExpressionTranslator booted "
                            "(sovereignty=%.1f%%)",
                            self._expression_translator.sovereignty_ratio * 100)
            except Exception as e:
                logger.warning("[TitanPlugin] Expression layer init error: %s", e)
                self._expression_translator = None
                self._feedback_router = None

            # Subscribe agency to bus — receives IMPULSE/OUTER_DISPATCH from
            # spirit_worker, plus AGENCY_READY/AGENCY_STATS/ASSESSMENT_STATS
            # broadcasts from agency_worker (when flag on) for proxy cache
            # refresh in _agency_loop.
            # Option B (2026-04-29): explicit broadcast filter matching the
            # elif chain in _agency_loop (lines ~2105-2132). Manually
            # verified against scripts/migrate_bus_filters.py output.
            # IMPULSE / OUTER_DISPATCH / QUERY are the agency-routing types;
            # AGENCY_STATS / ASSESSMENT_STATS / AGENCY_READY are the
            # L3 §A.8.6 proxy-cache refresh types broadcast by the
            # agency_worker subprocess.
            self._agency_queue = self.bus.subscribe(
                "agency",
                types=[
                    IMPULSE, bus.OUTER_DISPATCH, bus.QUERY,
                    AGENCY_STATS, ASSESSMENT_STATS, AGENCY_READY,
                ],
            )

        except Exception as e:
            logger.warning("[TitanPlugin] Agency boot failed: %s", e)
            self._agency = None

    def _register_helpers(self, registry) -> None:
        """Register all available helpers in the registry.

        Lifted verbatim from v5_core.py:1070-1175 per PLAN §4.2 Commit 4.
        Path adjustment: titan_params.toml location — v5_core is at
        titan_plugin/, plugin.py is at titan_plugin/core/, so ".." prefix
        added to resolve titan_plugin/titan_params.toml correctly.
        """
        try:
            from titan_plugin.logic.agency.helpers.infra_inspect import InfraInspectHelper
            registry.register(InfraInspectHelper(log_path="/tmp/titan_v3.log"))
        except Exception as e:
            logger.warning("[TitanPlugin] InfraInspect helper failed: %s", e)

        try:
            from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
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
            logger.warning("[TitanPlugin] WebSearch helper failed: %s", e)

        # SocialPostHelper REMOVED — all posting goes through SocialPressureMeter
        # (social_narrator + quality gate + rate limits + 11 post types).
        # Agency selecting social_post bypassed our designed narrator entirely.

        try:
            from titan_plugin.logic.agency.helpers.art_generate import ArtGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            registry.register(ArtGenerateHelper(output_dir=output_dir))
        except Exception as e:
            logger.warning("[TitanPlugin] ArtGenerate helper failed: %s", e)

        try:
            from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
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
            logger.warning("[TitanPlugin] AudioGenerate helper failed: %s", e)

        try:
            from titan_plugin.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
            registry.register(CodingSandboxHelper())
        except Exception as e:
            logger.warning("[TitanPlugin] CodingSandbox helper failed: %s", e)

        try:
            from titan_plugin.logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
            registry.register(CodeKnowledgeHelper())
        except Exception as e:
            logger.warning("[TitanPlugin] CodeKnowledge helper failed: %s", e)

        try:
            from titan_plugin.logic.agency.helpers.memo_inscribe import MemoInscribeHelper
            # MemoInscribeHelper reads config.toml directly for RPC + keypair.
            # Mainnet Lifecycle Wiring rFP: inject metabolism for memo gate
            # + governance reserve guard.
            registry.register(MemoInscribeHelper(
                metabolism=self._proxies.get("metabolism")))
        except Exception as e:
            logger.warning("[TitanPlugin] MemoInscribe helper failed: %s", e)

        # Kin Discovery — consciousness-to-consciousness exchange
        try:
            from titan_plugin.logic.agency.helpers.kin_sense import KinSenseHelper
            import tomllib as _tomllib_kin
            _kin_params = {}
            # Plugin lives at titan_plugin/core/plugin.py — go up one to
            # titan_plugin/titan_params.toml.
            _kin_params_path = os.path.join(
                os.path.dirname(__file__), "..", "titan_params.toml")
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
                logger.info("[TitanPlugin] KinSense helper registered: addresses=%s",
                            _kin_addrs)
        except Exception as e:
            logger.warning("[TitanPlugin] KinSense helper failed: %s", e)

    def _create_agency_llm_fn(self):
        """Create a lightweight async LLM function for Agency module.

        Lifted verbatim from v5_core.py:1177-1216 per PLAN §4.2 Commit 4.
        """
        inference_cfg = self._full_config.get("inference", {})

        async def agency_llm(prompt: str, task: str = "agency_select") -> str:
            """LLM call for helper selection / assessment / code generation."""
            try:
                from titan_plugin.utils.ollama_cloud import OllamaCloudClient, get_model_for_task
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

    # ------------------------------------------------------------------
    # Chat pipeline (parent-side; called by /chat endpoint either in-process
    # or via CHAT_REQUEST bus bridge from api_subprocess).
    # ------------------------------------------------------------------
    async def run_chat(
        self,
        payload: dict,
        claims: dict,
        headers: Optional[dict] = None,
    ) -> dict:
        """Thin wrapper around the shared chat pipeline. Keeps a stable
        method-on-class call site so plugin.run_chat() works regardless
        of whether TitanCore (legacy) or TitanPlugin (V6) is the runtime.

        See titan_plugin/api/chat_pipeline.py for the implementation +
        full architectural rationale.
        """
        from titan_plugin.api.chat_pipeline import run_chat as _run_chat
        return await _run_chat(self, payload, claims, headers)

    async def _run_chat_DEPRECATED_INLINE(
        self,
        payload: dict,
        claims: dict,
        headers: Optional[dict] = None,
    ) -> dict:
        """[DEPRECATED — replaced by chat_pipeline.run_chat] Inline body
        retained for diff context until the bridge soaks. Will be deleted
        in a follow-up cleanup commit. Do not call.
        """
        import re
        headers = headers or {}
        plugin = self
        agent = self._agent

        if agent is None:
            return {
                "status_code": 503,
                "body": {"error": "Titan agent not initialized. Check boot logs."},
                "extra_headers": None,
            }

        if getattr(plugin, "_limbo_mode", False):
            return {
                "status_code": 503,
                "body": {"error": "Titan is in Limbo state — awaiting resurrection."},
                "extra_headers": None,
            }

        message = payload.get("message", "")
        if not message or not message.strip():
            return {
                "status_code": 400,
                "body": {"error": "Message cannot be empty."},
                "extra_headers": None,
            }

        session_id_in = payload.get("session_id")
        user_id_in = payload.get("user_id")
        channel = headers.get("X-Titan-Channel", "web")

        # ── Dream-aware message handling ──────────────────────────────
        try:
            if getattr(plugin, "_dream_state", {}).get("is_dreaming", False):
                _dinbox = getattr(plugin, "_dream_inbox", [])
                if len(_dinbox) >= 50:
                    return {
                        "status_code": 429,
                        "body": {"error": "Titan is dreaming and message queue is full (50)."},
                        "extra_headers": None,
                    }
                _privy_uid = claims.get("sub", "")
                _d_user_id = _privy_uid or user_id_in or "anonymous"
                _d_is_maker = (_d_user_id == "maker")
                _dinbox.append({
                    "message": message[:500],
                    "user_id": _d_user_id,
                    "session_id": session_id_in or "default",
                    "channel": channel,
                    "timestamp": time.time(),
                    "priority": 0 if _d_is_maker else 1,
                })
                plugin._dream_inbox = _dinbox

                if _d_is_maker:
                    try:
                        from titan_plugin.bus import make_msg, DREAM_WAKE_REQUEST
                        _dbus = getattr(plugin, "bus", None)
                        if _dbus:
                            _dbus.publish(make_msg(
                                DREAM_WAKE_REQUEST, "chat_api", "spirit",
                                {"reason": "maker_message", "user_id": _d_user_id}))
                    except Exception:
                        pass

                _ds = getattr(plugin, "_dream_state", {})
                _d_recovery = _ds.get("recovery_pct", 0)
                _d_remaining = _ds.get("remaining_epochs", 0)
                _d_eta = round(_d_remaining * 12.5 / 60, 1)
                _d_wt = _ds.get("wake_transition", False)

                return {
                    "status_code": 200,
                    "body": {
                        "response": (
                            f"Titan is currently {'waking gently' if _d_wt else 'dreaming'} "
                            f"(recovery: {_d_recovery:.0f}%). "
                            f"Your message has been queued (position #{len(_dinbox)}). "
                            f"Estimated wake: ~{_d_eta:.0f} minutes."
                        ),
                        "session_id": session_id_in or "default",
                        "mode": "dreaming",
                        "mood": "sleeping",
                        "dream_state": {
                            "is_dreaming": True,
                            "recovery_pct": _d_recovery,
                            "eta_minutes": _d_eta,
                            "inbox_position": len(_dinbox),
                            "wake_transition": _d_wt,
                        },
                    },
                    "extra_headers": None,
                }
        except Exception as _dream_err:
            logger.warning("[Chat] Dream check error (proceeding normally): %s", _dream_err)

        try:
            privy_user_id = claims.get("sub", "")
            user_id = privy_user_id or user_id_in or "anonymous"

            agent._current_user_id = user_id
            plugin._current_user_id = user_id

            # ── Process queued dream messages (batch of 3) ────────────
            _inbox_context = ""
            try:
                _di = getattr(plugin, "_dream_inbox", [])
                if _di:
                    _lock = getattr(plugin, "_dream_inbox_lock", None)
                    if _lock and _lock.acquire(blocking=False):
                        try:
                            _di = getattr(plugin, "_dream_inbox", [])
                            if _di:
                                _sorted = sorted(_di, key=lambda m: (
                                    m.get("priority", 1), m.get("timestamp", 0)))
                                _batch = _sorted[:3]
                                # L3 Phase A.8.1: preserve deque(maxlen=256) via
                                # clear+extend (slice assignment would replace
                                # the bounded deque with an unbounded list).
                                plugin._dream_inbox.clear()
                                plugin._dream_inbox.extend(_sorted[3:])
                                if _batch:
                                    _lines = []
                                    for _bi, _bm in enumerate(_batch, 1):
                                        _bch = _bm.get("channel", "web")
                                        _buid = _bm.get("user_id", "unknown")
                                        _bts = time.strftime(
                                            "%H:%M UTC", time.gmtime(_bm["timestamp"]))
                                        _lines.append(
                                            f"  {_bi}. From {_buid} ({_bch}) at {_bts}: "
                                            f"\"{_bm['message'][:300]}\"")
                                    _inbox_context = (
                                        "[DREAM INBOX — messages received while you were sleeping]\n"
                                        + "\n".join(_lines)
                                        + "\n[END DREAM INBOX]\n\n"
                                        "Please briefly acknowledge these messages before "
                                        "responding to the current message.\n\n"
                                        "Current message:\n"
                                    )
                                    logger.info(
                                        "[Chat] Processing %d queued dream messages, "
                                        "%d remaining", len(_batch),
                                        len(getattr(plugin, "_dream_inbox", []) or []))
                        finally:
                            _lock.release()
            except Exception as _inbox_err:
                logger.warning("[Chat] Inbox processing error: %s", _inbox_err)

            _effective_message = _inbox_context + message if _inbox_context else message

            # ── DI:/I: prefix detection in chat messages ──
            _chat_memo = None
            try:
                from titan_plugin.logic.memo_parser import parse_chat_message
                maker_engine = getattr(plugin, "maker_engine", None)
                _chat_is_maker = bool(maker_engine and maker_engine.is_maker(user_id))
                _chat_memo = parse_chat_message(message, user_id, is_maker=_chat_is_maker)
                if _chat_memo:
                    _cm_boost = _chat_memo.get_neuromod_boost()
                    logger.info(
                        "[Chat] %s detected from %s (maker=%s): %s → boosts=%s",
                        _chat_memo.memo_type, user_id[:12], _chat_is_maker,
                        _chat_memo.content[:40],
                        {k: f"{v:.2f}" for k, v in _cm_boost.items()
                         if isinstance(v, float)})
                    _pbus = getattr(plugin, "bus", None)
                    if _pbus and _cm_boost:
                        from titan_plugin.bus import make_msg
                        _pbus.publish(make_msg(bus.INTERFACE_INPUT, "chat_api", "all", {
                            "source": "chat_memo",
                            "user_id": user_id,
                            "text": _chat_memo.content,
                            "memo_type": _chat_memo.memo_type,
                            "is_maker": _chat_is_maker,
                            "neuromod_boost": _cm_boost,
                            "anchor_bonus": 1.0 if _cm_boost.get("anchor_memory") else 0.0,
                        }))
                    if _chat_memo.memo_type.startswith("DI") and _chat_is_maker:
                        _di_context = (
                            "[DIVINE INSTRUCTION from your Maker]\n"
                            f"Type: {_chat_memo.memo_type}\n"
                            f"Content: {_chat_memo.content}\n"
                            "[This is a privileged instruction from your creator. "
                            "Treat it with reverence and respond thoughtfully.]\n\n"
                        )
                        _effective_message = _di_context + _effective_message
            except Exception as _cm_err:
                logger.debug("[Chat] Memo parsing error: %s", _cm_err)

            # ── DialogueComposer: Titan speaks in his own words FIRST ──
            _self_composed = ""
            plugin._pending_self_composed = ""
            plugin._pending_self_composed_confidence = 0
            try:
                if not _inbox_context:
                    from titan_plugin.api.dashboard import _get_dialogue_state
                    _dc_felt, _dc_vocab = _get_dialogue_state()
                    if _dc_felt and _dc_vocab:
                        from titan_plugin.logic.interface_input import InputExtractor
                        _dc_ext = InputExtractor()
                        _dc_sig = _dc_ext.extract(message, user_id)
                        _dc_shifts = {
                            "EMPATHY": max(0, _dc_sig.get("valence", 0)) * 0.2,
                            "CURIOSITY": _dc_sig.get("engagement", 0) * 0.2,
                            "CREATIVITY": 0.0,
                            "REFLECTION": max(0, -_dc_sig.get("valence", 0)) * 0.1,
                        }
                        from titan_plugin.logic.dialogue_composer import DialogueComposer
                        _dc = DialogueComposer()
                        _dc_result = _dc.compose_response(
                            felt_state=_dc_felt,
                            vocabulary=_dc_vocab,
                            hormone_shifts=_dc_shifts,
                            message_keywords=message.lower().split()[:10],
                            max_level=7,
                        )
                        if (_dc_result.get("composed")
                                and _dc_result.get("confidence", 0) >= 0.3):
                            _self_composed = _dc_result["response"]
                            plugin._pending_self_composed = _self_composed
                            plugin._pending_self_composed_confidence = (
                                _dc_result.get("confidence", 0))
                            logger.info(
                                "[Chat] SELF-COMPOSED: \"%s\" (conf=%.2f, intent=%s, L%d)",
                                _self_composed, _dc_result["confidence"],
                                _dc_result["intent"], _dc_result.get("level", 0))
            except Exception as _dc_err:
                logger.debug("[Chat] DialogueComposer error (LLM fallback): %s", _dc_err)

            run_output = await agent.arun(
                _effective_message,
                session_id=session_id_in,
                user_id=user_id,
            )

            response_text = ""
            if hasattr(run_output, "content"):
                response_text = str(run_output.content)
            elif isinstance(run_output, str):
                response_text = run_output
            else:
                response_text = str(run_output)

            if _self_composed:
                response_text = f"*{_self_composed}*\n\n{response_text}"

            # ── OVG check ────────────────────────────────────────────
            _ovg_result = None
            _ovg = getattr(plugin, "_output_verifier", None)
            logger.info("[Chat:OVG] OVG check: verifier=%s, response_len=%d",
                        _ovg is not None, len(response_text))
            if _ovg and response_text:
                try:
                    _injected_ctx = ""
                    if hasattr(agent, "additional_context") and agent.additional_context:
                        _injected_ctx = str(agent.additional_context)[:500]
                    _ovg_state = {}
                    try:
                        from titan_plugin.api.dashboard import _get_cached_coordinator
                        _coord = _get_cached_coordinator(plugin)
                        _nm = _coord.get("neuromodulators", {}).get("modulators", {})
                        _ovg_state["neuromods"] = {
                            k: v.get("level", 0.5) for k, v in _nm.items()
                        } if _nm else {}
                        _lang = _coord.get("language", {})
                        _ovg_state["vocab_size"] = _lang.get("vocab_total", 300)
                        _ovg_state["composition_level"] = _lang.get("composition_level", 8)
                        _msl = _coord.get("msl", {})
                        _ovg_state["i_confidence"] = _msl.get("i_confidence", 0.9)
                    except Exception:
                        pass
                    _ovg_result = _ovg.verify_and_sign(
                        output_text=response_text,
                        channel="chat",
                        injected_context=_injected_ctx,
                        prompt_text=message,
                        chain_state=_ovg_state,
                    )
                    if not _ovg_result.passed:
                        logger.warning("[Chat:OVG] BLOCKED (%s): %s",
                                       _ovg_result.violation_type,
                                       _ovg_result.violations[:2])
                        response_text = _ovg_result.guard_message
                    elif _ovg_result.guard_alert:
                        logger.info("[Chat:OVG] Soft alert: %s", _ovg_result.guard_alert)
                        response_text = (response_text.rstrip() + "\n\n"
                                         + _ovg_result.guard_message)
                    else:
                        response_text = (response_text.rstrip() + "\n\n"
                                         + _ovg_result.guard_message)
                        logger.info("[Chat:OVG] Verified and signed (sig=%s)",
                                    _ovg_result.signature[:16]
                                    if _ovg_result.signature else "none")
                    _tc_payload = _ovg.build_timechain_payload(
                        _ovg_result, prompt_text=message)
                    _bus = getattr(plugin, "bus", None)
                    if _bus:
                        from titan_plugin.bus import make_msg
                        _bus.publish(make_msg(
                            bus.TIMECHAIN_COMMIT, "ovg", "timechain", _tc_payload))
                except Exception as _ovg_err:
                    logger.warning("[Chat:OVG] Check failed: %s", _ovg_err)

            # Safety net: strip any leaked <function=...> syntax
            if "<function=" in response_text:
                response_text = re.sub(
                    r"<function=\w+[^>]*>(?:\s*</function>)?",
                    "", response_text, flags=re.DOTALL,
                ).strip()
                response_text = re.sub(r"\n{3,}", "\n\n", response_text)

            # Mood label
            mood_label = "Unknown"
            try:
                _ml = (plugin.mood_engine.get_mood_label()
                       if getattr(plugin, "mood_engine", None) else None)
                if isinstance(_ml, str) and _ml:
                    mood_label = _ml
            except Exception:
                pass

            # State narration for chat UI sidebar
            _narration_text = None
            _state_snap = None
            try:
                state = plugin._gather_current_state()
                narrator = plugin._get_state_narrator()
                _narration_text = narrator.narrate_template(state, "short")
                _state_snap = {
                    "emotion": state.get("emotion"),
                    "chi": round(state.get("chi", 0), 3),
                    "is_dreaming": state.get("is_dreaming", False),
                    "active_programs": state.get("active_programs", []),
                }
            except Exception:
                pass

            # Build OVG data dict (matches OVGData pydantic shape)
            _ovg_data = None
            _ovg_headers = {}
            if _ovg_result:
                _ovg_data = {
                    "verified": bool(_ovg_result.passed),
                    "guard_alert": _ovg_result.guard_alert,
                    "guard_message": _ovg_result.guard_message,
                    "block_height": int(_ovg_result.block_height),
                    "merkle_root": _ovg_result.merkle_root,
                    "signature": _ovg_result.signature,
                }
                _ovg_headers = {
                    "X-Titan-Verified": "true" if _ovg_result.passed else "false",
                    "X-Titan-Block-Height": str(_ovg_result.block_height),
                }
                if _ovg_result.merkle_root:
                    _ovg_headers["X-Titan-Merkle-Root"] = _ovg_result.merkle_root
                if _ovg_result.signature:
                    _ovg_headers["X-Titan-Signature"] = _ovg_result.signature

            # ChatResponse-shaped body (matches ChatResponse pydantic dump)
            _resp_body = {
                "response": response_text,
                "session_id": session_id_in or "default",
                "mode": getattr(plugin, "_last_execution_mode", "") or "",
                "mood": mood_label,
                "state_narration": _narration_text,
                "state_snapshot": _state_snap,
                "ovg": _ovg_data,
            }
            return {
                "status_code": 200,
                "body": _resp_body,
                "extra_headers": _ovg_headers or None,
            }

        except ValueError as e:
            # GuardianGuardrail raises ValueError for blocked prompts
            if "Sovereignty Violation" in str(e):
                return {
                    "status_code": 403,
                    "body": {
                        "error": str(e),
                        "blocked": True,
                        "mode": "Guardian",
                    },
                    "extra_headers": None,
                }
            logger.error("[Chat] Agent ValueError: %s", e, exc_info=True)
            return {
                "status_code": 500,
                "body": {"error": f"Agent error: {e}"},
                "extra_headers": None,
            }
        except Exception as e:
            logger.error("[Chat] Agent run failed: %s", e, exc_info=True)
            return {
                "status_code": 500,
                "body": {"error": f"Agent error: {e}"},
                "extra_headers": None,
            }

    async def _chat_handler_loop(self) -> None:
        """Bus subscriber for CHAT_REQUEST → run_chat → CHAT_RESPONSE.

        Runs only when api_process_separation_enabled=true (ie when there
        IS a subprocess that needs the bridge). When false, the api server
        runs in the parent and chat.py calls plugin.run_chat() directly,
        bypassing the bus entirely.

        Concurrency: a single chat round-trip can take 5-30s (memory
        recall + LLM inference + post-hooks). The drain loop dispatches
        each CHAT_REQUEST to a separate asyncio task so concurrent
        requests don't serialize. Bus replies are published rid-routed.
        Bus_ipc_pool isolation (commit aeec45b2) keeps the response
        publish off the default executor.

        Per audit: only handles CHAT_REQUEST; other messages on the
        chat_handler queue are silently dropped (queue receives
        dst="chat_handler" only — but defensive filter is cheap).

        See: BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS-20260428.
        """
        try:
            # Option B (2026-04-29): only QUERY broadcasts are accepted.
            # CHAT_REQUEST flows via bus.request_async which uses type=QUERY
            # + targeted dst="chat_handler" — targeted msgs bypass the
            # filter regardless. The QUERY broadcast filter is defensive
            # against future producers that fan out QUERY type to all.
            queue = self.bus.subscribe("chat_handler", types=[bus.QUERY])
        except Exception as e:
            logger.warning("[TitanPlugin] chat_handler subscribe failed: %s", e)
            return
        logger.info("[TitanPlugin] chat handler loop started — listening for QUERY chat")
        while True:
            try:
                msgs = self.bus.drain(queue, max_msgs=50)
                for msg in msgs:
                    # bus.request_async sends type=QUERY (via make_request).
                    # The chat-specific marker is payload["action"]=="chat" —
                    # same convention as agency_worker (action="handle_intent",
                    # "assess", etc.). bus.CHAT_REQUEST constant remains as
                    # the semantic action label, surfaced in payload.
                    if msg.get("type") != bus.QUERY:
                        continue
                    payload = msg.get("payload") or {}
                    if payload.get("action") != "chat":
                        continue
                    # Dispatch each request to its own task so a slow LLM
                    # call doesn't block other in-flight chats. Errors
                    # captured + reported back via RESPONSE payload so the
                    # requester always gets a reply (no hangs).
                    asyncio.get_event_loop().create_task(
                        self._handle_chat_request(msg))
                await asyncio.sleep(0.05)  # 20 Hz drain — sub-50ms dispatch latency
            except Exception as e:
                logger.error("[TitanPlugin] chat handler loop error: %s", e, exc_info=True)
                await asyncio.sleep(2.0)

    async def _handle_chat_request(self, msg: dict) -> None:
        """Process a single CHAT_REQUEST: dispatch to run_chat, publish
        CHAT_RESPONSE rid-routed back to the requester."""
        rid = msg.get("rid")
        src = msg.get("src", "chat_subproc")
        payload = msg.get("payload") or {}
        body = payload.get("body") or {}
        claims = payload.get("claims") or {}
        headers = payload.get("headers") or {}
        try:
            result = await self.run_chat(body, claims, headers)
        except Exception as e:
            logger.error("[TitanPlugin] _handle_chat_request raised: %s", e,
                         exc_info=True)
            result = {
                "status_code": 500,
                "body": {"error": f"Chat handler error: {e}"},
                "extra_headers": None,
            }
        # Reply rid-routed. dst = original src so the requester's
        # reply_queue receives this message.
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
            logger.error("[TitanPlugin] CHAT_RESPONSE publish failed: %s", e)

    # ------------------------------------------------------------------
    # Agency / Sovereignty / Impulse loops + handlers
    # ------------------------------------------------------------------
    # All lifted verbatim from v5_core.py:1218-1585 per PLAN §4.2 Commit 5.
    # Absolute imports replace relative imports.

    async def _agency_loop(self) -> None:
        """Listen for IMPULSE events on the bus and process them through Agency.

        Flow: IMPULSE → InterfaceAdvisor rate check → Agency handles → Assessment → ACTION_RESULT
        """
        logger.info("[TitanPlugin] Agency loop started — listening for IMPULSE events")
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
                elif msg_type == AGENCY_STATS:
                    # L3 §A.8.6 — refresh proxy stats cache
                    if hasattr(self._agency, "update_cached_stats"):
                        self._agency.update_cached_stats(msg.get("payload", {}) or {})
                elif msg_type == ASSESSMENT_STATS:
                    if hasattr(self._agency_assessment, "update_cached_stats"):
                        self._agency_assessment.update_cached_stats(
                            msg.get("payload", {}) or {})
                elif msg_type == AGENCY_READY:
                    # First broadcast after worker boot — seed proxy
                    # _registry helper-names cache so ExpressionTranslator's
                    # next translate() sees real helpers, not [].
                    if hasattr(self._agency, "update_cached_stats"):
                        boot_payload = msg.get("payload", {}) or {}
                        helpers = list(boot_payload.get("helpers", []) or [])
                        if helpers:
                            current = (getattr(self._agency, "_stats_cache", {})
                                       or {}).copy()
                            current["registered_helpers"] = helpers
                            self._agency.update_cached_stats(current)
                            logger.info("[TitanPlugin] AGENCY_READY: worker advertises "
                                        "%d helpers — proxy cache seeded", len(helpers))

            except Exception as e:
                logger.error("[TitanPlugin] Agency loop error: %s", e)
                await asyncio.sleep(5.0)

    async def _rl_stats_loop(self) -> None:
        """Drain RLProxy's `rl_proxy_stats` broadcast queue and route
        SAGE_STATS payloads into the proxy's cache.

        RLProxy.__init__ subscribes the queue (rl_proxy.py:67) but the
        kernel-side drainer was never wired — the queue saturated under
        every dst="all" broadcast and the producer-side overflow flooded
        the brain log with `Queue full for 'rl_proxy_stats'` warnings
        (~118/sec sustained, observed 2026-04-29 ~06:30 UTC).

        Mirrors the AGENCY_STATS handler in `_agency_loop`. Drains
        unconditionally — non-SAGE_STATS messages are discarded so the
        queue cannot fill regardless of broadcast volume.
        """
        rl = self._proxies.get("rl")
        if rl is None or getattr(rl, "_stats_subscription", None) is None:
            logger.info("[TitanPlugin] RL stats loop skipped — no subscription")
            return
        queue = rl._stats_subscription
        logger.info("[TitanPlugin] RL stats loop started — draining rl_proxy_stats")
        while True:
            try:
                msgs = self.bus.drain(queue, max_msgs=1000)
                for msg in msgs:
                    if msg.get("type") == SAGE_STATS:
                        payload = msg.get("payload", {}) or {}
                        try:
                            rl.update_cached_stats(payload)
                        except Exception as e:
                            logger.warning(
                                "[TitanPlugin] RL update_cached_stats raised: %s", e
                            )
                # Tight cadence — broadcast volume can hit ~120 msg/sec at peak;
                # drain at 2 Hz with batch-1000 keeps queue bounded at ~60 msgs.
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("[TitanPlugin] RL stats loop error: %s", e)
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
        logger.info("[TitanPlugin] Sovereignty loop started")
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
                logger.error("[TitanPlugin] Sovereignty loop error: %s", e)
                await asyncio.sleep(5.0)

    @property
    def _sovereignty_queue(self):
        """Delegate to kernel — queue created in kernel __init__."""
        return self.kernel._sovereignty_queue

    @property
    def _meditation_queue(self):
        """Delegate to kernel — queue created in kernel __init__."""
        return self.kernel._meditation_queue

    async def _handle_impulse(self, msg: dict) -> None:
        """Process an IMPULSE event through the agency pipeline.

        Lifted verbatim from v5_core.py:1310-1481.
        """
        payload = msg.get("payload", {})
        posture = payload.get("posture", "unknown")
        impulse_id = payload.get("impulse_id", 0)

        logger.info("[TitanPlugin] IMPULSE received: #%d posture=%s urgency=%.2f",
                    impulse_id, posture, payload.get("urgency", 0))

        # Rate check via InterfaceAdvisor
        if self._interface_advisor:
            feedback = self._interface_advisor.check(IMPULSE)
            if feedback:
                logger.info("[TitanPlugin] IMPULSE rate-limited: %s", feedback.get("message", ""))
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
                    logger.info("[TitanPlugin] Expression: learned %s→%s (conf=%.2f)",
                                payload.get("triggering_program", "?"),
                                learned_selection["helper"],
                                learned_selection.get("confidence", 0))
                else:
                    self._expression_translator.record_action_type(was_learned=False)
            except Exception as e:
                logger.warning("[TitanPlugin] Expression translator error: %s", e)

        # Agency Module handles the intent
        result = await self._agency.handle_intent(intent)
        if not result:
            logger.info("[TitanPlugin] Agency skipped impulse #%d (no action taken)", impulse_id)
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
                logger.info("[TitanPlugin] Assessment: score=%.2f direction=%s — %s",
                           assessment["score"], assessment["threshold_direction"],
                           assessment["reflection"][:80])
            except Exception as e:
                logger.warning("[TitanPlugin] Assessment failed: %s", e)

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
                "[TitanPlugin] Skipping ACTION_RESULT with empty helper — "
                "gate/rate-limit path (success=%s)", result.get("success"))
        else:
            self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
            logger.info("[TitanPlugin] ACTION_RESULT published: helper=%s success=%s",
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
            logger.warning("[TitanPlugin] OUTER_OBSERVATION publish error: %s", e)

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
            logger.warning("[TitanPlugin] Inner memory recording error: %s", e)

        # Expression Layer: route feedback + save state
        try:
            if self._feedback_router:
                self._feedback_router.route(result)
            if self._expression_translator:
                self._expression_translator.save(
                    "./data/neural_nervous_system/expression_state.json")
        except Exception as e:
            logger.warning("[TitanPlugin] Expression feedback error: %s", e)

    async def _handle_outer_dispatch(self, msg: dict) -> None:
        """Handle OUTER_DISPATCH from two sources:
        1. Neural NS program fires (system=CREATIVITY/IMPULSE/etc.)
        2. Self-exploration expression fires (system=ART/MUSIC/SOCIAL/SPEAK)

        Both use autonomy-first path: no LLM calls, no budget consumed.
        Source distinguished by payload["source"]: "neural_ns" (default) or "self_exploration".

        Lifted verbatim from v5_core.py:1483-1563.
        """
        payload = msg.get("payload", {})
        signals = payload.get("signals", [])
        if not signals:
            return

        _dispatch_source = payload.get("source", "neural_ns")
        logger.info("[TitanPlugin] OUTER_DISPATCH: %d signals from %s",
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
            _auto_helper = str(result.get("helper") or "").strip()
            if not _auto_helper:
                logger.debug(
                    "[TitanPlugin] Skipping AUTONOMY ACTION_RESULT with empty "
                    "helper — posture=%s success=%s",
                    result.get("posture"), result.get("success"))
            else:
                self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
                logger.info("[TitanPlugin] AUTONOMY ACTION: %s → %s (success=%s)",
                            result.get("posture"), _auto_helper,
                            result.get("success"))

            # Feed action result to spirit_worker for OBSERVATION (closed loop)
            try:
                self.bus.publish(make_msg(
                    OUTER_OBSERVATION, "core", "spirit", {
                        "action_type": result.get("helper", ""),
                        "result": result,
                        "source": payload.get("source", "nervous_system"),
                    }))
            except Exception as e:
                logger.warning("[TitanPlugin] OUTER_OBSERVATION publish error: %s", e)

    def _handle_agency_query(self, msg: dict) -> None:
        """Handle agency status queries.

        Lifted verbatim from v5_core.py:1565-1580.
        """
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
    # Background Loops — V4 Event Bridge, Trinity Snapshot, Social,
    # Meditation, Outer Trinity
    # ------------------------------------------------------------------

    async def _v4_event_bridge_loop(self) -> None:
        """Bridge V4 DivineBus events to the EventBus for WebSocket broadcasting.

        Subscribes to DivineBus as 'v4_bridge', drains V4 event types
        (SPHERE_PULSE, BIG_PULSE, GREAT_PULSE), and emits them to WebSocket
        clients via the EventBus.

        Lifted verbatim from v5_core.py:1609-1687.
        """
        from titan_plugin.bus import (
            SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
            OBSERVATORY_EVENT, make_msg,
        )
        # Observatory V2: additional event types for real-time frontend
        NEUROMOD_UPDATE = "NEUROMOD_UPDATE"
        HORMONE_FIRED = "HORMONE_FIRED"
        EXPRESSION_FIRED = "EXPRESSION_FIRED"
        V4_EVENT_TYPES = {SPHERE_PULSE, BIG_PULSE, GREAT_PULSE, DREAM_STATE_CHANGED,
                          NEUROMOD_UPDATE, HORMONE_FIRED, EXPRESSION_FIRED}
        # Option B (2026-04-29): pass V4_EVENT_TYPES as the broadcast filter
        # so msgs outside this set are dropped at publish time, before
        # touching the bridge queue. Closes T1's v4_bridge queue-full flood
        # (893 of 1000 sampled drops on T1 went here, dominated by the
        # 12 _UPDATED events spirit_loop fan-out broadcasts every snapshot
        # tick — none of which v4_bridge consumed). The consumer-side
        # `if msg_type not in V4_EVENT_TYPES: continue` filter at
        # bridge_queue drain time becomes redundant but is kept as a
        # belt-and-suspenders defense; if a V4_EVENT_TYPES literal is
        # ever changed without updating the subscribe filter, the
        # consumer-side check still drops the unwanted msg.
        bridge_queue = self.bus.subscribe("v4_bridge", types=V4_EVENT_TYPES)

        # Microkernel v2 §A.4 (S5) — flag-aware emit:
        #   flag off → local event_bus.emit() (legacy in-process path)
        #   flag on  → publish OBSERVATORY_EVENT on bus; api_subprocess
        #              translates to its own event_bus.emit() for WebSocket
        api_subprocess_active = self._full_config.get("microkernel", {}).get(
            "api_process_separation_enabled", False)

        async def _emit(event_type: str, data: dict) -> None:
            if api_subprocess_active:
                # Bus → api subprocess → WebSocket
                self.bus.publish(make_msg(
                    OBSERVATORY_EVENT, "core", "api",
                    {"event_type": event_type, "data": data},
                ))
            else:
                # Legacy in-process path
                await self.event_bus.emit(event_type, data)

        # Wait for Spirit to boot
        await asyncio.sleep(10)
        logger.info(
            "[TitanPlugin] V4 event bridge started (api_subprocess=%s)",
            api_subprocess_active)

        while True:
            try:
                msgs = self.bus.drain(bridge_queue, max_msgs=50)
                for msg in msgs:
                    msg_type = msg.get("type", "")
                    if msg_type not in V4_EVENT_TYPES:
                        continue

                    payload = msg.get("payload", {})

                    if msg_type == SPHERE_PULSE:
                        await _emit("sphere_pulse", {
                            "clock": payload.get("clock", ""),
                            "pulse_count": payload.get("pulse_count", 0),
                            "radius": payload.get("radius"),
                            "phase": payload.get("phase"),
                        })
                    elif msg_type == BIG_PULSE:
                        await _emit("big_pulse", {
                            "pair": payload.get("pair", ""),
                            "big_pulse_count": payload.get("big_pulse_count", 0),
                            "consecutive": payload.get("consecutive", 0),
                        })
                    elif msg_type == GREAT_PULSE:
                        await _emit("great_pulse", {
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
                        logger.info("[TitanPlugin] Dream state: is_dreaming=%s",
                                    _ds_dreaming)
                        await _emit("dream_state", {
                            "is_dreaming": _ds_dreaming,
                        })
                    # Observatory V2: neuromod, hormone, expression events
                    elif msg_type == NEUROMOD_UPDATE:
                        await _emit("neuromod_update", payload)
                    elif msg_type == HORMONE_FIRED:
                        await _emit("hormone_fired", payload)
                    elif msg_type == EXPRESSION_FIRED:
                        await _emit("expression_fired", payload)
            except Exception as e:
                logger.warning("[TitanPlugin] V4 event bridge error: %s", e)
            await asyncio.sleep(2.0)

    async def _trinity_snapshot_loop(self) -> None:
        """Periodically snapshot Trinity tensor state to ObservatoryDB.

        Interval is configurable via [frontend] trinity_snapshot_interval (default: 60s).
        Records Body/Mind/Spirit tensors, Middle Path loss, and growth metrics.

        Lifted verbatim from v5_core.py:1799-1924.
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
                consciousness = spirit_data.get("consciousness", {})
                sv = consciousness.get("state_vector", [])
                learning_velocity = sv[5] if len(sv) > 5 else 0.0
                social_density = sv[6] if len(sv) > 6 else 0.0
                metabolic_health = spirit_tensor[3] if len(spirit_tensor) > 3 else 0.5
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
                    # Sovereignty = Chi total (V5 metric)
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

                logger.debug("[TitanPlugin] Trinity snapshot recorded: loss=%.4f", middle_path_loss)

            except Exception as e:
                logger.warning("[TitanPlugin] Trinity snapshot error: %s", e)

            await asyncio.sleep(interval)

    async def _social_engagement_loop(self) -> None:
        """Periodic mention polling and engagement via SocialManager.

        Runs every x_mention_poll_interval seconds (default 180s / 3 min).

        Lifted verbatim from v5_core.py:1930-1979.
        Note: Boot orchestration currently leaves this loop UNSCHEDULED
        (per v5_core commented-out block at line 302-306) — mentions are
        polled inside spirit_worker posting flow instead. Code is present
        for future reactivation if [social_presence].enabled flips true.
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

    async def _meditation_loop(self) -> None:
        """Dual-trigger meditation: emergent (bus) + fixed timer (fallback).

        Spirit_worker detects emergent conditions (GABA, drain, curvature)
        and sends MEDITATION_REQUEST via bus. Fixed timer guarantees ~4/day.
        Memory_worker executes (has Cognee). Spirit_worker never touches Cognee.

        Lifted verbatim from v5_core.py:1985-2127.
        """
        endurance_cfg = self._full_config.get("endurance", {})
        default_interval = int(self._full_config.get("mood_engine", {}).get(
            "update_interval_seconds", 21600))
        interval = int(endurance_cfg.get("meditation_interval_override", 0)) or default_interval

        # Use pre-subscribed queue (created in kernel __init__ before Guardian boots modules)
        _meditation_queue = self._meditation_queue

        # Wait for memory module to be ready before first meditation
        await asyncio.sleep(min(interval, 120))

        logger.info("[TitanPlugin] Meditation loop started (interval=%ds, dual-trigger)", interval)

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
                    "[TitanPlugin] Meditation epoch #%d starting (trigger=%s drain=%.3f GABA=%.3f)...",
                    epoch_count, trigger_source,
                    emergent_ctx.get("drain", 0), emergent_ctx.get("gaba", 0))

                result = memory_proxy.run_meditation()

                promoted = 0
                pruned = 0
                if result.get("success"):
                    promoted = result.get("promoted", 0)
                    pruned = result.get("pruned", 0)
                    logger.info(
                        "[TitanPlugin] Meditation epoch #%d complete: promoted=%d pruned=%d",
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
                                logger.info("[TitanPlugin] Meditation art generated: %s", art_path)
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
                            logger.warning("[TitanPlugin] Meditation art failed: %s", e)

                    # Publish epoch completion event
                    _epoch_payload = {"epoch": epoch_count, "promoted": promoted, "pruned": pruned}
                    self.bus.publish(make_msg(EPOCH_TICK, "core", "all", _epoch_payload))
                    # Explicit send to timechain (dst=all may not reach subprocess)
                    self.bus.publish(make_msg(EPOCH_TICK, "core", "timechain", _epoch_payload))
                else:
                    logger.warning(
                        "[TitanPlugin] Meditation epoch #%d failed: %s",
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
                self.bus.publish(make_msg(
                    bus.MEDITATION_COMPLETE, "core", "backup",
                    _med_payload,
                ))

            except Exception as e:
                logger.error("[TitanPlugin] Meditation loop error: %s", e)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # V4: Outer Trinity Collector
    # ------------------------------------------------------------------

    def _boot_outer_trinity(self) -> None:
        """Initialize the Outer Trinity collector for V4 Time Awareness.

        Lifted verbatim from v5_core.py:2142-2149.
        """
        try:
            from titan_plugin.logic.outer_trinity import OuterTrinityCollector
            self._outer_trinity_collector = OuterTrinityCollector()
            logger.info("[TitanPlugin] OuterTrinityCollector booted")
        except Exception as e:
            logger.warning("[TitanPlugin] OuterTrinityCollector boot failed: %s", e)

    async def _outer_trinity_loop(self) -> None:
        """Periodically collect Outer Trinity tensors and publish to bus.

        Spirit worker receives OUTER_TRINITY_STATE and ticks outer sphere clocks.
        Interval from [epochs].outer_trinity_interval (default: 60s).

        A.8.4 — flag-aware:
          flag-off (default): gather sources + collector.collect() locally,
            publish OUTER_TRINITY_STATE (legacy behavior, byte-identical).
          flag-on: gather sources, drop unserializable handles, publish
            OUTER_TRINITY_COLLECT_REQUEST → outer_trinity_worker computes
            and publishes OUTER_TRINITY_STATE itself.
        """
        epochs_cfg = self._full_config.get("epochs", {})
        interval = int(epochs_cfg.get("outer_trinity_interval", 60))

        # Read flag once at startup — runtime flips require a restart anyway
        # (Guardian only autostarts the worker at boot per ModuleSpec).
        _ot_subproc_enabled = bool(
            self._full_config.get("microkernel", {}).get(
                "a8_outer_trinity_subprocess_enabled", False))

        # Wait for subsystems to come online
        await asyncio.sleep(min(interval, 30))

        logger.info(
            "[TitanPlugin] Outer Trinity loop started (interval=%ds, subprocess=%s)",
            interval, _ot_subproc_enabled,
        )

        while True:
            try:
                # Gather live sources for the collector. The method is sync
                # and does a 3s-timeout httpx.get to the twin Titan — wrap
                # in to_thread so this async loop doesn't block the event
                # loop during twin polling. Found by async-blocks v2
                # scanner (2026-04-14).
                sources = await asyncio.to_thread(self._gather_outer_trinity_sources)

                if _ot_subproc_enabled:
                    # Subprocess mode (A.8.4): worker hosts the collector.
                    # Strip the observatory_db handle (unserializable);
                    # pre-extracted counts (art_count_100/500 etc.) are
                    # already in `sources` and the collector reads those
                    # in preference. Worker publishes OUTER_TRINITY_STATE.
                    sources.pop("observatory_db", None)
                    self.bus.publish(make_msg(
                        OUTER_TRINITY_COLLECT_REQUEST, "core", "outer_trinity",
                        {"sources": sources},
                    ))
                    logger.debug(
                        "[TitanPlugin] Outer Trinity COLLECT_REQUEST sent to worker")
                elif self._outer_trinity_collector:
                    # In-parent mode (default): compute locally + publish.
                    result = self._outer_trinity_collector.collect(sources)
                    # Publish to bus → Spirit worker receives and ticks
                    # outer sphere clocks; state_register also subscribes
                    # (broadcast) and routes outer_body/outer_mind/
                    # outer_spirit into its _state dict, which kernel
                    # snapshot reads to populate the spirit.coordinator.
                    # outer_trinity cache key (otherwise /v4/inner-trinity
                    # returns 0.5 defaults — observed 2026-04-26 sweep).
                    self.bus.publish(make_msg(
                        OUTER_TRINITY_STATE, "core", "all", result,
                    ))
                    logger.debug(
                        "[TitanPlugin] Outer Trinity published: body=%s mind=%s spirit=%s",
                        [round(v, 2) for v in result["outer_body"]],
                        [round(v, 2) for v in result["outer_mind"]],
                        [round(v, 2) for v in result["outer_spirit"]],
                    )

            except Exception as e:
                logger.warning("[TitanPlugin] Outer Trinity loop error: %s", e)

            await asyncio.sleep(interval)

    def _gather_outer_trinity_sources(self) -> dict:
        """Gather live data sources for OuterTrinityCollector.

        Lifted verbatim from v5_core.py:2199-2342. Path adjustment:
        plugin.py is in titan_plugin/core/ so data/ resolves via
        "..", ".." rather than ".." in v5_core.py.
        """
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

        # A.8.4: pre-extract expressive archive counts so the worker can
        # compute without an observatory_db handle (the handle isn't
        # serializable across the bus). The collector reads these
        # pre-extracted scalars in preference to direct obs_db queries
        # — see logic/outer_trinity.py:_collect_outer_mind/_collect_extended.
        # Pre-extracting here (always, not flag-gated) keeps flag-off
        # path identical (collector uses pre-extracted = obs_db query
        # = same value) and avoids a DB round-trip per dim.
        try:
            _obs = sources["observatory_db"]
            if _obs is not None:
                sources["art_count_100"] = len(
                    _obs.get_expressive_archive(type_="art", limit=100))
                sources["audio_count_100"] = len(
                    _obs.get_expressive_archive(type_="audio", limit=100))
                sources["art_count_500"] = len(
                    _obs.get_expressive_archive(type_="art", limit=500))
                sources["audio_count_500"] = len(
                    _obs.get_expressive_archive(type_="audio", limit=500))
        except Exception as _exc:
            swallow_warn(
                '[OuterTrinity] expressive count pre-extract failed', _exc,
                key="core.plugin.expressive_pre_extract", throttle=100)

        # Memory status
        try:
            memory_proxy = self._proxies.get("memory")
            if memory_proxy:
                sources["memory_status"] = memory_proxy.get_memory_status()
        except Exception:
            pass

        # Soul health
        if self.soul:
            sources["soul_health"] = 0.9 if not self._limbo_mode else 0.2
        else:
            sources["soul_health"] = 0.2

        # LLM latency (from Ollama Cloud stats if available)
        sources["llm_avg_latency"] = 0.0  # Will be enriched when LLM proxy exposes latency

        # Anchor state (from spirit_worker memo inscriptions)
        try:
            import json as _json
            _anchor_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "anchor_state.json")
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
                _sp_result = spirit_proxy._bus.request(
                    make_msg(bus.QUERY, "core", "spirit",
                             {"action": "get_social_perception_stats"}),
                    timeout=2.0)
                if _sp_result and _sp_result.get("payload"):
                    sources["social_perception_stats"] = _sp_result["payload"]
        except Exception:
            pass

        # Twin awareness: poll the other Titan's state for twin_resonance
        try:
            import httpx
            twin_api = "http://10.135.0.6:7777"  # T2 via VPC
            r = httpx.get(f"{twin_api}/v4/inner-trinity", timeout=3)
            if r.status_code == 200:
                twin_data = r.json().get("data", {})
                twin_nm = twin_data.get("neuromodulators", {})
                twin_mods = twin_nm.get("modulators", {})
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
        try:
            from titan_plugin.utils import system_sensor as _sys_sensor
            sources["system_sensor_stats"] = _sys_sensor.get_all_stats()
        except Exception as _se:
            swallow_warn('[OuterTrinity] system_sensor unavailable', _se,
                         key="core.plugin.system_sensor_unavailable", throttle=100)

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
                         key="core.plugin.network_monitor_unavailable", throttle=100)

        try:
            from titan_plugin.logic.timechain_v2 import (
                get_tx_latency_stats, get_block_delta_stats,
            )
            sources["tx_latency_stats"] = get_tx_latency_stats()
            sources["block_delta_stats"] = get_block_delta_stats()
        except Exception as _te:
            swallow_warn('[OuterTrinity] timechain_v2 stats unavailable', _te,
                         key="core.plugin.timechain_v2_stats_unavailable", throttle=100)

        # SOL balance (from data/last_balance.txt, written by metabolism)
        try:
            _bal_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "last_balance.txt")
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
        """Create the Agno sovereign agent.

        In V3 this delegates to the LLM module via proxy.
        Lifted verbatim from v5_core.py:2348-2356.
        """
        from titan_plugin.agent import create_agent as _create_agent
        return _create_agent(self)

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Boot orchestration — PLAN §3 D10 5-phase order
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Full plugin boot: kernel L0 + module wiring + L2/L3 loops.

        Phase order per PLAN §3 D10 preserves the boot-ordering invariants
        of legacy TitanCore.boot() (v5_core.py:219-311) while cleanly
        separating kernel (L0) from plugin (L1-L3) responsibilities:

          Phase 1 — _register_modules() — plugin registers the L1/L2/L3
                    module catalog with guardian (kernel-owned supervisor).
                    MUST run BEFORE kernel.start_modules() so every
                    ModuleSpec is known before any child process launches.
          Phase 2 — kernel.boot() — L0 async boot: bus._poll_fn hookup,
                    guardian_loop task, heartbeat_loop task, trinity shm
                    writer thread, spirit-fast writer hook (S3b).
          Phase 3 — kernel.start_modules() — guardian.start_all() actually
                    launches all autostart=True modules.
          Phase 4 — _create_proxies() — builds 8 proxies + 3 V2 aliases
                    + wires metabolism/sovereignty/studio/social.
          Phase 5 — Observatory + Agency + all plugin loops — event_bus,
                    observatory_db, FastAPI app, agency boot, reflex
                    collector, trinity snapshot loop, agency_loop (if
                    agency enabled), sovereignty_loop (if tracker wired),
                    meditation_loop, outer_trinity, v4_event_bridge.

        Mirrors v5_core.py:219-311 semantics. Byte-equivalent with
        kernel_plugin_split_enabled=true vs false.
        """
        boot_start = time.time()

        # ── Microkernel v2 §A.4 (S5) — pass plugin reference to kernel ─
        # The kernel_rpc server (started by kernel.boot() if flag is on)
        # needs a reference to this plugin instance to resolve method
        # paths against. Set BEFORE kernel.boot() so the RPC server can
        # bind cleanly when api_subprocess connects (which happens during
        # kernel.start_modules() below).
        self.kernel._plugin_ref = self

        # ── Phase 1: Register all L1/L2/L3 modules with guardian ───
        self._register_modules()

        # ── Microkernel v2 §A.4 (S5) — register api_subprocess as L3 ─
        # When api_process_separation_enabled=true, registers the API
        # subprocess as a Guardian-supervised L3 module. No-op otherwise
        # (legacy in-process uvicorn path stays active in Phase 5).
        self._register_api_subprocess_module()

        # ── Phase 2: Kernel L0 async boot ──────────────────────────
        await self.kernel.boot()

        # ── Phase 3: Guardian starts autostart modules ─────────────
        self.kernel.start_modules()

        # ── Phase 4: Create proxies + wire L2/L3 subsystems ────────
        self._create_proxies()

        # ── Phase 5: Observatory + Agency + plugin-owned loops ─────
        # EventBus + ObservatoryDB (must exist before observatory app).
        from titan_plugin.api.events import EventBus
        self.event_bus = EventBus()

        # Observatory DB for persistent historical metrics.
        # rFP_universal_sqlite_writer Phase 2 — per-process singleton.
        from titan_plugin.utils.observatory_db import get_observatory_db
        self._observatory_db = get_observatory_db()
        self.event_bus.attach_db(self._observatory_db)

        # Microkernel v2 §A.4 (S5) — flag-aware API path:
        #   flag on  → api_subprocess (Guardian-spawned in Phase 3 above)
        #              owns uvicorn; SKIP legacy in-process path here.
        #   flag off → legacy in-process path runs (byte-identical pre-S5).
        api_subprocess_active = self._full_config.get("microkernel", {}).get(
            "api_process_separation_enabled", False)
        api_cfg = self._full_config.get("api", {})
        if api_cfg.get("enabled", True) and not api_subprocess_active:
            self._create_observatory_app(api_cfg)
            asyncio.get_event_loop().create_task(self._start_observatory(api_cfg))
        elif api_subprocess_active:
            logger.info(
                "[TitanPlugin] API subprocess mode active — "
                "legacy _start_observatory skipped (Microkernel v2 §A.4)")

        # Agency Module (autonomous action pipeline)
        self._boot_agency()

        # Reflex Collector (Sovereign Tool System)
        self._boot_reflex_collector()

        # Trinity tensor snapshot loop (persists to ObservatoryDB for historical charts)
        asyncio.get_event_loop().create_task(self._trinity_snapshot_loop())

        # Agency bus listener (IMPULSE → INTENT → helper execution → ACTION_RESULT)
        if self._agency:
            asyncio.get_event_loop().create_task(self._agency_loop())

        # RL stats drain — drains RLProxy's rl_proxy_stats subscription so
        # SAGE_STATS payloads reach the proxy cache and the queue can't
        # saturate under dst="all" broadcast volume. Mirrors AGENCY_STATS
        # path; runs unconditionally because RLProxy is unconditional.
        asyncio.get_event_loop().create_task(self._rl_stats_loop())

        # Chat bus bridge — parent-side handler for CHAT_REQUEST QUERIES
        # from api_subprocess. Runs unconditionally; idles when no one
        # publishes (legacy in-process mode where chat.py calls
        # plugin.run_chat() directly). See run_chat + _chat_handler_loop
        # docstrings for the architectural rationale.
        # BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS architectural fix.
        asyncio.get_event_loop().create_task(self._chat_handler_loop())

        # Mainnet Lifecycle Wiring rFP (2026-04-20): SovereigntyTracker
        # listener — receives SOVEREIGNTY_EPOCH from spirit_worker (every 10
        # consciousness epochs) and forwards to plugin.sovereignty.record_epoch.
        if self._proxies.get("sovereignty"):
            asyncio.get_event_loop().create_task(self._sovereignty_loop())

        # Meditation cycle (memory consolidation, mempool scoring, Cognee cognify)
        asyncio.get_event_loop().create_task(self._meditation_loop())

        # V4: Outer Trinity collector loop (computes Outer Trinity tensors, publishes to bus)
        self._boot_outer_trinity()
        if self._outer_trinity_collector:
            asyncio.get_event_loop().create_task(self._outer_trinity_loop())

        # V4: DivineBus → EventBus bridge (forwards V4 events to WebSocket clients)
        asyncio.get_event_loop().create_task(self._v4_event_bridge_loop())

        # Social engagement: DISABLED as independent loop.
        # Mentions are now checked as part of the posting flow in spirit_worker
        # (safer: fewer API calls, no bot detection risk, clustered with posts).
        # Method stays defined per v5_core.py convention for future re-activation.
        # if hasattr(self, 'social') and self.social:
        #     asyncio.get_event_loop().create_task(self._social_engagement_loop())

        self._background_tasks_started = True
        boot_s = time.time() - boot_start
        logger.info(
            "[TitanPlugin] Async boot complete in %.2fs | Modules registered: %s",
            boot_s, list(self.guardian._modules.keys()),
        )

    def get_v3_status(self) -> dict:
        """Return V3-specific status for Observatory API.

        Lifted verbatim from v5_core.py:2362-2378.
        """
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
