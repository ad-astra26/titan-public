"""
titan_hcl/core/plugin.py — TitanHCL (L1-L3 coordinator).

Thin coordinator holding a `kernel: TitanKernel` reference plus the L2/L3
state and loops:
  - Proxies (memory/rl/llm/body/mind/spirit/media/timechain + 13 @property facades)
  - Agency subsystem (autonomous action pipeline)
  - OutputVerifier (external output gate)
  - Outer trinity collector
  - EventBus + ObservatoryDB + observatory FastAPI app
  - Agno agent
  - Dream inbox (API-side message queue during dream cycles)
  - Parent async loops: agency, v4_event_bridge, trinity_snapshot,
    publish_outer_sources (sovereignty/meditation/social_engagement loops
    retired to workers — D-SPEC-57/60/104)

Uses the kernel for all L0 services (bus, guardian, state_register,
registry_bank, soul, network, identity). The kernel never restarts;
this coordinator may be replaced or re-attached during Phase B shadow
core swap.

Compat @property facade (bus, guardian, soul, _full_config, ...) makes
TitanHCL duck-type-identical to the legacy TitanCore for dashboard +
agent code. Zero dashboard code changes required.

This commit (#3 — plugin skeleton + module registration + proxies)
lands the __init__, compat properties, _register_modules (380-line
module catalog lifted from v5_core.py:317-696), and _create_proxies.
Wire helpers + observatory + agency + async loops + boot orchestration
arrive in commits 4-6.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §A.1
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.2 + §3 D1+D9+D10
  - titan_hcl/core/kernel.py (the L0 paired class)
"""
import asyncio
import logging
import os
import time
from typing import Optional

from titan_hcl.bus import (
    ACTION_RESULT,
    AGENCY_READY,
    AGENCY_STATS,
    ASSESSMENT_STATS,
    EPOCH_TICK,
    IMPULSE,
    OUTER_OBSERVATION,
    SAGE_STATS,
    make_msg,
)
from titan_hcl.core.kernel import TitanKernel
from titan_hcl.guardian import ModuleSpec
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
)
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl import bus

logger = logging.getLogger(__name__)


# PERSISTENCE_BY_DESIGN: TitanHCL._proxies / _agency / _*_mode fields
# are runtime bootstrap state — constructed from kernel + config at boot.
# Proxy objects are not self-owned state to persist.
class TitanHCL:
    """
    L1-L3 Coordinator — owns proxies, agency, observatory, agno, dream inbox.

    Usage (from scripts/titan_hcl.py flag-branch per PLAN §4.4):
        kernel = TitanKernel(wallet_path)
        plugin = TitanHCL(kernel)
        await plugin.boot()  # orchestrates kernel.boot() + module wiring

    Compat shape: dashboard.py + agent code treat TitanHCL as the
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
        # _interface_advisor parent instance RETIRED v1.8.5 §4.H (D-SPEC-59,
        # 2026-05-15) — InterfaceAdvisor now lives in interface_advisor_worker
        # subprocess; parent reads rate state from SHM via
        # InterfaceAdvisorStateReader (sub-µs G18, 100ms cache).
        self._interface_advisor_reader = None

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
                from titan_hcl.proxies.output_verifier_proxy import OutputVerifierProxy
                self._output_verifier = OutputVerifierProxy(self.bus)
                logger.info("[TitanHCL] OutputVerifier using subprocess proxy "
                            "(A.8.3 flag enabled)")
            except Exception as _ovg_err:
                logger.warning("[TitanHCL] OutputVerifierProxy init failed: %s",
                               _ovg_err)
        else:
            try:
                from titan_hcl.logic.output_verifier import OutputVerifier
                _tc_dir = os.path.join("data", "timechain")
                _titan_id = kernel.config.get("info_banner", {}).get("titan_id") or kernel.titan_id
                _wallet_path = kernel.config.get("network", {}).get(
                    "wallet_keypair_path", "data/titan_identity_keypair.json")
                self._output_verifier = OutputVerifier(
                    titan_id=_titan_id, data_dir=_tc_dir, keypair_path=_wallet_path)
            except Exception as _ovg_err:
                logger.warning("[TitanHCL] OutputVerifier init failed: %s", _ovg_err)

        # ── State ────────────────────────────────────────────────────
        self._last_execution_mode = "Shadow"
        self._is_meditating = False
        self._background_tasks_started = False
        self._observatory_app = None
        self._agent = None

        # Phase C v1.8.2 (D-SPEC-56) per rFP_titan_hcl_l2_separation_strategy.md §4.I:
        # `_dream_inbox` deque + `_dream_state` dict DELETED — dream state
        # ownership moved to dream_state_worker (G21 single writer of
        # dream_state.bin SHM slot). Chat handler reads is_dreaming via
        # DreamStateReader (sub-µs G18 SHM-direct, 100ms TTL cache). Chat-during-
        # dream buffering happens in dream_state_worker via DREAM_INBOX_ENQUEUE
        # bus events; drains on dream_end via DREAM_INBOX_REPLAY → chat handler
        # re-processes the buffered messages. See dream_state_worker.py.

        # EventBus / ObservatoryDB — populated by boot()
        self.event_bus = None
        self._observatory_db = None

        # rFP_trinity_130d_awakening Phase 2 — inner perception state owns
        # the AudioPerception / VisualPerception / AmbientChangeMonitor
        # trackers + _last_create_ts. Producers feed inner_mind[5,7,9] +
        # outer_spirit ANANDA[41]. Lazily started by boot() once
        # system_sensor is available.
        self._inner_perception = None

        # rFP_trinity_130d_awakening Phase 2 — outer spirit history aggregator.
        # Owns env_adapt + graceful_rest + circadian_alignment +
        # dream_recall trackers (SPEC §23.9 SAT[11], CHIT[25,26], ANANDA[40]).
        # CHIT[29] self_trajectory is worker-local in outer_spirit_worker.
        self._outer_spirit_history = None

        logger.info(
            "[TitanHCL] Coordinator constructed (kernel_id=%s, limbo=%s)",
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
    # TitanHCL as if it were TitanHCL. Returns None when module
    # not loaded, so endpoints can degrade gracefully.

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
        from titan_hcl.modules.memory_worker import memory_worker_main
        from titan_hcl.modules.recorder_worker import recorder_worker_main
        from titan_hcl.modules.llm_worker import llm_worker_main
        from titan_hcl.modules.agno_worker import agno_worker_main
        from titan_hcl.modules.body_worker import body_worker_main
        from titan_hcl.modules.mind_worker import mind_worker_main
        # spirit_worker retired (D-SPEC-116) — its L3 engines live in
        # cognitive_worker / Rust daemons; the heartbeat stub is gone.
        from titan_hcl.modules.media_worker import media_worker_main
        from titan_hcl.modules.language_worker import language_worker_main
        from titan_hcl.modules.meta_teacher_worker import meta_teacher_worker_main
        from titan_hcl.modules.cgn_worker import cgn_worker_main
        from titan_hcl.modules.knowledge_worker import knowledge_worker_main
        from titan_hcl.modules.emot_cgn_worker import emot_cgn_worker_main
        from titan_hcl.modules.timechain_worker import timechain_worker_main
        from titan_hcl.modules.backup_worker import backup_worker_main
        from titan_hcl.persistence_entry import imw_main

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
        from titan_hcl.modules.output_verifier_worker import (
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
            # rFP_worker_broadcast_topics_completion §4.A Batch 1 (2026-05-10):
            # output_verifier consumes ONLY targeted QUERY messages (RPC
            # pattern: callers publish dst="output_verifier"). Drain at
            # output_verifier_worker.py:168-171 explicitly drops anything
            # that isn't QUERY. reply_only=True excludes broadcasts at the
            # broker, removing fan-out cost for high-rate Phase C types.
            reply_only=True,
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # not on hot ARC path; respawn-OK
        ))

        # Phase A.S8 outer_{body,mind,spirit} Python workers RETIRED (Phase C
        # dissolution C.8, 2026-05-22, no-shim). Under l0_rust_enabled=true
        # (production fleet) the Rust titan-outer-{body,mind,spirit}-rs daemons
        # own the outer tensor slots (§7.1) and were the only canonical writers;
        # the Python workers were the legacy flag=false path (never spawned under
        # flag-on per C.0) + carried a latent outer_body_5d.bin dual-writer. Their
        # source data now flows SHM-direct via the in-parent sensor sidecars +
        # outer_source_assembly helper (no OUTER_SOURCES_SNAPSHOT broadcast).

        # Reflex Worker — L3 §A.8.5 subprocess extraction.
        # Hosts a stateless ReflexCollector (no executors registered)
        # that performs the aggregation step on each QUERY(action=
        # "aggregate") from the parent's ReflexProxy. Cooldowns are
        # synced from each request's payload (parent owns the truth).
        # Default OFF — when flag flips, parent's reflex_collector
        # becomes a ReflexProxy (see _boot_reflex_collector + proxies/
        # reflex_proxy.py).
        from titan_hcl.modules.reflex_worker import reflex_worker_main
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
            # rFP_worker_broadcast_topics_completion §4.A Batch 1 (2026-05-10):
            # reflex drain at reflex_worker.py:134-137 explicitly drops
            # anything that isn't SHUTDOWN or QUERY (both targeted). Pure
            # RPC pattern — reply_only=True is the canonical declaration.
            reply_only=True,
            start_method="spawn" if _spawn_grad else "fork",
            b2_1_swap_critical=False,  # per-call processor; respawn-OK
        ))

        # Agency Worker — L3 §A.8.6 subprocess extraction.
        # Hosts AgencyModule + SelfAssessment + HelperRegistry + 8 helpers
        # + LLM fn. Handles work-RPC QUERY(action="handle_intent" |
        # "dispatch_from_nervous_signals" | "assess") via bus.request_async
        # with bounded timeout — async-friendly proxies in parent keep the
        # event loop unblocked during the worker's LLM round-trip.
        # State queries (agency_stats, assessment_stats) migrated to SHM
        # via agency_state.bin + assessment_state.bin (Phase C Session 3
        # §4.B.2/§4.B.3 publishers + Session 5 §4.D.4 handler retirement).
        # Default OFF — when flag flips, parent's _agency / _agency_assessment
        # become AgencyProxy + AssessmentProxy (see core/plugin.py:_boot_agency).
        from titan_hcl.modules.agency_worker import agency_worker_main
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
            # rFP_worker_broadcast_topics_completion §4.A Batch 1 (2026-05-10):
            # agency_worker drain at agency_worker.py:442-445 explicitly
            # drops anything that isn't SHUTDOWN or QUERY (both targeted).
            # Pre-existing comment "needs broadcasts (none currently consumed)"
            # was a categorization gap — none ARE consumed. Pure RPC pattern;
            # reply_only=True is canonical.
            reply_only=True,
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
        from titan_hcl.modules.warning_monitor_worker import (
            warning_monitor_worker_main,
        )
        _wm_cfg = self._full_config.get("warning_monitor", {})
        # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
        # warning_monitor drain at modules/warning_monitor_worker.py:209
        # consumes one broadcast type (SILENT_SWALLOW_REPORT).
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
            broadcast_topics=[bus.SILENT_SWALLOW_REPORT],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        ))

        # health_monitor_worker — SPEC v1.12.0 §9.B + D-SPEC-67
        # (rFP_health_monitor_worker.md, Maker greenlit 2026-05-17).
        # Pluggable L3 health-monitor framework: discovers HealthCheckPlugin
        # subclasses under titan_hcl.health.* at boot, filters by
        # applies_on, schedules each at its cadence_s. MVP plugin: social_x
        # (canonical_poller-only — active on T1 today). SOLE-sanctioned
        # heal path = bus HEAL_REQUEST → owning worker (social_worker for
        # social_x.refresh_session) → HEAL_RESULT reply. Worker boots
        # uniformly on T1+T2+T3; T2/T3 currently have zero plugins after
        # applies_on filter but still publish heartbeats so future
        # per-Titan plugins (backup_arweave et al.) graduate without
        # boot-wiring changes.
        from titan_hcl.modules.health_monitor_worker import (
            health_monitor_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="health_monitor",
            layer="L3",  # observability + advisory service
            entry_fn=health_monitor_worker_main,
            config=self._full_config,  # full config — plugins resolve sections
            rss_limit_mb=150,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[bus.MODULE_SHUTDOWN],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
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
        # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
        # rl drain at modules/rl_worker.py:136-147 consumes one broadcast
        # type (SAGE_RECORD_TRANSITION); MODULE_SHUTDOWN + QUERY are targeted.
        self.guardian.register(ModuleSpec(
            name="recorder",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (IQL chain learning)
            entry_fn=recorder_worker_main,
            config=self._full_config.get("stealth_sage", {}),
            rss_limit_mb=3000,
            autostart=_a8_sage_subproc_enabled,  # §A.8.7: autostart when flag-on
            lazy=not _a8_sage_subproc_enabled,   # §A.8.7: eager when flag-on
            broadcast_topics=[bus.SAGE_RECORD_TRANSITION],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # LLM/Inference module (Agno agent — ~500MB)
        # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
        # llm drain at modules/llm_worker.py:103-112 consumes one broadcast
        # type (LLM_TEACHER_REQUEST); MODULE_SHUTDOWN + QUERY are targeted.
        self.guardian.register(ModuleSpec(
            name="llm",
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (Agno inference, human-time)
            entry_fn=llm_worker_main,
            config=self._full_config.get("inference", {}),
            rss_limit_mb=1000,
            autostart=True,  # Changed: Language Teacher needs llm at boot
            lazy=False,
            heartbeat_timeout=120.0,  # LLM calls can block 30s+; match Spirit/Memory timeout
            broadcast_topics=[bus.LLM_TEACHER_REQUEST],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # SPEC v1.17.0 / D-SPEC-72 — agno_worker hosts the Agno Agent + chat
        # pipeline (PreHook + PostHook + tools + guardrails + AsyncSqliteDb
        # session store). Receives CHAT_REQUEST / CHAT_STREAM_REQUEST from
        # api_subprocess via agno_proxy; dispatches via agent.arun() in-worker.
        # See SPEC §9.B agno_worker block for the full contract.
        # Merged inference + agent config so the worker's _init_worker_plugin_and_agent
        # can resolve provider via §9.F.1 inference module.
        _agno_cfg = {
            **self._full_config.get("inference", {}),
            **self._full_config.get("agent", {}),
        }
        # Surface the [agent] block as a nested key for downstream usage too
        _agno_cfg["agent"] = self._full_config.get("agent", {})
        _agno_cfg["inference"] = self._full_config.get("inference", {})
        # ζ.0/ζ.1 (D-SPEC-79, 2026-05-18) — propagate [chat] section so the
        # ChatTierClassifier in agno_hooks.PreHook sees [[chat.tiers]] blocks.
        # Without this the worker subprocess only got inference+agent and the
        # classifier fell through to "passthrough" with all features on —
        # which defeated the whole point of ζ.1 feature gating.
        _agno_cfg["chat"] = self._full_config.get("chat", {})
        self.guardian.register(ModuleSpec(
            name="agno_worker",
            layer="L2",  # Microkernel v2 §A.5 — L2 module (chat pipeline owner)
            entry_fn=agno_worker_main,
            config=_agno_cfg,
            # D-SPEC-78 (Phase 2 Chunk α, 2026-05-18) — RE-BUMPED 600 → 1000
            # after partial root-cause work.
            #
            # First attempt (2026-05-18 morning) bumped 600 → 1000 as a pure
            # band-aid for 735MB production peak. Maker pushed back
            # (`feedback_no_rss_band_aid_understand_root_cause.md`) — find root
            # cause, not bump cap.
            #
            # Investigation found: `titan_hcl.logic.cgn` was transitively
            # pulling PyTorch (libtorch 53MB + libtriton 46MB) + pyarrow
            # (~22MB) when arc/session.py grounding ran during chat.
            #
            # Partial fix (commit 1d3f80cc): split cgn.py → `cgn_types.py`
            # (pure dataclasses, no torch) + `cgn.py` (nn.Module classes).
            # Non-torch callers now import from `cgn_types`. Result on T3:
            # agno_worker BOOT RSS dropped 622MB → 60MB (huge win — confirms
            # the cgn import was the boot-path culprit).
            #
            # HOWEVER: chat-time RSS still grows to 643MB on T3 (above the
            # 600MB cap → Guardian kill-loop → /chat 504). Something during
            # the actual chat flow still loads heavy libs — NOT cgn.
            # Candidates not yet investigated: agno's session DB warm-up,
            # research tool's sage_researcher chain, embedder loading.
            #
            # Per Maker direction 2026-05-18 ("if /chat is broken we need to
            # fix it"), restore rss_limit_mb=1000 to unblock the fleet
            # cascade. The chat-time RSS growth investigation is tracked as
            # rFP §9.1 Chunk α-2 (deeper py-spy + import-trace at peak-chat).
            # NOT a quiet band-aid — explicitly scoped + tracked.
            rss_limit_mb=1000,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,  # Background heartbeat thread bypasses arun delays
            broadcast_topics=[
                bus.CHAT_REQUEST,
                bus.CHAT_STREAM_REQUEST,
                bus.KERNEL_EPOCH_TICK,
                bus.SAVE_NOW,
                # RFP_phase_c_titan_hcl_cleanup Phase A (2026-05-21): agno_worker
                # is the new DREAM_INBOX_REPLAY consumer (re-answers messages
                # buffered during dream) — moved from the retired parent
                # _v4_event_bridge_loop. dream_state_worker broadcasts it dst="all".
                bus.DREAM_INBOX_REPLAY,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=True,  # data/agno_sessions.db is critical-data per §11.H
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
        # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
        # body drain at modules/body_worker.py:255-303 consumes 4 broadcasts
        # (FILTER_DOWN, FOCUS_NUDGE, CONVERSATION_STIMULUS, INTERFACE_INPUT);
        # MODULE_SHUTDOWN + QUERY are targeted.
        self.guardian.register(ModuleSpec(
            name="body",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT somatic)
            entry_fn=body_worker_main,
            config=body_config,
            rss_limit_mb=800,   # was 500; fork-inherited parent memory grew from ~250MB to ~400MB+ (2026-04-17)
            autostart=True,  # Body senses must always be active
            lazy=False,
            broadcast_topics=[
                bus.FILTER_DOWN, bus.FOCUS_NUDGE,
                bus.CONVERSATION_STIMULUS, bus.INTERFACE_INPUT,
            ],
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
        # mind drain at modules/mind_worker.py consumes these broadcasts;
        # MODULE_SHUTDOWN + QUERY are targeted. OUTER_SOURCES_SNAPSHOT dropped
        # (Phase C C.7/C.8): mind_worker now reads outer-source state SHM-direct.
        self.guardian.register(ModuleSpec(
            name="mind",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT cognitive)
            entry_fn=mind_worker_main,
            config=mind_config,
            rss_limit_mb=700,   # was 500; fork-inherited parent RSS ~400MB left only 100MB headroom — caused T3 cascade (2026-04-17). Memory profiling tool (DEFERRED TOP) will identify real optimization targets.
            autostart=True,  # Mind senses should always be active
            lazy=False,
            broadcast_topics=[
                bus.FILTER_DOWN, bus.FOCUS_NUDGE,
                bus.CONVERSATION_STIMULUS, bus.INTERFACE_INPUT,
                bus.SENSE_VISUAL, bus.SENSE_AUDIO,
            ],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # Spirit module (Consciousness + V4 Sphere Clocks + Enrichment + Neural NS + Experience Orchestrator)
        # filter_down_v5 + titan_self were defined in titan_params.toml but
        # never plumbed here, so V5's publish_enabled flag was always
        # stuck at its False default regardless of toml edits. Caught 2026-04-16
        # when attempting to flip publish_enabled and observing V5 kept
        # reporting False from /v4/filter-down-status.
        # spirit_worker ModuleSpec RETIRED (D-SPEC-116, 2026-05-22). It had been a
        # heartbeat-only stub since D8-3; under l0_rust_enabled=true the L3 cognitive
        # engines live in cognitive_worker + Rust daemons. The 6 orphaned dst="spirit"
        # flows it used to drop were repointed/restored/retired in this same change:
        #   MEMORY_RECALL_PERTURBATION → neuromod + cognitive (nudge/i_depth/working_mem)
        #   REFLEX_REWARD → NS_REWARD (cognitive); TEACHER_SIGNALS → cognitive (MSL+nudge)
        #   OUTER_OBSERVATION → cognitive (signal_engagement); STATE_SNAPSHOT + RATE_LIMIT
        #   retired (superseded / log-only). See AUDIT_phase_d_spirit_worker_retirement.

        # cognitive_worker (L2) — Phase C C-S8 4B (chunk 8E skeleton, 2026-05-05).
        # Hosts the L3 cognitive engines (Reasoning, MetaReasoning, Dreaming,
        # InnerTrinityCoordinator, PiHeartbeat, NeuralNervousSystem,
        # ObservableEngine, ExpressionManager). Active under l0_rust_enabled=true
        # ONLY — the whole Phase C microkernel (Rust trinity daemons + this L2
        # engine host) is gated on the flag. The legacy l0_rust_enabled=false
        # consciousness path (spirit_worker_main) was RETIRED in D-SPEC-116; the
        # flag-false rollback mode no longer ships a consciousness host (no-shim).
        # Boot order: cognitive_worker reads the trinity tensors body/mind publish
        # + the Rust-owned slots, so it boots after body/mind in the autostart seq.
        # See SPEC §1 glossary + §9.B Python tree (NEW v0.1.8) +
        # PLAN_microkernel_phase_c_s8_cognitive_worker_extraction.md §2.2.
        if self._full_config.get("microkernel", {}).get("l0_rust_enabled", False):
            cognitive_worker_config = {
                "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
                # Microkernel flag passthrough — cognitive_worker_main checks it
                # defensively even though registration is already gated.
                "microkernel": self._full_config.get("microkernel", {}),
                # Banner config for titan_id resolution.
                "info_banner": self._full_config.get("info_banner", {}),
                # Engine configs read from titan_params.toml inside the worker
                # via _load_toml_section helper — no need to thread them here.
                # titan_vm config kept here for InnerTrinityCoordinator's
                # internal NervousSystem (lightweight VM context).
                "titan_vm": self._full_config.get("titan_vm", {}),
            }
            from titan_hcl.modules.cognitive_worker import (
                cognitive_worker_main,
                _COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
            )
            # expression_worker registration (NEW per §4.B Track 3, 2026-05-15).
            # Sequenced AFTER cognitive_worker because expression_worker
            # subscribes to KERNEL_EPOCH_TICK from cognitive_worker.
            from titan_hcl.modules.expression_worker import (
                expression_worker_main,
                _EXPRESSION_WORKER_SUBSCRIBE_TOPICS,
            )
            self.guardian.register(ModuleSpec(
                name="cognitive_worker",
                layer="L2",  # Microkernel v2 §A.5 — L2 (cognitive engine host)
                entry_fn=cognitive_worker_main,
                config=cognitive_worker_config,
                rss_limit_mb=2000,  # ReasoningEngine + MetaReasoningEngine + V5 NS net
                autostart=True,     # Required for /v4/* cognitive routes to populate
                lazy=False,
                heartbeat_timeout=120.0,  # Cognitive epoch can include LLM calls
                start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
                # chunk 8M.3 (2026-05-05): broadcast_topics filter — closes the
                # 8-min-stale-heartbeat backpressure class identified in
                # rFP_phase_c_observatory_data_pipeline.md §2.4 / §1.4. Without
                # this filter, broker fanned-out every dst="all" event (incl.
                # high-rate trinity tensors / filter_down cascades) into
                # cognitive_worker's queue, blocking guardian_HCL heartbeat
                # delivery in the reverse direction → 10/25 modules stuck in
                # state=starting. Mirrors the worker-side topics= passed to
                # setup_worker_bus inside cognitive_worker_main.
                broadcast_topics=_COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
            ))

            # expression_worker (L2) — §4.B Track 3 extraction
            # (SHIPPED 2026-05-15 per rFP_titan_hcl_l2_separation_strategy
            # §4.B + SPEC §9.B expression_worker block + D-SPEC-NN).
            # Owns ExpressionManager + 6 composites (SPEAK/ART/MUSIC/SOCIAL/
            # KIN_SENSE/LONGING) + composite-ledger + expression_state.bin
            # SHM publisher. Drives evaluate_all on KERNEL_EPOCH_TICK
            # received from cognitive_worker (Maker Q2 — preserves
            # adaptive 1-30s consciousness epoch coupling). Absorbs
            # D8-3 catalyst-producer site #8 (strong_composition) per
            # the "D8 RETIREMENT PREREQUISITE" block at the top of the
            # strategy rFP.
            self.guardian.register(ModuleSpec(
                name="expression_worker",
                layer="L2",
                entry_fn=expression_worker_main,
                config={
                    "data_dir": self._full_config.get(
                        "memory_and_storage", {}).get("data_dir", "./data"),
                    "microkernel": self._full_config.get("microkernel", {}),
                    "info_banner": self._full_config.get("info_banner", {}),
                },
                rss_limit_mb=400,   # ExpressionManager + 6 composites is light
                autostart=True,
                lazy=False,
                heartbeat_timeout=60.0,
                broadcast_topics=_EXPRESSION_WORKER_SUBSCRIBE_TOPICS,
                start_method="spawn" if _spawn_grad else "fork",
            ))

            # outer_interface_worker (L2) — Track 2 of
            # rFP_phase_c_self_improvement_subsystem_migration (SPEC v1.3.1 §9.B).
            # Hosts OuterInterface (composition_engine + narrator + advisor + decoder),
            # dynamic word-recipes registry, kin signature/society broadcast surface,
            # self-exploration cadence driver. Active under l0_rust_enabled=true
            # AND outer_interface_worker_enabled=true ONLY; legacy spirit_worker_main
            # owns OuterInterface under l0_rust=false per Maker D3 (b).
            # NOTE: this registration is the TitanHCL (microkernel v2 split)
            # mirror of the TitanCore registration in legacy_core.py — both
            # paths must be kept in sync. See SPEC §9.B v1.3.1 + D-SPEC-38.
            if self._full_config.get("microkernel", {}).get("outer_interface_worker_enabled", True):
                outer_interface_worker_config = {
                    "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
                    "microkernel": self._full_config.get("microkernel", {}),
                    "info_banner": self._full_config.get("info_banner", {}),
                    "outer_interface":   self._full_config.get("outer_interface", {}),
                    "self_exploration":  self._full_config.get("self_exploration", {}),
                    "action_decoder":    self._full_config.get("action_decoder", {}),
                    "action_narrator":   self._full_config.get("action_narrator", {}),
                    "kin":               self._full_config.get("kin", {}),
                }
                from titan_hcl.modules.outer_interface_worker import outer_interface_worker_main
                self.guardian.register(ModuleSpec(
                    name="outer_interface_worker",
                    layer="L2",
                    entry_fn=outer_interface_worker_main,
                    config=outer_interface_worker_config,
                    rss_limit_mb=500,
                    autostart=True,
                    lazy=False,
                    heartbeat_timeout=60.0,
                    broadcast_topics=[
                        bus.REASONING_STATS_UPDATED, bus.NEUROMOD_STATS_UPDATED,
                        bus.CHI_UPDATED, bus.KERNEL_EPOCH_TICK,
                        bus.EXPRESSION_FIRED, bus.CONVERSATION_STIMULUS,
                        bus.SPEAK_REQUEST_PENDING, bus.GREAT_KIN_PULSE,
                    ],
                    b2_1_swap_critical=False,
                    start_method="spawn" if _spawn_grad else "fork",
                ))

            # self_reflection_worker (L2) — Track 2 of
            # rFP_phase_c_self_improvement_subsystem_migration (SPEC v1.3.1 §9.B).
            # Hosts SelfReasoningEngine + CodingExplorer (+ CodingSandboxHelper child
            # subprocess via PR_SET_PDEATHSIG) + PredictionEngine (relocated from
            # cognitive_worker per Track 1 drift correction, commit 51e5cfbf).
            # Active under l0_rust_enabled=true AND self_reflection_worker_enabled=true
            # ONLY; legacy spirit_worker_main owns SelfReasoning + CodingExplorer
            # under l0_rust=false. Closes the last 3 engines blocking D8-3.
            # NOTE: TitanHCL mirror of legacy_core.py registration — both
            # paths must be kept in sync. SPEC §9.B v1.3.1 + D-SPEC-38.
            if self._full_config.get("microkernel", {}).get("self_reflection_worker_enabled", True):
                self_reflection_worker_config = {
                    "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
                    "microkernel": self._full_config.get("microkernel", {}),
                    "info_banner": self._full_config.get("info_banner", {}),
                    "self_reflection":  self._full_config.get("self_reflection", {}),
                    "self_reasoning":   self._full_config.get("self_reasoning", {}),
                    "prediction":       self._full_config.get("prediction_engine", {}),
                    "cgn":              self._full_config.get("cgn", {}),
                }
                from titan_hcl.modules.self_reflection_worker import self_reflection_worker_main
                self.guardian.register(ModuleSpec(
                    name="self_reflection_worker",
                    layer="L2",
                    entry_fn=self_reflection_worker_main,
                    config=self_reflection_worker_config,
                    rss_limit_mb=800,
                    autostart=True,
                    lazy=False,
                    heartbeat_timeout=90.0,
                    broadcast_topics=[
                        bus.REASONING_STATS_UPDATED, bus.META_REASONING_STATS_UPDATED,
                        bus.EXPERIENCE_STIMULUS, bus.DREAMING_STATE_UPDATED,
                        bus.CGN_CROSS_INSIGHT, bus.KERNEL_EPOCH_TICK,
                        # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §9.B
                        # + D-SPEC-70 v1.15.0 — cognitive_worker fires META_INTROSPECT_REQUEST
                        # (fire-and-forget per §8.0.ter D-SPEC-48) per META INTROSPECT action;
                        # self_reflection_worker handler runs sr.introspect(**payload),
                        # persists to data/inner_memory.db.self_insights via _persist_insight(),
                        # writes result to inner_self_insight.bin SHM slot. Closes F-8.
                        bus.META_INTROSPECT_REQUEST,
                    ],
                    b2_1_swap_critical=True,
                    start_method="spawn" if _spawn_grad else "fork",
                ))

        # social_worker (L2) — Phase C-S9 (chunks 9A-9K shipped 2026-05-12).
        # Hosts SocialXGateway + ArchetypeDispatcher + SocialPressureMeter +
        # SOCIAL_CATALYST + KIN_SIGNAL + SOCIAL_RECEIVED bus subscribers +
        # social_x_state.bin SHM publisher + X_POST_PUBLISHED publisher +
        # per-Titan archetype recency boost. Active under
        # microkernel.social_worker_enabled=true ONLY — the legacy spirit_worker
        # owns X-posting under flag-false per Maker D3(b). NOT gated on
        # l0_rust_enabled — independent of broader Phase C migration.
        # See PLAN_microkernel_phase_c_s9_social_worker_extraction.md.
        if self._full_config.get("microkernel", {}).get("social_worker_enabled", False):
            social_worker_config = {
                "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
                "microkernel": self._full_config.get("microkernel", {}),
                "info_banner": self._full_config.get("info_banner", {}),
                # social_x section carries: gateway db path, archetype configs,
                # canonical_poller_titan_id, recency-boost tunables, post limits.
                "social_x": self._full_config.get("social_x", {}),
            }
            from titan_hcl.modules.social_worker import social_worker_main
            self.guardian.register(ModuleSpec(
                name="social_worker",
                layer="L2",
                entry_fn=social_worker_main,
                config=social_worker_config,
                rss_limit_mb=500,
                autostart=True,
                lazy=False,
                heartbeat_timeout=120.0,
                broadcast_topics=[
                    bus.EXPRESSION_FIRED,
                    bus.MEDITATION_COMPLETE,
                    bus.KIN_SIGNAL,
                    bus.SOCIAL_RECEIVED,
                    bus.SOCIAL_CATALYST,
                    bus.MENTION_RECEIVED,
                    bus.FELT_EXPERIENCE_CAPTURED,
                    bus.ENGAGEMENT_SNAPSHOT_TAKEN,
                ],
                start_method="spawn" if _spawn_grad else "fork",
            ))

        # social_graph_worker — extracted from mind_worker per
        # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1, 2026-05-14).
        # Hosts SocialGraph (Phase 13 Sage Socialite) + data/social_graph.db +
        # social_graph_state.bin SHM publisher (G21 single-writer).
        # ALWAYS-ON autostart — no flag-gate (replaces the legacy alias rot).
        # broadcast_topics minimal: dispatch arrives as dst="social_graph"
        # bus.QUERY (not a broadcast), only MODULE_SHUTDOWN + SAVE_NOW are
        # broadcasts the worker needs to consume.
        from titan_hcl.modules.social_graph_worker import (
            social_graph_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="social_graph",
            layer="L2",
            entry_fn=social_graph_worker_main,
            config={
                "data_dir": self._full_config.get(
                    "memory_and_storage", {}).get("data_dir", "./data"),
                "info_banner": self._full_config.get("info_banner", {}),
                "social_graph": self._full_config.get("social_graph", {}),
            },
            rss_limit_mb=150,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.MODULE_SHUTDOWN,
                bus.SAVE_NOW,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=True,
        ))

        # metabolism_worker — extracted from titan_HCL inline wire per
        # rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51 (v1.7.2, 2026-05-14).
        # Hosts MetabolismController + metabolism_state.bin SHM publisher
        # (G21 single-writer). ALWAYS-ON autostart — no flag-gate
        # (replaces the inline `_wire_metabolism` body which previously
        # constructed MetabolismController as an in-process attribute).
        # Subscribes to bus.QUERY (dst=metabolism) for evaluate_gate /
        # async state queries + SOLANA_BALANCE_UPDATED for responsive
        # tier refresh.
        from titan_hcl.modules.metabolism_worker import (
            metabolism_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="metabolism",
            layer="L2",
            entry_fn=metabolism_worker_main,
            config={
                "growth_metrics": self._full_config.get("growth_metrics", {}),
                "network": self._full_config.get("network", {}),
                "info_banner": self._full_config.get("info_banner", {}),
            },
            rss_limit_mb=100,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.SOLANA_BALANCE_UPDATED,
                bus.MODULE_SHUTDOWN,
                bus.SAVE_NOW,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # life_force_worker — extracted per rFP_titan_hcl_l2_separation_strategy §4.G
        # + D-SPEC-57 (v1.8.3, 2026-05-15). Maker-greenlit Q1-Q6 inline.
        # Hosts LifeForceEngine (Chi Λ 3×3 Trinity vitality math) extracted
        # from cognitive_worker chunk 8M.6 Track 1 drift. Owns
        # life_force_state.bin SHM slot (G21 single-writer, 1Hz cadence,
        # 4096B msgpack), publishes LIFE_FORCE_UPDATED / CHI_UPDATED
        # (producer flipped from cognitive_worker) / FATIGUE_LEVEL_CRITICAL /
        # NEUROMOD_EXTERNAL_NUDGE(source=life_force_chi_health — closes §4.Q
        # D-SPEC-54 orphan nudge). ALWAYS-ON autostart — no flag-gate.
        # See SPEC v1.8.3 §9.B `life_force_worker` block + PLAN b8e27b9d.
        from titan_hcl.modules.life_force_worker import (
            life_force_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="life_force",
            layer="L2",
            entry_fn=life_force_worker_main,
            config={
                "life_force": self._full_config.get("life_force", {}),
                "info_banner": self._full_config.get("info_banner", {}),
            },
            rss_limit_mb=100,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.KERNEL_EPOCH_TICK,
                bus.DREAM_STATE_CHANGED,
                bus.MEDITATION_COMPLETE,
                bus.EXPRESSION_FIRED,
                bus.NEUROMOD_STATS_UPDATED,
                bus.MODULE_SHUTDOWN,
                bus.SAVE_NOW,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # studio_worker — extracted per rFP_titan_hcl_l2_separation_strategy §4.K
        # + D-SPEC-63 (v1.9.4, 2026-05-16). Maker-greenlit Q1-Q4 inline.
        # (Renumbered from D-SPEC-57 / v1.8.3 → D-SPEC-63 / v1.9.4 at merge time
        # to resolve collisions with the parallel §4.D meditation_worker +
        # §4.G life_force_worker + backup unified + §4.L/§4.N/§4.H sessions
        # that landed D-SPEC-57..62 first.)
        # Owns StudioCoordinator (creative-render pipelines), data/studio_exports/
        # output directories (G21 single-writer), studio_state.bin SHM slot.
        # Adopts D-SPEC-46 event+Future-registry pattern for slow renders so
        # ALL work-RPC paths stay ≤5s per G19 strict (renders are async events,
        # NOT work-RPC; gallery is ≤2s allowlisted; stats is SHM-direct).
        # ALWAYS-ON autostart — no flag-gate.
        # See SPEC v1.9.4 §9.B `studio_worker` block + PLAN c44129ae.
        from titan_hcl.modules.studio_worker import studio_worker_main
        self.guardian.register(ModuleSpec(
            name="studio",
            layer="L2",
            entry_fn=studio_worker_main,
            config={
                "titan_id": self._full_config.get("network", {}).get(
                    "titan_id"),
                "expressive": self._full_config.get("expressive", {}),
                "inference": self._full_config.get("inference", {}),
            },
            rss_limit_mb=200,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.KERNEL_EPOCH_TICK,
                bus.MODULE_SHUTDOWN,
                bus.SAVE_NOW,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # dream_state_worker — extracted per rFP_titan_hcl_l2_separation_strategy §4.I
        # + D-SPEC-56 (v1.8.2, 2026-05-15). Maker-greenlit Q1-Q6 inline.
        # Owns dream_state.bin SHM slot (G21 single-writer), DREAM_STATE_CHANGED
        # canonical publisher (closes the latent Phase C silent-emit fleet-wide
        # bug — sole emitter was dead spirit_worker.py:3006/3007/3143/3144 under
        # l0_rust_enabled=true since cognitive_worker drives the dream lifecycle
        # but never emitted DREAM_STATE_CHANGED), _dream_inbox queue
        # (chat-during-dream buffering, deque maxlen=50, drains via
        # DREAM_INBOX_REPLAY on dream_end), and DREAM_WAKE_REQUEST routing hub
        # (forwards to cognitive_worker via DREAM_WAKE_FORWARD).
        # ALWAYS-ON autostart — no flag-gate.
        # See SPEC v1.8.2 §9.B `dream_state_worker` block + PLAN d4b6b37e.
        from titan_hcl.modules.dream_state_worker import (
            dream_state_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="dream_state",
            layer="L2",
            entry_fn=dream_state_worker_main,
            config={
                "titan_id": self._full_config.get("network", {}).get(
                    "titan_id"),
            },
            rss_limit_mb=200,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.DREAMING_STATE_UPDATED,
                bus.KERNEL_EPOCH_TICK,
                bus.MODULE_SHUTDOWN,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # observatory_worker — extracted per RFP_phase_c_titan_hcl_cleanup
        # Phase A+B (Track 2, 2026-05-21). Maker-greenlit inline.
        # Owns the two residual Observatory-output PRODUCTION loops carved out
        # of core/plugin.py: (A) the V4 real-time event bridge — translates
        # broadcast pulse/dream/neuromod/hormone/expression events to
        # OBSERVATORY_EVENT (dst="api") for the api_subprocess WebSocket fan-out
        # (D-SPEC-82: the api process is broadcast-free, so this MUST run in a
        # worker, not the api process); (B) the periodic ObservatoryDB history
        # snapshot loop (trinity/growth/v4/vital), reads SHM-direct (G18/INV-4),
        # writes via the observatory_writer daemon (IMW). NOT a writer-owner —
        # write serialization stays with observatory_writer. The
        # DREAM_INBOX_REPLAY → CHAT_REQUEST orchestration moved to agno_worker;
        # this worker only mirrors the replay to a WS event.
        # See SPEC §9.B `observatory_worker` block + RFP_phase_c_titan_hcl_cleanup.
        from titan_hcl.modules.observatory_worker import (
            observatory_worker_main,
            V4_EVENT_TYPES as _OBSERVATORY_V4_EVENT_TYPES,
        )
        self.guardian.register(ModuleSpec(
            name="observatory",
            layer="L3",
            entry_fn=observatory_worker_main,
            config={
                "titan_id": self._full_config.get("network", {}).get(
                    "titan_id"),
                "frontend": self._full_config.get("frontend", {}),
            },
            rss_limit_mb=150,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                *_OBSERVATORY_V4_EVENT_TYPES,
                bus.MODULE_SHUTDOWN,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # meditation_worker — extracted per rFP_titan_hcl_l2_separation_strategy §4.D
        # + D-SPEC-57 (v1.8.3, 2026-05-15). Maker-greenlit Q1-Q5 inline.
        # Owns the full meditation lifecycle: _meditation_tracker dict + M3
        # emergent driver + MeditationWatchdog + orchestrator loop + phase
        # state machine + post-completion side effects + meditation_state.bin
        # SHM slot writer (G21 single writer).
        # See SPEC v1.8.3 §9.B `meditation_worker` block.
        from titan_hcl.modules.meditation_worker import (
            meditation_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="meditation",
            layer="L2",
            entry_fn=meditation_worker_main,
            config={
                "titan_id": self._full_config.get("network", {}).get(
                    "titan_id"),
            },
            rss_limit_mb=150,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.MEDITATION_REQUEST,
                bus.MEDITATION_FORCE_END,
                bus.EXPRESSION_FIRED,
                bus.KERNEL_EPOCH_TICK,
                # MODULE_READY removed v1.29.0 — meditation_worker no longer
                # waits on memory's MODULE_READY broadcast per SPEC §11.G.2.5
                # (Guardian's pre-start activation guarantees dep readiness).
                bus.SAVE_NOW,
                bus.MODULE_SHUTDOWN,
            ],
            # SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — dependency-driven activation.
            # memory_worker is registered with autostart=False + lazy=True. Without
            # ENSURE_RUNNING here, meditation_worker's MEDITATION_REQUEST handler
            # blocks 300s on memory's MODULE_READY because no subprocess can reach
            # MemoryProxy._ensure_started (parent-process-only). Pre-§4.D extraction
            # this worked through plugin.py main; post-§4.D meditation_worker is a
            # subprocess. Guardian now pre-starts memory before meditation enters
            # its main loop.
            dependencies=[
                Dependency(
                    name="memory",
                    kind=DependencyKind.MODULE,
                    severity=DependencySeverity.CRITICAL,
                    action=DependencyAction.ENSURE_RUNNING,
                    check=lambda: self.guardian.is_running("memory"),
                ),
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # sovereignty_worker — extracted per rFP_titan_hcl_l2_separation_strategy
        # §4.L + D-SPEC-60 (v1.9.1, 2026-05-15; renumbered from v1.8.3/D-SPEC-57
        # at merge time due to collision with parallel §4.D meditation session).
        # Owns SovereigntyTracker (M10 GREAT CYCLE convergence tracker — 222 LOC,
        # logic/sovereignty.py) + data/sovereignty_state.json. Subscribes
        # SOVEREIGNTY_EPOCH + SOVEREIGNTY_CONFIRM_MAKER (api/webhook.py producer)
        # + SOVEREIGNTY_INCREMENT_GREAT_CYCLE (api/maker.py producer) — closes
        # latent api_subprocess kernel_rpc serialization gap on M10 Mainnet
        # Lifecycle mutator paths.
        # See SPEC v1.9.1 §9.B `sovereignty_worker` block.
        from titan_hcl.modules.sovereignty_worker import (
            sovereignty_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="sovereignty",
            layer="L2",
            entry_fn=sovereignty_worker_main,
            config={},
            rss_limit_mb=150,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.MODULE_SHUTDOWN,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # interface_advisor_worker — extracted per rFP_titan_hcl_l2_separation
        # _strategy §4.H + D-SPEC-62 (v1.9.3, 2026-05-15; renumbered from
        # v1.8.5/D-SPEC-59 at merge time due to collision with parallel §4.D
        # meditation + §4.G life_force + backup unified sessions). Maker-greenlit
        # Path Y (SHM-rate-oracle pattern) inline. Owns InterfaceAdvisor (162
        # LOC per-msg-type sliding-window rate limiter, logic/interface_advisor.py)
        # + interface_advisor_state.bin SHM slot (G21 single-writer; 10Hz cap).
        # Parent `_handle_impulse` reads SHM via InterfaceAdvisorStateReader
        # (sub-µs G18, 100ms cache) instead of in-proc advisor.check().
        # See SPEC v1.9.3 §9.B `interface_advisor_worker` block.
        from titan_hcl.modules.interface_advisor_worker import (
            interface_advisor_worker_main,
        )
        self.guardian.register(ModuleSpec(
            name="interface_advisor",
            layer="L2",
            entry_fn=interface_advisor_worker_main,
            config={
                "titan_id": self._full_config.get("network", {}).get(
                    "titan_id"),
            },
            rss_limit_mb=100,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=[
                bus.MODULE_SHUTDOWN,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            critical_data_writer=False,
        ))

        # Phase A+B compatibility (2026-05-08, rFP_trinity_130d_phase2_5_closure):
        # The original chunk 8M.1 registration block was nested inside the
        # `l0_rust_enabled` gate, which left T1+T2 (Phase A+B Python only,
        # l0_rust_enabled=false) with hormonal_state.bin / neuromod_state.bin
        # / titanvm_registers.bin slots EMPTY. spirit_proxy.get_trinity()
        # post-Phase-C-Session-1 is SHM-direct — it reads those slots — so
        # hormone_levels was being returned as zeros, cascading through
        # outer_sources broadcast → mind_worker / outer_*_worker plugin_cache
        # → 16+ inner_mind/outer_spirit dims classified PARTIAL.
        #
        # The workers themselves gate on `shm_*_enabled` (their own per-worker
        # flag, all default true in titan_params.toml). So they're safe to
        # register independent of l0_rust_enabled — that flag is about the
        # Rust kernel-rs binary, not the Python state writers.
        #
        # SPEC §9.B Python tree + §9.C contract. Boot order: workers spawn
        # alongside others; the SHM-direct producers don't depend on
        # cognitive_worker boot order (they tick independently).
        # broadcast_topics minimal — these workers only consume MODULE_SHUTDOWN
        # from the bus.
        from titan_hcl.modules.ns_worker import ns_worker_main
        from titan_hcl.modules.neuromod_worker import neuromod_worker_main
        from titan_hcl.modules.hormonal_worker import hormonal_worker_main

        _state_worker_config = {
            # Pass-through full config — workers read their own
            # [neuromodulators] / [hormonal_pressure] / [neural_nervous_system]
            # sections + microkernel.shm_*_enabled flags via _build_*_system.
            **self._full_config,
            # data_dir helper used by hormonal_worker to load/save persisted state.
            "data_dir": self._full_config.get("memory_and_storage", {}).get(
                "data_dir", "./data"),
        }
        _mk = self._full_config.get("microkernel", {}) or {}

        # ns_module / neuromod_module / hormonal_module — state-slot owners
        # per SPEC §7.1 row 574. Each consumes:
        #   • KERNEL_EPOCH_TICK / EPOCH_TICK (broadcast `dst=all` from
        #     kernel-rs / coordinator) — drives per-epoch slot write
        #     (see {ns,neuromod,hormonal}_worker.py `is_epoch_tick` branch).
        #   • MODULE_SHUTDOWN — clean exit.
        # Per rFP_worker_broadcast_topics_completion §4.C (post-stopgap
        # retirement), broadcasts not in `broadcast_topics` are filtered
        # out by the broker, so the epoch tick MUST be declared here
        # explicitly. Pre-2026-05-12 the workers ran on the Python
        # subscribe-all fallback (and on the Rust broker which didn't
        # filter at all — discovered via 816,560 dropped msgs on T3 soak
        # gate); both paths are now closed by the §4.C contract.
        _STATE_WORKER_BROADCAST_TOPICS = [
            bus.MODULE_SHUTDOWN,
            bus.KERNEL_EPOCH_TICK,
            bus.EPOCH_TICK,
            # SAVE_NOW added 2026-05-15 — hormonal_worker / neuromod_worker /
            # ns_worker durability invariant. Pre-fix, hormonal_worker only
            # saved its state to data/hormonal_state.json on graceful
            # MODULE_SHUTDOWN — process kills silently lost accumulated
            # hormone state across restarts. Hormonal_worker now saves
            # every 30s + on SAVE_NOW; subscribing here ensures B.1
            # shadow_swap orchestrator + manual checkpoint requests reach
            # the worker.
            bus.SAVE_NOW,
        ]
        # ns_module + hormonal_module gain extra broadcast subs per
        # rFP_phase_c_impulse_engine_d8_3_migration §3.A.1 + §3.B.7-B.8:
        #   - ns_module subscribes ACTION_RESULT (IMPULSE outcome learning)
        #   - hormonal_module subscribes HORMONE_STIMULUS (cross-worker bridge)
        # ACTION_RESULT is published with dst="all" (plugin.py:2879 verbatim);
        # HORMONE_STIMULUS is published with dst="hormonal_module" but
        # per-Titan broker routing depends on broadcast_topics filter.
        _NS_WORKER_BROADCAST_TOPICS = _STATE_WORKER_BROADCAST_TOPICS + [
            bus.ACTION_RESULT,
        ]
        # Note: NS-program urgencies flow cross-process via the
        # `ns_program_urgencies_input.bin` SHM slot (G18-pure per SPEC
        # §7.1 + D-SPEC-68 v1.13.0), NOT a bus event. ns_worker polls
        # the slot each tick — no ModuleSpec subscription needed.
        _HORMONAL_WORKER_BROADCAST_TOPICS = _STATE_WORKER_BROADCAST_TOPICS + [
            bus.HORMONE_STIMULUS,
        ]
        if _mk.get("shm_ns_enabled", True):
            self.guardian.register(ModuleSpec(
                name="ns_module",
                layer="L2",  # NeuralNervousSystem owner — L2 cognitive support
                entry_fn=ns_worker_main,
                config=_state_worker_config,
                rss_limit_mb=400,   # NeuralReflexNet + 11 program registers; lean
                autostart=True,     # Required for /v4/nervous-system + titanvm_registers slot
                lazy=False,
                heartbeat_timeout=60.0,
                start_method="spawn" if _spawn_grad else "fork",
                broadcast_topics=_NS_WORKER_BROADCAST_TOPICS,
            ))
        if _mk.get("shm_neuromod_enabled", True):
            self.guardian.register(ModuleSpec(
                name="neuromod_module",
                layer="L2",  # NeuromodulatorSystem owner
                entry_fn=neuromod_worker_main,
                config=_state_worker_config,
                rss_limit_mb=400,
                autostart=True,     # Required for neuromod_state.bin slot freshness
                lazy=False,
                heartbeat_timeout=60.0,
                start_method="spawn" if _spawn_grad else "fork",
                broadcast_topics=_STATE_WORKER_BROADCAST_TOPICS,
            ))
        if _mk.get("shm_hormonal_enabled", True):
            self.guardian.register(ModuleSpec(
                name="hormonal_module",
                layer="L2",  # HormonalSystem owner
                entry_fn=hormonal_worker_main,
                config=_state_worker_config,
                rss_limit_mb=400,
                autostart=True,     # Required for hormonal_state.bin slot freshness
                lazy=False,
                heartbeat_timeout=60.0,
                start_method="spawn" if _spawn_grad else "fork",
                broadcast_topics=_HORMONAL_WORKER_BROADCAST_TOPICS,
            ))

        # Media module (image/audio perception — lazy, starts on first media)
        media_config = {
            "queue_dir": os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "media_queue"
            ),
        }
        # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
        # media drain at modules/media_worker.py:126-130 explicitly drops
        # anything that isn't SHUTDOWN or QUERY (both targeted). Pure
        # RPC-reply pattern — reply_only=True is canonical.
        self.guardian.register(ModuleSpec(
            name="media",
            layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (expression: speech/art/music)
            entry_fn=media_worker_main,
            config=media_config,
            rss_limit_mb=800,  # was 700 (was 500); fork-inherited parent memory grew (2026-04-17)
            autostart=True,   # Always on — art/audio generated frequently
            lazy=False,
            heartbeat_timeout=180.0,  # Image/audio digest can block 30-90s
            reply_only=True,
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        ))

        # Language module (composition, teaching, vocabulary — higher cognitive)
        language_config = {
            **self._full_config.get("language", {}),
            "data_dir": self._full_config.get("memory_and_storage", {}).get("data_dir", "./data"),
        }
        # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
        # See legacy_core.py for full type list; mirror exactly here.
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
            broadcast_topics=[
                bus.SPEAK_REQUEST, bus.LLM_TEACHER_RESPONSE,
                bus.META_LANGUAGE_RESULT, bus.MAKER_NARRATION_REQUEST,
                bus.CGN_DREAM_CONSOLIDATE, bus.CGN_CROSS_INSIGHT,
                bus.CGN_WEIGHTS_MAJOR, bus.CGN_KNOWLEDGE_RESP,
                bus.QUERY_RESPONSE, bus.SOCIAL_PERCEPTION,
                bus.CGN_SOCIAL_TRANSITION, bus.CGN_HAOV_VERIFY_REQ,
                bus.META_REASON_RESPONSE, bus.EPOCH_TICK,
                # Track 2 D2 (v1.3.1): WORD_PERTURBATION_HINT consumer for SPEAK
                # quality chain. Mirrors legacy_core.py registration. SPEC §8.5
                # D-SPEC-38.
                bus.WORD_PERTURBATION_HINT,
            ],
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
            # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
            # one broadcast type (META_CHAIN_COMPLETE).
            broadcast_topics=[bus.META_CHAIN_COMPLETE],
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
        # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4):
        # See legacy_core.py for full type list; mirror exactly here.
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
            broadcast_topics=[
                "CGN_TRANSITION", "CGN_REGISTER",
                "CGN_CONSOLIDATE", "CGN_SURPRISE",
                bus.CGN_HAOV_VERIFY_RSP, bus.CGN_INFERENCE_REQ,
                bus.CGN_KNOWLEDGE_REQ,
            ],
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
        # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4):
        # See legacy_core.py for full type list; mirror exactly here.
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
            broadcast_topics=[
                bus.CGN_KNOWLEDGE_REQ, bus.META_REASON_RESPONSE,
                bus.CGN_KNOWLEDGE_USAGE, bus.CGN_HAOV_VERIFY_REQ,
                bus.SEARCH_PIPELINE_BUDGET_RESET, bus.CGN_WEIGHTS_MAJOR,
                bus.CGN_CROSS_INSIGHT, bus.KNOWLEDGE_QUERY_CONCEPT,
                bus.KNOWLEDGE_SEARCH, bus.KNOWLEDGE_CONCEPTS_FOR_PERSON,
            ],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        # TimeChain — Proof of Thought memory chain
        timechain_config = {
            **self._full_config.get("timechain", {}),
        }
        # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4 — heaviest):
        # See legacy_core.py for full 23-type list; mirror exactly here.
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
            broadcast_topics=[
                # System-upgrade events (5)
                bus.SYSTEM_UPGRADE_QUEUED, bus.SYSTEM_UPGRADE_STARTING,
                bus.SYSTEM_RESUMED, bus.SYSTEM_UPGRADE_PENDING_DEFERRED,
                bus.SYSTEM_UPGRADE_THOUGHT,
                # Core timechain events (7)
                bus.TIMECHAIN_COMMIT, bus.EPOCH_TICK, bus.DREAM_STATE_CHANGED,
                bus.MEDITATION_COMPLETE, bus.EXPRESSION_FIRED,
                bus.TIMECHAIN_STATUS, bus.TIMECHAIN_QUERY,
                # Timechain query ops (5)
                bus.TIMECHAIN_RECALL, bus.TIMECHAIN_CHECK,
                bus.TIMECHAIN_COMPARE, bus.TIMECHAIN_AGGREGATE,
                bus.TIMECHAIN_SIMILAR,
                # Contract events (6)
                bus.CONTRACT_DEPLOY, bus.CONTRACT_LIST, bus.CONTRACT_STATUS,
                bus.CONTRACT_PROPOSE, bus.CONTRACT_APPROVE, bus.CONTRACT_VETO,
            ],
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
            rss_limit_mb=500,     # Phase 5 / 5G (2026-05-19): reverted 1200 → 500 after streaming
                                  # encoders shipped (5A). Previously bumped 800 → 1200 was an anti-pattern
                                  # (per `feedback_no_rss_band_aid_understand_root_cause`) masking the real
                                  # bug: full_ship.py:38 loaded entire file into Python bytes object via
                                  # `patch = f.read()`. For inner_memory.db (1.1 GB) that meant 1.1 GB held
                                  # in RAM; ship_tier accumulated ALL files' patch_bytes simultaneously
                                  # → ~1700-2000 MB peak RSS for a 30 MB output tarball (67× amplification).
                                  # With 5A streaming (patch_path on disk, tar.addfile reads from fd in
                                  # blocksize chunks, no Python bytes materialization), backup_worker peak
                                  # RSS expected ~200-350 MB. 500 MB is generous + acts as regression sentinel.
            autostart=True,       # Must be ready by 1st meditation of the day
            lazy=False,
            heartbeat_timeout=600.0,  # Personality+TimeChain tarball build + upload can take 4-6 min
            reply_only=False,     # Subscribes to MEDITATION_COMPLETE + BACKUP_TRIGGER_MANUAL broadcasts
            start_method="spawn" if _spawn_ref else "fork",  # S6 (§A.3)
            b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
            # 2026-04-30 — broadcast topic filter (closes 09:55 backup-queue
            # flood regression from spawn_graduated activation: SPHERE_PULSE/
            # SPIRIT_STATE/SAGE_STATS broadcasts saturating queue capacity).
            # Only these 2 broadcast types reach backup; targeted dst="backup"
            # messages (SAVE_NOW, MODULE_SHUTDOWN, RESUME) are unaffected by
            # broadcast filter — broker routes them by name match.
            broadcast_topics=[
                bus.MEDITATION_COMPLETE,
                bus.BACKUP_TRIGGER_MANUAL,
            ],
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
        # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
        # See legacy_core.py for full type list; mirror exactly here.
        from titan_hcl.logic.emot_kin_protocol import (
            KIN_EMOT_STATE_MSG_TYPE)
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
            broadcast_topics=[
                bus.EMOT_CHAIN_EVIDENCE, bus.FELT_CLUSTER_UPDATE,
                bus.META_REASON_RESPONSE, bus.CGN_HAOV_VERIFY_REQ,
                bus.CGN_CROSS_INSIGHT, KIN_EMOT_STATE_MSG_TYPE,
                bus.HORMONE_FIRED,
                # bus.CGN_BETA_SNAPSHOT RETIRED v1.14.0 / D-SPEC-69 — flows
                # via cgn_beta_state.bin SHM slot now (G18-pure per
                # rFP_dead_dim_wiring_fix §2.F).
                # RFP_meta-reasoning_CGN_FIX.md §8 Stage 1 — direct
                # substrate subscriptions for the 2 substrate keys with
                # clean ordered-vector schemas. Retires the
                # spirit_worker `_attach_emot_producer_ctx` bridge for
                # these 2 keys (rFP §1.2 DEAD-DIM root cause: spirit
                # bridge broke post fleet Phase C migration 2026-05-14).
                # All 5 carry flat ordered-vector payloads tailored
                # to HDBSCAN bundle assembly (distinct from the verbose
                # NEUROMOD_STATS_UPDATED / TOPOLOGY_STATE_UPDATED /
                # per-layer SPHERE_PULSE existing events, which feed
                # dashboards). Together these close DEAD-DIM ×3-6
                # fleet-wide (rFP §1.2 root cause: post fleet Phase C
                # migration 2026-05-14 the spirit_worker bridge stopped
                # attaching → meta_engine._last_* attrs zero → ctx
                # zero-fill → HDBSCAN sees zero variance).
                # bus.TRAJECTORY_UPDATE RETIRED v1.14.0 / D-SPEC-69 — flows via
                # trajectory_state.bin SHM slot now (G18-pure per rFP_dead_dim_wiring_fix §2.E).
                bus.NS_URGENCIES_UPDATE,      # ns_worker → emot_cgn (NEW v1.9.5)
                bus.SPACE_TOPOLOGY_UPDATE,    # cognitive_worker → emot_cgn (NEW v1.9.5)
                bus.NEUROMOD_LEVELS_UPDATE,   # neuromod_worker → emot_cgn (NEW v1.9.5)
                bus.PI_PHASE_UPDATE,          # cognitive_worker → emot_cgn (NEW v1.9.5)
            ],
            start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        ))

        logger.info("[TitanHCL] Registered %d supervised modules", len(self.guardian._modules))

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
                "[TitanHCL] api_subprocess NOT registered "
                "(microkernel.api_process_separation_enabled=False) — "
                "legacy in-process uvicorn will start in Phase 5")
            return

        from titan_hcl.api.api_subprocess import api_subprocess_main
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
            # 2026-05-01: bumped 300 → 800MB. Original 300 assumed
            # "asyncio + sqlite3 + msgpack only; stays lean" but the
            # api subprocess actually loads full FastAPI + dashboard
            # (60+ endpoints) + chat + maker + webhook + RPC client +
            # uvicorn = 600-700MB baseline observed today. 300 caused
            # crash-loop disable (5 restarts in 600s → Guardian disable
            # for 600s → T1 API offline cycle). Bump is a tourniquet —
            # the underlying baseline-too-high investigation is part
            # of the multi-week RSS battle (BUG-PARENT-MEMORY-LEAK-
            # HOST-OOM thread). Real fix needs heap-dump on stable
            # api process to identify allocator hotspots.
            rss_limit_mb=800,
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            layer="L3",
            # Phase D D-SPEC-82 (2026-05-18): the 16 *_UPDATED broadcast
            # subscriptions that fed the api_subprocess's BusSubscriber
            # were retired with the entire bus-cache → CachedState
            # pipeline. api_subprocess now reads SHM-direct via
            # ShmReaderBank + TitanStateAccessor (Preamble G18). Only
            # targeted dst="api" messages remain — OBSERVATORY_EVENT
            # (websocket bridge), CHAT_RESPONSE / CHAT_STREAM_CHUNK
            # (rid-routed Futures), RELOAD — and those bypass the
            # broadcast filter via name-based routing.
            broadcast_topics=[],
            # reply_only=True closes the D-SPEC-82 follow-on regression: with
            # broadcast_topics=[] AND reply_only=False the socket broker logs
            # a WARN+drop for EVERY dst="all" broadcast routed to this
            # subscriber (broker.rs:690 "empty broadcast_topics AND
            # reply_only=false"). On the fleet that flooded /var/log/syslog to
            # ~40GB. Since the api subprocess consumes ZERO broadcasts (only
            # targeted dst="api", which bypass the reply_only check per SPEC
            # §8.2 v1.4.0 D-SPEC-42), reply_only=True is the contracted
            # silent-skip path — the WARN's own remedy text.
            reply_only=True,
        ))
        logger.info(
            "[TitanHCL] api_subprocess registered as L3 module "
            "(api_process_separation_enabled=True; uvicorn moves to subprocess)")

    # ------------------------------------------------------------------
    # Proxy Creation — lifted verbatim from v5_core.py:697-735
    # ------------------------------------------------------------------

    def _create_proxies(self) -> None:
        """Create proxy objects that bridge V2 API calls to V3 bus-supervised modules.

        Lifted verbatim from v5_core.py:697-735 per PLAN §4.2 Commit 3.
        `self.bus` and `self.guardian` work via compat @property delegates.
        """
        from titan_hcl.proxies.memory_proxy import MemoryProxy
        from titan_hcl.proxies.rl_proxy import RLProxy
        from titan_hcl.proxies.llm_proxy import LLMProxy
        from titan_hcl.proxies.mind_proxy import MindProxy
        from titan_hcl.proxies.body_proxy import BodyProxy
        from titan_hcl.proxies.media_proxy import MediaProxy
        from titan_hcl.proxies.timechain_proxy import TimechainProxy
        from titan_hcl.proxies.social_graph_proxy import SocialGraphProxy
        from titan_hcl.proxies.agno_proxy import AgnoProxy
        # rFP_phase_c_state_read_unification §B.5 — SpiritProxy retired
        # 2026-05-18 in favor of ShmReaderBank.compose_trinity which reads
        # PURELY from Rust L0+L1 canonical SHM slots per the Maker
        # directive. The bank is the canonical trinity composer fleet-wide.
        from titan_hcl.api.shm_reader_bank import ShmReaderBank

        # Lazy modules — start on first use
        self._proxies["memory"] = MemoryProxy(self.bus, self.guardian)
        self._proxies["recorder"] = RLProxy(self.bus, self.guardian)
        self._proxies["llm"] = LLMProxy(self.bus, self.guardian)

        # Always-on modules — already started by Guardian
        self._proxies["mind"] = MindProxy(self.bus, self.guardian)
        self._proxies["body"] = BodyProxy(self.bus, self.guardian)
        # ShmReaderBank is the trinity composer post-Phase-B.5 (no proxy)
        self._shm_reader_bank = ShmReaderBank()

        # Media module (lazy — starts on first use)
        self._proxies["media"] = MediaProxy(self.bus, self.guardian)

        # TimeChain v2 Consumer API proxy
        self._proxies["timechain"] = TimechainProxy(self.bus, self.guardian)

        # V2-compatible aliases (so dashboard/agent code finds what it expects)
        self._proxies["mood_engine"] = self._proxies["mind"]  # mind proxy has get_mood_label()
        self._proxies["gatekeeper"] = self._proxies["recorder"]     # recorder proxy has evaluate()
        # social_graph — dedicated proxy + dedicated subprocess per
        # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1).
        # Replaces the legacy MindProxy alias rot that surfaced as
        # AttributeError 'MindProxy' object has no attribute
        # 'record_interaction_async' on every chat post-hook fleet-wide
        # (Maker 2026-05-12: "no right to be in microkernel Phase C
        # architecture that must be lean and fast").
        self._proxies["social_graph"] = SocialGraphProxy(self.bus, self.guardian)

        # SPEC v1.17.0 / D-SPEC-72 — agno_proxy installation. AgnoProxy wraps
        # the bus-RPC round-trip to agno_worker for /chat + /v4/pitch-chat
        # endpoints. Cached at app.state.agno_proxy by api factory at boot
        # (api/__init__.py reads plugin._proxies["agno"] → assigns to app.state).
        # 90s request timeout matches the existing chat-arun Layer-1 closure
        # for BUG-CHAT-AGENT-ARUN-HANG-T3-PHASE-C; allowlisted in
        # phase_c_rpc_exemptions.yaml as `agno_proxy → agno_worker` work-RPC.
        self._proxies["agno"] = AgnoProxy(self.bus, request_timeout_s=90.0)

        # ── V2 Subsystems (direct instances in Core) ──────────────────
        # _wire_metabolism/studio/social arrive in commit 4.
        # _wire_sovereignty RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
        # SovereigntyTracker now lives in sovereignty_worker subprocess.
        self._wire_metabolism()
        self._wire_meditation()  # §4.D v1.8.3 D-SPEC-57
        self._wire_life_force()  # §4.G v1.8.4 D-SPEC-58
        # _wire_sovereignty RETIRED v1.9.1 §4.L (D-SPEC-60, 2026-05-15) —
        # SovereigntyTracker now lives in sovereignty_worker subprocess.
        self._wire_studio()
        self._wire_social()

        logger.info("[TitanHCL] Created %d proxies", len(self._proxies))

    # ------------------------------------------------------------------
    # V2 Subsystem wiring — lifted verbatim from v5_core.py:737-869
    # ------------------------------------------------------------------

    def _wire_metabolism(self) -> None:
        """Install MetabolismProxy in self._proxies['metabolism'].

        Per rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51
        (SPEC v1.7.2, 2026-05-14). The inline MetabolismController
        instantiation that lived here previously has moved into
        `metabolism_worker` (separate subprocess, registered via the
        ModuleSpec block in `_register_modules`). titan_HCL holds only
        the proxy that exposes the same public surface — hot reads via
        SHM (sub-ms) + work-RPC for evaluate_gate + async state queries.

        The legacy `self.soul.set_metabolism(metabolism)` reverse-
        injection is REMOVED (Maker-locked 2026-05-14 "Replace with SHM
        read from soul"). Soul now constructs its own
        `MetabolismShmReader` for sub-ms tier/feature-flag reads — see
        `titan_hcl/core/soul.py`.
        """
        try:
            from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
            self._proxies["metabolism"] = MetabolismProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] MetabolismProxy installed (rFP §4.J + "
                "D-SPEC-51 — MetabolismController now hosted in "
                "metabolism_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] MetabolismProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_meditation(self) -> None:
        """Install MeditationProxy in self._proxies['meditation'].

        Per rFP_titan_hcl_l2_separation_strategy §4.D + D-SPEC-57
        (SPEC v1.8.3, 2026-05-15). The dual-process meditation
        orchestration (spirit_worker M3 driver + watchdog + tracker +
        MEDITATION_COMPLETE handler; plugin.py `_meditation_loop` +
        `_meditation_queue` pre-subscription; legacy_core mirror) has
        moved into `meditation_worker` (separate subprocess). titan_HCL
        holds only the proxy that exposes the public surface — get_tracker
        / get_watchdog_health / force_end (G18 SHM reads + fire-and-forget
        MEDITATION_FORCE_END bus publish).
        """
        try:
            from titan_hcl.proxies.meditation_proxy import MeditationProxy
            self._proxies["meditation"] = MeditationProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] MeditationProxy installed (rFP §4.D + "
                "D-SPEC-57 — meditation_tracker + driver + watchdog + "
                "orchestrator now hosted in meditation_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] MeditationProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_life_force(self) -> None:
        """Install LifeForceProxy in self._proxies['life_force'].

        Per rFP_titan_hcl_l2_separation_strategy §4.G + D-SPEC-58
        (SPEC v1.8.4, 2026-05-15). LifeForceEngine (Chi Λ 3×3 Trinity
        vitality math) was hosted in cognitive_worker chunk 8M.6 as Track 1
        drift since 2026-05-10; v1.8.4 extracts it into a dedicated
        life_force_worker subprocess. titan_HCL holds only the proxy that
        exposes chi state — hot reads via SHM (sub-µs) + work-RPC for
        get_stats / get_chi_history / get_contemplation_status.
        """
        try:
            from titan_hcl.proxies.life_force_proxy import LifeForceProxy
            self._proxies["life_force"] = LifeForceProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] LifeForceProxy installed (rFP §4.G + "
                "D-SPEC-58 — LifeForceEngine now hosted in "
                "life_force_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] LifeForceProxy wiring failed: %s", e,
                exc_info=True)

    # _wire_sovereignty RETIRED v1.9.1 §4.L (D-SPEC-60, 2026-05-15) —
    # SovereigntyTracker now lives in titan_hcl/modules/sovereignty_worker.py
    # as Guardian-supervised L2 subprocess.

    def _wire_studio(self) -> None:
        """Install StudioProxy in self._proxies['studio'].

        Per rFP_titan_hcl_l2_separation_strategy §4.K + D-SPEC-57
        (SPEC v1.8.3, 2026-05-15). The inline StudioCoordinator
        instantiation that lived here previously has moved into
        `studio_worker` (separate subprocess, registered via the
        ModuleSpec block in `_register_modules`). titan_HCL holds only
        the proxy that exposes the same public surface — fire-and-forget
        request_* + _with_completion variants (D-SPEC-46 Future-registry)
        + get_gallery_async (work-RPC ≤2s) + get_stats (SHM-direct).

        The provider-specific OllamaCloudClient injection at
        `__init__.py:238` is REMOVED — studio_worker constructs zero
        provider clients (Maker direction 2026-05-15 Q2). Haiku
        generation routes via the canonical llm_proxy.distill work-RPC
        inside the worker, abstracting over any future inference provider.
        """
        try:
            from titan_hcl.proxies.studio_proxy import StudioProxy
            self._proxies["studio"] = StudioProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] StudioProxy installed (rFP §4.K + "
                "D-SPEC-57 — StudioCoordinator now hosted in "
                "studio_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] StudioProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_social(self) -> None:
        """Wire SocialManager in Core (degraded mode — no API keys, but structure in place).

        Lifted verbatim from v5_core.py:838-869 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.expressive.social import SocialManager
            sage_cfg = self._full_config.get("stealth_sage", {})
            # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1,
            # 2026-05-14). The parent process no longer instantiates an
            # in-process SocialGraph — that hosting moved to
            # social_graph_worker (its own subprocess; G21 single-writer
            # of data/social_graph.db). SocialManager receives the
            # SocialGraphProxy instead: same async API surface (the
            # `*_async` methods SocialManager calls all exist on the
            # proxy), routed via bus.request_async work-RPC.
            social_graph = self._proxies.get("social_graph")
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
            logger.info("[TitanHCL] SocialManager wired (dry_run=%s)", social._dry_run)
        except Exception as e:
            logger.warning("[TitanHCL] SocialManager wiring failed: %s", e)

    # ------------------------------------------------------------------
    # Observatory API (reuses existing V2 create_app)
    # ------------------------------------------------------------------

    def _create_observatory_app(self, api_cfg: dict):
        """Create the Observatory FastAPI app synchronously.

        Lifted verbatim from v5_core.py:935-944 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.api import create_app
            app = create_app(self, self.event_bus, api_cfg)
            self._observatory_app = app
            return app
        except Exception as e:
            logger.warning("[TitanHCL] Observatory app creation failed: %s", e)
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
            logger.warning("[TitanHCL] Observatory could not bind port")
        except Exception as e:
            logger.warning("[TitanHCL] Observatory failed: %s", e)

    def reload_api(self) -> dict:
        """Hot-reload API routes by rebuilding the FastAPI app and swapping it.

        Returns dict with reload status. The uvicorn server keeps running —
        only the ASGI app reference changes. Zero downtime.

        Lifted verbatim from v5_core.py:965-988 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.api import reload_api_app
            old_app = self._observatory_app
            new_app = reload_api_app(old_app)
            self._observatory_app = new_app

            # Swap the app in the running uvicorn server
            if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
                self._uvicorn_server.config.app = new_app
                # Also update the loaded_app which uvicorn uses for serving
                if hasattr(self._uvicorn_server, 'config'):
                    self._uvicorn_server.config.loaded_app = new_app

            logger.info("[TitanHCL] API hot-reloaded — routes updated, server continuous")
            return {"status": "ok", "reloaded": True}
        except Exception as e:
            logger.error("[TitanHCL] API reload failed: %s", e)
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
            from titan_hcl.logic.reflex_executors import register_reflex_executors
            from titan_hcl.params import get_params

            reflex_cfg = get_params("reflexes")

            _reflex_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_reflex_subprocess_enabled", False))
            if _reflex_subproc_enabled:
                from titan_hcl.proxies.reflex_proxy import ReflexProxy
                self.reflex_collector = ReflexProxy(self.bus, reflex_cfg)
                logger.info(
                    "[TitanHCL] ReflexCollector using subprocess proxy "
                    "(A.8.5 flag enabled)")
            else:
                from titan_hcl.logic.reflexes import ReflexCollector
                self.reflex_collector = ReflexCollector(reflex_cfg)

            # Register executors — they wrap existing subsystems. Works
            # on both ReflexCollector + ReflexProxy (proxy inherits
            # register_executor + _executors from base).
            count = register_reflex_executors(self.reflex_collector, self)
            logger.info("[TitanHCL] ReflexCollector booted: %d executors, threshold=%.2f",
                        count, self.reflex_collector.fire_threshold)
        except Exception as e:
            logger.warning("[TitanHCL] ReflexCollector boot failed: %s", e)
            self.reflex_collector = None

    def _boot_agency(self) -> None:
        """Initialize Agency — local module OR subprocess proxy (L3 §A.8.6).

        Original behavior (flag off, default): instantiates AgencyModule +
        SelfAssessment + HelperRegistry + 8 helpers in parent.

        Subprocess mode (microkernel.a8_agency_subprocess_enabled=true):
        instantiates AgencyProxy + AssessmentProxy that bus.request() into
        agency_worker — all LLM calls + helper.execute() awaits run in
        the worker subprocess, parent event loop never blocks on them.

        InterfaceAdvisor MOVED to interface_advisor_worker subprocess in
        v1.8.5 §4.H (D-SPEC-59, 2026-05-15) per `feedback_phase_c_break_
        monolith_ethos.md` — every L2 carve under Phase C earns its place
        via hot-reload + restart-isolation + own §9.B block. Parent reads
        rate state from `interface_advisor_state.bin` SHM slot via
        InterfaceAdvisorStateReader (sub-µs G18, 100ms cache); rate checks
        emit IMPULSE_RECEIVED bus event (fire-and-forget P3) to worker.

        ExpressionTranslator stays in parent — it only needs the
        helper-names list (worker advertises via AGENCY_READY +
        AGENCY_STATS broadcast → proxy._registry facade).
        """
        try:
            from titan_hcl.logic.interface_advisor_reader import (
                InterfaceAdvisorStateReader,
            )

            agency_cfg = self._full_config.get("agency", {})
            if not agency_cfg.get("enabled", True):
                logger.info("[TitanHCL] Agency disabled by config")
                return

            # InterfaceAdvisor MOVED to interface_advisor_worker subprocess
            # (v1.8.5 §4.H, D-SPEC-59). Parent reads rate state from SHM.
            self._interface_advisor_reader = InterfaceAdvisorStateReader()

            # L3 §A.8.6 — flag-routed agency residency.
            agency_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_agency_subprocess_enabled", False))

            if agency_subproc_enabled:
                from titan_hcl.proxies.agency_proxy import AgencyProxy
                from titan_hcl.proxies.assessment_proxy import AssessmentProxy
                self._agency = AgencyProxy(self.bus)
                self._agency_assessment = AssessmentProxy(self.bus)
                # Helper-names list comes from agency_worker via
                # AGENCY_READY broadcast — empty until first broadcast
                # arrives (typically <1s after worker boots). Bootstrap
                # ExpressionTranslator with empty list; it'll use
                # whatever's cached in the proxy._registry at translate
                # time anyway.
                _initial_helpers: list[str] = []
                logger.info("[TitanHCL] Agency using subprocess proxy "
                            "(A.8.6 flag enabled) — helpers list will populate "
                            "from AGENCY_READY broadcast")
            else:
                from titan_hcl.logic.agency.registry import HelperRegistry
                from titan_hcl.logic.agency.module import AgencyModule
                from titan_hcl.logic.agency.assessment import SelfAssessment

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
                logger.info("[TitanHCL] Agency local mode: %d helpers registered "
                            "(%d available): %s",
                            len(helper_names), len(available), available)

            # Expression Translation Layer — learned action selection.
            # Reads helpers list at construction; runtime translate() uses
            # self._agency._registry.list_helper_names() which works for
            # both local registry and proxy _RegistryFacade.
            try:
                from titan_hcl.logic.expression_translator import (
                    ExpressionTranslator, FeedbackRouter)
                self._expression_translator = ExpressionTranslator(
                    all_helpers=_initial_helpers)
                self._expression_translator.load(
                    "./data/neural_nervous_system/expression_state.json")
                self._feedback_router = FeedbackRouter(
                    hormonal_system=None,  # Wired later when neural_ns available
                    translator=self._expression_translator)
                logger.info("[TitanHCL] ExpressionTranslator booted "
                            "(sovereignty=%.1f%%)",
                            self._expression_translator.sovereignty_ratio * 100)
            except Exception as e:
                logger.warning("[TitanHCL] Expression layer init error: %s", e)
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
            # 2026-05-19 BOOT_TRACE diagnostic — confirms agency subscription
            # registered. Without this log, we cannot tell whether the
            # subscriber is eligible for IMPULSE cross-process delivery.
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] _boot_agency complete — "
                "agency_queue=%s subscribed for IMPULSE/OUTER_DISPATCH/"
                "QUERY/AGENCY_STATS/ASSESSMENT_STATS/AGENCY_READY",
                type(self._agency_queue).__name__)

        except Exception as e:
            logger.warning("[TitanHCL] Agency boot failed: %s", e)
            self._agency = None

    def _register_helpers(self, registry) -> None:
        """Register all available helpers in the registry.

        Lifted verbatim from v5_core.py:1070-1175 per PLAN §4.2 Commit 4.
        Path adjustment: titan_params.toml location — v5_core is at
        titan_hcl/, plugin.py is at titan_hcl/core/, so ".." prefix
        added to resolve titan_hcl/titan_params.toml correctly.
        """
        try:
            from titan_hcl.logic.agency.helpers.infra_inspect import InfraInspectHelper
            registry.register(InfraInspectHelper(log_path="/tmp/titan_v3.log"))
        except Exception as e:
            logger.warning("[TitanHCL] InfraInspect helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.web_search import WebSearchHelper
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
            logger.warning("[TitanHCL] WebSearch helper failed: %s", e)

        # SocialPostHelper REMOVED — all posting goes through SocialPressureMeter
        # (social_narrator + quality gate + rate limits + 11 post types).
        # Agency selecting social_post bypassed our designed narrator entirely.

        try:
            from titan_hcl.logic.agency.helpers.art_generate import ArtGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            registry.register(ArtGenerateHelper(output_dir=output_dir))
        except Exception as e:
            logger.warning("[TitanHCL] ArtGenerate helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.audio_generate import AudioGenerateHelper
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
            logger.warning("[TitanHCL] AudioGenerate helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
            registry.register(CodingSandboxHelper())
        except Exception as e:
            logger.warning("[TitanHCL] CodingSandbox helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
            registry.register(CodeKnowledgeHelper())
        except Exception as e:
            logger.warning("[TitanHCL] CodeKnowledge helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.memo_inscribe import MemoInscribeHelper
            # MemoInscribeHelper reads config.toml directly for RPC + keypair.
            # Mainnet Lifecycle Wiring rFP: inject metabolism for memo gate
            # + governance reserve guard.
            registry.register(MemoInscribeHelper(
                metabolism=self._proxies.get("metabolism")))
        except Exception as e:
            logger.warning("[TitanHCL] MemoInscribe helper failed: %s", e)

        # Kin Discovery — consciousness-to-consciousness exchange
        try:
            from titan_hcl.logic.agency.helpers.kin_sense import KinSenseHelper
            import tomllib as _tomllib_kin
            _kin_params = {}
            # Plugin lives at titan_hcl/core/plugin.py — go up one to
            # titan_hcl/titan_params.toml.
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
                logger.info("[TitanHCL] KinSense helper registered: addresses=%s",
                            _kin_addrs)
        except Exception as e:
            logger.warning("[TitanHCL] KinSense helper failed: %s", e)

    def _create_agency_llm_fn(self):
        """Create a lightweight async LLM function for Agency module.

        Phase 3 Chunk χ-bis (D-SPEC-88, 2026-05-18) — direct OllamaCloudClient
        + Venice fallback REPLACED by /v4/llm-distill round-trip. Mirrors
        the agency_worker.py subprocess path so both parent + worker share
        the same centralized LLM gateway. All LLM traffic appears in
        llm_state.bin.
        """
        inference_cfg = self._full_config.get("inference", {})
        api_cfg = self._full_config.get("api", {}) or {}
        _api_port = int(api_cfg.get("port", 7777))
        _api_base = f"http://127.0.0.1:{_api_port}"
        _internal_key = api_cfg.get("internal_key", "") or ""

        async def agency_llm(prompt: str, task: str = "agency_select") -> str:
            """LLM call for helper selection / assessment / code generation."""
            try:
                from titan_hcl.inference import get_model_for_task
                from titan_hcl.logic.llm_distill_client import (
                    distill_via_http_async)
                model = get_model_for_task(task)
                max_tok = 800 if task == "agency_code_gen" else 200
                result = await distill_via_http_async(
                    text=prompt,
                    instruction="",
                    api_base=_api_base,
                    internal_key=_internal_key,
                    model=model,
                    max_tokens=max_tok,
                    consumer=f"agency.{task}",
                    timeout_s=30.0,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning("[Agency LLM] /v4/llm-distill failed: %s", e)

            raise RuntimeError("No LLM available for agency")

        return agency_llm

    # ------------------------------------------------------------------
    # plugin.run_chat() RETIRED in Phase C v1.17.0 (D-SPEC-72).
    # The chat pipeline now lives in agno_worker subprocess; /chat +
    # /v4/pitch-chat handlers route through agno_proxy.chat() per
    # SPEC §9.B agno_worker block + §9.F.2 llm_pipeline contract.
    # See titan-docs/rFP_agno_worker_and_llm_libraries_extraction.md.
    # ------------------------------------------------------------------



    async def _guardian_handler_loop(self) -> None:
        """Bus subscriber for QUERY dst="guardian" admin requests.

        Closes both BUG-HOT-RELOAD-CODE-LOADING and BUG-GUARDIAN-CONTROL-
        COMMANDS-ORPHAN. Implements the missing receiver for the long-
        documented Phase C admin pattern: api_subprocess publishes QUERY
        with `action ∈ {"restart_module", "start_module", "stop_module"}`
        + payload, this loop dispatches to Guardian methods on the parent
        process, RESPONSE rid-routed back via bus.

        Concurrency: each admin request may take 5-30s (worker restart +
        boot), so requests are dispatched to separate asyncio tasks so a
        slow restart doesn't serialize subsequent admin calls.
        """
        try:
            queue = self.bus.subscribe("guardian", types=[bus.QUERY])
        except Exception as e:
            logger.warning("[TitanHCL] guardian handler subscribe failed: %s", e)
            return
        logger.info("[TitanHCL] guardian handler loop started — listening for admin QUERY")
        while True:
            try:
                msgs = self.bus.drain(queue, max_msgs=20)
                for msg in msgs:
                    if msg.get("type") != bus.QUERY:
                        continue
                    payload = msg.get("payload") or {}
                    action = payload.get("action")
                    if action not in (
                        "restart_module", "start_module", "stop_module",
                        # SPEC §8.3 Phase B — per-module hot-reload (D-SPEC-50).
                        # `Guardian.reload_module()` is async; dispatched via
                        # asyncio.create_task in `_handle_guardian_request`.
                        "reload_module",
                    ):
                        continue
                    asyncio.get_event_loop().create_task(
                        self._handle_guardian_request(msg))
                await asyncio.sleep(0.1)  # 10 Hz drain — admin ops are rare
            except Exception as e:
                logger.error("[TitanHCL] guardian handler loop error: %s",
                             e, exc_info=True)
                await asyncio.sleep(2.0)

    async def _handle_guardian_request(self, msg: dict) -> None:
        """Process one Guardian admin request.

        Supported actions:
          - `restart_module(name, reason, start_method=None)` → calls
             `Guardian.restart_module()` (BUG-HOT-RELOAD-CODE-LOADING).
          - `start_module(name)` → `Guardian.start()` (BUG-GUARDIAN-CONTROL).
          - `stop_module(name, reason)` → `Guardian.stop()` (BUG-GUARDIAN-CONTROL).
        """
        rid = msg.get("rid")
        src = msg.get("src", "api")
        payload = msg.get("payload") or {}
        action = payload.get("action")
        inner = payload.get("payload") or {}
        result: dict = {"ok": False, "error": "unknown_action"}
        try:
            if action == "restart_module":
                name = inner.get("name")
                reason = inner.get("reason", "bus request")
                start_method = inner.get("start_method")
                # Off-load to executor — Guardian.restart_module() blocks
                # in stop() + start() (worker SIGTERM + new spawn). Keep
                # the asyncio loop free for concurrent admin requests.
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.restart_module(
                        name=name, reason=reason,
                        start_method=start_method))
            elif action == "start_module":
                name = inner.get("name")
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.start(name))
                result = {"ok": bool(ok), "module": name, "error": None}
            elif action == "stop_module":
                name = inner.get("name")
                reason = inner.get("reason", "bus request")
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.stop(name, reason=reason))
                result = {"ok": True, "module": name, "error": None}
            elif action == "reload_module":
                # SPEC §8.3 Phase B — per-module hot-reload (D-SPEC-50).
                # `Guardian.reload_module()` is async — awaitable directly.
                # The dispatch path (chat_bridge → QUERY → here) is what
                # `arch_map reload-module` and POST /v4/admin/reload-module
                # use to initiate from outside the parent process.
                name = inner.get("name")
                new_module_path = inner.get("new_module_path")
                timeout_s = float(inner.get("timeout_s", 30.0))
                result = await self.guardian.reload_module(
                    module_name=name,
                    new_module_path=new_module_path,
                    timeout_s=timeout_s,
                )
                # reload_module returns {swap_id, module_name, status,
                # reason, total_elapsed_ms, ts} per SPEC §8.3 — add `ok`
                # flag for caller convenience without altering shape.
                result["ok"] = (result.get("status") == "ready")
        except Exception as e:
            logger.error("[TitanHCL] guardian handler %s raised: %s",
                         action, e, exc_info=True)
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            self.bus.publish({
                "type": bus.RESPONSE,
                "src": "guardian",
                "dst": src,
                "rid": rid,
                "payload": result,
                "ts": time.time(),
            })
            # SPEC §8.0.ter — flush before returning so the RESPONSE
            # reaches the requester's reply_queue before the handler
            # completes. Same rationale as _handle_chat_request flush:
            # the requester is awaiting via bus.request_async + rid.
            self.bus.flush(timeout=2.0)
        except Exception as e:
            logger.error("[TitanHCL] guardian RESPONSE publish failed: %s", e)

    # ------------------------------------------------------------------
    # Agency / Sovereignty / Impulse loops + handlers
    # ------------------------------------------------------------------
    # All lifted verbatim from v5_core.py:1218-1585 per PLAN §4.2 Commit 5.
    # Absolute imports replace relative imports.

    async def _agency_loop(self) -> None:
        """Listen for IMPULSE events on the bus and process them through Agency.

        Flow: IMPULSE → InterfaceAdvisor rate check → Agency handles → Assessment → ACTION_RESULT
        """
        # 2026-05-19 BOOT_TRACE — WARNING level so it survives journal flood
        logger.warning("[TitanHCL] [BOOT_TRACE] _agency_loop entered — "
                       "agency_queue type=%s",
                       type(getattr(self, "_agency_queue", None)).__name__)
        _bt_polls = 0
        _bt_msgs = 0
        _bt_last_log = time.time()
        while True:
            try:
                # Poll bus for IMPULSE messages addressed to agency or broadcast
                msg = None
                try:
                    msg = self._agency_queue.get_nowait()
                except Exception:
                    pass

                # 2026-05-19 BOOT_TRACE heartbeat — every 60s emit poll count
                # so we can see the loop is alive AND whether it's receiving
                # any messages. Cheap (1 log/min).
                _bt_polls += 1
                _now_bt = time.time()
                if _now_bt - _bt_last_log >= 60.0:
                    logger.warning(
                        "[TitanHCL] [BOOT_TRACE] _agency_loop alive — "
                        "polls=%d msgs_received=%d in last 60s",
                        _bt_polls, _bt_msgs)
                    _bt_polls = 0
                    _bt_msgs = 0
                    _bt_last_log = _now_bt

                if not msg:
                    await asyncio.sleep(2.0)  # Poll every 2s (impulses are rare)
                    continue

                _bt_msgs += 1
                msg_type = msg.get("type", "")
                # 2026-05-19 BOOT_TRACE — every msg received logged
                logger.warning(
                    "[TitanHCL] [BOOT_TRACE] msg received type=%s src=%s "
                    "dst=%s", msg_type, msg.get("src"), msg.get("dst"))

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
                            logger.info("[TitanHCL] AGENCY_READY: worker advertises "
                                        "%d helpers — proxy cache seeded", len(helpers))

            except Exception as e:
                logger.error("[TitanHCL] Agency loop error: %s", e)
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
        rl = self._proxies.get("recorder")
        if rl is None or getattr(rl, "_stats_subscription", None) is None:
            logger.info("[TitanHCL] RL stats loop skipped — no subscription")
            return
        queue = rl._stats_subscription
        logger.info("[TitanHCL] RL stats loop started — draining rl_proxy_stats")
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
                                "[TitanHCL] RL update_cached_stats raised: %s", e
                            )
                # Tight cadence — broadcast volume can hit ~120 msg/sec at peak;
                # drain at 2 Hz with batch-1000 keeps queue bounded at ~60 msgs.
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("[TitanHCL] RL stats loop error: %s", e)
                await asyncio.sleep(5.0)

    # _sovereignty_loop RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
    # SOVEREIGNTY_EPOCH consumption + tracker.record_epoch() + 100-message
    # criteria-snapshot log all moved to sovereignty_worker subprocess.

    # _sovereignty_queue property RETIRED v1.8.3 §4.L (D-SPEC-57) — kernel.py
    # bus.subscribe("sovereignty", types=[SOVEREIGNTY_EPOCH]) also retired
    # (worker subscribes directly via its own bus client).

    # _meditation_queue property REMOVED — meditation_worker subprocess
    # subscribes to MEDITATION_REQUEST via its own bus client (D-SPEC-57).

    async def _handle_impulse(self, msg: dict) -> None:
        """Process an IMPULSE event through the agency pipeline.

        Lifted verbatim from v5_core.py:1310-1481.
        """
        payload = msg.get("payload", {})
        posture = payload.get("posture", "unknown")
        impulse_id = payload.get("impulse_id", 0)

        logger.info("[TitanHCL] IMPULSE received: #%d posture=%s urgency=%.2f",
                    impulse_id, posture, payload.get("urgency", 0))

        # Rate check via SHM-rate-oracle (v1.8.5 §4.H, D-SPEC-59).
        # InterfaceAdvisor lives in interface_advisor_worker subprocess;
        # parent reads SHM snapshot sub-µs + (on within-limits) emits
        # IMPULSE_RECEIVED to worker so it records the timestamp + republishes.
        if self._interface_advisor_reader:
            feedback = self._interface_advisor_reader.check(IMPULSE, source="spirit")
            if feedback:
                # Phase D (D-SPEC-116): the RATE_LIMIT→spirit notify emit was
                # removed with spirit_worker (its handler only ever logged). The
                # rate-limit enforcement itself (the early return below) and the
                # local log line above are the real behavior — both retained.
                logger.info(
                    "[TitanHCL] IMPULSE rate-limited: current_rate=%d limit=%d",
                    feedback.get("current_rate", 0), feedback.get("limit", 0))
                return
            # Within-limits — fire-and-forget IMPULSE_RECEIVED to worker so
            # it records the timestamp in its sliding-window deque and
            # republishes interface_advisor_state.bin SHM (rate-throttled).
            try:
                self.bus.publish(make_msg(
                    bus.IMPULSE_RECEIVED, "core", "interface_advisor",
                    {"msg_type": IMPULSE, "source": "spirit",
                     "client_ts": time.time()},
                ))
            except Exception as e:
                logger.debug(
                    "[TitanHCL] IMPULSE_RECEIVED emit failed: %s", e)

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
                    logger.info("[TitanHCL] Expression: learned %s→%s (conf=%.2f)",
                                payload.get("triggering_program", "?"),
                                learned_selection["helper"],
                                learned_selection.get("confidence", 0))
                else:
                    self._expression_translator.record_action_type(was_learned=False)
            except Exception as e:
                logger.warning("[TitanHCL] Expression translator error: %s", e)

        # Agency Module handles the intent
        result = await self._agency.handle_intent(intent)
        if not result:
            logger.info("[TitanHCL] Agency skipped impulse #%d (no action taken)", impulse_id)
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
                logger.info("[TitanHCL] Assessment: score=%.2f direction=%s — %s",
                           assessment["score"], assessment["threshold_direction"],
                           assessment["reflection"][:80])
            except Exception as e:
                logger.warning("[TitanHCL] Assessment failed: %s", e)

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
                "[TitanHCL] Skipping ACTION_RESULT with empty helper — "
                "gate/rate-limit path (success=%s)", result.get("success"))
        else:
            self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
            logger.info("[TitanHCL] ACTION_RESULT published: helper=%s success=%s",
                        _helper, result.get("success"))

        # Feed action result to cognitive_worker for OBSERVATION (closed loop).
        # Phase D (D-SPEC-116): repointed from the retired spirit_worker; only
        # the X-engagement→MSL leg is live there now (others superseded/re-homed).
        try:
            self.bus.publish(make_msg(
                OUTER_OBSERVATION, "core", "cognitive_worker", {
                    "action_type": result.get("helper", ""),
                    "result": result,
                    "source": "impulse",
                }))
        except Exception as e:
            logger.warning("[TitanHCL] OUTER_OBSERVATION publish error: %s", e)

        # Record in Inner Memory (Phase M: action chain + event markers)
        try:
            if not hasattr(self, '_inner_memory'):
                from titan_hcl.logic.inner_memory import InnerMemoryStore
                self._inner_memory = InnerMemoryStore("./data/inner_memory.db")
            mem = self._inner_memory
            if mem:
                # Coalesce None → "" so `action_chains.helper TEXT NOT NULL`
                # never receives NULL. `.get(key, default)` only honors the
                # default on missing keys, not explicit None values.
                helper_name = result.get("helper") or ""
                # Bug B closure 2026-05-13: assessment score lives at
                # result["assessment"]["score"] (set above ~line 2864),
                # NOT result["score"]. Reading the wrong key left every
                # action_chains.score = 0.0 (14573 rows), which broke
                # practiced_response pool B (filters on
                # action_chains.score>=0.7). Fall back to
                # result["score"] for callers that bypass assessment.
                _assessment_score = (
                    (result.get("assessment") or {}).get("score")
                    if (result.get("assessment") or {}).get("score") is not None
                    else result.get("score", 0.0)
                )
                mem.record_action_chain(
                    impulse_id=result.get("impulse_id", 0),
                    triggering_program=result.get("triggering_program", ""),
                    posture=result.get("posture", ""),
                    helper=helper_name,
                    success=result.get("success", False),
                    score=_assessment_score,
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
                        # Same fix as action_chains above — assessment
                        # score lives at result["assessment"]["score"].
                        assessment_score=_assessment_score,
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
                        # Inner perception fan-out happens via observatory_db
                        # hook (utils/observatory_db.record_expressive). One
                        # hook on the canonical write site covers all callers.
                elif helper_name == "infra_inspect":
                    mem.record_event("inspect", program="VIGILANCE")
                elif helper_name == "kin_sense":
                    mem.record_event("kin", program="EMPATHY")
        except Exception as e:
            logger.warning("[TitanHCL] Inner memory recording error: %s", e)

        # Expression Layer: route feedback + save state
        try:
            if self._feedback_router:
                self._feedback_router.route(result)
            if self._expression_translator:
                self._expression_translator.save(
                    "./data/neural_nervous_system/expression_state.json")
        except Exception as e:
            logger.warning("[TitanHCL] Expression feedback error: %s", e)

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
        logger.info("[TitanHCL] OUTER_DISPATCH: %d signals from %s",
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
                    "[TitanHCL] Skipping AUTONOMY ACTION_RESULT with empty "
                    "helper — posture=%s success=%s",
                    result.get("posture"), result.get("success"))
            else:
                self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
                logger.info("[TitanHCL] AUTONOMY ACTION: %s → %s (success=%s)",
                            result.get("posture"), _auto_helper,
                            result.get("success"))

            # Feed action result to cognitive_worker for OBSERVATION (closed
            # loop). Phase D (D-SPEC-116): repointed from the retired
            # spirit_worker (X-engagement→MSL leg; others superseded/re-homed).
            try:
                self.bus.publish(make_msg(
                    OUTER_OBSERVATION, "core", "cognitive_worker", {
                        "action_type": result.get("helper", ""),
                        "result": result,
                        "source": payload.get("source", "nervous_system"),
                    }))
            except Exception as e:
                logger.warning("[TitanHCL] OUTER_OBSERVATION publish error: %s", e)

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
            if self._interface_advisor_reader:
                # v1.8.5 §4.H (D-SPEC-59): stats come from SHM via the
                # compat-shim get_stats() on InterfaceAdvisorStateReader.
                stats["advisor"] = self._interface_advisor_reader.get_stats()
            self.bus.publish(make_msg(bus.RESPONSE, "core", src, stats, rid))

    # ------------------------------------------------------------------
    # Background Loops — Social, Meditation, Outer Trinity
    # ------------------------------------------------------------------
    # _v4_event_bridge_loop + _trinity_snapshot_loop EXTRACTED 2026-05-21
    # (RFP_phase_c_titan_hcl_cleanup Phase A+B) → observatory_worker
    # (titan_hcl/modules/observatory_worker.py). The parent no longer
    # bridges bus events to OBSERVATORY_EVENT nor writes ObservatoryDB history;
    # both are owned by the L3 observatory_worker. No resurrection.
    # ------------------------------------------------------------------

    # _social_engagement_loop REMOVED 2026-05-21 (D-SPEC-106). It was dead
    # code — its create_task was commented out at boot (never scheduled), and
    # mention polling + engagement is owned by social_worker (per-Titan
    # canonical-poller mode: MENTION_RECEIVED / SOCIAL_RECEIVED dispatch +
    # SHM publish, see titan_hcl/modules/social_worker.py). The parent's
    # ad-hoc SocialManager.monitor_and_engage poll loop (lifted from the
    # retired v5_core.py) is superseded; no resurrection.
    # ------------------------------------------------------------------
    # _meditation_loop REMOVED — extracted to meditation_worker subprocess per
    # rFP_titan_hcl_l2_separation_strategy §4.D + D-SPEC-57 (SPEC v1.8.3,
    # 2026-05-15). The dual-trigger orchestrator (bus MEDITATION_REQUEST +
    # fixed timer fallback), memory readiness probe, run_meditation_async
    # 300s G19 work-RPC, observatory record, MEDITATION_COMPLETE 3-target
    # fan-out, EPOCH_TICK emit — all now live in
    # titan_hcl/modules/meditation_worker.py. The _meditation_queue
    # pre-subscription in kernel.__init__ is RETIRED too (next chunk).
    # ~225 LOC retired from this file per D-SPEC-57 Q5 aggressive-cleanup.
    # Studio art generation now lives in meditation_worker as well — wired
    # via studio_proxy.generate_meditation_art_with_completion (v1.9.4 §4.K
    # D-SPEC-63 closure of post-§4.D regression).
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Agent (Agno) — delegates to LLM module
    # ------------------------------------------------------------------

    def create_agent(self):
        """Create the Agno sovereign agent.

        In V3 this delegates to the LLM module via proxy.
        Lifted verbatim from v5_core.py:2348-2356.
        """
        # D-SPEC-72: factory moved to modules/agno_agent_factory.py. Under
        # the Phase C path this method is rarely called (api_worker routes
        # /chat through agno_proxy → agno_worker.arun); retained for
        # back-compat with the legacy MCP / direct-invocation paths.
        from titan_hcl.modules.agno_agent_factory import create_agent as _create_agent
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
        from titan_hcl.api.events import EventBus
        self.event_bus = EventBus()

        # Observatory DB for persistent historical metrics.
        # rFP_universal_sqlite_writer Phase 2 — per-process singleton.
        from titan_hcl.utils.observatory_db import get_observatory_db
        self._observatory_db = get_observatory_db()
        self.event_bus.attach_db(self._observatory_db)

        # rFP_trinity_130d_awakening Phase 2 — start InnerPerceptionState.
        # AmbientChangeMonitor samples (cpu_thermal, circadian) at 1Hz on
        # a daemon thread; AudioPerception / VisualPerception are populated
        # by ``_notify_expressive_create`` at every record_expressive site.
        try:
            from titan_hcl.logic.inner_perception import InnerPerceptionState
            from titan_hcl.utils import system_sensor as _sys_sensor

            def _ambient_sampler():
                # Both producers already exist on system_sensor (utils/system_sensor.py).
                # cpu_thermal ∈ [0,1], circadian_phase ∈ [0,1]; sum ∈ [0,2].
                return (
                    _sys_sensor.get_cpu_thermal(),
                    _sys_sensor.get_circadian_phase(),
                )

            self._inner_perception = InnerPerceptionState(_ambient_sampler)
            self._inner_perception.start()
            # Register the obs_db record_expressive hook ONCE — every art /
            # audio / music / text emission flows through record_expressive,
            # so this single registration covers all callers (helpers,
            # meditation, future writers). Best-effort; hook errors do NOT
            # propagate into archival.
            try:
                self._observatory_db._on_expressive_create_hook = (
                    self._inner_perception.notify_create)
            except Exception:
                pass
            logger.info("[TitanHCL] InnerPerceptionState started "
                        "(ambient=1Hz; audio/visual via obs_db hook)")
        except Exception as _ip_err:
            logger.warning("[TitanHCL] InnerPerceptionState start failed: %s",
                           _ip_err, exc_info=True)
            self._inner_perception = None

        # rFP_trinity_130d_awakening Phase 2 — outer-spirit history aggregator.
        # ExperientialMemory access via cognitive_worker's coordinator (lazy
        # lookup so we don't fail boot if cognitive_worker isn't up yet).
        # Phase B.5: spirit_proxy retired — the coordinator lives in
        # cognitive_worker post-D8-3; we look it up directly on plugin
        # state. ExperientialMemory was previously surfaced via
        # spirit_proxy._coordinator attribute (legacy V3-inline path);
        # under Phase C the coordinator instance lives in cognitive_worker
        # and is only exposed via its SHM publishers — outer-spirit
        # history aggregator now reads from in-proc coordinator handle if
        # available else None (None disables aggregation gracefully).
        try:
            from titan_hcl.logic.outer_spirit_history import OuterSpiritHistory

            def _e_mem_lookup():
                # Phase C: coordinator instance owned by cognitive_worker
                # subprocess; not accessible from parent. Surface inline
                # only if a parent-side coordinator handle is wired
                # (legacy Phase A+B path).
                coord = getattr(self, "_coordinator", None)
                if coord is None:
                    return None
                return getattr(coord, "_experiential_memory", None) or getattr(coord, "e_mem", None)

            self._outer_spirit_history = OuterSpiritHistory(_e_mem_lookup)
            logger.info("[TitanHCL] OuterSpiritHistory started "
                        "(env_adapt + graceful_rest + circadian + dream_recall)")
        except Exception as _osh_err:
            logger.warning("[TitanHCL] OuterSpiritHistory start failed: %s",
                           _osh_err, exc_info=True)
            self._outer_spirit_history = None

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
                "[TitanHCL] API subprocess mode active — "
                "legacy _start_observatory skipped (Microkernel v2 §A.4)")

        # Agency Module (autonomous action pipeline)
        # 2026-05-19 BOOT_TRACE — confirms boot reaches Phase 5 agency block
        logger.warning("[TitanHCL] [BOOT_TRACE] calling _boot_agency() now")
        self._boot_agency()
        logger.warning(
            "[TitanHCL] [BOOT_TRACE] _boot_agency() returned, "
            "self._agency=%s self._agency_queue=%s",
            type(getattr(self, "_agency", None)).__name__,
            type(getattr(self, "_agency_queue", None)).__name__)

        # Reflex Collector (Sovereign Tool System)
        self._boot_reflex_collector()

        # Trinity snapshot history → observatory_worker (RFP Phase B, 2026-05-21).

        # Agency bus listener (IMPULSE → INTENT → helper execution → ACTION_RESULT)
        if self._agency:
            # 2026-05-19 BOOT_TRACE — confirms create_task is reached
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] scheduling _agency_loop task — "
                "self._agency=%s", type(self._agency).__name__)
            asyncio.get_event_loop().create_task(self._agency_loop())
        else:
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] _agency_loop NOT scheduled — "
                "self._agency is None (Agency boot failed or disabled)")

        # RL stats drain — drains RLProxy's rl_proxy_stats subscription so
        # SAGE_STATS payloads reach the proxy cache and the queue can't
        # saturate under dst="all" broadcast volume. Mirrors AGENCY_STATS
        # path; runs unconditionally because RLProxy is unconditional.
        asyncio.get_event_loop().create_task(self._rl_stats_loop())

        # Chat bus bridge RETIRED in Phase C v1.17.0 (D-SPEC-72) — replaced
        # by agno_worker subprocess + agno_proxy.chat() per SPEC §9.B. The
        # api_subprocess now publishes CHAT_REQUEST with dst="agno_worker"
        # (not dst="chat_handler"), so this parent-side bridge was dead code.

        # Guardian bus bridge — parent-side handler for QUERY dst="guardian"
        # admin operations from api_subprocess. Same Phase C "events/commands
        # over bus" pattern as _chat_handler_loop. Closes BOTH
        # BUG-HOT-RELOAD-CODE-LOADING AND BUG-GUARDIAN-CONTROL-COMMANDS-ORPHAN
        # in one architectural fix: api_subprocess can now invoke
        # guardian.restart_module() (with optional `start_method="spawn"`
        # override for true code reload) over the bus, AND the historically
        # orphan GUARDIAN_{START,STOP,RESTART}_REQUEST publishers from
        # command_sender.py finally have a receiver. SPEC §[KERNEL_RPC] +
        # Phase C G19 — work-RPC pattern with rid-routed RESPONSE.
        asyncio.get_event_loop().create_task(self._guardian_handler_loop())

        # Sovereignty listener RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
        # SOVEREIGNTY_EPOCH is now consumed by sovereignty_worker subprocess
        # directly (registered via guardian.register(ModuleSpec(name="sovereignty",
        # ...)) above). spirit_worker.py:3845 already emits with dst="sovereignty",
        # so no producer change needed.

        # Meditation cycle (memory consolidation, mempool scoring, Cognee cognify)
        # REMOVED — extracted to meditation_worker subprocess per D-SPEC-57.

        # Phase C dissolution (2026-05-22): the OUTER_SOURCES_SNAPSHOT broadcast
        # (_publish_outer_sources_loop) is RETIRED — it carried STATE over the
        # bus (G18 violation) and its only live consumers (the Python outer_*
        # workers) are unspawned under l0_rust_enabled=true. Outer source data
        # now flows SHM-direct: each sidecar assembles via the §9.F
        # outer_source_assembly helper + re-homed breath trackers (below).

        # Phase C C-S6: outer sensor refresh sidecars (SPEC §9.D).
        # These are in-process asyncio tasks that snapshot the canonical
        # `sources` dict (per `_gather_outer_trinity_sources`) into msgpack-
        # encoded sensor_cache_outer_*.bin shm slots that titan-outer-
        # {body,mind,spirit}-rs Rust daemons read. Sidecars run
        # UNCONDITIONALLY (not flag-gated per PLAN §1.1 item 9 + SPEC §9.D
        # + §11.B line 1236) — writing to a slot Rust daemons don't yet
        # read is zero-cost and gives a soak window before C-S7 first
        # flag-flip. Each sidecar has its own in-process restart loop
        # (in-process exception handler per SPEC §11.B line 1236).
        try:
            import threading
            import traceback as _tb_mod
            from titan_hcl.logic import outer_body_sensor_refresh as _obsr
            from titan_hcl.logic import outer_mind_sensor_refresh as _omsr
            from titan_hcl.logic import outer_spirit_sensor_refresh as _ossr
            from titan_hcl.logic.outer_source_assembly import (
                OuterSourceContext, OuterHeavyStatsRefresher)
            from titan_hcl.logic.outer_sidecar_providers import (
                make_outer_body_provider, make_outer_mind_provider,
                make_outer_spirit_provider)
            from titan_hcl.core.state_registry import resolve_titan_id as _rtid
            # Phase C dissolution (2026-05-22): each sidecar assembles its source
            # data SHM-direct via the §9.F helper + re-homed breath trackers,
            # replacing the parent _gather_outer_sources + the OUTER_SOURCES_SNAPSHOT
            # bus broadcast (G18). One in-process heavy-stats refresher (G20) feeds
            # the DB-count keys + OSH's heavy dream_recall off the sidecar hot path.
            _osa_tid = _rtid()
            _osa_data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data")
            self._outer_heavy_refresher = OuterHeavyStatsRefresher(
                titan_id=_osa_tid, data_dir=_osa_data_dir,
                is_x_gateway=(str(_osa_tid).upper() == "T1"),
                outer_spirit_history=self._outer_spirit_history)
            self._outer_heavy_refresher.start()
            # inner_perception_state.bin publisher (C.7) — InnerPerception is
            # parent-resident hardware; publish it SHM-direct so mind_worker
            # reads it without the retired OUTER_SOURCES_SNAPSHOT broadcast (G18,
            # G21 single-writer = this parent thread).
            try:
                from titan_hcl.logic.inner_perception_state_publisher import (
                    InnerPerceptionStatePublisher)
                self._inner_perception_state_pub = InnerPerceptionStatePublisher(
                    titan_id=_osa_tid)

                def _inner_perception_publish_loop() -> None:
                    time.sleep(15)
                    while True:
                        try:
                            ip = getattr(self, "_inner_perception", None)
                            self._inner_perception_state_pub.publish(ip)
                        except Exception as _ipe:
                            logger.debug(
                                "[InnerPerceptionStatePub] loop: %s", _ipe)
                        time.sleep(1.0)
                threading.Thread(
                    target=_inner_perception_publish_loop,
                    name="inner_perception_state_publisher",
                    daemon=True).start()
            except Exception as _ip_pub_err:
                logger.error(
                    "[TitanHCL] inner_perception_state publisher boot failed "
                    "— inner_mind feeling[5/7/9] fall back to defaults: %s",
                    _ip_pub_err, exc_info=True)
            # expression_state.bin publisher (relocated from the deleted
            # _ensure_heavy_stats_refresher, C.8). Under l0_rust_enabled=true
            # expression_worker owns the slot (G21) → parent loop SKIPPED;
            # the flag-OFF legacy path keeps the parent publisher.
            _l0_rust_on = bool(self._full_config.get("microkernel", {}).get(
                "l0_rust_enabled", False))
            if _l0_rust_on:
                logger.info(
                    "[TitanHCL] expression_state.bin publisher SKIPPED under "
                    "l0_rust_enabled=true — owned by expression_worker (G21)")
                self._expression_state_pub = None
            else:
                try:
                    from titan_hcl.logic.expression_state_publisher import (
                        ExpressionStatePublisher)
                    self._expression_state_pub = ExpressionStatePublisher(
                        titan_id=_osa_tid)
                except Exception as _exp_pub_err:
                    logger.error(
                        "[TitanHCL] ExpressionStatePublisher BOOT FAILED — "
                        "inner_spirit expression dims fall back to defaults: %s",
                        _exp_pub_err, exc_info=True)
                    self._expression_state_pub = None

                def _expression_state_publish_loop() -> None:
                    time.sleep(15)
                    while True:
                        try:
                            pub = getattr(self, "_expression_state_pub", None)
                            if pub is not None:
                                pub.publish(
                                    getattr(self, "_expression_translator", None),
                                    getattr(self, "_expression_manager", None))
                        except Exception as _e:
                            logger.debug("[ExpressionStatePub] loop: %s", _e)
                        time.sleep(1.0)
                threading.Thread(
                    target=_expression_state_publish_loop,
                    name="expression_state_publisher", daemon=True).start()
                logger.info(
                    "[TitanHCL] expression_state.bin publisher started "
                    "(1Hz; l0_rust=false legacy path)")
            _osa_ctx = OuterSourceContext(
                shm_bank=self._shm_reader_bank, titan_id=_osa_tid,
                data_dir=_osa_data_dir, start_time=self._start_time,
                bus_stats_provider=lambda: self.bus.stats,
                observatory_db=self._observatory_db,
                heavy_stats=self._outer_heavy_refresher.cache,
                outer_spirit_history=self._outer_spirit_history)
            self._outer_body_sensor_sidecar = _obsr.OuterBodySensorRefresh(
                sources_provider=make_outer_body_provider(
                    _osa_ctx, _obsr.SOURCE_KEYS))
            self._outer_mind_sensor_sidecar = _omsr.OuterMindSensorRefresh(
                sources_provider=make_outer_mind_provider(
                    _osa_ctx, _omsr.SOURCE_KEYS))
            self._outer_spirit_sensor_sidecar = _ossr.OuterSpiritSensorRefresh(
                sources_provider=make_outer_spirit_provider(
                    _osa_ctx, _ossr.SOURCE_KEYS))

            # rFP_phase_c_close_all_runtime_gaps chunk 9H: each sidecar
            # runs in its own daemon thread with its own asyncio loop.
            # The previous architecture (3 × `asyncio.create_task` on the
            # shared main loop) deterministically reproduced a scheduling
            # bug under titan-kernel-rs where ONLY outer_body's run()
            # entered (verified live on T3 2026-05-06: body's "starting"
            # log fired 57s after task creation; mind+spirit "starting"
            # logs never appeared, no traceback). Per-thread isolation
            # sidesteps any main-loop scheduling drama. Threads are
            # `daemon=True` so they exit cleanly with the parent.
            # `_gather_outer_sources` is synchronous + reads in-process
            # state already designed for cross-thread access (proxy
            # registry uses parking_lot/threading.Lock locks).
            def _run_sidecar_thread(_sidecar, _name):
                try:
                    asyncio.run(_sidecar.run())
                except Exception:
                    logger.critical(
                        "[TitanHCL] sidecar thread %s crashed:\n%s",
                        _name, _tb_mod.format_exc(),
                    )
            for _sidecar, _name in (
                (self._outer_body_sensor_sidecar, "body"),
                (self._outer_mind_sensor_sidecar, "mind"),
                (self._outer_spirit_sensor_sidecar, "spirit"),
            ):
                _t = threading.Thread(
                    target=_run_sidecar_thread,
                    args=(_sidecar, _name),
                    name=f"outer_{_name}_sensor_refresh",
                    daemon=True,
                )
                _t.start()
            logger.info(
                "[TitanHCL] outer sensor refresh sidecars started in "
                "dedicated threads (body=%.1fs / mind=%.1fs / spirit=%.1fs cadence)",
                self._outer_body_sensor_sidecar._refresh_period_s,
                self._outer_mind_sensor_sidecar._refresh_period_s,
                self._outer_spirit_sensor_sidecar._refresh_period_s,
            )
        except Exception as e:
            logger.warning(
                "[TitanHCL] outer sensor refresh sidecar boot failed "
                "(non-fatal — Rust outer daemons read last-known): %s", e)

        # V4 event bridge → observatory_worker (RFP Phase A, 2026-05-21).

        # Social engagement: owned by social_worker (per-Titan canonical-poller
        # mention/DM polling + dispatch). The parent's _social_engagement_loop
        # was removed 2026-05-21 (D-SPEC-106) — it was never scheduled.

        self._background_tasks_started = True
        boot_s = time.time() - boot_start
        logger.info(
            "[TitanHCL] Async boot complete in %.2fs | Modules registered: %s",
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
