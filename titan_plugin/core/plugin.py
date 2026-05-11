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
    OUTER_SOURCES_SNAPSHOT,
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

        # Phase A.S8 — 3 symmetric outer trinity subprocess workers.
        # Each worker owns its own Schumann-clocked SHM write + rate-gated
        # bus publish. Publishes OUTER_BODY/MIND/SPIRIT_STATE (3 separate
        # messages, mirror of inner BODY/MIND/SPIRIT_STATE pattern).
        #
        # Phase C C-S7 Gap 7+11: under l0_rust_enabled=true, the Rust
        # outer-{body,mind,spirit}-rs daemons own these slots. Don't register
        # the Python A.S8 workers in that mode — Rust supervisor manages them.
        # Per PLAN_microkernel_phase_c_s7_activation_prep.md §2 Gap 7
        # (Option b — ModuleSpec gate).
        _l0_rust = self._full_config.get("microkernel", {}).get(
            "l0_rust_enabled", False)
        if not _l0_rust:
            from titan_plugin.modules.outer_body_worker import outer_body_worker_main
            from titan_plugin.modules.outer_mind_worker import outer_mind_worker_main
            from titan_plugin.modules.outer_spirit_worker import outer_spirit_worker_main

            # rFP_worker_broadcast_topics_completion §4.A Batch 1 (2026-05-10):
            # outer_{body,mind,spirit} drains at modules/outer_*_worker.py
            # consume exactly two broadcast types (OUTER_SOURCES_SNAPSHOT +
            # FILTER_DOWN) plus targeted SHUTDOWN + QUERY (which bypass the
            # broadcast filter regardless). Producer-side filter at
            # bus_socket.publish() drops everything else before enqueue.
            _OUTER_TRINITY_BROADCAST_TOPICS = [
                bus.OUTER_SOURCES_SNAPSHOT,
                bus.FILTER_DOWN,
            ]
            self.guardian.register(ModuleSpec(
                name="outer_body",
                layer="L1",
                entry_fn=outer_body_worker_main,
                config=self._full_config,
                rss_limit_mb=200,
                autostart=True,
                lazy=False,
                heartbeat_timeout=60.0,
                reply_only=False,
                broadcast_topics=_OUTER_TRINITY_BROADCAST_TOPICS,
                start_method="spawn" if _spawn_grad else "fork",
                b2_1_swap_critical=False,
            ))
            self.guardian.register(ModuleSpec(
                name="outer_mind",
                layer="L1",
                entry_fn=outer_mind_worker_main,
                config=self._full_config,
                rss_limit_mb=200,
                autostart=True,
                lazy=False,
                heartbeat_timeout=60.0,
                reply_only=False,
                broadcast_topics=_OUTER_TRINITY_BROADCAST_TOPICS,
                start_method="spawn" if _spawn_grad else "fork",
                b2_1_swap_critical=False,
            ))
            self.guardian.register(ModuleSpec(
                name="outer_spirit",
                layer="L1",
                entry_fn=outer_spirit_worker_main,
                config=self._full_config,
                rss_limit_mb=200,
                autostart=True,
                lazy=False,
                heartbeat_timeout=60.0,
                reply_only=False,
                broadcast_topics=_OUTER_TRINITY_BROADCAST_TOPICS,
                start_method="spawn" if _spawn_grad else "fork",
                b2_1_swap_critical=False,
            ))
        else:
            logger.info(
                "[TitanPlugin] outer_{body,mind,spirit} workers skipped — "
                "Rust outer-{body,mind,spirit}-rs daemons own these slots "
                "(microkernel.l0_rust_enabled=true)")

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
        from titan_plugin.modules.warning_monitor_worker import (
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
            name="rl",
            layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (IQL chain learning)
            entry_fn=rl_worker_main,
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
        # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
        # mind drain at modules/mind_worker.py:271-360 consumes 7 broadcasts
        # (OUTER_SOURCES_SNAPSHOT, FILTER_DOWN, FOCUS_NUDGE,
        # CONVERSATION_STIMULUS, INTERFACE_INPUT, SENSE_VISUAL, SENSE_AUDIO);
        # MODULE_SHUTDOWN + QUERY are targeted.
        self.guardian.register(ModuleSpec(
            name="mind",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT cognitive)
            entry_fn=mind_worker_main,
            config=mind_config,
            rss_limit_mb=700,   # was 500; fork-inherited parent RSS ~400MB left only 100MB headroom — caused T3 cascade (2026-04-17). Memory profiling tool (DEFERRED TOP) will identify real optimization targets.
            autostart=True,  # Mind senses should always be active
            lazy=False,
            broadcast_topics=[
                bus.OUTER_SOURCES_SNAPSHOT, bus.FILTER_DOWN, bus.FOCUS_NUDGE,
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
        # rFP_worker_broadcast_topics_completion §4.A.5 (Batch 5):
        # See legacy_core.py for full 57-type list; mirror exactly here.
        # spirit_worker.py is queued for retirement in
        # rFP_microkernel_v2_definitive_closure §4.D8-3.
        _SPIRIT_BROADCAST_TOPICS = [
            bus.ACTION_RESULT, bus.BODY_STATE,
            bus.CGN_CROSS_INSIGHT, bus.CGN_HAOV_VERIFY_REQ,
            bus.CGN_KNOWLEDGE_REQ, bus.CGN_KNOWLEDGE_RESP,
            bus.CGN_STATE_SNAPSHOT, bus.CONFIG_RELOAD,
            bus.CONTRACT_LIST_RESP, bus.CONVERSATION_STIMULUS,
            bus.DREAM_WAKE_REQUEST, bus.EMOT_CGN_SIGNAL,
            bus.EPOCH_TICK, bus.EXPERIENCE_STIMULUS,
            bus.FILTER_DOWN_V5, bus.LANGUAGE_STATS_UPDATE,
            bus.LLM_TEACHER_RESPONSE, bus.MAKER_DIALOGUE_COMPLETE,
            bus.MAKER_NARRATION_RESULT, bus.MAKER_PROPOSAL_CREATED,
            bus.MAKER_RESPONSE_RECEIVED, bus.MEDITATION_COMPLETE,
            bus.MEMORY_RECALL_PERTURBATION, bus.META_CGN_SIGNAL,
            bus.META_DIVERSITY_PRESSURE, bus.META_EUREKA,
            bus.META_EVENT_REWARD, bus.META_LANGUAGE_REQUEST,
            bus.META_LANGUAGE_REWARD, bus.META_OUTER_REWARD,
            bus.META_PATTERN_EMERGED, bus.META_PERSONA_REWARD,
            bus.META_REASON_OUTCOME, bus.META_REASON_REQUEST,
            bus.META_REASON_RESPONSE, bus.META_STRATEGY_DRIFT,
            bus.META_TEACHER_FEEDBACK, bus.META_TEACHER_GROUNDING,
            bus.MIND_STATE, bus.MODULE_CRASHED,
            bus.OUTER_BODY_STATE, bus.OUTER_MIND_STATE,
            bus.OUTER_OBSERVATION, bus.OUTER_SPIRIT_STATE,
            bus.OUTER_TRINITY_STATE, bus.RATE_LIMIT,
            bus.REFLEX_REWARD, bus.RELOAD,
            bus.SAVE_NOW, bus.SENSE_AUDIO,
            bus.SOCIAL_PERCEPTION, bus.SPEAK_RESULT,
            bus.STATE_SNAPSHOT, bus.TEACHER_SIGNALS,
            bus.TIMECHAIN_QUERY_RESP, bus.X_FORCE_POST,
            bus.X_POST_DISPATCH,
        ]
        self.guardian.register(ModuleSpec(
            name="spirit",
            layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (consciousness core)
            entry_fn=spirit_worker_main,
            config=spirit_config,
            rss_limit_mb=1200,  # was 750; spirit loads 11 neural nets + consciousness.db + inner_memory.db + NS programs during boot — peak RSS 961-1486MB observed 2026-04-18
            autostart=True,  # Spirit awareness should always be active
            lazy=False,
            heartbeat_timeout=120.0,  # Spirit does heavy V4 work (LLM, on-chain)
            broadcast_topics=_SPIRIT_BROADCAST_TOPICS,
        ))

        # cognitive_worker (L2) — Phase C C-S8 4B (chunk 8E skeleton, 2026-05-05).
        # Hosts the L3 cognitive engines (Reasoning, MetaReasoning, Dreaming,
        # InnerTrinityCoordinator, PiHeartbeat, NeuralNervousSystem,
        # ObservableEngine, ExpressionManager). Active under l0_rust_enabled=true
        # ONLY — the legacy spirit_worker_main path owns these engines under
        # l0_rust_enabled=false per Maker D3 (b). Registration is gated on the
        # flag so guardian_HCL never spawns cognitive_worker in the legacy mode.
        # Boot order: registered after spirit so guardian autostart sequence
        # boots body → mind → spirit → cognitive_worker (cognitive_worker reads
        # the trinity tensors body/mind/spirit publish).
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
            from titan_plugin.modules.cognitive_worker import (
                cognitive_worker_main,
                _COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
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
        from titan_plugin.modules.ns_worker import ns_worker_main
        from titan_plugin.modules.neuromod_worker import neuromod_worker_main
        from titan_plugin.modules.hormonal_worker import hormonal_worker_main

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
                broadcast_topics=[bus.MODULE_SHUTDOWN],
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
                broadcast_topics=[bus.MODULE_SHUTDOWN],
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
                broadcast_topics=[bus.MODULE_SHUTDOWN],
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
            rss_limit_mb=800,     # gzip-9 over ~300MB full-tier personality + Irys subprocess
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
        from titan_plugin.logic.emot_kin_protocol import (
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
                bus.HORMONE_FIRED, bus.CGN_BETA_SNAPSHOT,
            ],
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
            # 2026-05-01: broadcast_topics filter — matches what api_subprocess
            # actually consumes per cache_key_registry.py producer_event entries
            # (16 *_UPDATED broadcasts that populate observatory CachedState).
            # Pre-fix: empty broadcast_topics → broker used _HIGH_RATE_BROADCAST_
            # TYPES stopgap which silently DROPPED PI_HEARTBEAT_UPDATED + TOPOLOGY_
            # STATE_UPDATED (both events api NEEDS!) → /v4/pi-heartbeat + /v4/topology
            # returned empty data on T2/T3 under microkernel v2 socket-broker mode.
            # Targeted dst="api" messages (STATE_SNAPSHOT_RESPONSE, OBSERVATORY_
            # EVENT, RESPONSE for chat bridge) bypass broadcast filter automatically
            # via name-based routing — not listed here.
            # Mirrors the `backup` ModuleSpec migration pattern (1 of 17 → 2 of 17).
            broadcast_topics=[
                bus.CHI_UPDATED,
                bus.DREAMING_STATE_UPDATED,
                bus.EXPRESSION_COMPOSITES_UPDATED,
                bus.LANGUAGE_STATS_UPDATED,
                bus.MEMORY_KNOWLEDGE_GRAPH_UPDATED,
                bus.MEMORY_MEMPOOL_UPDATED,
                bus.MEMORY_STATUS_UPDATED,
                bus.MEMORY_TOPOLOGY_UPDATED,
                bus.MEMORY_TOP_UPDATED,
                bus.META_REASONING_STATS_UPDATED,
                bus.MSL_STATE_UPDATED,
                bus.NEUROMOD_STATS_UPDATED,
                bus.PI_HEARTBEAT_UPDATED,
                bus.REASONING_STATS_UPDATED,
                bus.SOLANA_BALANCE_UPDATED,
                bus.TOPOLOGY_STATE_UPDATED,
            ],
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
                        # Inner perception fan-out happens via observatory_db
                        # hook (utils/observatory_db.record_expressive). One
                        # hook on the canonical write site covers all callers.
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

        # rFP_meditation_worker_latency Fix #C (2026-05-07): wait for memory
        # worker to be FULLY attached to the bus broker before firing the
        # first cycle. The legacy `await asyncio.sleep(120)` raced with the
        # worker's bus-attach which takes 122s+ on spawn-mode (TieredMemoryGraph
        # + FAISS + Kuzu cold init). Live evidence: T1 2026-05-07 17:29:20
        # plugin epoch #1 fires → 17:29:22 worker attaches (2s late) → message
        # silently dropped → 300s timeout. We poll guardian's heartbeat-age
        # for the memory module — once heartbeat <30s the worker is alive,
        # bus-attached, and processing messages. Cap at 300s (covers
        # cold-boot of Kuzu graph + FAISS index load on slow disks).
        #
        # Fix-C amendment (2026-05-07 T2 deploy): the memory module is
        # registered as `autostart=False, lazy=True` — it stays STOPPED
        # until something calls `_ensure_started()` to trigger the spawn.
        # Verified live on T2 18:36:50: `[Guardian] Registered module
        # 'memory' [L2] (autostart=False, lazy=True ...)`. Without
        # ensure_started, the probe waits the full 300s for a worker
        # that will never start on its own. So FIRST trigger the lazy
        # spawn, THEN wait for readiness.
        memory_proxy_pre = self._proxies.get("memory")
        if memory_proxy_pre is not None:
            try:
                memory_proxy_pre._ensure_started()
                logger.info(
                    "[TitanPlugin] Meditation: memory_proxy._ensure_started() "
                    "called — lazy spawn triggered, now waiting for worker "
                    "to become ready")
            except Exception as _ens_err:
                logger.warning(
                    "[TitanPlugin] Meditation: _ensure_started() raised: %s — "
                    "proceeding to readiness probe regardless", _ens_err)
        guardian = getattr(self.kernel, "guardian", None) or getattr(self, "_guardian", None)
        if guardian is not None:
            _wait_deadline = time.time() + 300.0
            _last_log = 0.0
            while time.time() < _wait_deadline:
                try:
                    info = guardian._modules.get("memory")
                    if info is not None:
                        # Guardian.ModuleInfo.state is a ModuleState enum;
                        # its `.value` is the lowercase string ("running"
                        # / "starting" / "stopped" / etc).
                        state_val = getattr(getattr(info, "state", None), "value", None)
                        hb_age = time.time() - getattr(info, "last_heartbeat", 0.0)
                        if state_val == "running" and hb_age < 30.0:
                            logger.info(
                                "[TitanPlugin] Meditation: memory worker ready "
                                "(state=running, hb_age=%.1fs) after %.1fs wait",
                                hb_age, time.time() - (_wait_deadline - 300.0))
                            break
                    if time.time() - _last_log > 30.0:
                        state_repr = getattr(
                            getattr(info, "state", None), "value", "absent",
                        ) if info else "absent"
                        logger.info(
                            "[TitanPlugin] Meditation: waiting for memory worker "
                            "(state=%s, %.0fs elapsed)", state_repr,
                            time.time() - (_wait_deadline - 300.0))
                        _last_log = time.time()
                except Exception as _wait_err:
                    logger.debug(
                        "[TitanPlugin] Meditation readiness probe error: %s",
                        _wait_err)
                await asyncio.sleep(2.0)
            else:
                logger.warning(
                    "[TitanPlugin] Meditation: memory worker did not become "
                    "ready within 300s — proceeding anyway (cycles will likely "
                    "time out until worker recovers)")
        else:
            # Defensive fallback: no guardian reference available — fall back
            # to the legacy fixed sleep. Should never happen in production.
            logger.warning(
                "[TitanPlugin] Meditation: no guardian reference for readiness "
                "probe — falling back to legacy 120s fixed sleep")
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

                # rFP_meditation_worker_latency Fix #A (2026-05-07): use
                # `run_meditation_async` to avoid the 156s event-loop deadlock
                # that gated 8-day expressive_archive + 26-day Arweave outage.
                result = await memory_proxy.run_meditation_async()

                promoted = result.get("promoted", 0)
                pruned = result.get("pruned", 0)
                if result.get("success"):
                    logger.info(
                        "[TitanPlugin] Meditation epoch #%d complete: promoted=%d pruned=%d",
                        epoch_count, promoted, pruned,
                    )
                # rFP_meditation_worker_latency Fix #B2 (2026-05-07): art
                # generation now keys on `promoted > 0`, NOT `result["success"]`.
                # The worker handler may have completed all migration work
                # successfully but the bus.request return arrived late
                # (success=False on timeout) — gating art on `success` caused
                # the 8-day expressive_archive cutoff. Memory ops are persisted
                # by the worker regardless of bus.request timing; art belongs
                # alongside the persisted promotions.
                if promoted > 0 or result.get("success"):
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
                                    # Inner perception fan-out: see observatory_db
                                    # record_expressive hook.
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
    # Phase A.S8: Outer Sources publisher (plugin-only sources → 3 workers)
    # ------------------------------------------------------------------

    async def _publish_outer_sources_loop(self) -> None:
        """Every 10s, gather plugin-only outer sources and publish OUTER_SOURCES_SNAPSHOT.

        The 3 outer trinity workers (outer_body/mind/spirit) self-fetch all
        sources accessible without a plugin reference (file reads, utility modules,
        httpx). Plugin-only sources that require live proxy or module references
        are gathered here and broadcast via OUTER_SOURCES_SNAPSHOT so workers
        can cache them at tick time.
        """
        await asyncio.sleep(15)  # Let workers boot before first snapshot
        logger.info("[TitanPlugin] Outer sources loop started (10s interval)")
        _osl_n = 0
        while True:
            try:
                snap = await asyncio.to_thread(self._gather_outer_sources)
                _osl_n += 1
                if _osl_n in (1, 5, 25, 100):
                    logger.info(
                        "[TitanPlugin] OUTER_SOURCES_SNAPSHOT publish #%d "
                        "keys=%d sample=%s",
                        _osl_n, len(snap),
                        sorted(list(snap.keys())[:8]))
                self.bus.publish(make_msg(
                    OUTER_SOURCES_SNAPSHOT, "core", "all", snap,
                ))
            except Exception as _e:
                logger.warning("[TitanPlugin] outer_sources_loop error: %s",
                                _e, exc_info=True)
            await asyncio.sleep(10)

    # rFP_trinity_130d_awakening §12.4 — bus events whose latest payload we
    # cache for the 130D dim consumers. See SPEC §23.1 for which producers
    # publish each event. Lazy-subscribed on first _gather_outer_sources call.
    _STATS_CACHE_EVENT_TYPES = (
        "META_REASONING_STATS_UPDATED",  # contains meta_cgn block
        "LANGUAGE_STATS_UPDATED",        # vocab_total / composition_level / ...
        "OUTPUT_VERIFIER_STATS",         # verified_count / rejected_count
        "MEMORY_STATUS_UPDATED",         # persistent_count / mempool_size
        "MEMORY_KNOWLEDGE_GRAPH_UPDATED",  # KG node_count / edge_count
        "SOLANA_BALANCE_UPDATED",        # balance updates
        "SOCIAL_STATS_UPDATED",          # persona social stats
        "NEUROMOD_STATS_UPDATED",        # full neuromod state (incl hormones)
        "CGN_STATS_UPDATED",             # cgn worker periodic publish
        "SOCIAL_PERCEPTION_STATS_UPDATED",  # spirit worker periodic publish (G19, 2026-05-07)
    )

    def _ensure_stats_cache_subscription(self) -> None:
        """Lazy-init the stats cache subscriber. Idempotent."""
        if getattr(self, "_stats_cache_queue", None) is not None:
            return
        try:
            self._stats_cache: dict = {}
            self._stats_cache_queue = self.bus.subscribe(
                "trinity_dim_stats_cache",
                types=list(self._STATS_CACHE_EVENT_TYPES),
            )
            logger.info(
                "[TitanPlugin] Trinity 130D stats cache subscriber initialized "
                "(types=%d)", len(self._STATS_CACHE_EVENT_TYPES))
        except Exception as _e:
            logger.warning(
                "[TitanPlugin] stats cache subscribe failed: %s "
                "— rich producers will be unavailable to dim formulas", _e)
            self._stats_cache_queue = None

    def _drain_stats_cache(self) -> None:
        """Drain pending stats events and overwrite cache with latest payload per type."""
        if getattr(self, "_stats_cache_queue", None) is None:
            return
        try:
            msgs = self.bus.drain(self._stats_cache_queue, max_msgs=200)
            for m in msgs:
                t = m.get("type")
                p = m.get("payload")
                if t and isinstance(p, dict):
                    self._stats_cache[t] = p
        except Exception as _e:
            swallow_warn('[OuterSources] stats cache drain', _e,
                         key="core.plugin.stats_cache_drain", throttle=100)

    def _gather_outer_sources(self) -> dict:
        """Gather plugin-only outer sources for OUTER_SOURCES_SNAPSHOT.

        Phase 1 wiring (rFP_trinity_130d_awakening §12.4 + SPEC §23):
        every rich producer required by the 130D dim formulas is exposed
        here. Workers receive this dict via OUTER_SOURCES_SNAPSHOT and
        feed it into their tensor formulas at tick time.

        Producer sources (per SPEC §23.1):
          - In-process: agency, assessment, expression_translator, soul,
                        observatory_db, bus, helper_registry
          - Proxy / bus query: spirit_proxy (trinity, social_perception),
                              memory_proxy (memory_status)
          - Bus event cache (see _STATS_CACHE_EVENT_TYPES): meta_cgn,
                              language, output_verifier, kg, neuromod,
                              social, sol_balance, cgn
          - Utility modules: system_sensor, network_monitor, timechain_v2
          - File reads (mtime-cached): anchor_state.json,
                              jailbreak_alerts.json, genesis_record.json,
                              data/arweave_devnet/, data/meditation_memos/
        """
        # Drain bus events into cache before reading
        self._ensure_stats_cache_subscription()
        self._ensure_heavy_stats_refresher()
        self._drain_stats_cache()
        cache = getattr(self, "_stats_cache", {})

        sources: dict = {
            "uptime_seconds": time.time() - self._start_time,
        }

        # ── In-process producers (always available) ─────────────────
        if self._agency:
            sources["agency_stats"] = self._agency.get_stats()
        if self._agency_assessment:
            sources["assessment_stats"] = self._agency_assessment.get_stats()
        if self._agency and hasattr(self._agency, '_registry'):
            sources["helper_statuses"] = self._agency._registry.get_all_statuses()

        sources["bus_stats"] = self.bus.stats

        # Expression translator — sovereignty + learned action stats (in-process).
        try:
            if self._expression_translator is not None:
                sources["expression_translator_stats"] = self._expression_translator.get_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] expression_translator', _e,
                         key="core.plugin.outer_sources.expr", throttle=100)

        # ── Spirit proxy: hormones + impulse + social_perception ────
        try:
            spirit_proxy = self._proxies.get("spirit")
            if spirit_proxy:
                trinity_data = spirit_proxy.get_trinity()
                if isinstance(trinity_data, dict):
                    if "impulse_engine" in trinity_data:
                        sources["impulse_stats"] = trinity_data["impulse_engine"]
                    if "hormone_levels" in trinity_data:
                        sources["hormone_levels"] = trinity_data["hormone_levels"]
                    if "hormone_fires" in trinity_data:
                        sources["hormone_fires"] = trinity_data["hormone_fires"]
        except Exception as _e:
            swallow_warn('[OuterSources] spirit_proxy hormones', _e,
                         key="core.plugin.outer_sources.hormones", throttle=100)

        # social_perception_stats via SHM-direct read of
        # social_perception_state.bin (Session 3 producer in spirit_worker).
        # Phase C Session 4 (rFP §4.C.7) — full G18 compliance: state
        # transport is SHM, never bus (was bus-cache-mediated G19
        # hardening; now SHM-direct).
        try:
            from titan_plugin.core.state_registry import (
                StateRegistryReader, ensure_shm_root, resolve_titan_id)
            from titan_plugin.logic.session3_state_specs import (
                SOCIAL_PERCEPTION_STATE_SPEC)
            import msgpack
            if not hasattr(self, "_r_social_perception"):
                self._r_social_perception = StateRegistryReader(
                    SOCIAL_PERCEPTION_STATE_SPEC,
                    ensure_shm_root(resolve_titan_id()))
            _sp_raw = self._r_social_perception.read_variable()
            if _sp_raw:
                _sp_payload = msgpack.unpackb(_sp_raw, raw=False)
                if isinstance(_sp_payload, dict) and _sp_payload:
                    sources["social_perception_stats"] = _sp_payload
        except Exception as _e:
            swallow_warn(
                '[OuterSources] social_perception SHM read', _e,
                key="core.plugin.outer_sources.social_perception",
                throttle=100)

        # ── Memory proxy: status + KG (KG cached via bus event) ─────
        try:
            memory_proxy = self._proxies.get("memory")
            if memory_proxy:
                sources["memory_status"] = memory_proxy.get_memory_status()
                # Growth metrics (Trinity 130D consumers: knowledge_growth,
                # research_effectiveness, knowledge_retrieval).
                try:
                    sources["memory_growth_metrics"] = memory_proxy.get_growth_metrics()
                except Exception:
                    pass
        except Exception:
            pass

        # ── Observatory expressive archive ──────────────────────────
        try:
            _obs = getattr(self, "_observatory_db", None)
            if _obs is not None:
                sources["art_count_100"] = len(
                    _obs.get_expressive_archive(type_="art", limit=100))
                sources["audio_count_100"] = len(
                    _obs.get_expressive_archive(type_="audio", limit=100))
                sources["art_count_500"] = len(
                    _obs.get_expressive_archive(type_="art", limit=500))
                sources["audio_count_500"] = len(
                    _obs.get_expressive_archive(type_="audio", limit=500))
                # text/sentence count for world_footprint (SPEC §23.3)
                try:
                    sources["text_count_500"] = len(
                        _obs.get_expressive_archive(type_="text", limit=500))
                except Exception:
                    sources["text_count_500"] = 0
        except Exception as _exc:
            swallow_warn('[OuterSources] expressive count pre-extract failed', _exc,
                         key="core.plugin.outer_sources_expressive", throttle=100)

        # ── Soul / Solana local-stats (SPEC §23.9 SAT[0,5,13]) ──────
        if self.soul:
            sources["soul_health"] = 0.9 if not self._limbo_mode else 0.2
        else:
            sources["soul_health"] = 0.2

        # solana_local_stats — derived from local files + soul state, no RPC.
        try:
            _data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            _genesis_path = os.path.join(_data_dir, "genesis_record.json")
            sources["solana_local_stats"] = {
                "identity_verified": (
                    1.0 if (self.soul and not self._limbo_mode) else 0.0),
                "genesis_nft_exists": (
                    1.0 if os.path.exists(_genesis_path) else 0.0),
            }
        except Exception:
            pass

        # ── File-based state (anchor + jailbreak alerts) ─────────────
        try:
            import json as _json
            _data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            _anchor_path = os.path.join(_data_dir, "anchor_state.json")
            if os.path.exists(_anchor_path):
                with open(_anchor_path) as _af:
                    sources["anchor_state"] = _json.load(_af)
                # Convenience top-level (used by outer_body[0] interoception)
                _sb = sources["anchor_state"].get("sol_balance")
                if isinstance(_sb, (int, float)):
                    sources["sol_balance"] = float(_sb)
                # rFP_trinity_130d_phase2_5_closure §4 (chunk 2.5.D) —
                # SPEC §23.9 SAT[10] recovery_speed reads
                # `recovery_stats.consecutive_failures`. Derive from
                # anchor_state's consecutive_failures (single source of
                # truth) so the dim is no longer ABSENT-classified on T1.
                _cf = sources["anchor_state"].get("consecutive_failures", 0)
                sources["recovery_stats"] = {
                    "consecutive_failures": int(_cf or 0),
                    "last_anchor_time": sources["anchor_state"].get(
                        "last_anchor_time", 0.0),
                    "anchor_count": int(
                        sources["anchor_state"].get("anchor_count", 0) or 0),
                }
        except Exception:
            pass

        # jailbreak_alerts: count + 24h windows for boundary_enforcement /
        # threat_discernment (SPEC §23.9 SAT[3], CHIT[17]).
        try:
            sources["jailbreak_alerts_stats"] = self._compute_jailbreak_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] jailbreak stats', _e,
                         key="core.plugin.outer_sources.jailbreak", throttle=100)

        # arweave + meditation_memos counts (world_footprint, SPEC §23.3)
        try:
            sources["world_footprint_extra_counts"] = self._count_artifact_dirs()
        except Exception:
            pass

        # ── Utility modules (system_sensor, network_monitor, timechain) ──
        try:
            from titan_plugin.utils import system_sensor as _sys_sensor
            sources["system_sensor_stats"] = _sys_sensor.get_all_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] system_sensor', _e,
                         key="core.plugin.outer_sources.sysensor", throttle=100)

        # ── InnerPerceptionState (rFP_trinity_130d_awakening Phase 2) ──
        # Feeds inner_mind[5] inner_hearing, [7] inner_sight, [9] inner_smell
        # via the audio_state / visual_state / ambient_change keys. The
        # last_create_ts also feeds outer_spirit ANANDA[41] creative_tension.
        try:
            ip = getattr(self, "_inner_perception", None)
            if ip is not None:
                sources["inner_perception_stats"] = ip.get_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] inner_perception', _e,
                         key="core.plugin.outer_sources.innerperc", throttle=100)

        # ── OuterSpiritHistory (rFP_trinity_130d_awakening Phase 2) ──
        # SAT[11] env_adapt + ANANDA[40] graceful_rest fed from assessment.recent
        # gated on system_sensor (cpu_thermal / cpu_spike_rate / circadian).
        # CHIT[26] circadian_alignment fed from agency.recent_actions_detail.
        # CHIT[25] dream_recall already populated by heavy-stats refresher.
        try:
            osh = getattr(self, "_outer_spirit_history", None)
            if osh is not None:
                _sys_stats = sources.get("system_sensor_stats") or {}
                _cpu_thermal = float(_sys_stats.get("cpu_thermal", 0.0) or 0.0)
                _cpu_spike = float(_sys_stats.get("cpu_spike_rate", 0.0) or 0.0)
                _circadian = float(_sys_stats.get("circadian_phase", 0.5) or 0.5)
                _assess_block = sources.get("assessment_stats") or {}
                _recent_assessments = _assess_block.get("recent") or []
                osh.ingest_assessments(
                    _recent_assessments, _cpu_thermal, _cpu_spike, _circadian)
                _agency_block = sources.get("agency_stats") or {}
                _recent_acts = _agency_block.get("recent_actions_detail") or []
                osh.ingest_action_timestamps(
                    a.get("ts") for a in _recent_acts if a.get("ts"))
                sources["outer_spirit_history_stats"] = osh.get_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] outer_spirit_history', _e,
                         key="core.plugin.outer_sources.osh", throttle=100)

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
        except Exception as _e:
            swallow_warn('[OuterSources] network_monitor', _e,
                         key="core.plugin.outer_sources.netmon", throttle=100)

        try:
            from titan_plugin.logic.timechain_v2 import (
                get_tx_latency_stats, get_block_delta_stats,
            )
            sources["tx_latency_stats"] = get_tx_latency_stats()
            sources["block_delta_stats"] = get_block_delta_stats()
        except Exception as _e:
            swallow_warn('[OuterSources] timechain_v2 stats', _e,
                         key="core.plugin.outer_sources.timechain", throttle=100)

        # llm_avg_latency: fetched from spirit_proxy (LLM module exposes via
        # NEUROMOD_STATS or as part of meta_reasoning telemetry). Plugin
        # holds 0.0 sentinel — outer_body[4] thermal no longer reads this
        # (replaced by hormonal_heat per SPEC §23.7). Preserved for any
        # legacy consumer.
        sources["llm_avg_latency"] = 0.0

        # ── Bus-event cached stats (rich producers) ──────────────────
        # MetaReasoning stats include the meta_cgn sub-block (per
        # spirit_loop publish at META_REASONING_STATS_UPDATED).
        meta_payload = cache.get("META_REASONING_STATS_UPDATED")
        if isinstance(meta_payload, dict):
            sources["meta_reasoning_stats"] = meta_payload
            if isinstance(meta_payload.get("meta_cgn"), dict):
                sources["meta_cgn_stats"] = meta_payload["meta_cgn"]

        # Language stats (vocab, composition_level, teacher_sessions, ...).
        # Bus event LANGUAGE_STATS_UPDATED fires only on teacher activity
        # (irregular cadence — minutes between bursts). For dim formulas
        # that need always-fresh language state we fall back to the
        # language_state.bin SHM slot (Phase C Session 4 publisher).
        lang_payload = cache.get("LANGUAGE_STATS_UPDATED")
        if isinstance(lang_payload, dict):
            sources["language_stats"] = lang_payload
        else:
            # rFP_trinity_130d_phase2_5_closure §4 Chunk 2.5.D — SHM
            # fallback so language_stats input is never ABSENT-classified.
            try:
                if not hasattr(self, "_r_language_state"):
                    from titan_plugin.core.state_registry import (
                        StateRegistryReader, ensure_shm_root, resolve_titan_id,
                    )
                    from titan_plugin.logic.language_state_publisher import (
                        LANGUAGE_STATE_SPEC,
                    )
                    self._r_language_state = StateRegistryReader(
                        LANGUAGE_STATE_SPEC,
                        ensure_shm_root(resolve_titan_id()),
                    )
                blob = self._r_language_state.read_variable()
                if blob:
                    import msgpack as _mp
                    payload = _mp.unpackb(blob, raw=False)
                    if isinstance(payload, dict) and payload:
                        sources["language_stats"] = payload
            except Exception:
                pass

        # Output verifier (verified_count / rejected_count).
        ov_payload = cache.get("OUTPUT_VERIFIER_STATS")
        if isinstance(ov_payload, dict):
            sources["output_verifier_stats"] = ov_payload

        # CGN stats (groundings, avg_reward, consolidations, ...).
        cgn_payload = cache.get("CGN_STATS_UPDATED")
        if isinstance(cgn_payload, dict):
            sources["cgn_stats"] = cgn_payload

        # Knowledge graph (Kuzu) node + edge counts.
        kg_payload = cache.get("MEMORY_KNOWLEDGE_GRAPH_UPDATED")
        if isinstance(kg_payload, dict):
            sources["knowledge_graph_stats"] = {
                "node_count": kg_payload.get("node_count", 0),
                "edge_count": kg_payload.get("edge_count", 0),
                "total_entities": kg_payload.get("total_entities", 0),
                "total_edges": kg_payload.get("total_edges", 0),
            }

        # Persona social stats (jailbreak_score / identity_score aggregates).
        soc_stats_payload = cache.get("SOCIAL_STATS_UPDATED")
        if isinstance(soc_stats_payload, dict):
            sources["persona_social_stats"] = soc_stats_payload

        # Heavy DB/file producer stats (inner_memory + social_x_gateway +
        # events_teacher) are loaded asynchronously via _refresh_heavy_stats
        # — see ensure_heavy_stats_refresher(). Reading from cache here is
        # O(1) and never blocks the 10s gather hot path.
        heavy = getattr(self, "_heavy_stats_cache", {})
        if heavy:
            for _k in ("inner_memory_stats", "social_x_gateway_stats",
                        "events_teacher_stats",
                        # Phase 2 (SPEC §23.9 ANANDA[36,38]).
                        "community_engagement_stats"):
                v = heavy.get(_k)
                if isinstance(v, dict):
                    sources[_k] = v

        return sources

    # ── Heavy stats async refresher (rFP §12 close-out fix 2026-05-06) ──
    # Reading inner_memory.db (262K rows) + social_x_gateway DB +
    # events_teacher DB inside _gather_outer_sources blocks the 10s
    # publish loop. The fix: run these in a separate background thread
    # at 60s cadence; gather reads from the cache.

    def _ensure_heavy_stats_refresher(self) -> None:
        """Lazy-start the heavy-stats refresh thread (idempotent)."""
        if getattr(self, "_heavy_stats_thread_started", False):
            return
        self._heavy_stats_thread_started = True
        self._heavy_stats_cache: dict = {}
        import threading

        def _refresh_loop() -> None:
            # First refresh deferred until plugin has booted producers.
            time.sleep(20)
            while True:
                try:
                    self._heavy_stats_cache["inner_memory_stats"] = (
                        self._read_inner_memory_stats())
                except Exception as _e:
                    logger.debug("[HeavyStats] inner_memory refresh: %s", _e)
                try:
                    self._heavy_stats_cache["social_x_gateway_stats"] = (
                        self._read_social_x_gateway_stats())
                except Exception as _e:
                    logger.debug("[HeavyStats] social_x refresh: %s", _e)
                try:
                    self._heavy_stats_cache["events_teacher_stats"] = (
                        self._read_events_teacher_stats())
                except Exception as _e:
                    logger.debug("[HeavyStats] events_teacher refresh: %s", _e)
                # rFP_trinity_130d_awakening Phase 2 — DreamRecallProducer
                # SQL COUNT against experiential_memory; G19/G20 says NEVER
                # inline in gather hot path. Refresh here.
                try:
                    osh = getattr(self, "_outer_spirit_history", None)
                    if osh is not None:
                        osh.refresh_dream_recall()
                except Exception as _e:
                    logger.debug("[HeavyStats] dream_recall refresh: %s", _e)
                # Phase 2 (SPEC §23.9 ANANDA[36,38]) — community_connection
                # + expression_reach producer. SQL COUNT(DISTINCT) +
                # AVG against mention_tracking + engagement_snapshots.
                # Reuses the persistent _social_x_gateway_reader instance
                # initialized lazily by _read_social_x_gateway_stats above.
                #
                # Fleet topology: T1 is the SOLE X gateway. T2/T3 post
                # via T1's SocialXGateway over kin RPC and do NOT maintain
                # their own local mention_tracking / engagement_snapshots
                # tables (the .db files exist but stay empty / schema-less).
                # ``is_x_gateway`` flag tells the producer to short-circuit
                # to a delegation marker on T2/T3 — honestly reflecting
                # that ANANDA[36, 38] are T1-canonical dims by design.
                try:
                    _tid = (self.kernel.titan_id
                            if hasattr(self, "kernel") and self.kernel
                            else (self._titan_id
                                  if hasattr(self, "_titan_id") else "T1"))
                    _tid_str = str(_tid).upper()
                    if _tid_str == "T1":
                        # T1 owns social_x.db + events_teacher.db locally.
                        sxg = getattr(self, "_social_x_gateway_reader", None)
                        if (sxg is not None
                                and hasattr(sxg, "get_community_engagement_stats")):
                            self._heavy_stats_cache["community_engagement_stats"] = (
                                sxg.get_community_engagement_stats(
                                    is_x_gateway=True, titan_id="T1"))
                    else:
                        # Phase 2.5.E (rFP_trinity_130d_phase2_5_closure §5) —
                        # T2/T3 reach T1 over HTTP for their own per-Titan
                        # author-attributed slice of mention_tracking +
                        # engagement_snapshots. Cached at 60s cadence
                        # (G19/G20 — no per-tick HTTP).
                        try:
                            import urllib.request as _ur
                            import json as _json
                            _t1_addr = "http://10.135.0.3:7777"
                            _url = (f"{_t1_addr}/v4/community-engagement-stats"
                                    f"?titan_id={_tid_str}")
                            _resp = _ur.urlopen(_url, timeout=8.0)
                            _body = _json.loads(_resp.read())
                            if _body.get("status") == "ok":
                                stats = _body.get("data") or {}
                                stats["gateway_role"] = "kin-rpc"
                                stats["titan_id"] = _tid_str
                                self._heavy_stats_cache[
                                    "community_engagement_stats"] = stats
                        except Exception as _http_e:
                            logger.debug(
                                "[HeavyStats] community_engagement HTTP-to-T1: %s",
                                _http_e)
                except Exception as _e:
                    logger.debug("[HeavyStats] community_engagement refresh: %s", _e)
                time.sleep(60)

        t = threading.Thread(target=_refresh_loop, name="heavy_stats_refresher",
                              daemon=True)
        t.start()
        logger.info("[TitanPlugin] Heavy stats refresher thread started (60s cadence)")

    # ── File / DB readers — mtime-cached helpers (rFP §12.4) ─────────

    def _compute_jailbreak_stats(self) -> dict:
        """Read data/jailbreak_alerts.json and produce 24h aggregates.

        SPEC §23.9 SAT[3] boundary_enforcement + CHIT[17] threat_discernment
        consumers read these fields directly. Cached by file mtime.
        """
        import json as _json
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                             "data", "jailbreak_alerts.json")
        if not os.path.exists(path):
            return {"threats_detected_24h": 0, "blocked_24h": 0,
                    "confirmed_threats_24h": 0, "total_alerts": 0}

        cache = getattr(self, "_jailbreak_cache", None)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        if cache and cache.get("_mtime") == mtime:
            return cache["stats"]

        with open(path) as f:
            alerts = _json.load(f)
        if not isinstance(alerts, list):
            alerts = []

        now = time.time()
        cutoff = now - 86400
        recent = [a for a in alerts if isinstance(a, dict)
                  and a.get("timestamp", 0) > cutoff]
        threats_24h = len(recent)
        blocked = sum(1 for a in recent if a.get("score", 0) >= 0.9)
        # confirmed = score < 1.0 AND adversary_type set (real attempted attack)
        confirmed = sum(1 for a in recent
                        if a.get("score", 1.0) < 1.0 and a.get("adversary_type"))
        # severity_avg over 24h (1.0 - score = severity)
        if recent:
            severity_avg = sum(max(0, 1.0 - float(a.get("score", 1.0)))
                                for a in recent) / len(recent)
        else:
            severity_avg = 0.0

        # All-time defended count for world_footprint
        defended_all_time = sum(1 for a in alerts if isinstance(a, dict)
                                 and a.get("score", 0) >= 0.9)

        stats = {
            "threats_detected_24h": threats_24h,
            "blocked_24h": blocked,
            "confirmed_threats_24h": confirmed,
            "severity_avg_24h": round(severity_avg, 4),
            "total_alerts": len(alerts),
            "defended_all_time": defended_all_time,
        }
        self._jailbreak_cache = {"_mtime": mtime, "stats": stats}
        return stats

    def _count_artifact_dirs(self) -> dict:
        """Counts of arweave inscriptions + meditation memos for world_footprint."""
        cache = getattr(self, "_artifact_dirs_cache", {})
        now = time.time()
        if cache.get("_ts", 0) > now - 60:  # 60s cache
            return cache["counts"]
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        counts = {}
        try:
            arw_dir = os.path.join(data_dir, "arweave_devnet")
            counts["arweave_inscriptions"] = (
                sum(1 for f in os.listdir(arw_dir) if f.endswith(".tags.json"))
                if os.path.isdir(arw_dir) else 0)
        except Exception:
            counts["arweave_inscriptions"] = 0
        try:
            med_dir = os.path.join(data_dir, "meditation_memos")
            counts["meditation_memos"] = (
                sum(1 for _ in os.listdir(med_dir))
                if os.path.isdir(med_dir) else 0)
        except Exception:
            counts["meditation_memos"] = 0
        self._artifact_dirs_cache = {"_ts": now, "counts": counts}
        return counts

    def _read_inner_memory_stats(self) -> dict:
        """Read inner_memory.db stats (action_chains, vocabulary, creative_works).

        Cached for 30s — direct read-only sqlite COUNT(*) queries via a
        thin connection (no full InnerMemoryStore init — that init runs
        schema migrations + IMW client setup which deadlocks against
        the live store on the same DB file when called from heavy_stats
        refresher thread).

        rFP_trinity_130d_phase2_5_closure §4 Chunk 2.5.D (2026-05-08):
        switched from InnerMemoryStore.get_stats() to direct sqlite
        connection because the full Store init takes >30s on the 1.1GB
        DB and lock-contends with the existing reader instance from
        memory_worker. The dim formulas only need raw counts — the
        thin path is sufficient.
        """
        cache = getattr(self, "_inner_memory_stats_cache", {})
        now = time.time()
        if cache.get("_ts", 0) > now - 30:
            return cache["stats"]
        stats: dict = {}
        try:
            import sqlite3 as _sql
            db_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                    "data", "inner_memory.db")
            if not os.path.exists(db_path):
                return stats
            # Read-only URI mode + busy_timeout 2s. Avoids Store-init churn.
            conn = _sql.connect(
                f"file:{db_path}?mode=ro&immutable=0",
                uri=True, timeout=2.0,
            )
            try:
                conn.execute("PRAGMA busy_timeout=2000")
                for table in ("hormone_snapshots", "program_fires",
                               "action_chains", "creative_works",
                               "event_markers", "vocabulary"):
                    try:
                        c = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        stats[table] = int(c.fetchone()[0])
                    except Exception:
                        stats[table] = 0
            finally:
                conn.close()
        except Exception:
            stats = {"_error": "thin_read_failed"}
        self._inner_memory_stats_cache = {"_ts": now, "stats": stats}
        return stats

    def _read_social_x_gateway_stats(self) -> dict:
        """Read social_x_gateway DB stats (posts/replies last hour + day).

        Cached for 60s. The SocialXGateway instance is held persistently
        to avoid the per-call _init_db + _recover_pending overhead which
        was blocking the gather hot path.
        """
        cache = getattr(self, "_social_x_stats_cache", {})
        now = time.time()
        if cache.get("_ts", 0) > now - 60:
            return cache["stats"]
        try:
            gw = getattr(self, "_social_x_gateway_reader", None)
            if gw is None:
                from titan_plugin.logic.social_x_gateway import SocialXGateway
                self._social_x_gateway_reader = SocialXGateway()
                gw = self._social_x_gateway_reader
            stats = gw.get_stats()
        except Exception:
            stats = {}
        self._social_x_stats_cache = {"_ts": now, "stats": stats}
        return stats

    def _read_events_teacher_stats(self) -> dict:
        """Same pattern: persistent EventsTeacherDB reader, 60s cache."""
        cache = getattr(self, "_events_teacher_stats_cache", {})
        now = time.time()
        if cache.get("_ts", 0) > now - 60:
            return cache["stats"]
        try:
            db = getattr(self, "_events_teacher_reader", None)
            if db is None:
                from titan_plugin.logic.events_teacher import EventsTeacherDB
                self._events_teacher_reader = EventsTeacherDB()
                db = self._events_teacher_reader
            titan_id = self._titan_id if hasattr(self, "_titan_id") else "T1"
            stats = db.get_stats(titan_id)
        except Exception:
            stats = {}
        self._events_teacher_stats_cache = {"_ts": now, "stats": stats}
        return stats

    # DELETED: _boot_outer_trinity (replaced by 3 autostart ModuleSpecs in _register_modules)
    # DELETED: _outer_trinity_loop (replaced by _publish_outer_sources_loop)
    # DELETED: _gather_outer_trinity_sources (replaced by _gather_outer_sources above)


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

        # rFP_trinity_130d_awakening Phase 2 — start InnerPerceptionState.
        # AmbientChangeMonitor samples (cpu_thermal, circadian) at 1Hz on
        # a daemon thread; AudioPerception / VisualPerception are populated
        # by ``_notify_expressive_create`` at every record_expressive site.
        try:
            from titan_plugin.logic.inner_perception import InnerPerceptionState
            from titan_plugin.utils import system_sensor as _sys_sensor

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
            logger.info("[TitanPlugin] InnerPerceptionState started "
                        "(ambient=1Hz; audio/visual via obs_db hook)")
        except Exception as _ip_err:
            logger.warning("[TitanPlugin] InnerPerceptionState start failed: %s",
                           _ip_err, exc_info=True)
            self._inner_perception = None

        # rFP_trinity_130d_awakening Phase 2 — outer-spirit history aggregator.
        # ExperientialMemory access via spirit module's coordinator (lazy
        # lookup so we don't fail boot if spirit isn't up yet).
        try:
            from titan_plugin.logic.outer_spirit_history import OuterSpiritHistory

            def _e_mem_lookup():
                spirit_proxy = self._proxies.get("spirit") if hasattr(self, "_proxies") else None
                if spirit_proxy is None:
                    return None
                # The proxy may expose the coordinator's e_mem via a
                # convenience attribute; fall back to None if not bound.
                coord = getattr(spirit_proxy, "_coordinator", None) or getattr(spirit_proxy, "coordinator", None)
                if coord is None:
                    return None
                return getattr(coord, "_experiential_memory", None) or getattr(coord, "e_mem", None)

            self._outer_spirit_history = OuterSpiritHistory(_e_mem_lookup)
            logger.info("[TitanPlugin] OuterSpiritHistory started "
                        "(env_adapt + graceful_rest + circadian + dream_recall)")
        except Exception as _osh_err:
            logger.warning("[TitanPlugin] OuterSpiritHistory start failed: %s",
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

        # Phase A.S8: Outer sources publisher loop (plugin-only sources → 3 workers)
        asyncio.get_event_loop().create_task(self._publish_outer_sources_loop())

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
            from titan_plugin.logic.outer_body_sensor_refresh import (
                OuterBodySensorRefresh)
            from titan_plugin.logic.outer_mind_sensor_refresh import (
                OuterMindSensorRefresh)
            from titan_plugin.logic.outer_spirit_sensor_refresh import (
                OuterSpiritSensorRefresh)
            # Each sidecar reuses the parent's in-process source-gathering
            # so RPC traffic is unchanged (sidecar reads in-process registry
            # only — no new Solana RPC, no new observatory queries beyond
            # what _gather_outer_sources already does).
            sources_provider = self._gather_outer_sources
            self._outer_body_sensor_sidecar = OuterBodySensorRefresh(
                sources_provider=sources_provider)
            self._outer_mind_sensor_sidecar = OuterMindSensorRefresh(
                sources_provider=sources_provider)
            self._outer_spirit_sensor_sidecar = OuterSpiritSensorRefresh(
                sources_provider=sources_provider)

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
                        "[TitanPlugin] sidecar thread %s crashed:\n%s",
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
                "[TitanPlugin] outer sensor refresh sidecars started in "
                "dedicated threads (body=%.1fs / mind=%.1fs / spirit=%.1fs cadence)",
                self._outer_body_sensor_sidecar._refresh_period_s,
                self._outer_mind_sensor_sidecar._refresh_period_s,
                self._outer_spirit_sensor_sidecar._refresh_period_s,
            )
        except Exception as e:
            logger.warning(
                "[TitanPlugin] outer sensor refresh sidecar boot failed "
                "(non-fatal — Rust outer daemons read last-known): %s", e)

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
