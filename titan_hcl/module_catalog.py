"""
titan_hcl/module_catalog.py — Canonical module catalog for guardian_hcl.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0.

Owns the 47-entry ModuleSpec registration sequence that previously lived in
titan_hcl/core/plugin.py:_register_modules (now DELETED — pure cutover, no
shim, per Maker `feedback_no_shim_old_path_must_be_deleted`).

`build_catalog` is invoked by scripts/guardian_hcl.py inside the
guardian_hcl process. Worker entry functions are imported per-section
(matches the original lazy-import pattern) so unused modules incur no
boot-time import cost.

Signature:
  - `bus`: DivineBus (guardian_hcl's local hub, broker-attached)
  - `guardian`: Guardian instance whose .register() is called 47 times
  - `config`: full merged config dict (loaded via load_titan_config())
  - `titan_id`: canonical Titan identifier (T1/T2/T3)
  - `kernel`: optional kernel-like object — historically `self.kernel`
    references were rare and never load-bearing for the ModuleSpec
    creation itself. Under Phase 6, guardian_hcl has no TitanKernel;
    `kernel=None` is the canonical call.

Mechanical transformations applied during the extraction:
  self._full_config → config
  self.guardian    → guardian
  self.kernel      → kernel
  self.bus         → bus
"""
from __future__ import annotations

import logging
import os

from titan_hcl import bus as _bus_constants
from titan_hcl.guardian_hcl import ModuleSpec
# §11.G.2.5 dep-activation primitives (D-SPEC-90) — used by ModuleSpec
# dependencies fields in the carved catalog (mirrors plugin.py imports).
from titan_hcl.supervision import (
    Dependency, DependencyAction, DependencyKind, DependencySeverity,
)

logger = logging.getLogger(__name__)


def _mod_dep(name: str) -> Dependency:
    """Phase 11 §11.I.8 / D-SPEC-141 (Chunk 11G) — declare an
    ENSURE_RUNNING critical MODULE dep. Used by the §3H.10 dep matrix
    population below.

    Equivalent to:
        Dependency(
            name=name,
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        )

    Honoured by:
      - Orchestrator._compute_boot_order — topological sort respects
        MODULE-kind deps so a child only spawns after its parent reaches
        the boot order.
      - OrchestratorDepActivationMixin._activate_dependencies (§11.G.2.5
        D-SPEC-90) — recursive `start()` of any STOPPED dep before the
        dependent spawns; bounded by
        SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S (30s).
    """
    return Dependency(
        name=name,
        kind=DependencyKind.MODULE,
        severity=DependencySeverity.CRITICAL,
        action=DependencyAction.ENSURE_RUNNING,
    )


def build_catalog(bus, guardian, config, *, titan_id: str, kernel=None) -> None:
    """Register the full Titan module catalog with Guardian. See module docstring."""
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
    _spawn_grad = config.get("microkernel", {}).get(
        "spawn_graduated_workers_enabled", False)

    # IMW — Inner Memory Writer Service. Registered FIRST so other modules
    # that write to inner_memory.db can connect on startup.
    # Autostart only when persistence.enabled=true in config.toml.
    _persistence_cfg = config.get("persistence", {})
    _imw_enabled = bool(_persistence_cfg.get("enabled", False))
    _data_dir_raw = config.get("memory_and_storage", {}).get("data_dir", "./data")
    imw_config = {
        **_persistence_cfg,
        "db_path": _persistence_cfg.get("db_path") or (_data_dir_raw.rstrip("/") + "/inner_memory.db"),
    }
    guardian.register(ModuleSpec(
        name="imw",
        layer="L1",  # Microkernel v2 §A.5 — L1 persistence service (writer for inner_memory.db, L1's DB)
        entry_fn=imw_main,
        config=imw_config,
        rss_limit_mb=300,  # asyncio + sqlite3 + msgpack only; stays lean
        autostart=_imw_enabled,   # Only start when master switch flipped on
        lazy=False,
        heartbeat_timeout=60.0,
        reply_only=True,  # IMW IPC is via unix socket, not bus broadcasts
        # Phase 6 / D-SPEC-135 / v1.62.0: IMW must spawn (not fork) because
        # imw_main runs an asyncio loop + heartbeat thread + bus-watcher
        # thread in parallel. Under fork-mode from guardian_hcl (which has
        # 4+ background threads of its own — BusSocketClient writer +
        # reader, GuardianStatePublisher loop, module_ready_publisher,
        # lifecycle dispatcher, supervision loop), the child inherits a
        # locked-mp.Queue / locked-mp.Lock state from threads that no
        # longer exist post-fork. Live evidence (T3 2026-05-26): 4 of 5
        # IMW workers HANG on the first `send_queue.put(MODULE_READY)`
        # call indefinitely (PRE-PUBLISH stderr log emits, POST-PUBLISH
        # never does) → /health stays DEGRADED forever for 5 IMW writers.
        # spawn forces a fresh interpreter; no inherited lock state.
        start_method="spawn",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
        # imw is a reply_only Unix-socket persistence daemon — no SHM lifecycle
        # slot (§11.I.5); excluded from the /v6/readiness roster.
        reports_lifecycle_slot=False,
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
        config.get("microkernel", {}).get(
            "a8_output_verifier_subprocess_enabled", False))
    guardian.register(ModuleSpec(
        name="output_verifier",
        layer="L2",  # security/verification gate — L2 service
        entry_fn=output_verifier_worker_main,
        config=config,
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
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('timechain'),
        ],
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
        config.get("microkernel", {}).get(
            "a8_reflex_subprocess_enabled", False))
    guardian.register(ModuleSpec(
        name="reflex",
        layer="L3",  # reflex aggregation — L3 service per rFP §A.8.5
        entry_fn=reflex_worker_main,
        config=config,
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
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
        config.get("microkernel", {}).get(
            "a8_agency_subprocess_enabled", False))
    guardian.register(ModuleSpec(
        name="agency_worker",
        layer="L3",  # impulse decoder + helper execution — L3 service
        entry_fn=agency_worker_main,
        config=config,
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
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
    _wm_cfg = config.get("warning_monitor", {})
    # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
    # warning_monitor drain at modules/warning_monitor_worker.py:209
    # consumes one broadcast type (SILENT_SWALLOW_REPORT).
    guardian.register(ModuleSpec(
        name="warning_monitor",
        layer="L3",  # observability service — same tier as observatory
        entry_fn=warning_monitor_worker_main,
        config=_wm_cfg,
        rss_limit_mb=300,    # tail + small in-memory aggregations only
        autostart=True,       # Always on — visibility is non-negotiable
        lazy=False,
        heartbeat_timeout=120.0,
        reply_only=False,
        broadcast_topics=[_bus_constants.SILENT_SWALLOW_REPORT],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        b2_1_swap_critical=False,  # M5: light-state worker; respawn-OK
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
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
    guardian.register(ModuleSpec(
        name="health_monitor",
        layer="L3",  # observability + advisory service
        entry_fn=health_monitor_worker_main,
        config=config,  # full config — plugins resolve sections
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[_bus_constants.MODULE_SHUTDOWN],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
    ))

    # Observatory Writer Service — second IMW instance for observatory.db.
    # Same imw_main entry function, different config: own socket, own WAL,
    # own metrics file (auto-namespaced by name), own primary+shadow DBs.
    # Scoped per rFP_observatory_writer_service (drafted 2026-04-21);
    # microkernel-v2-aligned (L3 DB ownership). Default OFF — Maker flips
    # `enabled=true` in [persistence_observatory] when ready for Phase 1.
    _obs_persistence_cfg = config.get("persistence_observatory", {})
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
    guardian.register(ModuleSpec(
        name="observatory_writer",
        layer="L3",  # Microkernel v2 §A.5 — L3 persistence service (writer for observatory.db, L3's DB)
        entry_fn=imw_main,                    # reuse same entry — fully parameterized by config
        config=_obs_writer_config,
        rss_limit_mb=300,
        autostart=_obs_writer_enabled,        # default OFF until Maker flips
        lazy=False,
        heartbeat_timeout=60.0,
        reply_only=True,                       # Unix-socket IPC, no bus broadcasts
        # Phase 6 / D-SPEC-135: spawn-mode (same rationale as 'imw' above —
        # multi-threaded asyncio + heartbeat + bus-watcher; fork from
        # multi-threaded guardian_hcl deadlocks on inherited locks).
        start_method="spawn",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
        # Unix-socket persistence daemon — no SHM lifecycle slot (§11.I.5).
        reports_lifecycle_slot=False,
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
        _w_cfg_section = config.get(_w_section, {})
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
        guardian.register(ModuleSpec(
            name=_w_name,
            layer="L3",
            entry_fn=imw_main,
            config=_w_writer_config,
            rss_limit_mb=300,
            autostart=_w_enabled,
            lazy=False,
            heartbeat_timeout=60.0,
            reply_only=True,
            # Phase 6 / D-SPEC-135: spawn-mode (same rationale as 'imw' /
            # 'observatory_writer' above).
            start_method="spawn",
            # Unix-socket persistence daemon — no SHM lifecycle slot (§11.I.5).
            reports_lifecycle_slot=False,
        ))

    # Memory module (FAISS + Kuzu + DuckDB)
    memory_config = {
        **config.get("inference", {}),
        **config.get("memory_and_storage", {}),
    }
    guardian.register(ModuleSpec(
        name="memory",
        layer="L2",  # Microkernel v2 §A.5 — L2 cognitive substrate (FAISS+Kuzu+DuckDB)
        entry_fn=memory_worker_main,
        config=memory_config,
        rss_limit_mb=2000,  # FAISS + Kuzu + DuckDB: ~1100-1200MB steady on T1 (40min uptime). VmRSS includes mmap'd DB pages from page cache, which inflates the reading on rapid restarts. Old 1400MB limit was sized for the Cognee era and caused T2 doom-loop.
        # Phase 11 §11.I.8 / Chunk 11G (§3H.10) — memory was lazy=True under
        # the pre-Phase-11 catalog (started on first MemoryProxy use). The
        # §3H.10 matrix PROMOTES it to MANDATORY because /chat + dream + recall
        # all need it before fleet_ready latches; a lazy first-spawn during
        # the first /chat re-introduces the timechain-scan-style cold-boot
        # latency that Phase 11 is designed to eliminate.
        autostart=True,
        lazy=False,
        heartbeat_timeout=120.0,  # Memory queries can block for 30s+
        reply_only=True,  # Memory only needs targeted QUERY messages, not broadcasts
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority. Maker 2026-05-28:
        # HEAVY worker (FAISS/Kuzu/DuckDB cold-load) → Phase B so it doesn't gate
        # fleet_ready or pile cold-import CPU on mandatory boot. Consumers read
        # memory_state SHM (graceful-empty until it boots); boot cap bounds the
        # Phase B spike.
        boot_priority="post_boot",
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
        config.get("microkernel", {}).get(
            "a8_sage_scholar_gatekeeper_subprocess_enabled", False))
    # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
    # rl drain at modules/rl_worker.py:136-147 consumes one broadcast
    # type (SAGE_RECORD_TRANSITION); MODULE_SHUTDOWN + QUERY are targeted.
    guardian.register(ModuleSpec(
        name="recorder",
        layer="L2",  # Microkernel v2 §A.5 — L2 higher cognition (IQL chain learning)
        entry_fn=recorder_worker_main,
        config=config.get("stealth_sage", {}),
        rss_limit_mb=3000,
        autostart=_a8_sage_subproc_enabled,  # §A.8.7: autostart when flag-on
        lazy=not _a8_sage_subproc_enabled,   # §A.8.7: eager when flag-on
        broadcast_topics=[_bus_constants.SAGE_RECORD_TRANSITION],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # LLM/Inference module (Agno agent — ~500MB)
    # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
    # llm drain at modules/llm_worker.py:103-112 consumes one broadcast
    # type (LLM_TEACHER_REQUEST); MODULE_SHUTDOWN + QUERY are targeted.
    guardian.register(ModuleSpec(
        name="llm",
        layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (Agno inference, human-time)
        entry_fn=llm_worker_main,
        config=config.get("inference", {}),
        rss_limit_mb=1000,
        autostart=True,  # Changed: Language Teacher needs llm at boot
        lazy=False,
        heartbeat_timeout=120.0,  # LLM calls can block 30s+; match Spirit/Memory timeout
        broadcast_topics=[_bus_constants.LLM_TEACHER_REQUEST],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # SPEC v1.17.0 / D-SPEC-72 — agno_worker hosts the Agno Agent + chat
    # pipeline (PreHook + PostHook + tools + guardrails + AsyncSqliteDb
    # session store). Receives CHAT_REQUEST / CHAT_STREAM_REQUEST from
    # api_subprocess via agno_proxy; dispatches via agent.arun() in-worker.
    # See SPEC §9.B agno_worker block for the full contract.
    # Merged inference + agent config so the worker's _init_worker_plugin_and_agent
    # can resolve provider via §9.F.1 inference module.
    _agno_cfg = {
        **config.get("inference", {}),
        **config.get("agent", {}),
    }
    # Surface the [agent] block as a nested key for downstream usage too
    _agno_cfg["agent"] = config.get("agent", {})
    _agno_cfg["inference"] = config.get("inference", {})
    # ζ.0/ζ.1 (D-SPEC-79, 2026-05-18) — propagate [chat] section so the
    # ChatTierClassifier in agno_hooks.PreHook sees [[chat.tiers]] blocks.
    # Without this the worker subprocess only got inference+agent and the
    # classifier fell through to "passthrough" with all features on —
    # which defeated the whole point of ζ.1 feature gating.
    _agno_cfg["chat"] = config.get("chat", {})
    # Phase 8 (D-SPEC-PHASE8) — propagate [synthesis] so agno_worker's
    # delegate_live wiring (config[synthesis][skill][delegate_live]) sees the
    # per-Titan ~/.titan/microkernel_<id>.toml override. Without this the
    # agno tool's match_procedural_skill always reads the config default
    # (False) — the T3 canary's delegate_live=true never takes effect.
    _agno_cfg["synthesis"] = config.get("synthesis", {})
    guardian.register(ModuleSpec(
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
        # Phase 11 §11.I.5 (Chunk 11N) — bumped 60 → 180 because the
        # legitimate cold-boot path is Agent build (~30s) + eager OVG
        # warmup (~30s on T2/T3 devnet, ~130s on T1 mainnet 50MB chain).
        # The 60s budget cut it too fine — observed heartbeat-timeout
        # restart loop on T3 cascade 2026-05-27. 180s gives 3× cold-boot
        # headroom while keeping a tight cap on truly-stuck workers
        # (active SHM heartbeat starts BEFORE Agent build per Phase 11
        # §11.I.5, so this is a defense-in-depth bound — the SHM-direct
        # probe in §11.I.3 is the primary readiness check).
        heartbeat_timeout=180.0,
        broadcast_topics=[
            _bus_constants.CHAT_REQUEST,
            _bus_constants.CHAT_STREAM_REQUEST,
            _bus_constants.KERNEL_EPOCH_TICK,
            _bus_constants.SAVE_NOW,
            # RFP_phase_c_titan_hcl_cleanup Phase A (2026-05-21): agno_worker
            # is the new DREAM_INBOX_REPLAY consumer (re-answers messages
            # buffered during dream) — moved from the retired parent
            # _v4_event_bridge_loop. dream_state_worker broadcasts it dst="all".
            _bus_constants.DREAM_INBOX_REPLAY,
            # Phase 8 (D-SPEC-PHASE8): agno_worker subscribes to skill-
            # lifecycle events for cosmetic log + match-tool refresh
            # signals. META_SKILL_REJECTED is load-bearing — a delegated
            # skill that mid-flight gets rejected should NOT keep firing.
            _bus_constants.META_SKILL_COMPILED,
            _bus_constants.META_SKILL_REJECTED,
            # Phase 8 fold-in (P8.Y): CGN lexicon snapshot refresh. agno
            # refreshes plugin.cgn_lexicon on this event so the P7
            # _ground_for_goal_hook returns real concept_ids.
            _bus_constants.CGN_LEXICON_UPDATED,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=True,  # data/agno_sessions.db is critical-data per §11.H
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority. Maker 2026-05-28:
        # HEAVY worker (agno framework + LLM agent build) → Phase B so chat
        # readiness doesn't gate fleet_ready; the boot cap bounds its spike.
        boot_priority="post_boot",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('memory'),
            _mod_dep('output_verifier'),
            _mod_dep('timechain'),
        ],
    ))

    # Body module (5DT somatic sensors — lightweight, always-on)
    # Note: RSS includes inherited parent process memory from fork (~250MB),
    # so limit must account for that baseline.
    body_config = {
        **config.get("body", {}),
        "api_port": int(config.get("api", {}).get("port", 7777)),
        # Microkernel v2 Phase A §A.7 / §L1 — shm feature flags. Must
        # be passed through so body_worker's _read_flag() and
        # RegistryBank.is_enabled() can resolve
        # microkernel.shm_body_fast_enabled correctly. Same pattern
        # as spirit_worker (mirror the comment there). Without this,
        # the body shm fast-path is silently no-op even with the
        # flag flipped true in titan_params.toml.
        "microkernel": config.get("microkernel", {}),
    }
    # rFP_worker_broadcast_topics_completion §4.A.2 (Batch 2):
    # body drain at modules/body_worker.py:255-303 consumes 4 broadcasts
    # (FILTER_DOWN, FOCUS_NUDGE, CONVERSATION_STIMULUS, INTERFACE_INPUT);
    # MODULE_SHUTDOWN + QUERY are targeted.
    guardian.register(ModuleSpec(
        name="body",
        layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT somatic)
        entry_fn=body_worker_main,
        config=body_config,
        rss_limit_mb=800,   # was 500; fork-inherited parent memory grew from ~250MB to ~400MB+ (2026-04-17)
        autostart=True,  # Body senses must always be active
        lazy=False,
        broadcast_topics=[
            _bus_constants.FILTER_DOWN, _bus_constants.FOCUS_NUDGE,
            _bus_constants.CONVERSATION_STIMULUS, _bus_constants.INTERFACE_INPUT,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
    ))

    # Mind module (MoodEngine, SocialGraph — ~200MB)
    mind_config = {
        "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
        # Microkernel v2 Phase A §A.7 / §L1 — shm feature flags
        # (same passthrough pattern as body/spirit; without this,
        # mind shm fast-path is silently no-op).
        "microkernel": config.get("microkernel", {}),
    }
    # mind drain at modules/mind_worker.py consumes these broadcasts;
    # MODULE_SHUTDOWN + QUERY are targeted. OUTER_SOURCES_SNAPSHOT dropped
    # (Phase C C.7/C.8): mind_worker now reads outer-source state SHM-direct.
    guardian.register(ModuleSpec(
        name="mind",
        layer="L1",  # Microkernel v2 §A.5 — L1 Trinity daemon (5DT cognitive)
        entry_fn=mind_worker_main,
        config=mind_config,
        rss_limit_mb=700,   # was 500; fork-inherited parent RSS ~400MB left only 100MB headroom — caused T3 cascade (2026-04-17). Memory profiling tool (DEFERRED TOP) will identify real optimization targets.
        autostart=True,  # Mind senses should always be active
        lazy=False,
        broadcast_topics=[
            _bus_constants.FILTER_DOWN, _bus_constants.FOCUS_NUDGE,
            _bus_constants.CONVERSATION_STIMULUS, _bus_constants.INTERFACE_INPUT,
            _bus_constants.SENSE_VISUAL, _bus_constants.SENSE_AUDIO,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
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
    if config.get("microkernel", {}).get("l0_rust_enabled", False):
        cognitive_worker_config = {
            "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
            # Microkernel flag passthrough — cognitive_worker_main checks it
            # defensively even though registration is already gated.
            "microkernel": config.get("microkernel", {}),
            # Banner config for titan_id resolution.
            "info_banner": config.get("info_banner", {}),
            # Engine configs read from titan_params.toml inside the worker
            # via _load_toml_section helper — no need to thread them here.
            # titan_vm config kept here for InnerTrinityCoordinator's
            # internal NervousSystem (lightweight VM context).
            "titan_vm": config.get("titan_vm", {}),
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
        guardian.register(ModuleSpec(
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
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority. Maker
            # 2026-05-28: HEAVY worker (269KB, torch reasoning engine cold-load)
            # → Phase B so it doesn't gate fleet_ready or pile cold-import CPU on
            # mandatory boot; the boot concurrency cap bounds its spike.
            boot_priority="post_boot",
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
            dependencies=[
                _mod_dep('memory'),
            ],
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
        guardian.register(ModuleSpec(
            name="expression_worker",
            layer="L2",
            entry_fn=expression_worker_main,
            config={
                "data_dir": config.get(
                    "memory_and_storage", {}).get("data_dir", "./data"),
                "microkernel": config.get("microkernel", {}),
                "info_banner": config.get("info_banner", {}),
            },
            rss_limit_mb=400,   # ExpressionManager + 6 composites is light
            autostart=True,
            lazy=False,
            heartbeat_timeout=60.0,
            broadcast_topics=_EXPRESSION_WORKER_SUBSCRIBE_TOPICS,
            start_method="spawn" if _spawn_grad else "fork",
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
            boot_priority="post_boot",
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
            dependencies=[
                _mod_dep('cognitive_worker'),
            ],
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
        if config.get("microkernel", {}).get("outer_interface_worker_enabled", True):
            outer_interface_worker_config = {
                "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
                "microkernel": config.get("microkernel", {}),
                "info_banner": config.get("info_banner", {}),
                "outer_interface":   config.get("outer_interface", {}),
                "self_exploration":  config.get("self_exploration", {}),
                "action_decoder":    config.get("action_decoder", {}),
                "action_narrator":   config.get("action_narrator", {}),
                "kin":               config.get("kin", {}),
            }
            from titan_hcl.modules.outer_interface_worker import outer_interface_worker_main
            guardian.register(ModuleSpec(
                name="outer_interface_worker",
                layer="L2",
                entry_fn=outer_interface_worker_main,
                config=outer_interface_worker_config,
                rss_limit_mb=500,
                autostart=True,
                lazy=False,
                heartbeat_timeout=60.0,
                broadcast_topics=[
                    _bus_constants.REASONING_STATS_UPDATED, _bus_constants.NEUROMOD_STATS_UPDATED,
                    _bus_constants.CHI_UPDATED, _bus_constants.KERNEL_EPOCH_TICK,
                    _bus_constants.EXPRESSION_FIRED, _bus_constants.CONVERSATION_STIMULUS,
                    _bus_constants.SPEAK_REQUEST_PENDING, _bus_constants.GREAT_KIN_PULSE,
                ],
                b2_1_swap_critical=False,
                start_method="spawn" if _spawn_grad else "fork",
                # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
                boot_priority="post_boot",
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
        if config.get("microkernel", {}).get("self_reflection_worker_enabled", True):
            self_reflection_worker_config = {
                "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
                "microkernel": config.get("microkernel", {}),
                "info_banner": config.get("info_banner", {}),
                "self_reflection":  config.get("self_reflection", {}),
                "self_reasoning":   config.get("self_reasoning", {}),
                "prediction":       config.get("prediction_engine", {}),
                "cgn":              config.get("cgn", {}),
            }
            from titan_hcl.modules.self_reflection_worker import self_reflection_worker_main
            guardian.register(ModuleSpec(
                name="self_reflection_worker",
                layer="L2",
                entry_fn=self_reflection_worker_main,
                config=self_reflection_worker_config,
                rss_limit_mb=800,
                autostart=True,
                lazy=False,
                heartbeat_timeout=90.0,
                broadcast_topics=[
                    _bus_constants.REASONING_STATS_UPDATED, _bus_constants.META_REASONING_STATS_UPDATED,
                    _bus_constants.EXPERIENCE_STIMULUS, _bus_constants.DREAMING_STATE_UPDATED,
                    _bus_constants.CGN_CROSS_INSIGHT, _bus_constants.KERNEL_EPOCH_TICK,
                    # rFP_meta_reasoning_self_reasoning_resolver_migration / SPEC §9.B
                    # + D-SPEC-70 v1.15.0 — cognitive_worker fires META_INTROSPECT_REQUEST
                    # (fire-and-forget per §8.0.ter D-SPEC-48) per META INTROSPECT action;
                    # self_reflection_worker handler runs sr.introspect(**payload),
                    # persists to data/inner_memory.db.self_insights via _persist_insight(),
                    # writes result to inner_self_insight.bin SHM slot. Closes F-8.
                    _bus_constants.META_INTROSPECT_REQUEST,
                ],
                b2_1_swap_critical=True,
                start_method="spawn" if _spawn_grad else "fork",
                # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
                boot_priority="post_boot",
                # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
                dependencies=[
                    _mod_dep('cognitive_worker'),
                ],
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
    if config.get("microkernel", {}).get("social_worker_enabled", False):
        social_worker_config = {
            "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
            "microkernel": config.get("microkernel", {}),
            "info_banner": config.get("info_banner", {}),
            # social_x section carries: gateway db path, archetype configs,
            # canonical_poller_titan_id, recency-boost tunables, post limits.
            "social_x": config.get("social_x", {}),
        }
        from titan_hcl.modules.social_worker import social_worker_main
        guardian.register(ModuleSpec(
            name="social_worker",
            layer="L2",
            entry_fn=social_worker_main,
            config=social_worker_config,
            rss_limit_mb=500,
            autostart=True,
            lazy=False,
            heartbeat_timeout=120.0,
            broadcast_topics=[
                _bus_constants.EXPRESSION_FIRED,
                _bus_constants.MEDITATION_COMPLETE,
                _bus_constants.KIN_SIGNAL,
                _bus_constants.SOCIAL_RECEIVED,
                _bus_constants.SOCIAL_CATALYST,
                _bus_constants.MENTION_RECEIVED,
                _bus_constants.FELT_EXPERIENCE_CAPTURED,
                _bus_constants.ENGAGEMENT_SNAPSHOT_TAKEN,
            ],
            start_method="spawn" if _spawn_grad else "fork",
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
            boot_priority="post_boot",
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
            dependencies=[
                _mod_dep('social_graph'),
            ],
        ))

    # social_graph_worker — extracted from mind_worker per
    # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1, 2026-05-14).
    # Hosts SocialGraph (Phase 13 Sage Socialite) + data/social_graph.db +
    # social_graph_state.bin SHM publisher (G21 single-writer).
    # ALWAYS-ON autostart — no flag-gate (replaces the legacy alias rot).
    # broadcast_topics minimal: dispatch arrives as dst="social_graph"
    # _bus_constants.QUERY (not a broadcast), only MODULE_SHUTDOWN + SAVE_NOW are
    # broadcasts the worker needs to consume.
    from titan_hcl.modules.social_graph_worker import (
        social_graph_worker_main,
    )
    guardian.register(ModuleSpec(
        name="social_graph",
        layer="L2",
        entry_fn=social_graph_worker_main,
        config={
            "data_dir": config.get(
                "memory_and_storage", {}).get("data_dir", "./data"),
            "info_banner": config.get("info_banner", {}),
            "social_graph": config.get("social_graph", {}),
        },
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=True,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # metabolism_worker — extracted from titan_HCL inline wire per
    # rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51 (v1.7.2, 2026-05-14).
    # Hosts MetabolismController + metabolism_state.bin SHM publisher
    # (G21 single-writer). ALWAYS-ON autostart — no flag-gate
    # (replaces the inline `_wire_metabolism` body which previously
    # constructed MetabolismController as an in-process attribute).
    # Subscribes to _bus_constants.QUERY (dst=metabolism) for evaluate_gate /
    # async state queries + SOLANA_BALANCE_UPDATED for responsive
    # tier refresh.
    from titan_hcl.modules.metabolism_worker import (
        metabolism_worker_main,
    )
    guardian.register(ModuleSpec(
        name="metabolism",
        layer="L2",
        entry_fn=metabolism_worker_main,
        config={
            "growth_metrics": config.get("growth_metrics", {}),
            "network": config.get("network", {}),
            "info_banner": config.get("info_banner", {}),
        },
        rss_limit_mb=100,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.SOLANA_BALANCE_UPDATED,
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # journey_persistence_worker — P0.5 / D-SPEC-131 §G5.1 (PLAN
    # §6.5.6). Sole L2 consumer of BODY_BALANCE_GIFT + MIND_BALANCE_GIFT
    # events published by the body/mind Rust daemons on their own
    # sphere clock's balanced rising-edge (sub-1% of Schumann ticks).
    # Translates each gift to one row of `trinity_journey_gifts` in
    # consciousness.db (already §24 Arweave-backed → inherits sovereignty).
    # Best-effort delivery: SQL or queue overflow → warn-and-drop; not
    # load-bearing for any tick.
    from titan_hcl.modules.journey_persistence_worker import (
        journey_persistence_worker_main,
    )
    _journey_cfg = {
        "info_banner": config.get("info_banner", {}),
        "consciousness_db": config.get(
            "memory_and_storage", {}
        ).get("consciousness_db", "./data/consciousness.db"),
    }
    guardian.register(ModuleSpec(
        name="journey_persistence",
        layer="L2",
        entry_fn=journey_persistence_worker_main,
        # Phase 6 / D-SPEC-135 sizing correction (2026-05-26): the original
        # P0.5/P0.6-C ModuleSpec set rss_limit_mb=60 but live evidence on
        # T2 + T3 fleet boots showed actual VmRSS at boot = 73 MB (Python
        # 3.12 interpreter + sqlite3 + msgpack + asyncio import baseline is
        # ~55-65 MB on its own; worker logic + IMW client + buffer push it
        # to 70-80 MB). 60 MB caused boot-loop kills → /health DEGRADED.
        # 150 MB matches realistic baseline + ~2× headroom for SQLite page
        # cache + journal-replay temporaries. Per `feedback_no_rss_band_aid_understand_root_cause`
        # the limit reflects ACTUAL observed memory, not a guess.
        config=_journey_cfg,
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.BODY_BALANCE_GIFT,
            _bus_constants.MIND_BALANCE_GIFT,
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # corrective_events_persistence_worker — P0.6-C / D-SPEC-132 §6.6.6
    # (PLAN §6.6.6). Sole L2 consumer of EXTREME_IMBALANCE_DETECTED +
    # CORRECTIVE_NUDGE events. Pairs them by (source_part, side,
    # dominant_dim_idx) into one trinity_corrective_events row per cycle.
    # Best-effort delivery: orphan fires (no matching nudge within 5s)
    # persisted with nudge_* NULL.
    from titan_hcl.modules.corrective_events_persistence_worker import (
        corrective_events_persistence_worker_main,
    )
    guardian.register(ModuleSpec(
        name="corrective_events_persistence",
        layer="L2",
        entry_fn=corrective_events_persistence_worker_main,
        config=_journey_cfg,  # same consciousness_db path + info_banner
        # Phase 6 / D-SPEC-135 sizing correction (2026-05-26): see
        # journey_persistence rationale above — same Python+sqlite baseline.
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.EXTREME_IMBALANCE_DETECTED,
            _bus_constants.CORRECTIVE_NUDGE,
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
    guardian.register(ModuleSpec(
        name="life_force",
        layer="L2",
        entry_fn=life_force_worker_main,
        config={
            "life_force": config.get("life_force", {}),
            "info_banner": config.get("info_banner", {}),
        },
        rss_limit_mb=100,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.KERNEL_EPOCH_TICK,
            _bus_constants.DREAM_STATE_CHANGED,
            _bus_constants.MEDITATION_COMPLETE,
            _bus_constants.EXPRESSION_FIRED,
            _bus_constants.NEUROMOD_STATS_UPDATED,
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
    guardian.register(ModuleSpec(
        name="studio",
        layer="L2",
        entry_fn=studio_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
            "expressive": config.get("expressive", {}),
            "inference": config.get("inference", {}),
        },
        rss_limit_mb=200,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.KERNEL_EPOCH_TICK,
            _bus_constants.MODULE_SHUTDOWN,
            _bus_constants.SAVE_NOW,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
    guardian.register(ModuleSpec(
        name="dream_state",
        layer="L2",
        entry_fn=dream_state_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
        },
        rss_limit_mb=200,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.DREAMING_STATE_UPDATED,
            _bus_constants.KERNEL_EPOCH_TICK,
            _bus_constants.MODULE_SHUTDOWN,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # synthesis_worker — Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0
    # §25 / §9.B `synthesis_worker` block). 2026-05-23. L2 worker, sole
    # writer (G21 / INV-Syn-3) for activation_state DuckDB + synth_status.bin
    # SHM watermark. Cross-process consumers read activation_state via the
    # BridgeRecall pattern (G18 / INV-Syn-4 — watermark-gated). Phase 1
    # producers of MEMORY_RETRIEVAL_USED: core/memory.py._cognee_search
    # (use-gated emit per item passed to LLM context). 60s recompute
    # interval; titan_id resolved per-Titan via state_registry.
    from titan_hcl.modules.synthesis_worker import synthesis_worker_main
    guardian.register(ModuleSpec(
        name="synthesis",
        layer="L2",
        entry_fn=synthesis_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
            # G21 / INV-Syn-3: synthesis_worker owns synthesis.duckdb
            # (NOT titan_memory.duckdb — that's memory_worker's R/W
            # territory; sharing it across workers triggers DuckDB's
            # cross-process R/W lock rejection).
            "memory_db_path": os.path.join(
                config.get("memory_and_storage", {}).get(
                    "data_dir", "./data"),
                "synthesis.duckdb"),
            # Phase 4 FU-2 — Ollama Cloud provider config for the
            # ConsolidationPass LLM proposer. Threaded through from the
            # main [inference] block so synthesis_worker doesn't have to
            # re-merge config. titan_hcl.inference.get_provider("ollama_cloud", cfg)
            # consumes this dict.
            "inference": dict(config.get("inference", {}) or {}),
            # Phase 5 (D-SPEC-PHASE5) — [synthesis] subtable threaded
            # through so per-Titan overrides (~/.titan/microkernel_<id>.toml
            # [synthesis] block) reach synthesis_worker_main. Currently
            # consumed keys: `fork_gc_live` (bool; default False per
            # Maker decision 2026-05-27 — dry-run until soak validates).
            "synthesis": dict(config.get("synthesis", {}) or {}),
        },
        # FU-3 — bumped from 200 to 240. Root-cause: Phase 4 added the
        # Kuzu spine mmap (+~3MB) + consolidation thread + LLM provider
        # state (+~5MB Ollama Cloud httpx client + tokenizers cache). The
        # 200MB cap was set in Phase 1 when synthesis_worker had no spine
        # + no LLM provider. NOT a leak (per
        # feedback_no_rss_band_aid_understand_root_cause.md): the 30MB
        # margin is real accounted-for memory; 240MB matches steady-state.
        #
        # 2026-05-27 Phase-6-soak follow-up: T1 (mainnet) live evidence
        # showed steady-state RSS = 262MB after 109 activation_state rows +
        # 19 standing bundles loaded from synthesis.duckdb. DuckDB column-
        # store + page cache for the loaded data weighs more than the FU-3
        # estimate accounted for. Bumping 240 → 350 (262 actual + ~33%
        # headroom for further growth as the spine accumulates more
        # bundles + activation rows during normal operation). Backed by
        # observed RSS, not a guess. NOT a band-aid: Maker memory
        # `feedback_no_rss_band_aid_understand_root_cause` is satisfied
        # by the FU-3 + this measurement chain — each bump has a
        # documented accounted source.
        rss_limit_mb=350,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.MEMORY_RETRIEVAL_USED,
            # Operator-closure C1 (SPEC §25.9): per-TURN knowledge-moment signal
            # from agno (post-LLM). synthesis_worker's SovereigntyRatioMeter
            # records the per-turn denominator from this (not per-item).
            _bus_constants.KNOWLEDGE_MOMENT,
            # Operator-closure C2 (W7): chat-time self-oracle tool (coding_
            # sandbox) ships its pre-computed verdict → OracleRouter companion
            # buffer → dream-boundary OracleVerdictBatch flush (coverage).
            _bus_constants.TOOL_CALL_VERDICT_RECORD,
            # Phase 2 D-P2-4: standing-contract maintenance event, single
            # consumer = synthesis_worker (sole writer of
            # association_bundles). Post-seal contract hook in
            # timechain_v2.Mempool/BlockBuilder publishes.
            _bus_constants.MAINTAIN_BUNDLE,
            # Phase 4 §P4.G: dream-boundary consolidation pass trigger.
            # dream_state_worker emits this on sleep/wake transitions
            # (v1.8.2 D-SPEC-56 canonical producer). synthesis_worker is
            # an INDEPENDENT listener (INV-11 restart-isolation; does
            # not attach to cognitive_worker's off-tick suite).
            _bus_constants.DREAM_STATE_CHANGED,
            # Phase 5 (D-SPEC-PHASE5): hypothesis-fork lifecycle command
            # surface. POST /v6/synthesis/forks/* endpoints publish this
            # to the bus (dst="synthesis") and the worker dispatches
            # create/record_exploration_tx/graduate_manual/abandon/sweep
            # ops to HypothesisForkStore (sole writer per INV-Syn-8). The
            # worker emits SYNTHESIS_FORK_COMMAND_RESULT carrying
            # request_id so callers can correlate.
            _bus_constants.SYNTHESIS_FORK_COMMAND,
            # Phase 7 (D-SPEC-PHASE7): working-memory buffer command surface.
            # agno_worker publishes this on every BufferCache write-through;
            # synthesis_worker (sole writer per INV-Syn-16) persists the row
            # to `actr_buffers` + atomic-writes buffers_snapshot.json.
            _bus_constants.SYNTHESIS_BUFFER_COMMAND,
            # Phase 8 (D-SPEC-PHASE8): procedural skill miner + verifier
            # lifecycle. META_SKILL_COMPILATION_CANDIDATE wakes the miner
            # at dream_boundary (emitted by actr_procedural_skill_proposer
            # SC). META_SKILL_COMPILED / VERIFIED / REJECTED / SOFT_RETIRED
            # are emitted BY synthesis_worker — listed here so subscribers
            # (agno_worker, Observatory) can opt into them. CGN_LEXICON_UPDATED
            # is emitted by cgn_worker; synthesis_worker subscribes only for
            # observability (does not consume).
            _bus_constants.META_SKILL_COMPILATION_CANDIDATE,
            _bus_constants.META_SKILL_COMPILED,
            _bus_constants.META_SKILL_VERIFIED,
            _bus_constants.META_SKILL_REJECTED,
            _bus_constants.META_SKILL_SOFT_RETIRED,
            # Phase 9 (D-SPEC-PHASE9): SKILL_REPAIR_FORK_SPAWNED is emitted BY
            # synthesis_worker (SkillFailureTracker, §9.3) — listed so agno +
            # Observatory can opt in. USER_FEEDBACK_SIGNAL is emitted by
            # agno_worker on explicit thumbs-up/down; synthesis_worker is the
            # sole consumer (INV-Syn-24 Tier-2 override via UserFeedbackOverride).
            _bus_constants.SKILL_REPAIR_FORK_SPAWNED,
            _bus_constants.USER_FEEDBACK_SIGNAL,
            _bus_constants.KERNEL_EPOCH_TICK,
            _bus_constants.MODULE_SHUTDOWN,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('timechain'),
            _mod_dep('memory'),
        ],
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
    guardian.register(ModuleSpec(
        name="observatory",
        layer="L3",
        entry_fn=observatory_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
            "frontend": config.get("frontend", {}),
        },
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            *_OBSERVATORY_V4_EVENT_TYPES,
            _bus_constants.MODULE_SHUTDOWN,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
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
    guardian.register(ModuleSpec(
        name="meditation",
        layer="L2",
        entry_fn=meditation_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
        },
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.MEDITATION_REQUEST,
            _bus_constants.MEDITATION_FORCE_END,
            _bus_constants.EXPRESSION_FIRED,
            _bus_constants.KERNEL_EPOCH_TICK,
            # MODULE_READY removed v1.29.0 — meditation_worker no longer
            # waits on memory's MODULE_READY broadcast per SPEC §11.G.2.5
            # (Guardian's pre-start activation guarantees dep readiness).
            _bus_constants.SAVE_NOW,
            _bus_constants.MODULE_SHUTDOWN,
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
                check=lambda: guardian.is_running("memory"),
            ),
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 adds timechain to
            # meditation's deps (meditation persists its anchor block via
            # TimeChain on every cycle close).
            _mod_dep("timechain"),
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
    guardian.register(ModuleSpec(
        name="sovereignty",
        layer="L2",
        entry_fn=sovereignty_worker_main,
        config={},
        rss_limit_mb=150,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.MODULE_SHUTDOWN,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('timechain'),
        ],
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
    guardian.register(ModuleSpec(
        name="interface_advisor",
        layer="L2",
        entry_fn=interface_advisor_worker_main,
        config={
            "titan_id": config.get("network", {}).get(
                "titan_id"),
        },
        rss_limit_mb=100,
        autostart=True,
        lazy=False,
        heartbeat_timeout=60.0,
        broadcast_topics=[
            _bus_constants.MODULE_SHUTDOWN,
        ],
        start_method="spawn" if _spawn_grad else "fork",
        critical_data_writer=False,
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
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
        **config,
        # data_dir helper used by hormonal_worker to load/save persisted state.
        "data_dir": config.get("memory_and_storage", {}).get(
            "data_dir", "./data"),
    }
    _mk = config.get("microkernel", {}) or {}

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
        _bus_constants.MODULE_SHUTDOWN,
        _bus_constants.KERNEL_EPOCH_TICK,
        _bus_constants.EPOCH_TICK,
        # SAVE_NOW added 2026-05-15 — hormonal_worker / neuromod_worker /
        # ns_worker durability invariant. Pre-fix, hormonal_worker only
        # saved its state to data/hormonal_state.json on graceful
        # MODULE_SHUTDOWN — process kills silently lost accumulated
        # hormone state across restarts. Hormonal_worker now saves
        # every 30s + on SAVE_NOW; subscribing here ensures B.1
        # shadow_swap orchestrator + manual checkpoint requests reach
        # the worker.
        _bus_constants.SAVE_NOW,
    ]
    # ns_module + hormonal_module gain extra broadcast subs per
    # rFP_phase_c_impulse_engine_d8_3_migration §3.A.1 + §3.B.7-B.8:
    #   - ns_module subscribes ACTION_RESULT (IMPULSE outcome learning)
    #   - hormonal_module subscribes HORMONE_STIMULUS (cross-worker bridge)
    # ACTION_RESULT is published with dst="all" (plugin.py:2879 verbatim);
    # HORMONE_STIMULUS is published with dst="hormonal_module" but
    # per-Titan broker routing depends on broadcast_topics filter.
    _NS_WORKER_BROADCAST_TOPICS = _STATE_WORKER_BROADCAST_TOPICS + [
        _bus_constants.ACTION_RESULT,
    ]
    # Note: NS-program urgencies flow cross-process via the
    # `ns_program_urgencies_input.bin` SHM slot (G18-pure per SPEC
    # §7.1 + D-SPEC-68 v1.13.0), NOT a bus event. ns_worker polls
    # the slot each tick — no ModuleSpec subscription needed.
    _HORMONAL_WORKER_BROADCAST_TOPICS = _STATE_WORKER_BROADCAST_TOPICS + [
        _bus_constants.HORMONE_STIMULUS,
        # expression_worker → hormonal_worker depletion-on-fire bridge
        # (2026-06-01): restores the consumption→refractory loop the Phase C
        # split severed. Published with dst="hormonal_module"; broker routing
        # needs the topic in this broadcast filter (same as HORMONE_STIMULUS).
        _bus_constants.HORMONE_CONSUME,
    ]
    if _mk.get("shm_ns_enabled", True):
        guardian.register(ModuleSpec(
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
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
            boot_priority="post_boot",
        ))
    if _mk.get("shm_neuromod_enabled", True):
        guardian.register(ModuleSpec(
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
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
            boot_priority="post_boot",
        ))
    if _mk.get("shm_hormonal_enabled", True):
        guardian.register(ModuleSpec(
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
            # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
            boot_priority="post_boot",
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
    guardian.register(ModuleSpec(
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
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # Language module (composition, teaching, vocabulary — higher cognitive)
    language_config = {
        **config.get("language", {}),
        "data_dir": config.get("memory_and_storage", {}).get("data_dir", "./data"),
    }
    # rFP_worker_broadcast_topics_completion §4.A.3 (Batch 3):
    # See legacy_core.py for full type list; mirror exactly here.
    guardian.register(ModuleSpec(
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
            _bus_constants.SPEAK_REQUEST, _bus_constants.LLM_TEACHER_RESPONSE,
            _bus_constants.META_LANGUAGE_RESULT, _bus_constants.MAKER_NARRATION_REQUEST,
            _bus_constants.CGN_DREAM_CONSOLIDATE, _bus_constants.CGN_CROSS_INSIGHT,
            _bus_constants.CGN_WEIGHTS_MAJOR, _bus_constants.CGN_KNOWLEDGE_RESP,
            _bus_constants.QUERY_RESPONSE, _bus_constants.SOCIAL_PERCEPTION,
            _bus_constants.CGN_SOCIAL_TRANSITION, _bus_constants.CGN_HAOV_VERIFY_REQ,
            _bus_constants.META_REASON_RESPONSE, _bus_constants.EPOCH_TICK,
            # Track 2 D2 (v1.3.1): WORD_PERTURBATION_HINT consumer for SPEAK
            # quality chain. Mirrors legacy_core.py registration. SPEC §8.5
            # D-SPEC-38.
            _bus_constants.WORD_PERTURBATION_HINT,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('llm'),
        ],
    ))

    # Meta-Reasoning Teacher (rFP_titan_meta_reasoning_teacher.md)
    # Philosopher-critic observing META_CHAIN_COMPLETE from spirit.
    # Per rFP §11 migration: deploys with enabled=false by default.
    # Worker starts (for observability + pre-load); critique logic is
    # gated on config['enabled'] inside the teacher's sampling check.
    _meta_teacher_cfg = config.get("meta_teacher", {})
    meta_teacher_config = {
        **_meta_teacher_cfg,
        "inference": config.get("inference", {}),
        "data_dir": config.get("memory_and_storage", {}).get(
            "data_dir", "./data"),
    }
    guardian.register(ModuleSpec(
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
        broadcast_topics=[_bus_constants.META_CHAIN_COMPLETE],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('llm'),
        ],
    ))

    # CGN Cognitive Kernel (shared V(s) + per-consumer Q(s,a) + HAOV + Sigma)
    cgn_config = {
        "state_dir": "data/cgn",
        "db_path": config.get("memory_and_storage", {}).get("data_dir", "./data") + "/inner_memory.db",
        "shm_path": "/dev/shm/cgn_live_weights.bin",
        "online_consolidation_every": 50,
        "shm_write_on_every_outcome": True,
        **config.get("cgn", {}),
    }
    # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4):
    # See legacy_core.py for full type list; mirror exactly here.
    guardian.register(ModuleSpec(
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
            _bus_constants.CGN_HAOV_VERIFY_RSP, _bus_constants.CGN_INFERENCE_REQ,
            _bus_constants.CGN_KNOWLEDGE_REQ,
            # Phase 8 fold-in (P8.Y): CGN_LEXICON_UPDATED — cgn_worker
            # is the SOLE emitter. Listed here so the bus broker doesn't
            # silently drop the topic. Payload: {ts, lexicon_size, snapshot_path}.
            _bus_constants.CGN_LEXICON_UPDATED,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # Knowledge Worker (4th CGN consumer — knowledge acquisition + Stealth Sage)
    _data_dir = config.get("memory_and_storage", {}).get("data_dir", "./data")
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
        **config.get("stealth_sage", {}),
        # Inference (for LLM distillation)
        **{k: v for k, v in config.get("inference", {}).items()
           if k.startswith("ollama_cloud") or k == "inference_provider"},
        # Twitter API (for X research path)
        "twitterapi_io_key": config.get(
            "twitter_social", {}).get("twitterapi_io_key", ""),
        # Knowledge Pipeline v2 (router/cache/health/budgets/alerts)
        **config.get("knowledge_pipeline", {}),
    }
    # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4):
    # See legacy_core.py for full type list; mirror exactly here.
    guardian.register(ModuleSpec(
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
            _bus_constants.CGN_KNOWLEDGE_REQ, _bus_constants.META_REASON_RESPONSE,
            _bus_constants.CGN_KNOWLEDGE_USAGE, _bus_constants.CGN_HAOV_VERIFY_REQ,
            _bus_constants.SEARCH_PIPELINE_BUDGET_RESET, _bus_constants.CGN_WEIGHTS_MAJOR,
            _bus_constants.CGN_CROSS_INSIGHT, _bus_constants.KNOWLEDGE_QUERY_CONCEPT,
            _bus_constants.KNOWLEDGE_SEARCH, _bus_constants.KNOWLEDGE_CONCEPTS_FOR_PERSON,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    # TimeChain — Proof of Thought memory chain
    timechain_config = {
        **config.get("timechain", {}),
    }
    # rFP_worker_broadcast_topics_completion §4.A.4 (Batch 4 — heaviest):
    # See legacy_core.py for full 23-type list; mirror exactly here.
    guardian.register(ModuleSpec(
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
            _bus_constants.SYSTEM_UPGRADE_QUEUED, _bus_constants.SYSTEM_UPGRADE_STARTING,
            _bus_constants.SYSTEM_RESUMED, _bus_constants.SYSTEM_UPGRADE_PENDING_DEFERRED,
            _bus_constants.SYSTEM_UPGRADE_THOUGHT,
            # Core timechain events (7)
            _bus_constants.TIMECHAIN_COMMIT, _bus_constants.EPOCH_TICK, _bus_constants.DREAM_STATE_CHANGED,
            _bus_constants.MEDITATION_COMPLETE, _bus_constants.EXPRESSION_FIRED,
            _bus_constants.TIMECHAIN_STATUS, _bus_constants.TIMECHAIN_QUERY,
            # Timechain query ops (5)
            _bus_constants.TIMECHAIN_RECALL, _bus_constants.TIMECHAIN_CHECK,
            _bus_constants.TIMECHAIN_COMPARE, _bus_constants.TIMECHAIN_AGGREGATE,
            _bus_constants.TIMECHAIN_SIMILAR,
            # Contract events (6)
            _bus_constants.CONTRACT_DEPLOY, _bus_constants.CONTRACT_LIST, _bus_constants.CONTRACT_STATUS,
            _bus_constants.CONTRACT_PROPOSE, _bus_constants.CONTRACT_APPROVE, _bus_constants.CONTRACT_VETO,
            # Phase 14 §9.B — timechain_guardian heal path: owning worker for
            # HEAL_REQUEST(action="reseed_primary_fork") → idempotent reseed.
            _bus_constants.HEAL_REQUEST,
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
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
    _spawn_ref = config.get("microkernel", {}).get(
        "spawn_reference_worker_enabled", False)
    guardian.register(ModuleSpec(
        name="backup",
        layer="L3",  # Microkernel v2 §A.5 — L3 pluggable (on-chain anchoring + 3-2-1 cold storage)
        entry_fn=backup_worker_main,
        config=config,  # full config — reads [backup]/[network]/[info_banner]/[mainnet_budget]/[memory_and_storage]
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
            _bus_constants.MEDITATION_COMPLETE,
            _bus_constants.BACKUP_TRIGGER_MANUAL,
        ],
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('timechain'),
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
    guardian.register(ModuleSpec(
        name="emot_cgn",
        layer="L2",  # Microkernel v2 §A.5 — L2 CGN consumer (emotional grounding)
        entry_fn=emot_cgn_worker_main,
        config={
            **config.get("emot_cgn", {}),
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
            _bus_constants.EMOT_CHAIN_EVIDENCE, _bus_constants.FELT_CLUSTER_UPDATE,
            _bus_constants.META_REASON_RESPONSE, _bus_constants.CGN_HAOV_VERIFY_REQ,
            _bus_constants.CGN_CROSS_INSIGHT, KIN_EMOT_STATE_MSG_TYPE,
            _bus_constants.HORMONE_FIRED,
            # _bus_constants.CGN_BETA_SNAPSHOT RETIRED v1.14.0 / D-SPEC-69 — flows
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
            # _bus_constants.TRAJECTORY_UPDATE RETIRED v1.14.0 / D-SPEC-69 — flows via
            # trajectory_state.bin SHM slot now (G18-pure per rFP_dead_dim_wiring_fix §2.E).
            _bus_constants.NS_URGENCIES_UPDATE,      # ns_worker → emot_cgn (NEW v1.9.5)
            _bus_constants.SPACE_TOPOLOGY_UPDATE,    # cognitive_worker → emot_cgn (NEW v1.9.5)
            _bus_constants.NEUROMOD_LEVELS_UPDATE,   # neuromod_worker → emot_cgn (NEW v1.9.5)
            _bus_constants.PI_PHASE_UPDATE,          # cognitive_worker → emot_cgn (NEW v1.9.5)
        ],
        start_method="spawn" if _spawn_grad else "fork",  # B.2.1 graduation
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="post_boot",
    ))

    logger.info("[TitanHCL] Registered %d supervised modules", len(guardian._modules))

# ------------------------------------------------------------------
# Microkernel v2 §A.4 (S5) — api_subprocess module registration
# ------------------------------------------------------------------


# ── api_subprocess registration ──
    # Phase 6 (D-SPEC-135 / v1.62.0): api is ALWAYS a separate Guardian-
    # supervised process — there is no "in-process uvicorn" path anymore
    # because there is no titan_hcl plugin process to host it in. The
    # legacy `microkernel.api_process_separation_enabled` flag check is
    # gone: under Phase 6, process separation is mandatory by SPEC
    # §11.B.4 INV-PROC-4+5 (titan_hcl_api owns its own PID + crash domain).

    # Phase 6 (D-SPEC-135 / v1.62.0): api entry now lives in
    # titan_hcl/api/api_main.py:entry which sets setproctitle('titan_hcl_api')
    # before delegating to the unchanged api_subprocess_main body (INV-PROC-1).
    from titan_hcl.api.api_main import entry as api_main_entry
    api_cfg = config.get("api", {})
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
        "microkernel": config.get("microkernel", {}),
        # network block carries vault_program_id, RPC URLs, premium_rpc
        # — used by vault PDA derivation + /health vault check.
        "network": config.get("network", {}),
        # mainnet/devnet flag and other env-related settings the api
        # endpoints may need (frontend mode toggles).
        "frontend": config.get("frontend", {}),
    }
    guardian.register(ModuleSpec(
        # Phase 6 / D-SPEC-135: bus subscriber name remains "api" so
        # OBSERVATORY_EVENT routing (dst="api") + CHAT_RESPONSE rid-routed
        # paths keep working unchanged. ps identity differentiation comes
        # from setproctitle('titan_hcl_api') inside api_main.entry (the
        # entry_fn below) per INV-PROC-1+4.
        name="api",
        entry_fn=api_main_entry,
        config=sub_config,
        # Phase 6 / D-SPEC-135 / INV-PROC-5: start_method="spawn" forces a
        # fresh interpreter so api crash isolation is preserved across the
        # python kernel-rs → guardian_hcl → titan_hcl_api process chain.
        # (fork would inherit titan_hcl's locked Condition state.)
        start_method="spawn",
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
        # Phase 11 §11.I.1 / D-SPEC-141 — api is now a kernel-rs peer
        # spawned via `scripts/titan_hcl_api.py` (INV-PROC-3 / INV-PROC-5,
        # independent crash domain from titan_hcl). The ModuleSpec stays
        # in the catalog so Supervisor.monitor_tick has the heartbeat /
        # rss / layer / restart_on_crash metadata to track it, and so
        # /v6/* readouts continue to enumerate it. autostart=False stops
        # this Orchestrator from spawning the api as a Guardian-supervised
        # child (which would race with the kernel-rs peer-spawn).
        autostart=False,
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
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.
        boot_priority="mandatory",
        # Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:
        dependencies=[
            _mod_dep('agno_worker'),
            _mod_dep('memory'),
        ],
    ))
    logger.info(
        "[guardian_hcl] titan_hcl_api registered as L3 module "
        "(name=titan_hcl_api, start_method=spawn, INV-PROC-1+4+5)")

    # ── Phase 11 §11.I.3 / Chunk 11H — wire per-module probes ────────
    # Attach probe_fn for the 10 heaviest workers per RFP §3H.2. Other
    # modules keep `probe_fn=None` → the worker-side handler returns
    # ProbeResult.ok_() per §11.I.2 trivial-pass contract. Done as a
    # post-registration mutation (rather than 10 inline `probe_fn=` kwarg
    # additions) so the matrix-driven application is a single audit unit
    # alongside the §3H.10 dep matrix above (Chunk 11G).
    #
    # Bodies are shell-pass in 11H (Chunk 11H scope); each worker's real
    # liveness body lands in 11I when the worker's recv-loop adopts
    # handle_module_probe_request + the worker exposes the module-level
    # sentinels the probe inspects.
    from titan_hcl.probes import PROBE_REGISTRY
    _probe_misses: list[str] = []
    for _probe_name, _probe_fn in PROBE_REGISTRY.items():
        info = guardian._modules.get(_probe_name)
        if info is None:
            _probe_misses.append(_probe_name)
            continue
        info.spec.probe_fn = _probe_fn
    if _probe_misses:
        logger.warning(
            "[guardian_hcl] Phase 11 §11.I.3 / 11H probe wiring: %d "
            "modules not registered in catalog (flag-gated off?) — "
            "skipping probe attachment: %s",
            len(_probe_misses), sorted(_probe_misses))
    logger.info(
        "[guardian_hcl] Phase 11 §11.I.3 / 11H probes wired on %d/10 "
        "heaviest workers (per RFP §3H.2)",
        len(PROBE_REGISTRY) - len(_probe_misses))
