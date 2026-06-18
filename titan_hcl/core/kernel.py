"""
titan_hcl/core/kernel.py — TitanKernel (L0 microkernel).

Owns the foundational infrastructure that never restarts:
  - TitanBus (DivineBus — unified IPC)
  - Guardian (module supervisor)
  - StateRegister (legacy in-process state buffer, read path)
  - RegistryBank (/dev/shm state registry framework — Microkernel v2 §A.2)
  - SovereignSoul + HybridNetworkClient (identity)
  - DiskHealthMonitor + BusHealthMonitor
  - Trinity/Neuromod/Epoch shm writers (Microkernel v2 §A.2)
  - Spirit-fast 70.47 Hz shm writer hook (Microkernel v2 §A.7 / §L1)

Does NOT contain (per rFP "What L0 Does NOT Contain"):
  - LLM calls (L3 — plugin)
  - Reasoning logic (L2 — plugin)
  - Database connections (handled by L1/L2/L3 workers in their own processes)
  - HTTP API (L3 — plugin; will fully separate in S5)
  - Heavy Python libraries (torch, faiss — those live in L1/L2 workers)

This class is paired with `titan_hcl.core.plugin.TitanHCL` which holds
the L2/L3 coordinator state (proxies, agency, observatory, agno, dream
inbox) and orchestrates the full boot sequence via `plugin.boot()`.

When `microkernel.kernel_plugin_split_enabled=false` (default), the legacy
`titan_hcl.legacy_core.TitanCore` monolith is used instead. The split
path is byte-behavior-equivalent to the legacy path.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §L0 + §A.1
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.1 + §3 (D1-D10)
  - titan-docs/sessions/SESSION_20260424_microkernel_phase_a_s1_s2_shipped.md
"""
import asyncio
import inspect
import hashlib
import logging
import os
import threading
import time
from typing import Optional

from titan_hcl.bus import (
    DISK_CRITICAL,
    DISK_EMERGENCY,
    DISK_RECOVERED,
    DISK_WARNING,
    MODULE_HEARTBEAT,
    SOLANA_BALANCE_UPDATED,
    DivineBus,
    make_msg,
)
# Phase 6 / D-SPEC-135 / v1.62.0: Guardian moved to a separate process
# (scripts/guardian_hcl.py). The kernel holds a thin bus client mirroring
# Guardian's public surface — see titan_hcl/guardian_hcl_client.py.
from titan_hcl.guardian_hcl_client import GuardianHCLClient
from titan_hcl.params import load_titan_params

logger = logging.getLogger(__name__)


# ── Microkernel v2 Phase C C-S7 — in-process subscriber names ───────
#
# When microkernel.l0_rust_enabled=true, titan-kernel-rs owns the bus
# broker and the Python plugin connects as a client. The Rust broker
# routes targeted messages (dst != "all") by name — workers publishing
# `{"dst": "guardian", ...}` are delivered ONLY to the subscriber
# registered under name="guardian".
#
# The plugin's in-process subscribers (Guardian, Core, Meditation,
# Sovereignty, Kernel, Plugin) are NOT separate processes — they are
# objects inside the plugin process. For them to receive targeted
# messages from remote workers under l0_rust=true, the plugin must
# register one BusSocketClient per name that workers address. Each
# client is a separate broker connection but lives in the same Python
# process; an inbound dispatcher thread per client drains its inbound
# queue and re-injects messages into the in-process bus via
# publish_in_process.
#
# Without this list, MODULE_HEARTBEAT / MODULE_READY (always
# dst="guardian") would never reach the Guardian and worker liveness
# detection breaks immediately under l0_rust mode.
#
# "titan_HCL" is the canonical plugin identity (also the outbound
# publisher attached to bus.publish() routing).
IN_PROCESS_SUBSCRIBER_NAMES: tuple[str, ...] = (
    "titan_HCL",     # canonical plugin identity + outbound publisher
    # "guardian" KEPT (D-SPEC-151, 2026-06-08) — the plugin's `_guardian_handler_loop`
    # (titan_hcl/core/plugin.py) consumes the ADMIN QUERY (restart/start/stop/
    # reload_module) on dst="guardian" via this alias; that is titan_hcl's job
    # (it owns spawn+restart per D-SPEC-146) and is the ORIGINAL hot-reload /
    # restart-module design. The fleet-wide MODULE_HEARTBEAT flood was NOT this
    # alias — it was the Orchestrator separately subscribing "guardian" and never
    # draining (its monitor_tick drain moved to guardian_hcl at the 2026-05-28
    # peer-spawn split). Fix = Orchestrator(subscribe_guardian=False) in
    # scripts/titan_hcl.py (the undrained queue is gone); _guardian_handler_loop
    # keeps draining its own "guardian" queue at 10 Hz (QUERY handled, heartbeats
    # discarded — original behavior, no accumulation). Liveness (heartbeat→restart
    # decisions) is guardian_hcl's OWN client; spawn-side EXECUTION (MODULE_*_REQUEST
    # incl. reload/adopt) routes to "guardian_hcl_lifecycle" (the Orchestrator alias).
    "guardian",      # ADMIN QUERY (restart/start/stop/reload_module) → _guardian_handler_loop
    "core",          # core-loop messages
    # "meditation" RETIRED v1.9.5 §X1 D-SPEC-64 (2026-05-16) — pre-§4.D this
    # alias on titan_HCL's connection routed dst="meditation" to the
    # in-process `_meditation_queue` consumed by plugin._meditation_loop.
    # Post-§4.D (SPEC v1.8.3 / D-SPEC-57, 2026-05-15) meditation became a
    # separate worker subprocess (`titan_hcl/modules/meditation_worker.py`)
    # with its OWN BusSocketClient primary name="meditation". Keeping the
    # stale alias caused:
    #   (1) Duplicate broker subscribers under "meditation" (parent alias +
    #       worker primary) — phantom deliveries to the parent that had no
    #       in-process consumer.
    #   (2) The echo-prevention filter at `_bus_client_inbound_dispatcher`
    #       below silently DROPPED every MODULE_HEARTBEAT / MODULE_READY
    #       published by meditation_worker (src="meditation") because
    #       `src in plugin_names` matched the stale alias — Guardian's
    #       `_modules["meditation"].last_heartbeat` was never updated,
    #       trapping the worker in `state=starting` forever post-boot.
    # Same closure applies to "sovereignty" below.
    # "sovereignty" RETIRED v1.9.5 §X1 D-SPEC-64 (2026-05-16) — pre-§4.L
    # this alias routed dst="sovereignty" to the in-process `_sovereignty_queue`
    # consumed by plugin._sovereignty_loop. Post-§4.L (SPEC v1.9.1 /
    # D-SPEC-60, 2026-05-15) sovereignty became a separate worker subprocess
    # (`titan_hcl/modules/sovereignty_worker.py`). Stale alias caused
    # the same echo-prevention drop class as meditation above. Retired
    # in the same commit.
    "kernel",        # kernel-internal RPC + broadcasts
)

# SPEC §9.B (titan_HCL block, lines 1155-1163) — enumerated bus subscriptions
# for the Python parent process's single broker connection. ALL kernel-internal
# subscribers (5 names after "titan_HCL" in IN_PROCESS_SUBSCRIBER_NAMES + 11
# proxy aliases in KERNEL_PROXY_ALIASES) share titan_HCL's ONE connection per
# SPEC §9.B (`guardian_HCL ... within titan_HCL's bus client`). The broker
# only fans broadcasts when `msg_type ∈ TITAN_HCL_BROADCAST_TOPICS`; aliases
# inherit the connection's filter per SPEC §8.2 v1.3.0 multi-name semantics.
#
# Drift between this list and SPEC §9.B is a CI failure (lockstep test
# `tests/test_titan_hcl_topics_matches_spec_92b.py` parses §9.B and asserts
# exact set equality). Adding a new in-process consumer requires updating
# both SPEC §9.B AND this constant in the same commit (per Rule 0 + the
# v1.4.0 D-SPEC-42 amendment closure).
#
# Closes the architectural regression caused by `_HIGH_RATE_BROADCAST_TYPES`
# stopgap retirement (rFP_worker_broadcast_topics_completion §4.C, 2026-05-12):
# pre-fix, 6 separate BusSocketClient connections each subscribed-all and
# the stopgap blocked high-rate types; post-stopgap, every broadcast WARN+
# dropped on 6 connections × every type. This list (12 topics) is the
# SPEC-correct opt-in declaration replacing subscribe-all.
TITAN_HCL_BROADCAST_TOPICS: tuple[str, ...] = (
    # REQUIRED (kernel + L2 substrate) per SPEC §9.B titan_HCL block:
    "KERNEL_EPOCH_TICK",                # drives reasoning chains, MSL, dreaming
    "KERNEL_SHUTDOWN_ANNOUNCE",         # graceful shutdown cascade
    "KERNEL_BOOT_GENERATION_CHANGED",   # authkey re-derive, module re-bootstrap
    "UNIFIED_SPIRIT_SELF_ASSEMBLED",    # SELF freshness for L2 consumers
    "BODY_STATE",                       # current L2 modules (CGN, EMOT-CGN, ...)
    "MIND_STATE",                       # current L2 modules
    "SPIRIT_STATE",                     # current L2 modules
    "SWAP_HANDOFF",                     # B.2.1 spawn-mode supervision transfer
    "SWAP_HANDOFF_CANCELED",            # B.2.1 transfer cancellation
    "ADOPTION_ACK",                     # response to spawn-mode ADOPTION_REQUEST
    # REQUIRED (agency action pipeline) — v1.31.0 D-SPEC-92.
    # Closes BUG-IMPULSE-PIPELINE-DEAD-PHASE-C: ns_module fires IMPULSE every
    # ~300s (cooldown working) but the parent's `_agency_loop` (subscriber
    # "agency", types=[IMPULSE, OUTER_DISPATCH, ...] at plugin.py:2452)
    # never received them because broker dropped IMPULSE for titan_HCL prior
    # to this list extension. Same gap killed AGENCY_STATS/ASSESSMENT_STATS/
    # AGENCY_READY proxy cache refresh from agency_worker.
    "IMPULSE",                          # ns_module → _agency_loop._handle_impulse
    "OUTER_DISPATCH",                   # outer_trinity → _handle_outer_dispatch (lockstep)
    "AGENCY_STATS",                     # agency_worker → AgencyProxy.update_cached_stats
    "ASSESSMENT_STATS",                 # agency_worker → AssessmentProxy.update_cached_stats
    "AGENCY_READY",                     # agency_worker boot → ExpressionTranslator helpers list
    # V4 frontend SSE event bridge — RELOCATED to observatory_worker 2026-05-21
    # (RFP_phase_c_titan_hcl_cleanup Phase A). The parent's "v4_bridge"
    # subscriber + its 7 SSE broadcast types (BIG_PULSE, GREAT_PULSE,
    # DREAM_STATE_CHANGED, DREAM_INBOX_REPLAY, NEUROMOD_UPDATE, HORMONE_FIRED,
    # EXPRESSION_FIRED) moved to the L3 observatory_worker, which now subscribes
    # them via its own ModuleSpec.broadcast_topics and translates to
    # OBSERVATORY_EVENT. DREAM_INBOX_REPLAY's chat re-emission moved to
    # agno_worker. titan_HCL no longer consumes any of these → dropped here.
    # The former OPTIONAL block (SPHERE_PULSE / SPHERE_EPOCH_TICK /
    # TRINITY_SUBSTRATE_GROUND_UP / TRINITY_SUBSTRATE_FILTER_DOWN) was also
    # dropped in v1.46.0 (D-SPEC-108): SPHERE_PULSE's only parent consumer was
    # the now-relocated v4_bridge (its live consumers — observatory_worker +
    # resonance/sphere_clock producers — are in Phase C workers/logic, not the
    # titan_HCL connection); the other three had ZERO producers/consumers
    # fleet-wide (vestigial). No parent in-process subscriber reads any of them.
    # REQUIRED (Trinity 130D dim stats cache) — v1.31.0 D-SPEC-92.
    # The parent's "trinity_dim_stats_cache" subscriber (plugin.py:3640,
    # types=_STATS_CACHE_EVENT_TYPES) caches rich producer state for dim
    # formulas in outer_mind / outer_spirit / inner_mind. All 10 *_STATS_UPDATED
    # broadcasts come from worker subprocesses; without these the dim cache
    # stays empty fleet-wide → dim formulas fall through to SPEC_DEFAULT for
    # any dim that reads a rich producer payload. Cascade gap: dim values
    # plateau on stale defaults regardless of producer state.
    "META_REASONING_STATS_UPDATED",     # meta_cgn block → outer_spirit dim formulas
    "LANGUAGE_STATS_UPDATED",           # vocab_total / composition_level → outer_mind
    "OUTPUT_VERIFIER_STATS",            # verified/rejected counts → outer_spirit
    "MEMORY_STATUS_UPDATED",            # persistent_count / mempool_size → outer_mind
    "MEMORY_KNOWLEDGE_GRAPH_UPDATED",   # KG node/edge counts → outer_mind
    "SOLANA_BALANCE_UPDATED",           # SOL balance → metabolic dim formulas
    "SOCIAL_STATS_UPDATED",             # persona social stats → outer_spirit
    "NEUROMOD_STATS_UPDATED",           # full neuromod state → multiple dim formulas
    "CGN_STATS_UPDATED",                # CGN grounded_density → outer_mind
    "SOCIAL_PERCEPTION_STATS_UPDATED",  # social sentiment_ema → inner_mind
)

# Kernel-side proxy reply-queue names that need broker-side fanout routing
# under Phase C / l0_rust_enabled=true. Each proxy subscribes in-process
# via bus.subscribe("<x>_proxy", reply_only=True) and uses bus.request_async()
# to send a QUERY to its paired worker. The worker publishes a RESPONSE with
# dst="<x>_proxy". For that RESPONSE to deliver, the Rust broker needs a
# subscription registered for the proxy name pointing to the parent Python
# process. We register all these names as ALIASES on the titan_HCL bus
# client's single connection via additive BUS_SUBSCRIBE (Rust broker
# multi-name extension shipped 2026-05-12, SPEC §8.2 v0.1.9). Closes
# BUG-PHASE-C-BUS-FANOUT-MULTI-NAME-20260512.
#
# Why aliases instead of separate connections: spawning one BusSocketClient
# per proxy name (the 2026-05-12 first-attempt fix) caused 17× broadcast
# amplification — every dst="all" broadcast was delivered to 17 inbound
# queues, only 1 of which (titan_HCL) actually relays. Under chat load,
# the proxy clients' dispatchers couldn't drain fast enough to catch the
# targeted RESPONSE in the 5s proxy timeout window. Aliases on a single
# connection eliminate the amplification while preserving broker-side
# routing for all 11 proxy names.
KERNEL_PROXY_ALIASES: tuple[str, ...] = (
    "output_verifier_proxy",   # OVG verify_and_sign / build_timechain_payload
    "agency_proxy",            # agency_worker dispatch_from_nervous_signals / assess
    "assessment_proxy",        # gatekeeper assessment work-RPC
    "media_proxy",             # media generation work-RPC
    "llm_proxy",               # LLM inference work-RPC
    "reflex_proxy",            # reflex arc execution work-RPC
    "timechain_proxy",         # timechain verify / anchor work-RPC
    "spirit_proxy",            # spirit-worker proxy work-RPC (legacy paths still active)
    "memory_proxy",            # memory recall / log work-RPC
    "mind_proxy",              # mind-worker proxy work-RPC
    "rl_proxy",                # IQL training queue + stats reply queue
    "social_graph_proxy",      # social_graph_worker proxy (v1.7.1, D-SPEC-50)
    "metabolism_proxy",        # metabolism_worker proxy (v1.7.2, D-SPEC-51)
    "meditation_proxy",        # meditation_worker proxy (v1.8.3, D-SPEC-57)
    "life_force_proxy",        # life_force_worker proxy (v1.8.4, D-SPEC-58)
    "studio_proxy",            # studio_worker proxy (v1.9.4, D-SPEC-63)
    "studio_render_proxy",     # studio_worker render-completion broadcast subscriber (v1.9.4, D-SPEC-63)
    "agno_proxy",              # agno_worker chat/chat_stream work-RPC (v1.17.0, D-SPEC-72)
)


# ── Microkernel v2 §A.4 (S5) — kernel RPC exposed methods ───────────
#
# Canonical list of dotted method paths the API subprocess can call via
# the kernel_rpc Unix-socket RPC. Derived from `arch_map api-status`
# audit of titan_hcl/api/{dashboard,chat,maker,webhook}.py for
# patterns matching `<plugin_var>.X.Y(...)` and `<plugin_var>.X` runtime
# access (vs `from titan_hcl.X import ...` module imports, which are
# direct in the API process and don't need RPC).
#
# Adding a new endpoint that needs a new plugin attribute → add it
# here. The drift-detection test (tests/test_kernel_rpc_exposed_methods.py
# — commit #14) statically analyzes the API code and asserts every
# `plugin.X` access pattern is in this set.
#
# Bus-backed proxies (body, mind, spirit, memory, rl, llm, media,
# timechain, gatekeeper, mood_engine, social_graph) work cross-process
# via the bus's existing mp.Queue mechanism — they don't go through
# kernel_rpc. Listed here for completeness because endpoint code reaches
# them via `plugin.X` paths; the RPC server resolves them as references
# and the proxy's bus.request() does the actual cross-process call.
#
# Module-import paths (plugin.core, plugin.logic, plugin.utils,
# plugin.api, plugin.expressive) are NOT here — they're imported
# directly in the API process via `from titan_hcl.X import ...`.
KERNEL_RPC_EXPOSED_METHODS: frozenset[str] = frozenset({
    # Kernel-owned (L0) — Soul, Guardian, Bus, network
    "soul",
    "soul.evolve_soul",
    "soul.get_active_directives",
    "soul.current_gen",
    "soul._maker_pubkey",
    "soul._nft_address",
    "guardian",
    "guardian.get_status",
    "guardian.get_modules_by_layer",
    "guardian.layer_stats",
    "guardian.start",
    "guardian.enable",
    "bus",
    "bus.publish",
    "bus.request",
    "bus.stats",
    # Microkernel v2 Phase B.1 — Shadow Core Swap
    # Top-level and "kernel.X" prefixed paths both exposed (api_subprocess
    # accesses via titan_state.kernel.shadow_swap_orchestrate which
    # generates kernel_rpc path "kernel.shadow_swap_orchestrate").
    "shadow_swap_orchestrate",
    "shadow_swap_status",
    "hibernate_runtime",
    "restore_from_snapshot",
    "kernel_version",
    "dump_heap",
    "dump_tracemalloc",
    "dump_thread_inventory",
    "kernel",
    "kernel.shadow_swap_orchestrate",
    "kernel.shadow_swap_status",
    "kernel.hibernate_runtime",
    "kernel.restore_from_snapshot",
    "kernel.kernel_version",
    "kernel.dump_heap",
    "kernel.dump_tracemalloc",
    "kernel.dump_thread_inventory",
    # Microkernel v2 Phase B.2.1 — broker stats for /v4/state.bus_broker
    # + orchestrator's adoption-wait check + arch_map bus-status.
    "bus_broker_stats",
    "kernel.bus_broker_stats",
    # Phase C C-S7 Component 2 (2026-05-05) — bus_health monitor read access
    # for /v4/bus-health and /health.bus_health under l0_rust_enabled=true.
    # Without these, the kernel_rpc proxy refuses bus_health attribute lookup
    # and Component 1's kernel_rpc fallback returns empty dict instead of the
    # real BusHealthMonitor.snapshot() data.
    "bus_health",
    "bus_health.snapshot",
    "bus_health.update_queue_depths",
    # Phase A retrofit (2026-04-27) — swap-aware proxy interlock.
    # Guardian.start() inside the kernel calls these directly (no RPC),
    # but api_subprocess endpoints / arch_map may also probe via RPC.
    "is_shadow_swap_active",
    "kernel.is_shadow_swap_active",
    "wait_for_swap_completion",
    "kernel.wait_for_swap_completion",
    "network",
    "network.get_balance",
    "network.get_raw_account_data",
    "network.premium_rpc",
    "network.pubkey",
    "network.rpc_urls",
    # Plugin-owned in-memory state
    "_full_config",
    "_full_config.get",
    "_limbo_mode",
    "_start_time",
    "_is_meditating",
    "_last_commit_signature",
    "_last_execution_mode",
    "_last_research_sources",
    "_current_user_id",
    "_pending_self_composed",
    "_pending_self_composed_confidence",
    # v1.8.2 (D-SPEC-56): `_dream_inbox` deque + `.clear` + `.extend` removed
    # from kernel_rpc allowlist — chat-during-dream buffering moved to
    # dream_state_worker via DREAM_INBOX_ENQUEUE bus events; the inbox queue
    # no longer lives on the plugin object.
    "_proxies",
    "_proxies.get",
    "_agency",
    "_agency.get_stats",
    "_agency_assessment",
    "_agency_assessment.get_stats",
    "_interface_advisor",
    "_interface_advisor.get_stats",
    "_gather_current_state",
    "_get_state_narrator",
    # Plugin methods
    "reload_api",
    "get",
    "get_v3_status",
    "run_chat",
    # Plugin module-attribute references
    "maker",
    "backup",
    "backup._last_personality_date",
    "backup._last_soul_date",
    "backup._meditation_count",
    "backup.get_latest_backup_record",
    "metabolism",
    "metabolism.get_directive_alignment",
    "metabolism.get_learning_velocity",
    "metabolism.get_metabolic_health",
    "metabolism.get_social_density",
    # 2026-04-30 — closes ASGI 500 errors on /v4/metabolism/evaluate-gate
    # observed at ~1.7/min post bus_ipc_socket+spawn_graduated flip. The
    # endpoint at dashboard.py:1217 was added 2026-04-28 (BUG-METABOLISM-
    # EVALUATE-GATE-500) but never registered the methods it calls in the
    # kernel_rpc EXPOSED_METHODS set. Worked in legacy in-process api mode
    # (direct attribute access on TitanHCL); fails through kernel_rpc
    # proxy when api_process_separation_enabled=true (production path).
    "metabolism.evaluate_gate",
    # MEDITATION-WORK-RPC-SYNC-AUDIT (2026-05-26) — async sibling exposed
    # for the FastAPI endpoint `metabolism_evaluate_gate()` in dashboard.py.
    # Per SPEC Preamble G19, async callers MUST use the async sibling;
    # the kernel_rpc proxy needs the method allowlisted same as the sync.
    "metabolism.evaluate_gate_async",
    "metabolism.get_metabolic_tier",
    "metabolism.get_gates_enforced",
    "metabolism.get_last_gate_decision_reason",
    "metabolism.get_gate_decision_summary",
    "studio",
    "config_loader",
    "params",
    "persistence",
    "social",
    # Bus-backed proxies (route via bus's own IPC, not kernel_rpc;
    # listed for getattr resolution from API endpoint code)
    "memory",
    "memory._cognee_ready",
    "memory._node_store",
    "memory._node_store.items",
    "memory.fetch_mempool",
    "memory.fetch_mempool_for_observatory",  # rFP_bus_payload_contracts §3.1
    "memory.fetch_social_metrics",
    "memory.get_coordinator",
    "memory.get_knowledge_graph",
    "memory.get_memory_status",
    "memory.get_neuromod_state",
    "memory.get_ns_state",
    "memory.get_persistent_count",
    "memory.get_reasoning_state",
    "memory.get_top_memories",
    "memory.get_top_memories_for_observatory",  # rFP_bus_payload_contracts §3.1
    "memory.get_topology",
    "memory.inject_memory",
    "mood_engine",
    "mood_engine.get_mood_label",
    "mood_engine.get_mood_valence",
    "mood_engine.previous_mood",
    "mood_engine.force_zen",
    "gatekeeper",
    # Phase 2.5.E (rFP_trinity_130d_phase2_5_closure §4 Chunk E) —
    # community attribution endpoint exposure. /v4/community-engagement-stats
    # in dashboard.py reads plugin._social_x_gateway_reader; under
    # api_process_separation_enabled=true this is a kernel_rpc proxy
    # access that requires the attr + method to be in EXPOSED_METHODS.
    # Without these the endpoint returns 503 MethodNotExposed fleet-wide
    # (T1+T2+T3) and outer_spirit ANANDA[6,8] read default 0.0 instead of
    # real community_engagement_stats. Maker-flagged 2026-05-12 — Phase
    # 2.5.E shipped 2026-05-08 but EXPOSED_METHODS wiring was missed.
    "_social_x_gateway_reader",
    "_social_x_gateway_reader.get_community_engagement_stats",
    # WebSocket EventBus (relocates to API subprocess in S5; kept here
    # for legacy in-process path compatibility and emit-from-kernel mirror)
    "event_bus",
    "event_bus.emit",
    "event_bus.subscriber_count",
})


# PERSISTENCE_BY_DESIGN: TitanKernel._config is runtime bootstrap state
# (loaded from config.toml + ~/.titan/secrets.toml); it is not self-owned
# mutable state to persist.
class TitanKernel:
    """
    L0 microkernel — foundational infrastructure that never restarts.

    Usage (via TitanHCL — see titan_hcl.core.plugin):
        kernel = TitanKernel(wallet_path)
        plugin = TitanHCL(kernel)
        await plugin.boot()  # orchestrates kernel.boot() + module wiring

    Kernel boot sequence (commit 2, next — see PLAN §4.1 Commit 2):
        await kernel.boot()
          └─ bus._poll_fn hookup, _guardian_loop task, _heartbeat_loop task,
             _start_spirit_shm_writer hook (trinity/topology Python writers
             retired — Rust daemons own those slots, config-shm Phase D)

    This commit (#1 — kernel skeleton) lands __init__ only. Boot + loops
    arrive in commit 2.
    """

    def __init__(self, wallet_path: str):
        self._boot_start = time.time()

        # ── Load config ──────────────────────────────────────────────
        self._config = self._load_full_config()

        # ── Divine Bus ───────────────────────────────────────────────
        self.bus = DivineBus(maxsize=10000)
        # Option B (2026-04-29): both subscribers are reply_only=True so
        # they're already excluded from dst="all" broadcasts; types=[]
        # is documentation that no broadcasts are expected here. Targeted
        # dst="core" and dst="meditation" msgs (RPC, MEDITATION_REQUEST,
        # etc.) bypass the filter and reach the queue normally.
        self._core_queue = self.bus.subscribe(
            "core", reply_only=True, types=[])
        # _meditation_queue pre-subscription RETIRED v1.8.3 §4.D (D-SPEC-57,
        # 2026-05-15) — meditation_worker subprocess subscribes to
        # MEDITATION_REQUEST via its own bus client. The "meditation" dst is
        # now owned by the worker.
        # _sovereignty_queue RETIRED v1.9.1 §4.L (D-SPEC-60, 2026-05-15) —
        # sovereignty_worker subprocess subscribes to SOVEREIGNTY_EPOCH +
        # SOVEREIGNTY_CONFIRM_MAKER + SOVEREIGNTY_INCREMENT_GREAT_CYCLE directly
        # via its own bus client. spirit_worker.py:3845 producer + dst="sovereignty"
        # routing unchanged; worker registration in plugin.py provides the
        # subscriber name match.

        # ── StateRegister (real-time state buffer) ──────────────────
        from titan_hcl.logic.state_register import StateRegister
        self.state_register = StateRegister()
        # Phase D (D-SPEC-116): the STATE_SNAPSHOT snapshot-publish loop (and its
        # spirit_enrichment.micro_tick_interval knob) was retired with
        # spirit_worker; StateRegister now only maintains the real-time buffer
        # the kernel reads for TITAN_SELF topology composition.
        self.state_register.start(self.bus)

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
        from titan_hcl.core.state_registry import RegistryBank, resolve_titan_id
        self._titan_id = resolve_titan_id()
        self.registry_bank = RegistryBank(
            titan_id=self._titan_id, config=self._config,
        )

        # ── Guardian (Phase 6 / D-SPEC-135 / v1.62.0) ────────────────
        # Pre-Phase-6: `self.guardian = Guardian(self.bus, config=...)` —
        # in-process L1 supervisor that constructed workers via mp.Process,
        # held a kernel back-reference for shadow-swap interlock, drained
        # worker send queues, and ran monitor_tick on the kernel event loop.
        # Phase 6: Guardian lives in a SEPARATE PROCESS (scripts/guardian_hcl.py)
        # spawned by titan-kernel-rs. The kernel holds a thin bus client that
        # forwards lifecycle mutations (start/stop/restart/reload) to
        # guardian_hcl via dst="guardian_hcl_lifecycle"/dst="guardian" and
        # serves status reads from a cache populated by MODULE_*+SUPERVISION_*
        # events. _kernel_ref no longer applies — cross-process swap interlock
        # would need a bus-mediated protocol; left as a no-op pending a
        # follow-on RFP. INV-PROC-2.
        self.guardian = GuardianHCLClient(self.bus)
        # Shadow-swap completion signaling — endpoints / proxies block on this
        # event during a swap and resume when orchestrator finishes (success
        # or rollback). Initial state: set (no swap in flight = "done").
        self._shadow_swap_lock = threading.Lock()
        self._shadow_swap_active: Optional[str] = None
        self._shadow_swap_progress: dict[str, dict] = {}
        self._shadow_swap_history: dict[str, dict] = {}
        self._shadow_swap_done_event = threading.Event()
        self._shadow_swap_done_event.set()

        # ── Disk Health Monitor ──────────────────────────────────────
        # Background thread publishing DISK_WARNING/CRITICAL/EMERGENCY on
        # edge-detected transitions. On EMERGENCY, triggers graceful
        # Guardian.stop_all() via shutdown_fn hook. Protects against the
        # 2026-04-14 disk-full cascade pattern.
        from titan_hcl.core.disk_health import DiskHealthMonitor

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
            # Graceful all-worker stop — Guardian's own cleanup path runs
            # on a worker thread (commit f19a354) so this cannot deadlock
            # the event loop.
            logger.error("[TitanKernel] Initiating graceful shutdown: %s", reason)
            try:
                self.guardian.stop_all(reason=reason)
            except Exception as e:
                logger.error("[TitanKernel] shutdown stop_all error: %s", e)

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
        from titan_hcl.core.bus_health import BusHealthMonitor, set_global_monitor

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
        self._wallet_path_raw = wallet_path
        resolved_wallet = self._resolve_wallet(wallet_path)
        if resolved_wallet is None:
            self._limbo_mode = True
            logger.warning("[TitanKernel] No keypair — LIMBO MODE")

        # Boot Soul (lightweight — just Ed25519 keys, no network calls)
        if not self._limbo_mode:
            from titan_hcl.core.soul import SovereignSoul
            from titan_hcl.core.network import HybridNetworkClient
            network_cfg = self._config.get("network", {})
            self.network = HybridNetworkClient(config=network_cfg)
            self.soul = SovereignSoul(resolved_wallet, self.network, config=network_cfg)
        else:
            self.network = None
            self.soul = None

        # Shared stop event for shm writer threads.
        self._shm_writer_stop_evt: Optional[threading.Event] = None
        # Microkernel v2 §A.4 (S5) — kernel_rpc server holder (set in
        # _start_kernel_rpc when api_process_separation_enabled flag is on).
        # _plugin_ref is set by TitanHCL.boot() before kernel.boot() runs
        # so the RPC server can resolve method paths against the plugin.
        self._rpc_server = None
        self._plugin_ref = None

        # Process uptime anchor — consumed by _heartbeat_loop. Distinct
        # from _boot_start (which times sync __init__ duration).
        self._start_time = time.time()

        boot_ms = (time.time() - self._boot_start) * 1000
        logger.info(
            "[TitanKernel] L0 sync init complete in %.0fms (titan_id=%s, limbo=%s)",
            boot_ms, self._titan_id, self._limbo_mode,
        )

    # ------------------------------------------------------------------
    # Boot (async) — L0-only, called by TitanHCL.boot() per D10
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """L0 async boot: bus poll hookup, guardian health loop, heartbeat,
        shm writer threads.

        Does NOT call guardian.start_all() — Plugin registers modules
        first, then calls `kernel.start_modules()`. Does NOT create the
        observatory app, event bus, or any L2/L3 loops — those are
        Plugin's responsibility.

        Called from TitanHCL.boot() per PLAN §3 D10 boot-order
        invariants.
        """
        # ── bus poll hook — RETIRED Phase 10K (rFP §3G) ──────────────
        # Pre-Phase-6: `bus._poll_fn = self.guardian.drain_send_queues`
        # drained worker→bus queues whenever the bus needed to dispatch a
        # pending proxy QUERY/RESPONSE. Under Phase 6 worker send queues live
        # in guardian_hcl process — the kernel has no queues to drain, and
        # GuardianHCLClient.drain_send_queues became a no-op. The hookup is now
        # removed entirely: `bus._poll_fn` defaults to None and the bus drain
        # loop (bus.py:1585 `if self._poll_fn:`) is None-guarded, so dropping it
        # is a behavior-preserving cleanup.

        loop = asyncio.get_event_loop()

        # ── L0 SHM publisher loop (Phase 6 — _guardian_loop carved) ──
        # Phase 6: Guardian.monitor_tick + drain_send_queues moved to
        # scripts/guardian_hcl.py. Guardian state publisher (G21 single
        # writer of guardian_state.bin) ALSO moved there. The kernel still
        # owns soul_state.bin + network_state.bin publication — that lives
        # in _l0_state_publish_loop now.
        loop.create_task(self._l0_state_publish_loop())

        # Kernel heartbeat publisher (every 10s)
        loop.create_task(self._heartbeat_loop())

        # Memory hygiene daemon (every 60s by default; gates on titan_params
        # [microkernel] memory_hygiene_interval_s — set to 0 to disable).
        loop.create_task(self._memory_hygiene_loop())

        # Microkernel v2 Phase A §A.2 — Trinity + topology shm writers.
        # Phase C/D: l0_rust is permanently true → titan-unified-spirit-rs owns
        # trinity_state.bin and Rust trinity-rs owns topology_30d.bin (G21
        # single-writer). The legacy Python writers (_start_trinity_shm_writer /
        # _start_topology_shm_writer) only ever ran on the dead l0_rust=false
        # path and were retired here (config-shm Phase D).
        logger.info(
            "[TitanKernel] trinity_shm_writer + topology_shm_writer skipped — "
            "Rust spirit/trinity daemons own trinity_state.bin + topology_30d.bin")

        # Microkernel v2 Phase A §A.7 — spirit-fast writer hook (no-op placeholder).
        # Actual 70.47 Hz writes happen inside spirit_worker subprocess (D7).
        # Under l0_rust_enabled=true, spirit_worker enters SHIM mode (no writes)
        # and titan-inner-spirit-rs owns inner_spirit_45d.bin. The hook itself
        # is just an INFO-log breadcrumb so we leave it called either way.
        self._start_spirit_shm_writer()

        # Microkernel v2 Phase A §A.2 part 2 (S4) — immutable identity
        # shm registry. One-shot write of titan_id + maker_pubkey +
        # kernel_instance_nonce. Stable within kernel lifetime; nonce
        # changes on every kernel restart (enables Phase B shadow-core
        # worker reattach detection per PLAN §2.4).
        self._write_identity_shm()

        # Microkernel v2 Phase A §A.4 (S5) — kernel_rpc Unix-socket server.
        # Listens on /tmp/titan_kernel_{titan_id}.sock; the API subprocess
        # (when api_process_separation_enabled flag is on) connects via
        # HMAC handshake and issues msgpack-framed RPC calls. Server runs
        # in a daemon thread so it doesn't block the async event loop.
        # No-op when flag off — legacy in-process API path stays active.
        self._start_kernel_rpc()

        # Microkernel v2 Phase B.2 — Unix-socket pub/sub broker (workers
        # connect from separate processes and survive kernel swaps).
        # Authkey derived from identity keypair via HKDF-SHA256 (no
        # persistent secret on disk; resurrection-safe). DivineBus.publish()
        # additionally fans out to broker subscribers. No-op when
        # microkernel.bus_ipc_socket_enabled=false (default).
        self._start_bus_socket_broker()

        # M1-H4 (2026-04-26) — periodic SOL balance fetch + publish.
        # DivineBus race is fixed (bus.py _lock + race tests pass), but a
        # SECOND issue surfaced: even with the bus race resolved, the
        # publisher's first publish during boot still destabilizes
        # api_subprocess uvicorn on T2/T3 (T1 survives but Recv-Q
        # accumulates). Root cause not yet pinned — could be
        # multiprocessing.Queue.put racing with subprocess recv, or the
        # solana SDK's requests connection pool init colliding with
        # uvicorn startup. Default OFF until next session implements
        # delayed-first-publish (wait for Guardian api READY signal).
        # rFP §3.5 — delayed-first-publish ships in the publisher itself
        # (kernel._start_balance_publisher waits balance_publisher_first_delay_s
        # before the first emit). The flag stays opt-in for one rollout cycle:
        # flip on T1 first, soak 24h, then enable on T2/T3. Until the OBS gate
        # passes, default = OFF.
        if self._config.get("microkernel", {}).get(
                "balance_publisher_enabled", False):
            self._start_balance_publisher()
        else:
            logger.info(
                "[BalancePublisher] disabled — set "
                "microkernel.balance_publisher_enabled=true to enable "
                "(delayed-first-publish fix is in place)")

        # BUG-VAULT-COMMITS-NOT-LANDING (2026-04-29) — bus-bridge for
        # vault commits. memory_worker subprocess runs MeditationEpoch with
        # network_client=None (deployer keypair stays in main process for
        # security); on-chain TX submission is delegated to this loop. See
        # ANCHOR_REQUEST docstring in titan_hcl/bus.py for the wire
        # contract. The loop is no-op in limbo mode (no keypair) — request
        # gets a clean error response so memory_worker falls back to
        # MEDITATION_LOCAL signature.
        loop.create_task(self._anchor_request_loop())

        logger.info("[TitanKernel] Async boot complete — L0 loops running")

    def start_modules(self) -> None:
        """No-op under Phase 6 (D-SPEC-135).

        Pre-Phase-6: this called `guardian.start_all()` which spawned every
        autostart=True ModuleSpec via mp.Process inside the kernel process.
        Phase 6 moves the catalog + start_all() to guardian_hcl process —
        that process autostarts modules itself at its own boot (scripts/
        guardian_hcl.py:run). Plugin.boot() still calls start_modules()
        for surface compatibility; we keep it as a no-op so the boot
        sequence comment in plugin.py stays load-bearing.
        """
        logger.info(
            "[TitanKernel] start_modules() is a no-op under Phase 6 — "
            "guardian_hcl owns autostart.")

    async def shutdown(self, reason: str = "shutdown") -> None:
        """Graceful L0 stop — signal shm writer threads, stop disk health,
        stop all supervised modules.

        Safe to call repeatedly; each subsystem's stop path is idempotent.
        """
        logger.info("[TitanKernel] Shutdown initiated: %s", reason)
        if self._shm_writer_stop_evt is not None:
            self._shm_writer_stop_evt.set()
        # SPEC §8.0.ter — flush kernel's outbound bus buffer BEFORE
        # stopping the broker + clients. publish() returns after enqueue
        # per §8.0.ter; without an explicit flush, final shutdown
        # publishes (MODULE_SHUTDOWN, last heartbeats, swan-song state
        # snapshots) sitting in the writer's buffer would be lost when
        # _stop_bus_socket_clients() tears down the writer thread.
        # 2s budget is generous — healthy steady-state buffer drains in
        # microseconds; this only delays shutdown if the broker is
        # currently backpressured.
        try:
            self.bus.flush(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[TitanKernel] bus.flush during shutdown raised: %s "
                "— proceeding (final frames may be lost)", e)
        # Microkernel v2 §A.4 (S5) — stop kernel_rpc server (idempotent).
        self._stop_kernel_rpc()
        # Microkernel v2 Phase B.2 — stop bus_socket broker (idempotent).
        self._stop_bus_socket_broker()
        # Phase C C-S7 — stop bus_socket clients (idempotent).
        self._stop_bus_socket_clients()
        # M1-H4 — stop balance publisher.
        self._stop_balance_publisher()
        try:
            self.disk_health.stop()
        except Exception as e:
            logger.warning("[TitanKernel] disk_health.stop error: %s", e)
        # RFP_supervision_lifecycle §7.D — graceful drain. stop_all() runs each
        # module's save_first stop (WORKER_SHUTDOWN_GRACE_S per SPEC §18.4); it
        # returns the stopped count. NOTE: clean-shutdown CONFIRMATION for the
        # restart gate is NOT a marker written here — the systemd-supervised
        # process is the RUST kernel (titan-kernel-rs), so this Python path is not
        # the SIGTERM handler. The canonical clean-vs-forced signal is systemd's
        # exit Result/ExecMainStatus per SPEC §11.B (0/143=clean, 137=SIGKILL),
        # which the manage script checks post-stop (titan_common.sh
        # _verify_graceful_shutdown). See RFP §7.D.
        try:
            self.guardian.stop_all(reason=reason)
        except Exception as e:
            logger.warning("[TitanKernel] guardian.stop_all error: %s", e)

    # ------------------------------------------------------------------
    # Private L0 loops (event-loop tasks)
    # ------------------------------------------------------------------

    async def _l0_state_publish_loop(self) -> None:
        """Publish kernel-owned L0 SHM state slots at 1 Hz.

        Phase 6 (D-SPEC-135, v1.62.0): the legacy _guardian_loop was carved.
        Guardian.monitor_tick + drain_send_queues + guardian_state.bin
        publication moved to scripts/guardian_hcl.py. The kernel still owns
        soul_state.bin + network_state.bin (parent-process secrets and RPC
        state) — those publish here on the same 1 Hz cadence.

        G21 single-writer is preserved: only this loop writes soul_state.bin
        + network_state.bin; only guardian_hcl writes guardian_state.bin.
        """
        soul_pub = None
        network_pub = None
        try:
            from titan_hcl.logic.soul_state_publisher import SoulStatePublisher
            from titan_hcl.logic.network_state_publisher import NetworkStatePublisher
            from titan_hcl.core.state_registry import resolve_titan_id as _resolve_tid_a4_k
            _a4_tid_k = _resolve_tid_a4_k()
            soul_pub = SoulStatePublisher(titan_id=_a4_tid_k)
            network_pub = NetworkStatePublisher(titan_id=_a4_tid_k)
            soul_pub.publish(self.soul)  # cold-boot first publish
            network_pub.publish(self.network)
            logger.info(
                "[TitanKernel] Phase 6 L0 SHM publishers attached: "
                "soul_state + network_state (G21 single-writer). "
                "guardian_state.bin owned by guardian_hcl process.")
        except Exception as _err:
            logger.warning(
                "[TitanKernel] L0 SHM publisher init failed: %s — "
                "api_subprocess will read cold-boot stubs from those slots",
                _err)

        while True:
            try:
                if soul_pub is not None:
                    soul_pub.publish(self.soul)
                if network_pub is not None:
                    network_pub.publish(self.network)
            except Exception as e:
                logger.error("[TitanKernel] L0 state publish error: %s", e)
            await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Publish kernel heartbeat to the bus (every 10s)."""
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

    async def _memory_hygiene_loop(self) -> None:
        """Periodic gc.collect() + glibc malloc_trim(0) — keeps process RSS
        bounded by reclaiming allocator memory back to the OS.

        2026-05-01 — interim measure ahead of Phase C C-S7 (Rust kernel
        ownership swap, expected to deliver order-of-magnitude RSS reduction).
        Python's allocator + glibc arenas (even with MALLOC_ARENA_MAX=2) tend
        to hold freed memory in process address space rather than returning
        it to the OS. Under sustained request load this manifests as RSS
        appearing to "leak" — actually transient spikes that don't decay back
        to baseline cleanly. `gc.collect()` runs cyclic GC; `malloc_trim(0)`
        forces glibc to release fully-freed arena pages.

        Runs every `[microkernel] memory_hygiene_interval_s` seconds (default
        60s). Set to 0 in titan_params.toml to disable. Per-cycle cost: ~10-50ms
        depending on heap size.
        """
        interval_s = float(self._config.get("microkernel", {}).get(
            "memory_hygiene_interval_s", 60.0))
        if interval_s <= 0:
            logger.info("[MemHygiene] disabled (interval_s=%s)", interval_s)
            return
        logger.info("[MemHygiene] starting (interval_s=%.1f)", interval_s)
        # Lazy-load libc so the kernel boot path does not pay the cost on
        # systems where libc.so.6 is absent (containers, alpine, tests).
        libc = None
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError as e:
            logger.warning("[MemHygiene] libc.so.6 unavailable (%s) — "
                           "running gc.collect() only", e)
        while True:
            await asyncio.sleep(interval_s)
            try:
                import gc
                t0 = time.time()
                n_collected = gc.collect()
                t1 = time.time()
                trim_result = -1
                if libc is not None:
                    try:
                        trim_result = libc.malloc_trim(0)
                    except Exception as trim_err:  # noqa: BLE001
                        logger.debug(
                            "[MemHygiene] malloc_trim failed: %s", trim_err)
                t2 = time.time()
                logger.info(
                    "[MemHygiene] gc=%d freed (%.1fms) trim=%d (%.1fms)",
                    n_collected, (t1 - t0) * 1000,
                    trim_result, (t2 - t1) * 1000)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemHygiene] cycle failed: %s", e)

    # ------------------------------------------------------------------
    # Vault anchor bus-bridge (BUG-VAULT-COMMITS-NOT-LANDING — 2026-04-29)
    # ------------------------------------------------------------------

    async def _anchor_request_loop(self) -> None:
        """Listen for ANCHOR_REQUEST from memory_worker and submit on-chain TX.

        Microkernel v2's memory_worker subprocess runs
        ``MeditationEpoch(network_client=None)`` so meditation cycles cannot
        sign + submit TXes themselves. This loop is the bridge: memory_worker
        emits ``ANCHOR_REQUEST`` with the meditation's state_root + payload,
        kernel builds the vault commit instructions and submits via
        ``self.network`` (which holds the deployer keypair), and replies via
        ``bus.RESPONSE`` matched on the request's ``rid``.

        The loop subscribes ``reply_only=True`` so it does not receive
        broadcast state messages — only targeted ANCHOR_REQUEST frames.

        Wire contract documented at ``titan_hcl/bus.py`` near
        ``ANCHOR_REQUEST``.
        """
        from queue import Empty
        from titan_hcl.bus import ANCHOR_REQUEST

        queue = self.bus.subscribe("kernel", reply_only=True)
        logger.info("[TitanKernel] Anchor request loop subscribed (kernel queue)")

        while True:
            try:
                msg = await asyncio.to_thread(queue.get, True, 1.0)
            except Empty:
                continue
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                break
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("[TitanKernel] anchor queue read error: %s", e)
                await asyncio.sleep(1.0)
                continue

            if msg.get("type") != ANCHOR_REQUEST:
                # Stale RESPONSE landing in our queue — ignore.
                continue

            try:
                await self._handle_anchor_request(msg)
            except Exception as e:
                logger.warning(
                    "[TitanKernel] anchor handler error: %s", e, exc_info=True,
                )
                # Best-effort error response so memory_worker doesn't deadlock
                # the meditation cycle waiting on a reply.
                self._publish_anchor_response(
                    msg.get("src", "memory"), msg.get("rid"), None,
                    f"handler_exception: {type(e).__name__}",
                )

    async def _handle_anchor_request(self, msg: dict) -> None:
        """Build vault commit instructions, submit TX, reply with signature."""
        # rFP_meditation_worker_latency Option 1 instrumentation:
        # capture per-phase timing to pinpoint the kernel-side gap between
        # ANCHOR_REQUEST emit (worker) and "[Network] Sending tx" (kernel).
        _t_entry = time.time()
        _ts_emit = float(msg.get("ts", 0.0) or (msg.get("payload", {}) or {}).get("ts", _t_entry))
        _bus_dispatch_ms = (_t_entry - _ts_emit) * 1000.0 if _ts_emit > 0 else -1.0
        payload = msg.get("payload", {}) or {}
        src = msg.get("src", "memory")
        rid = msg.get("rid")
        state_root = payload.get("state_root", "") or ""
        payload_json = payload.get("payload", "") or ""
        promoted_count = int(payload.get("promoted_count", 0) or 0)
        logger.info(
            "[TitanKernel] [LAT] anchor_handler entry: rid=%s bus_dispatch=%.1fms "
            "promoted=%d state_root=%s",
            (rid or "")[:8], _bus_dispatch_ms, promoted_count, state_root[:16],
        )

        # Limbo / no-keypair guard. Reply with explicit error so
        # memory_worker falls back to MEDITATION_LOCAL.
        if self._limbo_mode or self.network is None:
            logger.info(
                "[TitanKernel] Anchor request received in limbo mode — "
                "no TX submission (state_root=%s, promoted=%d)",
                state_root[:16], promoted_count,
            )
            self._publish_anchor_response(src, rid, None, "limbo_mode_no_network")
            return

        vault_program_id = self._config.get("network", {}).get(
            "vault_program_id", "")
        if not vault_program_id:
            logger.info(
                "[TitanKernel] Anchor request: no vault_program_id configured "
                "— skipping TX (state_root=%s)", state_root[:16],
            )
            self._publish_anchor_response(src, rid, None, "no_vault_program_id")
            return

        # Lazy-init MeditationEpoch helper for instruction-building only.
        # ``memory_graph=None`` is safe — _build_commit_instructions only
        # uses self.network.pubkey + self._vault_program_id + 3 helpers
        # (_get_timechain_merkle / _get_vault_latest_root /
        # _compute_sovereignty_bp) which read disk + RPC, never self.memory.
        helper = getattr(self, "_anchor_helper", None)
        if helper is None:
            from titan_hcl.logic.meditation import MeditationEpoch
            helper = MeditationEpoch(
                memory_graph=None,
                network_client=self.network,
                config=self._config.get("inference", {}) or {},
            )
            helper._vault_program_id = vault_program_id
            self._anchor_helper = helper

        # Build instructions off the event loop — sync DB reads + sync httpx.
        # rFP_meditation_worker_latency Option 1: time the to_thread queue +
        # build phase separately, so we know whether the gap is executor
        # saturation vs. _build_commit_instructions itself.
        _t_pre_build = time.time()
        logger.info(
            "[TitanKernel] [LAT] anchor pre-build: rid=%s pre_build_setup=%.3fs",
            (rid or "")[:8], _t_pre_build - _t_entry,
        )
        try:
            instructions = await asyncio.to_thread(
                helper._build_commit_instructions, state_root, payload_json,
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Anchor _build_commit_instructions failed: %s", e,
            )
            self._publish_anchor_response(
                src, rid, None, f"build_failed: {type(e).__name__}",
            )
            return
        _t_post_build = time.time()
        logger.info(
            "[TitanKernel] [LAT] anchor post-build: rid=%s build_total=%.3fs "
            "(includes to_thread queue + _build_commit_instructions)",
            (rid or "")[:8], _t_post_build - _t_pre_build,
        )

        if not instructions:
            logger.info(
                "[TitanKernel] Anchor build returned no instructions — "
                "skipping TX (state_root=%s, promoted=%d)",
                state_root[:16], promoted_count,
            )
            self._publish_anchor_response(src, rid, None, "no_instructions")
            return

        # Submit TX. send_sovereign_transaction is async + handles priority
        # fee, retries, and budget enforcement (network._check_budget_exceeded).
        try:
            tx_signature = await self.network.send_sovereign_transaction(
                instructions, priority="HIGH",
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Anchor send_sovereign_transaction failed: %s", e,
            )
            self._publish_anchor_response(
                src, rid, None, f"send_failed: {type(e).__name__}",
            )
            return

        if tx_signature:
            logger.info(
                "[TitanKernel] Vault anchor TX landed (sig=%s, promoted=%d, "
                "root=%s)",
                tx_signature[:16], promoted_count, state_root[:16],
            )
            self._publish_anchor_response(src, rid, tx_signature, None)
        else:
            # send_sovereign_transaction returned None — likely budget
            # exceeded or RPC outage. Both already logged inside network.py.
            logger.warning(
                "[TitanKernel] Vault anchor TX returned None signature "
                "(budget exceeded? RPC down?)",
            )
            self._publish_anchor_response(src, rid, None, "tx_returned_none")

    def _publish_anchor_response(
        self,
        dst: str,
        rid: Optional[str],
        tx_signature: Optional[str],
        error: Optional[str],
    ) -> None:
        """Publish ANCHOR response (bus.RESPONSE matched on rid)."""
        from titan_hcl.bus import RESPONSE
        self.bus.publish(make_msg(
            RESPONSE, "kernel", dst,
            {"tx_signature": tx_signature, "error": error},
            rid=rid,
        ))

    # ------------------------------------------------------------------
    # Shm writer threads (Microkernel v2 §A.2 + §A.7)
    # ------------------------------------------------------------------

    def _start_spirit_shm_writer(self) -> None:
        """Microkernel v2 Phase A §A.7 — spirit-fast shm writer hook (S3b).

        Placeholder method for architectural symmetry with
        _start_trinity_shm_writer. The ACTUAL 70.47 Hz write happens
        inside spirit_worker subprocess (PLAN D7 — 70 Hz bus traffic
        would flood the bus; 45D tensor is already computed in-process
        at spirit_worker.py:2036 via collect_spirit_45d()).

        This method exists so:
          1. Kernel boot logs reflect all active shm paths (visibility).
          2. Future refactors or flag flips have a single kernel-side
             hook to attach to (e.g., to aggregate spirit-fast seq metrics
             in /v4/kernel-status).
          3. Symmetry: Trinity/Neuromod/Epoch writers all announced at
             kernel boot; spirit-fast follows the same shape.

        Emits a single INFO log on boot noting the flag state for the
        current kernel process.
        """
        # Phase C/D: l0_rust permanently true → spirit_worker is a SHIM and
        # Rust titan-inner-spirit-rs owns inner_spirit_45d.bin. The legacy
        # spirit_worker-owned branch was retired (config-shm Phase D).
        logger.info(
            "[TitanKernel] Spirit-fast shm writer: SHIM mode "
            "(spirit_worker no-ops; Rust titan-inner-spirit-rs owns "
            "inner_spirit_45d.bin) — l0_rust_enabled=true"
        )

    def _write_identity_shm(self) -> None:
        """Microkernel v2 Phase A §A.2 part 2 (S4) — immutable identity shm.

        Writes [titan_id:32 | maker_pubkey:32 | kernel_instance_nonce:32]
        to /dev/shm/titan_{id}/identity.bin exactly once at kernel boot.

        - titan_id + maker_pubkey are stable across kernel restarts.
        - kernel_instance_nonce is random per boot (secrets.token_bytes).
          Enables (a) worker reattach detection for Phase B shadow-core
          swap, (b) external monitoring distinguishing "same Titan, new
          kernel instance" from "same running kernel", (c) cross-process
          consistency checks for child processes.

        Feature-flag gated. No-op if shm_identity_enabled=false.

        Per PLAN §2.4: nonce never persists to disk — ephemeral-per-kernel
        by design. self._kernel_instance_nonce is set as a side effect for
        future API exposure (/v4/kernel-status can surface the nonce).
        """
        import secrets
        import numpy as _np

        from titan_hcl.core.state_registry import IDENTITY

        if not self.registry_bank.is_enabled(IDENTITY):
            logger.info(
                "[TitanKernel] Identity shm writer skipped "
                "(microkernel.shm_identity_enabled=False)")
            return

        # 32B titan_id (UTF-8, NUL-padded)
        tid_bytes = self.titan_id.encode("utf-8")[:32]
        tid_bytes = tid_bytes + b"\x00" * (32 - len(tid_bytes))

        # 32B maker_pubkey (Ed25519 raw; zero-filled if no maker_pubkey)
        mk_bytes = b"\x00" * 32
        if self.soul is not None:
            try:
                maker_pk = getattr(self.soul, "_maker_pubkey", None)
                if maker_pk is not None:
                    raw = bytes(maker_pk)
                    mk_bytes = raw[:32] + b"\x00" * max(0, 32 - len(raw))
            except Exception as e:
                logger.warning(
                    "[TitanKernel] maker_pubkey serialization failed: %s", e)

        # 32B kernel instance nonce — random per boot
        self._kernel_instance_nonce = secrets.token_bytes(32)

        payload = tid_bytes + mk_bytes + self._kernel_instance_nonce  # 96B
        arr = _np.frombuffer(payload, dtype=_np.uint8)
        try:
            self.registry_bank.writer(IDENTITY).write(arr)
            logger.info(
                "[TitanKernel] Identity shm written "
                "(titan_id=%s, maker_pubkey=%s..., kernel_nonce=%s...)",
                self.titan_id,
                mk_bytes[:4].hex(),
                self._kernel_instance_nonce[:4].hex(),
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Identity shm write failed: %s", e, exc_info=True)

    def _start_kernel_rpc(self) -> None:
        """Microkernel v2 Phase A §A.4 (S5) — kernel_rpc Unix-socket server.

        Listens on /tmp/titan_kernel_{titan_id}.sock for the API subprocess.
        Per-boot 32-byte authkey at /tmp/titan_kernel_{titan_id}.authkey.
        HMAC-SHA256 challenge-response on connect; msgpack-framed
        request/response thereafter.

        Feature-flag gated. No-op when api_process_separation_enabled=false
        (legacy in-process API path stays active).

        Server runs in a daemon thread (KernelRPCServer.serve_forever) so
        the async event loop is never blocked by accept()/recv() calls.

        Sets self._rpc_server (for stop_kernel_rpc) and self._plugin_ref
        (set by TitanHCL.boot() before this method runs).
        """
        if not self._config.get("microkernel", {}).get(
                "api_process_separation_enabled", False):
            logger.info(
                "[TitanKernel] kernel_rpc skipped "
                "(microkernel.api_process_separation_enabled=False)")
            return

        if not getattr(self, "_plugin_ref", None):
            logger.warning(
                "[TitanKernel] kernel_rpc cannot start — _plugin_ref not set "
                "(TitanHCL.boot() must assign self.kernel._plugin_ref before "
                "calling kernel.boot())")
            return

        from titan_hcl.core.kernel_rpc import KernelRPCServer
        try:
            # Capture the kernel's running asyncio loop so the RPC server
            # can await coroutine results (e.g. network.get_balance) on it
            # rather than spinning up a fresh loop per call.
            try:
                kernel_loop = asyncio.get_running_loop()
            except RuntimeError:
                kernel_loop = None
            self._rpc_server = KernelRPCServer(
                plugin_ref=self._plugin_ref,
                titan_id=self.titan_id,
                exposed_methods=KERNEL_RPC_EXPOSED_METHODS,
                kernel_loop=kernel_loop,
            )
            t = threading.Thread(
                target=self._rpc_server.serve_forever,
                daemon=True,
                name="kernel-rpc-server",
            )
            t.start()
            logger.info(
                "[TitanKernel] kernel_rpc server started "
                "(socket=%s, %d exposed methods)",
                self._rpc_server.sock_path,
                len(KERNEL_RPC_EXPOSED_METHODS))
        except Exception as e:
            logger.warning(
                "[TitanKernel] kernel_rpc start failed: %s", e, exc_info=True)
            self._rpc_server = None

    def _stop_kernel_rpc(self) -> None:
        """Graceful kernel_rpc shutdown — called from kernel.shutdown."""
        rpc = getattr(self, "_rpc_server", None)
        if rpc is not None:
            try:
                rpc.stop()
            except Exception as e:
                logger.warning(
                    "[TitanKernel] kernel_rpc stop error: %s", e)

    # ── Microkernel v2 Phase B.2 — Bus IPC broker ────────────────────────

    def _start_bus_socket_broker(self) -> None:
        """Start the bus client connections to the Rust kernel-rs broker.

        Under fleet-wide Phase C (since 2026-05-14), titan-kernel-rs owns
        the bus broker. Python plugin connects as one BusSocketClient per
        IN_PROCESS_SUBSCRIBER_NAMES entry so that targeted messages
        (dst="guardian" etc.) delivered by the Rust broker by name reach
        the plugin's in-process subscribers via the per-client inbound
        dispatcher.

        Per D8-1 retirement (2026-05-16): the Python `BusSocketServer`
        legacy Phase B.2 broker (~937 LOC at `bus_socket.py:460-1371`) +
        its `microkernel.bus_ipc_socket_enabled` flag-gated boot block
        here are DELETED. Verified dead via `ss -xlp` (kernel-rs binds
        `/tmp/titan_bus_<id>.sock` fleet-wide). The legacy Python broker
        was never started under Phase C anyway.
        """
        self._start_bus_socket_clients()

    def is_shadow_swap_active(self) -> bool:
        """Phase A retrofit (2026-04-27): True iff a shadow swap is in flight.

        Read by Guardian.start() to defer proxy lazy-starts during a swap
        (prevents mid-swap worker resurrection that holds DB locks). Also
        consulted by /maker/upgrade-status, arch_map, and any caller that
        wants to render swap state.

        Thread-safe via _shadow_swap_lock.
        """
        with self._shadow_swap_lock:
            return self._shadow_swap_active is not None

    def wait_for_swap_completion(self, timeout: float = 60.0) -> bool:
        """Phase A retrofit (2026-04-27): block until any in-flight swap
        completes (success OR rollback OR error).

        Returns True immediately if no swap is active. Returns True when
        the orchestrator finishes within `timeout` seconds. Returns False
        on timeout.

        Used by Guardian.start() so proxy lazy-start threads block during
        a swap window — they wake up automatically when the swap settles
        and proceed against whichever kernel won. Autonomous: no exception
        thrown, no user retry needed.

        Thread-safety: returns immediately if no swap (no lock-wait).
        Otherwise waits on a threading.Event signaled by _run_swap finally.
        """
        if not self.is_shadow_swap_active():
            return True
        return self._shadow_swap_done_event.wait(timeout=timeout)

    def bus_broker_stats(self) -> Optional[dict]:
        """Phase B.2.1 — return broker.stats() or None if no broker running.

        Exposed via KERNEL_RPC_EXPOSED_METHODS so api_subprocess (separate
        process per S5) can read it through the kernel_rpc proxy. The
        dashboard calls this for /v4/state.bus_broker; orchestrator's
        adoption-wait probes guardian.get_status() for the adopted set
        (see C5) and uses bus_broker stats for subscriber-count gating
        in HealthCriteria.

        Returns:
            dict with sock_path + subscriber_count + subscribers list, or
            None when bus_ipc_socket_enabled=false / broker not running.
        """
        broker = getattr(self, "_bus_broker", None)
        if broker is None:
            return None
        try:
            return broker.stats()
        except Exception:  # noqa: BLE001
            return None

    def _stop_bus_socket_broker(self) -> None:
        """Graceful broker shutdown — called from kernel.shutdown."""
        broker = getattr(self, "_bus_broker", None)
        if broker is not None:
            try:
                self.bus.detach_broker()
            except Exception:  # noqa: BLE001
                pass
            try:
                broker.stop(timeout=2.0)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[TitanKernel] bus_socket broker stop error: %s", e)
            self._bus_broker = None
            # Clear env vars so any future fork/spawn after kernel shutdown
            # falls back to legacy mode rather than connecting to a closed sock
            from titan_hcl.core.worker_bus_bootstrap import (
                ENV_BUS_KEYPAIR_PATH,
                ENV_BUS_SOCKET_PATH,
                ENV_BUS_TITAN_ID,
            )
            for k in (ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH):
                os.environ.pop(k, None)

    # ── Phase C C-S7 — outbound clients (l0_rust_enabled=true mode) ─────

    def _start_bus_socket_clients(self) -> None:
        """Phase C C-S7: connect plugin to Rust broker per SPEC §9.B titan_HCL.

        Per SPEC §9.B (titan_HCL block + guardian_HCL "hosted in titan_HCL ...
        within titan_HCL's bus client"): the Python parent process opens
        EXACTLY ONE BusSocketClient to the Rust broker. All other kernel-
        internal subscriber names (guardian, core, meditation, sovereignty,
        kernel — the 5 non-titan_HCL entries of IN_PROCESS_SUBSCRIBER_NAMES)
        + all 11 proxy reply-queue names (KERNEL_PROXY_ALIASES) are
        registered as ADDITIVE ALIASES on titan_HCL's single connection,
        per SPEC §8.2 v1.3.0 multi-name semantics (D-SPEC-39).

        Subscriber intent declarations on the single connection:
          * Primary BUS_SUBSCRIBE: payload.name="titan_HCL",
            payload.topics=TITAN_HCL_BROADCAST_TOPICS (SPEC §9.B enumerated
            12-topic list), payload.reply_only=False.
          * 16 alias BUS_SUBSCRIBE frames (5 in-process + 11 proxy):
            payload.name=<alias>, payload.topics=[], payload.reply_only=
            inherit from primary connection (False).

        Closes 2026-05-12 architectural regression: pre-fix this method
        spawned 6 separate BusSocketClient connections each with
        topics=None (subscribe-all), masked by the
        `_HIGH_RATE_BROADCAST_TYPES` stopgap that rFP_worker_broadcast_topics_completion §4.C retired same session. Post-stopgap-retirement,
        each broadcast WARN+dropped on 6 connections × every type → 150K
        WARN/h on T1+T2 → api event-loop saturation. The single-connection
        + explicit-topics design here matches SPEC §9.B exactly (verified
        by lockstep test `tests/test_titan_hcl_topics_matches_spec_92b.py`).

        Identity authkey derivation mirrors the legacy broker path.
        Default-off path unaffected — this method only runs when
        l0_rust_enabled=true.
        """
        try:
            identity_secret = self._load_identity_secret_for_bus()
        except Exception as e:
            logger.warning(
                "[TitanKernel] bus_socket clients cannot start — "
                "identity secret unavailable: %s", e, exc_info=True)
            return

        from titan_hcl.core.bus_authkey import derive_bus_authkey
        from titan_hcl.core.bus_socket import BusSocketClient, bus_sock_path
        from titan_hcl.core.worker_bus_bootstrap import (
            ENV_BUS_KEYPAIR_PATH,
            ENV_BUS_SOCKET_PATH,
            ENV_BUS_TITAN_ID,
        )

        try:
            # rFP_phase_c_bus_authkey_contract_fix.md — info is constant b"titan-bus"
            authkey = derive_bus_authkey(identity_secret)
        except Exception as e:
            logger.warning(
                "[TitanKernel] bus_socket clients cannot start — "
                "authkey derivation failed: %s", e, exc_info=True)
            return

        sock_path = bus_sock_path(self.titan_id)
        # SPEC §9.B: single connection. `_bus_clients` is a dict keyed by
        # primary name only; aliases are tracked inside BusSocketClient
        # itself (its `_aliases` set, re-fired on reconnect per D-SPEC-39).
        self._bus_clients: dict[str, "BusSocketClient"] = {}
        # Stop event shared by all dispatcher threads — set on shutdown.
        self._bus_clients_stop_event = threading.Event()
        self._bus_client_dispatchers: list[threading.Thread] = []

        # Single primary connection — opens with SPEC §9.B-enumerated topics.
        # Aliases (5 in-process names + 11 proxy reply queues) follow below.
        try:
            titan_hcl_client = BusSocketClient(
                titan_id=self.titan_id,
                authkey=authkey,
                name="titan_HCL",
                sock_path=sock_path,
                topics=list(TITAN_HCL_BROADCAST_TOPICS),
                reply_only=False,  # SPEC §9.B: titan_HCL IS a broadcast consumer
            )
            titan_hcl_client.start()
            self._bus_clients["titan_HCL"] = titan_hcl_client
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[TitanKernel] primary bus client 'titan_HCL' failed to "
                "start: %s — outbound publishes will not reach Rust broker",
                e, exc_info=True)
            return

        # Single inbound dispatcher — relays broadcasts + targeted dst=<any
        # alias> messages into the in-process bus. Replaces the 6-thread
        # per-name-connection dispatcher topology.
        t = threading.Thread(
            target=self._bus_client_inbound_dispatcher,
            args=(titan_hcl_client, "titan_HCL"),
            daemon=True,
            name="bus-inbound-titan_HCL",
        )
        t.start()
        self._bus_client_dispatchers.append(t)

        # Attach the canonical client for outbound publishes from plugin code.
        self.bus.attach_client(titan_hcl_client)

        # Aliases — SPEC §9.B kernel-internal subscriber names (5) +
        # SPEC §8.2 v1.3.0 multi-name proxy reply queues (11).
        #
        # Each alias is sent as an ADDITIVE BUS_SUBSCRIBE frame over
        # titan_HCL's existing connection. The broker treats these as
        # entries in BrokerSubscriber.aliases; fanout for dst=<alias>
        # routes to titan_HCL, which the inbound dispatcher re-injects
        # into publish_in_process → the in-process subscriber's queue.
        #
        # The 5 in-process aliases (guardian, core, meditation, sovereignty,
        # kernel) are required because spirit_loop / agency / etc. publish
        # with dst="guardian" (MODULE_HEARTBEAT routing per SPEC §9.B
        # guardian_HCL "Subscribes (within titan_HCL's bus client):"), and
        # without these alias registrations the Rust broker would silently
        # drop those targeted messages.
        in_process_aliases = tuple(
            n for n in IN_PROCESS_SUBSCRIBER_NAMES if n != "titan_HCL"
        )
        all_aliases = in_process_aliases + KERNEL_PROXY_ALIASES
        for alias in all_aliases:
            try:
                # subscribe_alias tracks the alias in client._aliases AND
                # fires a BUS_SUBSCRIBE frame with payload.name=<alias>,
                # topics=[], reply_only=<inherited from primary connection>.
                # Tracking ensures aliases are re-fired on every reconnect
                # (SPEC §8.2 v1.3.0); without tracking, a connection
                # drop+reconnect would lose all alias registrations.
                titan_hcl_client.subscribe_alias(alias)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[TitanKernel] alias subscribe '%s' failed: %s",
                    alias, e, exc_info=True)
        logger.info(
            "[TitanKernel] titan_HCL bus client subscribed (SPEC §9.B): "
            "%d primary topics + %d in-process aliases + %d proxy aliases "
            "= 1 connection (was 6 pre-2026-05-12 SPEC alignment)",
            len(TITAN_HCL_BROADCAST_TOPICS),
            len(in_process_aliases),
            len(KERNEL_PROXY_ALIASES))

        # Set env vars so Guardian-spawned workers connect to the same
        # Rust broker socket (kept identical to legacy broker path).
        os.environ[ENV_BUS_SOCKET_PATH] = str(sock_path)
        os.environ[ENV_BUS_TITAN_ID] = self.titan_id
        os.environ[ENV_BUS_KEYPAIR_PATH] = str(self.soul.wallet_path)

        logger.info(
            "[TitanKernel] bus_socket client started (SPEC §9.B compliant: "
            "1 connection × %d names) on %s",
            1 + len(all_aliases), sock_path)

    def _bus_client_inbound_dispatcher(
        self, client, client_name: str,
    ) -> None:
        """Drain titan_HCL's inbound queue → inject into in-process bus.

        Per SPEC §9.B (titan_HCL block + guardian_HCL "within titan_HCL's
        bus client"): there is EXACTLY ONE BusSocketClient connection +
        ONE dispatcher thread post-2026-05-12. All in-process subscribers
        and all 11 proxy aliases share the same connection via SPEC §8.2
        v1.3.0 additive BUS_SUBSCRIBE.

        Echo prevention: messages whose `src` is one of
        IN_PROCESS_SUBSCRIBER_NAMES are dropped — they originated in this
        process (we published them outbound and the Rust broker echoed
        them back to our own client connection).

        Broadcast deduplication: NO LONGER NEEDED post-SPEC-§9.B-alignment.
        Pre-2026-05-12 there were 6 dispatcher threads (one per per-name
        connection); broadcasts arrived 6× and we deduped by gating on
        `client_name == "titan_HCL"`. Now: 1 connection → 1 dispatcher →
        every broadcast arrives exactly once → no dedup needed.

        Targeted messages (dst != "all" / "") were filtered by the broker
        to this client (broker delivers to subs whose name OR alias
        matches dst). Re-inject and let publish_in_process resolve the
        dst against in-process _subscribers.
        """
        from queue import Empty as QueueEmpty

        sq = client.inbound_queue()
        # Echo prevention set — primary name + all known aliases. Workers
        # publishing dst="<alias>" would echo back to us; drop those.
        plugin_names = frozenset(IN_PROCESS_SUBSCRIBER_NAMES) | frozenset(KERNEL_PROXY_ALIASES)
        stop_evt = self._bus_clients_stop_event

        while not stop_evt.is_set():
            try:
                msg = sq.get(timeout=0.5)
            except QueueEmpty:
                continue
            except Exception:  # noqa: BLE001
                # Client closed mid-get; loop checks stop_evt next iteration.
                continue

            if not isinstance(msg, dict):
                continue

            # Echo prevention — drop self-published messages.
            src = msg.get("src", "")
            if src in plugin_names:
                continue

            try:
                self.bus.publish_in_process(msg)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[TitanKernel] bus client '%s' dispatcher: "
                    "publish_in_process raised", client_name)

    def _stop_bus_socket_clients(self) -> None:
        """Graceful client shutdown — called from kernel.shutdown.

        Mirrors _stop_bus_socket_broker. Sets the dispatcher stop event,
        detaches the bus client, calls .stop() on each client (idempotent),
        joins dispatchers with a small timeout, clears env vars.
        """
        clients = getattr(self, "_bus_clients", None)
        if not clients:
            return
        stop_evt = getattr(self, "_bus_clients_stop_event", None)
        if stop_evt is not None:
            stop_evt.set()
        try:
            self.bus.detach_client()
        except Exception:  # noqa: BLE001
            pass
        for name, client in clients.items():
            try:
                client.stop(timeout=2.0)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[TitanKernel] bus client '%s' stop error: %s", name, e)
        # Join dispatchers (daemons; short join to avoid shutdown stall)
        for t in getattr(self, "_bus_client_dispatchers", []):
            try:
                t.join(timeout=0.5)
            except Exception:  # noqa: BLE001
                pass
        self._bus_clients = {}
        self._bus_client_dispatchers = []
        from titan_hcl.core.worker_bus_bootstrap import (
            ENV_BUS_KEYPAIR_PATH,
            ENV_BUS_SOCKET_PATH,
            ENV_BUS_TITAN_ID,
        )
        for k in (ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH):
            os.environ.pop(k, None)
        logger.info("[TitanKernel] bus_socket clients stopped")

    def _load_identity_secret_for_bus(self) -> bytes:
        """Read the identity keypair file and return the 32-byte Ed25519
        secret_seed for HKDF input.

        Per `PLAN_microkernel_phase_c_s2_kernel.md §7.3` +
        `rFP_phase_c_bus_authkey_contract_fix.md`: HKDF IKM is the 32-byte
        secret_seed — NOT the full 64-byte Solana keypair. Solana CLI
        byte-array format is 64 bytes (seed[0..32] + pub_key[32..64]); we
        slice the first 32 to match Rust kernel's
        `Identity::load_from_disk_with_hint` (titan-rust/crates/titan-core/src/identity.rs:327).

        Raises if soul / wallet_path is missing — caller should handle.
        """
        if self.soul is None or not getattr(self.soul, "wallet_path", None):
            raise RuntimeError("soul not booted; identity wallet path unavailable")
        import json as _json
        from pathlib import Path as _Path
        path = _Path(self.soul.wallet_path)
        if not path.exists():
            raise FileNotFoundError(f"identity keypair file not found: {path}")
        data = _json.loads(path.read_text())
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"identity keypair file shape unexpected: {path}")
        full = bytes(int(b) & 0xFF for b in data)
        if len(full) == 32:
            return full
        if len(full) == 64:
            return full[:32]
        raise ValueError(
            f"identity keypair length {len(full)} unrecognized "
            f"(expected 32-byte seed or 64-byte Solana keypair)")


    # ------------------------------------------------------------------
    # SOL balance publisher (M1-H4) — periodic fetch + bus emit
    # ------------------------------------------------------------------
    def _start_balance_publisher(self) -> None:
        """Background thread: fetch SOL balance every 60s + publish
        SOLANA_BALANCE_UPDATED on the bus. The balance value is written
        SHM-direct into the `network_state.bin` slot (D-SPEC-71 Phase A
        single-writer = network_state writer in monitor_tick); the bus
        event is retained for cross-process consumers that need a
        change-edge (e.g. metabolism tier re-evaluation).

        Uses the SYNC solana.rpc.api.Client (same pattern as
        logic.trinity_anchor.maybe_anchor_trinity) — NOT the async
        HybridNetworkClient — because event loops can't be safely shared
        across threads. The async client gets bound to whichever loop
        first calls it; running it from a freshly-spawned thread loop
        causes the api_subprocess to fall over (observed 2026-04-26 first
        deploy attempt: uvicorn died, port 7777 unbound, api in
        crash-loop). The sync RPC call is bounded (60s cadence) and
        cheap (~50-500ms).

        rFP_observatory_data_loading_v1 §3.5 fix (2026-04-26): first
        publish is delayed by `microkernel.balance_publisher_first_delay_s`
        (default 30s). The first deploy crashed api_subprocess uvicorn
        on T2/T3 because the kernel-side balance publish fired during the
        api boot window — multiprocessing.Queue.put + uvicorn startup +
        solana SDK requests pool init all racing for the same resources.
        Waiting until api_subprocess is reliably up (Guardian heartbeats
        the api module after ~10-15s; 30s gives safety margin) avoids
        the race entirely. Cleaner than waiting on a MODULE_READY signal
        because Guardian can also restart api mid-flight; a fresh-after-30s
        publish lands on a stable subprocess in either case.
        """
        if self._limbo_mode or self.network is None:
            logger.info(
                "[BalancePublisher] skipped — limbo mode (no network client)")
            return

        self._balance_publisher_stop = threading.Event()

        first_delay_s = float(self._config.get("microkernel", {}).get(
            "balance_publisher_first_delay_s", 30.0))
        publish_interval_s = float(self._config.get("microkernel", {}).get(
            "balance_publisher_interval_s", 60.0))

        # Boot-prime fires INSIDE the publisher thread (below) so its
        # publish has thread-startup latency (~few ms) rather than blocking
        # this synchronous boot path.

        def _balance_loop():
            # Boot-prime BEFORE the 30s wait: publish last-known sol_balance
            # from data/anchor_state.json (SPEC §11.H entry #19) so /status
            # serves a real value during the warmup window. Closes the
            # Pitch UI boot-warmup "STARVATION / Metabolic Crisis"
            # black-and-white display caused by sol_balance=0.0 (initial
            # in-memory state) being classified as HIBERNATION by the
            # metabolism layer before BalancePublisher's first RPC fetch
            # lands. Failure-mode (file missing or unreadable) is non-fatal —
            # /status falls back to 0.0 for the wait window, then the
            # periodic fetcher populates fresh.
            try:
                self._prime_balance_from_anchor_state()
            except Exception as _prime_err:  # noqa: BLE001
                logger.warning(
                    "[BalancePublisher] boot-prime from anchor_state.json "
                    "failed (non-fatal): %s. /status will return "
                    "sol_balance=0.0 until first fetch in ~%.0fs.",
                    _prime_err, first_delay_s)

            # Delayed first publish (rFP §3.5) — let api_subprocess
            # uvicorn fully come up before the kernel→api bus traffic
            # spike of a sync solana RPC + bus.publish + queue put.
            if first_delay_s > 0:
                if self._balance_publisher_stop.wait(first_delay_s):
                    return  # stopped during delay
            try:
                from solana.rpc.api import Client as SolanaClient
            except Exception as e:
                logger.warning(
                    "[BalancePublisher] solana SDK unavailable: %s", e)
                return
            net_cfg = self._config.get("network", {})
            # Truthy fallback (NOT dict.get(key, default)) so an empty-string
            # `premium_rpc_url = ""` override falls through to public_rpc_urls.
            # Closes BUG-T2T3-SOL-BALANCE-ZERO-20260512: T2/T3 microkernel TOML
            # set `premium_rpc_url = ""` (key PRESENT, value empty) which
            # dict.get returned as `""`, then SolanaClient("") raised
            # empty-message URL errors on every fetch — silent failure visible
            # only as `[BalancePublisher] fetch/publish failed:` (empty %s).
            # Matches `feedback_empty_dict_truthiness_trap.md` pattern.
            rpc_url = (
                net_cfg.get("premium_rpc_url")
                or (net_cfg.get("public_rpc_urls") or [
                    "https://api.mainnet-beta.solana.com"])[0]
            )
            client = SolanaClient(rpc_url)
            pubkey = None
            try:
                pubkey = self.network.pubkey()
            except Exception:
                pubkey = getattr(self.network, "_pubkey", None)
            if pubkey is None:
                logger.warning(
                    "[BalancePublisher] no pubkey on network client — "
                    "publisher exiting")
                return
            # First fetch after the boot-stabilization delay.
            self._publish_balance_once(client, pubkey)
            while not self._balance_publisher_stop.wait(publish_interval_s):
                self._publish_balance_once(client, pubkey)

        t = threading.Thread(
            target=_balance_loop, daemon=True, name="balance-publisher")
        t.start()
        self._balance_publisher_thread = t
        logger.info(
            "[BalancePublisher] started — first publish in %.0fs, then every "
            "%.0fs (sync solana client)",
            first_delay_s, publish_interval_s)

    def _prime_balance_from_anchor_state(self) -> None:
        """Boot-prime: load last-known `sol_balance` from anchor_state.json
        and publish it as SOLANA_BALANCE_UPDATED so consumers see a real
        value during the first_delay_s + RPC-latency window.

        File: `data/anchor_state.json` (SPEC §11.H entry #19) — written by
        the anchor flow on every anchor cycle, contains the most recent
        sol_balance the system observed. On a fresh kernel boot this is
        typically <60s stale.

        Behavior:
          • File missing / unreadable / malformed → no-op (caller logs warning).
          • `sol_balance` field missing or non-numeric → no-op.
          • `sol_balance == 0.0` → publish ANYWAY (this is a valid wallet
            state; only deviation from the "no prior balance" case is the
            file existing, which we treat as a known zero).
          • Successful publish goes via the bus, just like the periodic
            fetcher — same path means same SHM `network_state.bin` writer.

        Failure mode: bus.publish() may queue rather than deliver if the
        SHM `network_state.bin` writer is mid-init (boot race). That's the
        same race the original first_delay_s=30s was designed to avoid
        for the fetcher path. Mitigation: publish IS thread-safe + the
        slot writer overwrites on next fetch (~30s later) — UI never sees
        sol_balance=0.0 → black-and-white for the full warmup window.

        Closes the boot-warmup UX issue: Pitch UI briefly displayed
        STARVATION / Metabolic Crisis (black-and-white) on T1 restart
        because metabolism classified the initial in-memory
        sol_balance=0.0 as HIBERNATION before BalancePublisher's first
        fetch landed.
        """
        import json
        import os

        # data_dir resolved via config (same pattern as other state files).
        # No state-file lookup → assume default project-root data dir.
        anchor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "data", "anchor_state.json")
        if not os.path.exists(anchor_path):
            logger.info(
                "[BalancePublisher] boot-prime skipped — %s does not exist "
                "(fresh install or anchor never fired). /status will return "
                "sol_balance=0.0 until first fetch.",
                anchor_path)
            return
        try:
            with open(anchor_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"anchor_state.json unreadable/malformed: {e!r}") from e
        bal = state.get("sol_balance")
        if not isinstance(bal, (int, float)):
            logger.info(
                "[BalancePublisher] boot-prime skipped — sol_balance field "
                "in anchor_state.json is %r (not numeric)", bal)
            return
        bal_float = float(bal)
        payload = {"balance": bal_float}
        self.bus.publish(make_msg(
            SOLANA_BALANCE_UPDATED, "kernel", "all", payload))
        ts = state.get("last_anchor_time")
        age_s = (time.time() - ts) if isinstance(ts, (int, float)) else None
        logger.info(
            "[BalancePublisher] boot-primed sol_balance=%.6f from "
            "anchor_state.json (age=%s) — published SOLANA_BALANCE_UPDATED; "
            "first RPC fetch in %.0fs will refresh",
            bal_float,
            f"{age_s:.0f}s" if age_s is not None else "unknown",
            float(self._config.get("microkernel", {}).get(
                "balance_publisher_first_delay_s", 30.0)))

    def _publish_balance_once(self, client, pubkey) -> None:
        """One sync balance fetch + bus publish. Wrapped for try/except."""
        try:
            resp = client.get_balance(pubkey)
            lamports = getattr(resp, "value", 0) or 0
            balance = float(lamports) / 1_000_000_000
            payload = {"balance": balance}
            self.bus.publish(make_msg(
                SOLANA_BALANCE_UPDATED, "kernel", "all", payload))
            # Write the fetched balance back onto the network client so its
            # `balance` + `is_available` properties report real values. The
            # network_state_publisher reads these (getattr(network, "balance"/
            # "is_available")) to fill network_state.bin → /status.sol_balance
            # + /health RPC_CONNECTIVITY. Without this, the slot stayed at
            # balance_sol=0 / network_available=False despite a successful
            # fetch (the bus publish alone never reached the publisher's read).
            try:
                if self.network is not None:
                    import time as _t
                    self.network._balance_cache = balance
                    self.network._balance_cache_ts = _t.time()
                    self.network._network_available = True
            except Exception:
                pass
        except Exception as e:
            # `%r` + exception type so empty-message failures (e.g. URL
            # parse error from a misconfigured RPC) are not silent. Closes
            # the second half of BUG-T2T3-SOL-BALANCE-ZERO-20260512 — the
            # bare `%s` previously logged "fetch/publish failed:" with no
            # diagnostic content for 4+ days before discovery.
            rpc_endpoint = getattr(client, "_provider", None)
            rpc_endpoint = getattr(rpc_endpoint, "endpoint_uri",
                                   getattr(client, "endpoint_uri", "?"))
            logger.warning(
                "[BalancePublisher] fetch/publish failed (rpc=%s): %s(%r)",
                rpc_endpoint, type(e).__name__, e)

    def _stop_balance_publisher(self) -> None:
        evt = getattr(self, "_balance_publisher_stop", None)
        if evt is not None:
            evt.set()

    # ------------------------------------------------------------------
    # Read-only @property accessors (KernelView interface contract)
    # ------------------------------------------------------------------
    # Narrow, typed read surface for upper layers. Mutation of kernel
    # state goes through explicit methods (boot, start_modules, shutdown
    # — added in commit 2). See titan_hcl.core.kernel_interface.

    @property
    def config(self) -> dict:
        """Read-only view of the loaded config.

        Plugin + upper-layer read access. Kernel owns the canonical dict;
        mutation is discouraged. Hot config changes flow via SHM now (the
        kernel config daemon re-seeds per-section slots on a config edit;
        workers re-apply on heartbeat) — see RFP_config_as_shm_state.
        """
        return self._config

    @property
    def titan_id(self) -> str:
        """Resolved titan identifier (T1 / T2 / T3 / ...).

        Source of truth: data/titan_identity.json (canonical precedence
        chain). Set once at __init__, immutable for the process lifetime.
        """
        return self._titan_id

    @property
    def limbo_mode(self) -> bool:
        """True when no keypair could be resolved — degraded operation.

        In limbo: self.soul is None, self.network is None. Kernel still
        boots and runs L0 services (bus, guardian, shm); L2/L3 subsystems
        that depend on the wallet degrade gracefully.
        """
        return self._limbo_mode

    # ------------------------------------------------------------------
    # Microkernel v2 Phase B.1 — Shadow Core Swap (rFP §347-357)
    # ------------------------------------------------------------------

    @property
    def kernel_version(self) -> str:
        """Short identifier for the kernel code currently running.

        Used in RuntimeSnapshot.kernel_version + the system-fork TimeChain
        block (kernel_version_from / kernel_version_to). Falls back to a
        sentinel if git is unavailable.

        Cached on first access — cheap; we only need a single read per
        hibernate event.
        """
        cached = getattr(self, "_kernel_version_cache", None)
        if cached:
            return cached
        version = "unknown"
        try:
            import subprocess
            here = os.path.dirname(__file__)
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=here, capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                version = result.stdout.strip() or "unknown"
        except Exception as e:
            logger.debug("[TitanKernel] kernel_version git lookup failed: %s", e)
        self._kernel_version_cache = version
        return version

    def dump_heap(self, top_types: int = 30, top_containers: int = 20) -> dict:
        """Live heap snapshot of the parent (kernel) process.

        Aggregates `gc.get_objects()` by type and surfaces the largest
        unbounded containers. Aggregate-only — never object content.

        Cost: ~1-5 s wall-clock on a 2 GB heap. Endpoint callers MUST
        run via `asyncio.to_thread` to avoid blocking the event loop.

        Returns the dict produced by `take_heap_snapshot()` plus
        `pid` and `process="parent"`.
        """
        from titan_hcl.core.profiler import take_heap_snapshot
        snap = take_heap_snapshot(top_types=top_types,
                                   top_containers=top_containers)
        snap["pid"] = os.getpid()
        snap["process"] = "parent"
        return snap

    def dump_tracemalloc(self, top_n: int = 30,
                         key_type: str = "filename",
                         diff: bool = True) -> dict:
        """Live tracemalloc snapshot of the parent (kernel) process.

        Returns top file:line allocators by size (or by growth since
        boot if `diff=True`). Requires tracemalloc to have been started
        at boot via `[profiling] tracemalloc_enabled = true`.

        Per worker_stability_audit 2026-04-27: this is the canonical
        path to find C-level memory leaks invisible to gc.get_objects().

        Args:
          top_n:    number of top allocators to return
          key_type: "filename" or "lineno"
          diff:     if True, return growth since boot baseline

        Returns dict with: `pid`, `process="parent"`, `tracemalloc_active`,
        and either `top` (sorted by size) or `diff` (sorted by growth).
        """
        import tracemalloc as _tm
        result: dict = {
            "pid": os.getpid(),
            "process": "parent",
            "tracemalloc_active": _tm.is_tracing(),
        }
        if not _tm.is_tracing():
            result["error"] = ("tracemalloc not running — set "
                                "[profiling] tracemalloc_enabled=true + restart")
            return result
        # `_profiling_collector` is set on the plugin, not kernel; locate it
        # via the parent process's plugin_ref (set when kernel_rpc started).
        collector = None
        plugin_ref = getattr(self, "_plugin_ref", None)
        if plugin_ref is not None:
            collector = getattr(plugin_ref, "_profiling_collector", None)
        if collector is None:
            # Fall back to ad-hoc snapshot (no diff baseline available)
            snap = _tm.take_snapshot()
            stats = snap.statistics(key_type)
            result["fallback"] = "no _profiling_collector — boot-time baseline missing"
            result["top"] = [
                {"file": str(s.traceback), "size_mb": round(s.size / 1048576, 2),
                 "size_bytes": s.size, "count": s.count}
                for s in stats[:top_n]
            ]
            return result
        if diff:
            result["diff"] = collector.get_diff_stats(n=top_n, key_type=key_type)
        else:
            result["top"] = collector.get_top_stats(n=top_n, key_type=key_type)
        result["summary"] = collector.get_summary()
        return result

    def dump_thread_inventory(self) -> dict:
        """Live thread inventory of the parent (kernel) process.

        Snapshots `threading.enumerate()` and groups by name-prefix to
        surface where the parent's threads come from. Used by `arch_map
        thread-pool --parent` for the rFP A.8 §6 measurement-driven
        residency audit, and by tests/test_a8_thread_count.py as the
        regression baseline.

        Returns:
          {
            "pid": <parent pid>,
            "process": "parent",
            "total": <int>,
            "threads": [
              {"name": str, "ident": int, "daemon": bool, "alive": bool},
              ...
            ],
            "by_prefix": {<prefix>: <count>, ...},
          }
        """
        import threading as _threading
        threads = list(_threading.enumerate())
        rows = []
        by_prefix: dict[str, int] = {}
        for t in threads:
            name = t.name or "<unnamed>"
            rows.append({
                "name": name,
                "ident": t.ident,
                "daemon": bool(t.daemon),
                "alive": t.is_alive(),
            })
            # Group by trimming a per-instance ID suffix at the LAST
            # "-" / ":" / "_" separator (e.g., "shadow-swap-deadbeef" →
            # "shadow-swap"). rpartition isolates the trailing token; we
            # only collapse if the trailing token is ≥6 chars of hex.
            prefix = name
            for sep in ("-", ":", "_"):
                if sep in prefix:
                    head, _, tail = prefix.rpartition(sep)
                    if len(tail) >= 6 and all(
                            c in "0123456789abcdef" for c in tail.lower()):
                        prefix = head
                    break
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        return {
            "pid": os.getpid(),
            "process": "parent",
            "total": len(rows),
            "threads": rows,
            "by_prefix": dict(sorted(
                by_prefix.items(), key=lambda kv: -kv[1])),
        }

    def hibernate_runtime(
        self,
        event_id: str,
        snapshot_path: Optional[str] = None,
    ) -> str:
        """Serialize kernel runtime state to disk for a shadow swap.

        Called by the shadow-swap orchestrator (scripts/shadow_swap.py)
        after readiness wait completes and before sending HIBERNATE to
        workers. The orchestrator passes in the upgrade `event_id` (UUID4)
        so this snapshot's path links to the system-fork TimeChain block.

        Returns the path the snapshot was written to (for the orchestrator
        to pass to the shadow kernel via --restore-from).

        This method does NOT pause the bus or the kernel. Workers continue
        normally; the snapshot is just a metadata file. The actual quiescence
        happens via the HIBERNATE bus message handled per-worker.
        """
        from titan_hcl.core import shadow_protocol as sp

        # ── Collect snapshot fields ──
        soul_gen = 0
        if self.soul is not None:
            soul_gen = int(getattr(self.soul, "current_gen", 0) or 0)

        # Registry seqs (informational — shadow re-opens same /dev/shm files
        # so seqs naturally continue. Dict captured for OBS-fidelity diagnostics).
        registry_seqs: dict[str, int] = {}
        bank = getattr(self, "registry_bank", None)
        if bank is not None:
            for name, writer in getattr(bank, "_writers", {}).items():
                seq = getattr(writer, "_seq", None)
                if seq is None:
                    seq = getattr(writer, "last_seq", 0)
                try:
                    registry_seqs[name] = int(seq)
                except (TypeError, ValueError):
                    registry_seqs[name] = 0

        # Guardian module roster — capture names so shadow can verify superset
        guardian_modules: list[str] = []
        if self.guardian is not None:
            modules_dict = getattr(self.guardian, "_modules", None)
            if modules_dict:
                guardian_modules = sorted(modules_dict.keys())

        # Bus subscriber count — informational; shadow re-subscribes on boot.
        bus_subscriber_count = 0
        if self.bus is not None:
            subs = getattr(self.bus, "_subscribers", {})
            bus_subscriber_count = sum(len(v) for v in subs.values())

        snap = sp.RuntimeSnapshot(
            kernel_version=self.kernel_version,
            soul_current_gen=soul_gen,
            titan_id=self._titan_id,
            registry_seqs=registry_seqs,
            guardian_modules=guardian_modules,
            bus_subscriber_count=bus_subscriber_count,
            written_at=time.time(),
            event_id=event_id,
        )

        # ── Serialize ──
        if snapshot_path is None:
            target = sp.default_snapshot_path()
        else:
            from pathlib import Path
            target = Path(snapshot_path)
        path_written = sp.serialize_snapshot(snap, target)

        logger.info(
            "[TitanKernel] hibernate_runtime: event_id=%s kernel_version=%s "
            "soul_gen=%d titan_id=%s modules=%d registries=%d bus_subs=%d → %s",
            event_id[:8], snap.kernel_version, soul_gen, self._titan_id,
            len(guardian_modules), len(registry_seqs), bus_subscriber_count,
            path_written,
        )
        return str(path_written)

    def restore_from_snapshot(
        self,
        snapshot_path: str,
        *,
        max_age_seconds: float = 300.0,
    ) -> dict:
        """Verify a runtime snapshot at boot time (--restore-from).

        Called from `scripts/titan_hcl.py` during shadow-kernel boot,
        AFTER soul + guardian are initialized but BEFORE start_modules().

        Returns a dict with the snapshot's event_id + verification result.
        Caller (titan_hcl) is responsible for:
          - Logging the verification outcome to the brain log
          - Publishing SYSTEM_RESUMED on the bus once start_modules completes
            (we publish later, not here, so the bus is fully wired first)

        If verify_compatible() fails, this method LOGS the reason but does
        NOT raise — the kernel continues a clean boot. The orchestrator's
        rollback path triggers via HIBERNATE_CANCEL on the OLD kernel.
        Refusing to boot would leave the system without a kernel at all.
        """
        from titan_hcl.core import shadow_protocol as sp

        try:
            snap = sp.deserialize_snapshot(snapshot_path)
        except FileNotFoundError:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: file not found at %s — "
                "continuing clean boot", snapshot_path,
            )
            return {"verified": False, "reason": "file_not_found", "event_id": ""}
        except Exception as e:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: deserialize failed (%s) — "
                "continuing clean boot", e,
            )
            return {"verified": False, "reason": f"deserialize_error:{e}", "event_id": ""}

        # Compat check — collect target's module roster
        target_modules: list[str] = []
        if self.guardian is not None:
            target_modules = sorted(getattr(self.guardian, "_modules", {}).keys())

        ok, reason = sp.verify_compatible(
            snap,
            target_titan_id=self._titan_id,
            target_modules=target_modules,
            max_age_seconds=max_age_seconds,
        )

        if ok:
            logger.info(
                "[TitanKernel] restore_from_snapshot: VERIFIED event_id=%s "
                "kernel_version_from=%s soul_gen_from=%d age=%.1fs",
                snap.event_id[:8], snap.kernel_version, snap.soul_current_gen,
                time.time() - snap.written_at,
            )
        else:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: REFUSED event_id=%s reason=%s — "
                "continuing clean boot (orchestrator should detect via missing SYSTEM_RESUMED)",
                snap.event_id[:8] if snap.event_id else "?", reason,
            )

        return {
            "verified": ok,
            "reason": reason,
            "event_id": snap.event_id,
            "kernel_version_from": snap.kernel_version,
            "soul_gen_from": snap.soul_current_gen,
            "age_seconds": time.time() - snap.written_at,
        }

    def shadow_swap_orchestrate(self, reason: str = "manual",
                                grace: float = 120.0,
                                b2_1_forced: bool = False) -> dict:
        """B.1 §7 — KICKOFF entrypoint for the shadow swap protocol.

        Spawns the orchestrator in a background thread + returns
        immediately with {event_id, outcome="started", ...}. The kernel's
        main thread continues normally (Guardian drain, RPC service,
        bus routing) so workers can ACK the swap protocol without
        starvation.

        Caller polls /maker/upgrade-status (or kernel.shadow_swap_status)
        to track progress + retrieve the final result.

        Refuses if microkernel.shadow_swap_enabled=false OR if another
        swap is currently active.
        """
        # Phase C/D: l0_rust is permanently true, and Rust kernel-rs has no
        # BUS_HANDOFF / shadow_swap_orchestrate / hibernate symbols yet, so the
        # B.1 + B.2.1 shadow-swap protocol cannot complete — this UNCONDITIONALLY
        # refuses. Operators upgrade via systemd restart. The protocol body below
        # is PRESERVED (currently unreachable) for the future Rust BUS_HANDOFF
        # work that will re-enable it (config-shm Phase D — Maker call 2026-06-18).
        logger.warning(
            "[TitanKernel] shadow_swap_orchestrate refused — Rust kernel has no "
            "BUS_HANDOFF yet; use systemd restart instead.")
        return {
            "outcome": "error",
            "failure_reason": "l0_rust_enabled_shadow_swap_unsupported",
            "phase": "preflight",
            "event_id": "",
            "elapsed_seconds": 0.0,
        }

        # Flag check — refuse if shadow_swap_enabled is false (default)
        flag = (self._config.get("microkernel", {})
                            .get("shadow_swap_enabled", False))
        if not bool(flag):
            return {
                "outcome": "error",
                "failure_reason": "shadow_swap_enabled_flag_off",
                "phase": "preflight",
                "event_id": "",
                "elapsed_seconds": 0.0,
            }

        # Thread-safe single-active-swap guard
        if not hasattr(self, "_shadow_swap_lock"):
            self._shadow_swap_lock = threading.Lock()
            self._shadow_swap_active = None  # event_id of running swap
            self._shadow_swap_progress = {}  # event_id → live progress dict
            self._shadow_swap_history = {}   # event_id → final result

        with self._shadow_swap_lock:
            if self._shadow_swap_active:
                return {
                    "outcome": "error",
                    "failure_reason": "another_swap_active",
                    "active_event_id": self._shadow_swap_active,
                    "phase": "preflight",
                    "event_id": "",
                }
            from titan_hcl.core import shadow_protocol as _sp
            event_id = _sp.new_event_id()
            self._shadow_swap_active = event_id
            # Phase A retrofit (2026-04-27): clear done event so threads
            # entering wait_for_swap_completion() during the swap actually
            # block. Set again in _run_swap finally on completion.
            self._shadow_swap_done_event.clear()
            self._shadow_swap_progress[event_id] = {
                "event_id": event_id,
                "outcome": "running",
                "phase": "preflight",
                "reason": reason,
                "started_at": time.time(),
                "elapsed_seconds": 0.0,
            }

        def _run_swap():
            try:
                from titan_hcl.core.shadow_orchestrator import orchestrate_shadow_swap
                result = orchestrate_shadow_swap(
                    self, reason=reason, grace=grace, event_id=event_id,
                    b2_1_forced=b2_1_forced,
                )
                with self._shadow_swap_lock:
                    self._shadow_swap_history[event_id] = result
                    # Keep only last 5 in memory; full history is in audit jsonl
                    if len(self._shadow_swap_history) > 5:
                        oldest = sorted(
                            self._shadow_swap_history.keys(),
                            key=lambda k: self._shadow_swap_history[k].get("started_at", 0),
                        )[0]
                        self._shadow_swap_history.pop(oldest, None)
            except Exception as e:
                logger.exception("[shadow_swap] background thread crashed: %s", e)
                with self._shadow_swap_lock:
                    self._shadow_swap_history[event_id] = {
                        "event_id": event_id, "outcome": "error",
                        "failure_reason": f"thread_crashed:{e}",
                        "phase": "?", "started_at": time.time(),
                    }
            finally:
                with self._shadow_swap_lock:
                    if self._shadow_swap_active == event_id:
                        self._shadow_swap_active = None
                    self._shadow_swap_progress.pop(event_id, None)
                # Phase A retrofit (2026-04-27): signal swap completion to
                # any threads blocked on wait_for_swap_completion() — proxy
                # lazy-starts deferred during the swap can now proceed.
                self._shadow_swap_done_event.set()

        thread = threading.Thread(
            target=_run_swap, daemon=False,
            name=f"shadow-swap-{event_id[:8]}",
        )
        thread.start()

        # Return kickoff result immediately — caller polls for progress.
        return {
            "event_id": event_id,
            "outcome": "started",
            "phase": "preflight",
            "reason": reason,
            "started_at": time.time(),
            "elapsed_seconds": 0.0,
            "poll_endpoint": "/maker/upgrade-status",
        }

    def shadow_swap_status(self, event_id: str = "") -> dict:
        """B.1 §7 — read live progress / final result of a shadow swap.

        With event_id: returns that swap's state (live if active, or
        from history if completed). Without: returns the most recent
        active swap, or last completed swap.
        """
        if not hasattr(self, "_shadow_swap_lock"):
            return {"outcome": "no_swaps_yet", "history": []}

        with self._shadow_swap_lock:
            active = self._shadow_swap_active
            if event_id:
                if event_id in self._shadow_swap_progress:
                    return dict(self._shadow_swap_progress[event_id])
                if event_id in self._shadow_swap_history:
                    return dict(self._shadow_swap_history[event_id])
                return {"outcome": "not_found", "event_id": event_id}
            # No event_id — return current active OR last completed
            if active and active in self._shadow_swap_progress:
                return dict(self._shadow_swap_progress[active])
            if self._shadow_swap_history:
                last_eid = max(
                    self._shadow_swap_history.keys(),
                    key=lambda k: self._shadow_swap_history[k].get("started_at", 0),
                )
                return dict(self._shadow_swap_history[last_eid])
            return {"outcome": "no_swaps_yet"}

    # ------------------------------------------------------------------
    # Static helpers (lifted from TitanCore for verbatim semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full_config() -> dict:
        """Load the full merged Titan config (config.toml + ~/.titan/secrets.toml)."""
        return load_titan_params()

    def _resolve_wallet(self, wallet_path: str) -> Optional[str]:
        """Resolve wallet keypair (lifted from TitanCore._resolve_wallet).

        Precedence:
          1. Hardware-bound encrypted keypair (data/soul_keypair.enc) —
             decrypt via utils.crypto.decrypt_for_machine and persist to
             runtime_keypair.json.
          2. Plain wallet_path if it exists on disk.
          3. Genesis-record-only fallback (limbo mode) — returns None.
          4. wallet_path (non-existent) — degraded mode.
        """
        enc_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "soul_keypair.enc")
        )
        if os.path.exists(enc_path):
            try:
                from titan_hcl.utils.crypto import decrypt_for_machine
                with open(enc_path, "rb") as f:
                    encrypted = f.read()
                key_bytes = decrypt_for_machine(encrypted)
                import json
                runtime_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "data", "runtime_keypair.json"
                )
                with open(runtime_path, "w") as f:
                    json.dump(list(key_bytes), f)
                logger.info("[TitanKernel] Warm reboot: hardware-bound keypair decrypted.")
                return runtime_path
            except Exception as e:
                logger.warning("[TitanKernel] Hardware-bound keypair failed: %s", e)

        if os.path.exists(wallet_path):
            return wallet_path

        genesis_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "genesis_record.json")
        )
        if os.path.exists(genesis_path):
            return None

        logger.info("[TitanKernel] No keypair at %s — degraded mode.", wallet_path)
        return wallet_path
