"""
Divine Bus — IPC message router for Titan V3.0 microkernel.

Phase 1 uses multiprocessing.Queue for simplicity (~50K msg/s).
Graduates to Unix domain sockets in Phase 2+ if throughput demands it.

Message envelope:
    {
        "type": str,       # message type (BODY_STATE, MIND_STATE, MODULE_READY, etc.)
        "src": str,        # source module name
        "dst": str,        # destination module name or "all" for broadcast
        "ts": float,       # time.time()
        "rid": str | None, # request ID for QUERY/RESPONSE pairs
        "payload": dict,   # type-specific data
    }
"""
import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from collections import OrderedDict
from queue import Empty, Full, Queue as ThreadQueue
from multiprocessing import Queue as MPQueue
from typing import Callable, Iterable, Optional, Union

from .shared_blackboard import SharedBlackboard
from .core import bus_census as _census  # opt-in instrumentation (TITAN_BUS_CENSUS=1)
from .utils.silent_swallow import swallow_warn  # Pattern C — visibility for cross-process emit failures
from titan_plugin.utils.silent_swallow import swallow_warn

# Type alias: either threading or multiprocessing Queue
AnyQueue = Union[ThreadQueue, MPQueue]

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Bus IPC dedicated thread pool
# ──────────────────────────────────────────────────────────────────────
# 2026-04-29 — RCA of T1 thread-pool saturation (250% peak: 64 busy + 192
# queued). Default asyncio executor (64 workers) serves ~151 to_thread
# sites including Observatory snapshots, DB queries, and bus.request reply
# waits. Under Observatory poll bursts, the pool saturates → bus.request
# reply waits get queued → 3s default timeout exceeded → cascade:
#   - BUG-DIVINEBUS-SPIRIT-PROXY-TIMEOUTS (333 timeouts/18h on T2)
#   - BUG-DASHBOARD-BUS-ATTR-ERRORS (cascade in /v4/signal-concept etc.)
#   - Observatory data widgets show empty/loading
#
# Architectural invariant: bus IPC reply waits are LATENCY-SENSITIVE
# (3s timeout) and must NOT queue behind LATENCY-INSENSITIVE work
# (Observatory snapshots, 1-5s OK). Solution: dedicated small pool.
#
# Sized at 8 workers (configurable). Saturation of THIS pool is a
# CRITICAL signal — means real bus contention, not Observatory load.
#
# Async callers use `bus.request_async()` which routes through this
# pool. Sync `bus.request()` callers retain previous behavior (run in
# whichever thread the caller is on — typically default pool).
_bus_ipc_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_bus_ipc_pool_lock = threading.Lock()
_BUS_IPC_DEFAULT_WORKERS = 8


def get_bus_ipc_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Returns the dedicated thread pool for bus IPC reply waits.

    Lazy-init on first access. Size from titan_params.toml
    [runtime.pools].bus_ipc_workers, default 8. Threads named
    `bus-ipc-N` so pool stats endpoint can distinguish them.

    Saturation of this pool is a CRITICAL alert — see RCA above.
    """
    global _bus_ipc_pool
    if _bus_ipc_pool is not None:
        return _bus_ipc_pool
    with _bus_ipc_pool_lock:
        if _bus_ipc_pool is not None:
            return _bus_ipc_pool
        size = _BUS_IPC_DEFAULT_WORKERS
        try:
            from titan_plugin.config_loader import load_titan_config
            cfg = load_titan_config() or {}
            size = int(cfg.get("runtime", {}).get("pools", {}).get(
                "bus_ipc_workers", _BUS_IPC_DEFAULT_WORKERS))
        except Exception:
            pass
        _bus_ipc_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, size),
            thread_name_prefix="bus-ipc",
        )
        logger.info("[DivineBus] bus_ipc_pool initialized — max_workers=%d", size)
        return _bus_ipc_pool

# ── Message type constants ──────────────────────────────────────────

# Tensor updates (periodic)
BODY_STATE = "BODY_STATE"
MIND_STATE = "MIND_STATE"
SPIRIT_STATE = "SPIRIT_STATE"

# Sense data (event-driven)
# RESERVED: SENSE_INPUT is the generic superclass type — specific channels
# (SENSE_VISUAL / SENSE_AUDIO) are what's actually published today.
SENSE_INPUT = "SENSE_INPUT"
SENSE_VISUAL = "SENSE_VISUAL"
SENSE_AUDIO = "SENSE_AUDIO"

# Counterpart messages
FOCUS_NUDGE = "FOCUS_NUDGE"
# LEGACY:INTUITION_SUGGEST/OUTCOME + FILTER_DOWN (V3) are v1 names.
# Active intuition path: direct IntuitionEngine.suggest/record_outcome calls
# from spirit_worker. Active filter-down path: FILTER_DOWN_V5.
INTUITION_SUGGEST = "INTUITION_SUGGEST"
INTUITION_OUTCOME = "INTUITION_OUTCOME"  # LEGACY:
FILTER_DOWN = "FILTER_DOWN"              # LEGACY:

# Interface messages (Step 5: human ↔ digital plane)
INTERFACE_INPUT = "INTERFACE_INPUT"

# Agency messages (Step 7: autonomous impulse → action)
IMPULSE = "IMPULSE"
# RESERVED: INTENT was designed as an intermediate stage between IMPULSE and
# ACTION_RESULT; current pipeline goes straight IMPULSE → action → ACTION_RESULT.
# Kept as reserved hook-point for future two-phase agency (impulse→intent→action).
INTENT = "INTENT"
ACTION_RESULT = "ACTION_RESULT"
RATE_LIMIT = "RATE_LIMIT"

# L3 §A.8.6 — Agency subprocess broadcasts. agency_worker emits AGENCY_READY
# on boot (parent's _agency_loop flips proxy state to bus-routed) and
# AGENCY_STATS / ASSESSMENT_STATS every 60s (parent's _agency_loop refreshes
# proxy cached attrs so dashboard /v3/agency reads return fresh stats without
# a per-call bus round-trip).
AGENCY_READY = "AGENCY_READY"
AGENCY_STATS = "AGENCY_STATS"
ASSESSMENT_STATS = "ASSESSMENT_STATS"

# Control messages
MODULE_READY = "MODULE_READY"
MODULE_HEARTBEAT = "MODULE_HEARTBEAT"
MODULE_SHUTDOWN = "MODULE_SHUTDOWN"
MODULE_CRASHED = "MODULE_CRASHED"
EPOCH_TICK = "EPOCH_TICK"

# Mainnet Lifecycle Wiring rFP (2026-04-20): SovereigntyTracker wiring.
# Spirit_worker publishes SOVEREIGNTY_EPOCH every 10 consciousness epochs
# with current neuromod snapshot + dev_age + great_pulse_fired. Main
# process v5_core subscribes and calls plugin.sovereignty.record_epoch.
# Effective resolution is 10:1 sampled (5000-sample convergence window
# = ~50k actual epochs ≈ 10-14h). Adequate for long-horizon convergence.
SOVEREIGNTY_EPOCH = "SOVEREIGNTY_EPOCH"

# Disk health — edge-detected state transitions only (never continuous
# telemetry). Published by DiskHealthMonitor when free-space crosses
# hysteresis-protected thresholds. EMERGENCY triggers Guardian.stop_all.
# RESERVED: DiskHealthMonitor class exists but is not instantiated in the
# current boot path — constants are reserved for when the monitor is wired.
DISK_WARNING = "DISK_WARNING"
DISK_CRITICAL = "DISK_CRITICAL"
DISK_EMERGENCY = "DISK_EMERGENCY"
DISK_RECOVERED = "DISK_RECOVERED"  # emitted when state improves back to HEALTHY
# RESERVED: RL_CYCLE_REQUEST — reserved hook point for offline IQL cycle
# scheduling (not yet wired).
RL_CYCLE_REQUEST = "RL_CYCLE_REQUEST"

# Vault anchor request — wired 2026-04-29 (BUG-VAULT-COMMITS-NOT-LANDING).
# In microkernel v2, meditation runs in memory_worker subprocess, which has
# `network_client=None` (deployer keypair stays in main process for security).
# Memory_worker emits ANCHOR_REQUEST to the kernel main process, which holds
# `self.network` (HybridNetworkClient) and the keypair. Kernel builds the
# vault commit instructions via MeditationEpoch._build_commit_instructions,
# submits the TX, and replies via bus.RESPONSE matched on `rid`.
#
# Wire contract:
#   ANCHOR_REQUEST: src=memory, dst=kernel, rid=<unique>
#     payload = {"state_root": str, "payload": str (json),
#                "promoted_count": int, "ts": float}
#   RESPONSE:       src=kernel, dst=memory, rid=<matches request>
#     payload = {"tx_signature": str | None, "error": str | None}
#
# The kernel handler is non-blocking on the parent event loop — TX
# submission happens in `asyncio.to_thread` so daemon threads + main loop
# stay responsive. Memory_worker waits up to 30s for the response with
# heartbeats interleaved so Guardian doesn't kill it during the wait.
ANCHOR_REQUEST = "ANCHOR_REQUEST"

# Microkernel v2 Layer 2 (2026-04-28) — Sage subprocess migration.
# Parent (TitanPlugin / SageGuardian) publishes transition records via this
# message; rl_worker handles them by calling its in-subprocess SageRecorder.
# Closes BUG-SAGE-INSTANTIATED-IN-PARENT (parent no longer holds the
# 2GB LazyMemmapStorage). Payload is kwargs-shaped to match
# SageRecorder.record_transition signature.
SAGE_RECORD_TRANSITION = "SAGE_RECORD_TRANSITION"

# Microkernel v2 §A.8.7 — Sage Scholar/Gatekeeper consolidation (2026-04-28).
# Parent (legacy `__init__.py` path OR V6 `core/plugin.py` path via RLProxy)
# routes IQL training + gatekeeper routing to rl_worker via these QUERY
# actions. Worker owns Recorder + Scholar + Gatekeeper.
# - SAGE_GATE_DECIDE  — request type alias for QUERY action="decide_execution_mode"
# - SAGE_IQL_TRAIN_STEP — request type alias for QUERY action="dream"
# - SAGE_STATS — broadcast every 60s from rl_worker (sovereignty_score etc.)
# - SAGE_READY — boot signal (mirrors A.8.3/A.8.5 dual-emit pattern)
SAGE_GATE_DECIDE = "SAGE_GATE_DECIDE"
SAGE_IQL_TRAIN_STEP = "SAGE_IQL_TRAIN_STEP"
SAGE_STATS = "SAGE_STATS"
SAGE_READY = "SAGE_READY"

# V5 Dual-Layer Nervous System
OUTER_DISPATCH = "OUTER_DISPATCH"  # Outer program → Agency direct dispatch
OUTER_OBSERVATION = "OUTER_OBSERVATION"  # Action result → Spirit worker observation feedback

# Hot-Reload (Tier 2)
RELOAD = "RELOAD"          # Request worker module reload
# RESERVED: RELOAD_COMPLETE — workers currently confirm via MODULE_READY
# after restart instead of a separate COMPLETE signal. Kept for future
# two-phase reload protocol (request → confirm distinct from ready).
RELOAD_COMPLETE = "RELOAD_COMPLETE"  # Worker confirms reload success
CONFIG_RELOAD = "CONFIG_RELOAD"  # Hot-reload titan_params.toml without restart

# Microkernel v2 Phase A §A.4 (S5) — kernel→api WebSocket bridge
# Replaces direct plugin.event_bus.emit() calls when the API runs as a
# Guardian-supervised L3 subprocess. Kernel publishes; API subprocess
# subscribes and translates each message to event_bus.emit() for its
# WebSocket subscribers. Payload shape: {"event_type": str, "data": dict}.
# See PLAN_microkernel_phase_a_s5.md §2.5 + D5.
OBSERVATORY_EVENT = "OBSERVATORY_EVENT"  # kernel/spirit → api (WebSocket emit translation)

# Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25) — bus-cached
# state population for the api_subprocess. Kernel publishes a periodic
# bulk snapshot; api_subprocess BusSubscriber consumes and updates its
# CachedState dict that endpoint code reads via TitanStateAccessor.
# See PLAN_microkernel_phase_a_s5_amendment.md.
STATE_SNAPSHOT_REQUEST = "STATE_SNAPSHOT_REQUEST"      # api → kernel (bootstrap)
STATE_SNAPSHOT_RESPONSE = "STATE_SNAPSHOT_RESPONSE"    # kernel → api (full state dump)

# ── Chat bus bridge (BUG-CHAT-AGENT-NOT-INITIALIZED-API-SUBPROCESS) ──
# 2026-04-29 — When microkernel.api_process_separation_enabled=true, the
# api subprocess holds a kernel_rpc proxy to plugin but cannot reach the
# Agno agent (would double LLM clients + drift state + lose tool wiring).
# These constants define a rid-routed bus protocol so api_subprocess can
# forward /chat requests to a chat_handler in the parent process where
# the real agent lives. Pre/post hooks (gatekeeper, memory recall, RL
# recording, OVG, dream inbox) all run in parent — the subprocess only
# does Privy auth + payload assembly + JSON serialization.
#
# Bus dst convention: chat_handler subscribes as "chat_handler"; api
# subprocess sends as src="chat_subproc". Reply rid matches request.
CHAT_REQUEST = "CHAT_REQUEST"      # api_subproc → chat_handler (forward chat)
CHAT_RESPONSE = "CHAT_RESPONSE"    # chat_handler → api_subproc (response)

# ── Microkernel v2 Phase B.1 — Shadow Core Swap ────────────────────
# State-preserving atomic kernel restart with auto-readiness orchestration.
# rFP §347-357 + PLAN_microkernel_phase_b1_shadow_swap.md.
# All flag-gated by microkernel.shadow_swap_enabled (default false).
# Bus dst convention: orchestrator subscribes as "shadow_swap"; workers
# match per-name; lifecycle events broadcast to all so the observatory
# dashboard + spirit (self-aware thoughts) both see them.

# Lifecycle events — broadcast (dst="all"), Titan-aware
SYSTEM_UPGRADE_QUEUED = "SYSTEM_UPGRADE_QUEUED"                  # orchestrator → all (upgrade scheduled)
SYSTEM_UPGRADE_PENDING = "SYSTEM_UPGRADE_PENDING"                # orchestrator → all (still waiting on blockers — emitted every 5s during readiness wait)
SYSTEM_UPGRADE_PENDING_DEFERRED = "SYSTEM_UPGRADE_PENDING_DEFERRED"  # orchestrator → all (120s grace exceeded — upgrade NOT firing, deferred)
SYSTEM_UPGRADE_STARTING = "SYSTEM_UPGRADE_STARTING"              # orchestrator → all (readiness clear, hibernate firing now)
SYSTEM_RESUMED = "SYSTEM_RESUMED"                                # shadow kernel → all (post-swap, on the new kernel's bus)

# Readiness query/response (orchestrator ↔ workers, request/reply with rid)
UPGRADE_READINESS_QUERY = "UPGRADE_READINESS_QUERY"              # orchestrator → all workers
UPGRADE_READINESS_REPORT = "UPGRADE_READINESS_REPORT"            # workers → "shadow_swap" (HARD/SOFT blockers + ETA)

# Hibernation protocol (orchestrator ↔ workers)
HIBERNATE = "HIBERNATE"                                          # orchestrator → all workers (save state + checksum)
HIBERNATE_ACK = "HIBERNATE_ACK"                                  # workers → "shadow_swap" (state_path + checksum + elapsed_ms)
HIBERNATE_CANCEL = "HIBERNATE_CANCEL"                            # orchestrator → all (rollback — resume from hibernation, shadow boot failed)

# ── Microkernel v2 Phase B.2 — Bus IPC Migration ───────────────────
# Unix-domain pub/sub broker; workers connect outward and survive kernel
# swaps. Per PLAN_microkernel_phase_b2_ipc.md §4 (D2/D3) + §5 backpressure.
# All gated by microkernel.bus_ipc_socket_enabled (default false).

# Subscription protocol (worker ↔ broker)
BUS_SUBSCRIBE = "BUS_SUBSCRIBE"          # worker → broker on (re)connect (registers name + topics)
BUS_UNSUBSCRIBE = "BUS_UNSUBSCRIBE"      # worker → broker (drop topics)

# Heartbeat (broker ↔ workers, every PING_INTERVAL_S=5s; 3 missed → broker drops)
BUS_PING = "BUS_PING"                    # broker → worker (every 5s)
BUS_PONG = "BUS_PONG"                    # worker → broker (auto-reply)

# Backpressure observability (broker → all, debounced 1×/60s per slow subscriber)
BUS_SLOW_CONSUMER = "BUS_SLOW_CONSUMER"  # broker → all (when sub's drop_rate_60s > 5%)

# Self-awareness signal (kernel → all, last act before shadow swap exit)
BUS_HANDOFF = "BUS_HANDOFF"              # kernel → all (informs workers a swap is happening; socket reattach is mechanical)

# Phase B.2.1 — worker-supervision-transfer protocol (workers literally outlive kernel swap)
BUS_WORKER_ADOPT_REQUEST = "BUS_WORKER_ADOPT_REQUEST"  # worker → shadow Guardian (rid-routed; payload: name, pid, start_method, boot_ts)
BUS_WORKER_ADOPT_ACK = "BUS_WORKER_ADOPT_ACK"          # shadow Guardian → worker (rid-matched; payload: status="adopted"|"rejected", reason, shadow_pid)
BUS_HANDOFF_CANCELED = "BUS_HANDOFF_CANCELED"          # kernel → all (P-2c unwind: swap aborted, re-arm PDEATHSIG, restore strict watcher)

# Phase C C-S2: canonical message names per SPEC §8.3 + §8.4 + §3 D13/D14/D15.
# Both legacy (above) and canonical names exist during Phase C; the broker
# drift bridge dual-emits so subscribers can listen on either. C-S8 deletes
# the legacy entries. Per PLAN_microkernel_phase_c_s2_kernel.md §12.6.
SWAP_HANDOFF = "SWAP_HANDOFF"                          # canonical for BUS_HANDOFF (D13)
SWAP_HANDOFF_CANCELED = "SWAP_HANDOFF_CANCELED"        # canonical for BUS_HANDOFF_CANCELED (D13)
ADOPTION_REQUEST = "ADOPTION_REQUEST"                  # canonical for BUS_WORKER_ADOPT_REQUEST (D14)
ADOPTION_ACK = "ADOPTION_ACK"                          # canonical for BUS_WORKER_ADOPT_ACK (D14)
KERNEL_EPOCH_TICK = "KERNEL_EPOCH_TICK"                # canonical for EPOCH_TICK (D15)

# V4 Time Awareness messages
OUTER_TRINITY_STATE = "OUTER_TRINITY_STATE"
# A.8.4 — parent → outer_trinity worker (fire-and-forget; sources payload pre-flattened by parent)
OUTER_TRINITY_COLLECT_REQUEST = "OUTER_TRINITY_COLLECT_REQUEST"
SPHERE_PULSE = "SPHERE_PULSE"
BIG_PULSE = "BIG_PULSE"
GREAT_PULSE = "GREAT_PULSE"
# LEGACY:FILTER_DOWN_V4 superseded by FILTER_DOWN_V5 (162D TITAN_SELF).
# V4 FilterDown engine is still instantiated for back-compat but the bus
# message type is no longer published — V5 carries the live multipliers.
FILTER_DOWN_V4 = "FILTER_DOWN_V4"

# V4 Sovereign Reflex Arc messages
CONVERSATION_STIMULUS = "CONVERSATION_STIMULUS"  # chat → workers (input features)
# LEGACY:REFLEX_SIGNAL + REFLEX_RESULT were the v1 reflex arc carrier types.
# Current path: TitanVM evaluates reflexes inline and emits REFLEX_REWARD
# directly to FilterDown. Kept for future multi-worker reflex arbitration.
REFLEX_SIGNAL = "REFLEX_SIGNAL"                  # LEGACY:workers → collector (Intuition signals)
REFLEX_RESULT = "REFLEX_RESULT"                  # LEGACY:collector → consciousness (fired reflexes)
REFLEX_REWARD = "REFLEX_REWARD"                  # TitanVM → FilterDown (interaction reward score)
STATE_SNAPSHOT = "STATE_SNAPSHOT"                  # StateRegister → Spirit (full 30DT for enrichment)
OBSERVABLES_SNAPSHOT = "OBSERVABLES_SNAPSHOT"      # spirit → state_register (rFP #1: 30D space topology + dict)
TITAN_SELF_STATE = "TITAN_SELF_STATE"              # consciousness → broadcast (rFP #2: 162D TITAN_SELF)
FILTER_DOWN_V5 = "FILTER_DOWN_V5"                  # spirit → workers (rFP #2: V5 162D-driven multipliers)

# Trinity Dream Cycle messages
DREAM_STATE_CHANGED = "DREAM_STATE_CHANGED"  # spirit → all (sleep/wake transition)
DREAM_WAKE_REQUEST = "DREAM_WAKE_REQUEST"    # chat_api → spirit (maker gentle wake)

# State-update events for api_subprocess BusSubscriber (microkernel v2 §A.4)
# Consumer maps these to CachedState keys via EVENT_TO_CACHE_KEY in
# titan_plugin/api/bus_subscriber.py. dst="all" so the message reaches the
# api subprocess + any future consumer (e.g. observatory_writer).
CHI_UPDATED = "CHI_UPDATED"                                    # spirit → all (chi.state)
PI_HEARTBEAT_UPDATED = "PI_HEARTBEAT_UPDATED"                  # spirit → all (pi_heartbeat.state)
DREAMING_STATE_UPDATED = "DREAMING_STATE_UPDATED"              # spirit → all (dreaming.state)
META_REASONING_STATS_UPDATED = "META_REASONING_STATS_UPDATED"  # spirit → all (meta_reasoning.state)
MSL_STATE_UPDATED = "MSL_STATE_UPDATED"                        # reserved (msl.state) — no producer yet
SOLANA_BALANCE_UPDATED = "SOLANA_BALANCE_UPDATED"              # kernel → all (network.balance)
MEMORY_STATUS_UPDATED = "MEMORY_STATUS_UPDATED"                # memory → all (memory.status)
MEMORY_MEMPOOL_UPDATED = "MEMORY_MEMPOOL_UPDATED"              # memory → all (memory.mempool)
MEMORY_TOP_UPDATED = "MEMORY_TOP_UPDATED"                      # memory → all (memory.top)
MEMORY_TOPOLOGY_UPDATED = "MEMORY_TOPOLOGY_UPDATED"            # memory → all (memory.topology)
MEMORY_KNOWLEDGE_GRAPH_UPDATED = "MEMORY_KNOWLEDGE_GRAPH_UPDATED"  # memory → all (memory.knowledge_graph)
TOPOLOGY_STATE_UPDATED = "TOPOLOGY_STATE_UPDATED"              # spirit → all (topology.state — Batch E)
REASONING_STATS_UPDATED = "REASONING_STATS_UPDATED"            # spirit → all (reasoning.state)
EXPRESSION_COMPOSITES_UPDATED = "EXPRESSION_COMPOSITES_UPDATED"  # spirit → all (expression.composites)
NEUROMOD_STATS_UPDATED = "NEUROMOD_STATS_UPDATED"              # spirit → all (neuromods.full)
# rFP_observatory_data_loading_v1 Phase 4 — declared here so producer
# wiring (Persona Social / Language Teacher / Meta-Teacher / CGN periodic
# stats publishers) can import the constant. Until producers ship, these
# are unused at runtime; the cache_key_registry tracks each as kind="missing".
SOCIAL_STATS_UPDATED = "SOCIAL_STATS_UPDATED"                  # persona → all (social.stats)
LANGUAGE_STATS_UPDATED = "LANGUAGE_STATS_UPDATED"              # language teacher → all (language.stats)
META_TEACHER_STATS_UPDATED = "META_TEACHER_STATS_UPDATED"      # meta-teacher → all (meta_teacher.stats)
CGN_STATS_UPDATED = "CGN_STATS_UPDATED"                        # cgn → all (cgn.stats)

# M8: External Intent (wallet observer → spirit → neuromod boost)
# RESERVED: EXTERNAL_INTENT — WalletObserver helper class exists but
# polling loop not yet wired into boot. Donation → neuromod boost path
# awaits wallet_observer spawn site (tracked in mainnet master plan M8).
EXTERNAL_INTENT = "EXTERNAL_INTENT"              # wallet_observer → spirit (DI/I/donation)

# Observatory V2: real-time frontend events (spirit → v4_bridge → WebSocket)
NEUROMOD_UPDATE = "NEUROMOD_UPDATE"      # spirit → v4_bridge (every Tier 2 tick)
HORMONE_FIRED = "HORMONE_FIRED"          # spirit → v4_bridge (on program fire)
EXPRESSION_FIRED = "EXPRESSION_FIRED"    # spirit → v4_bridge (on composite fire)

# Language Worker messages (spirit ↔ language process)
SPEAK_REQUEST = "SPEAK_REQUEST"          # spirit → language (compose from felt-state)
SPEAK_RESULT = "SPEAK_RESULT"            # language → spirit (sentence + perturbation deltas)
TEACHER_SIGNALS = "TEACHER_SIGNALS"      # language → spirit (MSL signals + vocab updates)
LANGUAGE_STATS_UPDATE = "LANGUAGE_STATS_UPDATE"  # language → spirit (periodic stats broadcast)

# Query/Response (sync via request-id)
QUERY = "QUERY"
RESPONSE = "RESPONSE"

# META-CGN cross-consumer signal (v3 rewire, rFP_meta_cgn_v3_clean_rewire.md).
# Emitted via emit_meta_cgn_signal() helper below — NEVER use make_msg directly
# for this type, because the helper enforces the architectural invariants:
# edge-detected triggers, rate budget, SIGNAL_TO_PRIMITIVE mapping check.
META_CGN_SIGNAL = "META_CGN_SIGNAL"

# EMOT-CGN cross-consumer signal (rFP_emot_cgn_v2.md).
# Emitted via emit_emot_cgn_signal() helper below — same invariant
# guarantees as META-CGN (orphan detection + rate gate). Consumed by
# handle_cross_consumer_signal in spirit_worker which routes to
# meta_engine._emot_cgn (pre-Phase-1.6e) OR the emot_cgn worker (post).
EMOT_CGN_SIGNAL = "EMOT_CGN_SIGNAL"

# EMOT-CGN worker inputs (rFP_emot_cgn_v2.md §10 standalone-worker ADR,
# Phase 1.6d). Producer → emot_cgn_worker messages for the EVENT channel.
# STATE queries go through shm-mirror (`emot_shm_protocol.py`), NOT the bus.
#
# EMOT_CHAIN_EVIDENCE: meta_reasoning emits per chain conclude.
#   src="spirit", dst="emot_cgn"
#   payload: {chain_id, dominant_at_start, dominant_at_end, terminal_reward, ctx}
# FELT_CLUSTER_UPDATE: spirit_worker emits per felt-tensor emit (or
#   meta_reasoning._start_chain for the current simpler path).
#   src="spirit", dst="emot_cgn"
#   payload: {feature_vec_150d: list[float] OR felt_tensor_130d: list[float]}
EMOT_CHAIN_EVIDENCE = "EMOT_CHAIN_EVIDENCE"
FELT_CLUSTER_UPDATE = "FELT_CLUSTER_UPDATE"

# CGN lightweight state snapshot for consumption by TitanVM programs
# (rFP_titan_vm_v2 Phase 2 §3.8). cgn_worker emits every 10 ticks with a
# dict payload: {grounded_density, active_haovs, reasoning_V, language_V,
# social_V, emotional_V, coding_V, last_ground_ts_<consumer>, …}.
# Consumer: spirit_worker subscriber writes payload into
# state_register.cgn_state; InnerTrinityCoordinator threads the snapshot
# into nervous_system.evaluate() as the "cgn.*" observable namespace.
# Routing: src="cgn", dst="spirit". Not rate-limited (once every 10 cgn
# ticks ≈ 0.2 Hz — negligible bus traffic).
CGN_STATE_SNAPSHOT = "CGN_STATE_SNAPSHOT"

# CGN_BETA_SNAPSHOT — cgn_worker emits periodic per-consumer V snapshot
# (§23.6a of rFP_emot_cgn_v2). Payload: {"values_by_consumer": {name: V,
# ...}} where V is the 64-window reward EMA per consumer (proxy for
# dominant Q-value), ordered for the 8 CGN_CONSUMERS. emot_cgn_worker
# consumes this to populate bundle's cgn_beta_states_8d (previously
# dead — 8/210 dims of HDBSCAN input). Emitted at same cadence as
# CGN_STATE_SNAPSHOT (every N transitions, ~0.2 Hz). Routing: src="cgn",
# dst="emot_cgn".
CGN_BETA_SNAPSHOT = "CGN_BETA_SNAPSHOT"

# Bus backpressure — edge-detected when worker queue depths cross > 30%
# (enter) or < 20% (exit) threshold. Published by BusHealthMonitor only on
# state transitions. Respects bus-clean invariant (discrete events only).
# RESERVED: BusHealthMonitor.update_queue_depths is defined but currently
# unused (also surfaced in orphan list). Constant reserved for when the
# monitor's periodic sample loop is wired.
BUS_BACKPRESSURE = "BUS_BACKPRESSURE"

# ── Meta-Reasoning Consumer Service Layer (F-phase rFP) ─────────────
# Bidirectional consumer↔meta request/response/outcome protocol.
# See `titan-docs/rFP_meta_service_interface.md` §4.
#
# Routing invariants (per `feedback_bus_dst_must_have_subscriber.md`):
#   - META_REASON_REQUEST  : dst="spirit" — meta_service lives in spirit_worker
#   - META_REASON_RESPONSE : dst=<consumer_home_worker> — usually "spirit"
#                            (social, emot, dreaming, reflection, self_model live
#                            there); "language", "knowledge", "cgn" own their
#                            workers. Client helper resolves the mapping.
#   - META_REASON_OUTCOME  : dst="spirit" — meta accumulates signed reward per
#                            (consumer, primitive, sub_mode) tuple.
#
# Payload shapes are documented in rFP §4.1. Schema validation lives in
# `titan_plugin/logic/meta_service_client.py`.
META_REASON_REQUEST = "META_REASON_REQUEST"
META_REASON_RESPONSE = "META_REASON_RESPONSE"
META_REASON_OUTCOME = "META_REASON_OUTCOME"

# ── TimeChain SIMILAR primitive (F-phase §9.2) ──────────────────────
# Semantic-embedding similarity search over TimeChain blocks. Only genuinely-new
# TimeChain primitive (RECALL/CHECK/COMPARE/AGGREGATE already ship per rFP
# §9.1). dst="timechain" (timechain_worker); response type TIMECHAIN_SIMILAR_RESP
# returned to src.
TIMECHAIN_SIMILAR = "TIMECHAIN_SIMILAR"
TIMECHAIN_SIMILAR_RESP = "TIMECHAIN_SIMILAR_RESP"

# ── Meta-Reasoning Teacher (rFP_titan_meta_reasoning_teacher.md) ─────
# Philosopher-critic observer for Titan's meta-reasoning chains.
#
# META_CHAIN_COMPLETE: spirit (meta_reasoning._conclude_chain) emits after
#   each chain conclude. dst="all" — meta_teacher subscribes; also useful
#   for observability. Emitted unconditionally (cheap); teacher worker
#   decides whether to critique (sampling + rate cap).
#
# META_TEACHER_FEEDBACK: meta_teacher_worker emits after critique.
#   dst="spirit" — spirit_worker routes to chain_iql.apply_external_reward
#   and publishes to observers. Gated on config enabled=true.
#
# META_TEACHER_GROUNDING: meta_teacher_worker emits per-primitive grounding
#   nudge after critique. dst="spirit" — spirit_worker routes to
#   meta_cgn.handle_teacher_grounding (Beta posterior nudge, mirrors
#   handle_incoming_cross_insight weighting).
META_CHAIN_COMPLETE = "META_CHAIN_COMPLETE"
META_TEACHER_FEEDBACK = "META_TEACHER_FEEDBACK"
META_TEACHER_GROUNDING = "META_TEACHER_GROUNDING"
# Phase D.1 (rFP_meta_teacher_v2): cross-Titan teaching exchange. Network
# transport is HTTP (DivineBus is in-process only); these bus types exist
# for INTRA-Titan observability — telemetry of issued/answered queries
# surfaces in bus census and can be inspected without opening the JSONL log.
# INTENTIONAL_BROADCAST: dst="all" so both observatory + meta-teacher worker
# can subscribe without coupling.
META_TEACHER_PEER_QUERY = "META_TEACHER_PEER_QUERY"
META_TEACHER_PEER_RESPONSE = "META_TEACHER_PEER_RESPONSE"


def make_msg(
    msg_type: str,
    src: str,
    dst: str,
    payload: Optional[dict] = None,
    rid: Optional[str] = None,
) -> dict:
    """Create a properly formatted bus message."""
    return {
        "type": msg_type,
        "src": src,
        "dst": dst,
        "ts": time.time(),
        "rid": rid,
        "payload": payload or {},
    }


def make_request(src: str, dst: str, payload: dict) -> dict:
    """Create a QUERY message with a unique request ID for sync request/response."""
    return make_msg(QUERY, src, dst, payload, rid=str(uuid.uuid4()))


class DivineBus:
    """
    Central message router using Queue per module.

    Uses threading.Queue for in-process communication (reliable, fast).
    Uses multiprocessing.Queue when cross-process IPC is needed.
    Each module subscribes by name and gets its own Queue.
    Publishing routes to the destination module's queue(s) or broadcasts to all.
    """

    # State message types → routed to blackboard (latest-value, no queue pressure)
    STATE_MSG_TYPES = {
        BODY_STATE, MIND_STATE, SPIRIT_STATE,
        OUTER_TRINITY_STATE, SPHERE_PULSE, FILTER_DOWN_V4,
    }

    def __init__(self, maxsize: int = 1000, multiprocess: bool = False):
        self._maxsize = maxsize
        self._multiprocess = multiprocess
        self._blackboard = SharedBlackboard()
        # Microkernel v2 Phase B.2 — optional Unix-socket pub/sub broker.
        # Set via attach_broker() after kernel boots. When non-None, publish()
        # ALSO routes to the broker so worker-side BusSocketClient subscribers
        # (in separate processes) receive the message. Internal subscribers
        # (in-process kernel components) still get the message via ThreadQueue
        # in the legacy code path. None = legacy mode (no socket bus).
        self._broker = None
        # module_name → list[Queue]  (a module may have multiple subscribers)
        self._subscribers: dict[str, list[AnyQueue]] = {}
        # All registered module names (for broadcast)
        self._modules: set[str] = set()
        # Modules that only receive targeted messages (excluded from dst="all" broadcasts)
        self._reply_only: set[str] = set()
        # 2026-04-29 Option B — per-subscriber msg_type filter for broadcasts.
        # Map: module_name → frozenset of msg_types that subscriber WANTS
        # to receive via dst="all" broadcast. Missing key = legacy wildcard
        # (receive every broadcast). Present key (incl. empty frozenset)
        # = filter applied; broadcasts whose msg_type is not in the set are
        # dropped at publish time, before the queue is touched. Targeted
        # dst="<name>" messages (RPC, replies) bypass this filter completely.
        # Discovered via 1.2M queue-full drops on T1 in v2 mode caused by
        # spirit_loop fan-out broadcasting ~12 _UPDATED events to dst="all"
        # → every subscriber's queue (v4_bridge, guardian, rl_proxy_stats,
        # ...) flooded with msgs they don't consume → MODULE_HEARTBEAT drops
        # at guardian, DREAMING_STATE_UPDATED drops at api → /health.is_dreaming
        # missing.
        self._broadcast_filters: dict[str, frozenset[str]] = {}
        # 2026-04-26 — RLock protecting _subscribers / _modules / _reply_only.
        # subscribe()/unsubscribe()/publish() all touch these from different
        # threads (kernel-boot subscribe vs balance-publisher / spirit_worker
        # / api-bus-listener publish). Without the lock, publish()'s
        # iteration over _subscribers.items() could race with a concurrent
        # subscribe() and raise "dictionary changed size during iteration".
        # H4 first-deploy on T1 hit this exact race during boot and left
        # api_subprocess unable to bind :7777. RLock so re-entrant publish
        # paths (e.g. _try_put → census callback) don't deadlock.
        self._lock = threading.RLock()
        self._stats = {
            "published": 0, "dropped": 0, "routed": 0,
            # Option B (2026-04-29): broadcasts dropped by per-subscriber
            # type filter — zero queue cost, separate from "dropped" (which
            # counts queue-full drops).
            "filtered_broadcasts": 0,
        }
        # Optional callback called during request() polling to drain worker send queues.
        # Set by TitanCore to guardian.drain_send_queues so responses flow back.
        self._poll_fn: Optional[Callable] = None
        # Per-(src,dst) timeout warning throttle (audit fix I-014, 2026-04-08).
        # Prevents boot-time race spam from masking real persistent timeout issues.
        # L3 Phase A.8.1: OrderedDict + cap at 2048 entries (FIFO eviction) so
        # stale (src,dst) pairs from long-dead modules can't accumulate.
        self._timeout_warned_at: "OrderedDict[tuple[str, str], float]" = OrderedDict()
        self._TIMEOUT_WARNED_MAX = 2048
        # Bus census (Phase E.1): if enabled via TITAN_BUS_CENSUS=1, register
        # this bus as the depth sampler so the census writer can sample
        # qsize() of every subscriber queue periodically.
        _census.register_depth_sampler(self.sample_census_depths)

    def subscribe(self, module_name: str, reply_only: bool = False,
                  types: Optional[Iterable[str]] = None) -> AnyQueue:
        """Register a module and return its dedicated receive queue.

        Args:
            module_name: Unique name for this subscriber.
            reply_only: If True, this queue only receives targeted messages
                        (e.g. RESPONSE with dst=module_name) and is excluded
                        from dst="all" broadcasts.  Use for proxy reply queues
                        that never consume broadcast state messages.
            types: Option B (2026-04-29) — opt-in msg_type filter for
                   dst="all" broadcasts. None (default) = legacy wildcard
                   (receive every broadcast — required during incremental
                   migration). Iterable of msg_type strings = receive only
                   those types via broadcast. Empty iterable = receive no
                   broadcasts (similar to reply_only=True for the broadcast
                   path; reply_only is preferred when targeted msgs are
                   also unwanted). Targeted dst="<module_name>" messages
                   are ALWAYS delivered regardless of this filter — RPC,
                   replies, and direct sends never filtered.

                   When two subscribe() calls use the same module_name with
                   different types, the union is taken (any caller listed
                   in types receives that msg_type). reply_only union also
                   applies — once a name is reply_only, it stays that way.
        """
        if self._multiprocess:
            q: AnyQueue = MPQueue(maxsize=self._maxsize)
        else:
            q = ThreadQueue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.setdefault(module_name, []).append(q)
            self._modules.add(module_name)
            if reply_only:
                self._reply_only.add(module_name)
            # Filter merge (union) so dual-path subscribe sites (legacy_core
            # vs core/plugin for the same name) don't accidentally narrow
            # each other's filter. Wildcard (types=None) does NOT install
            # a filter — keeps full backward-compat for un-migrated callers.
            if types is not None:
                new_set = frozenset(types)
                existing = self._broadcast_filters.get(module_name)
                if existing is None:
                    self._broadcast_filters[module_name] = new_set
                else:
                    self._broadcast_filters[module_name] = existing | new_set
            queue_count = len(self._subscribers[module_name])
            filter_size = (
                len(self._broadcast_filters[module_name])
                if module_name in self._broadcast_filters else None
            )
        logger.info(
            "[DivineBus] Module '%s' subscribed (queues: %d, reply_only=%s, "
            "broadcast_filter=%s)",
            module_name, queue_count, reply_only,
            "wildcard" if filter_size is None else f"{filter_size} types",
        )
        return q

    def unsubscribe(self, module_name: str, q: AnyQueue) -> None:
        """Remove a specific queue for a module."""
        with self._lock:
            if module_name in self._subscribers:
                try:
                    self._subscribers[module_name].remove(q)
                except ValueError:
                    pass
                if not self._subscribers[module_name]:
                    del self._subscribers[module_name]
                    self._modules.discard(module_name)
                    # Clear filter + reply_only when the last queue for
                    # this name goes away — keeps state in lockstep with
                    # subscription lifetime.
                    self._broadcast_filters.pop(module_name, None)
                    self._reply_only.discard(module_name)
        logger.info("[DivineBus] Module '%s' unsubscribed", module_name)

    @property
    def blackboard(self) -> SharedBlackboard:
        """Access the shared blackboard for latest-value state reads."""
        return self._blackboard

    def get_broadcast_filter(self, module_name: str) -> Optional[frozenset[str]]:
        """Option B introspection — return the broadcast filter for a name.

        Returns:
            None if the subscriber has no filter (legacy wildcard — receives
            every broadcast). frozenset of msg_type strings otherwise (incl.
            empty set, which means receive no broadcasts).

        Used by tests + `arch_map dead-wiring` to verify the declared filter
        matches the msg_types the subscriber actually handles in code.
        """
        with self._lock:
            return self._broadcast_filters.get(module_name)

    def publish(self, msg: dict) -> int:
        """
        Route a message to destination queue(s).
        State messages also written to blackboard (latest-value).

        Returns the number of queues that received the message.
        """
        dst = msg.get("dst", "")
        msg_type = msg.get("type", "")
        self._stats["published"] += 1
        _census.record_emission(msg_type, dst)
        delivered = 0

        # State messages → write to blackboard (latest-value, zero backpressure)
        if msg_type in self.STATE_MSG_TYPES:
            bb_key = f"{msg.get('src', 'unknown')}_{msg_type}"
            self._blackboard.write(bb_key, msg)

        if dst == "all":
            # Broadcast to every registered module (except sender and reply-only).
            # Hold _lock during iteration to prevent "dictionary changed size
            # during iteration" if subscribe()/unsubscribe() runs concurrently.
            # _try_put uses put_nowait (non-blocking) so the critical section
            # stays short — no Queue.put backpressure inside the lock.
            #
            # Option B (2026-04-29): also skip subscribers whose broadcast
            # filter excludes msg_type. Filter check is fast (frozenset
            # membership) so it stays inside the lock with minimal cost,
            # and filtered drops never touch the queue → zero backpressure.
            src = msg.get("src", "")
            filtered_count = 0
            with self._lock:
                snapshot = []
                for mod_name, queues in self._subscribers.items():
                    if mod_name == src:
                        continue
                    if mod_name in self._reply_only:
                        continue
                    flt = self._broadcast_filters.get(mod_name)
                    if flt is not None and msg_type not in flt:
                        filtered_count += 1
                        continue
                    snapshot.append((mod_name, list(queues)))
            if filtered_count:
                self._stats["filtered_broadcasts"] += filtered_count
            for mod_name, queues in snapshot:
                for q in queues:
                    if self._try_put(q, msg, subscriber=mod_name):
                        delivered += 1
        else:
            with self._lock:
                queues = list(self._subscribers.get(dst, []))
            if queues:
                for q in queues:
                    if self._try_put(q, msg, subscriber=dst):
                        delivered += 1
            else:
                logger.debug("[DivineBus] No subscriber for dst='%s', msg type=%s",
                             dst, msg.get("type"))

        # Microkernel v2 Phase B.2 — fan out to the Unix-socket broker
        # (worker subscribers connected externally). Internal subscribers
        # already received the message via the legacy queue path above; the
        # broker handles its own subscriber list (workers in separate
        # processes) without us tracking them here. Failures are logged but
        # do not affect the in-process delivery count.
        broker = self._broker
        if broker is not None:
            try:
                broker.publish(msg)
            except Exception:  # noqa: BLE001
                logger.exception("[DivineBus] broker.publish raised; in-process delivery unaffected")

        self._stats["routed"] += delivered
        return delivered

    def publish_in_process(self, msg: dict) -> int:
        """Phase B.2.1 (2026-04-27) — broker callback path.

        Same as publish() but does NOT re-forward to the socket broker.
        Called by BusSocketServer when a worker publishes a message that
        kernel-side in-process subscribers (shadow_swap orchestrator,
        Guardian, etc.) need to see. Skipping the broker forward is what
        prevents the worker → broker → kernel → broker → worker loop.

        Returns the number of in-process subscribers that received the msg.
        """
        msg = dict(msg)  # defensive copy
        msg.setdefault("ts", time.time())
        src = msg.get("src", "?")
        dst = msg.get("dst", "all")
        msg_type = msg.get("type", "")

        with self._lock:
            self._stats["published"] += 1

        delivered = 0
        if dst == "all":
            # Option B filter applied here too — broker-callback path is
            # the v2 mode's hot route from worker → kernel-side subscribers.
            filtered_count = 0
            with self._lock:
                snapshot = []
                for mod_name, queues in self._subscribers.items():
                    if mod_name == src:
                        continue
                    if mod_name in self._reply_only:
                        continue
                    flt = self._broadcast_filters.get(mod_name)
                    if flt is not None and msg_type not in flt:
                        filtered_count += 1
                        continue
                    snapshot.append((mod_name, list(queues)))
            if filtered_count:
                self._stats["filtered_broadcasts"] += filtered_count
            for mod_name, queues in snapshot:
                for q in queues:
                    if self._try_put(q, msg, subscriber=mod_name):
                        delivered += 1
        else:
            with self._lock:
                queues = list(self._subscribers.get(dst, []))
            for q in queues:
                if self._try_put(q, msg, subscriber=dst):
                    delivered += 1
        self._stats["routed"] += delivered
        return delivered

    # ── Microkernel v2 Phase B.2 — broker attach/detach ──────────────────

    def attach_broker(self, broker) -> None:
        """Attach a started BusSocketServer so publish() also fans out to
        cross-process subscribers. Idempotent (re-attach replaces).

        Called by TitanKernel._start_bus_socket_broker after the broker is
        bound and accept loop is running. Not used in legacy (mp.Queue) mode.
        """
        self._broker = broker
        logger.info("[DivineBus] socket broker attached (path=%s)",
                    getattr(broker, "sock_path", "?"))

    def detach_broker(self) -> None:
        """Detach the broker (kernel shutdown). Bus reverts to in-process-only
        delivery for any final messages. Idempotent."""
        if self._broker is not None:
            logger.info("[DivineBus] socket broker detached")
            self._broker = None

    @property
    def has_socket_broker(self) -> bool:
        """True if a Unix-socket broker is attached (Phase B.2 mode)."""
        return self._broker is not None

    def _try_put(self, q: AnyQueue, msg: dict, subscriber: str = "?") -> bool:
        """Non-blocking put. Returns True on success, False if queue full."""
        try:
            q.put_nowait(msg)
            return True
        except Full:
            self._stats["dropped"] += 1
            _census.record_drop(subscriber, msg.get("type", ""))
            logger.warning("[DivineBus] Queue full for '%s', dropped msg type=%s dst=%s",
                           subscriber, msg.get("type"), msg.get("dst"))
            return False

    def drain(self, q: AnyQueue, max_msgs: int = 100) -> list[dict]:
        """Read up to max_msgs from a queue (non-blocking)."""
        msgs = []
        for _ in range(max_msgs):
            try:
                msgs.append(q.get_nowait())
            except Empty:
                break
        if msgs:
            _census.record_drain("?", len(msgs))
        return msgs

    def sample_census_depths(self) -> None:
        """Sample qsize() of all subscriber queues into census log.

        Called periodically by Guardian when TITAN_BUS_CENSUS=1.
        Safe to call from parent process; worker calls only see own queues.
        Snapshots _subscribers under lock so the external consumer can
        iterate without racing concurrent subscribe()/unsubscribe().
        """
        with self._lock:
            snapshot = {k: list(v) for k, v in self._subscribers.items()}
        _census.sample_queue_depths(snapshot)

    def request(
        self, src: str, dst: str, payload: dict, timeout: float = 10.0, reply_queue: Optional[AnyQueue] = None
    ) -> Optional[dict]:
        """
        Synchronous request/response over the bus.

        Publishes a QUERY, waits for a matching RESPONSE on the reply_queue.
        Caller must provide their own reply_queue (from subscribe()).
        """
        msg = make_request(src, dst, payload)
        rid = msg["rid"]
        if reply_queue is None:
            logger.error("[DivineBus] request() requires a reply_queue")
            return None

        self.publish(msg)

        deadline = time.time() + timeout
        while time.time() < deadline:
            # Drain worker send queues so responses flow back through bus
            if self._poll_fn:
                try:
                    self._poll_fn()
                except Exception:
                    pass
            try:
                reply = reply_queue.get(timeout=min(0.2, max(0.01, deadline - time.time())))
                if reply.get("type") == RESPONSE and reply.get("rid") == rid:
                    return reply
                # Not our reply — only keep RESPONSE messages (discard stale broadcasts)
                if reply.get("type") == RESPONSE:
                    self._try_put(reply_queue, reply)
                # else: silently discard broadcast messages that leaked into reply queue
            except Empty:
                continue

        # 2026-04-08 audit fix (I-014): rate-limit timeout warnings per (src, dst)
        # pair to avoid noise from boot-time race conditions (e.g., memory_proxy
        # queries firing before memory is fully initialized — common on T3 boot).
        # Real persistent timeout issues will still fire (every 30s), preserving
        # signal on T1 mainnet. Only the first occurrence of each pair logs as
        # WARNING; subsequent within the throttle window log at DEBUG.
        pair = (src, dst)
        now = time.time()
        last = self._timeout_warned_at.get(pair, 0)
        if now - last >= 30.0:
            logger.warning("[DivineBus] Request timed out: %s → %s (rid=%s)", src, dst, rid)
            # L3 Phase A.8.1: OrderedDict + FIFO cap at 2048 entries.
            self._timeout_warned_at[pair] = now
            self._timeout_warned_at.move_to_end(pair)
            while len(self._timeout_warned_at) > self._TIMEOUT_WARNED_MAX:
                self._timeout_warned_at.popitem(last=False)
        else:
            logger.debug("[DivineBus] Request timed out (throttled): %s → %s (rid=%s)",
                         src, dst, rid)
        return None

    async def request_async(
        self,
        src: str,
        dst: str,
        payload: dict,
        timeout: float = 10.0,
        reply_queue: Optional[AnyQueue] = None,
    ) -> Optional[dict]:
        """Async wrapper around request() — routes the blocking reply wait
        through the dedicated bus_ipc_pool (NOT the default asyncio pool).

        WHY: see module-level get_bus_ipc_pool() docstring. Bus reply waits
        are latency-sensitive (3s default timeout) and must not queue behind
        Observatory snapshot builds (1-5s) sharing the default 64-worker
        pool. Pool saturation observed 2026-04-29 hit 250% (64 busy + 192
        queued) on T1 — bus replies were timing out as a cascade.

        ALL async proxy paths (`agency_proxy.handle_intent`,
        `assessment_proxy.assess`, `rl_proxy.dream`, etc.) MUST call this,
        not `await asyncio.to_thread(self.request, ...)`. Pre-commit
        scanner enforces — see scripts/scan_bus_request_async_misuse.py.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_bus_ipc_pool(),
            self.request,
            src, dst, payload, timeout, reply_queue,
        )

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def modules(self) -> set[str]:
        return set(self._modules)


# ----------------------------------------------------------------------
# emit_meta_cgn_signal — v3 rewire helper
# ----------------------------------------------------------------------
# Replaces the Phase 2 helper (which was reverted in commit f19a354).
# Enforces the three architectural invariants required by
# rFP_meta_cgn_v3_clean_rewire.md:
#
#   1. Rate budget — each call passes a min_interval_s; emissions faster
#      than that are silently dropped (rate_drop counter incremented).
#   2. SIGNAL_TO_PRIMITIVE mapping check — producer calls for an unmapped
#      (consumer, event_type) tuple are dropped with WARN on first
#      occurrence; prevents orphan signals (the Phase 2 5-orphan bug).
#   3. BusHealthMonitor recording — every accepted emission is counted
#      and surfaced via /v4/bus-health.

_emit_gate_last_ts: dict[tuple, float] = {}
import threading as _threading
_emit_gate_lock = _threading.Lock()


def emit_meta_cgn_signal(
    sender,
    src: str,
    consumer: str,
    event_type: str,
    intensity: float = 1.0,
    domain: Optional[str] = None,
    narrative_context: Optional[dict] = None,
    reason: Optional[str] = None,
    min_interval_s: float = 0.5,
) -> bool:
    """Emit a META_CGN_SIGNAL with architectural invariants enforced.

    Polymorphic `sender` argument (duck-typed):
      - If it has `put_nowait` — treated as a worker send_queue (multiprocessing.Queue).
        Message is put on the queue; Guardian drain loop forwards to bus and records
        emission in parent's BusHealthMonitor.
      - If it has `publish` — treated as the main-process DivineBus. Message is
        published directly; record_emission is called here since we're already
        in the parent process where the monitor singleton lives.

    This duck-typed dispatch keeps all META-CGN invariants (orphan check, rate
    gate, schema, record_emission) in ONE place regardless of which emission
    path (worker or main-process endpoint) triggered the signal.

    Returns True if the emission was sent, False if dropped (rate gate,
    orphan, or queue-full). Callers don't need to handle the return —
    drops are surfaced via /v4/bus-health."""
    # 1. Mapping check — refuse to emit orphan signals.
    try:
        from .logic.meta_cgn import SIGNAL_TO_PRIMITIVE
    except ImportError:
        SIGNAL_TO_PRIMITIVE = None

    if SIGNAL_TO_PRIMITIVE is not None and (consumer, event_type) not in SIGNAL_TO_PRIMITIVE:
        try:
            from .core.bus_health import get_global_monitor
            m = get_global_monitor()
            if m is not None:
                m.record_orphan(consumer, event_type)
        except Exception:
            pass
        logger.warning(
            "[emit_meta_cgn_signal] REFUSED orphan emission from %s: "
            "(%s, %s) has no SIGNAL_TO_PRIMITIVE mapping. reason=%s",
            src, consumer, event_type, reason,
        )
        return False

    # 2. Rate gate — drop if we emitted too recently for this tuple.
    key = (src, consumer, event_type)
    now = time.time()
    with _emit_gate_lock:
        last = _emit_gate_last_ts.get(key, 0.0)
        if now - last < min_interval_s:
            try:
                from .core.bus_health import get_global_monitor
                m = get_global_monitor()
                if m is not None:
                    m.record_rate_drop(src, consumer, event_type)
            except Exception:
                pass
            return False
        _emit_gate_last_ts[key] = now

    # 3. Build message (shape identical for both transports).
    payload = {
        "consumer": consumer,
        "event_type": event_type,
        "intensity": float(max(0.0, min(1.0, intensity))),
    }
    if domain is not None:
        payload["domain"] = str(domain)[:40]
    if narrative_context is not None:
        payload["narrative_context"] = narrative_context
    if reason is not None:
        payload["reason"] = str(reason)[:120]

    # dst="spirit": META_CGN_SIGNAL is consumed by handle_cross_consumer_signal
    # at spirit_worker.py:8526, which runs inside the "spirit" subprocess
    # (meta_engine + _meta_cgn live there). No module is registered as "meta",
    # so dst="meta" silently dropped at DivineBus.publish (routing step).
    # Caught 2026-04-19: 14k+ emissions observed by Guardian drain but
    # signals_received=0 on meta_cgn — see HAOV signal-starvation fix.
    msg = {
        "type": META_CGN_SIGNAL,
        "src": src,
        "dst": "spirit",
        "ts": now,
        "rid": None,
        "payload": payload,
    }

    # 4. Dispatch via duck-typed sender.
    _is_main_bus = False
    try:
        if hasattr(sender, "put_nowait"):
            # Worker path: multiprocessing.Queue-like. Guardian drain loop
            # will observe + record emission in parent's BusHealthMonitor.
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            # Main-process path: DivineBus. Publish directly; we'll record
            # emission here (step 5) since Guardian drain only observes
            # worker send_queues.
            sender.publish(msg)
            _is_main_bus = True
        else:
            logger.warning(
                "[emit_meta_cgn_signal] Unknown sender type %s — cannot dispatch "
                "(%s, %s) from %s",
                type(sender).__name__, consumer, event_type, src)
            return False
    except Exception as _disp_err:
        swallow_warn(
            f"[emit_meta_cgn_signal] dispatch failed ({consumer}, "
            f"{event_type}) from {src}",
            _disp_err, key="bus.emit_meta_cgn_signal")
        return False

    # 5. Record successful emission.
    # Worker path: Guardian drain also calls record_emission → we'd double-count
    # if we did it here too. Skip it. For main-process path, we ARE the source
    # of the observation, so record here.
    try:
        from .core.bus_health import get_global_monitor
        m = get_global_monitor()
        if m is not None and _is_main_bus:
            m.record_emission(src, consumer, event_type, intensity)
    except Exception:
        pass

    return True


# ----------------------------------------------------------------------
# emit_emot_cgn_signal — EMOT-CGN signal emission helper (rFP_emot_cgn_v2)
# ----------------------------------------------------------------------
# Mirrors emit_meta_cgn_signal exactly but uses EMOT_CGN_SIGNAL type +
# EMOT_SIGNAL_TO_PRIMITIVE mapping. This gives us the same orphan
# detection guard for free — every producer path goes through this
# single choke point.
#
# dst="spirit": EMOT_CGN_SIGNAL is consumed by handle_cross_consumer_signal
# in spirit_worker which routes to meta_engine._emot_cgn.


def emit_emot_cgn_signal(
    sender,
    src: str,
    consumer: str,
    event_type: str,
    intensity: float = 1.0,
    domain: Optional[str] = None,
    narrative_context: Optional[dict] = None,
    reason: Optional[str] = None,
    min_interval_s: float = 0.5,
) -> bool:
    """Emit an EMOT_CGN_SIGNAL with same invariant guards as emit_meta_cgn_signal.

    Returns True if sent, False if dropped (rate gate, orphan, or queue-full).
    """
    # 1. Mapping check — refuse orphan emissions.
    try:
        from .logic.emot_cgn import EMOT_SIGNAL_TO_PRIMITIVE
    except ImportError:
        EMOT_SIGNAL_TO_PRIMITIVE = None

    if EMOT_SIGNAL_TO_PRIMITIVE is not None and (consumer, event_type) not in EMOT_SIGNAL_TO_PRIMITIVE:
        try:
            from .core.bus_health import get_global_monitor
            m = get_global_monitor()
            if m is not None:
                m.record_orphan(consumer, event_type)
        except Exception:
            pass
        logger.warning(
            "[emit_emot_cgn_signal] REFUSED orphan emission from %s: "
            "(%s, %s) has no EMOT_SIGNAL_TO_PRIMITIVE mapping. reason=%s",
            src, consumer, event_type, reason,
        )
        return False

    # 2. Rate gate.
    key = ("emot", src, consumer, event_type)
    now = time.time()
    with _emit_gate_lock:
        last = _emit_gate_last_ts.get(key, 0.0)
        if now - last < min_interval_s:
            try:
                from .core.bus_health import get_global_monitor
                m = get_global_monitor()
                if m is not None:
                    m.record_rate_drop(src, consumer, event_type)
            except Exception:
                pass
            return False
        _emit_gate_last_ts[key] = now

    # 3. Build message.
    payload = {
        "consumer": consumer,
        "event_type": event_type,
        "intensity": float(max(0.0, min(1.0, intensity))),
    }
    if domain is not None:
        payload["domain"] = str(domain)[:40]
    if narrative_context is not None:
        payload["narrative_context"] = narrative_context
    if reason is not None:
        payload["reason"] = str(reason)[:120]

    msg = {
        "type": EMOT_CGN_SIGNAL,
        "src": src,
        "dst": "spirit",
        "ts": now,
        "rid": None,
        "payload": payload,
    }

    # 4. Dispatch.
    _is_main_bus = False
    try:
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
            _is_main_bus = True
        else:
            logger.warning(
                "[emit_emot_cgn_signal] Unknown sender type %s — cannot dispatch "
                "(%s, %s) from %s",
                type(sender).__name__, consumer, event_type, src)
            return False
    except Exception as _disp_err:
        swallow_warn(
            f"[emit_emot_cgn_signal] dispatch failed ({consumer}, "
            f"{event_type}) from {src}",
            _disp_err, key="bus.emit_emot_cgn_signal")
        return False

    # 5. Record.
    try:
        from .core.bus_health import get_global_monitor
        m = get_global_monitor()
        if m is not None and _is_main_bus:
            m.record_emission(src, consumer, event_type, intensity)
    except Exception:
        pass

    return True


# ----------------------------------------------------------------------
# EMOT-CGN worker producers (Phase 1.6d — rFP_emot_cgn_v2 §10 ADR)
# ----------------------------------------------------------------------
# Simple structured producers for EVENT channel messages to emot_cgn worker.
# No rate-gate / orphan-detection needed for these (unlike META-CGN's
# SIGNAL_TO_PRIMITIVE orphan guard): cadence is per-chain (~1 per 2-3 min)
# and dst="emot_cgn" is a single well-known subscriber, not a mapping table.


def emit_emot_chain_evidence(
    sender, src: str, chain_id: int,
    dominant_at_start: str, dominant_at_end: str,
    terminal_reward: float, ctx: Optional[dict] = None,
) -> bool:
    """Send EMOT_CHAIN_EVIDENCE to emot_cgn worker. Called from
    meta_reasoning._conclude_chain per chain. Returns True if sent."""
    try:
        msg = {
            "type": EMOT_CHAIN_EVIDENCE,
            "src": src, "dst": "emot_cgn", "ts": time.time(), "rid": None,
            "payload": {
                "chain_id": int(chain_id),
                "dominant_at_start": str(dominant_at_start),
                "dominant_at_end": str(dominant_at_end),
                "terminal_reward": float(terminal_reward),
                "ctx": dict(ctx) if ctx else {},
            },
        }
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
        else:
            return False
        return True
    except Exception as e:
        swallow_warn("[emit_emot_chain_evidence] failed", e,
                     key="bus.emit_emot_chain_evidence")
        return False


def emit_meta_chain_complete(
    sender, src: str, chain_id: int, primitives_used: list,
    primitive_transitions: list, chain_length: int, domain: str,
    task_success: float, chain_iql_confidence: float,
    start_epoch: int, conclude_epoch: int,
    context_summary: Optional[dict] = None,
    haov_hypothesis_id: Optional[str] = None,
    final_observation: Optional[dict] = None,
    outer_summary: Optional[dict] = None,
    step_arguments: Optional[list] = None,
) -> bool:
    """Send META_CHAIN_COMPLETE to meta_teacher (and any other subscribers).
    Called from meta_reasoning._conclude_chain after all existing hooks.
    Emitted unconditionally — teacher worker decides whether to critique.

    rFP_meta_teacher_v2 Phase A: outer_summary (distilled outer_context) and
    step_arguments (per-step primitive+sub+arg_summary) are optional payload
    extensions. Legacy consumers that don't read these fields are unaffected.
    When outer-layer is inactive or chain had no outer_context, outer_summary
    is None — teacher falls back to syntactic critique.

    Returns True if sent."""
    try:
        payload: dict = {
            "chain_id": int(chain_id),
            "primitives_used": list(primitives_used),
            "primitive_transitions": [
                (str(a), str(b)) for a, b in primitive_transitions
            ],
            "chain_length": int(chain_length),
            "domain": str(domain or "general"),
            "task_success": float(max(0.0, min(1.0, task_success))),
            "chain_iql_confidence": float(
                max(0.0, min(1.0, chain_iql_confidence))),
            "start_epoch": int(start_epoch),
            "conclude_epoch": int(conclude_epoch),
            "context_summary": dict(context_summary) if context_summary else {},
            "haov_hypothesis_id": haov_hypothesis_id,
            "final_observation": dict(final_observation) if final_observation else {},
        }
        # Phase A additions — only included when non-None / non-empty so
        # legacy-chain payloads stay byte-for-byte compatible.
        if outer_summary:
            payload["outer_summary"] = dict(outer_summary)
        if step_arguments:
            payload["step_arguments"] = list(step_arguments)
        msg = {
            "type": META_CHAIN_COMPLETE,
            "src": src, "dst": "all", "ts": time.time(), "rid": None,
            "payload": payload,
        }
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
        else:
            return False
        return True
    except Exception as e:
        swallow_warn("[emit_meta_chain_complete] failed", e,
                     key="bus.emit_meta_chain_complete")
        return False


def emit_felt_cluster_update(
    sender, src: str, feature_vec_150d: Optional[list] = None,
    felt_tensor_130d: Optional[list] = None,
) -> bool:
    """Send FELT_CLUSTER_UPDATE to emot_cgn worker. Called from spirit
    (or meta_reasoning._start_chain). Callers can pass EITHER the full
    150D feature vector OR just the 130D felt tensor — worker builds
    the 150D from 130D using its own context if only 130D provided.
    Returns True if sent."""
    try:
        payload = {}
        if feature_vec_150d is not None:
            payload["feature_vec_150d"] = list(feature_vec_150d)
        if felt_tensor_130d is not None:
            payload["felt_tensor_130d"] = list(felt_tensor_130d)
        if not payload:
            return False
        msg = {
            "type": FELT_CLUSTER_UPDATE,
            "src": src, "dst": "emot_cgn", "ts": time.time(), "rid": None,
            "payload": payload,
        }
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
        else:
            return False
        return True
    except Exception as e:
        swallow_warn("[emit_felt_cluster_update] failed", e,
                     key="bus.emit_felt_cluster_update")
        return False


# ----------------------------------------------------------------------
# record_send_drop — per-minute aggregated drop telemetry (Phase A §10.E)
# ----------------------------------------------------------------------
# Before this helper, every worker's _send_msg logged WARNING on every
# dropped put_nowait. The 2026-04-14 AM outage produced 184 670 such
# warnings in hours, hiding the actual signal. This aggregator keeps the
# log clean during steady state and produces a single INFO line per
# minute when drops occur, listing count + top-5 (src, dst:type) pairs.
#
# Each worker subprocess has its own module-level state (Python's
# per-process import). Aggregation is per-worker, which matches the
# blast-radius of a worker's own queue-full events.

_drop_window_start: float = 0.0
_drop_counts: dict[tuple, int] = {}  # (src, dst, msg_type) -> count
_drop_total_since_boot: int = 0
_drop_lock = _threading.Lock()
_DROP_WINDOW_SECONDS = 60.0


def record_send_drop(src: str, dst: str, msg_type: str) -> None:
    """Record a dropped send_queue.put failure. Logs aggregated summary
    once per minute, not per-drop. Thread-safe (lock held briefly).

    Call from each worker's _send_msg except-branch instead of
    `logger.warning(...)` per drop. Silent during steady state.
    """
    global _drop_window_start, _drop_total_since_boot
    now = time.time()
    with _drop_lock:
        if _drop_window_start == 0.0:
            _drop_window_start = now

        key = (src, dst, msg_type)
        _drop_counts[key] = _drop_counts.get(key, 0) + 1
        _drop_total_since_boot += 1

        if now - _drop_window_start >= _DROP_WINDOW_SECONDS:
            # Flush + reset
            total = sum(_drop_counts.values())
            n_pairs = len(_drop_counts)
            top5 = sorted(_drop_counts.items(), key=lambda kv: -kv[1])[:5]
            pairs_str = ", ".join(
                f"{k[0]}→{k[1]}:{k[2]}={v}" for k, v in top5
            )
            logger.info(
                "[DropAggregator] %ds window: %d drops across %d pairs. "
                "Top: %s | total since boot: %d",
                int(_DROP_WINDOW_SECONDS), total, n_pairs,
                pairs_str, _drop_total_since_boot,
            )
            _drop_counts.clear()
            _drop_window_start = now


def get_drop_stats() -> dict:
    """Return current drop aggregator state for diagnostics / /v4/bus-health."""
    with _drop_lock:
        return {
            "total_since_boot": _drop_total_since_boot,
            "current_window_drops": sum(_drop_counts.values()),
            "current_window_pairs": len(_drop_counts),
            "window_started_at": _drop_window_start,
        }

# ── Unregistered literals — registered 2026-04-28 audit ─────────────
# These message types were used as string literals across the codebase
# without being registered as constants here. The arch_map dead-wiring
# scanner v2.4 (commit 4493b41e) caught 77 such literals on the
# 2026-04-28 Phase C contract audit; this section closes the drift.
#
# The scanner now emits HIGH-severity `bus_literal_msg_type` findings
# for any literal looking like a bus msg type (UPPER_SNAKE_CASE) that
# isn't in this file. Future literal drift fails CI.
#
# Spec entries in titan_plugin/bus_specs.py are optional: messages here
# default to P2 + no-coalesce (the safe default per Phase B.2 §D6).
# Add a spec entry only if a message genuinely needs P0/P1/P3 or
# coalesce semantics — judgment per-message.

# Backup / save lifecycle
BACKUP_TRIGGER_MANUAL = "BACKUP_TRIGGER_MANUAL"

# Bus protocol — supervision transfer
BUS_HANDOFF_ACK = "BUS_HANDOFF_ACK"

# CGN protocol
CGN_CROSS_INSIGHT     = "CGN_CROSS_INSIGHT"
CGN_DREAM_CONSOLIDATE = "CGN_DREAM_CONSOLIDATE"
CGN_HAOV_VERIFY_REQ   = "CGN_HAOV_VERIFY_REQ"
CGN_HAOV_VERIFY_RSP   = "CGN_HAOV_VERIFY_RSP"
CGN_INFERENCE_REQ     = "CGN_INFERENCE_REQ"
CGN_KNOWLEDGE_REQ     = "CGN_KNOWLEDGE_REQ"
CGN_KNOWLEDGE_RESP    = "CGN_KNOWLEDGE_RESP"
CGN_KNOWLEDGE_USAGE   = "CGN_KNOWLEDGE_USAGE"
CGN_SOCIAL_TRANSITION = "CGN_SOCIAL_TRANSITION"
CGN_WEIGHTS_MAJOR     = "CGN_WEIGHTS_MAJOR"

# Events teacher
EVENTS_WINDOW_POLL = "EVENTS_WINDOW_POLL"

# Experience playground
EXPERIENCE_STIMULUS = "EXPERIENCE_STIMULUS"

# GREAT pulse / kin
GREAT_KIN_PULSE = "GREAT_KIN_PULSE"

# Knowledge graph queries
KNOWLEDGE_CONCEPTS_FOR_PERSON = "KNOWLEDGE_CONCEPTS_FOR_PERSON"
KNOWLEDGE_QUERY_CONCEPT       = "KNOWLEDGE_QUERY_CONCEPT"
KNOWLEDGE_SEARCH              = "KNOWLEDGE_SEARCH"

# LLM teacher request/response
LLM_TEACHER_REQUEST  = "LLM_TEACHER_REQUEST"
LLM_TEACHER_RESPONSE = "LLM_TEACHER_RESPONSE"

# Maker dialogue + narration + proposals
MAKER_DIALOGUE_COMPLETE = "MAKER_DIALOGUE_COMPLETE"
MAKER_NARRATION_REQUEST = "MAKER_NARRATION_REQUEST"
MAKER_NARRATION_RESULT  = "MAKER_NARRATION_RESULT"
MAKER_PROPOSAL_CREATED  = "MAKER_PROPOSAL_CREATED"
MAKER_RESPONSE_RECEIVED = "MAKER_RESPONSE_RECEIVED"

# Meditation lifecycle
MEDITATION_COMPLETE        = "MEDITATION_COMPLETE"
MEDITATION_HEALTH_ALERT    = "MEDITATION_HEALTH_ALERT"
MEDITATION_RECOVERY_TIER_1 = "MEDITATION_RECOVERY_TIER_1"
MEDITATION_RECOVERY_TIER_2 = "MEDITATION_RECOVERY_TIER_2"
MEDITATION_REQUEST         = "MEDITATION_REQUEST"

# Memory ops
MEMORY_ADD                 = "MEMORY_ADD"
MEMORY_RECALL_PERTURBATION = "MEMORY_RECALL_PERTURBATION"

# Meta-reasoning rewards + signals
META_DIVERSITY_PRESSURE = "META_DIVERSITY_PRESSURE"
META_EUREKA             = "META_EUREKA"
META_EVENT_REWARD       = "META_EVENT_REWARD"
META_LANGUAGE_REQUEST   = "META_LANGUAGE_REQUEST"
META_LANGUAGE_RESULT    = "META_LANGUAGE_RESULT"
META_LANGUAGE_REWARD    = "META_LANGUAGE_REWARD"
META_OUTER_REWARD       = "META_OUTER_REWARD"
META_PATTERN_EMERGED    = "META_PATTERN_EMERGED"
META_PERSONA_REWARD     = "META_PERSONA_REWARD"
META_STRATEGY_DRIFT     = "META_STRATEGY_DRIFT"

# Outer trinity ready
OUTER_TRINITY_READY = "OUTER_TRINITY_READY"

# Output verifier
OUTPUT_VERIFIER_READY = "OUTPUT_VERIFIER_READY"
OUTPUT_VERIFIER_STATS = "OUTPUT_VERIFIER_STATS"

# Query / response infra
QUERY_RESPONSE = "QUERY_RESPONSE"

# Reflex worker ready
REFLEX_READY = "REFLEX_READY"

# Save lifecycle
SAVE_DONE = "SAVE_DONE"
SAVE_NOW  = "SAVE_NOW"

# Search pipeline
SEARCH_PIPELINE_BUDGET_RESET = "SEARCH_PIPELINE_BUDGET_RESET"

# Self-exploration / self-prediction
SELF_EXPLORE_TRIGGER     = "SELF_EXPLORE_TRIGGER"
SELF_PREDICTION_VERIFIED = "SELF_PREDICTION_VERIFIED"

# Silent-swallow report
SILENT_SWALLOW_REPORT = "SILENT_SWALLOW_REPORT"

# Social perception
SOCIAL_PERCEPTION = "SOCIAL_PERCEPTION"

# System upgrade thoughts
SYSTEM_UPGRADE_THOUGHT = "SYSTEM_UPGRADE_THOUGHT"

# TimeChain contracts
CONTRACT_APPROVE   = "CONTRACT_APPROVE"
CONTRACT_DEPLOY    = "CONTRACT_DEPLOY"
CONTRACT_LIST      = "CONTRACT_LIST"
CONTRACT_LIST_RESP = "CONTRACT_LIST_RESP"
CONTRACT_PROPOSE   = "CONTRACT_PROPOSE"
CONTRACT_REJECTED  = "CONTRACT_REJECTED"
CONTRACT_STATUS    = "CONTRACT_STATUS"
CONTRACT_VETO      = "CONTRACT_VETO"

# TimeChain operations
TIMECHAIN_AGGREGATE  = "TIMECHAIN_AGGREGATE"
TIMECHAIN_CHECK      = "TIMECHAIN_CHECK"
TIMECHAIN_CHECKPOINT = "TIMECHAIN_CHECKPOINT"
TIMECHAIN_COMMIT     = "TIMECHAIN_COMMIT"
TIMECHAIN_COMMITTED  = "TIMECHAIN_COMMITTED"
TIMECHAIN_COMPARE    = "TIMECHAIN_COMPARE"
TIMECHAIN_QUERY      = "TIMECHAIN_QUERY"
TIMECHAIN_QUERY_RESP = "TIMECHAIN_QUERY_RESP"
TIMECHAIN_RECALL     = "TIMECHAIN_RECALL"
TIMECHAIN_REJECTED   = "TIMECHAIN_REJECTED"
TIMECHAIN_STATUS     = "TIMECHAIN_STATUS"

# Warning monitor pulses
WARNING_PULSE = "WARNING_PULSE"

# X / social_x dispatch
X_FORCE_POST    = "X_FORCE_POST"
X_POST_DISPATCH = "X_POST_DISPATCH"

