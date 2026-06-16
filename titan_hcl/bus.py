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
from titan_hcl.utils.silent_swallow import swallow_warn

# Type alias: either threading or multiprocessing Queue
AnyQueue = Union[ThreadQueue, MPQueue]

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Kernel-internal subscriber discriminator (Microkernel v2 Phase B.2 §D12)
# ──────────────────────────────────────────────────────────────────────
# Per PLAN_microkernel_phase_b2_ipc.md §D12: under socket mode, kernel-internal
# subscribers (Guardian, kernel control loops, proxy reply queues, etc.) get
# an in-process ThreadQueue (faster, no broker hop). Worker subscribers (in
# separate processes) SHOULD use `setup_worker_bus` → BusSocketClient instead
# of calling `bus.subscribe()` directly.
#
# This explicit allowlist + suffix rules is the single source of truth for
# the discrimination. Update it in lockstep with new kernel-side subscribers.
#
# Audited 2026-05-02 against every `bus.subscribe(...)` call site in
# titan_hcl/. Workers do NOT call subscribe — they receive via
# `setup_worker_bus` / SocketQueue path. So in steady state, every reachable
# subscribe() call is kernel-internal; the discriminator returns True for all
# of them. The check exists to surface contract violations LOUD if a future
# caller (or accidental worker-side use) trips it.

_KERNEL_INTERNAL_NAMES: frozenset[str] = frozenset({
    # Guardian self
    "guardian",
    # Kernel control loops (titan_hcl/core/kernel.py:267-279,690).
    # Phase 10K (rFP §3G): "meditation" + "sovereignty" removed — they became
    # separate worker subprocesses (D-SPEC-57/60/64) with their own
    # BusSocketClient; no kernel-side subscriber uses those names anymore
    # (plugin.py:1441 confirms the sovereignty alias is retired).
    "core", "kernel",
    # Plugin control loops (titan_hcl/core/plugin.py:1470,2099,2640 +
    # legacy_core.py:1193,1386,1848). NOTE: "agency" (NOT "agency_queue" —
    # the Python attr is `_agency_queue` but the bus name is "agency").
    "agency", "chat_handler", "v4_bridge",
    # State register (titan_hcl/logic/state_register.py:430)
    "state_register",
    # RLProxy stats subscription (titan_hcl/proxies/rl_proxy.py:79).
    # Doesn't match the _proxy suffix because it's a SECOND subscription
    # made by rl_proxy.py for SAGE_STATS broadcasts, with a distinct queue
    # name from the proxy's reply queue. Kernel-side, broadcasts from the
    # rl worker's spawn process.
    "rl_proxy_stats",
    # Legacy in-process API (api_process_separation_enabled=false path)
    "api",
})

# Suffixes that mark a subscriber as kernel-internal:
# - "_proxy"  → kernel-side proxy reply queue (12 proxies in titan_hcl/proxies/)
# - "_query"  → reflex_executors.py query reply queues
_KERNEL_INTERNAL_SUFFIXES: tuple[str, ...] = ("_proxy", "_query")


def _is_kernel_internal(module_name: str) -> bool:
    """Phase B.2 §D12 discriminator — True if subscriber is kernel-side.

    Kernel-internal subscribers get an in-process Queue (legacy ThreadQueue
    path). Non-kernel-internal callers under socket mode are a contract
    violation — workers should use setup_worker_bus, not bus.subscribe.

    See `_KERNEL_INTERNAL_NAMES` and `_KERNEL_INTERNAL_SUFFIXES` for the
    canonical allowlist. Update those in lockstep with new kernel-side
    subscribers.
    """
    if module_name in _KERNEL_INTERNAL_NAMES:
        return True
    return module_name.endswith(_KERNEL_INTERNAL_SUFFIXES)


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
            from titan_hcl.config_loader import load_titan_config
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

# Phase 11 (SPEC §11.I / D-SPEC-141 / v1.65.0):
# MODULE_ERROR is the SINGLE structured-error topic — every worker / helper /
# provider that raises publishes a ModuleError envelope here (P1, dst="all",
# non-blocking per §8.0.ter). Subscribers: guardian_hcl (restart decisions),
# warning_monitor (trend detection), observatory_worker (errors_state.bin SHM).
# Wire payload = ModuleError.as_wire_dict() (see titan_hcl/errors.py).
MODULE_ERROR = "MODULE_ERROR"
# Backpressure notification — broker rate-limits per-module ModuleError emissions
# to 10/s when sustained rate exceeds 100/s (per RFP §3H.3); this event signals
# the flood-state to observers so they can render degraded-mode UX.
MODULE_ERROR_FLOOD = "MODULE_ERROR_FLOOD"
# Probe contract per §11.I.2/§11.I.3 — bus-RPC, not state propagation.
# titan_hcl polls per-module SHM slots at 1Hz; on state=BOOTED it dispatches
# MODULE_PROBE_REQUEST(name, probe_id) to the worker (P0, dst=worker). Worker
# runs its probe_fn, writes state=RUNNING/UNHEALTHY + last_probe_result to its
# SHM slot, AND replies MODULE_PROBE_RESPONSE(probe_id, result: ProbeResult)
# correlation_id-routed back to titan_hcl. Both signals must agree (SHM + RPC).
MODULE_PROBE_REQUEST = "MODULE_PROBE_REQUEST"
MODULE_PROBE_RESPONSE = "MODULE_PROBE_RESPONSE"

# Synthesis Engine Phase 1 (D-SPEC-123 / SPEC v1.56.0 §25):
# MEMORY_RETRIEVAL_USED is the use-gated activation event (INV-Syn-5): emitted
# by retrieval consumers (Phase 1.5+ producers: agno post-hook, RECALL ops)
# ONLY for items the LLM actually cited / acted upon — never mere surfacing.
# synthesis_worker is the sole consumer; payload =
#     {"item_id": "kuzu:NODE_42" | "tc:0xabc..." | "mem:42" | "skill:S7",
#      "ts": <monotonic seconds>}
# SYNTHESIS_RECOMPUTE_DONE is observability-only — emitted by synthesis_worker
# after each 60s recompute pass with {"items_recomputed": N, "duration_ms": M}.
MEMORY_RETRIEVAL_USED = "MEMORY_RETRIEVAL_USED"
SYNTHESIS_RECOMPUTE_DONE = "SYNTHESIS_RECOMPUTE_DONE"
# Operator-closure (SPEC §25.9) — ONE per-turn knowledge-moment signal emitted
# post-LLM by agno (the CitedUseDetector boundary), carrying {needed, satisfied,
# ts}. synthesis_worker's SovereigntyRatioMeter records the per-TURN denominator
# from this (replacing the per-ITEM MEMORY_RETRIEVAL_USED inflation). MEMORY_
# RETRIEVAL_USED stays the per-ITEM reinforcement signal (record_access).
KNOWLEDGE_MOMENT = "KNOWLEDGE_MOMENT"
# Operator-closure C2 (W7) — a chat-time self-oracle tool (coding_sandbox in
# agno) ships its PRE-COMPUTED companion verdict to synthesis_worker's
# OracleRouter for the dream-boundary OracleVerdictBatch flush (no re-exec).
# Payload: {parent_tool_call_tx, oracle_id, verdict, evidence_ref, latency_ms, ts}.
TOOL_CALL_VERDICT_RECORD = "TOOL_CALL_VERDICT_RECORD"
# Operator-closure telemetry (2026-06-01) — the §3 operator RECALL runs in
# agno_worker (chat) + cognitive_worker (per-epoch), each with its OWN
# RuleEvaluator + EngineRecall instance, so the §18 chi/retrieval metrics
# (read off synthesis_worker's idle local evaluator + an un-fed latency ring)
# were structurally blind to the work actually happening. The recall paths now
# emit this fire-and-forget P3 sample after each recall; synthesis_worker
# aggregates it into the retrieval latency ring + cross-process chi totals so
# /v6/synthesis/metrics reflects the real loop. Metrics-only (INV-Syn-25:
# derived, non-authoritative). Payload: {latency_ms, chi_spent, evaluations,
# hits, fork, source, ts}.
RETRIEVAL_SAMPLE = "RETRIEVAL_SAMPLE"

# ── Outer Meta-Reasoning Self-Learning (RFP_synthesis_self_learning_meta_
# reasoning Phase 1 / §7.A) — the verifiable-lane closed loop. The reward is
# ASYNC (the oracle verdict arrives at the dream-flush, NOT at turn time), so
# the decision and the reward are two events joined on `parent_tool_call_tx`
# by the self_learning_worker (INV-OML-7; the join is the load-bearing detail).
#
# agno DECIDE path → self_learning_worker. Emitted AFTER the backstop runs (so
# parent_tool_call_tx is known). Payload: {parent_tool_call_tx, features (the
# OuterFeatures float list), action (int index), goal_class, turn_id, ts}.
SELF_LEARN_DECISION = "SELF_LEARN_DECISION"
# synthesis_worker (OracleRouter score-event flush) → self_learning_worker. The
# joined reward for a stashed decision. Payload: {parent_tool_call_tx, reward
# (+1 verified-true / −1 false), oracle_id, goal_class, ts}.
SELF_LEARN_REWARD = "SELF_LEARN_REWARD"
# self_learning_worker → synthesis_worker. A distilled winning macro-strategy to
# persist under the `Self` spine node (Reasoning node + SELF_HAS_REASONING edge)
# via the single SynthesisWriter (INV-Syn-19/28 / INV-OML-8). Payload:
# {signature (float list), goal_class, b_i, c, time_cost, use_count, verified
# (bool → timechain anchor), label, ts}.
SELF_LEARN_MACRO_READY = "SELF_LEARN_MACRO_READY"
# self_learning_worker (idle EXPLORE loop, metabolically gated) → the background
# idle chat loop. Ask the real pipeline to pose an uncertain verifiable problem
# so its reward flows back through the normal join (exploration never on a live
# user turn — INV-OML-9). Payload: {goal_class, prompt_hint, ts}.
SELF_LEARN_EXPLORE_REQUEST = "SELF_LEARN_EXPLORE_REQUEST"
# ── Phase D / §7.D-knowledge (DK.1) — the sovereign LLM-Wiki research→declarative-
# concept COMPOUNDING loop, continuous-at-confirm (INV-OML-12). Two hops, each
# worker does what it owns:
# RESEARCH_CONFIRMED: agno PreHook (EEL-A `detect_confirmation=="confirm"` branch)
#   → memory_worker (promotion/DB owner). The user just confirmed a researched
#   answer; promote+anchor THAT single `acquired:research` mempool node NOW (reuse
#   `migrate_to_persistent` + `_anchor_promoted_node` → local-timechain tx_hash +
#   sidecar) instead of waiting for the 6h meditation epoch. Payload:
#   {node_id, user_prompt, agent_response, acquired_source, user_id, felt, ts}.
RESEARCH_CONFIRMED = "RESEARCH_CONFIRMED"
# RESEARCH_CONCEPT_SEED: memory_worker (after the per-node anchor) → synthesis_worker
#   (sole spine writer, INV-Syn-7). Seed/refine the declarative `Engram` concept for
#   the now-anchored finding via the single SynthesisWriter path (LLM-librarian names
#   it over the VERIFIED content — never authors the fact; GD10). Payload:
#   {tx_hash (the finding's anchor — deref target, INV-OML-10), content, domain_hint,
#   felt_coverage, ts}.
RESEARCH_CONCEPT_SEED = "RESEARCH_CONCEPT_SEED"
# ── Phase B (§7.B, C1′) — the NON-verifiable lane. agno PostHook → synthesis_worker
# AFTER the response is generated (a direct/research/IDK turn has no oracle). The
# synthesis single-writer graphs a `Reasoning(kind='turn')` record under SELF →
# LEARNING → REASONING with reward=NULL (pending); the turn-judge (B.2) / a user-
# Maker rating (B.3) scores it later, keyed by the same reasoning_id. Payload:
# {reasoning_id, prompt, response, action (int index), goal_class, features (float
# list), lane, user_id, ts}.
TURN_REASONING_RECORD = "TURN_REASONING_RECORD"
# §7.B (B.3) — the feedback endpoint → synthesis_worker, MAKER ONLY. A Maker rating
# of a non-verifiable turn is graphed as a `MakerAssessment` node under `Self`
# (SELF_HAS_MAKER_ASSESSMENT) — the searchable Maker↔Titan bond (ordinary-user
# feedback is reward-only, NOT graphed). The reward itself rides SELF_LEARN_REWARD
# (source: maker|user). Payload: {reasoning_id, score, scale, reward, turn_summary, ts}.
MAKER_ASSESSMENT_RECORD = "MAKER_ASSESSMENT_RECORD"

# RFP_missions_and_the_maker_model §7.1 — a fact ABOUT the Maker (what Titan has
# learned about the human he is bonded to) → synthesis_worker, persisted as a
# `MakerFact` under the `Maker` hub (Self -[SELF_HAS_MAKER]-> Maker -[MAKER_HAS_FACT]->
# MakerFact). Distinct from MAKER_ASSESSMENT_RECORD (that's the Maker rating Titan; this
# is Titan modelling the Maker). Sovereign knowledge (INV-MIS-SOVEREIGN-KNOWLEDGE) +
# epistemic-honest (provenance + confidence<1.0). Produced by the gated post-Maker-turn
# extractor. Payload: {category, value, provenance, confidence, source_turn, ts}.
MAKER_FACT_RECORD = "MAKER_FACT_RECORD"

# Phase 2 standing-contract event (PLAN_synthesis_engine_Phase2.md 2B,
# D-P2-4): emitted by the post-seal contract hook in
# timechain_v2.Mempool/BlockBuilder for every TX sealed that matches an
# active contract whose action is `maintain_bundle`. Single consumer =
# synthesis_worker (sole writer G21 / INV-Syn-3) of
# data/synthesis.duckdb / association_bundles. Payload:
#     {"entity_class": "user" | "topic" | "skill" | ...,
#      "entity_id":    "<hash>" | "<tag>" | "<skill_id>",
#      "fork":         "conversation" | "procedural" | ...,
#      "tx_hash":      "<hex>",
#      "epoch_id":     <int>,
#      "ts":           <wall-clock seconds>,
#      "significance": <float>}
# Standing contracts convert RECALL of a bundled entity into a READ
# (single SELECT) per arch §12.2.
MAINTAIN_BUNDLE = "MAINTAIN_BUNDLE"

# Phase 5 §P5.A-G — hypothesis-fork lifecycle command-and-control surface.
# Published by api process (POST /v6/synthesis/forks/*) on Maker-authenticated
# requests; consumed by synthesis_worker (sole writer per INV-Syn-8).
# Eventual-consistent: after the command lands, synthesis_worker exports a
# fresh forks_snapshot.json before returning to the recv loop so the next
# api GET reflects the new state (no 60s wait).
#
# Payload shape (op-discriminated):
#   {op: "create",  intent: str, root_anchor?: str, parent_concept_id?: str,
#                   request_id: str}
#   {op: "record_exploration_tx", fork_id: str, tx_hash: str,
#                   request_id: str}
#   {op: "graduate_manual", fork_id: str, concept_name?: str, evidence_ref: str,
#                   request_id: str}
#   {op: "abandon", fork_id: str, reason?: str, request_id: str}
#   {op: "sweep",   dry_run?: bool, request_id: str}
SYNTHESIS_FORK_COMMAND = "SYNTHESIS_FORK_COMMAND"
SYNTHESIS_FORK_COMMAND_RESULT = "SYNTHESIS_FORK_COMMAND_RESULT"

# SPEC §25.6 / D-SPEC-PHASE7 (v1.66.0) — ACT-R working-memory buffer commands.
# Emitted by agno_worker on every BufferCache.set/clear write-through;
# synthesis_worker (sole writer per INV-Syn-16) persists the row to
# `actr_buffers` on synthesis.duckdb + atomic-writes buffers_snapshot.json.
# INV-4 single-canonical-write-path extends through this bus command —
# no producer bypasses synthesis_worker as the sole writer.
# Payload shapes:
#   {op: "set",   chat_id, buffer_name, content, concept_ids, ts}
#   {op: "clear", chat_id, buffer_name, ts}
SYNTHESIS_BUFFER_COMMAND = "SYNTHESIS_BUFFER_COMMAND"

# Phase 8 §P8.I (D-SPEC-PHASE8): meta-fork bus events surfacing the
# procedural pipeline + skill miner lifecycle. Synthesis_worker is the
# sole emitter for all of these except META_SKILL_COMPILATION_CANDIDATE
# which the actr_procedural_skill_proposer SC emits at dream_boundary
# to wake the ProceduralMiner.
META_SKILL_COMPILATION_CANDIDATE = "META_SKILL_COMPILATION_CANDIDATE"
META_SKILL_COMPILED = "META_SKILL_COMPILED"
META_SKILL_VERIFIED = "META_SKILL_VERIFIED"
META_SKILL_REJECTED = "META_SKILL_REJECTED"
META_SKILL_SOFT_RETIRED = "META_SKILL_SOFT_RETIRED"
# Phase 8 fold-in (P8.Y / P7 CGN lexicon exporter): emitted by cgn_worker
# on every CGN vocabulary mutation event + on 5-min snapshot exporter
# cadence; agno_worker subscribes to refresh its in-process lexicon cache
# so the P7 _ground_for_goal_hook returns real concept_ids.
CGN_LEXICON_UPDATED = "CGN_LEXICON_UPDATED"

# Phase 9 §P9.F (D-SPEC-PHASE9): meta-reasoning integration + repair forks.
# SKILL_REPAIR_FORK_SPAWNED — emitted by synthesis_worker's SkillFailureTracker
# when a delegated skill hits N consecutive failures (§9.3 / §11.5); payload =
#     {"skill_id", "fork_id", "parent_concept_id", "root_anchor", "kind",
#      "consecutive_failures", "ts"}. agno_worker subscribes (log surface).
# USER_FEEDBACK_SIGNAL — emitted by agno_worker on explicit user thumbs-up/down
# (INV-Syn-24 Tier-2); payload =
#     {"tool_call_tx", "verdict": "positive"|"negative", "source": "explicit",
#      "skill_id"?, "ts"}. Single consumer = synthesis_worker (UserFeedbackOverride).
# NOTE (INV-Syn-23): MEMORY_RETRIEVAL_USED now carries `used_by_llm: bool` — only
# True triggers record_access reinforcement; False = surfaced-not-cited telemetry.
SKILL_REPAIR_FORK_SPAWNED = "SKILL_REPAIR_FORK_SPAWNED"
USER_FEEDBACK_SIGNAL = "USER_FEEDBACK_SIGNAL"

# SPEC §8.3 Phase B (rFP_phase_c_bus_delivery_continuity_and_hot_reload §4):
# per-module hot-reload protocol. REQUEST emitted by Maker CLI / future D9
# Guardian; ACK emitted by parent Guardian with status transitions
# spawning → adopted → ready (or failed / rolled_back on error). The ACK
# is pre-listed in BOOT_BUFFERED_TYPES (Phase A) so the broker absorbs it
# if the initiator's reply subscription transiently lapses during reload.
MODULE_RELOAD_REQUEST = "MODULE_RELOAD_REQUEST"
MODULE_RELOAD_ACK = "MODULE_RELOAD_ACK"
# Phase 6 (SPEC §11.B.4 / D-SPEC-135 / v1.62.0) — GuardianHCLClient (thin
# bus client in titan_hcl process) forwards lifecycle mutations to
# guardian_hcl via these targeted messages. guardian_hcl subscribes via
# scripts/guardian_hcl.py:_handle_module_lifecycle_requests and invokes
# Guardian.{start, stop, restart_module} on receipt.
MODULE_START_REQUEST = "MODULE_START_REQUEST"
MODULE_STOP_REQUEST = "MODULE_STOP_REQUEST"
MODULE_RESTART_REQUEST = "MODULE_RESTART_REQUEST"
EPOCH_TICK = "EPOCH_TICK"

# Phase C C-S7 (2026-05-05) — supervision messages per SPEC §8.1 +
# §11.B + §11.G.6. Emitted by guardian_HCL when Python L2/L3 modules
# crash, restart, escalate, or hit dependency-blocked respawn states.
# Cross-language unified with Rust supervisors (titan-kernel-rs's
# KernelChildSupervisor + titan-trinity-rs's UnifiedSpiritSupervisor +
# titan-unified-spirit-rs's DaemonSupervisor) — same wire format,
# same JSONL log destination (`data/supervision.jsonl`).
SUPERVISION_CHILD_DOWN = "SUPERVISION_CHILD_DOWN"
SUPERVISION_CHILD_RESTARTED = "SUPERVISION_CHILD_RESTARTED"
SUPERVISION_ESCALATION = "SUPERVISION_ESCALATION"
SUPERVISION_ESCALATION_RESPONSE = "SUPERVISION_ESCALATION_RESPONSE"
SUPERVISION_DEPENDENCY_BLOCKED = "SUPERVISION_DEPENDENCY_BLOCKED"
SUPERVISION_DEPENDENCY_RECOVERED = "SUPERVISION_DEPENDENCY_RECOVERED"
SUPERVISION_DEPENDENCY_DEGRADED = "SUPERVISION_DEPENDENCY_DEGRADED"
# SPEC §11.G.2.5 (D-SPEC-90, v1.29.0) — Guardian emits before recursively starting
# a STOPPED ENSURE_RUNNING dep at the dependent's first-start. P1 informational.
SUPERVISION_DEPENDENCY_ACTIVATING = "SUPERVISION_DEPENDENCY_ACTIVATING"

# Microkernel v2 Phase B.2 §D9 (2026-05-02) — broker peer-process death
# signal. Published by BusSocketServer when its smart-liveness algorithm
# (Tier 1-4) detects that a subscriber's peer PID is dead via os.kill(pid, 0).
# Authoritative + faster than Guardian's process.is_alive() polling
# (which runs every 1s); broker often catches the death within tens of ms.
#
# Payload schema:
#     {"name": str,          # "memory" / "cgn" / etc, OR "anon-N" if pre-subscribe
#      "pid": int,           # peer PID from SO_PEERCRED at accept time
#      "was_anon": bool,     # True if connection died before sending BUS_SUBSCRIBE
#      "silent_for_s": float} # seconds since last_pong_ts at purge time
#
# dst="guardian"; consumed by Guardian._process_guardian_messages →
# triggers self.restart(name) for named workers (idempotent with the
# polling-path restart via Guardian._module_lock).
BUS_PEER_DIED = "BUS_PEER_DIED"

# Mainnet Lifecycle Wiring rFP (2026-04-20): GreatCycleTracker wiring.
# Spirit_worker publishes SOVEREIGNTY_EPOCH every 10 consciousness epochs
# with current neuromod snapshot + dev_age + great_pulse_fired.
# sovereignty_worker (SPEC v1.8.3 §9.B / D-SPEC-57) subscribes and calls
# tracker.record_epoch(...). Effective resolution is 10:1 sampled
# (5000-sample convergence window = ~50k actual epochs ≈ 10-14h).
# Adequate for long-horizon convergence.
SOVEREIGNTY_EPOCH = "SOVEREIGNTY_EPOCH"

# §4.L sovereignty_worker carve (SPEC v1.8.3 / D-SPEC-57, 2026-05-15):
# fire-and-forget Maker-confirm event emitted from api/webhook.py on
# verified Maker directive (Helius webhook, Ed25519 signature checked).
# Replaces the pre-carve `getattr(plugin, "sovereignty").confirm_maker()`
# direct-call which silent-no-opped under Phase C api_subprocess
# kernel_rpc serialization gap. Idempotent — repeated emits no-op once
# _maker_confirmed=True. Payload: {tx_signature: str, ts: float}.
SOVEREIGNTY_CONFIRM_MAKER = "SOVEREIGNTY_CONFIRM_MAKER"

# §4.L sovereignty_worker carve (SPEC v1.8.3 / D-SPEC-57, 2026-05-15):
# fire-and-forget great-cycle-increment event emitted from api/maker.py
# on Resurrection Protocol successful completion (subprocess
# returncode == 0). Replaces the pre-carve
# `getattr(plugin, "sovereignty").increment_great_cycle()` direct-call
# which silent-no-opped under Phase C api_subprocess kernel_rpc
# serialization gap. Increments _great_cycle by 1 + persists JSON.
# Payload: {ts: float, source: str}.
SOVEREIGNTY_INCREMENT_GREAT_CYCLE = "SOVEREIGNTY_INCREMENT_GREAT_CYCLE"

# §4.H interface_advisor_worker carve (SPEC v1.8.5 / D-SPEC-59,
# 2026-05-15): fire-and-forget per-message-type record event emitted
# from parent `_handle_impulse` (and any other caller seeking rate
# accounting) → interface_advisor_worker. Worker records timestamp in
# the sliding-window deque + republishes interface_advisor_state.bin
# SHM slot. On rate-exceeded the worker emits RATE_LIMIT back to source.
# Payload: {msg_type: str, source: str, client_ts: float}. P3 priority.
IMPULSE_RECEIVED = "IMPULSE_RECEIVED"

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

# SAGE_RECORD_TRANSITION / SAGE_GATE_DECIDE / SAGE_IQL_TRAIN_STEP / SAGE_STATS /
# SAGE_READY RETIRED with the offline-RL subsystem
# (RFP_synthesis_decision_authority P1) — the recorder/scholar/gatekeeper
# (IQL/torch) workers + their bus contracts are gone. Execution-mode routing is
# the grounded router; sovereignty is the ONE S = 0.7E+0.3V.

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
CHAT_REQUEST = "CHAT_REQUEST"      # api_subproc → agno_worker (D-SPEC-72 — was chat_handler pre-v1.17.0; dst rerouted, semantics preserved)
CHAT_RESPONSE = "CHAT_RESPONSE"    # agno_worker → api_subproc (D-SPEC-72 — was chat_handler pre-v1.17.0)
CHAT_STREAM_REQUEST = "CHAT_STREAM_REQUEST"   # api_subproc → agno_worker (SSE path; D-SPEC-72 NEW)
CHAT_STREAM_CHUNK = "CHAT_STREAM_CHUNK"       # agno_worker → api_subproc (per-token; correlation_id; D-SPEC-72 NEW)
AGNO_WORKER_READY = "AGNO_WORKER_READY"       # agno_worker → guardian_HCL (once on boot; D-SPEC-72 NEW)

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
# Phase A.S8 (2026-04-30) — outer trinity 3 separate STATE messages
# (mirror inner BODY/MIND/SPIRIT_STATE; explicit OUTER_ prefix for code readability).
OUTER_BODY_STATE = "OUTER_BODY_STATE"
OUTER_MIND_STATE = "OUTER_MIND_STATE"
OUTER_SPIRIT_STATE = "OUTER_SPIRIT_STATE"
# OUTER_SOURCES_SNAPSHOT RETIRED (Phase C dissolution C.8, 2026-05-22): it
# broadcast STATE over the bus (G18 violation). Outer-source data now flows
# SHM-direct via the in-parent sensor sidecars + outer_source_assembly helper.
# DEPRECATED — kept for legacy_core.py import compat (legacy boot fallback path).
# Active microkernel v2 path uses the 3 separate STATE messages above.
OUTER_TRINITY_STATE = "OUTER_TRINITY_STATE"
# A.8.4 — parent → outer_trinity worker (fire-and-forget; sources payload pre-flattened by parent)
OUTER_TRINITY_COLLECT_REQUEST = "OUTER_TRINITY_COLLECT_REQUEST"
SPHERE_PULSE = "SPHERE_PULSE"
BIG_PULSE = "BIG_PULSE"
GREAT_PULSE = "GREAT_PULSE"
# P0.5 / D-SPEC-131 §G5.1 UP-leg gift events published by titan-{inner,outer}-{body,mind}-rs
# on their own sphere clock's balanced rising-edge. Spirit daemons subscribe;
# journey_persistence_worker on Python L2 also subscribes for SQL durability.
BODY_BALANCE_GIFT = "BODY_BALANCE_GIFT"
MIND_BALANCE_GIFT = "MIND_BALANCE_GIFT"
# P0.6-C / D-SPEC-132 §6.6 polarity-homeostat corrective event chain.
# body/mind → spirit on imbalance detection; spirit → body/mind on nudge.
EXTREME_IMBALANCE_DETECTED = "EXTREME_IMBALANCE_DETECTED"
CORRECTIVE_NUDGE = "CORRECTIVE_NUDGE"
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
# v1.8.2 (D-SPEC-56): dream_state_worker is the canonical DREAM_STATE_CHANGED
# publisher (sourced from DREAMING_STATE_UPDATED transition detection). Producer
# changed from dead spirit_worker.py:3006/3143 under Phase C l0_rust_enabled=true.
# DREAM_WAKE_REQUEST destination changed from "spirit" → "dream_state_worker"
# (worker forwards via DREAM_WAKE_FORWARD to cognitive_worker which calls
# coordinator.dreaming.request_wake() in-process).
DREAM_STATE_CHANGED = "DREAM_STATE_CHANGED"  # dream_state_worker → all + → timechain (sleep/wake transition; v1.8.2 producer)
DREAM_WAKE_REQUEST = "DREAM_WAKE_REQUEST"    # chat_api + world_observer → dream_state_worker (v1.8.2 dst)
DREAM_WAKE_FORWARD = "DREAM_WAKE_FORWARD"    # dream_state_worker → cognitive (v1.8.2 NEW — forwards to coordinator.dreaming.request_wake)
DREAM_INBOX_ENQUEUE = "DREAM_INBOX_ENQUEUE"  # chat_api → dream_state_worker (v1.8.2 NEW — chat-during-dream buffer)
DREAM_INBOX_REPLAY = "DREAM_INBOX_REPLAY"    # dream_state_worker → chat_api (v1.8.2 NEW — drain queue on dream_end)
# Force-dream command (post-§4.I D8-3 cleanup, 2026-05-15 evening): admin/maker
# path for testing / maintenance / inspection. Was published by CommandSender
# at command_sender.py:120 (dst="spirit") but its handler lived in the deleted
# spirit_worker BEGIN_DREAMING coord_event block (chunk I8 cleanup) →
# orphaned. Re-wired to cognitive_worker which owns DreamingEngine via
# InnerTrinityCoordinator. cognitive_worker subscriber → coordinator.dreaming.
# request_dream(reason) in-process → next check_transition AWAKE branch
# returns BEGIN_DREAMING (bypasses wake_inertia + drive gate, matching FORCE
# SLEEP semantics).
FORCE_DREAM_REQUEST = "FORCE_DREAM_REQUEST"  # CommandSender → cognitive (was → spirit; rewired post-§4.I)

# Studio Worker messages (v1.8.3 — D-SPEC-57)
# rFP_titan_hcl_l2_separation_strategy §4.K — adopts D-SPEC-46 (memory_worker
# Phase B) event+Future-registry pattern for slow renders so ALL work-RPC paths
# stay ≤5s per G19 strict. ZERO new phase_c_rpc_exemptions.yaml entries.
# STUDIO_RENDER_REQUEST = caller (cognitive_worker / reflex_executors / agno_tools /
# meditation / plugin) → studio_worker (one-way work request, uuid request_id).
# STUDIO_RENDER_COMPLETED = studio_worker → all (broadcast with matching request_id;
# StudioProxy._RenderCompletionRegistry resolves matching Future for _with_completion
# callers). STUDIO_WORKER_READY = once at boot lifecycle.
STUDIO_WORKER_READY = "STUDIO_WORKER_READY"        # studio_worker → guardian_HCL (v1.8.3 NEW — boot ready)
STUDIO_RENDER_REQUEST = "STUDIO_RENDER_REQUEST"    # any caller → studio_worker (v1.8.3 NEW — slow-render request)
STUDIO_RENDER_COMPLETED = "STUDIO_RENDER_COMPLETED"  # studio_worker → all (v1.8.3 NEW — render done, with request_id)

# State-update events — change-edge notifications for cross-process
# consumers (e.g. trinity 130D dim formulas in plugin._STATS_CACHE_EVENT_TYPES,
# observatory_writer, metabolism tier re-evaluation). Phase D D-SPEC-82
# retired the api_subprocess BusSubscriber consumer; api_subprocess now
# reads canonical state from SHM slots per Preamble G18.
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
EXPRESSION_TRANSLATOR_STATS_UPDATED = "EXPRESSION_TRANSLATOR_STATS_UPDATED"  # parent (translator owner) → expression_worker (L3 housekeeping closure 2026-05-26)
NEUROMOD_STATS_UPDATED = "NEUROMOD_STATS_UPDATED"              # spirit → all (neuromods.full)
# Producer-event change-edge notifications. Consumed by trinity 130D dim
# formulas (plugin._STATS_CACHE_EVENT_TYPES) + observatory writers; api
# subprocess reads canonical state SHM-direct per Preamble G18.
SOCIAL_STATS_UPDATED = "SOCIAL_STATS_UPDATED"                  # persona → all (social.stats)
LANGUAGE_STATS_UPDATED = "LANGUAGE_STATS_UPDATED"              # language teacher → all (language.stats)
META_TEACHER_STATS_UPDATED = "META_TEACHER_STATS_UPDATED"      # meta-teacher → all (meta_teacher.stats)
CGN_STATS_UPDATED = "CGN_STATS_UPDATED"                        # cgn → all (cgn.stats)
SOCIAL_PERCEPTION_STATS_UPDATED = "SOCIAL_PERCEPTION_STATS_UPDATED"  # spirit → all (social_perception.stats) — replaces sync bus.request lookup (G19, 2026-05-07)

# M8: External Intent (wallet observer → spirit → neuromod boost)
# RESERVED: EXTERNAL_INTENT — WalletObserver helper class exists but
# polling loop not yet wired into boot. Donation → neuromod boost path
# awaits wallet_observer spawn site (tracked in mainnet master plan M8).
EXTERNAL_INTENT = "EXTERNAL_INTENT"              # wallet_observer → spirit (DI/I/donation)

# Observatory V2: real-time frontend events (spirit → v4_bridge → WebSocket)
NEUROMOD_UPDATE = "NEUROMOD_UPDATE"      # spirit → v4_bridge (every Tier 2 tick)
HORMONE_FIRED = "HORMONE_FIRED"          # spirit → v4_bridge (on program fire)
EXPRESSION_FIRED = "EXPRESSION_FIRED"    # spirit → v4_bridge (on composite fire)

# Cross-worker neuromod external nudge (NEW 2026-05-15 per §4.Q neuromod_worker.evaluate migration).
# Direct producer→neuromod_worker bridge for the 7 apply_external_nudge call sites
# previously inlined in spirit_worker.py (DELETED with §4.Q aggressive cleanup):
#   producers: cognitive_worker × 6 (MSL concept valence ×2, FILTER_DOWN reasoning ×2,
#              META eureka DA burst, SPIRIT_SELF nudge) + outer_interface_worker × 1
#              (self-exploration seek_novelty DA+NE bump)
#   consumer:  neuromod_worker → NeuromodulatorSystem.apply_external_nudge(...)
#   payload:   {"nudge_map": dict[str, float], "max_delta": float,
#               "developmental_age": float, "source": str}
NEUROMOD_EXTERNAL_NUDGE = "NEUROMOD_EXTERNAL_NUDGE"

# Cross-worker hormone stimulus (NEW per rFP_phase_c_impulse_engine_d8_3_migration §2.D).
# Direct producer→hormonal_worker bridge for stimulus accumulation; replaces the
# legacy in-process call `neural_nervous_system._hormonal.get_hormone(name).accumulate(...)`
# when HormonalSystem is owned by hormonal_worker (per master plan §10 D15).
# Payload schema:
#   {
#     "hormone_name": str,    # NS_PROGRAMS canonical name (REFLEX/FOCUS/INTUITION/IMPULSE/...)
#     "stimulus": float,      # accumulation magnitude (typically deficit * scale factor)
#     "dt": float,            # integration timestep (default 0.1s)
#     "src": str,             # producer identity for telemetry ("ns_worker._run_impulse", etc.)
#     "ts": float,            # producer timestamp
#   }
# Priority lane: P1 (coalesce-by-(titan_id, hormone_name) — stimulus to same hormone
# within one tick window naturally folds).
HORMONE_STIMULUS = "HORMONE_STIMULUS"    # ns_worker → hormonal_worker (per-hormone stimulus accumulation)
# expression_worker → hormonal_worker: per-hormone depletion when an EXPRESSION
# composite fires. Restores the monolith's consumption→refractory feedback loop
# that the Phase C process split severed — expression_manager.evaluate_all used
# to deplete the in-process HormonalSystem on fire, but the worker now runs with
# hormonal_system=None (cross-process), so the consumption dict was discarded
# and composites fired every tick (EXPRESSION.SOCIAL runaway, 2026-06-01).
HORMONE_CONSUME = "HORMONE_CONSUME"      # expression_worker → hormonal_worker (per-hormone depletion on composite fire)

# Language Worker messages (spirit ↔ language process)
# SPEAK_REQUEST schema (v1.2.1 extension per SPEC §8.5 D-SPEC-38):
#   {
#     "request_id": str,
#     "candidate_words": [str],
#     "word_perturbations": dict[str, float] | None,  # NEW v1.2.1 — optional;
#         # filled by language_worker from recent WORD_PERTURBATION_HINT
#         # (≤200ms old) when outer_interface_worker is active. Backwards-
#         # compatible: consumers tolerate missing/None field and fall back
#         # to un-perturbed SPEAK (pre-v1.2.1 behavior).
#     "ts": float,
#     ... (existing felt-state payload fields, unchanged)
#   }
SPEAK_REQUEST = "SPEAK_REQUEST"          # spirit → language (compose from felt-state)
SPEAK_RESULT = "SPEAK_RESULT"            # language → spirit (sentence + perturbation deltas)
TEACHER_SIGNALS = "TEACHER_SIGNALS"      # language → spirit (MSL signals + vocab updates)
LANGUAGE_STATS_UPDATE = "LANGUAGE_STATS_UPDATE"  # language → spirit (periodic stats broadcast)

# Track 2 outer_interface_worker / cognitive_worker SPEAK gating (NEW v1.2.1 per
# SPEC §8.5 D-SPEC-38 — rFP_phase_c_self_improvement_subsystem_migration §2.A.7).
# Closes T3 SPEAK quality regression observed since 2026-05-10 deploy.
#
# Producer/consumer chain:
#   1. cognitive_worker (per SPEAK candidate) checks cache.outer_interface.advisor
#      (populated by ADVISOR_REFRACTORY_STATE). If refractory → skip.
#   2. Otherwise cognitive_worker emits SPEAK_REQUEST_PENDING with candidate_words.
#   3. outer_interface_worker consumes pending → computes
#      narrator.get_word_perturbation(w) per w → publishes WORD_PERTURBATION_HINT.
#   4. cognitive_worker emits SPEAK_REQUEST; language_worker consumes both
#      SPEAK_REQUEST + most-recent WORD_PERTURBATION_HINT (≤200ms TTL) and fills
#      SPEAK_REQUEST.word_perturbations downstream.
ADVISOR_REFRACTORY_STATE = "ADVISOR_REFRACTORY_STATE"  # outer_interface_worker → cognitive_worker (on change; coalesce=("titan_id",))
WORD_PERTURBATION_HINT = "WORD_PERTURBATION_HINT"      # outer_interface_worker → language_worker (per pending; TTL≤200ms consumer-side)
SPEAK_REQUEST_PENDING = "SPEAK_REQUEST_PENDING"        # expression_worker → cognitive_worker + outer_interface_worker (Tier-1 SPEAK detection; consumer assembles full SPEAK_REQUEST)


# expression_worker (§4.B Track 3 — extracted from cognitive_worker
# 2026-05-15 per rFP_titan_hcl_l2_separation_strategy.md §4.B).
EXPRESSION_WORKER_READY = "EXPRESSION_WORKER_READY"    # expression_worker → all (informational, on boot)

# NS_REWARD — cognitive_worker subscribes and calls
# neural_nervous_system.record_outcome(reward, program, source) on its
# in-process NS instance. Replaces direct call from cognitive_worker
# Block 8 (which is now in expression_worker, cross-process). Payload:
#   {
#     "reward":  float (0..1),          # min(1.0, urge)
#     "program": str,                   # NS_PROGRAMS canonical name (CREATIVITY/EMPATHY/...)
#     "source":  str,                   # producer-side label (e.g. "composite.art")
#     "ts":      float,
#   }
NS_REWARD = "NS_REWARD"                                # expression_worker → cognitive_worker (composite → NS program reward)

# EXPERIENCE_RECORD — restores the Record stage of the ExperienceOrchestrator
# Record→Distill→Bias loop, deleted with spirit_worker.py legacy body (72f95a6b
# D8-3) and never re-homed under Phase C. Per-worker producers (agency / social /
# language for the original 4 domains; knowledge / self_reflection / cognitive for
# Phase 1 enrichment) emit the SEMANTIC content they own; cognitive_worker — sole
# owner of the ExperienceOrchestrator + experience_orchestrator.db (G21) — ENRICHES
# on receipt with the in-proc consciousness 130d inner-state + HormonalShmReader
# snapshot + the domain plugin's perception key, then calls record_outcome().
#
# Bus-hygiene invariants (rFP §3.1, Maker constraint 2026-05-21):
#   - P2, no coalesce, TARGETED dst="cognitive_worker" (NEVER dst="all"; mirrors
#     SOCIAL_CATALYST — avoids the EXPRESSION_FIRED dst=all broadcast-flood pattern).
#   - Non-blocking publish (§8.0.ter); event-gated (never per-tick/per-epoch);
#     drop-safe (P2 drop-oldest, no producer stall); per-producer min-interval guard.
# Payload (producer-owned semantic content only):
#   {
#     "domain":        str,    # plugin domain key ("language"/"communication"/...)
#     "action_taken":  str,    # human-readable action label
#     "outcome_score": float,  # 0..1 success/quality signal
#     "context":       dict,   # small JSON-serializable provenance
#     "epoch_id":      int,    # producer's best-known epoch (0 if N/A)
#     "ts":            float,
#   }
EXPERIENCE_RECORD = "EXPERIENCE_RECORD"                # producers → cognitive_worker (Record stage of distillation loop)

# Track 2 outer_interface_worker periodic stats publishers (NEW v1.2.1 per
# SPEC §9.B outer_interface_worker Bus publications row). Coalesced at the
# broker by ("titan_id",) so under backpressure the freshest snapshot wins.
# Drive /v4/self-exploration, /v4/kin-signature, /v4/kin-society routes
# (chunk A9).
OUTER_INTERFACE_STATS_UPDATED = "OUTER_INTERFACE_STATS_UPDATED"  # 2.5s coalesced — /v4/self-exploration
KIN_SIGNATURE_UPDATED = "KIN_SIGNATURE_UPDATED"                  # 2.5s coalesced — /v4/kin-signature
KIN_SOCIETY_UPDATED = "KIN_SOCIETY_UPDATED"                      # 10s coalesced  — /v4/kin-society

# Track 2 self_reflection_worker publishers (NEW v1.2.1 per SPEC §9.B
# self_reflection_worker Bus publications row). Coalesced *_STATS_UPDATED
# at the broker by ("titan_id",) so freshest snapshot wins under backpressure;
# on-event SELF_REASONING_INSIGHT / CODING_INSIGHT / PREDICTION_GENERATED
# remain ordered (P2 default).
SELF_REFLECTION_STATS_UPDATED = "SELF_REFLECTION_STATS_UPDATED"  # 2.5s coalesced — /v4/self-reflection
SELF_REASONING_INSIGHT = "SELF_REASONING_INSIGHT"                # on insight    — cognitive_worker meta-feed
CODING_EXPLORER_STATS_UPDATED = "CODING_EXPLORER_STATS_UPDATED"  # 5s coalesced  — /v4/coding-explorer
CODING_INSIGHT = "CODING_INSIGHT"                                # on insight    — cgn_module cross-insight propagation
PREDICTION_STATS_UPDATED = "PREDICTION_STATS_UPDATED"            # 2.5s coalesced — /v4/prediction (Track 1 drift corrected)
PREDICTION_GENERATED = "PREDICTION_GENERATED"                    # on prediction — cognitive_worker novelty consumer

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

# TRAJECTORY_UPDATE — cognitive_worker emits 2D consciousness trajectory
# (state_132d[130:132] = curvature + density) every meta-reasoning chain
# conclude. RFP_meta-reasoning_CGN_FIX.md §8 — closes EmotCGN
# `trajectory` DEAD-DIM by giving emot_cgn_worker a direct subscription
# instead of routing through the retiring spirit_worker
# `_attach_emot_producer_ctx` hook. Payload: {"trajectory_2d": [c, d]}.
# Routing: src="cognitive_worker", dst="emot_cgn". Coalesce-by-titan_id
# at ~0.2 Hz (chain-conclude cadence).
TRAJECTORY_UPDATE = "TRAJECTORY_UPDATE"

# NS_URGENCIES_UPDATE — ns_worker emits 11-program urgency snapshot at
# 1Hz (KERNEL_EPOCH_TICK cadence). RFP_meta-reasoning_CGN_FIX.md §8 —
# closes EmotCGN `ns_urgencies` DEAD-DIM by giving emot_cgn_worker a
# direct subscription instead of routing through the retiring
# spirit_worker `_attach_emot_producer_ctx` hook. Payload:
# {"urgencies_by_program": {REFLEX: float, FOCUS: float, ...}, "ts":
# float}. Order normalized to emot_bundle_protocol.NS_PROGRAMS by the
# consumer. Routing: src="ns_worker", dst="emot_cgn". Coalesce-by-
# titan_id at 1Hz.
NS_URGENCIES_UPDATE = "NS_URGENCIES_UPDATE"

# SPACE_TOPOLOGY_UPDATE — cognitive_worker emits 30D digital-body space
# topology (outer_lower_topology_10D + inner_lower_topology_10D +
# whole/unified_spirit_topology_10D). RFP_meta-reasoning_CGN_FIX.md §8
# Stage 2 — closes EmotCGN `space_topology` DEAD-DIM, retires the
# spirit_worker `_attach_emot_producer_ctx` 30D-vector attach hook.
# Payload: {"space_topology_30d": [f×30]}. Routing: src=
# "cognitive_worker", dst="emot_cgn". Coalesce-by-titan_id at ~0.2 Hz.
SPACE_TOPOLOGY_UPDATE = "SPACE_TOPOLOGY_UPDATE"

# NEUROMOD_LEVELS_UPDATE — neuromod_worker emits the 6-modulator level
# tuple (DA, 5HT, NE, ACh, Endorphin, GABA) as an ordered flat vector,
# distinct from the verbose NEUROMOD_STATS_UPDATED (which carries the
# full (6,4) level/gain/phasic/tonic matrix + metadata for dashboard
# consumption). RFP_meta-reasoning_CGN_FIX.md §8 Stage 2 — closes
# EmotCGN `neuromod_state` DEAD-DIM with a clean 6D feed independent
# of the dashboard schema. Payload: {"levels_6d": [DA, 5HT, NE, ACh,
# Endorphin, GABA]}. Routing: src="neuromod_worker", dst="emot_cgn".
# Coalesce-by-titan_id at 0.4 Hz (matches existing 2.5s coalesce).
NEUROMOD_LEVELS_UPDATE = "NEUROMOD_LEVELS_UPDATE"

# PI_PHASE_UPDATE — cognitive_worker emits the 6D sphere-clock phase
# tuple (inner_body, outer_body, inner_mind, outer_mind, inner_spirit,
# outer_spirit) per consciousness epoch tick. Sourced from
# coordinator._sphere_clocks_snapshot (already read from SHM in
# cognitive_worker per Phase C G18). RFP_meta-reasoning_CGN_FIX.md §8
# Stage 2 — closes EmotCGN `pi_phase` DEAD-DIM, retires the
# spirit_worker `_attach_emot_producer_ctx` pi_phase attach hook.
# Payload: {"pi_phase_6d": [f×6]}. Routing: src="cognitive_worker",
# dst="emot_cgn". Coalesce-by-titan_id at consciousness epoch cadence
# (~1 Hz).
PI_PHASE_UPDATE = "PI_PHASE_UPDATE"

# Note: the cross-process NS-program urgency transport from
# cognitive_worker → ns_worker uses a dedicated SHM slot
# `ns_program_urgencies_input.bin` (G18-pure, see SPEC §7.1 + D-SPEC-68
# v1.13.0), NOT a bus event. ns_worker reads the slot each tick and
# applies peak-hold-decay; cognitive_worker is the G21 single-writer.
# Pattern mirrors `neuromod_inputs.bin` (§4.Q D-SPEC-57) +
# `life_force_inputs.bin` (§4.G D-SPEC-57).

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
# `titan_hcl/logic/meta_service_client.py`.
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
        OUTER_BODY_STATE, OUTER_MIND_STATE, OUTER_SPIRIT_STATE,
        OUTER_TRINITY_STATE, SPHERE_PULSE, FILTER_DOWN_V4,
    }

    # 2026-05-09 — High-rate broadcast types that flood unmigrated subscribers
    # (legacy wildcard mode = no broadcast_filter registered). Mirrors the
    # bus_socket.py:_HIGH_RATE_BROADCAST_TYPES stopgap to also cover the
    # Phase C path: Rust kernel-rs broker → Python `_client` (attach_client) →
    # DivineBus.publish() → in-process subscribers. Without this stopgap,
    # Phase C T3 (l0_rust_enabled=true) bypasses the bus_socket filter entirely
    # because messages enter via _client.publish callback, not via the socket
    # broker that owns _HIGH_RATE_BROADCAST_TYPES. Result on T3 2026-05-09:
    # 125k SPIRIT_STATE drops/10min, /v4/chi + /v4/nervous-system + /v4/neuromodulators
    # stuck on bootstrap defaults. Phase A+B Python publishers (T1+T2) are below
    # Schumann rate so this stopgap is a no-op for them. Phase D D-SPEC-82
    # retired the STATE_SNAPSHOT_RESPONSE pipeline that originally amplified
    # the impact on api subscribers — endpoints now read SHM-direct.
    #
    # Stopgap until rFP_bus_broadcast_filter_migration ships per-worker
    # ModuleSpec.broadcast_topics for every spawn_graduated worker. See BUGS.md
    # BUG-BUS-PER-WORKER-BROADCAST-FILTER-MIGRATION-INCOMPLETE-20260430.
    _HIGH_RATE_BROADCAST_TYPES = frozenset({
        SPHERE_PULSE,                           # 6 clocks × ~12 Hz Schumann pulses
        "PI_HEARTBEAT_UPDATED",                 # ~10 Hz π-heartbeat
        "BIG_PULSE",                            # frequent state aggregation
        BODY_STATE,                             # Schumann body 7.83 Hz
        MIND_STATE,                             # Schumann mind 23.49 Hz
        SPIRIT_STATE,                           # Schumann spirit × 9 = 70.47 Hz
        "TOPOLOGY_STATE_UPDATED",               # frequent topology snapshots
        # Phase C Rust types (no Python module constants — Rust-only publishers)
        "INNER_SPIRIT_FILTER_DOWN",             # 70.47 Hz from inner-spirit-rs
        "UNIFIED_SPIRIT_FILTER_DOWN",           # GLOBAL filter at Schumann
        "UNIFIED_SPIRIT_SELF_ASSEMBLED",        # 70.47 Hz from unified-spirit-rs
        "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",   # high-freq from trinity-rs
    })

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
        # Microkernel v2 Phase C C-S7 — optional outbound BusSocketClient.
        # Set via attach_client() when microkernel.l0_rust_enabled=true: the
        # Rust kernel-rs binary owns the broker socket, so the Python plugin
        # connects as a client instead of running its own broker. publish()
        # routes to the client (and Rust broker fans out to remote workers)
        # whenever _broker is None and _client is not None. Mutually exclusive
        # with _broker — only one path is active per kernel.
        self._client = None
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
            # Phase B.2 §D12 (2026-05-02): non-kernel-internal subscribe()
            # calls under socket mode. Should be 0 in steady state — workers
            # use setup_worker_bus, kernel components are kernel-internal.
            # Non-zero = contract violation; surfaces in arch_map.
            "non_kernel_internal_subscribe_under_socket": 0,
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

        Microkernel v2 Phase B.2 §D7+§D12 dual-mode contract (2026-05-02):
        Under socket mode (broker attached), kernel-internal subscribers
        receive an in-process Queue (existing behavior). External (worker-
        process) subscribers SHOULD NOT call this method directly — they
        attach to the broker via `setup_worker_bus`. If a non-kernel-internal
        name calls subscribe() while a broker is attached, that is a contract
        violation: we log a loud warning, increment a stat counter, and still
        return a regular Queue (defensive — avoid crashing legitimate kernel
        components). Once the field counter shows zero hits across the fleet
        for a soak window, we can graduate the violation to a hard raise.

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
        # Phase B.2 §D12 (2026-05-02): contract check — under socket mode,
        # non-kernel-internal callers should not call subscribe() directly.
        # Workers must use setup_worker_bus → BusSocketClient. Hitting this
        # branch is a contract violation; log loud + count for visibility.
        if self._broker is not None and not _is_kernel_internal(module_name):
            with self._lock:
                self._stats["non_kernel_internal_subscribe_under_socket"] += 1
            logger.warning(
                "[DivineBus] CONTRACT VIOLATION: bus.subscribe('%s', reply_only=%s) "
                "called under socket mode for a non-kernel-internal name. "
                "Workers must use setup_worker_bus → BusSocketClient. "
                "Returning in-process Queue defensively, but this caller is "
                "off-contract per PLAN_microkernel_phase_b2_ipc.md §D12. "
                "Update _KERNEL_INTERNAL_NAMES if this name IS kernel-internal.",
                module_name, reply_only,
            )
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
            high_rate_skip = (
                msg_type in self._HIGH_RATE_BROADCAST_TYPES
            )
            with self._lock:
                snapshot = []
                for mod_name, queues in list(self._subscribers.items()):
                    if mod_name == src:
                        continue
                    if mod_name in self._reply_only:
                        continue
                    flt = self._broadcast_filters.get(mod_name)
                    # 2026-05-09 stopgap: high-rate types only deliver to
                    # subscribers that EXPLICITLY opted in via broadcast_filter.
                    # Wildcard subscribers (flt is None) skip these to prevent
                    # Phase C T3-style queue overflow. See class-level
                    # _HIGH_RATE_BROADCAST_TYPES comment for full context.
                    if high_rate_skip and flt is None:
                        filtered_count += 1
                        continue
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
        #
        # Phase C C-S7: under l0_rust_enabled=true the broker is owned by
        # titan-kernel-rs and we have a client attached instead. Both modes
        # are mutually exclusive — only one of (_broker, _client) is set.
        broker = self._broker
        if broker is not None:
            try:
                broker.publish(msg)
            except Exception:  # noqa: BLE001
                logger.exception("[DivineBus] broker.publish raised; in-process delivery unaffected")
        else:
            client = self._client
            if client is not None:
                try:
                    client.publish(msg)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[DivineBus] client.publish raised; in-process delivery unaffected")

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
            high_rate_skip = (
                msg_type in self._HIGH_RATE_BROADCAST_TYPES
            )
            with self._lock:
                snapshot = []
                for mod_name, queues in list(self._subscribers.items()):
                    if mod_name == src:
                        continue
                    if mod_name in self._reply_only:
                        continue
                    flt = self._broadcast_filters.get(mod_name)
                    # 2026-05-09 stopgap mirror — same as publish().
                    if high_rate_skip and flt is None:
                        filtered_count += 1
                        continue
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

    # ── Microkernel v2 Phase C C-S7 — outbound client attach/detach ─────

    def attach_client(self, client) -> None:
        """Attach an outbound BusSocketClient (Phase C C-S7 / l0_rust mode).

        Used when microkernel.l0_rust_enabled=true: titan-kernel-rs owns
        the broker socket; the Python plugin connects as a client instead
        of starting its own broker. publish() will route via the client
        when _broker is None.

        Mutually exclusive with attach_broker — only one of the two paths
        is active per kernel. Idempotent (re-attach replaces).
        """
        self._client = client
        logger.info(
            "[DivineBus] outbound bus client attached (name=%s, sock=%s)",
            getattr(client, "name", "?"),
            getattr(client, "sock_path", "?"))

    def detach_client(self) -> None:
        """Detach the outbound client (kernel shutdown). Idempotent."""
        if self._client is not None:
            logger.info("[DivineBus] outbound bus client detached")
            self._client = None

    @property
    def has_socket_client(self) -> bool:
        """True if an outbound client is attached (Phase C C-S7 mode)."""
        return self._client is not None

    def flush(self, timeout: float = 5.0) -> bool:
        """SPEC §8.0.ter passthrough — block until outbound buffer drained.

        Returns True if drained within timeout; False on timeout. Returns
        True immediately if no socket client is attached (no buffer to
        drain in pure in-process mode). Use for the rare send-completion
        sites — graceful shutdown, RPC reply emits — that need the wire
        write completed before continuing. Most callers should NOT use
        this; the whole point of §8.0.ter is fire-and-forget from the
        publisher's perspective.
        """
        client = self._client
        if client is None:
            return True
        try:
            return client.flush(timeout=timeout)
        except Exception:  # noqa: BLE001
            logger.exception("[DivineBus] flush raised — treating as drained")
            return True

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

    # Phase C Session 5 (rFP §4.D.7) — DivineBus.request() is DEPRECATED.
    # State lookup MUST go via SHM (Preamble G18). Work-RPC MUST go via
    # bus.request_async with explicit timeout (Preamble G19). The list of
    # legitimate sync bus.request call sites is enumerated in
    # titan-docs/specs/phase_c_rpc_exemptions.yaml — every call site there has
    # documented rationale + bounded timeout. NEW use of bus.request()
    # is a SPEC violation (G22 — no new sync-RPC patterns post-Phase-C).
    #
    # Implementation: emit a one-shot DeprecationWarning per (file, line)
    # call site so existing exempted sites can keep working without log
    # flood, while NEW calls get an immediate, visible warning.
    _DEPRECATION_WARNED: "set[tuple[str, int]]" = set()

    def request(
        self, src: str, dst: str, payload: dict, timeout: float = 10.0, reply_queue: Optional[AnyQueue] = None
    ) -> Optional[dict]:
        """
        Synchronous request/response over the bus.

        **DEPRECATED (Phase C Session 5, rFP §4.D.7).** Use SHM-direct
        read via StateRegistryReader for state lookup (Preamble G18) or
        bus.request_async with explicit timeout for work-RPC
        (Preamble G19). New use is a SPEC violation per G22.
        Existing legitimate call sites are enumerated in
        titan-docs/specs/phase_c_rpc_exemptions.yaml.

        Publishes a QUERY, waits for a matching RESPONSE on the reply_queue.
        Caller must provide their own reply_queue (from subscribe()).
        """
        # One-shot deprecation warning per call site (file, line).
        try:
            import inspect
            import warnings as _warnings
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                caller = frame.f_back
                site = (caller.f_code.co_filename, caller.f_lineno)
                if site not in DivineBus._DEPRECATION_WARNED:
                    DivineBus._DEPRECATION_WARNED.add(site)
                    _warnings.warn(
                        f"DivineBus.request() is deprecated (Phase C Session 5 "
                        f"rFP §4.D.7). Caller: {site[0]}:{site[1]}. Use SHM "
                        f"via StateRegistryReader for state lookup (G18) or "
                        f"bus.request_async with explicit timeout for "
                        f"work-RPC (G19). See "
                        f"titan-docs/specs/phase_c_rpc_exemptions.yaml for the "
                        f"allowlist of legitimate sync sites.",
                        DeprecationWarning, stacklevel=2,
                    )
        except Exception:
            # Never let deprecation telemetry break the call path itself.
            pass

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
    # Per D-SPEC-52 (v1.7.3, 2026-05-14): record on BOTH worker-path
    # (socket mode under Phase C) AND main-process path. The original
    # `_is_main_bus` gate was added when Guardian.drain_send_queues was
    # the worker-path observer — that drain is dead for socket-mode
    # workers, so without recording here the producer-side telemetry is
    # permanently empty (`/v4/bus-health.producers = []` fleet-wide pre-fix).
    # Recording here is correct because emit_meta_cgn_signal is the choke
    # point: every emit goes through this function exactly once, no
    # double-count risk under Phase C.
    try:
        from .core.bus_health import get_global_monitor
        m = get_global_monitor()
        if m is not None:
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
    # Per D-SPEC-52 (v1.7.3, 2026-05-14): record on BOTH worker-path
    # (socket mode under Phase C) AND main-process path — Guardian drain
    # is dead for socket-mode workers; see emit_meta_cgn_signal step 5.
    try:
        from .core.bus_health import get_global_monitor
        m = get_global_monitor()
        if m is not None:
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
    context_signature: Optional[list] = None,
    binding_outcome: Optional[dict] = None,
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
        # Phase G additions (RFP_cgn_enhancements §9.3) — the numeric context
        # signature the policy used, so the teacher mints a binding in the SAME
        # cosine space; and the per-chain binding outcome so the teacher (sole
        # writer) applies the G.iv recognized/produced counters. Both optional —
        # legacy-chain payloads stay byte-for-byte compatible when absent.
        if context_signature is not None:
            payload["context_signature"] = list(context_signature)
        if binding_outcome:
            payload["binding_outcome"] = dict(binding_outcome)
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


# Bus-hygiene §3.1 (rFP_experience_distillation_phase_c): per-domain coalescing
# guard so a domain event storm in any one producer process cannot become a bus
# storm. Per-process state (each worker is its own process).
EXPERIENCE_RECORD_MIN_INTERVAL_S = 2.0
_experience_record_last_emit: dict = {}


def emit_experience_record(
    sender, src: str, domain: str, action_taken: str, outcome_score: float,
    context: Optional[dict] = None, epoch_id: int = 0,
    min_interval_s: Optional[float] = None, coalesce_key: Optional[str] = None,
) -> bool:
    """Emit a targeted EXPERIENCE_RECORD frame — Record stage of the
    ExperienceOrchestrator Record→Distill→Bias loop (rFP_experience_distillation_
    phase_c_restoration_and_enrichment.md). Per-worker producers call this with
    the SEMANTIC content they own; cognitive_worker enriches + records.

    Bus-hygiene invariants (§3.1, Maker constraint 2026-05-21):
      - TARGETED dst="cognitive_worker" (P2, NEVER dst="all" — avoids the
        EXPRESSION_FIRED dst=all broadcast-flood pattern).
      - Non-blocking (put_nowait / client publish — §8.0.ter).
      - Event-gated by a per-domain min-interval coalescing guard.
      - Drop-safe: a full queue / failed put returns False with no producer stall.

    `sender` may be a worker send_queue (put_nowait) or a bus client (publish).
    Returns True if emitted, False if coalesced or dropped.
    """
    now = time.time()
    iv = (EXPERIENCE_RECORD_MIN_INTERVAL_S
          if min_interval_s is None else min_interval_s)
    if not str(domain):
        return False
    # Coalescing key defaults to domain; producers with multiple distinct
    # experience sub-streams in one domain (e.g. language compose vs comprehend)
    # pass an explicit key so the streams rate-limit independently rather than
    # starving each other (bus hygiene without cross-stream loss).
    key = str(coalesce_key) if coalesce_key else str(domain)
    if now - _experience_record_last_emit.get(key, 0.0) < iv:
        return False  # coalesced — bus hygiene
    msg = {
        "type": EXPERIENCE_RECORD,
        "src": src,
        "dst": "cognitive_worker",
        "ts": now,
        "rid": None,
        "payload": {
            "domain": str(domain),
            "action_taken": str(action_taken)[:200],
            "outcome_score": float(outcome_score),
            "context": context if isinstance(context, dict) else {},
            "epoch_id": int(epoch_id or 0),
            "ts": now,
        },
    }
    try:
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
        else:
            return False
        _experience_record_last_emit[key] = now
        return True
    except Exception as e:
        swallow_warn("[emit_experience_record] failed", e,
                     key="bus.emit_experience_record")
        return False


# ----------------------------------------------------------------------
# publish_module_error — Phase 11 / SPEC §11.I.4 typed-error envelope
# ----------------------------------------------------------------------
# Single structured-error path per locked D-SPEC-141. Producers wrap exceptions
# either via `@with_error_envelope` (Chunk 11C decorator) or call this helper
# directly. Rate-gated to 100 envelopes/s per (module_name, error_code) tuple;
# excess emissions are dropped + counted, and the first drop in a 1s window
# triggers MODULE_ERROR_FLOOD notification so observers can render degraded UX.
#
# Wire contract: msgpack-serializable ModuleError.as_wire_dict() on topic
# MODULE_ERROR (P1, dst="all", non-blocking per §8.0.ter).

# Per-(module_name, error_code) rate-gate state.
_module_error_rate_lock = _threading.Lock()
_module_error_last_window_ts: dict[tuple, float] = {}  # (module_name, error_code) → window start
_module_error_window_count: dict[tuple, int] = {}      # (module_name, error_code) → count in window
_module_error_flood_last_emit: dict[tuple, float] = {} # (module_name, error_code) → last MODULE_ERROR_FLOOD ts
MODULE_ERROR_RATE_WINDOW_S: float = 1.0
MODULE_ERROR_RATE_LIMIT_PER_WINDOW: int = 100  # 100/s sustained; excess dropped
MODULE_ERROR_FLOOD_NOTIFY_MIN_INTERVAL_S: float = 5.0  # don't spam flood notifications


def publish_module_error(sender, error) -> bool:
    """Publish a ModuleError envelope on the MODULE_ERROR topic.

    Args:
        sender: Either a worker send_queue (has `put_nowait`) OR a DivineBus /
                bus-client (has `publish`). Duck-typed dispatch matches the
                existing emit_* helper convention in this module.
        error:  A `titan_hcl.errors.ModuleError` instance.

    Returns:
        True if the envelope was sent, False if rate-gated or send failed.

    Rate-gating (RFP §3H.3):
        Up to MODULE_ERROR_RATE_LIMIT_PER_WINDOW emissions per
        (module_name, error_code) per MODULE_ERROR_RATE_WINDOW_S. Excess
        emissions are dropped silently. The first drop in a window
        publishes one MODULE_ERROR_FLOOD notification (subject to a
        MODULE_ERROR_FLOOD_NOTIFY_MIN_INTERVAL_S debounce per same tuple).
    """
    # Import here to avoid a hard import cycle (errors.py is leaf; bus.py is core).
    from .errors import ModuleError as _ME

    if not isinstance(error, _ME):
        # Defensive: refuse to publish anything that isn't a ModuleError.
        # This guards against the common "I called publish_module_error
        # with a raw dict" mistake.
        return False

    now = time.time()
    key = (error.module_name, error.error_code)

    # 1. Rate gate (per-(module, error_code) sliding 1s window).
    with _module_error_rate_lock:
        window_start = _module_error_last_window_ts.get(key, 0.0)
        if now - window_start >= MODULE_ERROR_RATE_WINDOW_S:
            # Window rolled — reset.
            _module_error_last_window_ts[key] = now
            _module_error_window_count[key] = 0
            window_start = now
        _module_error_window_count[key] = _module_error_window_count.get(key, 0) + 1
        over_limit = _module_error_window_count[key] > MODULE_ERROR_RATE_LIMIT_PER_WINDOW
        last_flood_ts = _module_error_flood_last_emit.get(key, 0.0)
        should_notify_flood = (
            over_limit
            and (now - last_flood_ts) >= MODULE_ERROR_FLOOD_NOTIFY_MIN_INTERVAL_S
        )
        if should_notify_flood:
            _module_error_flood_last_emit[key] = now

    # 2. Optional flood notification (one per debounce window).
    if should_notify_flood:
        flood_msg = {
            "type": MODULE_ERROR_FLOOD,
            "src": error.module_name,
            "dst": "all",
            "ts": now,
            "rid": None,
            "payload": {
                "module_name": error.module_name,
                "error_code": error.error_code,
                "count_in_window": int(_module_error_window_count[key]),
                "window_s": float(MODULE_ERROR_RATE_WINDOW_S),
                "limit_per_window": int(MODULE_ERROR_RATE_LIMIT_PER_WINDOW),
            },
        }
        try:
            if hasattr(sender, "put_nowait"):
                sender.put_nowait(flood_msg)
            elif hasattr(sender, "publish"):
                sender.publish(flood_msg)
        except Exception as e:
            swallow_warn("[publish_module_error] flood-notify drop", e,
                         key="bus.publish_module_error_flood")

    # 3. Drop the envelope if over limit.
    if over_limit:
        return False

    # 4. Publish the envelope on MODULE_ERROR (P1, dst="all", non-blocking).
    msg = {
        "type": MODULE_ERROR,
        "src": error.module_name,
        "dst": "all",
        "ts": error.ts,
        "rid": error.correlation_id,
        "payload": error.as_wire_dict(),
    }
    try:
        if hasattr(sender, "put_nowait"):
            sender.put_nowait(msg)
        elif hasattr(sender, "publish"):
            sender.publish(msg)
        else:
            return False
        return True
    except Exception as e:
        swallow_warn("[publish_module_error] send failed", e,
                     key="bus.publish_module_error")
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
# Spec entries in titan_hcl/bus_specs.py are optional: messages here
# default to P2 + no-coalesce (the safe default per Phase B.2 §D6).
# Add a spec entry only if a message genuinely needs P0/P1/P3 or
# coalesce semantics — judgment per-message.

# Backup / save lifecycle
BACKUP_TRIGGER_MANUAL = "BACKUP_TRIGGER_MANUAL"

# Bus protocol — supervision transfer
BUS_HANDOFF_ACK = "BUS_HANDOFF_ACK"

# Inner↔Outer Felt-Teaching Bridge (RFP_inner_outer_felt_teaching_bridge §7.3)
# synthesis_worker → felt_teaching_worker: a decomposed Engram Object has no CGN
# felt-grounding (a gap) → a propose-only teaching candidate. Payload:
# {object_label, felt_state, source_engram, source_version, domain_hint}.
ENGRAM_FELT_CANDIDATE = "ENGRAM_FELT_CANDIDATE"

# CGN protocol
CGN_CONCEPT_GROUNDED  = "CGN_CONCEPT_GROUNDED"  # cgn → cognitive_worker: a concept matured across ≥2 consumers (Phase B Level-B trigger, RFP_cgn_enhancements §9.2)
CGN_CROSS_INSIGHT     = "CGN_CROSS_INSIGHT"
CGN_DREAM_CONSOLIDATE = "CGN_DREAM_CONSOLIDATE"
CGN_HAOV_VERIFY_REQ   = "CGN_HAOV_VERIFY_REQ"
CGN_HAOV_VERIFY_RSP   = "CGN_HAOV_VERIFY_RSP"
CGN_HAOV_RULE_APPLIED = "CGN_HAOV_RULE_APPLIED"  # consumer → cgn_worker: a verified haov rule was APPLIED to an action (C2/C3); increments source tracker's used_for_action (RFP_cgn_loop_closure §7.D — closes learning→behaviour, INV-LOOP-6)
CGN_INFERENCE_REQ     = "CGN_INFERENCE_REQ"
CGN_KNOWLEDGE_REQ     = "CGN_KNOWLEDGE_REQ"
CGN_KNOWLEDGE_RESP    = "CGN_KNOWLEDGE_RESP"
CGN_KNOWLEDGE_USAGE   = "CGN_KNOWLEDGE_USAGE"
CGN_SOCIAL_TRANSITION = "CGN_SOCIAL_TRANSITION"
CGN_WEIGHTS_MAJOR     = "CGN_WEIGHTS_MAJOR"

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

# D-SPEC-77 (SPEC v1.18.0) — explicit topic-named LLM work-RPCs.
# Replace the legacy `bus.QUERY action="distill"` / `action="score"`
# patterns (where target was llm_worker but the action was muxed via
# QUERY payload). Explicit topics give callers + scanners a clear
# producer/consumer contract. Each request supports optional
# `pre_hook: bool` / `post_hook: bool` flags so the llm_worker can
# invoke `llm_pipeline.compose_pre` / `verify_post_async` internally
# — making llm_worker the canonical "talk through the truth-gate"
# surface for non-agno generation (X replies via social_x_gateway,
# autonomous language pipeline, future avatars).
LLM_DISTILL_REQUEST  = "LLM_DISTILL_REQUEST"
LLM_DISTILL_RESPONSE = "LLM_DISTILL_RESPONSE"
LLM_SCORE_REQUEST    = "LLM_SCORE_REQUEST"
LLM_SCORE_RESPONSE   = "LLM_SCORE_RESPONSE"

# Maker dialogue + narration + proposals
MAKER_DIALOGUE_COMPLETE = "MAKER_DIALOGUE_COMPLETE"
MAKER_NARRATION_REQUEST = "MAKER_NARRATION_REQUEST"
MAKER_NARRATION_RESULT  = "MAKER_NARRATION_RESULT"
MAKER_PROPOSAL_CREATED  = "MAKER_PROPOSAL_CREATED"
MAKER_RESPONSE_RECEIVED = "MAKER_RESPONSE_RECEIVED"

# Meditation lifecycle
MEDITATION_COMPLETE        = "MEDITATION_COMPLETE"
MEDITATION_FORCE_END       = "MEDITATION_FORCE_END"      # NEW v1.8.3 §4.D — dashboard → meditation worker, Maker manual abort
MEDITATION_HEALTH_ALERT    = "MEDITATION_HEALTH_ALERT"
MEDITATION_INTERRUPTED     = "MEDITATION_INTERRUPTED"    # NEW v1.8.3 §4.D — meditation worker → all, abnormal termination
MEDITATION_PHASE_CHANGED   = "MEDITATION_PHASE_CHANGED"  # NEW v1.8.3 §4.D — meditation worker → all, phase state-machine transition
MEDITATION_RECOVERY_TIER_1 = "MEDITATION_RECOVERY_TIER_1"
MEDITATION_RECOVERY_TIER_2 = "MEDITATION_RECOVERY_TIER_2"
MEDITATION_REQUEST         = "MEDITATION_REQUEST"

# Memory ops
MEMORY_ADD                 = "MEMORY_ADD"
MEMORY_INGEST_COMPLETED    = "MEMORY_INGEST_COMPLETED"           # memory worker → all (broadcast, completion ack for INGEST_REQUEST; carries node_id + weight + effective_weight, filtered by request_id)
MEMORY_INGEST_REQUEST      = "MEMORY_INGEST_REQUEST"             # producers → memory worker (one-way, no RPC; replaces work-RPC `add` action — Phase B rFP §3.4.1)
MEMORY_MEMPOOL_ADD         = "MEMORY_MEMPOOL_ADD"                # chat → memory worker (one-way, no RPC)
MEMORY_REINFORCE_NODE      = "MEMORY_REINFORCE_NODE"            # EEL-A2 confirm/dispute → reinforce mempool node (one-way)
MEMORY_TICK_CONFIRMATION   = "MEMORY_TICK_CONFIRMATION"         # EEL-A2 neutral → tick a pending node's confirmation window (one-way)
MEMORY_RECALL_PERTURBATION = "MEMORY_RECALL_PERTURBATION"

# Meta-reasoning rewards + signals
META_DIVERSITY_PRESSURE = "META_DIVERSITY_PRESSURE"
META_EUREKA             = "META_EUREKA"
META_EVENT_REWARD       = "META_EVENT_REWARD"
# META_INTROSPECT_REQUEST — P2 fire-and-forget trigger from cognitive_worker's
# _prim_introspect to self_reflection_worker. Payload: {sub_mode,
# effective_sub_mode, epoch, neuromods, msl_data, reasoning_stats,
# language_stats, coordinator_data, state_132d, ts}. Handler runs
# SelfReasoningEngine.introspect(**payload) → persists to
# data/inner_memory.db.self_insights via _persist_insight() → writes result
# to inner_self_insight.bin SHM slot for cognitive_worker's next tick.
# SPEC §9.B / D-SPEC-70 v1.15.0 — closes F-8 fleet-wide.
META_INTROSPECT_REQUEST = "META_INTROSPECT_REQUEST"
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

# C-S5 Python L2 worker boot signals — peers know the slot writers are
# live (sibling triad with REFLEX_READY pattern).
HORMONAL_READY = "HORMONAL_READY"
NEUROMOD_READY = "NEUROMOD_READY"
NS_READY = "NS_READY"

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

# Phase C-S9 social_worker (rFP_titan_hcl_l2_separation_strategy §4.C +
# PLAN_microkernel_phase_c_s9_social_worker_extraction §11.2). 12 new event
# types — see PLAN §11.2 table for full direction/priority/coalesce taxonomy.
# All P3 (social tier per bus_specs.py docstring §1). No coalesce — each
# event is a distinct signal, not a state-update.
#
# §4.C-spec'd subscriptions (existed in spec, never registered):
KIN_SIGNAL                       = "KIN_SIGNAL"                        # spirit/cognitive → social
SOCIAL_RECEIVED                  = "SOCIAL_RECEIVED"                   # social → all (DM/mention)
# Catalyst event — ONE generic event with `type` in payload. Replaces the
# in-process _x_catalysts.append flow at 8 producer sites (chunk 9I rewires).
# Catalyst-type taxonomy lives in payload not in event name (easier extension,
# fewer bus_specs entries, only one subscriber today). Known types in payload:
# eureka_spirit, dream_summary, kin_resonance, emotion_shift, onchain_anchor,
# vulnerability, strong_composition, plus variable Maker-forced types.
SOCIAL_CATALYST                  = "SOCIAL_CATALYST"                   # producers → social
# Publishers (post-success notification + community state changes):
X_POST_PUBLISHED                 = "X_POST_PUBLISHED"                  # social → all (engagement reaper, KIN_RESONANCE coord, Observatory)
SOCIAL_GRAPH_UPDATE              = "SOCIAL_GRAPH_UPDATE"               # social → all (community_registry change)
# rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1, 2026-05-14).
# social_graph_worker publishes 5 informational events. Stats bulk payload via
# SHM slot social_graph_state.bin (§7.1) per rFP_bus_payload_contracts §3.1 —
# *_STATS_UPDATED is a notification only.
SOCIAL_GRAPH_READY               = "SOCIAL_GRAPH_READY"                # social_graph → all (boot signal)
SOCIAL_GRAPH_STATS_UPDATED       = "SOCIAL_GRAPH_STATS_UPDATED"        # social_graph → all (5s coalesced; bulk via SHM)
SOCIAL_INTERACTION_RECORDED      = "SOCIAL_INTERACTION_RECORDED"       # social_graph → all (per record_interaction write)
SOCIAL_DONATION_RECORDED         = "SOCIAL_DONATION_RECORDED"          # social_graph → all (per record_donation write)
SOCIAL_INSPIRATION_RECORDED      = "SOCIAL_INSPIRATION_RECORDED"       # social_graph → all (per record_inspiration write)

# rFP_titan_hcl_l2_separation_strategy.md §4.J + D-SPEC-51 (SPEC v1.7.2,
# 2026-05-14) — metabolism_worker bus contract. METABOLIC_TIER_CHANGED is
# P1 (on every tier transition; consumed by life_force_worker §4.G future,
# dashboard, social_x_gateway, observatory_writer). GATE_DECISION_RECORDED
# is P3 coalesce-by-feature (per evaluate_gate; authoritative ring buffer
# writer is metabolism_worker per Maker-locked G19-strict design).
# METABOLIC_STATS_UPDATED is P3 coalesce-by-type (1Hz notification; bulk
# via metabolism_state.bin SHM slot per rFP_bus_payload_contracts §3.1).
METABOLIC_TIER_CHANGED           = "METABOLIC_TIER_CHANGED"            # metabolism → all (on tier transition)
GATE_DECISION_RECORDED           = "GATE_DECISION_RECORDED"            # metabolism → all (per evaluate_gate)
METABOLIC_STATS_UPDATED          = "METABOLIC_STATS_UPDATED"           # metabolism → all (1Hz; bulk via SHM)

# Life-force / Chi (v1.8.3 §4.G — D-SPEC-57). Bus event surface for the
# new life_force_worker (Python L2). LIFE_FORCE_UPDATED is P3 1Hz
# coalesce-by-titan_id notification; bulk via life_force_state.bin SHM
# slot per rFP_bus_payload_contracts §3.1. FATIGUE_LEVEL_CRITICAL is P1
# single-shot on _metabolic_drain ≥ 0.7 upward crossing (edge-debounced,
# resets when drain ≤ 0.6 to avoid threshold-edge oscillation). Publish-
# only this rFP per Maker Q6; consumer wiring deferred to follow-up.
LIFE_FORCE_UPDATED               = "LIFE_FORCE_UPDATED"                # life_force_worker → all (1Hz; bulk via life_force_state.bin SHM slot)
FATIGUE_LEVEL_CRITICAL           = "FATIGUE_LEVEL_CRITICAL"            # life_force_worker → all (single-shot on drain≥0.7 upward, edge-debounced)

# Per-Titan polling broadcasts (canonical poller → fleet consumers, chunks 9N-9O):
MENTION_RECEIVED                 = "MENTION_RECEIVED"                  # canonical poller → other social_workers
FELT_EXPERIENCE_CAPTURED         = "FELT_EXPERIENCE_CAPTURED"          # canonical poller → other social_workers
ENGAGEMENT_SNAPSHOT_TAKEN        = "ENGAGEMENT_SNAPSHOT_TAKEN"         # canonical poller → other social_workers
KNOWLEDGE_REUSE_HIT              = "KNOWLEDGE_REUSE_HIT"               # agno knowledge-cache reuse → synthesis chain_reuse (Affective §7.C)

# rFP_health_monitor_worker.md + D-SPEC-67 (SPEC v1.12.0, 2026-05-17) — pluggable
# L3 health-monitor framework bus surface. SOLE-sanctioned heal path = bus
# (health_monitor → owning worker via HEAL_REQUEST; owning worker → health_monitor
# via HEAL_RESULT). Preserves `feedback_social_x_gateway_post_is_sole_sanctioned
# _x_path.md` — health_monitor never instantiates a second SocialXGateway in its
# own process; refresh_session always runs inside social_worker against the live
# in-proc gateway state. Future plugins (backup_arweave → backup_worker,
# language_teacher → language_worker, meditation → meditation_worker) follow the
# identical bus-mediated heal contract.
HEALTH_CHECK_RESULT              = "HEALTH_CHECK_RESULT"               # health_monitor → all (per (plugin, layer) per pass; P2 coalesce-by-(plugin,layer))
HEAL_REQUEST                     = "HEAL_REQUEST"                      # health_monitor → <owning_worker> (P2 targeted; correlation_id; 60s reply timeout)
HEAL_RESULT                      = "HEAL_RESULT"                       # <owning_worker> → health_monitor (P2 targeted; reply with success/reason)
HEALTH_HEAL_ATTEMPT              = "HEALTH_HEAL_ATTEMPT"               # health_monitor → all (P2 broadcast; emitted AFTER HEAL_RESULT or timeout)
HEALTH_HEAL_FAILED               = "HEALTH_HEAL_FAILED"                # health_monitor → all (P1 broadcast; daily-cap exhaustion or 3 consecutive failures; triggers Maker alert)

