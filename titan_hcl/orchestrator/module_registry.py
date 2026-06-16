"""
titan_hcl.orchestrator.module_registry — dataclasses + supervision-log helper.

Phase 6 (D-SPEC-135) carved this out of titan_hcl/guardian.py into
guardian_hcl/module_registry.py. Phase 11 §11.I.1 (D-SPEC-141 / v1.65.0)
relocates it under titan_hcl/orchestrator/ alongside the Orchestrator class.
Dataclasses + free functions only — no orchestrator behaviour here. Imports
preserved verbatim so dataclass field defaults / annotations resolve
identically.
"""
"""
Guardian — Module supervisor for Titan V4.0 microkernel.

Manages the lifecycle of supervised module processes:
  - Start/stop/restart individual modules
  - Monitor heartbeats (kill/restart on timeout)
  - Track RSS per module (restart on threshold breach)
  - Provide module status to Core via the Divine Bus
  - Sliding-window restart tracking (prevents infinite restart loops)
  - Per-module heartbeat timeout (Spirit needs longer for heavy V4 work)

Each module runs as a separate multiprocessing.Process with its own
memory space, communicating exclusively through the Divine Bus.
"""
import asyncio
import logging
import os
import queue as _queue_mod
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from queue import Empty
from typing import Callable, Optional

from titan_hcl.bus import (
    AnyQueue,
    BUS_PEER_DIED,
    BUS_WORKER_ADOPT_ACK,
    BUS_WORKER_ADOPT_REQUEST,
    DivineBus,
    MODULE_CRASHED,
    MODULE_HEARTBEAT,
    MODULE_READY,
    MODULE_RELOAD_ACK,
    MODULE_RELOAD_REQUEST,
    MODULE_SHUTDOWN,
    SUPERVISION_CHILD_DOWN,
    SUPERVISION_CHILD_RESTARTED,
    SUPERVISION_DEPENDENCY_ACTIVATING,
    SUPERVISION_DEPENDENCY_BLOCKED,
    SUPERVISION_DEPENDENCY_DEGRADED,
    SUPERVISION_DEPENDENCY_RECOVERED,
    SUPERVISION_ESCALATION,
    make_msg,
)
from titan_hcl import bus
from titan_hcl._phase_c_constants import (
    ADOPTION_TIMEOUT_S,
    MODULE_RELOAD_DEFAULT_TIMEOUT_S,
    MODULE_RELOAD_HAPPY_PATH_S,
    SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S,
)
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
    EscalationDecision,
    ReasonRecord,
    SupervisionReason,
    classify_exit_code,
    kernel_default_decision,
    most_common_reason,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = 10.0       # seconds between expected heartbeats
HEARTBEAT_TIMEOUT = 90.0        # seconds before declaring a module dead (mainnet-safe: ~Schumann×27)
DEFAULT_RSS_LIMIT_MB = 1500     # per-module RSS limit (MB)
RESTART_BACKOFF_BASE = 2.0      # exponential backoff base (seconds)
MAX_RESTARTS_IN_WINDOW = 5      # max restarts allowed in the sliding window
RESTART_WINDOW_SECONDS = 600.0  # 10-minute sliding window for restart tracking
SUSTAINED_UPTIME_RESET = 300.0  # 5 minutes of uptime before restart count resets
REENABLE_COOLDOWN_S = 180.0    # RFP_supervision_lifecycle §7.C — 3min (was 600) auto-re-enable cooldown
# CPU-aware heartbeat (added 2026-04-21) — when heartbeat times out, sample
# /proc/<pid>/stat CPU time. If CPU grew ≥ MIN_CPU_DELTA_FOR_ALIVE since last
# sample, the module is alive-but-CPU-starved (not deadlocked). Defer restart
# for up to MAX_STARVED_CYCLES wallclock heartbeat windows; then force-restart
# (bounded grace prevents runaway hang on a truly stuck module).
MIN_CPU_DELTA_FOR_ALIVE = 1.0   # seconds of CPU time per heartbeat window proves liveness
MAX_STARVED_CYCLES = 5          # how many consecutive starved-but-alive cycles to tolerate
# Bumped 3 → 5 on 2026-04-21 after observing both T2+T3 media modules hit
# grace-exhausted-restart once each during the same 75-min ARC iter-3 slot.
# 5 cycles ≈ 5 minutes wallclock grace under monitor_tick=5s — should bridge
# typical ARC tail without leaving truly-stuck modules hanging too long.




class ModuleState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    # Phase 11 (SPEC §11.I.2 / D-SPEC-141): worker has finished in-process
    # scaffolding; titan_hcl will detect this state via 1Hz SHM poll and
    # dispatch MODULE_PROBE_REQUEST.
    BOOTED = "booted"
    # Phase 11: worker received MODULE_PROBE_REQUEST and is running probe_fn.
    PROBING = "probing"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    CRASHED = "crashed"
    DISABLED = "disabled"


@dataclass
class ModuleSpec:
    """Specification for a supervised module."""
    name: str
    entry_fn: Callable  # function(bus_queue, config) → runs in child process
    config: dict = field(default_factory=dict)
    rss_limit_mb: int = DEFAULT_RSS_LIMIT_MB
    autostart: bool = False      # start immediately on Guardian boot
    lazy: bool = True            # start on first use (via proxy)
    restart_on_crash: bool = True
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT  # per-module override
    reply_only: bool = False     # if True, skip dst="all" broadcasts (only receive targeted msgs)
    # SPEC §11.B.4 INV-PROC-5 / §11.B.5 (2026-06-02) — kernel-supervised peer.
    # When True, this module's liveness + restart are owned SOLELY by
    # titan-kernel-rs (kernel_supervisor.rs spawns/health-gates/drains/respawns
    # it), NOT by guardian_hcl. The ModuleSpec stays in the catalog so it is
    # still enumerable in /v6/* readouts (heartbeat / rss / layer metadata), but
    # the L1 Supervisor.monitor_tick MUST NOT police it (a shm_pid_dead check
    # races the kernel's own respawn / zero-downtime swap → spurious
    # MODULE_RESTART_REQUEST → a doomed orchestrator spawn that loses the port
    # and zombies), and Orchestrator.start() MUST refuse it (the kernel is the
    # sole spawner; an orchestrator spawn would collide with the kernel peer).
    # Set True ONLY for the L3 `titan_hcl_api` peer (the lone kernel-spawned
    # entry in the Guardian module catalog).
    kernel_supervised: bool = False
    # Microkernel v2 Phase A §A.5 (2026-04-24): layer assignment.
    # Canonical values in titan_hcl._layer_canon.LAYER_CANON.
    # Validated in Guardian.register(). Used by arch_map, dashboard,
    # and layer-aware crash logging.
    layer: str = "L3"
    # Microkernel v2 Phase A §A.3 (2026-04-25, S6): spawn vs fork
    # start method. Default "fork" preserves Guardian's current
    # byte-identical behavior. When set to "spawn", Guardian boots
    # the worker via multiprocessing.get_context("spawn") — fresh
    # interpreter, no parent COW inheritance. Saves ~200 MB RSS per
    # worker (fork baseline ~265 MB → spawn baseline ~50-80 MB).
    # Unknown values fall back to "fork" with a WARNING (Guardian
    # never crashes boot on a misconfigured ModuleSpec).
    start_method: str = "fork"
    # Microkernel v2 Phase B.2 (2026-04-30): broadcast topic filter.
    # When non-empty + bus_ipc_socket_enabled=true, the broker filters
    # `dst="all"` broadcasts at publish time so only messages with
    # `type` in this list reach the subscriber. Empty list = legacy
    # "subscribe-all" (every broadcast delivered) — preserved for
    # backward compatibility with workers not yet migrated.
    # Closes the per-subscriber flood class identified 2026-04-30
    # (backup queue receiving SPHERE_PULSE/SPIRIT_STATE/etc. it never
    # consumes). See bus_socket.py:563 BrokerSubscriber.publish docs.
    #
    # ⚠️ NOT a CPU lever (PROFILING.md F1, under-load --gil sweep 2026-05-30):
    # the "recv_exact 45-100%" that motivated this filter was py-spy WITHOUT
    # --gil counting BLOCKED bus-receive threads — the named workers burn 0%
    # CPU idle AND under load. Any value here is bus-throughput / msgpack-unpack
    # churn / correctness, NOT baseline CPU. Do not justify migration on CPU.
    broadcast_topics: list = field(default_factory=list)
    # Microkernel v2 Phase B.2.1 §M5 (2026-04-27 PM): adoption criticality.
    # When True (default): worker MUST adopt for shadow swap to succeed —
    # holds heavy in-process state (FAISS, DuckDB, audit chain, neural
    # nets, vocabulary) that would be expensive to re-load. When False:
    # nice-to-adopt — orchestrator declares swap successful regardless;
    # if this worker doesn't adopt by timeout, it's left to self-SIGTERM
    # via supervision daemon's bus-as-supervision check, and shadow's
    # Guardian respawns it fresh post-swap (light-state workers like
    # autonomous writers, observability aggregators, periodic backup
    # daemons).
    b2_1_swap_critical: bool = True
    # SG6 (2026-05-14) — Phase 13 Sage Socialite social_graph_worker
    # extraction (rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50).
    # Marker for modules that own a critical-data SHM slot (e.g.,
    # social_graph_state.bin G21 single-writer). Used by:
    #   - shadow_swap_orchestrator.py: gates SWAP_CHECKPOINT_REQUEST scope
    #     to only critical-data writers (per SPEC §8.3 / §12.E.1)
    #   - backup_worker: prioritizes critical-data slot snapshotting in
    #     incremental capture cadence
    # Closed BUG-MODULESPEC-CRITICAL-DATA-WRITER-FIELD-MISSING-20260514
    # (kwarg added in plugin.py:1030 + legacy_core.py:911 by SG6 commit
    # but field declaration was missed — TypeError on T2 Phase C boot).
    critical_data_writer: bool = False
    # Phase C C-S7 (2026-05-05) — declarative dependencies per SPEC §11.G.1.
    # Default empty → no pre-respawn dep check (legacy behavior). Future
    # commits populate per-module (e.g., social_module.dependencies =
    # [Dependency("x_api_reachable", EXTERNAL_SVC, SOFT, ...)]).
    dependencies: list[Dependency] = field(default_factory=list)
    # Phase 11 (SPEC §11.I.3 / D-SPEC-141): optional per-module readiness probe.
    # When None, the module gets a trivial pass-through probe
    # (`lambda bus: ProbeResult.ok_()`) — legacy modules require no migration
    # to remain compatible. When provided, must complete ≤2s wall time + must
    # be pure observation (no state mutation, no main-thread locks, no asyncio
    # block). Probes run on a titan_hcl-side thread; communicate with target
    # worker via MODULE_PROBE_REQUEST/RESPONSE bus-RPC pair.
    # Signature: `Callable[[BusClient], titan_hcl.core.module_state.ProbeResult]`.
    # Populated per-worker in Chunk 11H (10 heaviest workers).
    probe_fn: Optional[Callable] = None
    # Phase 11 (SPEC §11.I.8 / D-SPEC-141): per-module boot-priority partition.
    # - MANDATORY:          part of Phase A; gates fleet_ready=true SHM publication.
    # - OPTIONAL_POST_BOOT: scheduled in Phase B background after fleet ready.
    # - LAZY:               never auto-started; pre-activated via §11.G.2.5
    #                       ENSURE_RUNNING from a consumer dep.
    # New modules default to MANDATORY (preserves current pre-Phase-11
    # behaviour). Today's `lazy=True` migrates 1:1 to LAZY in Chunk 11G.
    # Stored as a string here to avoid an import cycle into core.module_state;
    # canonical values match BootPriority enum values
    # ("mandatory" / "post_boot" / "lazy"). Validation lives in 11F orchestrator.
    boot_priority: str = "mandatory"
    # Phase 11 §11.I.5 (2026-05-28): does this module publish a
    # `module_<name>_state.bin` lifecycle slot (STARTING→BOOTED→PROBING→
    # RUNNING)? Default True — standard supervised workers do. False for the
    # imw-class persistence-writer daemons (imw / observatory_writer / the
    # universal-sqlite-writer loop): they're reply_only Unix-socket daemons
    # supervised via socket-heartbeat, NOT the SHM lifecycle, so they never
    # write a slot. Such modules are EXCLUDED from the /v6/readiness roster —
    # else they'd read as a permanent not_booted (running fine, just not
    # lifecycle-slot-reporting) and trip a false "mandatory MISSING".
    reports_lifecycle_slot: bool = True


@dataclass
class ModuleInfo:
    """Runtime state for a supervised module."""
    spec: ModuleSpec
    state: ModuleState = ModuleState.STOPPED
    process: Optional[Process] = None
    pid: Optional[int] = None
    queue: Optional[AnyQueue] = None     # module's receive queue (bus→worker)
    send_queue: Optional[AnyQueue] = None  # module's send queue (worker→bus)
    last_heartbeat: float = 0.0
    start_time: float = 0.0
    restart_count: int = 0
    last_restart: float = 0.0
    rss_mb: float = 0.0
    restart_timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_RESTARTS_IN_WINDOW + 1))
    ready_time: float = 0.0  # when MODULE_READY was received
    disabled_at: float = 0.0  # when module was disabled (for auto-re-enable cooldown)
    # CPU-aware heartbeat (added 2026-04-21 after iter-3 ARC load triggered
    # cascading media-module restart loops on shared T2/T3 VPS — modules
    # were CPU-starved, not deadlocked, but wallclock heartbeat fired anyway).
    last_cpu_time: float = 0.0           # /proc/<pid>/stat utime+stime sample (seconds)
    last_cpu_sample_ts: float = 0.0      # when last_cpu_time was sampled
    consecutive_starved_cycles: int = 0  # heartbeat misses where CPU grew (alive-but-starved)
    consecutive_rss_over_cycles: int = 0  # cycles RssAnon > rss_limit — THROTTLE not respawn (INV-SUP-1/2, RFP §7.B/F)
    cpu_delta_s: float = 0.0             # SPEC §1339 — per-interval CPU seconds self-reported in MODULE_HEARTBEAT
    # Microkernel v2 Phase B.2.1 (2026-04-27): worker supervision-transfer.
    # When True, this ModuleInfo refers to an externally-spawned worker
    # (adopted from a prior kernel via BUS_WORKER_ADOPT_REQUEST). We do NOT
    # own info.process (it stays None); cleanup uses os.kill instead of
    # process.kill(). First heartbeat after adoption resets clocks fresh.
    adopted: bool = False
    adopt_ts: float = 0.0                # when adoption completed (for telemetry)
    # Phase C C-S7 (2026-05-05) — rolling reason buffer (last 16) per
    # SPEC §11.B step 3, used to compute most_common_reason for
    # SUPERVISION_ESCALATION payloads (§11.B.1 step 1).
    reason_buffer: deque = field(default_factory=lambda: deque(maxlen=16))
    # Phase C C-S7 — track in-flight escalations + dependency-blocked state.
    last_escalation_id: Optional[str] = None
    blocked_dependency: Optional[str] = None
    blocked_since: float = 0.0
    # SPEC §11.B.3 (Phase B, D-SPEC-49) — per-module hot-reload state.
    # Set True by Guardian.reload_module() on entry; cleared on terminal
    # status (ready / failed / rolled_back). While True, monitor_tick MUST
    # skip restart paths (heartbeat-timeout / CPU-starvation / RSS-overflow)
    # for this module's OLD pid; restart_async() from any other caller
    # returns None. Bounded by the orchestrator-level timeout
    # MODULE_RELOAD_DEFAULT_TIMEOUT_S=30s so supervision authority is
    # always recoverable. Single-threaded — only the orchestrator
    # mutates this under Guardian._reload_lock.
    reload_in_flight: bool = False


@dataclass
class ReloadState:
    """SPEC §11.B.3 (D-SPEC-49) — orchestrator state for one in-flight
    per-module hot-reload. Owned by Guardian._reloads_in_flight (keyed by
    module_name). Lifetime is from `reload_module()` entry to terminal
    MODULE_RELOAD_ACK emission; the orchestrator deletes the entry under
    `Guardian._reload_lock` before returning.

    The two queues route ADOPTION_REQUEST + MODULE_READY frames from
    `_process_guardian_messages` to the orchestrator thread without
    blocking message processing. They are bounded but generously sized
    (8 frames) because at most one of each is expected per reload.
    """
    swap_id: str
    module_name: str
    old_pid: int
    new_module_path: Optional[str]
    started_ts: float
    status: str = "spawning"          # spawning → adopted → ready | failed | rolled_back
    new_process: Optional[Process] = None
    new_pid: Optional[int] = None
    new_queue: Optional[AnyQueue] = None
    new_send_queue: Optional[AnyQueue] = None
    new_recv_queue_registered: bool = False
    error: Optional[str] = None
    failed_step: Optional[str] = None
    # Inter-thread routing — _process_guardian_messages fills adoption_q so the
    # orchestrator thread (running on _restart_executor via _reload_module_sync)
    # doesn't block the bus drain loop.
    adoption_q: "_queue_mod.Queue" = field(
        default_factory=lambda: _queue_mod.Queue(maxsize=8)
    )
    # NOTE: the legacy `ready_q` field was REMOVED 2026-06-01 — it held the
    # MODULE_READY bus event the reload-completion wait used pre-Phase-11. That
    # broadcast was deleted (D-SPEC-141 locked D1); reload now waits on the
    # NEW worker's SHM `state=running` slot via `_wait_for_module_running`
    # (§11.I.2/§11.I.6). Nothing fed ready_q after the migration, so the field
    # was dead + the reload-completion wait silently timed out. No shim.


def _append_meta_cgn_emission_log(msg: dict, payload: dict) -> None:
    """Append one META_CGN_SIGNAL emission event to the persistent JSONL log.

    Called from the Guardian drain loop after each drained META_CGN_SIGNAL.
    Schema v1: {ts, src, consumer, event_type, intensity, domain, reason,
                schema_version}. Missing optional fields recorded as null.
    """
    import json
    event = {
        "ts": float(msg.get("ts", time.time())),
        "src": str(msg.get("src", "unknown")),
        "consumer": str(payload.get("consumer", "")),
        "event_type": str(payload.get("event_type", "")),
        "intensity": float(payload.get("intensity", 1.0)),
        "domain": (str(payload["domain"])[:40] if payload.get("domain") else None),
        "reason": (str(payload["reason"])[:200] if payload.get("reason") else None),
        "schema_version": 1,
    }
    # Ensure parent dir exists (cheap, idempotent)
    _dir = os.path.dirname(_META_CGN_EMISSION_LOG_PATH)
    if _dir and not os.path.isdir(_dir):
        os.makedirs(_dir, exist_ok=True)
    with open(_META_CGN_EMISSION_LOG_PATH, "a") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")
