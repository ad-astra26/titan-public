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
import logging
import time
import uuid
from queue import Empty, Full, Queue as ThreadQueue
from multiprocessing import Queue as MPQueue
from typing import Callable, Optional, Union

from .shared_blackboard import SharedBlackboard
from .core import bus_census as _census  # opt-in instrumentation (TITAN_BUS_CENSUS=1)

# Type alias: either threading or multiprocessing Queue
AnyQueue = Union[ThreadQueue, MPQueue]

logger = logging.getLogger(__name__)

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

# Control messages
MODULE_READY = "MODULE_READY"
MODULE_HEARTBEAT = "MODULE_HEARTBEAT"
MODULE_SHUTDOWN = "MODULE_SHUTDOWN"
MODULE_CRASHED = "MODULE_CRASHED"
EPOCH_TICK = "EPOCH_TICK"

# Disk health — edge-detected state transitions only (never continuous
# telemetry). Published by DiskHealthMonitor when free-space crosses
# hysteresis-protected thresholds. EMERGENCY triggers Guardian.stop_all.
# RESERVED: DiskHealthMonitor class exists but is not instantiated in the
# current boot path — constants are reserved for when the monitor is wired.
DISK_WARNING = "DISK_WARNING"
DISK_CRITICAL = "DISK_CRITICAL"
DISK_EMERGENCY = "DISK_EMERGENCY"
DISK_RECOVERED = "DISK_RECOVERED"  # emitted when state improves back to HEALTHY
# RESERVED: RL_CYCLE_REQUEST + ANCHOR_REQUEST — reserved hook points for
# offline IQL cycle scheduling and on-chain anchor scheduler (not yet wired).
RL_CYCLE_REQUEST = "RL_CYCLE_REQUEST"
ANCHOR_REQUEST = "ANCHOR_REQUEST"

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

# V4 Time Awareness messages
OUTER_TRINITY_STATE = "OUTER_TRINITY_STATE"
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

# Bus backpressure — edge-detected when worker queue depths cross > 30%
# (enter) or < 20% (exit) threshold. Published by BusHealthMonitor only on
# state transitions. Respects bus-clean invariant (discrete events only).
# RESERVED: BusHealthMonitor.update_queue_depths is defined but currently
# unused (also surfaced in orphan list). Constant reserved for when the
# monitor's periodic sample loop is wired.
BUS_BACKPRESSURE = "BUS_BACKPRESSURE"


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
        # module_name → list[Queue]  (a module may have multiple subscribers)
        self._subscribers: dict[str, list[AnyQueue]] = {}
        # All registered module names (for broadcast)
        self._modules: set[str] = set()
        # Modules that only receive targeted messages (excluded from dst="all" broadcasts)
        self._reply_only: set[str] = set()
        self._stats = {"published": 0, "dropped": 0, "routed": 0}
        # Optional callback called during request() polling to drain worker send queues.
        # Set by TitanCore to guardian.drain_send_queues so responses flow back.
        self._poll_fn: Optional[Callable] = None
        # Per-(src,dst) timeout warning throttle (audit fix I-014, 2026-04-08)
        # Prevents boot-time race spam from masking real persistent timeout issues.
        self._timeout_warned_at: dict[tuple[str, str], float] = {}
        # Bus census (Phase E.1): if enabled via TITAN_BUS_CENSUS=1, register
        # this bus as the depth sampler so the census writer can sample
        # qsize() of every subscriber queue periodically.
        _census.register_depth_sampler(self.sample_census_depths)

    def subscribe(self, module_name: str, reply_only: bool = False) -> AnyQueue:
        """Register a module and return its dedicated receive queue.

        Args:
            module_name: Unique name for this subscriber.
            reply_only: If True, this queue only receives targeted messages
                        (e.g. RESPONSE with dst=module_name) and is excluded
                        from dst="all" broadcasts.  Use for proxy reply queues
                        that never consume broadcast state messages.
        """
        if self._multiprocess:
            q: AnyQueue = MPQueue(maxsize=self._maxsize)
        else:
            q = ThreadQueue(maxsize=self._maxsize)
        self._subscribers.setdefault(module_name, []).append(q)
        self._modules.add(module_name)
        if reply_only:
            self._reply_only.add(module_name)
        logger.info("[DivineBus] Module '%s' subscribed (queues: %d, reply_only=%s)",
                     module_name, len(self._subscribers[module_name]), reply_only)
        return q

    def unsubscribe(self, module_name: str, q: AnyQueue) -> None:
        """Remove a specific queue for a module."""
        if module_name in self._subscribers:
            try:
                self._subscribers[module_name].remove(q)
            except ValueError:
                pass
            if not self._subscribers[module_name]:
                del self._subscribers[module_name]
                self._modules.discard(module_name)
            logger.info("[DivineBus] Module '%s' unsubscribed", module_name)

    @property
    def blackboard(self) -> SharedBlackboard:
        """Access the shared blackboard for latest-value state reads."""
        return self._blackboard

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
            # Broadcast to every registered module (except sender and reply-only)
            src = msg.get("src", "")
            for mod_name, queues in self._subscribers.items():
                if mod_name == src or mod_name in self._reply_only:
                    continue
                for q in queues:
                    if self._try_put(q, msg, subscriber=mod_name):
                        delivered += 1
        elif dst in self._subscribers:
            for q in self._subscribers[dst]:
                if self._try_put(q, msg, subscriber=dst):
                    delivered += 1
        else:
            logger.debug("[DivineBus] No subscriber for dst='%s', msg type=%s",
                         dst, msg.get("type"))

        self._stats["routed"] += delivered
        return delivered

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
        """
        _census.sample_queue_depths(self._subscribers)

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
            self._timeout_warned_at[pair] = now
        else:
            logger.debug("[DivineBus] Request timed out (throttled): %s → %s (rid=%s)",
                         src, dst, rid)
        return None

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
        logger.debug(
            "[emit_meta_cgn_signal] dispatch failed (%s, %s) from %s: %s",
            consumer, event_type, src, _disp_err)
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
