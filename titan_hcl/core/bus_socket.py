"""
bus_socket — Unix-domain pub/sub broker for Microkernel v2 Phase B.2.

Replaces DivineBus's mp.Queue backend with a single broker socket bound by
the kernel. Workers connect outward, hold a persistent connection,
re-publish their subscription list on (re)connect, and survive kernel
swaps without restarting. The kernel is replaced under live workers;
their Schumann ticks at 7.83/23.49/70.47 Hz never pause.

This module ships:
  - BoundedRing — per-subscriber ring buffer with P0 reserve + coalesce
  - BrokerSubscriber — per-connection state held by the broker
  - BusSocketServer — accept loop, handshake, dispatch, ping/pong heartbeat

Client side (BusSocketClient + SocketQueue) lands in C5.

Wire protocol (every frame, both directions):
  [4 bytes: little-endian uint32 length]
  [N bytes: msgpack(envelope_dict)]

Connect lifecycle:
  1. Worker connects to /tmp/titan_bus_<titan_id>.sock
  2. Broker sends CHALLENGE_SIZE random bytes
  3. Worker sends HMAC-SHA256(authkey, challenge)
  4. Broker verifies; on mismatch closes connection
  5. Worker sends one or more BUS_SUBSCRIBE frames listing module name + topics
  6. Bidirectional message loop: worker publishes / broker dispatches / both
     side periodically exchange BUS_PING / BUS_PONG (5s)

Threading model (~2N+2 for N workers):
  - 1 accept thread (rate-limited) — accepts new conns, spawns recv thread
  - 1 recv thread per subscriber — blocking recv_frame on its socket
  - 1 send thread per subscriber — drains its ring, batches >=5 msgs, sendall
  - 1 ping thread — every 5s, BUS_PING all subscribers; close stale (3 missed)

Phase C: this whole module ports to async Rust (tokio + rmp_serde + RustCrypto
hmac); same wire format; tests/test_frame_parity.py locks the contract.
"""
from __future__ import annotations

import logging
import os
import random
import secrets
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Callable, Optional

import msgpack

from titan_hcl.bus_specs import coalesce_key, get_spec
from titan_hcl.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    compute_hmac,
    constant_time_eq,
    recv_exact,
    recv_frame,
    send_frame,
)
from titan_hcl import bus

logger = logging.getLogger(__name__)

# ── Tunables (keep modest; broker is a hot path) ───────────────────────────

DEFAULT_RING_CAPACITY = 1024
DEFAULT_P0_RESERVE = 64
PING_INTERVAL_S = 5.0
PING_TIMEOUT_S = 15.0           # 3 missed pings → escalate to PID check (NOT instant purge anymore)
ACCEPT_RATE_LIMIT_PER_S = 50

# Microkernel v2 Phase B.2 §D9 (2026-05-02) — smart-liveness tunables.
# See `BUG-BUS-PING-PONG-TIGHT-TIMEOUT-VS-HEAVY-WORKER-INIT-20260502`
# for full context. Replaces the tight 15s purge that killed heavy-init
# workers (rl=19.7s, cgn=31s, memory=52s, backup=184s) during their boot
# phase.
#
# Algorithm (in _ping_loop):
#   Tier 1: silent_for ≤ PING_TIMEOUT_S          → alive, send ping
#   Tier 2: PID dead (os.kill(pid, 0) raised)    → purge instant + publish
#                                                  BUS_PEER_DIED so Guardian
#                                                  restarts (faster than
#                                                  Guardian's 1Hz polling)
#   Tier 3: anon AND silent > ANON_GRACE_S       → purge "anon_subscribe_timeout"
#                                                  (60s ceiling on a code-path
#                                                  duration; never user work)
#   Tier 4: named, silent, PID alive             → KEEP (busy worker; trust
#                                                  Guardian's CPU-grew check
#                                                  to detect deadlock)
# ── SPEC §8.0.bis boot-window buffer constants (D-SPEC pending) ─────────
# Mirror of Rust `boot_buffer::BOOT_BUFFER_MAX_FRAMES_PER_DST` +
# `BOOT_BUFFER_TTL_S` + `BOOT_BUFFERED_TYPES`. Cross-language parity:
# Python broker (T1+T2 Phase A+B) and Rust broker (T3 Phase C) MUST
# produce identical buffer behavior. Per Phase A of
# rFP_phase_c_bus_delivery_continuity_and_hot_reload.
BOOT_BUFFER_MAX_FRAMES_PER_DST = 32
BOOT_BUFFER_TTL_S = 60.0  # aligns with SUPERVISION_BOOT_TIMEOUT_S
# Targeted message types eligible for boot-window buffering. Restricted
# to lifecycle + supervision frames where late-delivery is preferable
# to drop. Application RPCs, broadcasts, swap-protocol messages are
# NOT buffered. Identical to Rust `boot_buffer::BOOT_BUFFERED_TYPES`.
BOOT_BUFFERED_TYPES = frozenset({
    # Module lifecycle (§8.1)
    "MODULE_READY",
    "MODULE_HEARTBEAT",
    "MODULE_SHUTDOWN",
    "MODULE_CRASHED",
    # Supervision lifecycle (§8.1)
    "SUPERVISION_CHILD_DOWN",
    "SUPERVISION_CHILD_RESTARTED",
    # Service-ready broadcasts when targeted
    "AGENCY_READY",
    "NS_READY",
    "MEMORY_READY",
    # Phase B reload ACK (reserved per §8.3)
    "MODULE_RELOAD_ACK",
})

ANON_SUBSCRIBE_GRACE_S = 120.0  # tier-3 ceiling — bound on _send_subscribe_frame latency.
                                # Bumped 60s→120s 2026-05-02 after Stage 1 fixed the
                                # fork-at-locked-mp.Queue bug — `llm` was the lone
                                # remaining stuck worker (anon-purged at 60s); 120s
                                # absorbs boot-time GIL variance for any heavy worker
                                # whose connection_loop thread contended for scheduling.
                                # Still bounded (kills truly-stuck connections), still
                                # well-tested at unit level, no user-controlled work.


# ── Msgpack default-encoder for non-native payload types ──────────────────
#
# BUG-BUS-IPC-SPIRIT-MALFORMED-FRAME-20260428 root-cause hypothesis
# (audit 2026-04-28 PM late): worker reply payloads — especially from
# spirit_worker's _handle_query (build_trinity_snapshot, get_stats() on
# intuition/sphere_clock/resonance/etc.) — frequently embed numpy
# scalars (np.float32, np.int64), numpy arrays, torch tensors, sets,
# frozensets, datetime, pathlib.Path, or dataclass instances. msgpack's
# default packb(...) with no `default=` callback raises TypeError on
# any of those, but our _raw_send catches only (ConnectionError,
# OSError) — TypeError propagates out and either crashes the publisher
# thread silently OR gets caught by an outer try/except that re-tries
# in a way that produces a torn / partially-serialized frame on the
# wire (msgpack.unpackb on the broker side then raises ValueError →
# "[bus_socket] malformed frame from spirit; closing").
#
# This `default=` callback converts the common offenders to msgpack-
# native types BEFORE serialization, so the wire bytes are always
# valid msgpack. Decoded by any msgpack reader (Python, Rust, Go) with
# no custom hook needed → forward-compat with Phase C Rust kernel
# parity vectors (the wire format remains pure msgpack; only the
# encoder pre-converts).
#
# Everything else (custom classes without msgpack-native fields) falls
# through to repr() so the frame still serializes cleanly + the
# offending value is debuggable in the broker hex-dump if anything
# else slips through.

def _pack_default(obj):
    """msgpack ``default=`` callback — coerces non-native types to native.

    Order is fastest-checks-first since this fires once per
    non-native value during publish (a hot path).
    """
    # numpy scalars + arrays — most common offender on Titan reply payloads
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:  # noqa: BLE001 — numpy not installed / weird build
        pass
    # set / frozenset → list (msgpack has no set type)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    # bytes/bytearray/memoryview already msgpack-native — passthrough
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    # torch tensor — convert via tolist() if torch is available
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:  # noqa: BLE001 — torch not installed / weird build
        pass
    # dataclass — convert to dict (recursively encodable)
    try:
        import dataclasses as _dc
        if _dc.is_dataclass(obj) and not isinstance(obj, type):
            return _dc.asdict(obj)
    except Exception:  # noqa: BLE001
        pass
    # pathlib.Path → str
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:  # noqa: BLE001
        pass
    # datetime → ISO-8601 str
    try:
        import datetime as _dt
        if isinstance(obj, (_dt.date, _dt.datetime, _dt.time)):
            return obj.isoformat()
    except Exception:  # noqa: BLE001
        pass
    # Last-resort fallback: repr() the object so the frame still serializes.
    # Loses round-trip fidelity but keeps the wire format valid + makes
    # the offending value visible in receiver-side logs (debugger can
    # match on the repr string to find the producer).
    return repr(obj)


def _coerce_keys_to_str(obj):
    """Recursively coerce all dict keys to strings.

    msgpack with `strict_map_key=True` (the broker-side default) rejects
    int/tuple/list/frozenset keys at unpack time even though packb accepts
    them. Per-publisher sanitization is brittle — every worker that
    constructs a dict with a non-str key (subsystem.get_stats() returning
    {(a,b): v}, np.int64 keys, etc.) silently produces malformed frames.

    Sanitizing at the broker boundary closes the whole class of bugs at
    once: every frame that crosses the Unix socket has guaranteed-str
    keys. Mirrors the convention used by spirit_state._sanitize_dict_keys
    (tuple → "a:b" / "a:b:c", others → str(k)).

    Closes BUG-BUS-IPC-SPIRIT-MALFORMED-FRAME-20260428 (third+ occurrence)
    at the architectural level instead of per-publisher.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                out[k] = _coerce_keys_to_str(v)
            elif isinstance(k, tuple):
                if len(k) == 2:
                    out[f"{k[0]}:{k[1]}"] = _coerce_keys_to_str(v)
                else:
                    out[":".join(str(p) for p in k)] = _coerce_keys_to_str(v)
            else:
                out[str(k)] = _coerce_keys_to_str(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_coerce_keys_to_str(x) for x in obj]
    return obj


class BusContractViolation(Exception):
    """Raised when a bus message fails its declared contract at SEND time.

    Per `rFP_bus_payload_contracts.md` (2026-05-01): SEND-time validation
    fail-loud — log ERROR + don't pack. Bug surfaces immediately at the
    producer instead of producing a malformed frame somewhere down the line.
    """


def _packb_safe(msg) -> bytes:
    """msgpack.packb wrapper — broker-boundary safety pass.

    Layered defenses:
      1. **Bus contract validation** (rFP §4 Chunk 4) — when msg["type"] has
         a contract in `bus_contracts.REGISTRY`, validate payload against
         the declared Pydantic schema + check size against max_payload_bytes.
         Fail-loud on violation — log ERROR + raise BusContractViolation.
         Producer caller MUST handle (see _raw_send / broker fanout — they
         catch + drop the frame so bus stays alive).
      2. `_coerce_keys_to_str` recursively rewrites every dict key as str
         so strict_map_key=True consumers always unpack cleanly.
      3. `default=_pack_default` handles non-standard VALUES (numpy,
         torch, dataclass, Path, datetime, repr fallback).

    All bus_socket producer sites must call this (workers via
    BusSocketClient, broker fanout, drift bridge). Per-publisher
    sanitization remains as belt-and-suspenders defense-in-depth.
    """
    # Layer 1 — contract validation (best-effort import to avoid hard
    # dependency cycle in environments where api package may not be loaded
    # — e.g., minimal kernel boot before plugin attaches).
    msg_type = msg.get("type", "") if isinstance(msg, dict) else ""
    if msg_type:
        try:
            from titan_hcl.api.bus_contracts import get_contract
            contract = get_contract(msg_type)
        except Exception:  # noqa: BLE001
            contract = None
        if contract is not None:
            payload = msg.get("payload", {}) if isinstance(msg, dict) else {}
            try:
                contract.schema.model_validate(payload)
            except Exception as schema_err:  # noqa: BLE001 — pydantic ValidationError or anything else
                logger.error(
                    "[BusContract] %s payload schema violation — DROPPING frame "
                    "(producer=%s, error=%s)",
                    msg_type, contract.producer_module, schema_err)
                raise BusContractViolation(
                    f"{msg_type} payload schema violation: {schema_err}"
                ) from schema_err

    # Layer 2+3 — pack with key coercion + value defaults
    packed = msgpack.packb(_coerce_keys_to_str(msg), default=_pack_default)

    # Layer 1 (size check) — runs after pack so we measure actual wire size.
    # Done after pack rather than estimated before so we catch packed-form
    # bloat (e.g., float-list expansion). Cheap because we already have bytes.
    if msg_type:
        try:
            from titan_hcl.api.bus_contracts import get_contract
            contract = get_contract(msg_type)
        except Exception:  # noqa: BLE001
            contract = None
        if contract is not None and len(packed) > contract.max_payload_bytes:
            logger.error(
                "[BusContract] %s payload exceeds max_payload_bytes — "
                "DROPPING frame (size=%d, limit=%d, producer=%s). "
                "Bulk data belongs in SHM or RPC, not the bus.",
                msg_type, len(packed), contract.max_payload_bytes,
                contract.producer_module)
            raise BusContractViolation(
                f"{msg_type} payload {len(packed)}B exceeds limit "
                f"{contract.max_payload_bytes}B"
            )
    return packed
SEND_BATCH_THRESHOLD = 5        # batch this many or more msgs into one frame
SEND_FLUSH_TIMEOUT_S = 0.05     # wait at most this long for batching to fill
SLOW_CONSUMER_WARN_INTERVAL_S = 60.0
SLOW_CONSUMER_DROP_RATE_THRESHOLD = 0.05  # 5%


# ── Broker socket path resolution ──────────────────────────────────────────


def bus_sock_path(titan_id: str) -> Path:
    """Per-titan bus broker socket path. Mirrors kernel_rpc per-titan convention.

    Critical for T2/T3 shared VPS — without per-titan suffix, both Titans'
    kernels would attempt to bind the same socket and conflict.
    """
    return Path(f"/tmp/titan_bus_{titan_id}.sock")


# ── BoundedRing — per-subscriber ring with P0 reserve + coalesce ───────────


class BoundedRing:
    """Bounded queue with P0 reserve region.

    The non-P0 deque has maxlen=capacity-p0_reserve; appending past full
    drops the oldest (deque semantics). The p0_reserve deque has maxlen=
    p0_reserve; same semantics — but priority dispatch logic in the broker
    chooses which deque to append to.

    Coalesce works by mutating the message dict in place (the dict reference
    in the ring stays the same; its content is updated). This means
    broker.publish on a coalesce-key collision does NOT consume a new ring
    slot — the existing slot's message just gets refreshed content.

    Thread-safety: caller holds a lock while mutating. Designed to be small
    and easy to port to crossbeam::ArrayQueue in Phase C.
    """

    __slots__ = ("_main", "_p0", "_capacity", "_p0_reserve")

    def __init__(self, capacity: int = DEFAULT_RING_CAPACITY,
                 p0_reserve: int = DEFAULT_P0_RESERVE) -> None:
        if p0_reserve >= capacity:
            raise ValueError(f"p0_reserve {p0_reserve} >= capacity {capacity}")
        self._capacity = capacity
        self._p0_reserve = p0_reserve
        self._main: deque = deque(maxlen=capacity - p0_reserve)
        self._p0: deque = deque(maxlen=p0_reserve)

    def append_main(self, msg: dict) -> bool:
        """Append a non-P0 message. Returns True if appended without eviction,
        False if the deque was full and the oldest was evicted (drop event)."""
        evicted = len(self._main) == self._main.maxlen
        self._main.append(msg)
        return not evicted

    def append_p0(self, msg: dict) -> bool:
        """Append a P0 message into the reserved region. Returns False if even
        the P0 reserve was already full (extreme overflow — should not happen
        in normal operation; broker logs it as critical)."""
        evicted = len(self._p0) == self._p0_reserve
        self._p0.append(msg)
        return not evicted

    def main_is_full(self) -> bool:
        return len(self._main) == self._main.maxlen

    def is_empty(self) -> bool:
        return len(self._main) == 0 and len(self._p0) == 0

    def __len__(self) -> int:
        return len(self._main) + len(self._p0)

    def pop_for_send(self, max_msgs: int) -> list[dict]:
        """Pop up to max_msgs messages, P0 first then main. FIFO within each."""
        out: list[dict] = []
        while self._p0 and len(out) < max_msgs:
            out.append(self._p0.popleft())
        while self._main and len(out) < max_msgs:
            out.append(self._main.popleft())
        return out


# ── BrokerSubscriber — per-connection state ────────────────────────────────


@dataclass
class BrokerSubscriber:
    """All state the broker holds about one connected worker.

    coalesce_index: maps coalesce-key tuple → message dict that lives in the
    ring. On a coalesce hit, broker mutates that dict in place; ring slot is
    NOT consumed twice. When the send thread pops the message, it removes
    the entry from coalesce_index by id-check (caller's responsibility).
    """

    name: str                        # module/worker name (used as bus dst)
    conn: socket.socket
    addr: str                        # description for logging ("anon-1234")
    ring: BoundedRing = field(default_factory=BoundedRing)
    coalesce_index: dict[tuple, dict] = field(default_factory=dict)
    subscribed_topics: set[str] = field(default_factory=set)  # bus.publish dst targets
    # D-SPEC-42 (SPEC v1.4.0, 2026-05-12) — subscriber intent flag.
    # When True, this subscriber receives ONLY targeted dst=name (or dst=alias)
    # messages — never dst="all" broadcasts. Broker `publish()` silently skips
    # reply_only subscribers from the broadcast fan-out (no log, no drop
    # counter — they're not in the broadcast contract by design). Mirrors the
    # in-process `bus.py:_reply_only` set, which has honored this semantic
    # since 2026-04-30. Set by BUS_SUBSCRIBE handler from the payload's
    # `reply_only` field. Connection-level (last value wins on multi-name
    # subscribe per v1.3.0).
    reply_only: bool = False
    last_pong_ts: float = field(default_factory=time.time)
    last_recv_ts: float = field(default_factory=time.time)  # any frame, not just PONG
    drop_count_60s: int = 0
    recv_count_60s: int = 0
    last_warning_ts: float = 0.0
    last_window_reset_ts: float = field(default_factory=time.time)
    has_data_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    recv_thread: Optional[threading.Thread] = None
    send_thread: Optional[threading.Thread] = None
    # Microkernel v2 Phase B.2 §D9 (2026-05-02) — peer process identity from
    # SO_PEERCRED at accept time. Enables tier-2 PID liveness check in
    # _ping_loop without coupling to Guardian (broker uses OS as oracle).
    # None on platforms without SO_PEERCRED (degrades to time-based heuristic).
    peer_pid: Optional[int] = None
    accept_ts: float = field(default_factory=time.time)  # for anon-grace bound


# ── BusSocketClient + SocketQueue (C5) ──────────────────────────────────────

# Reconnect backoff
RECONNECT_JITTER_INITIAL_MIN_S = 0.05
RECONNECT_JITTER_INITIAL_MAX_S = 0.15
RECONNECT_BACKOFF_BASE_S = 0.1
RECONNECT_BACKOFF_MAX_S = 2.0

# SPEC §8.0.ter (rFP_bus_socket_outbound_writer_thread.md, v1.6.0, 2026-05-14).
# These will move to SPEC_titan_architecture_constants.toml under
# [bus.client_outbound] in Chunk 5 of the rFP.
#
# OUTBOUND_BUFFER_HIGH_WATER — frame count threshold for the rate-limited WARN.
# Buffer is unbounded (§8.0 P0 never-drop), so this is a backpressure SIGNAL,
# not a cap. Healthy steady-state depth is <10 frames; >1000 means a slow
# downstream consumer that operators need to see via warning_monitor.
OUTBOUND_BUFFER_HIGH_WATER = 1000
# Rate-limit window for the high-water WARN — one log line per client per
# minute. Without this, sustained backpressure would flood the log.
OUTBOUND_BUFFER_HIGH_WATER_WARN_INTERVAL_S = 60.0
# Writer thread idle-poll fallback. Writer wakes on _outbound_event OR
# every N seconds. Idle polling lets the writer recover if the event was
# missed (defensive; the event is set on every enqueue + on (re)connect).
OUTBOUND_WRITER_IDLE_POLL_S = 1.0


class BusSocketClient:
    """Worker-side persistent connection to the broker.

    Holds a single connection. Owns one connection thread that runs the
    full state machine: connect → handshake → subscribe → recv loop →
    on EOF/error reconnect with jittered exponential backoff.

    The worker's main loop interacts with the inbound queue via SocketQueue
    (returned from `inbound_queue()`). Outbound publishes go via `publish()`.

    Reconnect is invisible to the worker: after the kernel swap, ~50-150ms
    later the recv loop is back and messages flow again. The worker's
    SocketQueue.get() may see a brief gap but otherwise unchanged.
    """

    def __init__(self, titan_id: str, authkey: bytes, name: str,
                 sock_path: Optional[Path] = None,
                 topics: Optional[list[str]] = None,
                 reply_only: bool = False,
                 inbound_capacity: int = DEFAULT_RING_CAPACITY,
                 on_subscribe_ack: Optional[Callable[[], None]] = None) -> None:
        if not titan_id:
            raise ValueError("titan_id required")
        if len(authkey) == 0:
            raise ValueError("authkey required")
        if not name:
            raise ValueError("name required (subscriber identity)")
        self.titan_id = titan_id
        self._authkey = authkey
        self.name = name
        self.sock_path = Path(sock_path) if sock_path else bus_sock_path(titan_id)
        self._initial_topics = list(topics or [])
        # SPEC §8.0.bis worker contract C2: callback invoked after every
        # successful BUS_SUBSCRIBE handshake (initial boot + every
        # reconnect). Used by `setup_worker_bus()` to re-emit MODULE_READY
        # so Guardian's state machine sees the worker as running even
        # if the initial MODULE_READY was lost in the bootstrap race
        # window. Idempotent: Guardian treats re-emit on state=running
        # as no-op. None = no callback (test fixtures, isolated clients).
        # Per Phase A of rFP_phase_c_bus_delivery_continuity_and_hot_reload.
        self._on_subscribe_ack: Optional[Callable[[], None]] = on_subscribe_ack
        # D-SPEC-42 (SPEC v1.4.0, 2026-05-12) — declare subscriber intent
        # to the broker. reply_only=True ↔ this connection consumes ONLY
        # targeted dst=<name>/dst=<alias> messages (RPC replies + control),
        # never dst="all" broadcasts. Broker silently skips reply_only
        # subscribers from broadcast fan-out. Persisted locally + re-fired
        # on every reconnect (same pattern as _initial_topics).
        self._reply_only = bool(reply_only)
        self._inbound: deque = deque()
        # SPEC §8.0.quat parity (D-SPEC-131-Py, RFP_Phase_C_python_fix) — single
        # wake primitive on a predicate (data arrived OR stop requested), so
        # callers re-check authoritative state on every wakeup instead of
        # interpreting a dual-purpose Event. The Condition uses _inbound_lock
        # as its underlying lock — zero new lock contention.
        self._inbound_lock = threading.Lock()
        self._wake_cond = threading.Condition(self._inbound_lock)
        self._inbound_capacity = inbound_capacity
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._conn_thread: Optional[threading.Thread] = None
        # Subscription state owned locally — re-published on every reconnect
        self._topics: set[str] = set(self._initial_topics)
        self._topics_lock = threading.Lock()
        # Multi-name BUS_SUBSCRIBE aliases (SPEC §8.2 v1.3.0). Each entry
        # is a non-primary subscriber name this connection should ALSO be
        # registered under on the broker. On every (re)connect, the
        # connection loop fires the primary BUS_SUBSCRIBE first (sets
        # broker.sub.name) then fires one BUS_SUBSCRIBE per alias (each
        # additive on broker.sub.aliases). Without re-firing on reconnect,
        # the post-reconnect broker subscriber loses all alias entries
        # while the stale-but-still-tracked zombie subscriber retains
        # them — and the broker fanout delivers RESPONSE messages to the
        # zombie's send queue, where they're lost. Fix: track aliases
        # locally + re-fire on every reconnect alongside topics.
        self._aliases: set[str] = set()
        self._aliases_lock = threading.Lock()
        # Set right after the primary BUS_SUBSCRIBE (with topics) is sent on
        # the current connection; cleared on disconnect. subscribe_alias()
        # gates its immediate send on this so an alias frame can never be the
        # FIRST BUS_SUBSCRIBE the broker sees on a connection — otherwise the
        # broker promotes the alias name to primary with empty topics and
        # WARN-spams every dst="all" broadcast until the primary subscribe
        # lands (BUG-BROKER-ORPHAN-SUB-WARN-FROM-ALIAS-REGISTRATION-20260526).
        self._primary_subscribed = threading.Event()
        self._reconnect_count = 0
        # SPEC §8.0.ter outbound buffer — canonical write path
        # (rFP_bus_socket_outbound_writer_thread.md, 2026-05-14, v1.6.0).
        #
        # Holds pre-packed msgpack bytes. `_raw_send` validates + packs
        # synchronously on the publisher's thread (so bus contract
        # violations fail-fast at the producer per rFP §4 Chunk 4), then
        # appends to this buffer and signals the writer thread. The
        # writer thread does the actual `send_frame()` socket I/O —
        # publisher NEVER blocks on socket per §8.0.ter guarantee.
        #
        # Unbounded (no `maxlen=`) because SPEC §8.0 P0 forbids dropping.
        # Backpressure surfaces as a rate-limited high-water WARN at
        # OUTBOUND_BUFFER_HIGH_WATER frames (default 1000), monitored
        # via warning_monitor per §11.B. Realistic peak buffer depth in
        # healthy steady-state is <10 frames — high-water indicates
        # something is wrong upstream (broker stall, slow consumer).
        #
        # History: originally added 2026-04-30 (Phase B.2 IPC §D8) as a
        # 256-frame deque just for the disconnect-reconnect race window,
        # held dicts, was drained synchronously from _connection_loop on
        # reconnect via _flush_outbound_buffer. Generalized 2026-05-14
        # to be the canonical write path for ALL publishes (closes the
        # publish-from-asyncio MainThread deadlock class observed on
        # T3 2026-05-14 via py-spy after D-SPEC-42 connection collapse
        # commit 78fbf356 tipped the latent backpressure issue).
        self._outbound_buffer: deque = deque()
        self._outbound_lock = threading.Lock()
        # Event signaled by `_raw_send` after each enqueue + on
        # (re)connect from `_connection_loop`. Writer thread waits on
        # this with an idle poll fallback (OUTBOUND_WRITER_IDLE_POLL_S
        # = 1.0s) so reconnects + new frames both wake it promptly.
        self._outbound_event = threading.Event()
        # Dedicated writer thread (started in `start()`, joined in `stop()`).
        # Runs `_writer_loop()` which is the only code path performing
        # `send_frame()` socket I/O. All callers of `publish()` /
        # `_raw_send()` return without touching the socket.
        self._writer_thread: Optional[threading.Thread] = None
        # Rate-limit timestamp for the high-water WARN — one WARN per
        # client per OUTBOUND_BUFFER_HIGH_WATER_WARN_INTERVAL_S (60s).
        # Avoids log flood if backpressure persists for hours.
        self._high_water_last_warn_ts: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._conn_thread is not None and self._conn_thread.is_alive():
            return
        self._stop_event.clear()
        # SPEC §8.0.ter: writer thread MUST start before connection thread
        # so any publish() that lands during the connect window is drained
        # promptly once the socket goes live (rather than waiting for the
        # connection thread to call _flush_outbound_buffer after handshake).
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True,
            name=f"bus-writer-{self.name}")
        self._writer_thread.start()
        self._conn_thread = threading.Thread(
            target=self._connection_loop, daemon=True,
            name=f"bus-client-{self.name}")
        self._conn_thread.start()

    def flush(self, timeout: float = 5.0) -> bool:
        """SPEC §8.0.ter — block caller until outbound buffer is drained.

        For the rare callers that need send-completion semantics (graceful
        shutdown sequences, RPC reply emits where the reply MUST be on the
        wire before the handler returns). Most callers should NOT use this
        — the whole point of §8.0.ter is that publish() is fire-and-forget
        from the caller's perspective.

        Approach: signal the writer thread (idempotent) then wait for the
        buffer to empty. The writer thread does the actual drain via
        send_frame() on its own thread, so this method blocks the caller
        but never the writer.

        Returns True if buffer drained within `timeout`. Returns False on
        timeout — caller can either retry, log, or proceed (the writer
        thread keeps draining in background regardless).

        Per Preamble G19, the default timeout (5.0s) matches the work-RPC
        budget. Callers can request smaller timeouts (e.g. 1-2s) for
        latency-sensitive paths like RPC reply emits.
        """
        deadline = time.monotonic() + timeout
        # Initial wake — covers the case where the writer is currently
        # sleeping on the idle poll and our enqueue completed too recently
        # for it to have noticed.
        self._outbound_event.set()
        while time.monotonic() < deadline:
            with self._outbound_lock:
                if not self._outbound_buffer:
                    return True
            # Short sleep — gives the writer thread CPU to drain.
            # 1ms granularity is well under the 5s default budget and
            # well above the writer's wake latency (microseconds).
            time.sleep(0.001)
        # One last check after the timeout fires, in case the writer
        # drained on the very last iteration.
        with self._outbound_lock:
            return not self._outbound_buffer

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        # Wake the writer thread so it observes _stop_event and exits.
        # Idempotent — set() on an already-set event is a no-op.
        self._outbound_event.set()
        with self._sock_lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
        # Wake any pending get() callers — they re-check the predicate and
        # observe _stop_event.is_set() = True, raising QueueEmpty.
        with self._inbound_lock:
            self._wake_cond.notify_all()
        if self._conn_thread is not None and self._conn_thread.is_alive():
            self._conn_thread.join(timeout=timeout)
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)

    def wait_until_connected(self, timeout: float = 5.0) -> bool:
        return self._connected_event.wait(timeout=timeout)

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    # ── Subscription ──────────────────────────────────────────────────────

    def subscribe(self, topics: list[str]) -> None:
        with self._topics_lock:
            self._topics.update(topics)
        self._send_subscribe_frame(list(topics))

    def subscribe_alias(self, alias_name: str) -> None:
        """Register an ADDITIONAL subscriber name on this connection (SPEC
        §8.2 v1.3.0 multi-name BUS_SUBSCRIBE). The broker treats this as
        adding `alias_name` to BrokerSubscriber.aliases, so fanout for
        `dst=alias_name` will deliver to this connection alongside any
        delivery targeting the connection's primary `name`.

        Persisted in self._aliases and re-fired on every reconnect so
        the broker's post-reconnect view of this connection retains all
        alias subscriptions (without this, the connection_loop's
        _send_subscribe_frame only re-publishes the primary name +
        topics, dropping aliases on the floor).

        Idempotent — re-adding the same alias does nothing extra.
        """
        if not alias_name or alias_name == self.name:
            return
        with self._aliases_lock:
            self._aliases.add(alias_name)
        # Only send the alias frame NOW if the primary BUS_SUBSCRIBE (with
        # topics) has already gone out on the current connection. At boot the
        # connection is still establishing (client.start() connects async) and
        # callers invoke this synchronously right after — sending now would
        # buffer the alias frame into the FIFO _outbound_buffer AHEAD of the
        # connect sequence's primary subscribe (connection_loop), so the broker
        # would promote this alias to primary with empty topics + reply_only=
        # false and WARN-drop every broadcast until the primary lands. Instead
        # we register in self._aliases above; the connect sequence fires the
        # primary subscribe FIRST then one alias frame per self._aliases entry,
        # guaranteeing alias-after-primary ordering. Runtime alias adds (post
        # primary subscribe) still send immediately.
        # (BUG-BROKER-ORPHAN-SUB-WARN-FROM-ALIAS-REGISTRATION-20260526.)
        if self._primary_subscribed.is_set():
            self._send_alias_subscribe_frame(alias_name)

    def _send_alias_subscribe_frame(self, alias_name: str) -> None:
        """Send a BUS_SUBSCRIBE frame with payload.name=<alias_name>. Empty
        topics; alias frames are about registering an additional dst-route
        only, not extending the broadcast filter. SPEC §8.2 v1.4.0
        (D-SPEC-42): same reply_only as the primary connection — intent is
        a connection-level property, so every BUS_SUBSCRIBE frame this
        connection sends MUST carry the same value (broker stores last
        value sent, but consistent values keep the view stable on
        reconnect)."""
        msg = {
            "type": "BUS_SUBSCRIBE", "src": self.name, "dst": "broker",
            "payload": {
                "name": alias_name,
                "topics": [],
                "reply_only": self._reply_only,
            },
        }
        self._raw_send(msg)

    def unsubscribe(self, topics: list[str]) -> None:
        with self._topics_lock:
            self._topics.difference_update(topics)
        self._send_unsubscribe_frame(list(topics))

    def _send_subscribe_frame(self, topics: list[str]) -> None:
        # SPEC §8.2 v1.4.0 (D-SPEC-42): payload carries the connection's
        # reply_only intent. Connection-level property — same value on every
        # (re)subscribe + on every alias subscribe so the broker's view
        # converges deterministically on a single intent per connection.
        msg = {
            "type": "BUS_SUBSCRIBE", "src": self.name, "dst": "broker",
            "payload": {
                "name": self.name,
                "topics": topics,
                "reply_only": self._reply_only,
            },
        }
        self._raw_send(msg)

    def _send_unsubscribe_frame(self, topics: list[str]) -> None:
        msg = {
            "type": "BUS_UNSUBSCRIBE", "src": self.name, "dst": "broker",
            "payload": {"topics": topics},
        }
        self._raw_send(msg)

    # ── Outbound publish ──────────────────────────────────────────────────

    def publish(self, msg: dict) -> bool:
        """Send a message to the broker. Returns True on success, False if
        the connection is currently broken (worker should accept that publish
        attempts during a kernel swap may briefly fail)."""
        return self._raw_send(msg)

    def _raw_send(self, msg: dict) -> bool:
        """SPEC §8.0.ter publish path — non-blocking by construction.

        Validates + packs the frame on the caller's thread (so bus contract
        violations per rFP_bus_payload_contracts.md §4 surface fail-loud at
        the producer), then appends pre-packed bytes to `_outbound_buffer`
        and signals the writer thread. NEVER performs socket I/O on the
        caller's thread — the writer thread is the sole `send_frame()`
        caller post-§8.0.ter.

        Performance budget per §8.0.ter: ≤10 µs p99 warm cache. Hot path
        is msgpack pack + size check (via `_packb_safe`) + lock-append +
        event-set. No sleeping, no I/O, no waiting.

        Returns True after enqueue. Returns False ONLY if the frame fails
        validation (BusContractViolation, unpicklable payload) — in which
        case the frame is dropped and a log line is emitted; caller can
        treat False as "this specific frame won't be delivered, but the
        client is still healthy and other publishes will work."
        """
        # ── Pack + validate on caller's thread (fail-fast at producer) ──
        # `_packb_safe` enforces bus contracts (rFP_bus_payload_contracts §4):
        # size ≤ MAX_PAYLOAD_BYTES, schema validation for typed contracts,
        # type-coercion for non-standard values. Raising here is the SPEC-
        # correct accountability point — the producer caller learns about
        # the violation. Deferring this to the writer thread would lose the
        # producer-context association.
        try:
            packed = _packb_safe(msg)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[bus_client] msgpack pack failed; dropping frame "
                "(type=%s, msg keys=%s)",
                msg.get("type") if isinstance(msg, dict) else "?",
                list(msg.keys())[:10] if isinstance(msg, dict) else "?")
            return False
        # ── Enqueue + signal writer (no socket I/O on this thread) ──
        # Acquire the lock for the minimal critical section: append +
        # depth check. Release before signaling the event so the writer
        # can re-acquire the lock immediately on wake.
        with self._outbound_lock:
            self._outbound_buffer.append(packed)
            depth = len(self._outbound_buffer)
        # High-water WARN (rate-limited per-client). Emitted from the
        # publisher's thread so the log line carries the producing module's
        # context. Operator path: warning_monitor consumes this and
        # surfaces via /v4/warning-monitor (Chunk 4 of the rFP).
        if depth >= OUTBOUND_BUFFER_HIGH_WATER:
            now = time.monotonic()
            if (now - self._high_water_last_warn_ts
                    > OUTBOUND_BUFFER_HIGH_WATER_WARN_INTERVAL_S):
                self._high_water_last_warn_ts = now
                logger.warning(
                    "[bus_client.%s] outbound buffer high water: %d frames "
                    "queued (threshold=%d). Writer thread may be blocked "
                    "by slow broker drain.",
                    self.name, depth, OUTBOUND_BUFFER_HIGH_WATER)
        self._outbound_event.set()
        return True

    def _flush_outbound_buffer(self) -> None:
        """SPEC §8.0.ter — drain outbound buffer (used by _connection_loop
        immediately after a (re)connect to flush queued frames ASAP, and
        by the writer thread on every wake).

        Shared drain implementation between two contexts:
          - `_connection_loop` calls this synchronously after `_connected_event.set()`
            so frames buffered during the connect window go out immediately.
          - `_writer_loop` calls this on every wake (event or idle poll) so
            new frames from `_raw_send` go out without waiting for a reconnect.

        Either context can race the other safely: the outbound_lock guards
        the buffer; only one of the two threads will own the lock at a time.

        Buffer holds pre-packed msgpack bytes (`_raw_send` packs synchronously
        for fail-fast validation per rFP_bus_payload_contracts §4). This drain
        is therefore pure I/O — no pack, no validation, just `send_frame`.

        Drain pattern: peek-then-popleft-on-success. The frame stays in the
        buffer until its `send_frame()` call returns cleanly, then it's
        atomically popped from the head. This preserves two invariants:

          1. `len(buffer)` always reflects frames-not-yet-on-the-wire,
             so `flush()` correctly waits for actual wire transmission
             rather than "writer has the snapshot."
          2. If the writer crashes mid-drain, only the in-flight frame is
             lost; the rest stay safely buffered for the next drain pass.

        On send failure (ConnectionError / OSError): leave the failing frame
        at the head of the buffer and return. `_connection_loop` reconnect →
        writer wakes via _outbound_event → next drain picks up where we left
        off in FIFO order.
        """
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        flushed = 0
        while True:
            # Peek the head frame under lock. We don't popleft until the
            # send succeeds so `len(buffer)` accurately reflects un-sent
            # frames at all times (flush() depends on this invariant).
            with self._outbound_lock:
                if not self._outbound_buffer:
                    break
                packed = self._outbound_buffer[0]
            try:
                send_frame(sock, packed)
            except (ConnectionError, OSError) as e:
                # Connection broke mid-flush. Leave the failing frame at
                # the head of the buffer; `_connection_loop` reconnect →
                # writer thread retries from this exact frame in FIFO order.
                logger.debug(
                    "[bus_client] flush interrupted for '%s' "
                    "(buffer depth %d): %s — reconnect will retry from "
                    "the head frame",
                    self.name, len(self._outbound_buffer), e)
                return
            except Exception:  # noqa: BLE001
                # Defensive: send_frame's documented exceptions are
                # ConnectionError / OSError (handled above) + ValueError
                # for oversized frames (caught at pack time by _packb_safe).
                # Anything else here is a bug — log + drop the offending
                # frame so one bad frame doesn't wedge the writer.
                logger.exception(
                    "[bus_client] flush dropped frame for '%s' "
                    "(buffer depth %d)",
                    self.name, len(self._outbound_buffer))
                # Drop the frame regardless of send status — same semantics
                # as the success path below.
            # Atomic popleft on success (or drop-on-bad-frame). Defensive
            # check: if some other thread emptied the buffer or replaced
            # the head between the peek and now, popleft would surface a
            # bug; we silently no-op in that case to stay defensive.
            with self._outbound_lock:
                if self._outbound_buffer and self._outbound_buffer[0] is packed:
                    self._outbound_buffer.popleft()
            flushed += 1
        if flushed:
            # INFO at large batches (typical of post-reconnect flush);
            # DEBUG at small batches (typical of writer-thread steady state)
            # to avoid log noise from healthy operation.
            level = logger.info if flushed >= 5 else logger.debug
            level("[bus_client] worker '%s' drained %d frame(s)",
                  self.name, flushed)

    def _writer_loop(self) -> None:
        """SPEC §8.0.ter writer thread — sole `send_frame()` caller post-fix.

        Loop until `_stop_event` set:
          1. Wait on `_outbound_event` (or `OUTBOUND_WRITER_IDLE_POLL_S`
             timeout for defensive idle polling — covers signal misses).
          2. Clear the event BEFORE draining so any concurrent enqueue
             that happens during drain still triggers a follow-up wake.
          3. Call `_flush_outbound_buffer()` which is the shared FIFO drain.

        Exceptions inside the loop are caught + logged + the loop continues
        — a single bad frame or transient I/O error must not kill the
        thread (otherwise publishers silently queue forever).

        Idle CPU cost: zero (event-driven). Wake latency on enqueue:
        microseconds (single threading.Event signal + wake).
        """
        logger.info(
            "[bus_client.%s] writer thread started (SPEC §8.0.ter)",
            self.name)
        while not self._stop_event.is_set():
            # Wait for either a new enqueue (_outbound_event.set() from
            # _raw_send) or a reconnect (_outbound_event.set() from
            # _connection_loop after _sock is established). Idle poll
            # is a defensive fallback — if _outbound_event was somehow
            # missed, the 1s timeout still lets us drain any queued
            # frames the next cycle.
            self._outbound_event.wait(timeout=OUTBOUND_WRITER_IDLE_POLL_S)
            # Clear BEFORE drain so any enqueue that races with the
            # drain still leaves the event set for the next iteration.
            # (set() on an already-set event is a no-op, so producers
            # can safely signal without coordination.)
            self._outbound_event.clear()
            try:
                self._flush_outbound_buffer()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[bus_client.%s] writer loop iteration raised — "
                    "continuing", self.name)
        logger.info(
            "[bus_client.%s] writer thread stopped", self.name)

    # ── Inbound queue (SocketQueue API) ───────────────────────────────────

    def inbound_queue(self) -> "SocketQueue":
        """Return a SocketQueue wrapper for the worker's main loop."""
        return SocketQueue(self)

    # ── Connection state machine ──────────────────────────────────────────

    def _connection_loop(self) -> None:
        backoff_attempt = 0
        while not self._stop_event.is_set():
            sock: Optional[socket.socket] = None
            try:
                # Initial connect: small jitter to avoid thundering herd
                if backoff_attempt == 0:
                    delay = random.uniform(
                        RECONNECT_JITTER_INITIAL_MIN_S,
                        RECONNECT_JITTER_INITIAL_MAX_S,
                    )
                else:
                    base = RECONNECT_BACKOFF_BASE_S * (2 ** (backoff_attempt - 1))
                    capped = min(base, RECONNECT_BACKOFF_MAX_S)
                    # Jitter ±25%
                    delay = capped * random.uniform(0.75, 1.25)
                if self._stop_event.wait(delay):
                    return
                sock = self._open_and_handshake()
                with self._sock_lock:
                    self._sock = sock
                # Re-publish subscription list on every (re)connect
                with self._topics_lock:
                    topics = list(self._topics)
                self._send_subscribe_frame(topics)
                # Primary subscribe is now on the wire for this connection —
                # alias frames sent from here on (the loop below + any runtime
                # subscribe_alias) correctly land AFTER it on the broker.
                self._primary_subscribed.set()
                # SPEC §8.2 v1.3.0 — re-publish alias subscriptions too.
                # Without this, the post-reconnect broker subscriber loses
                # the multi-name alias entries while the zombie subscriber
                # from the prior connection retains them (until the broker
                # eventually purges the zombie); fanout for dst=<alias>
                # routes to the zombie's send queue → RESPONSE lost.
                with self._aliases_lock:
                    aliases = list(self._aliases)
                for alias_name in aliases:
                    try:
                        self._send_alias_subscribe_frame(alias_name)
                    except Exception:  # noqa: BLE001
                        # Send buffer fallback — _raw_send already buffers
                        # if the socket isn't ready. We don't want one bad
                        # alias to abort the rest of reconnect.
                        pass
                # SPEC §8.0.bis worker contract C2: re-emit MODULE_READY
                # after every successful (re)subscribe. Closes the
                # bootstrap-race class: even if the initial MODULE_READY
                # was lost in the Guardian-subscribe window, every
                # reconnect (including post-broker-swap, post-adoption,
                # network reconnect) re-establishes Guardian's view of
                # the worker as running. Idempotent: Guardian treats
                # re-emit on state=running as no-op.
                # Per Phase A of rFP_phase_c_bus_delivery_continuity_and_hot_reload.
                if self._on_subscribe_ack is not None:
                    try:
                        self._on_subscribe_ack()
                    except Exception:  # noqa: BLE001
                        # Hook must NEVER crash the connection loop —
                        # log and continue. Worker functionality must
                        # not depend on this re-emit succeeding.
                        logger.warning(
                            "[%s] on_subscribe_ack hook raised — ignoring "
                            "(connection loop unaffected)",
                            self.name, exc_info=True)
                self._connected_event.set()
                if backoff_attempt > 0:
                    self._reconnect_count += 1
                # Phase B.2 IPC §D8 + SPEC §8.0.ter outbound buffer flush.
                # Drain any messages publish() buffered while the socket
                # was establishing — preserves first-publish (MODULE_READY,
                # initial state, etc.) for light-init workers that race
                # past the connection thread. Synchronous immediate drain
                # so reconnects don't add a 0-1s wait for the writer
                # thread's idle poll. Signaling _outbound_event AFTER
                # the synchronous drain ensures the writer thread takes
                # over any further enqueues without contending for the
                # buffer during the post-reconnect critical window.
                self._flush_outbound_buffer()
                self._outbound_event.set()
                # Recv loop blocks here until EOF/error
                self._recv_loop(sock)
            except (ConnectionError, OSError, ValueError) as e:
                logger.debug("[bus_client] connection attempt failed: %s", e)
            finally:
                self._connected_event.clear()
                # Next connection must re-send the primary subscribe before any
                # alias frame — re-arm the gate so subscribe_alias defers again.
                self._primary_subscribed.clear()
                with self._sock_lock:
                    if self._sock is sock:
                        self._sock = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            backoff_attempt = min(backoff_attempt + 1, 8)

    def _open_and_handshake(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(self.sock_path))
        challenge = recv_exact(s, CHALLENGE_SIZE)
        response = compute_hmac(self._authkey, challenge)
        s.sendall(response)
        s.settimeout(None)  # blocking from here on
        return s

    def _recv_loop(self, sock: socket.socket) -> None:
        while not self._stop_event.is_set():
            try:
                payload = recv_frame(sock)
            except (ConnectionError, OSError, ValueError):
                return
            try:
                msg = msgpack.unpackb(payload, raw=False)
            except (ValueError, msgpack.UnpackException) as _unpack_err:
                # Symmetric diagnostic to broker-side recv loop above.
                _hex_head = payload[:64].hex() if payload else "<empty>"
                _hex_tail = payload[-16:].hex() if len(payload) > 64 else ""
                logger.warning(
                    "[bus_client] malformed frame from broker; closing "
                    "(unpack=%s, payload_len=%d, head_hex=%s%s)",
                    _unpack_err, len(payload), _hex_head,
                    f", tail_hex={_hex_tail}" if _hex_tail else "")
                return
            if not isinstance(msg, dict):
                logger.warning("[bus_client] non-dict msg from broker; closing "
                               "(got type=%s, repr=%.200r)",
                               type(msg).__name__, msg)
                return
            self._handle_inbound(msg)

    def _handle_inbound(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == bus.BUS_PING:
            # Reply with PONG
            self._raw_send({"type": "BUS_PONG", "src": self.name, "dst": "broker",
                            "payload": {}})
            return
        if mtype == "BUS_BATCH":
            # Unwrap and deliver each (recursive — each inner frame
            # goes through _handle_inbound's MODULE_SHUTDOWN pid-target
            # filter below).
            for inner in msg.get("msgs", []):
                self._handle_inbound(inner)
            return
        # SPEC §8.1 + §11.B.3.1 (D-SPEC-93, v1.32.0) — pid-targeted
        # MODULE_SHUTDOWN. During reload's dual-pid name-aliased
        # subscription window, Guardian sets payload.target_pid=old_pid
        # so the just-spawned NEW worker (subscribed under the same
        # dst=module_name) drops the frame and only OLD acts.
        # Default (target_pid absent or null) preserves legacy all-
        # subscribers-honor behavior.
        if mtype == bus.MODULE_SHUTDOWN:
            target_pid = (msg.get("payload") or {}).get("target_pid")
            if target_pid is not None and target_pid != os.getpid():
                return
        self._deliver_to_inbound(msg)

    def _deliver_to_inbound(self, msg: dict) -> None:
        with self._inbound_lock:
            if len(self._inbound) >= self._inbound_capacity:
                # Drop oldest — match broker's eviction semantics so worker
                # doesn't accumulate stale state when it can't keep up
                self._inbound.popleft()
            self._inbound.append(msg)
            self._wake_cond.notify_all()


class SocketQueue:
    """Drop-in stand-in for multiprocessing.Queue / queue.Queue used in worker
    code. Wraps a BusSocketClient's inbound deque + event so .get() blocks
    just like the previous mp.Queue.

    Worker code path that did:
        q = bus.subscribe("body_worker")
        msg = q.get(timeout=1.0)
    keeps working unchanged when bus is in socket mode — DivineBus.subscribe
    returns SocketQueue(bus_client) instead of mp.Queue.
    """

    __slots__ = ("_client",)

    def __init__(self, client: BusSocketClient) -> None:
        self._client = client

    def get(self, timeout: Optional[float] = None) -> dict:
        """Block until a message is available; raise queue.Empty on timeout.

        Uses Condition.wait_for with a multi-predicate (data arrived OR client
        stopping). Predicate is re-evaluated on every wakeup, so a stop signal
        cannot be mistaken for data and a spurious wake cannot return stale
        state. SPEC §8.0.quat parity (D-SPEC-131-Py)."""
        client = self._client
        deadline = time.time() + timeout if timeout is not None else None
        predicate = lambda: bool(client._inbound) or client._stop_event.is_set()
        with client._inbound_lock:
            if deadline is None:
                client._wake_cond.wait_for(predicate)
            else:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise QueueEmpty()
                if not client._wake_cond.wait_for(predicate, timeout=remaining):
                    raise QueueEmpty()
            # Predicate matched. Drain data if any; otherwise stop signal won.
            if client._inbound:
                return client._inbound.popleft()
            raise QueueEmpty()

    def get_nowait(self) -> dict:
        with self._client._inbound_lock:
            if not self._client._inbound:
                raise QueueEmpty()
            return self._client._inbound.popleft()

    def put_nowait(self, msg: dict) -> None:
        """Outbound — publish via broker. Worker's bus.publish API."""
        self._client.publish(msg)

    def put(self, msg: dict, timeout: Optional[float] = None) -> None:
        """Outbound. timeout is ignored (broker-side queue is bounded; we
        do not wait for ack — same fire-and-forget semantics as mp.Queue.put_nowait
        the worker code already uses)."""
        self._client.publish(msg)

    def qsize(self) -> int:
        with self._client._inbound_lock:
            return len(self._client._inbound)

    def empty(self) -> bool:
        return self.qsize() == 0

