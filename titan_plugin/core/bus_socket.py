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

from titan_plugin.bus_specs import coalesce_key, get_spec
from titan_plugin.core._frame import (
    AUTH_TAG_SIZE,
    CHALLENGE_SIZE,
    compute_hmac,
    constant_time_eq,
    recv_exact,
    recv_frame,
    send_frame,
)
from titan_plugin import bus

logger = logging.getLogger(__name__)

# ── Tunables (keep modest; broker is a hot path) ───────────────────────────

DEFAULT_RING_CAPACITY = 1024
DEFAULT_P0_RESERVE = 64
PING_INTERVAL_S = 5.0
PING_TIMEOUT_S = 15.0           # 3 missed pings → close
ACCEPT_RATE_LIMIT_PER_S = 50


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


def _packb_safe(msg) -> bytes:
    """msgpack.packb wrapper using _pack_default.

    All bus_socket producer sites should call this rather than
    msgpack.packb directly. Defense-in-depth against malformed-frame
    cascades (BUG-BUS-IPC-SPIRIT-MALFORMED-FRAME-20260428).
    """
    return msgpack.packb(msg, default=_pack_default)
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
    last_pong_ts: float = field(default_factory=time.time)
    drop_count_60s: int = 0
    recv_count_60s: int = 0
    last_warning_ts: float = 0.0
    last_window_reset_ts: float = field(default_factory=time.time)
    has_data_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    recv_thread: Optional[threading.Thread] = None
    send_thread: Optional[threading.Thread] = None


# ── BusSocketServer ─────────────────────────────────────────────────────────


class BusSocketServer:
    """The broker. Bound to /tmp/titan_bus_<titan_id>.sock by the kernel."""

    def __init__(self, titan_id: str, authkey: bytes,
                 sock_path: Optional[Path] = None,
                 ring_capacity: int = DEFAULT_RING_CAPACITY,
                 p0_reserve: int = DEFAULT_P0_RESERVE,
                 on_inbound_publish: Optional[Callable[[dict], None]] = None) -> None:
        if not titan_id:
            raise ValueError("titan_id required")
        if len(authkey) == 0:
            raise ValueError("authkey required (non-empty bytes)")
        self.titan_id = titan_id
        self._authkey = authkey
        self.sock_path = Path(sock_path) if sock_path else bus_sock_path(titan_id)
        self._ring_capacity = ring_capacity
        self._p0_reserve = p0_reserve
        # Phase B.2.1 fix (2026-04-27): when a worker publishes a message via
        # the broker, the broker fanouts to other broker subscribers (other
        # workers). It does NOT — by default — relay the message back into
        # the kernel's in-process DivineBus, so kernel-side subscribers like
        # the shadow_swap orchestrator never see worker → kernel messages
        # (UPGRADE_READINESS_REPORT, BUS_HANDOFF_ACK, BUS_WORKER_ADOPT_REQUEST,
        # etc.). The kernel passes a callback here that re-enters DivineBus
        # *without* re-forwarding to the broker (avoiding a loop). When None,
        # this stays an isolated worker-to-worker bus (used by tests).
        self._on_inbound_publish = on_inbound_publish
        # M-investigate (2026-04-27 PM): instrumentation for the transient
        # post-handoff disconnect bug. When BUS_HANDOFF is published via
        # broker.publish(), record the timestamp. If a subscriber is
        # subsequently purged within 10s, log a TRANSIENT_HANDOFF_DROP
        # warning so we can correlate worker drops with the swap protocol.
        self._last_handoff_publish_ts: float = 0.0

        self._server_sock: Optional[socket.socket] = None
        self._subscribers: dict[str, BrokerSubscriber] = {}  # name → sub
        self._subs_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._anon_counter = 0
        self._accept_window_ts = 0.0
        self._accept_window_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind the socket, start accept + ping threads. Idempotent-failsafe:
        unlinks any stale socket file from a previous crashed kernel."""
        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError as e:
                logger.warning("[bus_socket] could not unlink stale %s: %s",
                               self.sock_path, e)
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.sock_path))
        os.chmod(self.sock_path, 0o600)
        s.listen(64)
        self._server_sock = s
        self._stop_event.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="bus-broker-accept")
        self._accept_thread.start()
        self._ping_thread = threading.Thread(
            target=self._ping_loop, daemon=True, name="bus-broker-ping")
        self._ping_thread.start()
        logger.info("[bus_socket] broker listening on %s", self.sock_path)

    def stop(self, timeout: float = 2.0) -> None:
        """Close all client connections, unbind the socket, unlink the file."""
        self._stop_event.set()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        # Close all subscriber connections
        with self._subs_lock:
            subs = list(self._subscribers.values())
            self._subscribers.clear()
        for sub in subs:
            self._purge_subscriber(sub, log_reason="server_stop")
        # Unlink socket file
        try:
            if self.sock_path.exists():
                self.sock_path.unlink()
        except OSError:
            pass
        # Wait for threads
        for t in (self._accept_thread, self._ping_thread):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)

    # ── Accept loop (with rate limit) ─────────────────────────────────────

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)  # so we wake to check _stop_event
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    logger.exception("[bus_socket] accept failed")
                return
            if not self._allow_accept_now():
                logger.warning("[bus_socket] accept rate limit hit; refusing %s",
                               self._next_anon())
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            t = threading.Thread(
                target=self._handle_client, args=(conn,),
                daemon=True, name="bus-broker-client",
            )
            t.start()

    def _allow_accept_now(self) -> bool:
        """Token-bucket-like: ACCEPT_RATE_LIMIT_PER_S accepts per rolling 1-sec window."""
        now = time.time()
        if now - self._accept_window_ts >= 1.0:
            self._accept_window_ts = now
            self._accept_window_count = 0
        if self._accept_window_count >= ACCEPT_RATE_LIMIT_PER_S:
            return False
        self._accept_window_count += 1
        return True

    def _next_anon(self) -> str:
        self._anon_counter += 1
        return f"anon-{self._anon_counter}"

    # ── Per-client handler ────────────────────────────────────────────────

    def _handle_client(self, conn: socket.socket) -> None:
        """Handshake, install subscriber, run recv loop until EOF/error."""
        addr = self._next_anon()
        try:
            if not self._handshake(conn):
                conn.close()
                return
        except (ConnectionError, OSError, ValueError) as e:
            logger.warning("[bus_socket] handshake failed for %s: %s", addr, e)
            try:
                conn.close()
            except OSError:
                pass
            return

        sub = BrokerSubscriber(name=addr, conn=conn, addr=addr,
                               ring=BoundedRing(self._ring_capacity, self._p0_reserve))
        # Send + recv threads
        sub.send_thread = threading.Thread(
            target=self._send_loop, args=(sub,),
            daemon=True, name=f"bus-broker-send-{addr}")
        sub.recv_thread = threading.Thread(
            target=self._recv_loop, args=(sub,),
            daemon=True, name=f"bus-broker-recv-{addr}")
        with self._subs_lock:
            self._subscribers[sub.name] = sub
        sub.send_thread.start()
        sub.recv_thread.start()

    def _handshake(self, conn: socket.socket) -> bool:
        challenge = secrets.token_bytes(CHALLENGE_SIZE)
        conn.sendall(challenge)
        client_hmac = recv_exact(conn, AUTH_TAG_SIZE)
        expected = compute_hmac(self._authkey, challenge)
        return constant_time_eq(client_hmac, expected)

    # ── Recv loop (per-subscriber thread) ─────────────────────────────────

    def _recv_loop(self, sub: BrokerSubscriber) -> None:
        try:
            while not self._stop_event.is_set() and not sub.closed:
                try:
                    payload = recv_frame(sub.conn)
                except (ConnectionError, OSError, ValueError):
                    break
                try:
                    msg = msgpack.unpackb(payload, raw=False)
                except (ValueError, msgpack.UnpackException) as _unpack_err:
                    # BUG-BUS-IPC-SPIRIT-MALFORMED-FRAME-20260428 diagnostic:
                    # Hex-dump first 64 bytes of the offending payload + the
                    # last 16 bytes (truncation tail check). This is a hot
                    # path for the failure mode but rare in steady state, so
                    # the cost is negligible + the data is invaluable for
                    # root-causing the next reproduction.
                    _hex_head = payload[:64].hex() if payload else "<empty>"
                    _hex_tail = payload[-16:].hex() if len(payload) > 64 else ""
                    logger.warning(
                        "[bus_socket] malformed frame from %s; closing "
                        "(unpack=%s, payload_len=%d, head_hex=%s%s)",
                        sub.name, _unpack_err, len(payload), _hex_head,
                        f", tail_hex={_hex_tail}" if _hex_tail else "")
                    break
                if not isinstance(msg, dict):
                    logger.warning("[bus_socket] non-dict msg from %s; closing "
                                   "(got type=%s, repr=%.200r)",
                                   sub.name, type(msg).__name__, msg)
                    break
                self._handle_inbound(sub, msg)
        finally:
            self._purge_subscriber(sub, log_reason="recv_loop_end")

    def _handle_inbound(self, sub: BrokerSubscriber, msg: dict) -> None:
        mtype = msg.get("type")
        # First, BUS_SUBSCRIBE registers this connection's identity + topics
        if mtype == bus.BUS_SUBSCRIBE:
            payload = msg.get("payload") or {}
            new_name = payload.get("name") or sub.name
            topics = payload.get("topics") or []
            with self._subs_lock:
                # Promote sub from anon-X to its real name (replacing any prior)
                if new_name != sub.name:
                    self._subscribers.pop(sub.name, None)
                    sub.name = new_name
                    self._subscribers[new_name] = sub
                sub.subscribed_topics.update(topics)
            return
        if mtype == bus.BUS_UNSUBSCRIBE:
            payload = msg.get("payload") or {}
            topics = payload.get("topics") or []
            with self._subs_lock:
                sub.subscribed_topics.difference_update(topics)
            return
        if mtype == bus.BUS_PONG:
            sub.last_pong_ts = time.time()
            return
        # Otherwise, this is a worker publishing — broker dispatches it
        self.publish(msg, _from_subscriber=sub)
        # Phase B.2.1 fix (2026-04-27): also relay into the kernel's in-process
        # DivineBus so kernel-side subscribers (shadow_swap orchestrator,
        # Guardian, etc.) receive worker → kernel messages. Callback is wired
        # to DivineBus.publish_in_process — same in-process delivery path as
        # publish() but skips re-forwarding to this broker (avoids a loop).
        # No-op when on_inbound_publish is None (test fixtures, isolated brokers).
        cb = self._on_inbound_publish
        if cb is not None:
            try:
                cb(msg)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[bus_socket] on_inbound_publish callback raised; "
                    "broker fanout unaffected")

    # ── Publish + dispatch (called from kernel internals OR from a worker) ─

    # Phase C C-S2: drift-bridge dual-emit pairs (legacy ↔ canonical) per
    # SPEC §3 D13/D14/D15 + PLAN_microkernel_phase_c_s2_kernel.md §12.7.
    # Same logic ships in titan-bus::drift_bridge.rs §9.5; both brokers
    # behave identically. C-S8 deletes both.
    _PHASE_C_BRIDGE_PAIRS = {
        # D13: BUS_HANDOFF / BUS_HANDOFF_CANCELED ↔ SWAP_HANDOFF / SWAP_HANDOFF_CANCELED
        "BUS_HANDOFF": "SWAP_HANDOFF",
        "SWAP_HANDOFF": "BUS_HANDOFF",
        "BUS_HANDOFF_CANCELED": "SWAP_HANDOFF_CANCELED",
        "SWAP_HANDOFF_CANCELED": "BUS_HANDOFF_CANCELED",
        # D14: BUS_WORKER_ADOPT_* ↔ ADOPTION_*
        "BUS_WORKER_ADOPT_REQUEST": "ADOPTION_REQUEST",
        "ADOPTION_REQUEST": "BUS_WORKER_ADOPT_REQUEST",
        "BUS_WORKER_ADOPT_ACK": "ADOPTION_ACK",
        "ADOPTION_ACK": "BUS_WORKER_ADOPT_ACK",
        # D15: EPOCH_TICK ↔ KERNEL_EPOCH_TICK
        "EPOCH_TICK": "KERNEL_EPOCH_TICK",
        "KERNEL_EPOCH_TICK": "EPOCH_TICK",
    }

    def publish(self, msg: dict, *, _from_subscriber: Optional[BrokerSubscriber] = None) -> None:
        """Route a message to every matching subscriber.

        Routing rules (mirror existing DivineBus):
          - If msg["dst"] == "all" or empty: deliver to every subscriber that
            either subscribed_topics is empty (legacy "subscribe-all") or has
            msg["type"] in subscribed_topics
          - Otherwise: deliver to the subscriber whose name == msg["dst"]
            (if any). Optional msg["type"] check still applies for safety.

        Phase C C-S2 (per SPEC §3 D13/D14/D15 + PLAN §12.7): drift-bridge
        dual-emit. After fanout to the original `msg["type"]`, if the type
        is in `_PHASE_C_BRIDGE_PAIRS`, fanout a copy with the bridged name
        so subscribers listening on either legacy or canonical name
        receive the message. C-S8 deletes this bridge.
        """
        # M-investigate (2026-04-27 PM): track last BUS_HANDOFF publish for
        # transient-drop correlation in _purge_subscriber.
        if msg.get("type") == bus.BUS_HANDOFF:
            self._last_handoff_publish_ts = time.time()
        dst = msg.get("dst") or "all"
        with self._subs_lock:
            subs = list(self._subscribers.values())
        for sub in subs:
            if sub is _from_subscriber:
                continue  # don't echo to publisher
            if dst != "all" and sub.name != dst:
                continue
            self._enqueue_to(sub, msg)

        # Phase C drift-bridge dual-emit. Only re-fans-out for the bridged
        # name; original fanout above already delivered the canonical type.
        bridged_type = self._PHASE_C_BRIDGE_PAIRS.get(msg.get("type", ""))
        if bridged_type is not None:
            bridged = dict(msg)
            bridged["type"] = bridged_type
            for sub in subs:
                if sub is _from_subscriber:
                    continue
                if dst != "all" and sub.name != dst:
                    continue
                self._enqueue_to(sub, bridged)

    def _enqueue_to(self, sub: BrokerSubscriber, msg: dict) -> None:
        spec = get_spec(msg.get("type", ""))
        with sub.lock:
            sub.recv_count_60s += 1
            self._maybe_reset_window(sub)
            # Coalesce path
            if spec.coalesce is not None:
                key = coalesce_key(spec, msg)
                if key in sub.coalesce_index:
                    # Mutate existing dict in place — no new ring slot consumed
                    sub.coalesce_index[key].clear()
                    sub.coalesce_index[key].update(msg)
                    sub.has_data_event.set()
                    return
                # New key — append + record
                if spec.priority == 0:
                    appended_clean = sub.ring.append_p0(msg)
                else:
                    appended_clean = sub.ring.append_main(msg)
                sub.coalesce_index[key] = msg
                if not appended_clean:
                    self._record_drop(sub, msg, reason="ring_evict_on_append")
                sub.has_data_event.set()
                return
            # Non-coalesce — events
            if spec.priority == 0:
                ok = sub.ring.append_p0(msg)
                if not ok:
                    self._record_drop(sub, msg, reason="p0_reserve_full")
            elif spec.priority == 3:
                # Drop NEWEST under hard pressure
                if sub.ring.main_is_full():
                    self._record_drop(sub, msg, reason="p3_drop_newest")
                else:
                    sub.ring.append_main(msg)
            else:
                # P1 / P2 — drop oldest (deque maxlen does this naturally)
                ok = sub.ring.append_main(msg)
                if not ok:
                    self._record_drop(sub, msg, reason="ring_evict_on_append")
            sub.has_data_event.set()

    def _record_drop(self, sub: BrokerSubscriber, msg: dict, *, reason: str) -> None:
        sub.drop_count_60s += 1
        # Slow-consumer warning (debounced)
        denom = max(sub.recv_count_60s, 1)
        rate = sub.drop_count_60s / denom
        if rate > SLOW_CONSUMER_DROP_RATE_THRESHOLD:
            now = time.time()
            if now - sub.last_warning_ts > SLOW_CONSUMER_WARN_INTERVAL_S:
                sub.last_warning_ts = now
                logger.warning(
                    "[bus_socket] BUS_SLOW_CONSUMER name=%s drops=%d/%d (%.1f%%) reason=%s",
                    sub.name, sub.drop_count_60s, denom, rate * 100, reason)

    def _maybe_reset_window(self, sub: BrokerSubscriber) -> None:
        now = time.time()
        if now - sub.last_window_reset_ts > 60.0:
            sub.drop_count_60s = 0
            sub.recv_count_60s = 0
            sub.last_window_reset_ts = now

    # ── Send loop (per-subscriber thread) ─────────────────────────────────

    def _send_loop(self, sub: BrokerSubscriber) -> None:
        try:
            while not self._stop_event.is_set() and not sub.closed:
                # Wait for data or short timeout (so we wake to check stop)
                if not sub.has_data_event.wait(SEND_FLUSH_TIMEOUT_S):
                    continue
                with sub.lock:
                    if sub.ring.is_empty():
                        sub.has_data_event.clear()
                        continue
                    batch = sub.ring.pop_for_send(max_msgs=64)
                    # Remove popped messages from coalesce_index by id
                    for popped in batch:
                        spec = get_spec(popped.get("type", ""))
                        if spec.coalesce is not None:
                            key = coalesce_key(spec, popped)
                            if (key in sub.coalesce_index
                                    and sub.coalesce_index[key] is popped):
                                del sub.coalesce_index[key]
                    if sub.ring.is_empty():
                        sub.has_data_event.clear()
                # Outside the lock — actually send
                if not batch:
                    continue
                if len(batch) >= SEND_BATCH_THRESHOLD:
                    payload = _packb_safe({"type": "BUS_BATCH", "msgs": batch})
                else:
                    # Send each individually for low-load latency
                    for m in batch:
                        try:
                            send_frame(sub.conn, _packb_safe(m))
                        except (ConnectionError, OSError):
                            sub.closed = True
                            return
                    continue
                try:
                    send_frame(sub.conn, payload)
                except (ConnectionError, OSError):
                    sub.closed = True
                    return
        except Exception:
            logger.exception("[bus_socket] send loop crash for %s", sub.name)
        finally:
            sub.closed = True

    # ── Ping loop ─────────────────────────────────────────────────────────

    def _ping_loop(self) -> None:
        while not self._stop_event.wait(PING_INTERVAL_S):
            now = time.time()
            with self._subs_lock:
                subs = list(self._subscribers.values())
            ping_msg = {"type": "BUS_PING", "src": "broker", "dst": "all", "ts": now,
                        "payload": {}}
            for sub in subs:
                if sub.closed:
                    continue
                if (now - sub.last_pong_ts) > PING_TIMEOUT_S:
                    logger.warning("[bus_socket] BUS_PONG timeout for %s; closing",
                                   sub.name)
                    self._purge_subscriber(sub, log_reason="pong_timeout")
                    continue
                # Enqueue ping into sub's ring
                self._enqueue_to(sub, ping_msg)

    # ── Subscriber teardown ───────────────────────────────────────────────

    def _purge_subscriber(self, sub: BrokerSubscriber, *, log_reason: str) -> None:
        if sub.closed:
            return
        sub.closed = True
        sub.has_data_event.set()  # wake send loop so it can exit
        try:
            sub.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sub.conn.close()
        except OSError:
            pass
        with self._subs_lock:
            self._subscribers.pop(sub.name, None)
        # M-investigate (2026-04-27 PM): if the purge happens within 10s of
        # the last BUS_HANDOFF publish AND the reason isn't an explicit
        # B.2.1 disconnect, this is the transient-drop-at-handoff bug we
        # observed (4 workers dropping at 18:24:36). Log at WARNING so it
        # surfaces in arch_map errors. Helps next-session investigation.
        if log_reason != "b2_1_handoff" and self._last_handoff_publish_ts > 0:
            elapsed_since_handoff = time.time() - self._last_handoff_publish_ts
            if elapsed_since_handoff < 10.0:
                logger.warning(
                    "[bus_socket] TRANSIENT_HANDOFF_DROP: subscriber %s purged "
                    "%.2fs after BUS_HANDOFF (reason=%s) — root cause TBD",
                    sub.name, elapsed_since_handoff, log_reason,
                )
                return  # avoid duplicate purge log
        logger.info("[bus_socket] subscriber %s purged (%s)", sub.name, log_reason)

    def disconnect_subscribers(self, names: list[str], *, reason: str = "kernel_handoff") -> list[str]:
        """Phase B.2.1 (2026-04-27 PM) — proactively disconnect specific subscribers.

        After BUS_HANDOFF + shadow's broker is bound on the same socket
        path, the OLD kernel's broker still owns existing FD connections
        (TCP-like persistence — unlinking the socket file doesn't kill
        live connections). For spawn-mode workers to reconnect to the
        SHADOW broker (and adopt-register), the old kernel's broker must
        actively close their connections.

        Workers' BusSocketClient detects the disconnect (recv frame error)
        and triggers its reconnect loop → re-attaches to /tmp/titan_bus_<id>.sock
        which is now bound by shadow. Worker's supervision daemon then
        observes reconnect_count++ and calls request_adoption(state).

        Args:
            names:  subscriber names to disconnect (typically the spawn-mode
                    workers that acked BUS_HANDOFF).
            reason: log line for the purge event.

        Returns the list of names that were actually disconnected (intersect
        of `names` with currently-registered subscribers — others are no-ops).
        """
        with self._subs_lock:
            to_close = [
                self._subscribers[n] for n in names if n in self._subscribers
            ]
        purged: list[str] = []
        for sub in to_close:
            self._purge_subscriber(sub, log_reason=reason)
            purged.append(sub.name)
        if purged:
            logger.info(
                "[bus_socket] disconnect_subscribers: closed %d connections "
                "(reason=%s, names=%s)",
                len(purged), reason, sorted(purged),
            )
        return purged

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot for arch_map bus-status. Cheap; takes the subs_lock briefly."""
        with self._subs_lock:
            return {
                "sock_path": str(self.sock_path),
                "subscriber_count": len(self._subscribers),
                "subscribers": [
                    {
                        "name": s.name,
                        "ring_size": len(s.ring),
                        "drop_count_60s": s.drop_count_60s,
                        "recv_count_60s": s.recv_count_60s,
                        "topics": sorted(s.subscribed_topics),
                        "last_pong_age_s": time.time() - s.last_pong_ts,
                    }
                    for s in self._subscribers.values()
                ],
            }


# ── BusSocketClient + SocketQueue (C5) ──────────────────────────────────────

# Reconnect backoff
RECONNECT_JITTER_INITIAL_MIN_S = 0.05
RECONNECT_JITTER_INITIAL_MAX_S = 0.15
RECONNECT_BACKOFF_BASE_S = 0.1
RECONNECT_BACKOFF_MAX_S = 2.0


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
                 inbound_capacity: int = DEFAULT_RING_CAPACITY) -> None:
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
        self._inbound: deque = deque()
        self._inbound_event = threading.Event()
        self._inbound_lock = threading.Lock()
        self._inbound_capacity = inbound_capacity
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._conn_thread: Optional[threading.Thread] = None
        # Subscription state owned locally — re-published on every reconnect
        self._topics: set[str] = set(self._initial_topics)
        self._topics_lock = threading.Lock()
        self._reconnect_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._conn_thread is not None and self._conn_thread.is_alive():
            return
        self._stop_event.clear()
        self._conn_thread = threading.Thread(
            target=self._connection_loop, daemon=True,
            name=f"bus-client-{self.name}")
        self._conn_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
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
        # Wake any pending get() callers
        self._inbound_event.set()
        if self._conn_thread is not None and self._conn_thread.is_alive():
            self._conn_thread.join(timeout=timeout)

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

    def unsubscribe(self, topics: list[str]) -> None:
        with self._topics_lock:
            self._topics.difference_update(topics)
        self._send_unsubscribe_frame(list(topics))

    def _send_subscribe_frame(self, topics: list[str]) -> None:
        msg = {
            "type": "BUS_SUBSCRIBE", "src": self.name, "dst": "broker",
            "payload": {"name": self.name, "topics": topics},
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
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            send_frame(sock, _packb_safe(msg))
            return True
        except (ConnectionError, OSError):
            return False
        except Exception:  # noqa: BLE001
            # Defensive: if _packb_safe itself raises (e.g., a type its
            # default-callback can't coerce that also breaks repr()), don't
            # crash the worker thread — log + drop the frame so the
            # connection survives. Without this guard, TypeError from a
            # genuinely unencodable payload would propagate and either
            # crash the publisher or produce torn frames on the wire
            # (the original BUG-BUS-IPC-SPIRIT-MALFORMED-FRAME mode).
            logger.exception(
                "[bus_client] msgpack pack failed; dropping frame "
                "(type=%s, msg keys=%s)",
                msg.get("type") if isinstance(msg, dict) else "?",
                list(msg.keys())[:10] if isinstance(msg, dict) else "?")
            return False

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
                self._connected_event.set()
                if backoff_attempt > 0:
                    self._reconnect_count += 1
                # Recv loop blocks here until EOF/error
                self._recv_loop(sock)
            except (ConnectionError, OSError, ValueError) as e:
                logger.debug("[bus_client] connection attempt failed: %s", e)
            finally:
                self._connected_event.clear()
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
            # Unwrap and deliver each
            for inner in msg.get("msgs", []):
                self._deliver_to_inbound(inner)
            return
        self._deliver_to_inbound(msg)

    def _deliver_to_inbound(self, msg: dict) -> None:
        with self._inbound_lock:
            if len(self._inbound) >= self._inbound_capacity:
                # Drop oldest — match broker's eviction semantics so worker
                # doesn't accumulate stale state when it can't keep up
                self._inbound.popleft()
            self._inbound.append(msg)
        self._inbound_event.set()


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

        Mirrors Queue.get(timeout=N) semantics — including the partial-wait
        loop pattern (event may be falsely set after a drain elsewhere)."""
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            with self._client._inbound_lock:
                if self._client._inbound:
                    msg = self._client._inbound.popleft()
                    if not self._client._inbound:
                        self._client._inbound_event.clear()
                    return msg
                self._client._inbound_event.clear()
            if self._client._stop_event.is_set():
                raise QueueEmpty()
            if deadline is None:
                self._client._inbound_event.wait()
            else:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise QueueEmpty()
                if not self._client._inbound_event.wait(timeout=remaining):
                    raise QueueEmpty()

    def get_nowait(self) -> dict:
        with self._client._inbound_lock:
            if not self._client._inbound:
                raise QueueEmpty()
            msg = self._client._inbound.popleft()
            if not self._client._inbound:
                self._client._inbound_event.clear()
            return msg

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

