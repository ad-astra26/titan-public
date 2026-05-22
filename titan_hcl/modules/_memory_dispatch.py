"""
Memory worker dispatch infrastructure — Phase A of
`PLAN_phase_c_memory_worker_concurrent_dispatch.md` (rFP §3.4.1).

Splits the previously-serial `memory_worker.py` main loop into:
  * `_read_pool` (8 workers) — READ_ACTIONS dispatch
  * `_writer_pool` (1 worker, serial) — WRITE_ACTIONS + MEMORY_ADD / MEMORY_MEMPOOL_ADD
  * dedicated `_meditation_thread` (lazy spawn) — SPECIAL_ACTIONS (run_meditation)

Each thread owns its own asyncio loop (via `ensure_thread_loop`) so coroutine
calls don't share a single loop across threads (which would race the existing
`_periodic_publish_loop` thread that already drives the main loop).

The main loop in `memory_worker_main` remains the SOLE `recv_queue` reader.
Threads that need a bus-bridge round-trip (the meditation thread's
ANCHOR_REQUEST/RESPONSE flow) register their rid in `InFlightRegistry`; the
main loop, on every incoming msg, gives the registry first dibs at resolving
the rid before normal action dispatch.

Phase C SPEC compliance (load-bearing):
  * G18 — state transport stays SHM (`memory_state.bin` publisher untouched).
  * G19 — every action handler is a true work-RPC; timeouts unchanged in
    Phase A (Phase B tightens to ≤5s per rFP §3.4.1).
  * G20 — pre-warmed cache is Phase B (LRU on `query`); Phase A delivers the
    pool isolation that lets reads run while writes are in flight.
  * G21 — `memory_state.bin` retains single-writer (`_periodic_publish_loop`).
  * G22 — no new `action="get_*"` handlers introduced.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Hashable, Optional

logger = logging.getLogger(__name__)


# ── Phase B (rFP §3.4.1) — query LRU+TTL cache ────────────────────────────


class QueryCache:
    """Thread-safe LRU+TTL cache for memory_worker `query` action results.

    Phase B G20 closure: pre-warmed cache in front of FAISS+Kuzu+DuckDB so
    repeated identical queries on the chat hot path return in <1ms instead
    of paying the full FAISS read budget every time. Bounded `maxsize`
    keeps memory pressure deterministic; `ttl_s` keeps results fresh enough
    that producer-side writes are visible within one TTL window even
    without explicit invalidation (PLAN B5 verbatim — TTL-only invalidation
    is the canonical pattern for semantic-search caches).

    Implementation: plain `OrderedDict` keyed on the cache key + a per-entry
    timestamp. Single `threading.Lock` because read-pool workers may hit
    the cache concurrently. All operations O(1) amortized except the
    bounded eviction sweep on `_set` which is O(1) average via popitem.
    """

    def __init__(self, maxsize: int = 256, ttl_s: float = 60.0) -> None:
        if maxsize <= 0:
            raise ValueError("QueryCache maxsize must be > 0")
        if ttl_s <= 0:
            raise ValueError("QueryCache ttl_s must be > 0")
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._store: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Return cached value or None on miss / expiry."""
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if now - ts > self._ttl_s:
                # Expired — drop and miss.
                self._store.pop(key, None)
                self._misses += 1
                return None
            # LRU touch: move to end (most-recently-used).
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.time()
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (now, value)
                return
            self._store[key] = (now, value)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict LRU
                self._evictions += 1

    def invalidate(self) -> int:
        """Drop every entry. Returns count cleared. Used after writes if
        the worker wants strict consistency (Phase B uses TTL-only by
        default; this is here for tests + future use)."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "ttl_s": self._ttl_s,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0 else 0.0
                ),
            }


# ── Action categorization ──────────────────────────────────────────────────
# Source of truth for which thread pool runs each action. Kept here (not in
# memory_worker.py) so tests can import + assert classification without
# importing the worker entry function.

READ_ACTIONS = frozenset({
    "query",
    "fetch_mempool",
    "fetch_mempool_observatory",
    "top_memories",
    "top_memories_observatory",
    "topology",
    "knowledge_graph",
    "count",
    "status",
})

WRITE_ACTIONS = frozenset({
    # Phase B (rFP §3.4.1 D-SPEC-44, 2026-05-13): `add` RETIRED — migrated
    # to one-way MEMORY_INGEST_REQUEST event (handled by
    # _handle_memory_ingest_request on the writer pool, dispatched via
    # ActionRouter's msg_type branch — NOT via bus.QUERY action). Proxy's
    # add_memory + 3 spirit_worker producers all migrated; orphan handler
    # in memory_worker.py deleted per G-RPC-4.
    # Phase A (commit 72932cf4, 2026-05-12): `add_to_mempool` RETIRED —
    # similar pattern via MEMORY_MEMPOOL_ADD event.
    # This frozenset is intentionally EMPTY: every write now flows through
    # a msg_type-keyed bus event, not the work-RPC action channel.
})

SPECIAL_ACTIONS = frozenset({
    "run_meditation",     # dedicated thread (Maker decision Q2, 2026-05-12)
})


# ── Per-thread asyncio loop registry ───────────────────────────────────────

_thread_local = threading.local()


def ensure_thread_loop() -> asyncio.AbstractEventLoop:
    """Return the asyncio event loop owned by the calling thread.

    Each worker thread (read pool worker, writer pool worker, meditation
    thread) creates its own loop on first call; reuses it across subsequent
    submissions on the same thread. Avoids the "asyncio loop already running
    in another thread" hazard that the pre-Phase-A serial dispatcher avoided
    by virtue of having only one loop on the main thread.
    """
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    return loop


def close_thread_loop() -> None:
    """Close the calling thread's asyncio loop (called from pool shutdown)."""
    loop = getattr(_thread_local, "loop", None)
    if loop is not None and not loop.is_closed():
        try:
            loop.close()
        except Exception:
            pass
    _thread_local.loop = None


# ── In-flight rid registry ─────────────────────────────────────────────────

class InFlightRegistry:
    """Routes RESPONSE messages to threads waiting on bus.request round-trips.

    Phase C constraint (load-bearing): the main loop is the sole `recv_queue`
    reader. The meditation thread's ANCHOR_REQUEST/RESPONSE bus-bridge can no
    longer pull directly from `recv_queue` (it would race the main loop).

    Pattern:
      1. Thread allocates a `rid` + `Future` via `register(rid)`.
      2. Thread sends its request through `send_queue` with that `rid`.
      3. Thread calls `future.result(timeout=...)` and blocks.
      4. Main loop, on every incoming msg, calls `resolve(msg)` first. If the
         msg's rid matches a registered Future, `resolve` sets the result and
         returns True so the main loop skips normal action dispatch for that
         message.

    Concurrent-safety: a single `threading.Lock` guards the futures dict.
    `Future.set_result` itself is thread-safe (`concurrent.futures.Future`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    def register(self, rid: str) -> Future:
        if not rid:
            raise ValueError("InFlightRegistry.register requires non-empty rid")
        fut: Future = Future()
        with self._lock:
            if rid in self._futures:
                raise RuntimeError(
                    f"InFlightRegistry: rid already in-flight: {rid}")
            self._futures[rid] = fut
        return fut

    def resolve(self, msg: dict) -> bool:
        """Return True if msg matches an in-flight rid.

        Caller (main loop) should skip normal dispatch for messages that
        resolve a registered Future — those bytes are owned by the thread
        that registered the rid, not by the action dispatcher.
        """
        rid = msg.get("rid")
        if not rid:
            return False
        with self._lock:
            fut = self._futures.pop(rid, None)
        if fut is None:
            return False
        try:
            fut.set_result(msg)
        except concurrent.futures.InvalidStateError:
            # Future was already cancelled/resolved; nothing more to do.
            pass
        return True

    def cancel(self, rid: str) -> None:
        """Drop a registered rid without resolving its Future. Used when the
        waiting thread times out and no longer cares about the response."""
        with self._lock:
            self._futures.pop(rid, None)

    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._futures)


# ── Worker context bundle ──────────────────────────────────────────────────

@dataclasses.dataclass
class WorkerContext:
    """State passed to every action handler.

    `write_lock` is acquired by writer-pool handlers + meditation thread
    around any call that mutates persistent state (FAISS index / Kuzu /
    DuckDB). Reads do NOT acquire this lock — FAISS read-during-write is
    safe (copy-on-write segment under write). If that assumption ever
    breaks, upgrade to a reader-writer lock without changing dispatch.

    `query_cache` is the Phase B (rFP §3.4.1) LRU+TTL cache for the
    `query` action; default None for tests that don't need it. Production
    `memory_worker_main` constructs a single shared cache and threads it
    through here.
    """
    memory: object              # TieredMemoryGraph (avoid import cycle)
    send_queue: object          # multiprocessing.Queue
    name: str                   # "memory"
    config: dict
    in_flight: InFlightRegistry
    write_lock: threading.RLock
    query_cache: Optional["QueryCache"] = None


# ── Action router ──────────────────────────────────────────────────────────

# Type alias: an action handler takes (msg, ctx) and returns None.
ActionHandler = Callable[[dict, WorkerContext], None]


class ActionRouter:
    """Dispatches incoming worker messages to the right thread / pool.

    The router is the only object the main loop talks to (besides
    `InFlightRegistry.resolve`). It does NOT read `recv_queue` — that's the
    main loop's exclusive job.

    Construction:
        ctx = WorkerContext(memory, send_queue, "memory", config, registry, lock)
        router = ActionRouter(
            ctx,
            handle_query=_handle_query,            # bus.QUERY action dispatch
            handle_memory_add=_handle_memory_add,  # bus.MEMORY_ADD event
            handle_mempool_add=_handle_mempool_add,  # bus.MEMORY_MEMPOOL_ADD event
        )
        # main loop:
        router.dispatch(msg)
        # shutdown:
        router.shutdown(timeout=5.0)
    """

    def __init__(
        self,
        ctx: WorkerContext,
        handle_query: ActionHandler,
        handle_memory_add: ActionHandler,
        handle_mempool_add: ActionHandler,
        handle_memory_ingest_request: Optional[ActionHandler] = None,
        max_read_workers: int = 8,
    ) -> None:
        self.ctx = ctx
        self._handle_query = handle_query
        self._handle_memory_add = handle_memory_add
        self._handle_mempool_add = handle_mempool_add
        # Optional — Phase B (rFP §3.4.1) introduces MEMORY_INGEST_REQUEST.
        # Pre-Phase-B tests + early callers may omit; if a request arrives
        # without a registered handler, dispatch logs + drops to avoid
        # hanging the producer (which is fire-and-forget anyway).
        self._handle_memory_ingest_request = handle_memory_ingest_request

        self._read_pool = ThreadPoolExecutor(
            max_workers=max_read_workers,
            thread_name_prefix="mem-read",
            initializer=ensure_thread_loop,
        )
        self._writer_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mem-write",
            initializer=ensure_thread_loop,
        )

        # Meditation: dedicated thread + dedicated queue. Lazy spawn on
        # first run_meditation call (saves resources when meditation isn't
        # requested in this worker lifetime).
        self._meditation_thread: Optional[threading.Thread] = None
        self._meditation_queue: queue.Queue = queue.Queue()
        self._meditation_stop = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────

    def dispatch(self, msg: dict) -> None:
        """Route a single message to the right pool / thread.

        Caller (main loop) has already given `InFlightRegistry.resolve` the
        first shot at the message. Anything that lands here is genuine
        dispatch traffic. Caller should NOT pass MODULE_SHUTDOWN or
        Microkernel-B.1/B.2.1 swap messages — those stay inline on the main
        loop per existing semantics.
        """
        # Import bus lazily to avoid cycles + per the lazy-imports rule.
        from titan_hcl import bus

        msg_type = msg.get("type", "")

        if msg_type == bus.MEMORY_ADD:
            self._writer_pool.submit(
                self._safe_run, self._handle_memory_add, msg)
            return
        if msg_type == bus.MEMORY_INGEST_REQUEST:
            if self._handle_memory_ingest_request is None:
                logger.warning(
                    "[MemoryDispatch] MEMORY_INGEST_REQUEST received but no "
                    "handler registered — dropping (router was constructed "
                    "without handle_memory_ingest_request kwarg)")
                return
            self._writer_pool.submit(
                self._safe_run, self._handle_memory_ingest_request, msg)
            return
        if msg_type == bus.MEMORY_MEMPOOL_ADD:
            self._writer_pool.submit(
                self._safe_run, self._handle_mempool_add, msg)
            return
        if msg_type == bus.QUERY:
            payload = msg.get("payload", {}) or {}
            action = payload.get("action", "")
            pool = self._select_pool(action)
            if pool == "meditation":
                self._ensure_meditation_thread()
                self._meditation_queue.put(msg)
                return
            if pool == "write":
                self._writer_pool.submit(
                    self._safe_run, self._handle_query, msg)
                return
            if pool == "read":
                self._read_pool.submit(
                    self._safe_run, self._handle_query, msg)
                return
            # Unknown action — send error response so caller doesn't wait
            # for its timeout. Don't import _send_response (cycle); build the
            # response shape inline.
            self._send_error_response(
                msg, f"unknown action: {action}")
            return

        logger.warning(
            "[MemoryDispatch] router.dispatch called for unhandled msg_type=%s "
            "(main loop should handle this inline)",
            msg_type)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop accepting new work and join all threads/pools."""
        logger.info("[MemoryDispatch] shutdown starting")
        # Stop meditation thread (queue sentinel) before pools so it
        # doesn't continue to enqueue writes onto a closing pool.
        self._meditation_stop.set()
        try:
            self._meditation_queue.put_nowait(None)
        except Exception:
            pass
        if self._meditation_thread is not None:
            self._meditation_thread.join(timeout=timeout)
            if self._meditation_thread.is_alive():
                logger.warning(
                    "[MemoryDispatch] meditation thread did not stop in %.1fs",
                    timeout)

        # ThreadPoolExecutor.shutdown(cancel_futures=True) drops queued
        # work without running it; running tasks finish naturally. Phase A
        # accepts this — the alternative (force-kill mid-task) would risk
        # half-written FAISS state.
        self._read_pool.shutdown(wait=True, cancel_futures=True)
        self._writer_pool.shutdown(wait=True, cancel_futures=True)
        logger.info("[MemoryDispatch] shutdown complete")

    # ── Introspection (used by tests + diagnostics) ────────────────────

    @staticmethod
    def classify(action: str) -> str:
        """Pure classification — returns 'read', 'write', 'meditation', or 'unknown'."""
        if action in SPECIAL_ACTIONS:
            return "meditation"
        if action in WRITE_ACTIONS:
            return "write"
        if action in READ_ACTIONS:
            return "read"
        return "unknown"

    def _select_pool(self, action: str) -> str:
        return self.classify(action)

    # ── Internals ──────────────────────────────────────────────────────

    def _safe_run(self, handler: ActionHandler, msg: dict) -> None:
        """Pool-submitted wrapper that swallows handler exceptions so a
        single bad message can't poison a pool worker. Handlers are
        expected to send their own RESPONSE; on uncaught exception we
        send an error response so the caller doesn't hang on timeout."""
        try:
            handler(msg, self.ctx)
        except Exception as e:
            logger.error(
                "[MemoryDispatch] handler raised for msg_type=%s action=%s: %s",
                msg.get("type"),
                (msg.get("payload") or {}).get("action"),
                e, exc_info=True)
            self._send_error_response(msg, f"handler error: {type(e).__name__}: {e}")

    def _send_error_response(self, msg: dict, error: str) -> None:
        """Send a RESPONSE with `{"error": ...}` payload directly via the
        send_queue. Avoids importing memory_worker._send_response (cycle)."""
        from titan_hcl import bus
        import time as _time
        rid = msg.get("rid")
        src = msg.get("src", "")
        try:
            self.ctx.send_queue.put_nowait({
                "type": bus.RESPONSE,
                "src": self.ctx.name,
                "dst": src,
                "ts": _time.time(),
                "rid": rid,
                "payload": {"error": error},
            })
        except Exception:
            from titan_hcl.bus import record_send_drop
            record_send_drop(self.ctx.name, src, bus.RESPONSE)

    def _ensure_meditation_thread(self) -> None:
        if self._meditation_thread is not None and self._meditation_thread.is_alive():
            return
        self._meditation_stop.clear()
        self._meditation_thread = threading.Thread(
            target=self._meditation_loop,
            name="mem-meditation",
            daemon=True,
        )
        self._meditation_thread.start()
        logger.info("[MemoryDispatch] meditation thread spawned (lazy)")

    def _meditation_loop(self) -> None:
        """Long-lived dedicated thread for run_meditation per Maker decision
        2026-05-12 (option b — separate from writer pool for importance +
        debuggability). Acquires its own asyncio loop on first iteration."""
        ensure_thread_loop()
        while not self._meditation_stop.is_set():
            try:
                msg = self._meditation_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                # Shutdown sentinel
                break
            self._safe_run(self._handle_query, msg)
