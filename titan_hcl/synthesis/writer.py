"""SynthesisWriter — the single in-process persistence authority (Option C).

Root cause this closes (AUDIT_synthesis_engine_crashloop_concurrency_20260602.md):
the synthesis_worker shared ONE non-thread-safe ``duckdb.Connection`` (plus the
Kuzu spine graph and the FAISS shards) across six daemon threads guarded by
disjoint per-store locks. DuckDB/Kuzu/FAISS handles are NOT thread-safe;
concurrent ``.execute()`` from two threads corrupts native memory → SIGSEGV →
guardian respawn → crash-loop. The sole-writer invariants (INV-Syn-3/7/8/16/19)
were honored at the PROCESS level but violated at the THREAD level.

The fix realizes "sole writer" at the THREAD level: ONE writer thread is the
sole *invoker* of every native handle. All other threads (recv, recompute, the
dream orchestrator) submit closures; the writer drains them serially. Because a
single thread can never call ``.execute()`` concurrently with itself, the race
is impossible BY CONSTRUCTION — and there are no locks, so there is no
lock-ordering deadlock surface (the failure mode a multi-lock fix would carry).

Enforcement, not convention: connections are handed out via :func:`guard_conn`,
a thin proxy that raises ``WriterThreadViolation`` if ``.execute()`` is called
off the writer thread. A forgotten ``submit()`` therefore fails LOUDLY in tests
instead of segfaulting in production.

SPEC: the intra-process single-writer-thread invariant ratified alongside this
module (ARCHITECTURE_synthesis_engine.md §6 / new INV-Syn). G21 "one slot, one
writer" — at the thread level.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default timeout for submit_sync / flush. Generous: the writer only does fast
# native ops (SQL execute, faiss.add) — never LLM/compute (that stays on the
# caller thread). A blocked submit_sync beyond this is a real bug, not slowness.
DEFAULT_SYNC_TIMEOUT_S = 30.0

# Boot-only timeout for the heavy, boot-CRITICAL native ops (the one
# ``duckdb.connect`` that opens data/synthesis.duckdb + its schema/load).
# Root cause (2026-06-04, T1 mainnet): when synthesis RESPAWNS during a
# boot-grace cascade (dozens of modules restarting at once → box-wide CPU/I/O
# starvation), the writer thread cannot be scheduled to run the connect within
# the steady-state 30s → ``TimeoutError`` → the worker crashes on init and
# synthesis is silently non-functional (process up, init dead). Boot is exactly
# when the box is most contended, so the boot connect must tolerate transient
# congestion. Generous (3×) + the guardian respawn loop is the retry mechanism
# (a fresh process each time → no double-connect). Steady-state submit_sync stays
# 30s so chat-path blockages still surface fast as the real bugs they are.
BOOT_SYNC_TIMEOUT_S = 90.0

# Sentinel enqueued by close() to stop the drain loop after all prior ops run.
_STOP = object()


class WriterThreadViolation(RuntimeError):
    """Raised when a guarded native handle is touched off the writer thread.

    Converts the silent SIGSEGV (concurrent handle use) into a loud,
    test-catchable error. If you see this, a handle op was not routed through
    SynthesisWriter.submit()/submit_sync()."""


class SynthesisWriter:
    """Single-thread serial executor for all synthesis native-handle ops.

    Not tied to any specific handle — it is the one thread permitted to touch
    the DuckDB connection, the Kuzu spine graph, AND the FAISS shards, so every
    op submitted here is serialized against every other regardless of which
    substrate it touches.

    Usage::

        writer = SynthesisWriter(name="T3")
        conn = guard_conn(writer, duckdb.connect(path))   # boot, single-threaded
        writer.start()
        # ... from any thread:
        writer.submit(lambda: conn.execute("INSERT ...", params))      # fire-and-forget
        fid = writer.submit_sync(lambda: _create_fork_returning_id())  # blocking, returns value
        writer.flush()                                                 # barrier
        # shutdown:
        writer.submit_sync(lambda: conn.execute("CHECKPOINT"))
        writer.close()
    """

    def __init__(self, name: str = "", maxsize: int = 0) -> None:
        self._name = name
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._run, name=f"synthesis-writer-{name}", daemon=True)
        self._thread_ident: Optional[int] = None
        self._started = False
        self._closed = threading.Event()
        # Set by the thread once it is live, so start() can hand back a writer
        # whose _thread_ident is populated for the guard proxy.
        self._ready = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self) -> "SynthesisWriter":
        if self._started:
            return self
        self._started = True
        self._thread.start()
        # Block until the thread records its ident — guard proxies created
        # before this returns will then have a valid writer-thread ident.
        self._ready.wait(timeout=5.0)
        return self

    def _run(self) -> None:
        self._thread_ident = threading.get_ident()
        self._ready.set()
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                fn, fut = item
                try:
                    result = fn()
                    if fut is not None:
                        fut.set_result(result)
                except BaseException as exc:  # noqa: BLE001 — surface ALL failures
                    if fut is not None:
                        fut.set_exception(exc)
                    else:
                        # Fire-and-forget op failed: NEVER swallow (the json
                        # NameError + consolidation SQL bugs hid for weeks
                        # exactly this way). directive_error_visibility.
                        logger.warning(
                            "[synthesis_writer:%s] queued op failed: %s",
                            self._name, exc, exc_info=True)
            finally:
                self._q.task_done()

    def close(self, timeout: float = 10.0) -> None:
        """Stop accepting ops, drain everything already queued, join the
        thread. Submit any final CHECKPOINT/close ops BEFORE calling this."""
        if not self._started or self._closed.is_set():
            return
        self._closed.set()
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)

    # ── submission API ─────────────────────────────────────────────────
    def on_writer_thread(self) -> bool:
        return threading.get_ident() == self._thread_ident

    def _assert_on_thread(self, what: str) -> None:
        if threading.get_ident() != self._thread_ident:
            raise WriterThreadViolation(
                f"{what} called off the synthesis writer thread "
                f"(current={threading.current_thread().name!r}). "
                f"Route it through SynthesisWriter.submit()/submit_sync().")

    def submit(self, fn: Callable[[], Any]) -> None:
        """Fire-and-forget: enqueue fn() to run on the writer thread. Returns
        immediately. Op failures are logged at WARN (never swallowed)."""
        if self._closed.is_set():
            raise RuntimeError("SynthesisWriter is closed — submit rejected")
        self._q.put((fn, None))

    def submit_sync(self, fn: Callable[[], Any],
                    timeout: float = DEFAULT_SYNC_TIMEOUT_S) -> Any:
        """Enqueue fn() and BLOCK until the writer runs it; returns its result
        or re-raises its exception. For the bounded set of callers that need
        the value now (create_fork→id, graduate→tx, consistency-sensitive
        reads). If already on the writer thread, run inline to avoid deadlock."""
        if self.on_writer_thread():
            return fn()
        if self._closed.is_set():
            raise RuntimeError("SynthesisWriter is closed — submit_sync rejected")
        fut: "Future[Any]" = Future()
        self._q.put((fn, fut))
        return fut.result(timeout=timeout)

    def flush(self, timeout: float = DEFAULT_SYNC_TIMEOUT_S) -> None:
        """Barrier: block until every op enqueued before this call has run.
        Used between the judge and miner dream passes so the judge's scored_by
        writes are durably visible before the miner reads (INV-Syn-21)."""
        self.submit_sync(lambda: None, timeout=timeout)


class InlineWriter:
    """Synchronous, thread-less SynthesisWriter implementation.

    Runs every submitted op inline on the calling thread. This is the default
    a store falls back to when no real writer is injected — i.e. in unit tests,
    which construct stores with a raw (unguarded) connection and exercise them
    single-threaded. Production ALWAYS injects the real threaded
    :class:`SynthesisWriter`, so the store code path is uniform
    (``self._writer.submit_sync(...)``) with no production dual-mode branch.

    NOT a shim for an old production path — it is a legitimate synchronous
    writer for single-threaded contexts (per feedback_no_shim: the production
    path fully uses the threaded writer; this is the test/inline path)."""

    def start(self) -> "InlineWriter":
        return self

    def on_writer_thread(self) -> bool:
        return True

    def _assert_on_thread(self, what: str) -> None:
        # No-op: an inline writer runs ops on the caller thread, so any thread
        # is "the writer thread". Keeps guard_conn() usable in single-threaded
        # tests without a real writer thread.
        return None

    def submit(self, fn: Callable[[], Any]) -> None:
        fn()

    def submit_sync(self, fn: Callable[[], Any],
                    timeout: float = DEFAULT_SYNC_TIMEOUT_S) -> Any:
        return fn()

    def flush(self, timeout: float = DEFAULT_SYNC_TIMEOUT_S) -> None:
        return None

    def close(self, timeout: float = 10.0) -> None:
        return None


def resolve_writer(writer: Any) -> Any:
    """Return ``writer`` if provided, else a fresh :class:`InlineWriter`.
    Stores use this so a test that omits the writer still gets a working
    (synchronous) one, while production injects the threaded writer."""
    return writer if writer is not None else InlineWriter()


def on_writer(method: Callable) -> Callable:
    """Decorator: run a store method on its ``self._writer`` thread.

    Apply to any method whose body is a PURE native-handle op (DuckDB/Kuzu/
    FAISS) with no heavy compute — it then executes serially on the single
    writer thread, so it can never race another handle user. The store must
    expose ``self._writer`` (a SynthesisWriter or InlineWriter).

    Re-entrancy is safe: ``submit_sync`` runs inline when already on the writer
    thread, so a decorated method calling another decorated method does not
    deadlock. Do NOT use this on methods that embed/compute heavily (split the
    compute off and submit only the handle op — e.g. ActivationStore
    .recompute_and_persist, SynthesisVectorStore.add_text)."""
    import functools

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Prefer `self._db_writer` when the store already uses `self._writer`
        # for something else (HypothesisForkStore / EngramStore hold the
        # bus-only OuterMemoryWriter on `self._writer`); else `self._writer`
        # IS the SynthesisWriter (ActivationStore, buffer/skill/vector stores).
        w = getattr(self, "_db_writer", None) or self._writer
        return w.submit_sync(lambda: method(self, *args, **kwargs))

    return wrapper


class _GuardedConn:
    """Thread-guard proxy over a DuckDB/Kuzu Connection.

    Delegates every attribute to the wrapped connection, but intercepts the
    mutating/execution entry points (``execute``/``executemany``) to assert the
    call is on the writer thread — raising :class:`WriterThreadViolation`
    otherwise. Result objects (``.fetchall()`` etc.) are returned as-is; they
    are consumed inside the same submitted closure, on the writer thread."""

    __slots__ = ("_writer", "_conn")

    def __init__(self, writer: SynthesisWriter, conn: Any) -> None:
        object.__setattr__(self, "_writer", writer)
        object.__setattr__(self, "_conn", conn)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self._writer._assert_on_thread("Connection.execute")
        return self._conn.execute(*args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        self._writer._assert_on_thread("Connection.executemany")
        return self._conn.executemany(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Passthrough for non-execution attributes (close, checkpoint, cursor,
        # commit helpers, etc.). DuckDB/Kuzu transactions in this codebase are
        # SQL-driven via .execute("BEGIN"/"COMMIT"/"ROLLBACK"), so the guard on
        # execute() already covers the transaction path.
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._conn, name, value)


def guard_conn(writer: SynthesisWriter, conn: Any) -> Any:
    """Wrap a DuckDB/Kuzu connection so any off-writer-thread ``execute`` raises
    WriterThreadViolation. Hand the guarded object to the stores; they may hold
    the reference freely — it can only be *invoked* inside a writer closure."""
    return _GuardedConn(writer, conn)
