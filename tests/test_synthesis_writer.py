"""Unit tests for the single-writer-thread persistence discipline (Option C).

Covers the SynthesisWriter / InlineWriter / guard_conn / @on_writer primitives
that close the synthesis crash-loop (AUDIT_synthesis_engine_crashloop_
concurrency_20260602.md): all DuckDB/Kuzu/FAISS handle ops serialize on one
writer thread, so concurrent .execute() (the SIGSEGV cause) is impossible by
construction, and the guard makes a forgotten submit() fail loudly.
"""
import threading
import time

import pytest

from titan_hcl.synthesis.writer import (
    InlineWriter,
    SynthesisWriter,
    WriterThreadViolation,
    guard_conn,
    on_writer,
    resolve_writer,
)


@pytest.fixture
def writer():
    w = SynthesisWriter("test").start()
    yield w
    w.close()


def test_submit_sync_returns_result(writer):
    assert writer.submit_sync(lambda: 21 * 2) == 42


def test_submit_sync_reraises_exception(writer):
    def boom():
        raise ValueError("kaboom")
    with pytest.raises(ValueError, match="kaboom"):
        writer.submit_sync(boom)
    # writer survives a failed op
    assert writer.submit_sync(lambda: "alive") == "alive"


def test_fire_and_forget_then_flush(writer):
    out = []
    for i in range(50):
        writer.submit(lambda i=i: out.append(i))
    writer.flush()
    assert out == list(range(50))  # serial → FIFO order preserved


def test_ops_never_run_concurrently(writer):
    """The core guarantee: a single writer thread → no two ops overlap."""
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def op():
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.002)
        with lock:
            active["n"] -= 1

    barrier = threading.Barrier(8)

    def submitter():
        barrier.wait()
        for _ in range(20):
            writer.submit(op)

    threads = [threading.Thread(target=submitter) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.flush()
    assert active["max"] == 1, f"ops overlapped (max concurrency {active['max']})"


def test_guard_conn_off_thread_raises(writer):
    class _FakeConn:
        def execute(self, *a, **k):
            return "ran"
    gc = guard_conn(writer, _FakeConn())
    # On a non-writer thread → raises.
    with pytest.raises(WriterThreadViolation):
        gc.execute("SELECT 1")
    # Routed through the writer → passes.
    assert writer.submit_sync(lambda: gc.execute("SELECT 1")) == "ran"


def test_guard_conn_passthrough_non_execute(writer):
    class _FakeConn:
        attr = "x"

        def close(self):
            return "closed"
    gc = guard_conn(writer, _FakeConn())
    # Non-execute attributes pass through without the guard.
    assert gc.attr == "x"
    assert gc.close() == "closed"


def test_inline_writer_runs_inline():
    w = InlineWriter()
    assert w.on_writer_thread() is True
    assert w.submit_sync(lambda: 7) == 7
    seen = []
    w.submit(lambda: seen.append(1))
    assert seen == [1]
    w.flush()
    w.close()


def test_resolve_writer_defaults_to_inline():
    assert isinstance(resolve_writer(None), InlineWriter)
    w = SynthesisWriter("x").start()
    assert resolve_writer(w) is w
    w.close()


def test_on_writer_decorator_routes_and_is_reentrant(writer):
    """A @on_writer method runs on the writer thread; a decorated method
    calling another decorated method runs inline (no deadlock)."""
    class Store:
        def __init__(self, w):
            self._writer = w
            self.thread_ids = []

        @on_writer
        def outer(self):
            self.thread_ids.append(threading.get_ident())
            return self.inner() + 1  # nested decorated call

        @on_writer
        def inner(self):
            self.thread_ids.append(threading.get_ident())
            return 10

    s = Store(writer)
    assert s.outer() == 11
    # Both ran on the SAME (writer) thread, and NOT the caller thread.
    assert len(set(s.thread_ids)) == 1
    assert s.thread_ids[0] != threading.get_ident()


def test_on_writer_prefers_db_writer_attr(writer):
    """When a store already uses self._writer for something else, the decorator
    falls back to self._db_writer (HypothesisForkStore / EngramStore case)."""
    class Store:
        def __init__(self, w):
            self._writer = object()       # NOT a writer (e.g. OuterMemoryWriter)
            self._db_writer = w

        @on_writer
        def op(self):
            return threading.get_ident()

    s = Store(writer)
    assert s.op() != threading.get_ident()  # ran on the writer thread


def test_close_drains_then_rejects(writer):
    out = []
    writer.submit(lambda: out.append("a"))
    writer.close()           # drains 'a' then stops
    assert out == ["a"]
    with pytest.raises(RuntimeError):
        writer.submit(lambda: out.append("b"))
