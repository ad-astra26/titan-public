"""Phase 1 (2026-05-31) — verify the backup cascade runs OFF the recv-loop thread.

Root cause being guarded: the unified_v2 cascade used to run via
loop.run_until_complete(...) synchronously on the recv-loop thread, blocking the
worker from reading its bus socket for the multi-minute diff/upload → broker
silent-hang-defense BrokenPipe → Guardian shm_pid_dead → restart loop. These
tests pin the contract of `_dispatch_backup_offloop`:
  1. it returns to the caller IMMEDIATELY (does not block on the handler),
  2. the handler actually runs (in a separate thread), and
  3. single-flight: a second dispatch while one is in flight is SKIPPED.
"""
from __future__ import annotations

import asyncio
import threading
import time

from titan_hcl.modules import backup_worker


def _fresh_state():
    return {
        "_backup_lock": threading.Lock(),
        "loop": asyncio.new_event_loop(),
    }


def test_dispatch_returns_immediately_and_runs_handler_in_another_thread():
    state = _fresh_state()
    caller_thread = threading.current_thread().ident
    ran = {"thread": None, "done": False}
    started = threading.Event()

    def handler(_state, _msg):
        ran["thread"] = threading.current_thread().ident
        started.set()
        time.sleep(0.4)          # simulate the long diff/upload
        ran["done"] = True

    t0 = time.time()
    backup_worker._dispatch_backup_offloop(state, handler, {"type": "X"})
    elapsed = time.time() - t0

    # Caller is NOT blocked by the 0.4s handler — the whole point of Phase 1.
    assert elapsed < 0.2, f"dispatch blocked the caller for {elapsed:.2f}s"
    assert started.wait(timeout=2.0), "handler never started"
    # Handler ran on a DIFFERENT (daemon) thread, not the caller's.
    assert ran["thread"] is not None and ran["thread"] != caller_thread
    # And it does finish.
    time.sleep(0.6)
    assert ran["done"] is True
    state["loop"].close()


def test_single_flight_skips_overlapping_dispatch():
    state = _fresh_state()
    runs = {"n": 0}
    gate = threading.Event()

    def slow_handler(_state, _msg):
        runs["n"] += 1
        gate.wait(timeout=2.0)   # hold the lock until released

    # First dispatch acquires the lock + blocks in the handler.
    backup_worker._dispatch_backup_offloop(state, slow_handler, {"type": "A"})
    time.sleep(0.1)              # let the first thread acquire + enter handler
    # Second dispatch while the first is in flight → must be SKIPPED.
    backup_worker._dispatch_backup_offloop(state, slow_handler, {"type": "B"})
    time.sleep(0.1)
    assert runs["n"] == 1, "single-flight failed: overlapping backup ran"

    gate.set()                   # release the first handler
    time.sleep(0.3)
    # Lock released → a fresh dispatch runs again.
    done = threading.Event()

    def quick(_s, _m):
        runs["n"] += 1
        done.set()

    backup_worker._dispatch_backup_offloop(state, quick, {"type": "C"})
    assert done.wait(timeout=2.0)
    assert runs["n"] == 2
    state["loop"].close()


def test_handler_exception_releases_the_lock():
    """A failing cascade must not wedge the single-flight lock forever."""
    state = _fresh_state()
    done = threading.Event()

    def boom(_s, _m):
        raise RuntimeError("simulated cascade failure")

    backup_worker._dispatch_backup_offloop(state, boom, {"type": "A"})
    time.sleep(0.3)
    # Lock must be free again despite the exception.
    assert state["_backup_lock"].acquire(blocking=False), "lock leaked after exception"
    state["_backup_lock"].release()

    def ok(_s, _m):
        done.set()
    backup_worker._dispatch_backup_offloop(state, ok, {"type": "B"})
    assert done.wait(timeout=2.0)
    state["loop"].close()
