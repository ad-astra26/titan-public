"""RFP_load_adaptive_inference_routing §7.B0 — TRUE concurrent chat dispatch.

Two coupled changes, tested here against the REAL code (no agno/TitanHCL stack):

  • B0-state — the per-turn WorkerPlugin fields are backed by a ContextVar bag
    (`agno_worker_plugin._RequestScopedAttr` + `enter_request_scope`). Concurrent
    chats interleave across `await arun()` on ONE asyncio loop; without scoping,
    chat B's `_current_user_id` would overwrite chat A's between A's PreHook and
    A's PostHook → every synthesis record cross-contaminated. The load-bearing
    test asserts NO bleed AND that the harness actually DETECTS bleed when the
    scope is removed (so a green result is meaningful, not vacuous).

  • B0-dispatch — `_ChatLoopDispatcher` runs the loop forever in a daemon thread
    and schedules chats via `run_coroutine_threadsafe` (non-blocking) → `arun()`s
    overlap (wall ≈ max, not sum). Exceptions in a scheduled chat task are
    surfaced via the done-callback (INV-CC-ERRORS-SURFACE), never swallowed.

Invariants: INV-CC-CONCURRENT, INV-CC-NO-BLEED, INV-CC-BOUNDED,
INV-CC-ERRORS-SURFACE, INV-CC-ORDERING.

Run: python -m pytest tests/test_b0_concurrent_dispatch.py -v -p no:anchorpy
"""
import asyncio
import contextvars
import logging
import time

import pytest

from titan_hcl.modules.agno_worker import _ChatLoopDispatcher
from titan_hcl.modules.agno_worker_plugin import (
    WorkerPlugin,
    enter_request_scope,
    _REQUEST_SCOPED_FIELDS,
)


# ════════════════════════════════════════════════════════════════════════
# B0-state — no cross-contamination (the load-bearing gate)
# ════════════════════════════════════════════════════════════════════════

# The exact per-turn fields a chat sets then reads back a turn later. Chosen to
# span the three classes the bug corrupts: identity (user_id/is_maker/did_hash),
# the reasoning record id, and a presence/telemetry field.
def _turn_inputs(i: int) -> dict:
    return {
        "_current_user_id": f"user_{i}",
        "_current_session_id": f"sess_{i}",
        "_current_channel": "web" if i % 2 else "app",
        "_current_is_maker": (i == 0),
        "_current_did_hash": f"did_{i}",
        "_last_reasoning_id": f"rid_{i}",
        "_telemetry_trigger_id": f"trig_{i}",
        "_pre_chat_user_id": f"user_{i}",
    }


async def _fake_turn(plugin, i: int, scoped: bool, started: asyncio.Event,
                     barrier_n: int, arrived: list, release: asyncio.Event):
    """Mimic one chat: (PreHook) install scope + set per-turn fields → (await
    arun, where OTHER turns run + overwrite the shared plugin) → (PostHook) read
    the fields back. Returns what THIS turn observed."""
    if scoped:
        enter_request_scope()
    inputs = _turn_inputs(i)
    for k, v in inputs.items():
        setattr(plugin, k, v)

    # Force maximal interleaving: every turn parks here until ALL turns have set
    # their fields, so a non-scoped run is GUARANTEED to read a clobbered value.
    arrived.append(i)
    if len(arrived) >= barrier_n:
        release.set()
    await release.wait()
    await asyncio.sleep(0)  # extra yield — let the scheduler interleave reads

    observed = {k: getattr(plugin, k) for k in inputs}
    return i, inputs, observed


async def _run_turns(plugin, n: int, scoped: bool):
    release = asyncio.Event()
    arrived: list = []
    started = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _fake_turn(plugin, i, scoped, started, n, arrived, release))
        for i in range(n)
    ]
    return await asyncio.gather(*tasks)


def test_no_cross_contamination_scoped():
    """LOAD-BEARING: N concurrent distinct-user turns, each reads back ITS OWN
    per-turn fields after a fully-interleaved await — zero bleed (INV-CC-NO-BLEED)."""
    plugin = WorkerPlugin(bus_client=None, config={})
    results = asyncio.run(_run_turns(plugin, n=24, scoped=True))
    for i, inputs, observed in results:
        assert observed == inputs, (
            f"turn {i} cross-contaminated: set {inputs} but read {observed}")


def test_harness_detects_bleed_without_scope():
    """Control: the SAME interleaving WITHOUT enter_request_scope DOES corrupt
    the fields — proves the load-bearing test isn't vacuously green."""
    plugin = WorkerPlugin(bus_client=None, config={})
    results = asyncio.run(_run_turns(plugin, n=24, scoped=False))
    bled = [i for i, inputs, observed in results if observed != inputs]
    # With a barrier forcing all writes before any read, every turn except the
    # last writer reads the shared (last) value → overwhelming contamination.
    assert bled, ("expected cross-contamination without request scope — the "
                  "harness cannot prove isolation if it can't detect the bug")


def test_every_declared_field_is_isolated():
    """Each of the 38 declared request-scoped fields is genuinely task-local —
    set distinct values in two contexts, assert no leak across them (G-STUB
    companion: the descriptor set must actually cover every declared name)."""
    plugin = WorkerPlugin(bus_client=None, config={})

    def writer(tag):
        enter_request_scope()
        for f in _REQUEST_SCOPED_FIELDS:
            setattr(plugin, f, f"{f}:{tag}")
        return {f: getattr(plugin, f) for f in _REQUEST_SCOPED_FIELDS}

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()
    seen_a = ctx_a.run(writer, "A")
    seen_b = ctx_b.run(writer, "B")
    # re-read inside each context AFTER the other has written
    reread_a = ctx_a.run(lambda: {f: getattr(plugin, f)
                                   for f in _REQUEST_SCOPED_FIELDS})
    for f in _REQUEST_SCOPED_FIELDS:
        assert reread_a[f] == f"{f}:A", f"{f} bled B→A: {reread_a[f]!r}"
        assert seen_b[f] == f"{f}:B"


# ════════════════════════════════════════════════════════════════════════
# B0-dispatch — overlap, bounded, errors-surface, ordering
# ════════════════════════════════════════════════════════════════════════

def _fresh_dispatcher():
    loop = asyncio.new_event_loop()
    d = _ChatLoopDispatcher(loop)
    d.start()
    return loop, d


def test_concurrent_overlap_wall_approx_max():
    """INV-CC-CONCURRENT: 5 stub chats each sleeping 0.20s, scheduled (not
    run_until_complete'd), overlap → wall ≈ 0.20s, NOT 1.0s."""
    loop, d = _fresh_dispatcher()
    try:
        SLEEP, N = 0.20, 5

        async def stub_chat(i):
            await asyncio.sleep(SLEEP)
            return i

        t0 = time.time()
        futs = [d.schedule(stub_chat(i)) for i in range(N)]
        results = sorted(f.result(timeout=5) for f in futs)
        wall = time.time() - t0

        assert results == list(range(N))
        assert wall < SLEEP * (N / 2), (
            f"chats did not overlap: wall={wall:.3f}s for {N}x{SLEEP}s "
            f"(serial would be {SLEEP*N:.3f}s)")
    finally:
        d.stop()
        loop.close()


def test_bounded_by_semaphore():
    """INV-CC-BOUNDED: a Semaphore(k) around arun caps simultaneous in-flight
    chats at k even when 4k are scheduled at once."""
    loop, d = _fresh_dispatcher()
    try:
        K = 3
        sem = asyncio.Semaphore(K)
        live = {"now": 0, "peak": 0}

        async def stub_chat():
            async with sem:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
                await asyncio.sleep(0.05)
                live["now"] -= 1

        futs = [d.schedule(stub_chat()) for _ in range(K * 4)]
        for f in futs:
            f.result(timeout=5)
        assert live["peak"] <= K, f"peak {live['peak']} exceeded cap {K}"
        assert live["peak"] >= 2, "semaphore serialized everything — not concurrent"
    finally:
        d.stop()
        loop.close()


def test_chat_task_exception_is_surfaced(caplog):
    """INV-CC-ERRORS-SURFACE: a chat task that raises is LOGGED (not swallowed)
    and dropped from the in-flight set."""
    loop, d = _fresh_dispatcher()
    try:
        async def boom():
            raise RuntimeError("synthetic chat failure")

        with caplog.at_level(logging.ERROR):
            fut = d.schedule(boom())
            with pytest.raises(RuntimeError):
                fut.result(timeout=5)
            # let the done-callback run on the loop thread
            deadline = time.time() + 2.0
            while d.in_flight_count and time.time() < deadline:
                time.sleep(0.01)

        assert d.in_flight_count == 0, "crashed future not dropped from set"
        assert any("chat task crashed" in r.getMessage() for r in caplog.records), (
            "chat-task exception was not surfaced to the log")
    finally:
        d.stop()
        loop.close()


def test_nonchat_blocking_result_preserves_ordering():
    """INV-CC-ORDERING: a non-chat op the recv thread WAITS on (.result()) runs
    to completion before the recv thread proceeds — the dream-replay pattern."""
    loop, d = _fresh_dispatcher()
    try:
        order = []

        async def maintenance():
            await asyncio.sleep(0.05)
            order.append("maintenance_done")

        d.schedule(maintenance()).result(timeout=5)   # blocks, like dream replay
        order.append("recv_continued")
        assert order == ["maintenance_done", "recv_continued"]
    finally:
        d.stop()
        loop.close()
