"""Chunk B.1 (RFP_meta-reasoning_CGN_FIX.md §4.1) — verify MetaService's
Session 3 live-dispatch infrastructure: asyncio loop thread, pending
response registry, _dispatch_to_resolver, handle_response correlation,
sweep_timeouts.

The async coroutine dispatch pattern is the SPEC-correct path per
Preamble G19 + §8.0.ter D-SPEC-48. Tests exercise the full
resolver-coroutine → publish → await Future → response → emit flow with
a mocked send_queue + mocked async resolver.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import pytest

from titan_hcl.logic.meta_recruitment import MetaRecruitment
from titan_hcl.logic.meta_service import (
    MetaService,
    QUESTION_TYPE_TO_PRIMITIVE,
    _PendingResponseRegistry,
    _default_sub_mode_for,
)


# ──────────────────────────────────────────────────────────────────────
# _PendingResponseRegistry — async future correlation cache
# ──────────────────────────────────────────────────────────────────────


def _make_loop_thread():
    """Spin up a daemon loop thread for registry tests. Caller must stop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    th = threading.Thread(target=_run, daemon=True, name="pytest-loop")
    th.start()
    ready.wait(timeout=2.0)
    return loop, th


def _stop_loop(loop, th):
    loop.call_soon_threadsafe(loop.stop)
    th.join(timeout=2.0)


def test_pending_registry_next_correlation_id_unique():
    cids = {_PendingResponseRegistry.next_correlation_id() for _ in range(100)}
    assert len(cids) == 100, "uuid4 collision (statistically impossible)"
    for cid in cids:
        assert isinstance(cid, str) and len(cid) == 32


def test_pending_registry_register_and_resolve_round_trip():
    loop, th = _make_loop_thread()
    try:
        reg = _PendingResponseRegistry(loop)
        # Register from inside the loop (resolver-coroutine context).
        async def _make_and_await(cid):
            fut = reg.register(cid, meta={"category": "test"})
            return await asyncio.wait_for(fut, timeout=2.0)

        cid = reg.next_correlation_id()
        fut_future = asyncio.run_coroutine_threadsafe(_make_and_await(cid), loop)
        # Wait briefly for the coroutine to call register()
        time.sleep(0.05)
        assert reg.size() == 1
        # Resolve from the main thread (simulates bus handler).
        assert reg.resolve(cid, {"output": "hello"}) is True
        # Coroutine should now complete with the resolved payload.
        result = fut_future.result(timeout=2.0)
        assert result == {"output": "hello"}
        # Registry empty after resolve.
        assert reg.size() == 0
    finally:
        _stop_loop(loop, th)


def test_pending_registry_resolve_unknown_returns_false():
    loop, th = _make_loop_thread()
    try:
        reg = _PendingResponseRegistry(loop)
        assert reg.resolve("nonexistent-cid", {"x": 1}) is False
    finally:
        _stop_loop(loop, th)


def test_pending_registry_sweep_timeouts():
    loop, th = _make_loop_thread()
    try:
        reg = _PendingResponseRegistry(loop)
        # Register two entries with controlled dispatch_ts.
        async def _register(cid):
            reg.register(cid, meta={"category": "test"})
        asyncio.run_coroutine_threadsafe(_register("old-cid"), loop).result(2.0)
        # Backdate so it's stale
        reg._meta["old-cid"]["dispatch_ts"] = time.time() - 10.0
        asyncio.run_coroutine_threadsafe(_register("fresh-cid"), loop).result(2.0)

        stale = reg.sweep_timeouts(timeout_s=5.0)
        assert stale == ["old-cid"]
    finally:
        _stop_loop(loop, th)


def test_pending_registry_discard():
    loop, th = _make_loop_thread()
    try:
        reg = _PendingResponseRegistry(loop)

        async def _register_and_check(cid):
            reg.register(cid)
            return reg.size()

        size = asyncio.run_coroutine_threadsafe(
            _register_and_check("x"), loop).result(2.0)
        assert size == 1
        reg.discard("x")
        assert reg.size() == 0
    finally:
        _stop_loop(loop, th)


# ──────────────────────────────────────────────────────────────────────
# QUESTION_TYPE_TO_PRIMITIVE + _default_sub_mode_for
# ──────────────────────────────────────────────────────────────────────


def test_question_type_to_primitive_covers_all_known():
    from titan_hcl.logic.meta_service_client import KNOWN_QUESTION_TYPES
    # Every KNOWN question_type must have a primitive mapping (else
    # _dispatch_to_resolver hits the dry-run fallback at runtime).
    missing = KNOWN_QUESTION_TYPES - set(QUESTION_TYPE_TO_PRIMITIVE.keys())
    assert not missing, f"question_types without primitive mapping: {missing}"


def test_default_sub_mode_for_returns_first_entry_per_primitive():
    from titan_hcl.logic.meta_reasoning import SUB_MODES
    for primitive, modes in SUB_MODES.items():
        assert _default_sub_mode_for(primitive) == modes[0]


def test_default_sub_mode_for_unknown_returns_none():
    assert _default_sub_mode_for("NOT_A_PRIMITIVE") is None


# ──────────────────────────────────────────────────────────────────────
# MetaService lifecycle — asyncio loop boot + close
# ──────────────────────────────────────────────────────────────────────


def test_meta_service_dispatch_loop_starts_and_closes_cleanly():
    ms = MetaService()
    try:
        assert ms._loop is not None
        assert ms._loop.is_running()
        assert ms._loop_thread.is_alive()
        assert ms._pending_registry is ms.pending_registry
        # New stats keys present
        for k in ("dispatches_scheduled", "dispatches_resolved",
                  "dispatches_timed_out", "responses_correlated"):
            assert k in ms._stats
    finally:
        ms.close()
    # After close, loop thread should have exited within the join window.
    assert not ms._loop_thread.is_alive()


def test_meta_service_close_is_idempotent():
    ms = MetaService()
    ms.close()
    ms.close()  # second call must not raise


# ──────────────────────────────────────────────────────────────────────
# End-to-end resolver-dispatch round trip with a mock async resolver
# ──────────────────────────────────────────────────────────────────────


class _MockResponseEmitter:
    """Collects emitted META_REASON_RESPONSE messages for assertion."""

    def __init__(self):
        self.responses: List[Dict[str, Any]] = []

    def __call__(self, msg: Dict[str, Any]) -> None:
        self.responses.append(msg)


def _make_async_resolver_that_uses_registry(
    pending_registry, published_list, dst="cognitive_worker",
    kind="reasoning", succeed=True,
):
    """Build a stand-in async resolver that mimics what the real Session 3
    resolvers do: publish (via append to list, no real bus), register a
    pending Future via the registry, await it, return formatted result.
    """

    async def _resolver(name: str, ctx: dict) -> dict:
        cid = pending_registry.next_correlation_id()
        fut = pending_registry.register(
            cid, meta={"category": kind, "name": name})
        published_list.append({
            "dst": dst, "kind": kind, "name": name, "correlation_id": cid,
            "consumer_id": ctx.get("consumer_id", ""),
        })
        try:
            response = await asyncio.wait_for(fut, timeout=3.0)
            if response.get("_timeout"):
                return {"success": False, "output": None,
                        "recruiter": f"{kind}.{name}",
                        "reason": "awaited future signalled timeout",
                        "failure_mode": "resolver_timeout"}
            if not succeed:
                return {"success": False, "output": None,
                        "recruiter": f"{kind}.{name}",
                        "reason": "mock configured to fail"}
            return {"success": True, "output": response,
                    "recruiter": f"{kind}.{name}",
                    "reason": "live_dispatch_test"}
        except asyncio.TimeoutError:
            return {"success": False, "output": None,
                    "recruiter": f"{kind}.{name}",
                    "reason": "asyncio.TimeoutError raised",
                    "failure_mode": "resolver_timeout"}

    return _resolver


def _build_request_msg(consumer_id="reasoning", question_type="formulate_strategy"):
    return {
        "src": consumer_id,
        "payload": {
            "consumer_id": consumer_id,
            "question_type": question_type,
            "request_id": uuid.uuid4().hex,
            "context_vector": [0.5] * 30,
            "time_budget_ms": 200,
        },
    }


def test_dispatch_to_resolver_round_trip_success():
    """Full flow: handle_request → dispatch → resolver publishes →
    handle_response correlates → coroutine completes → META_REASON_RESPONSE
    emitted with insight from the response payload."""
    emitter = _MockResponseEmitter()
    recruitment = MetaRecruitment()
    ms = MetaService(response_emitter=emitter, recruitment=recruitment)
    published: List[Dict[str, Any]] = []
    try:
        # FORMULATE.define has 3 candidate recruiters in the catalog
        # (reasoning.DECOMPOSE / language_reasoner.formulate_query /
        # pattern_primitives.extract_structure). Thompson selector picks
        # one — register ALL three categories so any pick succeeds.
        for cat in ("reasoning", "language_reasoner", "pattern_primitives"):
            resolver = _make_async_resolver_that_uses_registry(
                ms.pending_registry, published, kind=cat)
            recruitment.register_resolver(cat, resolver)

        msg = _build_request_msg(question_type="formulate_strategy")
        request_id = msg["payload"]["request_id"]

        # Fire request — handle_request returns immediately; dispatch is async
        result = ms.handle_request(msg)
        assert result is None

        # Wait for the resolver to publish (background loop is processing)
        deadline = time.time() + 2.0
        while not published and time.time() < deadline:
            time.sleep(0.01)
        assert published, "resolver did not publish within 2s"
        cid = published[0]["correlation_id"]

        # Simulate target worker's CGN_KNOWLEDGE_RESP arrival
        resp_msg = {
            "payload": {
                "correlation_id": cid,
                "result": "computed-answer",
                "confidence": 0.85,
            }
        }
        assert ms.handle_response(resp_msg) is True

        # Wait for META_REASON_RESPONSE emission
        deadline = time.time() + 2.0
        while not emitter.responses and time.time() < deadline:
            time.sleep(0.01)
        assert emitter.responses, "no META_REASON_RESPONSE emitted"
        resp = emitter.responses[0]
        assert resp["payload"]["request_id"] == request_id
        assert resp["payload"]["failure_mode"] is None
        assert resp["payload"]["insight"]["result"] == "computed-answer"

        # Stats reflect the success path
        assert ms._stats["dispatches_scheduled"] >= 1
        assert ms._stats["dispatches_resolved"] >= 1
        assert ms._stats["responses_correlated"] >= 1
    finally:
        ms.close()


def test_dispatch_resolver_timeout_emits_failure_mode():
    """If the response never arrives, sweep_timeouts() resolves the Future
    with a _timeout=True payload; the resolver returns failure_mode=
    resolver_timeout; META_REASON_RESPONSE is emitted accordingly."""
    emitter = _MockResponseEmitter()
    recruitment = MetaRecruitment()
    ms = MetaService(response_emitter=emitter, recruitment=recruitment)
    # Tight timeout so the test runs fast
    ms._dispatch_timeout_s = 0.1
    published: List[Dict[str, Any]] = []
    try:
        # Register all 3 candidate categories for FORMULATE.define so any
        # Thompson pick succeeds.
        for cat in ("reasoning", "language_reasoner", "pattern_primitives"):
            resolver = _make_async_resolver_that_uses_registry(
                ms.pending_registry, published, kind=cat)
            recruitment.register_resolver(cat, resolver)

        msg = _build_request_msg(question_type="formulate_strategy")
        ms.handle_request(msg)

        # Wait for the resolver to publish
        deadline = time.time() + 2.0
        while not published and time.time() < deadline:
            time.sleep(0.01)
        assert published

        # Don't send a response. Wait past timeout, then sweep.
        time.sleep(0.15)
        timed_out = ms.sweep_timeouts()
        assert timed_out >= 1

        # Wait for META_REASON_RESPONSE
        deadline = time.time() + 2.0
        while not emitter.responses and time.time() < deadline:
            time.sleep(0.01)
        assert emitter.responses
        resp = emitter.responses[0]
        assert resp["payload"]["failure_mode"] == "resolver_timeout"
        assert ms._stats["dispatches_timed_out"] >= 1
    finally:
        ms.close()


def test_dispatch_no_recruitment_falls_back_to_dry_run():
    """Without a recruitment layer wired, _dispatch_to_resolver falls back
    to _resolve_dry_run for Session 1 backward compatibility."""
    emitter = _MockResponseEmitter()
    ms = MetaService(response_emitter=emitter, recruitment=None)
    try:
        msg = _build_request_msg(question_type="formulate_strategy")
        ms.handle_request(msg)
        # Dry-run is synchronous — emission happens before handle_request returns
        assert emitter.responses
        assert emitter.responses[0]["payload"]["failure_mode"] == "not_yet_implemented"
        assert ms._stats["requests_dry_run_resolved"] >= 1
    finally:
        ms.close()


def test_dispatch_no_resolver_registered_emits_unavailable():
    """If recruitment has no resolver for the selected category, dispatch
    surfaces failure_mode=resolver_unavailable."""
    emitter = _MockResponseEmitter()
    recruitment = MetaRecruitment()  # no resolvers registered
    ms = MetaService(response_emitter=emitter, recruitment=recruitment)
    try:
        msg = _build_request_msg(question_type="formulate_strategy")
        ms.handle_request(msg)
        # handle_request returns immediately; dispatch synchronous fallback
        # path emits resolver_unavailable inline.
        deadline = time.time() + 1.0
        while not emitter.responses and time.time() < deadline:
            time.sleep(0.01)
        assert emitter.responses
        resp = emitter.responses[0]
        assert resp["payload"]["failure_mode"] == "resolver_unavailable"
        assert ms._stats["dispatches_resolver_unavailable"] >= 1
    finally:
        ms.close()


def test_dispatch_handle_response_unknown_correlation_id_drops():
    """handle_response with an unknown correlation_id increments
    responses_uncorrelated and returns False (no crash)."""
    emitter = _MockResponseEmitter()
    ms = MetaService(response_emitter=emitter)
    try:
        ok = ms.handle_response({"payload": {"correlation_id": "ghost-cid"}})
        assert ok is False
        assert ms._stats["responses_uncorrelated"] == 1
    finally:
        ms.close()


def test_dispatch_handle_response_no_correlation_id_returns_false():
    emitter = _MockResponseEmitter()
    ms = MetaService(response_emitter=emitter)
    try:
        ok = ms.handle_response({"payload": {"output": "no-cid-here"}})
        assert ok is False
    finally:
        ms.close()
