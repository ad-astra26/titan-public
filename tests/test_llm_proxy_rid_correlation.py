"""Regression test for the 2026-05-23 LLMProxy rid-correlation bug.

Before the fix, `LLMProxy._await_response` only filtered responses by
`message_type` — not by `rid`. When two distill calls were in flight on
the shared `_reply_queue` (e.g. in-kernel `social_x_composer.distill()`
+ HTTP `/v4/llm-distill` from `events_teacher`), the first matching-type
response in the queue was returned to ANY waiter regardless of which
request it actually was for.

This caused 7 X posts on T2/T3 (2026-05-23) to leak `events_teacher`'s
verbatim JSON distillation as the tweet body. The 4 affected post_types
were `world_mirror`, `bilingual`, `outer_rumination`, `practiced_response`,
`vulnerability` × 2 — proof the bug was at the LLM-proxy layer, not the
archetype layer (every caller of the shared proxy was vulnerable).

SPEC anchors for the canonical correlation-id pattern:
  - §8.2 D-SPEC-65 (correlation_id-keyed cache for multi-request demux)
  - bus.py:1446-1471 (DivineBus.request: filter by type AND rid, put back
    non-matching responses)

This test simulates the racing scenario: two distill awaiters concurrently
poll the same reply_queue while responses for BOTH arrive in arbitrary
order. The test asserts each awaiter receives the response carrying its
own rid — never the other's.
"""
from __future__ import annotations

import asyncio
import queue
import uuid

import pytest

from titan_hcl.bus import LLM_DISTILL_RESPONSE
from titan_hcl.proxies.llm_proxy import LLMProxy


class _FakeBus:
    """Minimal DivineBus stand-in that hands the proxy a shared Queue."""

    def __init__(self):
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=64)

    def subscribe(self, module_name, reply_only=False):
        return self._q

    def publish(self, msg):  # noop — test injects responses directly
        pass

    def inject_response(self, rid, result_text):
        """Drop an LLM_DISTILL_RESPONSE into the shared reply queue."""
        self._q.put({
            "type": LLM_DISTILL_RESPONSE,
            "src": "llm",
            "dst": "llm_proxy",
            "ts": 0.0,
            "rid": rid,
            "payload": {"result": result_text, "model": "test",
                        "elapsed_ms": 1.0, "ovg": None},
        })


class _FakeGuardian:
    def start(self, *args, **kwargs):
        pass


def _make_proxy():
    return LLMProxy(_FakeBus(), _FakeGuardian())


@pytest.mark.asyncio
async def test_await_response_filters_by_rid_not_just_type():
    """When two responses of the same type land in the shared reply queue
    with different rids, _await_response must hand each caller its OWN
    response — not the first one it sees."""
    proxy = _make_proxy()

    rid_a = uuid.uuid4().hex
    rid_b = uuid.uuid4().hex
    proxy._bus.inject_response(rid=rid_b, result_text="B-response-for-events_teacher-shape")
    proxy._bus.inject_response(rid=rid_a, result_text="A-response-for-social_x-prose")

    result_a = await proxy._await_response(
        LLM_DISTILL_RESPONSE, timeout=3.0, result_key="result",
        expected_rid=rid_a)
    result_b = await proxy._await_response(
        LLM_DISTILL_RESPONSE, timeout=3.0, result_key="result",
        expected_rid=rid_b)

    assert result_a == "A-response-for-social_x-prose"
    assert result_b == "B-response-for-events_teacher-shape"


@pytest.mark.asyncio
async def test_await_response_puts_back_non_matching_rid():
    """A reply with the right type but wrong rid must be put back into the
    queue so the rightful waiter can find it later. This mirrors
    DivineBus.request at bus.py:1467-1468 — the SPEC-canonical pattern."""
    proxy = _make_proxy()

    rid_a = uuid.uuid4().hex
    rid_b = uuid.uuid4().hex

    # Only B's response is in the queue; A's waiter must put it back and
    # ultimately time out instead of stealing B's payload.
    proxy._bus.inject_response(rid=rid_b, result_text="B-payload")

    result_a = await proxy._await_response(
        LLM_DISTILL_RESPONSE, timeout=1.0, result_key="result",
        expected_rid=rid_a)
    # A should NOT have received B's payload, just an empty (timeout) string.
    assert result_a == ""

    # B's response must still be in the queue, claimable by B's waiter.
    result_b = await proxy._await_response(
        LLM_DISTILL_RESPONSE, timeout=1.0, result_key="result",
        expected_rid=rid_b)
    assert result_b == "B-payload"


@pytest.mark.asyncio
async def test_await_response_legacy_unfiltered_path_still_works():
    """With `expected_rid=None`, the legacy type-only path remains. This
    keeps a graceful fallback for callers that haven't migrated yet (and
    documents that omitting rid is unsafe under concurrency — see the
    bug context above)."""
    proxy = _make_proxy()
    proxy._bus.inject_response(rid="some-rid", result_text="legacy-grab")

    result = await proxy._await_response(
        LLM_DISTILL_RESPONSE, timeout=1.0, result_key="result")
    # Without rid filter, first matching type wins (pre-fix behavior).
    assert result == "legacy-grab"
