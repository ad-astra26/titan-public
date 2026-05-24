"""Test ensure_started_async_safe handles guardian=None gracefully.

D-SPEC-123 follow-up (2026-05-23). Previously crashed with
`AttributeError: 'NoneType' object has no attribute 'is_running'` when
agno_worker_plugin constructed MemoryProxy(b, None). The crash blocked
the entire Phase 1 memory.query → _cognee_search → MEMORY_RETRIEVAL_USED
emit chain from agno chats.
"""
from __future__ import annotations

import pytest

from titan_hcl.proxies._start_safe import ensure_started_async_safe


def test_guardian_none_returns_true_no_crash():
    """guardian=None is the agno_worker_plugin pattern — must return
    True (optimistic 'assume started') and NOT raise AttributeError."""
    result = ensure_started_async_safe(
        guardian=None, module="memory", proxy_id=12345,
        proxy_label="MemoryProxy",
    )
    assert result is True


def test_real_guardian_with_is_running_true():
    """When guardian is a real object with is_running returning True,
    return True (module already running)."""
    class FakeGuardian:
        def is_running(self, module):
            return True

    result = ensure_started_async_safe(
        guardian=FakeGuardian(), module="memory", proxy_id=1,
        proxy_label="MemoryProxy",
    )
    assert result is True


def test_real_guardian_with_is_started_preferred():
    """getattr prefers is_started over is_running when both exist —
    keeps the existing API contract."""
    started_called = []

    class FakeGuardian:
        def is_started(self, module):
            started_called.append(module)
            return True

        def is_running(self, module):
            pytest.fail("is_running should not be called when is_started exists")

    result = ensure_started_async_safe(
        guardian=FakeGuardian(), module="memory", proxy_id=1,
        proxy_label="MemoryProxy",
    )
    assert result is True
    assert started_called == ["memory"]
