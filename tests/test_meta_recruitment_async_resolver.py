"""Step 1 (RFP_meta-reasoning_CGN_FIX.md Chunk A) — verify MetaRecruitment.
register_resolver accepts both sync and async callables.

Async is preferred per SPEC Preamble G19+G22 + §8.0.ter D-SPEC-48 and per
RFP_meta-reasoning_CGN_FIX.md §4.1. Sync remains allowed for the existing 10
Session 2 shell resolvers and any explicitly documented exception.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import pytest

from titan_hcl.logic.meta_recruitment import (
    MetaRecruitment,
    ResolverCallable,  # type alias — verifies the type is exported
)


# ──────────────────────────────────────────────────────────────────────


def test_register_resolver_accepts_sync_callable():
    """Existing Session 2 sync-resolver registration still works."""
    mr = MetaRecruitment()

    def sync_resolver(name: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {"success": True, "output": {"name": name, "kind": "sync"}}

    mr.register_resolver("test_sync", sync_resolver)
    assert mr._resolvers["test_sync"] is sync_resolver


def test_register_resolver_accepts_async_callable():
    """Session 3 (PART A) async-resolver registration works."""
    mr = MetaRecruitment()

    async def async_resolver(name: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {"success": True, "output": {"name": name, "kind": "async"}}

    mr.register_resolver("test_async", async_resolver)
    assert mr._resolvers["test_async"] is async_resolver


def test_register_resolver_logs_sync_vs_async(caplog):
    """Boot log line distinguishes sync vs async — eases live debugging."""
    mr = MetaRecruitment()

    def sync_resolver(name, ctx):
        return None

    async def async_resolver(name, ctx):
        return None

    with caplog.at_level(logging.INFO, logger="titan_hcl.logic.meta_recruitment"):
        mr.register_resolver("test_sync_log", sync_resolver)
        mr.register_resolver("test_async_log", async_resolver)

    sync_lines = [r for r in caplog.records if "test_sync_log" in r.getMessage()]
    async_lines = [r for r in caplog.records if "test_async_log" in r.getMessage()]
    assert sync_lines, "sync resolver registration should log"
    assert async_lines, "async resolver registration should log"
    assert "sync" in sync_lines[0].getMessage()
    assert "async" in async_lines[0].getMessage()


def test_register_resolver_rejects_non_callable():
    """Non-callable resolver_fn still raises ValueError."""
    mr = MetaRecruitment()
    with pytest.raises(ValueError, match="must be callable"):
        mr.register_resolver("bad", "not_a_function")  # type: ignore[arg-type]


def test_async_resolver_round_trip_via_asyncio_run():
    """Async resolver can be awaited and returns the expected dict.

    Verifies the Chunk B dispatcher pattern: store the async callable,
    later invoke it via `asyncio.run(resolver(name, ctx))`. The MetaService
    dispatcher will use the same pattern with an inspect.iscoroutine check.
    """
    mr = MetaRecruitment()

    async def async_resolver(name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)  # yield once to confirm we're really async
        return {"success": True, "output": {"name": name, "ctx_size": len(ctx)}}

    mr.register_resolver("rt", async_resolver)
    fn = mr._resolvers["rt"]
    result = asyncio.run(fn("HELLO", {"k": 1, "v": 2}))
    assert result == {"success": True, "output": {"name": "HELLO", "ctx_size": 2}}


def test_resolver_callable_type_alias_exists():
    """`ResolverCallable` type alias is exported for downstream consumers
    (e.g., meta_resolvers.py factories will type-annotate against it).
    """
    # Both sync and async signatures should be assignable to ResolverCallable
    sync_fn: ResolverCallable = lambda name, ctx: {"success": True, "output": {}}

    async def async_fn(name, ctx):
        return {"success": True, "output": {}}

    async_fn_typed: ResolverCallable = async_fn  # noqa: F841 — type-check only
    # No runtime assertion beyond the type-check successfully importing
    assert callable(sync_fn)
