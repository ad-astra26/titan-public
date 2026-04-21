"""Tests for rFP_social_graph_async_safety Option A+ (async companions + lock).

Covers rFP §6.1:
 - test_async_and_sync_coexist: same instance used by both paths returns consistent data.
 - test_concurrent_record_interaction_races: 50 concurrent async record_interaction
   for the same user; assert interaction_count == 50 (race detector).
 - test_read_lock_free: a concurrent async get_stats completes while a writer holds
   the lock (reads don't wait on writers).
 - test_subprocess_sync_caller_unaffected: direct sync ledger_record still works
   when the async API is also in use.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from titan_plugin.core.social_graph import SocialGraph


@pytest.fixture
def sg():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        graph = SocialGraph(db_path=tmp.name)
        yield graph
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass


def test_async_and_sync_coexist(sg):
    """Sync write followed by async read returns the same profile."""
    async def _run():
        sg.get_or_create_user("alice", platform="x", display_name="Alice")
        sg.record_interaction("alice", quality=0.8)
        profile = await sg.get_or_create_user_async("alice")
        assert profile.user_id == "alice"
        assert profile.interaction_count == 1
        stats = await sg.get_stats_async()
        assert stats["users"] == 1

    asyncio.run(_run())


def test_concurrent_record_interaction_races(sg):
    """50 concurrent async record_interaction calls against the same user must
    all be serialized; final interaction_count must equal 50."""
    async def _run():
        sg.get_or_create_user("bob", platform="x")
        await asyncio.gather(*[
            sg.record_interaction_async("bob", quality=0.9)
            for _ in range(50)
        ])
        profile = await sg.get_or_create_user_async("bob")
        assert profile.interaction_count == 50, (
            f"Expected 50, got {profile.interaction_count} — race condition")

    asyncio.run(_run())


def test_read_lock_free(sg):
    """A long-running writer must not block concurrent readers (WAL lets
    readers proceed; the async companion lock is writer-only)."""
    async def _run():
        sg.get_or_create_user("carol", platform="x")

        writer_entered = asyncio.Event()
        writer_release = asyncio.Event()

        async def slow_writer():
            # Take the lock and hold it explicitly to simulate a long writer.
            async with sg._get_write_lock():
                writer_entered.set()
                await writer_release.wait()
                await asyncio.to_thread(sg.record_interaction, "carol", 0.7)

        writer_task = asyncio.create_task(slow_writer())
        await writer_entered.wait()

        t0 = time.monotonic()
        stats = await sg.get_stats_async()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        assert stats["users"] == 1
        # Reader should not be blocked by writer; allow 100ms slack for to_thread.
        assert elapsed_ms < 100.0, (
            f"Reader waited {elapsed_ms:.1f} ms while writer held lock — "
            "read path is not lock-free")

        writer_release.set()
        await writer_task

    asyncio.run(_run())


def test_subprocess_sync_caller_unaffected(sg):
    """A sync caller (simulating subprocess/cron context) can still write via
    the sync API while async callers are also using the class."""
    async def _run():
        # Async caller establishes user.
        await sg.get_or_create_user_async("dave", platform="x")
        # Sync caller writes ledger (mimics mind_worker / consciousness tick).
        sg.ledger_record("tweet_1", "dave", "reply", mention_text="hello")
        # Async read sees sync write.
        count = await sg.ledger_total_today_async("reply")
        assert count == 1

    asyncio.run(_run())
