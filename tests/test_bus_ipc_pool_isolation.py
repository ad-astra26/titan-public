"""
2026-04-29 — Bus IPC pool isolation regression tests.

After T1 thread-pool saturation hit 250% (64 busy + 192 queued) under
Observatory poll bursts, bus.request reply waits got queued behind heavy
work and timed out, cascading into BUG-DIVINEBUS-SPIRIT-PROXY-TIMEOUTS +
BUG-DASHBOARD-BUS-ATTR-ERRORS + Observatory data-loading issues.

Fix: dedicated bus_ipc_pool (8 workers) for `bus.request_async()` reply
waits. These tests lock the architectural invariant: bus IPC must NOT
share the default asyncio pool with Observatory snapshot work.

Three layers of enforcement:
  1. Existence — get_bus_ipc_pool() returns a dedicated ThreadPoolExecutor
  2. Routing — DivineBus.request_async() runs the reply wait on bus-ipc
  3. Pattern hygiene — async proxies use request_async, never to_thread(request)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import re
import threading
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestBusIpcPoolExists(unittest.TestCase):
    """Layer 1 — the dedicated pool must exist and be configurable."""

    def test_get_bus_ipc_pool_returns_thread_pool(self):
        from titan_plugin.bus import get_bus_ipc_pool
        pool = get_bus_ipc_pool()
        self.assertIsInstance(pool, concurrent.futures.ThreadPoolExecutor)

    def test_pool_is_singleton(self):
        from titan_plugin.bus import get_bus_ipc_pool
        a = get_bus_ipc_pool()
        b = get_bus_ipc_pool()
        self.assertIs(a, b, "get_bus_ipc_pool must return the same pool every call")

    def test_pool_size_matches_default_or_config(self):
        from titan_plugin.bus import get_bus_ipc_pool, _BUS_IPC_DEFAULT_WORKERS
        pool = get_bus_ipc_pool()
        # _max_workers is a public-enough internal of stdlib ThreadPoolExecutor
        self.assertGreaterEqual(pool._max_workers, 1)
        # If config wasn't loaded, expect the default
        # (config layer can override; we just assert sane bounds)
        self.assertLessEqual(pool._max_workers, 64,
                             "bus_ipc pool sized larger than 64 defeats the "
                             "isolation purpose (default pool is 64)")

    def test_pool_threads_named_distinctly(self):
        """Threads must have a recognizable name prefix so /v4/thread-pool
        and py-spy can distinguish bus-ipc workers from default workers."""
        from titan_plugin.bus import get_bus_ipc_pool
        pool = get_bus_ipc_pool()
        # Submit a no-op so a thread spawns
        pool.submit(lambda: None).result(timeout=2.0)
        # Inspect any thread that picked up the work
        names = [t.name for t in threading.enumerate()
                 if t.name.startswith("bus-ipc")]
        self.assertGreater(len(names), 0,
                           "bus_ipc pool must spawn threads named bus-ipc-N "
                           "(found none in threading.enumerate())")


class TestBusRequestAsyncRoutesThroughPool(unittest.TestCase):
    """Layer 2 — request_async() must use bus_ipc_pool, not default."""

    def test_request_async_method_exists(self):
        from titan_plugin.bus import DivineBus
        self.assertTrue(hasattr(DivineBus, "request_async"))
        self.assertTrue(asyncio.iscoroutinefunction(DivineBus.request_async))

    def test_request_async_calls_get_bus_ipc_pool(self):
        """Source-inspection: request_async body references
        get_bus_ipc_pool. The runtime check (that the wait actually
        happens on a bus-ipc thread) is in
        test_request_async_runs_on_bus_ipc_thread below."""
        from titan_plugin.bus import DivineBus
        src = inspect.getsource(DivineBus.request_async)
        self.assertIn("get_bus_ipc_pool", src,
                      "request_async must route through get_bus_ipc_pool")
        # Negative: must NOT use the default pool (run_in_executor(None,...))
        self.assertNotRegex(src, r"run_in_executor\s*\(\s*None\s*,",
                            "request_async must not pass None as executor "
                            "(would route through default pool, defeating "
                            "isolation)")

    def test_request_async_runs_on_bus_ipc_thread(self):
        """Runtime: when await request_async, the inner sync request()
        executes on a thread named bus-ipc-N. Without a real bus this
        test stubs request() to capture its execution thread name."""
        from titan_plugin import bus as bus_mod
        from titan_plugin.bus import DivineBus

        captured = {}

        def fake_request(self, src, dst, payload, timeout, reply_queue):
            captured["thread_name"] = threading.current_thread().name
            return None

        async def runner():
            db = DivineBus.__new__(DivineBus)
            # request_async is a method on DivineBus; bind our fake to it
            return await DivineBus.request_async(
                db, "test_src", "test_dst", {}, 1.0, None
            )

        # Patch DivineBus.request for the duration of this test
        original = DivineBus.request
        DivineBus.request = fake_request
        try:
            asyncio.run(runner())
        finally:
            DivineBus.request = original

        self.assertIn("thread_name", captured,
                      "fake request was never called")
        self.assertTrue(captured["thread_name"].startswith("bus-ipc"),
                        f"request ran on wrong thread: "
                        f"{captured['thread_name']!r} — expected bus-ipc-N")


class TestProxyHygiene(unittest.TestCase):
    """Layer 3 — async proxy paths must use request_async, not to_thread(request)."""

    def _read(self, path):
        return (REPO_ROOT / path).read_text()

    def test_agency_proxy_uses_request_async(self):
        src = self._read("titan_plugin/proxies/agency_proxy.py")
        self.assertIn("self._bus.request_async(", src)
        self.assertNotRegex(
            src, r"asyncio\.to_thread\(\s*self\._bus\.request\b",
            "agency_proxy still uses to_thread(self._bus.request) — "
            "must migrate to await self._bus.request_async(...)")

    def test_assessment_proxy_uses_request_async(self):
        src = self._read("titan_plugin/proxies/assessment_proxy.py")
        self.assertIn("self._bus.request_async(", src)
        self.assertNotRegex(
            src, r"asyncio\.to_thread\(\s*self\._bus\.request\b",
            "assessment_proxy still uses to_thread(self._bus.request)")

    def test_rl_proxy_uses_request_async(self):
        src = self._read("titan_plugin/proxies/rl_proxy.py")
        self.assertIn("self._bus.request_async(", src)
        self.assertNotRegex(
            src, r"asyncio\.to_thread\(\s*self\._bus\.request\b",
            "rl_proxy still uses to_thread(self._bus.request)")

    def test_no_proxy_uses_to_thread_for_bus_request(self):
        """Project-wide invariant: no proxy may use
        asyncio.to_thread(...bus.request...). Scanner enforces."""
        violations = []
        proxies_dir = REPO_ROOT / "titan_plugin" / "proxies"
        for f in proxies_dir.glob("*.py"):
            text = f.read_text()
            # Match asyncio.to_thread( ... bus.request ... ) on the same
            # logical statement (allowing newlines via re.DOTALL)
            for m in re.finditer(
                r"asyncio\.to_thread\([^)]*?\b(?:_bus|bus)\.request\b",
                text, re.DOTALL,
            ):
                violations.append(f"{f.name}: {m.group(0)[:80]}")
        self.assertEqual(violations, [],
                         "Found asyncio.to_thread(_bus.request) in proxies — "
                         "must migrate to await self._bus.request_async(...). "
                         f"Violations: {violations}")


class TestPoolStatsEndpointMultiPool(unittest.TestCase):
    """The /v4/thread-pool endpoint must report both pools, not just default."""

    def test_endpoint_emits_pools_dict(self):
        from titan_plugin.api import dashboard
        src = inspect.getsource(dashboard.thread_pool_stats)
        self.assertIn('"pools"', src, "endpoint must emit a 'pools' key")
        self.assertIn('"default"', src, "endpoint must report default pool")
        self.assertIn('"bus_ipc"', src, "endpoint must report bus_ipc pool")


if __name__ == "__main__":
    unittest.main()
