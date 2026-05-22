"""
Tests for /health warm-cache + /health/light endpoint (2026-05-05 closure).

Coverage:
- /health/light registered + returns ok with no subsystem checks
- _build_vault_status_sync handles missing config gracefully
- _start_vault_warmer + _start_health_warmer are idempotent
- _get_vault_status_cached + _get_health_summary_cached return cached data
- /health endpoint falls back to bounded sync builder on cold boot
- Lazy-start kicks off warmer chain on first /health call
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from titan_hcl.api import dashboard as _dash


class TestHealthLightEndpoint(unittest.TestCase):
    def test_health_light_route_registered(self):
        routes = [r.path for r in _dash.router.routes if hasattr(r, "path")]
        self.assertIn("/health/light", routes)

    def test_health_light_does_not_invoke_build_health_snapshot(self):
        """The whole point of /health/light is to skip subsystem checks.
        Verify the handler body (post-docstring) doesn't reference heavy
        builders. Strip docstring before checking — references in docs
        are explanatory, not invocations.
        """
        import inspect
        src = inspect.getsource(_dash.health_light)
        # Strip the triple-quoted docstring to get just executable body
        import re
        body = re.sub(r'""".*?"""', '', src, count=1, flags=re.DOTALL)
        self.assertNotIn("_build_health_snapshot_sync", body)
        self.assertNotIn("_get_vault_status_cached", body)
        self.assertNotIn("guardian", body.lower())
        self.assertNotIn("vault", body.lower())


class TestVaultStatusWarmer(unittest.TestCase):
    def setUp(self):
        _dash._vault_status_cache["data"] = None
        _dash._vault_status_cache["updated_at"] = 0.0
        _dash._vault_warmer_started["flag"] = False

    def test_builder_returns_none_when_vault_program_id_missing(self):
        """No vault program_id configured → returns None (not an error)."""
        plugin = MagicMock()
        plugin.config = {"network": {}}
        # Ensure the early-return path triggers
        result = _dash._build_vault_status_sync(plugin)
        self.assertIsNone(result)

    def test_builder_returns_none_when_solana_unavailable(self):
        plugin = MagicMock()
        plugin.config = {"network": {"vault_program_id": "abc"}}
        with patch("titan_hcl.utils.solana_client.is_available", return_value=False):
            result = _dash._build_vault_status_sync(plugin)
        self.assertIsNone(result)

    def test_warmer_idempotent(self):
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            plugin = MagicMock()
            _dash._start_vault_warmer(plugin)
            _dash._start_vault_warmer(plugin)
            self.assertEqual(mock_thread.call_count, 1)

    def test_cached_getter_returns_data_when_warm(self):
        _dash._vault_status_cache["data"] = {"commit_count": 5}
        result = _dash._get_vault_status_cached()
        self.assertEqual(result, {"commit_count": 5})

    def test_cached_getter_returns_none_when_cold(self):
        self.assertIsNone(_dash._get_vault_status_cached())

    def test_warmer_interval_is_30s(self):
        """Vault state changes only on meditation cycles (~5-15min); 30s is plenty."""
        self.assertEqual(_dash._VAULT_WARMER_INTERVAL_S, 30.0)


class TestHealthSummaryWarmer(unittest.TestCase):
    def setUp(self):
        _dash._health_summary_cache["data"] = None
        _dash._health_summary_cache["updated_at"] = 0.0
        _dash._health_warmer_started["flag"] = False
        _dash._vault_warmer_started["flag"] = False

    def test_builder_function_exists(self):
        self.assertTrue(callable(_dash._build_health_snapshot_sync))

    def test_warmer_idempotent(self):
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            plugin = MagicMock()
            _dash._start_health_warmer(plugin)
            _dash._start_health_warmer(plugin)
            # 2 threads expected: vault warmer + health warmer (one each)
            # Total Thread() calls: 2 (one per first-time start)
            self.assertEqual(mock_thread.call_count, 2)

    def test_warmer_starts_vault_dependency(self):
        """_start_health_warmer should kick off _start_vault_warmer too."""
        with patch.object(_dash, "_start_vault_warmer") as mock_vault_start, \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            plugin = MagicMock()
            _dash._start_health_warmer(plugin)
            mock_vault_start.assert_called_once_with(plugin)

    def test_cached_getter_returns_data_when_warm(self):
        _dash._health_summary_cache["data"] = {"status": "ACTIVE"}
        self.assertEqual(_dash._get_health_summary_cached(), {"status": "ACTIVE"})

    def test_cached_getter_returns_none_when_cold(self):
        self.assertIsNone(_dash._get_health_summary_cached())

    def test_warmer_interval_is_5s(self):
        """Health is the most-polled endpoint; 5s keeps cache fresh."""
        self.assertEqual(_dash._HEALTH_WARMER_INTERVAL_S, 5.0)


class TestHealthHandlerCachePath(unittest.TestCase):
    """The /health handler is the canonical fast path — verify cache reads."""

    def setUp(self):
        _dash._health_summary_cache["data"] = None
        _dash._health_warmer_started["flag"] = False
        _dash._vault_warmer_started["flag"] = False

    def test_health_route_registered(self):
        routes = [r.path for r in _dash.router.routes if hasattr(r, "path")]
        self.assertIn("/health", routes)

    def test_handler_uses_cache_first(self):
        """Verify the handler source has the warm-cache shortcut at the top."""
        import inspect
        src = inspect.getsource(_dash.health_check)
        # Cache-read should appear before the cold-boot path
        cache_idx = src.find("_get_health_summary_cached")
        cold_idx = src.find("_build_health_snapshot_sync")
        self.assertGreater(cache_idx, 0, "missing _get_health_summary_cached call")
        self.assertGreater(cold_idx, 0, "missing _build_health_snapshot_sync call")
        self.assertLess(cache_idx, cold_idx,
                        "cache-read must be checked before cold-boot fallback")

    def test_handler_starts_warmer_lazily(self):
        """First /health call should kick off _start_health_warmer."""
        import inspect
        src = inspect.getsource(_dash.health_check)
        self.assertIn("_start_health_warmer", src)

    def test_handler_has_15s_cold_boot_bound(self):
        """Cold-boot path must be wrapped in asyncio.wait_for(timeout=1.5)."""
        import inspect
        src = inspect.getsource(_dash.health_check)
        self.assertIn("timeout=1.5", src)
        self.assertIn("asyncio.wait_for", src)


# TestWatchdogScriptUsesLightProbe retired 2026-05-16 — services_watchdog.sh
# was removed (cron retired 2026-05-14 Phase C migration; systemd + HCL
# Guardian own supervision). t{1,2,3}_manage.sh probes /health/light
# directly without the watchdog-script intermediary.


if __name__ == "__main__":
    unittest.main()
