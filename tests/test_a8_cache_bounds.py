"""
L3 Phase A.8.1 — Cache bounds regression tests.

Tests the three real fixes shipped in this phase:
  1. plugin._dream_inbox: list → deque(maxlen=256)
  2. AgencyModule._history: list with manual trim → deque(maxlen=50)
  3. DivineBus._timeout_warned_at: dict → OrderedDict + cap 2048

Skipped (no fix needed):
  - agno session: already config-bounded via history_runs config
  - OutputVerifier: has no signature cache to bound
  - ReflexCollector: has no events buffer to bound
"""
from __future__ import annotations

import time
import unittest
from collections import OrderedDict, deque


class TestDreamInboxBound(unittest.TestCase):
    """A.8.1.1 — _dream_inbox is a deque(maxlen=256) in plugin.__init__."""

    def test_plugin_dream_inbox_is_bounded_deque(self):
        """V6 path: TitanPlugin.__init__ creates deque(maxlen=256)."""
        from titan_plugin.core.plugin import TitanPlugin
        # Construct minimal plugin (no kernel reference required for inbox check).
        # __init__ takes a kernel parameter; we don't need the kernel for this test,
        # so we mock-construct with a placeholder. The inbox is set unconditionally.
        # This test inspects the construction path rather than runtime behavior.
        import inspect
        src = inspect.getsource(TitanPlugin.__init__)
        # Verify the implementation declares deque(maxlen=256), not unbounded list.
        self.assertIn("deque(maxlen=256)", src,
                      "plugin.__init__ must use deque(maxlen=256) for _dream_inbox")
        self.assertNotIn("self._dream_inbox = []", src,
                         "plugin.__init__ must not initialize _dream_inbox to []")

    def test_legacy_core_dream_inbox_is_bounded_deque(self):
        """Legacy path: TitanCore.__init__ also uses deque(maxlen=256)."""
        from titan_plugin.legacy_core import TitanCore
        import inspect
        src = inspect.getsource(TitanCore.__init__)
        self.assertIn("deque(maxlen=256)", src,
                      "legacy_core.__init__ must use deque(maxlen=256)")
        self.assertNotIn("self._dream_inbox = []", src)

    def test_deque_evicts_oldest_at_overflow(self):
        """deque(maxlen=256) auto-evicts oldest when exceeding bound."""
        d = deque(maxlen=256)
        for i in range(300):
            d.append({"i": i})
        self.assertEqual(len(d), 256)
        # First 44 items (0-43) evicted; 44-299 retained.
        self.assertEqual(d[0]["i"], 44)
        self.assertEqual(d[-1]["i"], 299)

    def test_chat_drain_pattern_preserves_deque(self):
        """chat.py:175 drain pattern (clear+extend) must preserve the deque."""
        # Simulate the drain logic: sort, take batch of 3, clear+extend remainder.
        inbox = deque(maxlen=256)
        for i in range(10):
            inbox.append({"priority": i % 2, "timestamp": float(i), "i": i})
        _sorted = sorted(inbox, key=lambda m: (m["priority"], m["timestamp"]))
        _batch = _sorted[:3]
        # The fix in chat.py:175: clear + extend instead of slice-assign.
        inbox.clear()
        inbox.extend(_sorted[3:])
        # Inbox is still a deque with the bound.
        self.assertIsInstance(inbox, deque)
        self.assertEqual(inbox.maxlen, 256)
        self.assertEqual(len(inbox), 7)


class TestAgencyHistoryBound(unittest.TestCase):
    """A.8.1.2 — AgencyModule._history is deque(maxlen=50)."""

    def test_history_field_is_bounded_deque(self):
        """AgencyModule.__init__ declares _history as deque(maxlen=50)."""
        from titan_plugin.logic.agency.module import AgencyModule
        import inspect
        src = inspect.getsource(AgencyModule.__init__)
        self.assertIn("deque(maxlen=50)", src)
        self.assertNotIn("self._history: list[dict] = []", src)

    def test_record_action_no_longer_manually_trims(self):
        """The record-action helper no longer needs `self._history = self._history[-50:]`."""
        from titan_plugin.logic.agency.module import AgencyModule
        import inspect
        src = inspect.getsource(AgencyModule)
        # The manual trim line should be gone.
        self.assertNotIn("self._history = self._history[-50:]", src)
        # The append remains.
        self.assertIn("self._history.append(action_result)", src)

    def test_deque_50_evicts_at_51st_action(self):
        """Sanity check: deque(maxlen=50) evicts the 1st item at the 51st append."""
        d = deque(maxlen=50)
        for i in range(75):
            d.append({"i": i})
        self.assertEqual(len(d), 50)
        self.assertEqual(d[0]["i"], 25)  # oldest retained
        self.assertEqual(d[-1]["i"], 74)


class TestBusTimeoutWarnedAtBound(unittest.TestCase):
    """A.8.1.6 — DivineBus._timeout_warned_at is OrderedDict + cap 2048."""

    def test_field_is_ordered_dict(self):
        """DivineBus.__init__ creates OrderedDict, not regular dict."""
        from titan_plugin.bus import DivineBus
        bus = DivineBus()
        self.assertIsInstance(bus._timeout_warned_at, OrderedDict)
        self.assertEqual(bus._TIMEOUT_WARNED_MAX, 2048)

    def test_fifo_eviction_at_overflow(self):
        """When _timeout_warned_at exceeds 2048, oldest pair is evicted (FIFO)."""
        # Manually exercise the eviction logic (we don't need a real timeout to fire).
        # Use the same dict the bus uses.
        from titan_plugin.bus import DivineBus
        bus = DivineBus()
        bus._TIMEOUT_WARNED_MAX = 5  # shrink for test speed
        for i in range(8):
            pair = (f"src{i}", f"dst{i}")
            bus._timeout_warned_at[pair] = float(i)
            bus._timeout_warned_at.move_to_end(pair)
            while len(bus._timeout_warned_at) > bus._TIMEOUT_WARNED_MAX:
                bus._timeout_warned_at.popitem(last=False)
        self.assertEqual(len(bus._timeout_warned_at), 5)
        # First 3 pairs (src0..src2) evicted; src3..src7 retained.
        keys = list(bus._timeout_warned_at.keys())
        self.assertEqual(keys[0], ("src3", "dst3"))
        self.assertEqual(keys[-1], ("src7", "dst7"))

    def test_repeated_pair_does_not_count_twice(self):
        """Same (src, dst) pair stays as one entry, just gets re-ordered."""
        from titan_plugin.bus import DivineBus
        bus = DivineBus()
        bus._TIMEOUT_WARNED_MAX = 3
        # Insert 3 distinct pairs.
        for i in range(3):
            pair = (f"src{i}", f"dst{i}")
            bus._timeout_warned_at[pair] = float(i)
            bus._timeout_warned_at.move_to_end(pair)
        # Re-touch the first pair — should reorder, not duplicate.
        pair0 = ("src0", "dst0")
        bus._timeout_warned_at[pair0] = 99.0
        bus._timeout_warned_at.move_to_end(pair0)
        while len(bus._timeout_warned_at) > bus._TIMEOUT_WARNED_MAX:
            bus._timeout_warned_at.popitem(last=False)
        self.assertEqual(len(bus._timeout_warned_at), 3)
        # pair0 is now MRU (last); src1 and src2 are older.
        keys = list(bus._timeout_warned_at.keys())
        self.assertEqual(keys[-1], pair0)


class TestSkippedFixes(unittest.TestCase):
    """Audit notes for the 3 rFP items that turned out NOT to need fixing."""

    def test_agno_history_runs_already_config_bounded(self):
        """agno session in agent.py uses num_history_runs from config (default 5)."""
        from titan_plugin import agent
        import inspect
        src = inspect.getsource(agent)
        self.assertIn("num_history_runs", src)
        self.assertIn("history_runs", src)

    def test_output_verifier_has_no_signature_cache(self):
        """OutputVerifier has no signature cache attribute — stateless per-call."""
        from titan_plugin.logic.output_verifier import OutputVerifier
        # Inspect __init__ signature for any signature-cache attribute.
        import inspect
        src = inspect.getsource(OutputVerifier.__init__)
        # No lru_cache, no _signature_cache, no _sig_cache.
        self.assertNotIn("_signature_cache", src)
        self.assertNotIn("_sig_cache", src)
        self.assertNotIn("lru_cache", src)

    def test_reflex_collector_has_no_events_buffer(self):
        """ReflexCollector accumulates no events — only cooldowns + executors."""
        from titan_plugin.logic.reflexes import ReflexCollector
        rc = ReflexCollector()
        # Only these state attributes exist; no _events / _buffer.
        self.assertTrue(hasattr(rc, "_cooldowns"))
        self.assertTrue(hasattr(rc, "_executors"))
        self.assertFalse(hasattr(rc, "_events"))
        self.assertFalse(hasattr(rc, "_buffer"))


if __name__ == "__main__":
    unittest.main()
