"""Tests — synthesis.turn_index_store (Phase 3 P3.B).

Per-chat-session monotonic turn_index counter; persists across restart;
LRU-evicts at MAX_TRACKED_SESSIONS; soft-fails to 0 on any error.
"""
from __future__ import annotations

import json
import os
import unittest

from titan_hcl.synthesis import turn_index_store as tis


class TestTurnIndexStore(unittest.TestCase):

    def setUp(self):
        # Each test gets its own tmp state file.
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmpdir, "ti.json")
        tis.set_state_path(self.state_path)

    def tearDown(self):
        tis.clear_cache_for_test()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── Core monotonic behavior ───────────────────────────────────

    def test_first_call_returns_zero(self):
        self.assertEqual(tis.next_turn_index("chat-1"), 0)

    def test_monotonic_increment_per_session(self):
        for expected in range(5):
            self.assertEqual(tis.next_turn_index("session-A"), expected)

    def test_independent_sessions_have_independent_counters(self):
        for _ in range(3):
            tis.next_turn_index("alpha")
        for _ in range(7):
            tis.next_turn_index("beta")
        self.assertEqual(tis.peek_turn_index("alpha"), 2)
        self.assertEqual(tis.peek_turn_index("beta"), 6)

    def test_empty_chat_id_returns_zero_no_persist(self):
        self.assertEqual(tis.next_turn_index(""), 0)
        self.assertEqual(tis.next_turn_index(None), 0)
        # No file written for empty keys.
        self.assertFalse(os.path.exists(self.state_path))

    def test_peek_returns_none_for_unknown(self):
        self.assertIsNone(tis.peek_turn_index("never-seen"))

    def test_peek_does_not_advance(self):
        tis.next_turn_index("X")  # → 0
        self.assertEqual(tis.peek_turn_index("X"), 0)
        self.assertEqual(tis.peek_turn_index("X"), 0)
        self.assertEqual(tis.next_turn_index("X"), 1)  # peek didn't bump

    # ── Persistence across simulated restart ──────────────────────

    def test_state_persists_to_json_file(self):
        tis.next_turn_index("persisted-session")  # → 0
        tis.next_turn_index("persisted-session")  # → 1
        tis.next_turn_index("persisted-session")  # → 2
        # File should exist + carry the last value.
        self.assertTrue(os.path.exists(self.state_path))
        with open(self.state_path) as f:
            data = json.load(f)
        self.assertEqual(data["persisted-session"], 2)

    def test_simulated_restart_resumes_continuity(self):
        # Pre-restart: advance to 3
        for _ in range(4):
            tis.next_turn_index("continuity-test")
        self.assertEqual(tis.peek_turn_index("continuity-test"), 3)
        # Simulate process restart — clear cache, leave file on disk
        tis.clear_cache_for_test()
        tis.set_state_path(self.state_path)
        # First call after restart should resume at 4 (prev + 1)
        self.assertEqual(tis.next_turn_index("continuity-test"), 4)

    def test_corrupt_state_file_starts_fresh(self):
        """Corrupted JSON → log + start with empty cache, no exception."""
        with open(self.state_path, "w") as f:
            f.write("not-valid-json{{{")
        tis.clear_cache_for_test()
        tis.set_state_path(self.state_path)
        # First call on corrupt-state-load returns 0 cleanly.
        self.assertEqual(tis.next_turn_index("after-corrupt"), 0)

    def test_non_dict_state_file_starts_fresh(self):
        with open(self.state_path, "w") as f:
            json.dump(["not", "a", "dict"], f)
        tis.clear_cache_for_test()
        tis.set_state_path(self.state_path)
        self.assertEqual(tis.next_turn_index("x"), 0)

    # ── LRU eviction at cap ───────────────────────────────────────

    def test_lru_evicts_oldest_when_cap_hit(self):
        """When MAX_TRACKED_SESSIONS hit, oldest key is dropped."""
        # Temporarily lower the cap for test speed.
        orig_cap = tis.MAX_TRACKED_SESSIONS
        tis.MAX_TRACKED_SESSIONS = 5
        try:
            tis.clear_cache_for_test()
            tis.set_state_path(self.state_path)
            for i in range(5):
                tis.next_turn_index(f"sess-{i}")
            # sess-0 is the oldest. Adding sess-5 should evict it.
            tis.next_turn_index("sess-5")
            self.assertIsNone(tis.peek_turn_index("sess-0"))
            # Others survive.
            self.assertIsNotNone(tis.peek_turn_index("sess-1"))
            self.assertIsNotNone(tis.peek_turn_index("sess-5"))
        finally:
            tis.MAX_TRACKED_SESSIONS = orig_cap


if __name__ == "__main__":
    unittest.main()
