"""Tests for Layer 2 — Shadow DB integrity gate (_check_shadow_db_integrity).

2026-04-27 PM (T2-shadow-swap-fix session): the shadow health gate now
runs PRAGMA quick_check on every SQLite in shadow's data dir before nginx
swap. This codifies the protection so a malformed shadow DB can never
again be committed to production (closes 2026-04-26 timechain incident).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_orchestrator as so


def _make_db(path: Path, healthy: bool = True) -> None:
    """Create a tiny SQLite at `path`. If healthy=False, write garbage to
    corrupt the page header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    c.execute("INSERT INTO t (val) VALUES ('hello')")
    c.commit()
    c.close()
    if not healthy:
        # Corrupt the SQLite header (first 16 bytes is the magic string).
        # Truncate file to half its size — quick_check will detect.
        sz = path.stat().st_size
        with open(path, "r+b") as f:
            f.seek(sz // 2)
            f.write(b"\x00" * 4096)


class TestShadowDBIntegrityGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Mimic the shadow_data_dir layout: <root>/data_shadow_<port>/
        self.port = 7779
        self.shadow_dir = self.root / f"data_shadow_{self.port}"
        # Shim: monkey-patch shadow_data_dir.shadow_data_dir_for_port to
        # our tmpdir-relative location for this test.
        from titan_plugin.core import shadow_data_dir as sdd
        self._orig_sdd_for_port = sdd.shadow_data_dir_for_port
        sdd.shadow_data_dir_for_port = lambda port, root=".": (
            self.root / f"data_shadow_{port}"
        )
        # Also shim project_root in the orchestrator path resolution.
        # _check_shadow_db_integrity computes project_root from
        # __file__.parents[2]. We can't change that, but we CAN make
        # shadow_data_dir_for_port ignore root and return our path —
        # which is what the shim above does.

    def tearDown(self):
        from titan_plugin.core import shadow_data_dir as sdd
        sdd.shadow_data_dir_for_port = self._orig_sdd_for_port
        self._tmp.cleanup()

    def test_disabled_returns_none(self):
        """When shadow_db_integrity_check_enabled=False, function is no-op."""
        criteria = so.HealthCriteria(shadow_db_integrity_check_enabled=False)
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertIsNone(result)

    def test_missing_shadow_dir(self):
        """Missing shadow dir → fail with clear error."""
        # shadow dir doesn't exist (we never made it)
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertFalse(result["pass"])
        self.assertIn("shadow_dir_missing", result["error"])

    def test_empty_dir_passes_silently(self):
        """No DBs to check → trivial pass with checked_count=0."""
        self.shadow_dir.mkdir()
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertTrue(result["pass"])
        self.assertEqual(result["checked_count"], 0)

    def test_all_healthy_passes(self):
        """3 healthy SQLite DBs → pass, all checked."""
        self.shadow_dir.mkdir()
        for name in ("inner_memory.db", "consciousness.db", "social_graph.db"):
            _make_db(self.shadow_dir / name, healthy=True)
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertTrue(result["pass"], f"expected pass, got {result}")
        self.assertEqual(result["checked_count"], 3)
        self.assertEqual(result["failed_count"], 0)

    def test_corrupt_db_fails_gate(self):
        """One corrupt DB → fail, listed in failed[]."""
        self.shadow_dir.mkdir()
        _make_db(self.shadow_dir / "inner_memory.db", healthy=True)
        _make_db(self.shadow_dir / "consciousness.db", healthy=False)
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertFalse(result["pass"], "expected fail with corrupt DB")
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(len(result["failed"]), 1)
        # The failed entry names the corrupt file
        self.assertEqual(result["failed"][0]["name"], "consciousness.db")

    def test_timechain_index_subdir_checked(self):
        """The deep timechain/index.db is checked (it's the one that
        actually corrupted on 2026-04-26)."""
        self.shadow_dir.mkdir()
        _make_db(self.shadow_dir / "inner_memory.db", healthy=True)
        _make_db(self.shadow_dir / "timechain" / "index.db", healthy=False)
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertFalse(result["pass"])
        # The failed entry names the relative path including subdir
        names = [f["name"] for f in result["failed"]]
        self.assertIn("timechain/index.db", names)

    def test_multi_corrupt_all_listed(self):
        """Multiple corrupt DBs → all listed in failed[]."""
        self.shadow_dir.mkdir()
        _make_db(self.shadow_dir / "a.db", healthy=False)
        _make_db(self.shadow_dir / "b.db", healthy=False)
        _make_db(self.shadow_dir / "c.db", healthy=True)
        criteria = so.HealthCriteria()
        result = so._check_shadow_db_integrity(self.port, criteria)
        self.assertFalse(result["pass"])
        self.assertEqual(result["failed_count"], 2)
        self.assertEqual(result["checked_count"], 1)


if __name__ == "__main__":
    unittest.main()
