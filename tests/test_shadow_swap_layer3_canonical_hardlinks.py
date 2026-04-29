"""Tests for Layer 3 — break_canonical_db_hardlinks.

2026-04-27 PM (T2-shadow-swap-fix session): before each shadow swap, we
break leftover SQLite/DuckDB hardlinks between data/ and any prior
data.OLD.<ts>/. Without this, multiple swaps grow link_count chains
across data/ ↔ data.OLD.* — concrete state on T1 today: 28k+ files
have link_count=2.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_data_dir as sdd


class TestBreakCanonicalDBHardlinks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.canonical = self.root / "data"
        self.canonical.mkdir()
        self.old = self.root / "data.OLD.20260427_184938"
        self.old.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _create_hardlinked_db(self, name: str):
        """Create canonical/<name> + old/<name> as hardlinks (shared inode)."""
        src = self.canonical / name
        src.write_bytes(b"sqlite-fake-content")
        os.link(src, self.old / name)
        # Verify hardlink semantics
        assert src.stat().st_ino == (self.old / name).stat().st_ino
        assert src.stat().st_nlink == 2

    def test_breaks_db_hardlinks(self):
        """SQLite *.db files in canonical have link>1 → broken."""
        self._create_hardlinked_db("inner_memory.db")
        self._create_hardlinked_db("consciousness.db")

        broken = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(broken, 2)

        # Canonical now has its own inode
        for name in ("inner_memory.db", "consciousness.db"):
            self.assertEqual((self.canonical / name).stat().st_nlink, 1,
                             f"{name} should be sole-link post-break")
            self.assertNotEqual(
                (self.canonical / name).stat().st_ino,
                (self.old / name).stat().st_ino,
                f"{name} canonical and old should be distinct inodes")
            # Content preserved
            self.assertEqual((self.canonical / name).read_bytes(),
                             b"sqlite-fake-content")
            # data.OLD/ side preserved (and now its OWN inode)
            self.assertEqual((self.old / name).read_bytes(),
                             b"sqlite-fake-content")
            self.assertEqual((self.old / name).stat().st_nlink, 1,
                             "data.OLD copy is now sole-link too")

    def test_breaks_duckdb_hardlinks(self):
        """*.duckdb files also handled."""
        self._create_hardlinked_db("titan_memory.duckdb")
        broken = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(broken, 1)
        self.assertEqual(
            (self.canonical / "titan_memory.duckdb").stat().st_nlink, 1)

    def test_skips_already_sole_link(self):
        """DBs already at link_count=1 are left alone."""
        # File exists in canonical only — link_count=1
        (self.canonical / "fresh.db").write_bytes(b"new-data")
        broken = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(broken, 0)

    def test_idempotent_second_call_is_noop(self):
        """After breaking, re-running returns 0."""
        self._create_hardlinked_db("a.db")
        first = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(first, 1)
        second = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(second, 0)

    def test_breaks_all_top_level_files_defense_in_depth(self):
        """Defense-in-depth (2026-04-28 PM late, post BUG-T1-INNER-MEMORY-
        CORRUPTION): break_canonical_db_hardlinks now walks ALL top-level
        files and breaks any with nlink>1 — not just *.db / *.duckdb.
        Pattern-match was leaky (missed *.db-wal, *.db-shm) and corrupted
        T1's inner_memory.db. Symmetric to _break_db_hardlinks(dst)."""
        # Hardlinked JSON (formerly skipped, now broken)
        (self.canonical / "config.json").write_bytes(b"{}")
        os.link(self.canonical / "config.json", self.old / "config.json")
        # Hardlinked DB
        self._create_hardlinked_db("inner_memory.db")

        broken = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(broken, 2,
            "BOTH the .db AND the .json should be broken — defense-in-depth")
        # JSON now has its own inode
        self.assertNotEqual(
            (self.canonical / "config.json").stat().st_ino,
            (self.old / "config.json").stat().st_ino,
            "JSON hardlink MUST also break (defense-in-depth)")
        # DB no longer hardlinked
        self.assertNotEqual(
            (self.canonical / "inner_memory.db").stat().st_ino,
            (self.old / "inner_memory.db").stat().st_ino)

    def test_handles_timechain_index_subdir(self):
        """timechain/index.db (subdir) is the actual incident file —
        must be explicitly handled."""
        tc = self.canonical / "timechain"
        tc.mkdir()
        old_tc = self.old / "timechain"
        old_tc.mkdir()
        (tc / "index.db").write_bytes(b"timechain-index")
        os.link(tc / "index.db", old_tc / "index.db")
        self.assertEqual((tc / "index.db").stat().st_nlink, 2)

        broken = sdd.break_canonical_db_hardlinks(self.canonical)
        self.assertEqual(broken, 1)
        self.assertEqual((tc / "index.db").stat().st_nlink, 1)
        self.assertNotEqual(
            (tc / "index.db").stat().st_ino,
            (old_tc / "index.db").stat().st_ino)

    def test_missing_canonical_returns_zero(self):
        """Calling on a path that doesn't exist returns 0, no exception."""
        broken = sdd.break_canonical_db_hardlinks(self.root / "does_not_exist")
        self.assertEqual(broken, 0)


if __name__ == "__main__":
    unittest.main()
