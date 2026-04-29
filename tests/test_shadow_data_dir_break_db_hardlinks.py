"""Tests for Fix B — _break_db_hardlinks in shadow_data_dir.

Codifies the 2026-04-27 PM T2-shadow-swap-investigation finding: SQLite
WAL-mode mainfiles do NOT break the hardlink on first write (writes go
to a separate `-wal` file, mainfile stays untouched until checkpoint).
After cp -al we must replace top-level *.db + *.duckdb with real copies
so the shadow's IMW/DuckDB writers see independent inodes.

T1's `data/inner_memory.db` post-yesterday's swap had link count=2,
sharing inode with `data.OLD.20260427_*/inner_memory.db` — concrete
evidence that the prior code did NOT break the link.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_data_dir as sdd


class TestBreakDbHardlinks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "data"
        self.dst = self.root / "shadow"
        self.src.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _create_files(self, names):
        for name in names:
            (self.src / name).write_bytes(b"sqlite-fake-content")

    def _hardlink_dst(self):
        """Mirror of cp -al for predictable test setup."""
        self.dst.mkdir()
        for f in self.src.iterdir():
            if f.is_file():
                os.link(f, self.dst / f.name)

    def test_breaks_db_hardlinks(self):
        """*.db files in dst end up with separate inodes from src."""
        self._create_files(["inner_memory.db", "consciousness.db", "social_graph.db"])
        self._hardlink_dst()

        # Sanity: hardlinked before fix
        for name in ("inner_memory.db", "consciousness.db", "social_graph.db"):
            src_inode = (self.src / name).stat().st_ino
            dst_inode = (self.dst / name).stat().st_ino
            self.assertEqual(src_inode, dst_inode, f"{name} should be hardlinked pre-fix")

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 3, "Expected 3 hardlinks broken")

        # After fix: separate inodes
        for name in ("inner_memory.db", "consciousness.db", "social_graph.db"):
            src_inode = (self.src / name).stat().st_ino
            dst_inode = (self.dst / name).stat().st_ino
            self.assertNotEqual(src_inode, dst_inode,
                                f"{name} should have its own inode post-fix")

    def test_breaks_duckdb_hardlinks(self):
        """*.duckdb files are also rewritten."""
        self._create_files(["titan_memory.duckdb"])
        self._hardlink_dst()

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 1)

        src_inode = (self.src / "titan_memory.duckdb").stat().st_ino
        dst_inode = (self.dst / "titan_memory.duckdb").stat().st_ino
        self.assertNotEqual(src_inode, dst_inode)

    def test_preserves_content(self):
        """Real-copy preserves the bytes (no data loss)."""
        content = b"important-titan-memory-state"
        (self.src / "inner_memory.db").write_bytes(content)
        self._hardlink_dst()

        sdd._break_db_hardlinks(self.dst)
        self.assertEqual((self.dst / "inner_memory.db").read_bytes(), content)

    def test_skips_already_separate_inodes(self):
        """Files that already have separate inodes (e.g. after reflink) are
        left alone — function is a no-op for them, returns 0 broken."""
        self._create_files(["inner_memory.db"])
        self.dst.mkdir()
        # Real copy (not hardlink) → separate inode
        import shutil
        shutil.copy2(self.src / "inner_memory.db", self.dst / "inner_memory.db")

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 0, "Already-separate inodes should not be re-copied")

    def test_skips_subdirectories(self):
        """Only top-level *.db files are touched; subdirs left alone."""
        self._create_files(["inner_memory.db"])
        sub = self.src / "backups"
        sub.mkdir()
        (sub / "old.db").write_bytes(b"old-backup")

        self.dst.mkdir()
        os.link(self.src / "inner_memory.db", self.dst / "inner_memory.db")
        sub_dst = self.dst / "backups"
        sub_dst.mkdir()
        os.link(sub / "old.db", sub_dst / "old.db")

        broken = sdd._break_db_hardlinks(self.dst)
        # Top-level inner_memory.db broken; backups/old.db left as hardlink
        self.assertEqual(broken, 1)
        self.assertEqual(
            (sub / "old.db").stat().st_ino,
            (sub_dst / "old.db").stat().st_ino,
            "Subdir files should remain hardlinked (not concurrently written)")

    def test_breaks_all_top_level_files_defense_in_depth(self):
        """Defense-in-depth (2026-04-28 PM late):
        ALL top-level hardlinks are broken regardless of extension.

        Previous behavior (extension-pattern) was leaky — *.db-wal /
        *.db-shm were not covered, causing T1's inner_memory.db to corrupt
        when the shadow's WAL recovery wrote to the shared -wal file.
        New behavior: walk every top-level file, break any with nlink>1.
        """
        # Mix of file types — DB family + auxiliaries + non-DB:
        for name in (
            "inner_memory.db", "inner_memory.db-wal", "inner_memory.db-shm",
            "consciousness.db", "titan_memory.duckdb", "knowledge_graph.kuzu",
            "config.json", "faiss.index", "telemetry.jsonl", "state.bin",
        ):
            (self.src / name).write_bytes(b"x" * 64)
        self._hardlink_dst()

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 10,
            "All 10 top-level files should have their hardlinks broken")

        # Every top-level file should have its own inode now:
        for name in (
            "inner_memory.db", "inner_memory.db-wal", "inner_memory.db-shm",
            "consciousness.db", "titan_memory.duckdb", "knowledge_graph.kuzu",
            "config.json", "faiss.index", "telemetry.jsonl", "state.bin",
        ):
            self.assertNotEqual(
                (self.src / name).stat().st_ino,
                (self.dst / name).stat().st_ino,
                f"{name} should have its own inode post-fix (defense-in-depth)")

    def test_breaks_sqlite_wal_and_shm_hardlinks_specifically(self):
        """Regression test for BUG-T1-INNER-MEMORY-CORRUPTION (2026-04-28).

        The 2026-04-28 15:00:34 swap on T1 corrupted inner_memory.db
        because the shadow process's SQLite WAL recovery wrote to the
        STILL-HARDLINKED inner_memory.db-wal file. Pattern matching only
        broke *.db / *.duckdb / *.kuzu / *.kuzu.wal. This test enforces
        that *.db-wal and *.db-shm are now covered by the defense-in-depth
        walk-all-top-level approach.
        """
        for name in (
            "inner_memory.db", "inner_memory.db-wal", "inner_memory.db-shm",
            "observatory.db", "observatory.db-wal", "observatory.db-shm",
        ):
            (self.src / name).write_bytes(b"sqlite-mock")
        self._hardlink_dst()

        # Sanity: hardlinked pre-fix
        for name in ("inner_memory.db-wal", "inner_memory.db-shm",
                     "observatory.db-wal", "observatory.db-shm"):
            self.assertEqual(
                (self.src / name).stat().st_ino,
                (self.dst / name).stat().st_ino,
                f"{name} should be hardlinked pre-fix")

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 6, "All 6 SQLite family files should break")

        for name in ("inner_memory.db-wal", "inner_memory.db-shm",
                     "observatory.db-wal", "observatory.db-shm"):
            self.assertNotEqual(
                (self.src / name).stat().st_ino,
                (self.dst / name).stat().st_ino,
                f"{name} hardlink MUST be broken — see "
                "BUG-T1-INNER-MEMORY-CORRUPTION root-cause analysis")

    def test_breaks_arbitrary_unknown_extensions(self):
        """A future DB type with an unknown extension is also covered."""
        for name in ("future_db.xyz", "experimental_state.qqq", "weird_no_extension"):
            (self.src / name).write_bytes(b"future-content")
        self._hardlink_dst()

        broken = sdd._break_db_hardlinks(self.dst)
        self.assertEqual(broken, 3,
            "Defense-in-depth: any top-level hardlink is broken regardless "
            "of extension — covers future DB types we don't know about yet")


class TestCopyDataDirIntegratesBreak(unittest.TestCase):
    """End-to-end: copy_data_dir on the hardlink path now calls _break_db_hardlinks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "data"
        self.dst = self.root / "shadow"
        self.src.mkdir()
        (self.src / "inner_memory.db").write_bytes(b"x" * 1024)
        (self.src / "config.json").write_bytes(b"{}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_hardlink_path_breaks_db_inodes(self):
        # Force hardlink (skip reflink — most CI ext4 doesn't support it)
        ok, method = sdd.copy_data_dir(self.src, self.dst, use_reflink=False)
        self.assertTrue(ok)
        # method may be "hardlink" on ext4 or "copy" if -l also fails;
        # either way, *.db must be a separate inode if dst exists
        self.assertIn(method, ("hardlink", "copy"))

        if method == "hardlink":
            # Defense-in-depth (2026-04-28 PM late, post BUG-T1-INNER-MEMORY-
            # CORRUPTION): every top-level file is broken from its hardlink
            # to the source — not just *.db. JSON included.
            self.assertNotEqual(
                (self.src / "inner_memory.db").stat().st_ino,
                (self.dst / "inner_memory.db").stat().st_ino,
                "After hardlink-copy, inner_memory.db should have its own inode")
            self.assertNotEqual(
                (self.src / "config.json").stat().st_ino,
                (self.dst / "config.json").stat().st_ino,
                "Defense-in-depth: even non-DB top-level files break their "
                "hardlinks now (eliminates pattern-leak corruption mode)")


if __name__ == "__main__":
    unittest.main()
