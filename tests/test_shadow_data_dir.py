"""Tests for shadow_data_dir + lock_polling modules (BUG-B1-SHARED-LOCKS fix).

Per-shadow data directory isolation + strict lock-release polling. Verifies:
- resolve_data_path honors TITAN_DATA_DIR env var
- copy_data_dir produces a working copy via reflink/hardlink/copy fallback chain
- swap_data_dirs rotates atomically with rollback on failure
- cleanup helpers are idempotent
- lock_polling correctly identifies file holders + times out cleanly
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_data_dir as sdd
from titan_plugin.core import lock_polling


class TestResolveDataPath(unittest.TestCase):
    def setUp(self):
        # Save + clear TITAN_DATA_DIR for predictability
        self._saved = os.environ.pop("TITAN_DATA_DIR", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["TITAN_DATA_DIR"] = self._saved

    def test_unset_env_returns_default_data_path(self):
        """TITAN_DATA_DIR unset → no rewrite (original kernel behavior)."""
        self.assertEqual(sdd.resolve_data_path("data/foo.db"), "data/foo.db")
        self.assertEqual(sdd.resolve_data_path("foo.db"), "data/foo.db")
        self.assertEqual(sdd.resolve_data_path("data"), "data")

    def test_set_env_redirects_paths(self):
        """TITAN_DATA_DIR set → rewrite to that base."""
        os.environ["TITAN_DATA_DIR"] = "/tmp/my_shadow"
        self.assertEqual(sdd.resolve_data_path("data/foo.db"), "/tmp/my_shadow/foo.db")
        self.assertEqual(sdd.resolve_data_path("foo.db"), "/tmp/my_shadow/foo.db")
        self.assertEqual(sdd.resolve_data_path("data"), "/tmp/my_shadow")

    def test_handles_nested_subdirs(self):
        os.environ["TITAN_DATA_DIR"] = "/tmp/sd"
        self.assertEqual(sdd.resolve_data_path("data/run/imw.sock"), "/tmp/sd/run/imw.sock")


class TestShadowDataDirForPort(unittest.TestCase):
    def test_distinct_per_port(self):
        a = sdd.shadow_data_dir_for_port(7779)
        b = sdd.shadow_data_dir_for_port(7777)
        self.assertNotEqual(a, b)
        self.assertEqual(a.name, "data_shadow_7779")

    def test_root_relative(self):
        p = sdd.shadow_data_dir_for_port(7779, root="/var/titan")
        self.assertEqual(str(p), "/var/titan/data_shadow_7779")


class TestCopyDataDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")
        self.src = Path(self.tmp) / "data"
        self.dst = Path(self.tmp) / "data_shadow_7779"
        # Build a small fake data dir
        self.src.mkdir()
        (self.src / "titan_memory.duckdb").write_bytes(b"fake duckdb content")
        (self.src / "subdir").mkdir()
        (self.src / "subdir" / "nested.txt").write_text("nested content")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_happy_path(self):
        ok, method = sdd.copy_data_dir(self.src, self.dst)
        self.assertTrue(ok, msg=f"method: {method}")
        self.assertIn(method, ("reflink", "hardlink", "copy"))
        # Dst exists with same files
        self.assertTrue((self.dst / "titan_memory.duckdb").exists())
        self.assertTrue((self.dst / "subdir" / "nested.txt").exists())
        self.assertEqual(
            (self.dst / "subdir" / "nested.txt").read_text(),
            "nested content",
        )

    def test_refuses_when_dst_exists(self):
        self.dst.mkdir()
        ok, method = sdd.copy_data_dir(self.src, self.dst)
        self.assertFalse(ok)
        self.assertEqual(method, "dst_exists")

    def test_refuses_when_src_missing(self):
        ok, method = sdd.copy_data_dir(Path(self.tmp) / "nonexistent", self.dst)
        self.assertFalse(ok)
        self.assertEqual(method, "src_missing")

    def test_falls_through_to_copy_if_hardlink_fails(self):
        """Cross-FS hardlinks fail; must fall through to plain cp -a."""
        # Patch the module-level subprocess.run reference. Reflink + hardlink
        # both fail (rc=1); only the bare cp -a (last fallback) succeeds.
        original_run = subprocess.run
        def fake_run(args, *a, **kw):
            if "--reflink=always" in args:
                return mock.MagicMock(returncode=1, stderr=b"")
            # The hardlink attempt has "-l" as a flag (not as part of "-al")
            if "-l" in args and "--reflink=always" not in args:
                return mock.MagicMock(returncode=1, stderr=b"")
            # Last-resort plain cp -a falls through to real subprocess
            return original_run(args, *a, **kw)
        with mock.patch.object(sdd.subprocess, "run", side_effect=fake_run):
            ok, method = sdd.copy_data_dir(self.src, self.dst, use_reflink=True)
        self.assertTrue(ok, msg=f"method: {method}")
        self.assertEqual(method, "copy")


class TestSwapDataDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")
        self.canonical = Path(self.tmp) / "data"
        self.shadow = Path(self.tmp) / "data_shadow_7779"
        self.canonical.mkdir()
        (self.canonical / "old.txt").write_text("old")
        self.shadow.mkdir()
        (self.shadow / "new.txt").write_text("new")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_happy_path(self):
        ok, msg = sdd.swap_data_dirs(self.canonical, self.shadow)
        self.assertTrue(ok, msg=msg)
        # Canonical now has shadow's content
        self.assertTrue((self.canonical / "new.txt").exists())
        self.assertFalse((self.canonical / "old.txt").exists())
        # Backup exists with old content
        backups = list(Path(self.tmp).glob("data.OLD.*"))
        self.assertEqual(len(backups), 1)
        self.assertTrue((backups[0] / "old.txt").exists())
        # Shadow gone
        self.assertFalse(self.shadow.exists())

    def test_refuses_when_shadow_missing(self):
        shutil.rmtree(self.shadow)
        ok, msg = sdd.swap_data_dirs(self.canonical, self.shadow)
        self.assertFalse(ok)
        self.assertEqual(msg, "shadow_dir_missing")

    def test_works_when_canonical_missing(self):
        """Edge case: first-ever swap with no prior canonical (testbed scenario)."""
        shutil.rmtree(self.canonical)
        ok, msg = sdd.swap_data_dirs(self.canonical, self.shadow)
        self.assertTrue(ok)
        self.assertTrue((self.canonical / "new.txt").exists())


class TestCleanupShadowDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_removes_existing(self):
        target = Path(self.tmp) / "shadow"
        target.mkdir()
        (target / "x.txt").write_text("data")
        self.assertTrue(sdd.cleanup_shadow_dir(target))
        self.assertFalse(target.exists())

    def test_idempotent_when_missing(self):
        self.assertTrue(sdd.cleanup_shadow_dir(Path(self.tmp) / "nonexistent"))


class TestCleanupOldBackups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")
        self.canonical = Path(self.tmp) / "data"
        # Create 5 backups
        for i in range(5):
            (Path(self.tmp) / f"data.OLD.2026010{i}_120000").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_keeps_most_recent_n(self):
        removed = sdd.cleanup_old_backups(self.canonical, keep_count=2)
        self.assertEqual(removed, 3)
        remaining = sorted(Path(self.tmp).glob("data.OLD.*"))
        self.assertEqual(len(remaining), 2)
        # Most-recent timestamps remain
        names = [p.name for p in remaining]
        self.assertIn("data.OLD.20260103_120000", names)
        self.assertIn("data.OLD.20260104_120000", names)


class TestLockPollingFuserHolders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")
        self.test_file = Path(self.tmp) / "lockable.txt"
        self.test_file.write_text("hello")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_empty_for_unheld_file(self):
        holders = lock_polling._fuser_holders(self.test_file)
        self.assertEqual(holders, [])

    def test_returns_empty_for_missing_file(self):
        self.assertEqual(
            lock_polling._fuser_holders(Path(self.tmp) / "missing"),
            [],
        )

    def test_detects_held_file(self):
        """Open the test file and verify fuser sees us as a holder."""
        if shutil.which("fuser") is None:
            self.skipTest("fuser not available")
        # Open file + keep fd alive while we poll
        fp = open(self.test_file, "r")
        try:
            holders = lock_polling._fuser_holders(self.test_file)
            self.assertIn(os.getpid(), holders)
        finally:
            fp.close()


class TestPollLocksReleased(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="titan_test_")
        self.test_file = Path(self.tmp) / "lockable.txt"
        self.test_file.write_text("hello")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_immediately_when_files_unheld(self):
        ok, diag = lock_polling.poll_locks_released(
            files=[self.test_file], timeout=2.0,
        )
        self.assertTrue(ok)
        self.assertTrue(diag["released"])

    def test_times_out_when_locked(self):
        if shutil.which("fuser") is None:
            self.skipTest("fuser not available")
        fp = open(self.test_file, "r")
        try:
            ok, diag = lock_polling.poll_locks_released(
                files=[self.test_file], timeout=0.5, poll_interval=0.1,
            )
        finally:
            fp.close()
        self.assertFalse(ok)
        self.assertFalse(diag["released"])
        self.assertIn(str(self.test_file), diag["still_held"])

    def test_excludes_pids(self):
        """If holder is in exclude_pids, treat as released."""
        if shutil.which("fuser") is None:
            self.skipTest("fuser not available")
        fp = open(self.test_file, "r")
        try:
            ok, diag = lock_polling.poll_locks_released(
                files=[self.test_file], timeout=1.0,
                exclude_pids={os.getpid()},
            )
        finally:
            fp.close()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
