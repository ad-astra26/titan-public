"""Tests for titan_plugin/core/worker_lifecycle.py — orphan prevention.

Two complementary defenses are tested:

1. install_parent_death_signal() — calls Linux prctl(PR_SET_PDEATHSIG).
   Hard to unit-test the actual kernel behavior (would need real fork +
   parent kill + race timing). We test the call returns True on Linux,
   handles libc-missing gracefully, and the post-prctl race-check fires
   when ppid is already 1.

2. start_parent_watcher() — daemon thread that polls getppid() and
   self-signals on reparent. Tested by patching os.getppid() to return 1
   and asserting os.kill is invoked with the right signal.

3. install_full_protection() — convenience wrapper. Tested for return
   shape + idempotency.

End-to-end orphan-prevention behavior (real fork() + kill -9 parent +
verify child exits) is left for integration testing — running it in
pytest would race the test runner's own subprocess management.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
import unittest
from unittest import mock

# Ensure project root on sys.path for direct invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import worker_lifecycle as wl


class TestInstallParentDeathSignal(unittest.TestCase):
    def test_returns_true_on_linux(self):
        """On a real Linux box, prctl should succeed."""
        if sys.platform != "linux":
            self.skipTest("prctl is Linux-only")
        # Don't actually want to set PDEATHSIG on the test runner — patch
        # libc.prctl to return 0 (success) without side effects.
        fake_libc = mock.MagicMock()
        fake_libc.prctl.return_value = 0
        with mock.patch.object(wl.ctypes, "CDLL", return_value=fake_libc):
            # Also patch getppid so the race-check doesn't fire (we're not
            # actually orphaned in the test)
            with mock.patch.object(wl.os, "getppid", return_value=os.getpid()):
                ok = wl.install_parent_death_signal(sig=signal.SIGTERM)
        self.assertTrue(ok)
        # Verify prctl was called with the right constants
        fake_libc.prctl.assert_called_once_with(
            wl._PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0,
        )

    def test_returns_false_when_libc_missing(self):
        """If ctypes.CDLL fails (non-Linux, broken libc), return False not crash."""
        with mock.patch.object(wl.ctypes, "CDLL", side_effect=OSError("not found")):
            ok = wl.install_parent_death_signal()
        self.assertFalse(ok)

    def test_returns_false_when_prctl_rejects(self):
        """Non-zero rc from prctl → False, no exception."""
        fake_libc = mock.MagicMock()
        fake_libc.prctl.return_value = -1
        with mock.patch.object(wl.ctypes, "CDLL", return_value=fake_libc):
            with mock.patch.object(wl.ctypes, "get_errno", return_value=22):
                ok = wl.install_parent_death_signal()
        self.assertFalse(ok)

    def test_race_check_fires_when_already_orphaned(self):
        """If ppid == 1 immediately after prctl, self-signal explicitly.

        Models the race where parent dies between fork() and prctl()
        completing in child. PDEATHSIG can't fire (parent already gone),
        so the race-check is the only defense.
        """
        fake_libc = mock.MagicMock()
        fake_libc.prctl.return_value = 0
        # Simulate orphaned state: getppid returns 1
        with mock.patch.object(wl.ctypes, "CDLL", return_value=fake_libc), \
             mock.patch.object(wl.os, "getppid", return_value=1), \
             mock.patch.object(wl.os, "kill") as mock_kill:
            ok = wl.install_parent_death_signal(sig=signal.SIGTERM)
        self.assertTrue(ok)  # prctl itself succeeded
        # Self-signal must have fired
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


class TestStartParentWatcher(unittest.TestCase):
    """Each test passes its own stop_event so we can clean up between cases —
    daemon threads otherwise leak across the test suite and a leftover
    watcher's getppid() patch sees the NEXT test's mock and double-fires."""

    def test_does_not_signal_when_parent_alive(self):
        """ppid != 1 → watcher loops, never signals."""
        stop = threading.Event()
        with mock.patch.object(wl.os, "getppid", return_value=os.getpid()), \
             mock.patch.object(wl.os, "kill") as mock_kill:
            t = wl.start_parent_watcher(interval=0.01, stop_event=stop)
            time.sleep(0.05)  # let watcher loop a few times
            stop.set()
            t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        mock_kill.assert_not_called()

    def test_signals_when_orphaned(self):
        """ppid == 1 → SIGTERM self, watcher exits."""
        stop = threading.Event()
        with mock.patch.object(wl.os, "getppid", return_value=1), \
             mock.patch.object(wl.os, "kill") as mock_kill:
            t = wl.start_parent_watcher(interval=0.01, sig=signal.SIGTERM,
                                        stop_event=stop)
            t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_uses_custom_signal(self):
        """Custom sig propagates to os.kill call."""
        stop = threading.Event()
        with mock.patch.object(wl.os, "getppid", return_value=1), \
             mock.patch.object(wl.os, "kill") as mock_kill:
            t = wl.start_parent_watcher(interval=0.01, sig=signal.SIGUSR1,
                                        stop_event=stop)
            t.join(timeout=2.0)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGUSR1)

    def test_returns_started_thread(self):
        """Returned thread is alive + named for observability."""
        stop = threading.Event()
        with mock.patch.object(wl.os, "getppid", return_value=os.getpid()):
            t = wl.start_parent_watcher(interval=10.0, stop_event=stop)
        try:
            self.assertTrue(t.is_alive())
            self.assertEqual(t.name, "parent_watcher")
            self.assertTrue(t.daemon)
        finally:
            stop.set()
            t.join(timeout=1.0)


class TestInstallFullProtection(unittest.TestCase):
    def test_returns_status_dict(self):
        """Convenience wrapper returns a status dict for boot logs."""
        fake_libc = mock.MagicMock()
        fake_libc.prctl.return_value = 0
        with mock.patch.object(wl.ctypes, "CDLL", return_value=fake_libc), \
             mock.patch.object(wl.os, "getppid", return_value=os.getpid()):
            status = wl.install_full_protection()
        self.assertIn("pdeathsig_installed", status)
        self.assertIn("watcher_started", status)
        self.assertTrue(status["pdeathsig_installed"])
        self.assertTrue(status["watcher_started"])

    def test_idempotent(self):
        """Calling twice is harmless (no exceptions, both report success)."""
        fake_libc = mock.MagicMock()
        fake_libc.prctl.return_value = 0
        with mock.patch.object(wl.ctypes, "CDLL", return_value=fake_libc), \
             mock.patch.object(wl.os, "getppid", return_value=os.getpid()):
            s1 = wl.install_full_protection()
            s2 = wl.install_full_protection()
        self.assertEqual(s1["pdeathsig_installed"], s2["pdeathsig_installed"])
        # Both watchers running (wasteful but not incorrect)
        self.assertTrue(s2["watcher_started"])

    def test_pdeathsig_failure_does_not_block_watcher(self):
        """Even if prctl fails, watcher still starts as backup."""
        with mock.patch.object(wl.ctypes, "CDLL", side_effect=OSError("nope")), \
             mock.patch.object(wl.os, "getppid", return_value=os.getpid()):
            status = wl.install_full_protection()
        self.assertFalse(status["pdeathsig_installed"])
        self.assertTrue(status["watcher_started"])  # backup still running


class TestEndToEndOrphanScenario(unittest.TestCase):
    """Integration test: real fork, real kill, verify child dies.

    Skipped by default because it racy against pytest's own subprocess
    management. Set TITAN_RUN_LIFECYCLE_E2E=1 to enable.
    """
    def test_real_orphan_child_dies(self):
        if not os.environ.get("TITAN_RUN_LIFECYCLE_E2E"):
            self.skipTest("set TITAN_RUN_LIFECYCLE_E2E=1 to run")
        if sys.platform != "linux":
            self.skipTest("Linux-only")

        # Spawn a small parent process whose child will install protection.
        # We then SIGKILL the parent and verify the child exits within a bound.
        import subprocess
        script = """
import os, signal, time, sys
sys.path.insert(0, %r)
from titan_plugin.core.worker_lifecycle import install_full_protection
pid = os.fork()
if pid == 0:
    install_full_protection(watcher_interval=0.5)
    # Child loops forever. Should be killed when parent dies.
    while True: time.sleep(0.1)
else:
    print(pid, flush=True)
    time.sleep(60)
""" % os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        proc = subprocess.Popen([sys.executable, "-c", script],
                                stdout=subprocess.PIPE)
        child_pid = int(proc.stdout.readline().decode().strip())
        # Wait for child to install protection
        time.sleep(1.0)
        # SIGKILL parent — child should receive PDEATHSIG SIGTERM
        proc.kill()
        proc.wait()
        # Verify child is gone within a few seconds (PDEATHSIG should be instant;
        # watcher has up to 0.5s polling interval as backup).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)  # signal 0 = existence check
                time.sleep(0.1)
            except OSError:
                # Child gone — pass
                return
        self.fail(f"child {child_pid} survived parent kill — orphan prevention failed")


if __name__ == "__main__":
    unittest.main()
