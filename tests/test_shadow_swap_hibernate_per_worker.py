"""Tests for Fix A — per-worker HIBERNATE publish in shadow_orchestrator.

Codifies the 2026-04-27 PM T2-shadow-swap-investigation finding: HIBERNATE
must reach reply_only=True workers (IMW, observatory_writer) so they can
hibernate cleanly and release DB locks BEFORE shadow's writers boot.
The prior `dst="all"` broadcast skipped reply_only modules (bus.py:512).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.core import shadow_orchestrator as so
from titan_plugin import bus as bus_mod


class _FakeBus:
    """Minimal bus that records every publish."""

    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(dict(msg))
        return 0

    def drain(self, *_args, **_kwargs):
        return []


class _FakeInbox:
    pass


class TestHibernatePerWorkerPublish(unittest.TestCase):
    def setUp(self):
        self.bus = _FakeBus()
        self.inbox = _FakeInbox()
        self.result = so.SwapResult(event_id="evt-test", reason="test")

    def _run_phase(self, expected_workers):
        # _phase_hibernate calls helpers (_drain_messages etc.) that need a
        # kernel; we only care about the publish loop, so wrap it.
        with mock.patch.object(so, "_drain_messages", return_value=[]):
            try:
                so._phase_hibernate(
                    self.bus, self.inbox, self.result,
                    expected_workers=expected_workers,
                    kernel=None,
                )
            except Exception:
                # Function may raise after publishes when kernel=None and
                # no acks arrive. We only care about the publishes.
                pass

    def test_hibernate_published_per_worker_dst(self):
        """Every expected_worker receives HIBERNATE via dst=<worker_name>."""
        workers = ["imw", "observatory_writer", "memory", "spirit"]
        self._run_phase(workers)

        hibernate_msgs = [
            m for m in self.bus.published if m.get("type") == bus_mod.HIBERNATE
        ]
        dsts = sorted(m["dst"] for m in hibernate_msgs)
        self.assertEqual(dsts, sorted(workers),
                         f"Expected HIBERNATE per worker, got dsts={dsts}")

    def test_no_hibernate_with_dst_all(self):
        """The legacy dst='all' HIBERNATE broadcast is gone."""
        self._run_phase(["imw", "spirit"])
        hibernate_msgs = [
            m for m in self.bus.published if m.get("type") == bus_mod.HIBERNATE
        ]
        for m in hibernate_msgs:
            self.assertNotEqual(
                m.get("dst"), "all",
                f"Expected per-worker dst, found dst='all': {m}")

    def test_imw_receives_hibernate(self):
        """Regression test: IMW must be in the publish list (it's reply_only=True
        and was previously skipped by the dst='all' broadcast)."""
        self._run_phase(["imw", "memory", "spirit"])
        imw_hibernates = [
            m for m in self.bus.published
            if m.get("type") == bus_mod.HIBERNATE and m.get("dst") == "imw"
        ]
        self.assertEqual(len(imw_hibernates), 1,
                         "IMW must receive exactly one HIBERNATE; got "
                         f"{len(imw_hibernates)}")

    def test_other_signals_still_broadcast(self):
        """SYSTEM_UPGRADE_STARTING + BUS_HANDOFF stay as dst='all'.
        They reach socket-broker subscribers (no reply_only filter at
        broker) and don't need per-worker fanout."""
        self._run_phase(["imw"])
        sys_upgrade = [
            m for m in self.bus.published
            if m.get("type") == bus_mod.SYSTEM_UPGRADE_STARTING
        ]
        bus_handoff = [
            m for m in self.bus.published if m.get("type") == bus_mod.BUS_HANDOFF
        ]
        self.assertEqual(len(sys_upgrade), 1)
        self.assertEqual(sys_upgrade[0]["dst"], "all")
        self.assertEqual(len(bus_handoff), 1)
        self.assertEqual(bus_handoff[0]["dst"], "all")

    def test_empty_expected_workers_publishes_no_hibernates(self):
        """Edge case: empty expected_workers list → no HIBERNATE publishes."""
        self._run_phase([])
        hibernate_msgs = [
            m for m in self.bus.published if m.get("type") == bus_mod.HIBERNATE
        ]
        self.assertEqual(len(hibernate_msgs), 0)


if __name__ == "__main__":
    unittest.main()
