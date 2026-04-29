"""
Tests for titan_plugin.core.readiness_reporter — B.1 §5+§6 helper.

Covers the four B.1 message handlers + trivial-reporter factory.
PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §5 + §6
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock

import pytest

from titan_plugin import bus
from titan_plugin.core import shadow_protocol as sp
from titan_plugin.core.readiness_reporter import (
    ReadinessReporter,
    trivial_reporter,
)


def make_send_queue():
    """Use a real Queue so we can introspect what got published."""
    return queue.Queue(maxsize=100)


# ── Basic dispatch ─────────────────────────────────────────────────

class TestDispatch:
    def test_handles_returns_true_for_b1_types(self):
        r = ReadinessReporter("spirit", "L1", make_send_queue())
        for t in [
            bus.UPGRADE_READINESS_QUERY, bus.HIBERNATE, bus.HIBERNATE_CANCEL,
            bus.SYSTEM_UPGRADE_QUEUED, bus.SYSTEM_UPGRADE_PENDING,
            bus.SYSTEM_UPGRADE_PENDING_DEFERRED, bus.SYSTEM_UPGRADE_STARTING,
            bus.SYSTEM_RESUMED,
        ]:
            assert r.handles(t), f"should handle {t}"

    def test_handles_returns_false_for_unrelated(self):
        r = ReadinessReporter("spirit", "L1", make_send_queue())
        assert not r.handles(bus.RELOAD)
        assert not r.handles(bus.MODULE_HEARTBEAT)
        assert not r.handles(bus.BODY_STATE)


# ── Readiness query → report ───────────────────────────────────────

class TestReadinessQuery:
    def test_empty_blockers_reports_ready(self):
        q = make_send_queue()
        r = ReadinessReporter("spirit", "L1", q,
                              blocker_cb=lambda: ([], []))
        r.handle({"type": bus.UPGRADE_READINESS_QUERY, "rid": "abc"})

        out = q.get_nowait()
        assert out["type"] == bus.UPGRADE_READINESS_REPORT
        assert out["src"] == "spirit"
        assert out["dst"] == "shadow_swap"
        assert out["rid"] == "abc"
        assert out["payload"]["ready"] is True
        assert out["payload"]["hard"] == []
        assert out["payload"]["soft"] == []

    def test_hard_blocker_reports_not_ready(self):
        q = make_send_queue()
        r = ReadinessReporter(
            "api_subprocess", "L3", q,
            blocker_cb=lambda: (
                [sp.HardBlocker(name="x_post_in_flight", eta_seconds=8.4, since=0.0)],
                [],
            ),
        )
        r.handle({"type": bus.UPGRADE_READINESS_QUERY, "rid": "xyz"})

        out = q.get_nowait()
        assert out["payload"]["ready"] is False
        assert len(out["payload"]["hard"]) == 1
        assert out["payload"]["hard"][0]["name"] == "x_post_in_flight"

    def test_soft_blocker_reports_not_ready(self):
        q = make_send_queue()
        r = ReadinessReporter(
            "spirit", "L1", q,
            blocker_cb=lambda: (
                [],
                [sp.SoftBlocker(name="reasoning_chain", eta_seconds=4.0,
                                metadata={"chain_id": 4521})],
            ),
        )
        r.handle({"type": bus.UPGRADE_READINESS_QUERY})
        out = q.get_nowait()
        assert out["payload"]["ready"] is False
        assert out["payload"]["soft"][0]["metadata"]["chain_id"] == 4521

    def test_blocker_cb_exception_defaults_to_ready(self, caplog):
        q = make_send_queue()

        def boom():
            raise RuntimeError("simulated")

        r = ReadinessReporter("spirit", "L1", q, blocker_cb=boom)
        r.handle({"type": bus.UPGRADE_READINESS_QUERY})
        out = q.get_nowait()
        # Failsafe: if blocker_cb raises, we report ready (don't block upgrade
        # because of a buggy reporter)
        assert out["payload"]["ready"] is True


# ── Hibernate → ack ────────────────────────────────────────────────

class TestHibernate:
    def test_hibernate_ack_with_state_paths(self, tmp_path):
        # Real files so checksum can be computed
        f1 = tmp_path / "spirit_state.json"
        f1.write_bytes(b'{"epoch": 42}')
        f2 = tmp_path / "sphere_clock.json"
        f2.write_bytes(b'{"radius": 0.7}')

        q = make_send_queue()
        r = ReadinessReporter(
            "spirit", "L1", q,
            save_state_cb=lambda: [str(f1), str(f2)],
        )
        r.handle({"type": bus.HIBERNATE, "rid": "h1",
                  "payload": {"event_id": "EID123"}})

        out = q.get_nowait()
        assert out["type"] == bus.HIBERNATE_ACK
        assert out["payload"]["src"] == "spirit"
        assert out["payload"]["layer"] == "L1"
        assert out["payload"]["state_paths"] == [str(f1), str(f2)]
        assert out["payload"]["event_id"] == "EID123"
        # Checksum is non-empty 64-char SHA-256 hex
        assert len(out["payload"]["state_checksum"]) == 64
        assert out["payload"]["elapsed_ms"] > 0

        # Reporter signals exit
        assert r.hibernating is True
        assert r.should_exit() is True

    def test_hibernate_with_empty_state_paths(self):
        q = make_send_queue()
        r = ReadinessReporter("body", "L1", q, save_state_cb=lambda: [])
        r.handle({"type": bus.HIBERNATE, "payload": {}})

        out = q.get_nowait()
        assert out["payload"]["state_checksum"] == ""  # no files
        assert r.hibernating is True

    def test_hibernate_save_state_exception_still_acks(self):
        q = make_send_queue()

        def boom():
            raise RuntimeError("save failed")

        r = ReadinessReporter("body", "L1", q, save_state_cb=boom)
        r.handle({"type": bus.HIBERNATE, "payload": {}})

        # Failsafe: still sends ACK (orchestrator can detect partial save
        # via empty state_paths + zero checksum). Worker still exits.
        out = q.get_nowait()
        assert out["type"] == bus.HIBERNATE_ACK
        assert out["payload"]["state_paths"] == []
        assert r.hibernating is True

    def test_hibernate_cancel_resets_hibernating(self):
        q = make_send_queue()
        r = ReadinessReporter("body", "L1", q)
        # Pre-hibernate: cancel is a no-op
        r.handle({"type": bus.HIBERNATE_CANCEL, "payload": {}})
        assert r.hibernating is False
        # After hibernate: cancel is logged but doesn't reverse state
        # (worker is about to exit; CANCEL race-window only)
        r.handle({"type": bus.HIBERNATE, "payload": {}})
        q.get_nowait()  # drain ACK
        assert r.hibernating is True
        r.handle({"type": bus.HIBERNATE_CANCEL, "payload": {}})
        assert r.hibernating is True  # already past the point


# ── Self-aware thought callbacks ───────────────────────────────────

class TestThoughtCallbacks:
    def test_queued_thought_emitted(self):
        thoughts = []
        cb = lambda t, p: thoughts.append((t, p))
        r = ReadinessReporter("spirit", "L1", make_send_queue(), thought_cb=cb)
        r.handle({
            "type": bus.SYSTEM_UPGRADE_QUEUED,
            "payload": {"event_id": "E1", "reason": "test upgrade"},
        })
        assert len(thoughts) == 1
        text, payload = thoughts[0]
        assert "upgrade approaching" in text.lower()
        assert payload["phase"] == "queued"
        assert payload["event_id"] == "E1"

    def test_starting_thought_emitted(self):
        thoughts = []
        r = ReadinessReporter(
            "spirit", "L1", make_send_queue(),
            thought_cb=lambda t, p: thoughts.append((t, p)),
        )
        r.handle({"type": bus.SYSTEM_UPGRADE_STARTING, "payload": {}})
        assert len(thoughts) == 1
        assert "resting" in thoughts[0][0].lower()
        assert thoughts[0][1]["phase"] == "starting"

    def test_resumed_thought_includes_kernel_versions_and_gap(self):
        thoughts = []
        r = ReadinessReporter(
            "spirit", "L1", make_send_queue(),
            thought_cb=lambda t, p: thoughts.append((t, p)),
        )
        r.handle({
            "type": bus.SYSTEM_RESUMED,
            "payload": {
                "kernel_version_from": "abc12345",
                "kernel_version_to": "def67890",
                "gap_seconds": 2.7,
            },
        })
        text = thoughts[0][0]
        assert "abc12345" in text
        assert "def67890" in text
        assert "2.7" in text
        assert "preserved" in text.lower()

    def test_deferred_thought_emitted(self):
        thoughts = []
        r = ReadinessReporter(
            "spirit", "L1", make_send_queue(),
            thought_cb=lambda t, p: thoughts.append((t, p)),
        )
        r.handle({
            "type": bus.SYSTEM_UPGRADE_PENDING_DEFERRED,
            "payload": {"blockers": [{"name": "x"}, {"name": "y"}]},
        })
        text = thoughts[0][0]
        assert "deferred" in text.lower()
        assert "2 blocker" in text

    def test_thought_cb_exception_doesnt_break_handler(self):
        def boom(*a, **kw):
            raise RuntimeError("thought failed")
        r = ReadinessReporter(
            "spirit", "L1", make_send_queue(), thought_cb=boom,
        )
        # Should NOT raise out of handle()
        r.handle({"type": bus.SYSTEM_UPGRADE_QUEUED, "payload": {}})

    def test_no_thought_cb_pending_silently_ignored(self):
        r = ReadinessReporter("spirit", "L1", make_send_queue())
        # No thought_cb set — message should be silently absorbed
        r.handle({"type": bus.SYSTEM_UPGRADE_PENDING, "payload": {}})
        # No exception, no output queue write expected


# ── Trivial reporter factory ───────────────────────────────────────

class TestTrivialReporter:
    def test_always_reports_ready(self):
        q = make_send_queue()
        r = trivial_reporter("body", "L1", q)
        r.handle({"type": bus.UPGRADE_READINESS_QUERY})
        assert q.get_nowait()["payload"]["ready"] is True

    def test_handles_hibernate_with_save_state(self, tmp_path):
        f = tmp_path / "body_state.json"
        f.write_bytes(b'{"x": 1}')
        q = make_send_queue()
        r = trivial_reporter("body", "L1", q, save_state_cb=lambda: [str(f)])
        r.handle({"type": bus.HIBERNATE, "payload": {"event_id": "E1"}})
        out = q.get_nowait()
        assert out["type"] == bus.HIBERNATE_ACK
        assert out["payload"]["state_paths"] == [str(f)]
        assert r.hibernating is True

    def test_no_thought_cb_means_silent_lifecycle_events(self):
        q = make_send_queue()
        r = trivial_reporter("body", "L1", q)
        for t in [bus.SYSTEM_UPGRADE_QUEUED, bus.SYSTEM_UPGRADE_STARTING,
                  bus.SYSTEM_RESUMED, bus.SYSTEM_UPGRADE_PENDING_DEFERRED]:
            r.handle({"type": t, "payload": {}})
        # No outputs expected — body shouldn't emit upgrade thoughts
        assert q.empty()
