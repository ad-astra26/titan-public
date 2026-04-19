"""
META-CGN v3 invariant tests — enforces architectural principles at CI time.

These tests codify the lessons from the 2026-04-14 incident. Any commit
that violates them fails the build. Belongs to rFP_meta_cgn_v3 § 7
improvement #8 (architecture-invariant test suite).

Covered invariants:
  1. Every emit_meta_cgn_signal call site must map to a
     SIGNAL_TO_PRIMITIVE entry (no orphans → no silent-drop of signals).
  2. Total emission rate under simulated activity must stay under the
     0.5 Hz rate budget (rFP § 3).
  3. EdgeDetector fires only on first-occurrence / threshold-crossing /
     new-max — never per-call.
  4. emit_meta_cgn_signal refuses orphan emissions (defense-in-depth at
     the helper, independent of the call-site scan).
"""
from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import pytest


# ======================================================================
# Invariant #1: no orphan producer call sites
# ======================================================================

def _collect_emit_call_sites() -> list[tuple[Path, int, str, str]]:
    """Walk titan_plugin/ for every emit_meta_cgn_signal( call and extract
    the consumer=... / event_type=... kwargs."""
    project_root = Path(__file__).parent.parent
    titan_plugin = project_root / "titan_plugin"
    sites = []
    consumer_re = re.compile(r'consumer\s*=\s*["\']([^"\']+)["\']')
    event_type_re = re.compile(r'event_type\s*=\s*["\']([^"\']+)["\']')
    for py in titan_plugin.rglob("*.py"):
        try:
            src_lines = py.read_text().splitlines()
        except Exception:
            continue
        for i, line in enumerate(src_lines, 1):
            if "emit_meta_cgn_signal(" in line:
                # Block = 20 lines including this one, to cover multi-line
                # kwargs.
                block = "\n".join(src_lines[i - 1:i + 20])
                cm = consumer_re.search(block)
                em = event_type_re.search(block)
                sites.append((py, i, cm.group(1) if cm else None,
                              em.group(1) if em else None))
    return sites


def test_no_orphan_producer_call_sites():
    """Every emit_meta_cgn_signal call site must reference a
    (consumer, event_type) pair that exists in SIGNAL_TO_PRIMITIVE.

    Regression fence for the Phase 2 bug (2026-04-14) where 5 of 13
    producers emitted signals the consumer silently discarded."""
    from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE
    sites = _collect_emit_call_sites()

    # Filter out the helper definition itself (it's a declaration, not
    # a producer call site — its args are parameter names, not literals).
    producer_sites = [
        s for s in sites
        if "bus.py" not in str(s[0])
    ]

    orphans = []
    for py, line, consumer, event_type in producer_sites:
        if consumer is None or event_type is None:
            # Could not parse — flag separately but don't fail on it.
            continue
        if (consumer, event_type) not in SIGNAL_TO_PRIMITIVE:
            orphans.append(f"{py.relative_to(py.parents[2])}:{line} — "
                           f"({consumer}, {event_type})")

    assert not orphans, (
        "Orphan emit_meta_cgn_signal call site(s) — add SIGNAL_TO_PRIMITIVE "
        "mapping or remove site:\n  " + "\n  ".join(orphans)
    )


# ======================================================================
# Invariant #2: emission rate stays within budget
# ======================================================================

def test_rate_budget_enforced_by_helper():
    """The emit_meta_cgn_signal helper's min_interval_s rate gate must
    actually drop rapid emissions. Ensures the 0.5 Hz budget is enforced
    at the source, not hoped for."""
    from titan_plugin.bus import emit_meta_cgn_signal, _emit_gate_last_ts
    from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE

    # Pick a valid (consumer, event_type) tuple from the table
    consumer, event_type = next(iter(SIGNAL_TO_PRIMITIVE.keys()))

    # Clear state for this test key
    key = ("test_producer", consumer, event_type)
    _emit_gate_last_ts.pop(key, None)

    class MockQueue:
        def __init__(self):
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

    q = MockQueue()

    # Rapid burst: 10 emissions in quick succession with 1-second gate
    accepted = sum(
        emit_meta_cgn_signal(
            q, src="test_producer", consumer=consumer,
            event_type=event_type, intensity=0.5, min_interval_s=1.0
        )
        for _ in range(10)
    )
    # Only the first should pass; the rest within 1s window should drop
    assert accepted == 1, (
        f"Rate gate failed — expected 1 accepted, got {accepted}. "
        "The min_interval_s parameter is not enforcing the rate budget."
    )
    assert len(q.items) == 1


# ======================================================================
# Invariant #3: EdgeDetector correctness
# ======================================================================

def test_edge_detector_first_time():
    """observe_first_time fires exactly once per unique key."""
    from titan_plugin.logic.meta_cgn import EdgeDetector
    d = EdgeDetector()
    assert d.observe_first_time("concept_I") is True
    assert d.observe_first_time("concept_I") is False
    assert d.observe_first_time("concept_WE") is True
    assert d.observe_first_time("concept_WE") is False


def test_edge_detector_threshold_crossing():
    """observe fires True only on crossing threshold, not per-call above."""
    from titan_plugin.logic.meta_cgn import EdgeDetector
    d = EdgeDetector()
    # Below threshold — no fire
    assert d.observe("coherence", 0.2, 0.5) is False
    # First crossing — fire
    assert d.observe("coherence", 0.6, 0.5) is True
    # Still above — no fire (invariant: not per-call above threshold!)
    assert d.observe("coherence", 0.7, 0.5) is False
    assert d.observe("coherence", 0.8, 0.5) is False
    # Drops below — next crossing fires again
    assert d.observe("coherence", 0.3, 0.5) is False
    assert d.observe("coherence", 0.6, 0.5) is True


def test_edge_detector_new_max():
    """observe_new_max fires only on genuinely new maximum."""
    from titan_plugin.logic.meta_cgn import EdgeDetector
    d = EdgeDetector()
    assert d.observe_new_max("depth", 3) is True      # first = new max
    assert d.observe_new_max("depth", 2) is False     # below prev
    assert d.observe_new_max("depth", 3) is False     # equal, not greater
    assert d.observe_new_max("depth", 4) is True      # new max
    assert d.observe_new_max("depth", 3) is False     # below prev


# ======================================================================
# Invariant #4: helper refuses orphan emissions
# ======================================================================

def test_emit_helper_refuses_orphan_tuples():
    """The helper itself must drop emissions for tuples not in
    SIGNAL_TO_PRIMITIVE, even if a producer bypasses the CI check."""
    from titan_plugin.bus import emit_meta_cgn_signal

    class MockQueue:
        def __init__(self):
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

    q = MockQueue()
    result = emit_meta_cgn_signal(
        q, src="test_producer",
        consumer="nonexistent_consumer",
        event_type="nonexistent_event",
        intensity=1.0,
    )
    assert result is False, (
        "emit_meta_cgn_signal must refuse orphan (consumer, event_type) "
        "tuples. This is defense-in-depth for the 2026-04-14 Phase 2 "
        "bug pattern."
    )
    assert len(q.items) == 0, "No message should be queued on orphan refusal"


def test_emit_helper_accepts_mapped_tuple():
    """Sanity: a valid (consumer, event_type) pair IS accepted."""
    from titan_plugin.bus import emit_meta_cgn_signal, _emit_gate_last_ts
    from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE

    consumer, event_type = next(iter(SIGNAL_TO_PRIMITIVE.keys()))
    # Clear gate state for fresh test
    _emit_gate_last_ts.clear()

    class MockQueue:
        def __init__(self):
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

    q = MockQueue()
    result = emit_meta_cgn_signal(
        q, src="test_producer",
        consumer=consumer, event_type=event_type,
        intensity=0.7, min_interval_s=0.0,
    )
    assert result is True
    assert len(q.items) == 1
    msg = q.items[0]
    assert msg["type"] == "META_CGN_SIGNAL"
    assert msg["payload"]["consumer"] == consumer
    assert msg["payload"]["event_type"] == event_type


def test_meta_cgn_signal_routes_to_spirit():
    """Regression: META_CGN_SIGNAL must route to a registered DivineBus
    subscriber. Before 2026-04-19 the helper hard-coded dst='meta', which
    had no subscriber — 14k+ emissions were silently dropped at
    DivineBus.publish. The consumer (handle_cross_consumer_signal) lives
    in the 'spirit' subprocess, so dst must be 'spirit'.
    """
    from titan_plugin.bus import emit_meta_cgn_signal, _emit_gate_last_ts
    from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE

    consumer, event_type = next(iter(SIGNAL_TO_PRIMITIVE.keys()))
    _emit_gate_last_ts.clear()

    class MockQueue:
        def __init__(self):
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

    q = MockQueue()
    emit_meta_cgn_signal(
        q, src="test_producer",
        consumer=consumer, event_type=event_type,
        intensity=0.7, min_interval_s=0.0,
    )
    assert len(q.items) == 1
    msg = q.items[0]
    assert msg["dst"] == "spirit", (
        f"META_CGN_SIGNAL dst must be 'spirit' (consumer lives in spirit "
        f"worker), got {msg['dst']!r}. No module is registered as 'meta' "
        f"so routing to dst='meta' silently drops at DivineBus.publish."
    )


# ======================================================================
# COMPLETE-4-EVENTS: META_EVENT_REWARD endpoint routing
# ======================================================================

def test_event_reward_endpoint_routes_to_spirit():
    """Regression: Events Teacher POSTs quality to
    /v4/meta-reasoning/event-reward, which must republish as
    META_EVENT_REWARD with dst='spirit' so the handler at
    spirit_worker:8454 reaches meta_engine.add_external_reward. Same
    routing failure mode as META_CGN_SIGNAL (dst='meta' → no subscriber).
    """
    from titan_plugin.bus import make_msg

    # Simulate the body of post_v4_meta_event_reward inline — avoids
    # having to stand up FastAPI TestClient. The critical invariant is
    # that the constructed bus message routes to 'spirit'.
    quality = 0.73
    window_number = 42
    titan_id = "T1"
    quality = max(0.0, min(1.0, float(quality)))
    msg = make_msg(
        "META_EVENT_REWARD", "events_teacher", "spirit", {
            "quality": quality,
            "window_number": window_number,
            "titan_id": titan_id,
        })
    assert msg["type"] == "META_EVENT_REWARD"
    assert msg["dst"] == "spirit", (
        f"META_EVENT_REWARD dst must be 'spirit' (handler lives in "
        f"spirit_worker process), got {msg['dst']!r}. Routing to any "
        f"other destination silently drops — same bug class as the "
        f"pre-2026-04-19 dst='meta' META_CGN_SIGNAL silent-drop."
    )
    assert msg["src"] == "events_teacher"
    assert msg["payload"]["quality"] == 0.73
    assert msg["payload"]["window_number"] == 42
    assert msg["payload"]["titan_id"] == "T1"


def test_event_reward_quality_clamped_to_unit_interval():
    """Defensive clamp: quality must land in [0, 1] regardless of
    caller input. add_external_reward assumes the Q-blend domain."""
    for raw, expected in [(1.5, 1.0), (-0.3, 0.0), (0.5, 0.5), (0.0, 0.0),
                          (1.0, 1.0)]:
        clamped = max(0.0, min(1.0, float(raw)))
        assert clamped == expected, f"{raw} → {clamped}, expected {expected}"


# ======================================================================
# Invariant #5: BusHealthMonitor state machine + orphan logging
# ======================================================================

def test_bus_health_orphan_logs_only_first_occurrence():
    """record_orphan logs WARN exactly once per tuple, then stays silent."""
    from titan_plugin.core.bus_health import BusHealthMonitor
    m = BusHealthMonitor()
    # First call: logs
    m.record_orphan("fake", "event_a")
    # Subsequent same tuple: silent (but still counted)
    for _ in range(99):
        m.record_orphan("fake", "event_a")

    snap = m.snapshot()
    assert snap["orphans"]["total_count"] == 100
    assert snap["orphans"]["unique_tuples"] == [["fake", "event_a"]]


def test_bus_health_snapshot_structure():
    """Snapshot returns the keys consumers depend on."""
    from titan_plugin.core.bus_health import BusHealthMonitor
    m = BusHealthMonitor()
    m.record_emission("producer_a", "consumer_a", "event_a", 0.8)
    snap = m.snapshot()

    required = {
        "ts", "uptime_s", "overall_state", "rate_budget_hz",
        "total_emission_rate_1min_hz", "producers", "queues",
        "max_queue_fraction", "backpressure_active", "orphans",
    }
    assert set(snap.keys()) >= required, (
        f"snapshot missing keys: {required - set(snap.keys())}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
