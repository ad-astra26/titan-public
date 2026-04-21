"""BUG-CGN-SILENT-UNREGISTERED-CONSUMER Phase 0 observability tests.

Background. language_worker.py sends CGN_TRANSITION for "language" (7 sites)
and "social" (2 sites) but never CGN_REGISTERs them. On Titans without
historical state to resurrect those names into self._consumers (T2/T3),
record_outcome() silently no-oped — the for-loop scanned transitions, found
no match, exited without logging. Hidden T1↔T2/T3 learning divergence
(consumer_freq T1: {"language": 11622, "social": 654}; T2/T3: {}).

Phase 0 = visibility ONLY (no behavior change). These tests verify the
observability primitives are wired correctly.
"""

from __future__ import annotations

import logging
import tempfile

import pytest

from titan_plugin.logic.cgn import ConceptGroundingNetwork


@pytest.fixture
def cgn():
    with tempfile.TemporaryDirectory() as state_dir:
        yield ConceptGroundingNetwork(state_dir=state_dir)


def test_unregistered_consumer_counted(cgn):
    """record_outcome with an unregistered consumer increments the counter."""
    cgn.record_outcome("language", "concept_xyz", 0.7)
    cgn.record_outcome("language", "concept_abc", 0.5)
    cgn.record_outcome("social", "concept_def", 0.3)
    telemetry = cgn.get_silent_drop_telemetry()
    assert telemetry["unregistered"] == {"language": 2, "social": 1}
    # Unmatched is for REGISTERED consumers without pending transition.
    assert telemetry["unmatched"] == {}
    # Registered list is empty until anyone calls register_consumer.
    assert telemetry["registered"] == []


def test_warn_throttle_per_consumer(cgn, caplog):
    """First call WARNs; subsequent calls within throttle window are silent."""
    caplog.set_level(logging.WARNING, logger="titan.cgn")
    cgn.record_outcome("language", "c1", 0.1)
    cgn.record_outcome("language", "c2", 0.1)
    cgn.record_outcome("language", "c3", 0.1)
    warns = [r for r in caplog.records
             if "UNREGISTERED consumer='language'" in r.getMessage()]
    # Throttle interval defaults to 60s; only the first should have logged.
    assert len(warns) == 1, (
        f"expected 1 throttled WARN for 'language', got {len(warns)}: "
        f"{[r.getMessage() for r in warns]}")


def test_throttle_independent_per_consumer(cgn, caplog):
    """Each unregistered consumer gets its own throttle window."""
    caplog.set_level(logging.WARNING, logger="titan.cgn")
    cgn.record_outcome("language", "c1", 0.1)
    cgn.record_outcome("social", "c1", 0.1)
    msgs = [r.getMessage() for r in caplog.records
            if "UNREGISTERED consumer=" in r.getMessage()]
    # Both consumers' first call should have logged.
    assert any("'language'" in m for m in msgs), msgs
    assert any("'social'" in m for m in msgs), msgs


def test_telemetry_exposes_registered_consumers(cgn):
    """get_silent_drop_telemetry includes the current registered set for diff."""
    from titan_plugin.logic.cgn import CGNConsumerConfig
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    telemetry = cgn.get_silent_drop_telemetry()
    assert "reasoning" in telemetry["registered"]


def test_no_behavior_change_for_registered_consumer(cgn):
    """Registered consumer's record_outcome path unchanged.

    The function should still no-op silently when there's no matching
    transition (the existing benign case). It should NOT increment
    unregistered, but MAY increment unmatched.
    """
    from titan_plugin.logic.cgn import CGNConsumerConfig
    cgn.register_consumer(CGNConsumerConfig(name="reasoning"))
    cgn.record_outcome("reasoning", "no_such_concept", 0.5)
    telemetry = cgn.get_silent_drop_telemetry()
    assert telemetry["unregistered"] == {}, "registered consumer must not count as unregistered"
    assert telemetry["unmatched"] == {"reasoning": 1}
