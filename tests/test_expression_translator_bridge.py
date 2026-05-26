"""tests/test_expression_translator_bridge.py — L3 housekeeping closure.

Pins the cross-process ExpressionTranslator stats bridge:
  - bus event constant EXPRESSION_TRANSLATOR_STATS_UPDATED exists
  - ExpressionStatePublisher.publish accepts translator_stats kwarg
  - Snapshot path produces a payload with real (non-default) values

Closes the long-standing SPEC line 2577 "ExpressionTranslator migration
(deferred)" via a low-overhead bus-event bridge (translator stays in
the main plugin process — its translate() is on the impulse-handling
hot path; an RPC migration would add unacceptable latency).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.logic.expression_state_publisher import ExpressionStatePublisher


class TestExpressionTranslatorBridge:
    def test_bus_event_constant_exists(self):
        # Pin the new event name — if removed, expression_worker
        # subscribe list silently no-ops via the optional-topic guard,
        # but the bridge would break silently.
        assert hasattr(bus, "EXPRESSION_TRANSLATOR_STATS_UPDATED")
        assert (bus.EXPRESSION_TRANSLATOR_STATS_UPDATED ==
                "EXPRESSION_TRANSLATOR_STATS_UPDATED")

    def test_publisher_accepts_translator_stats_kwarg(self):
        pub = ExpressionStatePublisher.__new__(ExpressionStatePublisher)
        pub._publish_count = 0
        pub._publish_success = 0
        pub._encode_fails = 0
        pub._write_fails = 0
        pub._oversize_fails = 0
        # Direct payload calculation — bypass the SHM write path.
        stats = {
            "sovereignty_ratio": 0.65,
            "learned_actions": 42,
            "llm_actions": 30,
            "total_actions": 72,
            "top_mappings": [{"helper": "speak", "score": 0.9}],
            "total_learned_pairs": 18,
            "posture_authenticity_ratio_30": 0.78,
        }
        payload = pub._compute_payload(
            translator=None, manager=None, translator_stats=stats)
        assert payload["sovereignty_ratio"] == pytest.approx(0.65)
        assert payload["learned_actions"] == 42
        assert payload["llm_actions"] == 30
        assert payload["total_actions"] == 72
        assert payload["total_learned_pairs"] == 18
        assert payload["posture_authenticity_ratio_30"] == pytest.approx(0.78)
        assert payload["top_mappings"] == [{"helper": "speak", "score": 0.9}]

    def test_publisher_partial_snapshot_uses_defaults_for_missing(self):
        # Cold-start safety: parent may send a snapshot before all
        # counters populated. Missing keys must default to zero.
        pub = ExpressionStatePublisher.__new__(ExpressionStatePublisher)
        pub._publish_count = 0
        pub._publish_success = 0
        pub._encode_fails = 0
        pub._write_fails = 0
        pub._oversize_fails = 0
        partial = {"sovereignty_ratio": 0.42}
        payload = pub._compute_payload(
            translator=None, manager=None, translator_stats=partial)
        assert payload["sovereignty_ratio"] == pytest.approx(0.42)
        assert payload["learned_actions"] == 0
        assert payload["llm_actions"] == 0
        assert payload["total_actions"] == 0
        assert payload["total_learned_pairs"] == 0
        assert payload["posture_authenticity_ratio_30"] == 0.0
        assert payload["top_mappings"] == []

    def test_publisher_stats_override_translator(self):
        # When BOTH translator and translator_stats are passed,
        # translator_stats wins — that's the cross-process snapshot
        # the L3 bridge is for.
        fake_translator = MagicMock()
        fake_translator.get_stats.return_value = {
            "sovereignty_ratio": 0.10,  # would-be value from object
        }
        fake_translator.posture_authenticity_ratio_30.return_value = 0.10

        pub = ExpressionStatePublisher.__new__(ExpressionStatePublisher)
        pub._publish_count = 0
        pub._publish_success = 0
        pub._encode_fails = 0
        pub._write_fails = 0
        pub._oversize_fails = 0
        snapshot = {
            "sovereignty_ratio": 0.90,  # snapshot value
            "posture_authenticity_ratio_30": 0.90,
        }
        payload = pub._compute_payload(
            translator=fake_translator, manager=None,
            translator_stats=snapshot)
        # Snapshot wins over the live object.
        assert payload["sovereignty_ratio"] == pytest.approx(0.90)
        assert payload["posture_authenticity_ratio_30"] == pytest.approx(0.90)
        # And the object's get_stats should NOT have been called.
        fake_translator.get_stats.assert_not_called()

    def test_publisher_falls_back_to_translator_object_without_snapshot(self):
        # Legacy l0_rust=false path: parent owns the publisher and
        # passes the translator object directly (no snapshot). The
        # publish path must work as before.
        fake_translator = MagicMock()
        fake_translator.get_stats.return_value = {
            "sovereignty_ratio": 0.55,
            "learned_actions": 10,
        }
        fake_translator.posture_authenticity_ratio_30.return_value = 0.40

        pub = ExpressionStatePublisher.__new__(ExpressionStatePublisher)
        pub._publish_count = 0
        pub._publish_success = 0
        pub._encode_fails = 0
        pub._write_fails = 0
        pub._oversize_fails = 0
        payload = pub._compute_payload(
            translator=fake_translator, manager=None,
            translator_stats=None)
        assert payload["sovereignty_ratio"] == pytest.approx(0.55)
        assert payload["learned_actions"] == 10
        assert payload["posture_authenticity_ratio_30"] == pytest.approx(0.40)
        fake_translator.get_stats.assert_called_once()

    def test_publisher_cold_start_no_translator_no_snapshot(self):
        # Both None — must publish stub payload (cold-start safe per
        # the publisher's contract).
        pub = ExpressionStatePublisher.__new__(ExpressionStatePublisher)
        pub._publish_count = 0
        pub._publish_success = 0
        pub._encode_fails = 0
        pub._write_fails = 0
        pub._oversize_fails = 0
        payload = pub._compute_payload(
            translator=None, manager=None, translator_stats=None)
        assert payload["sovereignty_ratio"] == 0.0
        assert payload["learned_actions"] == 0
        assert payload["total_actions"] == 0
        assert payload["posture_authenticity_ratio_30"] == 0.0
