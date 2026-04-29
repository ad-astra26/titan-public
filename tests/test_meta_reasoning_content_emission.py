"""Tests for META_CHAIN_COMPLETE Phase A payload extension.

Per rFP_meta_teacher_v2_content_awareness_memory.md §2 + §4:
  - outer_summary + step_arguments ride along in META_CHAIN_COMPLETE
  - legacy payloads (no outer_context) still emit without the new fields
  - emit helper signature backwards-compatible (kwargs default to None/[])
"""
from __future__ import annotations

from queue import Queue

from titan_plugin.bus import emit_meta_chain_complete


def _base_kwargs(**overrides):
    base = {
        "src": "spirit",
        "chain_id": 42,
        "primitives_used": ["FORMULATE", "RECALL"],
        "primitive_transitions": [("FORMULATE", "RECALL")],
        "chain_length": 2,
        "domain": "social",
        "task_success": 0.6,
        "chain_iql_confidence": 0.5,
        "start_epoch": 100,
        "conclude_epoch": 105,
        "context_summary": {},
        "haov_hypothesis_id": None,
        "final_observation": {},
    }
    base.update(overrides)
    return base


class TestLegacyBackwardsCompat:
    def test_no_outer_no_steps_emits_without_fields(self):
        q: Queue = Queue()
        ok = emit_meta_chain_complete(q, **_base_kwargs())
        assert ok is True
        msg = q.get_nowait()
        pl = msg["payload"]
        assert "outer_summary" not in pl
        assert "step_arguments" not in pl

    def test_none_outer_empty_steps_still_excluded(self):
        q: Queue = Queue()
        emit_meta_chain_complete(
            q, **_base_kwargs(),
            outer_summary=None,
            step_arguments=[],
        )
        msg = q.get_nowait()
        pl = msg["payload"]
        assert "outer_summary" not in pl
        assert "step_arguments" not in pl

    def test_legacy_fields_still_present(self):
        q: Queue = Queue()
        emit_meta_chain_complete(q, **_base_kwargs())
        pl = q.get_nowait()["payload"]
        for key in ("chain_id", "primitives_used", "primitive_transitions",
                     "chain_length", "domain", "task_success",
                     "chain_iql_confidence", "start_epoch", "conclude_epoch",
                     "context_summary", "haov_hypothesis_id",
                     "final_observation"):
            assert key in pl


class TestPhaseAFieldsRide:
    def test_outer_summary_rides(self):
        q: Queue = Queue()
        emit_meta_chain_complete(
            q, **_base_kwargs(),
            outer_summary={"primary_person": "@jkacrpto",
                            "current_topic": "sovereignty",
                            "felt_summaries": ["talked deep"]},
            step_arguments=[],
        )
        pl = q.get_nowait()["payload"]
        assert pl["outer_summary"]["primary_person"] == "@jkacrpto"
        assert pl["outer_summary"]["current_topic"] == "sovereignty"
        assert "step_arguments" not in pl  # empty list filtered

    def test_step_arguments_ride(self):
        q: Queue = Queue()
        emit_meta_chain_complete(
            q, **_base_kwargs(),
            outer_summary=None,
            step_arguments=[
                {"step": 0, "primitive": "FORMULATE", "sub_mode": "goal",
                 "arg_summary": "x"},
            ],
        )
        pl = q.get_nowait()["payload"]
        assert "outer_summary" not in pl
        assert pl["step_arguments"][0]["primitive"] == "FORMULATE"

    def test_both_ride_together(self):
        q: Queue = Queue()
        emit_meta_chain_complete(
            q, **_base_kwargs(),
            outer_summary={"current_topic": "t"},
            step_arguments=[{"step": 0, "primitive": "RECALL", "sub_mode": "e"}],
        )
        pl = q.get_nowait()["payload"]
        assert pl["outer_summary"]["current_topic"] == "t"
        assert len(pl["step_arguments"]) == 1

    def test_outer_summary_copies_not_references(self):
        q: Queue = Queue()
        os = {"current_topic": "t"}
        emit_meta_chain_complete(
            q, **_base_kwargs(), outer_summary=os, step_arguments=[])
        pl = q.get_nowait()["payload"]
        pl["outer_summary"]["current_topic"] = "MUTATED"
        assert os["current_topic"] == "t"  # original untouched
