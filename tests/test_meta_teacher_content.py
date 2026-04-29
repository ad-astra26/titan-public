"""Tests for titan_plugin.logic.meta_teacher_content — Phase A of
rFP_meta_teacher_v2_content_awareness_memory.md.

Covers:
  - build_teacher_outer_summary — shape, None-fallback, redaction caps,
    entity label stability, peer_cgn_beta extraction, felt_averages
  - build_step_arguments — shape, arg_summary derivation for each primitive,
    inner-state-only chains (no entity_refs)
  - render_chain_content_prompt_section — empty-when-empty, has content when
    populated, includes subject_pair_compatibility instruction
"""
from __future__ import annotations

from types import SimpleNamespace

from titan_plugin.logic.meta_teacher_content import (
    build_step_arguments,
    build_teacher_outer_summary,
    render_chain_content_prompt_section,
)


def _state(outer=None, refs=None, chain=None, results=None):
    """Build a minimal chain_state stand-in. MetaChainState shape only needs
    these four attrs for Phase A helpers."""
    return SimpleNamespace(
        outer_context=dict(outer) if outer else {},
        entity_refs=dict(refs) if refs else {},
        chain=list(chain) if chain else [],
        chain_results=list(results) if results else [],
    )


# ── build_teacher_outer_summary ────────────────────────────────────────────

class TestOuterSummary:
    def test_empty_outer_returns_none(self):
        assert build_teacher_outer_summary(_state()) is None
        assert build_teacher_outer_summary(_state(outer={}, refs={})) is None

    def test_non_state_object_returns_none(self):
        # graceful degradation: object without expected attrs
        class Bogus:
            pass
        assert build_teacher_outer_summary(Bogus()) is None

    def test_minimal_outer_with_entities(self):
        state = _state(
            outer={
                "person": {"handle": "@jkacrpto"},
                "felt_history": [
                    {"felt_summary": "talked about sovereignty", "relevance": 0.9,
                     "sentiment": 0.2, "arousal": 0.5},
                    {"felt_summary": "shared a joke", "relevance": 0.3,
                     "sentiment": 0.8, "arousal": 0.6},
                ],
                "sources_queried": ["person", "felt_person"],
                "sources_failed": [],
                "sources_timed_out": [],
                "fetch_ms": 142.0,
            },
            refs={"primary_person": "@jkacrpto", "current_topic": "sovereignty"},
        )
        out = build_teacher_outer_summary(state)
        assert out is not None
        assert out["primary_person"] == "@jkacrpto"
        assert out["current_topic"] == "sovereignty"
        assert out["sources_status"] == {"person": "ok", "felt_person": "ok"}
        assert out["felt_summaries"] == ["talked about sovereignty", "shared a joke"]
        assert abs(out["felt_averages"]["sentiment_avg"] - 0.5) < 0.01
        assert abs(out["felt_averages"]["relevance_avg"] - 0.6) < 0.01
        assert out["fetch_ms"] == 142.0

    def test_felt_summary_truncation_to_60_chars(self):
        long_text = "x" * 200
        state = _state(
            outer={
                "felt_history": [{"felt_summary": long_text, "relevance": 0.5}],
                "sources_queried": ["felt_person"],
            },
            refs={"primary_person": "@a"},
        )
        out = build_teacher_outer_summary(state)
        assert out is not None
        assert len(out["felt_summaries"][0]) == 60
        assert out["felt_summaries"][0] == "x" * 60

    def test_sources_status_categorizes_timeouts_and_failures(self):
        state = _state(
            outer={
                "sources_queried": ["person", "topic", "felt_person", "peer_cgn"],
                "sources_timed_out": ["topic"],
                "sources_failed": ["peer_cgn"],
                "fetch_ms": 200,
            },
            refs={"primary_person": "@a"},
        )
        out = build_teacher_outer_summary(state)
        assert out["sources_status"]["person"] == "ok"
        assert out["sources_status"]["topic"] == "timeout"
        assert out["sources_status"]["felt_person"] == "ok"
        assert out["sources_status"]["peer_cgn"] == "failed"

    def test_peer_cgn_beta_flat_map(self):
        state = _state(
            outer={
                "peer_cgn": {
                    "meta": {"sovereignty": 0.7},
                    "emot": {"sovereignty": 0.4},
                    # language has no "sovereignty" entry → None
                    "language": {"other_topic": 0.9},
                },
                "sources_queried": ["peer_cgn"],
            },
            refs={"current_topic": "sovereignty"},
        )
        out = build_teacher_outer_summary(state)
        assert out is not None
        assert out["peer_cgn_beta"]["meta"] == 0.7
        assert out["peer_cgn_beta"]["emot"] == 0.4
        assert out["peer_cgn_beta"]["language"] is None

    def test_peer_cgn_beta_case_insensitive(self):
        state = _state(
            outer={
                "peer_cgn": {"knowledge": {"AI Development": 0.55}},
                "sources_queried": ["peer_cgn"],
            },
            refs={"current_topic": "ai development"},  # different case
        )
        out = build_teacher_outer_summary(state)
        assert out["peer_cgn_beta"]["knowledge"] == 0.55

    def test_peer_cgn_beta_nested_concept_grounding(self):
        state = _state(
            outer={
                "peer_cgn": {
                    "meta": {"concept_grounding": {"chi_flow": 0.82}},
                    "emot": {"_primitives": {"chi_flow": {"beta": 0.31}}},
                },
                "sources_queried": ["peer_cgn"],
            },
            refs={"current_topic": "chi_flow"},
        )
        out = build_teacher_outer_summary(state)
        assert out["peer_cgn_beta"]["meta"] == 0.82
        assert out["peer_cgn_beta"]["emot"] == 0.31

    def test_peer_cgn_omitted_when_no_topic(self):
        state = _state(
            outer={"peer_cgn": {"meta": {"x": 0.5}}, "sources_queried": ["peer_cgn"]},
            refs={"primary_person": "@a"},
        )
        out = build_teacher_outer_summary(state)
        assert out is not None
        assert "peer_cgn_beta" not in out

    def test_titan_self_picks_only_relevant_fields(self):
        state = _state(
            outer={
                "titan_self_snapshot": {
                    "mood": "FLOW", "chi_remaining": 0.42, "pi_rate": 0.12,
                    "private_secret": "should_not_surface",
                    "internal_weights": [1, 2, 3],
                },
                "sources_queried": ["titan_self_snapshot"],
            },
            refs={"primary_person": "@a"},
        )
        out = build_teacher_outer_summary(state)
        ts = out["titan_self"]
        assert ts["mood"] == "FLOW"
        assert ts["chi_remaining"] == 0.42
        assert ts["pi_rate"] == 0.12
        assert "private_secret" not in ts
        assert "internal_weights" not in ts

    def test_inner_narrative_snippets(self):
        state = _state(
            outer={
                "inner_narrative": [
                    {"text": "Titan noticed a pattern in morning posts"},
                    {"snippet": "x" * 200},   # will truncate to 80
                    "raw string row also accepted",
                ],
                "sources_queried": ["inner_narrative_person"],
            },
            refs={"primary_person": "@a"},
        )
        out = build_teacher_outer_summary(state)
        narr = out["inner_narrative"]
        assert len(narr) == 2   # _INNER_NARR_TOP_N=2
        assert narr[0] == "Titan noticed a pattern in morning posts"
        assert len(narr[1]) == 80


# ── build_step_arguments ───────────────────────────────────────────────────

class TestStepArguments:
    def test_empty_chain_returns_empty(self):
        assert build_step_arguments(_state()) == []
        assert build_step_arguments(_state(chain=[], results=[])) == []

    def test_formulate_uses_result_goal_label(self):
        state = _state(
            chain=["FORMULATE.goal_setting"],
            results=[{"goal_label": "research sovereignty", "count": 1}],
            refs={"current_topic": "sovereignty"},
        )
        sa = build_step_arguments(state)
        assert len(sa) == 1
        assert sa[0]["primitive"] == "FORMULATE"
        assert sa[0]["sub_mode"] == "goal_setting"
        assert sa[0]["arg_summary"] == "research sovereignty"
        assert sa[0]["result_shape"]["count"] == 1

    def test_recall_entity_picks_primary_person(self):
        state = _state(
            chain=["RECALL.entity"],
            results=[{"count": 3, "wisdom_found": True}],
            refs={"primary_person": "@jkacrpto"},
        )
        sa = build_step_arguments(state)
        assert sa[0]["arg_summary"] == "@jkacrpto"
        assert sa[0]["result_shape"]["count"] == 3
        assert sa[0]["result_shape"]["wisdom_found"] is True

    def test_recall_topic_picks_current_topic(self):
        state = _state(
            chain=["RECALL.topic"],
            results=[{"count": 5}],
            refs={"current_topic": "sovereignty"},
        )
        assert build_step_arguments(state)[0]["arg_summary"] == "sovereignty"

    def test_hypothesize_uses_hypothesis_seed(self):
        state = _state(
            chain=["HYPOTHESIZE.extend_pattern"],
            results=[{"hypothesis_seed": "posts cluster by time-of-day"}],
        )
        assert (build_step_arguments(state)[0]["arg_summary"]
                == "posts cluster by time-of-day")

    def test_evaluate_renders_quality_label(self):
        state = _state(
            chain=["EVALUATE.assess"],
            results=[{"quality_score": 0.73}],
        )
        sa = build_step_arguments(state)
        assert sa[0]["arg_summary"] == "quality=0.73"
        assert sa[0]["result_shape"]["quality_score"] == 0.73

    def test_break_labels_with_step_index(self):
        state = _state(
            chain=["FORMULATE.init", "RECALL.entity", "BREAK.default"],
            results=[{}, {}, {}],
            refs={"primary_person": "@a"},
        )
        sa = build_step_arguments(state)
        assert sa[2]["arg_summary"] == "after_step_1"

    def test_inner_only_chain_still_produces_steps(self):
        # No entity_refs at all — chain should still surface primitives + subs
        state = _state(
            chain=["FORMULATE.goal_setting", "INTROSPECT.self"],
            results=[{"count": 1}, {"count": 1}],
        )
        sa = build_step_arguments(state)
        assert len(sa) == 2
        assert sa[0]["primitive"] == "FORMULATE"
        assert sa[1]["primitive"] == "INTROSPECT"
        # No arg_summary because no entity_refs AND no result labels — acceptable
        assert "arg_summary" not in sa[0] or not sa[0]["arg_summary"]

    def test_arg_summary_truncation_50_chars(self):
        long = "x" * 200
        state = _state(
            chain=["FORMULATE.goal"],
            results=[{"goal_label": long}],
        )
        sa = build_step_arguments(state)
        assert len(sa[0]["arg_summary"]) == 50

    def test_malformed_result_safe(self):
        # result is not a dict — should not blow up
        state = _state(
            chain=["FORMULATE.goal"],
            results=["not_a_dict"],
            refs={"current_topic": "x"},
        )
        sa = build_step_arguments(state)
        assert sa[0]["primitive"] == "FORMULATE"
        # arg fallback to entity_refs topic
        assert sa[0].get("arg_summary") == "x"

    def test_step_key_without_dot(self):
        state = _state(chain=["SPIRIT_SELF"], results=[{}])
        sa = build_step_arguments(state)
        assert sa[0]["primitive"] == "SPIRIT_SELF"
        assert sa[0]["sub_mode"] == ""


# ── render_chain_content_prompt_section ────────────────────────────────────

class TestRenderChainContentPromptSection:
    def test_empty_when_both_empty(self):
        assert render_chain_content_prompt_section(None, []) == ""
        assert render_chain_content_prompt_section({}, []) == ""

    def test_renders_outer_summary_fields(self):
        outer = {
            "primary_person": "@jkacrpto",
            "current_topic": "sovereignty",
            "felt_averages": {"sentiment_avg": 0.3, "arousal_avg": 0.7},
            "felt_summaries": ["talked deep", "short joke"],
            "peer_cgn_beta": {"meta": 0.7, "emot": 0.4, "language": None},
            "sources_status": {"person": "ok", "topic": "timeout"},
        }
        text = render_chain_content_prompt_section(outer, [])
        assert "Chain content:" in text
        assert "primary_person: @jkacrpto" in text
        assert "current_topic: sovereignty" in text
        assert "sentiment=0.3" in text
        assert "arousal=0.7" in text
        assert "- talked deep" in text
        assert "meta=0.7" in text
        assert "language" not in text or "language=None" not in text  # None filtered
        assert "sources_issues" in text  # timeout present
        # Subject-pair instruction present
        assert "subject_pair_compatibility" in text

    def test_renders_step_arguments(self):
        steps = [
            {"step": 0, "primitive": "FORMULATE", "sub_mode": "goal_setting",
             "arg_summary": "research X", "result_shape": {"count": 1}},
            {"step": 1, "primitive": "RECALL", "sub_mode": "entity",
             "arg_summary": "@a", "result_shape": {"found": True}},
        ]
        text = render_chain_content_prompt_section(None, steps)
        assert "steps:" in text
        assert "[0] FORMULATE.goal_setting" in text
        assert "arg=research X" in text
        assert "[1] RECALL.entity" in text

    def test_outer_only_no_issues_no_status_line(self):
        outer = {"primary_person": "@a",
                 "sources_status": {"person": "ok", "felt_person": "ok"}}
        text = render_chain_content_prompt_section(outer, [])
        assert "sources_issues" not in text


# ── Integration: build_user_prompt picks up content section ───────────────

class TestUserPromptWithContent:
    def _base_payload(self, **overrides):
        base = {
            "chain_id": 123,
            "primitives_used": ["FORMULATE", "RECALL"],
            "primitive_transitions": [("FORMULATE", "RECALL")],
            "chain_length": 2,
            "domain": "social",
            "task_success": 0.5,
            "chain_iql_confidence": 0.4,
            "context_summary": {
                "dominant_emotion": "WONDER", "chi_remaining": 0.5,
                "impasse_state": "none", "trigger_reason": "",
                "knowledge_injected": False,
            },
            "final_observation": {"chain_template": "", "unique_primitives": 2},
        }
        base.update(overrides)
        return base

    def test_legacy_payload_no_chain_content_section(self):
        from titan_plugin.logic.meta_teacher_prompts import build_user_prompt
        text = build_user_prompt(self._base_payload())
        assert "Chain content:" not in text

    def test_with_outer_summary_includes_chain_content(self):
        from titan_plugin.logic.meta_teacher_prompts import build_user_prompt
        payload = self._base_payload(
            outer_summary={
                "primary_person": "@jkacrpto",
                "current_topic": "sovereignty",
            },
            step_arguments=[
                {"step": 0, "primitive": "FORMULATE", "sub_mode": "goal",
                 "arg_summary": "explore sovereignty"},
                {"step": 1, "primitive": "RECALL", "sub_mode": "entity",
                 "arg_summary": "@jkacrpto"},
            ],
        )
        text = build_user_prompt(payload)
        assert "Chain content:" in text
        assert "@jkacrpto" in text
        assert "sovereignty" in text
        assert "[0] FORMULATE.goal" in text
        # Still contains v2 adherence line
        assert "NOT USED" in text

    def test_step_arguments_only_still_renders(self):
        from titan_plugin.logic.meta_teacher_prompts import build_user_prompt
        payload = self._base_payload(
            step_arguments=[
                {"step": 0, "primitive": "FORMULATE", "sub_mode": "goal",
                 "arg_summary": "x"},
            ],
        )
        text = build_user_prompt(payload)
        assert "Chain content:" in text
        assert "[0] FORMULATE.goal" in text
