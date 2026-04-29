"""Tests for titan_plugin.logic.meta_teacher.MetaTeacher.

Covers per rFP §10 "Tests" checklist:
  - evaluate_chain output shape (via parse_critique + build_feedback_payload)
  - reward-weight schedule advances correctly across critique count
  - sampling respects uncertainty_threshold + rate_cap + random_sample_rate
  - disabled teacher produces no bus traffic
"""
import random
import time

import pytest

from titan_plugin.logic.meta_teacher import MetaTeacher
from titan_plugin.logic.meta_teacher_prompts import (
    ALL_PRIMITIVES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    build_user_prompt,
)


def make_cfg(**overrides):
    base = {
        "enabled": True,
        "sample_mode": "uncertainty_plus_random",
        "uncertainty_threshold": 0.4,
        "random_sample_rate": 0.15,
        "max_critiques_per_hour": 30,
        "domain_balance_floor": 0.05,
        "reward_weight": 0.05,
        "reward_weight_cap": 0.30,
        "grounding_weight": 0.15,
        "ramp_phase1_critiques": 1000,
        "ramp_phase2_critiques": 1500,
        "llm_timeout_s": 30.0,
        "task_key": "meta_teacher",
    }
    base.update(overrides)
    return base


def make_payload(**overrides):
    base = {
        "chain_id": 42,
        "primitives_used": ["FORMULATE", "RECALL", "HYPOTHESIZE", "EVALUATE"],
        "primitive_transitions": [
            ("FORMULATE", "RECALL"),
            ("RECALL", "HYPOTHESIZE"),
            ("HYPOTHESIZE", "EVALUATE"),
        ],
        "chain_length": 4,
        "domain": "social",
        "task_success": 0.72,
        "chain_iql_confidence": 0.6,
        "start_epoch": 100,
        "conclude_epoch": 105,
        "context_summary": {
            "dominant_emotion": "WONDER",
            "chi_remaining": 0.5,
            "impasse_state": "none",
            "trigger_reason": "explore",
            "knowledge_injected": False,
        },
        "haov_hypothesis_id": None,
        "final_observation": {"chain_template": "FORMULATE→RECALL→HYPOTHESIZE→EVALUATE", "unique_primitives": 4},
    }
    base.update(overrides)
    return base


class TestRewardWeightSchedule:
    def test_phase0_flat(self):
        t = MetaTeacher(make_cfg())
        assert t.compute_reward_weight() == pytest.approx(0.05)
        t.total_critiques = 500
        assert t.compute_reward_weight() == pytest.approx(0.05)

    def test_phase1_ramps_linearly(self):
        t = MetaTeacher(make_cfg())
        t.total_critiques = 1000
        assert t.compute_reward_weight() == pytest.approx(0.05, abs=1e-3)
        t.total_critiques = 1250  # midway
        assert t.compute_reward_weight() == pytest.approx(0.10, abs=1e-3)
        t.total_critiques = 1500
        assert t.compute_reward_weight() == pytest.approx(0.15, abs=1e-3)

    def test_phase2_stable(self):
        t = MetaTeacher(make_cfg())
        t.total_critiques = 5000
        assert t.compute_reward_weight() == pytest.approx(0.15)

    def test_hard_cap_never_exceeded(self):
        t = MetaTeacher(make_cfg(reward_weight_cap=0.99))  # try to bypass
        # Hard cap is min(0.30, config) — can't exceed 0.30
        assert t.reward_weight_cap == 0.30

    def test_config_cap_lower_than_hard(self):
        t = MetaTeacher(make_cfg(reward_weight_cap=0.10))
        assert t.reward_weight_cap == 0.10


class TestSampling:
    def test_disabled_never_samples(self):
        t = MetaTeacher(make_cfg(enabled=False))
        ok, reason = t.should_sample(make_payload())
        assert ok is False
        assert reason == "disabled"

    def test_uncertainty_gate_triggers(self):
        t = MetaTeacher(make_cfg(uncertainty_threshold=0.5))
        ok, reason = t.should_sample(make_payload(chain_iql_confidence=0.3))
        assert ok is True
        assert reason == "uncertainty"

    def test_uncertainty_above_threshold_falls_back_to_random(self):
        rng = random.Random(0)
        t = MetaTeacher(make_cfg(uncertainty_threshold=0.3, random_sample_rate=1.0))
        ok, reason = t.should_sample(
            make_payload(chain_iql_confidence=0.9), rng=rng)
        assert ok is True
        assert reason == "random"

    def test_random_only_mode(self):
        rng = random.Random(0)
        t = MetaTeacher(make_cfg(
            sample_mode="random_only", random_sample_rate=1.0))
        ok, reason = t.should_sample(
            make_payload(chain_iql_confidence=0.1), rng=rng)
        assert ok is True
        assert reason == "random"

    def test_uncertainty_only_mode(self):
        rng = random.Random(0)
        t = MetaTeacher(make_cfg(
            sample_mode="uncertainty_only", uncertainty_threshold=0.3))
        ok, reason = t.should_sample(
            make_payload(chain_iql_confidence=0.9), rng=rng)
        assert ok is False

    def test_rate_cap_enforced(self):
        t = MetaTeacher(make_cfg(max_critiques_per_hour=5, random_sample_rate=1.0))
        rng = random.Random(0)
        for _ in range(5):
            ok, _ = t.should_sample(make_payload(chain_iql_confidence=0.1), rng=rng)
            assert ok is True
            t._record_sample(make_payload())
        # 6th chain — rate cap trips
        ok, reason = t.should_sample(make_payload(chain_iql_confidence=0.1), rng=rng)
        assert ok is False
        assert reason == "rate_cap"


class TestParseCritique:
    def test_valid_json_parses(self):
        t = MetaTeacher(make_cfg())
        resp = (
            '{"quality_score": 0.8, "critique_categories": ["depth"], '
            '"critique_text": "Good chain.", "suggested_primitives": [], '
            '"confidence": 0.9, "principles_invoked": ["depth"]}'
        )
        critique = t.parse_critique(resp)
        assert critique is not None
        assert critique["quality_score"] == 0.8
        assert critique["critique_categories"] == ["depth"]
        assert critique["critique_text"] == "Good chain."
        assert critique["confidence"] == 0.9

    def test_json_wrapped_in_prose(self):
        t = MetaTeacher(make_cfg())
        resp = (
            "Here is my evaluation:\n"
            '{"quality_score": 0.7, "critique_categories": ["grounding"], '
            '"critique_text": "Good.", "suggested_primitives": [], '
            '"confidence": 0.8, "principles_invoked": []}\n'
            "Hope that helps!"
        )
        critique = t.parse_critique(resp)
        assert critique is not None
        assert critique["quality_score"] == 0.7

    def test_invalid_json_returns_none(self):
        t = MetaTeacher(make_cfg())
        assert t.parse_critique("") is None
        assert t.parse_critique("not json at all") is None
        assert t.parse_critique("{not valid json}") is None

    def test_clamps_out_of_range_values(self):
        t = MetaTeacher(make_cfg())
        resp = ('{"quality_score": 2.5, "critique_categories": [], '
                '"critique_text": "x", "suggested_primitives": [], '
                '"confidence": -0.5, "principles_invoked": []}')
        critique = t.parse_critique(resp)
        assert critique["quality_score"] == 1.0
        assert critique["confidence"] == 0.0


class TestBuildFeedbackPayload:
    def test_feedback_from_valid_critique(self):
        t = MetaTeacher(make_cfg())
        critique = {
            "quality_score": 0.75,
            "critique_categories": ["depth"],
            "critique_text": "x",
            "suggested_primitives": ["FORMULATE", "RECALL"],
            "confidence": 0.8,
            "principles_invoked": ["depth"],
        }
        fb = t.build_feedback_payload(make_payload(), critique)
        assert fb["chain_id"] == 42
        assert fb["quality_score"] == 0.75
        assert fb["llm_ok"] is True
        assert fb["reward_bonus"] == pytest.approx(0.05)  # phase 0 weight
        assert fb["suggested_primitives"] == ["FORMULATE", "RECALL"]

    def test_feedback_neutral_on_critique_none(self):
        t = MetaTeacher(make_cfg())
        fb = t.build_feedback_payload(make_payload(), None)
        assert fb["quality_score"] == 0.5
        assert fb["confidence"] == 0.0
        assert fb["reward_bonus"] == 0.0
        assert fb["llm_ok"] is False


class TestGroundingPayloads:
    def test_grounding_per_primitive(self):
        t = MetaTeacher(make_cfg())
        critique = {
            "quality_score": 0.7,
            "critique_categories": [],
            "critique_text": "x",
            "suggested_primitives": [],
            "confidence": 0.8,
            "principles_invoked": [],
        }
        payloads = t.build_grounding_payloads(make_payload(), critique)
        assert len(payloads) == 4  # one per primitive
        assert all(p["label_quality"] == 0.7 for p in payloads)
        assert all(p["grounding_weight"] == 0.15 for p in payloads)
        prim_ids = {p["primitive_id"] for p in payloads}
        assert prim_ids == {"FORMULATE", "RECALL", "HYPOTHESIZE", "EVALUATE"}

    def test_grounding_empty_when_disabled(self):
        t = MetaTeacher(make_cfg(enabled=False))
        critique = {
            "quality_score": 0.7, "critique_categories": [],
            "critique_text": "x", "suggested_primitives": [],
            "confidence": 0.8, "principles_invoked": [],
        }
        assert t.build_grounding_payloads(make_payload(), critique) == []

    def test_grounding_empty_when_critique_none(self):
        t = MetaTeacher(make_cfg())
        assert t.build_grounding_payloads(make_payload(), None) == []


class TestPromptBuilder:
    def test_user_prompt_has_required_fields(self):
        p = build_user_prompt(make_payload())
        assert "Chain 42" in p
        assert "domain=social" in p
        assert "FORMULATE" in p
        assert "reward: 0.720" in p
        assert "WONDER" in p

    def test_user_prompt_handles_hypothesis(self):
        p = build_user_prompt(make_payload(haov_hypothesis_id="H:123"))
        assert "Hypothesis formed: H:123" in p

    def test_system_prompt_is_nonempty(self):
        assert len(SYSTEM_PROMPT) > 200
        assert "DEPTH" in SYSTEM_PROMPT
        assert "GROUNDING" in SYSTEM_PROMPT


class TestAdoption:
    def test_adoption_ema_updates(self):
        t = MetaTeacher(make_cfg())
        # Seed starts at 0.5
        e1 = t.update_adoption("social", True)
        assert e1 == pytest.approx(0.55)  # 0.9 * 0.5 + 0.1 * 1.0
        e2 = t.update_adoption("social", False)
        assert e2 == pytest.approx(0.495)

    def test_multiple_domains_tracked_independently(self):
        t = MetaTeacher(make_cfg())
        t.update_adoption("social", True)
        t.update_adoption("knowledge", False)
        assert t.adoption_ema_by_domain["social"] > t.adoption_ema_by_domain["knowledge"]


class TestTelemetry:
    def test_telemetry_shape(self):
        t = MetaTeacher(make_cfg())
        tel = t.telemetry()
        assert "enabled" in tel
        assert "critiques_lifetime" in tel
        assert "current_reward_weight" in tel
        assert "reward_weight_schedule" in tel
        assert tel["reward_weight_schedule"] == "phase_0_flat"

    def test_schedule_phase_transitions(self):
        t = MetaTeacher(make_cfg())
        assert t._schedule_phase_name() == "phase_0_flat"
        t.total_critiques = 1250
        assert t._schedule_phase_name() == "phase_1_ramp"
        t.total_critiques = 2000
        assert t._schedule_phase_name() == "phase_2_stable"


# ─────────────────────────────────────────────────────────────────────
# v2 (2026-04-24) — MISSING-primitives-only + adoption intersection
# ─────────────────────────────────────────────────────────────────────

class TestV2PromptContract:
    def test_prompt_version_is_current(self):
        # v2 = diversity-failure fix (2026-04-24 morning)
        # v3 = teaching memory bump (rFP_meta_teacher_v2 Phase B, 2026-04-24)
        # OBS-meta-teacher-* gates reset on any version bump.
        assert SYSTEM_PROMPT_VERSION >= 2

    def test_all_primitives_set_has_9(self):
        # Guard against silent shrinkage.
        assert set(ALL_PRIMITIVES) == {
            "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
            "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF", "INTROSPECT",
        }

    def test_user_prompt_lists_used_and_not_used(self):
        """v2: build_user_prompt must emit both USED + NOT USED sections."""
        payload = make_payload()  # uses FORMULATE+RECALL+HYPOTHESIZE+EVALUATE
        prompt = build_user_prompt(payload)
        assert "Primitives USED in this chain:" in prompt
        assert "Primitives NOT USED in this chain:" in prompt
        # NOT_USED should contain BREAK, DELEGATE, SPIRIT_SELF, INTROSPECT, SYNTHESIZE
        for p in ["BREAK", "DELEGATE", "SPIRIT_SELF", "INTROSPECT", "SYNTHESIZE"]:
            assert p in prompt, f"expected {p} in NOT USED list"

    def test_system_prompt_mentions_missing_only_rule(self):
        assert "MISSING" in SYSTEM_PROMPT or "NOT USED" in SYSTEM_PROMPT


class TestV2ParseCritiqueDefensiveFilter:
    """parse_critique strips any suggested primitive that was already used."""

    def test_strips_violations_when_used_primitives_passed(self):
        t = MetaTeacher(make_cfg())
        # LLM disobeys v2 rule — suggests FORMULATE which was used
        resp = (
            '{"quality_score": 0.6, "critique_categories": ["depth"], '
            '"critique_text": "x", '
            '"suggested_primitives": ["FORMULATE", "SYNTHESIZE", "BREAK"], '
            '"confidence": 0.8, "principles_invoked": ["depth"]}'
        )
        critique = t.parse_critique(
            resp, used_primitives=["FORMULATE", "RECALL"])
        assert critique is not None
        # FORMULATE filtered out; SYNTHESIZE + BREAK kept
        assert critique["suggested_primitives"] == ["SYNTHESIZE", "BREAK"]
        assert critique.get("filtered_v2_violations") == 1

    def test_keeps_all_when_used_primitives_not_passed(self):
        """Backward compat: if used_primitives is None, no filter."""
        t = MetaTeacher(make_cfg())
        resp = (
            '{"quality_score": 0.6, "critique_categories": [], '
            '"critique_text": "x", '
            '"suggested_primitives": ["FORMULATE", "RECALL"], '
            '"confidence": 0.8, "principles_invoked": []}'
        )
        critique = t.parse_critique(resp)  # no used_primitives arg
        assert critique["suggested_primitives"] == ["FORMULATE", "RECALL"]
        assert "filtered_v2_violations" not in critique

    def test_caps_at_3_even_if_llm_returns_more(self):
        t = MetaTeacher(make_cfg())
        # LLM returns 5 non-violation suggestions — should still cap at 3
        resp = (
            '{"quality_score": 0.6, "critique_categories": [], '
            '"critique_text": "x", '
            '"suggested_primitives": '
            '["SYNTHESIZE", "BREAK", "DELEGATE", "INTROSPECT", "SPIRIT_SELF"], '
            '"confidence": 0.8, "principles_invoked": []}'
        )
        critique = t.parse_critique(
            resp, used_primitives=["FORMULATE", "RECALL"])
        assert len(critique["suggested_primitives"]) == 3
        assert critique["suggested_primitives"] == [
            "SYNTHESIZE", "BREAK", "DELEGATE"]

    def test_all_violations_yields_empty_list(self):
        t = MetaTeacher(make_cfg())
        # All suggested are already used
        resp = (
            '{"quality_score": 0.6, "critique_categories": [], '
            '"critique_text": "x", '
            '"suggested_primitives": ["FORMULATE", "RECALL"], '
            '"confidence": 0.8, "principles_invoked": []}'
        )
        critique = t.parse_critique(
            resp, used_primitives=["FORMULATE", "RECALL", "HYPOTHESIZE"])
        assert critique["suggested_primitives"] == []
        assert critique.get("filtered_v2_violations") == 2
