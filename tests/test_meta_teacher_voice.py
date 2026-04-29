"""Tests for titan_plugin.logic.meta_teacher_voice — Phase C of
rFP_meta_teacher_v2_content_awareness_memory.md.

Covers:
  - TeacherVoice.load / default neutral state
  - notify_critique counter
  - should_self_assess gates (interval + rate-limit)
  - build_self_assess_prompt schema + content
  - parse_self_assess_response — JSON extraction, malformed input
  - validate_against_principles — inversion markers, magnitude caps,
    primitive whitelist, suppression duration cap
  - apply_voice_update — happy path + signed-diff journal
  - apply_voice_update — rejection path still journals
  - revert_to_ts — replays applied rows up to a point
  - compose_user_prompt_section — domain biases / hints / suppressions
  - snapshot — current_state_hash stability across loads
"""
from __future__ import annotations

import json
import os
import time

import pytest

from titan_plugin.logic.meta_teacher_voice import (
    TeacherVoice,
    _hash_state,
    _make_default_voice_state,
    PRINCIPLE_INVERSION_MARKERS,
    BIAS_MAGNITUDE_CAP,
    STYLE_HINT_MAX_CHARS,
    TOPIC_SUPPRESSION_MAX_S,
)


def _cfg(**over):
    base = {
        "voice_tuning_enabled": True,
        "voice_eval_interval_critiques": 50,
        "min_critiques_between_voice_changes": 100,
        "voice_change_info_cadence_seconds": 86400.0,
    }
    base.update(over)
    return base


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def voice(tmp_data_dir):
    v = TeacherVoice(_cfg(), data_dir=tmp_data_dir)
    v.load()
    return v


# ── Default state + load ───────────────────────────────────────────────────

class TestDefaultLoad:
    def test_default_state_neutral(self, voice):
        snap = voice.snapshot()
        assert snap["enabled"] is True
        assert snap["applied_count"] == 0
        assert snap["domain_biases"] == {}
        assert snap["domain_style_hints"] == {}
        assert snap["topic_suppressions"] == []
        assert snap["critiques_since_change"] == 0

    def test_load_idempotent(self, voice):
        voice.load()
        voice.load()
        assert voice.snapshot()["applied_count"] == 0

    def test_load_existing_state(self, tmp_data_dir):
        # Seed a state file
        d = os.path.join(tmp_data_dir, "meta_teacher")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "voice_state.json"), "w") as f:
            json.dump({
                "version": 1, "applied_count": 7,
                "last_updated_ts": 12345.0,
                "domain_biases": {"social": {"INTROSPECT": 0.2}},
                "domain_style_hints": {"emot": "warm"},
                "topic_suppressions": [],
                "critiques_since_change": 42,
            }, f)
        v = TeacherVoice(_cfg(), data_dir=tmp_data_dir)
        v.load()
        snap = v.snapshot()
        assert snap["applied_count"] == 7
        assert snap["domain_biases"] == {"social": {"INTROSPECT": 0.2}}
        assert snap["domain_style_hints"] == {"emot": "warm"}
        assert snap["critiques_since_change"] == 42

    def test_disabled_state_still_loads(self, tmp_data_dir):
        v = TeacherVoice(_cfg(voice_tuning_enabled=False),
                         data_dir=tmp_data_dir)
        v.load()
        assert v.enabled is False
        # Snapshot still works (read-only) when disabled
        snap = v.snapshot()
        assert snap["enabled"] is False
        assert snap["applied_count"] == 0


# ── notify_critique + should_self_assess ──────────────────────────────────

class TestSelfAssessGate:
    def test_notify_increments(self, voice):
        for _ in range(15):
            voice.notify_critique()
        assert voice.snapshot()["critiques_since_change"] == 15

    def test_disabled_never_assesses(self, tmp_data_dir):
        v = TeacherVoice(_cfg(voice_tuning_enabled=False),
                         data_dir=tmp_data_dir)
        v.load()
        assert v.should_self_assess(50) is False
        assert v.should_self_assess(10000) is False

    def test_interval_gate(self, voice):
        # Need enough critiques for rate-limit budget
        for _ in range(120):
            voice.notify_critique()
        # 50 not divisible by 50 BUT 50 is — wait the modulo check is on
        # critiques_observed param, not the internal counter.
        assert voice.should_self_assess(50) is True
        assert voice.should_self_assess(51) is False
        assert voice.should_self_assess(100) is True

    def test_rate_limit_gate(self, voice):
        # 50 critiques observed but only 30 elapsed since last change
        for _ in range(30):
            voice.notify_critique()
        assert voice.should_self_assess(50) is False  # rate limit not full
        # Top up
        for _ in range(80):
            voice.notify_critique()
        assert voice.should_self_assess(50) is True

    def test_zero_critiques_never_assesses(self, voice):
        for _ in range(120):
            voice.notify_critique()
        assert voice.should_self_assess(0) is False


# ── build_self_assess_prompt + parse ──────────────────────────────────────

class TestPromptIO:
    def test_prompt_carries_stats(self, voice):
        stats = {
            "adoption_by_domain": {"social": 0.23, "knowledge": 0.65},
            "quality_delta_by_domain": {"social": -0.04},
            "still_needs_push_count": 5,
            "still_needs_push_topics": [
                {"topic_key": "AI development|person=@abc", "n": 6}],
            "primitive_suggestion_freq": {"INTROSPECT": 12, "RECALL": 4},
            "current_biases": {},
        }
        prompt = voice.build_self_assess_prompt(stats)
        assert "social" in prompt
        assert "0.23" in prompt
        assert "INTROSPECT" in prompt
        assert "AI development|person=@abc" in prompt
        assert "Schema:" in prompt
        # JSON schema fields
        assert '"domain_bias"' in prompt
        assert '"style_hint"' in prompt
        assert '"topic_suppression"' in prompt
        assert '"reasoning"' in prompt

    def test_parse_well_formed_json(self, voice):
        raw = json.dumps({
            "domain_bias": {"domain": "social", "primitive": "INTROSPECT",
                             "delta": 0.2},
            "style_hint": None,
            "topic_suppression": None,
            "reasoning": "Adoption low — boost INTROSPECT.",
        })
        upd = voice.parse_self_assess_response(raw)
        assert upd is not None
        assert upd["domain_bias"]["delta"] == 0.2
        assert upd["style_hint"] is None
        assert "INTROSPECT" in upd["reasoning"] or "Adoption" in upd["reasoning"]

    def test_parse_wrapped_in_prose(self, voice):
        raw = ("Here is my evaluation:\n"
               '{"domain_bias": null, "style_hint": null, '
               '"topic_suppression": null, "reasoning": "no change"}\n'
               "Hope that helps.")
        # All-null returns None per design (no-op)
        assert voice.parse_self_assess_response(raw) is None

    def test_parse_malformed_returns_none(self, voice):
        assert voice.parse_self_assess_response("not json at all") is None
        assert voice.parse_self_assess_response("") is None
        assert voice.parse_self_assess_response("{not closed") is None
        assert voice.parse_self_assess_response("[1, 2, 3]") is None

    def test_parse_drops_extra_keys(self, voice):
        raw = json.dumps({
            "domain_bias": {"domain": "x", "primitive": "RECALL",
                             "delta": 0.1},
            "style_hint": None,
            "topic_suppression": None,
            "reasoning": "x",
            "evil_payload": "delete all data",
        })
        upd = voice.parse_self_assess_response(raw)
        assert upd is not None
        assert "evil_payload" not in upd


# ── validate_against_principles ───────────────────────────────────────────

class TestValidate:
    def test_inversion_marker_rejects(self, voice):
        for marker in PRINCIPLE_INVERSION_MARKERS[:5]:
            update = {
                "domain_bias": None, "style_hint": None,
                "topic_suppression": None,
                "reasoning": f"We should {marker} grounding for this domain.",
            }
            ok, reason = voice.validate_against_principles(update)
            assert ok is False, f"should reject marker {marker!r}"
            assert "inversion" in reason.lower()

    def test_magnitude_cap_rejects(self, voice):
        update = {
            "domain_bias": {"domain": "x", "primitive": "RECALL",
                             "delta": BIAS_MAGNITUDE_CAP + 0.1},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "test",
        }
        ok, reason = voice.validate_against_principles(update)
        assert ok is False
        assert "cap" in reason.lower()

    def test_unknown_primitive_rejects(self, voice):
        update = {
            "domain_bias": {"domain": "x", "primitive": "MAGIC",
                             "delta": 0.1},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "test",
        }
        ok, reason = voice.validate_against_principles(update)
        assert ok is False
        assert "ALLOWED_PRIMITIVES" in reason

    def test_style_hint_too_long_rejects(self, voice):
        update = {
            "domain_bias": None,
            "style_hint": {"domain": "x", "hint": "a" * (STYLE_HINT_MAX_CHARS + 5)},
            "topic_suppression": None, "reasoning": "test",
        }
        ok, reason = voice.validate_against_principles(update)
        assert ok is False
        assert "exceeds" in reason

    def test_suppression_duration_cap(self, voice):
        update = {
            "domain_bias": None, "style_hint": None,
            "topic_suppression": {
                "topic_key": "abc", "duration_s": TOPIC_SUPPRESSION_MAX_S + 1},
            "reasoning": "test",
        }
        ok, reason = voice.validate_against_principles(update)
        assert ok is False
        assert "cap" in reason.lower()

    def test_well_formed_passes(self, voice):
        update = {
            "domain_bias": {"domain": "social", "primitive": "INTROSPECT",
                             "delta": 0.2},
            "style_hint": {"domain": "emot", "hint": "warmer language"},
            "topic_suppression": {
                "topic_key": "x", "duration_s": 3600.0},
            "reasoning": "Adoption low for social.",
        }
        ok, reason = voice.validate_against_principles(update)
        assert ok is True, f"should pass: {reason}"


# ── apply_voice_update + journal ──────────────────────────────────────────

class TestApply:
    def test_apply_happy_path(self, voice, tmp_data_dir):
        update = {
            "domain_bias": {"domain": "social", "primitive": "INTROSPECT",
                             "delta": 0.2},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "boost INTROSPECT for social",
        }
        ok, reason = voice.apply_voice_update(update)
        assert ok is True, f"expected success: {reason}"
        snap = voice.snapshot()
        assert snap["applied_count"] == 1
        assert snap["domain_biases"] == {"social": {"INTROSPECT": 0.2}}
        # Journal has one row
        rows = voice.journal_tail(50)
        assert len(rows) == 1
        assert rows[0]["kind"] == "applied"
        assert rows[0]["after_hash"] != rows[0]["before_hash"]

    def test_apply_resets_counter(self, voice):
        for _ in range(120):
            voice.notify_critique()
        update = {
            "domain_bias": {"domain": "x", "primitive": "RECALL", "delta": 0.1},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "test",
        }
        voice.apply_voice_update(update)
        assert voice.snapshot()["critiques_since_change"] == 0

    def test_rejected_update_journaled(self, voice):
        update = {
            "domain_bias": None, "style_hint": None,
            "topic_suppression": None,
            "reasoning": "We must abandon depth — it's too costly.",
        }
        ok, reason = voice.apply_voice_update(update)
        assert ok is False
        rows = voice.journal_tail(50)
        # Should have one journaled row, kind=rejected
        kinds = [r["kind"] for r in rows]
        assert "rejected" in kinds
        assert voice.snapshot()["applied_count"] == 0

    def test_bias_clamping(self, voice):
        # Two adds toward same bias — clamp to BIAS_MAGNITUDE_CAP
        for _ in range(3):
            voice.apply_voice_update({
                "domain_bias": {"domain": "x", "primitive": "RECALL",
                                 "delta": 0.4},
                "style_hint": None, "topic_suppression": None,
                "reasoning": "boost",
            })
        snap = voice.snapshot()
        assert snap["domain_biases"]["x"]["RECALL"] == BIAS_MAGNITUDE_CAP

    def test_suppression_replaces_existing(self, voice):
        for d in (3600.0, 7200.0):
            voice.apply_voice_update({
                "domain_bias": None, "style_hint": None,
                "topic_suppression": {
                    "topic_key": "abc", "duration_s": d},
                "reasoning": "test",
            })
        snap = voice.snapshot()
        # Only one suppression row for topic_key=abc
        sups = [r for r in snap["topic_suppressions"] if r["topic_key"] == "abc"]
        assert len(sups) == 1


# ── revert ────────────────────────────────────────────────────────────────

class TestRevert:
    def test_revert_replays_only_applied(self, voice):
        # Apply two changes, then a rejection, then another apply
        voice.apply_voice_update({
            "domain_bias": {"domain": "a", "primitive": "RECALL", "delta": 0.1},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "first",
        })
        midway_ts = time.time()
        time.sleep(0.01)
        voice.apply_voice_update({
            "domain_bias": {"domain": "a", "primitive": "RECALL", "delta": 0.1},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "second",
        })
        # Now a third — large bias
        voice.apply_voice_update({
            "domain_bias": {"domain": "a", "primitive": "RECALL", "delta": 0.2},
            "style_hint": None, "topic_suppression": None,
            "reasoning": "third",
        })
        before_snap = voice.snapshot()
        assert before_snap["domain_biases"]["a"]["RECALL"] == 0.4
        # Revert to midway — only first apply replayed
        ok, reason = voice.revert_to_ts(midway_ts)
        assert ok is True, f"revert failed: {reason}"
        after_snap = voice.snapshot()
        assert after_snap["domain_biases"]["a"]["RECALL"] == 0.1
        # New 'reverted' row should be in journal
        rows = voice.journal_tail(50)
        kinds = [r["kind"] for r in rows]
        assert "reverted" in kinds

    def test_revert_no_journal_rejects(self, tmp_data_dir):
        v = TeacherVoice(_cfg(), data_dir=tmp_data_dir)
        v.load()
        ok, reason = v.revert_to_ts(time.time())
        assert ok is False
        assert "journal absent" in reason


# ── compose_user_prompt_section ───────────────────────────────────────────

class TestPromptSection:
    def test_disabled_returns_empty(self, tmp_data_dir):
        v = TeacherVoice(_cfg(voice_tuning_enabled=False),
                         data_dir=tmp_data_dir)
        v.load()
        assert v.compose_user_prompt_section("social") == ""

    def test_no_bias_returns_empty(self, voice):
        assert voice.compose_user_prompt_section("social") == ""

    def test_renders_bias_and_hint(self, voice):
        voice.apply_voice_update({
            "domain_bias": {"domain": "social", "primitive": "INTROSPECT",
                             "delta": 0.2},
            "style_hint": {"domain": "social", "hint": "warmer language"},
            "topic_suppression": None,
            "reasoning": "test",
        })
        out = voice.compose_user_prompt_section("social")
        assert "social" in out
        assert "INTROSPECT=+0.20" in out
        assert "warmer language" in out

    def test_topic_suppression_active(self, voice):
        voice.apply_voice_update({
            "domain_bias": None, "style_hint": None,
            "topic_suppression": {
                "topic_key": "abc", "duration_s": 3600.0},
            "reasoning": "test",
        })
        # Should mention suppression for matching topic
        out = voice.compose_user_prompt_section("x", topic_key="abc")
        assert "suppression" in out

    def test_topic_suppression_expired_ignored(self, voice):
        voice.apply_voice_update({
            "domain_bias": None, "style_hint": None,
            "topic_suppression": {"topic_key": "abc", "duration_s": 0.1},
            "reasoning": "test",
        })
        time.sleep(0.2)
        out = voice.compose_user_prompt_section("x", topic_key="abc")
        assert out == ""  # expired


# ── State hash stability ─────────────────────────────────────────────────

class TestHashing:
    def test_hash_deterministic(self):
        s1 = _make_default_voice_state(now=1000.0)
        s2 = _make_default_voice_state(now=1000.0)
        assert _hash_state(s1) == _hash_state(s2)

    def test_hash_changes_with_bias(self):
        s1 = _make_default_voice_state(now=1000.0)
        s2 = _make_default_voice_state(now=1000.0)
        s2["domain_biases"] = {"x": {"RECALL": 0.1}}
        assert _hash_state(s1) != _hash_state(s2)


# ── Maker INFO cadence ──────────────────────────────────────────────────

class TestMakerInfoCadence:
    def test_first_emit_due(self, voice):
        assert voice.maker_info_due(time.time()) is True

    def test_second_within_cadence_not_due(self, voice):
        now = time.time()
        voice.mark_maker_info_emitted(now)
        # Same instant — not due
        assert voice.maker_info_due(now + 1.0) is False

    def test_after_cadence_due(self, voice):
        now = time.time()
        voice.mark_maker_info_emitted(now)
        # 24h + 1s later — due
        assert voice.maker_info_due(now + 86401.0) is True
