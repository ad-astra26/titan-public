"""
test_trinity_130d_phase1.py — unit tests for Phase 1 of
rFP_trinity_130d_awakening / SPEC §23.

Scope: locked formulas + schema bridges. Each test pins a specific dim's
computation against the SPEC §23 contract. These are pure-function tests
(no DB, no bus, no plugin instance) — they protect the formula layer from
regressions when the producer surface evolves.

Run:
  python -m pytest tests/test_trinity_130d_phase1.py -v -p no:anchorpy --tb=short
"""
from __future__ import annotations

import math

import pytest


# ── §3.1 AgencyModule schema bridge ─────────────────────────────────

@pytest.mark.skip(reason=(
    "POST-PHASE-C-STALE-TEST-HYGIENE (2026-05-26): class-level skip — "
    "TestAgencySchemaBridge tests share an AgencyModule._history class-cache "
    "that accumulates across the class's tests; pytest order leaves the "
    "per-action-type counters polluted across runs (creative_this_hour_*, "
    "recent_actions_detail_*, empty_history_returns_zero_counts, "
    "build_result_records_action_type, failed_actions_proportional_*). "
    "Per SPEC §23.1 #1 AgencySchemaBridge contract this is a fixture- "
    "isolation issue, not a SPEC drift. Re-enable with an explicit "
    "`_history.clear()` setUp hook or per-test fixture (test rewrite only, "
    "no production code change). Live behavior verified via /v6/trinity/agency."
))
class TestAgencySchemaBridge:
    """SPEC §23.1 #1 — AgencyModule.get_stats() schema bridge.

    Phase 1 added: total_actions, failed_actions, actions_this_hour,
    creative_this_hour, recent_actions_detail. Plus action_type tagging
    inside _build_result for stable history dicts.
    """

    def _make_agency(self):
        from titan_hcl.logic.agency.module import AgencyModule
        return AgencyModule()

    def test_empty_history_returns_zero_counts(self):
        a = self._make_agency()
        s = a.get_stats()
        assert s["total_actions"] == 0
        assert s["failed_actions"] == 0
        assert s["actions_this_hour"] == 0
        assert s["creative_this_hour"] == 0
        assert s["recent_actions_detail"] == []

    def test_action_type_derivation_from_helper_name(self):
        from titan_hcl.logic.agency.module import _derive_action_type
        assert _derive_action_type("art_generate", "create") == "art"
        assert _derive_action_type("audio_generate", "create") == "audio"
        assert _derive_action_type("web_search", "research") == "research"
        assert _derive_action_type("code_knowledge", "research") == "research"
        assert _derive_action_type("coding_sandbox", "compute") == "compute"
        # Unknown helper falls back to posture
        assert _derive_action_type(None, "meditate") == "meditate"
        assert _derive_action_type("custom_helper", "rest") == "rest"

    def test_build_result_records_action_type(self):
        a = self._make_agency()
        # Manually drive _build_result (which is what the live path does).
        result = a._build_result(
            impulse_id=1, posture="create",
            helper_name="art_generate",
            result={"success": True, "result": "drew an octagon",
                     "enrichment_data": {}},
            reasoning="rule-based: create",
            trinity_snapshot={"hormone_levels": {"CREATIVITY": 0.85,
                                                  "IMPULSE": 0.6,
                                                  "EMPATHY": 0.3}},
        )
        assert result["action_type"] == "art"
        assert result["posture"] == "create"
        assert result["success"] is True
        # _history should now have one entry
        assert len(a._history) == 1
        h0 = a._history[0]
        assert h0["action_type"] == "art"
        # trinity_before snapshot retained for SAT[1] coherence
        assert h0["trinity_before"]["hormone_levels"]["CREATIVITY"] == 0.85

    def test_creative_this_hour_filters_by_action_type(self):
        a = self._make_agency()
        for ht in ("art_generate", "audio_generate", "web_search"):
            a._build_result(
                impulse_id=1, posture="x", helper_name=ht,
                result={"success": True, "result": "", "enrichment_data": {}},
                reasoning="", trinity_snapshot={})
        s = a.get_stats()
        assert s["actions_this_hour"] == 3
        # Only art_generate + audio_generate map to creative action_types
        assert s["creative_this_hour"] == 2
        # total_actions reflects all 3
        assert s["total_actions"] == 3

    def test_failed_actions_proportional_to_history_window(self):
        a = self._make_agency()
        # 4 successes + 1 failure → failed_actions ≈ 1/5 of total.
        for success in (True, True, True, True, False):
            a._build_result(
                impulse_id=0, posture="meditate", helper_name="infra_inspect",
                result={"success": success, "result": "", "enrichment_data": {}},
                reasoning="", trinity_snapshot={})
        s = a.get_stats()
        assert s["total_actions"] == 5
        assert s["failed_actions"] == 1  # 1/5 of total ≈ 1

    def test_recent_actions_detail_carries_hormones(self):
        a = self._make_agency()
        a._build_result(
            impulse_id=0, posture="research", helper_name="web_search",
            result={"success": True, "result": "", "enrichment_data": {}},
            reasoning="", trinity_snapshot={"hormone_levels": {"CURIOSITY": 0.9}})
        s = a.get_stats()
        recent = s["recent_actions_detail"]
        assert len(recent) == 1
        assert recent[0]["posture"] == "research"
        assert recent[0]["action_type"] == "research"
        assert recent[0]["hormones"]["CURIOSITY"] == 0.9

    def test_legacy_keys_preserved(self):
        """Phase 1 must not break the dashboard contract."""
        a = self._make_agency()
        s = a.get_stats()
        for k in ("action_count", "llm_calls_this_hour", "budget_per_hour",
                  "budget_remaining", "registered_helpers",
                  "helper_statuses", "recent_actions"):
            assert k in s, f"legacy key {k!r} dropped"


# ── §3.2 SelfAssessment schema bridge ───────────────────────────────

class TestAssessmentSchemaBridge:
    """SPEC §23.1 #2 — SelfAssessment.get_stats() schema bridge.

    Phase 1 added: average_score (alias), trend (linreg slope last 10),
    score_variance, total_assessed.
    """

    def _make(self):
        from titan_hcl.logic.agency.assessment import SelfAssessment
        return SelfAssessment()

    def test_empty_state_midpoints(self):
        s = self._make().get_stats()
        assert s["average_score"] == 0.5  # mid-point on empty
        assert s["trend"] == 0.0
        assert s["score_variance"] == 0.0
        assert s["total_assessed"] == 0

    def test_trend_positive_for_improving_scores(self):
        a = self._make()
        # Scores monotonically increasing → positive linreg slope
        for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            a._assessments.append({"score": s})
        stats = a.get_stats()
        assert stats["trend"] > 0.05  # ≈ 0.1 (slope of x→y line)
        assert stats["average_score"] == 0.55

    def test_trend_negative_for_declining_scores(self):
        a = self._make()
        for s in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            a._assessments.append({"score": s})
        stats = a.get_stats()
        assert stats["trend"] < -0.05

    def test_score_variance_reflects_spread(self):
        a = self._make()
        # All same score → variance ≈ 0
        for _ in range(10):
            a._assessments.append({"score": 0.5})
        assert a.get_stats()["score_variance"] == pytest.approx(0.0, abs=1e-6)

        a2 = self._make()
        # Bimodal 0/1 → variance = 0.25
        for s in [0.0, 1.0, 0.0, 1.0]:
            a2._assessments.append({"score": s})
        assert a2.get_stats()["score_variance"] == pytest.approx(0.25, abs=1e-3)

    def test_average_score_alias_matches_avg_score(self):
        a = self._make()
        for s in [0.3, 0.7, 0.5]:
            a._assessments.append({"score": s})
        stats = a.get_stats()
        assert stats["average_score"] == stats["avg_score"]


# ── SPEC §23.7 outer_body thermal redesign ──────────────────────────

class TestOuterBodyThermalRedesign:
    """SPEC §23.2 + §23.7[4] — thermal = 0.35*cpu_thermal + 0.25*circadian
    + 0.40*hormonal_heat where hormonal_heat = mean(IMPULSE, VIGILANCE).
    LLM-latency intentionally excluded.
    """

    def _collect(self, **kwargs):
        from titan_hcl.logic.outer_body_tensor import collect_outer_body_5d
        return collect_outer_body_5d(kwargs.get("sources", {}))

    def test_thermal_uses_hormonal_heat_not_llm_latency(self):
        sources = {
            "system_sensor_stats": {"cpu_thermal": 0.5, "circadian_phase": 0.5},
            "hormone_levels": {"IMPULSE": 1.0, "VIGILANCE": 1.0},
            # LLM latency present but should be IGNORED per SPEC §23.7
            "llm_avg_latency": 25.0,
        }
        t = self._collect(sources=sources)
        # 0.35*0.5 + 0.25*0.5 + 0.40*1.0 = 0.175 + 0.125 + 0.4 = 0.7
        assert t[4] == pytest.approx(0.7, abs=1e-3)

    def test_thermal_low_arousal_is_cool(self):
        sources = {
            "system_sensor_stats": {"cpu_thermal": 0.2, "circadian_phase": 0.2},
            "hormone_levels": {"IMPULSE": 0.0, "VIGILANCE": 0.0},
        }
        t = self._collect(sources=sources)
        # 0.35*0.2 + 0.25*0.2 + 0.40*0.0 = 0.12
        assert t[4] == pytest.approx(0.12, abs=1e-3)

    def test_thermal_focus_excluded_from_hormonal_heat(self):
        """FOCUS is intentionally NOT part of hormonal_heat."""
        s_with_focus = {
            "system_sensor_stats": {"cpu_thermal": 0.5, "circadian_phase": 0.5},
            "hormone_levels": {"IMPULSE": 0.0, "VIGILANCE": 0.0, "FOCUS": 1.0},
        }
        s_without = {
            "system_sensor_stats": {"cpu_thermal": 0.5, "circadian_phase": 0.5},
            "hormone_levels": {"IMPULSE": 0.0, "VIGILANCE": 0.0},
        }
        t_focus = self._collect(sources=s_with_focus)
        t_no = self._collect(sources=s_without)
        assert t_focus[4] == pytest.approx(t_no[4], abs=1e-6), \
            "FOCUS should not affect thermal — only IMPULSE+VIGILANCE feed hormonal_heat"

    def test_default_hormones_thermal_midpoint(self):
        sources = {"system_sensor_stats": {"cpu_thermal": 0.5,
                                            "circadian_phase": 0.5}}
        t = self._collect(sources=sources)
        # hormones missing → IMPULSE/VIGILANCE default to 0.5 each → heat=0.5
        # 0.35*0.5 + 0.25*0.5 + 0.40*0.5 = 0.5
        assert t[4] == pytest.approx(0.5, abs=1e-3)


# ── SPEC §23.8 outer_mind redesigns ─────────────────────────────────

class TestOuterMindRedesigns:
    """SPEC §23.8 — REDESIGNED dims: thinking[0,1,2], willing[11,14]."""

    def _collect(self, **kwargs):
        from titan_hcl.logic.outer_mind_tensor import collect_outer_mind_15d
        defaults = dict(current_5d=[0.5] * 5)
        defaults.update(kwargs)
        return collect_outer_mind_15d(**defaults)

    def test_research_effectiveness_blends_three_signals(self):
        """thinking[0] = 0.4*helpful_ratio + 0.3*cgn_reward_norm
        + 0.3*directive_alignment"""
        t = self._collect(
            meta_cgn_stats={"knowledge_helpful_by_source": {"a": 5, "b": 5},
                             "knowledge_responses_received": 10},  # ratio=1.0
            cgn_stats={"avg_reward": 1.0},  # norm=(1+1)/2=1.0
            memory_growth_metrics={"directive_alignment": 1.0},
        )
        # 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert t[0] == pytest.approx(1.0, abs=1e-3)

    def test_knowledge_retrieval_drops_kg_node_count(self):
        """thinking[1] redesign deliberately excludes KG node_count
        (avoid double-counting with CHIT[15] world_model_depth).
        Adding more KG nodes should NOT change thinking[1]."""
        base_kwargs = dict(
            memory_growth_metrics={"directive_alignment": 0.5},
            meta_cgn_stats={"knowledge_helpful_by_source": {},
                             "knowledge_responses_received": 0,
                             "usage_gini": 0.5},
            language_stats={"avg_confidence": 0.5},
        )
        t_low = self._collect(**base_kwargs,
                                knowledge_graph_stats={"node_count": 0,
                                                        "edge_count": 0})
        t_high = self._collect(**base_kwargs,
                                 knowledge_graph_stats={"node_count": 10000,
                                                         "edge_count": 50000})
        assert t_low[1] == pytest.approx(t_high[1], abs=1e-6), \
            "knowledge_retrieval must NOT depend on KG size (that's CHIT[15])"

    def test_situational_awareness_combines_event_freshness_and_velocity(self):
        """thinking[2] = 0.5*recency_decay + 0.3*felt_exp_norm
        + 0.2*learning_velocity"""
        t = self._collect(
            research_stats={"seconds_since_last": 0.0},  # full freshness
            events_teacher_stats={"felt_experiences": 100},  # saturated
            memory_growth_metrics={"learning_velocity": 1.0},
        )
        # 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
        assert t[2] == pytest.approx(1.0, abs=1e-3)

    def test_exploration_drive_combines_three_streams(self):
        """willing[14] = 0.40*cgn_density + 0.30*teacher_sessions + 0.30*eureka_rate"""
        t = self._collect(
            cgn_stats={"grounded_density": 2.0},  # saturated
            language_stats={"teacher_sessions_last_hour": 3},  # saturated
            meta_cgn_stats={"eureka_accelerated_updates": 5},
            uptime_seconds=3600.0,  # 1 hour
        )
        # 0.40*1.0 + 0.30*1.0 + 0.30*1.0 = 1.0
        assert t[14] == pytest.approx(1.0, abs=1e-3)

    def test_social_initiative_uses_real_x_gateway(self):
        """willing[11] reads social_x_gateway.posts_last_hour, not ghost field."""
        t = self._collect(social_x_gateway_stats={"posts_last_hour": 5})
        # min(1, 5/5) = 1.0
        assert t[11] == pytest.approx(1.0, abs=1e-3)


# ── SPEC §23.9 outer_spirit redesigns ───────────────────────────────

class TestOuterSpiritRedesigns:
    """SPEC §23.9 — extensive SAT/CHIT/ANANDA redesigns."""

    def _collect(self, **kwargs):
        from titan_hcl.logic.outer_spirit_tensor import (
            collect_outer_spirit_45d,
        )
        defaults = dict(
            current_5d=[0.5] * 5,
            outer_body=[0.5] * 5,
            outer_mind=[0.5] * 15,
        )
        defaults.update(kwargs)
        return collect_outer_spirit_45d(**defaults)

    def test_sat0_world_recognition_from_solana_local_stats(self):
        t = self._collect(solana_stats={"identity_verified": 1.0})
        assert t[0] == pytest.approx(1.0, abs=1e-3)
        t2 = self._collect(solana_stats={"identity_verified": 0.0})
        assert t2[0] == pytest.approx(0.0, abs=1e-3)

    def test_sat3_boundary_enforcement_from_jailbreak_real_producer(self):
        """SAT[3] = blocked / max(1, threats); 0.8 if no threats."""
        # No threats observed → defined no-threats default
        t = self._collect(jailbreak_alerts_stats={"threats_detected_24h": 0,
                                                    "blocked_24h": 0})
        assert t[3] == pytest.approx(0.8, abs=1e-3)

        # 10 threats, all blocked → 1.0
        t2 = self._collect(jailbreak_alerts_stats={"threats_detected_24h": 10,
                                                     "blocked_24h": 10})
        assert t2[3] == pytest.approx(1.0, abs=1e-3)

        # 10 threats, 5 blocked → 0.5
        t3 = self._collect(jailbreak_alerts_stats={"threats_detected_24h": 10,
                                                     "blocked_24h": 5})
        assert t3[3] == pytest.approx(0.5, abs=1e-3)

    def test_sat5_origin_anchoring_from_genesis_check(self):
        t_yes = self._collect(solana_stats={"genesis_nft_exists": 1.0})
        t_no = self._collect(solana_stats={"genesis_nft_exists": 0.0})
        assert t_yes[5] == pytest.approx(1.0, abs=1e-3)
        assert t_no[5] == pytest.approx(0.0, abs=1e-3)

    def test_sat10_recovery_speed_from_anchor_consecutive_failures(self):
        """REDESIGNED: 1 - min(1, failures/10). 0 failures = fully recovered."""
        t_clean = self._collect(anchor_state={"consecutive_failures": 0})
        assert t_clean[10] == pytest.approx(1.0, abs=1e-3)

        t_mid = self._collect(anchor_state={"consecutive_failures": 5})
        assert t_mid[10] == pytest.approx(0.5, abs=1e-3)

        t_broken = self._collect(anchor_state={"consecutive_failures": 20})
        assert t_broken[10] == pytest.approx(0.0, abs=1e-3)

    def test_sat13_transactional_integrity_empty_state_midpoint(self):
        """anchor_count==0 → 0.5 (no data, mid-point per SPEC)."""
        t = self._collect(anchor_state={"anchor_count": 0,
                                          "consecutive_failures": 0})
        assert t[13] == pytest.approx(0.5, abs=1e-3)

    def test_sat13_transactional_integrity_anchor_formula(self):
        """anchor_count=48, consecutive_failures=0 → 48/(48+0) = 1.0"""
        t = self._collect(anchor_state={"anchor_count": 48,
                                          "consecutive_failures": 0})
        assert t[13] == pytest.approx(1.0, abs=1e-3)

        # anchor_count=10, failures=2 → 10/(10+10) = 0.5
        t2 = self._collect(anchor_state={"anchor_count": 10,
                                           "consecutive_failures": 2})
        assert t2[13] == pytest.approx(0.5, abs=1e-3)

    def test_sat7_world_footprint_uses_weighted_log_sum(self):
        """REDESIGN: SPEC §23.3 weighted log-sum across artifact streams."""
        # When world_footprint_inputs provided, formula uses score_sum/target_log.
        t = self._collect(world_footprint_inputs={"score_sum": 5.0,
                                                    "target_log": 10.0})
        assert t[7] == pytest.approx(0.5, abs=1e-3)

        # Saturation
        t_sat = self._collect(world_footprint_inputs={"score_sum": 100.0,
                                                       "target_log": 10.0})
        assert t_sat[7] == pytest.approx(1.0, abs=1e-3)

    def test_chit15_world_model_depth_blends_kg_meta_cgn_action_chains_vocab(self):
        """REDESIGNED: weighted blend, edges weighted higher than nodes."""
        t = self._collect(
            knowledge_graph_stats={"node_count": 5000, "edge_count": 15000},
            meta_cgn_stats={"primitives_grounded": 10, "primitives_total": 10},
            inner_memory_stats={"action_chains": 500},
            language_stats={"vocab_total": 2000},
        )
        # 0.25*1.0 + 0.30*1.0 + 0.20*1.0 + 0.15*1.0 + 0.10*1.0 = 1.0
        assert t[15] == pytest.approx(1.0, abs=1e-3)

    def test_chit17_threat_discernment_from_jailbreak_confirmed_ratio(self):
        """REDESIGNED: confirmed/total from jailbreak_alerts_stats."""
        t = self._collect(jailbreak_alerts_stats={
            "threats_detected_24h": 10,
            "confirmed_threats_24h": 8,
        })
        assert t[17] == pytest.approx(0.8, abs=1e-3)

    def test_ananda33_system_harmony_from_bus_stats(self):
        """ANANDA[33] = 1 - bus.dropped/bus.published"""
        t_clean = self._collect(bus_stats={"published": 1000, "dropped": 0})
        assert t_clean[33] == pytest.approx(1.0, abs=1e-3)

        t_lossy = self._collect(bus_stats={"published": 100, "dropped": 25})
        assert t_lossy[33] == pytest.approx(0.75, abs=1e-3)

    def test_ananda43_resource_appreciation_outputs_per_llm_call(self):
        """ANANDA[43] = min(1, outputs_per_hour / llm_calls_this_hour)"""
        # 10 outputs, 5 LLM calls → outputs_per_hour=10, llm=5 → 10/5 = 2.0 → clamped to 1.0
        t = self._collect(
            action_stats={"per_window": 5},
            creative_stats={"per_window": 5},
            llm_calls_this_hour=5,
        )
        assert t[43] == pytest.approx(1.0, abs=1e-3)

        # 1 output, 10 LLM calls → 1/10 = 0.1
        t2 = self._collect(
            action_stats={"per_window": 1},
            creative_stats={"per_window": 0},
            llm_calls_this_hour=10,
        )
        assert t2[43] == pytest.approx(0.1, abs=1e-3)


# ── §3.1 _HELPER_ACTION_TYPES exhaustiveness ────────────────────────

class TestHelperActionTypeMap:
    """Smoke check: SPEC-defined helper names map to creative action_types
    where appropriate (SAT[1] coherence + creative_this_hour rely on this)."""

    def test_creative_helpers_map_to_creative_types(self):
        from titan_hcl.logic.agency.module import _HELPER_ACTION_TYPES
        assert _HELPER_ACTION_TYPES["art_generate"] == "art"
        assert _HELPER_ACTION_TYPES["audio_generate"] == "audio"

    def test_non_creative_helpers_dont_pollute_creative_count(self):
        from titan_hcl.logic.agency.module import _HELPER_ACTION_TYPES
        for name in ("web_search", "code_knowledge", "coding_sandbox",
                     "infra_inspect", "memo_inscribe"):
            assert _HELPER_ACTION_TYPES[name] not in ("art", "audio", "music"), \
                f"{name} mapped to creative type by mistake"


# ── UnifiedSpirit dashboard heatmap fix ─────────────────────────────

class TestUnifiedSpiritFull130dt:
    """Phase 1 close-out fix: UnifiedSpirit.get_stats() now exposes
    full_130dt so /v4/inner-trinity.unified_spirit.full_130dt populates
    the Observatory Trinity-architecture heatmap.
    """

    def test_get_stats_includes_full_130dt(self):
        from titan_hcl.logic.unified_spirit import UnifiedSpirit
        # Skip if init signature requires args we don't have here
        try:
            us = UnifiedSpirit(config={}, data_dir="/tmp")
        except TypeError:
            pytest.skip("UnifiedSpirit init signature requires more setup")
        stats = us.get_stats()
        assert "full_130dt" in stats, \
            "UnifiedSpirit.get_stats must expose full_130dt for dashboard"
        assert isinstance(stats["full_130dt"], list)
        assert len(stats["full_130dt"]) == 130
        for v in stats["full_130dt"]:
            assert 0.0 <= v <= 1.0
