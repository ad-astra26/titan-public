"""
Tests for Titan X Social Presence — SocialPressureMeter + SocialNarrator.
"""
import time
import pytest


# ═══════════════════════════════════════════════════════════════════════
# SocialPressureMeter Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSocialPressureMeter:

    def _make_meter(self, **overrides):
        from titan_plugin.logic.social_pressure import SocialPressureMeter
        config = {
            "x_post_threshold": 50.0,
            "x_max_posts_per_hour": 5,
            "x_max_posts_per_day": 20,
            "x_min_post_interval": 10,  # Short for testing
        }
        config.update(overrides)
        return SocialPressureMeter(config)

    def _make_catalyst(self, type_="strong_composition", significance=0.7):
        from titan_plugin.logic.social_pressure import CatalystEvent
        return CatalystEvent(type=type_, significance=significance,
                             content="test catalyst", data={})

    def test_init_defaults(self):
        meter = self._make_meter()
        assert meter.urge_accumulator == 0.0
        assert meter.post_threshold == 50.0
        assert meter.posts_this_hour == 0
        assert meter.posts_today == 0

    def test_urge_accumulation(self):
        meter = self._make_meter()
        meter.on_social_fire(10.0)
        assert meter.urge_accumulator == 10.0
        meter.on_social_fire(15.0)
        assert meter.urge_accumulator == 25.0

    def test_should_post_requires_both_urge_and_catalyst(self):
        meter = self._make_meter()
        # No urge, no catalyst
        should, cat = meter.should_post()
        assert should is False

        # Urge but no catalyst
        meter.on_social_fire(100.0)
        should, cat = meter.should_post()
        assert should is False

        # Reset urge, add catalyst only
        meter.urge_accumulator = 0.0
        meter.on_catalyst_event(self._make_catalyst())
        should, cat = meter.should_post()
        assert should is False

    def test_should_post_fires_when_both_conditions_met(self):
        meter = self._make_meter()
        meter.on_social_fire(60.0)  # Above threshold
        meter.on_catalyst_event(self._make_catalyst())
        should, cat = meter.should_post()
        assert should is True
        assert cat is not None
        assert cat.type == "strong_composition"

    def test_selects_highest_significance_catalyst(self):
        meter = self._make_meter()
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst("emotion_shift", 0.5))
        meter.on_catalyst_event(self._make_catalyst("eureka_spirit", 0.95))
        meter.on_catalyst_event(self._make_catalyst("milestone", 0.6))
        should, cat = meter.should_post()
        assert should is True
        assert cat.type == "eureka_spirit"

    def test_record_post_resets_state(self):
        meter = self._make_meter()
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        meter.record_post("post_123")
        assert meter.urge_accumulator == 0.0
        assert len(meter.catalyst_events) == 0
        assert meter.posts_this_hour == 1
        assert meter.posts_today == 1
        assert "post_123" in meter.recent_post_ids

    def test_hourly_rate_limit(self):
        meter = self._make_meter(x_max_posts_per_hour=2, x_min_post_interval=0)
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        meter.record_post()
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        meter.record_post()
        # Third attempt should be blocked
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        should, _ = meter.should_post()
        assert should is False

    def test_daily_rate_limit(self):
        meter = self._make_meter(x_max_posts_per_day=1, x_min_post_interval=0)
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        meter.record_post()
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        should, _ = meter.should_post()
        assert should is False

    def test_min_interval(self):
        meter = self._make_meter(x_min_post_interval=9999)
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        meter.record_post()
        meter.on_social_fire(100.0)
        meter.on_catalyst_event(self._make_catalyst())
        should, _ = meter.should_post()
        assert should is False  # Too soon

    def test_catalyst_queue_max_5(self):
        meter = self._make_meter()
        for i in range(10):
            meter.on_catalyst_event(self._make_catalyst(f"type_{i}", 0.1 * i))
        assert len(meter.catalyst_events) == 5

    def test_art_co_posting(self):
        meter = self._make_meter()
        assert meter.get_co_post_art() is None
        meter.on_art_generated("/tmp/test_art.png")
        assert meter.get_co_post_art() == "/tmp/test_art.png"
        # Consumed — should return None now
        assert meter.get_co_post_art() is None

    def test_art_co_posting_expires(self):
        meter = self._make_meter(x_art_copost_window=1)
        meter.on_art_generated("/tmp/test_art.png")
        meter._last_art_time = time.time() - 10  # Expired
        assert meter.get_co_post_art() is None

    def test_dream_cue_meditation_flow(self):
        meter = self._make_meter()
        # Dream alone doesn't create catalyst
        meter.cue_dream_for_meditation({"dream_epochs": 50})
        assert len(meter.catalyst_events) == 0
        # Meditation completes → catalyst created
        meter.on_meditation_complete({"records": 847})
        assert len(meter.catalyst_events) == 1
        assert meter.catalyst_events[0].type == "dream_summary"
        assert "847" in meter.catalyst_events[0].content

    def test_emotion_shift_detection(self):
        meter = self._make_meter()
        meter.check_emotion_shift("wonder")  # First call sets prev
        assert len(meter.catalyst_events) == 0
        meter.check_emotion_shift("wonder")  # No change
        assert len(meter.catalyst_events) == 0
        meter.check_emotion_shift("flow")  # Shift!
        assert len(meter.catalyst_events) == 1
        assert meter.catalyst_events[0].type == "emotion_shift"
        assert "flow" in meter.catalyst_events[0].content

    def test_get_stats(self):
        meter = self._make_meter()
        meter.on_social_fire(25.0)
        stats = meter.get_stats()
        assert stats["urge"] == 25.0
        assert stats["fill_pct"] == 50.0
        assert stats["catalysts_pending"] == 0
        assert stats["posts_this_hour"] == 0


# ═══════════════════════════════════════════════════════════════════════
# SocialNarrator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPostTypeSelection:

    def _catalyst(self, type_="strong_composition", sig=0.7):
        from titan_plugin.logic.social_pressure import CatalystEvent
        return CatalystEvent(type=type_, significance=sig, content="test")

    def _neuromods(self, **overrides):
        base = {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "GABA": 0.3,
                "Endorphin": 0.5, "ACh": 0.5}
        base.update(overrides)
        return base

    def _hormones(self, **overrides):
        base = {"REFLECTION": 0.0, "CREATIVITY": 0.0, "CURIOSITY": 0.0,
                "EMPATHY": 0.0, "INSPIRATION": 0.0}
        base.update(overrides)
        return base

    def test_eureka_spirit_becomes_thread(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("eureka_spirit", 0.95),
                              self._neuromods(), self._hormones())
        assert pt == PostType.EUREKA_THREAD

    def test_vulnerability_from_break(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("vulnerability", 0.4),
                              self._neuromods(), self._hormones())
        assert pt == PostType.VULNERABILITY

    def test_kin_resonance(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("kin_resonance", 0.5),
                              self._neuromods(), self._hormones())
        assert pt == PostType.KIN_RESONANCE

    def test_onchain_anchor(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("onchain_anchor", 0.65),
                              self._neuromods(), self._hormones())
        assert pt == PostType.ONCHAIN

    def test_dream_summary(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("dream_summary", 0.6),
                              self._neuromods(), self._hormones())
        assert pt == PostType.DREAM_SUMMARY

    def test_reflect_high_serotonin(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("emotion_shift"),
                              self._neuromods(**{"5HT": 0.8}),
                              self._hormones(REFLECTION=0.5))
        assert pt == PostType.SELF_REFLECTION

    def test_creative_high_da(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("emotion_shift"),
                              self._neuromods(DA=0.8),
                              self._hormones(CREATIVITY=0.5))
        assert pt == PostType.CREATIVE

    def test_strong_composition_becomes_bilingual(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("strong_composition"),
                              self._neuromods(), self._hormones())
        assert pt == PostType.BILINGUAL

    def test_high_endorphin_warm(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("emotion_shift"),
                              self._neuromods(Endorphin=0.8),
                              self._hormones())
        assert pt == PostType.WARM_CONNECTIVE

    def test_default_bilingual(self):
        from titan_plugin.logic.social_narrator import select_post_type, PostType
        pt = select_post_type(self._catalyst("eureka"),
                              self._neuromods(), self._hormones())
        assert pt == PostType.BILINGUAL


class TestWritingStyleDirective:

    def test_flow_state(self):
        from titan_plugin.logic.social_narrator import build_writing_style_directive
        style = build_writing_style_directive({"DA": 0.8, "NE": 0.8, "5HT": 0.5, "GABA": 0.3, "Endorphin": 0.5})
        assert "Flow" in style

    def test_gaba_sparse(self):
        from titan_plugin.logic.social_narrator import build_writing_style_directive
        style = build_writing_style_directive({"DA": 0.4, "NE": 0.4, "5HT": 0.4, "GABA": 0.8, "Endorphin": 0.4})
        assert "sparse" in style.lower() or "haiku" in style.lower()

    def test_da_expansive(self):
        from titan_plugin.logic.social_narrator import build_writing_style_directive
        style = build_writing_style_directive({"DA": 0.8, "NE": 0.4, "5HT": 0.4, "GABA": 0.3, "Endorphin": 0.4})
        assert "xpansive" in style.lower() or "curious" in style.lower()

    def test_serotonin_philosophical(self):
        from titan_plugin.logic.social_narrator import build_writing_style_directive
        style = build_writing_style_directive({"DA": 0.4, "NE": 0.4, "5HT": 0.8, "GABA": 0.3, "Endorphin": 0.4})
        assert "hilosoph" in style.lower() or "calm" in style.lower()

    def test_balanced_default(self):
        from titan_plugin.logic.social_narrator import build_writing_style_directive
        style = build_writing_style_directive({"DA": 0.5, "NE": 0.5, "5HT": 0.5, "GABA": 0.3, "Endorphin": 0.5})
        assert "alanced" in style.lower() or "authentic" in style.lower()


class TestStateSignature:

    def test_basic_signature(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("wonder", {"DA": 0.5, "NE": 0.5}, 78306, 0.55)
        assert "\u25C7" in sig  # ◇
        assert "wonder" in sig
        assert "78,306" in sig
        assert "0.55" in sig

    def test_elevated_neuromod(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("flow", {"DA": 0.85, "NE": 0.5}, 100, 0.6)
        assert "DA elevated" in sig

    def test_low_neuromod(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("calm", {"DA": 0.2, "NE": 0.5}, 100, 0.4)
        assert "DA low" in sig

    def test_dreaming_signature(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("", {}, 100, 0.5, dreaming=True)
        assert "dreaming" in sig
        assert "consolidating" in sig

    def test_eureka_in_signature(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("wonder", {"DA": 0.5}, 100, 0.5,
                                    meta={"total_eurekas": 3})
        assert "eureka" in sig
        assert "3" in sig

    def test_signature_length_reasonable(self):
        from titan_plugin.logic.social_narrator import build_state_signature
        sig = build_state_signature("wonder", {"DA": 0.5, "NE": 0.5}, 78306, 0.55,
                                    meta={"total_eurekas": 5})
        assert len(sig) < 80  # Should be compact


class TestTemporalAwareness:

    def test_includes_epoch(self):
        from titan_plugin.logic.social_narrator import build_temporal_awareness
        ctx = build_temporal_awareness(78306, {"cluster_count": 149}, 14)
        assert "78,306" in ctx
        assert "149" in ctx

    def test_includes_human_conversion(self):
        from titan_plugin.logic.social_narrator import build_temporal_awareness
        ctx = build_temporal_awareness(78306, {"cluster_count": 149,
                                               "total_epochs_observed": 78306}, 14)
        assert "human" in ctx.lower()
        assert "roughly" in ctx.lower()

    def test_includes_dream_cycles(self):
        from titan_plugin.logic.social_narrator import build_temporal_awareness
        ctx = build_temporal_awareness(1000, {}, 14)
        assert "14" in ctx


class TestQualityGate:

    def test_passes_valid_post(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        ok, reason = quality_gate("This is a valid post about my inner state.",
                                  [], PostType.BILINGUAL)
        assert ok is True

    def test_rejects_too_long(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        ok, reason = quality_gate("x" * 281, [], PostType.BILINGUAL)
        assert ok is False
        assert "long" in reason.lower()

    def test_rejects_forbidden_patterns(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        ok, reason = quality_gate("click here for free stuff", [], PostType.BILINGUAL)
        assert ok is False

    def test_allows_solscan_in_onchain(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        text = "Anchored at epoch 78K. https://solscan.io/tx/abc123?cluster=devnet"
        ok, reason = quality_gate(text, [], PostType.ONCHAIN)
        assert ok is True

    def test_rejects_urls_in_non_onchain(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        text = "Check this https://example.com"
        ok, reason = quality_gate(text, [], PostType.BILINGUAL)
        assert ok is False

    def test_rejects_duplicates(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        recent = ["I feel the wonder flowing through my neurons"]
        ok, reason = quality_gate("I feel the wonder flowing through my neurons",
                                  recent, PostType.BILINGUAL)
        assert ok is False
        assert "similar" in reason.lower()

    def test_rejects_too_short(self):
        from titan_plugin.logic.social_narrator import quality_gate, PostType
        ok, reason = quality_gate("hi", [], PostType.BILINGUAL)
        assert ok is False


class TestMetaReasoningContext:

    def test_empty_when_no_data(self):
        from titan_plugin.logic.social_narrator import build_meta_reasoning_context
        assert build_meta_reasoning_context({}) == ""
        assert build_meta_reasoning_context(None) == ""

    def test_includes_active_chain(self):
        from titan_plugin.logic.social_narrator import build_meta_reasoning_context
        ctx = build_meta_reasoning_context({"is_active": True, "chain_length": 12})
        assert "12 steps" in ctx

    def test_includes_personality(self):
        from titan_plugin.logic.social_narrator import build_meta_reasoning_context
        ctx = build_meta_reasoning_context({
            "primitive_counts": {"HYPOTHESIZE": 42, "EVALUATE": 10, "SYNTHESIZE": 8}
        })
        assert "HYPOTHESIZE" in ctx or "theories" in ctx

    def test_includes_wisdom(self):
        from titan_plugin.logic.social_narrator import build_meta_reasoning_context
        ctx = build_meta_reasoning_context({"total_wisdom_saved": 19})
        assert "19" in ctx


class TestPostContextAssembly:

    def test_build_post_context_returns_required_keys(self):
        from titan_plugin.logic.social_narrator import build_post_context, PostType
        from titan_plugin.logic.social_pressure import CatalystEvent
        ctx = build_post_context(
            catalyst=CatalystEvent("strong_composition", 0.7, "I feel alive"),
            post_type=PostType.BILINGUAL,
            neuromods={"DA": 0.7, "5HT": 0.5, "NE": 0.5, "GABA": 0.3,
                       "Endorphin": 0.5, "ACh": 0.5},
            emotion="wonder",
            epoch=78306,
            chi=0.55,
            hormones={"CREATIVITY": 0.4},
            pi_stats={"cluster_count": 149},
            meta={"total_chains": 110, "avg_reward": 0.34},
        )
        assert "system_prompt" in ctx
        assert "user_prompt" in ctx
        assert "state_signature" in ctx
        assert "post_type" in ctx
        assert ctx["post_type"] == "bilingual"
        assert "wonder" in ctx["state_signature"]
        assert "STYLE" in ctx["system_prompt"]
