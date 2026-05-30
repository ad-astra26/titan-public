"""Tests for Phase C-S9 social_worker extraction.

PLAN_microkernel_phase_c_s9_social_worker_extraction §2.10 (chunk 9J).
Covers chunks 9A (skeleton), 9B (gateway boot), 9C (meter), 9D (bus events),
9E (SHM slot), 9F (X_POST_PUBLISHED), 9G (recency boost), 9H (Observatory
route schema), 9I (catalyst dual-mode + D8 markers).

Polling-mode tests (chunks 9M-9P) deferred to next session.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time

import pytest

from titan_hcl import bus
from titan_hcl.bus_specs import MSG_SPECS


# ─── Bus event registration (chunks 9D + 9I consolidation) ──────────────

class TestBusEvents:
    """All 8 social_worker bus event types (after 9I 5→1 consolidation)."""

    SOCIAL_WORKER_EVENTS = (
        "KIN_SIGNAL", "SOCIAL_RECEIVED",
        "SOCIAL_CATALYST",  # the 1 generic event (chunk 9I consolidation)
        "X_POST_PUBLISHED", "SOCIAL_GRAPH_UPDATE",
        "MENTION_RECEIVED", "FELT_EXPERIENCE_CAPTURED",
        "ENGAGEMENT_SNAPSHOT_TAKEN",
    )

    def test_all_constants_present_in_bus_module(self):
        for evt in self.SOCIAL_WORKER_EVENTS:
            assert hasattr(bus, evt), f"bus.{evt} missing"
            assert getattr(bus, evt) == evt, \
                f"bus.{evt} has unexpected value"

    def test_all_specs_registered_in_msg_specs(self):
        for evt in self.SOCIAL_WORKER_EVENTS:
            assert evt in MSG_SPECS, f"{evt} missing from MSG_SPECS"

    def test_all_specs_priority_p3(self):
        # All social-tier events are P3 per bus_specs.py docstring + PLAN §11.2
        for evt in self.SOCIAL_WORKER_EVENTS:
            assert MSG_SPECS[evt].priority == 3, \
                f"{evt} priority {MSG_SPECS[evt].priority} != 3"

    def test_no_coalesce_on_event_types(self):
        # Each event is a distinct signal, not a state-update — coalesce=None
        for evt in self.SOCIAL_WORKER_EVENTS:
            assert MSG_SPECS[evt].coalesce is None, \
                f"{evt} unexpectedly has coalesce={MSG_SPECS[evt].coalesce}"

    def test_chunk_9I_consolidation_no_per_type_catalyst_events(self):
        # Phase 1 of chunk 9D registered 5 SOCIAL_CATALYST_*; chunk 9I
        # consolidated to ONE generic SOCIAL_CATALYST. Old per-type names
        # must NOT exist (would indicate revert).
        OLD_PER_TYPE_NAMES = (
            "SOCIAL_CATALYST_EUREKA", "SOCIAL_CATALYST_DREAM_SUMMARY",
            "SOCIAL_CATALYST_KIN_RESONANCE", "SOCIAL_CATALYST_ART_GENERATED",
            "SOCIAL_CATALYST_EMOTION_SHIFT",
        )
        for old in OLD_PER_TYPE_NAMES:
            assert not hasattr(bus, old), \
                f"{old} should have been removed in chunk 9I consolidation"
            assert old not in MSG_SPECS, \
                f"{old} should have been removed from MSG_SPECS in chunk 9I"


# ─── social_worker module skeleton (chunk 9A) ──────────────────────────

class TestSocialWorkerSkeleton:

    def test_main_entry_function_exists(self):
        from titan_hcl.modules.social_worker import social_worker_main
        assert callable(social_worker_main)

    def test_subscribe_topics_complete(self):
        from titan_hcl.modules.social_worker import (
            _SOCIAL_WORKER_SUBSCRIBE_TOPICS)
        # 11 topics after D-SPEC-67 v1.12.0 (HEAL_REQUEST added for
        # health_monitor_worker MVP — social_worker is the first owning
        # worker for the HEAL_REQUEST contract, handles
        # action="refresh_session" against the live SocialXGateway):
        # MODULE_SHUTDOWN, SAVE_NOW (lifecycle)
        # EXPRESSION_FIRED, SOCIAL_RECEIVED, KIN_SIGNAL (§4.C)
        # SOCIAL_CATALYST (1 generic, post-9I)
        # X_FORCE_POST (D-SPEC-66 v1.11.0 PLAN §1.5 — Maker force-post
        #               subscriber-side catalyst, was dead at
        #               spirit_worker.py:7995 under fleet-wide Phase C)
        # MENTION_RECEIVED, FELT_EXPERIENCE_CAPTURED, ENGAGEMENT_SNAPSHOT_TAKEN
        # HEAL_REQUEST (D-SPEC-67 v1.12.0 — health_monitor_worker plugin
        #               framework; social_worker first HEAL_REQUEST owner)
        # MODULE_PROBE_REQUEST (Phase 11 §11.I.3 probe handler — added with the
        #               probe dispatcher; test count had not been bumped)
        # QUERY_RESPONSE (rFP_haov_efficacy_closure §3E C3 — cross-insights reply
        #               carrying language's verified HAOV concepts for engage-bias)
        assert len(_SOCIAL_WORKER_SUBSCRIBE_TOPICS) == 13, \
            f"expected 13 topics, got {len(_SOCIAL_WORKER_SUBSCRIBE_TOPICS)}"
        assert bus.SOCIAL_CATALYST in _SOCIAL_WORKER_SUBSCRIBE_TOPICS
        assert bus.MODULE_SHUTDOWN in _SOCIAL_WORKER_SUBSCRIBE_TOPICS
        assert bus.X_FORCE_POST in _SOCIAL_WORKER_SUBSCRIBE_TOPICS
        assert bus.HEAL_REQUEST in _SOCIAL_WORKER_SUBSCRIBE_TOPICS


# ─── Catalyst event dispatch (chunks 9D + 9I) ──────────────────────────

class TestCatalystDispatch:

    def _make_meter(self):
        from titan_hcl.logic.social_pressure import SocialPressureMeter
        meter = SocialPressureMeter({"x_post_threshold": 50.0})
        meter._boot_time = 0  # bypass 30s boot-settle
        # Clear any restored state for deterministic testing
        meter.catalyst_events = []
        meter.urge_accumulator = 0.0
        return meter

    def test_handle_catalyst_event_appends_to_meter(self):
        from titan_hcl.modules.social_worker import _handle_catalyst_event
        meter = self._make_meter()
        _handle_catalyst_event(meter, {
            "type": "eureka_spirit", "significance": 0.9,
            "content": "test eureka", "data": {"k": "v"},
        })
        assert len(meter.catalyst_events) == 1
        cat = meter.catalyst_events[0]
        assert cat.type == "eureka_spirit"
        assert cat.significance == 0.9
        assert cat.content == "test eureka"
        assert cat.data == {"k": "v"}

    def test_handle_catalyst_event_uses_payload_type_not_event_name(self):
        # Chunk 9I: type is in payload, NOT in event name (5→1 consolidation)
        from titan_hcl.modules.social_worker import _handle_catalyst_event
        meter = self._make_meter()
        for catalyst_type in ("dream_summary", "kin_resonance",
                              "onchain_anchor", "vulnerability"):
            _handle_catalyst_event(meter, {
                "type": catalyst_type, "significance": 0.5,
                "content": "test", "data": {},
            })
        types_in_meter = [c.type for c in meter.catalyst_events]
        assert "dream_summary" in types_in_meter
        assert "kin_resonance" in types_in_meter
        assert "onchain_anchor" in types_in_meter
        assert "vulnerability" in types_in_meter

    def test_handle_catalyst_event_handles_missing_fields(self):
        from titan_hcl.modules.social_worker import _handle_catalyst_event
        meter = self._make_meter()
        # Missing significance + content + data — should default gracefully
        _handle_catalyst_event(meter, {"type": "test"})
        assert len(meter.catalyst_events) == 1
        cat = meter.catalyst_events[0]
        assert cat.type == "test"
        assert cat.significance == 0.5  # default

    def test_handle_catalyst_event_swallows_meter_failures(self):
        # Robustness: dispatcher must not crash on bad meter
        from titan_hcl.modules.social_worker import _handle_catalyst_event
        # None meter — should not raise
        _handle_catalyst_event(None, {"type": "test"})  # would AttributeError
        # ↑ actually our handler imports CatalystEvent and tries to call
        # meter.on_catalyst_event(...) — None.on_catalyst_event raises but
        # is caught by try/except. Test just verifies no propagation.


# ─── SHM slot publisher (chunk 9E) ─────────────────────────────────────

class TestSHMSlotPublisher:

    def test_publisher_module_imports(self):
        from titan_hcl.logic.social_x_state_publisher import (
            SocialXStatePublisher, SOCIAL_X_STATE_SLOT, SOCIAL_X_STATE_SPEC)
        assert SOCIAL_X_STATE_SLOT == "social_x_state"
        # 2026-05-12: bumped 8192 → 32768 after the chunk 9E initial cap was
        # observed at 24KB live on T1 (per PLAN §2.5 "raise to live_bytes ×
        # 1.1 after first deploy observation per the spirit_supplemental_state
        # 58KB-vs-8KB lesson"). 32KB cap covers production worst-case +
        # margin. Constant lives in SPEC_titan_architecture_constants.toml
        # since 2026-05-12 (restored alongside EXPRESSION_STATE_* in
        # 86e01868 follow-up after parallel-session regen sweep dropped it).
        assert SOCIAL_X_STATE_SPEC.payload_bytes == 32768
        assert SOCIAL_X_STATE_SPEC.schema_version == 1

    def test_publisher_class_attributes(self):
        from titan_hcl.logic.social_x_state_publisher import (
            SocialXStatePublisher)
        assert SocialXStatePublisher.slot_name == "social_x_state"
        assert SocialXStatePublisher.slot_spec is not None

    def test_compute_payload_returns_required_keys(self):
        from titan_hcl.logic.social_x_state_publisher import (
            SocialXStatePublisher)
        pub = SocialXStatePublisher.__new__(SocialXStatePublisher)
        # Don't call __init__ (touches SHM); test the pure compute method
        state_refs = {
            "titan_id": "T1",
            "is_canonical_poller": True,
            "boot_ts": time.time(),
            "social_pressure_meter": None,
            "social_x_gateway": None,
        }
        payload = pub._compute_payload(state_refs)
        required_keys = {"titan_id", "current_urge", "post_threshold",
                         "posts_this_hour", "posts_today",
                         "next_allowed_post_ts", "catalysts_pending",
                         "last_archetype_fired", "last_post_ts",
                         "recent_posts", "is_canonical_poller",
                         "boot_grace_remaining_s", "ts"}
        assert required_keys.issubset(payload.keys())
        assert payload["titan_id"] == "T1"
        assert payload["is_canonical_poller"] is True


# ─── Recency boost (chunk 9G) ──────────────────────────────────────────

class TestRecencyBoost:

    def _make_dispatcher_with_db(self, fake_actions: list[tuple]):
        """fake_actions: list of (titan_id, post_type, posted_at_ts) tuples."""
        from titan_hcl.logic.social_x.dispatcher import ArchetypeDispatcher
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE actions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              action_type TEXT, status TEXT, titan_id TEXT,
              post_type TEXT, posted_at REAL, created_at REAL
            )""")
        now = time.time()
        for titan_id, post_type, ts in fake_actions:
            conn.execute(
                "INSERT INTO actions(action_type, status, titan_id, post_type, "
                "posted_at, created_at) VALUES ('post','verified',?,?,?,?)",
                (titan_id, post_type, ts, ts))
        conn.commit()
        conn.close()

        class FakeGateway:
            REFLECTABLE_POST_TYPES = frozenset()
            REFLECTION_EXCLUDED_POST_TYPES = frozenset()
        d = ArchetypeDispatcher(gateway=FakeGateway(),
                                  social_x_db_path=tmp.name)
        return d, tmp.name

    def _ctx(self, titan_id="T1"):
        class C: pass
        c = C()
        c.titan_id = titan_id
        return c

    def test_never_fired_archetypes_get_max_boost(self):
        d, path = self._make_dispatcher_with_db([])  # empty DB
        ordered = d._boosted_priority_order(self._ctx("T1"))
        # All 9 archetypes should be in boosted (all never-fired) — sorted DESC
        # by boost. Tied at max, stable sort preserves PRIORITY_ORDER tie-break.
        assert len(ordered) == 9
        os.unlink(path)

    def test_recently_fired_archetypes_drop_to_baseline(self):
        # world_mirror fired 1h ago — boost ≈ 0.004 < threshold 0.5 → baseline
        # self_watching never fired — boost = max → boosted tier
        from titan_hcl.logic.social_x.dispatcher import PRIORITY_ORDER
        now = time.time()
        d, path = self._make_dispatcher_with_db([
            ("T1", "world_mirror", now - 3600),  # 1h ago
        ])
        ordered = d._boosted_priority_order(self._ctx("T1"))
        # world_mirror should be in baseline tail (last, since baseline preserves
        # PRIORITY_ORDER). self_watching (never fired) should be in boosted head.
        sw_idx = ordered.index("self_watching")
        wm_idx = ordered.index("world_mirror")
        assert sw_idx < wm_idx, \
            f"self_watching ({sw_idx}) should come before world_mirror ({wm_idx})"
        os.unlink(path)

    def test_falls_back_to_priority_order_on_db_error(self):
        from titan_hcl.logic.social_x.dispatcher import (
            ArchetypeDispatcher, PRIORITY_ORDER)
        class FakeGateway:
            REFLECTABLE_POST_TYPES = frozenset()
            REFLECTION_EXCLUDED_POST_TYPES = frozenset()
        d = ArchetypeDispatcher(gateway=FakeGateway(),
                                  social_x_db_path="/nonexistent/db")
        ordered = d._boosted_priority_order(self._ctx("T1"))
        assert ordered == PRIORITY_ORDER

    def test_no_titan_id_returns_baseline(self):
        from titan_hcl.logic.social_x.dispatcher import PRIORITY_ORDER
        d, path = self._make_dispatcher_with_db([])
        class C: pass
        c = C()
        c.titan_id = ""  # empty
        ordered = d._boosted_priority_order(c)
        assert ordered == PRIORITY_ORDER
        os.unlink(path)

    def test_configurable_thresholds(self):
        # User-tunable in [social_x] config — chunk 9G + PLAN §11.3
        from titan_hcl.logic.social_x.dispatcher import ArchetypeDispatcher
        class FakeGateway:
            REFLECTABLE_POST_TYPES = frozenset()
            REFLECTION_EXCLUDED_POST_TYPES = frozenset()
        d = ArchetypeDispatcher(
            gateway=FakeGateway(), social_x_db_path="/tmp/x.db",
            recency_boost_per_day=0.2,    # double default
            recency_boost_threshold=0.7,  # higher than default
            recency_boost_max=2.0,         # higher than default
        )
        assert d.recency_boost_per_day == 0.2
        assert d.recency_boost_threshold == 0.7
        assert d.recency_boost_max == 2.0


# ─── X_POST_PUBLISHED publisher injection (chunk 9F) ───────────────────

class TestPostSuccessCallback:

    def test_gateway_has_set_post_success_callback_method(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        assert hasattr(SocialXGateway, "set_post_success_callback")
        assert hasattr(SocialXGateway, "_invoke_post_success_callback")

    def test_invoke_no_op_when_callback_unset(self):
        # Gateway is standalone — no callback set means no-op (legacy path)
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        # Use __new__ to skip __init__ side effects (DB creation etc.)
        g = SocialXGateway.__new__(SocialXGateway)
        g._post_success_callback = None
        # Should not raise
        g._invoke_post_success_callback(
            tweet_id="t123", titan_id="T1", post_type="world_mirror",
            archetype_candidate=None, status="verified")

    def test_invoke_calls_callback_with_correct_kwargs(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        g = SocialXGateway.__new__(SocialXGateway)
        captured = []

        def fake_callback(**kwargs):
            captured.append(kwargs)

        g._post_success_callback = fake_callback
        g._invoke_post_success_callback(
            tweet_id="t999", titan_id="T2", post_type="reflection",
            archetype_candidate=None, status="posted")
        assert len(captured) == 1
        assert captured[0]["tweet_id"] == "t999"
        assert captured[0]["titan_id"] == "T2"
        assert captured[0]["post_type"] == "reflection"
        assert captured[0]["status"] == "posted"
        assert captured[0]["archetype"] == ""  # no candidate
        assert captured[0]["pool"] == ""
        assert captured[0]["source_id"] == ""

    def test_invoke_extracts_archetype_metadata_from_candidate(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        from titan_hcl.logic.social_x.archetypes.base import ArchetypeCandidate
        g = SocialXGateway.__new__(SocialXGateway)
        captured = []
        g._post_success_callback = lambda **kw: captured.append(kw)
        # Real ArchetypeCandidate
        cand = ArchetypeCandidate(
            archetype="reflection", pool="A_recent_delta",
            source_id="tweet:12345",
            layers=[], layer_values={}, prompt_template="", prompt_values={},
        )
        g._invoke_post_success_callback(
            tweet_id="t1", titan_id="T1", post_type="reflection",
            archetype_candidate=cand, status="verified")
        assert captured[0]["archetype"] == "reflection"
        assert captured[0]["pool"] == "A_recent_delta"
        assert captured[0]["source_id"] == "tweet:12345"


# ─── Spirit_worker retirement (D-SPEC-116) ─────────────────────────────

class TestSpiritWorkerRetired:
    """D-SPEC-116 (2026-05-22): spirit_worker.py was fully DELETED. The
    interim D8-3 guards (legacy-body deleted, _emit_x_catalyst gone,
    _SocialWorkerOwnsX sentinel) are superseded by the ultimate invariant —
    the module no longer exists. SocialXGateway is the sole X path (social_worker)."""

    def test_spirit_worker_module_deleted(self):
        import importlib.util
        assert importlib.util.find_spec(
            "titan_hcl.modules.spirit_worker") is None, (
            "spirit_worker.py must stay deleted (D-SPEC-116); engines live "
            "in cognitive_worker + Rust daemons. Did the retirement revert?")


# ─── Smoke: gateway init + meter init via social_worker helpers ────────

class TestSocialWorkerHelpers:

    def test_init_pressure_meter_returns_meter_or_none(self):
        from titan_hcl.modules.social_worker import _init_pressure_meter
        meter = _init_pressure_meter({"social_presence": {"x_post_threshold": 50.0}})
        # On test env, meter may or may not boot — but result must not raise
        assert meter is None or hasattr(meter, "on_catalyst_event")
