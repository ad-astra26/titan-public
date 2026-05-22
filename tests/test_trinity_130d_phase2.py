"""
test_trinity_130d_phase2.py — unit tests for Phase 2 of
rFP_trinity_130d_awakening / SPEC §23.

Scope: 13 new dims + their producers. Each test pins a specific
formula against the SPEC §23 contract. Pure-function tests; no DB,
no bus, no plugin instance — they protect the producer + tensor
layers from regression as the architecture evolves.

Run:
  python -m pytest tests/test_trinity_130d_phase2.py -v -p no:anchorpy --tb=short
"""
from __future__ import annotations

import math
import time
from collections import deque

import pytest


# ─────────────────────────────────────────────────────────────────────
# §4.4 — InnerPerceptionState (SPEC §23.5 inner_mind[5,7,9])
# ─────────────────────────────────────────────────────────────────────

class TestAudioPerception:
    def test_record_and_count_recent(self):
        from titan_hcl.logic.inner_perception import AudioPerception
        ap = AudioPerception()
        ap.record_create()
        ap.record_create()
        assert ap.get_state()["creates_recent"] == 2

    def test_old_creates_excluded_from_window(self):
        from titan_hcl.logic.inner_perception import AudioPerception
        ap = AudioPerception()
        ap.record_create(ts=time.time() - 7200)  # 2h ago
        ap.record_create()
        assert ap.get_state(window_s=3600)["creates_recent"] == 1


class TestVisualPerception:
    def test_record_and_count(self):
        from titan_hcl.logic.inner_perception import VisualPerception
        vp = VisualPerception()
        for _ in range(3):
            vp.record_create()
        assert vp.get_state()["creates_recent"] == 3


class TestAmbientChangeMonitor:
    def test_cold_start_returns_zero(self):
        from titan_hcl.logic.inner_perception import AmbientChangeMonitor
        m = AmbientChangeMonitor(lambda: (0.5, 0.5))
        assert m.get_value() == 0.0  # n<5 cold-start

    def test_high_variance_saturates(self):
        from titan_hcl.logic.inner_perception import AmbientChangeMonitor
        m = AmbientChangeMonitor(lambda: (0.0, 0.0))
        # Manually push samples with high variance.
        m._history.extend([0.0, 1.5, 0.0, 1.5, 0.0, 1.5, 0.0, 1.5, 0.0, 1.5])
        v = m.get_value()
        assert v == 1.0  # saturation expected

    def test_low_variance_low_value(self):
        from titan_hcl.logic.inner_perception import AmbientChangeMonitor
        m = AmbientChangeMonitor(lambda: (0.5, 0.5))
        m._history.extend([1.0, 1.01, 1.02, 1.0, 0.99, 1.0, 1.0])
        v = m.get_value()
        assert v < 0.1  # near-zero variance → low value


class TestInnerPerceptionStateNotifyCreate:
    def test_audio_only_routes_to_audio(self):
        from titan_hcl.logic.inner_perception import InnerPerceptionState
        s = InnerPerceptionState(lambda: (0.5, 0.5))
        s.notify_create("audio")
        s.notify_create("music")
        s.notify_create("art")
        s.notify_create("text")  # no perception, but updates last_create_ts
        stats = s.get_stats()
        assert stats["audio_state"]["creates_recent"] == 2  # audio + music
        assert stats["visual_state"]["creates_recent"] == 1  # art
        assert stats["last_create_ts"] > 0

    def test_text_only_updates_last_create_ts(self):
        from titan_hcl.logic.inner_perception import InnerPerceptionState
        s = InnerPerceptionState(lambda: (0.5, 0.5))
        s.notify_create("text")
        stats = s.get_stats()
        assert stats["audio_state"]["creates_recent"] == 0
        assert stats["visual_state"]["creates_recent"] == 0
        assert stats["last_create_ts"] > 0


# ─────────────────────────────────────────────────────────────────────
# SPEC §23.5 — mind_tensor inner_mind feeling formulas
# ─────────────────────────────────────────────────────────────────────

class TestMindTensorPhase2Formulas:
    """Pin SPEC §23.5 inner_mind[5,7,9] formulas."""

    def test_inner_hearing_full_creates_no_ambient(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        v = collect_mind_15d(
            current_5d=[0.5] * 5,
            audio_state={"creates_recent": 5, "ambient": 0.0},
        )
        # 0.5*min(1, 5/5) + 0.5*0.0 = 0.5
        assert abs(v[5] - 0.5) < 1e-9

    def test_inner_hearing_zero_creates_high_ambient(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        v = collect_mind_15d(
            current_5d=[0.5] * 5,
            audio_state={"creates_recent": 0, "ambient": 0.8},
        )
        # 0.5*0 + 0.5*0.8 = 0.4
        assert abs(v[5] - 0.4) < 1e-9

    def test_inner_sight_5_creates_05_ambient(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        v = collect_mind_15d(
            current_5d=[0.5] * 5,
            visual_state={"creates_recent": 5, "ambient": 0.5},
        )
        # 0.5*1.0 + 0.5*0.5 = 0.75
        assert abs(v[7] - 0.75) < 1e-9

    def test_inner_smell_passthrough_ambient_change(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        v = collect_mind_15d(current_5d=[0.5] * 5, ambient_change=0.42)
        assert abs(v[9] - 0.42) < 1e-9

    def test_inner_smell_clamped(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        v = collect_mind_15d(current_5d=[0.5] * 5, ambient_change=2.5)
        assert v[9] == 1.0
        v2 = collect_mind_15d(current_5d=[0.5] * 5, ambient_change=-0.3)
        assert v2[9] == 0.0


# ─────────────────────────────────────────────────────────────────────
# §4.2 — Agency surrender_capacity producers
# ─────────────────────────────────────────────────────────────────────

class TestAgencySurrenderProducers:
    def _make_agency(self):
        from titan_hcl.logic.agency.module import AgencyModule
        return AgencyModule()

    def _push(self, m, helper, posture, success):
        res = {"success": success, "result": "", "enrichment_data": {},
               "error": None}
        return m._build_result(0, posture, helper, res, "", None)

    def test_failed_retry_rate_cold_start_is_zero(self):
        m = self._make_agency()
        assert m._compute_failed_retry_rate() == 0.0

    def test_heuristic_retry_detection(self):
        m = self._make_agency()
        # First fail, then same helper+posture within 30s = retry.
        self._push(m, "art_creator", "CREATIVITY", False)
        self._push(m, "art_creator", "CREATIVITY", False)  # retry, fail
        self._push(m, "art_creator", "CREATIVITY", True)   # retry, success
        # 2 retries, 1 failed → 0.5
        assert abs(m._compute_failed_retry_rate() - 0.5) < 1e-9

    def test_burst_frequency_cold_start_is_zero(self):
        m = self._make_agency()
        for _ in range(5):
            self._push(m, "infra_inspect", "VIGILANCE", True)
        # n<10 → cold-start
        assert m._compute_burst_frequency() == 0.0

    def test_burst_frequency_regular_cadence(self):
        m = self._make_agency()
        # Synthesize 15 evenly-spaced timestamps in the deque directly.
        m._action_timestamps.clear()
        for i in range(15):
            m._action_timestamps.append(1000.0 + i * 1.0)
        bf = m._compute_burst_frequency()
        # Coefficient of variation = 0 for perfectly even spacing.
        assert bf == 0.0

    def test_burst_frequency_clustered(self):
        m = self._make_agency()
        m._action_timestamps.clear()
        # Cluster at start, then long gap, then cluster — high CV.
        for i in range(10):
            m._action_timestamps.append(1000.0 + i * 0.1)
        m._action_timestamps.append(2000.0)  # huge gap
        bf = m._compute_burst_frequency()
        assert bf > 0.5  # bursty

    def test_get_stats_exposes_phase2_keys(self):
        m = self._make_agency()
        s = m.get_stats()
        assert "failed_retry_rate" in s
        assert "burst_frequency" in s


# ─────────────────────────────────────────────────────────────────────
# §4.5 — OuterSpiritHistory trackers (SPEC §23.9 SAT[11], CHIT[25,26,29], ANANDA[10])
# ─────────────────────────────────────────────────────────────────────

class TestEnvironmentalAdaptationTracker:
    def _make(self):
        from titan_hcl.logic.outer_spirit_history import (
            EnvironmentalAdaptationTracker)
        return EnvironmentalAdaptationTracker()

    def test_cold_start_neutral(self):
        assert self._make().compute() == 0.5

    def test_low_variance_high_adaptation(self):
        t = self._make()
        for s in [0.55, 0.55, 0.55, 0.55, 0.55, 0.55]:
            t.record(s, cpu_thermal=0.7)
        assert t.compute() > 0.95

    def test_high_variance_low_adaptation(self):
        t = self._make()
        for s in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
            t.record(s, cpu_thermal=0.7)
        assert t.compute() < 0.5

    def test_low_thermal_ignored(self):
        t = self._make()
        for s in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
            t.record(s, cpu_thermal=0.3)  # below 0.6 threshold
        # Nothing recorded → cold-start neutral
        assert t.compute() == 0.5


class TestGracefulRestTracker:
    def _make(self):
        from titan_hcl.logic.outer_spirit_history import GracefulRestTracker
        return GracefulRestTracker()

    def test_cold_start_neutral(self):
        assert self._make().compute() == 0.5

    def test_min_score_during_rest(self):
        t = self._make()
        for s in [0.7, 0.6, 0.8]:
            t.record(s, cpu_spike_rate=0.1, circadian_phase=0.1)
        assert t.compute() == 0.6  # min

    def test_only_rest_window_counts(self):
        t = self._make()
        # High spike → ignored
        t.record(0.1, cpu_spike_rate=0.8, circadian_phase=0.1)
        # High circadian → ignored
        t.record(0.1, cpu_spike_rate=0.1, circadian_phase=0.8)
        for s in [0.7, 0.7, 0.7]:
            t.record(s, cpu_spike_rate=0.1, circadian_phase=0.1)
        # Only the 3 rest-window entries → min=0.7
        assert t.compute() == 0.7


class TestCircadianAlignmentTracker:
    def _make(self):
        from titan_hcl.logic.outer_spirit_history import (
            CircadianAlignmentTracker)
        return CircadianAlignmentTracker()

    def test_cold_start_neutral(self):
        # RE-GROUNDED (rFP_trinity_dim_resonance): π-pulse cadence regularity.
        # < MIN_INTERVALS observed → 0.5.
        t = self._make()
        t.record_cluster(0, 0)    # baseline seed (no interval)
        t.record_cluster(1, 30)   # 1 interval — still < MIN_INTERVALS
        assert t.compute() == 0.5

    def test_steady_cadence_high_alignment(self):
        # Perfectly steady epochs-per-pulse (interval=30 each) → CV=0 → ~1.0.
        t = self._make()
        ep = 0
        for c in range(1, 11):
            ep += 30
            t.record_cluster(c, ep)
        assert t.compute() > 0.95

    def test_erratic_cadence_low_alignment(self):
        # Wildly varying intervals → high CV → low alignment.
        t = self._make()
        ep = 0
        for c, iv in enumerate([5, 200, 8, 300, 12, 250, 6, 180], start=1):
            ep += iv
            t.record_cluster(c, ep)
        steady = self._make()
        ep2 = 0
        for c in range(1, 11):
            ep2 += 30
            steady.record_cluster(c, ep2)
        assert t.compute() < steady.compute()

    def test_idempotent_same_pulse_count(self):
        """Repeated gathers with the same pulse_count record no new interval."""
        t = self._make()
        t.record_cluster(5, 100)   # baseline seed
        for _ in range(10):
            t.record_cluster(5, 130)  # same count → no new interval
        assert len(t._intervals) == 0


class TestSelfTrajectoryTracker:
    def _make(self):
        from titan_hcl.logic.outer_spirit_history import SelfTrajectoryTracker
        return SelfTrajectoryTracker()

    def test_cold_start_zero(self):
        assert self._make().compute() == 0.0

    def test_zero_movement_zero_trajectory(self):
        t = self._make()
        v = [0.5] * 45
        t.record_snapshot(v)
        t.record_snapshot(list(v))
        assert t.compute() == 0.0

    def test_movement_normalized(self):
        t = self._make()
        t.record_snapshot([0.5] * 45)
        t.record_snapshot([0.7] * 45)
        # L2 = sqrt(45 * 0.04) ≈ 1.342, /5.0 = 0.268
        v = t.compute()
        assert 0.26 < v < 0.28

    def test_clamped_to_one(self):
        t = self._make()
        t.record_snapshot([0.0] * 45)
        t.record_snapshot([1.0] * 45)
        # L2 = sqrt(45) ≈ 6.7, /5.0 = 1.34 → clamped to 1.0
        assert t.compute() == 1.0


class TestDreamRecallProducer:
    def test_cold_start_zero(self):
        from titan_hcl.logic.outer_spirit_history import DreamRecallProducer
        p = DreamRecallProducer(lambda: None)
        assert p.get_value() == 0.0

    def test_refresh_reads_e_mem(self):
        from titan_hcl.logic.outer_spirit_history import DreamRecallProducer

        class _StubEMem:
            def get_recall_ratio(self): return 0.42
        p = DreamRecallProducer(lambda: _StubEMem())
        p.refresh()
        assert abs(p.get_value() - 0.42) < 1e-9


class TestOuterSpiritHistoryAggregator:
    def test_idempotent_ingestion(self):
        from titan_hcl.logic.outer_spirit_history import OuterSpiritHistory
        h = OuterSpiritHistory(lambda: None)
        # Same assessments fed twice — should only land once
        recents = [{"score": 0.7, "ts": 100.0}, {"score": 0.6, "ts": 200.0}]
        h.ingest_assessments(recents, cpu_thermal=0.7,
                              cpu_spike_rate=0.1, circadian_phase=0.1)
        h.ingest_assessments(recents, cpu_thermal=0.7,
                              cpu_spike_rate=0.1, circadian_phase=0.1)
        # env_adapt deque should hold 2 entries (not 4)
        assert len(h.environmental_adaptation._scores) == 2

    def test_get_stats_shape(self):
        from titan_hcl.logic.outer_spirit_history import OuterSpiritHistory
        h = OuterSpiritHistory(lambda: None)
        s = h.get_stats()
        # v1.32.0 / D-SPEC-94 — outer_spirit_trajectory added to OSH
        # output to close CHIT[14] self_trajectory missing-field gap.
        assert set(s.keys()) == {
            "environmental_adaptation",
            "graceful_rest",
            "circadian_alignment",
            "dream_recall_ratio",
            "outer_spirit_trajectory",
        }

    def test_self_trajectory_in_get_stats_v1_32_0(self):
        """v1.32.0 / D-SPEC-94 — outer_spirit_trajectory populated from
        SelfTrajectoryTracker after 45D snapshots ingested via
        ``ingest_outer_spirit_45d``."""
        import time
        from titan_hcl.logic.outer_spirit_history import (
            OuterSpiritHistory, SELF_TRAJ_SNAPSHOT_INTERVAL_S)
        h = OuterSpiritHistory(lambda: None)
        # Cold start — no snapshots → trajectory 0.0
        assert h.get_stats()["outer_spirit_trajectory"] == 0.0
        # Ingest first snapshot
        h.ingest_outer_spirit_45d([0.5] * 45)
        # Force throttle past 30s for second ingest (overwrite internal ts)
        h._self_traj_last_ingest_ts = time.time() - (
            SELF_TRAJ_SNAPSHOT_INTERVAL_S + 1.0)
        h.ingest_outer_spirit_45d([0.6] * 45)
        # 2 snapshots → trajectory = L2(diff) / 5.0 = sqrt(45*0.01)/5 ≈ 0.134
        traj = h.get_stats()["outer_spirit_trajectory"]
        assert 0.05 < traj < 0.30, f"expected ~0.134, got {traj}"

    def test_ingest_outer_spirit_45d_throttle(self):
        """v1.32.0 / D-SPEC-94 — back-to-back ingests within
        ``SELF_TRAJ_SNAPSHOT_INTERVAL_S`` window should drop the second
        snapshot (deque stays at length 1)."""
        from titan_hcl.logic.outer_spirit_history import OuterSpiritHistory
        h = OuterSpiritHistory(lambda: None)
        h.ingest_outer_spirit_45d([0.1] * 45)
        h.ingest_outer_spirit_45d([0.9] * 45)  # within throttle window
        # Deque holds only the first; second was throttled out.
        assert len(h.self_trajectory._snapshots) == 1


# ─────────────────────────────────────────────────────────────────────
# SPEC §23.9 — outer_spirit Phase 2 dim formulas (tensor layer)
# ─────────────────────────────────────────────────────────────────────

class TestOuterSpiritPhase2Formulas:
    def _build(self, *, hist=None, acts=None, soc=None, hlvl=None,
               assess=None):
        from titan_hcl.logic.outer_spirit_tensor import (
            collect_outer_spirit_45d)
        return collect_outer_spirit_45d(
            current_5d=[0.5] * 5,
            outer_body=[0.5] * 5,
            outer_mind=[0.5] * 15,
            history=hist or {},
            action_stats=acts or {"success_rate": 0.8, "error_rate": 0.2},
            social_stats=soc,
            hormone_levels=hlvl,
            assessment_stats=assess or {"mean_score": 0.5},
        )

    def test_sat11_env_adapt_from_history(self):
        v = self._build(hist={"environmental_adaptation": 0.85})
        # SAT[11] = 45D index 11
        assert abs(v[11] - 0.85) < 1e-6

    def test_chit25_dream_recall_ratio_from_history(self):
        # chit[10] = 45D index 25; tensor formula multiplies by 0.6
        # and adds 0.4*body_coh. body_coh = mean([0.5]*5) = 0.5.
        # ratio=1.0 → 1.0*0.6 + 0.5*0.4 = 0.8
        v = self._build(hist={"dream_recall_ratio": 1.0})
        assert abs(v[25] - 0.8) < 1e-6

    def test_chit26_circadian_alignment_passthrough(self):
        v = self._build(hist={"circadian_alignment": 0.73})
        assert abs(v[26] - 0.73) < 1e-6

    def test_chit29_self_trajectory_passthrough(self):
        v = self._build(hist={"outer_spirit_trajectory": 0.4})
        assert abs(v[29] - 0.4) < 1e-6

    def test_ananda10_graceful_rest_passthrough(self):
        v = self._build(hist={"rest_performance_floor": 0.62})
        # ANANDA[10] = 45D index 30+10 = 40
        assert abs(v[40] - 0.62) < 1e-6

    def test_ananda11_creative_tension_formula(self):
        # CREATIVITY=0.8, dt=300 → 0.8 * min(1, 300/600) = 0.8*0.5 = 0.4
        v = self._build(
            hist={"seconds_since_last_create": 300.0},
            hlvl={"CREATIVITY": 0.8},
        )
        assert abs(v[41] - 0.4) < 1e-6

    def test_ananda11_saturates_after_10min(self):
        v = self._build(
            hist={"seconds_since_last_create": 1200.0},
            hlvl={"CREATIVITY": 0.7},
        )
        # min(1, 1200/600) = 1 → tension = 0.7
        assert abs(v[41] - 0.7) < 1e-6

    def test_ananda42_surrender_capacity_formula(self):
        # mean(failed_retry=0.6, 1-body_coh=0.5, burst=0.4) = 0.5
        # surrender = 1 - 0.5 = 0.5
        v = self._build(
            acts={"failed_retry_rate": 0.6, "burst_frequency": 0.4,
                  "success_rate": 0.8, "error_rate": 0.2},
        )
        # body_coh = mean([0.5]*5) = 0.5 → 1-body_coh = 0.5
        # mean(0.6, 0.5, 0.4) = 0.5 → surrender = 0.5
        assert abs(v[42] - 0.5) < 1e-6

    def test_ananda44_flow_state_gated_by_surrender(self):
        # body_coh=mind_coh=0.5, error_rate=0.2 (factor=0.8),
        # mean_score=0.5, surrender_gate from ANANDA[42].
        # acts: failed_retry=0, burst=0; body_coh=0.5; surrender = 1-(0+0.5+0)/3 = 0.833...
        # flow = 0.5 * 0.8 * 0.5 * 0.833... ≈ 0.1666...
        v = self._build(
            acts={"failed_retry_rate": 0.0, "burst_frequency": 0.0,
                  "success_rate": 0.8, "error_rate": 0.2},
            assess={"mean_score": 0.5},
        )
        expected_surrender = 1.0 - (0.0 + 0.5 + 0.0) / 3.0
        expected_flow = 0.5 * 0.8 * 0.5 * expected_surrender
        assert abs(v[44] - expected_flow) < 1e-6

    def test_ananda36_community_connection_normalized(self):
        # tensor: min(1, new_connections_per_window / 5.0)
        v = self._build(soc={"new_connections_per_window": 3})
        # 3/5 = 0.6
        assert abs(v[36] - 0.6) < 1e-6

    def test_ananda36_saturates_at_5(self):
        v = self._build(soc={"new_connections_per_window": 20})
        assert v[36] == 1.0

    def test_ananda38_expression_reach_passthrough(self):
        v = self._build(soc={"creative_engagement": 0.42})
        assert abs(v[38] - 0.42) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# §4.4 — SocialXGateway Phase 2 producer (community + reach)
# ─────────────────────────────────────────────────────────────────────

class TestSocialXGatewayCommunityProducer:
    def test_community_engagement_method_exists(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        assert hasattr(SocialXGateway, "get_community_engagement_stats")

    def test_method_returns_expected_shape(self):
        # Unit test against an in-memory schema: instantiate against a
        # tmp DB so the queries exercise real SQL paths.
        import tempfile, os
        from titan_hcl.logic.social_x_gateway import SocialXGateway

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "social_x.db")
            gw = SocialXGateway(db_path=db_path)
            stats = gw.get_community_engagement_stats()
            assert "distinct_handles_24h" in stats
            assert "mean_engagement_per_post_7d" in stats
            assert "expression_reach_norm" in stats
            assert "gateway_role" in stats
            # Empty DB → all zero
            assert stats["distinct_handles_24h"] == 0
            assert stats["expression_reach_norm"] == 0.0
            # Default is_x_gateway=True → "canonical"
            assert stats["gateway_role"] == "canonical"

    def test_non_x_gateway_returns_delegation_marker(self):
        """T2/T3 don't have local X data — producer must short-circuit
        when is_x_gateway=False, NOT touch the local DB."""
        import tempfile, os
        from titan_hcl.logic.social_x_gateway import SocialXGateway

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "social_x.db")
            gw = SocialXGateway(db_path=db_path)
            stats = gw.get_community_engagement_stats(is_x_gateway=False)
            # Delegation marker
            assert stats["gateway_role"] == "non-canonical"
            assert stats["distinct_handles_24h"] == 0
            assert stats["mean_engagement_per_post_7d"] == 0.0
            assert stats["expression_reach_norm"] == 0.0

    def test_x_gateway_true_does_query_db(self):
        """Sanity check that is_x_gateway=True still reaches the DB layer
        (so the canonical T1 path doesn't accidentally short-circuit)."""
        import tempfile, os
        from titan_hcl.logic.social_x_gateway import SocialXGateway

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "social_x.db")
            gw = SocialXGateway(db_path=db_path)
            stats = gw.get_community_engagement_stats(is_x_gateway=True)
            # Empty DB but query path executes (gateway_role="canonical").
            assert stats["gateway_role"] == "canonical"
