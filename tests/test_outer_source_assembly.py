"""
Tests for the Phase C outer-source dissolution (RFP_phase_c_titan_hcl_cleanup §2 Phase C):
  - outer_source_assembly.assemble_outer_sources — SHM-direct field mapping/transforms
  - outer_source_assembly.OuterHeavyStatsRefresher — cache shape
  - outer_sidecar_providers.make_outer_* — breath-key computation (re-homed trackers)
  - inner_perception_state_publisher.InnerPerceptionStatePublisher — payload

Run: python -m pytest tests/test_outer_source_assembly.py -v -p no:anchorpy
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from titan_hcl.logic.outer_source_assembly import (  # noqa: E402
    OuterSourceContext, assemble_outer_sources,
)
from titan_hcl.logic import outer_sidecar_providers as osp  # noqa: E402


class FakeBank:
    """Minimal ShmReaderBank stand-in returning canned slot payloads."""

    def read_agency_state(self):
        return {"total_actions": 10, "failed_actions": 2, "success_rate": 0.8,
                "helper_statuses": {"x": "ok"}}

    def read_assessment_state(self):
        return {"average_score": 0.7, "recent": [{"score": 0.7}]}

    def read_social_perception_state(self):
        return {"sentiment_ema": 0.6}

    def read_memory_state(self):
        return {"persistent_count": 100, "mempool_size": 5,
                "learning_velocity": 0.3, "directive_alignment": 0.9,
                "effective_nodes_24h": 50, "high_quality_count": 20,
                "kg_node_count": 200, "kg_edge_count": 400}

    def read_titanvm_registers(self):
        return {"programs": {"IMPULSE": {"urgency": 0.5},
                             "VIGILANCE": {"urgency": 0.4}}}

    def read_cgn_engine_state(self):
        return {"avg_reward": 0.55, "grounded_density": 0.3, "consolidations": 7}

    def read_output_verifier_state(self):
        return {"rejected_count": 3, "verified_count": 90}

    def read_language_state(self):
        return {"vocab_total": 1234, "composition_level": 2}

    def read_meta_reasoning_state(self):
        return {"meta_cgn": {
            "knowledge_helpful_ratio": 0.6, "usage_gini": 0.3,
            "eureka_accelerated_per_hour": 1.2, "primitives_total": 9,
            "primitives_grounded": 7, "status": "ok"}}  # status must be dropped

    def read_expression_state(self):
        return {"sovereignty_ratio": 0.8, "learned_actions": 4,
                "total_actions": 10, "composites": {
                    "ART": {"fire_count": 3}, "MUSIC": {"fire_count": 2},
                    "SPEAK": {"fire_count": 5}, "SOCIAL": {"fire_count": 1}}}

    def read_soul_state(self):
        return {"soul_initialized": True}

    def read_timechain_state(self):
        return {"total_blocks": 500, "recent_anchor_age_s": 120.0}

    def read_inner_perception_state(self):
        return {"audio_state": {"a": 1}, "visual_state": {"v": 1},
                "ambient_change": 0.2, "last_create_ts": 99.0}

    def read_pi_heartbeat(self):
        return {"pulse_count": 1000.0}

    def read_outer_spirit_45d(self):
        return {"values": [0.5] * 45}


def _ctx():
    return OuterSourceContext(shm_bank=FakeBank(), titan_id="T1",
                              data_dir="/tmp/nonexistent_data", start_time=0.0)


# ── assemble_outer_sources field mapping ────────────────────────────


def test_assemble_only_requested_keys():
    out = assemble_outer_sources({"agency_stats"}, _ctx())
    assert set(out.keys()) == {"agency_stats"}
    assert out["agency_stats"]["total_actions"] == 10


def test_helper_statuses_from_agency_slot():
    out = assemble_outer_sources({"helper_statuses"}, _ctx())
    assert out["helper_statuses"] == {"x": "ok"}


def test_knowledge_graph_name_transform():
    out = assemble_outer_sources({"knowledge_graph_stats"}, _ctx())
    # kg_node_count → node_count (the name the tensor consumes)
    assert out["knowledge_graph_stats"]["node_count"] == 200
    assert out["knowledge_graph_stats"]["edge_count"] == 400


def test_meta_cgn_projected_to_consumed_fields_only():
    out = assemble_outer_sources({"meta_cgn_stats"}, _ctx())
    mc = out["meta_cgn_stats"]
    assert mc["knowledge_helpful_ratio"] == 0.6
    assert mc["primitives_grounded"] == 7
    assert "status" not in mc  # non-consumed field dropped (keeps slot small)


def test_cgn_stats_from_engine_slot():
    out = assemble_outer_sources({"cgn_stats"}, _ctx())
    assert out["cgn_stats"]["avg_reward"] == 0.55
    assert out["cgn_stats"]["grounded_density"] == 0.3


def test_soul_health_derived_from_soul_initialized():
    out = assemble_outer_sources({"soul_health"}, _ctx())
    assert out["soul_health"] == 0.9


def test_memory_growth_under_both_keys():
    out = assemble_outer_sources({"memory_stats", "memory_growth_metrics"}, _ctx())
    assert out["memory_stats"]["learning_velocity"] == 0.3
    assert out["memory_growth_metrics"] == out["memory_stats"]


def test_inner_perception_from_slot():
    out = assemble_outer_sources({"inner_perception_stats"}, _ctx())
    assert out["inner_perception_stats"]["ambient_change"] == 0.2


def test_substrate_success_rate_derived():
    out = assemble_outer_sources({"substrate_success_rate", "agency_stats"}, _ctx())
    # (10-2)/10 = 0.8
    assert abs(out["substrate_success_rate"] - 0.8) < 1e-6


def test_llm_avg_latency_sentinel():
    assert assemble_outer_sources({"llm_avg_latency"}, _ctx())["llm_avg_latency"] == 0.0


# ── breath providers (re-homed trackers) ────────────────────────────


def test_body_provider_emits_breath_keys():
    prov = osp.make_outer_body_provider(_ctx(), ["agency_stats", "hormone_levels"])
    out = prov()
    assert "outer_body_change" in out          # ChangeBreathTracker
    assert isinstance(out["outer_body_change"], dict)
    # pi_heartbeat_hrv present (pulse_count available)
    assert "pi_heartbeat_hrv" in out


def test_mind_provider_emits_willing_window():
    prov = osp.make_outer_mind_provider(_ctx(), ["agency_stats", "output_verifier_stats"])
    out = prov()
    assert "willing_window" in out
    assert isinstance(out["willing_window"], dict)


def test_spirit_provider_emits_expr_window_and_self_change():
    prov = osp.make_outer_spirit_provider(_ctx(), ["expression_translator_stats"])
    out = prov()
    assert "expr_window" in out
    assert "outer_spirit_self_change" in out


# ── heavy refresher + inner_perception publisher ────────────────────


def test_heavy_refresher_constructs():
    from titan_hcl.logic.outer_source_assembly import OuterHeavyStatsRefresher
    r = OuterHeavyStatsRefresher(titan_id="T1", data_dir="/tmp/x", is_x_gateway=True)
    assert r.cache == {}
    assert r.titan_id == "T1"


def test_inner_perception_publisher_payload():
    from titan_hcl.logic.inner_perception_state_publisher import (
        InnerPerceptionStatePublisher)

    class FakeIP:
        def get_stats(self):
            return {"audio_state": {"a": 1}, "visual_state": {},
                    "ambient_change": 0.5, "last_create_ts": 12.0}

    pub = InnerPerceptionStatePublisher(titan_id="T1")
    payload = pub._compute_payload(FakeIP())
    assert payload["ambient_change"] == 0.5
    assert payload["audio_state"] == {"a": 1}
    assert "ts" in payload
    # stub path
    stub = pub._compute_payload(None)
    assert stub["ambient_change"] == 0.0


def test_cgn_publisher_handles_list_consumers():
    """CGN engine get_stats() can return `consumers` as a LIST (not dict);
    _compute_payload must not raise (found live on T2 2026-05-22 — the raw
    .items() left cgn_engine_state empty fleet-wide)."""
    from titan_hcl.logic.cgn_engine_state_publisher import CGNEngineStatePublisher

    class FakeCGN:
        def get_stats(self):
            return {"consumers": [{"name": "reasoning", "transitions": 3}],
                    "avg_reward": 0.5, "consolidations": 2}

        def get_vm_snapshot(self):
            return {"grounded_density": 0.4}

    pub = CGNEngineStatePublisher(titan_id="T1")
    payload = pub._compute_payload(FakeCGN())  # must not raise
    assert payload["avg_reward"] == 0.5
    assert payload["grounded_density"] == 0.4
    assert payload["consumers"]["reasoning"]["transitions"] == 3


def test_cgn_publisher_handles_str_list_consumers():
    """CGN get_stats() can return consumers as a LIST OF STRINGS (names only);
    _compute_payload must not raise (live T2 2026-05-22 hit str.get())."""
    from titan_hcl.logic.cgn_engine_state_publisher import CGNEngineStatePublisher

    class FakeCGN:
        def get_stats(self):
            return {"consumers": ["reasoning", "social", "meta"],
                    "avg_reward": 0.42, "consolidations": 1}

        def get_vm_snapshot(self):
            return {"grounded_density": 0.25}

    pub = CGNEngineStatePublisher(titan_id="T1")
    payload = pub._compute_payload(FakeCGN())  # must not raise
    assert payload["avg_reward"] == 0.42
    assert payload["grounded_density"] == 0.25
    assert payload["consumers"] == {"reasoning": {}, "social": {}, "meta": {}}
